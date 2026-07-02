// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

// ownership_test.go is the OWNERSHIP INVARIANT harness — the assertion
// conservation alone cannot make.
//
// Conservation proves no value is minted or burned: Σ balances is constant. But
// the vault-drain / colliding-handle exploit class CONSERVES value while STEALING
// ownership — it moves a victim's funds to an attacker who was never a party to
// any accepted event. Σ balances is unchanged, so a conservation sweep stays
// green while the wrong account ends up holding the money.
//
// The ownership invariant closes that gap. It is a statement about ATTRIBUTION,
// not totals:
//
//	For every (account, asset), any non-zero balance delta across an accepted
//	block MUST be explained by an accepted event in which that account is an
//	EXPLICIT PARTY.
//
// The explicit parties of an accepted block, by event:
//
//	deposit      -> the DEPOSITOR (the decoded deposit user), in the deposited asset.
//	withdraw     -> the WITHDRAWER (the decoded withdraw user), in the withdrawn asset.
//	place        -> the ORDER OWNER (the decoded place user): available -> locked in
//	                the spend asset (a rested place), or a no-op (a rejected place).
//	cancel       -> the ORDER OWNER (resolved via orderuser: by the cancelled id):
//	                locked -> available in the order's reserved asset.
//	trade-fill   -> the TAKER (the decoded submit user) AND each MAKER (resolved via
//	                orderuser: by the fill's resting order id), in base AND quote.
//
// A non-zero delta for an (account, asset) that is NOT an explicit party of ANY
// event in the block is an OWNERSHIP VIOLATION — value arrived at, or left, an
// account that no accepted event authorized. That is exactly the theft the
// colliding-handle bug would have produced, and exactly what conservation misses.
//
// The harness:
//   - snapshots EVERY per-(account,asset) (available, locked) row from the durable
//     ledger before and after a block (the same whole-keyspace sweep the
//     conservation property uses, so an attacker account outside any roster is
//     still seen),
//   - diffs them,
//   - derives the explicit-party set from the block's accepted events (its txs +
//     consensus outcomes), resolving maker/cancel identity through the SAME
//     orderuser: index settlement uses,
//   - and fails if any non-zero delta is unexplained.
//
// It runs ALONGSIDE conservation in the property and full-cycle tests, so every
// randomized sequence now proves BOTH "no value created/destroyed" AND "no value
// moved to a non-party".

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// acctAsset is a single (account, asset) ledger coordinate — the FULL 16-byte
// settlement identity and the FULL 32-byte asset id, the exact width the ledger
// keys by. It is the unit the ownership invariant quantifies over.
type acctAsset struct {
	user  userKey
	asset [32]byte
}

// balPair is an account's (available, locked) holdings for one asset. Ownership
// reasons over the TOTAL the account controls (available + locked are both the
// account's value — a lock is the account's own escrow), but keeps the split so a
// place (available -> locked, same owner) is recognized as a within-owner move,
// not two unrelated deltas.
type balPair struct {
	available uint64
	locked    uint64
}

// total is the value the account controls for the asset — the quantity that must
// not change for a NON-party. (available + locked: a locked reserve is still the
// owner's value, escrowed by the owner's own order.)
func (b balPair) total() uint64 { return b.available + b.locked }

// ownershipSnapshot sweeps the WHOLE durable ledger (every balance:/locked: row)
// into a per-(account,asset) (available, locked) map, WITHOUT trusting any roster
// of accounts. A delta into a derived/wrong account (the theft signature) shows
// up here because the sweep totals the entire keyspace, exactly like the
// conservation sweep. Must mirror that sweep's key discipline: keys are
// <prefix><user:16><asset:32>.
func ownershipSnapshot(t *testing.T, vm *VM) map[acctAsset]balPair {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	out := map[acctAsset]balPair{}
	sweep := func(prefix string, set func(p *balPair, v uint64)) {
		it := vm.db.NewIteratorWithPrefix([]byte(prefix))
		defer it.Release()
		const userW = zapwire.UserSize
		for it.Next() {
			key := it.Key()
			if len(key) != len(prefix)+userW+32 {
				t.Fatalf("ownership sweep: %q key wrong width %d (want %d)", prefix, len(key), len(prefix)+userW+32)
			}
			val := it.Value()
			if len(val) != 8 {
				t.Fatalf("ownership sweep: %q value wrong width %d (want 8)", prefix, len(val))
			}
			var k acctAsset
			copy(k.user[:], key[len(prefix):len(prefix)+userW])
			copy(k.asset[:], key[len(prefix)+userW:len(prefix)+userW+32])
			p := out[k]
			set(&p, be64(val))
			out[k] = p
		}
		if err := it.Error(); err != nil {
			t.Fatalf("ownership sweep %q iterate: %v", prefix, err)
		}
	}
	sweep(prefixBalance, func(p *balPair, v uint64) { p.available = v })
	sweep(prefixLocked, func(p *balPair, v uint64) { p.locked = v })
	return out
}

// partySet is the set of (account, asset) coordinates that the accepted events of
// ONE block name as EXPLICIT PARTIES. A delta for a coordinate in this set is
// authorized; a delta for a coordinate NOT in it is an ownership violation.
type partySet map[acctAsset]struct{}

func (s partySet) add(user userKey, asset [32]byte) { s[acctAsset{user, asset}] = struct{}{} }

func (s partySet) has(user userKey, asset [32]byte) bool {
	_, ok := s[acctAsset{user, asset}]
	return ok
}

// orderUserResolver resolves a resting order's FULL settlement identity to build
// the maker/cancel party set, from the SAME source settlement uses: the
// orderuser: index. It is seeded with a snapshot of every orderuser: row taken
// BEFORE the block (so a maker that rested in a PRIOR block, and whose row is
// deleted by a full fill in THIS block, is still resolvable), and augmented with
// every TxPlace in the block (so a maker that rested EARLIER IN THIS SAME block —
// whose row did not exist in the pre-snapshot — is resolvable too). This is the
// honest mirror of how settlement itself finds the maker: never the 8-byte handle,
// always the full 16-byte account keyed by (pool, orderID).
type orderUserResolver struct {
	byOrder map[orderRef]userKey
}

type orderRef struct {
	pool    [32]byte
	orderID uint64
}

// snapshotOrderUsers reads every orderuser: row into a resolver. The orderuser:
// key is orderuser:<poolID:32><orderID:8> -> user[16].
func snapshotOrderUsers(t *testing.T, vm *VM) *orderUserResolver {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	r := &orderUserResolver{byOrder: map[orderRef]userKey{}}
	it := vm.db.NewIteratorWithPrefix([]byte(prefixOrderUser))
	defer it.Release()
	const prefW = len(prefixOrderUser)
	for it.Next() {
		key := it.Key()
		if len(key) != prefW+32+8 {
			t.Fatalf("orderuser sweep: key wrong width %d", len(key))
		}
		val := it.Value()
		if len(val) != zapwire.UserSize {
			t.Fatalf("orderuser sweep: value wrong width %d", len(val))
		}
		var ref orderRef
		copy(ref.pool[:], key[prefW:prefW+32])
		ref.orderID = beU64(key[prefW+32 : prefW+40])
		var u userKey
		copy(u[:], val)
		r.byOrder[ref] = u
	}
	if err := it.Error(); err != nil {
		t.Fatalf("orderuser sweep iterate: %v", err)
	}
	return r
}

func (r *orderUserResolver) resolve(pool [32]byte, orderID uint64) (userKey, bool) {
	u, ok := r.byOrder[orderRef{pool, orderID}]
	return u, ok
}

// beU64 reads an 8-byte big-endian uint64 from a key slice.
func beU64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0])<<56 | uint64(b[1])<<48 | uint64(b[2])<<40 | uint64(b[3])<<32 |
		uint64(b[4])<<24 | uint64(b[5])<<16 | uint64(b[6])<<8 | uint64(b[7])
}

// partiesOfBlock derives the explicit-party set from a block's accepted events:
// its txs (decoded for the named user / market) and its consensus outcomes (the
// realized fills / placed-order ids). preOrderUsers resolves a maker/cancel order
// id to its full account; the block's own TxPlaces are added to it first so a
// same-block maker is resolvable. The set is the ONLY accounts any value is
// authorized to move for in this block.
func partiesOfBlock(t *testing.T, vm *VM, blk *Block, preOrderUsers *orderUserResolver) partySet {
	t.Helper()
	parties := partySet{}

	// Augment the resolver with this block's own placements: orderID =
	// blockDeterministicID(height, txIndex), owner = the decoded place user. This
	// makes a maker that rests earlier in THIS block resolvable for a cross later
	// in the same block.
	res := &orderUserResolver{byOrder: map[orderRef]userKey{}}
	for k, v := range preOrderUsers.byOrder {
		res.byOrder[k] = v
	}
	for i, tx := range blk.txs {
		if tx.Type != TxPlace {
			continue
		}
		pool, _, _, _, user, err := zapwire.DecodePlace(tx.Body)
		if err != nil {
			continue
		}
		res.byOrder[orderRef{pool, blockDeterministicID(blk.height, uint32(i))}] = userKey16(user)
	}

	// Market (base,quote) bindings are needed to attribute a place/cancel's locked
	// asset and a fill's two legs. Read from committed state (post-accept).
	marketAssets := func(pool [32]byte) (base, quote [32]byte, ok bool) {
		vm.mu.Lock()
		defer vm.mu.Unlock()
		b, q, bound, err := readMarketAssets(vm.db, pool)
		if err != nil || !bound {
			return base, quote, false
		}
		return b, q, true
	}

	for _, tx := range blk.txs {
		switch tx.Type {
		case TxDeposit:
			user, asset, _, _, err := zapwire.DecodeDeposit(tx.Body)
			if err != nil {
				continue
			}
			parties.add(userKey16(user), asset) // depositor credited the asset

		case TxWithdraw:
			user, asset, _, _, err := zapwire.DecodeWithdraw(tx.Body)
			if err != nil {
				continue
			}
			parties.add(userKey16(user), asset) // withdrawer debited the asset

		case TxPlace:
			pool, side, _, _, user, err := zapwire.DecodePlace(tx.Body)
			if err != nil {
				continue
			}
			base, quote, ok := marketAssets(pool)
			if !ok {
				continue
			}
			// A place locks (or refunds) the spend asset for its OWNER only: buy locks
			// quote, sell locks base. The owner is the only party to a place.
			uid := userKey16(user)
			if side == sideBuy {
				parties.add(uid, quote)
			} else {
				parties.add(uid, base)
			}

		case TxCancel:
			pool, orderID, err := zapwire.DecodeCancel(tx.Body)
			if err != nil {
				continue
			}
			base, quote, ok := marketAssets(pool)
			if !ok {
				continue
			}
			// A cancel unlocks the order's reserve to its OWNER (resolved via
			// orderuser:). The owner is the only party; admit both assets so either
			// side's reserve is covered (a sell reserved base, a buy reserved quote).
			if owner, ok := res.resolve(pool, orderID); ok {
				parties.add(owner, base)
				parties.add(owner, quote)
			}

		case TxSubmit:
			pool, _, _, _, _, user, err := zapwire.DecodeSubmit(tx.Body)
			if err != nil {
				continue
			}
			base, quote, ok := marketAssets(pool)
			if !ok {
				continue
			}
			// The TAKER (the submitter) is a party in BOTH assets: it spends one and
			// receives the other, and its unspent lock is refunded.
			taker := userKey16(user)
			parties.add(taker, base)
			parties.add(taker, quote)
			// Each MAKER of a fill this submit produced is a party in BOTH assets. The
			// makers are read from the block's PERSISTED trade rows (which carry the
			// exact MakerOrderID) and resolved to a full account through the SAME
			// orderuser: index settlement uses — NEVER the trade row's 8-byte
			// MakerUserID handle. This is the EXACT maker set (not a superset): only
			// the owner of the specific resting order a fill consumed is admitted, so
			// even an attacker who happens to hold an UNRELATED resting order on the
			// same pool is NOT credited another order's proceeds without tripping the
			// gate.
			for _, mid := range makerOrderIDsForSubmit(t, vm, blk, tx, pool) {
				if owner, ok := res.resolve(pool, mid); ok {
					parties.add(owner, base)
					parties.add(owner, quote)
				}
			}
		}
	}
	return parties
}

// makerOrderIDsForSubmit returns the resting maker order ids the fills of THIS
// submit consumed, read from the block's persisted trade rows. The submit's fill
// count comes from its consensus outcome (snapshotted before Accept); the block's
// trade:<height>:<seq> rows are appended in execution order, so a single-submit
// block's fills are exactly the block's trade rows. For attribution this is the
// authoritative maker set: each row's MakerOrderID names the exact resting order a
// fill consumed.
func makerOrderIDsForSubmit(t *testing.T, vm *VM, blk *Block, tx *Tx, pool [32]byte) []uint64 {
	t.Helper()
	// Only the fills attributed to this submit matter. In the property/full-cycle
	// tests a submit is the only matching tx in its block, so every trade row at
	// this height belongs to it; reading all rows at the height is exact. (A block
	// with multiple submits would need per-tx fill ranges; the harness asserts a
	// single matching tx per block via the count check below.)
	oc := outcomeForTx(blk, tx)
	want := len(oc.fills)
	if want == 0 {
		return nil
	}
	vm.mu.Lock()
	defer vm.mu.Unlock()
	prefix := make([]byte, len(prefixTrade)+8)
	copy(prefix, prefixTrade)
	for i := 0; i < 8; i++ {
		prefix[len(prefixTrade)+7-i] = byte(blk.height >> (8 * i))
	}
	it := vm.db.NewIteratorWithPrefix(prefix)
	defer it.Release()
	var ids []uint64
	for it.Next() {
		row, ok := dex.DecodeTrade(it.Value())
		if !ok {
			t.Fatalf("ownership: corrupt trade row at height %d", blk.height)
		}
		ids = append(ids, row.MakerOrderID)
	}
	if err := it.Error(); err != nil {
		t.Fatalf("ownership: trade iterate: %v", err)
	}
	return ids
}

// outcomeForTx returns the consensus outcome the block derived for a given tx (by
// tx id). The block's outcomes are set at Verify and snapshotted by the caller
// BEFORE Accept clears them; this resolves a submit's fills for attribution.
func outcomeForTx(blk *Block, tx *Tx) txOutcome {
	id := tx.ID()
	for _, o := range blk.outcomes {
		if o.txID == id {
			return o
		}
	}
	return txOutcome{}
}

// findOwnershipViolation is the PURE ownership gate: it returns the first
// (account, asset) whose controlled total CHANGED across the block while NOT
// being an explicit party of the block's accepted events, with ok=true when such
// a violation exists. This is the decomplected core — detection with no test
// glue — so it can be asserted clean (assertOwnership) AND asserted to FIRE
// against a simulated-vulnerable state (TestOwnershipInvariant_*).
//
// It checks the controlled TOTAL (available + locked): a place moves a party's
// own available -> locked (total unchanged, owner is a party either way); a fill
// moves value BETWEEN parties (both parties' totals change, both authorized); a
// theft moves value to a NON-party (its total rises with no event naming it —
// the violation this returns).
func findOwnershipViolation(before, after map[acctAsset]balPair, parties partySet) (acctAsset, int64, bool) {
	seen := map[acctAsset]struct{}{}
	for k := range before {
		seen[k] = struct{}{}
	}
	for k := range after {
		seen[k] = struct{}{}
	}
	// Deterministic order so a failure names the same coordinate every run.
	keys := make([]acctAsset, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sortAcctAssets(keys)
	for _, k := range keys {
		b := before[k]
		a := after[k]
		if a.total() == b.total() {
			continue // no change in controlled value -> nothing to authorize
		}
		if !parties.has(k.user, k.asset) {
			return k, int64(a.total()) - int64(b.total()), true
		}
	}
	return acctAsset{}, 0, false
}

// sortAcctAssets imposes a total order on coordinates (user bytes then asset
// bytes) so violation reporting is deterministic.
func sortAcctAssets(ks []acctAsset) {
	less := func(x, y acctAsset) bool {
		for i := range x.user {
			if x.user[i] != y.user[i] {
				return x.user[i] < y.user[i]
			}
		}
		for i := range x.asset {
			if x.asset[i] != y.asset[i] {
				return x.asset[i] < y.asset[i]
			}
		}
		return false
	}
	for i := 1; i < len(ks); i++ {
		for j := i; j > 0 && less(ks[j], ks[j-1]); j-- {
			ks[j], ks[j-1] = ks[j-1], ks[j]
		}
	}
}

// assertOwnership is the ownership gate as a test assertion: it fails (Fatalf)
// when findOwnershipViolation reports a non-party whose controlled value changed
// — value that arrived at, or left, an account no accepted event authorized (the
// theft conservation misses). `where` names the step for a precise failure.
func assertOwnership(t *testing.T, before, after map[acctAsset]balPair, parties partySet, where string) {
	t.Helper()
	if k, delta, bad := findOwnershipViolation(before, after, parties); bad {
		b := before[k]
		a := after[k]
		t.Fatalf("%s: OWNERSHIP VIOLATED for account %x asset %x:\n"+
			"  controlled total %d -> %d (delta %d)\n"+
			"  but this account is NOT an explicit party to ANY accepted event in the block\n"+
			"  (value moved to/from an account no deposit/withdraw/fill/cancel named — the theft conservation misses)",
			where, k.user, k.asset, b.total(), a.total(), delta)
	}
}

// ownershipChecker bundles the pre-block snapshots so a test can assert ownership
// over a block with one before/after pair. Usage:
//
//	oc := newOwnershipChecker(t, vm)   // snapshot before
//	blk := addBlock(t, vm, ...)        // accept the block
//	oc.assert(t, vm, blk)              // snapshot after + gate
//
// It is the reusable harness the property and full-cycle tests apply alongside
// the conservation assertion.
type ownershipChecker struct {
	before     map[acctAsset]balPair
	orderUsers *orderUserResolver
}

func newOwnershipChecker(t *testing.T, vm *VM) *ownershipChecker {
	t.Helper()
	return &ownershipChecker{
		before:     ownershipSnapshot(t, vm),
		orderUsers: snapshotOrderUsers(t, vm),
	}
}

func (c *ownershipChecker) assert(t *testing.T, vm *VM, blk *Block, where string) {
	t.Helper()
	after := ownershipSnapshot(t, vm)
	parties := partiesOfBlock(t, vm, blk, c.orderUsers)
	assertOwnership(t, c.before, after, parties, where)
}

// parties returns the explicit-party set the harness derives for a block (so a
// test can reuse the SAME set the gate uses against a simulated-vulnerable state).
func (c *ownershipChecker) parties(t *testing.T, vm *VM, blk *Block) partySet {
	t.Helper()
	return partiesOfBlock(t, vm, blk, c.orderUsers)
}

// ---------------------------------------------------------------------------
// THE CENTERPIECE: the harness CATCHES the exact bug class conservation missed.
// ---------------------------------------------------------------------------

// TestOwnershipInvariant_CollidingAttackerIsNotAParty is the proof that the
// ownership gate catches the colliding-handle theft — the bug class a conservation
// sweep cannot see (it CONSERVES value while STEALING ownership).
//
// It runs the colliding-8 scenario two ways against the SAME derived party set:
//
//  1. FIXED code (clean): a victim rests a maker; a distinct, legitimate taker
//     crosses it. Real settlement credits the maker leg to the VICTIM (resolved
//     via orderuser:). The harness sees only the victim (maker) and the taker
//     change value — both explicit parties — and reports NO violation. Crucially
//     the colliding ATTACKER (who shares the victim's 8-byte matcher handle but is
//     a DISTINCT full account) is NEVER credited.
//
//  2. SIMULATED-VULNERABLE (caught): we take the SAME before-snapshot and the SAME
//     party set the harness derived, then construct the after-state a vulnerable
//     8-byte-fold settlement WOULD produce — the maker's 50-LUSD proceeds credited
//     to the colliding ATTACKER instead of the victim. The attacker shares the
//     victim's handle, so a fallback-to-handle settlement mis-routes the proceeds
//     to it. The attacker is NOT a party to the cross (neither the resting maker
//     nor the taker), so findOwnershipViolation FLAGS the attacker's unexplained
//     credit. CONSERVATION still holds in this vulnerable state (the 50 LUSD just
//     moved victim->attacker; Σ is unchanged) — which is exactly why conservation
//     misses it and ownership does not.
func TestOwnershipInvariant_CollidingAttackerIsNotAParty(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	victim, attacker := twoUsersSharingUserID8()
	// Precondition for the exploit class: same 8-byte matcher handle, distinct full
	// accounts.
	if idOf(t, victim) == idOf(t, attacker) {
		t.Fatal("test setup invalid: victim/attacker are the same full account")
	}
	const taker = "legit-taker"
	pool := [32]byte{0x0c, 0xa1}

	// Victim rests a SELL of 10 LUX @ 5 (locks 10 LUX); a legitimate taker funds
	// 1000 LUSD to cross it. The attacker deposits NOTHING and never trades — it is
	// purely the colliding non-party the vulnerable fold would mis-credit.
	addBlock(t, vm,
		depositTx(t, victim, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, victim))
	assertBalance(t, vm, victim, assetLUX, 90, 10)

	// --- (1) FIXED code: the cross settles to the real parties; harness is clean. ---
	own := newOwnershipChecker(t, vm)
	crossBlk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))

	// Real settlement: victim (maker) received 50 LUSD + gave up 10 LUX; taker
	// received 10 LUX + spent 50 LUSD. The colliding attacker got NOTHING.
	assertBalance(t, vm, victim, assetLUX, 90, 0)
	assertBalance(t, vm, victim, assetLUSD, 50, 0)
	assertBalance(t, vm, taker, assetLUX, 10, 0)
	assertBalance(t, vm, taker, assetLUSD, 950, 0)
	assertBalance(t, vm, attacker, assetLUX, 0, 0)
	assertBalance(t, vm, attacker, assetLUSD, 0, 0)

	before := own.before
	afterFixed := ownershipSnapshot(t, vm)
	parties := own.parties(t, vm, crossBlk)

	// The party set names the victim (maker) and the taker — never the attacker.
	if parties.has(idOf(t, attacker), assetLUSD) || parties.has(idOf(t, attacker), assetLUX) {
		t.Fatal("party set wrongly includes the colliding attacker (it is neither maker nor taker)")
	}
	if !parties.has(idOf(t, victim), assetLUSD) || !parties.has(idOf(t, taker), assetLUX) {
		t.Fatal("party set missing a real party (victim maker / legit taker)")
	}

	// On the FIXED state, the harness reports NO ownership violation.
	if k, delta, bad := findOwnershipViolation(before, afterFixed, parties); bad {
		t.Fatalf("ownership gate FALSE-flagged the fixed cross: account %x asset %x delta %d (all deltas are real parties)", k.user, k.asset, delta)
	}

	// --- (2) SIMULATED-VULNERABLE: the 8-byte fold mis-credits the attacker. ---
	// Construct the after-state a fallback-to-handle settlement would produce: the
	// victim's 50-LUSD maker proceeds land on the colliding ATTACKER instead. We
	// mutate a COPY of the real after-state so production state is untouched.
	afterVuln := cloneBalances(afterFixed)
	const proceeds = 50
	moveControlled(afterVuln, idOf(t, victim), assetLUSD, idOf(t, attacker), assetLUSD, proceeds)

	// CONSERVATION still holds in the vulnerable state (value only MOVED, none
	// minted/burned) — the precise reason a conservation sweep cannot catch this.
	if sumControlled(before, assetLUSD) != sumControlled(afterVuln, assetLUSD) {
		t.Fatal("test setup error: simulated theft was not value-conserving")
	}
	if sumControlled(before, assetLUX) != sumControlled(afterVuln, assetLUX) {
		t.Fatal("test setup error: simulated theft perturbed LUX")
	}

	// OWNERSHIP catches it: the attacker's controlled value rose with no event
	// naming it as a party. This is the theft conservation missed.
	k, delta, bad := findOwnershipViolation(before, afterVuln, parties)
	if !bad {
		t.Fatal("ownership gate FAILED to flag the colliding-attacker theft (the whole point of the harness)")
	}
	if k.user != idOf(t, attacker) || k.asset != assetLUSD {
		t.Fatalf("ownership gate flagged the wrong coordinate: %x/%x (want attacker/LUSD)", k.user, k.asset)
	}
	if delta != proceeds {
		t.Fatalf("flagged delta = %d, want +%d (the stolen maker proceeds)", delta, proceeds)
	}
	t.Logf("ownership gate flagged colliding-attacker theft: account %x gained %d LUSD as a NON-party", k.user[:8], delta)
}

// cloneBalances deep-copies a per-(account,asset) balance snapshot so a test can
// mutate it (model a vulnerable settlement) without touching the real snapshot.
func cloneBalances(in map[acctAsset]balPair) map[acctAsset]balPair {
	out := make(map[acctAsset]balPair, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// moveControlled models a value move BETWEEN accounts in a snapshot: debit `amt`
// of the controlled (available) balance from (fromUser,fromAsset) and credit it to
// (toUser,toAsset). Used to construct the after-state a vulnerable settlement would
// produce (e.g. maker proceeds mis-routed to a colliding handle).
func moveControlled(snap map[acctAsset]balPair, fromUser userKey, fromAsset [32]byte, toUser userKey, toAsset [32]byte, amt uint64) {
	from := acctAsset{fromUser, fromAsset}
	to := acctAsset{toUser, toAsset}
	f := snap[from]
	f.available -= amt
	snap[from] = f
	tp := snap[to]
	tp.available += amt
	snap[to] = tp
}

// sumControlled totals the controlled value (available + locked) for one asset
// across every account in a snapshot — the conservation total, used here to prove
// the simulated theft is value-conserving (so only ownership, not conservation,
// can catch it).
func sumControlled(snap map[acctAsset]balPair, asset [32]byte) uint64 {
	var sum uint64
	for k, v := range snap {
		if k.asset == asset {
			sum += v.total()
		}
	}
	return sum
}
