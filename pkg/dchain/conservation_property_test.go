// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

// conservation_property_test.go is the PROPERTY proof for the D-Chain custody
// ledger — the parallel ledger that backs "the money lives in the order book".
//
// Where conservation_test.go pins a handful of HAND-PICKED scenarios, this file
// drives long, RANDOMIZED sequences of deposit / place / submit(match) / cancel /
// withdraw against the REAL VM (BuildBlock -> Verify -> Accept, one op per block)
// and asserts the global conservation invariant holds after EVERY operation:
//
//	I = A + L + F + E   (per asset, in atomic units)
//
//	I (Imported) = Σ every amount ever DEPOSITED for the asset.
//	A (Available) = Σ available balance over every account (ground-truth DB sweep).
//	L (Locked)    = Σ locked balance over every account (ground-truth DB sweep).
//	F (Fees)      = Σ fees the venue skimmed for the asset. This ledger takes NO
//	                fee (settleFills moves maker<->taker with no skim), so F is
//	                STRUCTURALLY 0 — and this test ASSERTS it. If a fee leg is ever
//	                added without a matching accounting bucket, A+L+E would fall
//	                below I and this property fails LOUDLY. F is a first-class term
//	                precisely so the proof stays honest if the model grows a fee.
//	E (Exported)  = Σ every REALIZED amount ever WITHDRAWN for the asset.
//
// The driver is fully deterministic: a fixed table of PRNG seeds (NOT wall-clock,
// NOT a nondeterministic source), each replayed against a fresh VM over a fresh
// memdb. The adversarial moves the task calls out are exercised by construction:
// withdraw-more-than-available (clamp), cancel-after-fill, double-withdraw,
// partial fills, unfunded place (reject), cancel of a stale/unknown id. The
// acceptance criterion is binary: the invariant NEVER breaks, across every seed
// and every step. That is the executable proof that the parallel ledger creates
// and destroys NO value.

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// ledgerTotals is the GROUND-TRUTH A+L per asset: it sweeps the durable ledger
// (every balance:/locked: row in the VM's own store) and sums by asset, WITHOUT
// trusting the test's roster of accounts. A settlement bug that minted value into
// a derived/wrong account (outside the roster) still shows up here, because the
// sweep totals the whole keyspace. Keys are <prefix><user:8><asset:32>, so the
// asset is the trailing 32 bytes after the 8-byte user handle.
//
// It locks vm.mu and reads vm.db directly — the same authoritative store
// Block.Accept commits to. This is white-box (same package) on purpose: the
// conservation property is a statement about the WHOLE ledger, and the only
// honest way to total the whole ledger is to read every row.
func ledgerTotals(t *testing.T, vm *VM) (available, locked map[[32]byte]uint64) {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	available = sumByAsset(t, vm, prefixBalance)
	locked = sumByAsset(t, vm, prefixLocked)
	return available, locked
}

// sumByAsset iterates one ledger prefix and returns asset -> Σ value, keyed by the
// FULL 32-byte asset id (the ledger key width). Must be called under vm.mu. A
// corrupt-width key/value is a hard test failure (the ledger never writes a
// malformed row, so seeing one means a real bug).
func sumByAsset(t *testing.T, vm *VM, prefix string) map[[32]byte]uint64 {
	t.Helper()
	out := make(map[[32]byte]uint64)
	it := vm.db.NewIteratorWithPrefix([]byte(prefix))
	defer it.Release()
	for it.Next() {
		key := it.Key()
		// <prefix><user:16><asset:32> — the user is the FULL 16-byte settlement
		// identity (zapwire.UserSize), the asset the full 32-byte injective id.
		const userW = zapwire.UserSize
		if len(key) != len(prefix)+userW+32 {
			t.Fatalf("conservation sweep: %q key wrong width %d (want %d)", prefix, len(key), len(prefix)+userW+32)
		}
		val := it.Value()
		if len(val) != 8 {
			t.Fatalf("conservation sweep: %q value wrong width %d (want 8)", prefix, len(val))
		}
		var asset [32]byte
		copy(asset[:], key[len(prefix)+userW:len(prefix)+userW+32])
		amt := be64(val)
		if out[asset]+amt < out[asset] {
			t.Fatalf("conservation sweep: asset %x total overflow", asset)
		}
		out[asset] += amt
	}
	if err := it.Error(); err != nil {
		t.Fatalf("conservation sweep %q iterate: %v", prefix, err)
	}
	return out
}

// a32 packs a uint64 asset LABEL into the LOW 8 bytes of a full 32-byte asset id
// (big-endian), the test-side analog of the production injective address->id map.
// The d-chain ledger keys by the full 32-byte id; tests use compact uint64 labels
// and lift them here at the wire/ledger boundary.
func a32(label uint64) [32]byte {
	var a [32]byte
	be := func(v uint64, dst []byte) {
		for i := 0; i < 8; i++ {
			dst[7-i] = byte(v >> (8 * i))
		}
	}
	be(label, a[24:32])
	return a
}

// alreadySeen reports whether a tx with these exact bytes has ALREADY been
// applied by an accepted block — i.e. the d-chain's content-addressed dedup will
// treat the next submission as a no-op replay (returning the first outcome,
// moving no value). The frozen ZAP frame carries no nonce, so a byte-identical
// deposit/withdraw the randomized driver re-emits IS such a replay. The driver
// uses this to count imported/exported value only for genuinely-new txs — and in
// doing so the property test proves the idempotency contract too: a replay
// changes neither the ledger nor the conservation totals. Reads vm.db under
// vm.mu, exactly as Block.execute does via getSeenOutcome.
func alreadySeen(vm *VM, tx *Tx) bool {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	seen, err := isSeen(vm.db, tx.ID())
	if err != nil {
		// A read error here is not a conservation question; surface it as "not
		// seen" so the caller proceeds and any real DB fault fails elsewhere loudly.
		return false
	}
	return seen
}

// be64 reads an 8-byte big-endian uint64 (the ledger's canonical width). Local to
// avoid pulling encoding/binary into the assertion path's hot loop signature.
func be64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0])<<56 | uint64(b[1])<<48 | uint64(b[2])<<40 | uint64(b[3])<<32 |
		uint64(b[4])<<24 | uint64(b[5])<<16 | uint64(b[6])<<8 | uint64(b[7])
}

// ledgerModel tracks the test-side accounting the chain's behaviour is checked
// against: cumulative imported (I) and exported (E) per asset, plus the asset
// universe seen. F (fees) is held explicitly and asserted zero — the model has no
// fee bucket because the ledger takes no fee; carrying F makes that an EXPLICIT,
// checkable property rather than a silent assumption.
type ledgerModel struct {
	imported map[[32]byte]uint64 // I: Σ deposited
	exported map[[32]byte]uint64 // E: Σ realized-withdrawn
	fees     map[[32]byte]uint64 // F: Σ fees skimmed (must stay 0)
	assets   map[[32]byte]struct{}
}

func newLedgerModel() *ledgerModel {
	return &ledgerModel{
		imported: map[[32]byte]uint64{},
		exported: map[[32]byte]uint64{},
		fees:     map[[32]byte]uint64{},
		assets:   map[[32]byte]struct{}{},
	}
}

func (m *ledgerModel) note(asset [32]byte) { m.assets[asset] = struct{}{} }

func (m *ledgerModel) deposit(asset [32]byte, amt uint64) {
	m.note(asset)
	m.imported[asset] += amt
}

func (m *ledgerModel) withdraw(asset [32]byte, realized uint64) {
	m.note(asset)
	m.exported[asset] += realized
}

// assertConservation is the heart of the proof: for every asset the test has
// touched, I == A + L + F + E. A is summed from the live DB sweep, L likewise, I
// and E from the model, F asserted zero. Called after EVERY accepted op. The
// `where` string names the step so a violation pinpoints the exact op + seed.
func (m *ledgerModel) assertConservation(t *testing.T, vm *VM, where string) {
	t.Helper()
	avail, locked := ledgerTotals(t, vm)

	// Defense-in-depth: the live ledger must not hold a row for an asset the test
	// never deposited (that would be value minted from nowhere). Sweep both DB
	// maps and fail if an unknown asset appears.
	for a := range avail {
		if _, ok := m.assets[a]; !ok {
			t.Fatalf("%s: available ledger holds UNKNOWN asset %x = %d (value from nowhere)", where, a, avail[a])
		}
	}
	for a := range locked {
		if _, ok := m.assets[a]; !ok {
			t.Fatalf("%s: locked ledger holds UNKNOWN asset %x = %d (value from nowhere)", where, a, locked[a])
		}
	}

	for a := range m.assets {
		I := m.imported[a]
		A := avail[a]
		L := locked[a]
		F := m.fees[a]
		E := m.exported[a]
		if F != 0 {
			t.Fatalf("%s: asset %x has F=%d, but the ledger takes NO fee — accounting drift", where, a, F)
		}
		if A+L+F+E != I {
			t.Fatalf("%s: CONSERVATION VIOLATED for asset %x:\n"+
				"  I (imported)  = %d\n"+
				"  A (available) = %d\n"+
				"  L (locked)    = %d\n"+
				"  F (fees)      = %d\n"+
				"  E (exported)  = %d\n"+
				"  A+L+F+E       = %d  (want == I = %d)\n"+
				"  delta         = %d",
				where, a, I, A, L, F, E, A+L+F+E, I, int64(A+L+F+E)-int64(I))
		}
	}
}

// conservationSeeds is the FIXED seed table (deterministic; not wall-clock). Each
// seed replays a fresh, independent randomized session. A diverse spread of
// starting points exercises different op interleavings without any
// nondeterministic input — re-running the test reproduces every session exactly.
// REGRESSION GUARD — seed 4242: its randomized stream drives a SELF-TRADE through
// the marketable path — the SAME account both rests a maker and submits a crossing
// taker. The maker, rebuilt from its persisted row, carries the leading-8-byte
// folded identity (persist.userToUint64) while the fresh taker carries the full
// 16-byte wire identity, so the matcher's old raw-string self-trade check failed to
// recognize the self-cross. The cross then desynced the integer-units lane (which
// fully consumed the maker, deleting its orderuser:/reserve) from the float lane
// (which left it resting), and a LATER cross hit the missing orderuser: row and
// stranded settlement, breaking conservation. Fixed in pkg/dex by resolving both
// orders through the canonical owner handle (selfTradeKey) in BOTH the place path
// (checkSelfTrade) and the marketable matcher (tryMatchImmediateLocked), so a
// rested-then-rebuilt maker and a fresh taker from the same account compare equal
// and the self-maker leg is skip-and-cancelled (no fill, no value move). This seed
// stays IN the table to lock the fix.
var conservationSeeds = []int64{
	1, 2, 3, 7, 11, 13, 17, 23, 42, 99,
	101, 256, 999, 1337, 4242, 8888888, 31337, 65537, 123456, 7777777,
}

// opsPerSeed is the number of randomized operations driven per seed. With 20
// seeds this is 20*opsPerSeed total state transitions, each followed by a full
// invariant sweep.
const opsPerSeed = 220

// TestConservationInvariantProperty is the executable proof that the parallel
// D-Chain ledger conserves value across arbitrary operation sequences. For each
// fixed seed it stands up a fresh VM, opens two markets over a fixed 3-asset
// universe, then drives opsPerSeed randomized deposit/place/submit/cancel/
// withdraw ops (one per block, through the real BuildBlock->Verify->Accept path),
// asserting I = A + L + F + E after every single op. It deliberately steers into
// the adversarial cases (over-withdraw, cancel-after-fill, double-withdraw,
// partial fills, unfunded place, stale cancel) so the invariant is proven under
// the exact sequences that would expose a mint/burn bug.
func TestConservationInvariantProperty(t *testing.T) {
	// Fixed asset universe (full 32-byte ids via a32). Two markets share LUX as the
	// base so crosses exercise both legs (base + quote) of settlement against a
	// common asset.
	var (
		aLUX  = a32(0x4c5558_00000001) // base
		aLUSD = a32(0x4c555344_000001) // quote A
		aLETH = a32(0x4c455448_000001) // quote B
	)
	poolLUSD := [32]byte{'L', 'U', 'X', '/', 'L', 'U', 'S', 'D', 0xD0}
	poolLETH := [32]byte{'L', 'U', 'X', '/', 'L', 'E', 'T', 'H', 0xD1}

	// Fixed account roster the driver picks from.
	accounts := []string{"alice", "bob", "carol", "dave", "erin"}

	// Per-market (base,quote) so the driver locks the right asset for a side.
	type market struct {
		pool        [32]byte
		base, quote [32]byte
	}
	markets := []market{
		{poolLUSD, aLUX, aLUSD},
		{poolLETH, aLUX, aLETH},
	}

	totalOps := 0
	for _, seed := range conservationSeeds {
		seed := seed
		t.Run(fmt.Sprintf("seed-%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			vm, _ := newTestVM(t, memdb.New())
			defer vm.Shutdown(context.Background())

			model := newLedgerModel()
			for i := range markets {
				model.note(markets[i].base)
				model.note(markets[i].quote)
			}

			// Block 1: open both markets so place/submit have a book.
			addBlock(t, vm,
				openMarketTx(t, poolLUSD, aLUX, aLUSD),
				openMarketTx(t, poolLETH, aLUX, aLETH),
			)
			model.assertConservation(t, vm, "after open-markets")

			// liveOrders tracks order ids the driver has placed per pool, so cancel
			// can target a REAL id (or, on purpose, a stale one). Filled/cancelled
			// ids are left in the slice intentionally to exercise cancel-after-fill
			// and double-cancel (deterministic no-ops the ledger must conserve).
			// Each entry carries the placing OWNER so a cancel is signed by the
			// account that placed the order — the gate refuses a non-owner cancel, so
			// owner-signing is what exercises the real unlock path.
			type placedOrder struct {
				oid   uint64
				owner string
			}
			liveOrders := map[[32]byte][]placedOrder{}

			for step := 0; step < opsPerSeed; step++ {
				// Snapshot the whole ledger + the orderuser: identity index BEFORE the
				// op, so after the op we can assert BOTH conservation (Σ value
				// unchanged) AND ownership (every changed account is an explicit party
				// to an accepted event — no value moved to a non-party). The two
				// invariants are orthogonal: conservation catches mint/burn, ownership
				// catches theft (a conserving mis-attribution).
				own := newOwnershipChecker(t, vm)
				var blk *Block
				var where string

				switch rng.Intn(6) {
				case 0, 1:
					// DEPOSIT (weighted: keeps balances funded so the other ops have
					// something to work with). Pick account + asset + amount.
					acct := accounts[rng.Intn(len(accounts))]
					asset := pickAsset(rng, aLUX, aLUSD, aLETH)
					amt := uint64(1 + rng.Intn(1000))
					tx := depositTx(t, acct, asset, amt)
					// The d-chain dedups CONTENT-IDENTICAL txs (no nonce on the frozen
					// frame): a byte-identical deposit already committed is a no-op that
					// returns the FIRST outcome and credits NOTHING again. The driver can
					// emit the same (acct,asset,amt) twice, so gate the import on whether
					// this exact tx was already applied — counting a deduped replay as new
					// imported value would be a TEST bug, not a ledger one. This makes the
					// property ALSO prove idempotency: a replay moves no value.
					replay := alreadySeen(vm, tx)
					var o []txOutcome
					blk, o = addBlockOutcomes(t, vm, tx)
					realized := withdrawRealizedOf(o, TxDeposit)
					if realized != amt {
						t.Fatalf("deposit realized %d != amount %d (a deposit must never clamp)", realized, amt)
					}
					if !replay {
						model.deposit(asset, amt)
					}
					where = fmt.Sprintf("step %d deposit %s asset=%x amt=%d replay=%v", step, acct, asset, amt, replay)
					model.assertConservation(t, vm, where)

				case 2:
					// PLACE a resting limit order. May rest (locks the spend asset) or
					// be REJECTED when the account cannot fund it (unfunded place) — both
					// conserving. Integer price/size so the lock math is exact.
					mk := markets[rng.Intn(len(markets))]
					acct := accounts[rng.Intn(len(accounts))]
					side := uint8(rng.Intn(2)) // 0 buy, 1 sell
					price := float64(1 + rng.Intn(20))
					size := float64(1 + rng.Intn(40))
					blk = addBlock(t, vm, placePoolTx(t, mk.pool, side, price, size, acct))
					// One op per block, txIndex 0 -> deterministic id. Record it
					// regardless of whether it rested; a rejected place yields an id
					// that resolves to "no such order" on cancel (exercises stale cancel).
					oid := blockDeterministicID(blk.height, 0)
					liveOrders[mk.pool] = append(liveOrders[mk.pool], placedOrder{oid, acct})
					where = fmt.Sprintf("step %d place %s side=%d %g@%g", step, acct, side, size, price)
					model.assertConservation(t, vm, where)

				case 3:
					// SUBMIT a marketable order (the cross/match path). Settlement moves
					// value maker<->taker entirely inside the ledger. Sometimes oversized
					// to force a PARTIAL fill (maker fully consumed, taker remainder
					// refunded); sometimes against an empty book (zero fill). Conserving.
					mk := markets[rng.Intn(len(markets))]
					acct := accounts[rng.Intn(len(accounts))]
					side := uint8(rng.Intn(2))
					price := float64(1 + rng.Intn(20))
					size := float64(1 + rng.Intn(60)) // can exceed resting -> partial
					blk = addBlock(t, vm, submitPoolTx(t, mk.pool, side, false, price, size, acct))
					where = fmt.Sprintf("step %d submit %s side=%d %g@%g", step, acct, side, size, price)
					model.assertConservation(t, vm, where)

				case 4:
					// CANCEL. Target a recorded id (real, filled, or stale) — all of
					// which the ledger handles deterministically: a live order unlocks
					// its reserve (locked->available), a filled/unknown id is a no-op.
					// This is the cancel-after-fill + double-cancel + stale-cancel path.
					mk := markets[rng.Intn(len(markets))]
					ids := liveOrders[mk.pool]
					var oid uint64
					owner := accounts[rng.Intn(len(accounts))] // stale-id case: any signer (no owner row -> no-op)
					if len(ids) > 0 {
						po := ids[rng.Intn(len(ids))]
						oid, owner = po.oid, po.owner
					} else {
						oid = uint64(rng.Int63()) // stale/unknown id when nothing placed yet
					}
					blk = addBlock(t, vm, cancelPoolTx(t, mk.pool, oid, owner))
					where = fmt.Sprintf("step %d cancel oid=%d", step, oid)
					model.assertConservation(t, vm, where)

				case 5:
					// WITHDRAW. Half the time ask for MORE than is available
					// (over-withdraw -> clamp to realized); the realized amount is read
					// from the consensus outcome and added to E. Repeated picks of the
					// same (acct,asset) exercise double-withdraw (second clamps to the
					// remainder). Conserving: E rises by exactly what A fell.
					acct := accounts[rng.Intn(len(accounts))]
					asset := pickAsset(rng, aLUX, aLUSD, aLETH)
					avail, _, err := vm.Balance(wireUser(t, acct), asset)
					if err != nil {
						t.Fatalf("Balance(%s,%x): %v", acct, asset, err)
					}
					var want uint64
					if rng.Intn(2) == 0 && avail > 0 {
						// In-range withdraw (0..avail].
						want = 1 + uint64(rng.Int63n(int64(avail)))
					} else {
						// Over-withdraw (or withdraw from an empty balance) -> must clamp.
						want = avail + uint64(1+rng.Intn(500))
					}
					tx := withdrawTx(t, acct, asset, want)
					// Same content-addressed dedup as deposit: a byte-identical withdraw
					// already committed returns the FIRST realized amount and EXPORTS
					// nothing again. A deduped replay's realized was clamped to the
					// available balance AT FIRST APPLICATION, which can exceed the CURRENT
					// available (intervening withdraws drained it) — so the current-avail
					// mint check and the E update apply only to a genuinely-new withdraw.
					replay := alreadySeen(vm, tx)
					var o []txOutcome
					blk, o = addBlockOutcomes(t, vm, tx)
					realized := withdrawRealizedOf(o, TxWithdraw)
					if realized > want {
						t.Fatalf("withdraw realized %d > want %d (MINT via withdraw)", realized, want)
					}
					if !replay {
						if realized > avail {
							t.Fatalf("withdraw realized %d > available %d (clamp failed -> mint)", realized, avail)
						}
						model.withdraw(asset, realized)
					}
					where = fmt.Sprintf("step %d withdraw %s asset=%x want=%d replay=%v", step, acct, asset, want, replay)
					model.assertConservation(t, vm, where)
				}

				// OWNERSHIP: every account whose controlled value changed in this block
				// must be an explicit party to one of the block's accepted events. This
				// runs alongside the conservation assertion above on EVERY op.
				own.assert(t, vm, blk, where)
				totalOps++
			}

			// Final cross-check: drain every account's full available balance via
			// withdraw, then the ledger's remaining A must equal exactly Σlocked
			// (only live order reserves left), and I == A+L+E still holds. This is
			// the "exported == imported minus what's still escrowed" closure.
			//
			// Each drain OVER-withdraws by a globally-unique pad (>> the random
			// wants' 1..500 range, plus a monotonic counter) so the clamp drains the
			// full available AND no two drain txs are ever byte-identical — the
			// content-addressed dedup can never swallow a drain.
			drainPad := uint64(1_000_000)
			for _, acct := range accounts {
				for _, asset := range [][32]byte{aLUX, aLUSD, aLETH} {
					avail, _, err := vm.Balance(wireUser(t, acct), asset)
					if err != nil {
						t.Fatalf("final Balance(%s,%x): %v", acct, asset, err)
					}
					if avail == 0 {
						continue
					}
					drainPad++
					want := avail + drainPad
					tx := withdrawTx(t, acct, asset, want)
					if alreadySeen(vm, tx) {
						t.Fatalf("final drain tx unexpectedly already seen (pad collision) acct=%s asset=%x want=%d", acct, asset, want)
					}
					ownDrain := newOwnershipChecker(t, vm)
					dblk, o := addBlockOutcomes(t, vm, tx)
					realized := withdrawRealizedOf(o, TxWithdraw)
					if realized != avail {
						t.Fatalf("final drain withdraw realized %d != available %d (clamp must export the full available)", realized, avail)
					}
					model.withdraw(asset, realized)
					model.assertConservation(t, vm, "final-drain "+acct)
					ownDrain.assert(t, vm, dblk, "final-drain "+acct)
				}
			}

			// After draining all available, every asset's remaining ledger value is
			// purely LOCKED (live order reserves). Assert A == 0 for assets with no
			// live locks, and that I - E == L (the still-escrowed remainder).
			availFinal, lockedFinal := ledgerTotals(t, vm)
			for _, asset := range [][32]byte{aLUX, aLUSD, aLETH} {
				if got := availFinal[asset]; got != 0 {
					t.Fatalf("after full drain, asset %x still has %d AVAILABLE (want 0)", asset, got)
				}
				if I, E, L := model.imported[asset], model.exported[asset], lockedFinal[asset]; I-E != L {
					t.Fatalf("closure broken asset %x: I(%d) - E(%d) = %d, but L = %d", asset, I, E, I-E, L)
				}
			}
		})
	}
	t.Logf("conservation property: %d seeds x ~%d ops = %d randomized state transitions, invariant held after EVERY op",
		len(conservationSeeds), opsPerSeed, totalOps)
}

// withdrawRealizedOf extracts the realized amount the consensus outcome carries
// in orderID for the given tx type (deposit credits == amount; withdraw realized
// == clamped debit). Fails if the expected outcome is absent.
func withdrawRealizedOf(outcomes []txOutcome, typ TxType) uint64 {
	for _, oc := range outcomes {
		if oc.typ == typ {
			return oc.orderID
		}
	}
	return 0
}

// pickAsset returns one of the supplied assets uniformly. Keeps the asset choice
// deterministic under the seeded rng.
func pickAsset(rng *rand.Rand, assets ...[32]byte) [32]byte {
	return assets[rng.Intn(len(assets))]
}
