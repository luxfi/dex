// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"

	"github.com/luxfi/database"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/vm/chains/atomic"
)

// drive.go is the PROPOSER-SIDE DRIVE of the native co-located settlement seam — the
// "keeper, but in-consensus". atomic.go is the EXECUTION half (executeImport /
// executeExport: how a C->D object is consumed and a D->C object is produced); this is
// the DRIVE half (what to import / export and when). Before this file the seam could
// only be driven by an external actor manually injecting TxImport/TxExport into the
// mempool (the atomic_seam_e2e_test harness plays exactly that role); NO production path
// generated them, so a real 0x9999 SubmitSwapIntent staged a C->D object that nothing
// imported — no taker funding, no crossing, no D->C settlement object, no DEXFill.
//
// The drive closes that gap WITHOUT a keeper and WITHOUT an external venue: during normal
// block production the VM ITSELF (BuildBlock, vm.go) enumerates the committed cross-chain
// state and emits the import/export txs, which every validator then re-verifies
// deterministically against the SAME committed state. It is keeper-less and consensus-safe
// BY DESIGN (atomic.go:47-55): the import credit is a pure function of the rooted TxImport
// body + the committed shared-memory object; the export object is a pure function of the
// rooted TxExport body + the escrow the import wrote. The drive only DECIDES which txs to
// emit; the txs themselves carry their full authority and are checked by executeImport /
// executeExport on every node, so a divergent proposer is rejected on the tx root.
//
// DETERMINISM CONTRACT (the matcher-at-Verify rule — non-determinism is a chain fork):
//   - The drive reads ONLY committed state: the C->D objects in shared memory (the source
//     chain has accepted), the seamintent: escrow index, and the ledger. No wallclock, no
//     mempool, no node-local counter, no external I/O.
//   - Every enumeration result is SORTED into a total order (imports by intentID, escrows
//     and outputs by their committed key order) before any tx is built, so two proposers
//     on the SAME committed state build byte-identical blocks (the leaderless-build
//     requirement) and a validator re-running execute derives the identical effects.
//   - The drive is bounded per block (maxSeamDrivePerBlock); an excess backlog is drained
//     across subsequent blocks (each leg is independently retryable — a not-yet-imported
//     intent is simply re-enumerated next block).

// SeamPendingTrait is the FIXED, owner-agnostic discovery trait every C->D swap-intent
// object is tagged with in shared memory, so the D-side proposer can ENUMERATE all pending
// intents without first knowing their owners. atomic.SharedMemory.Indexed can only look up
// keys that possess a KNOWN trait (state.go getKeys); the precompile's existing per-owner
// trait (native_staging.go) cannot be enumerated without the owner set, so the seam needs
// one shared constant the writer tags and the reader queries. It is a domain-separated
// 32-byte tag so it can never collide with a 20-byte owner trait.
//
// THE .v2 DOMAIN IS THE FLAG-DAY SWITCH, and it is why an old node and a new node can
// never disagree about a block. A v1 intent object carried VALUE ONLY (69 bytes); a v2
// intent carries value + the CLOB OPERATION (118 bytes). A node that cannot read the
// operation must never import the value — it would have to invent the taker's
// market/side/limit. Rotating the domain makes the two generations mutually INVISIBLE
// rather than mutually confusing, and the tx wire makes them mutually UNPARSEABLE:
//
//   - An OLD node enumerates the v1 trait, finds no v2 object, and emits no seam tx at
//     all. It keeps validating every non-seam block normally. Inert, never divergent.
//   - An OLD node handed a NEW node's seam block cannot execute it: the TxImport body
//     is now 150 bytes and its bodySize(TxImport) says 32, so parseTxFrame refuses the
//     frame (ErrTxMalformedAuth, "unexpected trailing bytes"), decodeTxList fails, and
//     ParseBlock fails. The block is rejected as UNPARSEABLE — the old node never
//     computes a state root for it, so it cannot compute a DIFFERENT one. It stalls.
//   - A NEW node enumerates the v2 trait, so a stale op-less v1 object from a
//     not-yet-upgraded C node is never imported. Fail closed; the taker's principal is
//     reclaimable on C at the intent deadline.
//
// So the changeover costs seam LIVENESS while the fleet is mixed, and can never cost
// SAFETY. It has no retroactive effect either: this seam has never produced a
// transaction on any network (0 events on 0x9999 since genesis on every chain), so
// there is no historical block anywhere containing a TxImport, and a new node
// re-syncing history parses all of it unchanged. The operational rule that follows is
// the whole flag day: upgrade every D validator BEFORE the first C-side v2 intent is
// minted. Until then the drive emits nothing, because there is nothing tagged v2 to
// find.
var SeamPendingTrait = func() []byte {
	d := sha256.Sum256([]byte("lux.dex.native.intent.pending.v2"))
	return d[:]
}()

// maxSeamDrivePerBlock bounds how many import (and, separately, export) legs one block's
// drive emits, so a large backlog cannot produce an unbounded block. The excess is picked
// up by the next block's drive (the enumeration is a pure function of the still-pending
// committed state, so nothing is lost — just deferred). Sized generously for the in-flight
// cross-chain swap count a single block realistically faces.
const maxSeamDrivePerBlock = 256

// driveSeamImports enumerates the committed C->D swap-intent objects in shared memory
// and returns a deterministic, intentID-sorted list of the txs that execute them: for
// each not-yet-imported intent, a TxImport CARRYING the object's bytes immediately
// followed by the TxIntentSubmit that places the taker's order. It is the autonomous
// replacement for a keeper's "watch IntentSubmitted -> inject ImportTx -> place order"
// loop, run inside BuildBlock.
//
// THE SUBMIT IS THE MISSING LEG. Before it, this function emitted the import alone: the
// value landed on D and nothing was ever done with it. No order was placed, so nothing
// crossed, so the export drive (which fires only on a settled intent) never fired, so
// Phase B had no object and the settle reverted. Emitting the submit alongside the
// import is what makes the seam reachable at all.
//
// BOTH LEGS IN ONE BLOCK, so there is no window in which value sits on D with no order
// against it. A block is one atomic unit: if the import rejects (replay, malformed
// object, unexecutable operation) the escrow is never written and the submit that
// follows it rejects too, having nothing to bind — the pair is all-or-nothing without
// needing to be one transaction. Keeping them as two transactions is what lets the
// submit flow through the IDENTICAL custody -> match -> settle path a signed dex_submit
// takes, instead of duplicating that machinery inside the import.
//
// Pure function of committed state: the pending set comes from shared memory (the C
// chain accepted the staging block) and the already-imported filter comes from the
// committed escrow index. The double-guard against re-import — the committed-escrow
// filter here AND executeImport's own replay-reject — means a re-enumerated intent
// (object still present because its consuming Remove has not yet been accepted) is
// imported at most once.
//
// Reading shared memory HERE is sound where reading it inside execute() was not: this
// runs only on the proposer, and everything it produces is committed in the block's own
// bytes (hashed into txRoot) for every validator to re-execute without touching shared
// memory again.
func (vm *VM) driveSeamImports() ([]*Tx, error) {
	if !vm.autoDriveSeam {
		return nil, nil
	}
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return nil, nil // seam not wired: nothing to drive (no mint)
	}

	keys, err := enumerateSeamPending(sm, vm.cChainID, maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: seam import enumerate: %w", err)
	}

	// Filter to un-imported intents, collecting their ids for a total sort.
	pending := make([]ids.ID, 0, len(keys))
	for _, k := range keys {
		if len(k) != 32 {
			continue // a key that is not an intent id is not ours (defensive)
		}
		var intentID ids.ID
		copy(intentID[:], k)
		_, imported, gerr := getSeamIntent(vm.db, intentID)
		if gerr != nil {
			return nil, gerr
		}
		if imported {
			continue // already imported in committed state: skip (no re-import)
		}
		pending = append(pending, intentID)
	}

	// DETERMINISM: total order by intentID, independent of shared-memory iteration order,
	// so every proposer on the same committed partition emits the same sequence.
	sort.Slice(pending, func(i, j int) bool {
		return bytes.Compare(pending[i][:], pending[j][:]) < 0
	})

	out := make([]*Tx, 0, 2*len(pending))
	for _, intentID := range pending {
		object, ok, oerr := readSeamObject(sm, vm.cChainID, intentID)
		if oerr != nil {
			return nil, oerr
		}
		if !ok {
			// Enumerated but unreadable, or not a canonical intent object (an op-less
			// v1 object, a truncated record). Emit nothing rather than a leg that can
			// only reject: the enumeration is a pure function of still-pending state,
			// so a genuinely pending intent is simply picked up in a later block.
			continue
		}
		imp, terr := NewTx(TxImport, EncodeSeamImportBody(intentID, object))
		if terr != nil {
			return nil, fmt.Errorf("dchain: build TxImport: %w", terr)
		}
		sub, terr := NewTx(TxIntentSubmit, EncodeSeamSubmitBody(intentID))
		if terr != nil {
			return nil, fmt.Errorf("dchain: build TxIntentSubmit: %w", terr)
		}
		out = append(out, imp, sub)
		if len(out) >= maxSeamDrivePerBlock {
			return out, nil
		}
	}
	return out, nil
}

// readSeamObject reads the C->D object recorded at [intentID] and returns it only if it
// is a canonical swap intent (118 bytes, railSwap, an executable operation). Everything
// the drive then puts on the wire is bytes it has already validated, so a leg it emits
// is one executeImport can bind.
func readSeamObject(sm atomic.SharedMemory, cChainID, intentID ids.ID) ([]byte, bool, error) {
	vals, err := sm.Get(cChainID, [][]byte{intentID[:]})
	if err != nil {
		return nil, false, fmt.Errorf("dchain: seam object read: %w", err)
	}
	if len(vals) != 1 || len(vals[0]) == 0 {
		return nil, false, nil // not present (not flushed / already consumed)
	}
	object := vals[0]
	_, _, amount, op, ok := decodeSeamIntentObject(object)
	if !ok || amount == 0 || !op.valid() {
		return nil, false, nil
	}
	return object, true, nil
}

// driveSeamExports enumerates the SETTLED intent escrows — those whose order has RUN —
// and returns deterministic TxExport txs that send everything the intent's own account
// holds back to C as D->C objects the 0x9999 ImportSettlement consumes: the realized
// proceeds, and the input the order did not spend.
//
// THE GATE IS THE RECORDED FACT, NOT A GUESS. This used to export only when the account
// held a non-AssetIn balance — an inference that "some other asset appeared, so the
// swap must have traded". The inference is wrong in both directions, and one of those
// directions mints:
//
//   - An order that filled NOTHING leaves only AssetIn, so it never satisfied the
//     heuristic and was never exported. The principal sat on D forever while C's
//     deadline reclaim refunded the same principal to the taker: paid twice.
//   - Any unrelated balance in the account satisfied the heuristic, exporting value
//     that was never this intent's.
//
// Both disappear together. seamIntentSubmitted is written by the submit leg itself, so
// "the order has run" is a fact recorded in committed state; and the account is the
// intent's OWN (seamAccount), so everything in it belongs to this intent by
// construction. Nothing to infer, nothing to mis-attribute.
//
// ATTRIBUTION + CONSERVATION: the export's recipient is the escrow's RECORDED 20-byte
// owner (executeExport binds it), so value can only return to the rightful taker; the
// debit draws the REALIZED ledger balance (executeExport refuses an over-debit) and a
// same-asset refund is capped by the intent's remaining principal — so C can never be
// credited more than D matched. The drive emits the refund leg BEFORE the proceeds legs
// so the per-taker cap is applied against the still-open escrow, and execute CLOSES the
// escrow once the account is fully drained (closeSeamIntentIfDrained), making each swap
// one-shot.
func (vm *VM) driveSeamExports() ([]*Tx, error) {
	if !vm.autoDriveSeam {
		return nil, nil
	}
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return nil, nil // seam not wired: an export cannot be written (no D->C object)
	}

	escrows, err := listSeamIntents(vm.db, seamIntentSubmitted, maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: seam export enumerate: %w", err)
	}

	out := make([]*Tx, 0, len(escrows))
	for _, e := range escrows {
		acct := seamAccount(e.IntentID)

		// The realized OUTPUT (non-AssetIn balances) and the leftover INPUT (AssetIn),
		// both read from committed state.
		outputs, oerr := listAccountOutputs(vm.db, acct, e.AssetIn)
		if oerr != nil {
			return nil, oerr
		}
		leftoverIn, lerr := getAvailable(vm.db, acct, e.AssetIn)
		if lerr != nil {
			return nil, lerr
		}

		// The spent witness the precompile's taker-authenticated MEV floor reads: the
		// AssetIn actually CONSUMED to produce the output = the locked principal minus
		// what is left. EXACT, because the account is this intent's alone — nothing
		// else could have moved either number.
		spent := uint64(0)
		if e.Remaining > leftoverIn {
			spent = e.Remaining - leftoverIn
		}

		// REFUND leg FIRST (same-asset, per-taker-capped, decrements Remaining) so it
		// runs against the still-open escrow.
		if leftoverIn > 0 {
			refund := leftoverIn
			if refund > e.Remaining {
				refund = e.Remaining
			}
			if refund > 0 {
				tx, terr := NewTx(TxExport, EncodeSeamExportBody(e.IntentID, e.AssetIn, refund, 0))
				if terr != nil {
					return nil, fmt.Errorf("dchain: build TxExport refund: %w", terr)
				}
				out = append(out, tx)
			}
		}

		// PROCEEDS legs (sorted by asset id via the committed key order); the draining
		// one triggers the escrow close in execute.
		for _, o := range outputs {
			tx, terr := NewTx(TxExport, EncodeSeamExportBody(e.IntentID, o.asset, o.amount, spent))
			if terr != nil {
				return nil, fmt.Errorf("dchain: build TxExport proceeds: %w", terr)
			}
			out = append(out, tx)
		}
		if len(out) >= maxSeamDrivePerBlock {
			return out, nil
		}
	}
	return out, nil
}

// hasSeamWork reports whether the drive would emit ANY tx from the current committed state
// — a pending un-imported intent, or an open escrow with realized output to export. It is
// the WaitForEvent trigger so the engine calls BuildBlock for a pure-seam block (one with
// no mempool txs). It short-circuits on the first match to stay cheap on the poll path.
func (vm *VM) hasSeamWork() bool {
	if !vm.autoDriveSeam {
		return false
	}
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return false
	}
	// Any un-imported pending intent?
	keys, err := enumerateSeamPending(sm, vm.cChainID, maxSeamDrivePerBlock)
	if err == nil {
		for _, k := range keys {
			if len(k) != 32 {
				continue
			}
			var intentID ids.ID
			copy(intentID[:], k)
			if _, imported, gerr := getSeamIntent(vm.db, intentID); gerr == nil && !imported {
				return true
			}
		}
	}
	// Any settled escrow still holding value to send back?
	escrows, eerr := listSeamIntents(vm.db, seamIntentSubmitted, 1)
	if eerr != nil {
		return false
	}
	return len(escrows) > 0
}

// closeSeamIntentIfDrained marks a SUBMITTED intent escrow seamIntentReclaimed once its
// own account has been FULLY drained (no remaining ledger balance), so a settled intent is
// not re-enumerated by a later export drive. Called from execute after a successful
// TxExport; order-independent (it fires on whichever export — proceeds or refund — empties
// the account), so it correctly closes a full fill (one proceeds leg), a partial fill
// (proceeds + refund), and a zero fill (refund only) alike. A still-funded account (more
// output not yet exported in this block) is left open for a subsequent leg.
func (vm *VM) closeSeamIntentIfDrained(db database.Database, intentID ids.ID) error {
	rec, exists, err := getSeamIntent(db, intentID)
	if err != nil {
		return err
	}
	if !exists || rec.Status != seamIntentSubmitted {
		return nil
	}
	drained, err := accountFullyDrained(db, seamAccount(intentID))
	if err != nil {
		return err
	}
	if !drained {
		return nil
	}
	rec.Status = seamIntentReclaimed
	return putSeamIntent(db, intentID, rec)
}

// --- committed-state enumeration helpers (pure reads; the drive's only inputs) ----------

// enumerateSeamPending walks the C->D shared-memory partition for THIS chain and returns
// the keys (intent ids) of every object tagged SeamPendingTrait, up to [max]. The object
// VALUE is not self-identifying (it carries owner/asset/amount but not its own key), so the
// keys are recovered through atomic.SharedMemory.Indexed's lastKey: each page advances by
// one element and surfaces that element's key as lastKey. The walk is the documented
// shared-memory enumeration (the GetAtomicUTXOs pagination, adapted to recover keys rather
// than parse self-identifying values). Deterministic over committed shared memory; the
// caller sorts the result so the page order does not matter.
func enumerateSeamPending(sm atomic.SharedMemory, peerChainID ids.ID, max int) ([][]byte, error) {
	keys := make([][]byte, 0, 16)
	seen := make(map[ids.ID]struct{}, 16)
	var startKey []byte
	limit := 1 // first page: surface the head element's key as lastKey
	for len(keys) < max {
		_, _, lastKey, err := sm.Indexed(peerChainID, [][]byte{SeamPendingTrait}, SeamPendingTrait, startKey, limit)
		if err != nil {
			return nil, err
		}
		if len(lastKey) == 0 {
			break // empty partition
		}
		if startKey != nil && bytes.Equal(lastKey, startKey) {
			break // tail reached: the page returned only the (inclusive) start element
		}
		if len(lastKey) == 32 {
			var id ids.ID
			copy(id[:], lastKey)
			if _, dup := seen[id]; !dup {
				seen[id] = struct{}{}
				keys = append(keys, append([]byte(nil), lastKey...))
			}
		}
		startKey = lastKey
		limit = 2 // subsequent pages re-include the inclusive start element + one new key
	}
	return keys, nil
}

// seamIntentEntry is one OPEN escrow paired with its intent id (the seamintent: key), the
// unit the export drive iterates.
type seamIntentEntry struct {
	IntentID  ids.ID
	Owner     common.Address
	AssetIn   [32]byte
	Remaining uint64
}

// listSeamIntents streams the seamintent: index and returns every escrow in [status], in
// intentID order (the prefix iterator already yields keys — hence intent ids — in ascending
// order, so the result is the total sort the drive needs without an extra sort). Bounded by
// [max].
func listSeamIntents(db database.Iteratee, status uint8, max int) ([]seamIntentEntry, error) {
	it := db.NewIteratorWithPrefix([]byte(prefixSeamIntent))
	defer it.Release()
	out := make([]seamIntentEntry, 0, 16)
	for it.Next() && len(out) < max {
		key := it.Key()
		if len(key) != len(prefixSeamIntent)+32 {
			continue
		}
		rec, derr := decodeSeamIntent(it.Value())
		if derr != nil {
			return nil, derr
		}
		if rec.Status != status {
			continue
		}
		var e seamIntentEntry
		copy(e.IntentID[:], key[len(prefixSeamIntent):])
		e.Owner = rec.Owner
		e.AssetIn = rec.AssetIn
		e.Remaining = rec.Remaining
		out = append(out, e)
	}
	return out, it.Error()
}

// accountOutput is one realized-output (asset, amount) leg of a cross-chain account.
type accountOutput struct {
	asset  [32]byte
	amount uint64
}

// listAccountOutputs returns an account's available balances for every asset OTHER than
// [assetIn], in asset-id order — the realized swap OUTPUT the export drive settles. The
// balance: ledger keys by <user:16><asset:32>, so a prefix scan over <user> yields the
// account's per-asset rows in ascending asset order (the total sort the drive needs).
func listAccountOutputs(db database.Iteratee, acct userKey, assetIn [32]byte) ([]accountOutput, error) {
	prefix := make([]byte, len(prefixBalance)+len(acct))
	copy(prefix, prefixBalance)
	copy(prefix[len(prefixBalance):], acct[:])

	it := db.NewIteratorWithPrefix(prefix)
	defer it.Release()
	out := make([]accountOutput, 0, 4)
	for it.Next() {
		key := it.Key()
		if len(key) != len(prefix)+32 {
			continue
		}
		var asset [32]byte
		copy(asset[:], key[len(prefix):])
		if asset == assetIn {
			continue // leftover input, handled as the refund leg, not output
		}
		v := it.Value()
		if len(v) != 8 {
			return nil, fmt.Errorf("dchain: corrupt balance len=%d", len(v))
		}
		amount := binary.BigEndian.Uint64(v)
		if amount == 0 {
			continue
		}
		out = append(out, accountOutput{asset: asset, amount: amount})
	}
	return out, it.Error()
}

// accountFullyDrained reports whether an account holds NO available balance in any asset (no
// balance: row under its prefix). writeUint64 deletes a row at zero, so a present row is a
// non-zero balance; the absence of any row is a fully-settled account.
func accountFullyDrained(db database.Iteratee, acct userKey) (bool, error) {
	prefix := make([]byte, len(prefixBalance)+len(acct))
	copy(prefix, prefixBalance)
	copy(prefix[len(prefixBalance):], acct[:])
	it := db.NewIteratorWithPrefix(prefix)
	defer it.Release()
	if it.Next() {
		return false, it.Error()
	}
	return true, it.Error()
}
