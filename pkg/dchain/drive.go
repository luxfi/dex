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
// COMPANION (coordinated cutover): the C-side flush MUST add this trait to the swap
// intent's Element.Traits — precompile/dex native_staging.go collectRange, where the
// PutRequests for the C->D object are built (today Traits=[owner] only). That is the ONE
// line that makes a production 0x9999 intent discoverable here; it is pinned to THIS
// constant (the same wire-golden discipline that pins encodeSeamObject). The native VM
// e2e models that corrected flush (writeCToDIntent tags the object with SeamPendingTrait).
var SeamPendingTrait = func() []byte {
	d := sha256.Sum256([]byte("lux.dex.native.intent.pending.v1"))
	return d[:]
}()

// maxSeamDrivePerBlock bounds how many import (and, separately, export) legs one block's
// drive emits, so a large backlog cannot produce an unbounded block. The excess is picked
// up by the next block's drive (the enumeration is a pure function of the still-pending
// committed state, so nothing is lost — just deferred). Sized generously for the in-flight
// cross-chain swap count a single block realistically faces.
const maxSeamDrivePerBlock = 256

// driveSeamImports enumerates the committed C->D swap-intent objects in shared memory and
// returns a deterministic, intentID-sorted list of TxImport txs for those NOT YET imported
// (no committed seamintent: escrow). It is the autonomous replacement for a keeper's
// "watch IntentSubmitted -> inject ImportTx" loop, run inside BuildBlock.
//
// Pure function of committed state: the pending set comes from shared memory (the C chain
// accepted the staging block) and the already-imported filter comes from the committed
// escrow index. The double-guard against re-import — the committed-escrow filter here AND
// executeImport's own replay-reject — means a re-enumerated intent (object still present
// because its consuming Remove has not yet been accepted) is imported at most once.
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
	// so every proposer on the same committed partition emits the same import sequence.
	sort.Slice(pending, func(i, j int) bool {
		return bytes.Compare(pending[i][:], pending[j][:]) < 0
	})

	out := make([]*Tx, 0, len(pending))
	for _, intentID := range pending {
		tx, err := NewTx(TxImport, EncodeSeamImportBody(intentID))
		if err != nil {
			return nil, fmt.Errorf("dchain: build TxImport: %w", err)
		}
		out = append(out, tx)
	}
	return out, nil
}

// driveSeamExports enumerates the OPEN intent escrows whose owner holds realized OUTPUT in
// committed state and returns deterministic TxExport txs that settle the realized proceeds
// (and any leftover input refund) back to C as D->C objects the 0x9999 ImportSettlement
// consumes. It is the autonomous replacement for a keeper's "watch the D fill -> inject the
// settlement export" loop, run inside BuildBlock AFTER the matcher has produced proceeds in
// a prior accepted block.
//
// THE OUTPUT GATE (why this never refunds a still-pending swap, and satisfies
// no-export-before-import): an intent is exported ONLY when its owner's cross-chain account
// holds a realized OUTPUT — a non-AssetIn balance > 0. Right after an import the account
// holds only the imported AssetIn (no output) so nothing is exported; only once a signed
// taker order has CROSSED (producing a different asset) does the swap's proceeds become
// exportable. An intent that was never imported has no escrow, is never enumerated, and is
// never exported.
//
// ATTRIBUTION + CONSERVATION: the export's recipient/owner is the escrow's RECORDED 20-byte
// owner (executeExport binds it), so value can only return to the rightful taker; the
// debit draws the REALIZED ledger balance (executeExport refuses an over-debit) and a
// same-asset refund is capped by the intent's remaining principal — so C can never be
// credited more than D matched. The drive emits the refund leg BEFORE the proceeds leg so
// the per-taker cap is applied against the open escrow, and execute then CLOSES the escrow
// once the account is fully drained (closeSeamIntentIfDrained), making each swap one-shot.
//
// SCOPE: the realistic cross-chain swap is an IOC taker (TxSubmit) against a single market,
// which resolves in one block to one output asset (+ optional leftover). The drive settles
// exactly that. A cross-chain account that mixes direct (non-imported) D activity, or a
// multi-hop swap with several output assets, is outside this first-cut drive and noted as a
// boundary (the spent witness is attributed to the single output; extra outputs remain the
// owner's D balance rather than being mis-bound to this intent's recorded MEV floor).
func (vm *VM) driveSeamExports() ([]*Tx, error) {
	if !vm.autoDriveSeam {
		return nil, nil
	}
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return nil, nil // seam not wired: an export cannot be written (no D->C object)
	}

	escrows, err := listOpenSeamIntents(vm.db, maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: seam export enumerate: %w", err)
	}

	out := make([]*Tx, 0, len(escrows))
	for _, e := range escrows {
		acct := crossChainAccount(e.Owner)

		// The realized OUTPUT (non-AssetIn balances) and the leftover INPUT (AssetIn),
		// both read from committed state. No output => the swap has not produced proceeds
		// yet => do not export (no premature refund; no-export-before-trade).
		outputs, oerr := listAccountOutputs(vm.db, acct, e.AssetIn)
		if oerr != nil {
			return nil, oerr
		}
		if len(outputs) == 0 {
			continue
		}
		leftoverIn, lerr := getAvailable(vm.db, acct, e.AssetIn)
		if lerr != nil {
			return nil, lerr
		}

		// The spent witness the precompile's taker-authenticated MEV floor reads: the
		// AssetIn actually CONSUMED to produce the output = original locked principal minus
		// the leftover input. Computed from committed state, so it is stable regardless of
		// the in-block leg order.
		spent := uint64(0)
		if e.Remaining > leftoverIn {
			spent = e.Remaining - leftoverIn
		}

		// REFUND leg FIRST (same-asset, per-taker-capped, decrements Remaining) so it runs
		// against the still-open escrow.
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

		// PROCEEDS legs (sorted by asset id via the committed key order); the draining one
		// triggers the escrow close in execute.
		for _, o := range outputs {
			tx, terr := NewTx(TxExport, EncodeSeamExportBody(e.IntentID, o.asset, o.amount, spent))
			if terr != nil {
				return nil, fmt.Errorf("dchain: build TxExport proceeds: %w", terr)
			}
			out = append(out, tx)
			if len(out) >= maxSeamDrivePerBlock {
				return out, nil
			}
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
	// Any open escrow with realized output?
	escrows, eerr := listOpenSeamIntents(vm.db, maxSeamDrivePerBlock)
	if eerr != nil {
		return false
	}
	for _, e := range escrows {
		outputs, oerr := listAccountOutputs(vm.db, crossChainAccount(e.Owner), e.AssetIn)
		if oerr == nil && len(outputs) > 0 {
			return true
		}
	}
	return false
}

// closeSeamIntentIfDrained marks an OPEN intent escrow seamIntentReclaimed once its owner's
// cross-chain account has been FULLY drained (no remaining ledger balance), so a settled
// intent is not re-enumerated by a later export drive and the owner can import a fresh
// intent. Called from execute after a successful TxExport; order-independent (it fires on
// whichever export — proceeds or refund — empties the account), so it correctly closes a
// full-fill (one proceeds leg) and a partial-fill (proceeds + refund) alike. A still-funded
// account (more output not yet exported in this block) is left open for a subsequent leg.
func (vm *VM) closeSeamIntentIfDrained(db database.Database, intentID ids.ID) error {
	rec, exists, err := getSeamIntent(db, intentID)
	if err != nil {
		return err
	}
	if !exists || rec.Status != seamIntentOpen {
		return nil
	}
	drained, err := accountFullyDrained(db, crossChainAccount(rec.Owner))
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
	Owner     [20]byte
	AssetIn   [32]byte
	Remaining uint64
}

// listOpenSeamIntents streams the seamintent: index and returns every OPEN escrow, in
// intentID order (the prefix iterator already yields keys — hence intent ids — in ascending
// order, so the result is the total sort the drive needs without an extra sort). Settled
// (reclaimed) escrows are skipped. Bounded by [max].
func listOpenSeamIntents(db database.Iteratee, max int) ([]seamIntentEntry, error) {
	it := db.NewIteratorWithPrefix([]byte(prefixSeamIntent))
	defer it.Release()
	out := make([]seamIntentEntry, 0, 16)
	for it.Next() && len(out) < max {
		key := it.Key()
		if len(key) != len(prefixSeamIntent)+32 {
			continue
		}
		v := it.Value()
		if len(v) != 20+32+8+1 {
			return nil, fmt.Errorf("dchain: corrupt seam intent len=%d", len(v))
		}
		if v[60] != seamIntentOpen {
			continue // settled: not exportable
		}
		var e seamIntentEntry
		copy(e.IntentID[:], key[len(prefixSeamIntent):])
		copy(e.Owner[:], v[0:20])
		copy(e.AssetIn[:], v[20:52])
		e.Remaining = binary.BigEndian.Uint64(v[52:60])
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
