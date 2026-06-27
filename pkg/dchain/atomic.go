// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic.go is the NATIVE co-located C<->D settlement seam: the leg that lets the
// in-luxd D-Chain (this VM) PRODUCE and CONSUME the same primary-network atomic
// shared-memory objects the 0x9999 precompile (precompile/dex) speaks. It is the
// D-side mirror of native_dchain_client.go:
//
//   - IMPORT (C->D): the precompile's SubmitSwapIntent locked the taker's tokenIn on
//     C and wrote a C->D atomic INTENT object keyed by intentID. This VM CONSUMES that
//     object (Get under the C chain, then a shared-memory Remove) and FUNDS the taker's
//     native D account with the recorded asset — the matcher then trades it natively.
//     D is funded ONLY by consuming a C->D object (no mint): the value was already
//     removed from the taker on C.
//   - EXPORT (D->C): once the native matcher has filled (settle.go), the taker's
//     realized proceeds / unfilled refund are EXPORTED back to C — debited from the D
//     ledger and written as a D->C atomic SETTLEMENT object the precompile's
//     ImportSettlement consumes EXACTLY ONCE. C is credited ONLY by consuming a D->C
//     object (no mint): the value was already debited on D.
//
// THE WIRE IS THE PRECOMPILE'S, byte-for-byte. encodeSeamObject / decodeSeamObject
// reproduce precompile/dex/native_wire.go's 69-byte object
//
//	rail(1) | owner(20) | asset(32) | amount(8) | spent(8)
//
// and DeriveIntentID reproduces its SHA-256 intent-id derivation, so an object this
// VM exports is consumable by ImportSettlement and an intent the precompile minted is
// importable here — ONE wire across precompile / proxy (chains/dexvm) / native. The
// proxy carried the SAME 69-byte value but behind its own txs codec; this is the
// native realization that needs no proxy and no external venue.
//
// CONSENSUS-SAFETY (why no extra root binding is needed, unlike the proxy). The proxy
// learned its fills from a LIVE external matcher and so had to fold the produced
// exports into its state root to catch a node whose relay diverged. This VM's matcher
// is IN-CONSENSUS and deterministic: the import credit is a pure function of the
// (rooted) TxImport body + the committed shared-memory object every validator reads
// identically; the export object is a pure function of the (rooted) TxExport body +
// the escrow record (itself written by the rooted import). The tx root already commits
// both, so honest validators produce byte-identical atomic operations and a divergent
// proposer is rejected on the tx root — there is no live, unrooted input to bind.

// seamRail is the cross-chain object's lane discriminator (wire byte 0), IDENTICAL to
// precompile/dex Rail. A swap fill/refund travels railSwap; an LP collect travels
// railLP. The native swap seam writes and consumes ONLY railSwap; the rail byte lets
// the precompile refuse a cross-rail object before it touches a pot (the H1 gate).
type seamRail uint8

const (
	railSwap seamRail = 0 // swap fill / refund lane (the native seam's only lane)
	railLP   seamRail = 1 // LP commit / collect lane (not produced here; named for the gate)
)

// seamObjectSize is the fixed shared-memory object width — IDENTICAL to
// precompile/dex exportedOutputSize9999 and chains/dexvm exportedOutputSize:
// rail(1) | owner(20) | asset(32) | amount(8) | spent(8) = 69 bytes.
const seamObjectSize = 1 + 20 + 32 + 8 + 8

// nativeIntentDomain scopes the intent-id derivation — IDENTICAL to precompile/dex
// nativeIntentDomain so the id this VM recomputes for an imported object matches the
// id the precompile minted it under.
const nativeIntentDomain = "lux.dex.native.intent.v1"

// Seam state schema (under the VM's init.DB namespace, committed in the block overlay
// alongside the ledger so the consume/credit/escrow land atomically with the block):
//
//	seamintent:<intentID:32> -> owner(20)|assetIn(32)|remaining(8)|status(1)
//	    the per-intent escrow written on IMPORT. It is BOTH the import replay guard
//	    (a second import of the same intentID finds it) AND the export binding (the
//	    full 20-byte owner the D->C object must name, the locked asset, and the
//	    remaining principal the same-asset refund leg is capped by — the D-side analog
//	    of the precompile's per-taker cap).
//	seamout:<outputID:32>    -> intentID(32)|asset(32)|amount(8)
//	    the per-export settlement record written on EXPORT, so a keeper/SDK can read
//	    which D->C object id to claim in the precompile's Phase-B hookData.
const (
	prefixSeamIntent = "seamintent:"
	prefixSeamOut    = "seamout:"
)

const (
	seamIntentOpen      uint8 = 0
	seamIntentReclaimed uint8 = 1
)

// seamIntentRecord is the decoded per-intent escrow.
type seamIntentRecord struct {
	Owner     common.Address // the taker's full 20-byte C address (the D->C object owner)
	AssetIn   [32]byte       // the locked input asset (the refund leg is capped in these units)
	Remaining uint64         // remaining locked principal not yet refunded
	Status    uint8          // seamIntentOpen / seamIntentReclaimed
}

var (
	// ErrSeamNoAtomicMemory: the seam needs cross-chain shared memory + a resolved C
	// chain id; a single-chain runtime cannot import/export (no mint, fail closed).
	ErrSeamNoAtomicMemory = errors.New("dchain: native seam requires cross-chain atomic memory + C-Chain id")
	// ErrSeamIntentReplay: a C->D intent id already imported (the escrow row exists).
	ErrSeamIntentReplay = errors.New("dchain: C->D intent already imported (replay)")
	// ErrSeamObjectMissing: no C->D object found in shared memory for the claimed id
	// (not flushed yet, or already consumed) — never credit an unbacked import.
	ErrSeamObjectMissing = errors.New("dchain: no C->D atomic object found for intent id")
	// ErrSeamObjectMalformed: the recorded object is not the canonical 69-byte width.
	ErrSeamObjectMalformed = errors.New("dchain: C->D atomic object malformed")
	// ErrSeamWrongRail: a C->D object whose rail is not railSwap reached the swap seam.
	ErrSeamWrongRail = errors.New("dchain: C->D atomic object is not a swap-rail object")
	// ErrSeamBadAmount: a zero (or unrepresentable) object amount.
	ErrSeamBadAmount = errors.New("dchain: C->D atomic object amount out of range")
	// ErrSeamNoIntent: an export names no open intent escrow for the recorded owner.
	ErrSeamNoIntent = errors.New("dchain: D->C export names no open intent escrow")
	// ErrSeamExportExceedsIntent: a same-asset (refund) export exceeds the intent's
	// remaining locked principal — the D-side per-taker cap (the export analog of the
	// precompile's ErrSettleExceedsIntent), so an over-refund can never draw on other
	// takers' pooled input.
	ErrSeamExportExceedsIntent = errors.New("dchain: D->C refund export exceeds the intent's remaining locked principal")
)

// encodeSeamObject serializes a cross-chain value object as the shared-memory value,
// BYTE-IDENTICAL with precompile/dex encodeAtomicObjectSpent and chains/dexvm
// encodeExportedOutput: rail(1) | owner(20) | asset(32) | amount(8) | spent(8). owner
// is the full 20-byte address; asset is the full injective AssetID (native = all-zero);
// amount is integer asset units; spent is the matched INPUT on a swap proceeds leg (0
// on a refund / import object).
func encodeSeamObject(rail seamRail, owner common.Address, asset [32]byte, amount, spent uint64) []byte {
	v := make([]byte, seamObjectSize)
	v[0] = byte(rail)
	copy(v[1:21], owner[:])
	copy(v[21:53], asset[:])
	binary.BigEndian.PutUint64(v[53:61], amount)
	binary.BigEndian.PutUint64(v[61:69], spent)
	return v
}

// decodeSeamObject is the inverse, IDENTICAL to precompile/dex decodeAtomicObject:
// ok=false for any value that is not EXACTLY the canonical width, so a corrupt record
// is never reinterpreted into a credit. The consumer binds to THIS recorded value.
func decodeSeamObject(v []byte) (rail seamRail, owner common.Address, asset [32]byte, amount, spent uint64, ok bool) {
	if len(v) != seamObjectSize {
		return 0, common.Address{}, [32]byte{}, 0, 0, false
	}
	rail = seamRail(v[0])
	copy(owner[:], v[1:21])
	copy(asset[:], v[21:53])
	amount = binary.BigEndian.Uint64(v[53:61])
	spent = binary.BigEndian.Uint64(v[61:69])
	return rail, owner, asset, amount, spent, true
}

// DeriveIntentID computes the deterministic id of a C->D atomic intent object — the
// shared-memory key the precompile PUTs it under (and that this VM's import consumes).
// It is BYTE-IDENTICAL to precompile/dex DeriveIntentID:
//
//	SHA-256( domain | networkID | cChainID | dChainID |
//	         account | assetIn | amountIn | marketID | nonce )
//
// Every component is fixed width. The native VM does not re-derive ids on the import
// path (it consumes by the key carried in the TxImport body); this is exported so the
// keeper / SDK / tests derive the SAME id the precompile minted, off-chain.
func DeriveIntentID(
	networkID uint32,
	cChainID, dChainID ids.ID,
	account common.Address,
	assetIn [32]byte,
	amountIn uint64,
	marketID [32]byte,
	nonce uint64,
) ids.ID {
	h := sha256.New()
	h.Write([]byte(nativeIntentDomain))
	var u4 [4]byte
	binary.BigEndian.PutUint32(u4[:], networkID)
	h.Write(u4[:])
	h.Write(cChainID[:])
	h.Write(dChainID[:])
	h.Write(account[:])
	h.Write(assetIn[:])
	var u8 [8]byte
	binary.BigEndian.PutUint64(u8[:], amountIn)
	h.Write(u8[:])
	h.Write(marketID[:])
	binary.BigEndian.PutUint64(u8[:], nonce)
	h.Write(u8[:])
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return ids.ID(out)
}

// deriveSeamOutputID computes the deterministic D->C object key from the exporting
// tx id and the leg index — SHA-256 over txID||index, the SAME UTXO-id derivation the
// proxy (chains/dexvm deriveUTXOID) and platformvm use, so the key is globally unique
// and the keeper recomputes it from the export tx.
func deriveSeamOutputID(exportTxID ids.ID, index uint32) ids.ID {
	var buf [36]byte
	copy(buf[0:32], exportTxID[:])
	binary.BigEndian.PutUint32(buf[32:36], index)
	return ids.ID(sha256.Sum256(buf[:]))
}

// crossChainAccount folds a 20-byte cross-chain C address to this VM's 16-byte ledger
// account — the SAME Account16 fold a signed D order uses, with the secp256k1 scheme
// (a C-Chain caller is an EVM/secp address). So an imported intent funds the EXACT
// account the taker's own signed D orders draw from, and there is ONE address->account
// derivation across the deposit, order, and seam paths.
func crossChainAccount(owner common.Address) userKey {
	return Account16(lx.AuthSecp256k1, owner)
}

// --- TxImport / TxExport body codecs ---------------------------------------------

// EncodeSeamImportBody builds a TxImport body: intentID[32] — the shared-memory key of
// the C->D object to consume. Exported for the keeper / SDK / tests.
func EncodeSeamImportBody(intentID ids.ID) []byte {
	b := make([]byte, seamImportBodySize)
	copy(b, intentID[:])
	return b
}

// EncodeSeamExportBody builds a TxExport body: intentID[32] | asset[32] | amount[8] |
// spent[8]. asset/amount are the D->C object's value; spent is the matched input
// witness on a proceeds leg (0 on a refund). Exported for the keeper / SDK / tests.
func EncodeSeamExportBody(intentID ids.ID, asset [32]byte, amount, spent uint64) []byte {
	b := make([]byte, seamExportBodySize)
	copy(b[0:32], intentID[:])
	copy(b[32:64], asset[:])
	binary.BigEndian.PutUint64(b[64:72], amount)
	binary.BigEndian.PutUint64(b[72:80], spent)
	return b
}

// decodeSeamExportBody parses a TxExport body. A short body is a malformed tx (block
// abort) — the fixed bodySize gate already rejects undersized frames at parse, so this
// only fails on internal misuse.
func decodeSeamExportBody(body []byte) (intentID ids.ID, asset [32]byte, amount, spent uint64, err error) {
	if len(body) < seamExportBodySize {
		return ids.Empty, [32]byte{}, 0, 0, fmt.Errorf("dchain: seam export body too short: %d", len(body))
	}
	copy(intentID[:], body[0:32])
	copy(asset[:], body[32:64])
	amount = binary.BigEndian.Uint64(body[64:72])
	spent = binary.BigEndian.Uint64(body[72:80])
	return intentID, asset, amount, spent, nil
}

// --- per-intent escrow (seamintent:) ---------------------------------------------

func seamIntentKey(intentID ids.ID) []byte {
	k := make([]byte, len(prefixSeamIntent)+32)
	copy(k, prefixSeamIntent)
	copy(k[len(prefixSeamIntent):], intentID[:])
	return k
}

func putSeamIntent(db database.KeyValueWriter, intentID ids.ID, r seamIntentRecord) error {
	v := make([]byte, 20+32+8+1)
	copy(v[0:20], r.Owner[:])
	copy(v[20:52], r.AssetIn[:])
	binary.BigEndian.PutUint64(v[52:60], r.Remaining)
	v[60] = r.Status
	return db.Put(seamIntentKey(intentID), v)
}

func getSeamIntent(db database.KeyValueReader, intentID ids.ID) (seamIntentRecord, bool, error) {
	v, err := db.Get(seamIntentKey(intentID))
	if err == database.ErrNotFound {
		return seamIntentRecord{}, false, nil
	}
	if err != nil {
		return seamIntentRecord{}, false, err
	}
	if len(v) != 20+32+8+1 {
		return seamIntentRecord{}, false, fmt.Errorf("dchain: corrupt seam intent len=%d", len(v))
	}
	var r seamIntentRecord
	copy(r.Owner[:], v[0:20])
	copy(r.AssetIn[:], v[20:52])
	r.Remaining = binary.BigEndian.Uint64(v[52:60])
	r.Status = v[60]
	return r, true, nil
}

// --- per-export settlement record (seamout:) -------------------------------------

func seamOutKey(outputID ids.ID) []byte {
	k := make([]byte, len(prefixSeamOut)+32)
	copy(k, prefixSeamOut)
	copy(k[len(prefixSeamOut):], outputID[:])
	return k
}

func putSeamOut(db database.KeyValueWriter, outputID, intentID ids.ID, asset [32]byte, amount uint64) error {
	v := make([]byte, 32+32+8)
	copy(v[0:32], intentID[:])
	copy(v[32:64], asset[:])
	binary.BigEndian.PutUint64(v[64:72], amount)
	return db.Put(seamOutKey(outputID), v)
}

// --- atomic request accumulator (the block's cross-chain mutation set) -----------
//
// atomicRequests collects the shared-memory removes (import) and puts (export) a
// block produces, applied atomically with the state batch at Accept (sm.Apply). It is
// the SAME platformvm acceptor pattern the proxy (chains/dexvm atomic.go) uses.

type atomicRequests struct {
	reqs map[ids.ID]*atomic.Requests
}

func newAtomicRequests() *atomicRequests {
	return &atomicRequests{reqs: make(map[ids.ID]*atomic.Requests)}
}

func (a *atomicRequests) empty() bool { return a == nil || len(a.reqs) == 0 }

func (a *atomicRequests) forChain(chainID ids.ID) *atomic.Requests {
	r, ok := a.reqs[chainID]
	if !ok {
		r = &atomic.Requests{}
		a.reqs[chainID] = r
	}
	return r
}

// seamSharedMemory returns this chain's atomic shared-memory handle (from the runtime
// the consensus engine wired at Initialize), or nil when no cross-chain memory was
// wired (single-chain dev / unit tests with no seam).
func (vm *VM) seamSharedMemory() atomic.SharedMemory {
	if vm.runtime == nil {
		return nil
	}
	return vm.runtime.GetSharedMemory()
}

// executeImport CONSUMES a C->D atomic intent object and FUNDS the taker's native D
// account with the recorded asset. It mirrors precompile SubmitSwapIntent's other
// side and the proxy executeImport, but credits the native ledger so the in-consensus
// matcher trades the funds:
//
//  1. replay guard: the intent escrow must not already exist (one import per intent).
//  2. read the RECORDED C->D object (Get under the C chain); missing => reject (never
//     credit an unbacked import). A nil shared memory / unresolved C chain => reject.
//  3. BIND to the recorded value: rail==railSwap, amount>0; the credited asset/owner
//     are the RECORDED object's, never declared by the tx (the tx carries only the id).
//  4. CREDIT available[crossChainAccount(owner), asset] += amount (fund the D account).
//  5. RECORD the per-intent escrow (owner/assetIn/remaining) for the export binding.
//  6. ACCUMULATE the shared-memory Remove under the C chain (applied at Accept), so the
//     object is consumed exactly once.
//
// It returns ok=false (with a nil error) for a deterministic per-tx REJECT — a missing
// object, wrong rail, replay, or unwired seam — so a not-yet-flushed intent can be
// retried in a later block rather than wedging the chain. A db fault returns an error
// (block abort). The block records a successful import under seen: (idempotent replay)
// but NOT a reject (retryable).
func (vm *VM) executeImport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, intentID ids.ID) (credited uint64, ok bool, err error) {
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return 0, false, nil // seam not wired: deterministic reject, no mint
	}

	// (1) replay guard — the escrow row is the durable consumed marker.
	if _, exists, gerr := getSeamIntent(db, intentID); gerr != nil {
		return 0, false, gerr
	} else if exists {
		return 0, false, nil // already imported (ErrSeamIntentReplay condition); reject
	}

	// (2) read the RECORDED object under the C chain partition.
	vals, gerr := sm.Get(vm.cChainID, [][]byte{intentID[:]})
	if gerr != nil {
		return 0, false, fmt.Errorf("dchain: seam import read: %w", gerr)
	}
	if len(vals) != 1 || len(vals[0]) == 0 {
		return 0, false, nil // not present (not flushed / already consumed): retryable reject
	}
	rail, owner, asset, amount, _, decOK := decodeSeamObject(vals[0])
	if !decOK {
		return 0, false, nil // malformed: reject (never reinterpret into a credit)
	}

	// (3) BIND: swap rail only, non-zero amount.
	if rail != railSwap || amount == 0 {
		return 0, false, nil
	}

	// (4) CREDIT the native D account (no mint — the value was removed from the taker
	// on C and is consumed here exactly once).
	acct := crossChainAccount(owner)
	if cerr := creditAvailable(db, acct, asset, amount); cerr != nil {
		return 0, false, cerr
	}

	// (5) RECORD the escrow (owner/assetIn/remaining) for the export binding + cap.
	if perr := putSeamIntent(db, intentID, seamIntentRecord{
		Owner:     owner,
		AssetIn:   asset,
		Remaining: amount,
		Status:    seamIntentOpen,
	}); perr != nil {
		return 0, false, perr
	}

	// (6) ACCUMULATE the Remove under the C chain (consumed once, applied at Accept).
	req := ar.forChain(vm.cChainID)
	req.RemoveRequests = append(req.RemoveRequests, intentID[:])
	return amount, true, nil
}

// executeExport DEBITS the taker's native D balance and writes a D->C settlement
// object the precompile's ImportSettlement consumes. It is driven AFTER the native
// matcher has settled (the proceeds / refund live in the taker's available balance),
// and binds the object to the intent so Phase B's checks pass:
//
//  1. load the intent escrow (must be open) — the export draws against THIS intent;
//     the D->C object's owner is the escrow's RECORDED 20-byte owner, never the tx, so
//     an export can never redirect value to a different account.
//  2. PER-TAKER CAP: a same-asset (refund) export (asset == escrow.AssetIn) is bounded
//     by the intent's remaining locked principal and decrements it — the D-side analog
//     of the precompile's per-taker cap. A proceeds export (different asset) is bounded
//     only by the available balance the matcher actually credited.
//  3. DEBIT available[crossChainAccount(owner), asset] -= amount (refuses an over-debit:
//     the export is backed by realized D value, so it can never credit more on C than D
//     matched — conservation).
//  4. ENCODE the D->C object (rail=railSwap, owner, asset, amount, spent) and ACCUMULATE
//     a Put under the C chain keyed by the deterministic outputID.
//  5. RECORD the settlement (seamout:) so a keeper reads the outputID to claim.
//
// spent is the matched INPUT that produced a proceeds output (0 on a refund) — the
// witness the precompile's taker-authenticated MEV floor checks. Returns ok=false for a
// deterministic per-tx reject (no intent, over-cap, insufficient balance, unwired seam).
func (vm *VM) executeExport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, exportTxID, intentID ids.ID, asset [32]byte, amount, spent uint64) (outputID ids.ID, ok bool, err error) {
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return ids.Empty, false, nil // seam not wired: reject
	}
	if amount == 0 {
		return ids.Empty, false, nil
	}

	// (1) the intent escrow must exist and be open.
	rec, exists, gerr := getSeamIntent(db, intentID)
	if gerr != nil {
		return ids.Empty, false, gerr
	}
	if !exists || rec.Status != seamIntentOpen {
		return ids.Empty, false, nil // ErrSeamNoIntent condition
	}

	// (2) PER-TAKER CAP on the same-asset refund leg (the export analog of the
	// precompile's ImportSettlement per-taker cap). A proceeds leg (different asset) is
	// not principal-capped — output units legitimately differ from input units.
	refund := asset == rec.AssetIn
	if refund && amount > rec.Remaining {
		return ids.Empty, false, nil // ErrSeamExportExceedsIntent condition
	}

	// (3) DEBIT the realized D balance (refuses over-debit: conservation — C can never be
	// credited more than D matched). The matcher credited exactly the proceeds/refund.
	acct := crossChainAccount(rec.Owner)
	if derr := debitAvailable(db, acct, asset, amount); derr != nil {
		if errors.Is(derr, ErrInsufficientAvailable) {
			return ids.Empty, false, nil // not enough realized value: reject
		}
		return ids.Empty, false, derr
	}

	// (4) ENCODE + ACCUMULATE the D->C Put under the C chain, keyed by outputID. owner
	// is the escrow's RECORDED owner so the precompile binds recOwner == the taker.
	outputID = deriveSeamOutputID(exportTxID, 0)
	obj := encodeSeamObject(railSwap, rec.Owner, asset, amount, spent)
	req := ar.forChain(vm.cChainID)
	req.PutRequests = append(req.PutRequests, &atomic.Element{
		Key:    outputID[:],
		Value:  obj,
		Traits: [][]byte{rec.Owner[:]}, // index by recipient (lux.Addressable)
	})

	// (5) decrement the remaining principal on a refund + RECORD the settlement.
	if refund {
		rec.Remaining -= amount
		if perr := putSeamIntent(db, intentID, rec); perr != nil {
			return ids.Empty, false, perr
		}
	}
	if perr := putSeamOut(db, outputID, intentID, asset, amount); perr != nil {
		return ids.Empty, false, perr
	}
	return outputID, true, nil
}

// commitSeamAtomic commits the block's accumulated cross-chain operations atomically
// with the state overlay — the single accept-time commit point (the platformvm /
// proxy commitAtomic pattern). With no shared memory or no atomic ops it falls back to
// the plain overlay commit. The overlay's CommitBatch is handed to sm.Apply so the
// shared-memory removes/puts and the chain state land in one atomic write; the overlay
// mem is then cleared (Abort) since its contents are already persisted via the batch.
func (vm *VM) commitSeamAtomic(overlay versionDB, ar *atomicRequests) error {
	if ar.empty() {
		return overlay.Commit()
	}
	sm := vm.seamSharedMemory()
	if sm == nil {
		// Atomic ops were accumulated but no shared memory is wired — this cannot
		// happen on the seam path (import/export reject when sm is nil), so a non-empty
		// ar with nil sm is a programming error. Commit state only and surface nothing
		// cross-chain rather than dropping the block.
		return overlay.Commit()
	}
	batch, err := overlay.CommitBatch()
	if err != nil {
		overlay.Abort()
		return err
	}
	if err := sm.Apply(ar.reqs, batch); err != nil {
		overlay.Abort()
		return fmt.Errorf("dchain: seam atomic apply: %w", err)
	}
	overlay.Abort()
	return nil
}

// versionDB is the minimal commit surface commitSeamAtomic needs from
// *versiondb.Database, kept as an interface so the seam commit is testable in isolation.
type versionDB interface {
	Commit() error
	CommitBatch() (database.Batch, error)
	Abort()
}
