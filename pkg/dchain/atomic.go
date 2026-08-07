// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/crypto/hash"
	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic.go is the NATIVE co-located C<->D settlement seam: the leg that lets the
// in-luxd D-Chain (this VM) PRODUCE and CONSUME the same primary-network atomic
// shared-memory objects the 0x9999 precompile (precompile/dex) speaks. It is the
// D-side mirror of native_dchain_client.go:
//
//   - IMPORT (C->D): the precompile's SubmitSwapOrder locked the taker's tokenIn on
//     C and wrote a C->D atomic ORDER object keyed by orderID. This VM CONSUMES that
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
// reproduce precompile/dex/native_wire.go's 69-byte VALUE object
//
//	rail(1) | owner(20) | asset(32) | amount(8) | spent(8)
//
// and DeriveOrderID reproduces its SHA-256 order-id derivation, so an object this
// VM exports is consumable by ImportSettlement and an order the precompile minted is
// importable here — ONE wire across precompile / proxy (chains/dexvm) / native. The
// proxy carried the SAME 69-byte value but behind its own txs codec; this is the
// native realization that needs no proxy and no external venue.
//
// A C->D SWAP ORDER CARRIES MORE THAN VALUE. Those 69 bytes are a value: "this much
// of this asset belongs to this owner". A value says what moves; it does not say what
// to DO with it. That is why this seam produced nothing: executeImport consumed the
// object, credited the taker's D account, and stopped, because it had been told
// nothing else. Nothing placed an order, so nothing crossed, so no proceeds existed,
// so driveSeamExports (which only fires on realized output) never fired, so Phase B
// had no object and the settle reverted. The seam was unreachable BY CONSTRUCTION.
//
// So a C->D swap order appends the OPERATION the taker authorized on C:
//
//	market(32) | side(1) | limitPrice(8) | size(8)   = 49 bytes
//
// giving a 118-byte order object. WIDTH IS THE DISCRIMINATOR, and there are exactly
// two canonical widths:
//
//	 69 -> a VALUE object: what this VM EXPORTS (D->C settlements). Instructs nothing.
//	118 -> a C->D SWAP ORDER: value + operation. What this VM IMPORTS.
//
// The value head is byte-for-byte unchanged, so the golden vectors pinning it stay
// valid and every existing consumer of those bytes is untouched. Anything that is
// neither width is refused rather than reinterpreted — in particular a 69-byte
// op-less object is NOT importable, because importing it would mean inventing the
// taker's market/side/limit, i.e. executing an order they never authorized.
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

// seamObjectSize is the fixed shared-memory VALUE object width — IDENTICAL to
// precompile/dex exportedOutputSize9999 and chains/dexvm exportedOutputSize:
// rail(1) | owner(20) | asset(32) | amount(8) | spent(8) = 69 bytes.
const seamObjectSize = 1 + 20 + 32 + 8 + 8

// seamSideBuy / seamSideSell are the CLOB sides carried in the operation segment.
// They are the matcher's OWN constants (sideBuy / sideSell in settle.go, zapwire's
// side byte), so the byte read out of a C->D object is the byte the order book
// consumes — no translation table, nothing to keep in sync.
const (
	seamSideBuy  uint8 = sideBuy
	seamSideSell uint8 = sideSell
)

// seamOpSize is the fixed OPERATION segment width: market(32) | side(1) |
// limitPrice(8) | size(8) = 49 bytes. IDENTICAL to precompile/dex seamOpSize.
const seamOpSize = 32 + 1 + 8 + 8

// seamOrderObjectSize is the fixed C->D SWAP ORDER object width: the 69-byte value
// head plus the 49-byte operation tail = 118 bytes. IDENTICAL to precompile/dex
// orderObjectSize9999.
const seamOrderObjectSize = seamObjectSize + seamOpSize

// seamOp is the CLOB operation a C->D swap order instructs — every field a value
// the taker fixed on C when they signed the swap, so this VM places their order
// without inventing any part of it.
//
// LimitPrice is REQUIRED (non-zero): every seam order is a LIMIT IOC order. A
// cross-chain taker cannot re-price mid-flight and their order executes in a block
// they do not control, so an unbounded market order across the seam is exactly the
// case the spent-witness MEV floor exists to refuse — and at a zero limit that floor
// is vacuous. C refuses a zero limit at submission; this VM refuses it at import. The
// same rule at both ends, so neither has to trust the other for it.
//
// isMarket and "which end the limit bounds" are NOT carried: every seam order is a
// limit order, and a BUY's limit is its ceiling while a SELL's is its floor. Both
// follow from the fields above, so each fact is declared exactly once.
type seamOp struct {
	Market     [32]byte
	Side       uint8
	LimitPrice uint64
	Size       uint64
}

// valid reports whether an operation is executable at all: a non-zero size, a
// non-zero limit, and a side the matcher knows. An invalid operation is refused at
// import — the order's principal stays on C, reclaimable at its deadline, rather
// than being credited on D against an order that can never run.
func (op seamOp) valid() bool {
	return op.Size != 0 && op.LimitPrice != 0 && op.LimitPrice <= maxPriceUnits &&
		(op.Side == seamSideBuy || op.Side == seamSideSell)
}

// nativeOrderDomain scopes the order-id derivation — IDENTICAL to precompile/dex
// nativeOrderDomain so the id this VM recomputes for an imported object matches the
// id the precompile minted it under.
//
// The literal is the hash input, not a label, so it does not track this identifier:
// changing a byte of it moves every order id the seam has ever derived.
const nativeOrderDomain = "lux.dex.native.intent.v1"

// Seam state schema (under the VM's init.DB namespace, committed in the block overlay
// alongside the ledger so the consume/credit/escrow land atomically with the block):
//
//	seamintent:<orderID:32> ->
//	        owner(20)|assetIn(32)|remaining(8)|status(1)|market(32)|side(1)|limit(8)|size(8)
//	    the per-order escrow written on IMPORT. It is the order's WHOLE life in one
//	    row: the import replay guard (a second import of the same orderID finds it),
//	    the export binding (the full 20-byte owner the D->C object must name, the
//	    locked asset, and the remaining principal the same-asset refund leg is capped
//	    by — the D-side analog of the precompile's per-taker cap), the OPERATION the
//	    seam submit runs, and the STATUS that sequences the three legs.
//
//	    STATUS IS THE STATE MACHINE, and it replaced a guess. The export drive used to
//	    infer "has this swap traded yet?" from the shape of the taker's balances — it
//	    exported only when a non-AssetIn balance appeared. That inference is wrong in
//	    both directions: an order that filled nothing leaves only AssetIn and would
//	    NEVER export, stranding the principal on D while C's deadline reclaim refunds
//	    it (a mint); and any unrelated asset in the account would trigger an export
//	    that is not this order's. Recording the fact instead of inferring it makes
//	    each leg fire exactly once:
//
//	        open -> submitted -> reclaimed
//	         ^         ^            ^
//	         |         |            +-- the seam account is drained; nothing more to export
//	         |         +-- the taker's order has RUN (filled, partially filled, or not
//	         |             at all); whatever is in the account now is the final answer
//	         +-- value imported, order not yet placed
//	seamout:<outputID:32>    -> orderID(32)|asset(32)|amount(8)
//	    the per-export settlement record written on EXPORT, so a keeper/SDK can read
//	    which D->C object id to claim in the precompile's Phase-B hookData.
//
// These are on-disk key bytes and do not track their identifier names: renaming
// "seamintent:" would orphan every escrow row already written.
const (
	prefixSeamOrder = "seamintent:"
	prefixSeamOut   = "seamout:"
)

const (
	seamOrderOpen      uint8 = 0
	seamOrderReclaimed uint8 = 1
	// seamOrderSubmitted: the taker's order has RUN. Whatever the seam account holds
	// from here on is the swap's final answer — proceeds, leftover input, or both —
	// and the export drive settles ALL of it back to C. This is the state that makes
	// a zero-fill swap exportable, closing the strand-then-reclaim mint.
	seamOrderSubmitted uint8 = 2
)

// seamOrderRecordSize is the fixed escrow row width:
// owner(20)|assetIn(32)|remaining(8)|status(1)|market(32)|side(1)|limit(8)|size(8).
const seamOrderRecordSize = 20 + 32 + 8 + 1 + 32 + 1 + 8 + 8

// seamOrderRecord is the decoded per-order escrow.
type seamOrderRecord struct {
	Owner     common.Address // the taker's full 20-byte C address (the D->C object owner)
	AssetIn   [32]byte       // the locked input asset (the refund leg is capped in these units)
	Remaining uint64         // remaining locked principal not yet refunded
	Status    uint8          // seamOrderOpen / seamOrderSubmitted / seamOrderReclaimed
	Op        seamOp         // the CLOB operation the taker authorized on C
}

var (
	// ErrSeamNoAtomicMemory: the seam needs cross-chain shared memory + a resolved C
	// chain id; a single-chain runtime cannot import/export (no mint, fail closed).
	ErrSeamNoAtomicMemory = errors.New("dchain: native seam requires cross-chain atomic memory + C-Chain id")
	// ErrSeamOrderReplay: a C->D order id already imported (the escrow row exists).
	ErrSeamOrderReplay = errors.New("dchain: C->D order already imported (replay)")
	// ErrSeamObjectMissing: no C->D object found in shared memory for the claimed id
	// (not flushed yet, or already consumed) — never credit an unbacked import.
	ErrSeamObjectMissing = errors.New("dchain: no C->D atomic object found for order id")
	// ErrSeamObjectMalformed: the recorded object is not the canonical 69-byte width.
	ErrSeamObjectMalformed = errors.New("dchain: C->D atomic object malformed")
	// ErrSeamWrongRail: a C->D object whose rail is not railSwap reached the swap seam.
	ErrSeamWrongRail = errors.New("dchain: C->D atomic object is not a swap-rail object")
	// ErrSeamBadAmount: a zero (or unrepresentable) object amount.
	ErrSeamBadAmount = errors.New("dchain: C->D atomic object amount out of range")
	// ErrSeamNoOrder: an export names no open order escrow for the recorded owner.
	ErrSeamNoOrder = errors.New("dchain: D->C export names no open order escrow")
	// ErrSeamExportExceedsOrder: a same-asset (refund) export exceeds the order's
	// remaining locked principal — the D-side per-taker cap (the export analog of the
	// precompile's ErrSettleExceedsOrder), so an over-refund can never draw on other
	// takers' pooled input.
	ErrSeamExportExceedsOrder = errors.New("dchain: D->C refund export exceeds the order's remaining locked principal")
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

// encodeSeamOrderObject serializes a C->D swap order — the canonical value head
// with spent=0 (a C->D leg has matched nothing yet) followed by the operation tail.
// BYTE-IDENTICAL with precompile/dex encodeOrderObject; the golden vector in
// atomic_seam_wire_test.go pins the two encoders against the same string.
func encodeSeamOrderObject(owner common.Address, asset [32]byte, amount uint64, op seamOp) []byte {
	v := make([]byte, 0, seamOrderObjectSize)
	v = append(v, encodeSeamObject(railSwap, owner, asset, amount, 0)...)
	tail := make([]byte, seamOpSize)
	copy(tail[0:32], op.Market[:])
	tail[32] = op.Side
	binary.BigEndian.PutUint64(tail[33:41], op.LimitPrice)
	binary.BigEndian.PutUint64(tail[41:49], op.Size)
	return append(v, tail...)
}

// decodeSeamOrderObject reads back a C->D swap order. ok=false for any width other
// than the canonical 118, or any rail other than railSwap — so a 69-byte VALUE object
// (an op-less order from a not-yet-upgraded C node, or a settlement object misrouted
// here) is REFUSED rather than imported with an invented operation. Fail closed: no
// operation, no import. The taker's principal stays locked on C and is reclaimable at
// the order's own deadline.
func decodeSeamOrderObject(v []byte) (owner common.Address, asset [32]byte, amount uint64, op seamOp, ok bool) {
	if len(v) != seamOrderObjectSize {
		return common.Address{}, [32]byte{}, 0, seamOp{}, false
	}
	rail, owner, asset, amount, _, headOK := decodeSeamObject(v[:seamObjectSize])
	if !headOK || rail != railSwap {
		return common.Address{}, [32]byte{}, 0, seamOp{}, false
	}
	tail := v[seamObjectSize:]
	copy(op.Market[:], tail[0:32])
	op.Side = tail[32]
	op.LimitPrice = binary.BigEndian.Uint64(tail[33:41])
	op.Size = binary.BigEndian.Uint64(tail[41:49])
	return owner, asset, amount, op, true
}

// DeriveOrderID computes the deterministic id of a C->D atomic order object — the
// shared-memory key the precompile PUTs it under (and that this VM's import consumes).
// It is BYTE-IDENTICAL to precompile/dex DeriveOrderID:
//
//	SHA-256( domain | networkID | cChainID | dChainID |
//	         account | assetIn | amountIn | marketID | nonce )
//
// Every component is fixed width. The native VM does not re-derive ids on the import
// path (it consumes by the key carried in the TxImport body); this is exported so the
// keeper / SDK / tests derive the SAME id the precompile minted, off-chain.
func DeriveOrderID(
	networkID uint32,
	cChainID, dChainID ids.ID,
	account common.Address,
	assetIn [32]byte,
	amountIn uint64,
	marketID [32]byte,
	nonce uint64,
) ids.ID {
	h := sha256.New()
	h.Write([]byte(nativeOrderDomain))
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

// seamAccount is the PER-ORDER ledger account a cross-chain swap runs in: the 16-byte
// fold of the order id, domain-separated so it can never collide with an Account16
// derived from a signer address.
//
// WHY NOT THE TAKER'S OWN ACCOUNT. The import used to credit Account16(secp, owner) —
// the same account the taker's own signed D orders draw from — so that the taker could
// place the order themselves. Now the seam places it, and sharing the account becomes a
// live conservation hole: the export drive settles the account's balances back to C,
// and it cannot tell this order's proceeds from the taker's unrelated native D
// balance. A taker holding 1000 ZOO natively and swapping in for 5 ZOO would have all
// 1005 exported, and C would credit 1005 out of the shared seam reserve while only 5
// of it was ever backed by this seam — draining other takers' pooled input. That is
// theft from the pool, reachable the moment the seam works.
//
// One account per order removes the attribution problem instead of bounding it: the
// account is born when the order is imported, only this order's order draws on it,
// and everything left in it at the end IS this order's answer. The spent witness
// (Remaining - leftover) becomes exact rather than inferred, the escrow closes on a
// truly empty account, and a taker's native D balance is untouchable by the seam.
//
// The D->C object's owner is still the escrow's RECORDED 20-byte C address
// (executeExport), so the account change is invisible to C: value returns to exactly
// the same place it always did.
func seamAccount(orderID ids.ID) userKey {
	d := hash.ComputeKeccak256Array([]byte(seamAccountDomain), orderID[:])
	var a userKey
	copy(a[:], d[:len(a)])
	return a
}

// seamAccountDomain scopes the order -> account fold. Distinct from accountDomain
// (auth.go), so a seam account and a signer's account live in disjoint spaces and no
// signature can ever authorize a draw on an order's escrow account.
const seamAccountDomain = "lux.dchain.seam.account.v1"

// --- TxImport / TxExport body codecs ---------------------------------------------

// EncodeSeamImportBody builds a TxImport body: orderID[32] | object[118] — the
// shared-memory key of the C->D object to consume, AND THE OBJECT'S OWN BYTES.
//
// WHY THE OBJECT TRAVELS IN THE TRANSACTION (the D-side half of the same defect the
// C side fixed in precompile v0.19.6). executeImport used to read the object out of
// shared memory with sm.Get, DURING execute(), while the consuming Remove landed at
// Accept. Those two facts cannot both hold. execute() is contracted to be a pure
// function of (committed state, this block's txs) and its output is compared against
// the block's claimed execRoot — but sm.Get reads live cross-chain memory that is
// outside the overlay and outside the parent root. Three concrete failures followed:
//
//   - A node re-executing an accepted block — bootstrapping, state-syncing, restarting
//     with no cached overlay — finds the object GONE (its own Accept removed it),
//     computes a rejected import where the network computed a credited one, derives a
//     different ledgerRoot, and dies on `execution root mismatch`. It can never sync
//     past the first swap the seam ever settles.
//   - Even live, an honest proposer's block is rejected by any validator that has not
//     yet seen the C-side flush: same block bytes, same committed D state, different
//     verdict, purely by timing.
//   - The read is a cross-process ZAP round-trip on the consensus path, taken while
//     Verify holds the VM lock.
//
// Carrying the bytes makes execution replayable forever with no access to mutable peer
// state. That the bytes are the REAL recorded object is then proven one level up, on
// the BLOCK (Block.verifySeamImports), where a mismatch rejects the block instead of
// diverging a root — the primary network's own import discipline, and exactly what
// evm/plugin/evm/dex_atomic_verify.go does for the C side.
//
// The ship rule did not weaken, it moved: D is still funded ONLY by consuming a real
// C->D object. What changed is where that is proven.
func EncodeSeamImportBody(orderID ids.ID, object []byte) []byte {
	b := make([]byte, seamImportBodySize)
	copy(b[0:32], orderID[:])
	copy(b[32:], object)
	return b
}

// decodeSeamImportBody splits a TxImport body into the order id and the carried
// object bytes. The fixed bodySize gate already rejects any other width at parse, so
// this only fails on internal misuse.
func decodeSeamImportBody(body []byte) (orderID ids.ID, object []byte, err error) {
	if len(body) != seamImportBodySize {
		return ids.Empty, nil, fmt.Errorf("dchain: seam import body width %d, want %d", len(body), seamImportBodySize)
	}
	copy(orderID[:], body[0:32])
	return orderID, body[32:], nil
}

// EncodeSeamExportBody builds a TxExport body: orderID[32] | asset[32] | amount[8] |
// spent[8]. asset/amount are the D->C object's value; spent is the matched input
// witness on a proceeds leg (0 on a refund). Exported for the keeper / SDK / tests.
func EncodeSeamExportBody(orderID ids.ID, asset [32]byte, amount, spent uint64) []byte {
	b := make([]byte, seamExportBodySize)
	copy(b[0:32], orderID[:])
	copy(b[32:64], asset[:])
	binary.BigEndian.PutUint64(b[64:72], amount)
	binary.BigEndian.PutUint64(b[72:80], spent)
	return b
}

// decodeSeamExportBody parses a TxExport body. A short body is a malformed tx (block
// abort) — the fixed bodySize gate already rejects undersized frames at parse, so this
// only fails on internal misuse.
func decodeSeamExportBody(body []byte) (orderID ids.ID, asset [32]byte, amount, spent uint64, err error) {
	if len(body) < seamExportBodySize {
		return ids.Empty, [32]byte{}, 0, 0, fmt.Errorf("dchain: seam export body too short: %d", len(body))
	}
	copy(orderID[:], body[0:32])
	copy(asset[:], body[32:64])
	amount = binary.BigEndian.Uint64(body[64:72])
	spent = binary.BigEndian.Uint64(body[72:80])
	return orderID, asset, amount, spent, nil
}

// --- per-order escrow (seamintent:) ---------------------------------------------

func seamOrderKey(orderID ids.ID) []byte {
	k := make([]byte, len(prefixSeamOrder)+32)
	copy(k, prefixSeamOrder)
	copy(k[len(prefixSeamOrder):], orderID[:])
	return k
}

func encodeSeamOrder(r seamOrderRecord) []byte {
	v := make([]byte, seamOrderRecordSize)
	copy(v[0:20], r.Owner[:])
	copy(v[20:52], r.AssetIn[:])
	binary.BigEndian.PutUint64(v[52:60], r.Remaining)
	v[60] = r.Status
	copy(v[61:93], r.Op.Market[:])
	v[93] = r.Op.Side
	binary.BigEndian.PutUint64(v[94:102], r.Op.LimitPrice)
	binary.BigEndian.PutUint64(v[102:110], r.Op.Size)
	return v
}

func decodeSeamOrder(v []byte) (seamOrderRecord, error) {
	if len(v) != seamOrderRecordSize {
		return seamOrderRecord{}, fmt.Errorf("dchain: corrupt seam order len=%d", len(v))
	}
	var r seamOrderRecord
	copy(r.Owner[:], v[0:20])
	copy(r.AssetIn[:], v[20:52])
	r.Remaining = binary.BigEndian.Uint64(v[52:60])
	r.Status = v[60]
	copy(r.Op.Market[:], v[61:93])
	r.Op.Side = v[93]
	r.Op.LimitPrice = binary.BigEndian.Uint64(v[94:102])
	r.Op.Size = binary.BigEndian.Uint64(v[102:110])
	return r, nil
}

func putSeamOrder(db database.KeyValueWriter, orderID ids.ID, r seamOrderRecord) error {
	return db.Put(seamOrderKey(orderID), encodeSeamOrder(r))
}

func getSeamOrder(db database.KeyValueReader, orderID ids.ID) (seamOrderRecord, bool, error) {
	v, err := db.Get(seamOrderKey(orderID))
	if err == database.ErrNotFound {
		return seamOrderRecord{}, false, nil
	}
	if err != nil {
		return seamOrderRecord{}, false, err
	}
	r, derr := decodeSeamOrder(v)
	if derr != nil {
		return seamOrderRecord{}, false, derr
	}
	return r, true, nil
}

// --- per-export settlement record (seamout:) -------------------------------------

func seamOutKey(outputID ids.ID) []byte {
	k := make([]byte, len(prefixSeamOut)+32)
	copy(k, prefixSeamOut)
	copy(k[len(prefixSeamOut):], outputID[:])
	return k
}

func putSeamOut(db database.KeyValueWriter, outputID, orderID ids.ID, asset [32]byte, amount uint64) error {
	v := make([]byte, 32+32+8)
	copy(v[0:32], orderID[:])
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

// executeImport CONSUMES a C->D atomic order object and FUNDS the taker's native D
// account with the recorded asset. It mirrors precompile SubmitSwapOrder's other
// side and the proxy executeImport, but credits the native ledger so the in-consensus
// matcher trades the funds:
//
//  1. replay guard: the order escrow must not already exist (one import per order).
//  2. BIND the object bytes the TRANSACTION carried — never a shared-memory read. The
//     credited owner/asset/amount and the recorded operation are the OBJECT's, never
//     declared separately by the tx. That those bytes are the real recorded object is
//     proven on the BLOCK (Block.verifySeamImports), not here.
//  3. GATE: the operation must be executable at all (a real market side, a non-zero
//     size, a non-zero limit). An order whose order could never run is refused before
//     any value is credited — the principal stays on C, reclaimable at its deadline.
//  4. CREDIT available[seamAccount(orderID), asset] += amount (fund the order's OWN
//     ledger account, isolated from the taker's native D balance).
//  5. RECORD the per-order escrow — owner/assetIn/remaining for the export binding and
//     cap, the OPERATION for the seam submit, status=open.
//  6. ACCUMULATE the shared-memory Remove under the C chain (applied at Accept), so the
//     object is consumed exactly once.
//
// It returns ok=false (with a nil error) for a deterministic per-tx REJECT — a
// malformed object, wrong rail, unexecutable operation, replay, or unwired seam — so a
// rejected leg never wedges the chain. A db fault returns an error (block abort). The
// block records a successful import under seen: (idempotent replay) but NOT a reject
// (retryable).
//
// Every input is now the block's own bytes or the committed overlay, so two nodes
// executing this block at any time, in any order, from any starting point, compute the
// identical result.
func (vm *VM) executeImport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, orderID ids.ID, object []byte) (credited uint64, ok bool, err error) {
	if vm.cChainID == ids.Empty {
		return 0, false, nil // seam not wired: deterministic reject, no mint
	}

	// (1) replay guard — the escrow row is the durable consumed marker.
	if _, exists, gerr := getSeamOrder(db, orderID); gerr != nil {
		return 0, false, gerr
	} else if exists {
		return 0, false, nil // already imported (ErrSeamOrderReplay condition); reject
	}

	// (2) BIND the CARRIED object. Width and rail are checked by the decoder: a
	// 69-byte value object has no operation and is refused rather than imported with
	// an invented one.
	owner, asset, amount, op, decOK := decodeSeamOrderObject(object)
	if !decOK || amount == 0 {
		return 0, false, nil // malformed: reject (never reinterpret into a credit)
	}

	// (3) GATE the operation.
	if !op.valid() {
		return 0, false, nil
	}

	// (4) CREDIT the order's OWN account (no mint — the value was removed from the
	// taker on C and is consumed here exactly once).
	acct := seamAccount(orderID)
	if cerr := creditAvailable(db, acct, asset, amount); cerr != nil {
		return 0, false, cerr
	}

	// (5) RECORD the escrow: the export binding + cap, and the operation to run.
	if perr := putSeamOrder(db, orderID, seamOrderRecord{
		Owner:     owner,
		AssetIn:   asset,
		Remaining: amount,
		Status:    seamOrderOpen,
		Op:        op,
	}); perr != nil {
		return 0, false, perr
	}

	// (6) ACCUMULATE the Remove under the C chain (consumed once, applied at Accept).
	req := ar.forChain(vm.cChainID)
	req.RemoveRequests = append(req.RemoveRequests, orderID[:])
	return amount, true, nil
}

// seamSubmitBody derives the zapwire Submit frame for an imported order — the ONE
// definition of "the order this order means". The drive uses it to decide there is
// work; execute uses it to run the order. Both derive it from the SAME committed
// escrow, so the operation is declared exactly once (in the object, at import) and
// never restated on the wire for the two sides to cross-check.
//
// isMarket is always false: every seam order is a LIMIT IOC order (seamOp.LimitPrice
// is required non-zero), so the cross is bounded by the taker's own limit and the
// unbounded market-buy path — which locks the account's entire balance and sweeps a
// caller-chosen size — is unreachable from the seam.
//
// The user is the order's OWN account, so the lock, the cross and the settlement all
// draw on exactly the value this order imported.
func seamSubmitBody(orderID ids.ID, rec seamOrderRecord) []byte {
	acct := seamAccount(orderID)
	return zapwire.EncodeSubmit(rec.Op.Market, rec.Op.Side, false, rec.Op.LimitPrice, rec.Op.Size, string(acct[:]))
}

// executeExport DEBITS the taker's native D balance and writes a D->C settlement
// object the precompile's ImportSettlement consumes. It is driven AFTER the native
// matcher has settled (the proceeds / refund live in the taker's available balance),
// and binds the object to the order so Phase B's checks pass:
//
//  1. load the order escrow (must be open) — the export draws against THIS order;
//     the D->C object's owner is the escrow's RECORDED 20-byte owner, never the tx, so
//     an export can never redirect value to a different account.
//  2. PER-TAKER CAP: a same-asset (refund) export (asset == escrow.AssetIn) is bounded
//     by the order's remaining locked principal and decrements it — the D-side analog
//     of the precompile's per-taker cap. A proceeds export (different asset) is bounded
//     only by the available balance the matcher actually credited.
//  3. DEBIT available[seamAccount(orderID), asset] -= amount (refuses an over-debit:
//     the export is backed by realized D value, so it can never credit more on C than D
//     matched — conservation).
//  4. ENCODE the D->C object (rail=railSwap, owner, asset, amount, spent) and ACCUMULATE
//     a Put under the C chain keyed by the deterministic outputID.
//  5. RECORD the settlement (seamout:) so a keeper reads the outputID to claim.
//
// spent is the matched INPUT that produced a proceeds output (0 on a refund) — the
// witness the precompile's taker-authenticated MEV floor checks. Returns ok=false for a
// deterministic per-tx reject (no order, over-cap, insufficient balance, unwired seam).
func (vm *VM) executeExport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, exportTxID, orderID ids.ID, asset [32]byte, amount, spent uint64) (outputID ids.ID, ok bool, err error) {
	sm := vm.seamSharedMemory()
	if sm == nil || vm.cChainID == ids.Empty {
		return ids.Empty, false, nil // seam not wired: reject
	}
	if amount == 0 {
		return ids.Empty, false, nil
	}

	// (1) the order escrow must exist and its ORDER MUST HAVE RUN. Exporting from an
	// `open` escrow would settle a swap whose order has not been placed yet — the
	// premature refund the old "does the account hold realized output?" heuristic was
	// there to prevent. Now it is the recorded fact, so a zero-fill swap (which
	// produces no output, and which that heuristic could therefore NEVER export)
	// settles too, instead of stranding the taker's principal on D.
	rec, exists, gerr := getSeamOrder(db, orderID)
	if gerr != nil {
		return ids.Empty, false, gerr
	}
	if !exists || rec.Status != seamOrderSubmitted {
		return ids.Empty, false, nil // ErrSeamNoOrder condition
	}

	// (2) PER-TAKER CAP on the same-asset refund leg (the export analog of the
	// precompile's ImportSettlement per-taker cap). A proceeds leg (different asset) is
	// not principal-capped — output units legitimately differ from input units.
	refund := asset == rec.AssetIn
	if refund && amount > rec.Remaining {
		return ids.Empty, false, nil // ErrSeamExportExceedsOrder condition
	}

	// (3) DEBIT the realized D balance (refuses over-debit: conservation — C can never be
	// credited more than D matched). The matcher credited exactly the proceeds/refund.
	acct := seamAccount(orderID)
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
		if perr := putSeamOrder(db, orderID, rec); perr != nil {
			return ids.Empty, false, perr
		}
	}
	if perr := putSeamOut(db, outputID, orderID, asset, amount); perr != nil {
		return ids.Empty, false, perr
	}
	return outputID, true, nil
}

// commitSeamAtomic commits the block's accumulated cross-chain operations at the
// single accept-time commit point. With no shared memory or no atomic ops it falls
// back to the plain overlay commit.
//
// NO BATCH. This used to hand the overlay's CommitBatch to sm.Apply so state and
// shared memory landed in one write — the in-process platformvm acceptor pattern.
// That pattern is unavailable to this VM: the D-Chain is an out-of-process plugin,
// its SharedMemory is the ZAP client, and a database.Batch cannot cross a process
// boundary. atomiczap.Apply refuses one outright (ErrBatchUnsupported), and there was
// NO batch-less path here — so every block carrying a seam op failed at Accept, which
// is fatal. The D-Chain would have halted on the first cross-chain order it ever
// imported.
//
// COMMIT FIRST, THEN APPLY. Without a shared batch, exactly-once is unreachable; the
// choice is at-most-once (commit first — a crash in the window between the two writes
// SKIPS an op) or at-least-once (apply first — a crash REPLAYS one). Replay is not
// survivable: chains/atomic's SetValue returns a duplicate-put error for a key already
// present, which is a fatal Accept, and a Put replayed after the peer consumed the
// object RE-CREATES it. A skip is survivable on both legs, because neither side's
// replay protection depends on the shared-memory op:
//
//   - a skipped Remove of an imported C->D order: the object lingers, but executeImport's
//     replay guard is the durable `seamintent:` escrow row in COMMITTED STATE, written
//     above. A re-import is refused. Inert garbage.
//   - a skipped Put of a D->C export: the taker's proceeds do not reach C. C's order
//     carries a finite deadline by construction, so the taker's locked principal is
//     reclaimable there; the D-side escrow is already drained, so nothing is minted.
//     Bounded, and loud.
//
// An Apply that ERRORS stays fatal on purpose: a node that cannot mutate shared memory
// must stop rather than run on with a divergent view of it.
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
	if err := overlay.Commit(); err != nil {
		return fmt.Errorf("dchain: seam state commit: %w", err)
	}
	if err := sm.Apply(ar.reqs); err != nil {
		return fmt.Errorf("dchain: seam atomic apply: %w", err)
	}
	return nil
}

// versionDB is the minimal commit surface commitSeamAtomic needs from
// *versiondb.Database, kept as an interface so the seam commit is testable in isolation.
type versionDB interface {
	Commit() error
	CommitBatch() (database.Batch, error)
	Abort()
}
