// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic.go is the D side of the C<->D funding rail. It has exactly ONE job: move
// ownership exactly once between chains.
//
// MOVE ONCE, TRADE MANY. Value crosses the boundary twice in its life — in, and out.
// Between those, D owns it outright and every order, match and trade is a local
// balance movement against the owner's own account. That is what makes D an exchange
// ledger rather than a remote matching oracle bolted onto C, and it is why a million
// trades cost zero crossings.
//
//	C --Export--> C->D --Import--> D --Export--> D->C --Import--> C
//
// Those four are the only transitions. A unit is spendable on exactly one chain at a
// time, because each transition consumes its input and produces its output.
//
//   - IMPORT (C->D): the 0x9999 precompile debited the depositor on C and wrote a
//     claim. This VM consumes it — credits the claim's BENEFICIARY, and accumulates
//     the shared-memory Remove. No mint: the value was already removed on C. The
//     credit lands in the beneficiary's OWN account, the one their signed orders draw
//     from, so the money is immediately tradable without another crossing.
//   - EXPORT (D->C): the owner signs a withdrawal. This VM debits their available
//     balance and writes a claim the precompile's import consumes exactly once. No
//     mint: the value was already debited here.
//
// THE WIRE IS 60 BYTES AND SAYS NOTHING ABOUT TRADING:
//
//	beneficiary(20) | asset(32) | amount(8)
//
// It is byte-identical with precompile/dex native_wire.go, pinned by the golden
// vector in atomic_wire_test.go. The two repos cannot import each other, so that
// vector is the whole cross-repo contract; a one-sided change fails at test time
// rather than in consensus.
//
// WHAT IS NOT HERE, DELIBERATELY. No market, side, price, size or execution witness
// rides this wire; no lane byte selects a trading rail; no slippage rule runs on the
// import path. An order is a D-local transaction against an already-funded account,
// and a cancel is a D-local unreserve. Braiding those into the crossing is what made
// a trade cost a cross-chain object; keeping them out is the whole point.
//
// CONSENSUS-SAFETY. The import credit is a pure function of the (rooted) tx body plus
// the committed consumed-set; the export claim is a pure function of the (rooted) tx
// body plus the committed ledger. Nothing reads live cross-chain memory during
// execution, so a node re-executing an accepted block — bootstrapping, state-syncing,
// restarting — computes the identical result forever. That the carried object is the
// REAL recorded one is proven one level up, on the block (Block.verifyImports), where
// a mismatch rejects the block instead of diverging a root.

// claimSize is the fixed funding-object width: beneficiary(20) | asset(32) |
// amount(8). IDENTICAL to precompile/dex claimSize.
const claimSize = 20 + 32 + 8

// encodeClaim serializes a funding object, BYTE-IDENTICAL with precompile/dex
// encodeClaim. beneficiary is the full 20-byte destination account; asset is the full
// injective AssetID (native = all-zero); amount is integer asset units.
func encodeClaim(beneficiary common.Address, asset [32]byte, amount uint64) []byte {
	v := make([]byte, claimSize)
	copy(v[0:20], beneficiary[:])
	copy(v[20:52], asset[:])
	binary.BigEndian.PutUint64(v[52:60], amount)
	return v
}

// decodeClaim is the exact inverse. ok=false for any width but the canonical one, so
// a corrupt record is never reinterpreted into a credit.
func decodeClaim(v []byte) (beneficiary common.Address, asset [32]byte, amount uint64, ok bool) {
	if len(v) != claimSize {
		return common.Address{}, [32]byte{}, 0, false
	}
	copy(beneficiary[:], v[0:20])
	copy(asset[:], v[20:52])
	amount = binary.BigEndian.Uint64(v[52:60])
	return beneficiary, asset, amount, true
}

// DeriveClaimID computes a funding claim's deterministic id — the shared-memory key
// the object is written under, and the key the destination consumes:
//
//	SHA-256( claimDomain | source | dest | sourceTx | index )
//
// ONE derivation, BOTH directions, BOTH repos — byte-identical with precompile/dex
// DeriveClaimID. Injective: a transaction id is unique on its chain and the index is
// unique within that transaction. (source, dest) scope the claim to one corridor, so
// an object minted for C->D can never be replayed as D->C.
func DeriveClaimID(source, dest, sourceTx ids.ID, index uint32) ids.ID {
	h := sha256.New()
	h.Write([]byte(claimDomain))
	h.Write(source[:])
	h.Write(dest[:])
	h.Write(sourceTx[:])
	var u4 [4]byte
	binary.BigEndian.PutUint32(u4[:], index)
	h.Write(u4[:])
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return ids.ID(out)
}

// claimDomain scopes the claim-id derivation — IDENTICAL to precompile/dex
// claimDomain so the id this VM recomputes matches the id C minted the object under.
//
// The literal is the hash input, not a label: changing a byte of it moves every claim
// id the rail has ever derived.
const claimDomain = "lux.dex.claim.v1"

// prefixClaim is the CONSUMED SET: one row per imported claim id, written in the
// block overlay alongside the credit so the consume and the credit land together.
// That is the whole safety property of the rail — if a claim could be marked consumed
// without the credit landing, value would vanish; if the credit could land without
// the mark, a retry, a reorg, a racing relayer or a restart could import the same
// claim again, and that is a double-spend of real money.
//
// It does not depend on the shared-memory Remove, so a Remove that is skipped by a
// crash between the state commit and the apply cannot cause a re-import.
//
// These are on-disk key bytes and do not track this identifier's name.
const prefixClaim = "claim:"

func claimKey(claimID ids.ID) []byte {
	k := make([]byte, len(prefixClaim)+32)
	copy(k, prefixClaim)
	copy(k[len(prefixClaim):], claimID[:])
	return k
}

func claimConsumed(db database.KeyValueReader, claimID ids.ID) (bool, error) {
	_, err := db.Get(claimKey(claimID))
	if err == database.ErrNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

func markClaimConsumed(db database.KeyValueWriter, claimID ids.ID) error {
	return db.Put(claimKey(claimID), []byte{1})
}

var (
	// ErrClaimBodyWidth: an import/export body that is not its fixed width. The
	// per-type bodySize gate already rejects these at parse, so this only fires on
	// internal misuse.
	ErrClaimBodyWidth = errors.New("dchain: cross-chain body width")
)

// --- tx body codecs ---------------------------------------------------------------

// importBodySize is the TxImport body width: claimID(32) | claim(60).
// exportBodySize is the TxExport body width: owner(16) | asset(32) | amount(8) |
// beneficiary(20).
const (
	importBodySize = 32 + claimSize
	exportBodySize = zapwire.UserSize + 32 + 8 + 20
)

// EncodeImportBody builds a TxImport body: the claim's shared-memory key AND THE
// CLAIM'S OWN BYTES.
//
// WHY THE OBJECT TRAVELS IN THE TRANSACTION. Execution used to read the object out of
// shared memory with sm.Get, DURING execute(), while the consuming Remove landed at
// Accept. Those two facts cannot both hold: execute() is contracted to be a pure
// function of (committed state, this block's txs) and its output is compared against
// the block's claimed execRoot, but sm.Get reads live cross-chain memory outside the
// overlay and outside the parent root. A node re-executing an accepted block finds
// the object GONE (its own Accept removed it), computes a rejected import where the
// network computed a credited one, and dies on `execution root mismatch` — it can
// never sync past the first crossing the rail ever makes.
//
// Carrying the bytes makes execution replayable forever with no access to mutable
// peer state. That the bytes are the REAL recorded object is proven one level up, on
// the BLOCK (Block.verifyImports), where a mismatch rejects the block instead of
// diverging a root. The rail's rule did not weaken, it moved: D is still funded ONLY
// by consuming a real C->D claim.
func EncodeImportBody(claimID ids.ID, object []byte) []byte {
	b := make([]byte, importBodySize)
	copy(b[0:32], claimID[:])
	copy(b[32:], object)
	return b
}

func decodeImportBody(body []byte) (claimID ids.ID, object []byte, err error) {
	if len(body) != importBodySize {
		return ids.Empty, nil, fmt.Errorf("%w: import %d, want %d", ErrClaimBodyWidth, len(body), importBodySize)
	}
	copy(claimID[:], body[0:32])
	return claimID, body[32:], nil
}

// EncodeExportBody builds a TxExport body: owner(16) | asset(32) | amount(8) |
// beneficiary(20).
//
// owner is the D account debited. It is ASSERTED on the wire and bound to the
// signature (txWireUser), exactly as a withdraw is, because an export removes value
// from a real account and names where it lands — so only that account's own signer may
// authorize one. beneficiary is the C address credited on the other side.
func EncodeExportBody(owner userKey, asset [32]byte, amount uint64, beneficiary common.Address) []byte {
	b := make([]byte, exportBodySize)
	copy(b[0:zapwire.UserSize], owner[:])
	copy(b[zapwire.UserSize:zapwire.UserSize+32], asset[:])
	binary.BigEndian.PutUint64(b[zapwire.UserSize+32:zapwire.UserSize+40], amount)
	copy(b[zapwire.UserSize+40:], beneficiary[:])
	return b
}

func decodeExportBody(body []byte) (owner userKey, asset [32]byte, amount uint64, beneficiary common.Address, err error) {
	if len(body) != exportBodySize {
		return owner, asset, 0, beneficiary, fmt.Errorf("%w: export %d, want %d", ErrClaimBodyWidth, len(body), exportBodySize)
	}
	copy(owner[:], body[0:zapwire.UserSize])
	copy(asset[:], body[zapwire.UserSize:zapwire.UserSize+32])
	amount = binary.BigEndian.Uint64(body[zapwire.UserSize+32 : zapwire.UserSize+40])
	copy(beneficiary[:], body[zapwire.UserSize+40:])
	return owner, asset, amount, beneficiary, nil
}

// --- cross-chain request accumulator ----------------------------------------------
//
// atomicRequests collects the shared-memory removes (import) and puts (export) a
// block produces, applied at Accept. It is the platformvm acceptor pattern.

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

// sharedMemory returns this chain's cross-chain handle (from the runtime the engine
// wired at Initialize), or nil when none was wired (single-chain dev / unit tests).
func (vm *VM) sharedMemory() atomic.SharedMemory {
	if vm.runtime == nil {
		return nil
	}
	return vm.runtime.GetSharedMemory()
}

// executeImport CONSUMES a C->D claim and CREDITS ITS BENEFICIARY. That is the whole
// operation — no order is created, no escrow is written, nothing is placed on a book.
// The value lands in the beneficiary's own account and becomes tradable by their own
// signed transactions, which is what "move once, trade many" means in practice.
//
//  1. REPLAY guard: the claim id must not already be in the consumed set.
//  2. BIND the claim bytes the TRANSACTION carried — never a shared-memory read. The
//     credited beneficiary/asset/amount are the OBJECT's, never declared separately.
//  3. CREDIT available[Account16(beneficiary), asset] += amount. No mint: the value
//     was removed from the depositor on C and is consumed here exactly once.
//  4. MARK the claim consumed, in the same overlay as the credit.
//  5. ACCUMULATE the shared-memory Remove under the C chain (applied at Accept).
//
// It returns ok=false with a nil error for a deterministic per-tx REJECT — malformed
// claim, zero amount, replay, or an unwired rail — so a rejected leg never wedges the
// chain. A db fault returns an error (block abort).
//
// PERMISSIONLESS BY CONSTRUCTION, which is why it carries no signature: delivering a
// claim can only credit the account the claim already names, so a relayer that
// refuses to deliver cannot strand anyone — any other participant can.
func (vm *VM) executeImport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, claimID ids.ID, object []byte) (credited uint64, ok bool, err error) {
	// BOTH ENDS OR NEITHER. The carried bytes are only ever a CLAIM about a recorded
	// object; what makes them value is that shared memory holds the same object, proven
	// on the block. With no shared memory that proof cannot be taken — so the credit
	// cannot be given, or 60 self-authored bytes are money. A C chain id alone is not
	// the rail: the plugin server sets one unconditionally, and every live network has
	// one today with no atomic server behind it.
	if vm.sharedMemory() == nil || vm.cChainID == ids.Empty {
		return 0, false, nil // rail not wired: deterministic reject, no mint
	}

	// (1) REPLAY guard.
	done, gerr := claimConsumed(db, claimID)
	if gerr != nil {
		return 0, false, gerr
	}
	if done {
		return 0, false, nil
	}

	// (2) BIND the CARRIED claim. Width is checked by the decoder; a zero amount is
	// not a value transfer (and is what an absent object zero-padded to width would
	// decode as), so it is refused here rather than credited.
	beneficiary, asset, amount, decOK := decodeClaim(object)
	if !decOK || amount == 0 {
		return 0, false, nil
	}

	// (3) CREDIT the beneficiary's OWN account. A C address is a secp256k1 identity by
	// construction, so it folds to the secp account — the same one their signed orders
	// draw from, with no second crossing between arriving and trading.
	acct := Account16(dex.AuthSecp256k1, beneficiary)
	if cerr := creditAvailable(db, acct, asset, amount); cerr != nil {
		return 0, false, cerr
	}

	// (4) MARK consumed, in the same overlay as the credit.
	if merr := markClaimConsumed(db, claimID); merr != nil {
		return 0, false, merr
	}

	// (5) ACCUMULATE the Remove under the C chain (applied at Accept).
	req := ar.forChain(vm.cChainID)
	req.RemoveRequests = append(req.RemoveRequests, claimID[:])
	return amount, true, nil
}

// executeExport DEBITS the owner's available balance and WRITES a C-bound claim the
// 0x9999 precompile consumes exactly once. That is the whole operation — it draws on
// no order, consults no escrow and carries no execution witness.
//
//  1. DEBIT available[owner, asset] -= amount. Refuses an over-debit, so C can never
//     be credited more than D held. RESERVED balance is untouchable here, as
//     everywhere: value backing an open order cannot also leave the chain, and
//     debitAvailable draws only the available lane.
//  2. ENCODE the claim naming the beneficiary and ACCUMULATE a Put under the C chain,
//     keyed by the deterministic claim id.
//
// The owner is bound to the transaction's signature by the auth gate before this
// runs, so an export can only ever spend its signer's own balance.
//
// Returns ok=false for a deterministic per-tx reject (unwired rail, zero amount,
// insufficient balance).
func (vm *VM) executeExport(db database.KeyValueReaderWriterDeleter, ar *atomicRequests, exportTxID ids.ID, owner userKey, asset [32]byte, amount uint64, beneficiary common.Address) (claimID ids.ID, ok bool, err error) {
	if vm.sharedMemory() == nil || vm.cChainID == ids.Empty {
		return ids.Empty, false, nil // rail not wired: reject
	}
	if amount == 0 || beneficiary == (common.Address{}) {
		return ids.Empty, false, nil
	}

	// (1) DEBIT the realized balance (no mint on C: the value leaves D first).
	if derr := debitAvailable(db, owner, asset, amount); derr != nil {
		if errors.Is(derr, ErrInsufficientAvailable) {
			return ids.Empty, false, nil
		}
		return ids.Empty, false, derr
	}

	// (2) ENCODE + ACCUMULATE the Put under the C chain.
	claimID = DeriveClaimID(vm.chainID, vm.cChainID, exportTxID, 0)
	req := ar.forChain(vm.cChainID)
	req.PutRequests = append(req.PutRequests, &atomic.Element{
		Key:    claimID[:],
		Value:  encodeClaim(beneficiary, asset, amount),
		Traits: [][]byte{beneficiary[:]}, // index by recipient (lux.Addressable)
	})
	return claimID, true, nil
}

// commitAtomic commits the block's cross-chain operations at the single accept-time
// commit point. With no shared memory or no operations it falls back to the plain
// overlay commit.
//
// NO BATCH. This used to hand the overlay's CommitBatch to sm.Apply so state and
// shared memory landed in one write — the in-process platformvm acceptor pattern.
// That pattern is unavailable here: the D-Chain is an out-of-process plugin, its
// SharedMemory is the ZAP client, and a database.Batch cannot cross a process
// boundary. atomiczap.Apply refuses one outright (ErrBatchUnsupported).
//
// COMMIT FIRST, THEN APPLY. Without a shared batch, exactly-once is unreachable; the
// choice is at-most-once (commit first — a crash between the two writes SKIPS an op)
// or at-least-once (apply first — a crash REPLAYS one). Replay is not survivable:
// SetValue returns a duplicate-put error for a key already present, which is a fatal
// Accept, and a Put replayed after the peer consumed the object RE-CREATES it. A skip
// is survivable on both legs, because neither side's replay protection depends on the
// shared-memory op:
//
//   - a skipped Remove of an imported claim: the object lingers, but the replay guard
//     is the durable consumed-set row in COMMITTED STATE, written above. A re-import
//     is refused. Inert garbage.
//   - a skipped Put of an export: the value does not reach C. Nothing is minted — D
//     debited it — and the owner's export is retryable, which is the ordinary repair
//     for a lost write.
//
// An Apply that ERRORS stays fatal on purpose: a node that cannot mutate shared
// memory must stop rather than run on with a divergent view of it.
func (vm *VM) commitAtomic(overlay versionDB, ar *atomicRequests) error {
	if ar.empty() {
		return overlay.Commit()
	}
	sm := vm.sharedMemory()
	if sm == nil {
		// Operations were accumulated with nothing to apply them to. Both producers
		// refuse when sm is nil, so reaching here means a new path started accumulating
		// without that guard — and this branch used to COMMIT THE STATE ANYWAY while
		// dropping the op, which is the "credit lands, remove never does" shape the
		// whole file exists to prevent. An unappliable op is the same condition as an
		// Apply that errors, and gets the same answer: stop.
		return errors.New("dchain: rail ops accumulated with no shared memory")
	}
	if err := overlay.Commit(); err != nil {
		return fmt.Errorf("dchain: rail state commit: %w", err)
	}
	if err := sm.Apply(ar.reqs); err != nil {
		return fmt.Errorf("dchain: rail apply: %w", err)
	}
	return nil
}

// versionDB is the minimal commit surface commitAtomic needs from
// *versiondb.Database, kept as an interface so the commit is testable in isolation.
type versionDB interface {
	Commit() error
	CommitBatch() (database.Batch, error)
	Abort()
}
