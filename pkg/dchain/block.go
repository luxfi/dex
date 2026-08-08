// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/luxfi/database"
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// block.go is the d-chain block: a height-ordered batch of DEX transactions. Its
// lifecycle is the heart of the matcher-at-Verify design:
//
//   - Verify  re-executes the block's txs against a versiondb OVERLAY on the
//             committed state, deriving the fills and the resulting book rows,
//             and checks the proposer's claimed execution root. Every validator
//             runs this, so a lying proposer (wrong fills, wrong root) is
//             rejected. The overlay is held on the block until Accept/Reject.
//   - Accept  commits the overlay to durable zapdb atomically (versiondb.Commit)
//             and advances the in-RAM book + meta watermarks.
//   - Reject  aborts the overlay (versiondb.Abort) and returns the txs to the
//             mempool.
//
// The block bytes are self-contained: [parent:32][height:8][ts:8][execRoot:32]
// [txlist]. The id is Checksum256 over the bytes.

// blockHeaderSize is the fixed prefix before the tx list.
const blockHeaderSize = 32 + 8 + 8 + 32 // parent + height + timestamp + execRoot

// Status bytes (mirror the consensus choices.Status small-int convention used by
// block.Block.Status() — Processing/Accepted/Rejected).
const (
	statusProcessing uint8 = 0
	statusAccepted   uint8 = 1
	statusRejected   uint8 = 2
)

// errZeroCustodyRef refuses a deposit/withdraw whose idempotency ref is all-zero
// (R3, defense-in-depth). The ref is the txHash that uniquely identifies the
// originating EVM tx and enters the seen: dedup key via tx.ID(); a committed EVM
// tx always has a unique non-zero hash, so a zero ref means a non-tx-identified
// (test-mock) or proxy fillRef=0 frame. A zero ref is a full-length 64B frame so
// length checks don't catch it — refusing here (every validator decodes the same
// frame and rejects identically, so it stays consensus-neutral) closes the door
// before any credit/debit and is a prerequisite for the ERC-20 rail.
var errZeroCustodyRef = errors.New("dchain: zero custody ref refused (no originating-tx identity)")

// Block is one d-chain block implementing block.Block.
type Block struct {
	vm *VM

	parentID  ids.ID
	height    uint64
	timestamp time.Time
	execRoot  [Size]byte
	txs       []*Tx

	bytes  []byte
	id     ids.ID
	status uint8

	// overlay is the versiondb staged during Verify, committed at Accept and
	// aborted at Reject. nil until Verify runs.
	overlay *versiondb.Database

	// outcomes are the per-tx consensus results derived by execute (during
	// Verify or Build). Accept resolves any parked ZAP handler from these. nil
	// until execute runs against this block.
	outcomes []txOutcome

	// atomic is the block's accumulated cross-chain shared-memory operations (the
	// funding rail's C->D removes / D->C puts), derived by execute and
	// applied atomically with the state overlay at Accept (atomic.go commitAtomic).
	// nil/empty for a block with no crossings. Stashed on the block alongside the
	// overlay so the same instance a later Accept resolves carries both.
	atomic *atomicRequests
}

// newBlock assembles a block from its fields and computes its bytes + id. The
// execRoot is the proposer's claim (set by BuildBlock to the locally-computed
// root; checked by Verify on every other validator).
func newBlock(vm *VM, parentID ids.ID, height uint64, ts time.Time, execRoot [Size]byte, txs []*Tx) *Block {
	b := &Block{
		vm:        vm,
		parentID:  parentID,
		height:    height,
		timestamp: ts,
		execRoot:  execRoot,
		txs:       txs,
		status:    statusProcessing,
	}
	b.bytes = b.marshal()
	b.id = ids.Checksum256(b.bytes)
	return b
}

// marshal encodes the block to its canonical bytes.
func (b *Block) marshal() []byte {
	body := encodeTxList(b.txs)
	out := make([]byte, blockHeaderSize+len(body))
	copy(out[0:32], b.parentID[:])
	binary.BigEndian.PutUint64(out[32:40], b.height)
	binary.BigEndian.PutUint64(out[40:48], uint64(b.timestamp.UnixNano()))
	copy(out[48:80], b.execRoot[:])
	copy(out[80:], body)
	return out
}

// parseBlock decodes block bytes into a Block bound to vm. It does NOT execute —
// execution happens in Verify.
func parseBlock(vm *VM, raw []byte) (*Block, error) {
	if len(raw) < blockHeaderSize {
		return nil, fmt.Errorf("dchain: block too short: %d", len(raw))
	}
	var parent ids.ID
	copy(parent[:], raw[0:32])
	height := binary.BigEndian.Uint64(raw[32:40])
	tsNanos := int64(binary.BigEndian.Uint64(raw[40:48]))
	var execRoot [Size]byte
	copy(execRoot[:], raw[48:80])
	txs, err := decodeTxList(raw[80:])
	if err != nil {
		return nil, err
	}
	b := &Block{
		vm:        vm,
		parentID:  parent,
		height:    height,
		timestamp: time.Unix(0, tsNanos).UTC(),
		execRoot:  execRoot,
		txs:       txs,
		status:    statusProcessing,
	}
	b.bytes = make([]byte, len(raw))
	copy(b.bytes, raw)
	b.id = ids.Checksum256(b.bytes)
	return b, nil
}

// block.Block interface.

func (b *Block) ID() ids.ID           { return b.id }
func (b *Block) Parent() ids.ID       { return b.parentID }
func (b *Block) ParentID() ids.ID     { return b.parentID }
func (b *Block) Height() uint64       { return b.height }
func (b *Block) Timestamp() time.Time { return b.timestamp }
func (b *Block) Status() uint8        { return b.status }
func (b *Block) Bytes() []byte        { return b.bytes }

// Verify re-executes the block against a versiondb overlay on the committed state
// and checks the proposer's execution root. The matcher runs here (not at
// Accept), so every validator independently derives the fills and can reject a
// proposer that lied. The overlay is retained on the block for Accept to commit.
//
// Execution is deterministic: applyTx supplies block-derived IDs/timestamps,
// SubmitMarketable/ConsensusAddOrder never mint. The resulting fills, book rows,
// and root are a pure function of (committed state, this block's txs).
func (b *Block) Verify(ctx context.Context) error {
	b.vm.mu.Lock()
	defer b.vm.mu.Unlock()

	// Parent must be the current last-accepted block: the d-chain is linear.
	if b.parentID != b.vm.lastAcceptedID {
		return fmt.Errorf("dchain: block parent %s != last accepted %s", b.parentID, b.vm.lastAcceptedID)
	}
	if b.height != b.vm.lastAcceptedHeight+1 {
		return fmt.Errorf("dchain: block height %d != expected %d", b.height, b.vm.lastAcceptedHeight+1)
	}

	// THE SHIP RULE, proven on the BLOCK. Execution binds the object bytes the
	// transaction carried, which is what makes it replayable; that those bytes are the
	// real recorded object is proven here, once, against shared memory. A forged object
	// costs the producer a BLOCK instead of forking a state root — and the producer
	// verifies its own block before proposing it, so a forgery never leaves the node.
	if err := b.verifyImports(); err != nil {
		return err
	}

	overlay := versiondb.New(b.vm.db)
	result, err := b.execute(ctx, overlay)
	if err != nil {
		overlay.Abort()
		return err
	}

	// Check the proposer's claimed root against the locally derived one.
	if result.root != b.execRoot {
		overlay.Abort()
		return fmt.Errorf("dchain: execution root mismatch: claimed %x derived %x", b.execRoot[:8], result.root[:8])
	}

	b.overlay = overlay
	b.outcomes = result.outcomes
	b.atomic = result.atomic
	return nil
}

// Accept commits the verified overlay to durable zapdb atomically and advances
// the VM's accepted state + in-RAM books. If Verify was skipped (overlay nil) it
// runs execution now (the leader's own block path is Build->Verify->Accept, so
// the overlay is normally present).
func (b *Block) Accept(ctx context.Context) error {
	b.vm.mu.Lock()
	defer b.vm.mu.Unlock()

	if b.overlay == nil {
		overlay := versiondb.New(b.vm.db)
		res, err := b.execute(ctx, overlay)
		if err != nil {
			overlay.Abort()
			return err
		}
		// AND CHECK THE ROOT HERE TOO. This branch is the BOOTSTRAP path — a node
		// replaying accepted history calls Accept without Verify — and it used to
		// commit whatever it derived while persisting the block's CLAIMED root. A node
		// that derived something different therefore wrote a ledger that disagreed with
		// its own meta:root and only discovered it one block later, on the next Verify,
		// with the divergence already committed. Checking here turns "sync silently
		// onto a different chain" into "stop at the block you could not reproduce",
		// which is the only safe answer and the thing that makes a bootstrap proof
		// meaningful: reaching the tip now means having re-derived every root on the
		// way to it.
		if res.root != b.execRoot {
			overlay.Abort()
			return fmt.Errorf("dchain: execution root mismatch at accept (height %d): claimed %x derived %x",
				b.height, b.execRoot[:8], res.root[:8])
		}
		b.overlay = overlay
		b.outcomes = res.outcomes
		b.atomic = res.atomic
	}

	// Persist the accept watermarks into the SAME overlay so the commit is one
	// atomic batch: rows + meta land together or not at all.
	if err := writeLastAccepted(b.overlay, b.id); err != nil {
		b.overlay.Abort()
		return err
	}
	if err := writeHeight(b.overlay, b.height); err != nil {
		b.overlay.Abort()
		return err
	}
	if err := writeRoot(b.overlay, b.execRoot); err != nil {
		b.overlay.Abort()
		return err
	}
	// Persist the head block's canonical bytes in the SAME overlay so the head
	// pointer and the block it names commit atomically. loadHead reconstructs this
	// block on restart so GetBlock(lastAccepted) — called by the engine right after
	// Initialize — succeeds once the chain has advanced past genesis.
	if err := writeHeadBlock(b.overlay, b.bytes); err != nil {
		b.overlay.Abort()
		return err
	}

	// Commit the overlay AND the block's cross-chain operations (C->D removes /
	// D->C puts) in one atomic write (atomic.go). With no crossings this is the plain
	// overlay commit; with them the shared-memory ops land atomically with the
	// state so a consumed order / produced settlement can never diverge from the
	// committed ledger.
	if err := b.vm.commitAtomic(b.overlay, b.atomic); err != nil {
		return fmt.Errorf("dchain: commit block %s: %w", b.id, err)
	}

	// Advance in-RAM authoritative state.
	b.vm.lastAcceptedID = b.id
	b.vm.lastAcceptedHeight = b.height
	b.vm.lastRoot = b.execRoot
	b.vm.acceptedBlocks[b.id] = b
	b.vm.heightIndex[b.height] = b.id
	delete(b.vm.processingBlocks, b.id) // now accepted; drop the processing entry
	b.applyToMemBooks()
	b.status = statusAccepted
	b.overlay = nil

	// Reclaim tombstones whose in-flight height is now accepted (R4): such a tx can
	// no longer be in flight, so its tombstone is dead weight. gcTombstones takes the
	// mempool's own lock (distinct from vm.mu, no re-entry) and is a no-op on the hot
	// path (no tombstones), so this bounds tombstone growth without taxing draining.
	b.vm.mempool.gcTombstones(b.height)

	// Resolve any parked ZAP handlers with their tx's consensus outcome. This is
	// the only place an outcome becomes final: a rejected block never reaches
	// here (its txs return to the mempool). On a follower with no local waiters
	// this is a no-op.
	for _, o := range b.outcomes {
		b.vm.outcomes.resolve(o)
	}
	b.outcomes = nil
	return nil
}

// Reject aborts the overlay and returns the block's txs to the mempool for
// reconsideration in a future block.
func (b *Block) Reject(ctx context.Context) error {
	b.vm.mu.Lock()
	defer b.vm.mu.Unlock()
	if b.overlay != nil {
		b.overlay.Abort()
		b.overlay = nil
	}
	b.status = statusRejected
	delete(b.vm.processingBlocks, b.id) // rejected; drop the processing entry
	// Every tx in a rejected block came from the mempool, so every one goes back.
	b.vm.mempool.Requeue(b.txs)
	return nil
}

// execResult carries what execute derived: the per-market resulting rows, the
// block's fills, the chained execution root, and the per-tx outcomes (so Accept
// can resolve any parked ZAP handler with its tx's consensus result).
type execResult struct {
	root     [Size]byte
	fills    []dex.DEXTrade
	rows     []dex.DEXOrder  // canonical resting rows across all touched markets
	outcomes []txOutcome     // one per tx, in block order
	atomic   *atomicRequests // cross-chain removes/puts to apply at Accept
}

// execute runs the block's txs against the overlay, writing order/market/trade
// rows into it and returning the derived fills + root. It rebuilds each touched
// market's book from the overlay (so it sees prior in-block effects and committed
// state), applies the tx, and writes back the affected rows. The in-RAM cached
// books are NOT mutated here — only at Accept (applyToMemBooks) — so a rejected
// block leaves the accelerator untouched.
//
// This is the single execution path shared by Verify and Accept: one body, no
// drift between "what was verified" and "what was committed".
func (b *Block) execute(ctx context.Context, overlay *versiondb.Database) (execResult, error) {
	// Books for markets touched in this block, rebuilt from the overlay on first
	// touch so in-block ordering is honored.
	books := map[[32]byte]*dex.OrderBook{}
	bookForPool := func(poolID [32]byte) (*dex.OrderBook, error) {
		if ob, ok := books[poolID]; ok {
			return ob, nil
		}
		symbol, exists, err := readMarketSymbol(overlay, poolID)
		if err != nil {
			return nil, err
		}
		if !exists {
			return nil, fmt.Errorf("dchain: tx for unknown market %x", poolID[:8])
		}
		ob, err := rebuildBookFromDB(overlay, poolID, symbol)
		if err != nil {
			return nil, err
		}
		books[poolID] = ob
		return ob, nil
	}

	var allFills []dex.DEXTrade
	var tradeSeq uint64
	// ar accumulates the block's cross-chain operations (C->D removes / D->C
	// puts); it is applied atomically with this overlay at Accept. Empty for a block
	// with no TxImport/TxExport.
	ar := newAtomicRequests()
	outcomes := make([]txOutcome, 0, len(b.txs))
	for i, tx := range b.txs {
		// Idempotency (all tx types): a tx whose id is already committed (or
		// applied earlier in THIS block) is a deterministic no-op returning the
		// FIRST outcome verbatim — no double deposit/withdraw/fill/place on a relay
		// retry of a dropped frame. Checked from the overlay so every validator and
		// the proposer's probe decide identically.
		txID := tx.ID()
		if prior, seen, err := getSeenOutcome(overlay, txID, tx.Type); err != nil {
			return execResult{}, err
		} else if seen {
			outcomes = append(outcomes, prior)
			continue
		}

		// ---- AUTHORIZATION GATE ----
		// A money-moving tx (place/cancel/submit/deposit/withdraw) MUST be signed by
		// the account it asserts, carry that account's next nonce, and — for a cancel
		// — be the resting order's owner. The gate runs against THIS overlay BEFORE
		// any value moves, so a forged/replayed/unauthorized command mutates nothing.
		// A failure is a deterministic per-tx reject (consensus-neutral: every
		// validator decodes the same frame and the same overlay nonce/owner, so all
		// reach the identical verdict), NOT a block abort. The nonce is consumed
		// (advanced in the overlay) only on full authorization, committing atomically
		// with the tx's effects.
		if tx.Type.requiresAuth() {
			rejected, aerr := b.authorizeTx(overlay, tx)
			if aerr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d authorize: %w", i, aerr)
			}
			if rejected {
				outcomes = append(outcomes, txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusRejected})
				continue
			}
		}

		// ---- Account-scoped custody txs (no poolId) ----
		switch tx.Type {
		case TxDeposit:
			// ref is decoded for completeness but the ledger does NOT key on it:
			// the ref already entered the idempotency identity via tx.ID() above
			// (Checksum256 over the full body, ref included). The credit arithmetic
			// is orthogonal to the dedup identity — it operates on (user,asset,amount)
			// exactly as before. This is the decomplect: the funds-safety property
			// (exactly-once per originating tx) lives in the content hash; the ledger
			// math stays in its own lane.
			user, asset, amount, ref, derr := zapwire.DecodeDeposit(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d deposit decode: %w", i, derr)
			}
			// Reject a zero-ref (unbound) custody credit before touching the ledger
			// (R3). A genuine EVM deposit always threads a unique non-zero txHash; a
			// zero ref is an unidentified frame that must not mint an unbacked credit.
			if ref == ([32]byte{}) {
				return execResult{}, fmt.Errorf("dchain: tx %d deposit: %w", i, errZeroCustodyRef)
			}
			realized, serr := creditDeposit(overlay, user, asset, amount)
			if serr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d deposit: %w", i, serr)
			}
			oc := txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusPlaced, orderID: realized}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue
		case TxWithdraw:
			// ref decoded for completeness; see the deposit note — the ref is in the
			// txID/seen: identity (so two GENUINELY distinct withdraws no longer
			// collide on seen: and replay a stale realized amount against an
			// already-paid vault), while debitWithdraw clamps to CURRENT available
			// independent of the ref.
			user, asset, amount, ref, derr := zapwire.DecodeWithdraw(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d withdraw decode: %w", i, derr)
			}
			// Reject a zero-ref (unbound) custody debit before touching the ledger or
			// releasing the vault (R3). A genuine EVM withdraw always threads a unique
			// non-zero txHash; a zero ref is an unidentified frame that must not
			// release value against an untracked identity.
			if ref == ([32]byte{}) {
				return execResult{}, fmt.Errorf("dchain: tx %d withdraw: %w", i, errZeroCustodyRef)
			}
			realized, serr := debitWithdraw(overlay, user, asset, amount)
			if serr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d withdraw: %w", i, serr)
			}
			// orderID carries the REALIZED amount the proxy must export (clamped to
			// available). A zero realized withdraw is StatusRejected (nothing to
			// export) so the caller does not build an empty export.
			st := uint8(zapwire.StatusPlaced)
			if realized == 0 {
				st = zapwire.StatusRejected
			}
			oc := txOutcome{txID: txID, typ: tx.Type, status: st, orderID: realized}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue

		case TxImport:
			// CONSUME a C->D funding claim and credit its beneficiary (atomic.go).
			// Authority is the claim itself, not a signature (TxImport.requiresAuth()
			// is false), so the auth gate above skipped it: a claim can only credit
			// the account it already names. The claim's BYTES ride in the body, so
			// this binds them and never reads shared memory — execution is a pure
			// function of the block and replays byte-identically forever.
			// Block.verifyImports proves those bytes are the real recorded object.
			claimID, object, derr := decodeImportBody(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d import decode: %w", i, derr)
			}
			credited, ok, ierr := b.vm.executeImport(overlay, ar, claimID, object)
			if ierr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d import: %w", i, ierr)
			}
			if !ok {
				// Deterministic per-tx reject (claim not flushed yet / replay / unwired
				// rail). NOT marked seen, so a not-yet-available claim retries later.
				outcomes = append(outcomes, txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusRejected})
				continue
			}
			oc := txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusPlaced, orderID: credited}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue

		case TxExport:
			// DEBIT the signer's available balance and write a C-bound claim the
			// 0x9999 precompile consumes exactly once (atomic.go). The auth gate above
			// already bound the body's owner to the recovered signer, so this can only
			// ever spend the signer's own money. The exporting tx id seeds the
			// deterministic claim id.
			owner, asset, amount, beneficiary, derr := decodeExportBody(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d export decode: %w", i, derr)
			}
			claimID, ok, eerr := b.vm.executeExport(overlay, ar, txID, owner, asset, amount, beneficiary)
			if eerr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d export: %w", i, eerr)
			}
			if !ok {
				outcomes = append(outcomes, txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusRejected})
				continue
			}
			// orderID carries a compact handle of the claim id for the ack; the full id
			// is recomputable from the export tx via DeriveClaimID.
			oc := txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusPlaced, orderID: binary.BigEndian.Uint64(claimID[24:32])}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue
		}

		// ---- Market-scoped txs (begin with poolId) ----
		poolID, ok := tx.poolID()
		if !ok {
			return execResult{}, fmt.Errorf("dchain: tx %d missing poolId", i)
		}

		switch tx.Type {
		case TxEnsureMarket:
			// Idempotent market creation: record existence keyed by poolId. The
			// symbol is the hex of the poolId (the venue's market identity is the
			// poolId; a human symbol is a display concern handled at the gateway).
			if _, exists, err := readMarketSymbol(overlay, poolID); err != nil {
				return execResult{}, err
			} else if !exists {
				if err := writeMarket(overlay, poolID, poolSymbol(poolID)); err != nil {
					return execResult{}, err
				}
			}
			oc := outcomeFromApply(tx, applyResult{})
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue
		case TxOpenMarket:
			pid, base, quote, derr := zapwire.DecodeOpenMarket(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d open-market decode: %w", i, derr)
			}
			// Ensure the market exists (idempotent) and bind its assets.
			if _, exists, err := readMarketSymbol(overlay, pid); err != nil {
				return execResult{}, err
			} else if !exists {
				if err := writeMarket(overlay, pid, poolSymbol(pid)); err != nil {
					return execResult{}, err
				}
			}
			if err := writeMarketAssets(overlay, pid, base, quote); err != nil {
				return execResult{}, err
			}
			oc := txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusPlaced}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue
		}

		ob, err := bookForPool(poolID)
		if err != nil {
			return execResult{}, err
		}

		// ---- CUSTODY GATE: lock the order's spend BEFORE matching ----
		// A place/submit can only proceed if the account has the AVAILABLE balance
		// to fund it. The lock moves available -> locked for the spend asset; the
		// matcher then draws from the locked balance (a fill spends locked, a
		// cancel/refund unlocks it). If the market's assets are bound and the lock
		// cannot be funded, the order is REJECTED here — it never matches, never
		// rests. (A market with unbound assets falls back to the no-custody path so
		// asset-less test/legacy markets still function; the live custody e2e binds
		// assets via dex_open_market.)
		base, quote, assetsBound, aerr := readMarketAssets(overlay, poolID)
		if aerr != nil {
			return execResult{}, aerr
		}
		locked, lockedOK, lerr := b.lockOrderSpend(overlay, tx, uint32(i), poolID, base, quote, assetsBound)
		if lerr != nil && !errors.Is(lerr, ErrDustPriceOrZeroLock) {
			// A real lock fault (decode / db) aborts the block; only the named
			// zero-notional reject below is a deterministic per-tx outcome.
			return execResult{}, fmt.Errorf("dchain: tx %d lock: %w", i, lerr)
		}
		if assetsBound && !lockedOK {
			// Order not funded for matching: either insufficient AVAILABLE balance, or
			// the market-constraint reject (ErrDustPriceOrZeroLock — an executable order
			// whose notional floors to zero lock). Both are DETERMINISTIC per-tx
			// rejects: no match, no rest, no lock held. The reason is the same frozen
			// function of state+wire on every validator, so the reject is
			// consensus-neutral. (Status is StatusRejected either way; the named error
			// documents the zero-notional case and is what the placement tests assert.)
			oc := txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusRejected}
			outcomes = append(outcomes, oc)
			if err := markSeen(overlay, txID, oc); err != nil {
				return execResult{}, err
			}
			continue
		}

		res, err := applyTx(ob, tx, b.height, b.timestamp, uint32(i))
		if err != nil {
			return execResult{}, err
		}

		// ---- SETTLE: move value for this tx's effects inside the ledger ----
		if assetsBound {
			if serr := b.settleOrderEffects(overlay, poolID, tx, uint32(i), res, base, quote, locked); serr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d settle: %w", i, serr)
			}
		}

		outcome := outcomeFromApply(tx, res)
		outcomes = append(outcomes, outcome)
		// Record the outcome under the seen-key so a later replay returns it
		// verbatim (commits atomically with the tx's effects in this overlay).
		if err := markSeen(overlay, txID, outcome); err != nil {
			return execResult{}, err
		}

		// Persist resting deltas to the overlay.
		switch {
		case res.Placed != nil:
			if err := putOrderRow(overlay, poolID, dex.OrderToRow(res.Placed)); err != nil {
				return execResult{}, err
			}
		case res.Canceled != 0:
			if err := deleteOrderRow(overlay, poolID, res.Canceled); err != nil {
				return execResult{}, err
			}
		}
		// A submit touches makers: rewrite/delete their rows.
		for _, m := range res.Touched {
			if m.ID == 0 {
				continue
			}
			row := dex.OrderToRow(m)
			row.OrderID = m.ID
			if err := putOrderRow(overlay, poolID, row); err != nil {
				return execResult{}, err
			}
		}
		// A submit that self-crossed cancelled its own resting makers: delete their
		// order: rows so the durable book matches the matcher's in-memory removal (the
		// reserve/orderuser: unlock is done in settleOrderEffects, in lockstep).
		for _, cid := range res.SelfCanceled {
			if cid == 0 {
				continue
			}
			if err := deleteOrderRow(overlay, poolID, cid); err != nil {
				return execResult{}, err
			}
		}
		// Record fills in the trade log + the block's fill set.
		for _, f := range res.Fills {
			row := dex.TradeToRow(f, res.TakerSide)
			if err := overlay.Put(tradeKey(b.height, tradeSeq), dex.EncodeTrade(row)); err != nil {
				return execResult{}, err
			}
			tradeSeq++
			allFills = append(allFills, row)
		}
	}

	// Persist each touched market's LastOrderID watermark and gather the
	// canonical resting rows for the book root.
	var allRows []dex.DEXOrder
	for poolID, ob := range books {
		if err := writeBookWatermark(overlay, poolID, ob.LastOrderID); err != nil {
			return execResult{}, err
		}
		allRows = append(allRows, dex.BookToRows(ob)...)
	}
	sortRowsCanonical(allRows)

	// Commit the CUSTODY LEDGER (balance:/locked:/orderasset:) into the execution
	// root so validators cannot finalize agreeing on the book while holding different
	// money — the money is a term in consensus, not an off-root side effect.
	lroot, lerr := ledgerRoot(overlay)
	if lerr != nil {
		return execResult{}, lerr
	}

	root, _, _, _ := ExecutionRoot(b.vm.lastRoot, allRows, allFills, b.txs, lroot, b.height)
	return execResult{root: root, fills: allFills, rows: allRows, outcomes: outcomes, atomic: ar}, nil
}

// lockOrderSpend reserves the asset a place/submit will spend, moving it from the
// account's AVAILABLE to LOCKED balance BEFORE matching. It is the custody gate:
// returns ok=false (and locks nothing) when the account cannot fund the order, so
// the caller rejects it without matching or resting. Returns the locked amount so
// the caller can refund/spend it precisely.
//
//   - PLACE: locks the full spend (buy -> ceil(size*price) quote, sell -> size
//     base) and records a per-order reserve (orderasset:) so a later cancel or a
//     maker fill unlocks/spends the exact remaining amount. A rested limit order
//     never crosses (ConsensusAddOrder), so its lock stays held until filled or
//     cancelled.
//   - SUBMIT: locks the taker's spend. A LIMIT submit locks the exact want
//     (fills occur at maker prices within the limit, so spend <= lock). A MARKET
//     submit locks ALL available of the spend asset (the worst-case spend is
//     bounded by the balance), and the unspent remainder is refunded after the
//     cross. No per-order reserve (IOC never rests).
//   - other tx types: no lock (returns 0, true).
//
// txIndex is the tx's position in the block — the SAME index the matcher uses to
// derive a placed order's deterministic id (applyTx -> blockDeterministicID). It is
// threaded in (not recovered by scanning b.txs) so the per-order reserve/identity
// rows key under the EXACT order id the matcher assigned, with no possibility of a
// pointer-aliased tx resolving to the wrong index (which would key the orderuser:/
// orderasset: rows under a different order id than the resting order, stranding the
// lock on cancel and tripping ErrMissingSettlementUser on a maker fill).
func (b *Block) lockOrderSpend(db *versiondb.Database, tx *Tx, txIndex uint32, poolID [32]byte, base, quote [32]byte, assetsBound bool) (uint64, bool, error) {
	if !assetsBound {
		return 0, true, nil // no-custody fallback for asset-less markets
	}
	switch tx.Type {
	case TxPlace:
		_, side, price, size, user, derr := zapwire.DecodePlace(tx.Body)
		if derr != nil {
			return 0, false, derr
		}
		// MARKET-CONSTRAINT GATE: an executable order whose notional floors to zero
		// lock is rejected with the NAMED reason (a sub-tick BUY price, or a sub-unit
		// SELL size). This is the "no free executable order" invariant — a zero-lock
		// order would rest/cross reserving NOTHING. A deterministic per-tx reject
		// (execute maps ErrDustPriceOrZeroLock to StatusRejected), not a block abort.
		if (marketConstraint{base, quote}).floorsToZeroLock(side, price, size) {
			return 0, false, ErrDustPriceOrZeroLock
		}
		lockAsset, lockAmt := orderLock(side, price, size, base, quote)
		if lockAmt == 0 {
			// Defense-in-depth: any remaining zero-lock executable order (a degenerate
			// size the predicate did not name) is rejected too. A resting order on a
			// custody market MUST carry BOTH a non-zero lock AND an orderuser: identity
			// row; allowing a zero-lock order to rest would leave a book row with no
			// orderuser: mapping — exactly the degraded path settlement must never hit.
			return 0, false, ErrDustPriceOrZeroLock
		}
		uid := userKey16(user)
		if err := lockFromAvailable(db, uid, lockAsset, lockAmt); err != nil {
			if errors.Is(err, ErrInsufficientAvailable) {
				return 0, false, nil
			}
			return 0, false, err
		}
		// Record the per-order reserve so cancel/fill unlocks the exact amount, and
		// the order's FULL settlement identity so a maker fill/cancel credits the
		// right account even after a restart (the GPU row carries only 8 user bytes).
		orderID := blockDeterministicID(b.height, txIndex)
		if err := putOrderLock(db, poolID, orderID, lockAsset, lockAmt); err != nil {
			return 0, false, err
		}
		if err := putOrderUser(db, poolID, orderID, uid); err != nil {
			return 0, false, err
		}
		return lockAmt, true, nil

	case TxSubmit:
		_, side, isMarket, limitPrice, size, user, derr := zapwire.DecodeSubmit(tx.Body)
		if derr != nil {
			return 0, false, derr
		}
		uid := userKey16(user)
		var lockAsset [32]byte
		var lockAmt uint64
		if side == sideBuy {
			lockAsset = quote
		} else {
			lockAsset = base
		}
		if isMarket && side == sideBuy {
			// Market buy: spend is unbounded by a limit -> lock all available quote;
			// the cross spends what it fills, the remainder is refunded.
			avail, err := getAvailable(db, uid, lockAsset)
			if err != nil {
				return 0, false, err
			}
			lockAmt = avail
		} else {
			// MARKET-CONSTRAINT GATE (limit submit): an executable LIMIT order whose
			// notional floors to zero lock is the "free executable order" hazard — a
			// sub-tick BUY price (ceil(size*price)==0) or a sub-unit SELL size could
			// cross a resting maker while reserving NOTHING, acquiring the maker's
			// asset for no spend. Reject it with the named reason BEFORE locking.
			// (A MARKET buy is excluded above: its zero-lock means zero balance, an
			// affordability reject, not a dust price.)
			if (marketConstraint{base, quote}).floorsToZeroLock(side, limitPrice, size) {
				return 0, false, ErrDustPriceOrZeroLock
			}
			_, lockAmt = orderLock(side, limitPrice, size, base, quote)
		}
		if lockAmt == 0 {
			// Nothing affordable to lock: a market buy with zero quote, or a zero
			// size. A SUBMIT is IOC — it NEVER rests, so (unlike a place) it creates
			// no resting book row and needs no orderuser: identity row; the "every
			// resting custody order has a lock + identity" invariant does not apply.
			// The taker's own settlement identity is always the live, full 16-byte
			// submit user, never an 8-byte handle. A zero lock means the matcher can
			// produce no settle-able fills (its exact-integer value lane is zero), so
			// settle is a no-op; treat as funded with a zero lock.
			return 0, true, nil
		}
		if err := lockFromAvailable(db, uid, lockAsset, lockAmt); err != nil {
			if errors.Is(err, ErrInsufficientAvailable) {
				return 0, false, nil
			}
			return 0, false, err
		}
		return lockAmt, true, nil

	default:
		return 0, true, nil
	}
}

// settleOrderEffects moves value for an order tx's matcher result inside the
// ledger, AFTER matching, against the same overlay. lockedAmt is what
// lockOrderSpend reserved for this tx (the taker's lock for a submit; the place's
// lock is held under its per-order reserve).
//
//   - CANCEL: unlock the cancelled order's remaining per-order reserve.
//   - PLACE rejected by the matcher (post-only would take) AFTER we locked:
//     refund the lock (unlock + delete the per-order reserve). A rested place
//     keeps its lock (held under the per-order reserve recorded at lock time).
//   - SUBMIT: settle each fill (taker locked spend -> maker available; maker
//     locked -> taker available), decrement each touched maker's per-order
//     reserve by the filled amount, then unlock the taker's unspent remainder.
//
// txIndex is the tx's block position (the matcher's order-id index), threaded in
// for the same reason as lockOrderSpend: the TxPlace refund path must delete the
// reserve/identity rows under the EXACT order id the lock recorded them under.
func (b *Block) settleOrderEffects(db *versiondb.Database, poolID [32]byte, tx *Tx, txIndex uint32, res applyResult, base, quote [32]byte, lockedAmt uint64) error {
	switch tx.Type {
	case TxCancel:
		// A cancel releases the resting order's reserve to its owner and drops its
		// identity row — the ONE canonical cancel-persist path (releaseOrderReserve),
		// shared with the self-trade-prevented removals a submit performs.
		return b.releaseOrderReserve(db, poolID, res.Canceled)

	case TxPlace:
		if res.Placed != nil {
			return nil // rested: lock stays held under the per-order reserve
		}
		// The matcher rejected the place AFTER we locked: refund the lock fully.
		_, side, price, size, user, derr := zapwire.DecodePlace(tx.Body)
		if derr != nil {
			return derr
		}
		lockAsset, lockAmt := orderLock(side, price, size, base, quote)
		if lockAmt == 0 {
			return nil
		}
		uid := userKey16(user)
		if err := unlockToAvailable(db, uid, lockAsset, lockAmt); err != nil {
			return err
		}
		// Remove the per-order reserve and identity row recorded at lock time.
		orderID := blockDeterministicID(b.height, txIndex)
		if err := putOrderLock(db, poolID, orderID, lockAsset, 0); err != nil {
			return err
		}
		return deleteOrderUser(db, poolID, orderID)

	case TxSubmit:
		_, side, _, _, _, user, derr := zapwire.DecodeSubmit(tx.Body)
		if derr != nil {
			return derr
		}
		uid := userKey16(user)
		// Move value for the fills (uses the matcher's exact integer lane). poolID
		// lets settleFills resolve each maker's FULL settlement identity from the
		// orderuser: row (restart-safe), not the 8-byte in-RAM maker string.
		spent, err := settleFills(db, poolID, side, base, quote, res.Fills)
		if err != nil {
			return err
		}
		// Makers the cross fully filled left the book (applySubmit tombstones them with
		// Status=Filled). Their per-order reserve is CLOSED and any ceil−floor residual
		// returned to the owner (a filled order can never be cancelled). Partial makers
		// keep their decremented reserve for a later cancel.
		fullyFilled := make(map[uint64]struct{}, len(res.Touched))
		for _, m := range res.Touched {
			if m != nil && m.ID != 0 && m.Status == dex.Filled {
				fullyFilled[m.ID] = struct{}{}
			}
		}
		if err := b.decrementMakerReserves(db, poolID, side, base, quote, res.Fills, fullyFilled); err != nil {
			return err
		}
		// Refund the taker's unspent lock remainder (IOC never rests).
		if lockedAmt > spent {
			var lockAsset [32]byte
			if side == sideBuy {
				lockAsset = quote
			} else {
				lockAsset = base
			}
			if err := unlockToAvailable(db, uid, lockAsset, lockedAmt-spent); err != nil {
				return err
			}
		}
		// Release any self-trade-prevented makers this cross removed: the taker
		// crossed its OWN resting liquidity, so those makers leave the book with no
		// fill and their reserve returns to the owner (locked -> available) via the
		// SAME cancel-persist path a TxCancel uses. Conservation-exact: no value is
		// created or destroyed, the reserve simply moves back to available.
		for _, cid := range res.SelfCanceled {
			if err := b.releaseOrderReserve(db, poolID, cid); err != nil {
				return err
			}
		}
		return nil

	default:
		return nil
	}
}

// releaseOrderReserve returns a resting order's per-order reserve to its recorded
// owner (locked -> available) and deletes its reserve + orderuser: identity rows —
// the single canonical cancel-persist primitive, shared by TxCancel and by a
// submit's self-trade-prevented maker removals. A zero id or an order with no
// recorded reserve is a no-op (nothing locked to return). The owner is resolved
// EXCLUSIVELY through the orderuser: index (the full 16-byte settlement identity),
// never the matcher's 8-byte in-RAM handle: a reserve with no orderuser: row is
// state corruption and fails closed (block rejected) rather than unlocking to a
// compact-handle-derived account — a collision there would be value-bearing theft.
func (b *Block) releaseOrderReserve(db *versiondb.Database, poolID [32]byte, orderID uint64) error {
	if orderID == 0 {
		return nil // unknown/no order: nothing to unlock
	}
	asset, amt, ok, err := getOrderLock(db, poolID, orderID)
	if err != nil {
		return err
	}
	if !ok || amt == 0 {
		return nil // no reserve recorded (pre-custody / asset-less market)
	}
	owner, ok, uerr := getOrderUser(db, poolID, orderID)
	if uerr != nil {
		return uerr
	}
	if !ok {
		return fmt.Errorf("%w: release of order %d in market %x", ErrMissingSettlementUser, orderID, poolID[:8])
	}
	if err := unlockToAvailable(db, owner, asset, amt); err != nil {
		return err
	}
	if err := putOrderLock(db, poolID, orderID, asset, 0); err != nil { // delete the reserve
		return err
	}
	return deleteOrderUser(db, poolID, orderID) // delete the identity row
}

// decrementMakerReserves reduces each filled maker's per-order reserve by the
// amount of its locked asset the cross consumed, so cancel-time unlock matches
// the remaining resting size. A maker on the opposite side of the taker locked:
// taker BUY -> maker SOLD base (reserve in base, reduced by fill base units);
// taker SELL -> maker BOUGHT base (reserve in quote, reduced by fill quote units).
//
// fullyFilled names the makers that LEFT THE BOOK on this cross (a fully-filled
// order the matcher removed). For those the per-order reserve is CLOSED here and
// any residual is RETURNED to the owner: a resting BUY locks ceil(size*price) quote
// but each fill charges floor(base*price) (settlement_units.go quoteUnitsFor), so
// after a full fill a ceil−floor residual remains in the reserve. A filled order
// can never be cancelled, so that residual would be FROZEN in locked forever. We
// release it (locked -> available) through the SAME primitive a cancel uses, so a
// full fill conserves the maker's over-lock exactly like a cancel would. A maker
// that is still resting (partial fill) keeps its decremented reserve for a later
// cancel.
func (b *Block) decrementMakerReserves(db *versiondb.Database, poolID [32]byte, takerSide uint8, base, quote [32]byte, fills []dex.Trade, fullyFilled map[uint64]struct{}) error {
	// Aggregate the consumed reserve per maker order id (a maker may fill across
	// several fills in one cross, though typically one).
	consumed := map[uint64]uint64{}
	for _, f := range fills {
		var makerOrderID, amt uint64
		if takerSide == sideBuy {
			makerOrderID = f.SellOrder
			if f.BaseUnits != nil && f.BaseUnits.IsUint64() {
				amt = f.BaseUnits.Uint64() // maker reserve is in base
			}
		} else {
			makerOrderID = f.BuyOrder
			if f.QuoteUnits != nil && f.QuoteUnits.IsUint64() {
				amt = f.QuoteUnits.Uint64() // maker reserve is in quote
			}
		}
		consumed[makerOrderID] += amt
	}
	for makerOrderID, amt := range consumed {
		asset, reserve, ok, err := getOrderLock(db, poolID, makerOrderID)
		if err != nil {
			return err
		}
		if !ok {
			continue // maker placed before custody / no reserve recorded
		}
		residual := uint64(0)
		if reserve > amt {
			residual = reserve - amt
		}
		if _, left := fullyFilled[makerOrderID]; left {
			// The order left the book: return the ceil−floor residual to the owner
			// (it can never be cancelled) and CLOSE the per-order rows. A zero residual
			// (e.g. a SELL maker whose base lock is exact) makes the unlock a no-op.
			owner, hasOwner, uerr := getOrderUser(db, poolID, makerOrderID)
			if uerr != nil {
				return uerr
			}
			if residual > 0 {
				if !hasOwner {
					return fmt.Errorf("%w: residual release for filled maker %d in market %x", ErrMissingSettlementUser, makerOrderID, poolID[:8])
				}
				if uerr := unlockToAvailable(db, owner, asset, residual); uerr != nil {
					return uerr
				}
			}
			if err := putOrderLock(db, poolID, makerOrderID, asset, 0); err != nil {
				return err
			}
			if err := deleteOrderUser(db, poolID, makerOrderID); err != nil {
				return err
			}
			continue
		}
		// Still resting (partial fill): keep the decremented reserve for a later cancel.
		if err := putOrderLock(db, poolID, makerOrderID, asset, residual); err != nil {
			return err
		}
	}
	return nil
}

// applyToMemBooks replays the block's resting effects onto the VM's cached in-RAM
// books at Accept. The cache is an accelerator; the durable rows are truth. We
// rebuild each touched market's cached book from the just-committed overlay state
// so the cache is exactly consistent with disk.
func (b *Block) applyToMemBooks() {
	touched := map[[32]byte]struct{}{}
	for _, tx := range b.txs {
		if poolID, ok := tx.poolID(); ok {
			touched[poolID] = struct{}{}
		}
	}
	for poolID := range touched {
		symbol, exists, err := readMarketSymbol(b.vm.db, poolID)
		if err != nil || !exists {
			continue
		}
		ob, err := rebuildBookFromDB(b.vm.db, poolID, symbol)
		if err != nil {
			continue
		}
		b.vm.books[poolID] = ob
	}
}

// rebuildBookFromDB streams the order:<poolID> rows from db and folds them into a
// fresh OrderBook via RowsToBook — the rebuildable-accelerator path. This is the
// SAME function used at startup (rebuild every book) and during execution (rebuild
// one market into the overlay), so there is one rebuild path.
func rebuildBookFromDB(db database.Iteratee, poolID [32]byte, symbol string) (*dex.OrderBook, error) {
	prefix := orderPrefixFor(poolID)
	it := db.NewIteratorWithPrefix(prefix)
	defer it.Release()

	var rows []dex.DEXOrder
	for it.Next() {
		row, ok := dex.DecodeRow(it.Value())
		if !ok {
			return nil, fmt.Errorf("dchain: corrupt order row under %x (len %d)", poolID[:8], len(it.Value()))
		}
		rows = append(rows, row)
	}
	if err := it.Error(); err != nil {
		return nil, err
	}
	ob := dex.RowsToBook(symbol, rows)
	// The persisted DEXOrder row stores the EXACT atomic-unit quantity and PriceInt
	// (persist.go), and RowToOrder restores SizeUnits/RemainingUnits/PriceUnits from
	// those integers verbatim — so a maker rebuilt from disk carries the identical
	// value lane the live one had, and settles a taker's cross to the unit. There is
	// no float re-derivation (the old restoreIntegerLane), so a rebuilt book is
	// byte-identical to the live one — a restart cannot fork the bookRoot.
	return ob, nil
}

// poolSymbol renders a poolId as its hex market symbol.
func poolSymbol(poolID [32]byte) string {
	const hexdigits = "0123456789abcdef"
	out := make([]byte, 64)
	for i, c := range poolID {
		out[i*2] = hexdigits[c>>4]
		out[i*2+1] = hexdigits[c&0x0f]
	}
	return string(out)
}

// verifyImports authenticates every funding claim this block's imports CARRIED,
// against the object shared memory actually holds. It is the block-level half
// of the reproducibility fix: execution binds tx-carried bytes (pure, replayable
// forever), and the proof that those bytes are real lives here, where a mismatch
// rejects the BLOCK rather than diverging a state root.
//
// This is the primary network's own import discipline — object in the transaction,
// shared memory consulted at Verify, consumption applied at Accept — and it is the same
// shape evm/plugin/evm/dex_atomic_verify.go gives the C side.
//
// THE BOOTSTRAP GATE IS REQUIRED, NOT AN OPTIMIZATION. A node replaying history has not
// necessarily applied the exporting chain's block yet: C and D bootstrap independently,
// so below the frontier the C->D object may legitimately be absent — not yet exported,
// or already consumed by this very block's own accepted Remove. Blocks below the
// frontier were already validated by the network under its finality rule; that
// acceptance is their authority. Above the frontier the object must be there, and must
// match byte for byte.
//
// KNOWN, INHERENT LIVENESS PROPERTY (shared with every shared-memory import model,
// coreth's ImportTx included): a validator caught up on D but LAGGING on C will not yet
// see the C-side flush and will reject an otherwise-valid D block. It does not halt —
// the block is rejected, not fatal, and re-verification succeeds once C catches up. The
// drive only ever emits an import for an object it just read out of shared memory, so
// in practice the producer has already observed it.
func (b *Block) verifyImports() error {
	if !b.vm.normalOp || b.vm.cChainID == ids.Empty {
		return nil
	}
	sm := b.vm.sharedMemory()
	for _, tx := range b.txs {
		if tx.Type != TxImport {
			continue
		}
		claimID, object, err := decodeImportBody(tx.Body)
		if err != nil {
			return fmt.Errorf("dchain: block %s: %w", b.id, err)
		}
		// NO HANDLE, NO PROOF, NO BLOCK. This used to return nil here for the whole
		// block, on the reasoning that execution rejects an import without a C chain id.
		// It rejects without a C chain ID, and the plugin server supplies one
		// unconditionally — so the reasoning named a condition that is never the one
		// that holds, and the authenticator waved through precisely the imports it could
		// not authenticate. A node with no shared memory cannot tell a recorded object
		// from a fabricated one; the only honest answer is to refuse the block. Blocks
		// carrying no import are unaffected, because the check is per-import: an absent
		// rail is not an error, crossing on one is.
		if sm == nil {
			return fmt.Errorf("dchain: import unprovable in block %s (height %d): claim %s carried with no shared memory wired",
				b.id, b.height, claimID)
		}
		vals, gerr := sm.Get(b.vm.cChainID, [][]byte{claimID[:]})
		if gerr != nil || len(vals) != 1 || len(vals[0]) == 0 {
			return fmt.Errorf("dchain: import unbacked in block %s (height %d): claim %s is not in shared memory: %w",
				b.id, b.height, claimID, gerr)
		}
		if !bytes.Equal(vals[0], object) {
			return fmt.Errorf("dchain: import forged in block %s (height %d): claim %s declared %x but shared memory holds %x",
				b.id, b.height, claimID, object[:8], vals[0][:min(8, len(vals[0]))])
		}
	}
	return nil
}

