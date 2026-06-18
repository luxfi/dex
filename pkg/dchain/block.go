// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"time"

	"github.com/luxfi/database"
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/dex/pkg/lx"
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

func (b *Block) ID() ids.ID            { return b.id }
func (b *Block) Parent() ids.ID        { return b.parentID }
func (b *Block) ParentID() ids.ID      { return b.parentID }
func (b *Block) Height() uint64        { return b.height }
func (b *Block) Timestamp() time.Time  { return b.timestamp }
func (b *Block) Status() uint8         { return b.status }
func (b *Block) Bytes() []byte         { return b.bytes }

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
		b.overlay = overlay
		b.outcomes = res.outcomes
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

	if err := b.overlay.Commit(); err != nil {
		return fmt.Errorf("dchain: commit block %s: %w", b.id, err)
	}

	// Advance in-RAM authoritative state.
	b.vm.lastAcceptedID = b.id
	b.vm.lastAcceptedHeight = b.height
	b.vm.lastRoot = b.execRoot
	b.vm.acceptedBlocks[b.id] = b
	b.vm.heightIndex[b.height] = b.id
	b.applyToMemBooks()
	b.status = statusAccepted
	b.overlay = nil

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
	b.vm.mempool.Requeue(b.txs)
	return nil
}

// execResult carries what execute derived: the per-market resulting rows, the
// block's fills, the chained execution root, and the per-tx outcomes (so Accept
// can resolve any parked ZAP handler with its tx's consensus result).
type execResult struct {
	root     [Size]byte
	fills    []lx.DEXTrade
	rows     []lx.DEXOrder // canonical resting rows across all touched markets
	outcomes []txOutcome   // one per tx, in block order
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
	books := map[[32]byte]*lx.OrderBook{}
	bookForPool := func(poolID [32]byte) (*lx.OrderBook, error) {
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

	var allFills []lx.DEXTrade
	var tradeSeq uint64
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

		// ---- Account-scoped custody txs (no poolId) ----
		switch tx.Type {
		case TxDeposit:
			user, asset, amount, derr := zapwire.DecodeDeposit(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d deposit decode: %w", i, derr)
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
			user, asset, amount, derr := zapwire.DecodeWithdraw(tx.Body)
			if derr != nil {
				return execResult{}, fmt.Errorf("dchain: tx %d withdraw decode: %w", i, derr)
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
		// assets via clob_open_market.)
		base, quote, assetsBound, aerr := readMarketAssets(overlay, poolID)
		if aerr != nil {
			return execResult{}, aerr
		}
		locked, lockedOK, lerr := b.lockOrderSpend(overlay, tx, poolID, base, quote, assetsBound)
		if lerr != nil {
			return execResult{}, fmt.Errorf("dchain: tx %d lock: %w", i, lerr)
		}
		if assetsBound && !lockedOK {
			// Insufficient available balance to fund the order: deterministic reject,
			// no match, no rest, no lock held.
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
			if serr := b.settleOrderEffects(overlay, poolID, tx, res, base, quote, locked); serr != nil {
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
			if err := putOrderRow(overlay, poolID, lx.OrderToRow(res.Placed)); err != nil {
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
			row := lx.OrderToRow(m)
			row.OrderID = m.ID
			if err := putOrderRow(overlay, poolID, row); err != nil {
				return execResult{}, err
			}
		}
		// Record fills in the trade log + the block's fill set.
		for _, f := range res.Fills {
			row := lx.TradeToRow(f, res.TakerSide)
			if err := overlay.Put(tradeKey(b.height, tradeSeq), lx.EncodeTrade(row)); err != nil {
				return execResult{}, err
			}
			tradeSeq++
			allFills = append(allFills, row)
		}
	}

	// Persist each touched market's LastOrderID watermark and gather the
	// canonical resting rows for the book root.
	var allRows []lx.DEXOrder
	for poolID, ob := range books {
		if err := writeBookWatermark(overlay, poolID, ob.LastOrderID); err != nil {
			return execResult{}, err
		}
		allRows = append(allRows, lx.BookToRows(ob)...)
	}
	sortRowsCanonical(allRows)

	root, _, _, _ := ExecutionRoot(b.vm.lastRoot, allRows, allFills, b.txs, b.height)
	return execResult{root: root, fills: allFills, rows: allRows, outcomes: outcomes}, nil
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
func (b *Block) lockOrderSpend(db *versiondb.Database, tx *Tx, poolID [32]byte, base, quote uint64, assetsBound bool) (uint64, bool, error) {
	if !assetsBound {
		return 0, true, nil // no-custody fallback for asset-less markets
	}
	switch tx.Type {
	case TxPlace:
		_, side, price, size, user, derr := zapwire.DecodePlace(tx.Body)
		if derr != nil {
			return 0, false, derr
		}
		lockAsset, lockAmt := orderLock(side, price, size, base, quote)
		if lockAmt == 0 {
			// Degenerate order (zero size / unrepresentable lock): let the matcher
			// reject it, lock nothing.
			return 0, true, nil
		}
		uid := userID8(user)
		if err := lockFromAvailable(db, uid, lockAsset, lockAmt); err != nil {
			if errors.Is(err, ErrInsufficientAvailable) {
				return 0, false, nil
			}
			return 0, false, err
		}
		// Record the per-order reserve so cancel/fill unlocks the exact amount.
		orderID := blockDeterministicID(b.height, txIndexOf(b, tx))
		if err := putOrderLock(db, poolID, orderID, lockAsset, lockAmt); err != nil {
			return 0, false, err
		}
		return lockAmt, true, nil

	case TxSubmit:
		_, side, isMarket, limitPrice, size, user, derr := zapwire.DecodeSubmit(tx.Body)
		if derr != nil {
			return 0, false, derr
		}
		uid := userID8(user)
		var lockAsset, lockAmt uint64
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
			_, lockAmt = orderLock(side, limitPrice, size, base, quote)
		}
		if lockAmt == 0 {
			// Nothing affordable to lock: a market buy with zero quote, or a zero
			// size. The matcher will produce no settle-able fills; treat as funded
			// with a zero lock (the cross yields nothing, settle is a no-op).
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
func (b *Block) settleOrderEffects(db *versiondb.Database, poolID [32]byte, tx *Tx, res applyResult, base, quote, lockedAmt uint64) error {
	switch tx.Type {
	case TxCancel:
		if res.Canceled == 0 {
			return nil // unknown cancel: no reserve to unlock
		}
		asset, amt, ok, err := getOrderLock(db, poolID, res.Canceled)
		if err != nil {
			return err
		}
		if !ok || amt == 0 {
			return nil
		}
		// The cancelled order's owner reserved `amt` of `asset`; return it.
		owner := userID8(canceledOwner(res))
		if err := unlockToAvailable(db, owner, asset, amt); err != nil {
			return err
		}
		return putOrderLock(db, poolID, res.Canceled, asset, 0) // delete the reserve

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
		uid := userID8(user)
		if err := unlockToAvailable(db, uid, lockAsset, lockAmt); err != nil {
			return err
		}
		// Remove the per-order reserve recorded at lock time.
		orderID := blockDeterministicID(b.height, txIndexOf(b, tx))
		return putOrderLock(db, poolID, orderID, lockAsset, 0)

	case TxSubmit:
		_, side, _, _, _, user, derr := zapwire.DecodeSubmit(tx.Body)
		if derr != nil {
			return derr
		}
		uid := userID8(user)
		// Move value for the fills (uses the matcher's exact integer lane).
		spent, err := settleFills(db, side, base, quote, res.Fills)
		if err != nil {
			return err
		}
		// Decrement each touched maker's per-order reserve by the base/quote it
		// spent on this cross, so a later cancel of a partially-filled maker unlocks
		// only the still-resting remainder (a fully-filled maker's reserve is
		// deleted). The maker's aggregate locked: was already spent by settleFills.
		if err := b.decrementMakerReserves(db, poolID, side, base, quote, res.Fills); err != nil {
			return err
		}
		// Refund the taker's unspent lock remainder (IOC never rests).
		if lockedAmt > spent {
			var lockAsset uint64
			if side == sideBuy {
				lockAsset = quote
			} else {
				lockAsset = base
			}
			if err := unlockToAvailable(db, uid, lockAsset, lockedAmt-spent); err != nil {
				return err
			}
		}
		return nil

	default:
		return nil
	}
}

// decrementMakerReserves reduces each filled maker's per-order reserve by the
// amount of its locked asset the cross consumed, so cancel-time unlock matches
// the remaining resting size. A maker on the opposite side of the taker locked:
// taker BUY -> maker SOLD base (reserve in base, reduced by fill base units);
// taker SELL -> maker BOUGHT base (reserve in quote, reduced by fill quote units).
func (b *Block) decrementMakerReserves(db *versiondb.Database, poolID [32]byte, takerSide uint8, base, quote uint64, fills []lx.Trade) error {
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
		newReserve := uint64(0)
		if reserve > amt {
			newReserve = reserve - amt
		}
		if err := putOrderLock(db, poolID, makerOrderID, asset, newReserve); err != nil {
			return err
		}
	}
	return nil
}

// txIndexOf returns the position of tx within the block (the txIndex the VM uses
// to derive a placed order's deterministic id). It scans by pointer identity; a
// block holds a handful of txs so the linear scan is negligible and keeps the
// helper free of an index parameter threaded through the lock/settle calls.
func txIndexOf(b *Block, tx *Tx) uint32 {
	for i, t := range b.txs {
		if t == tx {
			return uint32(i)
		}
	}
	return 0
}

// canceledOwner returns the user identity of a cancelled order. The matcher's
// cancel path removes the order; the owner is recovered from the order snapshot
// the apply result captured. (applyCancel records only the id today; the owner is
// read from the reserve's account at unlock by re-deriving from the order row —
// but since the reserve is keyed by order id and the owner is whoever placed it,
// we resolve the owner from the order row captured in res.) For the current
// applyCancel which does not carry the owner, the owner is the reserve's account
// — see settleOrderEffects, which reads the reserve and unlocks to the placing
// account. This helper returns the canceller from the result when available.
func canceledOwner(res applyResult) string {
	if res.CanceledOwner != "" {
		return res.CanceledOwner
	}
	return ""
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
func rebuildBookFromDB(db database.Iteratee, poolID [32]byte, symbol string) (*lx.OrderBook, error) {
	prefix := orderPrefixFor(poolID)
	it := db.NewIteratorWithPrefix(prefix)
	defer it.Release()

	var rows []lx.DEXOrder
	for it.Next() {
		row, ok := lx.DecodeRow(it.Value())
		if !ok {
			return nil, fmt.Errorf("dchain: corrupt order row under %x (len %d)", poolID[:8], len(it.Value()))
		}
		rows = append(rows, row)
	}
	if err := it.Error(); err != nil {
		return nil, err
	}
	ob := lx.RowsToBook(symbol, rows)
	// Restore the EXACT-INTEGER value lane on every rebuilt resting order. The
	// persisted DEXOrder row keeps quantities in fixed-point, not the matcher's
	// atomic-unit SizeUnits/RemainingUnits, so an order rebuilt from disk would
	// lose the integer lane — its fills would carry no BaseUnits/QuoteUnits and
	// the custody ledger could not settle value-exactly. We re-derive the lane
	// from the order's (now-restored) size so a maker resting across a restart, or
	// rebuilt into a fresh per-block book, settles a taker's cross to the unit.
	restoreIntegerLane(ob)
	return ob, nil
}

// restoreIntegerLane sets SizeUnits/RemainingUnits (the matcher's exact-integer
// quantity lane) on every resting order in a rebuilt book, derived from the
// order's float size/remaining via the same whole-unit conversion the consensus
// place/submit path uses (sizeToUnits). Without this a rebuilt maker has a nil
// lane and a taker crossing it produces fills with no integer truth (which the
// custody settle refuses). Idempotent: an order already carrying the lane is left
// as-is.
func restoreIntegerLane(ob *lx.OrderBook) {
	for _, o := range ob.Orders {
		if o == nil {
			continue
		}
		if o.SizeUnits == nil {
			if u := sizeToUnits(o.Size); u > 0 {
				o.SizeUnits = new(big.Int).SetUint64(u)
			}
		}
		if o.RemainingUnits == nil {
			rem := o.RemainingSize
			if rem == 0 {
				rem = o.Size
			}
			if u := sizeToUnits(rem); u > 0 {
				o.RemainingUnits = new(big.Int).SetUint64(u)
			}
		}
	}
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
