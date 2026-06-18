// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/binary"
	"fmt"
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
		poolID, ok := tx.poolID()
		if !ok {
			return execResult{}, fmt.Errorf("dchain: tx %d missing poolId", i)
		}

		if tx.Type == TxEnsureMarket {
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
			outcomes = append(outcomes, outcomeFromApply(tx, applyResult{}))
			continue
		}

		// Idempotency: an order op whose tx id is already committed (or applied
		// earlier in THIS block, since seen-marks are written to the overlay) is a
		// deterministic no-op. This is read from the overlay so every validator —
		// and the proposer's probe — make the identical decision. A deduped write
		// resolves its waiter as rejected (already-processed), never re-executes,
		// so a relay retry of a dropped clob_submit can never double-fill/place.
		txID := tx.ID()
		if seen, err := isSeen(overlay, txID); err != nil {
			return execResult{}, err
		} else if seen {
			outcomes = append(outcomes, txOutcome{txID: txID, typ: tx.Type, status: zapwire.StatusRejected})
			continue
		}
		if err := markSeen(overlay, txID); err != nil {
			return execResult{}, err
		}

		ob, err := bookForPool(poolID)
		if err != nil {
			return execResult{}, err
		}

		res, err := applyTx(ob, tx, b.height, b.timestamp, uint32(i))
		if err != nil {
			return execResult{}, err
		}
		outcomes = append(outcomes, outcomeFromApply(tx, res))

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
	return lx.RowsToBook(symbol, rows), nil
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
