// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"time"

	"github.com/luxfi/crypto/hash"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/zapwire"
)

// execute.go is the deterministic state-transition function. applyTx takes a tx,
// the in-memory book for its market, the block height, the block timestamp, and
// the tx's index in the block, and applies it — minting NOTHING from wall-clock
// or a process-local counter. Every value that participates in matched state is
// derived from block context, so two validators replaying the same ordered block
// produce byte-identical results.

// blockDeterministicID derives a resting order's ID from (height, txIndex). It
// is globally unique across the chain (height monotone, txIndex unique within a
// block) and identical on every validator. The top 32 bits are the height and
// the bottom 32 are the txIndex, with a +1 bias so an ID is never zero (zero is
// the "unset" sentinel the matcher rejects). For heights/indices beyond 32 bits
// this would alias; a venue block never holds 4 billion txs and a chain does not
// reach height 4 billion in any realistic horizon, but if it ever must, widen to
// a keccak-derived 64-bit id (kept simple here for a stable, debuggable id).
func blockDeterministicID(height uint64, txIndex uint32) uint64 {
	id := (height << 32) | uint64(txIndex)
	return id + 1
}

// applyResult is the outcome of applying one tx: the fills it produced (for a
// submit) and the resting-order delta to persist. The VM writes these to the
// versiondb overlay; on a place the placed order is the delta, on a cancel the
// cancelled order id is removed, on a submit the affected makers are updated.
type applyResult struct {
	// Fills are the trades a TxSubmit produced (empty for other tx types).
	Fills []lx.Trade
	// TakerSide is the side the submit took with (for encoding fill rows).
	TakerSide lx.Side
	// Placed is the order a TxPlace rested (nil otherwise).
	Placed *lx.Order
	// Canceled is the order id a TxCancel removed (0 otherwise).
	Canceled uint64
	// Touched are resting orders whose remaining changed due to a submit cross
	// (the makers); their rows must be rewritten/deleted.
	Touched []*lx.Order
}

// applyTx applies a single transaction to the book deterministically. It is a
// pure function of (tx, book state, height, ts, txIndex): no time.Now(), no
// LastOrderID mint. EnsureMarket is handled by the VM (it has no book effect
// beyond existence, recorded in state); applyTx handles place/cancel/submit.
//
// For TxPlace: the order's ID = blockDeterministicID(height, txIndex), its
// Timestamp = ts, then ConsensusAddOrder rests it.
// For TxCancel: CancelOrder removes the resting order.
// For TxSubmit: the marketable order also gets a deterministic ID/ts (ephemeral —
// it never rests) and SubmitMarketable crosses the book, returning its fills.
func applyTx(book *lx.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
	switch tx.Type {
	case TxPlace:
		return applyPlace(book, tx, height, ts, txIndex)
	case TxCancel:
		return applyCancel(book, tx)
	case TxSubmit:
		return applySubmit(book, tx, height, ts, txIndex)
	case TxEnsureMarket:
		// No book mutation; market existence is recorded in d-chain state by the
		// VM. Return an empty result.
		return applyResult{}, nil
	default:
		return applyResult{}, ErrUnknownTx
	}
}

// applyPlace decodes a zapwire Place body and rests it via the deterministic
// consensus entrypoint. Rejected orders (bad price/size, post-only-would-take,
// self-trade) yield an empty result with no error: a rejected order is a valid,
// deterministic outcome (no fills, no resting delta), not a chain-level failure.
func applyPlace(book *lx.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
	body := tx.Body
	// Place body: poolId[32] + side[1] + price[8] + size[8] + user[16].
	side := lx.Side(body[zapwire.PoolIDSize])
	price := zapwire.Float64(body[zapwire.PoolIDSize+1 : zapwire.PoolIDSize+9])
	size := zapwire.Float64(body[zapwire.PoolIDSize+9 : zapwire.PoolIDSize+17])
	user := string(trimNull(body[zapwire.PoolIDSize+17 : zapwire.PoolIDSize+17+zapwire.UserSize]))

	if !(price > 0) || !(size > 0) {
		return applyResult{}, nil // deterministic reject
	}
	order := &lx.Order{
		ID:        blockDeterministicID(height, txIndex),
		Type:      lx.Limit,
		Side:      side,
		Price:     price,
		Size:      size,
		User:      user,
		UserID:    user,
		Symbol:    book.Symbol,
		Timestamp: ts,
	}
	if book.ConsensusAddOrder(order) == 0 {
		return applyResult{}, nil // rejected (e.g. post-only would take)
	}
	return applyResult{Placed: order}, nil
}

// applyCancel decodes a zapwire Cancel body and removes the resting order. A
// cancel of an unknown order is a deterministic no-op (the matcher returns an
// error which we map to an empty result — every validator sees the same).
func applyCancel(book *lx.OrderBook, tx *Tx) (applyResult, error) {
	orderID := binary.BigEndian.Uint64(tx.Body[zapwire.PoolIDSize : zapwire.PoolIDSize+8])
	// Snapshot the order before cancel so we know it existed (for the delta).
	if book.GetOrder(orderID) == nil {
		return applyResult{}, nil
	}
	if err := book.CancelOrder(orderID); err != nil {
		return applyResult{}, nil
	}
	return applyResult{Canceled: orderID}, nil
}

// applySubmit decodes a zapwire Submit body and crosses the book. The marketable
// order gets a deterministic (height,txIndex) ID and the block ts so the fills it
// produces are byte-identical across validators; the order itself never rests
// (SubmitMarketable is IOC). Touched makers are captured BEFORE the cross so the
// VM can rewrite their rows; SubmitMarketable mutates resting orders in place.
func applySubmit(book *lx.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
	body := tx.Body
	// Submit body: poolId[32] + side[1] + isMarket[1] + price[8] + size[8] + user[16].
	side := lx.Side(body[zapwire.PoolIDSize])
	isMarket := body[zapwire.PoolIDSize+1] == 1
	limitPrice := zapwire.Float64(body[zapwire.PoolIDSize+2 : zapwire.PoolIDSize+10])
	size := zapwire.Float64(body[zapwire.PoolIDSize+10 : zapwire.PoolIDSize+18])
	user := string(trimNull(body[zapwire.PoolIDSize+18 : zapwire.PoolIDSize+18+zapwire.UserSize]))

	if !(size > 0) {
		return applyResult{}, nil
	}
	order := &lx.Order{
		ID:        blockDeterministicID(height, txIndex),
		Side:      side,
		Size:      size,
		User:      user,
		UserID:    user,
		Symbol:    book.Symbol,
		Timestamp: ts,
	}
	if isMarket {
		order.Type = lx.Market
	} else {
		if !(limitPrice > 0) {
			return applyResult{}, nil
		}
		order.Type = lx.Limit
		order.Price = limitPrice
	}

	fills, err := book.SubmitMarketable(order)
	if err != nil {
		return applyResult{}, nil // deterministic reject (e.g. invalid)
	}

	// Collect the makers touched by these fills so the VM rewrites their rows.
	touched := make([]*lx.Order, 0, len(fills))
	seen := make(map[uint64]struct{}, len(fills))
	for _, f := range fills {
		makerID := f.SellOrder
		if side == lx.Sell {
			makerID = f.BuyOrder
		}
		if _, dup := seen[makerID]; dup {
			continue
		}
		seen[makerID] = struct{}{}
		if m := book.GetOrder(makerID); m != nil {
			touched = append(touched, m)
		} else {
			// Maker fully filled and removed from the book: synthesize a
			// tombstone row so the VM deletes its persisted row.
			touched = append(touched, &lx.Order{ID: makerID, Status: lx.Filled})
		}
	}

	return applyResult{Fills: fills, TakerSide: side, Touched: touched}, nil
}

// idempotencyKey derives the d-chain dedup key for a submit/place from the wire
// user[16] and the per-user sequence. Dedup lives in STATE (this key), keeping
// the 66B/65B frames FROZEN — no nonce field is added to the wire. The seq is the
// caller's monotone counter; the VM stores seen keys and drops a replay.
func idempotencyKey(user [zapwire.UserSize]byte, seq uint64) [32]byte {
	var s [8]byte
	binary.BigEndian.PutUint64(s[:], seq)
	return hash.ComputeKeccak256Array(user[:], s[:])
}

// trimNull returns b with trailing NUL bytes removed (the inverse of zapwire's
// null padding for the user field).
func trimNull(b []byte) []byte {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] != 0 {
			return b[:i+1]
		}
	}
	return b[:0]
}
