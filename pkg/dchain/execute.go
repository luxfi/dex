// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"math/big"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// maxPriceUnits is the largest wire price the matcher's PriceInt (int64) grid can
// hold. A price at/above 2^63 would overflow PriceInt, so it is rejected at decode
// — deterministically, on every validator (the wire is an exact uint64, so there
// is no architecture-dependent float→int conversion left to diverge).
const maxPriceUnits = uint64(1)<<63 - 1

// newWireOrder builds an order from the wire's EXACT integers: PriceUnits is the
// authoritative PriceInt grid value, SizeUnits the authoritative atomic base-unit
// count. The float Price/Size are DERIVED projections for the matcher's crossing
// loop and legacy/display only — they are never the source of a consensus value,
// so no float→int conversion (whose out-of-range result is architecture-dependent)
// touches the state transition.
func newWireOrder(id uint64, side dex.Side, priceUnits, sizeUnits uint64, user, symbol string, ts time.Time) *dex.Order {
	return &dex.Order{
		ID:         id,
		Side:       side,
		PriceUnits: dex.PriceInt(priceUnits),
		Price:      float64(priceUnits) / float64(dex.PriceMultiplier),
		Size:       float64(sizeUnits),
		SizeUnits:  new(big.Int).SetUint64(sizeUnits),
		User:       user,
		UserID:     user,
		Symbol:     symbol,
		Timestamp:  ts,
	}
}

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
	Fills []dex.Trade
	// TakerSide is the side the submit took with (for encoding fill rows).
	TakerSide dex.Side
	// Placed is the order a TxPlace rested (nil otherwise).
	Placed *dex.Order
	// Canceled is the order id a TxCancel removed (0 otherwise). The cancelled
	// order's SETTLEMENT identity is NOT carried here: the custody unlock resolves
	// it exclusively through the orderuser: index by this order id (settle.go /
	// settleOrderEffects), never from the matcher's 8-byte in-RAM owner — so a
	// post-restart cancel unlocks to the SAME full account the lock lives under.
	Canceled uint64
	// Touched are resting orders whose remaining changed due to a submit cross
	// (the makers); their rows must be rewritten/deleted.
	Touched []*dex.Order
	// SelfCanceled are resting maker order ids the submit's cross removed for
	// self-trade prevention (the taker crossed its OWN resting liquidity). They
	// produced NO fill; the VM persists each removal through the SAME cancel path a
	// TxCancel uses — delete the order: row, release the reserve (locked ->
	// available), delete the orderuser: row — so the durable state matches the
	// in-memory book the matcher already updated (no orphan resting row a later
	// cross would trip on). Empty for every non-self-crossing submit.
	SelfCanceled []uint64
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
func applyTx(book *dex.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
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
func applyPlace(book *dex.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
	poolID, side, price, size, user, derr := zapwire.DecodePlace(tx.Body)
	_ = poolID
	if derr != nil {
		return applyResult{}, nil // malformed body: deterministic reject
	}
	// price is fixed-point ×1e8 (PriceInt), size is atomic base units — both exact
	// integers. Reject zero price/size or a price that overflows the PriceInt grid.
	if price == 0 || size == 0 || price > maxPriceUnits {
		return applyResult{}, nil // deterministic reject
	}
	order := newWireOrder(blockDeterministicID(height, txIndex), dex.Side(side), price, size, user, book.Symbol, ts)
	order.Type = dex.Limit
	if book.ConsensusAddOrder(order) == 0 {
		return applyResult{}, nil // rejected (e.g. post-only would take)
	}
	return applyResult{Placed: order}, nil
}

// applyCancel decodes a zapwire Cancel body and removes the resting order. A
// cancel of an unknown order is a deterministic no-op (the matcher returns an
// error which we map to an empty result — every validator sees the same). The
// cancelled order's owner is NOT captured here: the custody unlock resolves the
// full settlement identity through the orderuser: index (settleOrderEffects), so
// the matcher's 8-byte in-RAM owner never participates in a value move.
func applyCancel(book *dex.OrderBook, tx *Tx) (applyResult, error) {
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
func applySubmit(book *dex.OrderBook, tx *Tx, height uint64, ts time.Time, txIndex uint32) (applyResult, error) {
	poolID, side, isMarket, limitPrice, size, user, derr := zapwire.DecodeSubmit(tx.Body)
	_ = poolID
	if derr != nil {
		return applyResult{}, nil
	}
	if size == 0 {
		return applyResult{}, nil
	}
	// For a limit submit the price bounds the cross and must fit the PriceInt grid;
	// a pure market submit ignores it (sweeps unconditionally). Exact integers, no float.
	priceUnits := limitPrice
	if isMarket {
		priceUnits = 0
	} else if limitPrice == 0 || limitPrice > maxPriceUnits {
		return applyResult{}, nil
	}
	dside := dex.Side(side)
	order := newWireOrder(blockDeterministicID(height, txIndex), dside, priceUnits, size, user, book.Symbol, ts)
	if isMarket {
		order.Type = dex.Market
	} else {
		order.Type = dex.Limit
	}

	// Route through the GPU shadow gate. With the default EngineCPU this is
	// byte-for-byte SubmitMarketable; with EngineGPUVerified (a node-local opt-in)
	// the GPU kernel is ALSO dispatched and byte-compared against this CPU result,
	// which is what is committed REGARDLESS — so the engine choice can never change
	// the committed fills and a mixed GPU/CPU validator set cannot fork.
	fills, err := book.SubmitMarketableVerified(order, dex.MatchEngine())
	if err != nil {
		return applyResult{}, nil // deterministic reject (e.g. invalid)
	}

	// Drain the self-trade-prevented maker removals this cross performed. The taker
	// crossed its OWN resting liquidity; those makers left the book with no fill and
	// must be cancelled in durable state too (reserve released, orderuser:/order:
	// rows deleted) so the persisted book matches the matcher's in-memory book.
	selfCanceled := book.DrainSelfCanceled()

	// Collect the makers touched by these fills so the VM rewrites their rows.
	touched := make([]*dex.Order, 0, len(fills))
	seen := make(map[uint64]struct{}, len(fills))
	for _, f := range fills {
		makerID := f.SellOrder
		if dside == dex.Sell {
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
			touched = append(touched, &dex.Order{ID: makerID, Status: dex.Filled})
		}
	}

	return applyResult{Fills: fills, TakerSide: dside, Touched: touched, SelfCanceled: selfCanceled}, nil
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
