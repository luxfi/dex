// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"math/big"
	"time"
)

// source_orderBook.go is the V4 native OrderBook liquidity source — the in-process matcher.
// It is the router's PRIMARY source: a marketable order crosses the resting book,
// each fill settles value-exactly inside the custody ledger (taker locked spend ->
// maker available; maker locked -> taker available), and the proceeds are recorded
// as a C-surface credit in the joint journal. Every step is the proven dchain
// settlement logic, now called over the shared Store.
//
// The matcher runs in BOTH QuoteSwap (a non-mutating cross over a CLONED book) and
// ExecuteSwap (the mutating cross over the live book). Quote uses the same matcher
// against a deep-copied book so its predicted output equals what Execute produces —
// the router relies on quote==execute for the split it picks.

// OrderBookSource matches against the in-trie resting book. It holds no state; the book
// is rebuilt from the Store on each call (the rebuildable-accelerator discipline).
type OrderBookSource struct{}

// NewOrderBookSource returns the OrderBook liquidity source.
func NewOrderBookSource() *OrderBookSource { return &OrderBookSource{} }

// Kind identifies this as the OrderBook source.
func (*OrderBookSource) Kind() SourceKind { return SourceOrderBook }

// QuoteSwap rebuilds the market book, runs the matcher against a CLONE for up to
// remainingIn of the spend asset, and reports the output WITHOUT mutating state.
// Ok=false when the market is unknown, has no crossable depth, or the order yields
// no fills.
func (*OrderBookSource) QuoteSwap(db Store, req SwapRequest, remainingIn uint64) (Quote, error) {
	symbol, exists, err := ReadMarketSymbol(db, req.PoolID)
	if err != nil {
		return Quote{}, err
	}
	if !exists {
		return Quote{}, nil // no such market on the OrderBook: not an error, just no fill
	}
	book, err := RebuildBook(db, req.PoolID, symbol)
	if err != nil {
		return Quote{}, err
	}
	order := buildMarketableOrder(req, remainingIn)
	if order == nil {
		return Quote{}, nil // no-limit buy / un-fundable budget: no fill from this source
	}
	// SubmitMarketable mutates the book IN PLACE; QuoteSwap must not persist, and it
	// works on a freshly-rebuilt book that is discarded when this call returns, so the
	// live Store is untouched. (The book is a per-call value; nothing persists unless
	// ExecuteSwap writes it.)
	fills, err := book.SubmitMarketable(order)
	if err != nil {
		return Quote{}, nil // deterministic reject => no fill from this source
	}
	inUsed, outGot := fillTotals(fills, req.Side)
	if inUsed == 0 || outGot == 0 {
		return Quote{}, nil
	}
	return Quote{AmountInUsed: inUsed, AmountOut: outGot, Ok: true}, nil
}

// ExecuteSwap rebuilds the market book, crosses up to remainingIn, settles each
// fill inside the ledger (the integer-exact conservation + fail-closed maker
// identity), persists the touched maker rows, and records the taker's proceeds as a
// C-surface credit in the journal. Returns the fill and the input/output consumed.
//
// The taker's input lock is NOT performed here — the top-level ExecuteSwap locks
// the FULL swap input once (so a split across sources draws from one lock); this
// source only spends the taker's already-locked balance per fill and refunds
// nothing (the top-level refunds the unspent remainder after all sources run).
func (*OrderBookSource) ExecuteSwap(db Store, req SwapRequest, remainingIn uint64, j *Journal) (Fill, uint64, uint64, error) {
	symbol, exists, err := ReadMarketSymbol(db, req.PoolID)
	if err != nil {
		return Fill{}, 0, 0, err
	}
	if !exists {
		return Fill{}, 0, 0, nil
	}
	book, err := RebuildBook(db, req.PoolID, symbol)
	if err != nil {
		return Fill{}, 0, 0, err
	}
	order := buildMarketableOrder(req, remainingIn)
	if order == nil {
		return Fill{}, 0, 0, nil // no-limit buy / un-fundable budget: no fill
	}
	// Capture the makers touched by the cross BEFORE it mutates them (the matcher
	// edits resting orders in place), so we rewrite/delete their persisted rows.
	fills, err := book.SubmitMarketable(order)
	if err != nil {
		return Fill{}, 0, 0, nil // deterministic reject
	}
	if len(fills) == 0 {
		return Fill{}, 0, 0, nil
	}

	// Settle the fills inside the ledger: taker locked spend -> maker available;
	// maker locked -> taker available. Resolves each maker's FULL identity via the
	// orderuser: index (fail-closed). The taker's spend comes from the lock the
	// top-level ExecuteSwap already placed.
	spent, serr := SettleFills(db, req.PoolID, req.TakerUser, req.Side, req.Base, req.Quote, fills)
	if serr != nil {
		return Fill{}, 0, 0, serr
	}
	// Decrement each touched maker's per-order reserve by what the cross consumed.
	if derr := DecrementMakerReserves(db, req.PoolID, req.Side, req.Base, req.Quote, fills); derr != nil {
		return Fill{}, 0, 0, derr
	}

	// Persist the touched maker rows (rewrite remaining, delete fully filled).
	if perr := persistTouchedMakers(db, req.PoolID, book, fills, req.Side); perr != nil {
		return Fill{}, 0, 0, perr
	}
	// Advance the market's LastOrderID watermark from the (rebuilt + crossed) book.
	if werr := WriteBookWatermark(db, req.PoolID, book.LastOrderID); werr != nil {
		return Fill{}, 0, 0, werr
	}

	inUsed, outGot := fillTotals(fills, req.Side)
	// The taker's proceeds (outGot of the receive asset) are now in the taker's
	// ledger available (credited by SettleFills). Mirror them to the C surface: the
	// 0x9999 vault releases outGot of the output asset to the taker's EVM balance.
	outAsset := req.Base
	if req.Side == Buy {
		outAsset = req.Base // taker buys base
	} else {
		outAsset = req.Quote // taker sells base, receives quote
	}
	j.CreditTakerOutput(outAsset, outGot)
	_ = spent // spent == inUsed for the integer lane; kept for clarity

	return Fill{Trades: fills, Source: SourceOrderBook}, inUsed, outGot, nil
}

// buildMarketableOrder constructs the ephemeral IOC order for a swap. The matcher
// sizes orders in BASE units, so the two sides differ:
//
//   - SELL (taker spends base): the spend amount IS the base size directly —
//     order.Size = remainingIn base units. A MARKET sell sweeps all crossable bids;
//     a LIMIT sell stops at the taker's floor price. Exact-input on base.
//
//   - BUY (taker spends quote): the spend is a quote BUDGET, not a base count, and a
//     bare market buy with no price ceiling is an unbounded-slippage MEV hazard — so
//     a BUY REQUIRES a price limit (the caller passes the V4 SqrtPriceLimitX96-derived
//     LimitPrice). The base size the budget buys at the ceiling is
//     floor(remainingIn * PriceMultiplier / LimitPrice); the order is a LIMIT buy at
//     LimitPrice. The taker pays maker prices <= the ceiling, so the actual quote
//     spend never exceeds the locked budget, and any unspent quote (makers cheaper
//     than the ceiling) is refunded by the top-level orchestrator. A no-limit BUY
//     returns nil (rejected upstream) — never a free unbounded sweep.
//
// The id is the block-deterministic id and the timestamp is the block time (both
// from EVM block context via SwapRequest), so the fills are byte-identical on every
// validator. The order NEVER rests (IOC).
func buildMarketableOrder(req SwapRequest, remainingIn uint64) *Order {
	order := &Order{
		ID:        req.OrderID,
		Side:      req.Side,
		User:      matcherHint(req.TakerUser),
		UserID:    matcherHint(req.TakerUser),
		Symbol:    PoolSymbol(req.PoolID),
		Timestamp: time.Unix(0, req.TimestampN),
	}
	if req.Side == Sell {
		order.Size = float64(remainingIn)
		order.SizeUnits = new(big.Int).SetUint64(remainingIn)
		if req.LimitPrice > 0 {
			order.Type = Limit
			order.Price = float64(req.LimitPrice) / float64(PriceMultiplier)
		} else {
			order.Type = Market
		}
		return order
	}

	// BUY: require a price ceiling; size base from the quote budget at the ceiling.
	if req.LimitPrice <= 0 {
		return nil // no-limit buy is rejected upstream (BuyRequiresLimit)
	}
	baseUnits := buyBaseFromQuoteBudget(remainingIn, req.LimitPrice)
	if baseUnits == 0 {
		return nil // budget cannot buy one base unit at the ceiling
	}
	order.Type = Limit
	order.Price = float64(req.LimitPrice) / float64(PriceMultiplier)
	order.Size = float64(baseUnits)
	order.SizeUnits = new(big.Int).SetUint64(baseUnits)
	return order
}

// buyBaseFromQuoteBudget returns floor(quoteBudget * PriceMultiplier / limitPrice)
// — the base units a quote budget buys at the ceiling price (limitPrice is the
// PriceInt quote-per-base). 128-bit safe so quoteBudget*PriceMultiplier can't wrap.
func buyBaseFromQuoteBudget(quoteBudget uint64, limitPrice PriceInt) uint64 {
	if quoteBudget == 0 || limitPrice <= 0 {
		return 0
	}
	return mulDivU64(quoteBudget, uint64(PriceMultiplier), uint64(limitPrice))
}

// fillTotals sums a fill set into (inUsed, outGot) in the matcher's integer lane,
// from the taker's perspective. For a BUY the taker spends quote (in) and receives
// base (out); for a SELL the taker spends base (in) and receives quote (out).
func fillTotals(fills []Trade, side Side) (inUsed, outGot uint64) {
	for _, f := range fills {
		if f.BaseUnits == nil || f.QuoteUnits == nil || !f.BaseUnits.IsUint64() || !f.QuoteUnits.IsUint64() {
			continue
		}
		base := f.BaseUnits.Uint64()
		quote := f.QuoteUnits.Uint64()
		if side == Buy {
			inUsed += quote
			outGot += base
		} else {
			inUsed += base
			outGot += quote
		}
	}
	return inUsed, outGot
}

// persistTouchedMakers rewrites/deletes the persisted rows of every maker the cross
// touched, so the durable book reflects the post-cross resting set.
func persistTouchedMakers(db Store, poolID [32]byte, book *OrderBook, fills []Trade, takerSide Side) error {
	seen := map[uint64]struct{}{}
	for _, f := range fills {
		makerID := f.SellOrder
		if takerSide == Sell {
			makerID = f.BuyOrder
		}
		if _, dup := seen[makerID]; dup {
			continue
		}
		seen[makerID] = struct{}{}
		if m := book.GetOrder(makerID); m != nil {
			row := OrderToRow(m)
			row.OrderID = m.ID
			if err := PutOrderRow(db, poolID, row); err != nil {
				return err
			}
		} else {
			// Maker fully filled and removed from the book: delete its row.
			if err := DeleteOrderRow(db, poolID, makerID); err != nil {
				return err
			}
		}
	}
	return nil
}

// matcherHint renders an AccountID to the matcher's User/UserID string. This string
// is ONLY a matching hint: the matcher derives its 8-byte STP collision handle from
// it, and that handle is NEVER authoritative for value (settlement resolves the taker
// from the explicit AccountID and the maker from the orderuser: row). So the hint
// only needs to be a deterministic, injective-enough function of the account — the
// raw 32 account bytes as a string serve exactly that, with no lossy round-trip
// concern because settlement never reads it back.
func matcherHint(a AccountID) string { return string(a[:]) }
