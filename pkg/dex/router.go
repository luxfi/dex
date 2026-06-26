// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"math/big"

	"github.com/luxfi/dex/pkg/lx"
)

// router.go is THE on-chain smart-order-router — the real product. It routes one
// swap across the on-chain liquidity SOURCES, in-process and deterministically:
//
//  1. the V4 native OrderBook (the in-process matcher) FIRST;
//  2. the V2/V3 AMM pools (EVM contracts in the SAME cEVM) when the OrderBook has
//     insufficient depth — read their state and execute the swap in-process;
//  3. best-execution: take the better price across the OrderBook and the AMM, or split
//     the input across them when splitting beats either alone.
//
// Every source is a pure, replayable function of (prior state, input): no live
// external call, no clock. A validator re-running the same swap over the same
// prior state derives the IDENTICAL fills and the IDENTICAL balance delta, so the
// route is committed by the block's state root with no fill vector carried in the
// block bytes.
//
// EXTERNAL CEX is OUT of the first cut (a direct CEX call is non-deterministic and
// forks consensus). The Source abstraction does NOT preclude it: an RFQ/market-
// maker source whose on-chain quotes are backed by off-chain hedging plugs in
// LATER as another Source whose Quote/Execute read only signed, on-chain-committed
// quote state. The router never knows or cares which concrete source it routes to.

// RouteClass tags the privacy/visibility regime a swap routes under. Only
// ClassPublicDEX is built in the first cut. ClassFHEPrivate is the staged-next
// private-pool route (owner spec §13): a swap whose amounts are FHE-encrypted and
// matched in a private pool. The router carries the class so the source selection
// can later restrict an FHE swap to FHE-capable sources WITHOUT reshaping the
// router — design room, not built.
type RouteClass uint8

const (
	// ClassPublicDEX is the public route: plaintext amounts, the public OrderBook +
	// public AMM pools. The only class built now.
	ClassPublicDEX RouteClass = iota
	// ClassFHEPrivate is the private route (STAGED — not built): FHE-encrypted
	// amounts, matched in a private pool. Reserved so the router can route it to an
	// FHE source later without an interface change.
	ClassFHEPrivate
)

// SwapRequest is the in-block representation of one synchronous swap, fully
// derivable from the EVM calldata + prior committed state, so every validator
// reconstructs it identically. No field is a live query result.
type SwapRequest struct {
	// Market identity + the taker.
	PoolID    [32]byte  // = V4 PoolKey.ID()
	TakerUser AccountID // the taker's FULL 16-byte settlement identity (the spender)
	Side      lx.Side   // Buy (taker buys base, spends quote) or Sell
	Base      AssetID   // market base asset (full injective id)
	Quote     AssetID   // market quote asset (full injective id)

	// Amount + the deterministic order id/timestamp (from EVM block context).
	AmountIn   uint64 // exact-input asset-unit amount (asset = the spend side)
	OrderID    uint64 // = blockDeterministicID(height, callIndex); never zero
	TimestampN int64  // block timestamp (UnixNano), for the ephemeral order

	// LimitPrice is the taker's OWN worst-acceptable OrderBook price (quote-per-base, in
	// PriceInt grid units), derived from the V4 SqrtPriceLimitX96. 0 = no limit.
	// LimitIsUpper says which side it bounds (true = a BUY ceiling, false = a SELL
	// floor). Enforced on the realized proceeds price — the taker-authenticated MEV
	// floor — so a sandwiching counterparty cannot force a fill outside the limit.
	LimitPrice   lx.PriceInt
	LimitIsUpper bool

	// MinOut is the V4 slippage floor: the swap reverts if the total proceeds
	// across all sources is below it. 0 = no floor.
	MinOut uint64

	// Class is the route class (public OrderBook now; FHE-private staged). The first cut
	// only routes ClassPublicDEX.
	Class RouteClass
}

// Fill is one matched quantity from one source, in the matcher's authoritative
// integer lane. The router aggregates Fills across sources into the joint journal.
type Fill struct {
	// Trades are the OrderBook fills (for a OrderBook source) — they carry maker order ids so
	// settlement resolves each maker's full identity via orderuser:. Empty for an
	// AMM source (an AMM fill has no resting maker; see AMMBaseDelta/AMMQuoteDelta).
	Trades []lx.Trade

	// AMMBaseDelta / AMMQuoteDelta are the base/quote units an AMM source moved (an
	// AMM has no maker rows). The router credits/debits these against the pool's
	// reserves and the taker directly. Zero for a OrderBook source.
	AMMBaseDelta  uint64
	AMMQuoteDelta uint64

	// Source names which source produced this fill (for events/receipts).
	Source SourceKind
}

// SourceKind enumerates the on-chain liquidity sources the router routes across.
type SourceKind uint8

const (
	// SourceOrderBook is the V4 native OrderBook (the in-process matcher). The primary source.
	SourceOrderBook SourceKind = iota
	// SourceAMM is an in-cEVM V2/V3 AMM pool, read + executed in-process.
	SourceAMM
	// SourceRFQ is the STAGED external-CEX/market-maker source (on-chain quotes
	// backed by off-chain hedging). NOT built in the first cut; the kind is reserved
	// so a later source can self-identify without reshaping this enum.
	SourceRFQ
)

// Quote is a source's indicative answer for a swap: how much output it can give
// for a given input, and at what (worst) effective price, WITHOUT mutating state.
// The router compares quotes across sources to pick the best execution / split.
type Quote struct {
	// AmountInUsed is the input units this source would consume (<= the requested
	// remaining; a source may fill only part of the depth).
	AmountInUsed uint64
	// AmountOut is the output units this source yields for AmountInUsed.
	AmountOut uint64
	// Ok is false when the source cannot fill any of the requested input.
	Ok bool
}

// Source is one on-chain liquidity source the router routes across. It is a pure,
// replayable view+executor over the Store. Both methods read prior state only; a
// source NEVER issues a live external query (that forks consensus). The OrderBook
// source and the AMM source implement it; an RFQ source can implement it LATER
// over on-chain-committed quote state without the router changing.
type Source interface {
	// Kind identifies the source (for fills/events).
	Kind() SourceKind

	// QuoteSwap returns this source's indicative output for filling up to
	// remainingIn of the swap's spend asset, reading prior state only. Ok=false
	// when the source cannot fill any of remainingIn.
	QuoteSwap(db Store, req SwapRequest, remainingIn uint64) (Quote, error)

	// ExecuteSwap fills up to remainingIn against this source, MUTATING the Store
	// (the matcher's book/ledger rows for the OrderBook; the pool reserves for the AMM)
	// and appending the value moves to the journal. Returns the Fill and the input
	// units actually consumed. It MUST be consistent with QuoteSwap over the same
	// state (the router relies on quote==execute for the split it chose).
	ExecuteSwap(db Store, req SwapRequest, remainingIn uint64, j *Journal) (fill Fill, inUsed uint64, outGot uint64, err error)
}

// Router routes a swap across an ordered set of Sources, best-execution first.
// The order is the SOURCE PRIORITY when prices tie or a source is exhausted:
// OrderBook first (the native venue), then AMM. The router itself holds no state — it
// is a pure function of the sources + the Store.
type Router struct {
	sources []Source
}

// NewRouter builds a router over the given sources in priority order. The first
// cut wires [OrderBook, AMM]; an RFQ source appends LATER with no router change.
func NewRouter(sources ...Source) *Router {
	return &Router{sources: sources}
}

// Route executes the swap across the router's sources with best-execution, writing
// every value move into the journal. The algorithm is a deterministic greedy
// best-price fill:
//
//  1. While input remains and progress is possible:
//     a. QuoteSwap every source for the remaining input.
//     b. Pick the source with the best marginal price (most output per input
//     consumed), ties broken by source priority (OrderBook before AMM).
//     c. ExecuteSwap that source for the remaining input; subtract the consumed
//     input, add the produced output.
//  2. Stop when input is exhausted, no source can fill more, or this iteration
//     made no progress.
//
// This yields best-execution across V4 + AMM and naturally SPLITS the input when
// one source's depth runs out and another offers a better remaining price than no
// fill — the router walks each source's depth in price order. It is deterministic:
// the quote functions are pure over prior state, the tie-break is a fixed priority,
// so every validator picks the identical route.
//
// Returns the aggregate output, the input actually consumed, and the per-source
// fills (for events). Enforces the taker price limit and the MinOut floor; on a
// breach the caller reverts the whole swap (the journal is discarded, the EVM
// snapshot rolls back the lock).
func (r *Router) Route(db Store, req SwapRequest, j *Journal) (totalOut, totalInUsed uint64, fills []Fill, err error) {
	remainingIn := req.AmountIn
	// Guard against a pathological non-progressing loop: at most one round per
	// source plus a small margin (each round either consumes input or removes a
	// source from contention). len(sources)+remainingIn is a safe, finite bound;
	// in practice the loop runs a handful of rounds.
	maxRounds := len(r.sources)*2 + 1
	for round := 0; remainingIn > 0 && round < maxRounds; round++ {
		// (a) Quote every source for the remaining input.
		bestIdx := -1
		var best Quote
		for i, s := range r.sources {
			q, qerr := s.QuoteSwap(db, req, remainingIn)
			if qerr != nil {
				return 0, 0, nil, qerr
			}
			if !q.Ok || q.AmountInUsed == 0 || q.AmountOut == 0 {
				continue
			}
			if bestIdx == -1 || betterMarginalPrice(q, best, req.Side) {
				best, bestIdx = q, i
			}
		}
		if bestIdx == -1 {
			break // no source can fill any of the remaining input
		}

		// (c) Execute the best source for the remaining input.
		fill, inUsed, outGot, eerr := r.sources[bestIdx].ExecuteSwap(db, req, remainingIn, j)
		if eerr != nil {
			return 0, 0, nil, eerr
		}
		if inUsed == 0 && outGot == 0 {
			break // no progress (defensive — a quoted source yielded nothing)
		}
		fills = append(fills, fill)
		totalOut += outGot
		totalInUsed += inUsed
		if inUsed >= remainingIn {
			remainingIn = 0
		} else {
			remainingIn -= inUsed
		}
	}

	if totalOut == 0 {
		return 0, 0, nil, ErrNoLiquidity
	}

	// Taker-authenticated MEV floor: enforce the realized proceeds price against the
	// taker's OWN recorded limit. The realized price is the aggregate (out/in),
	// compared on the PriceInt grid the limit is expressed in.
	if err := enforceProceedsPriceFloor(req, totalInUsed, totalOut); err != nil {
		return 0, 0, nil, err
	}
	// V4 slippage floor.
	if req.MinOut > 0 && totalOut < req.MinOut {
		return 0, 0, nil, ErrPriceLimit
	}
	return totalOut, totalInUsed, fills, nil
}

// betterMarginalPrice reports whether quote a offers a better effective price than
// quote b for the taker, given the side. The effective price is output-per-input;
// more output per unit input is always better for the taker. Compared with cross-
// multiplication (no float): a is better iff a.out * b.in > b.out * a.in. Equal
// prices keep the incumbent (b) so source priority (the scan order, OrderBook first)
// breaks the tie.
func betterMarginalPrice(a, b Quote, _ lx.Side) bool {
	// a.out/a.in  >  b.out/b.in   <=>   a.out*b.in > b.out*a.in
	lhs := new(big.Int).Mul(new(big.Int).SetUint64(a.AmountOut), new(big.Int).SetUint64(b.AmountInUsed))
	rhs := new(big.Int).Mul(new(big.Int).SetUint64(b.AmountOut), new(big.Int).SetUint64(a.AmountInUsed))
	return lhs.Cmp(rhs) > 0
}

// enforceProceedsPriceFloor checks the realized aggregate proceeds against the
// taker's recorded price limit (the MEV floor). The realized price is out/in for a
// SELL (quote-per-base proceeds) and in/out for a BUY (quote-per-base paid). The
// limit is in PriceInt units (quote * PriceMultiplier / base). A breach reverts
// the whole swap.
func enforceProceedsPriceFloor(req SwapRequest, inUsed, out uint64) error {
	if req.LimitPrice <= 0 || inUsed == 0 || out == 0 {
		return nil
	}
	// Realized price as PriceInt = (quoteUnits * PriceMultiplier) / baseUnits.
	var quoteUnits, baseUnits uint64
	if req.Side == lx.Buy {
		// Taker bought base (out) by paying quote (in): price = in/out.
		quoteUnits, baseUnits = inUsed, out
	} else {
		// Taker sold base (in) for quote (out): price = out/in.
		quoteUnits, baseUnits = out, inUsed
	}
	realized := new(big.Int).Mul(new(big.Int).SetUint64(quoteUnits), priceMultiplierBig)
	realized.Quo(realized, new(big.Int).SetUint64(baseUnits))
	limit := big.NewInt(int64(req.LimitPrice))
	if req.LimitIsUpper {
		// BUY ceiling: realized price must not exceed the limit (taker won't overpay).
		if realized.Cmp(limit) > 0 {
			return ErrPriceLimit
		}
	} else {
		// SELL floor: realized price must not fall below the limit (taker won't undersell).
		if realized.Cmp(limit) < 0 {
			return ErrPriceLimit
		}
	}
	return nil
}
