// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import "errors"

// The dex error set. Each names a fail-closed condition: when settlement
// cannot proceed soundly, it returns one of these and the caller aborts the whole
// transition (the cEVM tx reverts, the dchain block is rejected) — value never
// moves on a degraded path.
var (
	// ErrInsufficientAvailable is returned when a debit/lock exceeds an account's
	// available balance. The custody gate maps it to a deterministic per-tx reject
	// (the order does not rest, nothing is locked).
	ErrInsufficientAvailable = errors.New("dex: insufficient available balance")

	// ErrInsufficientLocked is returned when an unlock/spend exceeds an account's
	// locked balance — it would mint against the ledger. Hard refusal.
	ErrInsufficientLocked = errors.New("dex: insufficient locked balance")

	// ErrMissingSettlementUser is the fail-closed sentinel: a fill against a
	// resting maker (or a cancel of a resting order) found NO orderuser: identity
	// row. Every order that rests on a balance-bearing market persists its
	// orderuser: row BEFORE it is matchable, so a missing row at settle time can
	// only be state corruption — never a normal path. There is NO fallback to the
	// matcher's 8-byte handle (that would make compact-handle collisions value-
	// bearing — a critical theft bug), so the transition is rejected.
	ErrMissingSettlementUser = errors.New("dex: resting order has no full settlement identity (orderuser: row missing)")

	// ErrFillMissingUnits is returned when a fill on the consensus path lacks the
	// authoritative integer lane (BaseUnits/QuoteUnits). The matcher sets it on
	// every consensus fill; a missing lane is a programming error, refused rather
	// than mis-settled from the lossy float view.
	ErrFillMissingUnits = errors.New("dex: fill missing integer units (BaseUnits/QuoteUnits)")

	// ErrFillUnitsOverflow is returned when a fill's integer lane exceeds uint64 —
	// the ledger is uint64-unit; an overflowing fill is refused, never truncated.
	ErrFillUnitsOverflow = errors.New("dex: fill integer units exceed uint64")

	// ErrDustPriceOrZeroLock is the named "no free executable order" reject: an
	// executable order (size>0, and for a limit price>0) whose notional floors to a
	// ZERO lock — a sub-tick BUY price (ceil(size*price)==0) or a sub-unit SELL
	// size. Such an order would rest/cross while reserving NOTHING. The custody gate
	// maps it to a deterministic per-tx reject.
	ErrDustPriceOrZeroLock = errors.New("dex: executable order locks zero (dust price or sub-unit size)")

	// ErrUnknownMarket is returned when a swap/route targets a market with no
	// recorded (base,quote) asset binding — value cannot be checked. Fail closed.
	ErrUnknownMarket = errors.New("dex: market has no asset binding")

	// ErrNoLiquidity is returned by the router when NO configured source can fill
	// any of the requested amount (neither the CLOB nor an AMM crosses). A swap
	// that fills nothing returns this so the caller reverts rather than locking the
	// taker's input against a zero fill.
	ErrNoLiquidity = errors.New("dex: no liquidity across any source for this swap")

	// ErrPriceLimit is the taker-authenticated MEV floor breach: the realized fill
	// price violated the taker's own worst-acceptable price (derived from the V4
	// SqrtPriceLimitX96). The whole swap reverts — a sandwiching counterparty/
	// proposer cannot force a fill outside the taker's limit.
	ErrPriceLimit = errors.New("dex: realized price violates taker price limit")

	// ErrBuyRequiresLimit is returned for an exact-input BUY (taker spends quote) with
	// no price limit. A bare market buy off a quote budget has unbounded slippage — a
	// sandwicher could price the taker arbitrarily badly — so a BUY MUST carry the V4
	// SqrtPriceLimitX96-derived ceiling. A SELL needs no limit (its base input is
	// exact and a market sell's worst case is bounded by the book).
	ErrBuyRequiresLimit = errors.New("dex: exact-input BUY requires a price limit (unbounded-slippage market buy refused)")
)
