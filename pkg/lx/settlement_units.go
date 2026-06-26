// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import "math/big"

// settlement_units.go is the SINGLE home for the exact-integer quantity lane the
// OrderBook / consensus path uses for value conservation. The matcher's price grid
// stays float64 (a ratio is fine as a float, and price-time priority is unchanged
// by it), but the SETTLED QUANTITY — the base units a fill moves and the quote
// units owed for them — must be exact: an 18-decimal token amount (e.g. 1e20)
// exceeds float64's 2^53 exact-integer ceiling, so deriving settlement from a
// float quantity mints or burns real on-chain value on every realistic trade.
//
// Decomplected (Rich Hickey): "price" (a ratio) and "quantity" (a count of atomic
// units) are different values and live in different lanes. The float lane keeps
// serving the legacy single-process API and the matcher's crossing decision; this
// integer lane owns the value-bearing quantity end to end (wire -> match -> fill
// -> settlement). Nothing here touches the float crossing logic.
//
// priceMultiplierBig is PriceMultiplier as a big.Int (the 8-decimal fixed-point
// scale the engine quotes prices in). quoteUnits = baseUnits * priceInt /
// PriceMultiplier in exact integer arithmetic.
var priceMultiplierBig = big.NewInt(PriceMultiplier)

// quoteUnitsFor computes the exact quote owed for baseUnits at the maker's resting
// price, in atomic quote units. The price is taken as its fixed-point integer
// form (priceInt = round(price * PriceMultiplier)) — the SAME PriceInt the book
// keys levels by — so the quote is a pure-integer function of the matched base and
// the resting price, identical on every validator.
//
// Rounding: integer division floors. quote = floor(baseUnits * priceInt /
// PriceMultiplier). Floor (truncation toward zero for non-negative operands) means
// the quote charged to a taker buy / credited to a taker sell never EXCEEDS the
// mathematical product, so the matcher never fabricates a sub-unit of quote. The
// dropped remainder (< 1 atomic unit) stays with the counterparty — value is
// conserved across the maker/taker boundary because BOTH legs are derived from the
// SAME floored integer (the taker pays exactly what the maker is credited).
func quoteUnitsFor(baseUnits *big.Int, priceInt PriceInt) *big.Int {
	if baseUnits == nil || baseUnits.Sign() <= 0 || priceInt <= 0 {
		return big.NewInt(0)
	}
	q := new(big.Int).Mul(baseUnits, big.NewInt(int64(priceInt)))
	q.Quo(q, priceMultiplierBig)
	return q
}

// ensureRemainingUnits lazily initializes an order's exact integer remainder from
// its SizeUnits the first time the matcher touches it on the OrderBook path. An order
// without SizeUnits (legacy float API) yields nil — the integer lane stays off and
// the caller falls back to the float quantity, preserving legacy behavior.
func ensureRemainingUnits(o *Order) *big.Int {
	if o.SizeUnits == nil {
		return nil
	}
	if o.RemainingUnits == nil {
		o.RemainingUnits = new(big.Int).Set(o.SizeUnits)
	}
	return o.RemainingUnits
}

// minBig returns a fresh big.Int equal to the smaller of a and b.
func minBig(a, b *big.Int) *big.Int {
	if a.Cmp(b) <= 0 {
		return new(big.Int).Set(a)
	}
	return new(big.Int).Set(b)
}
