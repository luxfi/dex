// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"fmt"
	"math/big"

	"github.com/luxfi/dex/pkg/lx"
)

// settle.go is the CUSTODY transition — the integer-exact value moves a matcher
// fill set produces, run against the same Store the order/ledger rows live in, so
// a balance move commits atomically with the match that produced it. It is a pure
// integer function of (fills, market assets, ledger state): no float arithmetic
// touches a value, so two homes (or two validators) replaying the same fills
// produce byte-identical balances.
//
// THE MODEL (one direction per leg, integer-exact, conserving):
//   - SUBMIT : the taker's spend is locked up-front (by the caller, via
//     LockFromAvailable), the matcher crosses, then for each fill move value:
//       taker.locked[spend] -= fill.spend ; maker.available[spend] += fill.spend
//       maker.locked[recv]  -= fill.recv  ; taker.available[recv]  += fill.recv
//     The caller then unlocks the taker's unspent remainder. Every unit removed
//     from one account's locked is added to exactly one other account's available
//     => sum(available)+sum(locked) is invariant.
//
// IDENTITY: the settlement identity is the WALLET (the full 16-byte AccountID).
// The TAKER is the submitter, whose full wallet is in hand this block. The MAKER
// is a RESTING order whose in-RAM user is only 8 bytes after a restart, so its
// full wallet is resolved EXCLUSIVELY through the orderuser: index by the maker's
// order id — there is NO fallback to the trade's 8-byte-derived carried user.
// Settlement FAILS CLOSED (ErrMissingSettlementUser) if full identity is
// unavailable; falling back to the matcher UserID would make compact-handle
// collisions value-bearing — a critical theft bug.

// PriceMultiplier mirrors lx.PriceMultiplier; exposed for the lock math callers.
var priceMultiplierBig = big.NewInt(lx.PriceMultiplier)

// SettleFills moves value for the fills a submit produced, INSIDE the ledger,
// from the matcher's AUTHORITATIVE integer lane (lx.Trade.BaseUnits/QuoteUnits).
// takerUser is the submitter's FULL 16-byte identity (always in hand — the taker
// never rests). takerSide is the submitting side; each fill's resting maker is
// resolved to its full identity via the orderuser: index (restart-safe). Using
// the matcher's own exact integers (not a reconstruction) guarantees the taker is
// charged EXACTLY what the maker is credited — conservation by construction.
//
// Returns the taker's total spend (so the caller unlocks the unspent remainder).
// Requires every fill to carry the integer lane; a fill without it is refused.
func SettleFills(db Store, poolID [32]byte, takerUser AccountID, takerSide lx.Side, base, quote AssetID, fills []lx.Trade) (takerSpent uint64, err error) {
	for _, f := range fills {
		if f.BaseUnits == nil || f.QuoteUnits == nil {
			return 0, ErrFillMissingUnits
		}
		if !f.BaseUnits.IsUint64() || !f.QuoteUnits.IsUint64() {
			return 0, ErrFillUnitsOverflow
		}
		baseUnits := f.BaseUnits.Uint64()
		quoteUnits := f.QuoteUnits.Uint64()
		makerID, merr := makerSettleKey(db, poolID, f, takerSide)
		if merr != nil {
			return 0, merr
		}

		if takerSide == lx.Buy {
			// Taker pays quote (from its locked quote) -> maker available quote.
			if err = SpendLocked(db, takerUser, quote, quoteUnits); err != nil {
				return 0, err
			}
			if err = CreditAvailable(db, makerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			// Maker pays base (from its locked base) -> taker available base.
			if err = SpendLocked(db, makerID, base, baseUnits); err != nil {
				return 0, err
			}
			if err = CreditAvailable(db, takerUser, base, baseUnits); err != nil {
				return 0, err
			}
			takerSpent += quoteUnits
		} else {
			// Taker pays base (from its locked base) -> maker available base.
			if err = SpendLocked(db, takerUser, base, baseUnits); err != nil {
				return 0, err
			}
			if err = CreditAvailable(db, makerID, base, baseUnits); err != nil {
				return 0, err
			}
			// Maker pays quote (from its locked quote) -> taker available quote.
			if err = SpendLocked(db, makerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			if err = CreditAvailable(db, takerUser, quote, quoteUnits); err != nil {
				return 0, err
			}
			takerSpent += baseUnits
		}
	}
	return takerSpent, nil
}

// makerSettleKey resolves a fill's MAKER (the resting order) to its full 16-byte
// settlement identity. The ONLY source is the orderuser: row keyed by the maker's
// resting order id — which survives a restart, unlike the in-RAM maker user
// string (8-byte after a row fold). There is NO fallback: a fill against a
// resting order with no orderuser: row FAILS CLOSED with ErrMissingSettlementUser
// (the transition is rejected, no value moves).
func makerSettleKey(db Store, poolID [32]byte, f lx.Trade, takerSide lx.Side) (AccountID, error) {
	makerOrderID := f.SellOrder // taker bought -> maker sold
	if takerSide == lx.Sell {
		makerOrderID = f.BuyOrder // taker sold -> maker bought
	}
	u, ok, err := GetOrderUser(db, poolID, makerOrderID)
	if err != nil {
		return AccountID{}, err
	}
	if !ok {
		return AccountID{}, fmt.Errorf("%w: maker order %d in market %x", ErrMissingSettlementUser, makerOrderID, poolID[:8])
	}
	return u, nil
}

// DecrementMakerReserves reduces each filled maker's per-order reserve by the
// amount of its locked asset the cross consumed, so a later cancel of a partially-
// filled maker unlocks only the still-resting remainder (a fully-filled maker's
// reserve is deleted). A maker on the opposite side of the taker locked: taker BUY
// -> maker SOLD base (reserve in base, reduced by fill base units); taker SELL ->
// maker BOUGHT base (reserve in quote, reduced by fill quote units). Drops the
// maker's orderuser: row in lockstep when its reserve reaches zero so no orphan
// identity survives a fully-filled order.
func DecrementMakerReserves(db Store, poolID [32]byte, takerSide lx.Side, base, quote AssetID, fills []lx.Trade) error {
	consumed := map[uint64]uint64{}
	for _, f := range fills {
		var makerOrderID, amt uint64
		if takerSide == lx.Buy {
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
		asset, reserve, ok, err := GetOrderLock(db, poolID, makerOrderID)
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
		if err := PutOrderLock(db, poolID, makerOrderID, asset, newReserve); err != nil {
			return err
		}
		if newReserve == 0 {
			if err := DeleteOrderUser(db, poolID, makerOrderID); err != nil {
				return err
			}
		}
	}
	return nil
}

// ---- lock math (the notional floor) ----

// QuoteUnitsCeil returns ceil(baseUnits * priceInt / PriceMultiplier) — the quote
// a BUY of baseUnits at limit priceInt could AT MOST owe. The matcher credits
// floor(...) per fill at the MAKER's price (<= the taker's limit for a buy), so
// the summed floored spend is <= this ceiled lock. Locking the ceil guarantees
// lock >= spend, so a settle never overspends the lock.
func QuoteUnitsCeil(baseUnits *big.Int, priceInt lx.PriceInt) uint64 {
	if baseUnits == nil || baseUnits.Sign() <= 0 || priceInt <= 0 {
		return 0
	}
	q := new(big.Int).Mul(baseUnits, big.NewInt(int64(priceInt)))
	q.Add(q, new(big.Int).Sub(priceMultiplierBig, big.NewInt(1)))
	q.Quo(q, priceMultiplierBig)
	if !q.IsUint64() {
		return 0
	}
	return q.Uint64()
}

// SizeToUnits converts a wire size float to atomic base units (whole-unit
// truncation toward zero — a fractional dust sub-unit is never minted).
func SizeToUnits(size float64) uint64 {
	if !(size > 0) {
		return 0
	}
	return uint64(size)
}

// PriceToInt converts a wire price float to the fixed-point PriceInt grid the
// matcher keys levels by, so the lock's price and the matcher's crossing price are
// the SAME integer.
func PriceToInt(price float64) lx.PriceInt {
	if !(price > 0) {
		return 0
	}
	return lx.PriceInt(price * lx.PriceMultiplier)
}

// OrderLock returns the (asset, amount) a place/limit-submit of (side, price,
// size) on a market with (base, quote) assets must lock. A buy locks quote =
// ceil(size*price); a sell locks base = size. amount 0 means "nothing to lock".
func OrderLock(side lx.Side, price, size float64, base, quote AssetID) (asset AssetID, amount uint64) {
	units := SizeToUnits(size)
	if units == 0 {
		return AssetID{}, 0
	}
	if side == lx.Buy {
		return quote, QuoteUnitsCeil(new(big.Int).SetUint64(units), PriceToInt(price))
	}
	return base, units
}

// FloorsToZeroLock reports whether an EXECUTABLE order (size>0, and for a limit
// price>0) on (base,quote) would lock ZERO — the named "free executable order"
// hazard. A non-executable order returns false (it is handled as malformed
// earlier), so the dust reject is reserved for the precise "executable yet free"
// case.
func FloorsToZeroLock(side lx.Side, price, size float64, base, quote AssetID) bool {
	if !(size > 0) {
		return false
	}
	if side == lx.Buy && !(price > 0) {
		return false
	}
	_, lockAmt := OrderLock(side, price, size, base, quote)
	return lockAmt == 0
}
