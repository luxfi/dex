// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"math/big"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
)

// settle.go is the CUSTODY transition — the part that makes "the money live in
// the order book". It runs INSIDE the deterministic block execution (block.go),
// against the same versiondb overlay the order/trade rows are written to, so a
// balance move commits atomically with the match that produced it. It is a pure
// integer function of (tx, matcher result, market assets, ledger state): no
// float arithmetic touches a value, so two validators replaying the same ordered
// block produce byte-identical balances and hence the same execution root.
//
// THE MODEL (one direction per leg, integer-exact, conserving):
//   - DEPOSIT  : available += amount.
//   - PLACE    : lock the spend asset (buy -> quote, sell -> base) from available
//                into locked; record the per-order reserve so cancel/fill can
//                unlock the exact remaining amount. Insufficient available =>
//                the order does NOT rest (rejected), nothing is locked.
//   - SUBMIT   : lock the taker's spend up-front (limit: exact want; market: all
//                available), MATCH, then for each fill move value:
//                  taker.locked[spend] -= fill.spend ; maker.available[spend] += fill.spend
//                  maker.locked[recv]  -= fill.recv  ; taker.available[recv]  += fill.recv
//                then unlock the taker's unspent remainder. Every unit removed
//                from one account's locked is added to exactly one other
//                account's available => sum(available)+sum(locked) is invariant.
//   - CANCEL   : unlock the order's remaining reserve (locked -> available).
//   - WITHDRAW : available -= min(want, available) (clamped); the realized amount
//                is what the proxy exports.
//
// AFFORDABILITY POLICY: a LIMIT order is funded ALL-OR-NOTHING — if available <
// the full lock want, the order is rejected (not partially funded). This makes
// "lock >= spend" hold by construction (fills for a buy occur at maker prices <=
// the taker's limit, so sum(spend) <= ceil(size*limit) = lock), so a settle can
// never spend more than was locked. MARKET orders lock all available of the
// spend asset; spend <= lock holds because you cannot spend what you do not have.

// userID8 folds a user-identity string to the 8-byte ledger/UserID handle,
// byte-identically to lx's row encoding (big-endian over the leading 8 bytes,
// zero-padded). The ledger key and the matcher's DEXTrade.MakerUserID/TakerUserID
// MUST agree on this fold, so settlement credits the same account the trade
// names. (lx's helper is unexported; replicating the trivial, canonical fold here
// keeps the change inside the d-chain boundary. TestUserIDFoldMatchesLx pins the
// parity.)
func userID8(user string) uint64 {
	var b [8]byte
	copy(b[:], user)
	return binary.BigEndian.Uint64(b[:])
}

// priceMultiplierBig mirrors lx.PriceMultiplier as a big.Int for exact lock math.
var priceMultiplierBig = big.NewInt(lx.PriceMultiplier)

// quoteUnitsCeil returns ceil(baseUnits * priceInt / PriceMultiplier) — the
// quote a BUY of baseUnits at limit priceInt could AT MOST owe. It is the CEIL
// counterpart of the matcher's floor quoteUnitsFor: the matcher credits/charges
// floor(...) per fill at the MAKER's price, and maker prices for a buy are <= the
// taker's limit, so the summed floored spend is <= this ceiled lock. Locking the
// ceil therefore guarantees lock >= spend, so a settle never overspends the lock.
func quoteUnitsCeil(baseUnits *big.Int, priceInt lx.PriceInt) uint64 {
	if baseUnits == nil || baseUnits.Sign() <= 0 || priceInt <= 0 {
		return 0
	}
	q := new(big.Int).Mul(baseUnits, big.NewInt(int64(priceInt)))
	// ceil(a/b) = (a + b - 1) / b for positive a,b.
	q.Add(q, new(big.Int).Sub(priceMultiplierBig, big.NewInt(1)))
	q.Quo(q, priceMultiplierBig)
	if !q.IsUint64() {
		return 0 // refuse a lock that cannot be represented (caller treats 0 as reject)
	}
	return q.Uint64()
}

// sizeToUnits converts a wire size float to atomic base units. On the consensus
// path sizes are whole atomic-unit counts (the venue quotes integer lot sizes);
// the conversion truncates toward zero, so a fractional dust sub-unit is never
// minted into the lock. It is deterministic over the deterministic float input.
func sizeToUnits(size float64) uint64 {
	if !(size > 0) {
		return 0
	}
	u := uint64(size)
	return u
}

// priceToInt converts a wire price float to the fixed-point PriceInt grid the
// matcher keys levels by (round(price * PriceMultiplier)), so the lock's price
// and the matcher's crossing price are the SAME integer.
func priceToInt(price float64) lx.PriceInt {
	if !(price > 0) {
		return 0
	}
	return lx.PriceInt(price * lx.PriceMultiplier)
}

// orderLock returns the (asset, amount) a place/limit-submit of (side, price,
// size) on a market with (base, quote) assets must lock. A buy locks quote =
// ceil(size * price); a sell locks base = size. amount 0 means "nothing to lock"
// (a degenerate order the matcher will reject anyway).
func orderLock(side uint8, price float64, size float64, base, quote uint64) (asset, amount uint64) {
	units := sizeToUnits(size)
	if units == 0 {
		return 0, 0
	}
	if side == sideBuy {
		return quote, quoteUnitsCeil(new(big.Int).SetUint64(units), priceToInt(price))
	}
	return base, units
}

// sideBuy / sideSell are the d-chain's local side constants (== zapwire/lx).
const (
	sideBuy  uint8 = 0
	sideSell uint8 = 1
)

// creditDeposit applies a TxDeposit: available[user,asset] += amount. Returns the
// amount credited (== amount; a deposit is never clamped).
func creditDeposit(db database.KeyValueReaderWriterDeleter, user string, asset, amount uint64) (uint64, error) {
	if err := creditAvailable(db, userID8(user), asset, amount); err != nil {
		return 0, err
	}
	return amount, nil
}

// debitWithdraw applies a TxWithdraw: available[user,asset] -= min(want,
// available). Returns the REALIZED amount debited so the proxy exports exactly
// what left the ledger (a withdraw can only export realized, un-escrowed
// balance). Clamping to available means an over-withdraw exports only what is
// there — never mints.
func debitWithdraw(db database.KeyValueReaderWriterDeleter, user string, asset, want uint64) (uint64, error) {
	uid := userID8(user)
	avail, err := getAvailable(db, uid, asset)
	if err != nil {
		return 0, err
	}
	amount := want
	if amount > avail {
		amount = avail
	}
	if amount == 0 {
		return 0, nil
	}
	if err := debitAvailable(db, uid, asset, amount); err != nil {
		return 0, err
	}
	return amount, nil
}

// settleFills moves value for the fills a submit produced, INSIDE the ledger,
// from the matcher's AUTHORITATIVE integer lane (lx.Trade.BaseUnits /
// QuoteUnits). takerSide is the submitting side; each fill's BuyUserID/SellUserID
// name the two accounts. Using the matcher's own exact integers (not a
// reconstruction) guarantees the taker is charged EXACTLY what the maker is
// credited — conservation across the maker/taker boundary by construction.
//
// For each fill:
//   - taker BUY  spends QuoteUnits (from taker locked quote) -> maker available
//     quote; the maker SOLD base, so maker locked base is spent -> taker
//     available base.
//   - taker SELL spends BaseUnits (from taker locked base) -> maker available
//     base; the maker BOUGHT base, so maker locked quote is spent -> taker
//     available quote.
//
// Returns the taker's total spend (so the caller unlocks the taker's unspent
// remainder). Requires every fill to carry the integer lane (BaseUnits/QuoteUnits
// non-nil); a fill without it is a programming error on the consensus path (the
// VM sets SizeUnits on every order) and is refused rather than silently
// mis-settled.
func settleFills(db database.KeyValueReaderWriterDeleter, takerSide uint8, base, quote uint64, fills []lx.Trade) (takerSpent uint64, err error) {
	for _, f := range fills {
		if f.BaseUnits == nil || f.QuoteUnits == nil {
			return 0, ErrFillMissingUnits
		}
		if !f.BaseUnits.IsUint64() || !f.QuoteUnits.IsUint64() {
			return 0, ErrFillUnitsOverflow
		}
		baseUnits := f.BaseUnits.Uint64()
		quoteUnits := f.QuoteUnits.Uint64()
		makerID := userID8(makerUserID(f, takerSide))
		takerID := userID8(takerUserID(f, takerSide))

		if takerSide == sideBuy {
			// Taker pays quote (from its locked quote) -> maker available quote.
			if err = spendLocked(db, takerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			if err = creditAvailable(db, makerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			// Maker pays base (from its locked base) -> taker available base.
			if err = spendLocked(db, makerID, base, baseUnits); err != nil {
				return 0, err
			}
			if err = creditAvailable(db, takerID, base, baseUnits); err != nil {
				return 0, err
			}
			takerSpent += quoteUnits
		} else {
			// Taker pays base (from its locked base) -> maker available base.
			if err = spendLocked(db, takerID, base, baseUnits); err != nil {
				return 0, err
			}
			if err = creditAvailable(db, makerID, base, baseUnits); err != nil {
				return 0, err
			}
			// Maker pays quote (from its locked quote) -> taker available quote.
			if err = spendLocked(db, makerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			if err = creditAvailable(db, takerID, quote, quoteUnits); err != nil {
				return 0, err
			}
			takerSpent += baseUnits
		}
	}
	return takerSpent, nil
}

// makerUserID returns the resting (maker) user string for a fill given the taker
// side. If the taker bought, the maker sold (SellUserID); else the maker bought
// (BuyUserID). Mirrors lx.persist.makerUser so the settled account matches the
// trade row.
func makerUserID(f lx.Trade, takerSide uint8) string {
	if takerSide == sideBuy {
		return pickUserID(f.SellUserID, f.Seller)
	}
	return pickUserID(f.BuyUserID, f.Buyer)
}

// takerUserID returns the taker user string for a fill given the taker side.
func takerUserID(f lx.Trade, takerSide uint8) string {
	if takerSide == sideBuy {
		return pickUserID(f.BuyUserID, f.Buyer)
	}
	return pickUserID(f.SellUserID, f.Seller)
}

// pickUserID prefers the explicit UserID, falling back to the user name (the
// matcher sets both equal on the consensus path).
func pickUserID(primary, fallback string) string {
	if primary != "" {
		return primary
	}
	return fallback
}
