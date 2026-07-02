// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"fmt"
	"math/big"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/dex"
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

// priceMultiplierBig mirrors dex.PriceMultiplier as a big.Int for exact lock math.
var priceMultiplierBig = big.NewInt(dex.PriceMultiplier)

// quoteUnitsCeil returns ceil(baseUnits * priceInt / PriceMultiplier) — the
// quote a BUY of baseUnits at limit priceInt could AT MOST owe. It is the CEIL
// counterpart of the matcher's floor quoteUnitsFor: the matcher credits/charges
// floor(...) per fill at the MAKER's price, and maker prices for a buy are <= the
// taker's limit, so the summed floored spend is <= this ceiled lock. Locking the
// ceil therefore guarantees lock >= spend, so a settle never overspends the lock.
func quoteUnitsCeil(baseUnits *big.Int, priceInt dex.PriceInt) uint64 {
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

// orderLock returns the (asset, amount) a place/limit-submit of (side, priceUnits,
// sizeUnits) on a market with (base, quote) assets must lock. Both inputs are the
// wire's EXACT integers: sizeUnits is atomic base units, priceUnits is the PriceInt
// grid (price × 1e8). A buy locks quote = ceil(size * price / 1e8); a sell locks
// base = size. amount 0 means "nothing to lock" (a degenerate order). asset is the
// FULL 32-byte id (base or quote), so the lock keys the SAME ledger row a deposit
// credited. No float64 touches this path — the lock is byte-identical on every arch.
func orderLock(side uint8, priceUnits uint64, sizeUnits uint64, base, quote [32]byte) (asset [32]byte, amount uint64) {
	if sizeUnits == 0 {
		return [32]byte{}, 0
	}
	if side == sideBuy {
		if priceUnits == 0 || priceUnits > uint64(1)<<63-1 {
			return [32]byte{}, 0
		}
		return quote, quoteUnitsCeil(new(big.Int).SetUint64(sizeUnits), dex.PriceInt(priceUnits))
	}
	return base, sizeUnits
}

// sideBuy / sideSell are the d-chain's local side constants (== zapwire/lx).
const (
	sideBuy  uint8 = 0
	sideSell uint8 = 1
)

// marketConstraint is the MINIMAL notional scaffold: just enough to express the
// one constraint a value-bearing DEX must enforce against integer flooring — an
// executable order MUST reserve a non-zero lock. It deliberately does NOT model a
// full fee/tick/lot/minNotional engine (that is a separate roadmap); the only
// invariant here is "arithmetic flooring can never create a FREE executable order".
//
// The notional floor IS the lock: a BUY's notional is ceil(size*price) quote, a
// SELL's is size base (orderLock). When that floors to zero for a positive-size,
// positive-price executable order — a sub-tick price whose priceInt rounds to 0,
// or a size that truncates below one atomic unit — the order would rest/cross while
// reserving NOTHING. floorsToZeroLock is the single predicate that names that
// condition; the placement gate rejects it with ErrDustPriceOrZeroLock.
type marketConstraint struct {
	base, quote [32]byte
}

// floorsToZeroLock reports whether an EXECUTABLE order (sizeUnits>0) on this market
// would lock ZERO — the zero-notional reject condition. It is exactly "orderLock
// yields a zero amount":
//
//	BUY  : priceUnits==0 (price 0) => quote lock ceil(size*price/1e8) == 0
//	SELL : sizeUnits==0                                                    (handled below)
//
// Because the wire now carries the EXACT PriceInt, the smallest positive price is
// priceUnits==1 (=1e-8), and ceil(size*1/1e8) >= 1 for any size>=1 — so a positive
// price can NEVER floor a buy to zero. Only priceUnits==0 does, which this gate
// catches (rejected with ErrDustPriceOrZeroLock). A zero size is not executable
// and is handled by the matcher's malformed reject, so it is excluded here.
func (mc marketConstraint) floorsToZeroLock(side uint8, priceUnits, sizeUnits uint64) bool {
	if sizeUnits == 0 {
		return false // not executable (zero size) — handled as malformed, not dust
	}
	_, lockAmt := orderLock(side, priceUnits, sizeUnits, mc.base, mc.quote)
	return lockAmt == 0
}

// creditDeposit applies a TxDeposit: available[user,asset] += amount. Returns the
// amount credited (== amount; a deposit is never clamped). asset is the FULL
// 32-byte injective id.
func creditDeposit(db database.KeyValueReaderWriterDeleter, user string, asset [32]byte, amount uint64) (uint64, error) {
	if err := creditAvailable(db, userKey16(user), asset, amount); err != nil {
		return 0, err
	}
	return amount, nil
}

// debitWithdraw applies a TxWithdraw: available[user,asset] -= min(want,
// available). Returns the REALIZED amount debited so the proxy exports exactly
// what left the ledger (a withdraw can only export realized, un-escrowed
// balance). Clamping to available means an over-withdraw exports only what is
// there — never mints. asset is the FULL 32-byte injective id.
func debitWithdraw(db database.KeyValueReaderWriterDeleter, user string, asset [32]byte, want uint64) (uint64, error) {
	uid := userKey16(user)
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
// from the matcher's AUTHORITATIVE integer lane (dex.Trade.BaseUnits /
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
//
// IDENTITY: the settlement identity is the WALLET (AccountID); UserID8 is only a
// matcher hint. The TAKER is the submitter, whose full 16-byte wallet is carried on
// the ephemeral order this block from the decoded dex_submit invocation context
// (never the 8-byte row handle — the taker never rests, so its full wallet is always
// in hand). The MAKER is a RESTING order whose in-RAM user is only 8 bytes after a
// restart (RowToOrder folds the GPU row's 8-byte UserID handle), so its full wallet
// is resolved EXCLUSIVELY through the orderuser: index by the maker's order id —
// there is NO fallback to the trade's 8-byte-derived carried user. This guarantees
// a maker fill credits the SAME full 16-byte wallet the maker's lock lives under,
// live or post-restart.
//
// Settlement fails closed if full account identity is unavailable. Falling back
// to matcher UserID would make compact-handle collisions value-bearing — a critical theft bug.
func settleFills(db database.KeyValueReaderWriterDeleter, poolID [32]byte, takerSide uint8, base, quote [32]byte, fills []dex.Trade) (takerSpent uint64, err error) {
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
		takerID := userKey16(takerUserID(f, takerSide))

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

// makerSettleKey resolves a fill's MAKER (the resting order) to its full 16-byte
// settlement identity. The ONLY source is the orderuser: state row keyed by the
// maker's resting order id — which survives a restart, unlike the in-RAM maker user
// string (8-byte after RowToOrder). There is NO fallback: a fill against a resting
// order with no orderuser: row FAILS CLOSED with ErrMissingSettlementUser (the block
// is rejected, no value moves). Every order that rests on a balance-bearing market
// persists its orderuser: row BEFORE it is matchable (lockOrderSpend), so a missing
// row at settle time can only be state corruption — never a normal path.
//
// Settlement fails closed if full account identity is unavailable. Falling back
// to matcher UserID would make compact-handle collisions value-bearing — a critical theft bug.
func makerSettleKey(db database.KeyValueReader, poolID [32]byte, f dex.Trade, takerSide uint8) (userKey, error) {
	makerOrderID := f.SellOrder // taker bought -> maker sold
	if takerSide == sideSell {
		makerOrderID = f.BuyOrder // taker sold -> maker bought
	}
	u, ok, err := getOrderUser(db, poolID, makerOrderID)
	if err != nil {
		return userKey{}, err
	}
	if !ok {
		return userKey{}, fmt.Errorf("%w: maker order %d in market %x", ErrMissingSettlementUser, makerOrderID, poolID[:8])
	}
	return u, nil
}

// takerUserID returns the taker user string for a fill given the taker side.
func takerUserID(f dex.Trade, takerSide uint8) string {
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
