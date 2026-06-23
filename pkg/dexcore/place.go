// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"fmt"
	"math/big"
	"time"

	"github.com/luxfi/dex/pkg/lx"
)

// place.go is the resting-liquidity path: a maker places a LIMIT order that rests
// on the book (the counterparty a taker's swap crosses). It is the custody-gated
// place from the proven dchain path, now over the shared Store: lock the spend
// asset, rest the order deterministically, persist the row + the per-order reserve
// (orderasset:) + the FULL settlement identity (orderuser:) so a later fill/cancel
// settles to the exact account even after a restart.
//
// Cancel is the inverse: unlock the order's remaining reserve to its recorded full
// identity (fail-closed if the orderuser: row is missing) and delete the rows.

// PlaceOrder rests a maker LIMIT order on the market. maker is the FULL 16-byte
// settlement identity; orderID is the block-deterministic id; ts is the block time.
// The spend asset is locked (buy -> ceil(size*price) quote, sell -> size base) and
// the per-order reserve + identity rows are recorded. Returns the resting order id.
//
// Rejects (deterministic, no error): an unaffordable lock (insufficient available)
// or a zero-notional executable order (dust price / sub-unit size) — the order does
// NOT rest and nothing is locked, signaled by ok=false. A post-only-would-take
// rejection also yields ok=false with the lock refunded.
func PlaceOrder(db Store, poolID [32]byte, maker AccountID, side lx.Side, price, size float64, orderID uint64, ts time.Time) (ok bool, err error) {
	base, quote, bound, err := ReadMarketAssets(db, poolID)
	if err != nil {
		return false, err
	}
	if !bound {
		return false, ErrUnknownMarket
	}
	if !(price > 0) || !(size > 0) {
		return false, nil // malformed: not an error, just no rest
	}
	// Custody gate: an executable order whose notional floors to zero lock is the
	// "free executable order" hazard — reject it.
	if FloorsToZeroLock(side, price, size, base, quote) {
		return false, ErrDustPriceOrZeroLock
	}
	lockAsset, lockAmt := OrderLock(side, price, size, base, quote)
	if lockAmt == 0 {
		return false, ErrDustPriceOrZeroLock
	}
	// Lock the spend. Insufficient available => deterministic reject (no rest).
	if err := LockFromAvailable(db, maker, lockAsset, lockAmt); err != nil {
		if err == ErrInsufficientAvailable {
			return false, nil
		}
		return false, err
	}

	// Rest the order via the deterministic consensus entrypoint, over a rebuilt book.
	symbol, exists, err := ReadMarketSymbol(db, poolID)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, ErrUnknownMarket
	}
	book, err := RebuildBook(db, poolID, symbol)
	if err != nil {
		return false, err
	}
	order := &lx.Order{
		ID:        orderID,
		Type:      lx.Limit,
		Side:      side,
		Price:     price,
		Size:      size,
		SizeUnits: new(big.Int).SetUint64(SizeToUnits(size)),
		User:      string(trimTrailingZeros(maker[:])),
		UserID:    string(trimTrailingZeros(maker[:])),
		Symbol:    symbol,
		Timestamp: ts,
	}
	if book.ConsensusAddOrder(order) == 0 {
		// Rejected (e.g. post-only would take): refund the lock, rest nothing.
		if uerr := UnlockToAvailable(db, maker, lockAsset, lockAmt); uerr != nil {
			return false, uerr
		}
		return false, nil
	}

	// Persist the resting row + the per-order reserve + the full settlement identity.
	row := lx.OrderToRow(order)
	row.OrderID = order.ID
	if err := PutOrderRow(db, poolID, row); err != nil {
		return false, err
	}
	if err := PutOrderLock(db, poolID, orderID, lockAsset, lockAmt); err != nil {
		return false, err
	}
	if err := PutOrderUser(db, poolID, orderID, maker); err != nil {
		return false, err
	}
	if err := WriteBookWatermark(db, poolID, book.LastOrderID); err != nil {
		return false, err
	}
	return true, nil
}

// CancelOrder removes a resting order and unlocks its remaining reserve to the
// order's recorded FULL settlement identity (orderuser:) — the ONLY source. A
// cancel of an order that holds a non-zero reserve but has NO orderuser: row is
// state corruption: FAIL CLOSED (ErrMissingSettlementUser), never unlock to a
// compact-handle-derived account. A cancel of an unknown order is a deterministic
// no-op (ok=false).
func CancelOrder(db Store, poolID [32]byte, orderID uint64) (ok bool, err error) {
	symbol, exists, err := ReadMarketSymbol(db, poolID)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, nil
	}
	book, err := RebuildBook(db, poolID, symbol)
	if err != nil {
		return false, err
	}
	if book.GetOrder(orderID) == nil {
		return false, nil // unknown order: no-op
	}
	if cerr := book.CancelOrder(orderID); cerr != nil {
		return false, nil // matcher reject: no-op
	}

	// Unlock the remaining reserve to the recorded full identity.
	asset, amt, hasLock, err := GetOrderLock(db, poolID, orderID)
	if err != nil {
		return false, err
	}
	if hasLock && amt > 0 {
		owner, hasUser, uerr := GetOrderUser(db, poolID, orderID)
		if uerr != nil {
			return false, uerr
		}
		if !hasUser {
			return false, fmt.Errorf("%w: cancel of order %d in market %x", ErrMissingSettlementUser, orderID, poolID[:8])
		}
		if err := UnlockToAvailable(db, owner, asset, amt); err != nil {
			return false, err
		}
	}
	// Delete the persisted row + the per-order reserve + the identity row.
	if err := DeleteOrderRow(db, poolID, orderID); err != nil {
		return false, err
	}
	if err := PutOrderLock(db, poolID, orderID, asset, 0); err != nil {
		return false, err
	}
	if err := DeleteOrderUser(db, poolID, orderID); err != nil {
		return false, err
	}
	if err := WriteBookWatermark(db, poolID, book.LastOrderID); err != nil {
		return false, err
	}
	return true, nil
}

// Deposit credits a maker/taker's available balance for an asset — the on-ramp the
// custody ledger draws from. In the precompile this mirrors a real ERC-20/native
// lock in the 0x9999 vault; here it is the ledger-side credit. The asset is the
// FULL injective id.
func Deposit(db Store, account AccountID, asset AssetID, amount uint64) error {
	return CreditAvailable(db, account, asset, amount)
}

// OpenMarket binds a market's (base,quote) assets and records its existence, so the
// custody gate value-checks orders. Idempotent.
func OpenMarket(db Store, poolID [32]byte, base, quote AssetID) error {
	if exists, err := MarketExists(db, poolID); err != nil {
		return err
	} else if !exists {
		if err := WriteMarket(db, poolID, PoolSymbol(poolID)); err != nil {
			return err
		}
	}
	return WriteMarketAssets(db, poolID, base, quote)
}
