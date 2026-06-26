// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"encoding/binary"
	"fmt"
	"math/big"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
)

// book.go is the resting-book schema + the rebuild-from-rows discipline. The
// in-RAM lx.OrderBook is a rebuildable ACCELERATOR; the durable order:* rows are
// truth. Both homes fold the same rows into the same book, so the matcher sees the
// identical liquidity. There is ONE rebuild path (RebuildBook), used at init (fold
// every market) and during a swap (fold the touched market).
//
//	order:<poolID:32><orderID:8>  -> DEXOrder (canonical fixed-point row)
//	book:<poolID:32>              -> uint64 LastOrderID watermark
//	market:<poolID:32>            -> symbol bytes (existence + display symbol)

const (
	prefixOrder  = "order:"
	prefixBook   = "book:"
	prefixMarket = "market:"
)

// orderKey builds order:<poolID:32><orderID:8>.
func orderKey(poolID [32]byte, orderID uint64) []byte {
	k := make([]byte, len(prefixOrder)+32+8)
	copy(k, prefixOrder)
	copy(k[len(prefixOrder):], poolID[:])
	binary.BigEndian.PutUint64(k[len(prefixOrder)+32:], orderID)
	return k
}

// orderPrefixFor builds the order:<poolID:32> iteration prefix for one market.
func orderPrefixFor(poolID [32]byte) []byte {
	k := make([]byte, len(prefixOrder)+32)
	copy(k, prefixOrder)
	copy(k[len(prefixOrder):], poolID[:])
	return k
}

// marketKey builds market:<poolID:32>.
func marketKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixMarket)+32)
	copy(k, prefixMarket)
	copy(k[len(prefixMarket):], poolID[:])
	return k
}

// bookKey builds book:<poolID:32> (the per-market LastOrderID watermark).
func bookKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixBook)+32)
	copy(k, prefixBook)
	copy(k[len(prefixBook):], poolID[:])
	return k
}

// PoolSymbol renders a poolId as its hex market symbol (the venue's market
// identity is the poolId; a human symbol is a display concern at the gateway).
func PoolSymbol(poolID [32]byte) string {
	const hexdigits = "0123456789abcdef"
	out := make([]byte, 64)
	for i, c := range poolID {
		out[i*2] = hexdigits[c>>4]
		out[i*2+1] = hexdigits[c&0x0f]
	}
	return string(out)
}

// WriteMarket records a market's existence + symbol (idempotent).
func WriteMarket(db database.KeyValueWriter, poolID [32]byte, symbol string) error {
	return db.Put(marketKey(poolID), []byte(symbol))
}

// ReadMarketSymbol returns the market's symbol, ok=false if the market does not
// exist.
func ReadMarketSymbol(db database.KeyValueReader, poolID [32]byte) (string, bool, error) {
	v, err := db.Get(marketKey(poolID))
	if err == database.ErrNotFound {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return string(v), true, nil
}

// MarketExists reports whether a market:<poolID> row is present.
func MarketExists(db database.KeyValueReader, poolID [32]byte) (bool, error) {
	_, ok, err := ReadMarketSymbol(db, poolID)
	return ok, err
}

// ReadBookWatermark returns the market's LastOrderID watermark (0 if absent).
func ReadBookWatermark(db database.KeyValueReader, poolID [32]byte) (uint64, error) {
	v, err := db.Get(bookKey(poolID))
	if err == database.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if len(v) != 8 {
		return 0, fmt.Errorf("dex: corrupt book watermark len=%d", len(v))
	}
	return binary.BigEndian.Uint64(v), nil
}

// WriteBookWatermark persists the market's LastOrderID watermark.
func WriteBookWatermark(db database.KeyValueWriter, poolID [32]byte, watermark uint64) error {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], watermark)
	return db.Put(bookKey(poolID), b[:])
}

// PutOrderRow writes a resting order row, or deletes it when the order is no
// longer live (filled, cancelled, or zero remaining). The book is the live resting
// set; spent rows are removed so iteration over order:* yields exactly the
// rebuildable liquidity.
func PutOrderRow(db database.KeyValueWriterDeleter, poolID [32]byte, row lx.DEXOrder) error {
	key := orderKey(poolID, row.OrderID)
	if row.Status == lx.DEXStatusFilled || row.Status == lx.DEXStatusCancelled || row.Remaining == 0 {
		return db.Delete(key)
	}
	return db.Put(key, lx.EncodeRow(row))
}

// DeleteOrderRow removes a resting order row (used on cancel).
func DeleteOrderRow(db database.KeyValueDeleter, poolID [32]byte, orderID uint64) error {
	return db.Delete(orderKey(poolID, orderID))
}

// RebuildBook streams the order:<poolID> rows from db and folds them into a fresh
// OrderBook via RowsToBook — the one rebuild path. The persisted DEXOrder row keeps
// quantities in fixed-point, not the matcher's atomic-unit lane, so we restore the
// exact-integer SizeUnits/RemainingUnits on every rebuilt order (else its fills
// would carry no integer truth and the custody settle would refuse them).
func RebuildBook(db database.Iteratee, poolID [32]byte, symbol string) (*lx.OrderBook, error) {
	prefix := orderPrefixFor(poolID)
	it := db.NewIteratorWithPrefix(prefix)
	defer it.Release()

	var rows []lx.DEXOrder
	for it.Next() {
		row, ok := lx.DecodeRow(it.Value())
		if !ok {
			return nil, fmt.Errorf("dex: corrupt order row under %x (len %d)", poolID[:8], len(it.Value()))
		}
		rows = append(rows, row)
	}
	if err := it.Error(); err != nil {
		return nil, err
	}
	ob := lx.RowsToBook(symbol, rows)
	restoreIntegerLane(ob)
	return ob, nil
}

// restoreIntegerLane sets SizeUnits/RemainingUnits (the matcher's exact-integer
// quantity lane) on every resting order in a rebuilt book, derived from the order's
// float size/remaining via the same whole-unit conversion the consensus path uses.
// Idempotent.
func restoreIntegerLane(ob *lx.OrderBook) {
	for _, o := range ob.Orders {
		if o == nil {
			continue
		}
		if o.SizeUnits == nil {
			if u := SizeToUnits(o.Size); u > 0 {
				o.SizeUnits = new(big.Int).SetUint64(u)
			}
		}
		if o.RemainingUnits == nil {
			rem := o.RemainingSize
			if rem == 0 {
				rem = o.Size
			}
			if u := SizeToUnits(rem); u > 0 {
				o.RemainingUnits = new(big.Int).SetUint64(u)
			}
		}
	}
}
