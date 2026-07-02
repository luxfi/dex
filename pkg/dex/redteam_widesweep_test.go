// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// RED-TEAM wide sweep, now a REGRESSION GUARD for the exact-integer fix.
//
// The wire carries exact integers (price = PriceInt, size = atomic base units)
// and the persisted row stores them verbatim (persist.go), so the fresh->restart
// projection is the IDENTITY: F(w)=w and G(F(w))=w, hence F(G(F(w)))=F(w) trivially.
// The former fork came from round(size*1e8)/Q64.64-of-float64 being non-idempotent
// under the fixedQtyToFloat reload; those lossy conversions no longer exist. This
// sweep hammers the FULL uint64 quantity range and the full PriceInt range —
// including values above 2^53 that float64 cannot represent — through the actual
// row codec and asserts byte-stability across a rebuild.
package dex

import (
	"bytes"
	"math/big"
	"math/rand"
	"testing"
	"time"
)

func TestRedteam_WideSweep_RestartIdempotence(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE)) // deterministic

	const symbol = "LUX-LUSD"
	const N = 5_000_000
	priceFails, qtyFails := 0, 0

	// Price codec: an exact PriceInt round-trips through the row with no drift.
	for i := 0; i < N; i++ {
		p := PriceInt(rng.Uint64() >> 1) // full non-negative int64 range
		if priceUnitsFromRow(priceRowFor(p)) != p {
			priceFails++
		}
	}

	// Quantity + full row: a resting order with a full-range uint64 size projects
	// to a row, rebuilds, and re-projects BYTE-IDENTICALLY.
	ts := time.Unix(0, 1_000_000_000).UTC()
	for i := 0; i < N/50; i++ {
		size := rng.Uint64() // includes >2^53, top bit set
		priceUnits := PriceInt(rng.Uint64() >> 1)
		ord := &Order{
			ID:            uint64(i) + 1,
			Type:          Limit,
			Side:          Sell,
			PriceUnits:    priceUnits,
			Price:         float64(int64(priceUnits)) / float64(PriceMultiplier),
			Size:          float64(size),
			RemainingSize: float64(size),
			SizeUnits:     new(big.Int).SetUint64(size),
			User:          "victim",
			UserID:        "victim",
			Symbol:        symbol,
			Timestamp:     ts,
			Status:        Open,
		}
		fresh := EncodeRow(OrderToRow(ord))
		// Rebuild from the row (the restart path) then re-serialize.
		rebuilt := EncodeRow(OrderToRow(RowToOrder(OrderToRow(ord))))
		if !bytes.Equal(fresh, rebuilt) {
			qtyFails++
		}
	}

	if priceFails > 0 || qtyFails > 0 {
		t.Fatalf("EXACT-INTEGER REGRESSION: priceFails=%d qtyFails=%d — a float64 round-trip "+
			"has re-entered the persist layer and can fork a restarted validator", priceFails, qtyFails)
	}
	t.Logf("GUARD OK: %d price + %d full-row samples all round-trip byte-identically (exact-integer wire, no restart fork)", N, N/50)
}
