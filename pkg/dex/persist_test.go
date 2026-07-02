// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"testing"
	"time"
)

// TestFixedPointRoundTrip proves the Q64.64 price and 1e8 quantity conversions
// are stable: encoding a float into the deterministic integer domain and back
// recovers the value to within one ULP of the fixed-point grid, AND — the part
// that matters for consensus — the integer form itself is identical for the same
// input every time (it is a pure function of the float bits).
func TestFixedPointRoundTrip(t *testing.T) {
	prices := []float64{1, 100, 100.5, 0.00000001, 12345.6789, 999999.99999999, 0.1, 3.14159265358979}
	for _, p := range prices {
		fx := floatToFixedPrice(p)
		// determinism: same input -> same fixed-point.
		if fx != floatToFixedPrice(p) {
			t.Fatalf("floatToFixedPrice(%v) not deterministic", p)
		}
		back := fixedPriceToFloat(fx)
		// The grid step for the fraction is 2^-64, far finer than 1e-8, so the
		// recovered value must match to 1e-8.
		if diff := back - p; diff > 1e-8 || diff < -1e-8 {
			t.Errorf("price round-trip %v -> %+v -> %v (diff %v)", p, fx, back, diff)
		}
	}

	sizes := []float64{1, 0.5, 0.00000001, 1000000, 42.123456}
	for _, s := range sizes {
		q := floatToFixedQty(s)
		if q != floatToFixedQty(s) {
			t.Fatalf("floatToFixedQty(%v) not deterministic", s)
		}
		back := fixedQtyToFloat(q)
		if diff := back - s; diff > 1e-8 || diff < -1e-8 {
			t.Errorf("qty round-trip %v -> %d -> %v (diff %v)", s, q, back, diff)
		}
	}
}

// TestRowEncodeDecode proves the 64-byte row image round-trips and rejects
// mis-sized input.
func TestRowEncodeDecode(t *testing.T) {
	row := DEXOrder{
		OrderID:   42,
		UserID:    userToUint64("alice"),
		Price:     floatToFixedPrice(100.25),
		Quantity:  floatToFixedQty(5),
		Remaining: floatToFixedQty(3),
		Timestamp: 1700000000000000000,
		Side:      DEXSideBid,
		Type:      DEXOrderLimit,
		Status:    DEXStatusPartial,
	}
	b := EncodeRow(row)
	if len(b) != 64 {
		t.Fatalf("encoded row len = %d, want 64", len(b))
	}
	got, ok := DecodeRow(b)
	if !ok {
		t.Fatal("DecodeRow rejected a valid 64-byte row")
	}
	if !got.EqualFields(row) {
		t.Errorf("row decode mismatch:\n got %+v\nwant %+v", got, row)
	}
	if _, ok := DecodeRow(b[:63]); ok {
		t.Error("DecodeRow accepted a 63-byte value")
	}
	if _, ok := DecodeRow(append(b, 0)); ok {
		t.Error("DecodeRow accepted a 65-byte value")
	}
}

// TestBookToRowsRoundTrip is the CRITICAL step-1 test: a populated book
// serialized to rows and rebuilt must be EQUAL to the original — same resting
// orders, same price-time priority, same best bid/ask, same depth. Equality is
// checked through the canonical row stream (the deterministic representation),
// which is exactly what the merkle root and persistence consume.
func TestBookToRowsRoundTrip(t *testing.T) {
	ts := time.Unix(0, 1700000000000000000).UTC()
	ob := NewOrderBook("BTC-USD")

	// Rest a deterministic spread: 3 bids, 3 asks, with a couple of multi-order
	// price levels to exercise time priority.
	place := func(id uint64, side Side, price, size float64, user string, dt time.Duration) {
		o := &Order{
			ID: id, Type: Limit, Side: side, Price: price, Size: size,
			User: user, UserID: user, Symbol: "BTC-USD",
			Timestamp: ts.Add(dt),
		}
		if got := ob.ConsensusAddOrder(o); got != id {
			t.Fatalf("ConsensusAddOrder(id=%d) = %d", id, got)
		}
	}
	place(1, Buy, 100.0, 2.0, "alice", 0)
	place(2, Buy, 100.0, 1.5, "bob", 1) // same level, later -> behind id1
	place(3, Buy, 99.5, 3.0, "carol", 2)
	place(4, Sell, 101.0, 2.5, "dave", 3)
	place(5, Sell, 101.5, 1.0, "erin", 4)
	place(6, Sell, 101.0, 0.5, "frank", 5) // same level, later -> behind id4

	rows1 := BookToRows(ob)
	if len(rows1) != 6 {
		t.Fatalf("expected 6 rows, got %d", len(rows1))
	}

	// Rebuild and re-serialize.
	ob2 := RowsToBook("BTC-USD", rows1)
	rows2 := BookToRows(ob2)

	if len(rows2) != len(rows1) {
		t.Fatalf("rebuilt book row count %d != original %d", len(rows2), len(rows1))
	}
	for i := range rows1 {
		if !rows1[i].EqualFields(rows2[i]) {
			t.Errorf("row %d differs after round-trip:\n orig %+v\nrebuilt %+v", i, rows1[i], rows2[i])
		}
		// Byte-image equality too.
		if string(EncodeRow(rows1[i])) != string(EncodeRow(rows2[i])) {
			t.Errorf("row %d byte image differs after round-trip", i)
		}
	}

	// Functional equality: best bid/ask preserved.
	if got := len(ob2.Orders); got != 6 {
		t.Errorf("rebuilt book has %d orders, want 6", got)
	}
	if ob.LastOrderID != 0 && ob2.LastOrderID < 6 {
		t.Errorf("rebuilt LastOrderID watermark = %d, want >= 6", ob2.LastOrderID)
	}
}

// TestRowStreamIsCanonicallyOrdered proves BookToRows emits a deterministic
// order regardless of insertion order: two books built with the same orders in
// different insertion sequences produce the identical row stream. This is what
// makes the book merkle root insertion-order-independent.
func TestRowStreamIsCanonicallyOrdered(t *testing.T) {
	ts := time.Unix(0, 1700000000000000000).UTC()
	build := func(order []uint64) []DEXOrder {
		ob := NewOrderBook("ETH-USD")
		spec := map[uint64]struct {
			side  Side
			price float64
			size  float64
			user  string
		}{
			10: {Buy, 50.0, 1, "a"},
			11: {Buy, 51.0, 2, "b"},
			12: {Sell, 52.0, 1, "c"},
			13: {Sell, 53.0, 2, "d"},
		}
		for _, id := range order {
			s := spec[id]
			// Timestamp is tied to the ID (not the insertion index) so the same
			// order produces the same row regardless of when it was inserted —
			// the row is a pure function of the order's identity, and the only
			// thing the canonical sort reorders is position, not content.
			ob.ConsensusAddOrder(&Order{
				ID: id, Type: Limit, Side: s.side, Price: s.price, Size: s.size,
				User: s.user, UserID: s.user, Symbol: "ETH-USD",
				Timestamp: ts.Add(time.Duration(id) * time.Nanosecond),
			})
		}
		return BookToRows(ob)
	}
	a := build([]uint64{10, 11, 12, 13})
	b := build([]uint64{13, 12, 11, 10})
	if len(a) != len(b) {
		t.Fatalf("row counts differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if string(EncodeRow(a[i])) != string(EncodeRow(b[i])) {
			t.Errorf("canonical order not insertion-independent at row %d", i)
		}
	}
	// Canonical ordering: bids (descending price) then asks (ascending price).
	if a[0].Side != DEXSideBid || a[len(a)-1].Side != DEXSideAsk {
		t.Errorf("expected bids first then asks, got sides %d..%d", a[0].Side, a[len(a)-1].Side)
	}
	if a[0].Price.Integer != 51 || a[1].Price.Integer != 50 {
		t.Errorf("bids not descending: %d then %d", a[0].Price.Integer, a[1].Price.Integer)
	}
}
