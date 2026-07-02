// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// RED-TEAM (consensus fork on restart) — now a REGRESSION GUARD for the fix.
//
// A RESTING order's CANONICAL ROW (the exact bytes orderLeafDigest hashes into
// the bookRoot) MUST be stable across a validator restart. The live path derives
// the row from the order's exact integer lane (SizeUnits/PriceUnits); a restart
// rebuilds that lane verbatim from the stored row (RowToOrder). With exact
// integers on the wire and in the row, fresh == restart byte-for-byte. The old
// float64 encoding (round(size*1e8) reloaded via fixedQtyToFloat) drifted by 1
// unit for reachable large sizes and forked the set on any restart; this guards
// that it stays fixed.
package dex

import (
	"bytes"
	"math/big"
	"testing"
)

// TestRedteam_RestartForksBookRoot rests a LARGE-size limit order (a base-unit
// count above float64's 2^53 exact-integer ceiling — the exact regime that used
// to fork), snapshots its canonical row, simulates a restart via RowsToBook, and
// re-snapshots. The bytes MUST be identical.
func TestRedteam_RestartForksBookRoot(t *testing.T) {
	const symbol = "LUX-LUSD"

	// A large exact base-unit size that float64 cannot represent exactly (and that
	// the old round(size*1e8) encoding would have clamped/drifted).
	const bigSize = uint64(0x0020000000000001) // 2^53 + 1, > float64 exact ceiling
	priceUnits := PriceInt(25) * PriceInt(PriceMultiplier)

	// --- FRESH validator: rest the order, compute its canonical row bytes. ---
	obFresh := NewOrderBook(symbol)
	ord := &Order{
		ID:            0xABCDEF01,
		Type:          Limit,
		Side:          Sell,
		PriceUnits:    priceUnits,
		Price:         float64(int64(priceUnits)) / float64(PriceMultiplier),
		Size:          float64(bigSize),
		RemainingSize: float64(bigSize),
		SizeUnits:     new(big.Int).SetUint64(bigSize),
		User:          "victimmaker",
		UserID:        "victimmaker",
		Symbol:        symbol,
		Timestamp:     nanosToTime(1_000_000_000),
		Status:        Open,
	}
	if obFresh.ConsensusAddOrder(ord) == 0 {
		t.Fatalf("fresh: order rejected by ConsensusAddOrder")
	}
	rowsFresh := BookToRows(obFresh)
	if len(rowsFresh) != 1 {
		t.Fatalf("fresh: expected 1 resting row, got %d", len(rowsFresh))
	}
	leafFresh := EncodeRow(rowsFresh[0])

	// The row committed the EXACT integer size — no clamp, no ×1e8 drift.
	if rowsFresh[0].Quantity != bigSize {
		t.Fatalf("row did not commit the exact size: got Quantity=%d want %d", rowsFresh[0].Quantity, bigSize)
	}

	// --- RESTARTED validator: rebuild from the SAME committed rows, re-serialize. ---
	obRestart := RowsToBook(symbol, rowsFresh)
	rowsRestart := BookToRows(obRestart)
	if len(rowsRestart) != 1 {
		t.Fatalf("restart: expected 1 resting row, got %d", len(rowsRestart))
	}
	leafRestart := EncodeRow(rowsRestart[0])

	if !bytes.Equal(leafFresh, leafRestart) {
		t.Fatalf(`RESTART FORK REGRESSION:
  a resting order's canonical row changed across a restart.
  fresh   Remaining = %d
  restart Remaining = %d
  fresh   row bytes = %x
  restart row bytes = %x`,
			rowsFresh[0].Remaining, rowsRestart[0].Remaining, leafFresh, leafRestart)
	}
	t.Logf("GUARD OK: canonical row byte-stable across restart for a 2^53+1 size (exact-integer wire, no restart fork)")
}
