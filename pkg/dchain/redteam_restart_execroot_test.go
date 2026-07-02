// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// RED-TEAM (consensus fork) at the ACTUAL dchain root function — now a REGRESSION
// GUARD for the fix: a resting order with a large EXACT wire size yields an
// IDENTICAL bookRoot / ExecutionRoot before vs after a restart-forced rebuild.
// The old float64 encoding (round(size*1e8) reloaded via fixedQtyToFloat) drifted
// by 1 unit and forked a restarted validator; the exact-integer row eliminates it.
package dchain

import (
	"math/big"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
)

func TestRedteam_RestartForksExecutionRoot(t *testing.T) {
	const symbol = "LUX-LUSD"
	// A large EXACT base-unit size above float64's 2^53 exact ceiling — the regime
	// that used to fork. The wire now carries this integer verbatim (no float64).
	const bigSize = uint64(0x0020000000000001) // 2^53 + 1
	priceUnits := dex.PriceInt(25) * dex.PriceInt(dex.PriceMultiplier)

	// FRESH validator: rest the order and serialize the canonical rows.
	obFresh := dex.NewOrderBook(symbol)
	ord := &dex.Order{
		ID:            0xABCDEF01,
		Type:          dex.Limit,
		Side:          dex.Sell,
		PriceUnits:    priceUnits,
		Price:         float64(int64(priceUnits)) / float64(dex.PriceMultiplier),
		Size:          float64(bigSize),
		RemainingSize: float64(bigSize),
		SizeUnits:     new(big.Int).SetUint64(bigSize),
		User:          "victimmaker",
		UserID:        "victimmaker",
		Symbol:        symbol,
		Timestamp:     time.Unix(0, 1_000_000_000).UTC(),
		Status:        dex.Open,
	}
	if obFresh.ConsensusAddOrder(ord) == 0 {
		t.Fatalf("fresh: order rejected")
	}
	rowsFresh := dex.BookToRows(obFresh)

	// RESTARTED validator: rebuild from the SAME committed rows (VM.Initialize
	// path), then re-serialize — exactly what the next block's root computation
	// consumes.
	obRestart := dex.RowsToBook(symbol, rowsFresh)
	rowsRestart := dex.BookToRows(obRestart)

	// Compute the ACTUAL consensus roots the VM commits (state.go).
	sortRowsCanonical(rowsFresh)
	sortRowsCanonical(rowsRestart)
	bookFresh := bookRoot(rowsFresh)
	bookRestart := bookRoot(rowsRestart)

	var parent, tr, txr, ledger [Size]byte
	execFresh := ComposeRoot(parent, bookFresh, tr, txr, ledger, 2)
	execRestart := ComposeRoot(parent, bookRestart, tr, txr, ledger, 2)

	if rowsFresh[0].Quantity != bigSize {
		t.Fatalf("row did not commit the exact size: Quantity=%d want %d", rowsFresh[0].Quantity, bigSize)
	}
	if bookFresh != bookRestart || execFresh != execRestart {
		t.Fatalf(`RESTART FORK REGRESSION (dchain root function):
  bookRoot fresh=%x restart=%x
  execRoot fresh=%x restart=%x
  A restarted validator derives a DIFFERENT execution root than a non-restarted
  one from the SAME committed order set — the exact-integer row round-trip broke.`,
			bookFresh, bookRestart, execFresh, execRestart)
	}
	t.Logf("GUARD OK: bookRoot/execRoot byte-stable across restart for a 2^53+1 size (exact-integer wire)")
}
