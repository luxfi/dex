// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

var testPool = [32]byte{0xaa, 0xbb, 0xcc}

func placeTx(t *testing.T, side uint8, price, size float64, user string) *Tx {
	t.Helper()
	a := acctFor(t, user)
	return a.signed(t, TxPlace, zapwire.EncodePlace(testPool, side, price, size, a.user))
}

func submitTx(t *testing.T, side uint8, isMarket bool, price, size float64, user string) *Tx {
	t.Helper()
	a := acctFor(t, user)
	return a.signed(t, TxSubmit, zapwire.EncodeSubmit(testPool, side, isMarket, price, size, a.user))
}

// applyAll replays a tx sequence into a fresh book, returning every fill as a
// canonical DEXTrade row and the resulting book rows. height/txIndex are derived
// exactly as the VM derives them (block height fixed, txIndex = position).
func applyAll(t *testing.T, height uint64, ts time.Time, txs []*Tx) ([]dex.DEXTrade, []dex.DEXOrder) {
	t.Helper()
	book := dex.NewOrderBook("BTC-USD")
	var fills []dex.DEXTrade
	for i, tx := range txs {
		res, err := applyTx(book, tx, height, ts, uint32(i))
		if err != nil {
			t.Fatalf("applyTx[%d] (%s): %v", i, tx.Type, err)
		}
		for _, f := range res.Fills {
			fills = append(fills, dex.TradeToRow(f, res.TakerSide))
		}
	}
	return fills, dex.BookToRows(book)
}

// TestApplyTxDeterminism is the step-3 determinism proof at the tx layer: the
// same tx sequence applied twice (same height/ts/index) yields byte-identical
// fills and book rows.
func TestApplyTxDeterminism(t *testing.T) {
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	txs := []*Tx{
		placeTx(t, zapwire.SideSell, 101.0, 5.0, "maker-a"),
		placeTx(t, zapwire.SideSell, 101.5, 3.0, "maker-b"),
		placeTx(t, zapwire.SideBuy, 99.0, 2.0, "maker-c"),
		submitTx(t, zapwire.SideBuy, false, 101.5, 6.0, "taker-x"),
		submitTx(t, zapwire.SideSell, true, 0, 1.0, "taker-y"),
		placeTx(t, zapwire.SideSell, 101.25, 4.0, "maker-d"),
	}

	f1, r1 := applyAll(t, 42, ts, txs)
	f2, r2 := applyAll(t, 42, ts, txs)

	if len(f1) == 0 {
		t.Fatal("expected fills from the submits, got none")
	}
	if len(f1) != len(f2) {
		t.Fatalf("fill count differs: %d vs %d", len(f1), len(f2))
	}
	for i := range f1 {
		if string(dex.EncodeTrade(f1[i])) != string(dex.EncodeTrade(f2[i])) {
			t.Errorf("fill[%d] not byte-identical across replays", i)
		}
	}
	if len(r1) != len(r2) {
		t.Fatalf("book row count differs: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if string(dex.EncodeRow(r1[i])) != string(dex.EncodeRow(r2[i])) {
			t.Errorf("book row[%d] not byte-identical across replays", i)
		}
	}
}

// TestBlockDeterministicID proves the id derivation is unique per (height,
// txIndex) and never zero.
func TestBlockDeterministicID(t *testing.T) {
	seen := map[uint64]struct{}{}
	for h := uint64(0); h < 4; h++ {
		for i := uint32(0); i < 4; i++ {
			id := blockDeterministicID(h, i)
			if id == 0 {
				t.Fatalf("id zero for height=%d idx=%d", h, i)
			}
			if _, dup := seen[id]; dup {
				t.Fatalf("id collision %d at height=%d idx=%d", id, h, i)
			}
			seen[id] = struct{}{}
		}
	}
}

// TestApplyPlaceRestsDeterministicOrder proves a place rests with the
// VM-supplied ID and timestamp (never minted), and the order is recoverable.
func TestApplyPlaceRestsDeterministicOrder(t *testing.T) {
	book := dex.NewOrderBook("BTC-USD")
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	tx := placeTx(t, zapwire.SideBuy, 100.0, 2.0, "alice")

	res, err := applyTx(book, tx, 7, ts, 3)
	if err != nil {
		t.Fatalf("applyTx: %v", err)
	}
	if res.Placed == nil {
		t.Fatal("expected a placed order")
	}
	wantID := blockDeterministicID(7, 3)
	if res.Placed.ID != wantID {
		t.Errorf("placed ID = %d, want deterministic %d", res.Placed.ID, wantID)
	}
	if !res.Placed.Timestamp.Equal(ts) {
		t.Errorf("placed timestamp = %v, want block ts %v", res.Placed.Timestamp, ts)
	}
	if o := book.GetOrder(wantID); o == nil {
		t.Error("placed order not resting in book")
	}
}

// TestApplyCancelRemovesResting proves cancel removes a resting order and an
// unknown cancel is a deterministic no-op.
func TestApplyCancelRemovesResting(t *testing.T) {
	book := dex.NewOrderBook("BTC-USD")
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	res, _ := applyTx(book, placeTx(t, zapwire.SideBuy, 100.0, 2.0, "alice"), 1, ts, 0)
	id := res.Placed.ID

	// applyTx is the pure book function (no auth gate); sign with the placer so the
	// frame is well-formed (NewTx refuses an unsigned money-moving tx).
	cancel := acctFor(t, "alice").signed(t, TxCancel, zapwire.EncodeCancel(testPool, id))
	cres, err := applyTx(book, cancel, 1, ts, 1)
	if err != nil {
		t.Fatalf("applyTx cancel: %v", err)
	}
	if cres.Canceled != id {
		t.Errorf("Canceled = %d, want %d", cres.Canceled, id)
	}

	// Unknown cancel: deterministic no-op.
	unknown := acctFor(t, "alice").signed(t, TxCancel, zapwire.EncodeCancel(testPool, 999999))
	ures, err := applyTx(book, unknown, 1, ts, 2)
	if err != nil {
		t.Fatalf("applyTx unknown cancel: %v", err)
	}
	if ures.Canceled != 0 {
		t.Errorf("unknown cancel reported Canceled=%d, want 0", ures.Canceled)
	}
}

// TestTxIDStable proves the TxID is a stable function of the canonical bytes.
func TestTxIDStable(t *testing.T) {
	tx := placeTx(t, zapwire.SideBuy, 100.0, 2.0, "alice")
	if tx.ID() != tx.ID() {
		t.Error("TxID not stable")
	}
	// A different body yields a different id.
	other := placeTx(t, zapwire.SideBuy, 100.0, 3.0, "alice")
	if tx.ID() == other.ID() {
		t.Error("distinct txs share a TxID")
	}
	// Round-trips through Bytes/ParseTx unchanged.
	back, err := ParseTx(tx.Bytes())
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}
	if back.ID() != tx.ID() {
		t.Error("TxID changed across Bytes/ParseTx round-trip")
	}
}
