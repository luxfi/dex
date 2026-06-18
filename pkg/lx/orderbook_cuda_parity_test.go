// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/rand"
	"testing"
	"time"
)

// TestMatchOrder_GPUMatchesCPU pins the field-equality contract between
// MatchOrderGPU and MatchOrderCPU. On hosts without CUDA the GPU path
// transparently falls back to CPU, so this test still passes (it proves
// CPU == CPU and exercises the dispatcher path). On Linux+NVIDIA hosts
// it proves the CUDA dispatch produces the same trade list and remaining
// quantity field-for-field.
//
// We compare via DEXTrade.EqualFields / DEXOrder.EqualFields, NOT `==`,
// so padding bytes are not allowed to influence the result. Padding is
// also explicitly cleared on both paths (host driver post-copy + Go
// MatchOrderCPU) so the production objects stay deterministic too.
func TestMatchOrder_GPUMatchesCPU(t *testing.T) {
	for seed := int64(1); seed <= 8; seed++ {
		seed := seed
		t.Run("", func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))

			incoming := DEXOrder{
				OrderID:   uint64(seed*1000 + 1),
				UserID:    1,
				Price:     DEXPrice{Integer: uint64(100 + rng.Intn(20)), Fraction: 0},
				Quantity:  uint64(50 + rng.Intn(450)),
				Timestamp: uint64(seed),
				Side:      DEXSideBid,
				Type:      DEXOrderLimit,
				Status:    DEXStatusOpen,
			}
			incoming.Remaining = incoming.Quantity

			bookSize := 16 + rng.Intn(64)
			book := make([]DEXOrder, bookSize)
			indices := make([]uint32, bookSize)
			for i := range book {
				book[i] = DEXOrder{
					OrderID:   uint64(seed*10000) + uint64(i),
					UserID:    2 + uint64(i)%4,
					Price:     DEXPrice{Integer: uint64(90 + rng.Intn(30)), Fraction: 0},
					Quantity:  uint64(10 + rng.Intn(100)),
					Timestamp: uint64(seed*100) + uint64(i),
					Side:      DEXSideAsk,
					Type:      DEXOrderLimit,
					Status:    DEXStatusOpen,
				}
				book[i].Remaining = book[i].Quantity
				indices[i] = uint32(i)
			}

			// Deep-copy book so CPU and GPU runs start from identical state.
			bookCPU := append([]DEXOrder(nil), book...)
			bookGPU := append([]DEXOrder(nil), book...)

			incomingCPU := incoming
			tradesCPU, remCPU := MatchOrderCPU(&incomingCPU, bookCPU, indices, 1_000_000, 9_999)

			incomingGPU := incoming
			tradesGPU, remGPU, err := MatchOrderGPU(&incomingGPU, bookGPU, indices, 1_000_000, 9_999)
			if err != nil {
				t.Fatalf("MatchOrderGPU error: %v", err)
			}

			if remCPU != remGPU {
				t.Fatalf("remaining mismatch: cpu=%d gpu=%d", remCPU, remGPU)
			}
			if len(tradesCPU) != len(tradesGPU) {
				t.Fatalf("trade count mismatch: cpu=%d gpu=%d", len(tradesCPU), len(tradesGPU))
			}
			for i := range tradesCPU {
				if !tradesCPU[i].EqualFields(tradesGPU[i]) {
					t.Fatalf("trade[%d] field mismatch:\n  cpu=%+v\n  gpu=%+v",
						i, tradesCPU[i], tradesGPU[i])
				}
			}
			for i := range bookCPU {
				if !bookCPU[i].EqualFields(bookGPU[i]) {
					t.Fatalf("book[%d] field mismatch:\n  cpu=%+v\n  gpu=%+v",
						i, bookCPU[i], bookGPU[i])
				}
			}
		})
	}
}

// clobTx is one operation in a deterministic CLOB transaction sequence: either
// rest a limit order (place) or cross the book with a marketable order (submit).
// IDs and timestamps are supplied explicitly — exactly as the d-chain VM supplies
// them from block context — so the replay is a pure function of the sequence.
type clobTx struct {
	submit    bool // false = place (rest), true = submit (cross)
	id        uint64
	side      Side
	price     float64
	size      float64
	user      string
	timestamp time.Time
}

// replaySequence runs a CLOB tx sequence through a fresh book via the
// deterministic consensus entrypoints (ConsensusAddOrder for places,
// SubmitMarketable for submits, with VM-supplied ID/timestamp). It returns the
// concatenated fills (as canonical DEXTrade rows, taker side bound in) and the
// final book as canonical DEXOrder rows — the two byte-streams a validator
// commits. It NEVER calls AddOrder (would mint) or MatchOrders (would stamp
// time.Now()).
func replaySequence(symbol string, txs []clobTx) (fills []DEXTrade, rows []DEXOrder) {
	ob := NewOrderBook(symbol)
	for _, tx := range txs {
		if tx.submit {
			o := &Order{
				ID: tx.id, Type: Limit, Side: tx.side, Price: tx.price, Size: tx.size,
				User: tx.user, UserID: tx.user, Symbol: symbol, Timestamp: tx.timestamp,
			}
			trades, err := ob.SubmitMarketable(o)
			if err != nil {
				continue
			}
			for _, tr := range trades {
				fills = append(fills, TradeToRow(tr, tx.side))
			}
		} else {
			o := &Order{
				ID: tx.id, Type: Limit, Side: tx.side, Price: tx.price, Size: tx.size,
				User: tx.user, UserID: tx.user, Symbol: symbol, Timestamp: tx.timestamp,
			}
			ob.ConsensusAddOrder(o)
		}
	}
	return fills, BookToRows(ob)
}

// TestConsensusReplayDeterminism is the CRITICAL step-2 determinism proof: two
// replays of the IDENTICAL CLOB tx sequence produce BYTE-IDENTICAL fills and
// BYTE-IDENTICAL book rows. This is the property a multi-validator d-chain
// depends on — every honest validator that replays the same ordered block
// derives the same fills and the same committed state. The consensus path
// (ConsensusAddOrder + SubmitMarketable) supplies ID/timestamp, so nothing
// non-deterministic (no LastOrderID mint, no time.Now()) leaks in.
func TestConsensusReplayDeterminism(t *testing.T) {
	base := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	ts := func(i int) time.Time { return base.Add(time.Duration(i) * time.Millisecond) }

	// A representative sequence: rest both sides, then cross with marketable
	// orders that partially fill across multiple price levels, then rest more.
	txs := []clobTx{
		{false, 1, Sell, 101.0, 5.0, "maker-a", ts(0)},
		{false, 2, Sell, 101.5, 3.0, "maker-b", ts(1)},
		{false, 3, Sell, 102.0, 4.0, "maker-c", ts(2)},
		{false, 4, Buy, 99.0, 2.0, "maker-d", ts(3)},
		{false, 5, Buy, 98.5, 6.0, "maker-e", ts(4)},
		{true, 6, Buy, 102.0, 7.0, "taker-x", ts(5)}, // crosses 101.0 + 101.5 levels
		{true, 7, Sell, 98.5, 3.0, "taker-y", ts(6)}, // crosses 99.0 + 98.5 levels
		{false, 8, Sell, 101.25, 2.0, "maker-f", ts(7)},
		{true, 9, Buy, 101.25, 1.0, "taker-z", ts(8)}, // partial against maker-f
	}

	fills1, rows1 := replaySequence("BTC-USD", txs)
	fills2, rows2 := replaySequence("BTC-USD", txs)

	// The sequence must actually produce fills, or the test proves nothing.
	if len(fills1) == 0 {
		t.Fatal("expected the marketable submits to produce fills; sequence produced none")
	}

	if len(fills1) != len(fills2) {
		t.Fatalf("fill count differs across replays: %d vs %d", len(fills1), len(fills2))
	}
	for i := range fills1 {
		if string(EncodeTrade(fills1[i])) != string(EncodeTrade(fills2[i])) {
			t.Errorf("fill[%d] NOT byte-identical across replays:\n  a=%+v\n  b=%+v",
				i, fills1[i], fills2[i])
		}
	}

	if len(rows1) != len(rows2) {
		t.Fatalf("book row count differs across replays: %d vs %d", len(rows1), len(rows2))
	}
	for i := range rows1 {
		if string(EncodeRow(rows1[i])) != string(EncodeRow(rows2[i])) {
			t.Errorf("book row[%d] NOT byte-identical across replays:\n  a=%+v\n  b=%+v",
				i, rows1[i], rows2[i])
		}
	}
}

// TestConsensusAddOrderRejectsNonDeterministicInput proves the guard refuses to
// mint: an order with a zero ID or zero timestamp is rejected (returns 0) rather
// than silently acquiring a process-local counter value or wall-clock stamp.
func TestConsensusAddOrderRejectsNonDeterministicInput(t *testing.T) {
	ob := NewOrderBook("BTC-USD")
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()

	if got := ob.ConsensusAddOrder(&Order{ID: 0, Type: Limit, Side: Buy, Price: 100, Size: 1, User: "a", Timestamp: ts}); got != 0 {
		t.Errorf("ConsensusAddOrder accepted a zero-ID order: returned %d", got)
	}
	if got := ob.ConsensusAddOrder(&Order{ID: 1, Type: Limit, Side: Buy, Price: 100, Size: 1, User: "a"}); got != 0 {
		t.Errorf("ConsensusAddOrder accepted a zero-timestamp order: returned %d", got)
	}
	// LastOrderID must not have moved — no minting happened.
	if ob.LastOrderID != 0 {
		t.Errorf("ConsensusAddOrder minted LastOrderID despite rejection: %d", ob.LastOrderID)
	}
	// A well-formed order is accepted with its supplied ID.
	if got := ob.ConsensusAddOrder(&Order{ID: 7, Type: Limit, Side: Buy, Price: 100, Size: 1, User: "a", Timestamp: ts}); got != 7 {
		t.Errorf("ConsensusAddOrder rejected a valid order: returned %d", got)
	}
}
