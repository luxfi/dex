// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import "testing"

// orderbook_truncation_test.go is the regression guard for the day-1 latent
// matcher panic: SubmitMarketable and MatchOrders used to derive their returned
// fills by re-slicing the shared trade log (ob.Trades[startOffset:]) using an
// offset captured BEFORE matching. But ob.Trades self-truncates at 100k entries
// (keeps the last 50k). On a sustained-hot market, a match that pushes the log
// across that boundary truncates it mid-match, leaving startOffset (taken against
// the pre-truncation length, e.g. 60001) pointing past the end of the now-50000
// slice — ob.Trades[60001:] panics "slice bounds out of range". The trade log
// feeds settlement, so the fix had to keep returning EXACTLY this match's fills.
//
// The fix captures fills as they are produced, independent of ob.Trades, so the
// returned slice is correct across the truncation boundary by construction. These
// tests fail (panic) on the old code and pass on the fixed code.
//
// No time.Sleep, no stubs: every fill below is a real cross against real resting
// liquidity through the production match path.

// tradeLogCap mirrors the trade-log retention policy in orderbook.go: the log is
// trimmed to the last tradeLogKeep entries once it exceeds tradeLogCap.
const (
	tradeLogCap  = 100000
	tradeLogKeep = 50000
)

// TestSubmitMarketableCrossesTruncationBoundary drives the consensus taker path
// (SubmitMarketable) past the trade-log truncation boundary with single-fill
// submits, then verifies the one submit that actually triggers truncation still
// returns exactly its fill. Pre-fix this submit panicked.
func TestSubmitMarketableCrossesTruncationBoundary(t *testing.T) {
	ob := NewOrderBook("HOT-USD")

	// One deep resting maker on the ask: every 1-lot taker buy fills against it,
	// so each SubmitMarketable produces exactly one trade. Size comfortably covers
	// the > tradeLogCap submits below.
	const totalTakers = tradeLogCap + 5000 // 105000 > 100000: forces a truncation
	ob.AddOrder(&Order{
		ID: 1, Type: Limit, Side: Sell, Price: 100, Size: float64(totalTakers + 10),
		User: "mm",
	})

	for i := 0; i < totalTakers; i++ {
		lenBefore := len(ob.Trades)
		fills, err := ob.SubmitMarketable(&Order{
			Type: Limit, Side: Buy, Price: 100, Size: 1, User: "taker",
		})
		if err != nil {
			t.Fatalf("submit %d: unexpected error: %v", i, err)
		}
		// Every single-lot taker takes exactly one lot from the one maker => 1 fill.
		if len(fills) != 1 {
			t.Fatalf("submit %d: got %d fills, want 1", i, len(fills))
		}
		if fills[0].Size != 1 {
			t.Fatalf("submit %d: fill size = %v, want 1", i, fills[0].Size)
		}
		if fills[0].Price != 100 {
			t.Fatalf("submit %d: fill price = %v, want 100", i, fills[0].Price)
		}
		// If this submit is the one that pushed the log over the cap, the log must
		// have been trimmed to exactly tradeLogKeep — i.e. we genuinely exercised
		// the boundary that used to make the pre-match offset stale. (The trim is
		// ob.Trades[len-tradeLogKeep:] AFTER appending the new fill, so the result
		// keeps the last tradeLogKeep entries, the new fill among them.)
		if lenBefore == tradeLogCap {
			if len(ob.Trades) != tradeLogKeep {
				t.Fatalf("submit %d: expected truncation to %d, got log len %d",
					i, tradeLogKeep, len(ob.Trades))
			}
		}
	}

	// We crossed the boundary, so the log must have been trimmed at least once
	// (final length <= cap despite totalTakers > cap fills having been produced).
	if len(ob.Trades) > tradeLogCap {
		t.Fatalf("trade log not truncated: len=%d > cap=%d", len(ob.Trades), tradeLogCap)
	}
}

// TestSubmitMarketableSweepAcrossTruncation is the strongest case: a SINGLE
// SubmitMarketable call sweeps many resting makers and crosses the truncation
// boundary WITHIN that one match. The fix must return every fill of this sweep,
// in order, with exact sizes — not a slice corrupted by the mid-match trim.
func TestSubmitMarketableSweepAcrossTruncation(t *testing.T) {
	ob := NewOrderBook("SWEEP-USD")

	// Phase 1: warm the log to a length in (tradeLogKeep, tradeLogCap) so that an
	// offset captured at the start of phase 2 would be > tradeLogKeep — the exact
	// condition that made the old re-slice go out of range after the trim. Warm with
	// one FRESH 1-lot maker per 1-lot taker: each cross is exact (no float drift from
	// repeatedly subtracting from a single deep maker's float Filled), so the sweep
	// in phase 2 stays a clean integer count and this test isolates the truncation
	// boundary — not the separate, pre-existing float-quantity imprecision the
	// integer settlement lane (settlement_units.go) exists to sidestep.
	const warm = tradeLogKeep + 5000 // 55000: > keep, < cap
	wid := uint64(1)
	for i := 0; i < warm; i++ {
		ob.AddOrder(&Order{
			ID: wid, Type: Limit, Side: Sell, Price: 100, Size: 1, User: "mm-warm",
		})
		wid++
		if _, err := ob.SubmitMarketable(&Order{
			Type: Limit, Side: Buy, Price: 100, Size: 1, User: "taker",
		}); err != nil {
			t.Fatalf("warm submit %d: %v", i, err)
		}
	}
	if got := len(ob.Trades); got != warm {
		t.Fatalf("after warmup: log len = %d, want %d (no trim should have happened yet)", got, warm)
	}

	// Phase 2: rest many 1-lot makers at one price, then ONE taker that sweeps all
	// of them in a single match. This single call appends sweepLots trades; since
	// warm + sweepLots > tradeLogCap, the trim fires mid-match and (pre-fix) makes
	// the start offset (== warm, which is > tradeLogKeep) stale -> panic.
	const sweepLots = (tradeLogCap - warm) + 5000 // pushes warm+sweep over the cap
	const sweepIDBase = 1_000_000                 // disjoint from the warm maker IDs above
	for j := 0; j < sweepLots; j++ {
		ob.AddOrder(&Order{
			ID: uint64(sweepIDBase + j), Type: Limit, Side: Sell, Price: 200, Size: 1, User: "mm-sweep",
		})
	}

	taker := &Order{
		Type: Limit, Side: Buy, Price: 200, Size: float64(sweepLots), User: "sweeper",
	}
	fills, err := ob.SubmitMarketable(taker)
	if err != nil {
		t.Fatalf("sweep submit: %v", err)
	}

	// Exact-fill assertions: one fill per swept maker, every fill 1 lot at 200,
	// total size == sweepLots == taker.Filled. This is exactly what settlement
	// consumes; the fix must return it intact across the mid-match trim.
	if len(fills) != sweepLots {
		t.Fatalf("sweep returned %d fills, want %d", len(fills), sweepLots)
	}
	var total float64
	for k, f := range fills {
		if f.Size != 1 {
			t.Fatalf("sweep fill %d: size = %v, want 1", k, f.Size)
		}
		if f.Price != 200 {
			t.Fatalf("sweep fill %d: price = %v, want 200", k, f.Price)
		}
		total += f.Size
	}
	if total != float64(sweepLots) {
		t.Fatalf("sweep total size = %v, want %d", total, sweepLots)
	}
	// Value-conservation: the fills returned to the settlement layer must account
	// for exactly what the taker filled — no fill dropped or duplicated by the trim.
	if total != taker.Filled {
		t.Fatalf("sweep fills total %v != taker.Filled %v (dropped/duplicated fills)", total, taker.Filled)
	}

	// The mid-match trim must have fired: warm + sweepLots > cap, yet the log is
	// now <= cap.
	if len(ob.Trades) > tradeLogCap {
		t.Fatalf("trade log not truncated during sweep: len=%d > cap=%d", len(ob.Trades), tradeLogCap)
	}
}

// TestMatchOrdersCrossesTruncationBoundary drives the legacy MatchOrders path
// past the truncation boundary inside a single call: many crossing maker/taker
// pairs generate > tradeLogCap trades, the log trims mid-loop, and MatchOrders
// must still return exactly the fills it generated. Pre-fix the final
// ob.Trades[startingTradeCount:] re-slice panicked.
func TestMatchOrdersCrossesTruncationBoundary(t *testing.T) {
	ob := NewOrderBook("MATCH-USD")

	// Rest pairs that all cross at the same price (buy >= sell). MatchOrders pairs
	// best bid with best ask repeatedly; with equal sizes each pairing yields one
	// full trade. pairs > tradeLogCap forces a trim inside the single MatchOrders
	// loop.
	const pairs = tradeLogCap + 3000 // 103000 trades in one MatchOrders call
	id := uint64(1)
	for i := 0; i < pairs; i++ {
		// Distinct users so self-trade prevention never cancels a pairing, and
		// distinct prices so price-time priority pairs them 1:1 deterministically.
		ob.Bids.addOrder(&Order{
			ID: id, Type: Limit, Side: Buy, Price: 100, Size: 1, User: "buyer",
		})
		id++
		ob.Asks.addOrder(&Order{
			ID: id, Type: Limit, Side: Sell, Price: 100, Size: 1, User: "seller",
		})
		id++
	}

	trades := ob.MatchOrders()

	if len(trades) != pairs {
		t.Fatalf("MatchOrders returned %d trades, want %d", len(trades), pairs)
	}
	for k, tr := range trades {
		if tr.Size != 1 {
			t.Fatalf("trade %d: size = %v, want 1", k, tr.Size)
		}
		if tr.Price != 100 {
			t.Fatalf("trade %d: price = %v, want 100", k, tr.Price)
		}
	}
	if len(ob.Trades) > tradeLogCap {
		t.Fatalf("trade log not truncated in MatchOrders: len=%d > cap=%d", len(ob.Trades), tradeLogCap)
	}
}
