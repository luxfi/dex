// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/rand"
	"sort"
	"testing"
)

// batch_match_test.go is the determinism proof for the batch match path: the
// parallel, tiled BatchMatcher output is BYTE-IDENTICAL to the naive one-at-a-time
// sequential path, for the same input ordering. It follows the match_verify.go
// pattern (an independent authority the fast path is byte-compared against) with
// the authority being the per-order sequential reference instead of the CPU
// oracle. No GPU, no cgo: MatchOrderCPU is the shared matching primitive.

// mkOrder builds a flat DEXOrder maker/taker at an integer price.
func mkOrder(id, userID uint64, side uint8, price, qty uint64) DEXOrder {
	o := DEXOrder{
		OrderID:  id,
		UserID:   userID,
		Price:    DEXPrice{Integer: price},
		Quantity: qty,
		Side:     side,
		Type:     DEXOrderLimit,
		Status:   DEXStatusOpen,
	}
	o.Remaining = qty
	o.ClearPadding()
	return o
}

// genBatch builds a randomized batch of nBooks independent books. Each book gets
// resting makers on BOTH sides with overlapping prices (so crosses actually
// happen) and a stream of takers on both sides. Takers are threaded into a single
// global tx order (Seq) that INTERLEAVES across books — each book's own takers
// stay in ascending Seq, but consecutive Seqs land in different books — so the
// test exercises reassembly of a genuinely interleaved stream, not per-book runs.
func genBatch(rng *rand.Rand, nBooks int) []BatchBook {
	books := make([]BatchBook, nBooks)
	orderID := uint64(1)
	for b := 0; b < nBooks; b++ {
		nMakers := 4 + rng.Intn(20)
		book := make([]DEXOrder, 0, nMakers)
		for i := 0; i < nMakers; i++ {
			userID := uint64(1 + rng.Intn(6))
			if rng.Intn(2) == 0 {
				// Bid around 90..99.
				book = append(book, mkOrder(orderID, userID, DEXSideBid, uint64(90+rng.Intn(10)), uint64(1+rng.Intn(20))))
			} else {
				// Ask around 100..110.
				book = append(book, mkOrder(orderID, userID, DEXSideAsk, uint64(100+rng.Intn(11)), uint64(1+rng.Intn(20))))
			}
			orderID++
		}
		books[b].Book = book
	}

	// Build the interleaved taker stream. Total takers scales with book count so
	// large batches carry real work.
	totalTakers := nBooks * (1 + rng.Intn(4))
	seq := 0
	for t := 0; t < totalTakers; t++ {
		b := rng.Intn(nBooks)
		var in BatchIncoming
		in.Seq = seq
		seq++
		userID := uint64(100 + rng.Intn(6))
		if rng.Intn(2) == 0 {
			// Aggressive/soft BUY: price high enough to cross some asks (or not).
			price := uint64(98 + rng.Intn(15)) // 98..112 straddles the ask band
			in.Order = mkOrder(1_000_000+uint64(seq), userID, DEXSideBid, price, uint64(1+rng.Intn(40)))
		} else {
			// SELL: price low enough to cross some bids (or not).
			price := uint64(88 + rng.Intn(14)) // 88..101 straddles the bid band
			in.Order = mkOrder(1_000_000+uint64(seq), userID, DEXSideAsk, price, uint64(1+rng.Intn(40)))
		}
		in.TradeBase = uint64(seq) * 1_000 // distinct trade-id space per taker
		in.Timestamp = uint64(1_700_000_000_000_000_000 + seq)
		books[b].Incoming = append(books[b].Incoming, in)
	}
	return books
}

// deepCopyBooks clones a batch so two match paths start from byte-identical state
// (MatchOrderCPU mutates the book slice in place).
func deepCopyBooks(src []BatchBook) []BatchBook {
	out := make([]BatchBook, len(src))
	for i, b := range src {
		nb := BatchBook{
			Book:     append([]DEXOrder(nil), b.Book...),
			Incoming: append([]BatchIncoming(nil), b.Incoming...),
		}
		out[i] = nb
	}
	return out
}

// refSequentialMatch is the INDEPENDENT authority: it crosses each book's takers
// one at a time, in order, with an inline (separately written) price-time index
// ordering, and returns results in ascending Seq. It uses no batch_match.go
// internals other than the shared MatchOrderCPU primitive, so a bug in the batch
// matcher's orchestration OR its index ordering diverges from this reference.
func refSequentialMatch(books []BatchBook) []BatchResult {
	var out []BatchResult
	for b := range books {
		bk := books[b].Book
		for _, in := range books[b].Incoming {
			// Independently derive opposite-side, eligible makers in price-time order.
			want := DEXSideAsk
			if in.Order.Side == DEXSideAsk {
				want = DEXSideBid
			}
			idx := make([]uint32, 0, len(bk))
			for i := range bk {
				if bk[i].Side != want {
					continue
				}
				if bk[i].Status != DEXStatusOpen && bk[i].Status != DEXStatusPartial {
					continue
				}
				if bk[i].Remaining == 0 {
					continue
				}
				idx = append(idx, uint32(i))
			}
			buyTaker := in.Order.Side == DEXSideBid
			sort.SliceStable(idx, func(x, y int) bool {
				px, py := bk[idx[x]].Price, bk[idx[y]].Price
				if px.Integer != py.Integer {
					if buyTaker {
						return px.Integer < py.Integer
					}
					return px.Integer > py.Integer
				}
				if px.Fraction != py.Fraction {
					if buyTaker {
						return px.Fraction < py.Fraction
					}
					return px.Fraction > py.Fraction
				}
				return bk[idx[x]].OrderID < bk[idx[y]].OrderID
			})
			inc := in.Order
			trades, rem := MatchOrderCPU(&inc, bk, idx, in.TradeBase, in.Timestamp)
			out = append(out, BatchResult{Seq: in.Seq, Trades: trades, Remaining: rem})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out
}

// assertResultsByteEqual fails if two result sets are not byte-identical: same
// count, same Seq order, same Remaining, and byte-identical trade rows.
func assertResultsByteEqual(t *testing.T, label string, got, want []BatchResult) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: result count %d != reference %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i].Seq != want[i].Seq {
			t.Fatalf("%s: result[%d] Seq %d != reference %d (reassembly reordered)", label, i, got[i].Seq, want[i].Seq)
		}
		if got[i].Remaining != want[i].Remaining {
			t.Fatalf("%s: result[%d] (Seq %d) remaining %d != reference %d", label, i, want[i].Seq, got[i].Remaining, want[i].Remaining)
		}
		if len(got[i].Trades) != len(want[i].Trades) {
			t.Fatalf("%s: result[%d] (Seq %d) trade count %d != reference %d", label, i, want[i].Seq, len(got[i].Trades), len(want[i].Trades))
		}
		for j := range want[i].Trades {
			if string(EncodeTrade(got[i].Trades[j])) != string(EncodeTrade(want[i].Trades[j])) {
				t.Fatalf("%s: result[%d] (Seq %d) trade[%d] NOT byte-identical:\n  got=%+v\n  ref=%+v",
					label, i, want[i].Seq, j, got[i].Trades[j], want[i].Trades[j])
			}
		}
	}
}

// assertBooksByteEqual fails if two post-cross batches' book state differs by a
// single byte (EncodeRow over every resting order in every book).
func assertBooksByteEqual(t *testing.T, label string, got, want []BatchBook) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: book count %d != reference %d", label, len(got), len(want))
	}
	for b := range want {
		if len(got[b].Book) != len(want[b].Book) {
			t.Fatalf("%s: book[%d] size %d != reference %d", label, b, len(got[b].Book), len(want[b].Book))
		}
		for i := range want[b].Book {
			if string(EncodeRow(got[b].Book[i])) != string(EncodeRow(want[b].Book[i])) {
				t.Fatalf("%s: book[%d] order[%d] NOT byte-identical after cross:\n  got=%+v\n  ref=%+v",
					label, b, i, got[b].Book[i], want[b].Book[i])
			}
		}
	}
}

// countFills totals the trades across a result set — used to assert the corpus
// actually crosses (a test over a batch that never fills proves nothing).
func countFills(res []BatchResult) int {
	n := 0
	for _, r := range res {
		n += len(r.Trades)
	}
	return n
}

// TestBatchMatchEqualsSequential is the core determinism proof: for a battery of
// randomized batches, the parallel + tiled BatchMatcher produces byte-identical
// fills, remaining, and post-cross book state as the independent one-at-a-time
// sequential reference. Small worker counts and a small tile size force real
// cross-book parallelism AND multiple tiles per batch, so the concurrency and the
// tile boundaries are both exercised.
func TestBatchMatchEqualsSequential(t *testing.T) {
	// Force tiny tiles (7 books/tile) and 4 workers so the 64+ book batches below
	// span many tiles and run genuinely concurrently — the exact conditions a
	// reordering/race bug would surface under.
	matcher := NewCPUBatchMatcher(4, 7)

	sizes := []int{1, 2, 8, 33, 64, 200}
	for _, nBooks := range sizes {
		nBooks := nBooks
		for seed := int64(1); seed <= 6; seed++ {
			seed := seed
			t.Run("", func(t *testing.T) {
				rng := rand.New(rand.NewSource(seed*1000 + int64(nBooks)))
				orig := genBatch(rng, nBooks)

				batchBooks := deepCopyBooks(orig)
				refBooks := deepCopyBooks(orig)

				got := matcher.MatchBatch(batchBooks)
				want := refSequentialMatch(refBooks)

				assertResultsByteEqual(t, "batch-vs-sequential", got, want)
				assertBooksByteEqual(t, "batch-vs-sequential", batchBooks, refBooks)

				// The corpus must actually cross, or the equality proves nothing.
				if nBooks >= 8 && countFills(want) == 0 {
					t.Fatalf("nBooks=%d seed=%d produced zero fills; corpus is degenerate", nBooks, seed)
				}
			})
		}
	}
}

// TestBatchMatchRunToRunDeterministic proves the batch path is stable across runs:
// two MatchBatch calls on independent copies of the same input yield byte-
// identical results and book state. This guards against scheduling- or
// map-iteration-induced nondeterminism sneaking into the concurrent path.
func TestBatchMatchRunToRunDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(20260701))
	orig := genBatch(rng, 128)

	a := deepCopyBooks(orig)
	b := deepCopyBooks(orig)

	resA := BatchMatchCPU.MatchBatch(a)
	resB := BatchMatchCPU.MatchBatch(b)

	assertResultsByteEqual(t, "run-to-run", resA, resB)
	assertBooksByteEqual(t, "run-to-run", a, b)
	if countFills(resA) == 0 {
		t.Fatal("run-to-run corpus produced zero fills; degenerate")
	}
}

// TestBatchTileBoundaryEqualsSingleTile proves tiling is transparent: the same
// batch matched with a tile size of 1 (every book its own tile), a small tile,
// and one giant tile all produce byte-identical output. Tile size is pure
// throughput, never a committed byte.
func TestBatchTileBoundaryEqualsSingleTile(t *testing.T) {
	rng := rand.New(rand.NewSource(424242))
	orig := genBatch(rng, 96)

	ref := deepCopyBooks(orig)
	want := refSequentialMatch(ref)

	for _, tile := range []int{1, 3, 32, 4096} {
		books := deepCopyBooks(orig)
		got := NewCPUBatchMatcher(4, tile).MatchBatch(books)
		assertResultsByteEqual(t, "tile-invariance", got, want)
		assertBooksByteEqual(t, "tile-invariance", books, ref)
	}
}

// --- benchmarks: batch (parallel) vs per-order (single-threaded sequential) ----
//
// Both start each op from a fresh deep copy (MatchOrderCPU mutates in place), so
// the copy cost is charged to both and the delta is honest. The CPU batch win is
// cross-book parallelism; the GPU backend that slots behind BatchMatcher adds the
// dispatch-tax elimination (one kernel launch per tile vs one per order).

// genDeepBatch builds a match-bound benchmark corpus: nBooks books, each with
// `depth` resting makers spread over a price ladder, and `takers` sweeping takers
// per book that cross a large fraction of the ladder. This makes the actual
// price-time matching (not allocation) the dominant cost, so the parallel speedup
// reflects real matching throughput.
func genDeepBatch(nBooks, depth, takers int) []BatchBook {
	rng := rand.New(rand.NewSource(7))
	books := make([]BatchBook, nBooks)
	id := uint64(1)
	for b := 0; b < nBooks; b++ {
		book := make([]DEXOrder, 0, depth)
		for i := 0; i < depth; i++ {
			// Half bids (below 1000), half asks (at/above 1000), a deep two-sided ladder.
			if i%2 == 0 {
				book = append(book, mkOrder(id, uint64(1+rng.Intn(8)), DEXSideAsk, uint64(1000+i), 5))
			} else {
				book = append(book, mkOrder(id, uint64(1+rng.Intn(8)), DEXSideBid, uint64(1000-i), 5))
			}
			id++
		}
		in := make([]BatchIncoming, 0, takers)
		for t := 0; t < takers; t++ {
			seq := b*takers + t
			var o DEXOrder
			if t%2 == 0 {
				// Aggressive BUY sweeps up the ask ladder.
				o = mkOrder(1_000_000+uint64(seq), uint64(100+t), DEXSideBid, uint64(1000+depth), uint64(depth)) // enough qty to sweep many levels
			} else {
				// Aggressive SELL sweeps down the bid ladder.
				o = mkOrder(1_000_000+uint64(seq), uint64(100+t), DEXSideAsk, uint64(1000-depth), uint64(depth))
			}
			in = append(in, BatchIncoming{Seq: seq, Order: o, TradeBase: uint64(seq) * 10_000, Timestamp: uint64(1_700_000_000_000_000_000 + seq)})
		}
		books[b] = BatchBook{Book: book, Incoming: in}
	}
	return books
}

// benchmarkBatchInput: 2048 books × 128-deep ladder × 8 sweeping takers each — a
// heavy, match-bound block. 16,384 total crosses.
func benchmarkBatchInput() []BatchBook { return genDeepBatch(2048, 128, 8) }

func BenchmarkBatchMatchCPU_PerOrderSequential(b *testing.B) {
	orig := benchmarkBatchInput()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		books := deepCopyBooks(orig)
		b.StartTimer()
		_ = refSequentialMatch(books)
	}
}

func BenchmarkBatchMatchCPU_Batch(b *testing.B) {
	orig := benchmarkBatchInput()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		books := deepCopyBooks(orig)
		b.StartTimer()
		_ = BatchMatchCPU.MatchBatch(books)
	}
	b.Logf("matcher=%s", BatchMatchCPU.Name())
}
