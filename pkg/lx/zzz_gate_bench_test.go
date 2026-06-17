// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// TEMPORARY benchmark gate file — NOT for commit. Deleted by the
// benchmark-gate run. Mirrors the data shapes used by
// orderbook_cuda_parity_test.go and amm_gpu_test.go so the measured
// flat-buffer CLOB and AMM paths are the SAME code the parity tests pin.

package lx

import (
	"math/rand"
	"testing"
)

// makeBook builds a deterministic flat book of `n` resting asks plus a
// single crossing bid taker whose quantity is large enough to sweep the
// whole book (worst case: one trade per book level).
func makeBook(n int, sweepAll bool) (DEXOrder, []DEXOrder, []uint32) {
	rng := rand.New(rand.NewSource(42))
	book := make([]DEXOrder, n)
	indices := make([]uint32, n)
	var totalQty uint64
	for i := range book {
		q := uint64(10 + rng.Intn(100))
		totalQty += q
		book[i] = DEXOrder{
			OrderID:   uint64(10000 + i),
			UserID:    2 + uint64(i)%4,
			Price:     DEXPrice{Integer: uint64(90 + rng.Intn(10)), Fraction: 0},
			Quantity:  q,
			Remaining: q,
			Timestamp: uint64(100 + i),
			Side:      DEXSideAsk,
			Type:      DEXOrderLimit,
			Status:    DEXStatusOpen,
		}
		indices[i] = uint32(i)
	}
	takerQty := totalQty // sweep the entire book → n fills
	if !sweepAll {
		takerQty = uint64(50) // small taker → ~1 fill
	}
	taker := DEXOrder{
		OrderID:   1,
		UserID:    1,
		Price:     DEXPrice{Integer: 200, Fraction: 0}, // crosses every ask
		Quantity:  takerQty,
		Remaining: takerQty,
		Timestamp: 9999,
		Side:      DEXSideBid,
		Type:      DEXOrderLimit,
		Status:    DEXStatusOpen,
	}
	return taker, book, indices
}

// BenchmarkGateCancel measures the OrderBook cancel path (rich engine):
// pre-populate a book, then cancel + re-add the same order each iter so
// the steady-state book depth stays constant. Reports cancels/sec.
func BenchmarkGateCancel(b *testing.B) {
	for _, depth := range []int{1000, 10000} {
		depth := depth
		b.Run(sizeStr(uintptr(depth)), func(b *testing.B) {
			book := NewOrderBook("CANCEL")
			book.EnableImmediateMatching = false
			for i := 0; i < depth; i++ {
				book.AddOrder(&Order{
					ID: uint64(i), Type: Limit, Side: Buy,
					Price: 100 - float64(i%50)/100, Size: 10,
					User: "u", Timestamp: time.Now(),
				})
			}
			// Use a dedicated order id we cancel+re-add each iteration.
			const probe = uint64(1 << 40)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				book.AddOrder(&Order{
					ID: probe, Type: Limit, Side: Buy,
					Price: 99.5, Size: 10, User: "u", Timestamp: time.Now(),
				})
				if err := book.CancelOrder(probe); err != nil {
					b.Fatalf("cancel: %v", err)
				}
			}
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "cancel+add/sec")
		})
	}
}

// BenchmarkGateMatchOrderCPU_Sweep measures the flat-buffer CLOB match
// primitive (the pure-Go oracle the CUDA/Metal kernel mirrors) when the
// taker sweeps the whole book — n fills per call. Reports orders/sec as
// fills (book levels matched) per second, the meaningful matching-throughput
// unit for this primitive.
func BenchmarkGateMatchOrderCPU_Sweep(b *testing.B) {
	for _, n := range []int{16, 64, 256, 1024} {
		n := n
		b.Run(sizeStr(uintptr(n)), func(b *testing.B) {
			taker, book0, indices := makeBook(n, true)
			// Per-iteration we must reset the book (MatchOrderCPU mutates it).
			books := make([][]DEXOrder, b.N)
			for i := range books {
				books[i] = append([]DEXOrder(nil), book0...)
			}
			b.ResetTimer()
			b.ReportAllocs()
			var fills int
			for i := 0; i < b.N; i++ {
				t := taker
				trades, _ := MatchOrderCPU(&t, books[i], indices, 1_000_000, 9_999)
				fills += len(trades)
			}
			b.StopTimer()
			// orders/sec = book levels matched per second across all iters.
			b.ReportMetric(float64(fills)/b.Elapsed().Seconds(), "fills/sec")
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(fills), "ns/fill")
		})
	}
}

// BenchmarkGateAMMBatchEval measures BatchEvalConstantProduct — the AMM
// xy=k batch primitive. With CGO+darwin and the metallib present this
// dispatches to Metal; otherwise it is the pure-Go CPU oracle. The build
// tag / fallback decides which symbol runs; this bench measures whichever
// is wired (report alongside GPU_DISABLE=1 to force the CPU comparison).
func BenchmarkGateAMMBatchEval(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144, 1048576} {
		n := n
		b.Run(sizeStr(uintptr(n)), func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			reserves := make([]ReservePair, n)
			amounts := make([]uint64, n)
			for i := 0; i < n; i++ {
				reserves[i] = ReservePair{
					ReserveX: rng.Uint64()%(1<<48) + 1,
					ReserveY: rng.Uint64()%(1<<48) + 1,
				}
				amounts[i] = rng.Uint64() % (1 << 32)
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := BatchEvalConstantProduct(reserves, amounts); err != nil {
					b.Fatalf("BatchEvalConstantProduct: %v", err)
				}
			}
			b.StopTimer()
			// evals/sec = pool evaluations per second.
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "evals/sec")
		})
	}
}

// BenchmarkGateAMMBatchEvalCPU forces the pure-Go oracle regardless of
// build, so we always have a CPU baseline to compare the Metal numbers
// against on the same host.
func BenchmarkGateAMMBatchEvalCPU(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144, 1048576} {
		n := n
		b.Run(sizeStr(uintptr(n)), func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			reserves := make([]ReservePair, n)
			amounts := make([]uint64, n)
			for i := 0; i < n; i++ {
				reserves[i] = ReservePair{
					ReserveX: rng.Uint64()%(1<<48) + 1,
					ReserveY: rng.Uint64()%(1<<48) + 1,
				}
				amounts[i] = rng.Uint64() % (1 << 32)
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := BatchEvalConstantProductCPU(reserves, amounts); err != nil {
					b.Fatalf("cpu: %v", err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "evals/sec")
		})
	}
}
