// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"testing"
	"time"
)

// match_bench_test.go measures the honest cost of the GPU matcher vs the CPU
// oracle on a realistic single marketable cross, and the cost of the verified
// gate (CPU + GPU + byte-compare) vs the CPU authority alone.
//
// Reading these numbers: the flat MatchOrderGPU ABI matches ONE taker against
// ONE book, and the matcher is inherently serial (price-time priority over a
// shared remaining), so the deterministic kernel runs a SINGLE GPU lane. For a
// single cross that is dispatch-bound and slower than the CPU — the GPU win is
// in BATCHING many independent books per dispatch (a future ABI), not in one
// serial cross. The kernel is wired for correctness + determinism today; the
// throughput story is honest about where the speedup lives.

func benchBook(depth int) ([]DEXOrder, []uint32, DEXOrder) {
	book := make([]DEXOrder, depth)
	for i := 0; i < depth; i++ {
		book[i] = ask(uint64(1000+i), 5, uint64(100+i))
	}
	idx := idxOf(depth)
	// A taker that crosses about half the book.
	taker := bid(1, uint64(5*(depth/2)), uint64(100+depth))
	return book, idx, taker
}

func benchMatchCPU(b *testing.B, depth int) {
	book, idx, taker := benchBook(depth)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		bk := append([]DEXOrder(nil), book...)
		tk := taker
		b.StartTimer()
		MatchOrderCPU(&tk, bk, idx, 1_000_000, 7_777)
	}
}

func benchMatchGPU(b *testing.B, depth int) {
	book, idx, taker := benchBook(depth)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		bk := append([]DEXOrder(nil), book...)
		tk := taker
		b.StartTimer()
		if _, _, err := MatchOrderGPU(&tk, bk, idx, 1_000_000, 7_777); err != nil {
			b.Fatalf("MatchOrderGPU: %v", err)
		}
	}
	b.Logf("backend=%s", clobBackendLabel())
}

func BenchmarkMatchCPU_d16(b *testing.B)  { benchMatchCPU(b, 16) }
func BenchmarkMatchGPU_d16(b *testing.B)  { benchMatchGPU(b, 16) }
func BenchmarkMatchCPU_d64(b *testing.B)  { benchMatchCPU(b, 64) }
func BenchmarkMatchGPU_d64(b *testing.B)  { benchMatchGPU(b, 64) }
func BenchmarkMatchCPU_d256(b *testing.B) { benchMatchCPU(b, 256) }
func BenchmarkMatchGPU_d256(b *testing.B) { benchMatchGPU(b, 256) }

// BenchmarkSubmitMarketable_CPU vs _Verified measures the consensus-path cost of
// running the GPU shadow gate (capture + CPU authority + GPU + byte-compare)
// against the CPU authority alone. The delta is what a GPU-verified validator
// pays per cross for continuous at-Verify proof that the kernel still agrees.
func benchSubmit(b *testing.B, engine VerifyEngine) {
	b.ReportAllocs()
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ob := NewOrderBook("BTC-USD")
		for j := 0; j < 32; j++ {
			o := &Order{ID: uint64(j + 1), Type: Limit, Side: Sell, Price: float64(100 + j),
				Size: 5, User: "m", UserID: "m", Symbol: "BTC-USD", Timestamp: ts.Add(time.Duration(j))}
			ob.ConsensusAddOrder(o)
		}
		taker := &Order{ID: 999, Type: Limit, Side: Buy, Price: 200, Size: 80,
			User: "tk", UserID: "tk", Symbol: "BTC-USD", Timestamp: ts}
		b.StartTimer()
		if _, err := ob.SubmitMarketableVerified(taker, engine); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSubmit_CPU(b *testing.B)      { benchSubmit(b, EngineCPU) }
func BenchmarkSubmit_Verified(b *testing.B) { benchSubmit(b, EngineGPUVerified) }
