package benchmark

import (
	"fmt"
	"testing"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/geth/common"
)

// mustSignOrder builds a SignedOrder with a fresh secp256k1 keypair, signs
// the SigningHash digest, and stamps the derived address as Sender.
// Used by BenchmarkBatchVerifyOrders.
func mustSignOrder(b testing.TB, id uint64) lx.SignedOrder {
	b.Helper()
	key, err := crypto.GenerateKey()
	if err != nil {
		b.Fatalf("generate key: %v", err)
	}
	pub := crypto.FromECDSAPub(&key.PublicKey)
	var addr common.Address
	copy(addr[:], crypto.Keccak256(pub[1:])[12:])

	so := lx.SignedOrder{
		Order: lx.Order{
			ID:       id,
			Symbol:   "BTC-USD",
			Type:     lx.Limit,
			Side:     lx.Buy,
			Price:    100 + float64(id%100)/10,
			Size:     1 + float64(id%5),
			ClientID: fmt.Sprintf("c-%d", id),
		},
		Sender: addr,
	}
	hash, err := so.SigningHash()
	if err != nil {
		b.Fatalf("signing hash: %v", err)
	}
	sig, err := crypto.Sign(hash[:], key)
	if err != nil {
		b.Fatalf("sign: %v", err)
	}
	copy(so.Sig[:], sig)
	return so
}

// BenchmarkOrderBook benchmarks order processing
func BenchmarkOrderBook(b *testing.B) {
	ob := lx.NewOrderBook("BENCH-USD")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		order := &lx.Order{
			ID:     uint64(i + 1),
			Symbol: "BENCH-USD",
			Type:   lx.Limit,
			Side:   lx.Side(i % 2),
			Price:  100 + float64(i%100)/10,
			Size:   1,
			UserID: "bench",
		}

		ob.AddOrder(order)
	}

	ordersPerSecond := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(ordersPerSecond, "orders/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/order")
}

// BenchmarkOrderBookParallel benchmarks parallel order processing
func BenchmarkOrderBookParallel(b *testing.B) {
	ob := lx.NewOrderBook("PARALLEL-USD")

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			order := &lx.Order{
				ID:     uint64(i + 1),
				Symbol: "PARALLEL-USD",
				Type:   lx.Limit,
				Side:   lx.Side(i % 2),
				Price:  100 + float64(i%100),
				Size:   1,
				UserID: fmt.Sprintf("user_%d", i%10),
			}
			ob.AddOrder(order)
			i++
		}
	})

	ordersPerSecond := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(ordersPerSecond, "orders/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/order")
}

// BenchmarkBatchVerifyOrders compares per-order CPU ecrecover against the
// GPU-batched path on 10k orders. The CPU side is the same per-order oracle
// the test uses as a reference; the "GPU" side calls the batch entry point
// (luxcpp/crypto cgo dispatch when cgo is enabled, otherwise the CPU oracle).
func BenchmarkBatchVerifyOrders(b *testing.B) {
	const N = 10_000

	// Build the corpus once outside the timer — signing 10k orders with
	// fresh keys takes seconds and would dominate every iteration.
	orders := make([]lx.SignedOrder, N)
	for i := 0; i < N; i++ {
		orders[i] = mustSignOrder(b, uint64(i+1))
	}

	b.Run("CPU_PerOrder", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Single-element batches → forces per-order dispatch.
			for j := range orders {
				if _, err := lx.BatchVerifyOrders(orders[j : j+1]); err != nil {
					b.Fatal(err)
				}
			}
		}
		b.ReportMetric(float64(b.N*N)/b.Elapsed().Seconds(), "verifies/sec")
	})

	b.Run("GPU_Batch", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := lx.BatchVerifyOrders(orders); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N*N)/b.Elapsed().Seconds(), "verifies/sec")
	})
}

// BenchmarkBatchAMM compares a Go for-loop CPU evaluation against the
// GPU-batched amm_xyk_batch_metal dispatch on 100k tuples.
func BenchmarkBatchAMM(b *testing.B) {
	const N = 100_000
	reserves := make([]lx.ReservePair, N)
	amounts := make([]uint64, N)
	for i := 0; i < N; i++ {
		reserves[i] = lx.ReservePair{
			ReserveX: uint64(1+i) * 1_000_000,
			ReserveY: uint64(2+i) * 500_000,
		}
		amounts[i] = uint64(1 + i%10_000)
	}

	b.Run("CPU_Loop", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := lx.BatchEvalConstantProductCPU(reserves, amounts); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N*N)/b.Elapsed().Seconds(), "evals/sec")
	})

	b.Run("GPU_Batch", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := lx.BatchEvalConstantProduct(reserves, amounts); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N*N)/b.Elapsed().Seconds(), "evals/sec")
	})
}

// LP-108: BenchmarkMLXEngine and BenchmarkPlanetScale removed.
//
// BenchmarkMLXEngine pretended to measure an MLX-accelerated engine
// but called the standard CPU OrderBook and reported a hardcoded
// `b.ReportMetric(597, "ns/order")` literal — a fabricated number,
// not a measurement. The "MLX engine" itself
// (pkg/engine/mlx_engine.go) was a `time.Sleep` stub also archived
// under archive/lp108-2026-05-04/.
//
// BenchmarkPlanetScale reported entirely made-up metrics (markets=5M,
// orders/sec=150M, ns/order=597, watts=370, orders/watt=405405) that
// were never measured by any code in this repo.
//
// Honest CPU benchmarks remain at BenchmarkOrderBook /
// BenchmarkOrderBookParallel above. Real C++ numbers are at
// luxcpp/dex/build/luxdex_bench (5.51 M orders/sec single-symbol).
