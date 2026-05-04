package benchmark

import (
	"fmt"
	"testing"

	"github.com/luxfi/dex/pkg/lx"
)

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
