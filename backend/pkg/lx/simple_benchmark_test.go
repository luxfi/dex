package lx

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// BenchmarkSimpleAddOrder tests simple order addition
func BenchmarkSimpleAddOrder(b *testing.B) {
	ob := NewOrderBook("BTC-USD")

	// Pre-populate to avoid matching
	for i := 0; i < 100; i++ {
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  40000 + float64(i),
			Size:   1.0,
			User:   "maker",
		})
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  60000 + float64(i),
			Size:   1.0,
			User:   "maker",
		})
	}

	orders := make([]*Order, b.N)
	for i := 0; i < b.N; i++ {
		orders[i] = &Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Limit,
			Price:  50000 + float64(i%100),
			Size:   1.0,
			User:   fmt.Sprintf("user%d", i%100),
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ob.AddOrder(orders[i])
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
}

// BenchmarkThroughput tests sustained throughput
func BenchmarkThroughput(b *testing.B) {
	testCases := []struct {
		name      string
		numOrders int
	}{
		{"1K_Orders", 1000},
		{"10K_Orders", 10000},
		{"100K_Orders", 100000},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			ob := NewOrderBook("BTC-USD")

			orders := make([]*Order, tc.numOrders)
			for i := 0; i < tc.numOrders; i++ {
				orders[i] = &Order{
					Symbol: "BTC-USD",
					Side:   Side(i % 2),
					Type:   Limit,
					Price:  45000 + float64(rand.Intn(10000)),
					Size:   rand.Float64() * 10,
					User:   fmt.Sprintf("user%d", i%1000),
				}
			}

			b.ResetTimer()
			b.ReportAllocs()

			start := time.Now()
			for i := 0; i < tc.numOrders; i++ {
				ob.AddOrder(orders[i])
			}
			elapsed := time.Since(start)

			throughput := float64(tc.numOrders) / elapsed.Seconds()
			latencyUs := elapsed.Microseconds() / int64(tc.numOrders)

			b.ReportMetric(throughput, "orders/sec")
			b.ReportMetric(float64(latencyUs), "μs/order")

			b.Logf("  Processed %d orders in %v", tc.numOrders, elapsed)
			b.Logf("  Throughput: %.0f orders/sec", throughput)
			b.Logf("  Latency: %d μs/order", latencyUs)
		})
	}
}

// BenchmarkComparison shows the improvement
func BenchmarkComparison(b *testing.B) {
	// Test string keys (simulating old implementation)
	b.Run("StringKeys", func(b *testing.B) {
		m := make(map[string]interface{})
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("%.8f", 50000.0+float64(i%1000))
			m[key] = i
		}
	})

	// Test integer keys (new implementation)
	b.Run("IntegerKeys", func(b *testing.B) {
		m := make(map[int64]interface{})
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			key := int64((50000.0 + float64(i%1000)) * 100000000)
			m[key] = i
		}
	})
}

// BenchmarkSnapshot tests snapshot performance
func BenchmarkSnapshot(b *testing.B) {
	ob := NewOrderBook("BTC-USD")

	// Add 1000 orders to each side
	for i := 0; i < 1000; i++ {
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  49000 + float64(i),
			Size:   1.0,
			User:   "user",
		})
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  51000 + float64(i),
			Size:   1.0,
			User:   "user",
		})
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ob.GetSnapshot()
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "snapshots/sec")
}
