package lx

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

// BenchmarkOptimizedAddOrder tests order addition performance
func BenchmarkOptimizedAddOrder(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		order := &Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Limit,
			Price:  50000 + float64(i%1000),
			Size:   1.0,
			User:   fmt.Sprintf("user%d", i%100),
		}
		ob.AddOrder(order)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
}

// BenchmarkOptimizedMatching tests order matching performance
func BenchmarkOptimizedMatching(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	
	// Pre-populate with orders
	for i := 0; i < 1000; i++ {
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  49000 + float64(i),
			Size:   1.0,
			User:   "maker",
		})
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  51000 + float64(i),
			Size:   1.0,
			User:   "maker",
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Add crossing orders that will match
		if i%2 == 0 {
			ob.AddOrder(&Order{
				Symbol: "BTC-USD",
				Side:   Buy,
				Type:   Limit,
				Price:  52000, // Will cross with asks
				Size:   0.1,
				User:   "taker",
			})
		} else {
			ob.AddOrder(&Order{
				Symbol: "BTC-USD",
				Side:   Sell,
				Type:   Limit,
				Price:  48000, // Will cross with bids
				Size:   0.1,
				User:   "taker",
			})
		}
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "matches/sec")
}

// BenchmarkOptimizedCancelOrder tests order cancellation performance
func BenchmarkOptimizedCancelOrder(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	orderIDs := make([]uint64, 0, b.N)
	
	// Add orders first
	for i := 0; i < b.N; i++ {
		order := &Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Limit,
			Price:  50000 + float64(i%1000),
			Size:   1.0,
			User:   "user",
		}
		orderID := ob.AddOrder(order)
		if orderID > 0 {
			orderIDs = append(orderIDs, orderID)
		}
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for _, orderID := range orderIDs {
		ob.CancelOrder(orderID)
	}
	
	b.ReportMetric(float64(len(orderIDs))/b.Elapsed().Seconds(), "cancels/sec")
}

// BenchmarkOptimizedGetSnapshot tests snapshot generation performance
func BenchmarkOptimizedGetSnapshot(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	
	// Pre-populate
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

// BenchmarkOptimizedConcurrent tests concurrent operations
func BenchmarkOptimizedConcurrent(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	numGoroutines := runtime.NumCPU()
	ordersPerGoroutine := b.N / numGoroutines
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	
	start := time.Now()
	
	for g := 0; g < numGoroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < ordersPerGoroutine; i++ {
				order := &Order{
					Symbol: "BTC-USD",
					Side:   Side(i % 2),
					Price:  50000 + float64(rand.Intn(1000)),
					Size:   rand.Float64() * 10,
					Type:   Limit,
					User:   fmt.Sprintf("user%d", id),
				}
				ob.AddOrder(order)
			}
		}(g)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	totalOrders := numGoroutines * ordersPerGoroutine
	b.ReportMetric(float64(totalOrders)/elapsed.Seconds(), "concurrent_orders/sec")
}

// BenchmarkOptimizedMarketOrder tests market order performance
func BenchmarkOptimizedMarketOrder(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	
	// Pre-populate with limit orders
	for i := 0; i < 1000; i++ {
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  49000 + float64(i),
			Size:   10.0,
			User:   "maker",
		})
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  51000 + float64(i),
			Size:   10.0,
			User:   "maker",
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		order := &Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Market,
			Size:   0.5,
			User:   "taker",
		}
		ob.AddOrder(order)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "market_orders/sec")
}

// BenchmarkPriceKeyComparison compares string vs integer price keys
func BenchmarkPriceKeyComparison(b *testing.B) {
	b.Run("IntegerKeys", func(b *testing.B) {
		m := make(map[int64]int)
		b.ResetTimer()
		b.ReportAllocs()
		
		for i := 0; i < b.N; i++ {
			key := int64(50000.12345678 * 100000000)
			m[key] = i
			_ = m[key]
		}
	})
	
	b.Run("StringKeys", func(b *testing.B) {
		m := make(map[string]int)
		b.ResetTimer()
		b.ReportAllocs()
		
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("%.8f", 50000.12345678)
			m[key] = i
			_ = m[key]
		}
	})
}

// BenchmarkMemoryAllocations tests memory allocation patterns
func BenchmarkMemoryAllocations(b *testing.B) {
	b.Run("IntegerConversion", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = int64(50000.12345678 * 100000000)
		}
	})
	
	b.Run("StringFormatting", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = fmt.Sprintf("%.8f", 50000.12345678)
		}
	})
}

// BenchmarkLargeOrderBook tests performance with many orders
func BenchmarkLargeOrderBook(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			ob := NewOrderBook("BTC-USD")
			
			// Pre-populate
			for i := 0; i < size; i++ {
				ob.AddOrder(&Order{
					Symbol: "BTC-USD",
					Side:   Side(i % 2),
					Type:   Limit,
					Price:  45000 + float64(i%10000),
					Size:   1.0,
					User:   "user",
				})
			}
			
			b.ResetTimer()
			b.ReportAllocs()
			
			// Measure operations on large book
			for i := 0; i < b.N; i++ {
				order := &Order{
					Symbol: "BTC-USD",
					Side:   Side(i % 2),
					Type:   Limit,
					Price:  50000 + float64(i%100),
					Size:   1.0,
					User:   "bench",
				}
				ob.AddOrder(order)
			}
			
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
		})
	}
}

// BenchmarkStressTest runs a comprehensive stress test
func BenchmarkStressTest(b *testing.B) {
	ob := NewOrderBook("BTC-USD")
	numOrders := 100000
	
	b.ResetTimer()
	
	start := time.Now()
	for i := 0; i < numOrders; i++ {
		order := &Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Limit,
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   rand.Float64() * 10,
			User:   fmt.Sprintf("user%d", i%1000),
		}
		ob.AddOrder(order)
	}
	elapsed := time.Since(start)
	
	ordersPerSec := float64(numOrders) / elapsed.Seconds()
	latencyUs := elapsed.Microseconds() / int64(numOrders)
	
	b.Logf("Stress Test Results:")
	b.Logf("  Total orders: %d", numOrders)
	b.Logf("  Time taken: %v", elapsed)
	b.Logf("  Throughput: %.0f orders/sec", ordersPerSec)
	b.Logf("  Avg latency: %d μs/order", latencyUs)
	b.Logf("  Book stats: %d bids, %d asks, %d trades", 
		len(ob.Bids.orders), len(ob.Asks.orders), len(ob.Trades))
}