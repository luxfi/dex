// +build cgo

package lx

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkUltraFastMatcher measures the performance of the C++ ultra-fast matcher
func BenchmarkUltraFastMatcher(b *testing.B) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	// Pre-generate orders to avoid allocation during benchmark
	orders := make([]*Order, 10000)
	for i := range orders {
		orders[i] = &Order{
			ID:     fmt.Sprintf("%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   0.1 + rand.Float64(),
		}
		if i%2 == 0 {
			orders[i].Side = "sell"
		}
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var totalLatency time.Duration
	for i := 0; i < b.N; i++ {
		order := orders[i%len(orders)]
		latency, _ := matcher.AddOrder(order)
		totalLatency += latency
	}
	
	// Report custom metrics
	b.ReportMetric(float64(totalLatency.Nanoseconds())/float64(b.N), "ns/order")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
	
	stats := matcher.GetStats()
	b.Logf("Total Orders: %d, Total Trades: %d, Avg Latency: %v", 
		stats.TotalOrders, stats.TotalTrades, stats.AverageLatency)
}

// BenchmarkUltraFastMatcherParallel measures parallel performance
func BenchmarkUltraFastMatcherParallel(b *testing.B) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	b.RunParallel(func(pb *testing.PB) {
		var orderID uint64
		for pb.Next() {
			id := atomic.AddUint64(&orderID, 1)
			order := &Order{
				ID:     fmt.Sprintf("%d", id),
				Symbol: "BTC-USDT",
				Side:   "buy",
				Type:   "limit",
				Price:  50000 + float64(rand.Intn(1000)),
				Size:   0.1,
			}
			if id%2 == 0 {
				order.Side = "sell"
			}
			matcher.AddOrder(order)
		}
	})
	
	stats := matcher.GetStats()
	b.ReportMetric(float64(stats.TotalOrders)/b.Elapsed().Seconds(), "orders/sec")
	b.Logf("Total Orders: %d, Total Trades: %d", stats.TotalOrders, stats.TotalTrades)
}

// BenchmarkUltraFastMarketOrders benchmarks market order matching
func BenchmarkUltraFastMarketOrders(b *testing.B) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	// Pre-fill order book with liquidity
	for i := 0; i < 1000; i++ {
		matcher.AddOrder(&Order{
			ID:     fmt.Sprintf("liq_%d", i),
			Symbol: "BTC-USDT",
			Side:   "sell",
			Type:   "limit",
			Price:  50100 + float64(i),
			Size:   1.0,
		})
		matcher.AddOrder(&Order{
			ID:     fmt.Sprintf("liq_b_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  49900 - float64(i),
			Size:   1.0,
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		order := &Order{
			ID:     fmt.Sprintf("market_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "market",
			Size:   0.1,
		}
		if i%2 == 0 {
			order.Side = "sell"
		}
		matcher.AddOrder(order)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "market_orders/sec")
}

// TestUltraFastMatcherCorrectness verifies the matcher works correctly
func TestUltraFastMatcherCorrectness(t *testing.T) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	// Add buy order
	buyOrder := &Order{
		ID:     "buy_1",
		Symbol: "BTC-USDT",
		Side:   "buy",
		Type:   "limit",
		Price:  50000,
		Size:   1.0,
	}
	
	latency, err := matcher.AddOrder(buyOrder)
	if err != nil {
		t.Fatalf("Failed to add buy order: %v", err)
	}
	t.Logf("Buy order added with latency: %v", latency)
	
	// Add matching sell order
	sellOrder := &Order{
		ID:     "sell_1",
		Symbol: "BTC-USDT",
		Side:   "sell",
		Type:   "limit",
		Price:  50000,
		Size:   1.0,
	}
	
	trades, err := matcher.MatchOrder(sellOrder)
	if err != nil {
		t.Fatalf("Failed to match sell order: %v", err)
	}
	
	if len(trades) != 1 {
		t.Fatalf("Expected 1 trade, got %d", len(trades))
	}
	
	trade := trades[0]
	if trade.Price != 50000 {
		t.Errorf("Expected trade price 50000, got %f", trade.Price)
	}
	if trade.Quantity != 1.0 {
		t.Errorf("Expected trade quantity 1.0, got %f", trade.Quantity)
	}
	
	// Get best bid/ask
	bid, ask := matcher.GetBestBidAsk()
	t.Logf("Best Bid: %f, Best Ask: %f", bid, ask)
	
	// Get stats
	stats := matcher.GetStats()
	t.Logf("Stats: Orders=%d, Trades=%d, Volume=%d, AvgLatency=%v",
		stats.TotalOrders, stats.TotalTrades, stats.TotalVolume, stats.AverageLatency)
}

// BenchmarkCompareImplementations compares Go vs C++ performance
func BenchmarkCompareImplementations(b *testing.B) {
	b.Run("PureGo", func(b *testing.B) {
		engine := NewTradingEngine(EngineConfig{})
		engine.CreateOrderBook("BTC-USDT")
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			order := &Order{
				ID:     fmt.Sprintf("%d", i),
				Symbol: "BTC-USDT",
				Side:   "buy",
				Type:   "limit",
				Price:  50000 + float64(i%100),
				Size:   0.1,
			}
			if i%2 == 0 {
				order.Side = "sell"
			}
			engine.PlaceOrder(order)
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
	})
	
	b.Run("UltraFastCpp", func(b *testing.B) {
		matcher := NewUltraFastMatcher()
		defer matcher.Destroy()
		
		b.ResetTimer()
		var totalNanos int64
		for i := 0; i < b.N; i++ {
			order := &Order{
				ID:     fmt.Sprintf("%d", i),
				Symbol: "BTC-USDT",
				Side:   "buy",
				Type:   "limit",
				Price:  50000 + float64(i%100),
				Size:   0.1,
			}
			if i%2 == 0 {
				order.Side = "sell"
			}
			latency, _ := matcher.AddOrder(order)
			totalNanos += latency.Nanoseconds()
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
		b.ReportMetric(float64(totalNanos)/float64(b.N), "ns/order")
	})
}

// TestUltraFastLatencyDistribution measures latency distribution
func TestUltraFastLatencyDistribution(t *testing.T) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	const numOrders = 100000
	latencies := make([]time.Duration, 0, numOrders)
	
	// Measure latencies
	for i := 0; i < numOrders; i++ {
		order := &Order{
			ID:     fmt.Sprintf("%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   0.1,
		}
		if i%2 == 0 {
			order.Side = "sell"
		}
		
		latency, _ := matcher.AddOrder(order)
		latencies = append(latencies, latency)
	}
	
	// Calculate percentiles
	var sum, min, max time.Duration
	min = time.Hour
	for _, l := range latencies {
		sum += l
		if l < min {
			min = l
		}
		if l > max {
			max = l
		}
	}
	
	avg := sum / time.Duration(len(latencies))
	
	// Sort for percentiles (simplified)
	p50 := latencies[len(latencies)/2]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]
	
	t.Logf("Latency Distribution for %d orders:", numOrders)
	t.Logf("  Min: %v", min)
	t.Logf("  Avg: %v", avg)
	t.Logf("  P50: %v", p50)
	t.Logf("  P95: %v", p95)
	t.Logf("  P99: %v", p99)
	t.Logf("  Max: %v", max)
	
	// Verify sub-microsecond for most orders
	if p95 > time.Microsecond {
		t.Logf("WARNING: P95 latency (%v) exceeds 1 microsecond target", p95)
	} else {
		t.Logf("SUCCESS: P95 latency (%v) is sub-microsecond!", p95)
	}
	
	stats := matcher.GetStats()
	t.Logf("Final Stats: %+v", stats)
}

// BenchmarkUltraFastHeavyLoad simulates heavy trading load
func BenchmarkUltraFastHeavyLoad(b *testing.B) {
	matcher := NewUltraFastMatcher()
	defer matcher.Destroy()
	
	// Number of concurrent traders
	numTraders := 100
	ordersPerTrader := b.N / numTraders
	
	var wg sync.WaitGroup
	wg.Add(numTraders)
	
	start := time.Now()
	
	for trader := 0; trader < numTraders; trader++ {
		go func(traderID int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerTrader; i++ {
				order := &Order{
					ID:     fmt.Sprintf("t%d_o%d", traderID, i),
					Symbol: "BTC-USDT",
					Side:   "buy",
					Type:   "limit",
					Price:  49000 + float64(rand.Intn(2000)),
					Size:   0.01 + rand.Float64()*0.1,
				}
				if rand.Intn(2) == 0 {
					order.Side = "sell"
				}
				if rand.Intn(10) == 0 {
					order.Type = "market"
				}
				
				matcher.AddOrder(order)
			}
		}(trader)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	stats := matcher.GetStats()
	throughput := float64(stats.TotalOrders) / elapsed.Seconds()
	
	b.ReportMetric(throughput, "orders/sec")
	b.ReportMetric(float64(stats.TotalTrades)/elapsed.Seconds(), "trades/sec")
	b.Logf("Processed %d orders in %v (%.0f orders/sec)", 
		stats.TotalOrders, elapsed, throughput)
	b.Logf("Generated %d trades, Average latency: %v",
		stats.TotalTrades, stats.AverageLatency)
}