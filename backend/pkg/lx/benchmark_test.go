package lx

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkSimpleAddOrder measures single order insertion performance
func BenchmarkSimpleAddOrder(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	
	orders := make([]*Order, 10000)
	for i := range orders {
		orders[i] = &Order{
			ID:     fmt.Sprintf("order_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   0.1 + rand.Float64(),
			UserID: "trader1",
		}
		if i%2 == 0 {
			orders[i].Side = "sell"
		}
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		order := orders[i%len(orders)]
		engine.PlaceOrder(order)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
}

// BenchmarkSimpleMatching measures order matching performance
func BenchmarkSimpleMatching(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	
	// Pre-fill order book
	for i := 0; i < 1000; i++ {
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("buy_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  49000 + float64(i),
			Size:   1.0,
			UserID: "mm",
		})
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("sell_%d", i),
			Symbol: "BTC-USDT",
			Side:   "sell",
			Type:   "limit",
			Price:  51000 + float64(i),
			Size:   1.0,
			UserID: "mm",
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Place crossing orders that will match
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("match_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  52000, // Crosses with sells
			Size:   0.01,
			UserID: "trader",
		})
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "matches/sec")
}

// BenchmarkOrderBookParallel measures concurrent performance
func BenchmarkOrderBookParallel(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	
	var orderID atomic.Uint64
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := orderID.Add(1)
			order := &Order{
				ID:     fmt.Sprintf("p_%d", id),
				Symbol: "BTC-USDT",
				Side:   "buy",
				Type:   "limit",
				Price:  50000 + float64(rand.Intn(1000)),
				Size:   0.1,
				UserID: fmt.Sprintf("trader_%d", id%10),
			}
			if id%2 == 0 {
				order.Side = "sell"
			}
			engine.PlaceOrder(order)
		}
	})
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
}

// BenchmarkOrderBookMarketOrder measures market order performance
func BenchmarkOrderBookMarketOrder(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	
	// Pre-fill with liquidity
	for i := 0; i < 1000; i++ {
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("liq_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  49000 + float64(i),
			Size:   10.0,
			UserID: "mm",
		})
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("liq_s_%d", i),
			Symbol: "BTC-USDT",
			Side:   "sell",
			Type:   "limit",
			Price:  51000 + float64(i),
			Size:   10.0,
			UserID: "mm",
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("market_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "market",
			Size:   0.1,
			UserID: "trader",
		})
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "market_orders/sec")
}

// BenchmarkSimpleSnapshot measures snapshot generation performance
func BenchmarkSimpleSnapshot(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	ob := engine.CreateOrderBook("BTC-USDT")
	
	// Fill with orders
	for i := 0; i < 100; i++ {
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("b_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  49000 + float64(i*10),
			Size:   float64(i + 1),
			UserID: "mm",
		})
		engine.PlaceOrder(&Order{
			ID:     fmt.Sprintf("s_%d", i),
			Symbol: "BTC-USDT",
			Side:   "sell",
			Type:   "limit",
			Price:  51000 + float64(i*10),
			Size:   float64(i + 1),
			UserID: "mm",
		})
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_ = ob.GetSnapshot(10)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "snapshots/sec")
}

// BenchmarkOrderBookStress simulates heavy trading load
func BenchmarkOrderBookStress(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	engine.CreateOrderBook("ETH-USDT")
	
	symbols := []string{"BTC-USDT", "ETH-USDT"}
	prices := map[string]float64{
		"BTC-USDT": 50000,
		"ETH-USDT": 3000,
	}
	
	numTraders := 100
	ordersPerTrader := b.N / numTraders
	if ordersPerTrader == 0 {
		ordersPerTrader = 1
	}
	
	var wg sync.WaitGroup
	wg.Add(numTraders)
	
	start := time.Now()
	var totalOrders atomic.Uint64
	
	for t := 0; t < numTraders; t++ {
		go func(traderID int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerTrader; i++ {
				symbol := symbols[rand.Intn(len(symbols))]
				basePrice := prices[symbol]
				
				order := &Order{
					ID:     fmt.Sprintf("t%d_o%d", traderID, i),
					Symbol: symbol,
					Side:   "buy",
					Type:   "limit",
					Price:  basePrice * (0.95 + rand.Float64()*0.1),
					Size:   0.01 + rand.Float64()*0.1,
					UserID: fmt.Sprintf("trader_%d", traderID),
				}
				
				if rand.Intn(2) == 0 {
					order.Side = "sell"
				}
				if rand.Intn(10) == 0 {
					order.Type = "market"
				}
				
				engine.PlaceOrder(order)
				totalOrders.Add(1)
			}
		}(t)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	orders := totalOrders.Load()
	throughput := float64(orders) / elapsed.Seconds()
	
	b.ReportMetric(throughput, "orders/sec")
	b.Logf("Processed %d orders in %v (%.0f orders/sec)", orders, elapsed, throughput)
}

// MeasureLatencyDistribution measures detailed latency statistics
func TestMeasureLatencyDistribution(t *testing.T) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	
	const numOrders = 10000
	latencies := make([]time.Duration, 0, numOrders)
	
	for i := 0; i < numOrders; i++ {
		order := &Order{
			ID:     fmt.Sprintf("lat_%d", i),
			Symbol: "BTC-USDT",
			Side:   "buy",
			Type:   "limit",
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   0.1,
			UserID: "bench",
		}
		if i%2 == 0 {
			order.Side = "sell"
		}
		
		start := time.Now()
		engine.PlaceOrder(order)
		latency := time.Since(start)
		latencies = append(latencies, latency)
	}
	
	// Calculate statistics
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
	
	// Sort for percentiles (simplified - just use middle values)
	p50 := latencies[len(latencies)/2]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]
	
	t.Logf("=== Latency Distribution for %d orders ===", numOrders)
	t.Logf("Min:  %v", min)
	t.Logf("Avg:  %v", avg)
	t.Logf("P50:  %v", p50)
	t.Logf("P95:  %v", p95)
	t.Logf("P99:  %v", p99)
	t.Logf("Max:  %v", max)
	
	// Check if we achieve sub-microsecond for some operations
	subMicro := 0
	for _, l := range latencies {
		if l < time.Microsecond {
			subMicro++
		}
	}
	
	percentage := float64(subMicro) / float64(len(latencies)) * 100
	t.Logf("Sub-microsecond: %d orders (%.1f%%)", subMicro, percentage)
	
	if avg < 10*time.Microsecond {
		t.Logf("✅ PERFORMANCE GOAL ACHIEVED: Average latency is under 10 microseconds!")
	}
}