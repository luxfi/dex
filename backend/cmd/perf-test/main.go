package main

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
)

func main() {
	fmt.Println("=== LX DEX Performance Test ===")
	fmt.Println("Testing order matching engine performance...")
	fmt.Println()

	// Create engine
	engine := lx.NewTradingEngine(lx.EngineConfig{})
	engine.CreateOrderBook("BTC-USDT")
	ob := engine.GetOrderBook("BTC-USDT")

	// Test 1: Single-threaded performance
	fmt.Println("Test 1: Single-threaded Order Insertion")
	testSingleThreaded(ob)

	// Test 2: Concurrent performance
	fmt.Println("\nTest 2: Concurrent Order Processing")
	testConcurrent(ob)

	// Test 3: Market order matching
	fmt.Println("\nTest 3: Market Order Matching Speed")
	testMarketOrders(ob)

	// Test 4: Latency distribution
	fmt.Println("\nTest 4: Latency Distribution Analysis")
	testLatencyDistribution(ob)

	fmt.Println("\n=== Performance Test Complete ===")
}

func testSingleThreaded(ob *lx.OrderBook) {
	const numOrders = 10000
	
	start := time.Now()
	
	for i := 0; i < numOrders; i++ {
		order := &lx.Order{
			ID:     uint64(i),
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Limit,
			Price:  50000 + float64(rand.Intn(1000)),
			Size:   0.1 + rand.Float64(),
		}
		if i%2 == 0 {
			order.Side = lx.Sell
		}
		
		ob.AddOrder(order)
	}
	
	elapsed := time.Since(start)
	throughput := float64(numOrders) / elapsed.Seconds()
	avgLatency := elapsed / time.Duration(numOrders)
	
	fmt.Printf("  Processed %d orders in %v\n", numOrders, elapsed)
	fmt.Printf("  Throughput: %.0f orders/sec\n", throughput)
	fmt.Printf("  Avg Latency: %v per order\n", avgLatency)
	
	if avgLatency < time.Microsecond {
		fmt.Printf("  ✅ SUB-MICROSECOND LATENCY ACHIEVED!\n")
	}
}

func testConcurrent(ob *lx.OrderBook) {
	const numThreads = 10
	const ordersPerThread = 1000
	
	var wg sync.WaitGroup
	wg.Add(numThreads)
	
	var totalOrders atomic.Uint64
	start := time.Now()
	
	for t := 0; t < numThreads; t++ {
		go func(threadID int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerThread; i++ {
				orderID := totalOrders.Add(1)
				order := &lx.Order{
					ID:     orderID,
					Symbol: "BTC-USDT",
					Side:   lx.Buy,
					Type:   lx.Limit,
					Price:  50000 + float64(rand.Intn(1000)),
					Size:   0.1,
				}
				if orderID%2 == 0 {
					order.Side = lx.Sell
				}
				
				ob.AddOrder(order)
			}
		}(t)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	orders := totalOrders.Load()
	throughput := float64(orders) / elapsed.Seconds()
	
	fmt.Printf("  Processed %d orders from %d threads in %v\n", orders, numThreads, elapsed)
	fmt.Printf("  Throughput: %.0f orders/sec\n", throughput)
}

func testMarketOrders(ob *lx.OrderBook) {
	// Pre-fill order book with liquidity
	for i := 0; i < 100; i++ {
		ob.AddOrder(&lx.Order{
			ID:     uint64(100000 + i),
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Limit,
			Price:  49000 + float64(i*10),
			Size:   10.0,
		})
		ob.AddOrder(&lx.Order{
			ID:     uint64(200000 + i),
			Symbol: "BTC-USDT",
			Side:   lx.Sell,
			Type:   lx.Limit,
			Price:  51000 + float64(i*10),
			Size:   10.0,
		})
	}
	
	const numMarketOrders = 1000
	start := time.Now()
	
	for i := 0; i < numMarketOrders; i++ {
		order := &lx.Order{
			ID:     uint64(300000 + i),
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Market,
			Size:   0.1,
		}
		if i%2 == 0 {
			order.Side = lx.Sell
		}
		
		ob.AddOrder(order)
	}
	
	elapsed := time.Since(start)
	throughput := float64(numMarketOrders) / elapsed.Seconds()
	avgLatency := elapsed / time.Duration(numMarketOrders)
	
	fmt.Printf("  Matched %d market orders in %v\n", numMarketOrders, elapsed)
	fmt.Printf("  Throughput: %.0f market orders/sec\n", throughput)
	fmt.Printf("  Avg Latency: %v per order\n", avgLatency)
}

func testLatencyDistribution(ob *lx.OrderBook) {
	const numSamples = 1000
	latencies := make([]time.Duration, numSamples)
	
	for i := 0; i < numSamples; i++ {
		order := &lx.Order{
			ID:     uint64(400000 + i),
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Limit,
			Price:  50000 + float64(rand.Intn(100)),
			Size:   0.01,
		}
		if i%2 == 0 {
			order.Side = lx.Sell
		}
		
		start := time.Now()
		ob.AddOrder(order)
		latencies[i] = time.Since(start)
	}
	
	// Calculate statistics
	var sum, min, max time.Duration
	min = time.Hour
	subMicro := 0
	
	for _, l := range latencies {
		sum += l
		if l < min {
			min = l
		}
		if l > max {
			max = l
		}
		if l < time.Microsecond {
			subMicro++
		}
	}
	
	avg := sum / time.Duration(numSamples)
	p50 := latencies[numSamples/2]
	p95 := latencies[numSamples*95/100]
	p99 := latencies[numSamples*99/100]
	
	fmt.Printf("  Samples: %d orders\n", numSamples)
	fmt.Printf("  Min:  %v\n", min)
	fmt.Printf("  Avg:  %v\n", avg)
	fmt.Printf("  P50:  %v\n", p50)
	fmt.Printf("  P95:  %v\n", p95)
	fmt.Printf("  P99:  %v\n", p99)
	fmt.Printf("  Max:  %v\n", max)
	
	percentage := float64(subMicro) / float64(numSamples) * 100
	fmt.Printf("  Sub-microsecond: %d orders (%.1f%%)\n", subMicro, percentage)
	
	if avg < 10*time.Microsecond {
		fmt.Printf("  ✅ ACHIEVED: Average latency under 10 microseconds!\n")
	}
	if p50 < time.Microsecond {
		fmt.Printf("  ✅ ACHIEVED: Median latency is sub-microsecond!\n")
	}
}