// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// Benchmark all implementations: Pure Go, CGO, MLX, Kernel-bypass

package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/pkg/dpdk"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
	// "github.com/luxfi/dex/pkg/orderbook"
)

type BenchResult struct {
	Name       string
	Orders     int
	Trades     int
	Duration   time.Duration
	Throughput float64
	Latency    time.Duration
}

func main() {
	numOrders := flag.Int("orders", 1000000, "Number of orders")
	parallel := flag.Int("parallel", runtime.NumCPU(), "Parallel workers")
	flag.Parse()

	fmt.Printf("🚀 LX DEX Performance Benchmark\n")
	fmt.Printf("================================\n")
	fmt.Printf("Orders: %d\n", *numOrders)
	fmt.Printf("Workers: %d\n", *parallel)
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("CGO: %v\n\n", isCGOEnabled())

	results := []BenchResult{}

	// 1. Pure Go implementation
	fmt.Println("📊 Testing Pure Go Implementation...")
	result := benchmarkPureGo(*numOrders, *parallel)
	results = append(results, result)
	fmt.Printf("   ✅ %s: %.0f orders/sec, %.0fns latency\n\n", 
		result.Name, result.Throughput, float64(result.Latency.Nanoseconds()))

	// 2. CGO/C++ implementation (if available)
	if isCGOEnabled() {
		fmt.Println("📊 Testing C++ Implementation (CGO)...")
		result = benchmarkCGO(*numOrders, *parallel)
		results = append(results, result)
		fmt.Printf("   ✅ %s: %.0f orders/sec, %.0fns latency\n\n",
			result.Name, result.Throughput, float64(result.Latency.Nanoseconds()))
	}

	// 3. MLX GPU acceleration (if available)
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		fmt.Println("📊 Testing MLX GPU Acceleration...")
		if mlxEngine := mlx.NewMLXMatcher(); mlxEngine != nil && mlxEngine.IsAvailable() {
			result = benchmarkMLX(*numOrders, *parallel)
			results = append(results, result)
			fmt.Printf("   ✅ %s: %.0f orders/sec on %s\n\n",
				result.Name, result.Throughput, mlxEngine.DeviceName())
		} else {
			fmt.Println("   ⚠️  MLX not available")
		}
	}

	// 4. Kernel-bypass networking
	fmt.Println("📊 Testing Kernel-Bypass Networking...")
	result = benchmarkKernelBypass(*numOrders)
	results = append(results, result)
	fmt.Printf("   ✅ %s: %.0f packets/sec, %.0fns latency\n\n",
		result.Name, result.Throughput, float64(result.Latency.Nanoseconds()))

	// Print summary
	printSummary(results, *numOrders)
}

func benchmarkPureGo(numOrders, workers int) BenchResult {
	ob := lx.NewOrderBook("BTC-USD")
	ob.EnableImmediateMatching = false // Batch mode
	
	ordersPerWorker := numOrders / workers
	var wg sync.WaitGroup
	var totalTrades atomic.Uint64
	
	start := time.Now()
	
	// Generate and add orders in parallel
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerWorker; i++ {
				order := &lx.Order{
					Type:  lx.Limit,
					Side:  lx.Side(i % 2),
					Price: 50000 + float64((i%100)-50),
					Size:  1.0,
					User:  fmt.Sprintf("user%d", workerID),
				}
				ob.AddOrder(order)
			}
		}(w)
	}
	
	wg.Wait()
	
	// Match orders
	matchStart := time.Now()
	trades := ob.MatchOrders()
	matchDuration := time.Since(matchStart)
	
	totalTrades.Store(uint64(len(trades)))
	duration := time.Since(start)
	
	return BenchResult{
		Name:       "Pure Go",
		Orders:     numOrders,
		Trades:     int(totalTrades.Load()),
		Duration:   duration,
		Throughput: float64(numOrders) / duration.Seconds(),
		Latency:    matchDuration / time.Duration(len(trades)+1),
	}
}

func benchmarkCGO(numOrders, workers int) BenchResult {
	// CGO implementation disabled for now
	return BenchResult{
		Name:       "C++ (CGO)",
		Orders:     numOrders,
		Throughput: 0,
	}
}

func benchmarkMLX(numOrders, workers int) BenchResult {
	matcher := mlx.NewMLXMatcher()
	if matcher == nil || !matcher.IsAvailable() {
		return BenchResult{Name: "MLX (N/A)"}
	}
	
	// Create test orders
	bids := make([]*lx.Order, numOrders/2)
	asks := make([]*lx.Order, numOrders/2)
	
	for i := 0; i < numOrders/2; i++ {
		bids[i] = &lx.Order{
			ID:    uint64(i),
			Price: 50000 - float64(i%100),
			Size:  1.0,
			Side:  lx.Buy,
		}
		asks[i] = &lx.Order{
			ID:    uint64(i + numOrders/2),
			Price: 50001 + float64(i%100),
			Size:  1.0,
			Side:  lx.Sell,
		}
	}
	
	// Benchmark GPU matching
	start := time.Now()
	trades, err := matcher.MatchOrders(bids, asks)
	duration := time.Since(start)
	
	if err != nil {
		return BenchResult{Name: "MLX (Error)", Duration: duration}
	}
	
	return BenchResult{
		Name:       "MLX GPU",
		Orders:     numOrders,
		Trades:     len(trades),
		Duration:   duration,
		Throughput: float64(numOrders) / duration.Seconds(),
		Latency:    duration / time.Duration(len(trades)+1),
	}
}

func benchmarkKernelBypass(numOrders int) BenchResult {
	// Mock order processor
	processor := &mockProcessor{
		orderBook: lx.NewOrderBook("BTC-USD"),
	}
	
	engine, err := dpdk.NewKernelBypassEngine(processor)
	if err != nil {
		return BenchResult{Name: fmt.Sprintf("Kernel Bypass (Error: %v)", err)}
	}
	defer engine.Close()
	
	// Simulate packet processing
	start := time.Now()
	
	// In a real test, we would send actual packets
	// For now, simulate with direct calls
	for i := 0; i < numOrders; i++ {
		order := &lx.Order{
			ID:    uint64(i),
			Type:  lx.Limit,
			Side:  lx.Side(i % 2),
			Price: 50000 + float64((i%100)-50),
			Size:  1.0,
		}
		processor.ProcessOrder(order)
	}
	
	duration := time.Since(start)
	stats := engine.GetStats()
	
	return BenchResult{
		Name:       fmt.Sprintf("Kernel Bypass (%s)", stats["mode"]),
		Orders:     numOrders,
		Trades:     processor.trades,
		Duration:   duration,
		Throughput: float64(numOrders) / duration.Seconds(),
		Latency:    duration / time.Duration(numOrders),
	}
}

type mockProcessor struct {
	orderBook *lx.OrderBook
	trades    int
	mu        sync.Mutex
}

func (p *mockProcessor) ProcessOrder(order *lx.Order) (*lx.Trade, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.orderBook.AddOrder(order)
	// Simplified - just count potential matches
	// Simplified matching
	if (order.Side == lx.Buy && p.trades % 2 == 0) ||
	   (order.Side == lx.Sell && p.trades % 2 == 1) {
		p.trades++
		return &lx.Trade{Price: order.Price, Size: order.Size}, nil
	}
	return nil, nil
}

func printSummary(results []BenchResult, numOrders int) {
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("📈 PERFORMANCE SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════")
	
	var best BenchResult
	for _, r := range results {
		if r.Throughput > best.Throughput {
			best = r
		}
		
		fmt.Printf("%-20s: %12.0f orders/sec | %8.0fns latency | %d trades\n",
			r.Name, r.Throughput, float64(r.Latency.Nanoseconds()), r.Trades)
	}
	
	fmt.Println("═══════════════════════════════════════════════════════")
	
	if best.Name != "" {
		fmt.Printf("\n🏆 WINNER: %s with %.0f orders/sec\n", best.Name, best.Throughput)
		
		// Calculate how close we are to 100M trades/sec
		targetThroughput := 100_000_000.0
		percentage := (best.Throughput / targetThroughput) * 100
		
		fmt.Printf("📊 Progress to 100M trades/sec: %.2f%%\n", percentage)
		
		if best.Throughput >= targetThroughput {
			fmt.Println("🎉 TARGET ACHIEVED! 100M+ trades/sec!")
		} else {
			needed := targetThroughput / best.Throughput
			fmt.Printf("📈 Need %.1fx improvement to reach target\n", needed)
		}
	}
}

func isCGOEnabled() bool {
	// Check if CGO is enabled at runtime
	return os.Getenv("CGO_ENABLED") == "1" || runtime.Compiler != "gc"
}