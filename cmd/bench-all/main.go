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

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
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
	fmt.Println("📊 Testing MLX GPU Acceleration...")
	mlxEngine, err := mlx.NewEngine(mlx.Config{Backend: mlx.BackendAuto})
	if err == nil && mlxEngine.IsGPUAvailable() {
		result = benchmarkMLX(*numOrders, *parallel)
		results = append(results, result)
		fmt.Printf("   ✅ %s: %.0f orders/sec on %s\n\n",
			result.Name, result.Throughput, mlxEngine.Device())
	} else {
		fmt.Println("   ⚠️  MLX GPU not available on this platform")
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
	engine, err := mlx.NewEngine(mlx.Config{Backend: mlx.BackendAuto})
	if err != nil || !engine.IsGPUAvailable() {
		return BenchResult{Name: "MLX (N/A)"}
	}

	// Benchmark GPU matching using the engine's benchmark method
	throughput := engine.Benchmark(numOrders)

	// Calculate values from throughput
	duration := time.Duration(float64(time.Second) * float64(numOrders) / throughput)
	latency := time.Duration(float64(time.Second) / throughput)

	engine.Close()

	return BenchResult{
		Name:       "MLX GPU",
		Orders:     numOrders,
		Trades:     0, // Not tracked in benchmark
		Duration:   duration,
		Throughput: throughput,
		Latency:    latency,
	}
}

func benchmarkKernelBypass(numOrders int) BenchResult {
	// Kernel bypass not available on this platform
	// Would require DPDK on Linux or similar technology
	return BenchResult{
		Name:       fmt.Sprintf("Kernel Bypass (Error: no kernel-bypass method available on %s)", runtime.GOOS),
		Orders:     0,
		Trades:     0,
		Duration:   0,
		Throughput: 0,
		Latency:    0,
	}
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
