package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/orderbook"
)

func main() {
	iterations := flag.Int("iter", 100000, "Number of iterations")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of concurrent workers")
	warmup := flag.Int("warmup", 10000, "Warmup iterations")
	implementation := flag.String("impl", "auto", "Implementation: go, cpp, auto (tests both)")
	flag.Parse()

	fmt.Println("🏁 DEX Performance Benchmark")
	fmt.Printf("📊 Iterations: %d | Workers: %d | Warmup: %d\n", *iterations, *workers, *warmup)
	fmt.Printf("🖥️  CPU Cores: %d | CGO: %s\n", runtime.NumCPU(), os.Getenv("CGO_ENABLED"))
	fmt.Println()

	switch *implementation {
	case "go":
		runBenchmark(orderbook.ImplGo, "Pure Go", *iterations, *workers, *warmup)
	case "cpp":
		if os.Getenv("CGO_ENABLED") == "1" {
			runBenchmark(orderbook.ImplCpp, "C++ (CGO)", *iterations, *workers, *warmup)
		} else {
			fmt.Println("❌ CGO not enabled. Run with: CGO_ENABLED=1 go run main.go -impl cpp")
		}
	case "auto":
		// Test both and compare
		results := []BenchResult{}
		
		// Always test Pure Go
		goResult := runBenchmark(orderbook.ImplGo, "Pure Go", *iterations, *workers, *warmup)
		results = append(results, goResult)
		
		// Test C++ if available
		if os.Getenv("CGO_ENABLED") == "1" {
			fmt.Println()
			cppResult := runBenchmark(orderbook.ImplCpp, "C++ (CGO)", *iterations, *workers, *warmup)
			results = append(results, cppResult)
			
			// Compare results
			printComparison(results)
		} else {
			fmt.Println("\n⚠️  CGO not enabled. To compare with C++:")
			fmt.Println("   CGO_ENABLED=1 go run main.go")
		}
	default:
		fmt.Printf("Unknown implementation: %s\n", *implementation)
		fmt.Println("Use: go, cpp, or auto")
	}
}

type BenchResult struct {
	Name       string
	Ops        int64
	Duration   time.Duration
	Throughput float64
	Latency    time.Duration
}

func runBenchmark(impl orderbook.Implementation, name string, iterations, workers, warmup int) BenchResult {
	fmt.Printf("=== Benchmarking %s ===\n", name)
	
	// Create orderbook
	ob := orderbook.NewOrderBook(orderbook.Config{
		Implementation: impl,
		Symbol:         "BTC/USD",
	})

	// Warmup
	fmt.Printf("  Warming up (%d ops)... ", warmup)
	for i := 0; i < warmup; i++ {
		order := &orderbook.Order{
			ID:        uint64(i),
			Symbol:    "BTC/USD",
			Side:      orderbook.OrderSide(i % 2),
			Price:     50000 + float64(i%100),
			Quantity:  1.0,
			Timestamp: time.Now(),
		}
		ob.AddOrder(order)
	}
	fmt.Println("✓")

	// Benchmark
	var totalOps int64
	var totalLatency int64
	start := time.Now()

	var wg sync.WaitGroup
	opsPerWorker := iterations / workers

	fmt.Printf("  Running %d workers... ", workers)
	
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startID := uint64(workerID*opsPerWorker + warmup)
			
			for i := 0; i < opsPerWorker; i++ {
				opStart := time.Now()
				
				// Mixed operations
				switch i % 4 {
				case 0, 1: // 50% adds
					order := &orderbook.Order{
						ID:        startID + uint64(i),
						Symbol:    "BTC/USD",
						Side:      orderbook.OrderSide(i % 2),
						Price:     50000 + float64(i%100),
						Quantity:  1.0,
						Timestamp: time.Now(),
					}
					ob.AddOrder(order)
				case 2: // 25% cancels
					ob.CancelOrder(startID + uint64(i-1))
				case 3: // 25% matches
					ob.MatchOrders()
				}
				
				latency := time.Since(opStart)
				atomic.AddInt64(&totalLatency, int64(latency))
				atomic.AddInt64(&totalOps, 1)
			}
		}(w)
	}

	wg.Wait()
	duration := time.Since(start)
	fmt.Println("✓")

	finalOps := atomic.LoadInt64(&totalOps)
	throughput := float64(finalOps) / duration.Seconds()
	avgLatency := time.Duration(0)
	if finalOps > 0 {
		avgLatency = time.Duration(atomic.LoadInt64(&totalLatency) / finalOps)
	}

	fmt.Printf("  ✅ Results:\n")
	fmt.Printf("     Operations:  %d\n", finalOps)
	fmt.Printf("     Duration:    %v\n", duration)
	fmt.Printf("     Throughput:  %.0f ops/sec\n", throughput)
	fmt.Printf("     Avg Latency: %v\n", avgLatency)

	return BenchResult{
		Name:       name,
		Ops:        finalOps,
		Duration:   duration,
		Throughput: throughput,
		Latency:    avgLatency,
	}
}

func printComparison(results []BenchResult) {
	if len(results) != 2 {
		return
	}
	
	fmt.Println("\n" + repeatStr("=", 60))
	fmt.Println("📊 PERFORMANCE COMPARISON")
	fmt.Println(repeatStr("=", 60))
	
	goResult := results[0]
	cppResult := results[1]
	
	speedup := cppResult.Throughput / goResult.Throughput
	latencyRatio := float64(goResult.Latency) / float64(cppResult.Latency)
	
	fmt.Printf("Throughput:  C++ is %.2fx %s than Go\n", 
		speedup, getFasterSlower(speedup))
	fmt.Printf("Latency:     C++ is %.2fx %s than Go\n", 
		latencyRatio, getFasterSlower(latencyRatio))
	
	fmt.Println(repeatStr("=", 60))
	
	if speedup > 2.0 {
		fmt.Println("🚀 C++ provides significant performance benefits!")
	} else if speedup > 1.2 {
		fmt.Println("📈 C++ provides moderate performance benefits")
	} else {
		fmt.Println("⚖️  Performance is comparable - use Go for simplicity")
	}
}

func getFasterSlower(ratio float64) string {
	if ratio >= 1.0 {
		return "faster"
	}
	return "slower"
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}