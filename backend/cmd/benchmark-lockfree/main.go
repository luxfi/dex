package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
)

// BenchmarkResult holds performance metrics
type BenchmarkResult struct {
	Name            string
	OrdersPerSec    float64
	TradesPerSec    float64
	LatencyP50      time.Duration
	LatencyP99      time.Duration
	LatencyP999     time.Duration
	MemoryMB        uint64
	GoroutineCount  int
}

func main() {
	var (
		orders     = flag.Int("orders", 100000, "Number of orders to process")
		duration   = flag.Duration("duration", 10*time.Second, "Test duration")
		books      = flag.Int("books", 10, "Number of order books")
		goroutines = flag.Int("goroutines", runtime.NumCPU()*2, "Number of goroutines")
	)
	flag.Parse()

	fmt.Println("=================================================")
	fmt.Println("     LX DEX Lock-Free Performance Benchmark")
	fmt.Println("=================================================")
	fmt.Printf("Target: Path to 100M trades/sec\n")
	fmt.Printf("Orders: %d, Duration: %v\n", *orders, *duration)
	fmt.Printf("Order books: %d, Goroutines: %d\n", *books, *goroutines)
	fmt.Printf("CPU cores: %d, GOMAXPROCS: %d\n", runtime.NumCPU(), runtime.GOMAXPROCS(0))
	fmt.Println("-------------------------------------------------")

	// Run benchmark series
	results := []BenchmarkResult{}
	
	// Test different configurations
	configs := []struct {
		name       string
		goroutines int
		books      int
	}{
		{"Single-threaded", 1, 1},
		{"Multi-threaded (cores)", runtime.NumCPU(), *books},
		{"Multi-threaded (2x cores)", runtime.NumCPU() * 2, *books},
		{"Hyper-threaded (4x cores)", runtime.NumCPU() * 4, *books},
		{"Max concurrency", *goroutines, *books},
	}

	for _, cfg := range configs {
		fmt.Printf("\nTesting: %s (%d goroutines, %d books)\n", 
			cfg.name, cfg.goroutines, cfg.books)
		result := runBenchmark(cfg.name, *orders, *duration, cfg.goroutines, cfg.books)
		results = append(results, result)
		printResult(result)
	}

	// Print comparison
	printComparison(results)
	
	// Project to cluster scale
	best := results[0]
	for _, r := range results {
		if r.OrdersPerSec > best.OrdersPerSec {
			best = r
		}
	}
	projectToCluster(best)
}

func runBenchmark(name string, numOrders int, duration time.Duration, numGoroutines int, numBooks int) BenchmarkResult {
	// Initialize order books
	books := make([]*lx.OrderBook, numBooks)
	for i := 0; i < numBooks; i++ {
		books[i] = lx.NewOrderBook(fmt.Sprintf("ASSET%d-USD", i))
	}

	// Metrics
	var (
		ordersProcessed atomic.Uint64
		tradesExecuted  atomic.Uint64
		latencies       []time.Duration
		latencyMu       sync.Mutex
	)

	// Create order queue
	orderQueue := make(chan lx.Order, numOrders)
	
	// Pre-generate orders
	log.Printf("Generating %d orders...", numOrders)
	orders := generateOrders(numOrders)
	
	// Start workers
	wg := sync.WaitGroup{}
	startTime := time.Now()
	done := make(chan bool)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)))
			
			for {
				select {
				case <-done:
					return
				case order := <-orderQueue:
					// Select random book
					bookIdx := localRand.Intn(numBooks)
					book := books[bookIdx]
					
					// Process order with timing
					start := time.Now()
					tradeCount := book.AddOrder(&order)
					latency := time.Since(start)
					
					// Update metrics
					ordersProcessed.Add(1)
					tradesExecuted.Add(tradeCount)
					
					// Sample latencies (1% to reduce overhead)
					if localRand.Float32() < 0.01 {
						latencyMu.Lock()
						latencies = append(latencies, latency)
						latencyMu.Unlock()
					}
				default:
					// Queue empty, check if done
					if time.Since(startTime) > duration {
						return
					}
					time.Sleep(time.Microsecond)
				}
			}
		}(i)
	}

	// Feed orders continuously
	go func() {
		idx := 0
		for time.Since(startTime) < duration {
			orderQueue <- orders[idx%len(orders)]
			idx++
		}
		close(done)
	}()

	// Wait for completion
	wg.Wait()
	elapsed := time.Since(startTime)

	// Calculate metrics
	totalOrders := ordersProcessed.Load()
	totalTrades := tradesExecuted.Load()
	ordersPerSec := float64(totalOrders) / elapsed.Seconds()
	tradesPerSec := float64(totalTrades) / elapsed.Seconds()

	// Calculate latency percentiles
	latencyMu.Lock()
	p50, p99, p999 := calculatePercentiles(latencies)
	latencyMu.Unlock()

	// Get memory stats
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return BenchmarkResult{
		Name:           name,
		OrdersPerSec:   ordersPerSec,
		TradesPerSec:   tradesPerSec,
		LatencyP50:     p50,
		LatencyP99:     p99,
		LatencyP999:    p999,
		MemoryMB:       memStats.Alloc / 1024 / 1024,
		GoroutineCount: runtime.NumGoroutine(),
	}
}

func generateOrders(count int) []lx.Order {
	orders := make([]lx.Order, count)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	for i := 0; i < count; i++ {
		side := lx.Buy
		if r.Float32() > 0.5 {
			side = lx.Sell
		}
		
		// Generate realistic price distribution
		basePrice := 50000.0
		spread := 1000.0
		price := basePrice + (r.Float64()-0.5)*spread
		
		// Add some market orders for immediate matching
		orderType := lx.Limit
		if r.Float32() < 0.1 { // 10% market orders
			orderType = lx.Market
			price = 0
		}
		
		orders[i] = lx.Order{
			ID:        uint64(i),
			Symbol:    "BTC-USD",
			Side:      side,
			Type:      orderType,
			Price:     price,
			Size:      r.Float64() * 10,
			User:      fmt.Sprintf("user-%d", r.Intn(1000)),
			Timestamp: time.Now(),
		}
	}
	
	return orders
}

func calculatePercentiles(latencies []time.Duration) (p50, p99, p999 time.Duration) {
	if len(latencies) == 0 {
		return
	}
	
	// Simple bubble sort for small arrays
	n := len(latencies)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if latencies[j] > latencies[j+1] {
				latencies[j], latencies[j+1] = latencies[j+1], latencies[j]
			}
		}
	}
	
	p50 = latencies[n*50/100]
	if n > 99 {
		p99 = latencies[n*99/100]
	} else {
		p99 = latencies[n-1]
	}
	if n > 999 {
		p999 = latencies[n*999/1000]
	} else {
		p999 = latencies[n-1]
	}
	
	return
}

func printResult(result BenchmarkResult) {
	fmt.Printf("  Orders/sec: %12.0f | Trades/sec: %12.0f | P50: %8v | P99: %8v\n",
		result.OrdersPerSec, result.TradesPerSec, result.LatencyP50, result.LatencyP99)
}

func printComparison(results []BenchmarkResult) {
	fmt.Println("\n📊 Performance Comparison")
	fmt.Println("================================================================================")
	fmt.Printf("%-30s | %12s | %12s | %10s | %10s | %8s\n",
		"Configuration", "Orders/sec", "Trades/sec", "P50", "P99", "Memory")
	fmt.Println("--------------------------------------------------------------------------------")
	
	baseline := results[0].OrdersPerSec
	for _, r := range results {
		improvement := (r.OrdersPerSec / baseline - 1) * 100
		fmt.Printf("%-30s | %12.0f | %12.0f | %10v | %10v | %6dMB",
			r.Name, r.OrdersPerSec, r.TradesPerSec, 
			r.LatencyP50, r.LatencyP99, r.MemoryMB)
		if improvement > 0 {
			fmt.Printf(" | +%.1f%%", improvement)
		}
		fmt.Println()
	}
}

func projectToCluster(best BenchmarkResult) {
	fmt.Println("\n🚀 Scaling Projection to 100M trades/sec")
	fmt.Println("================================================================================")
	
	// Current single-node performance
	fmt.Printf("Single Node Best Performance:\n")
	fmt.Printf("  Orders/sec: %12.0f\n", best.OrdersPerSec)
	fmt.Printf("  Trades/sec: %12.0f\n", best.TradesPerSec)
	fmt.Printf("  Latency P50: %v, P99: %v\n", best.LatencyP50, best.LatencyP99)
	
	// Calculate requirements for different scenarios
	scenarios := []struct {
		name       string
		efficiency float64
		overhead   float64
	}{
		{"Ideal (100% efficiency)", 1.0, 0.0},
		{"Realistic (85% efficiency)", 0.85, 0.15},
		{"Conservative (70% efficiency)", 0.70, 0.30},
	}
	
	fmt.Printf("\nNodes Required for 100M trades/sec:\n")
	for _, s := range scenarios {
		nodesRequired := 100_000_000 / (best.TradesPerSec * s.efficiency)
		fmt.Printf("  %s: %.0f nodes\n", s.name, nodesRequired)
	}
	
	// Project with current optimizations
	fmt.Printf("\nWith Planned Optimizations:\n")
	optimizations := []struct {
		name     string
		speedup  float64
	}{
		{"+ Lock-free DAG", 2.0},
		{"+ DPDK networking", 3.0},
		{"+ RDMA replication", 1.5},
		{"+ GPU matching", 5.0},
		{"+ FPGA filtering", 2.0},
	}
	
	projectedPerf := best.TradesPerSec
	for _, opt := range optimizations {
		projectedPerf *= opt.speedup
		nodesFor100M := 100_000_000 / (projectedPerf * 0.85)
		fmt.Printf("  %s: %.0fx speedup → %.0f nodes needed\n", 
			opt.name, projectedPerf/best.TradesPerSec, nodesFor100M)
	}
	
	// Final projection
	totalSpeedup := projectedPerf / best.TradesPerSec
	finalNodes := 100_000_000 / (projectedPerf * 0.85)
	
	fmt.Printf("\n📈 Final Projection:\n")
	fmt.Printf("  Total speedup: %.0fx\n", totalSpeedup)
	fmt.Printf("  Per-node performance: %.0f trades/sec\n", projectedPerf)
	fmt.Printf("  Nodes for 100M trades/sec: %.0f\n", finalNodes)
	
	if finalNodes <= 100 {
		fmt.Println("\n✅ TARGET ACHIEVABLE with 100 nodes or less!")
	} else {
		fmt.Printf("\n⚠️  Need %.0f nodes (%.0f more than target)\n", 
			finalNodes, finalNodes-100)
		fmt.Println("\nAdditional optimizations needed:")
		fmt.Println("  • Implement sharding across order books")
		fmt.Println("  • Use hardware timestamping")
		fmt.Println("  • Optimize memory allocation patterns")
		fmt.Println("  • Implement batched matching algorithms")
	}
}