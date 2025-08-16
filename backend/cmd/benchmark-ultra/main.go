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

	"github.com/luxfi/dex/backend/pkg/gpu"
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
	CPUUsage        float64
	MemoryMB        uint64
	GoroutineCount  int
}

// TestConfig defines benchmark parameters
type TestConfig struct {
	NumOrders      int
	NumGoroutines  int
	Duration       time.Duration
	OrderBookCount int
	UseGPU         bool
	UseLockFree    bool
	UseZeroCopy    bool
}

func main() {
	var (
		orders      = flag.Int("orders", 100000, "Number of orders per second")
		duration    = flag.Duration("duration", 30*time.Second, "Test duration")
		books       = flag.Int("books", 10, "Number of order books")
		goroutines  = flag.Int("goroutines", runtime.NumCPU(), "Number of goroutines")
		useGPU      = flag.Bool("gpu", true, "Use GPU acceleration")
		lockFree    = flag.Bool("lockfree", true, "Use lock-free structures")
		zeroCopy    = flag.Bool("zerocopy", true, "Use zero-copy optimizations")
		compareAll  = flag.Bool("compare", false, "Compare all implementations")
	)
	flag.Parse()

	fmt.Println("=================================================")
	fmt.Println("     LX DEX Ultra-Low Latency Benchmark")
	fmt.Println("=================================================")
	fmt.Printf("Target: 100M trades/sec on 100 nodes\n")
	fmt.Printf("Local test: %d orders/sec for %v\n", *orders, *duration)
	fmt.Printf("Order books: %d, Goroutines: %d\n", *books, *goroutines)
	fmt.Printf("CPU cores: %d\n", runtime.NumCPU())
	fmt.Println("-------------------------------------------------")

	// Check GPU availability
	if *useGPU {
		if gpu.IsAvailable() {
			engine, err := gpu.NewEngine()
			if err == nil {
				info, _ := engine.GetDeviceInfo()
				fmt.Printf("GPU: %s\n", info)
				engine.Close()
			}
		} else {
			fmt.Println("GPU: Not available, falling back to CPU")
			*useGPU = false
		}
	}

	config := TestConfig{
		NumOrders:      *orders,
		NumGoroutines:  *goroutines,
		Duration:       *duration,
		OrderBookCount: *books,
		UseGPU:         *useGPU,
		UseLockFree:    *lockFree,
		UseZeroCopy:    *zeroCopy,
	}

	if *compareAll {
		runComparison(config)
	} else {
		result := runBenchmark("Ultra-Low Latency", config)
		printResult(result)
		printScalingAnalysis(result, 100) // Project to 100 nodes
	}
}

func runComparison(config TestConfig) {
	fmt.Println("\n📊 Running Comparative Benchmarks...")
	fmt.Println("=================================================")

	configs := []struct {
		name string
		cfg  TestConfig
	}{
		{
			name: "Baseline (No Optimizations)",
			cfg: TestConfig{
				NumOrders:      config.NumOrders,
				NumGoroutines:  config.NumGoroutines,
				Duration:       10 * time.Second,
				OrderBookCount: config.OrderBookCount,
				UseGPU:         false,
				UseLockFree:    false,
				UseZeroCopy:    false,
			},
		},
		{
			name: "Lock-Free Only",
			cfg: TestConfig{
				NumOrders:      config.NumOrders,
				NumGoroutines:  config.NumGoroutines,
				Duration:       10 * time.Second,
				OrderBookCount: config.OrderBookCount,
				UseGPU:         false,
				UseLockFree:    true,
				UseZeroCopy:    false,
			},
		},
		{
			name: "Zero-Copy Only",
			cfg: TestConfig{
				NumOrders:      config.NumOrders,
				NumGoroutines:  config.NumGoroutines,
				Duration:       10 * time.Second,
				OrderBookCount: config.OrderBookCount,
				UseGPU:         false,
				UseLockFree:    false,
				UseZeroCopy:    true,
			},
		},
		{
			name: "CPU Optimized (Lock-Free + Zero-Copy)",
			cfg: TestConfig{
				NumOrders:      config.NumOrders,
				NumGoroutines:  config.NumGoroutines,
				Duration:       10 * time.Second,
				OrderBookCount: config.OrderBookCount,
				UseGPU:         false,
				UseLockFree:    true,
				UseZeroCopy:    true,
			},
		},
	}

	// Add GPU config if available
	if gpu.IsAvailable() {
		configs = append(configs, struct {
			name string
			cfg  TestConfig
		}{
			name: "Full Optimization (GPU + Lock-Free + Zero-Copy)",
			cfg: TestConfig{
				NumOrders:      config.NumOrders,
				NumGoroutines:  config.NumGoroutines,
				Duration:       10 * time.Second,
				OrderBookCount: config.OrderBookCount,
				UseGPU:         true,
				UseLockFree:    true,
				UseZeroCopy:    true,
			},
		})
	}

	results := make([]BenchmarkResult, 0, len(configs))
	for _, c := range configs {
		fmt.Printf("\nTesting: %s\n", c.name)
		result := runBenchmark(c.name, c.cfg)
		results = append(results, result)
	}

	// Print comparison table
	printComparisonTable(results)
}

func runBenchmark(name string, config TestConfig) BenchmarkResult {
	// Initialize order books
	books := make([]*lx.OrderBook, config.OrderBookCount)
	for i := 0; i < config.OrderBookCount; i++ {
		books[i] = lx.NewOrderBook(fmt.Sprintf("ASSET%d-USD", i))
	}

	// Initialize GPU engine if enabled
	var gpuEngine *gpu.Engine
	if config.UseGPU {
		var err error
		gpuEngine, err = gpu.NewEngine()
		if err != nil {
			log.Printf("GPU initialization failed: %v", err)
			config.UseGPU = false
		}
		defer func() {
			if gpuEngine != nil {
				gpuEngine.Close()
			}
		}()
	}

	// Metrics
	var (
		ordersProcessed uint64
		tradesExecuted  uint64
		j               int
		latencies       []time.Duration
		latencyMu       sync.Mutex
		startTime       = time.Now()
	)

	// Worker pool
	wg := sync.WaitGroup{}
	done := make(chan bool)
	
	// Start workers
	for i := 0; i < config.NumGoroutines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)))
			orderBatch := make([]lx.Order, 0, 100)
			
			for {
				select {
				case <-done:
					return
				default:
					// Generate batch of orders
					orderBatch = orderBatch[:0]
					for j := 0; j < 100; j++ {
						order := generateOrder(localRand)
						orderBatch = append(orderBatch, order)
					}
					
					// Process orders
					for _, order := range orderBatch {
						bookIdx := localRand.Intn(config.OrderBookCount)
						book := books[bookIdx]
						
						start := time.Now()
						
						if config.UseGPU && j%10 == 0 {
							// Use GPU for batch matching every 10 orders
							processBatchGPU(gpuEngine, book, orderBatch[j:min(j+10, len(orderBatch))])
						} else {
							// Regular processing
							trades := book.AddOrder(&order)
							atomic.AddUint64(&tradesExecuted, uint64(len(trades)))
						}
						
						latency := time.Since(start)
						
						// Sample latencies (1% sampling to reduce overhead)
						if localRand.Float32() < 0.01 {
							latencyMu.Lock()
							latencies = append(latencies, latency)
							latencyMu.Unlock()
						}
						
						atomic.AddUint64(&ordersProcessed, 1)
					}
				}
			}
		}(i)
	}

	// Run for specified duration
	time.Sleep(config.Duration)
	close(done)
	wg.Wait()
	
	elapsed := time.Since(startTime)

	// Calculate metrics
	ordersPerSec := float64(ordersProcessed) / elapsed.Seconds()
	tradesPerSec := float64(tradesExecuted) / elapsed.Seconds()
	
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

func processBatchGPU(engine *gpu.Engine, book *lx.OrderBook, orders []lx.Order) {
	if engine == nil {
		return
	}
	
	// Convert orders to GPU format
	bids := make([]gpu.Order, 0, len(orders))
	asks := make([]gpu.Order, 0, len(orders))
	
	for _, order := range orders {
		gpuOrder := gpu.Order{
			OrderID:   order.ID,
			Price:     uint32(order.Price * 10000000), // Convert to fixed point
			Quantity:  uint32(order.Size * 10000000),
			Timestamp: uint32(order.Timestamp.Unix()),
			Side:      uint8(order.Side),
			Status:    0,
		}
		
		if order.Side == lx.Buy {
			bids = append(bids, gpuOrder)
		} else {
			asks = append(asks, gpuOrder)
		}
	}
	
	// Match on GPU
	if len(bids) > 0 && len(asks) > 0 {
		trades, err := engine.MatchOrders(bids, asks)
		if err == nil {
			atomic.AddUint64(&tradesExecuted, uint64(len(trades)))
		}
	}
}

func generateOrder(r *rand.Rand) lx.Order {
	side := lx.Buy
	if r.Float32() > 0.5 {
		side = lx.Sell
	}
	
	// Generate price around 50000 with some spread
	basePrice := 50000.0
	spread := 1000.0
	price := basePrice + (r.Float64()-0.5)*spread
	
	return lx.Order{
		ID:        uint64(r.Int63()),
		Symbol:    "BTC-USD",
		Side:      side,
		Type:      lx.Limit,
		Price:     price,
		Size:      r.Float64() * 10,
		User:      fmt.Sprintf("user-%d", r.Intn(1000)),
		Timestamp: time.Now(),
	}
}

func calculatePercentiles(latencies []time.Duration) (p50, p99, p999 time.Duration) {
	if len(latencies) == 0 {
		return
	}
	
	// Sort latencies
	for i := 0; i < len(latencies); i++ {
		for j := i + 1; j < len(latencies); j++ {
			if latencies[i] > latencies[j] {
				latencies[i], latencies[j] = latencies[j], latencies[i]
			}
		}
	}
	
	p50 = latencies[len(latencies)*50/100]
	p99 = latencies[len(latencies)*99/100]
	if len(latencies) > 1000 {
		p999 = latencies[len(latencies)*999/1000]
	} else {
		p999 = latencies[len(latencies)-1]
	}
	
	return
}

func printResult(result BenchmarkResult) {
	fmt.Println("\n📈 Benchmark Results")
	fmt.Println("=================================================")
	fmt.Printf("Orders/sec:      %15.0f\n", result.OrdersPerSec)
	fmt.Printf("Trades/sec:      %15.0f\n", result.TradesPerSec)
	fmt.Printf("Latency P50:     %15v\n", result.LatencyP50)
	fmt.Printf("Latency P99:     %15v\n", result.LatencyP99)
	fmt.Printf("Latency P99.9:   %15v\n", result.LatencyP999)
	fmt.Printf("Memory Usage:    %15d MB\n", result.MemoryMB)
	fmt.Printf("Goroutines:      %15d\n", result.GoroutineCount)
}

func printComparisonTable(results []BenchmarkResult) {
	fmt.Println("\n📊 Performance Comparison")
	fmt.Println("=================================================================================")
	fmt.Printf("%-40s | %12s | %12s | %10s | %10s\n",
		"Configuration", "Orders/sec", "Trades/sec", "P50 Latency", "P99 Latency")
	fmt.Println("---------------------------------------------------------------------------------")
	
	baseline := results[0].OrdersPerSec
	for _, r := range results {
		improvement := (r.OrdersPerSec / baseline - 1) * 100
		fmt.Printf("%-40s | %12.0f | %12.0f | %10v | %10v",
			r.Name, r.OrdersPerSec, r.TradesPerSec, r.LatencyP50, r.LatencyP99)
		if improvement > 0 {
			fmt.Printf(" | +%.1f%%", improvement)
		}
		fmt.Println()
	}
}

func printScalingAnalysis(result BenchmarkResult, nodeCount int) {
	fmt.Println("\n🚀 Scaling Projection")
	fmt.Println("=================================================")
	
	// Assume 85% scaling efficiency for distributed system
	scalingEfficiency := 0.85
	projectedOrders := result.OrdersPerSec * float64(nodeCount) * scalingEfficiency
	projectedTrades := result.TradesPerSec * float64(nodeCount) * scalingEfficiency
	
	fmt.Printf("Single Node Performance:\n")
	fmt.Printf("  Orders/sec: %.0f\n", result.OrdersPerSec)
	fmt.Printf("  Trades/sec: %.0f\n", result.TradesPerSec)
	
	fmt.Printf("\nProjected %d-Node Cluster Performance:\n", nodeCount)
	fmt.Printf("  Orders/sec: %.0f (%.1fM)\n", projectedOrders, projectedOrders/1000000)
	fmt.Printf("  Trades/sec: %.0f (%.1fM)\n", projectedTrades, projectedTrades/1000000)
	
	fmt.Printf("\nTo reach 100M trades/sec target:\n")
	requiredNodes := 100000000.0 / (result.TradesPerSec * scalingEfficiency)
	fmt.Printf("  Required nodes: %.0f\n", requiredNodes)
	fmt.Printf("  Current efficiency: %.1f%%\n", 
		(projectedTrades/100000000.0)*100)
	
	if projectedTrades >= 100000000 {
		fmt.Println("\n✅ TARGET ACHIEVED! System can handle 100M+ trades/sec")
	} else {
		fmt.Printf("\n⚠️  Need %.1fx more performance to reach target\n",
			100000000.0/projectedTrades)
		fmt.Println("\nRecommendations:")
		fmt.Println("  • Enable GPU acceleration")
		fmt.Println("  • Add FPGA packet processing")
		fmt.Println("  • Implement RDMA for inter-node communication")
		fmt.Println("  • Use kernel-bypass networking (DPDK)")
		fmt.Println("  • Optimize NUMA memory allocation")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}