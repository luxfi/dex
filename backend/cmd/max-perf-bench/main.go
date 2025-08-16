package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/luxexchange/engine/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	target     = flag.String("target", "go", "Target engine: go, hybrid, cpp, ts")
	duration   = flag.Duration("duration", 60*time.Second, "Test duration")
	warmup     = flag.Duration("warmup", 5*time.Second, "Warmup duration")
	maxTraders = flag.Int("max-traders", 10000, "Maximum traders to test")
	step       = flag.Int("step", 500, "Trader increment step")
)

type BenchResult struct {
	Engine       string
	Traders      int
	Duration     time.Duration
	Orders       int64
	Throughput   float64
	AvgLatency   time.Duration
	P50Latency   time.Duration
	P95Latency   time.Duration
	P99Latency   time.Duration
	MaxLatency   time.Duration
	ErrorRate    float64
	CPUUsage     float64
	MemoryMB     float64
}

type LatencyTracker struct {
	samples []int64
	mu      sync.Mutex
}

func (lt *LatencyTracker) Add(ns int64) {
	lt.mu.Lock()
	lt.samples = append(lt.samples, ns)
	lt.mu.Unlock()
}

func (lt *LatencyTracker) GetPercentile(p float64) time.Duration {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	
	if len(lt.samples) == 0 {
		return 0
	}
	
	// Simple percentile calculation (not optimal but works)
	index := int(float64(len(lt.samples)) * p / 100.0)
	if index >= len(lt.samples) {
		index = len(lt.samples) - 1
	}
	
	return time.Duration(lt.samples[index])
}

func getEndpoint(engine string) string {
	switch engine {
	case "go":
		return "localhost:50051"
	case "hybrid":
		return "localhost:50052"
	case "cpp":
		return "localhost:50054"
	case "ts":
		return "localhost:50053"
	default:
		return "localhost:50051"
	}
}

func runBenchmark(engine string, numTraders int, testDuration time.Duration) *BenchResult {
	endpoint := getEndpoint(engine)
	
	// Create connection pool
	connPool := make([]*grpc.ClientConn, 20)
	for i := range connPool {
		conn, err := grpc.Dial(endpoint,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)),
		)
		if err != nil {
			log.Printf("Failed to connect to %s: %v", endpoint, err)
			return nil
		}
		defer conn.Close()
		connPool[i] = conn
	}
	
	// Metrics
	var totalOrders int64
	var totalErrors int64
	latencyTracker := &LatencyTracker{}
	
	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), testDuration)
	defer cancel()
	
	// Start time
	startTime := time.Now()
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	startMem := memStats.Alloc
	
	// Worker function
	worker := func(workerID int, wg *sync.WaitGroup) {
		defer wg.Done()
		
		client := pb.NewEngineServiceClient(connPool[workerID%len(connPool)])
		ticker := time.NewTicker(time.Millisecond * 10) // 100 orders/sec per worker
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				start := time.Now()
				_, err := client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
					Symbol:        "BTC-USD",
					Side:          pb.OrderSide(workerID%2 + 1),
					Type:          pb.OrderType_ORDER_TYPE_LIMIT,
					Quantity:      0.1,
					Price:         40000.0 + float64(workerID%100),
					ClientOrderId: fmt.Sprintf("bench-%d-%d", workerID, time.Now().UnixNano()),
				})
				
				latency := time.Since(start).Nanoseconds()
				latencyTracker.Add(latency)
				
				if err != nil {
					atomic.AddInt64(&totalErrors, 1)
				} else {
					atomic.AddInt64(&totalOrders, 1)
				}
			}
		}
	}
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < numTraders; i++ {
		wg.Add(1)
		go worker(i, &wg)
	}
	
	// Wait for completion
	wg.Wait()
	
	// Calculate results
	elapsed := time.Since(startTime)
	orders := atomic.LoadInt64(&totalOrders)
	errors := atomic.LoadInt64(&totalErrors)
	
	runtime.ReadMemStats(&memStats)
	memUsed := float64(memStats.Alloc-startMem) / 1024 / 1024
	
	// Calculate latencies
	avgLatency := time.Duration(0)
	if len(latencyTracker.samples) > 0 {
		var sum int64
		for _, s := range latencyTracker.samples {
			sum += s
		}
		avgLatency = time.Duration(sum / int64(len(latencyTracker.samples)))
	}
	
	return &BenchResult{
		Engine:     engine,
		Traders:    numTraders,
		Duration:   elapsed,
		Orders:     orders,
		Throughput: float64(orders) / elapsed.Seconds(),
		AvgLatency: avgLatency,
		P50Latency: latencyTracker.GetPercentile(50),
		P95Latency: latencyTracker.GetPercentile(95),
		P99Latency: latencyTracker.GetPercentile(99),
		MaxLatency: latencyTracker.GetPercentile(100),
		ErrorRate:  float64(errors) / float64(orders+errors) * 100,
		MemoryMB:   memUsed,
	}
}

func findMaxThroughput(engine string) *BenchResult {
	fmt.Printf("\n=== Finding Maximum Throughput for %s Engine ===\n", engine)
	
	var bestResult *BenchResult
	maxThroughput := 0.0
	
	// Binary search for maximum sustainable throughput
	low := 100
	high := *maxTraders
	
	for low <= high {
		traders := (low + high) / 2
		
		fmt.Printf("Testing with %d traders...\n", traders)
		
		// Warmup
		fmt.Printf("  Warming up for %v...\n", *warmup)
		_ = runBenchmark(engine, traders, *warmup)
		
		// Actual test
		fmt.Printf("  Running benchmark for %v...\n", *duration)
		result := runBenchmark(engine, traders, *duration)
		
		if result == nil {
			fmt.Printf("  Failed to connect\n")
			high = traders - 1
			continue
		}
		
		fmt.Printf("  Throughput: %.0f orders/sec, Error Rate: %.2f%%, Avg Latency: %v\n",
			result.Throughput, result.ErrorRate, result.AvgLatency)
		
		// Check if this is sustainable (low error rate, reasonable latency)
		if result.ErrorRate < 1.0 && result.P99Latency < 100*time.Millisecond {
			if result.Throughput > maxThroughput {
				maxThroughput = result.Throughput
				bestResult = result
			}
			low = traders + *step
		} else {
			high = traders - *step
		}
	}
	
	return bestResult
}

func main() {
	flag.Parse()
	
	fmt.Println("=== LX ENGINE MAXIMUM PERFORMANCE BENCHMARK ===")
	fmt.Printf("Target: %s\n", *target)
	fmt.Printf("Duration: %v per test\n", *duration)
	fmt.Printf("Max Traders: %d\n", *maxTraders)
	fmt.Println()
	
	engines := []string{*target}
	if *target == "all" {
		engines = []string{"go", "hybrid", "cpp", "ts"}
	}
	
	results := make([]*BenchResult, 0)
	
	for _, engine := range engines {
		result := findMaxThroughput(engine)
		if result != nil {
			results = append(results, result)
		}
	}
	
	// Print comparison table
	fmt.Println("\n=== MAXIMUM PERFORMANCE COMPARISON ===")
	fmt.Println("┌─────────┬──────────┬────────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────┐")
	fmt.Println("│ Engine  │ Traders  │ Throughput │ Avg Latency  │ P50 Latency  │ P95 Latency  │ P99 Latency  │ Error Rate │")
	fmt.Println("├─────────┼──────────┼────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┤")
	
	for _, r := range results {
		fmt.Printf("│ %-7s │ %8d │ %10.0f │ %12v │ %12v │ %12v │ %12v │ %9.2f%% │\n",
			r.Engine, r.Traders, r.Throughput, 
			r.AvgLatency, r.P50Latency, r.P95Latency, r.P99Latency,
			r.ErrorRate)
	}
	fmt.Println("└─────────┴──────────┴────────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────┘")
	
	// Find winner
	if len(results) > 1 {
		var winner *BenchResult
		for _, r := range results {
			if winner == nil || r.Throughput > winner.Throughput {
				winner = r
			}
		}
		
		fmt.Printf("\n🏆 WINNER: %s Engine with %.0f orders/sec using %d traders\n",
			winner.Engine, winner.Throughput, winner.Traders)
		
		// Calculate relative performance
		fmt.Println("\n📊 Relative Performance:")
		for _, r := range results {
			pct := r.Throughput / winner.Throughput * 100
			fmt.Printf("  %s: %.1f%% of maximum\n", r.Engine, pct)
		}
	}
	
	// System limits analysis
	fmt.Println("\n🔬 System Analysis:")
	fmt.Printf("  CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("  GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("  Memory Allocated: %.2f MB\n", float64(m.Alloc)/1024/1024)
	fmt.Printf("  Total Memory: %.2f MB\n", float64(m.Sys)/1024/1024)
	
	// Theoretical maximum
	if len(results) > 0 {
		best := results[0]
		theoretical := float64(best.Traders) * 100 // 100 orders/sec per trader
		efficiency := best.Throughput / theoretical * 100
		fmt.Printf("\n  Theoretical Max: %.0f orders/sec\n", theoretical)
		fmt.Printf("  Achieved: %.0f orders/sec (%.1f%% efficiency)\n", best.Throughput, efficiency)
		
		// Bottleneck analysis
		fmt.Println("\n🔍 Bottleneck Analysis:")
		if best.P99Latency > 50*time.Millisecond {
			fmt.Println("  ⚠️  High P99 latency indicates CPU or lock contention")
		}
		if best.ErrorRate > 0.5 {
			fmt.Println("  ⚠️  Error rate suggests connection pool or backpressure issues")
		}
		if efficiency < 80 {
			fmt.Println("  ⚠️  Low efficiency indicates system bottleneck")
		}
		if best.AvgLatency > 20*time.Millisecond {
			fmt.Println("  ⚠️  High average latency suggests GC pressure or I/O wait")
		}
	}
}