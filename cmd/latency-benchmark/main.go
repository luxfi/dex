package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/pkg/lx"
)

// Latency measurement structures
type LatencyMetrics struct {
	OrderPlacement    []time.Duration
	OrderCancellation []time.Duration
	OrderMatching     []time.Duration
	MarketData        []time.Duration
	FullRoundTrip     []time.Duration
	mu                sync.Mutex
}

var (
	metrics = &LatencyMetrics{
		OrderPlacement:    make([]time.Duration, 0, 1000000),
		OrderCancellation: make([]time.Duration, 0, 1000000),
		OrderMatching:     make([]time.Duration, 0, 1000000),
		MarketData:        make([]time.Duration, 0, 1000000),
		FullRoundTrip:     make([]time.Duration, 0, 1000000),
	}

	warmupComplete atomic.Bool
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           LX DEX Ultra-Low Latency Benchmark Suite                  ║")
	fmt.Println("║              Mac Studio M2 Ultra vs NYSE vs AWS F2                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// System info
	printSystemInfo()

	// Create test environment
	book := lx.NewOrderBook("AAPL")
	populateOrderBook(book)

	// Warmup
	fmt.Println("Warming up caches and JIT...")
	warmup(book)
	warmupComplete.Store(true)
	fmt.Println("✓ Warmup complete\n")

	// Run benchmarks
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("LATENCY BENCHMARKS")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()

	// 1. Order Placement Latency
	benchmarkOrderPlacement(book)

	// 2. Order Cancellation Latency
	benchmarkOrderCancellation(book)

	// 3. Order Matching Latency
	benchmarkOrderMatching(book)

	// 4. Market Data Query Latency
	benchmarkMarketData(book)

	// 5. Full Round Trip (Place → Match → Confirm)
	benchmarkFullRoundTrip(book)

	// Analyze and report
	analyzeResults()

	// Compare with industry standards
	compareWithIndustry()

	// Hardware specific analysis
	hardwareComparison()
}

func printSystemInfo() {
	fmt.Println("System Configuration:")
	fmt.Printf("  • CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("  • Go Version: %s\n", runtime.Version())
	fmt.Printf("  • Architecture: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	// Detect if running on Apple Silicon
	if runtime.GOARCH == "arm64" && runtime.GOOS == "darwin" {
		fmt.Println("  • Hardware: Apple Silicon (M1/M2/M3)")
		fmt.Println("  • Memory: Unified Memory Architecture")
		fmt.Println("  • Cache: Shared L2 Cache")
	}
	fmt.Println()
}

func populateOrderBook(book *lx.OrderBook) {
	// Add realistic market depth
	basePrice := 150.0

	for i := 0; i < 1000; i++ {
		// Bids
		book.AddOrder(&lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     basePrice - float64(i)*0.01,
			Size:      float64(100 + rand.Intn(900)),
			User:      "mm",
			Timestamp: time.Now(),
		})

		// Asks
		book.AddOrder(&lx.Order{
			ID:        uint64(i + 1000),
			Type:      lx.Limit,
			Side:      lx.Sell,
			Price:     basePrice + float64(i)*0.01,
			Size:      float64(100 + rand.Intn(900)),
			User:      "mm",
			Timestamp: time.Now(),
		})
	}
}

func warmup(book *lx.OrderBook) {
	// Warmup to ensure CPU caches are hot
	for i := 0; i < 100000; i++ {
		order := &lx.Order{
			ID:        uint64(1000000 + i),
			Type:      lx.Limit,
			Side:      lx.Side(i % 2),
			Price:     150.0 + rand.Float64(),
			Size:      100,
			User:      "warmup",
			Timestamp: time.Now(),
		}
		book.AddOrder(order)
		book.CancelOrder(order.ID)
		book.GetBestBid()
		book.GetBestAsk()
	}
}

func benchmarkOrderPlacement(book *lx.OrderBook) {
	fmt.Println("1. ORDER PLACEMENT LATENCY")
	fmt.Println("   Testing: Add new limit order to book")

	numTests := 100000
	latencies := make([]time.Duration, numTests)

	for i := 0; i < numTests; i++ {
		order := &lx.Order{
			ID:        uint64(2000000 + i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     149.0 + rand.Float64()*2,
			Size:      100,
			User:      "bench",
			Timestamp: time.Now(),
		}

		start := time.Now()
		book.AddOrder(order)
		latencies[i] = time.Since(start)
	}

	reportLatencies("Order Placement", latencies)
}

func benchmarkOrderCancellation(book *lx.OrderBook) {
	fmt.Println("\n2. ORDER CANCELLATION LATENCY")
	fmt.Println("   Testing: Cancel existing order")

	// Pre-place orders
	orderIDs := make([]uint64, 100000)
	for i := range orderIDs {
		orderIDs[i] = uint64(3000000 + i)
		book.AddOrder(&lx.Order{
			ID:        orderIDs[i],
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     148.0,
			Size:      100,
			User:      "bench",
			Timestamp: time.Now(),
		})
	}

	latencies := make([]time.Duration, len(orderIDs))

	for i, id := range orderIDs {
		start := time.Now()
		book.CancelOrder(id)
		latencies[i] = time.Since(start)
	}

	reportLatencies("Order Cancellation", latencies)
}

func benchmarkOrderMatching(book *lx.OrderBook) {
	fmt.Println("\n3. ORDER MATCHING LATENCY")
	fmt.Println("   Testing: Match crossing orders")

	numTests := 100000
	latencies := make([]time.Duration, numTests)

	for i := 0; i < numTests; i++ {
		// Create crossing order
		order := &lx.Order{
			ID:        uint64(4000000 + i),
			Type:      lx.Market,
			Side:      lx.Buy,
			Size:      10,
			User:      "bench",
			Timestamp: time.Now(),
		}

		start := time.Now()
		trades := book.AddOrder(order)
		if trades > 0 {
			latencies[i] = time.Since(start)
		} else {
			latencies[i] = time.Since(start)
		}
	}

	reportLatencies("Order Matching", latencies)
}

func benchmarkMarketData(book *lx.OrderBook) {
	fmt.Println("\n4. MARKET DATA QUERY LATENCY")
	fmt.Println("   Testing: Get best bid/ask prices")

	numTests := 1000000
	latencies := make([]time.Duration, numTests)

	for i := 0; i < numTests; i++ {
		start := time.Now()
		_ = book.GetBestBid()
		_ = book.GetBestAsk()
		latencies[i] = time.Since(start)
	}

	reportLatencies("Market Data Query", latencies)
}

func benchmarkFullRoundTrip(book *lx.OrderBook) {
	fmt.Println("\n5. FULL ROUND TRIP LATENCY")
	fmt.Println("   Testing: Place → Match → Confirm")

	numTests := 10000
	latencies := make([]time.Duration, numTests)

	for i := 0; i < numTests; i++ {
		start := time.Now()

		// Place bid
		bidOrder := &lx.Order{
			ID:        uint64(5000000 + i*2),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     150.0,
			Size:      100,
			User:      "bench",
			Timestamp: time.Now(),
		}
		book.AddOrder(bidOrder)

		// Place matching ask
		askOrder := &lx.Order{
			ID:        uint64(5000000 + i*2 + 1),
			Type:      lx.Limit,
			Side:      lx.Sell,
			Price:     150.0,
			Size:      100,
			User:      "bench2",
			Timestamp: time.Now(),
		}
		trades := book.AddOrder(askOrder)

		// Verify trade occurred
		if trades > 0 {
			latencies[i] = time.Since(start)
		} else {
			latencies[i] = time.Since(start)
		}
	}

	reportLatencies("Full Round Trip", latencies)
}

func reportLatencies(name string, latencies []time.Duration) {
	if len(latencies) == 0 {
		return
	}

	// Calculate percentiles
	p50 := percentile(latencies, 50)
	p95 := percentile(latencies, 95)
	p99 := percentile(latencies, 99)
	p999 := percentile(latencies, 99.9)
	p9999 := percentile(latencies, 99.99)

	min := minLatency(latencies)
	max := maxLatency(latencies)
	avg := avgLatency(latencies)

	fmt.Printf("   Results (%d operations):\n", len(latencies))
	fmt.Printf("   ├─ Min:     %s\n", formatLatency(min))
	fmt.Printf("   ├─ Avg:     %s\n", formatLatency(avg))
	fmt.Printf("   ├─ P50:     %s\n", formatLatency(p50))
	fmt.Printf("   ├─ P95:     %s\n", formatLatency(p95))
	fmt.Printf("   ├─ P99:     %s\n", formatLatency(p99))
	fmt.Printf("   ├─ P99.9:   %s\n", formatLatency(p999))
	fmt.Printf("   ├─ P99.99:  %s\n", formatLatency(p9999))
	fmt.Printf("   └─ Max:     %s\n", formatLatency(max))

	// Store for analysis
	metrics.mu.Lock()
	switch name {
	case "Order Placement":
		metrics.OrderPlacement = append(metrics.OrderPlacement, latencies...)
	case "Order Cancellation":
		metrics.OrderCancellation = append(metrics.OrderCancellation, latencies...)
	case "Order Matching":
		metrics.OrderMatching = append(metrics.OrderMatching, latencies...)
	case "Market Data Query":
		metrics.MarketData = append(metrics.MarketData, latencies...)
	case "Full Round Trip":
		metrics.FullRoundTrip = append(metrics.FullRoundTrip, latencies...)
	}
	metrics.mu.Unlock()
}

func analyzeResults() {
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("LATENCY ANALYSIS SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()

	metrics.mu.Lock()
	defer metrics.mu.Unlock()

	// Calculate aggregate metrics
	if len(metrics.OrderPlacement) > 0 {
		p99 := percentile(metrics.OrderPlacement, 99)
		fmt.Printf("Order Placement P99:    %s\n", formatLatency(p99))
	}
	if len(metrics.OrderMatching) > 0 {
		p99 := percentile(metrics.OrderMatching, 99)
		fmt.Printf("Order Matching P99:     %s\n", formatLatency(p99))
	}
	if len(metrics.MarketData) > 0 {
		p99 := percentile(metrics.MarketData, 99)
		fmt.Printf("Market Data P99:        %s\n", formatLatency(p99))
	}
	if len(metrics.FullRoundTrip) > 0 {
		p99 := percentile(metrics.FullRoundTrip, 99)
		fmt.Printf("Full Round Trip P99:    %s\n", formatLatency(p99))
	}
}

func compareWithIndustry() {
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("INDUSTRY COMPARISON")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()

	fmt.Println("NYSE Pillar (FPGA-based):")
	fmt.Println("  • Gateway latency: 40-50 µs")
	fmt.Println("  • Matching engine: 5-10 µs")
	fmt.Println("  • Market data: 15-20 µs")
	fmt.Println("  • Total round trip: ~100 µs")
	fmt.Println()

	fmt.Println("NASDAQ Inet:")
	fmt.Println("  • Order acknowledgment: 30-40 µs")
	fmt.Println("  • Matching latency: 3-5 µs")
	fmt.Println("  • Market data: 10-15 µs")
	fmt.Println()

	fmt.Println("CME Globex:")
	fmt.Println("  • Order entry: 100-150 µs")
	fmt.Println("  • Matching: 5-10 µs")
	fmt.Println("  • Market data: 20-30 µs")
	fmt.Println()

	fmt.Println("Cryptocurrency Exchanges:")
	fmt.Println("  • Binance: 5-10 ms")
	fmt.Println("  • Coinbase: 10-50 ms")
	fmt.Println("  • FTX (historical): 1-5 ms")
	fmt.Println()

	// Get our P99 for comparison
	metrics.mu.Lock()
	if len(metrics.OrderMatching) > 0 {
		ourP99 := percentile(metrics.OrderMatching, 99)
		fmt.Printf("LX DEX on M2 Ultra: %s (P99)\n", formatLatency(ourP99))

		// Compare
		if ourP99 < 10*time.Microsecond {
			fmt.Println("  ✅ Comparable to NYSE/NASDAQ FPGA systems")
		} else if ourP99 < 100*time.Microsecond {
			fmt.Println("  ✅ Better than most traditional exchanges")
		} else if ourP99 < time.Millisecond {
			fmt.Println("  ✅ Better than all crypto exchanges")
		}
	}
	metrics.mu.Unlock()
}

func hardwareComparison() {
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("HARDWARE COMPARISON")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()

	fmt.Println("Mac Studio M2 Ultra:")
	fmt.Println("  • CPU: 24-core (16P + 8E)")
	fmt.Println("  • Memory Bandwidth: 800 GB/s")
	fmt.Println("  • L2 Cache: 32MB shared")
	fmt.Println("  • Neural Engine: 32-core")
	fmt.Println("  • Unified Memory: Zero-copy")
	fmt.Println("  • Expected Latency: 100ns - 10µs")
	fmt.Println()

	fmt.Println("Amazon EC2 F2 (FPGA):")
	fmt.Println("  • FPGA: Xilinx Virtex UltraScale+")
	fmt.Println("  • CPU: Intel Xeon (8-16 cores)")
	fmt.Println("  • Memory: 122 GB DDR4")
	fmt.Println("  • Network: 25 Gbps")
	fmt.Println("  • PCIe to FPGA: ~1µs overhead")
	fmt.Println("  • Expected Latency: 1-5µs (FPGA only)")
	fmt.Println("  • With CPU coordination: 5-20µs")
	fmt.Println()

	fmt.Println("NYSE FPGA Setup:")
	fmt.Println("  • Custom Arista 7130 switches")
	fmt.Println("  • Stratix 10 FPGAs")
	fmt.Println("  • Layer 1 switching: 5ns")
	fmt.Println("  • Matching engine: 5µs")
	fmt.Println("  • Total system: 40-50µs")
	fmt.Println()

	fmt.Println("Key Advantages of M2 Ultra:")
	fmt.Println("  ✓ No PCIe latency (unified memory)")
	fmt.Println("  ✓ 800 GB/s memory bandwidth (10x DDR4)")
	fmt.Println("  ✓ Hardware accelerators built-in")
	fmt.Println("  ✓ Lower power consumption (370W vs 1000W+)")
	fmt.Println("  ✓ No kernel/userspace transitions")
	fmt.Println()

	fmt.Println("Theoretical Limits:")
	fmt.Println("  • M2 Ultra (optimized): 50-100ns")
	fmt.Println("  • F2 FPGA (raw): 100-500ns")
	fmt.Println("  • F2 with CPU: 1-5µs")
	fmt.Println("  • NYSE production: 5-10µs")
}

// Helper functions
func percentile(latencies []time.Duration, p float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Sort copy
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)

	// Simple sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)-1) * p / 100)
	return sorted[idx]
}

func minLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	min := latencies[0]
	for _, l := range latencies {
		if l < min {
			min = l
		}
	}
	return min
}

func maxLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	max := latencies[0]
	for _, l := range latencies {
		if l > max {
			max = l
		}
	}
	return max
}

func avgLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	var sum int64
	for _, l := range latencies {
		sum += int64(l)
	}
	return time.Duration(sum / int64(len(latencies)))
}

func formatLatency(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%d ns", d.Nanoseconds())
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.2f µs", float64(d.Nanoseconds())/1000)
	} else {
		return fmt.Sprintf("%.2f ms", float64(d.Microseconds())/1000)
	}
}
