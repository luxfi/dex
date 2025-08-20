package main

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"time"
	"unsafe"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
)

// Memory analysis for MLX matching engine and orderbook
func main() {
	// Force GC before starting
	runtime.GC()
	debug.FreeOSMemory()

	// Get initial memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("=== LX DEX Memory Analysis for 1M Markets ===")
	fmt.Printf("Initial memory: %.2f MB\n", float64(m.Alloc)/(1024*1024))
	fmt.Println()

	// Calculate theoretical memory requirements
	analyzeTheoreticalMemory()

	// Test actual memory usage with scaling
	fmt.Println("\n=== Actual Memory Usage Tests ===")

	// Test 1: Single orderbook memory
	testSingleOrderbook()

	// Test 2: 1000 orderbooks
	test1KOrderbooks()

	// Test 3: 10K orderbooks
	test10KOrderbooks()

	// Test 4: 100K orderbooks (if memory allows)
	test100KOrderbooks()

	// Test 5: MLX Engine memory usage
	testMLXEngineMemory()

	// Extrapolate to 1M markets
	extrapolateTo1Million()
}

func analyzeTheoreticalMemory() {
	fmt.Println("=== Theoretical Memory Analysis ===")
	fmt.Println()

	// Size of core structures
	fmt.Printf("Size of Order struct: %d bytes\n", unsafe.Sizeof(lx.Order{}))
	fmt.Printf("Size of Trade struct: %d bytes\n", unsafe.Sizeof(lx.Trade{}))
	fmt.Printf("Size of OrderBook struct: %d bytes\n", unsafe.Sizeof(lx.OrderBook{}))
	fmt.Printf("Size of PriceLevel struct: %d bytes\n", unsafe.Sizeof(lx.PriceLevel{}))

	// MLX structures
	fmt.Printf("Size of MLX Order: %d bytes\n", unsafe.Sizeof(mlx.Order{}))
	fmt.Printf("Size of MLX Trade: %d bytes\n", unsafe.Sizeof(mlx.Trade{}))

	fmt.Println("\n--- Per-Market Memory Requirements ---")

	// Assumptions:
	// - Average 100 active orders per market
	// - Average 20 price levels per side
	// - Average 1000 trades history

	orderSize := unsafe.Sizeof(lx.Order{})
	avgOrdersPerMarket := 100
	avgPriceLevels := 40 // 20 per side
	avgTradesHistory := 1000

	orderMemory := orderSize * uintptr(avgOrdersPerMarket)
	priceLevelMemory := unsafe.Sizeof(lx.PriceLevel{}) * uintptr(avgPriceLevels)
	tradeMemory := unsafe.Sizeof(lx.Trade{}) * uintptr(avgTradesHistory)
	orderbookOverhead := unsafe.Sizeof(lx.OrderBook{})

	// Maps and slices overhead
	mapsOverhead := uintptr(1024)  // Estimate for internal map structures
	slicesOverhead := uintptr(512) // Estimate for slice headers

	totalPerMarket := orderMemory + priceLevelMemory + tradeMemory +
		orderbookOverhead + mapsOverhead + slicesOverhead

	fmt.Printf("Orders memory: %d bytes (%d orders × %d bytes)\n",
		orderMemory, avgOrdersPerMarket, orderSize)
	fmt.Printf("Price levels memory: %d bytes\n", priceLevelMemory)
	fmt.Printf("Trade history memory: %d bytes\n", tradeMemory)
	fmt.Printf("Orderbook overhead: %d bytes\n", orderbookOverhead)
	fmt.Printf("Maps/slices overhead: %d bytes\n", mapsOverhead+slicesOverhead)
	fmt.Printf("\nTotal per market: %d bytes (%.2f KB)\n",
		totalPerMarket, float64(totalPerMarket)/1024)

	// Calculate for different scales
	fmt.Println("\n--- Projected Memory Usage ---")
	scales := []int{1000, 10000, 100000, 1000000}
	for _, numMarkets := range scales {
		totalBytes := totalPerMarket * uintptr(numMarkets)
		fmt.Printf("%d markets: %.2f GB\n",
			numMarkets, float64(totalBytes)/(1024*1024*1024))
	}
}

func testSingleOrderbook() {
	fmt.Println("\n--- Single Orderbook Test ---")

	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	book := lx.NewOrderBook("TEST/USD")

	// Add 100 orders
	for i := 0; i < 100; i++ {
		book.AddOrder(&lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     100 + float64(i%10),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	used := after - before

	fmt.Printf("Memory used by 1 orderbook with 100 orders: %d bytes (%.2f KB)\n",
		used, float64(used)/1024)
}

func test1KOrderbooks() {
	fmt.Println("\n--- 1,000 Orderbooks Test ---")

	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	books := make(map[string]*lx.OrderBook)

	for i := 0; i < 1000; i++ {
		symbol := fmt.Sprintf("MARKET%d/USD", i)
		book := lx.NewOrderBook(symbol)

		// Add 50 orders per book
		for j := 0; j < 50; j++ {
			book.AddOrder(&lx.Order{
				ID:        uint64(i*100 + j),
				Type:      lx.Limit,
				Side:      lx.Buy,
				Price:     100 + float64(j%10),
				Size:      10,
				User:      "user",
				Timestamp: time.Now(),
			})
		}

		books[symbol] = book
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	used := after - before

	fmt.Printf("Memory used by 1,000 orderbooks: %d bytes (%.2f MB)\n",
		used, float64(used)/(1024*1024))
	fmt.Printf("Average per orderbook: %.2f KB\n",
		float64(used)/(1000*1024))
}

func test10KOrderbooks() {
	fmt.Println("\n--- 10,000 Orderbooks Test ---")

	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	books := make(map[string]*lx.OrderBook, 10000)

	for i := 0; i < 10000; i++ {
		symbol := fmt.Sprintf("MKT%d", i)
		book := lx.NewOrderBook(symbol)

		// Add 20 orders per book (reduced for memory)
		for j := 0; j < 20; j++ {
			book.AddOrder(&lx.Order{
				ID:        uint64(i*100 + j),
				Type:      lx.Limit,
				Side:      lx.Buy,
				Price:     100 + float64(j%5),
				Size:      10,
				User:      "u",
				Timestamp: time.Now(),
			})
		}

		books[symbol] = book
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	used := after - before

	fmt.Printf("Memory used by 10,000 orderbooks: %d bytes (%.2f MB)\n",
		used, float64(used)/(1024*1024))
	fmt.Printf("Average per orderbook: %.2f KB\n",
		float64(used)/(10000*1024))
}

func test100KOrderbooks() {
	fmt.Println("\n--- 100,000 Orderbooks Test ---")
	fmt.Println("(This may take a while and use significant memory)")

	// Check available memory first
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	availableMB := (m.Sys - m.Alloc) / (1024 * 1024)

	if availableMB < 4000 { // Need at least 4GB free
		fmt.Printf("Insufficient memory available: %.2f MB. Skipping 100K test.\n",
			float64(availableMB))
		return
	}

	runtime.GC()
	runtime.ReadMemStats(&m)
	before := m.Alloc

	books := make(map[string]*lx.OrderBook, 100000)

	// Process in batches to avoid memory pressure
	batchSize := 10000
	for batch := 0; batch < 10; batch++ {
		for i := 0; i < batchSize; i++ {
			idx := batch*batchSize + i
			symbol := fmt.Sprintf("M%d", idx)
			book := lx.NewOrderBook(symbol)

			// Add minimal orders (10 per book)
			for j := 0; j < 10; j++ {
				book.AddOrder(&lx.Order{
					ID:        uint64(idx*100 + j),
					Type:      lx.Limit,
					Side:      lx.Buy,
					Price:     100,
					Size:      1,
					User:      "u",
					Timestamp: time.Now(),
				})
			}

			books[symbol] = book
		}

		// Progress indicator
		fmt.Printf("  Batch %d/10 complete\n", batch+1)
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	used := after - before

	fmt.Printf("Memory used by 100,000 orderbooks: %d bytes (%.2f GB)\n",
		used, float64(used)/(1024*1024*1024))
	fmt.Printf("Average per orderbook: %.2f KB\n",
		float64(used)/(100000*1024))
}

func testMLXEngineMemory() {
	fmt.Println("\n--- MLX Engine Memory Test ---")

	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	// Create MLX engine
	engine, err := mlx.NewEngine(mlx.Config{
		Backend:  mlx.BackendCPU,
		MaxBatch: 10000,
	})
	if err != nil {
		fmt.Printf("Failed to create MLX engine: %v\n", err)
		return
	}
	defer engine.Close()

	// Create batch of orders
	bids := make([]mlx.Order, 1000)
	asks := make([]mlx.Order, 1000)

	for i := 0; i < 1000; i++ {
		bids[i] = mlx.Order{
			ID:    uint64(i),
			Side:  0,
			Price: 100 - float64(i%10),
			Size:  10,
		}
		asks[i] = mlx.Order{
			ID:    uint64(i + 1000),
			Side:  1,
			Price: 101 + float64(i%10),
			Size:  10,
		}
	}

	// Run batch matching
	trades := engine.BatchMatch(bids, asks)

	runtime.ReadMemStats(&m)
	after := m.Alloc
	used := after - before

	fmt.Printf("MLX Engine memory overhead: %d bytes (%.2f KB)\n",
		used, float64(used)/1024)
	fmt.Printf("Trades generated: %d\n", len(trades))
}

func extrapolateTo1Million() {
	fmt.Println("\n=== Extrapolation to 1 Million Markets ===")
	fmt.Println()

	// Based on our tests, calculate average memory per market
	// Using conservative estimates

	// Different order book densities
	scenarios := []struct {
		name           string
		ordersPerBook  int
		bytesPerMarket int
	}{
		{"Light (10 orders/market)", 10, 25 * 1024},
		{"Medium (50 orders/market)", 50, 50 * 1024},
		{"Heavy (100 orders/market)", 100, 100 * 1024},
		{"Very Heavy (500 orders/market)", 500, 400 * 1024},
	}

	fmt.Println("Memory Requirements for 1M Markets:")
	fmt.Println("------------------------------------")

	for _, s := range scenarios {
		totalBytes := int64(s.bytesPerMarket) * 1000000
		totalGB := float64(totalBytes) / (1024 * 1024 * 1024)

		fmt.Printf("%s:\n", s.name)
		fmt.Printf("  Total Memory: %.2f GB\n", totalGB)
		fmt.Printf("  Per Market: %d KB\n", s.bytesPerMarket/1024)

		// Add MLX GPU memory if using unified memory
		if totalGB < 128 {
			fmt.Printf("  ✅ Fits in 128GB unified memory (Apple M2 Ultra)\n")
		} else if totalGB < 256 {
			fmt.Printf("  ⚠️  Requires 256GB+ memory\n")
		} else {
			fmt.Printf("  ❌ Requires distributed architecture\n")
		}
		fmt.Println()
	}

	// Optimization recommendations
	fmt.Println("=== Memory Optimization Strategies ===")
	fmt.Println()
	fmt.Println("1. Use memory pools to reduce allocation overhead")
	fmt.Println("2. Implement order book compression for inactive markets")
	fmt.Println("3. Use tiered storage (hot/warm/cold markets)")
	fmt.Println("4. Implement memory-mapped files for historical data")
	fmt.Println("5. Use delta compression for order book updates")
	fmt.Println("6. Implement market hibernation for low-activity pairs")
	fmt.Println("7. Use shared memory for cross-market data")
	fmt.Println("8. Implement intelligent prefetching for active markets")

	// Hardware recommendations
	fmt.Println("\n=== Hardware Recommendations for 1M Markets ===")
	fmt.Println()
	fmt.Println("Option 1: Apple Mac Studio M2 Ultra (Recommended)")
	fmt.Println("  - 192GB unified memory configuration")
	fmt.Println("  - Can handle 1M light markets (10-20 orders each)")
	fmt.Println("  - MLX GPU acceleration included")
	fmt.Println("  - Power efficient (370W max)")
	fmt.Println()
	fmt.Println("Option 2: Server with 256GB+ RAM")
	fmt.Println("  - AMD EPYC or Intel Xeon")
	fmt.Println("  - 256-512GB DDR5 RAM")
	fmt.Println("  - NVIDIA A100 80GB for GPU acceleration")
	fmt.Println("  - Higher power consumption (1000W+)")
	fmt.Println()
	fmt.Println("Option 3: Distributed Architecture")
	fmt.Println("  - Shard markets across multiple nodes")
	fmt.Println("  - 10 nodes × 100K markets each")
	fmt.Println("  - Each node with 32-64GB RAM")
	fmt.Println("  - Horizontal scaling capability")
}

// Helper function to format bytes
func formatBytes(bytes uint64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	} else if bytes < 1024*1024 {
		return fmt.Sprintf("%.2f KB", float64(bytes)/1024)
	} else if bytes < 1024*1024*1024 {
		return fmt.Sprintf("%.2f MB", float64(bytes)/(1024*1024))
	} else {
		return fmt.Sprintf("%.2f GB", float64(bytes)/(1024*1024*1024))
	}
}
