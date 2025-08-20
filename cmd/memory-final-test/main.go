package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/pkg/lx"
)

// Global counters
var (
	totalOrders   int64
	totalTrades   int64
	peakMemory    uint64
	currentMemory uint64
)

func main() {
	// Configure runtime
	runtime.GOMAXPROCS(runtime.NumCPU())
	debug.SetGCPercent(100)

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     LX DEX Production Memory Test - 21,000 Global Markets       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Configuration:")
	fmt.Println("  • 11,000 Securities (US Stocks, ETFs, International)")
	fmt.Println("  • 10,000 Crypto Markets (Spot, Futures, Perpetuals)")
	fmt.Println("  • Target Hardware: Mac Studio M2 Ultra (512GB)")
	fmt.Println("  • Expected Load: 1M+ orders, 100K+ trades/sec")
	fmt.Println()

	// Initial state
	runtime.GC()
	debug.FreeOSMemory()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	startMem := m.Alloc

	// Phase 1: Market Creation
	fmt.Println("【Phase 1】Creating Global Markets")
	fmt.Println("─────────────────────────────────")
	markets := createGlobalMarkets()

	runtime.GC()
	runtime.ReadMemStats(&m)
	phase1Mem := m.Alloc - startMem

	fmt.Printf("✓ Securities Markets: 11,000\n")
	fmt.Printf("✓ Crypto Markets: 10,000\n")
	fmt.Printf("✓ Memory Used: %.2f MB\n", float64(phase1Mem)/(1024*1024))
	fmt.Println()

	// Phase 2: Order Population (Realistic Distribution)
	fmt.Println("【Phase 2】Populating Order Books")
	fmt.Println("─────────────────────────────────")
	populateRealisticOrders(markets)

	runtime.GC()
	runtime.ReadMemStats(&m)
	phase2Mem := m.Alloc - startMem
	orderMem := phase2Mem - phase1Mem

	fmt.Printf("✓ Total Orders: %d\n", atomic.LoadInt64(&totalOrders))
	fmt.Printf("✓ Average Orders/Market: %.1f\n",
		float64(atomic.LoadInt64(&totalOrders))/float64(len(markets)))
	fmt.Printf("✓ Order Memory: %.2f GB\n", float64(orderMem)/(1024*1024*1024))
	fmt.Println()

	// Phase 3: Trading Simulation
	fmt.Println("【Phase 3】Trading Simulation")
	fmt.Println("─────────────────────────────────")
	simulateProduction(markets)

	runtime.GC()
	runtime.ReadMemStats(&m)
	finalMem := m.Alloc - startMem

	// Final Analysis
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        FINAL RESULTS                             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Market Statistics:")
	fmt.Printf("  • Total Markets: %d\n", len(markets))
	fmt.Printf("  • Total Orders: %d\n", atomic.LoadInt64(&totalOrders))
	fmt.Printf("  • Total Trades: %d\n", atomic.LoadInt64(&totalTrades))
	fmt.Println()

	fmt.Println("Memory Analysis:")
	fmt.Printf("  • Total Memory Used: %.2f GB\n", float64(finalMem)/(1024*1024*1024))
	fmt.Printf("  • Per Market Average: %.2f MB\n", float64(finalMem)/(float64(len(markets))*1024*1024))
	fmt.Printf("  • System Memory: %.2f GB\n", float64(m.Sys)/(1024*1024*1024))
	fmt.Printf("  • Heap Objects: %d\n", m.HeapObjects)
	fmt.Printf("  • GC Runs: %d\n", m.NumGC)
	fmt.Println()

	// Mac Studio Analysis
	analyzeMacStudio(finalMem, len(markets))

	// Performance Benchmarks
	runPerformanceBenchmarks(markets)
}

func createGlobalMarkets() map[string]*lx.OrderBook {
	markets := make(map[string]*lx.OrderBook, 21000)

	// Top US Securities
	topSecurities := []string{
		"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
		"V", "JPM", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC",
		"CVX", "ABBV", "PFE", "AVGO", "CSCO", "MRK", "TMO", "LLY", "PEP",
		"COST", "ORCL", "ACN", "ADBE", "NKE", "NFLX", "ABT", "TXN", "CRM",
		"DHR", "VZ", "NEE", "CMCSA", "XOM", "INTC", "WFC", "QCOM", "T",
		"AMD", "UPS", "PM", "RTX", "LOW", "INTU", "BMY", "HON", "SPGI",
		"ELV", "UNP", "CVS", "SCHW", "GS", "MS", "BA", "MDT", "AXP", "BLK",
		"AMGN", "CAT", "SBUX", "AMT", "IBM", "GE", "GILD", "DE", "MU", "LMT",
	}

	// Create securities
	for i := 0; i < 11000; i++ {
		var symbol string
		if i < len(topSecurities) {
			symbol = topSecurities[i]
		} else {
			// Generate remaining symbols
			symbol = fmt.Sprintf("SEC%04d", i)
		}
		markets[symbol] = lx.NewOrderBook(symbol)
	}

	// Top Crypto Markets
	topCrypto := []string{
		"BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "SOL/USD",
		"ADA/USD", "AVAX/USD", "DOGE/USD", "TRX/USD", "DOT/USD",
		"MATIC/USD", "SHIB/USD", "LTC/USD", "UNI/USD", "LINK/USD",
		"BCH/USD", "NEAR/USD", "XLM/USD", "ATOM/USD", "XMR/USD",
		"ETC/USD", "APT/USD", "FIL/USD", "LDO/USD", "ARB/USD",
		"ICP/USD", "HBAR/USD", "VET/USD", "MKR/USD", "QNT/USD",
	}

	// Create crypto markets
	for i := 0; i < 10000; i++ {
		var symbol string
		if i < len(topCrypto) {
			symbol = topCrypto[i]
		} else {
			// Generate synthetic pairs
			base := fmt.Sprintf("TOK%d", i)
			quote := []string{"USD", "USDT", "BTC", "ETH"}[i%4]
			symbol = fmt.Sprintf("%s/%s", base, quote)
		}
		markets[symbol] = lx.NewOrderBook(symbol)
	}

	return markets
}

func populateRealisticOrders(markets map[string]*lx.OrderBook) {
	// Market distribution (based on real-world data):
	// Top 20 markets: 1000-5000 orders each (very liquid)
	// Top 100 markets: 500-1000 orders each (liquid)
	// Top 1000 markets: 100-500 orders each (moderate)
	// Top 5000 markets: 20-100 orders each (thin)
	// Rest: 5-20 orders each (very thin)

	marketList := make([]*lx.OrderBook, 0, len(markets))
	for _, book := range markets {
		marketList = append(marketList, book)
	}

	// Use goroutines for parallel population
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	marketsPerWorker := len(marketList) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * marketsPerWorker
		end := start + marketsPerWorker
		if w == numWorkers-1 {
			end = len(marketList)
		}

		wg.Add(1)
		go func(books []*lx.OrderBook, startIdx int) {
			defer wg.Done()

			for i, book := range books {
				globalIdx := startIdx + i

				// Determine order count based on market liquidity tier
				var orderCount int
				switch {
				case globalIdx < 20:
					orderCount = 1000 + rand.Intn(4000) // 1000-5000 (ultra liquid)
				case globalIdx < 100:
					orderCount = 500 + rand.Intn(500) // 500-1000 (very liquid)
				case globalIdx < 1000:
					orderCount = 100 + rand.Intn(400) // 100-500 (liquid)
				case globalIdx < 5000:
					orderCount = 20 + rand.Intn(80) // 20-100 (moderate)
				default:
					orderCount = 5 + rand.Intn(15) // 5-20 (thin)
				}

				// Generate realistic price/size distribution
				midPrice := 100.0 + rand.Float64()*1000
				spread := midPrice * 0.0005 // 0.05% typical spread

				// Add bid orders (buy side)
				for j := 0; j < orderCount/2; j++ {
					price := midPrice - spread*(1+rand.Float64()*20)
					size := generateRealisticSize()

					order := &lx.Order{
						ID:        uint64(globalIdx*10000 + j),
						Type:      lx.Limit,
						Side:      lx.Buy,
						Price:     price,
						Size:      size,
						User:      fmt.Sprintf("mm%d", j%20), // Market makers
						Timestamp: time.Now(),
					}
					book.AddOrder(order)
					atomic.AddInt64(&totalOrders, 1)
				}

				// Add ask orders (sell side)
				for j := 0; j < orderCount/2; j++ {
					price := midPrice + spread*(1+rand.Float64()*20)
					size := generateRealisticSize()

					order := &lx.Order{
						ID:        uint64(globalIdx*10000 + orderCount/2 + j),
						Type:      lx.Limit,
						Side:      lx.Sell,
						Price:     price,
						Size:      size,
						User:      fmt.Sprintf("mm%d", j%20),
						Timestamp: time.Now(),
					}
					book.AddOrder(order)
					atomic.AddInt64(&totalOrders, 1)
				}
			}
		}(marketList[start:end], start)
	}

	wg.Wait()
}

func generateRealisticSize() float64 {
	// Realistic size distribution (power law)
	r := rand.Float64()
	switch {
	case r < 0.5:
		return float64(1 + rand.Intn(10)) // Small: 1-10
	case r < 0.8:
		return float64(10 + rand.Intn(90)) // Medium: 10-100
	case r < 0.95:
		return float64(100 + rand.Intn(900)) // Large: 100-1000
	default:
		return float64(1000 + rand.Intn(9000)) // Very large: 1000-10000
	}
}

func simulateProduction(markets map[string]*lx.OrderBook) {
	// Simulate realistic trading patterns
	marketList := make([]*lx.OrderBook, 0, len(markets))
	for _, book := range markets {
		marketList = append(marketList, book)
	}

	// Run for 5 seconds
	duration := 5 * time.Second
	endTime := time.Now().Add(duration)

	// Progress tracker
	startTime := time.Now()
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()

		for time.Now().Before(endTime) {
			<-ticker.C
			trades := atomic.LoadInt64(&totalTrades)
			elapsed := time.Since(startTime).Seconds()
			if elapsed > 0 {
				fmt.Printf("  Trades/sec: %.0f\n", float64(trades)/elapsed)
			}
		}
	}()

	// Trading workers
	var wg sync.WaitGroup
	numTraders := runtime.NumCPU() * 2

	for t := 0; t < numTraders; t++ {
		wg.Add(1)
		go func(traderID int) {
			defer wg.Done()

			for time.Now().Before(endTime) {
				// Pick market with realistic distribution
				// 80% of volume in top 20% of markets
				var marketIdx int
				if rand.Float64() < 0.8 {
					marketIdx = rand.Intn(len(marketList) / 5)
				} else {
					marketIdx = rand.Intn(len(marketList))
				}

				book := marketList[marketIdx]

				// Create market order
				order := &lx.Order{
					ID:        uint64(time.Now().UnixNano()),
					Type:      lx.Market,
					Side:      lx.Side(rand.Intn(2)),
					Size:      float64(1 + rand.Intn(100)),
					User:      fmt.Sprintf("trader%d", traderID),
					Timestamp: time.Now(),
				}

				trades := book.AddOrder(order)
				if trades > 0 {
					atomic.AddInt64(&totalTrades, int64(trades))
				}

				// Realistic trading frequency
				time.Sleep(time.Microsecond * time.Duration(10+rand.Intn(90)))
			}
		}(t)
	}

	wg.Wait()
}

func analyzeMacStudio(memoryUsed uint64, numMarkets int) {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           Mac Studio M2 Ultra (512GB) Capability                 ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	memPerMarketMB := float64(memoryUsed) / (float64(numMarkets) * 1024 * 1024)
	memUsedGB := float64(memoryUsed) / (1024 * 1024 * 1024)

	// Mac Studio configs
	configs := []struct {
		name      string
		memoryGB  int
		available int // After OS overhead
	}{
		{"Mac Studio M2 Max (64GB)", 64, 56},
		{"Mac Studio M2 Max (96GB)", 96, 86},
		{"Mac Studio M2 Ultra (128GB)", 128, 115},
		{"Mac Studio M2 Ultra (192GB)", 192, 175},
		{"Mac Studio M2 Ultra (512GB)", 512, 462},
	}

	fmt.Println("Current 21,000 Market Performance:")
	fmt.Printf("  • Memory per market: %.3f MB\n", memPerMarketMB)
	fmt.Printf("  • Total memory used: %.2f GB\n", memUsedGB)
	fmt.Printf("  • Percentage of 512GB: %.2f%%\n", memUsedGB*100/512)
	fmt.Println()

	fmt.Println("Capacity Analysis by Configuration:")
	for _, cfg := range configs {
		maxMarkets := int(float64(cfg.available*1024) / memPerMarketMB)
		fmt.Printf("\n%s:\n", cfg.name)
		fmt.Printf("  • Available memory: %d GB\n", cfg.available)
		fmt.Printf("  • Max markets: %d\n", maxMarkets)

		if maxMarkets >= 1000000 {
			fmt.Printf("  • Status: ✅ Can handle 1M+ markets\n")
		} else if maxMarkets >= 100000 {
			fmt.Printf("  • Status: ✅ Can handle 100K+ markets\n")
		} else if maxMarkets >= 21000 {
			fmt.Printf("  • Status: ✅ Can handle current load\n")
		} else {
			fmt.Printf("  • Status: ⚠️  Below current requirements\n")
		}
	}

	fmt.Println()
	fmt.Println("Scaling Projections:")
	scales := []int{21000, 50000, 100000, 250000, 500000, 1000000}
	for _, scale := range scales {
		memRequired := memPerMarketMB * float64(scale) / 1024
		fmt.Printf("  %7d markets: %7.1f GB", scale, memRequired)

		// Find minimum config needed
		minConfig := "Distributed"
		for _, cfg := range configs {
			if float64(cfg.available) >= memRequired {
				minConfig = cfg.name
				break
			}
		}
		fmt.Printf(" → %s\n", minConfig)
	}
}

func runPerformanceBenchmarks(markets map[string]*lx.OrderBook) {
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    Performance Benchmarks                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	marketList := make([]*lx.OrderBook, 0, len(markets))
	for _, book := range markets {
		marketList = append(marketList, book)
	}

	// Benchmark 1: Order placement
	start := time.Now()
	numOps := 100000
	for i := 0; i < numOps; i++ {
		book := marketList[i%len(marketList)]
		order := &lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     100.0,
			Size:      10,
			User:      "bench",
			Timestamp: time.Now(),
		}
		book.AddOrder(order)
	}
	orderTime := time.Since(start)

	// Benchmark 2: Best price queries
	start = time.Now()
	for i := 0; i < numOps; i++ {
		book := marketList[i%len(marketList)]
		_ = book.GetBestBid()
		_ = book.GetBestAsk()
	}
	queryTime := time.Since(start)

	// Benchmark 3: Snapshot generation
	start = time.Now()
	for i := 0; i < 1000; i++ {
		book := marketList[i%len(marketList)]
		_ = book.GetSnapshot()
	}
	snapshotTime := time.Since(start)

	fmt.Println("Operation Latencies:")
	fmt.Printf("  • Order Placement: %.2f µs/op (%d ops/sec)\n",
		float64(orderTime.Nanoseconds())/float64(numOps)/1000,
		int(float64(numOps)/orderTime.Seconds()))
	fmt.Printf("  • Best Price Query: %.2f µs/op (%d ops/sec)\n",
		float64(queryTime.Nanoseconds())/float64(numOps*2)/1000,
		int(float64(numOps*2)/queryTime.Seconds()))
	fmt.Printf("  • Snapshot Generation: %.2f µs/op\n",
		float64(snapshotTime.Nanoseconds())/1000000)

	fmt.Println()
	fmt.Println("Projected System Throughput:")
	fmt.Printf("  • Orders/second: %d\n", int(float64(numOps)/orderTime.Seconds()))
	fmt.Printf("  • Queries/second: %d\n", int(float64(numOps*2)/queryTime.Seconds()))
	fmt.Printf("  • With MLX GPU acceleration (100x): %d orders/sec\n",
		int(float64(numOps)/orderTime.Seconds())*100)
}
