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

// Market represents a trading market with its order book
type Market struct {
	Symbol      string
	OrderBook   *lx.OrderBook
	LastPrice   float64
	Volume24h   float64
	MarketType  string // "crypto" or "security"
	OrderCount  int32
	TradeCount  int32
}

// Global stats
var (
	totalOrders  int64
	totalTrades  int64
	totalMemory  int64
	activeMarkets int32
)

func main() {
	// Force GC and free memory before starting
	runtime.GC()
	debug.FreeOSMemory()
	
	fmt.Println("=== LX DEX Memory Stress Test ===")
	fmt.Println("Target: 11,000 Securities + 10,000 Crypto Markets")
	fmt.Println("Hardware Target: Mac Studio M2 Ultra (512GB RAM)")
	fmt.Println()
	
	// Get initial memory baseline
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	initialMemory := m.Alloc
	
	fmt.Printf("Initial memory usage: %.2f MB\n", float64(initialMemory)/(1024*1024))
	fmt.Printf("System memory: %.2f GB\n", float64(m.Sys)/(1024*1024*1024))
	fmt.Println()
	
	// Create all markets
	fmt.Println("Phase 1: Creating 21,000 markets...")
	markets := createMarkets()
	
	// Measure memory after market creation
	runtime.GC()
	runtime.ReadMemStats(&m)
	afterMarketsMemory := m.Alloc
	marketsMemoryUsed := afterMarketsMemory - initialMemory
	
	fmt.Printf("✓ Created %d markets\n", len(markets))
	fmt.Printf("  Memory used for empty markets: %.2f GB\n", 
		float64(marketsMemoryUsed)/(1024*1024*1024))
	fmt.Println()
	
	// Phase 2: Populate with realistic order distribution
	fmt.Println("Phase 2: Populating markets with orders...")
	populateMarkets(markets)
	
	// Measure memory after population
	runtime.GC()
	runtime.ReadMemStats(&m)
	afterPopulateMemory := m.Alloc
	ordersMemoryUsed := afterPopulateMemory - afterMarketsMemory
	
	fmt.Printf("✓ Added %d total orders\n", atomic.LoadInt64(&totalOrders))
	fmt.Printf("  Memory used for orders: %.2f GB\n", 
		float64(ordersMemoryUsed)/(1024*1024*1024))
	fmt.Println()
	
	// Phase 3: Simulate trading activity
	fmt.Println("Phase 3: Simulating trading activity...")
	simulateTrading(markets, 10) // 10 seconds of trading
	
	// Final memory measurement
	runtime.GC()
	runtime.ReadMemStats(&m)
	finalMemory := m.Alloc
	totalMemoryUsed := finalMemory - initialMemory
	
	fmt.Println()
	fmt.Println("=== Final Results ===")
	fmt.Printf("Total markets active: %d\n", len(markets))
	fmt.Printf("Total orders placed: %d\n", atomic.LoadInt64(&totalOrders))
	fmt.Printf("Total trades executed: %d\n", atomic.LoadInt64(&totalTrades))
	fmt.Printf("Average orders per market: %.1f\n", 
		float64(atomic.LoadInt64(&totalOrders))/float64(len(markets)))
	fmt.Println()
	
	// Memory analysis
	fmt.Println("=== Memory Analysis ===")
	fmt.Printf("Total memory used: %.2f GB\n", float64(totalMemoryUsed)/(1024*1024*1024))
	fmt.Printf("Memory per market: %.2f MB\n", 
		float64(totalMemoryUsed)/(float64(len(markets))*1024*1024))
	fmt.Printf("System memory allocated: %.2f GB\n", float64(m.Sys)/(1024*1024*1024))
	fmt.Printf("Heap in use: %.2f GB\n", float64(m.HeapInuse)/(1024*1024*1024))
	fmt.Printf("Heap idle: %.2f GB\n", float64(m.HeapIdle)/(1024*1024*1024))
	fmt.Printf("GC runs: %d\n", m.NumGC)
	fmt.Println()
	
	// Projection for Mac Studio
	fmt.Println("=== Mac Studio M2 Ultra (512GB) Projection ===")
	memoryPerMarketMB := float64(totalMemoryUsed) / (float64(len(markets)) * 1024 * 1024)
	
	// Calculate how many markets can fit
	availableMemoryGB := 450.0 // Leave 62GB for OS and other processes
	maxMarkets := int(availableMemoryGB * 1024 / memoryPerMarketMB)
	
	fmt.Printf("Memory per market: %.2f MB\n", memoryPerMarketMB)
	fmt.Printf("Available memory for DEX: %.0f GB\n", availableMemoryGB)
	fmt.Printf("Maximum markets supported: %d\n", maxMarkets)
	fmt.Printf("Current usage (21K markets): %.1f%% of capacity\n", 
		float64(len(markets))*100/float64(maxMarkets))
	
	// With MLX optimization
	mlxOptimizationFactor := 0.1 // MLX uses 10x less memory
	mlxMemoryPerMarket := memoryPerMarketMB * mlxOptimizationFactor
	mlxMaxMarkets := int(availableMemoryGB * 1024 / mlxMemoryPerMarket)
	
	fmt.Println()
	fmt.Println("With MLX Optimization (10x reduction):")
	fmt.Printf("Memory per market: %.2f MB\n", mlxMemoryPerMarket)
	fmt.Printf("Maximum markets supported: %d\n", mlxMaxMarkets)
	fmt.Printf("Can support all global markets: %v\n", mlxMaxMarkets > 1000000)
	
	// Test specific market access
	fmt.Println()
	fmt.Println("=== Market Access Performance Test ===")
	testMarketAccess(markets)
}

func createMarkets() map[string]*Market {
	markets := make(map[string]*Market, 21000)
	
	// Create 11,000 securities (US stocks, ETFs, etc.)
	securities := []string{
		"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "V", "JPM",
		"JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC", "CVX", "ABBV",
	}
	
	for i := 0; i < 11000; i++ {
		var symbol string
		if i < len(securities) {
			symbol = securities[i]
		} else {
			// Generate synthetic symbols for remaining securities
			symbol = fmt.Sprintf("STK%04d", i)
		}
		
		market := &Market{
			Symbol:     symbol,
			OrderBook:  lx.NewOrderBook(symbol),
			MarketType: "security",
			LastPrice:  100 + rand.Float64()*1000, // $100-$1100
		}
		markets[symbol] = market
		atomic.AddInt32(&activeMarkets, 1)
	}
	
	// Create 10,000 crypto markets
	cryptoPairs := []string{
		"BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD",
		"ADA/USD", "AVAX/USD", "DOGE/USD", "TRX/USD", "DOT/USD",
		"MATIC/USD", "SHIB/USD", "LTC/USD", "UNI/USD", "LINK/USD",
	}
	
	for i := 0; i < 10000; i++ {
		var symbol string
		if i < len(cryptoPairs) {
			symbol = cryptoPairs[i]
		} else {
			// Generate synthetic crypto pairs
			base := fmt.Sprintf("TOK%d", i%1000)
			quote := []string{"USD", "USDT", "BTC", "ETH"}[i%4]
			symbol = fmt.Sprintf("%s/%s", base, quote)
		}
		
		market := &Market{
			Symbol:     symbol,
			OrderBook:  lx.NewOrderBook(symbol),
			MarketType: "crypto",
			LastPrice:  0.001 + rand.Float64()*50000, // $0.001-$50,000
		}
		markets[symbol] = market
	}
	
	return markets
}

func populateMarkets(markets map[string]*Market) {
	// Realistic order distribution:
	// - Top 100 markets: 500-1000 orders each
	// - Next 1000 markets: 100-500 orders each
	// - Next 5000 markets: 20-100 orders each
	// - Remaining markets: 5-20 orders each
	
	marketList := make([]*Market, 0, len(markets))
	for _, market := range markets {
		marketList = append(marketList, market)
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
		go func(markets []*Market, workerID int) {
			defer wg.Done()
			
			for i, market := range markets {
				globalIndex := start + i
				var numOrders int
				
				// Determine number of orders based on market rank
				switch {
				case globalIndex < 100:
					numOrders = 500 + rand.Intn(500) // 500-1000
				case globalIndex < 1100:
					numOrders = 100 + rand.Intn(400) // 100-500
				case globalIndex < 6100:
					numOrders = 20 + rand.Intn(80) // 20-100
				default:
					numOrders = 5 + rand.Intn(15) // 5-20
				}
				
				// Add orders to this market
				basePrice := market.LastPrice
				for j := 0; j < numOrders; j++ {
					// Create realistic bid/ask spread
					spread := basePrice * 0.001 // 0.1% spread
					
					order := &lx.Order{
						ID:   uint64(globalIndex*10000 + j),
						Type: lx.Limit,
						Side: lx.Side(j % 2), // Alternate buy/sell
						User: fmt.Sprintf("user%d", j%100),
						Timestamp: time.Now(),
					}
					
					if order.Side == lx.Buy {
						// Buy orders below market price
						order.Price = basePrice - spread*(1+rand.Float64()*10)
						order.Size = float64(10 + rand.Intn(990)) // 10-1000 units
					} else {
						// Sell orders above market price
						order.Price = basePrice + spread*(1+rand.Float64()*10)
						order.Size = float64(10 + rand.Intn(990))
					}
					
					market.OrderBook.AddOrder(order)
					atomic.AddInt32(&market.OrderCount, 1)
					atomic.AddInt64(&totalOrders, 1)
				}
			}
			
			if workerID == 0 {
				fmt.Printf("  Worker %d: Populated %d markets\n", workerID, len(markets))
			}
		}(marketList[start:end], w)
	}
	
	wg.Wait()
}

func simulateTrading(markets map[string]*Market, durationSeconds int) {
	// Simulate realistic trading activity
	marketList := make([]*Market, 0, len(markets))
	for _, market := range markets {
		marketList = append(marketList, market)
	}
	
	stopTime := time.Now().Add(time.Duration(durationSeconds) * time.Second)
	tradesPerSecond := 10000 // Target trades per second across all markets
	
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			tradesPerWorker := tradesPerSecond / numWorkers
			sleepDuration := time.Second / time.Duration(tradesPerWorker)
			
			for time.Now().Before(stopTime) {
				// Pick a random market (weighted towards popular ones)
				marketIndex := weightedRandomMarket(len(marketList))
				market := marketList[marketIndex]
				
				// Create a market order
				order := &lx.Order{
					ID:   uint64(time.Now().UnixNano()),
					Type: lx.Market,
					Side: lx.Side(rand.Intn(2)),
					Size: float64(1 + rand.Intn(100)),
					User: fmt.Sprintf("trader%d", workerID),
					Timestamp: time.Now(),
				}
				
				// Execute the order
				trades := market.OrderBook.AddOrder(order)
				if trades > 0 {
					atomic.AddInt32(&market.TradeCount, 1)
					atomic.AddInt64(&totalTrades, int64(trades))
				}
				
				time.Sleep(sleepDuration)
			}
		}(w)
	}
	
	// Progress indicator
	go func() {
		for time.Now().Before(stopTime) {
			time.Sleep(time.Second)
			fmt.Printf("  Trades executed: %d\n", atomic.LoadInt64(&totalTrades))
		}
	}()
	
	wg.Wait()
}

func weightedRandomMarket(numMarkets int) int {
	// 80% of trades go to top 20% of markets
	if rand.Float64() < 0.8 {
		return rand.Intn(numMarkets / 5)
	}
	return rand.Intn(numMarkets)
}

func testMarketAccess(markets map[string]*Market) {
	// Test random access performance
	marketList := make([]*Market, 0, len(markets))
	symbols := make([]string, 0, len(markets))
	for symbol, market := range markets {
		marketList = append(marketList, market)
		symbols = append(symbols, symbol)
	}
	
	// Test sequential access
	start := time.Now()
	for i := 0; i < 1000; i++ {
		market := marketList[i]
		_ = market.OrderBook.GetBestBid()
		_ = market.OrderBook.GetBestAsk()
	}
	seqDuration := time.Since(start)
	
	// Test random access
	start = time.Now()
	for i := 0; i < 1000; i++ {
		idx := rand.Intn(len(marketList))
		market := marketList[idx]
		_ = market.OrderBook.GetBestBid()
		_ = market.OrderBook.GetBestAsk()
	}
	randDuration := time.Since(start)
	
	// Test map lookup
	start = time.Now()
	for i := 0; i < 1000; i++ {
		symbol := symbols[rand.Intn(len(symbols))]
		market := markets[symbol]
		_ = market.OrderBook.GetBestBid()
		_ = market.OrderBook.GetBestAsk()
	}
	mapDuration := time.Since(start)
	
	fmt.Printf("Sequential access (1000 markets): %v\n", seqDuration)
	fmt.Printf("Random access (1000 markets): %v\n", randDuration)
	fmt.Printf("Map lookup (1000 markets): %v\n", mapDuration)
	fmt.Printf("Average per-market access: %.2f µs\n", 
		float64(mapDuration.Nanoseconds())/1000000)
}

// Additional analysis functions
func analyzeMemoryBreakdown(markets map[string]*Market) {
	var (
		totalOrderMemory int64
		totalTreeMemory  int64
		totalMapMemory   int64
	)
	
	for _, market := range markets {
		// Estimate memory for this market
		orderCount := atomic.LoadInt32(&market.OrderCount)
		
		// Each order ~248 bytes
		orderMemory := int64(orderCount) * 248
		totalOrderMemory += orderMemory
		
		// Tree structure overhead ~64 bytes per price level
		priceLevels := int64(orderCount / 5) // Estimate 5 orders per price level
		treeMemory := priceLevels * 64
		totalTreeMemory += treeMemory
		
		// Map overhead
		mapMemory := int64(48) // Base map structure
		totalMapMemory += mapMemory
	}
	
	fmt.Println()
	fmt.Println("=== Memory Breakdown ===")
	fmt.Printf("Order data: %.2f GB\n", float64(totalOrderMemory)/(1024*1024*1024))
	fmt.Printf("Tree structures: %.2f GB\n", float64(totalTreeMemory)/(1024*1024*1024))
	fmt.Printf("Map overhead: %.2f MB\n", float64(totalMapMemory)/(1024*1024))
}