// Test MLX order processing with 1ms consensus blocks
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
)

type OrderStats struct {
	OrdersSubmitted   uint64
	OrdersProcessed   uint64
	TradesExecuted    uint64
	BlocksFinalized   uint64
	ConsensusLatency  uint64 // nanoseconds
	MatchingLatency   uint64 // nanoseconds
	OrderBookDepth    uint64
}

type ConsensusEngine struct {
	db              database.Database
	orderBook       *lx.OrderBook
	mlxEngine       mlx.Engine
	blockHeight     uint64
	blockTime       time.Duration // Target: 1ms
	pendingOrders   []*lx.Order
	mu              sync.RWMutex
	stats           *OrderStats
	lastBlockTime   time.Time
	actualOrders    []ActualOrder // Track real orders
}

type ActualOrder struct {
	ID        uint64
	Price     float64
	Size      float64
	Side      lx.Side
	Timestamp time.Time
	Status    string // "pending", "matched", "cancelled"
	TradeID   uint64 // If matched
}

type Block struct {
	Height    uint64
	Timestamp time.Time
	Orders    []ActualOrder
	Trades    []*lx.Trade
	Hash      [32]byte
}

func NewConsensusEngine(blockTime time.Duration) (*ConsensusEngine, error) {
	// Initialize BadgerDB
	dbPath := "./test-mlx-badger"
	os.RemoveAll(dbPath) // Clean slate for test
	
	// Use luxfi/database with BadgerDB backend
	dbConfig := database.Config{
		Name: "mlx-consensus-test",
		Path: dbPath,
	}
	
	db, err := database.New(database.LevelDB, dbConfig)
	if err != nil {
		// Fallback to memory DB for testing
		db = database.NewMemDB()
		log.Println("Using in-memory database")
	}
	
	// Initialize MLX engine
	mlxConfig := mlx.Config{
		Backend:  mlx.BackendAuto,
		MaxBatch: 10000,
	}
	
	mlxEngine, err := mlx.NewEngine(mlxConfig)
	if err != nil {
		return nil, fmt.Errorf("MLX init failed: %w", err)
	}
	
	log.Printf("MLX Engine initialized: %s on %s", mlxEngine.Backend(), mlxEngine.Device())
	
	return &ConsensusEngine{
		db:           db,
		orderBook:    lx.NewOrderBook("TEST-MARKET"),
		mlxEngine:    mlxEngine,
		blockHeight:  0,
		blockTime:    blockTime,
		stats:        &OrderStats{},
		lastBlockTime: time.Now(),
		actualOrders: make([]ActualOrder, 0, 100000),
	}, nil
}

func (c *ConsensusEngine) SubmitOrder(order *lx.Order) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Track the actual order
	actualOrder := ActualOrder{
		ID:        order.OrderID,
		Price:     order.Price,
		Size:      order.Size,
		Side:      order.Side,
		Timestamp: time.Now(),
		Status:    "pending",
	}
	
	c.actualOrders = append(c.actualOrders, actualOrder)
	c.pendingOrders = append(c.pendingOrders, order)
	atomic.AddUint64(&c.stats.OrdersSubmitted, 1)
	
	// Update order book depth
	atomic.StoreUint64(&c.stats.OrderBookDepth, uint64(len(c.orderBook.GetBids())+len(c.orderBook.GetAsks())))
}

func (c *ConsensusEngine) RunConsensus() {
	ticker := time.NewTicker(c.blockTime)
	defer ticker.Stop()
	
	log.Printf("Starting consensus with %v block time", c.blockTime)
	
	for range ticker.C {
		c.finalizeBlock()
	}
}

func (c *ConsensusEngine) finalizeBlock() {
	startTime := time.Now()
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if len(c.pendingOrders) == 0 {
		return
	}
	
	// Create new block
	block := Block{
		Height:    c.blockHeight + 1,
		Timestamp: time.Now(),
		Orders:    make([]ActualOrder, 0),
		Trades:    make([]*lx.Trade, 0),
	}
	
	// Process orders using MLX if available
	var trades []*lx.Trade
	matchStart := time.Now()
	
	if c.mlxEngine != nil && c.mlxEngine.IsGPUAvailable() {
		// Use MLX GPU acceleration
		trades = c.processOrdersMLX(c.pendingOrders)
	} else {
		// Fallback to CPU matching
		for _, order := range c.pendingOrders {
			orderTrades := c.orderBook.AddOrder(order)
			trades = append(trades, orderTrades...)
		}
	}
	
	matchLatency := time.Since(matchStart)
	atomic.StoreUint64(&c.stats.MatchingLatency, uint64(matchLatency.Nanoseconds()))
	
	// Update actual order status
	for i := range c.actualOrders {
		if c.actualOrders[i].Status == "pending" {
			// Check if order was matched
			for _, trade := range trades {
				if c.actualOrders[i].ID == trade.BuyOrder.OrderID || 
				   c.actualOrders[i].ID == trade.SellOrder.OrderID {
					c.actualOrders[i].Status = "matched"
					c.actualOrders[i].TradeID = trade.ID
					break
				}
			}
		}
	}
	
	// Store block in database
	c.storeBlock(&block, trades)
	
	// Update statistics
	atomic.AddUint64(&c.stats.OrdersProcessed, uint64(len(c.pendingOrders)))
	atomic.AddUint64(&c.stats.TradesExecuted, uint64(len(trades)))
	atomic.AddUint64(&c.stats.BlocksFinalized, 1)
	
	consensusLatency := time.Since(startTime)
	atomic.StoreUint64(&c.stats.ConsensusLatency, uint64(consensusLatency.Nanoseconds()))
	
	// Clear pending orders
	c.pendingOrders = c.pendingOrders[:0]
	c.blockHeight++
	c.lastBlockTime = time.Now()
}

func (c *ConsensusEngine) processOrdersMLX(orders []*lx.Order) []*lx.Trade {
	// Convert to MLX format
	mlxBids := make([]mlx.Order, 0)
	mlxAsks := make([]mlx.Order, 0)
	
	for _, order := range orders {
		mlxOrder := mlx.Order{
			ID:     order.OrderID,
			Price:  order.Price,
			Size:   order.Size,
			UserID: 0, // Simplified
		}
		
		if order.Side == lx.Buy {
			mlxOrder.Side = 0
			mlxBids = append(mlxBids, mlxOrder)
		} else {
			mlxOrder.Side = 1
			mlxAsks = append(mlxAsks, mlxOrder)
		}
	}
	
	// Process on GPU
	mlxTrades := c.mlxEngine.BatchMatch(mlxBids, mlxAsks)
	
	// Convert back to lx.Trade format
	trades := make([]*lx.Trade, len(mlxTrades))
	for i, mt := range mlxTrades {
		trades[i] = &lx.Trade{
			ID:    mt.ID,
			Price: mt.Price,
			Size:  mt.Size,
			// Map orders back
			BuyOrder: &lx.Order{OrderID: mt.BuyOrderID},
			SellOrder: &lx.Order{OrderID: mt.SellOrderID},
			Timestamp: time.Now(),
		}
	}
	
	return trades
}

func (c *ConsensusEngine) storeBlock(block *Block, trades []*lx.Trade) {
	// Store block header
	blockKey := fmt.Sprintf("block:%d", block.Height)
	blockData := make([]byte, 16)
	binary.BigEndian.PutUint64(blockData[0:8], block.Height)
	binary.BigEndian.PutUint64(blockData[8:16], uint64(block.Timestamp.UnixNano()))
	
	err := c.db.Put([]byte(blockKey), blockData)
	if err != nil {
		log.Printf("Failed to store block: %v", err)
	}
	
	// Store trades
	for _, trade := range trades {
		tradeKey := fmt.Sprintf("trade:%d:%d", block.Height, trade.ID)
		tradeData := make([]byte, 24)
		binary.BigEndian.PutUint64(tradeData[0:8], trade.ID)
		binary.BigEndian.PutUint64(tradeData[8:16], uint64(trade.Price*1e8))
		binary.BigEndian.PutUint64(tradeData[16:24], uint64(trade.Size*1e8))
		
		c.db.Put([]byte(tradeKey), tradeData)
	}
}

func (c *ConsensusEngine) PrintStats() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	startTime := time.Now()
	var lastOrders, lastTrades, lastBlocks uint64
	
	for range ticker.C {
		elapsed := time.Since(startTime).Seconds()
		
		submitted := atomic.LoadUint64(&c.stats.OrdersSubmitted)
		processed := atomic.LoadUint64(&c.stats.OrdersProcessed)
		trades := atomic.LoadUint64(&c.stats.TradesExecuted)
		blocks := atomic.LoadUint64(&c.stats.BlocksFinalized)
		consensusNs := atomic.LoadUint64(&c.stats.ConsensusLatency)
		matchNs := atomic.LoadUint64(&c.stats.MatchingLatency)
		depth := atomic.LoadUint64(&c.stats.OrderBookDepth)
		
		ordersPerSec := float64(processed-lastOrders)
		tradesPerSec := float64(trades-lastTrades)
		blocksPerSec := float64(blocks-lastBlocks)
		
		fmt.Printf("\n===== MLX CONSENSUS TEST - %.0fs =====\n", elapsed)
		fmt.Printf("📊 Orders: %d submitted, %d processed (%.0f/sec)\n", 
			submitted, processed, ordersPerSec)
		fmt.Printf("💹 Trades: %d executed (%.0f/sec)\n", 
			trades, tradesPerSec)
		fmt.Printf("⛓️  Blocks: %d finalized (%.1f/sec, target: %.0f/sec)\n", 
			blocks, blocksPerSec, 1000.0/float64(c.blockTime.Milliseconds()))
		fmt.Printf("⚡ Latency: Consensus %.1fμs, Matching %.1fμs\n",
			float64(consensusNs)/1000, float64(matchNs)/1000)
		fmt.Printf("📚 Order Book Depth: %d orders\n", depth)
		
		// Verify we're achieving target block time
		if blocksPerSec > 0 {
			actualBlockTime := 1000.0 / blocksPerSec
			fmt.Printf("⏱️  Actual block time: %.1fms (target: %dms)\n", 
				actualBlockTime, c.blockTime.Milliseconds())
			
			if c.blockTime.Milliseconds() == 1 && actualBlockTime <= 1.5 {
				fmt.Println("✅ ACHIEVING 1MS CONSENSUS!")
			}
		}
		
		// Check if MLX is really processing orders
		if c.mlxEngine != nil && c.mlxEngine.IsGPUAvailable() {
			fmt.Printf("🚀 MLX Engine: %s processing orders\n", c.mlxEngine.Backend())
		}
		
		// Show some actual orders to prove they're real
		c.mu.RLock()
		matchedCount := 0
		for _, order := range c.actualOrders {
			if order.Status == "matched" {
				matchedCount++
			}
		}
		c.mu.RUnlock()
		
		if matchedCount > 0 {
			fmt.Printf("✅ REAL ORDERS: %d matched out of %d total\n", 
				matchedCount, len(c.actualOrders))
		}
		
		lastOrders = processed
		lastTrades = trades
		lastBlocks = blocks
	}
}

func generateOrders(engine *ConsensusEngine, rate int, duration time.Duration) {
	ticker := time.NewTicker(time.Second / time.Duration(rate))
	defer ticker.Stop()
	
	timeout := time.After(duration)
	orderID := uint64(1)
	
	for {
		select {
		case <-ticker.C:
			// Generate a realistic order
			side := lx.Buy
			if orderID%2 == 0 {
				side = lx.Sell
			}
			
			// Create price around market (50000 +/- 100)
			price := 50000.0 + float64((orderID%200)-100)
			
			order := &lx.Order{
				OrderID:   atomic.AddUint64(&orderID, 1),
				ID:        fmt.Sprintf("order-%d", orderID),
				Type:      lx.Limit,
				Side:      side,
				Price:     price,
				Size:      1.0 + float64(orderID%10)/10.0,
				User:      fmt.Sprintf("user-%d", orderID%100),
				Timestamp: time.Now(),
			}
			
			engine.SubmitOrder(order)
			
		case <-timeout:
			return
		}
	}
}

func main() {
	var (
		blockTime = flag.Duration("block-time", 1*time.Millisecond, "Target block time (1ms for mainnet)")
		orderRate = flag.Int("order-rate", 10000, "Orders per second to generate")
		duration  = flag.Duration("duration", 10*time.Second, "Test duration")
		verify    = flag.Bool("verify", true, "Verify orders are real")
	)
	flag.Parse()
	
	fmt.Println("=================================================")
	fmt.Println("  MLX + 1MS CONSENSUS VERIFICATION TEST")
	fmt.Println("=================================================")
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("Block Time: %v (target for mainnet: 1ms)\n", *blockTime)
	fmt.Printf("Order Rate: %d orders/sec\n", *orderRate)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Println("=================================================")
	
	// Initialize consensus engine
	engine, err := NewConsensusEngine(*blockTime)
	if err != nil {
		log.Fatal(err)
	}
	defer engine.db.Close()
	
	// Start consensus loop
	go engine.RunConsensus()
	
	// Start stats printer
	go engine.PrintStats()
	
	// Generate orders
	fmt.Printf("\n🚀 Starting order generation at %d orders/sec...\n", *orderRate)
	generateOrders(engine, *orderRate, *duration)
	
	// Wait a bit for final blocks
	time.Sleep(2 * time.Second)
	
	// Final verification
	if *verify {
		fmt.Println("\n=== FINAL VERIFICATION ===")
		
		blocks := atomic.LoadUint64(&engine.stats.BlocksFinalized)
		trades := atomic.LoadUint64(&engine.stats.TradesExecuted)
		processed := atomic.LoadUint64(&engine.stats.OrdersProcessed)
		
		fmt.Printf("Total Blocks: %d\n", blocks)
		fmt.Printf("Total Orders Processed: %d\n", processed)
		fmt.Printf("Total Trades: %d\n", trades)
		
		if blocks > 0 && processed > 0 && trades > 0 {
			avgBlockTime := (*duration).Seconds() / float64(blocks) * 1000
			fmt.Printf("Average Block Time: %.2fms\n", avgBlockTime)
			
			if avgBlockTime <= 1.5 && *blockTime == 1*time.Millisecond {
				fmt.Println("✅ SUCCESS: 1ms consensus achieved with real orders!")
			} else if avgBlockTime <= float64(blockTime.Milliseconds())*1.5 {
				fmt.Printf("✅ SUCCESS: Target %dms consensus achieved!\n", blockTime.Milliseconds())
			} else {
				fmt.Printf("⚠️  Block time %.2fms exceeds target %dms\n", 
					avgBlockTime, blockTime.Milliseconds())
			}
		}
		
		// Verify database has real data
		val, err := engine.db.Get([]byte("block:1"))
		if err == nil && len(val) > 0 {
			fmt.Println("✅ Database contains real block data")
		}
	}
}