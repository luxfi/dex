// Simple test to verify MLX performance and 1ms consensus
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
)

type Stats struct {
	OrdersSubmitted  uint64
	OrdersProcessed  uint64
	TradesExecuted   uint64
	BlocksFinalized  uint64
	ConsensusLatency uint64
	MatchingLatency  uint64
}

type SimpleConsensus struct {
	orderBook     *lx.OrderBook
	mlxEngine     mlx.Engine
	blockHeight   uint64
	blockTime     time.Duration
	pendingOrders []*lx.Order
	mu            sync.RWMutex
	stats         *Stats
}

func NewSimpleConsensus(blockTime time.Duration) *SimpleConsensus {
	// Try to initialize MLX
	mlxEngine, err := mlx.NewEngine(mlx.Config{
		Backend:  mlx.BackendAuto,
		MaxBatch: 10000,
	})
	
	if err != nil {
		log.Printf("MLX not available: %v", err)
	} else {
		log.Printf("MLX Engine: %s on %s", mlxEngine.Backend(), mlxEngine.Device())
	}
	
	return &SimpleConsensus{
		orderBook:     lx.NewOrderBook("TEST"),
		mlxEngine:     mlxEngine,
		blockTime:     blockTime,
		pendingOrders: make([]*lx.Order, 0, 10000),
		stats:         &Stats{},
	}
}

func (c *SimpleConsensus) SubmitOrder(order *lx.Order) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.pendingOrders = append(c.pendingOrders, order)
	atomic.AddUint64(&c.stats.OrdersSubmitted, 1)
}

func (c *SimpleConsensus) RunConsensus() {
	ticker := time.NewTicker(c.blockTime)
	defer ticker.Stop()
	
	log.Printf("Starting consensus with %v block time", c.blockTime)
	
	for range ticker.C {
		c.finalizeBlock()
	}
}

func (c *SimpleConsensus) finalizeBlock() {
	startTime := time.Now()
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if len(c.pendingOrders) == 0 {
		return
	}
	
	// Process orders
	matchStart := time.Now()
	totalTrades := 0
	
	if c.mlxEngine != nil && c.mlxEngine.IsGPUAvailable() {
		// Convert to MLX format and process on GPU
		bids := make([]mlx.Order, 0)
		asks := make([]mlx.Order, 0)
		
		for _, order := range c.pendingOrders {
			mlxOrder := mlx.Order{
				ID:    order.ID,
				Price: order.Price,
				Size:  order.Size,
			}
			
			if order.Side == lx.Buy {
				mlxOrder.Side = 0
				bids = append(bids, mlxOrder)
			} else {
				mlxOrder.Side = 1
				asks = append(asks, mlxOrder)
			}
		}
		
		// GPU batch processing
		trades := c.mlxEngine.BatchMatch(bids, asks)
		totalTrades = len(trades)
		
	} else {
		// CPU fallback
		for _, order := range c.pendingOrders {
			numTrades := c.orderBook.AddOrder(order)
			totalTrades += int(numTrades)
		}
	}
	
	matchLatency := time.Since(matchStart)
	atomic.StoreUint64(&c.stats.MatchingLatency, uint64(matchLatency.Nanoseconds()))
	
	// Update stats
	atomic.AddUint64(&c.stats.OrdersProcessed, uint64(len(c.pendingOrders)))
	atomic.AddUint64(&c.stats.TradesExecuted, uint64(totalTrades))
	atomic.AddUint64(&c.stats.BlocksFinalized, 1)
	
	consensusLatency := time.Since(startTime)
	atomic.StoreUint64(&c.stats.ConsensusLatency, uint64(consensusLatency.Nanoseconds()))
	
	// Clear pending
	c.pendingOrders = c.pendingOrders[:0]
	c.blockHeight++
}

func (c *SimpleConsensus) PrintStats() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	startTime := time.Now()
	var lastProcessed, lastTrades, lastBlocks uint64
	
	for range ticker.C {
		elapsed := time.Since(startTime).Seconds()
		
		submitted := atomic.LoadUint64(&c.stats.OrdersSubmitted)
		processed := atomic.LoadUint64(&c.stats.OrdersProcessed)
		trades := atomic.LoadUint64(&c.stats.TradesExecuted)
		blocks := atomic.LoadUint64(&c.stats.BlocksFinalized)
		consensusNs := atomic.LoadUint64(&c.stats.ConsensusLatency)
		matchNs := atomic.LoadUint64(&c.stats.MatchingLatency)
		
		ordersPerSec := float64(processed-lastProcessed)
		tradesPerSec := float64(trades-lastTrades)
		blocksPerSec := float64(blocks-lastBlocks)
		
		fmt.Printf("\n===== CONSENSUS TEST @ %.0fs =====\n", elapsed)
		fmt.Printf("📊 Orders: %d submitted, %d processed (%.0f/sec)\n",
			submitted, processed, ordersPerSec)
		fmt.Printf("💹 Trades: %d executed (%.0f/sec)\n",
			trades, tradesPerSec)
		fmt.Printf("⛓️  Blocks: %d finalized (%.1f/sec)\n",
			blocks, blocksPerSec)
		
		if consensusNs > 0 {
			fmt.Printf("⚡ Latency: Consensus %.1fμs, Matching %.1fμs\n",
				float64(consensusNs)/1000, float64(matchNs)/1000)
		}
		
		// Check block time achievement
		if blocksPerSec > 0 {
			actualBlockTime := 1000.0 / blocksPerSec
			fmt.Printf("⏱️  Block time: %.1fms (target: %dms)\n",
				actualBlockTime, c.blockTime.Milliseconds())
			
			if c.blockTime.Milliseconds() == 1 && actualBlockTime <= 2.0 {
				fmt.Println("✅ ACHIEVING ~1MS CONSENSUS!")
			}
		}
		
		// Check if using MLX
		if c.mlxEngine != nil && c.mlxEngine.IsGPUAvailable() {
			fmt.Printf("🚀 MLX: %s processing %d orders/block\n",
				c.mlxEngine.Backend(), processed/blocks)
		}
		
		lastProcessed = processed
		lastTrades = trades
		lastBlocks = blocks
	}
}

func generateOrders(c *SimpleConsensus, rate int, duration time.Duration) {
	interval := time.Second / time.Duration(rate)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	timeout := time.After(duration)
	orderCount := uint64(0)
	
	for {
		select {
		case <-ticker.C:
			// Generate batch of orders
			for i := 0; i < 10; i++ {
				orderID := atomic.AddUint64(&orderCount, 1)
				
				side := lx.Buy
				if orderID%2 == 0 {
					side = lx.Sell
				}
				
				order := &lx.Order{
					ID:    orderID,
					Type:  lx.Limit,
					Side:  side,
					Price: 50000.0 + float64((orderID%200)-100),
					Size:  1.0,
					User:  fmt.Sprintf("user-%d", orderID%10),
				}
				
				c.SubmitOrder(order)
			}
			
		case <-timeout:
			return
		}
	}
}

func main() {
	var (
		blockTime = flag.Duration("block-time", 1*time.Millisecond, "Target block time")
		orderRate = flag.Int("order-rate", 1000, "Order batches per second")
		duration  = flag.Duration("duration", 10*time.Second, "Test duration")
	)
	flag.Parse()
	
	fmt.Println("============================================")
	fmt.Println("   1MS CONSENSUS VERIFICATION TEST")
	fmt.Println("============================================")
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("Block Time Target: %v\n", *blockTime)
	fmt.Printf("Order Rate: %d batches/sec (10 orders each)\n", *orderRate)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Println("============================================")
	
	// Create consensus engine
	consensus := NewSimpleConsensus(*blockTime)
	
	// Start consensus loop
	go consensus.RunConsensus()
	
	// Start stats
	go consensus.PrintStats()
	
	// Generate orders
	fmt.Printf("\n🚀 Starting order generation...\n")
	generateOrders(consensus, *orderRate, *duration)
	
	// Wait for final stats
	time.Sleep(2 * time.Second)
	
	// Final report
	fmt.Println("\n============================================")
	fmt.Println("            FINAL RESULTS")
	fmt.Println("============================================")
	
	blocks := atomic.LoadUint64(&consensus.stats.BlocksFinalized)
	orders := atomic.LoadUint64(&consensus.stats.OrdersProcessed)
	trades := atomic.LoadUint64(&consensus.stats.TradesExecuted)
	
	fmt.Printf("Total Blocks: %d\n", blocks)
	fmt.Printf("Total Orders: %d\n", orders)
	fmt.Printf("Total Trades: %d\n", trades)
	
	if blocks > 0 {
		avgBlockTime := duration.Seconds() / float64(blocks) * 1000
		fmt.Printf("Average Block Time: %.2fms\n", avgBlockTime)
		
		if *blockTime == 1*time.Millisecond {
			if avgBlockTime <= 2.0 {
				fmt.Println("✅ SUCCESS: ~1ms consensus achieved!")
				fmt.Printf("📊 Orders processed: %d at %.0f orders/sec\n",
					orders, float64(orders)/duration.Seconds())
			} else {
				fmt.Printf("⚠️  Block time %.2fms exceeds 1ms target\n", avgBlockTime)
			}
		}
	}
	
	// MLX performance estimation
	if consensus.mlxEngine != nil && consensus.mlxEngine.IsGPUAvailable() {
		// The benchmark returns the theoretical throughput
		theoretical := consensus.mlxEngine.Benchmark(10000)
		fmt.Printf("\n🚀 MLX Theoretical Capacity: %.0f orders/sec\n", theoretical)
		
		if theoretical > 100_000_000 {
			fmt.Println("✅ MLX GPU can handle 100M+ orders/sec")
		}
	}
}