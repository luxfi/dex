package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Lock-free order structure
type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

// Shard for parallel processing
type Shard struct {
	orders      []Order
	trades      []Trade
	ordersCount int64
	tradesCount int64
	mu          sync.RWMutex
}

type Trade struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

type TurboDEX struct {
	shards       []*Shard
	numShards    int
	ordersTotal  int64
	tradesTotal  int64
	orderIDGen   uint64
	tradeIDGen   uint64
	workers      int
	orderChannel chan Order
	tradeChannel chan Trade
}

var turboDEX *TurboDEX

func main() {
	port := flag.Int("port", 8080, "Server port")
	workers := flag.Int("workers", 0, "Number of workers (0 = auto-detect CPUs)")
	shards := flag.Int("shards", 0, "Number of shards (0 = 4x CPUs)")
	bufferSize := flag.Int("buffer", 100000, "Channel buffer size")
	flag.Parse()

	// Auto-detect optimal settings
	numCPU := runtime.NumCPU()
	if *workers == 0 {
		*workers = numCPU * 2 // Oversubscribe for I/O
	}
	if *shards == 0 {
		*shards = numCPU * 4 // More shards for less contention
	}

	runtime.GOMAXPROCS(numCPU)

	log.Printf("🚀 Starting TURBO DEX Server")
	log.Printf("⚡ CPU Cores: %d", numCPU)
	log.Printf("⚡ Workers: %d", *workers)
	log.Printf("⚡ Shards: %d", *shards)
	log.Printf("⚡ Buffer Size: %d", *bufferSize)
	log.Printf("⚡ Port: %d", *port)

	// Initialize TurboDEX
	turboDEX = &TurboDEX{
		shards:       make([]*Shard, *shards),
		numShards:    *shards,
		workers:      *workers,
		orderChannel: make(chan Order, *bufferSize),
		tradeChannel: make(chan Trade, *bufferSize),
	}

	// Initialize shards
	for i := 0; i < *shards; i++ {
		turboDEX.shards[i] = &Shard{}
	}

	// Start worker pool for order processing
	for i := 0; i < *workers; i++ {
		go orderProcessor(i)
	}

	// Start background stats printer
	go printStats()

	// HTTP handlers - minimal overhead
	http.HandleFunc("/order", handleOrderTurbo)
	http.HandleFunc("/orders/batch", handleBatchOrders)
	http.HandleFunc("/stats", handleStatsTurbo)
	http.HandleFunc("/health", handleHealthTurbo)
	http.HandleFunc("/reset", handleReset)

	log.Printf("✅ TURBO DEX ready at http://localhost:%d", *port)
	log.Printf("🔥 Optimized for maximum throughput!")
	
	// Use larger buffers for HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", *port),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1MB
	}
	
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func orderProcessor(id int) {
	for order := range turboDEX.orderChannel {
		// Determine shard based on order ID
		shardID := order.ID % uint64(turboDEX.numShards)
		shard := turboDEX.shards[shardID]

		// Fast processing without heavy locking
		shard.mu.Lock()
		shard.orders = append(shard.orders, order)
		atomic.AddInt64(&shard.ordersCount, 1)
		
		// Simple matching (can be optimized further)
		if len(shard.orders) > 1 {
			// Create synthetic trades for demo
			if atomic.LoadInt64(&shard.ordersCount)%2 == 0 {
				trade := Trade{
					ID:        atomic.AddUint64(&turboDEX.tradeIDGen, 1),
					Symbol:    order.Symbol,
					Price:     order.Price,
					Quantity:  order.Quantity,
					Timestamp: time.Now(),
				}
				shard.trades = append(shard.trades, trade)
				atomic.AddInt64(&shard.tradesCount, 1)
				atomic.AddInt64(&turboDEX.tradesTotal, 1)
			}
		}
		shard.mu.Unlock()

		atomic.AddInt64(&turboDEX.ordersTotal, 1)
	}
}

func handleOrderTurbo(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var order Order
	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Generate order ID atomically
	order.ID = atomic.AddUint64(&turboDEX.orderIDGen, 1)
	order.Timestamp = time.Now()

	// Non-blocking send to channel
	select {
	case turboDEX.orderChannel <- order:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id": order.ID,
			"status": "accepted",
		})
	default:
		// Channel full, reject order
		http.Error(w, "System overloaded", http.StatusServiceUnavailable)
	}
}

func handleBatchOrders(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var orders []Order
	if err := json.NewDecoder(r.Body).Decode(&orders); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	accepted := 0
	rejected := 0
	
	for i := range orders {
		orders[i].ID = atomic.AddUint64(&turboDEX.orderIDGen, 1)
		orders[i].Timestamp = time.Now()
		
		select {
		case turboDEX.orderChannel <- orders[i]:
			accepted++
		default:
			rejected++
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"accepted": accepted,
		"rejected": rejected,
		"total": len(orders),
	})
}

func handleStatsTurbo(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"orders": atomic.LoadInt64(&turboDEX.ordersTotal),
		"trades": atomic.LoadInt64(&turboDEX.tradesTotal),
		"shards": turboDEX.numShards,
		"workers": turboDEX.workers,
		"channel_size": len(turboDEX.orderChannel),
		"channel_cap": cap(turboDEX.orderChannel),
		"timestamp": time.Now(),
	}

	// Add per-shard stats
	shardStats := make([]map[string]int64, turboDEX.numShards)
	for i, shard := range turboDEX.shards {
		shardStats[i] = map[string]int64{
			"orders": atomic.LoadInt64(&shard.ordersCount),
			"trades": atomic.LoadInt64(&shard.tradesCount),
		}
	}
	stats["shard_stats"] = shardStats

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func handleHealthTurbo(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	// Reset all counters
	atomic.StoreInt64(&turboDEX.ordersTotal, 0)
	atomic.StoreInt64(&turboDEX.tradesTotal, 0)
	
	for _, shard := range turboDEX.shards {
		shard.mu.Lock()
		shard.orders = nil
		shard.trades = nil
		atomic.StoreInt64(&shard.ordersCount, 0)
		atomic.StoreInt64(&shard.tradesCount, 0)
		shard.mu.Unlock()
	}
	
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Reset complete"))
}

func printStats() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	lastOrders := int64(0)
	lastTrades := int64(0)
	lastTime := time.Now()

	for range ticker.C {
		orders := atomic.LoadInt64(&turboDEX.ordersTotal)
		trades := atomic.LoadInt64(&turboDEX.tradesTotal)
		
		now := time.Now()
		elapsed := now.Sub(lastTime).Seconds()
		
		orderRate := float64(orders - lastOrders) / elapsed
		tradeRate := float64(trades - lastTrades) / elapsed
		
		channelUsage := float64(len(turboDEX.orderChannel)) / float64(cap(turboDEX.orderChannel)) * 100
		
		log.Printf("📊 Stats: Orders=%d (%.0f/sec), Trades=%d (%.0f/sec), Channel=%.1f%%", 
			orders, orderRate, trades, tradeRate, channelUsage)
		
		lastOrders = orders
		lastTrades = trades
		lastTime = now
	}
}