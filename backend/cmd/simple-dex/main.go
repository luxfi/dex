package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type Order struct {
	ID        string    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

type OrderBook struct {
	mu          sync.RWMutex
	bids        []Order
	asks        []Order
	orders      map[string]*Order
	trades      []Trade
	ordersCount int64
	tradesCount int64
}

type Trade struct {
	ID        string    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

var orderBook = &OrderBook{
	orders: make(map[string]*Order),
}

func main() {
	port := flag.Int("port", 8080, "Server port")
	flag.Parse()

	log.Printf("🚀 Starting Simple DEX Server on port %d...", *port)

	// API endpoints
	http.HandleFunc("/order", handleOrder)
	http.HandleFunc("/orders", handleOrders)
	http.HandleFunc("/trades", handleTrades)
	http.HandleFunc("/stats", handleStats)
	http.HandleFunc("/health", handleHealth)

	// Start stats printer
	go printStats()

	log.Printf("✅ DEX Server ready at http://localhost:%d", *port)
	log.Printf("📊 Endpoints: /order (POST), /orders (GET), /trades (GET), /stats (GET)")
	
	if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
		log.Fatal(err)
	}
}

func handleOrder(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var order Order
	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	order.ID = fmt.Sprintf("ORD%d", time.Now().UnixNano())
	order.Timestamp = time.Now()

	orderBook.mu.Lock()
	orderBook.orders[order.ID] = &order
	atomic.AddInt64(&orderBook.ordersCount, 1)
	
	// Simple matching logic
	if order.Side == "buy" {
		orderBook.bids = append(orderBook.bids, order)
		// Check for matches with asks
		for i, ask := range orderBook.asks {
			if ask.Price <= order.Price && ask.Quantity > 0 && order.Quantity > 0 {
				// Create trade
				trade := Trade{
					ID:        fmt.Sprintf("TRD%d", time.Now().UnixNano()),
					Symbol:    order.Symbol,
					Price:     ask.Price,
					Quantity:  min(order.Quantity, ask.Quantity),
					Timestamp: time.Now(),
				}
				orderBook.trades = append(orderBook.trades, trade)
				atomic.AddInt64(&orderBook.tradesCount, 1)
				
				// Update quantities
				order.Quantity -= trade.Quantity
				orderBook.asks[i].Quantity -= trade.Quantity
			}
		}
	} else {
		orderBook.asks = append(orderBook.asks, order)
		// Check for matches with bids
		for i, bid := range orderBook.bids {
			if bid.Price >= order.Price && bid.Quantity > 0 && order.Quantity > 0 {
				// Create trade
				trade := Trade{
					ID:        fmt.Sprintf("TRD%d", time.Now().UnixNano()),
					Symbol:    order.Symbol,
					Price:     order.Price,
					Quantity:  min(order.Quantity, bid.Quantity),
					Timestamp: time.Now(),
				}
				orderBook.trades = append(orderBook.trades, trade)
				atomic.AddInt64(&orderBook.tradesCount, 1)
				
				// Update quantities
				order.Quantity -= trade.Quantity
				orderBook.bids[i].Quantity -= trade.Quantity
			}
		}
	}
	orderBook.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(order)
}

func handleOrders(w http.ResponseWriter, r *http.Request) {
	orderBook.mu.RLock()
	orders := make([]Order, 0, len(orderBook.orders))
	for _, order := range orderBook.orders {
		orders = append(orders, *order)
	}
	orderBook.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(orders)
}

func handleTrades(w http.ResponseWriter, r *http.Request) {
	orderBook.mu.RLock()
	trades := orderBook.trades
	orderBook.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(trades)
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"orders": atomic.LoadInt64(&orderBook.ordersCount),
		"trades": atomic.LoadInt64(&orderBook.tradesCount),
		"timestamp": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func printStats() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		orders := atomic.LoadInt64(&orderBook.ordersCount)
		trades := atomic.LoadInt64(&orderBook.tradesCount)
		log.Printf("📊 Stats: Orders=%d, Trades=%d", orders, trades)
	}
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}