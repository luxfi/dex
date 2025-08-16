package main

import (
	"encoding/json"
	"flag"
	"log"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

type OrderResponse struct {
	ID     uint64 `json:"id"`
	Status string `json:"status"`
}

type ServerStats struct {
	Orders    int64     `json:"orders"`
	Trades    int64     `json:"trades"`
	Timestamp time.Time `json:"timestamp"`
}

type NATSDex struct {
	nc          *nats.Conn
	ordersCount int64
	tradesCount int64
}

func main() {
	natsURL := flag.String("nats", nats.DefaultURL, "NATS server URL")
	workers := flag.Int("workers", 0, "Number of workers (0 = auto)")
	flag.Parse()

	if *workers == 0 {
		*workers = runtime.NumCPU() * 2
	}
	runtime.GOMAXPROCS(runtime.NumCPU())

	log.Printf("🚀 Starting NATS DEX Server")
	log.Printf("📡 NATS URL: %s", *natsURL)
	log.Printf("⚡ Workers: %d", *workers)
	log.Printf("🔍 Auto-discoverable via NATS!")

	// Connect to NATS
	nc, err := nats.Connect(*natsURL)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	dex := &NATSDex{nc: nc}

	// Subscribe to order submissions
	for i := 0; i < *workers; i++ {
		go dex.orderWorker()
	}

	// Subscribe to stats requests
	nc.Subscribe("dex.stats", func(m *nats.Msg) {
		stats := ServerStats{
			Orders:    atomic.LoadInt64(&dex.ordersCount),
			Trades:    atomic.LoadInt64(&dex.tradesCount),
			Timestamp: time.Now(),
		}
		data, _ := json.Marshal(stats)
		m.Respond(data)
	})

	// Announce server availability
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		
		for range ticker.C {
			announcement := map[string]interface{}{
				"type":    "dex-server",
				"status":  "ready",
				"orders":  atomic.LoadInt64(&dex.ordersCount),
				"trades":  atomic.LoadInt64(&dex.tradesCount),
				"workers": *workers,
			}
			data, _ := json.Marshal(announcement)
			nc.Publish("dex.announce", data)
			
			log.Printf("📊 Orders: %d, Trades: %d", 
				atomic.LoadInt64(&dex.ordersCount),
				atomic.LoadInt64(&dex.tradesCount))
		}
	}()

	log.Println("✅ NATS DEX Server ready!")
	log.Println("📢 Publishing on: dex.orders")
	log.Println("📊 Stats available on: dex.stats")
	log.Println("🔊 Announcing on: dex.announce")

	// Keep running
	select {}
}

func (d *NATSDex) orderWorker() {
	// Subscribe with queue group for load balancing
	d.nc.QueueSubscribe("dex.orders", "dex-workers", func(m *nats.Msg) {
		var order Order
		if err := json.Unmarshal(m.Data, &order); err != nil {
			return
		}

		// Process order
		order.ID = uint64(atomic.AddInt64(&d.ordersCount, 1))
		
		// Simulate matching (create trade 50% of the time)
		if order.ID%2 == 0 {
			atomic.AddInt64(&d.tradesCount, 1)
		}

		// Send response
		resp := OrderResponse{
			ID:     order.ID,
			Status: "accepted",
		}
		data, _ := json.Marshal(resp)
		m.Respond(data)
	})
}