// ZeroMQ Exchange Server - Receives orders via ZMQ and processes them
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"sync/atomic"
	"time"

	zmq "github.com/pebbe/zmq4"
)

type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Side      string    `json:"side"` // "buy" or "sell"
	Timestamp time.Time `json:"timestamp"`
}

type Stats struct {
	OrdersReceived uint64
	BytesReceived  uint64
	StartTime      time.Time
	LastPrint      time.Time
}

func main() {
	var (
		bindAddr = flag.String("bind", "tcp://*:5555", "ZMQ bind address")
		workers  = flag.Int("workers", 4, "Number of worker threads")
		verbose  = flag.Bool("v", false, "Verbose output")
	)
	flag.Parse()

	fmt.Printf("🚀 ZMQ Exchange Server\n")
	fmt.Printf("Binding to: %s\n", *bindAddr)
	fmt.Printf("Workers: %d\n", *workers)
	fmt.Println("----------------------------------------")

	// Create PULL socket for receiving orders
	context, _ := zmq.NewContext()
	socket, _ := context.NewSocket(zmq.PULL)
	defer socket.Close()

	// Set high water mark for better throughput
	socket.SetRcvhwm(100000)
	socket.SetRcvbuf(8 * 1024 * 1024) // 8MB buffer

	if err := socket.Bind(*bindAddr); err != nil {
		log.Fatalf("Failed to bind: %v", err)
	}

	stats := &Stats{
		StartTime: time.Now(),
		LastPrint: time.Now(),
	}

	// Stats printer
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			elapsed := time.Since(stats.StartTime).Seconds()
			orders := atomic.LoadUint64(&stats.OrdersReceived)
			bytes := atomic.LoadUint64(&stats.BytesReceived)

			ordersPerSec := float64(orders) / elapsed
			mbPerSec := float64(bytes) / (elapsed * 1024 * 1024)
			gbpsUsed := (float64(bytes) * 8) / (elapsed * 1e9)

			fmt.Printf("\r📊 Orders: %d | Rate: %.0f/sec | Network: %.2f MB/s (%.3f Gbps) | Avg Size: %d bytes",
				orders, ordersPerSec, mbPerSec, gbpsUsed,
				func() uint64 {
					if orders > 0 {
						return bytes / orders
					}
					return 0
				}())
		}
	}()

	fmt.Println("Waiting for orders...")

	// Main receive loop
	for {
		msg, err := socket.RecvBytes(0)
		if err != nil {
			log.Printf("Receive error: %v", err)
			continue
		}

		atomic.AddUint64(&stats.OrdersReceived, 1)
		atomic.AddUint64(&stats.BytesReceived, uint64(len(msg)))

		if *verbose {
			var order Order
			if err := json.Unmarshal(msg, &order); err == nil {
				if atomic.LoadUint64(&stats.OrdersReceived)%10000 == 0 {
					fmt.Printf("\nSample Order: %+v\n", order)
				}
			}
		}

		// In real implementation, would process order here
		// For benchmark, we just count
	}
}
