package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type Order struct {
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

var (
	totalOrders   int64
	totalAccepted int64
	totalErrors   int64
	serverFound   bool
)

func main() {
	natsURL := flag.String("nats", nats.DefaultURL, "NATS server URL")
	traders := flag.Int("traders", 0, "Number of traders (0 = auto)")
	rate := flag.Int("rate", 1000, "Orders per second per trader")
	duration := flag.Duration("duration", 30*time.Second, "Test duration")
	flag.Parse()

	if *traders == 0 {
		*traders = runtime.NumCPU() * 2
	}
	runtime.GOMAXPROCS(runtime.NumCPU())

	log.Printf("⚡ NATS Trader - Auto-Discovery Mode")
	log.Printf("📡 NATS URL: %s", *natsURL)
	log.Printf("👥 Traders: %d", *traders)
	log.Printf("📈 Rate: %d orders/sec per trader", *rate)
	log.Printf("⏱️  Duration: %v", *duration)
	log.Printf("🎯 Total target: %d orders/sec", *traders**rate)

	// Connect to NATS
	nc, err := nats.Connect(*natsURL)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	// Listen for server announcements
	nc.Subscribe("dex.announce", func(m *nats.Msg) {
		if !serverFound {
			var announcement map[string]interface{}
			if json.Unmarshal(m.Data, &announcement) == nil {
				if announcement["type"] == "dex-server" {
					serverFound = true
					log.Printf("✅ Found DEX server! Status: %s, Orders: %.0f, Trades: %.0f",
						announcement["status"],
						announcement["orders"],
						announcement["trades"])
				}
			}
		}
	})

	// Wait for server discovery
	log.Println("🔍 Looking for DEX server...")
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for !serverFound {
		select {
		case <-timeout:
			log.Fatalf("❌ No DEX server found after 10 seconds")
		case <-ticker.C:
			// Keep checking
		}
	}

	log.Println("🚀 Starting traders...")

	// Start traders
	var wg sync.WaitGroup
	wg.Add(*traders)

	startTime := time.Now()
	endTime := startTime.Add(*duration)

	for i := 0; i < *traders; i++ {
		go runTrader(nc, i, *rate, endTime, &wg)
	}

	// Print stats
	go printStats(startTime)

	// Wait for completion
	wg.Wait()

	// Final stats
	elapsed := time.Since(startTime).Seconds()
	finalOrders := atomic.LoadInt64(&totalOrders)
	finalAccepted := atomic.LoadInt64(&totalAccepted)
	finalErrors := atomic.LoadInt64(&totalErrors)

	// Get server stats
	msg, err := nc.Request("dex.stats", nil, 2*time.Second)
	var serverStats map[string]interface{}
	if err == nil {
		json.Unmarshal(msg.Data, &serverStats)
	}

	fmt.Println("\n============================================================")
	fmt.Println("📊 NATS TRADER RESULTS")
	fmt.Println("============================================================")
	fmt.Printf("Duration: %.1f seconds\n", elapsed)
	fmt.Printf("Orders Sent: %d\n", finalOrders)
	fmt.Printf("Orders Accepted: %d\n", finalAccepted)
	fmt.Printf("Errors: %d\n", finalErrors)
	fmt.Printf("Success Rate: %.1f%%\n", float64(finalAccepted)*100/float64(finalOrders))
	fmt.Printf("\n📈 THROUGHPUT:\n")
	fmt.Printf("  %.0f orders/sec\n", float64(finalOrders)/elapsed)
	fmt.Printf("  %.0f accepted/sec\n", float64(finalAccepted)/elapsed)

	if serverStats != nil {
		fmt.Printf("\n📊 SERVER STATS:\n")
		fmt.Printf("  Total Orders: %.0f\n", serverStats["orders"])
		fmt.Printf("  Total Trades: %.0f\n", serverStats["trades"])
	}

	throughput := float64(finalOrders) / elapsed
	fmt.Printf("\n🏆 Performance: ")
	switch {
	case throughput >= 100000:
		fmt.Println("🥇 GOLD (100K+ orders/sec)")
	case throughput >= 50000:
		fmt.Println("🥈 SILVER (50K+ orders/sec)")
	case throughput >= 10000:
		fmt.Println("🥉 BRONZE (10K+ orders/sec)")
	default:
		fmt.Printf("%.0f orders/sec\n", throughput)
	}
}

func runTrader(nc *nats.Conn, id int, rate int, endTime time.Time, wg *sync.WaitGroup) {
	defer wg.Done()

	sleepNs := time.Duration(1000000000 / rate)

	for time.Now().Before(endTime) {
		order := Order{
			Symbol:    "BTC/USD",
			Side:      []string{"buy", "sell"}[rand.Intn(2)],
			Price:     50000 + rand.Float64()*10000,
			Quantity:  rand.Float64() * 10,
			Timestamp: time.Now(),
		}

		data, _ := json.Marshal(order)

		// Send order and wait for response
		msg, err := nc.Request("dex.orders", data, 100*time.Millisecond)
		atomic.AddInt64(&totalOrders, 1)

		if err != nil {
			atomic.AddInt64(&totalErrors, 1)
		} else {
			var resp OrderResponse
			if json.Unmarshal(msg.Data, &resp) == nil {
				if resp.Status == "accepted" {
					atomic.AddInt64(&totalAccepted, 1)
				}
			}
		}

		// Rate limiting
		if rate < 10000 {
			time.Sleep(sleepNs)
		}
	}
}

func printStats(startTime time.Time) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	lastOrders := int64(0)

	for range ticker.C {
		orders := atomic.LoadInt64(&totalOrders)
		accepted := atomic.LoadInt64(&totalAccepted)
		errors := atomic.LoadInt64(&totalErrors)

		delta := orders - lastOrders
		elapsed := time.Since(startTime).Seconds()
		avgRate := float64(orders) / elapsed

		fmt.Printf("\r📡 Orders: %d | Rate: %d/sec | Avg: %.0f/sec | Accepted: %d | Errors: %d",
			orders, delta, avgRate, accepted, errors)

		lastOrders = orders
	}
}
