package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Order struct {
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Price    float64 `json:"price"`
	Quantity float64 `json:"quantity"`
}

type TurboTrader struct {
	id           int
	serverURL    string
	ordersCount  int64
	errorCount   int64
	client       *http.Client
	orderBuffer  []Order
	jsonBuffer   []byte
}

var (
	totalOrders int64
	totalErrors int64
	startTime   time.Time
)

func main() {
	serverURL := flag.String("server", "http://localhost:8080", "DEX server URL")
	tradersPerCore := flag.Int("traders-per-core", 2, "Traders per CPU core")
	batchSize := flag.Int("batch", 100, "Orders per batch")
	duration := flag.Duration("duration", 30*time.Second, "Test duration")
	noDelay := flag.Bool("no-delay", true, "Remove all delays for max throughput")
	useBatch := flag.Bool("use-batch", false, "Use batch order submission")
	connections := flag.Int("connections", 100, "Max connections per trader")
	flag.Parse()

	// Auto-detect CPUs and maximize usage
	numCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPU)
	numTraders := numCPU * *tradersPerCore

	log.Printf("🚀 Starting TURBO Trader")
	log.Printf("⚡ CPU Cores: %d", numCPU)
	log.Printf("⚡ Traders: %d (%d per core)", numTraders, *tradersPerCore)
	log.Printf("⚡ Batch Size: %d", *batchSize)
	log.Printf("⚡ Duration: %v", *duration)
	log.Printf("⚡ Server: %s", *serverURL)
	log.Printf("⚡ Max Connections: %d", *connections)
	if *noDelay {
		log.Printf("🔥 NO DELAY MODE - Maximum throughput!")
	}
	if *useBatch {
		log.Printf("📦 BATCH MODE - Sending orders in batches")
	}

	// Check server health
	resp, err := http.Get(*serverURL + "/health")
	if err != nil {
		log.Fatalf("❌ Server not reachable: %v", err)
	}
	resp.Body.Close()
	log.Println("✅ Server is healthy")

	// Create high-performance HTTP transport
	transport := &http.Transport{
		MaxIdleConns:        numTraders * *connections,
		MaxIdleConnsPerHost: *connections,
		MaxConnsPerHost:     *connections,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
		DisableKeepAlives:   false,
	}

	// Create traders with dedicated HTTP clients
	traders := make([]*TurboTrader, numTraders)
	for i := 0; i < numTraders; i++ {
		traders[i] = &TurboTrader{
			id:        i,
			serverURL: *serverURL,
			client: &http.Client{
				Transport: transport,
				Timeout:   5 * time.Second,
			},
			orderBuffer: make([]Order, *batchSize),
			jsonBuffer:  make([]byte, 0, 1024*10), // 10KB buffer
		}
	}

	// Start traders
	var wg sync.WaitGroup
	wg.Add(numTraders)
	
	startTime = time.Now()
	
	for _, trader := range traders {
		if *useBatch {
			go trader.runBatch(*batchSize, *noDelay, *duration, &wg)
		} else {
			go trader.runSingle(*noDelay, *duration, &wg)
		}
	}

	// Real-time stats printer
	go printDetailedStats(numTraders)

	// Wait for duration
	time.Sleep(*duration)
	
	// Signal stop (traders will stop on their own due to duration)
	log.Println("\n⏱️  Stopping traders...")
	
	// Wait for all traders to finish
	wg.Wait()
	
	// Collect final stats
	totalOrders = 0
	totalErrors = 0
	for _, trader := range traders {
		totalOrders += atomic.LoadInt64(&trader.ordersCount)
		totalErrors += atomic.LoadInt64(&trader.errorCount)
	}
	
	elapsed := time.Since(startTime).Seconds()
	
	// Print final results
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("📊 FINAL TURBO RESULTS")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("Duration: %.1f seconds\n", elapsed)
	fmt.Printf("Total Orders: %d\n", totalOrders)
	fmt.Printf("Total Errors: %d\n", totalErrors)
	fmt.Printf("Average Rate: %.0f orders/sec\n", float64(totalOrders)/elapsed)
	fmt.Printf("Per Trader: %.0f orders/sec\n", float64(totalOrders)/elapsed/float64(numTraders))
	fmt.Printf("Per Core: %.0f orders/sec\n", float64(totalOrders)/elapsed/float64(numCPU))
	
	// Get server stats
	resp, err = http.Get(*serverURL + "/stats")
	if err == nil {
		var stats map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&stats)
		resp.Body.Close()
		fmt.Printf("\nServer Stats:\n")
		fmt.Printf("  Orders: %v\n", stats["orders"])
		fmt.Printf("  Trades: %v\n", stats["trades"])
	}
	
	// Performance grade
	rate := float64(totalOrders) / elapsed
	fmt.Printf("\n🏆 Performance Grade: ")
	switch {
	case rate >= 100000:
		fmt.Println("💎 DIAMOND (100K+ orders/sec)")
	case rate >= 50000:
		fmt.Println("🥇 GOLD (50K+ orders/sec)")
	case rate >= 10000:
		fmt.Println("🥈 SILVER (10K+ orders/sec)")
	case rate >= 5000:
		fmt.Println("🥉 BRONZE (5K+ orders/sec)")
	default:
		fmt.Printf("📈 %.0f orders/sec\n", rate)
	}
}

func (t *TurboTrader) runSingle(noDelay bool, duration time.Duration, wg *sync.WaitGroup) {
	defer wg.Done()
	
	endTime := time.Now().Add(duration)
	
	// Pre-create order template
	order := Order{
		Symbol: "BTC/USD",
	}
	
	for time.Now().Before(endTime) {
		// Fast order generation
		order.Side = []string{"buy", "sell"}[rand.Intn(2)]
		order.Price = 50000 + rand.Float64()*10000
		order.Quantity = rand.Float64() * 10
		
		data, _ := json.Marshal(order)
		
		// Fast HTTP POST
		resp, err := t.client.Post(
			t.serverURL+"/order",
			"application/json",
			bytes.NewBuffer(data),
		)
		
		if err != nil {
			atomic.AddInt64(&t.errorCount, 1)
			atomic.AddInt64(&totalErrors, 1)
			continue
		}
		
		resp.Body.Close()
		
		if resp.StatusCode == http.StatusOK {
			atomic.AddInt64(&t.ordersCount, 1)
			atomic.AddInt64(&totalOrders, 1)
		} else {
			atomic.AddInt64(&t.errorCount, 1)
			atomic.AddInt64(&totalErrors, 1)
		}
		
		// No delay for maximum throughput
		if !noDelay {
			time.Sleep(time.Microsecond) // Tiny delay if needed
		}
	}
}

func (t *TurboTrader) runBatch(batchSize int, noDelay bool, duration time.Duration, wg *sync.WaitGroup) {
	defer wg.Done()
	
	endTime := time.Now().Add(duration)
	
	for time.Now().Before(endTime) {
		// Generate batch of orders
		for i := 0; i < batchSize; i++ {
			t.orderBuffer[i] = Order{
				Symbol:   "BTC/USD",
				Side:     []string{"buy", "sell"}[rand.Intn(2)],
				Price:    50000 + rand.Float64()*10000,
				Quantity: rand.Float64() * 10,
			}
		}
		
		data, _ := json.Marshal(t.orderBuffer[:batchSize])
		
		// Send batch
		resp, err := t.client.Post(
			t.serverURL+"/orders/batch",
			"application/json",
			bytes.NewBuffer(data),
		)
		
		if err != nil {
			atomic.AddInt64(&t.errorCount, int64(batchSize))
			atomic.AddInt64(&totalErrors, int64(batchSize))
			continue
		}
		
		// Parse response
		var result map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		
		if accepted, ok := result["accepted"].(float64); ok {
			atomic.AddInt64(&t.ordersCount, int64(accepted))
			atomic.AddInt64(&totalOrders, int64(accepted))
		}
		
		if rejected, ok := result["rejected"].(float64); ok {
			atomic.AddInt64(&t.errorCount, int64(rejected))
			atomic.AddInt64(&totalErrors, int64(rejected))
		}
		
		if !noDelay {
			time.Sleep(time.Microsecond)
		}
	}
}

func printDetailedStats(numTraders int) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	lastOrders := int64(0)
	lastErrors := int64(0)
	
	for range ticker.C {
		orders := atomic.LoadInt64(&totalOrders)
		errors := atomic.LoadInt64(&totalErrors)
		
		ordersDelta := orders - lastOrders
		errorsDelta := errors - lastErrors
		
		elapsed := time.Since(startTime).Seconds()
		avgRate := float64(orders) / elapsed
		
		fmt.Printf("\r⚡ Orders: %d | Rate: %d/sec | Avg: %.0f/sec | Errors: %d/sec | Total Errors: %d",
			orders, ordersDelta, avgRate, errorsDelta, errors)
		
		lastOrders = orders
		lastErrors = errors
	}
}