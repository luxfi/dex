package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"runtime"
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

var (
	totalOrders    int64
	totalErrors    int64
	totalBatches   int64
	totalRejected  int64
)

func main() {
	serverURL := flag.String("server", "http://localhost:8080", "Server URL")
	workersPerCore := flag.Int("workers", 10, "Workers per CPU core")
	batchSize := flag.Int("batch", 1000, "Orders per batch")
	duration := flag.Duration("duration", 20*time.Second, "Test duration")
	flag.Parse()

	numCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPU)
	numWorkers := numCPU * *workersPerCore

	log.Printf("⚡ DEX TRADER - High Performance Mode")
	log.Printf("⚡ CPU Cores: %d", numCPU)
	log.Printf("⚡ Workers: %d (%d per core)", numWorkers, *workersPerCore)
	log.Printf("⚡ Batch Size: %d orders", *batchSize)
	log.Printf("⚡ Target: %d orders/batch × %d workers", *batchSize, numWorkers)
	log.Printf("⚡ Server: %s", *serverURL)

	// Test server
	resp, err := http.Get(*serverURL + "/health")
	if err != nil {
		log.Fatalf("❌ Server not reachable: %v", err)
	}
	resp.Body.Close()

	// Shared HTTP client with massive connection pool
	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        numWorkers * 10,
			MaxIdleConnsPerHost: numWorkers * 10,
			MaxConnsPerHost:     numWorkers * 10,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  true,
		},
		Timeout: 10 * time.Second,
	}

	// Generate batch data once
	orders := generateBatch(*batchSize)
	batchData, _ := json.Marshal(orders)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	startTime := time.Now()
	stopTime := startTime.Add(*duration)

	// Launch workers
	for i := 0; i < numWorkers; i++ {
		go hammerWorker(i, client, *serverURL, batchData, stopTime, &wg)
	}

	// Stats printer
	go printStats(startTime)

	// Wait for completion
	wg.Wait()

	// Final results
	elapsed := time.Since(startTime).Seconds()
	finalOrders := atomic.LoadInt64(&totalOrders)
	finalErrors := atomic.LoadInt64(&totalErrors)
	finalBatches := atomic.LoadInt64(&totalBatches)
	finalRejected := atomic.LoadInt64(&totalRejected)

	fmt.Println("\n" + "============================================================")
	fmt.Println("⚡ DEX TRADER RESULTS")
	fmt.Println("============================================================")
	fmt.Printf("Duration: %.1f seconds\n", elapsed)
	fmt.Printf("Batches Sent: %d\n", finalBatches)
	fmt.Printf("Orders Accepted: %d\n", finalOrders)
	fmt.Printf("Orders Rejected: %d\n", finalRejected)
	fmt.Printf("Network Errors: %d\n", finalErrors)
	fmt.Printf("Success Rate: %.1f%%\n", float64(finalOrders)*100/float64(finalOrders+finalRejected+finalErrors))
	fmt.Printf("\n📊 THROUGHPUT:\n")
	fmt.Printf("  %.0f orders/sec\n", float64(finalOrders)/elapsed)
	fmt.Printf("  %.0f batches/sec\n", float64(finalBatches)/elapsed)
	fmt.Printf("  %.0f orders/sec/core\n", float64(finalOrders)/elapsed/float64(numCPU))
	
	rate := float64(finalOrders) / elapsed
	fmt.Printf("\n🏆 Performance: ")
	switch {
	case rate >= 1000000:
		fmt.Println("🌟 LEGENDARY (1M+ orders/sec)")
	case rate >= 500000:
		fmt.Println("💎 DIAMOND (500K+ orders/sec)")
	case rate >= 100000:
		fmt.Println("🥇 GOLD (100K+ orders/sec)")
	case rate >= 50000:
		fmt.Println("🥈 SILVER (50K+ orders/sec)")
	case rate >= 10000:
		fmt.Println("🥉 BRONZE (10K+ orders/sec)")
	default:
		fmt.Printf("%.0f orders/sec\n", rate)
	}
}

func hammerWorker(id int, client *http.Client, serverURL string, batchData []byte, stopTime time.Time, wg *sync.WaitGroup) {
	defer wg.Done()

	url := serverURL + "/orders/batch"
	
	for time.Now().Before(stopTime) {
		// Fire request
		resp, err := client.Post(url, "application/json", bytes.NewReader(batchData))
		if err != nil {
			atomic.AddInt64(&totalErrors, 1)
			continue
		}

		// Read response
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		atomic.AddInt64(&totalBatches, 1)

		// Parse result
		var result map[string]interface{}
		if json.Unmarshal(body, &result) == nil {
			if accepted, ok := result["accepted"].(float64); ok {
				atomic.AddInt64(&totalOrders, int64(accepted))
			}
			if rejected, ok := result["rejected"].(float64); ok {
				atomic.AddInt64(&totalRejected, int64(rejected))
			}
		}
		
		// No delay - hammer continuously
	}
}

func generateBatch(size int) []Order {
	orders := make([]Order, size)
	for i := 0; i < size; i++ {
		orders[i] = Order{
			Symbol:   "BTC/USD",
			Side:     []string{"buy", "sell"}[rand.Intn(2)],
			Price:    50000 + rand.Float64()*10000,
			Quantity: rand.Float64() * 10,
		}
	}
	return orders
}

func printStats(startTime time.Time) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	lastOrders := int64(0)
	
	for range ticker.C {
		orders := atomic.LoadInt64(&totalOrders)
		batches := atomic.LoadInt64(&totalBatches)
		rejected := atomic.LoadInt64(&totalRejected)
		errors := atomic.LoadInt64(&totalErrors)
		
		delta := orders - lastOrders
		elapsed := time.Since(startTime).Seconds()
		avgRate := float64(orders) / elapsed
		
		fmt.Printf("\r🔨 Orders: %d | Rate: %d/sec | Avg: %.0f/sec | Batches: %d | Rejected: %d | Errors: %d",
			orders, delta, avgRate, batches, rejected, errors)
		
		lastOrders = orders
	}
}