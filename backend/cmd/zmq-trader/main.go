// ZeroMQ Trader - Sends orders to exchange via ZMQ
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	zmq "github.com/pebbe/zmq4"
)

type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Side      string    `json:"side"`
	Timestamp time.Time `json:"timestamp"`
}

type TraderStats struct {
	OrdersSent  uint64
	BytesSent   uint64
	Errors      uint64
	LastOrderID uint64
}

func runTrader(id int, serverAddr string, ordersPerSec int, stats *TraderStats, wg *sync.WaitGroup) {
	defer wg.Done()

	context, _ := zmq.NewContext()
	socket, _ := context.NewSocket(zmq.PUSH)
	defer socket.Close()

	// Set high water mark
	socket.SetSndhwm(100000)
	socket.SetSndbuf(8 * 1024 * 1024) // 8MB buffer

	if err := socket.Connect(serverAddr); err != nil {
		log.Printf("Trader %d failed to connect: %v", id, err)
		return
	}

	symbols := []string{"BTC", "ETH", "SOL", "AVAX", "MATIC"}
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))

	ticker := time.NewTicker(time.Second / time.Duration(ordersPerSec))
	defer ticker.Stop()

	for range ticker.C {
		orderID := atomic.AddUint64(&stats.LastOrderID, 1)

		order := Order{
			ID:        orderID,
			Symbol:    symbols[rng.Intn(len(symbols))],
			Price:     100 + rng.Float64()*10,
			Quantity:  rng.Float64() * 100,
			Side:      []string{"buy", "sell"}[rng.Intn(2)],
			Timestamp: time.Now(),
		}

		data, err := json.Marshal(order)
		if err != nil {
			atomic.AddUint64(&stats.Errors, 1)
			continue
		}

		if _, err := socket.SendBytes(data, zmq.DONTWAIT); err != nil {
			atomic.AddUint64(&stats.Errors, 1)
		} else {
			atomic.AddUint64(&stats.OrdersSent, 1)
			atomic.AddUint64(&stats.BytesSent, uint64(len(data)))
		}
	}
}

func main() {
	var (
		serverAddr   = flag.String("server", "tcp://localhost:5555", "ZMQ server address")
		traders      = flag.Int("traders", 100, "Number of trader threads")
		ordersPerSec = flag.Int("rate", 100, "Orders per second per trader")
		duration     = flag.Duration("duration", 30*time.Second, "Test duration")
		verbose      = flag.Bool("v", false, "Verbose output")
	)
	flag.Parse()

	totalRate := *traders * *ordersPerSec

	fmt.Printf("🚀 ZMQ Trader Client\n")
	fmt.Printf("Server: %s\n", *serverAddr)
	fmt.Printf("Traders: %d\n", *traders)
	fmt.Printf("Rate per trader: %d orders/sec\n", *ordersPerSec)
	fmt.Printf("Total target rate: %d orders/sec\n", totalRate)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Println("----------------------------------------")

	stats := &TraderStats{}
	var wg sync.WaitGroup

	// Start traders
	startTime := time.Now()
	for i := 0; i < *traders; i++ {
		wg.Add(1)
		go runTrader(i, *serverAddr, *ordersPerSec, stats, &wg)
	}

	// Stats printer
	done := make(chan bool)
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				elapsed := time.Since(startTime).Seconds()
				sent := atomic.LoadUint64(&stats.OrdersSent)
				bytes := atomic.LoadUint64(&stats.BytesSent)
				errors := atomic.LoadUint64(&stats.Errors)

				ordersPerSec := float64(sent) / elapsed
				mbPerSec := float64(bytes) / (elapsed * 1024 * 1024)
				gbpsUsed := (float64(bytes) * 8) / (elapsed * 1e9)
				efficiency := (ordersPerSec / float64(totalRate)) * 100

				fmt.Printf("\r📈 Sent: %d | Rate: %.0f/sec (%.1f%% efficiency) | Network: %.2f MB/s (%.3f Gbps) | Errors: %d",
					sent, ordersPerSec, efficiency, mbPerSec, gbpsUsed, errors)

				if *verbose && sent%10000 == 0 {
					fmt.Printf("\n[Checkpoint] Orders sent: %d\n", sent)
				}
			case <-done:
				return
			}
		}
	}()

	// Wait for duration
	time.Sleep(*duration)
	close(done)

	// Final stats
	finalSent := atomic.LoadUint64(&stats.OrdersSent)
	finalBytes := atomic.LoadUint64(&stats.BytesSent)
	finalErrors := atomic.LoadUint64(&stats.Errors)
	totalElapsed := time.Since(startTime).Seconds()

	fmt.Printf("\n\n=== Final Results ===\n")
	fmt.Printf("Duration: %.2f seconds\n", totalElapsed)
	fmt.Printf("Orders Sent: %d\n", finalSent)
	fmt.Printf("Errors: %d (%.2f%%)\n", finalErrors, float64(finalErrors)/float64(finalSent+finalErrors)*100)
	fmt.Printf("Average Rate: %.0f orders/sec\n", float64(finalSent)/totalElapsed)
	fmt.Printf("Total Data: %.2f MB\n", float64(finalBytes)/(1024*1024))
	fmt.Printf("Network Usage: %.3f Gbps\n", (float64(finalBytes)*8)/(totalElapsed*1e9))
	fmt.Printf("Efficiency: %.1f%%\n", (float64(finalSent)/totalElapsed/float64(totalRate))*100)

	if finalSent > 0 {
		fmt.Printf("Average Message Size: %d bytes\n", finalBytes/finalSent)
		fmt.Printf("Average Latency: %.2f ms (estimated)\n", float64(*traders)/(float64(finalSent)/totalElapsed)*1000)
	}
}
