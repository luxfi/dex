package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	grpcEndpoint = flag.String("grpc", "localhost:50051", "gRPC server endpoint")
	numClients   = flag.Int("clients", 10, "Number of concurrent clients")
	numOrders    = flag.Int("orders", 1000, "Number of orders per client")
	duration     = flag.Duration("duration", 30*time.Second, "Test duration")
	symbol       = flag.String("symbol", "BTC-USD", "Trading symbol")
)

type Stats struct {
	submitted  int64
	cancelled  int64
	errors     int64
	latencySum int64
	latencyCount int64
}

func (s *Stats) RecordLatency(ns int64) {
	atomic.AddInt64(&s.latencySum, ns)
	atomic.AddInt64(&s.latencyCount, 1)
}

func (s *Stats) AverageLatency() time.Duration {
	count := atomic.LoadInt64(&s.latencyCount)
	if count == 0 {
		return 0
	}
	sum := atomic.LoadInt64(&s.latencySum)
	return time.Duration(sum / count)
}

func runClient(clientID int, stats *Stats, wg *sync.WaitGroup) {
	defer wg.Done()

	// Connect to gRPC server
	conn, err := grpc.Dial(*grpcEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Client %d: Failed to connect: %v", clientID, err)
		return
	}
	defer conn.Close()

	client := pb.NewEngineServiceClient(conn)
	ctx := context.Background()

	// Random generator for this client
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(clientID)))
	
	orderIDs := make([]string, 0, *numOrders)
	
	// Submit orders
	for i := 0; i < *numOrders; i++ {
		side := pb.OrderSide_ORDER_SIDE_BUY
		if rng.Float64() > 0.5 {
			side = pb.OrderSide_ORDER_SIDE_SELL
		}
		
		price := 40000.0 + rng.Float64()*10000.0 // $40k-$50k
		quantity := 0.01 + rng.Float64()*0.99     // 0.01-1.0 BTC
		
		start := time.Now()
		resp, err := client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
			Symbol:        *symbol,
			Side:          side,
			Type:          pb.OrderType_ORDER_TYPE_LIMIT,
			Quantity:      quantity,
			Price:         price,
			ClientOrderId: fmt.Sprintf("client-%d-order-%d", clientID, i),
		})
		latency := time.Since(start).Nanoseconds()
		stats.RecordLatency(latency)
		
		if err != nil {
			atomic.AddInt64(&stats.errors, 1)
		} else {
			atomic.AddInt64(&stats.submitted, 1)
			orderIDs = append(orderIDs, resp.OrderId)
		}
		
		// Random delay between orders (0-10ms)
		time.Sleep(time.Duration(rng.Intn(10)) * time.Millisecond)
	}
	
	// Cancel some orders
	for i := 0; i < len(orderIDs)/10; i++ { // Cancel 10% of orders
		idx := rng.Intn(len(orderIDs))
		start := time.Now()
		_, err := client.CancelOrder(ctx, &pb.CancelOrderRequest{
			OrderId: orderIDs[idx],
		})
		latency := time.Since(start).Nanoseconds()
		stats.RecordLatency(latency)
		
		if err != nil {
			atomic.AddInt64(&stats.errors, 1)
		} else {
			atomic.AddInt64(&stats.cancelled, 1)
		}
	}
}

func main() {
	flag.Parse()

	fmt.Println("=== LX ENGINE LOAD TEST ===")
	fmt.Printf("Endpoint: %s\n", *grpcEndpoint)
	fmt.Printf("Clients: %d\n", *numClients)
	fmt.Printf("Orders per client: %d\n", *numOrders)
	fmt.Printf("Total orders: %d\n", *numClients * *numOrders)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Println()

	stats := &Stats{}
	var wg sync.WaitGroup

	// Start monitoring
	done := make(chan bool)
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				submitted := atomic.LoadInt64(&stats.submitted)
				cancelled := atomic.LoadInt64(&stats.cancelled)
				errors := atomic.LoadInt64(&stats.errors)
				avgLatency := stats.AverageLatency()
				
				fmt.Printf("[%s] Submitted: %d, Cancelled: %d, Errors: %d, Avg Latency: %v\n",
					time.Now().Format("15:04:05"),
					submitted, cancelled, errors, avgLatency)
			case <-done:
				return
			}
		}
	}()

	// Run test
	start := time.Now()
	
	for i := 0; i < *numClients; i++ {
		wg.Add(1)
		go runClient(i, stats, &wg)
	}
	
	// Wait for completion or timeout
	go func() {
		wg.Wait()
		done <- true
	}()
	
	select {
	case <-done:
		fmt.Println("\nAll clients completed")
	case <-time.After(*duration):
		fmt.Println("\nTest duration reached")
	}
	
	// Final stats
	elapsed := time.Since(start)
	submitted := atomic.LoadInt64(&stats.submitted)
	cancelled := atomic.LoadInt64(&stats.cancelled)
	errors := atomic.LoadInt64(&stats.errors)
	avgLatency := stats.AverageLatency()
	
	fmt.Println("\n=== FINAL RESULTS ===")
	fmt.Printf("Duration: %v\n", elapsed)
	fmt.Printf("Orders Submitted: %d\n", submitted)
	fmt.Printf("Orders Cancelled: %d\n", cancelled)
	fmt.Printf("Errors: %d\n", errors)
	fmt.Printf("Average Latency: %v\n", avgLatency)
	fmt.Printf("Throughput: %.0f orders/second\n", float64(submitted)/elapsed.Seconds())
	
	if errors > 0 {
		fmt.Printf("\n⚠️  Warning: %d errors occurred during test\n", errors)
	} else {
		fmt.Println("\n✅ Test completed successfully with no errors")
	}
}