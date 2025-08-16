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

	pb "github.com/luxexchange/engine/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	grpcEndpoint = flag.String("grpc", "localhost:50051", "gRPC server endpoint")
	numTraders   = flag.Int("traders", 1000, "Number of concurrent traders")
	ordersPerSec = flag.Int("rate", 100, "Orders per second per trader")
	duration     = flag.Duration("duration", 60*time.Second, "Test duration")
	symbols      = flag.String("symbols", "BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LUX-USD", "Trading symbols (comma-separated)")
	useHybrid    = flag.Bool("hybrid", false, "Use hybrid CGO server on port 50052")
)

type TraderStats struct {
	submitted    int64
	cancelled    int64
	errors       int64
	latencySum   int64
	latencyCount int64
	maxLatency   int64
	minLatency   int64
}

func (s *TraderStats) RecordLatency(ns int64) {
	atomic.AddInt64(&s.latencySum, ns)
	atomic.AddInt64(&s.latencyCount, 1)
	
	// Update max
	for {
		old := atomic.LoadInt64(&s.maxLatency)
		if ns <= old || atomic.CompareAndSwapInt64(&s.maxLatency, old, ns) {
			break
		}
	}
	
	// Update min
	for {
		old := atomic.LoadInt64(&s.minLatency)
		if old != 0 && ns >= old {
			break
		}
		if atomic.CompareAndSwapInt64(&s.minLatency, old, ns) {
			break
		}
	}
}

func (s *TraderStats) AverageLatency() time.Duration {
	count := atomic.LoadInt64(&s.latencyCount)
	if count == 0 {
		return 0
	}
	sum := atomic.LoadInt64(&s.latencySum)
	return time.Duration(sum / count)
}

type Trader struct {
	id     int
	client pb.EngineServiceClient
	stats  *TraderStats
	rng    *rand.Rand
	symbol string
}

func (t *Trader) Run(ctx context.Context, ordersPerSecond int) {
	ticker := time.NewTicker(time.Second / time.Duration(ordersPerSecond))
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			t.submitOrder(ctx)
			
			// Occasionally cancel orders (10% chance)
			if t.rng.Float64() < 0.1 {
				t.cancelRandomOrder(ctx)
			}
		}
	}
}

func (t *Trader) submitOrder(ctx context.Context) {
	side := pb.OrderSide_ORDER_SIDE_BUY
	if t.rng.Float64() > 0.5 {
		side = pb.OrderSide_ORDER_SIDE_SELL
	}
	
	// Price around market (40k for BTC, scaled for others)
	basePrice := 40000.0
	if t.symbol == "ETH-USD" {
		basePrice = 2500.0
	} else if t.symbol == "SOL-USD" {
		basePrice = 100.0
	} else if t.symbol == "AVAX-USD" {
		basePrice = 35.0
	} else if t.symbol == "LUX-USD" {
		basePrice = 1.0
	}
	
	price := basePrice * (0.95 + t.rng.Float64()*0.1) // ±5% from base
	quantity := 0.001 + t.rng.Float64()*0.999        // 0.001 to 1.0
	
	start := time.Now()
	_, err := t.client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
		Symbol:        t.symbol,
		Side:          side,
		Type:          pb.OrderType_ORDER_TYPE_LIMIT,
		Quantity:      quantity,
		Price:         price,
		ClientOrderId: fmt.Sprintf("trader-%d-%d", t.id, time.Now().UnixNano()),
	})
	latency := time.Since(start).Nanoseconds()
	t.stats.RecordLatency(latency)
	
	if err != nil {
		atomic.AddInt64(&t.stats.errors, 1)
	} else {
		atomic.AddInt64(&t.stats.submitted, 1)
	}
}

func (t *Trader) cancelRandomOrder(ctx context.Context) {
	// In a real system, we'd track order IDs
	// For now, just simulate a cancel
	start := time.Now()
	_, err := t.client.CancelOrder(ctx, &pb.CancelOrderRequest{
		OrderId: fmt.Sprintf("order-%s", t.symbol),
	})
	latency := time.Since(start).Nanoseconds()
	t.stats.RecordLatency(latency)
	
	if err == nil {
		atomic.AddInt64(&t.stats.cancelled, 1)
	}
}

func main() {
	flag.Parse()
	
	// Parse symbols
	symbolList := []string{"BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LUX-USD"}
	
	endpoint := *grpcEndpoint
	if *useHybrid {
		endpoint = "localhost:50052"
	}
	
	fmt.Println("=== MEGA TRADER SIMULATION ===")
	fmt.Printf("Endpoint: %s\n", endpoint)
	fmt.Printf("Traders: %d\n", *numTraders)
	fmt.Printf("Orders/sec/trader: %d\n", *ordersPerSec)
	fmt.Printf("Total rate: %d orders/sec\n", *numTraders * *ordersPerSec)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Printf("Symbols: %v\n", symbolList)
	fmt.Println()
	
	// Create shared connection pool
	conns := make([]*grpc.ClientConn, 10) // 10 connections shared among traders
	for i := range conns {
		conn, err := grpc.Dial(endpoint,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)),
		)
		if err != nil {
			log.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()
		conns[i] = conn
	}
	
	// Create traders
	stats := &TraderStats{minLatency: int64(time.Hour)}
	traders := make([]*Trader, *numTraders)
	
	for i := 0; i < *numTraders; i++ {
		traders[i] = &Trader{
			id:     i,
			client: pb.NewEngineServiceClient(conns[i%len(conns)]),
			stats:  stats,
			rng:    rand.New(rand.NewSource(time.Now().UnixNano() + int64(i))),
			symbol: symbolList[i%len(symbolList)],
		}
	}
	
	// Start monitoring
	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()
	
	var wg sync.WaitGroup
	
	// Monitor goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		
		lastSubmitted := int64(0)
		lastTime := time.Now()
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				submitted := atomic.LoadInt64(&stats.submitted)
				cancelled := atomic.LoadInt64(&stats.cancelled)
				errors := atomic.LoadInt64(&stats.errors)
				avgLatency := stats.AverageLatency()
				maxLatency := time.Duration(atomic.LoadInt64(&stats.maxLatency))
				minLatency := time.Duration(atomic.LoadInt64(&stats.minLatency))
				
				// Calculate rate
				now := time.Now()
				elapsed := now.Sub(lastTime).Seconds()
				rate := float64(submitted-lastSubmitted) / elapsed
				lastSubmitted = submitted
				lastTime = now
				
				fmt.Printf("[%s] Orders: %d | Rate: %.0f/s | Cancelled: %d | Errors: %d | Latency: avg=%v min=%v max=%v\n",
					now.Format("15:04:05"),
					submitted, rate, cancelled, errors,
					avgLatency, minLatency, maxLatency)
			}
		}
	}()
	
	// Start all traders
	fmt.Printf("Starting %d traders...\n", *numTraders)
	start := time.Now()
	
	for _, trader := range traders {
		wg.Add(1)
		go func(t *Trader) {
			defer wg.Done()
			t.Run(ctx, *ordersPerSec)
		}(trader)
	}
	
	// Wait for completion
	wg.Wait()
	
	// Final stats
	elapsed := time.Since(start)
	submitted := atomic.LoadInt64(&stats.submitted)
	cancelled := atomic.LoadInt64(&stats.cancelled)
	errors := atomic.LoadInt64(&stats.errors)
	avgLatency := stats.AverageLatency()
	maxLatency := time.Duration(atomic.LoadInt64(&stats.maxLatency))
	minLatency := time.Duration(atomic.LoadInt64(&stats.minLatency))
	
	fmt.Println("\n=== FINAL RESULTS ===")
	fmt.Printf("Duration: %v\n", elapsed)
	fmt.Printf("Total Orders: %d\n", submitted)
	fmt.Printf("Orders Cancelled: %d\n", cancelled)
	fmt.Printf("Errors: %d (%.2f%%)\n", errors, float64(errors)*100/float64(submitted+errors))
	fmt.Printf("Throughput: %.0f orders/second\n", float64(submitted)/elapsed.Seconds())
	fmt.Printf("Latency: avg=%v min=%v max=%v\n", avgLatency, minLatency, maxLatency)
	
	if float64(errors)*100/float64(submitted+errors) > 1 {
		fmt.Printf("\n⚠️  Warning: Error rate above 1%%\n")
	} else {
		fmt.Printf("\n✅ Test completed successfully\n")
	}
	
	// Calculate theoretical max
	theoretical := float64(*numTraders) * float64(*ordersPerSec) * elapsed.Seconds()
	efficiency := float64(submitted) / theoretical * 100
	fmt.Printf("\nEfficiency: %.1f%% of theoretical maximum\n", efficiency)
}