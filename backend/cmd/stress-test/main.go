package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type StressTest struct {
	nc           *nats.Conn
	js           nats.JetStreamContext
	traders      int
	duration     time.Duration
	ordersCount  int64
	errorsCount  int64
	bytesCount   int64
	latencies    []time.Duration
	mu           sync.Mutex
}

type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
}

func main() {
	var (
		traders  = flag.Int("traders", 200, "Number of concurrent traders")
		duration = flag.Duration("duration", 60*time.Second, "Test duration")
		natsURL  = flag.String("nats", nats.DefaultURL, "NATS server URL")
		burst    = flag.Bool("burst", false, "Enable burst mode (no rate limiting)")
	)
	flag.Parse()

	fmt.Println("🔥 LX DEX Extreme Stress Test")
	fmt.Println("================================")
	fmt.Printf("Traders: %d\n", *traders)
	fmt.Printf("Duration: %v\n", *duration)
	fmt.Printf("Burst Mode: %v\n", *burst)
	fmt.Printf("CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Println("================================\n")

	// Connect to NATS
	nc, err := nats.Connect(*natsURL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(1*time.Second),
		nats.Timeout(5*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	js, err := nc.JetStream()
	if err != nil {
		log.Fatalf("Failed to get JetStream context: %v", err)
	}

	st := &StressTest{
		nc:        nc,
		js:        js,
		traders:   *traders,
		duration:  *duration,
		latencies: make([]time.Duration, 0, 1000000),
	}

	// Wait for server
	fmt.Println("⏳ Waiting for DEX server...")
	sub, err := nc.SubscribeSync("dex.heartbeat")
	if err != nil {
		log.Fatal(err)
	}
	
	for {
		msg, err := sub.NextMsg(2 * time.Second)
		if err == nil {
			fmt.Printf("🔍 Found DEX server: %s\n", string(msg.Data))
			break
		}
		nc.Publish("dex.ping", []byte("stress-test"))
	}
	sub.Unsubscribe()

	// Start monitoring
	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	// Memory monitoring
	go st.monitorMemory(ctx)

	// Stats reporter
	go st.reportStats(ctx)

	// Start traders
	fmt.Printf("\n🚀 Starting %d concurrent traders...\n\n", *traders)
	
	var wg sync.WaitGroup
	startTime := time.Now()

	for i := 0; i < *traders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			if *burst {
				st.runBurstTrader(ctx, id)
			} else {
				st.runTrader(ctx, id)
			}
		}(i)
		
		// Stagger start slightly
		if i%10 == 0 {
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Wait for completion
	wg.Wait()
	elapsed := time.Since(startTime)

	// Final report
	st.printFinalReport(elapsed)
}

func (st *StressTest) runTrader(ctx context.Context, id int) {
	symbols := []string{"BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"}
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))
	
	// Rate limiting: each trader sends 100-500 orders/sec
	rate := 100 + rng.Intn(400)
	ticker := time.NewTicker(time.Second / time.Duration(rate))
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			order := Order{
				ID:        uint64(atomic.AddInt64(&st.ordersCount, 1)),
				Symbol:    symbols[rng.Intn(len(symbols))],
				Side:      []string{"buy", "sell"}[rng.Intn(2)],
				Price:     1000 + rng.Float64()*1000,
				Quantity:  rng.Float64() * 100,
				Timestamp: time.Now(),
			}

			start := time.Now()
			_, err := st.js.PublishAsync("orders."+order.Symbol, []byte(fmt.Sprintf("%+v", order)))
			
			if err != nil {
				atomic.AddInt64(&st.errorsCount, 1)
			} else {
				latency := time.Since(start)
				st.recordLatency(latency)
				atomic.AddInt64(&st.bytesCount, 200) // Approximate message size
			}
		}
	}
}

func (st *StressTest) runBurstTrader(ctx context.Context, id int) {
	symbols := []string{"BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"}
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))
	
	// Burst mode: send as fast as possible
	for {
		select {
		case <-ctx.Done():
			return
		default:
			order := Order{
				ID:        uint64(atomic.AddInt64(&st.ordersCount, 1)),
				Symbol:    symbols[rng.Intn(len(symbols))],
				Side:      []string{"buy", "sell"}[rng.Intn(2)],
				Price:     1000 + rng.Float64()*1000,
				Quantity:  rng.Float64() * 100,
				Timestamp: time.Now(),
			}

			start := time.Now()
			_, err := st.js.PublishAsync("orders."+order.Symbol, []byte(fmt.Sprintf("%+v", order)))
			
			if err != nil {
				atomic.AddInt64(&st.errorsCount, 1)
				// Back off on errors
				time.Sleep(10 * time.Millisecond)
			} else {
				latency := time.Since(start)
				st.recordLatency(latency)
				atomic.AddInt64(&st.bytesCount, 200)
			}
		}
	}
}

func (st *StressTest) recordLatency(d time.Duration) {
	st.mu.Lock()
	defer st.mu.Unlock()
	
	// Only record a sample to avoid memory issues
	if len(st.latencies) < 1000000 {
		st.latencies = append(st.latencies, d)
	}
}

func (st *StressTest) monitorMemory(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			fmt.Printf("💾 Memory: Alloc=%v MB, Sys=%v MB, NumGC=%v\n",
				m.Alloc/1024/1024, m.Sys/1024/1024, m.NumGC)
		}
	}
}

func (st *StressTest) reportStats(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	startTime := time.Now()
	lastOrders := int64(0)
	lastBytes := int64(0)
	lastTime := startTime

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			orders := atomic.LoadInt64(&st.ordersCount)
			errors := atomic.LoadInt64(&st.errorsCount)
			bytes := atomic.LoadInt64(&st.bytesCount)

			// Calculate rates
			deltaTime := now.Sub(lastTime).Seconds()
			ordersPerSec := float64(orders-lastOrders) / deltaTime
			mbPerSec := float64(bytes-lastBytes) / (deltaTime * 1024 * 1024)
			
			// Calculate cumulative
			totalTime := now.Sub(startTime).Seconds()
			avgRate := float64(orders) / totalTime

			fmt.Printf("📊 Orders: %d | Rate: %.0f/s (avg: %.0f/s) | Network: %.2f MB/s | Errors: %d\n",
				orders, ordersPerSec, avgRate, mbPerSec, errors)

			lastOrders = orders
			lastBytes = bytes
			lastTime = now
		}
	}
}

func (st *StressTest) printFinalReport(elapsed time.Duration) {
	orders := atomic.LoadInt64(&st.ordersCount)
	errors := atomic.LoadInt64(&st.errorsCount)
	bytes := atomic.LoadInt64(&st.bytesCount)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📈 STRESS TEST RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	
	fmt.Printf("Duration:        %v\n", elapsed)
	fmt.Printf("Total Orders:    %d\n", orders)
	fmt.Printf("Total Errors:    %d (%.2f%%)\n", errors, float64(errors)/float64(orders)*100)
	fmt.Printf("Success Rate:    %.2f%%\n", (1-float64(errors)/float64(orders))*100)
	fmt.Printf("Throughput:      %.0f orders/sec\n", float64(orders)/elapsed.Seconds())
	fmt.Printf("Data Volume:     %.2f MB\n", float64(bytes)/(1024*1024))
	fmt.Printf("Network Rate:    %.2f MB/s\n", float64(bytes)/(elapsed.Seconds()*1024*1024))
	
	// Latency analysis
	if len(st.latencies) > 0 {
		st.mu.Lock()
		defer st.mu.Unlock()
		
		// Sort latencies
		sort.Slice(st.latencies, func(i, j int) bool {
			return st.latencies[i] < st.latencies[j]
		})
		
		p50 := st.latencies[len(st.latencies)*50/100]
		p95 := st.latencies[len(st.latencies)*95/100]
		p99 := st.latencies[len(st.latencies)*99/100]
		
		fmt.Println("\nLatency Percentiles:")
		fmt.Printf("  P50: %v\n", p50)
		fmt.Printf("  P95: %v\n", p95)
		fmt.Printf("  P99: %v\n", p99)
	}
	
	fmt.Println(strings.Repeat("=", 60))
	
	// Performance grade
	rate := float64(orders) / elapsed.Seconds()
	grade := "F"
	switch {
	case rate > 100000:
		grade = "A+ (World-class)"
	case rate > 50000:
		grade = "A (Excellent)"
	case rate > 20000:
		grade = "B (Good)"
	case rate > 10000:
		grade = "C (Average)"
	case rate > 5000:
		grade = "D (Below Average)"
	}
	
	fmt.Printf("\n🏆 Performance Grade: %s\n", grade)
}