// E2E FIX over ZeroMQ Benchmark against LX DEX Cluster
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	zmq "github.com/pebbe/zmq4"
)

// FIX Message Types
const (
	MsgTypeNewOrderSingle  = "D"
	MsgTypeExecutionReport = "8"
	MsgTypeMarketData      = "W"
	MsgTypeOrderCancelReq  = "F"
	MsgTypeOrderStatusReq  = "H"
	MsgTypeQuoteRequest    = "R"
)

// Binary FIX format for ultra-low latency
type BinaryFIXOrder struct {
	MsgType     byte    // 1 byte
	Symbol      [8]byte // 8 bytes fixed
	OrderID     uint64  // 8 bytes
	Price       uint64  // 8 bytes (fixed point with 8 decimals)
	Quantity    uint64  // 8 bytes
	Side        byte    // 1 byte (1=Buy, 2=Sell)
	OrderType   byte    // 1 byte (1=Market, 2=Limit, 3=Stop)
	TimeInForce byte    // 1 byte (0=Day, 1=IOC, 2=FOK, 3=GTC)
	Timestamp   int64   // 8 bytes
}

type TestResult struct {
	TestName          string        `json:"test_name"`
	Protocol          string        `json:"protocol"`
	Transport         string        `json:"transport"`
	MessagesProcessed int64         `json:"messages_processed"`
	Duration          time.Duration `json:"duration"`
	Throughput        float64       `json:"throughput_msgs_per_sec"`
	LatencyStats      LatencyStats  `json:"latency_stats"`
	ErrorCount        int64         `json:"error_count"`
	SuccessRate       float64       `json:"success_rate"`
}

type LatencyStats struct {
	Min    int64   `json:"min_us"`
	Avg    float64 `json:"avg_us"`
	P50    int64   `json:"p50_us"`
	P95    int64   `json:"p95_us"`
	P99    int64   `json:"p99_us"`
	P999   int64   `json:"p999_us"`
	Max    int64   `json:"max_us"`
	StdDev float64 `json:"stddev_us"`
}

func encodeBinaryFIX(order *BinaryFIXOrder) []byte {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, order)
	return buf.Bytes()
}

func decodeBinaryFIX(data []byte) (*BinaryFIXOrder, error) {
	order := &BinaryFIXOrder{}
	buf := bytes.NewReader(data)
	err := binary.Read(buf, binary.LittleEndian, order)
	return order, err
}

func generateOrder(orderID uint64) *BinaryFIXOrder {
	symbols := []string{"BTC-USD", "ETH-USD", "LUX-USD", "SOL-USD"}
	symbol := symbols[rand.Intn(len(symbols))]

	order := &BinaryFIXOrder{
		MsgType:     MsgTypeNewOrderSingle[0],
		OrderID:     orderID,
		Price:       uint64(50000+rand.Intn(1000)) * 100000000, // Fixed point 8 decimals
		Quantity:    uint64(rand.Float64()*10) * 100000000,
		Side:        byte(1 + rand.Intn(2)),
		OrderType:   2, // Limit order
		TimeInForce: 0, // Day
		Timestamp:   time.Now().UnixNano(),
	}

	// Copy symbol
	copy(order.Symbol[:], symbol)

	return order
}

func calculateLatencyStats(latencies []int64) LatencyStats {
	if len(latencies) == 0 {
		return LatencyStats{}
	}

	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})

	var sum int64
	for _, l := range latencies {
		sum += l
	}
	avg := float64(sum) / float64(len(latencies))

	// Calculate standard deviation
	var sumSquares float64
	for _, l := range latencies {
		diff := float64(l) - avg
		sumSquares += diff * diff
	}
	stdDev := 0.0
	if len(latencies) > 1 {
		stdDev = sumSquares / float64(len(latencies)-1)
	}

	return LatencyStats{
		Min:    latencies[0],
		Avg:    avg,
		P50:    latencies[len(latencies)*50/100],
		P95:    latencies[len(latencies)*95/100],
		P99:    latencies[len(latencies)*99/100],
		P999:   latencies[min(len(latencies)*999/1000, len(latencies)-1)],
		Max:    latencies[len(latencies)-1],
		StdDev: stdDev,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Test 1: Direct HTTP API Test
func testHTTPAPI(nodeURL string, messageCount int) TestResult {
	fmt.Printf("\n📡 Testing HTTP API: %s\n", nodeURL)

	var processed int64
	var errors int64
	latencies := make([]int64, 0, messageCount)

	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	start := time.Now()

	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	messagesPerWorker := messageCount / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int, count int) {
			defer wg.Done()

			for i := 0; i < count; i++ {
				orderStart := time.Now()

				// Simple order submission
				order := fmt.Sprintf("BUY,BTC-USD,%d,0.1,trader-%d",
					50000+rand.Intn(1000), workerID)

				resp, err := client.Post(nodeURL+"/order",
					"text/plain",
					bytes.NewBufferString(order))

				if err != nil {
					atomic.AddInt64(&errors, 1)
					continue
				}
				resp.Body.Close()

				latency := time.Since(orderStart).Microseconds()
				latencies = append(latencies, latency)
				atomic.AddInt64(&processed, 1)
			}
		}(w, messagesPerWorker)
	}

	wg.Wait()
	duration := time.Since(start)

	return TestResult{
		TestName:          "HTTP API Direct",
		Protocol:          "HTTP/1.1",
		Transport:         "TCP",
		MessagesProcessed: processed,
		Duration:          duration,
		Throughput:        float64(processed) / duration.Seconds(),
		LatencyStats:      calculateLatencyStats(latencies),
		ErrorCount:        errors,
		SuccessRate:       float64(processed) / float64(processed+errors) * 100,
	}
}

// Test 2: Binary FIX over ZeroMQ
func testFIXOverZMQ(zmqEndpoint string, messageCount int) TestResult {
	fmt.Printf("\n⚡ Testing Binary FIX over ZeroMQ: %s\n", zmqEndpoint)

	// Create ZMQ context
	context, err := zmq.NewContext()
	if err != nil {
		log.Printf("Failed to create ZMQ context: %v", err)
		return TestResult{TestName: "FIX/ZMQ", ErrorCount: int64(messageCount)}
	}
	defer context.Term()

	var processed int64
	var errors int64
	latencies := make([]int64, 0, messageCount)
	latencyMutex := &sync.Mutex{}

	start := time.Now()

	var wg sync.WaitGroup
	workers := runtime.NumCPU() * 2 // More workers for ZMQ
	messagesPerWorker := messageCount / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int, count int) {
			defer wg.Done()

			// Create DEALER socket for async req/rep
			socket, err := context.NewSocket(zmq.DEALER)
			if err != nil {
				atomic.AddInt64(&errors, int64(count))
				return
			}
			defer socket.Close()

			// Set socket options for low latency
			socket.SetSndhwm(10000)
			socket.SetRcvhwm(10000)
			socket.SetLinger(0)
			socket.SetTcpKeepalive(1)
			socket.SetTcpKeepaliveIdle(30)

			// Connect to endpoint
			if err := socket.Connect(zmqEndpoint); err != nil {
				atomic.AddInt64(&errors, int64(count))
				return
			}

			// Send orders
			for i := 0; i < count; i++ {
				orderID := uint64(workerID*1000000 + i)
				order := generateOrder(orderID)
				data := encodeBinaryFIX(order)

				orderStart := time.Now()

				// Send binary FIX message
				if _, err := socket.SendBytes(data, zmq.DONTWAIT); err != nil {
					atomic.AddInt64(&errors, 1)
					continue
				}

				// For benchmark, we don't wait for response (fire-and-forget)
				// In real scenario, you'd use ROUTER/DEALER pattern for async responses

				latency := time.Since(orderStart).Microseconds()
				latencyMutex.Lock()
				latencies = append(latencies, latency)
				latencyMutex.Unlock()

				atomic.AddInt64(&processed, 1)

				// Pace the sending slightly to avoid overwhelming
				if i%100 == 0 {
					time.Sleep(time.Microsecond * 10)
				}
			}
		}(w, messagesPerWorker)
	}

	wg.Wait()
	duration := time.Since(start)

	return TestResult{
		TestName:          "Binary FIX over ZeroMQ",
		Protocol:          "FIX Binary",
		Transport:         "ZeroMQ/TCP",
		MessagesProcessed: processed,
		Duration:          duration,
		Throughput:        float64(processed) / duration.Seconds(),
		LatencyStats:      calculateLatencyStats(latencies),
		ErrorCount:        errors,
		SuccessRate:       float64(processed) / float64(processed+errors) * 100,
	}
}

// Test 3: Batched Binary FIX over ZMQ
func testBatchedFIXOverZMQ(zmqEndpoint string, messageCount int, batchSize int) TestResult {
	fmt.Printf("\n🚀 Testing Batched FIX over ZeroMQ (batch=%d): %s\n", batchSize, zmqEndpoint)

	context, err := zmq.NewContext()
	if err != nil {
		log.Printf("Failed to create ZMQ context: %v", err)
		return TestResult{TestName: "Batched FIX/ZMQ", ErrorCount: int64(messageCount)}
	}
	defer context.Term()

	var processed int64
	var errors int64
	latencies := make([]int64, 0, messageCount/batchSize)

	// Create PUSH socket for one-way high throughput
	socket, err := context.NewSocket(zmq.PUSH)
	if err != nil {
		return TestResult{TestName: "Batched FIX/ZMQ", ErrorCount: int64(messageCount)}
	}
	defer socket.Close()

	// Optimize for throughput
	socket.SetSndhwm(100000)
	socket.SetSndtimeo(1000) // 1 second timeout

	if err := socket.Connect(zmqEndpoint); err != nil {
		return TestResult{TestName: "Batched FIX/ZMQ", ErrorCount: int64(messageCount)}
	}

	start := time.Now()

	batches := messageCount / batchSize
	for b := 0; b < batches; b++ {
		batchStart := time.Now()

		// Create batch buffer
		batchBuf := new(bytes.Buffer)
		binary.Write(batchBuf, binary.LittleEndian, uint32(batchSize))

		for i := 0; i < batchSize; i++ {
			order := generateOrder(uint64(b*batchSize + i))
			data := encodeBinaryFIX(order)
			batchBuf.Write(data)
		}

		// Send entire batch
		if _, err := socket.SendBytes(batchBuf.Bytes(), 0); err != nil {
			atomic.AddInt64(&errors, int64(batchSize))
		} else {
			atomic.AddInt64(&processed, int64(batchSize))
		}

		latency := time.Since(batchStart).Microseconds()
		latencies = append(latencies, latency/int64(batchSize)) // Per-message latency
	}

	duration := time.Since(start)

	return TestResult{
		TestName:          fmt.Sprintf("Batched FIX/ZMQ (batch=%d)", batchSize),
		Protocol:          "FIX Binary Batched",
		Transport:         "ZeroMQ/TCP",
		MessagesProcessed: processed,
		Duration:          duration,
		Throughput:        float64(processed) / duration.Seconds(),
		LatencyStats:      calculateLatencyStats(latencies),
		ErrorCount:        errors,
		SuccessRate:       float64(processed) / float64(processed+errors) * 100,
	}
}

// Test 4: E2E Cluster Test - Submit to all nodes
func testClusterE2E(nodeURLs []string, messageCount int) TestResult {
	fmt.Printf("\n🌐 Testing E2E Cluster (K=3): %v\n", nodeURLs)

	var processed int64
	var errors int64
	allLatencies := make([]int64, 0, messageCount)
	latencyMutex := &sync.Mutex{}

	start := time.Now()

	var wg sync.WaitGroup
	messagesPerNode := messageCount / len(nodeURLs)

	for nodeIdx, nodeURL := range nodeURLs {
		wg.Add(1)
		go func(idx int, url string, count int) {
			defer wg.Done()

			client := &http.Client{Timeout: 2 * time.Second}

			for i := 0; i < count; i++ {
				orderStart := time.Now()

				// Submit order
				order := fmt.Sprintf("BUY,BTC-USD,%d,0.1,node%d-trader%d",
					50000+rand.Intn(1000), idx, i)

				resp, err := client.Post(url+"/order",
					"text/plain",
					bytes.NewBufferString(order))

				if err != nil {
					atomic.AddInt64(&errors, 1)
					continue
				}
				resp.Body.Close()

				latency := time.Since(orderStart).Microseconds()

				latencyMutex.Lock()
				allLatencies = append(allLatencies, latency)
				latencyMutex.Unlock()

				atomic.AddInt64(&processed, 1)
			}
		}(nodeIdx, nodeURL, messagesPerNode)
	}

	wg.Wait()
	duration := time.Since(start)

	return TestResult{
		TestName:          "E2E Cluster Test (K=3)",
		Protocol:          "HTTP",
		Transport:         "TCP (3 nodes)",
		MessagesProcessed: processed,
		Duration:          duration,
		Throughput:        float64(processed) / duration.Seconds(),
		LatencyStats:      calculateLatencyStats(allLatencies),
		ErrorCount:        errors,
		SuccessRate:       float64(processed) / float64(processed+errors) * 100,
	}
}

func main() {
	var (
		mode         = flag.String("mode", "all", "Test mode: all, http, zmq, batch, cluster")
		messages     = flag.Int("messages", 10000, "Number of messages to send")
		batchSize    = flag.Int("batch", 100, "Batch size for batched mode")
		zmqEndpoint  = flag.String("zmq", "tcp://localhost:5555", "ZeroMQ endpoint")
		httpEndpoint = flag.String("http", "http://localhost:8080", "HTTP endpoint")
		saveResults  = flag.Bool("save", true, "Save results to file")
	)
	flag.Parse()

	fmt.Println("=================================================")
	fmt.Println("   LX DEX E2E FIX/ZMQ Benchmark Suite")
	fmt.Println("=================================================")
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Printf("Messages: %d\n", *messages)
	fmt.Printf("Batch Size: %d\n", *batchSize)
	fmt.Println()

	results := []TestResult{}

	// Run tests based on mode
	switch *mode {
	case "all":
		// Test 1: HTTP API
		results = append(results, testHTTPAPI(*httpEndpoint, *messages))

		// Test 2: Binary FIX over ZMQ
		results = append(results, testFIXOverZMQ(*zmqEndpoint, *messages))

		// Test 3: Batched FIX over ZMQ
		results = append(results, testBatchedFIXOverZMQ(*zmqEndpoint, *messages, *batchSize))

		// Test 4: Cluster E2E
		clusterNodes := []string{
			"http://localhost:8080",
			"http://localhost:8090",
			"http://localhost:8100",
		}
		results = append(results, testClusterE2E(clusterNodes, *messages))

	case "http":
		results = append(results, testHTTPAPI(*httpEndpoint, *messages))

	case "zmq":
		results = append(results, testFIXOverZMQ(*zmqEndpoint, *messages))

	case "batch":
		results = append(results, testBatchedFIXOverZMQ(*zmqEndpoint, *messages, *batchSize))

	case "cluster":
		clusterNodes := []string{
			"http://localhost:8080",
			"http://localhost:8090",
			"http://localhost:8100",
		}
		results = append(results, testClusterE2E(clusterNodes, *messages))
	}

	// Print results summary
	fmt.Println("\n=================================================")
	fmt.Println("   BENCHMARK RESULTS SUMMARY")
	fmt.Println("=================================================")
	fmt.Printf("\n%-30s | %12s | %10s | %10s | %10s | %8s\n",
		"Test", "Throughput", "Avg Latency", "P99 Latency", "Success", "Errors")
	fmt.Println(string(bytes.Repeat([]byte("-"), 100)))

	for _, r := range results {
		fmt.Printf("%-30s | %10.0f/s | %8.2f μs | %8.2f μs | %7.1f%% | %6d\n",
			r.TestName,
			r.Throughput,
			r.LatencyStats.Avg,
			float64(r.LatencyStats.P99),
			r.SuccessRate,
			r.ErrorCount)
	}

	// Find best performer
	if len(results) > 0 {
		best := results[0]
		for _, r := range results {
			if r.Throughput > best.Throughput {
				best = r
			}
		}

		fmt.Printf("\n🏆 Best Throughput: %s (%.0f msgs/sec)\n", best.TestName, best.Throughput)

		// Find lowest latency
		bestLatency := results[0]
		for _, r := range results {
			if r.LatencyStats.P99 < bestLatency.LatencyStats.P99 {
				bestLatency = r
			}
		}
		fmt.Printf("⚡ Lowest P99 Latency: %s (%d μs)\n", bestLatency.TestName, bestLatency.LatencyStats.P99)
	}

	// Save results
	if *saveResults {
		timestamp := time.Now().Format("20060102-150405")
		filename := fmt.Sprintf("e2e-results-%s.json", timestamp)

		data, _ := json.MarshalIndent(results, "", "  ")
		if err := os.WriteFile(filename, data, 0644); err != nil {
			log.Printf("Failed to save results: %v", err)
		} else {
			fmt.Printf("\n📊 Results saved to: %s\n", filename)
		}
	}

	// Performance recommendations
	fmt.Println("\n=================================================")
	fmt.Println("   PERFORMANCE RECOMMENDATIONS")
	fmt.Println("=================================================")

	if len(results) > 0 {
		avgThroughput := 0.0
		for _, r := range results {
			avgThroughput += r.Throughput
		}
		avgThroughput /= float64(len(results))

		fmt.Printf("\n📈 Average Throughput: %.0f msgs/sec\n", avgThroughput)

		if avgThroughput < 100000 {
			fmt.Println("\n⚠️  Performance is below 100K msgs/sec. Consider:")
			fmt.Println("   • Enable kernel bypass (DPDK)")
			fmt.Println("   • Use RDMA for node communication")
			fmt.Println("   • Increase batch sizes")
			fmt.Println("   • Optimize network buffers")
		} else if avgThroughput < 1000000 {
			fmt.Println("\n✅ Good performance! To reach 1M+ msgs/sec:")
			fmt.Println("   • Use binary protocols exclusively")
			fmt.Println("   • Enable GPU acceleration for matching")
			fmt.Println("   • Implement lock-free data structures")
		} else {
			fmt.Println("\n🎯 Excellent performance! You're achieving:")
			fmt.Printf("   • %.2fM messages/second\n", avgThroughput/1000000)
			fmt.Println("   • Near-hardware limits")
			fmt.Println("   • Ready for production deployment")
		}
	}
}
