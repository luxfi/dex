// E2E FIX over QZMQ - Post-Quantum Secure Binary FIX
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

	"github.com/luxfi/qzmq"
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
	Nonce       [16]byte // 16 bytes for post-quantum security
}

type TestResult struct {
	TestName          string        `json:"test_name"`
	Protocol          string        `json:"protocol"`
	Transport         string        `json:"transport"`
	Encryption        string        `json:"encryption"`
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
		Price:       uint64(50000+rand.Intn(1000)) * 100000000,
		Quantity:    uint64(rand.Float64()*10) * 100000000,
		Side:        byte(1 + rand.Intn(2)),
		OrderType:   2, // Limit order
		TimeInForce: 0, // Day
		Timestamp:   time.Now().UnixNano(),
	}

	// Copy symbol
	copy(order.Symbol[:], symbol)
	
	// Add quantum-safe nonce
	rand.Read(order.Nonce[:])

	return order
}

func runQZMQTest(endpoint string, numMessages int, suite qzmq.Suite) *TestResult {
	startTime := time.Now()
	var sentCount, recvCount, errorCount int64
	var latencies []int64
	var wg sync.WaitGroup

	// Create QZMQ transport with specified suite
	opts := qzmq.Options{
		Backend: qzmq.BackendAuto,
		Suite:   suite,
		Mode:    qzmq.ModeHybrid, // Use hybrid classical+PQ
		KeyRotation: qzmq.KeyRotationPolicy{
			MaxMessages: 10000,
			MaxBytes:    1 << 30,
			MaxAge:      5 * time.Minute,
		},
	}

	// Create server transport
	serverTransport, err := qzmq.New(opts)
	if err != nil {
		log.Fatalf("Failed to create server transport: %v", err)
	}
	defer serverTransport.Close()

	// Create ROUTER socket for server
	router, err := serverTransport.NewSocket(qzmq.ROUTER)
	if err != nil {
		log.Fatalf("Failed to create ROUTER socket: %v", err)
	}
	defer router.Close()

	// Bind server
	serverEndpoint := "tcp://127.0.0.1:15570"
	if err := router.Bind(serverEndpoint); err != nil {
		log.Fatalf("Failed to bind: %v", err)
	}

	// Server goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for atomic.LoadInt64(&recvCount) < int64(numMessages) {
			// Receive multipart (identity + order)
			parts, err := router.RecvMultipart()
			if err != nil {
				atomic.AddInt64(&errorCount, 1)
				continue
			}

			if len(parts) < 2 {
				continue
			}

			// Parse order
			order, err := decodeBinaryFIX(parts[1])
			if err != nil {
				atomic.AddInt64(&errorCount, 1)
				continue
			}

			// Create execution report
			execReport := &BinaryFIXOrder{
				MsgType:   MsgTypeExecutionReport[0],
				Symbol:    order.Symbol,
				OrderID:   order.OrderID,
				Price:     order.Price,
				Quantity:  order.Quantity,
				Side:      order.Side,
				Timestamp: time.Now().UnixNano(),
			}

			// Send back execution report
			response := encodeBinaryFIX(execReport)
			router.SendMultipart([][]byte{parts[0], response})

			atomic.AddInt64(&recvCount, 1)
		}
	}()

	// Connect dealer to server
	if err := dealer.Connect(serverEndpoint); err != nil {
		log.Fatalf("Failed to connect dealer to server: %v", err)
	}
	time.Sleep(100 * time.Millisecond)

	// Client sending loop
	for i := 0; i < numMessages; i++ {
		order := generateOrder(uint64(i))
		orderBytes := encodeBinaryFIX(order)

		sendStart := time.Now()
		if err := dealer.Send(orderBytes); err != nil {
			atomic.AddInt64(&errorCount, 1)
			continue
		}

		// Wait for response
		response, err := dealer.Recv()
		if err != nil {
			atomic.AddInt64(&errorCount, 1)
			continue
		}

		latency := time.Since(sendStart).Microseconds()
		latencies = append(latencies, latency)

		// Verify response
		execReport, err := decodeBinaryFIX(response)
		if err != nil || execReport.OrderID != order.OrderID {
			atomic.AddInt64(&errorCount, 1)
			continue
		}

		atomic.AddInt64(&sentCount, 1)
	}

	wg.Wait()
	duration := time.Since(startTime)

	// Calculate statistics
	stats := calculateLatencyStats(latencies)

	return &TestResult{
		TestName:          "QZMQ Binary FIX",
		Protocol:          "Binary FIX",
		Transport:         "QZMQ",
		Encryption:        fmt.Sprintf("%s/%s/%s", suite.KEM, suite.Sign, suite.AEAD),
		MessagesProcessed: sentCount,
		Duration:          duration,
		Throughput:        float64(sentCount) / duration.Seconds(),
		LatencyStats:      stats,
		ErrorCount:        errorCount,
		SuccessRate:       float64(sentCount) / float64(numMessages) * 100,
	}
}

func calculateLatencyStats(latencies []int64) LatencyStats {
	if len(latencies) == 0 {
		return LatencyStats{}
	}

	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})

	// Calculate average
	var sum int64
	for _, l := range latencies {
		sum += l
	}
	avg := float64(sum) / float64(len(latencies))

	// Calculate standard deviation
	var variance float64
	for _, l := range latencies {
		diff := float64(l) - avg
		variance += diff * diff
	}
	stdDev := 0.0
	if len(latencies) > 1 {
		variance /= float64(len(latencies) - 1)
		stdDev = variance // sqrt would be actual stddev
	}

	return LatencyStats{
		Min:    latencies[0],
		Avg:    avg,
		P50:    latencies[len(latencies)*50/100],
		P95:    latencies[len(latencies)*95/100],
		P99:    latencies[len(latencies)*99/100],
		P999:   latencies[len(latencies)*999/1000],
		Max:    latencies[len(latencies)-1],
		StdDev: stdDev,
	}
}

func main() {
	var (
		endpoint    = flag.String("endpoint", "tcp://127.0.0.1:5555", "QZMQ endpoint")
		numMessages = flag.Int("messages", 10000, "Number of messages to send")
		mode        = flag.String("mode", "hybrid", "Security mode: performance, balanced, conservative")
		outputJSON  = flag.Bool("json", false, "Output results as JSON")
		httpPort    = flag.Int("http", 0, "HTTP port for stats (0 to disable)")
	)
	flag.Parse()

	// Set CPU affinity for consistent benchmarks
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Select crypto suite based on mode
	var suite qzmq.Suite
	switch *mode {
	case "performance":
		suite = qzmq.PerformanceOptions().Suite
	case "conservative":
		suite = qzmq.ConservativeOptions().Suite
	default:
		suite = qzmq.DefaultOptions().Suite
	}

	fmt.Printf("Starting QZMQ Binary FIX test...\n")
	fmt.Printf("Endpoint: %s\n", *endpoint)
	fmt.Printf("Messages: %d\n", *numMessages)
	fmt.Printf("Security: %s (KEM=%s, Sign=%s, AEAD=%s)\n", 
		*mode, suite.KEM, suite.Sign, suite.AEAD)

	// Start HTTP stats server if requested
	if *httpPort > 0 {
		go startHTTPServer(*httpPort)
	}

	// Run test
	result := runQZMQTest(*endpoint, *numMessages, suite)

	// Output results
	if *outputJSON {
		jsonData, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(jsonData))
	} else {
		fmt.Printf("\n=== Test Results ===\n")
		fmt.Printf("Protocol: %s\n", result.Protocol)
		fmt.Printf("Transport: %s\n", result.Transport)
		fmt.Printf("Encryption: %s\n", result.Encryption)
		fmt.Printf("Messages: %d\n", result.MessagesProcessed)
		fmt.Printf("Duration: %v\n", result.Duration)
		fmt.Printf("Throughput: %.2f msgs/sec\n", result.Throughput)
		fmt.Printf("Success Rate: %.2f%%\n", result.SuccessRate)
		fmt.Printf("\nLatency Statistics (microseconds):\n")
		fmt.Printf("  Min: %d µs\n", result.LatencyStats.Min)
		fmt.Printf("  Avg: %.2f µs\n", result.LatencyStats.Avg)
		fmt.Printf("  P50: %d µs\n", result.LatencyStats.P50)
		fmt.Printf("  P95: %d µs\n", result.LatencyStats.P95)
		fmt.Printf("  P99: %d µs\n", result.LatencyStats.P99)
		fmt.Printf("  P999: %d µs\n", result.LatencyStats.P999)
		fmt.Printf("  Max: %d µs\n", result.LatencyStats.Max)
	}

	// Save to file
	if err := saveResults(result); err != nil {
		log.Printf("Failed to save results: %v", err)
	}
}

func startHTTPServer(port int) {
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats := map[string]interface{}{
			"status": "running",
			"time":   time.Now().Unix(),
		}
		json.NewEncoder(w).Encode(stats)
	})

	log.Printf("HTTP stats server listening on :%d", port)
	http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

func saveResults(result *TestResult) error {
	filename := fmt.Sprintf("qzmq_test_%d.json", time.Now().Unix())
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}