package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/luxexchange/engine/backend/pkg/orderbook"
)

// Sample FIX message data structure
type FIXMessage struct {
	MsgType      string
	Symbol       string
	OrderID      string
	Side         string
	OrderType    string
	Price        float64
	Quantity     float64
	Timestamp    time.Time
}

// BenchmarkResult holds the results of a benchmark run
type BenchmarkResult struct {
	EngineName       string
	MessagesProcessed int
	OrdersCreated    int
	TradesExecuted   int
	Duration         time.Duration
	Throughput       float64 // messages per second
	AvgLatency       time.Duration
	P99Latency       time.Duration
}

func main() {
	var (
		dataFile   = flag.String("data", "", "Path to FIX data file (CSV or FIX log)")
		dataURL    = flag.String("url", "", "URL to download FIX data from")
		engineType = flag.String("engine", "all", "Engine type: go, cpp, hybrid, all")
		iterations = flag.Int("iter", 1, "Number of iterations to run")
		warmup     = flag.Int("warmup", 1000, "Number of warmup messages")
		verbose    = flag.Bool("v", false, "Verbose output")
	)
	flag.Parse()

	// Download or load FIX data
	var messages []FIXMessage
	var err error

	if *dataURL != "" {
		messages, err = downloadFIXData(*dataURL)
		if err != nil {
			log.Fatalf("Failed to download FIX data: %v", err)
		}
	} else if *dataFile != "" {
		messages, err = loadFIXData(*dataFile)
		if err != nil {
			log.Fatalf("Failed to load FIX data: %v", err)
		}
	} else {
		// Generate synthetic FIX data
		messages = generateSyntheticFIXData(100000)
	}

	fmt.Printf("Loaded %d FIX messages\n", len(messages))

	// Run benchmarks
	results := []BenchmarkResult{}

	if *engineType == "all" || *engineType == "go" {
		result := benchmarkEngine("Pure Go", orderbook.ImplGo, messages, *iterations, *warmup, *verbose)
		results = append(results, result)
	}

	if *engineType == "all" || *engineType == "cpp" {
		if os.Getenv("CGO_ENABLED") == "1" {
			result := benchmarkEngine("Pure C++", orderbook.ImplCpp, messages, *iterations, *warmup, *verbose)
			results = append(results, result)
		} else {
			fmt.Println("Skipping C++ engine (CGO not enabled)")
		}
	}

	if *engineType == "all" || *engineType == "hybrid" {
		if os.Getenv("CGO_ENABLED") == "1" {
			// Hybrid uses C++ for hot paths but Go for orchestration
			result := benchmarkEngine("Hybrid Go/C++", orderbook.ImplCpp, messages, *iterations, *warmup, *verbose)
			results = append(results, result)
		}
	}

	// Print results
	printResults(results)
}

func benchmarkEngine(name string, impl orderbook.Implementation, messages []FIXMessage, iterations, warmup int, verbose bool) BenchmarkResult {
	fmt.Printf("\n=== Benchmarking %s Engine ===\n", name)

	var totalDuration time.Duration
	var totalOrders int
	var totalTrades int
	latencies := []time.Duration{}

	for iter := 0; iter < iterations; iter++ {
		if verbose {
			fmt.Printf("Iteration %d/%d\n", iter+1, iterations)
		}

		// Create order book
		ob := orderbook.NewOrderBook(orderbook.Config{
			Implementation: impl,
			Symbol:         "BTC-USD",
		})

		ordersCreated := 0
		tradesExecuted := 0
		iterStart := time.Now()

		for i, msg := range messages {
			// Skip warmup messages for timing
			msgStart := time.Now()

			// Process message
			switch msg.MsgType {
			case "D": // New Order Single
				order := &orderbook.Order{
					ID:       uint64(i),
					Symbol:   msg.Symbol,
					Price:    msg.Price,
					Quantity: msg.Quantity,
					Side:     parseSide(msg.Side),
					Type:     parseOrderType(msg.OrderType),
					Status:   orderbook.Pending,
					Timestamp: msg.Timestamp,
				}

				ob.AddOrder(order)
				ordersCreated++

				// Match orders
				trades := ob.MatchOrders()
				tradesExecuted += len(trades)

			case "F": // Order Cancel Request
				// Cancel order
				if id, err := strconv.ParseUint(msg.OrderID, 10, 64); err == nil {
					ob.CancelOrder(id)
				}

			case "G": // Order Cancel/Replace Request
				if id, err := strconv.ParseUint(msg.OrderID, 10, 64); err == nil {
					ob.ModifyOrder(id, msg.Price, msg.Quantity)
				}
			}

			if i >= warmup {
				latencies = append(latencies, time.Since(msgStart))
			}
		}

		iterDuration := time.Since(iterStart)
		totalDuration += iterDuration
		totalOrders += ordersCreated
		totalTrades += tradesExecuted

		if verbose {
			fmt.Printf("  Orders: %d, Trades: %d, Duration: %v\n", 
				ordersCreated, tradesExecuted, iterDuration)
		}
	}

	// Calculate statistics
	avgDuration := totalDuration / time.Duration(iterations)
	throughput := float64(len(messages)) / avgDuration.Seconds()
	
	avgLatency := time.Duration(0)
	if len(latencies) > 0 {
		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		avgLatency = total / time.Duration(len(latencies))
	}

	// Calculate P99 latency
	p99Latency := calculateP99(latencies)

	return BenchmarkResult{
		EngineName:        name,
		MessagesProcessed: len(messages) * iterations,
		OrdersCreated:     totalOrders,
		TradesExecuted:    totalTrades,
		Duration:          totalDuration,
		Throughput:        throughput,
		AvgLatency:        avgLatency,
		P99Latency:        p99Latency,
	}
}

func downloadFIXData(url string) ([]FIXMessage, error) {
	fmt.Printf("Downloading FIX data from %s...\n", url)
	
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Save to temp file
	tmpFile, err := os.CreateTemp("", "fix_data_*.csv")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmpFile.Name())

	_, err = io.Copy(tmpFile, resp.Body)
	if err != nil {
		return nil, err
	}

	return loadFIXData(tmpFile.Name())
}

func loadFIXData(filename string) ([]FIXMessage, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var messages []FIXMessage

	// Check if it's CSV or raw FIX
	scanner := bufio.NewScanner(file)
	if scanner.Scan() {
		firstLine := scanner.Text()
		file.Seek(0, 0) // Reset to beginning

		if strings.Contains(firstLine, ",") {
			// CSV format
			reader := csv.NewReader(file)
			
			// Skip header
			if _, err := reader.Read(); err != nil {
				return nil, err
			}

			for {
				record, err := reader.Read()
				if err == io.EOF {
					break
				}
				if err != nil {
					return nil, err
				}

				// Parse CSV columns
				// Assuming: MsgType,Symbol,OrderID,Side,OrderType,Price,Quantity,Timestamp
				price, _ := strconv.ParseFloat(record[5], 64)
				quantity, _ := strconv.ParseFloat(record[6], 64)
				timestamp, _ := time.Parse(time.RFC3339, record[7])

				messages = append(messages, FIXMessage{
					MsgType:   record[0],
					Symbol:    record[1],
					OrderID:   record[2],
					Side:      record[3],
					OrderType: record[4],
					Price:     price,
					Quantity:  quantity,
					Timestamp: timestamp,
				})
			}
		} else {
			// Raw FIX format - parse FIX messages
			scanner := bufio.NewScanner(file)
			for scanner.Scan() {
				msg := parseFIXMessage(scanner.Text())
				if msg != nil {
					messages = append(messages, *msg)
				}
			}
		}
	}

	return messages, nil
}

func parseFIXMessage(line string) *FIXMessage {
	// Simple FIX parser for common fields
	fields := strings.Split(line, string(byte(1))) // SOH separator
	
	msg := &FIXMessage{
		Timestamp: time.Now(),
	}

	for _, field := range fields {
		parts := strings.Split(field, "=")
		if len(parts) != 2 {
			continue
		}

		tag := parts[0]
		value := parts[1]

		switch tag {
		case "35": // MsgType
			msg.MsgType = value
		case "55": // Symbol
			msg.Symbol = value
		case "37": // OrderID
			msg.OrderID = value
		case "54": // Side
			msg.Side = value
		case "40": // OrderType
			msg.OrderType = value
		case "44": // Price
			msg.Price, _ = strconv.ParseFloat(value, 64)
		case "38": // Quantity
			msg.Quantity, _ = strconv.ParseFloat(value, 64)
		case "52": // SendingTime
			msg.Timestamp, _ = time.Parse("20060102-15:04:05", value)
		}
	}

	// Only return if we have a valid message
	if msg.MsgType != "" && msg.Symbol != "" {
		return msg
	}
	return nil
}

func generateSyntheticFIXData(count int) []FIXMessage {
	fmt.Printf("Generating %d synthetic FIX messages...\n", count)
	
	messages := make([]FIXMessage, count)
	basePrice := 50000.0 // BTC price
	
	for i := 0; i < count; i++ {
		// Mix of order types
		msgType := "D" // New Order
		if i%10 == 0 {
			msgType = "F" // Cancel
		} else if i%20 == 0 {
			msgType = "G" // Modify
		}

		// Randomize price around base
		spread := 100.0
		price := basePrice + (float64(i%100) - 50) * spread / 50
		
		// Randomize quantity
		quantity := float64(1 + i%10) * 0.1

		// Alternate sides
		side := "1" // Buy
		if i%2 == 0 {
			side = "2" // Sell
		}

		messages[i] = FIXMessage{
			MsgType:   msgType,
			Symbol:    "BTC-USD",
			OrderID:   fmt.Sprintf("%d", i),
			Side:      side,
			OrderType: "2", // Limit
			Price:     price,
			Quantity:  quantity,
			Timestamp: time.Now().Add(time.Duration(i) * time.Millisecond),
		}
	}

	return messages
}

func parseSide(side string) orderbook.OrderSide {
	if side == "1" {
		return orderbook.Buy
	}
	return orderbook.Sell
}

func parseOrderType(orderType string) orderbook.OrderType {
	if orderType == "1" {
		return orderbook.Market
	}
	return orderbook.Limit
}

func calculateP99(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Simple P99 calculation
	index := int(float64(len(latencies)) * 0.99)
	if index >= len(latencies) {
		index = len(latencies) - 1
	}

	return latencies[index]
}

func printResults(results []BenchmarkResult) {
	fmt.Println("\n=== Benchmark Results ===")
	fmt.Println("╔═══════════════════╦════════════════╦═══════════╦═══════════╦════════════╦════════════╦════════════╗")
	fmt.Println("║ Engine            ║ Messages/sec   ║ Orders    ║ Trades    ║ Avg Latency║ P99 Latency║ Duration   ║")
	fmt.Println("╠═══════════════════╬════════════════╬═══════════╬═══════════╬════════════╬════════════╬════════════╣")
	
	for _, r := range results {
		fmt.Printf("║ %-17s ║ %14.0f ║ %9d ║ %9d ║ %10v ║ %10v ║ %10v ║\n",
			r.EngineName,
			r.Throughput,
			r.OrdersCreated,
			r.TradesExecuted,
			r.AvgLatency,
			r.P99Latency,
			r.Duration,
		)
	}
	
	fmt.Println("╚═══════════════════╩════════════════╩═══════════╩═══════════╩════════════╩════════════╩════════════╝")

	// Find winner
	if len(results) > 1 {
		best := results[0]
		for _, r := range results[1:] {
			if r.Throughput > best.Throughput {
				best = r
			}
		}
		fmt.Printf("\n🏆 Best Performance: %s with %.0f messages/sec\n", best.EngineName, best.Throughput)
	}
}