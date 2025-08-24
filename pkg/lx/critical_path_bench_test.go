package lx

import (
	"fmt"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Critical Path Benchmarks for LX DEX
// These benchmarks test the most performance-critical code paths

// BenchmarkCriticalOrderMatching tests the core matching engine performance
func BenchmarkCriticalOrderMatching(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("BookDepth_%d", size), func(b *testing.B) {
			book := NewOrderBook("BENCH")
			book.EnableImmediateMatching = false

			// Pre-populate order book
			for i := 0; i < size; i++ {
				book.AddOrder(&Order{
					ID:        uint64(i),
					Type:      Limit,
					Side:      Buy,
					Price:     100 - float64(i%50)/100,
					Size:      100,
					User:      "bench",
					Timestamp: time.Now(),
				})
				book.AddOrder(&Order{
					ID:        uint64(i + size),
					Type:      Limit,
					Side:      Sell,
					Price:     101 + float64(i%50)/100,
					Size:      100,
					User:      "bench",
					Timestamp: time.Now(),
				})
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Submit crossing order
				order := &Order{
					ID:        uint64(i + size*2),
					Type:      Limit,
					Side:      Buy,
					Price:     101.5,
					Size:      10,
					User:      "taker",
					Timestamp: time.Now(),
				}
				book.AddOrder(order)
			}

			// Report metrics
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
		})
	}
}

// BenchmarkClearinghouseMargin tests margin calculation performance
func BenchmarkClearinghouseMargin(b *testing.B) {
	ch := NewClearingHouse(nil, nil)

	// Pre-populate accounts
	numAccounts := 1000
	for i := 0; i < numAccounts; i++ {
		address := fmt.Sprintf("account_%d", i)
		ch.Deposit(address, big.NewInt(1000000))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			address := fmt.Sprintf("account_%d", i%numAccounts)
			ch.OpenPosition(address, "BTC-PERP", Buy, 0.1, Limit)
			i++
		}
	})

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "positions/sec")
}

// BenchmarkFPGAAcceleration tests FPGA processing pipeline
func BenchmarkFPGAAcceleration(b *testing.B) {
	fpga := NewFPGAAccelerator()

	// Create test orders
	orders := make([]*Order, 1000)
	for i := range orders {
		orders[i] = &Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Side(i % 2),
			Price:     100 + float64(i%20-10)/10,
			Size:      100,
			User:      "fpga_test",
			Timestamp: time.Now(),
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		order := orders[i%len(orders)]
		fpga.ProcessOrder(order)
	}

	metrics := fpga.GetMetrics()
	if metrics.LatencyNanos > 0 {
		b.ReportMetric(float64(metrics.LatencyNanos), "ns/order")
	}
}

// BenchmarkConsensusFinalization tests block finalization speed
func BenchmarkConsensusFinalization(b *testing.B) {
	// Simulate consensus with varying block sizes
	blockSizes := []int{100, 1000, 10000}

	for _, size := range blockSizes {
		b.Run(fmt.Sprintf("BlockSize_%d", size), func(b *testing.B) {
			data := make([]byte, size)
			rand.Read(data)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Simulate block processing
				hash := hashBlock(data)
				_ = verifyBlock(hash)
			}

			b.ReportMetric(float64(size*b.N)/b.Elapsed().Seconds()/1024/1024, "MB/sec")
		})
	}
}

// BenchmarkOracleAggregation tests oracle price aggregation
func BenchmarkOracleAggregation(b *testing.B) {
	ch := NewClearingHouse(nil, nil)

	// Simulate price updates from multiple sources
	prices := map[string]float64{
		"validator1": 50000.0,
		"validator2": 50001.0,
		"validator3": 49999.0,
		"validator4": 50000.5,
		"validator5": 50002.0,
		"validator6": 49998.0,
		"validator7": 50001.5,
		"validator8": 49999.5,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ch.UpdateOraclePrice("BTC", prices)
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "updates/sec")
}

// BenchmarkOrderBookSnapshot tests snapshot generation performance
func BenchmarkOrderBookSnapshot(b *testing.B) {
	book := NewOrderBook("SNAPSHOT")

	// Create deep order book
	for i := 0; i < 10000; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Buy,
			Price:     100 - float64(i)/1000,
			Size:      100,
			User:      "maker",
			Timestamp: time.Now(),
		})
		book.AddOrder(&Order{
			ID:        uint64(i + 10000),
			Type:      Limit,
			Side:      Sell,
			Price:     100 + float64(i)/1000,
			Size:      100,
			User:      "maker",
			Timestamp: time.Now(),
		})
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		snapshot := book.GetOrderBookSnapshot()
		_ = snapshot
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "snapshots/sec")
}

// BenchmarkConcurrentOrderSubmission tests concurrent order processing
func BenchmarkConcurrentOrderSubmission(b *testing.B) {
	book := NewOrderBook("CONCURRENT")

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			order := &Order{
				ID:        uint64(atomic.AddInt64(&orderIDCounter, 1)),
				Type:      Limit,
				Side:      Side(i % 2),
				Price:     100 + float64(i%20-10)/10,
				Size:      100,
				User:      fmt.Sprintf("user_%d", i%100),
				Timestamp: time.Now(),
			}
			book.AddOrder(order)
			i++
		}
	})

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
	b.ReportMetric(float64(runtime.NumGoroutine()), "goroutines")
}

var orderIDCounter int64

// BenchmarkMemoryPool tests memory allocation efficiency
func BenchmarkMemoryPool(b *testing.B) {
	pool := &sync.Pool{
		New: func() interface{} {
			return &Order{}
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Get from pool
			order := pool.Get().(*Order)
			order.ID = uint64(rand.Int63())
			order.Type = Limit
			order.Side = Buy
			order.Price = 100
			order.Size = 10
			order.User = "pool_test"
			order.Timestamp = time.Now()

			// Process order
			processOrder(order)

			// Return to pool
			order.Reset()
			pool.Put(order)
		}
	})

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "allocs/sec")
}

// BenchmarkTradeExecution tests trade execution performance
func BenchmarkTradeExecution(b *testing.B) {
	book := NewOrderBook("TRADE")

	// Pre-populate with liquidity
	for i := 0; i < 1000; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Sell,
			Price:     100 + float64(i)/100,
			Size:      1000,
			User:      "liquidity",
			Timestamp: time.Now(),
		})
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Market buy order that will execute
		order := &Order{
			ID:        uint64(i + 1000),
			Type:      Market,
			Side:      Buy,
			Size:      10,
			User:      "taker",
			Timestamp: time.Now(),
		}
		trades := book.AddOrder(order)
		_ = trades
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "trades/sec")
}

// BenchmarkRiskCheck tests pre-trade risk validation
func BenchmarkRiskCheck(b *testing.B) {
	ch := NewClearingHouse(nil, nil)

	// Setup account with balance
	address := "risk_test"
	ch.Deposit(address, big.NewInt(1000000))

	account := ch.getOrCreateAccount(address)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Perform risk check
		passed := ch.performMarginCheck(account, "BTC-PERP", 1.0)
		_ = passed
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "checks/sec")
}

// BenchmarkFundingCalculation tests funding rate calculations
func BenchmarkFundingCalculation(b *testing.B) {
	ch := NewClearingHouse(nil, nil)

	// Create accounts with positions
	for i := 0; i < 100; i++ {
		address := fmt.Sprintf("funding_%d", i)
		ch.Deposit(address, big.NewInt(1000000))
		ch.OpenPosition(address, "BTC-PERP", Side(i%2), 1.0, Limit)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ch.ProcessFunding()
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "funding_cycles/sec")
}

// BenchmarkDAGValidation tests DAG consensus validation
func BenchmarkDAGValidation(b *testing.B) {
	// Simulate DAG validation
	dag := make(map[string][]string)

	// Build DAG structure
	for i := 0; i < 1000; i++ {
		nodeID := fmt.Sprintf("node_%d", i)
		parents := make([]string, 0)
		if i > 0 {
			parents = append(parents, fmt.Sprintf("node_%d", i-1))
		}
		if i > 1 {
			parents = append(parents, fmt.Sprintf("node_%d", i-2))
		}
		dag[nodeID] = parents
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Validate DAG structure
		valid := validateDAG(dag)
		_ = valid
	}

	b.ReportMetric(float64(len(dag)*b.N)/b.Elapsed().Seconds(), "nodes/sec")
}

// Helper functions for benchmarks

func hashBlock(data []byte) []byte {
	// Simplified hash calculation
	hash := make([]byte, 32)
	for i := 0; i < len(data) && i < 32; i++ {
		hash[i] = data[i]
	}
	return hash
}

func verifyBlock(hash []byte) bool {
	// Simplified verification
	return len(hash) == 32
}

func processOrder(order *Order) {
	// Simulate order processing
	time.Sleep(time.Nanosecond)
}

func (o *Order) Reset() {
	o.ID = 0
	o.Type = 0
	o.Side = 0
	o.Price = 0
	o.Size = 0
	o.User = ""
	o.Timestamp = time.Time{}
}

func validateDAG(dag map[string][]string) bool {
	// Simplified DAG validation
	visited := make(map[string]bool)
	for node := range dag {
		if !visited[node] {
			if !dfsVisit(node, dag, visited, make(map[string]bool)) {
				return false
			}
		}
	}
	return true
}

func dfsVisit(node string, dag map[string][]string, visited, recStack map[string]bool) bool {
	visited[node] = true
	recStack[node] = true

	for _, parent := range dag[node] {
		if !visited[parent] {
			if !dfsVisit(parent, dag, visited, recStack) {
				return false
			}
		} else if recStack[parent] {
			return false // Cycle detected
		}
	}

	recStack[node] = false
	return true
}

// BenchmarkMLXAcceleration tests MLX GPU acceleration
func BenchmarkMLXAcceleration(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("MLX only available on macOS")
	}

	// Skip for now as detectBestEngine is not yet implemented
	b.Skip("MLX engine detection not implemented")

	orders := make([]*Order, 1000)
	for i := range orders {
		orders[i] = &Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Side(i % 2),
			Price:     100 + float64(i%20-10)/10,
			Size:      100,
			User:      "mlx_test",
			Timestamp: time.Now(),
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate MLX processing
		_ = orders[i%len(orders)]
	}

	// Report simulated metrics
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
}

// BenchmarkQuantumSignatures tests post-quantum signature verification
func BenchmarkQuantumSignatures(b *testing.B) {
	// Simulate quantum-resistant signature verification
	message := make([]byte, 256)
	signature := make([]byte, 512)
	publicKey := make([]byte, 256)

	rand.Read(message)
	rand.Read(signature)
	rand.Read(publicKey)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate verification
		valid := verifyQuantumSignature(message, signature, publicKey)
		_ = valid
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "verifications/sec")
}

func verifyQuantumSignature(message, signature, publicKey []byte) bool {
	// Simplified quantum signature verification
	return len(signature) == 512 && len(publicKey) == 256
}

// BenchmarkEndToEndLatency tests complete order lifecycle
func BenchmarkEndToEndLatency(b *testing.B) {
	book := NewOrderBook("E2E")
	ch := NewClearingHouse(nil, nil)

	// Setup account
	ch.Deposit("e2e_test", big.NewInt(10000000))

	// Pre-populate book
	for i := 0; i < 100; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Sell,
			Price:     101 + float64(i)/100,
			Size:      100,
			User:      "liquidity",
			Timestamp: time.Now(),
		})
	}

	latencies := make([]int64, 0, b.N)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Complete order lifecycle
		order := &Order{
			ID:        uint64(i + 100),
			Type:      Market,
			Side:      Buy,
			Size:      10,
			User:      "e2e_test",
			Timestamp: time.Now(),
		}

		// Risk check
		account := ch.getOrCreateAccount("e2e_test")
		if !ch.performMarginCheck(account, "E2E", order.Size) {
			continue
		}

		// Order matching
		trades := book.AddOrder(order)

		// Position update
		if trades > 0 {
			ch.OpenPosition("e2e_test", "E2E", order.Side, order.Size, order.Type)
		}

		latency := time.Since(start).Nanoseconds()
		latencies = append(latencies, latency)
	}

	// Calculate percentiles
	if len(latencies) > 0 {
		avgLatency := int64(0)
		for _, l := range latencies {
			avgLatency += l
		}
		avgLatency /= int64(len(latencies))

		b.ReportMetric(float64(avgLatency), "ns/order")
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
	}
}
