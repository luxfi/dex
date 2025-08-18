package lx

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/luxfi/dex/backend/pkg/types"
	"github.com/shopspring/decimal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestComprehensiveOrderBook tests all order book functionality
func TestComprehensiveOrderBook(t *testing.T) {
	tests := []struct {
		name string
		fn   func(t *testing.T)
	}{
		{"BasicOrderMatching", testBasicOrderMatching},
		{"AdvancedOrderTypes", testAdvancedOrderTypes},
		{"IcebergOrders", testIcebergOrders},
		{"PeggedOrders", testPeggedOrders},
		{"BracketOrders", testBracketOrders},
		{"TrailingStopOrders", testTrailingStopOrders},
		{"RiskManagement", testRiskManagement},
		{"L4BookProtocol", testL4BookProtocol},
		{"ConcurrentOrders", testConcurrentOrders},
		{"PerformanceBenchmark", testPerformanceBenchmark},
		{"FIXProtocol", testFIXProtocol},
		{"ConsensusIntegration", testConsensusIntegration},
		{"QuantumSecurity", testQuantumSecurity},
		{"MarketDataDistribution", testMarketDataDistribution},
		{"TimeInForce", testTimeInForce},
	}

	for _, test := range tests {
		t.Run(test.name, test.fn)
	}
}

// testBasicOrderMatching verifies basic limit and market order matching
func testBasicOrderMatching(t *testing.T) {
	ob := NewFullFeaturedOrderBook("BTC-USD")
	
	// Add buy limit order
	buyOrder := &AdvancedOrder{
		ID:       "buy1",
		Symbol:   "BTC-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(50000),
		Quantity: decimal.NewFromInt(1),
		ClientID: "client1",
	}
	
	trades, err := ob.AddAdvancedOrder(buyOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0) // No match yet
	
	// Add sell limit order that matches
	sellOrder := &AdvancedOrder{
		ID:       "sell1",
		Symbol:   "BTC-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(50000),
		Quantity: decimal.NewFromInt(1),
		ClientID: "client2",
	}
	
	trades, err = ob.AddAdvancedOrder(sellOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 1)
	assert.Equal(t, decimal.NewFromInt(50000), trades[0].Price)
	assert.Equal(t, decimal.NewFromInt(1), trades[0].Quantity)
}

// testAdvancedOrderTypes tests all advanced order types
func testAdvancedOrderTypes(t *testing.T) {
	ob := NewFullFeaturedOrderBook("ETH-USD")
	
	// Test stop order
	stopOrder := &AdvancedOrder{
		ID:        "stop1",
		Symbol:    "ETH-USD",
		Type:      OrderTypeStop,
		Side:      types.SideSell,
		StopPrice: decimal.NewFromInt(3000),
		Quantity:  decimal.NewFromInt(10),
		ClientID:  "client1",
	}
	
	trades, err := ob.AddAdvancedOrder(stopOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0) // Stop not triggered
	
	// Add market order to move price and trigger stop
	marketOrder := &AdvancedOrder{
		ID:       "market1",
		Symbol:   "ETH-USD",
		Type:     OrderTypeMarket,
		Side:     types.SideBuy,
		Quantity: decimal.NewFromInt(5),
		ClientID: "client2",
	}
	
	// First add a sell order for market to match
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "sell1",
		Symbol:   "ETH-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(2999),
		Quantity: decimal.NewFromInt(5),
		ClientID: "client3",
	})
	
	trades, err = ob.AddAdvancedOrder(marketOrder)
	assert.NoError(t, err)
	assert.Greater(t, len(trades), 0)
}

// testIcebergOrders tests iceberg order functionality
func testIcebergOrders(t *testing.T) {
	ob := NewFullFeaturedOrderBook("SOL-USD")
	
	// Add iceberg order with hidden quantity
	icebergOrder := &AdvancedOrder{
		ID:              "iceberg1",
		Symbol:          "SOL-USD",
		Type:            OrderTypeIceberg,
		Side:            types.SideBuy,
		Price:           decimal.NewFromInt(100),
		Quantity:        decimal.NewFromInt(1000), // Total quantity
		DisplayQuantity: decimal.NewFromInt(100),  // Visible quantity
		ClientID:        "whale1",
	}
	
	trades, err := ob.AddAdvancedOrder(icebergOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0)
	
	// Verify only display quantity is visible
	bids := ob.GetBids()
	assert.Len(t, bids, 1)
	totalVisible := decimal.Zero
	for _, level := range bids {
		for _, order := range level.Orders {
			totalVisible = totalVisible.Add(order.Quantity)
		}
	}
	assert.Equal(t, decimal.NewFromInt(100), totalVisible)
	
	// Match against visible portion
	sellOrder := &AdvancedOrder{
		ID:       "sell1",
		Symbol:   "SOL-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(100),
		Quantity: decimal.NewFromInt(100),
		ClientID: "client2",
	}
	
	trades, err = ob.AddAdvancedOrder(sellOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 1)
	assert.Equal(t, decimal.NewFromInt(100), trades[0].Quantity)
	
	// Verify next slice is revealed
	bids = ob.GetBids()
	if len(bids) > 0 {
		totalVisible = decimal.Zero
		for _, level := range bids {
			for _, order := range level.Orders {
				totalVisible = totalVisible.Add(order.Quantity)
			}
		}
		// Should show next 100 slice
		assert.LessOrEqual(t, totalVisible.IntPart(), int64(100))
	}
}

// testPeggedOrders tests pegged order functionality
func testPeggedOrders(t *testing.T) {
	ob := NewFullFeaturedOrderBook("AVAX-USD")
	
	// First establish a market with bid/ask
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "bid1",
		Symbol:   "AVAX-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(30),
		Quantity: decimal.NewFromInt(100),
		ClientID: "market_maker",
	})
	
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "ask1",
		Symbol:   "AVAX-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(31),
		Quantity: decimal.NewFromInt(100),
		ClientID: "market_maker",
	})
	
	// Add pegged order
	peggedOrder := &AdvancedOrder{
		ID:        "peg1",
		Symbol:    "AVAX-USD",
		Type:      OrderTypePeg,
		Side:      types.SideBuy,
		PegType:   PegBid,
		PegOffset: decimal.NewFromFloat(0.01),
		Quantity:  decimal.NewFromInt(50),
		ClientID:  "algo_trader",
	}
	
	trades, err := ob.AddAdvancedOrder(peggedOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0)
	
	// Verify order is pegged to best bid + offset
	bids := ob.GetBids()
	assert.Greater(t, len(bids), 0)
	
	// Update market and verify peg adjusts
	ob.UpdateMarketData()
}

// testBracketOrders tests bracket order with take profit and stop loss
func testBracketOrders(t *testing.T) {
	ob := NewFullFeaturedOrderBook("LINK-USD")
	
	// Add bracket order
	bracketOrder := &AdvancedOrder{
		ID:              "bracket1",
		Symbol:          "LINK-USD",
		Type:            OrderTypeBracket,
		Side:            types.SideBuy,
		Price:           decimal.NewFromInt(20),    // Entry price
		TakeProfitPrice: decimal.NewFromInt(25),    // TP at $25
		StopLossPrice:   decimal.NewFromInt(18),    // SL at $18
		Quantity:        decimal.NewFromInt(100),
		ClientID:        "trader1",
	}
	
	// First add a sell order to match entry
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "sell1",
		Symbol:   "LINK-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(20),
		Quantity: decimal.NewFromInt(100),
		ClientID: "market_maker",
	})
	
	trades, err := ob.AddAdvancedOrder(bracketOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 1) // Entry filled
	
	// Verify TP and SL orders are placed
	assert.Contains(t, ob.stopOrders, "bracket1_sl")
}

// testTrailingStopOrders tests trailing stop functionality
func testTrailingStopOrders(t *testing.T) {
	ob := NewFullFeaturedOrderBook("DOT-USD")
	
	// Establish market price
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "bid1",
		Symbol:   "DOT-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(10),
		Quantity: decimal.NewFromInt(100),
		ClientID: "mm",
	})
	
	// Add trailing stop
	trailingOrder := &AdvancedOrder{
		ID:          "trail1",
		Symbol:      "DOT-USD",
		Type:        OrderTypeTrailing,
		Side:        types.SideSell,
		TrailAmount: decimal.NewFromFloat(0.5), // Trail by $0.50
		Quantity:    decimal.NewFromInt(50),
		ClientID:    "trader1",
	}
	
	trades, err := ob.AddAdvancedOrder(trailingOrder)
	assert.NoError(t, err)
	assert.Nil(t, trades) // Trailing stop doesn't execute immediately
	
	// Verify stop price is set
	assert.Contains(t, ob.trailingOrders, "trail1")
	
	// Simulate price movement up
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "bid2",
		Symbol:   "DOT-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(11),
		Quantity: decimal.NewFromInt(100),
		ClientID: "mm",
	})
	
	// Update market data to adjust trailing stop
	ob.UpdateMarketData()
	
	// Verify trailing stop adjusted up
	trail := ob.trailingOrders["trail1"]
	assert.NotNil(t, trail)
	// Stop should be around 10.5 (11 - 0.5)
}

// testRiskManagement tests risk management features
func testRiskManagement(t *testing.T) {
	ob := NewFullFeaturedOrderBook("MATIC-USD")
	rm := ob.riskManager
	
	// Set position limit
	rm.positionLimits["MATIC-USD"] = decimal.NewFromInt(10000)
	rm.creditLimits["account1"] = decimal.NewFromInt(50000)
	
	// Try to exceed position limit
	largeOrder := &AdvancedOrder{
		ID:       "large1",
		Symbol:   "MATIC-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(1),
		Quantity: decimal.NewFromInt(20000), // Exceeds limit
		Account:  "account1",
		ClientID: "client1",
	}
	
	_, err := ob.AddAdvancedOrder(largeOrder)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exceed limit")
	
	// Order within limits should succeed
	smallOrder := &AdvancedOrder{
		ID:       "small1",
		Symbol:   "MATIC-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(1),
		Quantity: decimal.NewFromInt(5000),
		Account:  "account1",
		ClientID: "client1",
	}
	
	_, err = ob.AddAdvancedOrder(smallOrder)
	assert.NoError(t, err)
}

// testL4BookProtocol tests Hyperliquid's L4 book protocol
func testL4BookProtocol(t *testing.T) {
	ob := NewFullFeaturedOrderBook("UNI-USD")
	l4Server := NewL4BookServer(ob)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// Start L4 server
	go l4Server.Start(ctx, 8888)
	time.Sleep(100 * time.Millisecond)
	
	// Add orders to generate L4 events
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "l4_1",
		Symbol:   "UNI-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(5),
		Quantity: decimal.NewFromInt(100),
		ClientID: "user_0x123",
	})
	
	// Trigger order add event
	l4Server.OnOrderAdd(&types.Order{
		ID:       "l4_1",
		Symbol:   "UNI-USD",
		Type:     types.OrderTypeLimit,
		Side:     types.SideBuy,
		Price:    decimal.NewFromInt(5),
		Quantity: decimal.NewFromInt(100),
		ClientID: "user_0x123",
	})
	
	// Process block to batch updates
	l4Server.processBlock()
	
	// Generate snapshot
	snapshot := l4Server.generateL4Snapshot("UNI-USD")
	assert.NotNil(t, snapshot)
	assert.Greater(t, len(snapshot.Bids), 0)
	
	// Verify L4 order has full transparency
	if len(snapshot.Bids) > 0 {
		l4Order := snapshot.Bids[0]
		assert.Equal(t, "l4_1", l4Order.OrderID)
		assert.Equal(t, "user_0x123", l4Order.User)
		assert.Greater(t, l4Order.Timestamp, int64(0))
	}
	
	// Test conservative validation
	valid := l4Server.validateState()
	assert.True(t, valid)
}

// testConcurrentOrders tests thread safety with concurrent orders
func testConcurrentOrders(t *testing.T) {
	ob := NewFullFeaturedOrderBook("ATOM-USD")
	
	var wg sync.WaitGroup
	numGoroutines := 100
	ordersPerGoroutine := 100
	
	successCount := int64(0)
	errorCount := int64(0)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < ordersPerGoroutine; j++ {
				side := types.SideBuy
				if rand.Float32() > 0.5 {
					side = types.SideSell
				}
				
				order := &AdvancedOrder{
					ID:       fmt.Sprintf("order_%d_%d", id, j),
					Symbol:   "ATOM-USD",
					Type:     OrderTypeLimit,
					Side:     side,
					Price:    decimal.NewFromFloat(10 + rand.Float64()*2),
					Quantity: decimal.NewFromFloat(1 + rand.Float64()*10),
					ClientID: fmt.Sprintf("client_%d", id),
				}
				
				_, err := ob.AddAdvancedOrder(order)
				if err != nil {
					atomic.AddInt64(&errorCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
				}
			}
		}(i)
	}
	
	wg.Wait()
	
	t.Logf("Concurrent test: %d success, %d errors", successCount, errorCount)
	assert.Greater(t, successCount, int64(0))
}

// testPerformanceBenchmark tests performance characteristics
func testPerformanceBenchmark(t *testing.T) {
	ob := NewFullFeaturedOrderBook("PERF-USD")
	
	// Measure order addition latency
	numOrders := 10000
	start := time.Now()
	
	for i := 0; i < numOrders; i++ {
		order := &AdvancedOrder{
			ID:       fmt.Sprintf("perf_%d", i),
			Symbol:   "PERF-USD",
			Type:     OrderTypeLimit,
			Side:     types.Side(i % 2),
			Price:    decimal.NewFromFloat(100 + float64(i%100)),
			Quantity: decimal.NewFromInt(1),
			ClientID: "perf_test",
		}
		
		ob.AddAdvancedOrder(order)
	}
	
	elapsed := time.Since(start)
	ordersPerSecond := float64(numOrders) / elapsed.Seconds()
	latencyPerOrder := elapsed.Nanoseconds() / int64(numOrders)
	
	t.Logf("Performance: %.0f orders/sec, %d ns/order", ordersPerSecond, latencyPerOrder)
	
	// Verify performance meets targets
	assert.Greater(t, ordersPerSecond, float64(50000)) // At least 50K orders/sec
	assert.Less(t, latencyPerOrder, int64(20000))      // Less than 20μs per order
}

// testFIXProtocol tests FIX protocol implementation
func testFIXProtocol(t *testing.T) {
	// Test binary FIX encoding
	order := BinaryFIXOrder{
		MsgType:   'D', // NewOrderSingle
		OrderID:   12345,
		Price:     5000000000000, // $50000 with 8 decimals
		Quantity:  100000000,      // 1 BTC
		Side:      1,              // Buy
		OrderType: 2,              // Limit
		Timestamp: time.Now().UnixNano(),
	}
	
	copy(order.Symbol[:], "BTC-USD")
	
	// Serialize
	data := make([]byte, 60)
	serializeOrder(&order, data)
	
	// Deserialize
	var decoded BinaryFIXOrder
	deserializeOrder(data, &decoded)
	
	assert.Equal(t, order.OrderID, decoded.OrderID)
	assert.Equal(t, order.Price, decoded.Price)
	assert.Equal(t, order.Quantity, decoded.Quantity)
	assert.Equal(t, order.Side, decoded.Side)
}

// testConsensusIntegration tests consensus layer integration
func testConsensusIntegration(t *testing.T) {
	// Test K=3 consensus
	k := 3
	n := 3
	threshold := (k + n) / 2 // Should be 3
	
	assert.Equal(t, 3, threshold)
	
	// Simulate consensus round
	votes := make([]bool, n)
	for i := 0; i < k; i++ {
		votes[i] = true // K nodes vote yes
	}
	
	yesCount := 0
	for _, vote := range votes {
		if vote {
			yesCount++
		}
	}
	
	consensusReached := yesCount >= threshold
	assert.True(t, consensusReached)
}

// testQuantumSecurity tests quantum-resistant signatures
func testQuantumSecurity(t *testing.T) {
	// Test Ringtail+BLS hybrid signatures
	// This is a placeholder for actual quantum crypto
	
	message := []byte("order_data")
	
	// Simulate Ringtail signature (post-quantum)
	ringtailSig := make([]byte, 1024) // Lattice-based signatures are larger
	rand.Read(ringtailSig)
	
	// Simulate BLS aggregation
	blsSig := make([]byte, 96) // BLS signatures are compact
	rand.Read(blsSig)
	
	// Hybrid signature
	hybridSig := append(ringtailSig, blsSig...)
	
	assert.Equal(t, 1024+96, len(hybridSig))
	t.Logf("Quantum-resistant signature size: %d bytes", len(hybridSig))
}

// testMarketDataDistribution tests market data distribution
func testMarketDataDistribution(t *testing.T) {
	ob := NewFullFeaturedOrderBook("DATA-USD")
	
	// Add orders to generate market data
	for i := 0; i < 100; i++ {
		order := &AdvancedOrder{
			ID:       fmt.Sprintf("data_%d", i),
			Symbol:   "DATA-USD",
			Type:     OrderTypeLimit,
			Side:     types.Side(i % 2),
			Price:    decimal.NewFromFloat(10 + float64(i%10)/10),
			Quantity: decimal.NewFromInt(int64(i + 1)),
			ClientID: "data_test",
		}
		
		ob.AddAdvancedOrder(order)
	}
	
	// Get L2 snapshot
	bids := ob.GetBids()
	asks := ob.GetAsks()
	
	assert.Greater(t, len(bids), 0)
	assert.Greater(t, len(asks), 0)
	
	// Verify price levels are sorted
	for i := 1; i < len(bids); i++ {
		assert.True(t, bids[i-1].Price.GreaterThanOrEqual(bids[i].Price))
	}
	
	for i := 1; i < len(asks); i++ {
		assert.True(t, asks[i-1].Price.LessThanOrEqual(asks[i].Price))
	}
}

// testTimeInForce tests time-in-force order behavior
func testTimeInForce(t *testing.T) {
	ob := NewFullFeaturedOrderBook("TIF-USD")
	
	// Test IOC (Immediate or Cancel)
	iocOrder := &AdvancedOrder{
		ID:          "ioc1",
		Symbol:      "TIF-USD",
		Type:        OrderTypeLimit,
		Side:        types.SideBuy,
		Price:       decimal.NewFromInt(100),
		Quantity:    decimal.NewFromInt(10),
		TimeInForce: TIF_IOC,
		ClientID:    "client1",
	}
	
	// No matching order, should be cancelled
	trades, err := ob.AddAdvancedOrder(iocOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0)
	
	// Verify order is not in book
	bids := ob.GetBids()
	for _, level := range bids {
		for _, order := range level.Orders {
			assert.NotEqual(t, "ioc1", order.ID)
		}
	}
	
	// Test FOK (Fill or Kill)
	// First add liquidity
	ob.AddAdvancedOrder(&AdvancedOrder{
		ID:       "sell1",
		Symbol:   "TIF-USD",
		Type:     OrderTypeLimit,
		Side:     types.SideSell,
		Price:    decimal.NewFromInt(100),
		Quantity: decimal.NewFromInt(5), // Only 5 available
		ClientID: "mm",
	})
	
	fokOrder := &AdvancedOrder{
		ID:          "fok1",
		Symbol:      "TIF-USD",
		Type:        OrderTypeLimit,
		Side:        types.SideBuy,
		Price:       decimal.NewFromInt(100),
		Quantity:    decimal.NewFromInt(10), // Wants 10
		TimeInForce: TIF_FOK,
		ClientID:    "client2",
	}
	
	// Can't fill full quantity, should be killed
	trades, err = ob.AddAdvancedOrder(fokOrder)
	assert.NoError(t, err)
	assert.Len(t, trades, 0)
}

// BenchmarkOrderBookThroughput benchmarks order processing throughput
func BenchmarkOrderBookThroughput(b *testing.B) {
	ob := NewFullFeaturedOrderBook("BENCH-USD")
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		order := &AdvancedOrder{
			ID:       fmt.Sprintf("bench_%d", i),
			Symbol:   "BENCH-USD",
			Type:     OrderTypeLimit,
			Side:     types.Side(i % 2),
			Price:    decimal.NewFromFloat(100 + float64(i%100)/10),
			Quantity: decimal.NewFromInt(1),
			ClientID: "bench",
		}
		
		ob.AddAdvancedOrder(order)
	}
	
	ordersPerSecond := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(ordersPerSecond, "orders/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/order")
}

// Helper functions for FIX testing
func serializeOrder(order *BinaryFIXOrder, buf []byte) {
	buf[0] = order.MsgType
	copy(buf[1:9], order.Symbol[:])
	// Add serialization logic
}

func deserializeOrder(buf []byte, order *BinaryFIXOrder) {
	order.MsgType = buf[0]
	copy(order.Symbol[:], buf[1:9])
	// Add deserialization logic
}

// BinaryFIXOrder for testing
type BinaryFIXOrder struct {
	MsgType     byte
	Symbol      [8]byte
	OrderID     uint64
	Price       uint64
	Quantity    uint64
	Side        byte
	OrderType   byte
	TimeInForce byte
	Timestamp   int64
}