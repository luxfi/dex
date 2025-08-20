// +build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/sdk/go/client"
)

const (
	testJSONRPCURL = "http://localhost:8080"
	testWSURL      = "ws://localhost:8081"
	testGRPCURL    = "localhost:50051"
)

// TestFullOrderLifecycle tests complete order flow
func TestFullOrderLifecycle(t *testing.T) {
	ctx := context.Background()
	
	// Create client
	c, err := client.NewClient(
		client.WithJSONRPCURL(testJSONRPCURL),
		client.WithWebSocketURL(testWSURL),
		client.WithGRPCURL(testGRPCURL),
	)
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Connect to gRPC for best performance
	err = c.ConnectGRPC(ctx)
	if err != nil {
		t.Logf("gRPC not available, using JSON-RPC: %v", err)
	}
	
	// Test 1: Place a buy limit order
	buyOrder := &client.Order{
		Symbol:      "BTC-USD",
		Type:        client.OrderTypeLimit,
		Side:        client.OrderSideBuy,
		Price:       50000.00,
		Size:        0.1,
		UserID:      "test-user-1",
		ClientID:    "test-order-001",
		TimeInForce: client.TimeInForceGTC,
	}
	
	buyResp, err := c.PlaceOrder(ctx, buyOrder)
	require.NoError(t, err)
	assert.NotZero(t, buyResp.OrderID)
	assert.Equal(t, "open", buyResp.Status)
	
	// Test 2: Place a matching sell order
	sellOrder := &client.Order{
		Symbol:      "BTC-USD",
		Type:        client.OrderTypeLimit,
		Side:        client.OrderSideSell,
		Price:       50000.00,
		Size:        0.1,
		UserID:      "test-user-2",
		ClientID:    "test-order-002",
		TimeInForce: client.TimeInForceGTC,
	}
	
	sellResp, err := c.PlaceOrder(ctx, sellOrder)
	require.NoError(t, err)
	assert.NotZero(t, sellResp.OrderID)
	
	// Orders should match immediately
	time.Sleep(100 * time.Millisecond)
	
	// Test 3: Verify trades were created
	trades, err := c.GetTrades(ctx, "BTC-USD", 10)
	require.NoError(t, err)
	assert.NotEmpty(t, trades)
	
	// Find our trade
	var ourTrade *client.Trade
	for _, trade := range trades {
		if trade.BuyOrderID == buyResp.OrderID || trade.SellOrderID == sellResp.OrderID {
			ourTrade = trade
			break
		}
	}
	
	require.NotNil(t, ourTrade, "Trade not found")
	assert.Equal(t, 50000.0, ourTrade.Price)
	assert.Equal(t, 0.1, ourTrade.Size)
}

// TestOrderTypes tests different order types
func TestOrderTypes(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(client.WithJSONRPCURL(testJSONRPCURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	testCases := []struct {
		name      string
		orderType client.OrderType
		side      client.OrderSide
		price     float64
		size      float64
	}{
		{"Limit Buy", client.OrderTypeLimit, client.OrderSideBuy, 49000, 0.05},
		{"Limit Sell", client.OrderTypeLimit, client.OrderSideSell, 51000, 0.05},
		{"Market Buy", client.OrderTypeMarket, client.OrderSideBuy, 0, 0.02},
		{"Market Sell", client.OrderTypeMarket, client.OrderSideSell, 0, 0.02},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			order := &client.Order{
				Symbol: "BTC-USD",
				Type:   tc.orderType,
				Side:   tc.side,
				Price:  tc.price,
				Size:   tc.size,
				UserID: fmt.Sprintf("test-user-%s", tc.name),
			}
			
			resp, err := c.PlaceOrder(ctx, order)
			require.NoError(t, err)
			assert.NotZero(t, resp.OrderID)
		})
	}
}

// TestWebSocketStreaming tests real-time data streaming
func TestWebSocketStreaming(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	c, err := client.NewClient(client.WithWebSocketURL(testWSURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Connect WebSocket
	err = c.ConnectWebSocket(ctx)
	require.NoError(t, err)
	
	// Subscribe to order book updates
	orderBookChan := make(chan *client.OrderBook, 10)
	err = c.SubscribeOrderBook("BTC-USD", func(ob *client.OrderBook) {
		select {
		case orderBookChan <- ob:
		default:
		}
	})
	require.NoError(t, err)
	
	// Subscribe to trades
	tradeChan := make(chan *client.Trade, 10)
	err = c.SubscribeTrades("BTC-USD", func(trade *client.Trade) {
		select {
		case tradeChan <- trade:
		default:
		}
	})
	require.NoError(t, err)
	
	// Place orders to generate events
	go func() {
		for i := 0; i < 5; i++ {
			order := &client.Order{
				Symbol: "BTC-USD",
				Type:   client.OrderTypeLimit,
				Side:   client.OrderSideBuy,
				Price:  49000 + float64(i*100),
				Size:   0.01,
				UserID: fmt.Sprintf("ws-test-user-%d", i),
			}
			c.PlaceOrder(ctx, order)
			time.Sleep(100 * time.Millisecond)
		}
	}()
	
	// Wait for events
	orderBookReceived := false
	tradeReceived := false
	
	for {
		select {
		case ob := <-orderBookChan:
			assert.NotNil(t, ob)
			orderBookReceived = true
			
		case trade := <-tradeChan:
			assert.NotNil(t, trade)
			tradeReceived = true
			
		case <-ctx.Done():
			assert.True(t, orderBookReceived, "No order book updates received")
			return
		}
		
		if orderBookReceived && tradeReceived {
			return // Success
		}
	}
}

// TestConcurrentOrders tests high-concurrency order placement
func TestConcurrentOrders(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(client.WithJSONRPCURL(testJSONRPCURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	numWorkers := 10
	ordersPerWorker := 100
	var wg sync.WaitGroup
	errors := make(chan error, numWorkers*ordersPerWorker)
	orderIDs := make(chan uint64, numWorkers*ordersPerWorker)
	
	// Launch workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerWorker; i++ {
				order := &client.Order{
					Symbol: "BTC-USD",
					Type:   client.OrderTypeLimit,
					Side:   client.OrderSide(i % 2), // Alternate buy/sell
					Price:  50000 + float64((i%20-10)*10),
					Size:   0.001,
					UserID: fmt.Sprintf("worker-%d", workerID),
				}
				
				resp, err := c.PlaceOrder(ctx, order)
				if err != nil {
					errors <- err
				} else {
					orderIDs <- resp.OrderID
				}
			}
		}(w)
	}
	
	// Wait for completion
	wg.Wait()
	close(errors)
	close(orderIDs)
	
	// Check results
	errorCount := 0
	for err := range errors {
		t.Logf("Order error: %v", err)
		errorCount++
	}
	
	orderCount := 0
	uniqueOrders := make(map[uint64]bool)
	for id := range orderIDs {
		orderCount++
		uniqueOrders[id] = true
	}
	
	assert.Equal(t, 0, errorCount, "Should have no errors")
	assert.Equal(t, numWorkers*ordersPerWorker, orderCount, "All orders should succeed")
	assert.Equal(t, orderCount, len(uniqueOrders), "All order IDs should be unique")
}

// TestOrderBookDepth tests order book depth and aggregation
func TestOrderBookDepth(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(client.WithJSONRPCURL(testJSONRPCURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Place multiple orders at different price levels
	for i := 0; i < 20; i++ {
		buyOrder := &client.Order{
			Symbol: "BTC-USD",
			Type:   client.OrderTypeLimit,
			Side:   client.OrderSideBuy,
			Price:  49900 - float64(i*10),
			Size:   0.1 + float64(i)*0.01,
			UserID: fmt.Sprintf("depth-test-buy-%d", i),
		}
		
		sellOrder := &client.Order{
			Symbol: "BTC-USD",
			Type:   client.OrderTypeLimit,
			Side:   client.OrderSideSell,
			Price:  50100 + float64(i*10),
			Size:   0.1 + float64(i)*0.01,
			UserID: fmt.Sprintf("depth-test-sell-%d", i),
		}
		
		_, err = c.PlaceOrder(ctx, buyOrder)
		require.NoError(t, err)
		
		_, err = c.PlaceOrder(ctx, sellOrder)
		require.NoError(t, err)
	}
	
	// Get order book with depth
	orderBook, err := c.GetOrderBook(ctx, "BTC-USD", 10)
	require.NoError(t, err)
	
	assert.LessOrEqual(t, len(orderBook.Bids), 10)
	assert.LessOrEqual(t, len(orderBook.Asks), 10)
	
	// Verify price ordering
	for i := 1; i < len(orderBook.Bids); i++ {
		assert.GreaterOrEqual(t, orderBook.Bids[i-1].Price, orderBook.Bids[i].Price,
			"Bids should be in descending price order")
	}
	
	for i := 1; i < len(orderBook.Asks); i++ {
		assert.LessOrEqual(t, orderBook.Asks[i-1].Price, orderBook.Asks[i].Price,
			"Asks should be in ascending price order")
	}
	
	// Verify spread
	if len(orderBook.Bids) > 0 && len(orderBook.Asks) > 0 {
		spread := orderBook.Spread()
		assert.Greater(t, spread, 0.0, "Spread should be positive")
	}
}

// TestOrderCancellation tests order cancellation
func TestOrderCancellation(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(client.WithJSONRPCURL(testJSONRPCURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Place an order
	order := &client.Order{
		Symbol:   "BTC-USD",
		Type:     client.OrderTypeLimit,
		Side:     client.OrderSideBuy,
		Price:    45000, // Far from market to avoid immediate fill
		Size:     1.0,
		UserID:   "cancel-test-user",
		ClientID: "cancel-test-001",
	}
	
	resp, err := c.PlaceOrder(ctx, order)
	require.NoError(t, err)
	assert.NotZero(t, resp.OrderID)
	
	// Cancel the order
	err = c.CancelOrder(ctx, resp.OrderID)
	require.NoError(t, err)
	
	// Try to cancel again (should fail or be idempotent)
	err = c.CancelOrder(ctx, resp.OrderID)
	// This may or may not error depending on implementation
	if err != nil {
		t.Logf("Double cancel error (expected): %v", err)
	}
}

// TestNodeHealth tests node health endpoints
func TestNodeHealth(t *testing.T) {
	// Test JSON-RPC health
	resp, err := http.Get(testJSONRPCURL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close()
	
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	
	var health map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&health)
	require.NoError(t, err)
	
	assert.Equal(t, "healthy", health["status"])
	assert.NotZero(t, health["block"])
	assert.NotZero(t, health["orders"])
}

// TestPerformanceMetrics tests that performance meets requirements
func TestPerformanceMetrics(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(
		client.WithJSONRPCURL(testJSONRPCURL),
		client.WithGRPCURL(testGRPCURL),
	)
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Try to connect to gRPC for best performance
	c.ConnectGRPC(ctx)
	
	// Measure order placement latency
	numOrders := 1000
	latencies := make([]time.Duration, numOrders)
	
	for i := 0; i < numOrders; i++ {
		order := &client.Order{
			Symbol: "BTC-USD",
			Type:   client.OrderTypeLimit,
			Side:   client.OrderSide(i % 2),
			Price:  50000 + float64((i%100)-50),
			Size:   0.001,
			UserID: "perf-test",
		}
		
		start := time.Now()
		_, err := c.PlaceOrder(ctx, order)
		latencies[i] = time.Since(start)
		
		require.NoError(t, err)
	}
	
	// Calculate statistics
	var totalLatency time.Duration
	var maxLatency time.Duration
	for _, lat := range latencies {
		totalLatency += lat
		if lat > maxLatency {
			maxLatency = lat
		}
	}
	
	avgLatency := totalLatency / time.Duration(numOrders)
	
	t.Logf("Performance Metrics:")
	t.Logf("  Average latency: %v", avgLatency)
	t.Logf("  Max latency: %v", maxLatency)
	t.Logf("  Throughput: %.0f orders/sec", float64(numOrders)/totalLatency.Seconds())
	
	// Assert performance requirements
	assert.Less(t, avgLatency, 10*time.Millisecond, "Average latency should be < 10ms")
	assert.Less(t, maxLatency, 100*time.Millisecond, "Max latency should be < 100ms")
}

// TestErrorHandling tests error scenarios
func TestErrorHandling(t *testing.T) {
	ctx := context.Background()
	
	c, err := client.NewClient(client.WithJSONRPCURL(testJSONRPCURL))
	require.NoError(t, err)
	defer c.Disconnect()
	
	// Test invalid order (negative price)
	invalidOrder := &client.Order{
		Symbol: "BTC-USD",
		Type:   client.OrderTypeLimit,
		Side:   client.OrderSideBuy,
		Price:  -100, // Invalid
		Size:   0.1,
		UserID: "error-test",
	}
	
	_, err = c.PlaceOrder(ctx, invalidOrder)
	assert.Error(t, err, "Should reject negative price")
	
	// Test invalid symbol
	invalidSymbol := &client.Order{
		Symbol: "INVALID-PAIR",
		Type:   client.OrderTypeLimit,
		Side:   client.OrderSideBuy,
		Price:  50000,
		Size:   0.1,
		UserID: "error-test",
	}
	
	_, err = c.PlaceOrder(ctx, invalidSymbol)
	// May or may not error depending on implementation
	if err != nil {
		t.Logf("Invalid symbol error (expected): %v", err)
	}
	
	// Test cancelling non-existent order
	err = c.CancelOrder(ctx, 999999999)
	assert.Error(t, err, "Should error when cancelling non-existent order")
}