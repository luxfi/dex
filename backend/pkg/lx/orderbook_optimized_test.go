package lx

import (
	"testing"
	"time"
)

// TestOptimizedOrderBook tests the basic functionality of the optimized order book
func TestOptimizedOrderBook(t *testing.T) {
	ob := NewOrderBook("BTC-USD")
	ob.EnableImmediateMatching = true // Enable aggressive matching for optimized mode

	// Test 1: Add buy order
	buyOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "user1",
	}

	orderID := ob.AddOrder(buyOrder)
	if orderID == 0 {
		t.Error("Failed to add buy order")
	}

	// Test 2: Add sell order that should match
	sellOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  49999,
		Size:   0.5,
		User:   "user2",
	}

	numTrades := ob.AddOrder(sellOrder)
	if numTrades == 0 {
		t.Error("Orders should have matched")
	}

	// Test 3: Check trade was created
	if len(ob.Trades) == 0 {
		t.Error("No trades created")
	}

	// Test 4: Market order
	marketOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Market,
		Size:   0.5,
		User:   "user3",
	}

	// Add a sell order first for market order to match
	ob.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  50100,
		Size:   1.0,
		User:   "user4",
	})

	numTrades = ob.AddOrder(marketOrder)
	if numTrades == 0 {
		t.Error("Market order should have matched")
	}

	// Test 5: Cancel order
	cancelOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  49000,
		Size:   1.0,
		User:   "user5",
	}

	cancelID := ob.AddOrder(cancelOrder)
	err := ob.CancelOrder(cancelID)
	if err != nil {
		t.Errorf("Failed to cancel order: %v", err)
	}

	// Test 6: Get snapshot
	snapshot := ob.GetSnapshot()
	if snapshot == nil || snapshot.Symbol != "BTC-USD" {
		t.Error("Failed to get snapshot")
	}

	// Test 7: Get depth
	depth := ob.GetDepth(5)
	if depth == nil {
		t.Error("Failed to get depth")
	}
}

// BenchmarkOptimizedOrderBook benchmarks the optimized order book
func BenchmarkOptimizedOrderBook(b *testing.B) {
	ob := NewOrderBook("BTC-USD")

	// Pre-populate with some orders
	for i := 0; i < 100; i++ {
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  float64(49000 + i*10),
			Size:   1.0,
			User:   "user",
		})
		ob.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  float64(51000 + i*10),
			Size:   1.0,
			User:   "user",
		})
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		order := &Order{
			Symbol:    "BTC-USD",
			Side:      Side(i % 2),
			Type:      Limit,
			Price:     50000 + float64(i%100),
			Size:      1.0,
			User:      "bench_user",
			Timestamp: time.Now(),
		}
		ob.AddOrder(order)
	}
}
