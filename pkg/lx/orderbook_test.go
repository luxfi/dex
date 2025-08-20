package lx

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestOrderBookBasicOperations(t *testing.T) {
	book := NewOrderBook("BTC-USD")
	book.EnableImmediateMatching = true

	// Test adding buy order
	buyOrder := &Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     50000,
		Size:      1.0,
		Timestamp: time.Now(),
	}

	orderID := book.AddOrder(buyOrder)
	if orderID != 1 {
		t.Errorf("Expected order ID 1 for first order, got %d", orderID)
	}

	// No trades should be created yet
	if len(book.GetTrades()) != 0 {
		t.Errorf("Expected no trades for first order, got %d", len(book.GetTrades()))
	}

	// Test adding sell order that matches
	sellOrder := &Order{
		ID:        2,
		Type:      Limit,
		Side:      Sell,
		Price:     50000,
		Size:      0.5,
		Timestamp: time.Now(),
	}

	orderID2 := book.AddOrder(sellOrder)
	if orderID2 != 2 {
		t.Errorf("Expected order ID 2 for sell order, got %d", orderID2)
	}

	// Check that a trade was created
	trades := book.GetTrades()
	if len(trades) != 1 {
		t.Errorf("Expected 1 trade, got %d", len(trades))
	}

	// Check trade was recorded correctly
	if len(trades) > 0 && trades[0].Size != 0.5 {
		t.Errorf("Expected trade size 0.5, got %f", trades[0].Size)
	}
}

func TestOrderBookBestPrices(t *testing.T) {
	book := NewOrderBook("ETH-USD")

	// Add buy orders
	book.AddOrder(&Order{
		ID: 1, Type: Limit, Side: Buy, Price: 3000, Size: 1.0,
		Timestamp: time.Now(),
	})
	book.AddOrder(&Order{
		ID: 2, Type: Limit, Side: Buy, Price: 3001, Size: 1.0,
		Timestamp: time.Now(),
	})

	// Add sell orders
	book.AddOrder(&Order{
		ID: 3, Type: Limit, Side: Sell, Price: 3002, Size: 1.0,
		Timestamp: time.Now(),
	})
	book.AddOrder(&Order{
		ID: 4, Type: Limit, Side: Sell, Price: 3003, Size: 1.0,
		Timestamp: time.Now(),
	})

	bestBid := book.GetBestBid()
	if bestBid != 3001 {
		t.Errorf("Expected best bid 3001, got %f", bestBid)
	}

	bestAsk := book.GetBestAsk()
	if bestAsk != 3002 {
		t.Errorf("Expected best ask 3002, got %f", bestAsk)
	}
}

func TestOrderBookCancelOrder(t *testing.T) {
	book := NewOrderBook("SOL-USD")

	order := &Order{
		ID: 1, Type: Limit, Side: Buy, Price: 100, Size: 10,
		Timestamp: time.Now(),
	}

	book.AddOrder(order)

	// Cancel the order
	err := book.CancelOrder(1)
	if err != nil {
		t.Errorf("Failed to cancel order: %v", err)
	}

	// Try to cancel again
	err = book.CancelOrder(1)
	if err == nil {
		t.Error("Expected error when canceling non-existent order")
	}
}

func TestOrderBookMarketOrder(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add liquidity
	book.AddOrder(&Order{
		ID: 1, Type: Limit, Side: Sell, Price: 50000, Size: 1.0,
		Timestamp: time.Now(),
	})
	book.AddOrder(&Order{
		ID: 2, Type: Limit, Side: Sell, Price: 50001, Size: 1.0,
		Timestamp: time.Now(),
	})

	// Market buy order
	marketOrder := &Order{
		ID: 3, Type: Market, Side: Buy, Size: 1.5,
		Timestamp: time.Now(),
	}

	// Market orders should get an ID and execute immediately
	orderID := book.AddOrder(marketOrder)
	if orderID != 3 {
		t.Errorf("Expected order ID 3 for market order, got %d", orderID)
	}

	// Check that trades were created
	trades := book.GetTrades()
	// We expect at least 2 trades from the market order matching the two limit orders
	if len(trades) < 2 {
		t.Errorf("Expected at least 2 trades, got %d", len(trades))
	}

	// Check total traded size from market order
	totalSize := 0.0
	for _, trade := range trades {
		totalSize += trade.Size
	}

	if totalSize != 1.5 {
		t.Errorf("Expected total trade size 1.5, got %f", totalSize)
	}
}

func TestOrderBookConcurrency(t *testing.T) {
	book := NewOrderBook("STRESS-TEST")
	book.EnableImmediateMatching = true

	numGoroutines := 100
	ordersPerGoroutine := 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < ordersPerGoroutine; j++ {
				order := &Order{
					ID:        uint64(id*ordersPerGoroutine + j),
					Type:      Limit,
					Side:      Side(rand.Intn(2)),
					Price:     50000 + float64(rand.Intn(1000)),
					Size:      rand.Float64() * 10,
					Timestamp: time.Now(),
				}

				book.AddOrder(order)

				// Randomly cancel some orders
				if rand.Float32() < 0.3 {
					book.CancelOrder(order.ID)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify book is in consistent state
	bestBid := book.GetBestBid()
	bestAsk := book.GetBestAsk()

	// Just verify we have some orders
	if bestBid == 0 && bestAsk == 0 {
		t.Errorf("No orders in book after concurrent operations")
	}
}

func TestOrderBookSelfTradePrevention(t *testing.T) {
	book := NewOrderBook("STP-TEST")
	book.EnableImmediateMatching = true

	// Add orders from same user
	userID := "trader1"

	book.AddOrder(&Order{
		ID: 1, Type: Limit, Side: Buy, Price: 100, Size: 10,
		UserID: userID, Timestamp: time.Now(),
	})

	// This should not match with the previous order (self-trade prevention)
	orderID := book.AddOrder(&Order{
		ID: 2, Type: Limit, Side: Sell, Price: 100, Size: 10,
		UserID: userID, Timestamp: time.Now(),
	})

	// Order should be rejected (ID = 0) due to self-trade
	if orderID != 0 {
		t.Error("Self-trade prevention failed: orders from same user matched")
	}

	// Order from different user should match and create order
	orderID2 := book.AddOrder(&Order{
		ID: 3, Type: Limit, Side: Sell, Price: 100, Size: 10,
		UserID: "trader2", Timestamp: time.Now(),
	})

	// Should succeed and return order ID
	if orderID2 != 3 {
		t.Errorf("Expected order ID 3 from different user, got %d", orderID2)
	}

	// Verify a trade was created by checking the order book state
	trades := book.GetTrades()
	if len(trades) != 1 {
		t.Errorf("Expected 1 trade to be created, got %d", len(trades))
	}
}

func TestOrderBookPartialFills(t *testing.T) {
	book := NewOrderBook("PARTIAL-TEST")
	book.EnableImmediateMatching = true

	// Large buy order
	book.AddOrder(&Order{
		ID: 1, Type: Limit, Side: Buy, Price: 100, Size: 100,
		Timestamp: time.Now(),
	})

	// Small sell orders that partially fill
	for i := 0; i < 5; i++ {
		orderID := book.AddOrder(&Order{
			ID: uint64(i + 2), Type: Limit, Side: Sell,
			Price: 100, Size: 10, Timestamp: time.Now(),
		})

		// Each order should be added successfully
		if orderID != uint64(i+2) {
			t.Errorf("Expected order ID %d for partial fill %d, got %d", i+2, i, orderID)
		}
	}

	// Verify all trades were created
	trades := book.GetTrades()
	if len(trades) != 5 {
		t.Errorf("Expected 5 trades from partial fills, got %d", len(trades))
	}

	// Check remaining order
	bestBid := book.GetBestBid()
	if bestBid != 100 {
		t.Errorf("Expected remaining buy order at 100, got %f", bestBid)
	}
}

func TestOrderBookSnapshot(t *testing.T) {
	book := NewOrderBook("SNAPSHOT-TEST")

	// Add some orders
	for i := 0; i < 10; i++ {
		book.AddOrder(&Order{
			ID: uint64(i), Type: Limit, Side: Buy,
			Price: 100 - float64(i), Size: 1.0,
			Timestamp: time.Now(),
		})
		book.AddOrder(&Order{
			ID: uint64(i + 10), Type: Limit, Side: Sell,
			Price: 101 + float64(i), Size: 1.0,
			Timestamp: time.Now(),
		})
	}

	snapshot := book.GetSnapshot()

	if snapshot.Symbol != "SNAPSHOT-TEST" {
		t.Errorf("Expected symbol SNAPSHOT-TEST, got %s", snapshot.Symbol)
	}

	if len(snapshot.Bids) != 10 {
		t.Errorf("Expected 10 bids, got %d", len(snapshot.Bids))
	}

	if len(snapshot.Asks) != 10 {
		t.Errorf("Expected 10 asks, got %d", len(snapshot.Asks))
	}

	// Check order is correct (best bid first)
	if len(snapshot.Bids) > 0 && snapshot.Bids[0].Price != 100 {
		t.Errorf("Expected best bid 100, got %f", snapshot.Bids[0].Price)
	}

	if len(snapshot.Asks) > 0 && snapshot.Asks[0].Price != 101 {
		t.Errorf("Expected best ask 101, got %f", snapshot.Asks[0].Price)
	}
}

func BenchmarkOrderBookAddOrder(b *testing.B) {
	book := NewOrderBook("BENCH")

	orders := make([]*Order, b.N)
	for i := 0; i < b.N; i++ {
		orders[i] = &Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Side(i % 2),
			Price:     50000 + float64(i%1000),
			Size:      1.0,
			Timestamp: time.Now(),
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		book.AddOrder(orders[i])
	}
}

func BenchmarkOrderBookConcurrent(b *testing.B) {
	book := NewOrderBook("BENCH-CONCURRENT")

	b.RunParallel(func(pb *testing.PB) {
		id := uint64(rand.Int63())
		for pb.Next() {
			order := &Order{
				ID:        id,
				Type:      Limit,
				Side:      Side(rand.Intn(2)),
				Price:     50000 + float64(rand.Intn(1000)),
				Size:      rand.Float64() * 10,
				Timestamp: time.Now(),
			}
			book.AddOrder(order)
			id++
		}
	})
}
