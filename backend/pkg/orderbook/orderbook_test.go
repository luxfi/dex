package orderbook

import (
	"testing"
)

func TestOrderBook(t *testing.T) {
	// Create order book
	ob := NewOrderBook(Config{
		Symbol:         "TEST",
		Implementation: ImplGo,
	})
	
	// Add buy order
	buyOrder := &Order{
		ID:       1,
		Symbol:   "TEST",
		Side:     Buy,
		Price:    100.0,
		Quantity: 10.0,
	}
	ob.AddOrder(buyOrder)
	
	// Add sell order
	sellOrder := &Order{
		ID:       2,
		Symbol:   "TEST",
		Side:     Sell,
		Price:    101.0,
		Quantity: 10.0,
	}
	ob.AddOrder(sellOrder)
	
	// Get best bid/ask
	bestBid := ob.GetBestBid()
	if bestBid != 100.0 {
		t.Errorf("Expected best bid 100.0, got %v", bestBid)
	}
	
	bestAsk := ob.GetBestAsk()
	if bestAsk != 101.0 {
		t.Errorf("Expected best ask 101.0, got %v", bestAsk)
	}
	
	// Test matching
	matchBuy := &Order{
		ID:       3,
		Symbol:   "TEST",
		Side:     Buy,
		Price:    101.0, // Crosses spread
		Quantity: 5.0,
	}
	ob.AddOrder(matchBuy)
	
	// Should have executed trade
	trades := ob.MatchOrders()
	if len(trades) == 0 {
		t.Error("Expected trade to execute")
	}
}

func BenchmarkOrderBookAddOrder(b *testing.B) {
	ob := NewOrderBook(Config{
		Symbol:         "BENCH",
		Implementation: ImplGo,
	})
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		order := &Order{
			ID:       uint64(i),
			Symbol:   "BENCH",
			Side:     OrderSide(i % 2),
			Price:    100.0 + float64(i%100),
			Quantity: 1.0,
		}
		ob.AddOrder(order)
	}
}

func BenchmarkOrderBookMatch(b *testing.B) {
	ob := NewOrderBook(Config{
		Symbol:         "BENCH",
		Implementation: ImplGo,
	})
	
	// Pre-populate order book
	for i := 0; i < 1000; i++ {
		order := &Order{
			ID:       uint64(i),
			Symbol:   "BENCH",
			Side:     OrderSide(i % 2),
			Price:    100.0 + float64(i%100),
			Quantity: 1.0,
		}
		ob.AddOrder(order)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ob.MatchOrders()
	}
}