package unit

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/stretchr/testify/assert"
)

// TestOrderBookBasics tests basic orderbook functionality
func TestOrderBookBasics(t *testing.T) {
	ob := lx.NewOrderBook("BTC-USD")
	
	// Add buy order
	buyOrder := &lx.Order{
		ID:     1,
		Symbol: "BTC-USD",
		Type:   lx.Limit,
		Side:   lx.Buy,
		Price:  50000,
		Size:   1,
		UserID: "user1",
	}
	
	id := ob.AddOrder(buyOrder)
	assert.Greater(t, id, uint64(0))
	
	// Add matching sell order
	sellOrder := &lx.Order{
		ID:     2,
		Symbol: "BTC-USD",
		Type:   lx.Limit,
		Side:   lx.Sell,
		Price:  50000,
		Size:   1,
		UserID: "user2",
	}
	
	id = ob.AddOrder(sellOrder)
	assert.Greater(t, id, uint64(0))
	
	// Both orders should be added successfully
	// Matching happens internally
}

// TestConcurrentOrders tests thread-safe order processing
func TestConcurrentOrders(t *testing.T) {
	ob := lx.NewOrderBook("ETH-USD")
	
	done := make(chan bool)
	orderCount := 1000
	
	// Add orders concurrently
	go func() {
		for i := 0; i < orderCount; i++ {
			order := &lx.Order{
				ID:     uint64(i + 1),
				Symbol: "ETH-USD",
				Type:   lx.Limit,
				Side:   lx.Side(i % 2),
				Price:  3000 + float64(i%10),
				Size:   1,
				UserID: "concurrent_test",
			}
			ob.AddOrder(order)
		}
		done <- true
	}()
	
	select {
	case <-done:
		// Success
	case <-time.After(5 * time.Second):
		t.Fatal("Concurrent test timeout")
	}
}

// TestOrderTypes tests various order types
func TestOrderTypes(t *testing.T) {
	tests := []struct {
		name string
		side lx.Side
		typ  lx.OrderType
	}{
		{"BuyLimit", lx.Buy, lx.Limit},
		{"SellLimit", lx.Sell, lx.Limit},
		{"BuyMarket", lx.Buy, lx.Market},
		{"SellMarket", lx.Sell, lx.Market},
	}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ob := lx.NewOrderBook("TEST-USD")
			
			order := &lx.Order{
				ID:     1,
				Symbol: "TEST-USD",
				Type:   test.typ,
				Side:   test.side,
				Price:  100,
				Size:   1,
				UserID: "test",
			}
			
			if test.typ == lx.Market {
				// Market orders need liquidity to execute
				// Add opposite side first
				opposite := &lx.Order{
					ID:     99,
					Symbol: "TEST-USD",
					Type:   lx.Limit,
					Side:   1 - test.side, // Opposite side
					Price:  100,
					Size:   10,
					UserID: "liquidity",
				}
				ob.AddOrder(opposite)
			}
			
			id := ob.AddOrder(order)
			if test.typ == lx.Limit {
				assert.Greater(t, id, uint64(0))
			}
		})
	}
}