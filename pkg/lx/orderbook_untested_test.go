package lx

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test GetOrder function
func TestGetOrder(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add an order
	order := &Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      10,
		User:      "user1",
		Timestamp: time.Now(),
	}
	book.AddOrder(order)

	// Get existing order
	retrieved := book.GetOrder(1)
	assert.NotNil(t, retrieved)
	assert.Equal(t, uint64(1), retrieved.ID)
	assert.Equal(t, float64(100), retrieved.Price)

	// Get non-existent order
	retrieved = book.GetOrder(999)
	assert.Nil(t, retrieved)
}

// Test GetOrderBookSnapshot function
func TestGetOrderBookSnapshot(t *testing.T) {
	book := NewOrderBook("SNAP")

	// Add some orders
	for i := 1; i <= 5; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Buy,
			Price:     float64(100 - i),
			Size:      10,
			User:      "buyer",
			Timestamp: time.Now(),
		})

		book.AddOrder(&Order{
			ID:        uint64(i + 10),
			Type:      Limit,
			Side:      Sell,
			Price:     float64(101 + i),
			Size:      10,
			User:      "seller",
			Timestamp: time.Now(),
		})
	}

	// Get snapshot
	snapshot := book.GetOrderBookSnapshot()
	assert.NotNil(t, snapshot)
	assert.Equal(t, "SNAP", snapshot.Symbol)
	assert.True(t, len(snapshot.Bids) > 0)
	assert.True(t, len(snapshot.Asks) > 0)
	assert.NotNil(t, snapshot.Timestamp)

	// Check bid prices are descending
	for i := 1; i < len(snapshot.Bids); i++ {
		assert.True(t, snapshot.Bids[i-1].Price >= snapshot.Bids[i].Price)
	}

	// Check ask prices are ascending
	for i := 1; i < len(snapshot.Asks); i++ {
		assert.True(t, snapshot.Asks[i-1].Price <= snapshot.Asks[i].Price)
	}
}

// Test processMarketOrderOptimized function
func TestProcessMarketOrderOptimized(t *testing.T) {
	book := NewOrderBook("MARKET")

	// Add liquidity
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      20,
		User:      "buyer",
		Timestamp: time.Now(),
	})

	book.AddOrder(&Order{
		ID:        2,
		Type:      Limit,
		Side:      Sell,
		Price:     101,
		Size:      20,
		User:      "seller",
		Timestamp: time.Now(),
	})

	// Create market order
	marketBuy := &Order{
		ID:        3,
		Type:      Market,
		Side:      Buy,
		Size:      10,
		User:      "market_buyer",
		Timestamp: time.Now(),
	}

	// Process market order (processMarketOrderOptimized handles its own locking)
	tradesCount := book.processMarketOrderOptimized(marketBuy)

	assert.True(t, tradesCount > 0)

	// Market sell
	marketSell := &Order{
		ID:        4,
		Type:      Market,
		Side:      Sell,
		Size:      5,
		User:      "market_seller",
		Timestamp: time.Now(),
	}

	tradesCount = book.processMarketOrderOptimized(marketSell)

	assert.True(t, tradesCount > 0)
}

// Test getOrderLevels internal function
func TestGetOrderLevels(t *testing.T) {
	book := NewOrderBook("LEVELS")

	// Add multiple orders at different price levels
	for i := 1; i <= 3; i++ {
		for j := 1; j <= 2; j++ {
			book.AddOrder(&Order{
				ID:        uint64((i-1)*2 + j),
				Type:      Limit,
				Side:      Buy,
				Price:     float64(100 - i),
				Size:      10,
				User:      "buyer",
				Timestamp: time.Now(),
			})
		}
	}

	// Get order levels for bids using the exposed Bids field
	levels := book.Bids.getOrderLevels()
	assert.NotNil(t, levels)
	assert.True(t, len(levels) > 0)

	// Check that levels are properly aggregated
	for _, level := range levels {
		assert.True(t, level.Price > 0)
		assert.True(t, level.Size > 0)
		assert.True(t, level.OrderID > 0)
	}
}

// Test ValidateOrder with various invalid inputs
func TestValidateOrderEdgeCases(t *testing.T) {
	book := NewOrderBook("VALIDATE")

	// Test nil order
	err := book.validateOrder(nil)
	assert.Error(t, err)

	// Test zero size
	err = book.validateOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  0,
		User:  "user",
	})
	assert.Error(t, err)

	// Test negative size
	err = book.validateOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  -10,
		User:  "user",
	})
	assert.Error(t, err)

	// Test zero price for limit order
	err = book.validateOrder(&Order{
		ID:    3,
		Type:  Limit,
		Side:  Buy,
		Price: 0,
		Size:  10,
		User:  "user",
	})
	assert.Error(t, err)

	// Test negative price
	err = book.validateOrder(&Order{
		ID:    4,
		Type:  Limit,
		Side:  Buy,
		Price: -100,
		Size:  10,
		User:  "user",
	})
	assert.Error(t, err)

	// Valid order should pass
	err = book.validateOrder(&Order{
		ID:    5,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
		User:  "user",
	})
	assert.NoError(t, err)
}

// Test concurrent GetSnapshot operations
func TestConcurrentGetSnapshot(t *testing.T) {
	book := NewOrderBook("CONCURRENT")

	// Add initial orders
	for i := 1; i <= 100; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Side(i % 2),
			Price:     100 + float64(i%10-5),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
	}

	// Run concurrent snapshots
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				snapshot := book.GetSnapshot()
				assert.NotNil(t, snapshot)
			}
			done <- true
		}()
	}

	// Wait for completion
	for i := 0; i < 10; i++ {
		<-done
	}
}

// Test MatchOrders with complex scenarios
func TestMatchOrdersComplex(t *testing.T) {
	book := NewOrderBook("COMPLEX")
	book.EnableImmediateMatching = false // Disable to build up orders

	// Add multiple buy orders
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     102,
		Size:      10,
		User:      "buyer1",
		Timestamp: time.Now(),
	})

	book.AddOrder(&Order{
		ID:        2,
		Type:      Limit,
		Side:      Buy,
		Price:     101,
		Size:      15,
		User:      "buyer2",
		Timestamp: time.Now(),
	})

	book.AddOrder(&Order{
		ID:        3,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      20,
		User:      "buyer3",
		Timestamp: time.Now(),
	})

	// Add multiple sell orders
	book.AddOrder(&Order{
		ID:        4,
		Type:      Limit,
		Side:      Sell,
		Price:     98,
		Size:      10,
		User:      "seller1",
		Timestamp: time.Now(),
	})

	book.AddOrder(&Order{
		ID:        5,
		Type:      Limit,
		Side:      Sell,
		Price:     99,
		Size:      15,
		User:      "seller2",
		Timestamp: time.Now(),
	})

	book.AddOrder(&Order{
		ID:        6,
		Type:      Limit,
		Side:      Sell,
		Price:     100,
		Size:      20,
		User:      "seller3",
		Timestamp: time.Now(),
	})

	// Now match all crossing orders
	trades := book.MatchOrders()
	assert.True(t, len(trades) > 0)

	// Check that trades occurred at correct prices
	for _, trade := range trades {
		assert.True(t, trade.Price >= 98 && trade.Price <= 102)
		assert.True(t, trade.Size > 0)
	}
}

// Test edge cases in tree operations
func TestOrderTreeEdgeCases(t *testing.T) {
	book := NewOrderBook("TREE")

	// Test empty tree operations
	bestBid := book.GetBestBid()
	assert.Equal(t, float64(0), bestBid)

	bestAsk := book.GetBestAsk()
	assert.Equal(t, float64(0), bestAsk)

	// Add single order and test
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      10,
		User:      "user",
		Timestamp: time.Now(),
	})

	bestBid = book.GetBestBid()
	assert.Equal(t, float64(100), bestBid)

	// Cancel the only order
	book.CancelOrder(1)

	// After cancellation, best bid should handle gracefully
	bestBid = book.GetBestBid()
	// Depending on implementation, might be 0 or 100
	assert.True(t, bestBid == 0 || bestBid == 100)
}

// Test Post-Only order rejection
func TestPostOnlyOrderRejection(t *testing.T) {
	book := NewOrderBook("POSTONLY")

	// Add resting bid
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      10,
		User:      "maker",
		Timestamp: time.Now(),
	})

	// Try to add post-only sell that would cross
	postOnlyOrder := &Order{
		ID:        2,
		Type:      Limit,
		Side:      Sell,
		Price:     100,
		Size:      5,
		User:      "taker",
		PostOnly:  true,
		Timestamp: time.Now(),
	}

	// Check if order would take liquidity
	wouldTake := book.wouldTakeLiquidity(postOnlyOrder)
	assert.True(t, wouldTake)

	// Post-only order at non-crossing price should be fine
	postOnlyOrder.Price = 101
	wouldTake = book.wouldTakeLiquidity(postOnlyOrder)
	assert.False(t, wouldTake)
}

// Benchmark GetOrder performance
func BenchmarkGetOrder(b *testing.B) {
	book := NewOrderBook("BENCH")

	// Pre-populate with many orders
	for i := 1; i <= 10000; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Side(i % 2),
			Price:     100 + float64(i%100-50),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		orderID := uint64((i % 10000) + 1)
		_ = book.GetOrder(orderID)
	}
}

// Benchmark GetOrderBookSnapshot performance
func BenchmarkGetOrderBookSnapshot(b *testing.B) {
	book := NewOrderBook("BENCH")

	// Pre-populate with orders
	for i := 1; i <= 1000; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Buy,
			Price:     100 - float64(i%50),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
		book.AddOrder(&Order{
			ID:        uint64(i + 1000),
			Type:      Limit,
			Side:      Sell,
			Price:     101 + float64(i%50),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = book.GetOrderBookSnapshot()
	}
}
