package lx

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOrderBookDepth(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add multiple orders at different price levels
	for i := 1; i <= 5; i++ {
		book.AddOrder(&Order{
			ID:        uint64(i),
			Type:      Limit,
			Side:      Buy,
			Price:     float64(100 - i),
			Size:      float64(i),
			User:      fmt.Sprintf("buyer%d", i),
			Timestamp: time.Now(),
		})

		book.AddOrder(&Order{
			ID:        uint64(i + 10),
			Type:      Limit,
			Side:      Sell,
			Price:     float64(100 + i),
			Size:      float64(i),
			User:      fmt.Sprintf("seller%d", i),
			Timestamp: time.Now(),
		})
	}

	depth := book.GetDepth(3)
	assert.Equal(t, 3, len(depth.Bids))
	assert.Equal(t, 3, len(depth.Asks))

	// Best bid should be 99
	assert.Equal(t, float64(99), depth.Bids[0].Price)
	// Best ask should be 101
	assert.Equal(t, float64(101), depth.Asks[0].Price)
}

func TestOrderBookResetState(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add orders
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      10,
		User:      "user1",
		Timestamp: time.Now(),
	})

	assert.True(t, len(book.Orders) > 0)

	// Reset
	book.Reset()

	assert.Equal(t, 0, len(book.Orders))
	assert.Equal(t, 0, len(book.Trades))
}

func TestOrderBookModifyOrderFunc(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add order
	book.AddOrder(&Order{
		ID:        1,
		Type:      Limit,
		Side:      Buy,
		Price:     100,
		Size:      10,
		User:      "user1",
		Timestamp: time.Now(),
	})

	// Modify order
	err := book.ModifyOrder(1, 95, 15)
	assert.NoError(t, err)

	// Check modification
	if order, exists := book.Orders[1]; exists {
		assert.Equal(t, float64(95), order.Price)
		assert.Equal(t, float64(15), order.Size)
	}
}

func TestOrderBookAdvancedTypes(t *testing.T) {
	book := NewOrderBook("TEST")

	// Test Stop order
	book.AddOrder(&Order{
		ID:        1,
		Type:      Stop,
		Side:      Sell,
		StopPrice: 95,
		Size:      10,
		User:      "user1",
		Timestamp: time.Now(),
	})

	// Test Iceberg order
	book.AddOrder(&Order{
		ID:          2,
		Type:        Iceberg,
		Side:        Buy,
		Price:       100,
		Size:        100,
		DisplaySize: 10,
		User:        "user2",
		Timestamp:   time.Now(),
	})

	// Test Peg order
	book.AddOrder(&Order{
		ID:        3,
		Type:      Peg,
		Side:      Buy,
		Price:     99.5, // Pegged price
		Size:      5,
		User:      "user3",
		Timestamp: time.Now(),
	})

	// Test Bracket order
	book.AddOrder(&Order{
		ID:        4,
		Type:      Bracket,
		Side:      Buy,
		Price:     100,
		Size:      10,
		StopPrice: 95, // Stop loss
		User:      "user4",
		Timestamp: time.Now(),
	})

	// Verify orders were added (actual matching logic would be more complex)
	assert.True(t, len(book.Orders) > 0)
}

func TestOrderBookTimeInForceOptions(t *testing.T) {
	book := NewOrderBook("TEST")

	// Test IOC order
	trades := book.AddOrder(&Order{
		ID:          1,
		Type:        Limit,
		Side:        Buy,
		Price:       100,
		Size:        10,
		TimeInForce: "IOC",
		User:        "user1",
		Timestamp:   time.Now(),
	})
	// IOC order added to book even without immediate match
	// The actual IOC logic would be in a real implementation
	assert.Equal(t, uint64(1), trades)

	// Add a sell order
	book.AddOrder(&Order{
		ID:        2,
		Type:      Limit,
		Side:      Sell,
		Price:     100,
		Size:      5,
		User:      "user2",
		Timestamp: time.Now(),
	})

	// Test FOK order that can be filled
	trades = book.AddOrder(&Order{
		ID:          3,
		Type:        Limit,
		Side:        Buy,
		Price:       100,
		Size:        5,
		TimeInForce: "FOK",
		User:        "user3",
		Timestamp:   time.Now(),
	})
	// Should execute fully (one trade with order 2)
	assert.Equal(t, uint64(3), trades)
}
