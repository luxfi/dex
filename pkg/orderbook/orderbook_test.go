package orderbook

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGoOrderBook(t *testing.T) {
	config := Config{Symbol: "BTC/USDC"}
	ob := NewGoOrderBook(config)

	assert.NotNil(t, ob)
	assert.Equal(t, "BTC/USDC", ob.symbol)
	assert.NotNil(t, ob.orders)
	assert.Equal(t, uint64(1), ob.nextID)
}

func TestAddOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "ETH/USDC"})

	order := &Order{
		Price:    3000.0,
		Quantity: 10.0,
		Side:     Buy,
	}

	id := ob.AddOrder(order)
	assert.Equal(t, uint64(1), id)
	assert.Equal(t, uint64(1), order.ID)

	// Add another order
	order2 := &Order{
		Price:    3100.0,
		Quantity: 5.0,
		Side:     Sell,
	}
	id2 := ob.AddOrder(order2)
	assert.Equal(t, uint64(2), id2)
}

func TestAddOrderInvalid(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Nil order
	id := ob.AddOrder(nil)
	assert.Equal(t, uint64(0), id)

	// Zero quantity
	id = ob.AddOrder(&Order{Price: 100.0, Quantity: 0, Side: Buy})
	assert.Equal(t, uint64(0), id)

	// Negative quantity
	id = ob.AddOrder(&Order{Price: 100.0, Quantity: -1, Side: Buy})
	assert.Equal(t, uint64(0), id)

	// Negative price
	id = ob.AddOrder(&Order{Price: -100.0, Quantity: 1, Side: Buy})
	assert.Equal(t, uint64(0), id)
}

func TestCancelOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	order := &Order{Price: 50000.0, Quantity: 1.0, Side: Buy}
	id := ob.AddOrder(order)

	// Cancel existing order
	success := ob.CancelOrder(id)
	assert.True(t, success)

	// Cancel again should fail
	success = ob.CancelOrder(id)
	assert.False(t, success)

	// Cancel non-existent order
	success = ob.CancelOrder(999)
	assert.False(t, success)
}

func TestModifyOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	order := &Order{Price: 50000.0, Quantity: 1.0, Side: Buy}
	id := ob.AddOrder(order)

	// Modify existing order
	success := ob.ModifyOrder(id, 51000.0, 2.0)
	assert.True(t, success)

	// Verify modification
	ob.mu.RLock()
	modifiedOrder := ob.orders[id]
	ob.mu.RUnlock()
	assert.Equal(t, 51000.0, modifiedOrder.Price)
	assert.Equal(t, 2.0, modifiedOrder.Quantity)

	// Modify non-existent order
	success = ob.ModifyOrder(999, 50000.0, 1.0)
	assert.False(t, success)
}

func TestMatchOrders(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add buy order at 50000
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Add sell order at 49900 (crossing the spread)
	ob.AddOrder(&Order{Price: 49900.0, Quantity: 0.5, Side: Sell})

	// Match orders
	trades := ob.MatchOrders()

	require.Len(t, trades, 1)
	assert.Equal(t, 49900.0, trades[0].Price)
	assert.Equal(t, 0.5, trades[0].Quantity)
}

func TestMatchOrdersNoMatch(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add buy order at 49000
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})

	// Add sell order at 51000 (no cross)
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	// No match should occur
	trades := ob.MatchOrders()
	assert.Empty(t, trades)
}

func TestMatchOrdersMultipleFills(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "ETH/USDC"})

	// Large buy order
	ob.AddOrder(&Order{Price: 3000.0, Quantity: 10.0, Side: Buy})

	// Multiple small sell orders
	ob.AddOrder(&Order{Price: 2900.0, Quantity: 2.0, Side: Sell})
	ob.AddOrder(&Order{Price: 2950.0, Quantity: 3.0, Side: Sell})
	ob.AddOrder(&Order{Price: 2980.0, Quantity: 2.0, Side: Sell})

	// Should match all sells
	trades := ob.MatchOrders()
	require.Len(t, trades, 3)

	totalFilled := 0.0
	for _, trade := range trades {
		totalFilled += trade.Quantity
	}
	assert.Equal(t, 7.0, totalFilled)
}

func TestGetBestBid(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// No bids
	bid := ob.GetBestBid()
	assert.Equal(t, 0.0, bid)

	// Add bids
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49500.0, Quantity: 2.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49200.0, Quantity: 1.5, Side: Buy})

	bid = ob.GetBestBid()
	assert.Equal(t, 49500.0, bid)
}

func TestGetBestAsk(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// No asks
	ask := ob.GetBestAsk()
	assert.Equal(t, 0.0, ask)

	// Add asks
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})
	ob.AddOrder(&Order{Price: 50500.0, Quantity: 2.0, Side: Sell})
	ob.AddOrder(&Order{Price: 50800.0, Quantity: 1.5, Side: Sell})

	ask = ob.GetBestAsk()
	assert.Equal(t, 50500.0, ask)
}

func TestGetSpread(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Empty book - spread is 0 because both best bid and ask are 0
	bestBid := ob.GetBestBid()
	bestAsk := ob.GetBestAsk()
	assert.Equal(t, 0.0, bestBid)
	assert.Equal(t, 0.0, bestAsk)

	// Add orders
	ob.AddOrder(&Order{Price: 49500.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 50500.0, Quantity: 1.0, Side: Sell})

	bestBid = ob.GetBestBid()
	bestAsk = ob.GetBestAsk()
	spread := bestAsk - bestBid
	assert.Equal(t, 1000.0, spread)
}

func TestGetMidPrice(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Empty book
	bestBid := ob.GetBestBid()
	bestAsk := ob.GetBestAsk()
	assert.Equal(t, 0.0, bestBid)
	assert.Equal(t, 0.0, bestAsk)

	// Add orders
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	bestBid = ob.GetBestBid()
	bestAsk = ob.GetBestAsk()
	mid := (bestBid + bestAsk) / 2
	assert.Equal(t, 50000.0, mid)
}

func TestGetDepth(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add multiple orders
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49500.0, Quantity: 2.0, Side: Buy})
	ob.AddOrder(&Order{Price: 50500.0, Quantity: 1.5, Side: Sell})
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 2.5, Side: Sell})

	depth := ob.GetDepth(10)

	assert.NotNil(t, depth)
	assert.Len(t, depth.Bids, 2)
	assert.Len(t, depth.Asks, 2)
}

func TestGetVolume(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	assert.Equal(t, uint64(0), ob.GetVolume())

	// Execute a trade to add volume
	// Use quantity >= 1 to ensure uint64 cast produces > 0
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 5.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 3.0, Side: Sell})
	trades := ob.MatchOrders()

	// Verify a trade occurred (buy at 50000 matches sell at 49000)
	require.Len(t, trades, 1)
	assert.Equal(t, 3.0, trades[0].Quantity)

	// Volume should be 3 (uint64(3.0) = 3)
	assert.Equal(t, uint64(3), ob.GetVolume())
}

func TestConcurrentOperations(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})
	var wg sync.WaitGroup

	// Concurrent adds
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(price float64) {
			defer wg.Done()
			ob.AddOrder(&Order{
				Price:    price,
				Quantity: 1.0,
				Side:     Buy,
			})
		}(float64(49000 + i))
	}

	// Concurrent reads
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ob.GetBestBid()
			ob.GetBestAsk()
		}()
	}

	wg.Wait()

	// Should have 100 orders
	ob.mu.RLock()
	count := len(ob.orders)
	ob.mu.RUnlock()
	assert.Equal(t, 100, count)
}

func TestOrderTimestamp(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	order := &Order{Price: 50000.0, Quantity: 1.0, Side: Buy}
	assert.True(t, order.Timestamp.IsZero())

	ob.AddOrder(order)
	assert.False(t, order.Timestamp.IsZero())
}

func TestNewOrderBook(t *testing.T) {
	// Test the factory function
	config := Config{Symbol: "SOL/USDC"}
	ob := NewOrderBook(config)
	assert.NotNil(t, ob)
}

func TestOrderBookInterface(t *testing.T) {
	// Verify GoOrderBook implements OrderBook interface
	var _ OrderBook = &GoOrderBook{}
}

func TestDepthLevels(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add many orders
	for i := 0; i < 20; i++ {
		ob.AddOrder(&Order{Price: float64(49000 - i*100), Quantity: 1.0, Side: Buy})
		ob.AddOrder(&Order{Price: float64(51000 + i*100), Quantity: 1.0, Side: Sell})
	}

	// Request only 5 levels
	depth := ob.GetDepth(5)
	assert.LessOrEqual(t, len(depth.Bids), 5)
	assert.LessOrEqual(t, len(depth.Asks), 5)
}

func TestMatchPartialFill(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Buy order larger than sell
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 5.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 2.0, Side: Sell})

	trades := ob.MatchOrders()
	require.Len(t, trades, 1)
	assert.Equal(t, 2.0, trades[0].Quantity)

	// Buy order should still be in the book with remaining quantity
	ob.mu.RLock()
	remaining := 0.0
	for _, o := range ob.orders {
		if o.Side == Buy {
			remaining = o.Quantity
		}
	}
	ob.mu.RUnlock()
	assert.Equal(t, 3.0, remaining)
}

func BenchmarkAddOrder(b *testing.B) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})
	for i := 0; i < b.N; i++ {
		ob.AddOrder(&Order{
			Price:    50000.0 + float64(i%1000),
			Quantity: 1.0,
			Side:     Buy,
		})
	}
}

func BenchmarkMatchOrders(b *testing.B) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Pre-populate
	for i := 0; i < 1000; i++ {
		ob.AddOrder(&Order{Price: float64(49000 + i), Quantity: 1.0, Side: Buy})
		ob.AddOrder(&Order{Price: float64(51000 + i), Quantity: 1.0, Side: Sell})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ob.MatchOrders()
	}
}

// Test canceling a sell order (covers removeAsk)
func TestCancelSellOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a sell order
	order := &Order{Price: 51000.0, Quantity: 1.0, Side: Sell}
	id := ob.AddOrder(order)

	// Verify it was added
	ob.mu.RLock()
	_, exists := ob.orders[id]
	askCount := len(ob.asks)
	ob.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, 1, askCount)

	// Cancel the sell order
	success := ob.CancelOrder(id)
	assert.True(t, success)

	// Verify it was removed
	ob.mu.RLock()
	_, exists = ob.orders[id]
	askCount = len(ob.asks)
	ob.mu.RUnlock()
	assert.False(t, exists)
	assert.Equal(t, 0, askCount)
}

// Test modifying a sell order
func TestModifySellOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a sell order
	order := &Order{Price: 51000.0, Quantity: 1.0, Side: Sell}
	id := ob.AddOrder(order)

	// Modify the sell order
	success := ob.ModifyOrder(id, 52000.0, 2.0)
	assert.True(t, success)

	// Verify modification
	ob.mu.RLock()
	modifiedOrder := ob.orders[id]
	ob.mu.RUnlock()
	assert.Equal(t, 52000.0, modifiedOrder.Price)
	assert.Equal(t, 2.0, modifiedOrder.Quantity)
}

// Test matching where sell order is larger than buy
func TestMatchSellLargerThanBuy(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Small buy order
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Large sell order at crossing price
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 5.0, Side: Sell})

	trades := ob.MatchOrders()
	require.Len(t, trades, 1)
	assert.Equal(t, 1.0, trades[0].Quantity)

	// Sell order should still be in book with remaining quantity
	ob.mu.RLock()
	remaining := 0.0
	for _, o := range ob.orders {
		if o.Side == Sell {
			remaining = o.Quantity
		}
	}
	ob.mu.RUnlock()
	assert.Equal(t, 4.0, remaining)
}

// Test depth levels with more asks than requested
func TestGetDepthManyAsks(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add many sell orders
	for i := 0; i < 20; i++ {
		ob.AddOrder(&Order{Price: float64(51000 + i*100), Quantity: 1.0, Side: Sell})
	}

	// Request only 5 levels
	depth := ob.GetDepth(5)
	assert.LessOrEqual(t, len(depth.Asks), 5)
}

// Test inserting bids at various positions
func TestInsertBidAtDifferentPositions(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add bids in different price order to test insertion logic
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy}) // Middle insertion

	// Best bid should be 51000
	bestBid := ob.GetBestBid()
	assert.Equal(t, 51000.0, bestBid)
}

// Test inserting asks at various positions
func TestInsertAskAtDifferentPositions(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add asks in different price order to test insertion logic
	ob.AddOrder(&Order{Price: 53000.0, Quantity: 1.0, Side: Sell})
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})
	ob.AddOrder(&Order{Price: 52000.0, Quantity: 1.0, Side: Sell}) // Middle insertion

	// Best ask should be 51000 (lowest)
	bestAsk := ob.GetBestAsk()
	assert.Equal(t, 51000.0, bestAsk)
}

// Test modify sell order price to cause re-sorting
func TestModifySellOrderPriceChange(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add multiple sell orders
	id1 := ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})
	ob.AddOrder(&Order{Price: 52000.0, Quantity: 1.0, Side: Sell})

	// Best ask should be 51000
	bestAsk := ob.GetBestAsk()
	assert.Equal(t, 51000.0, bestAsk)

	// Modify first order to higher price
	success := ob.ModifyOrder(id1, 53000.0, 1.0)
	assert.True(t, success)

	// Best ask should now be 52000
	bestAsk = ob.GetBestAsk()
	assert.Equal(t, 52000.0, bestAsk)
}

// Test cancel order that was already matched
func TestCancelAlreadyMatchedOrder(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add and fully match an order
	buyID := ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})
	ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Sell})
	ob.MatchOrders()

	// Try to cancel the fully matched order
	success := ob.CancelOrder(buyID)
	// Should fail because order was fully filled and removed
	assert.False(t, success)
}

// Test GetDepth with orphaned bid IDs (covers lines 203-204)
func TestGetDepthWithOrphanedBidIDs(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a real order
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Manually insert orphaned ID into bids (simulates data corruption)
	ob.mu.Lock()
	ob.bids = append(ob.bids, 9999) // Non-existent order ID
	ob.mu.Unlock()

	// GetDepth should handle the orphaned ID gracefully
	depth := ob.GetDepth(10)
	assert.NotNil(t, depth)
	// Should still have the valid bid
	assert.Equal(t, 1, len(depth.Bids))
}

// Test GetDepth with orphaned ask IDs (covers lines 214-215)
func TestGetDepthWithOrphanedAskIDs(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a real order
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	// Manually insert orphaned ID into asks (simulates data corruption)
	ob.mu.Lock()
	ob.asks = append(ob.asks, 8888) // Non-existent order ID
	ob.mu.Unlock()

	// GetDepth should handle the orphaned ID gracefully
	depth := ob.GetDepth(10)
	assert.NotNil(t, depth)
	// Should still have the valid ask
	assert.Equal(t, 1, len(depth.Asks))
}

// Test GetDepth with nil order in map (covers lines 203-204, 214-215)
func TestGetDepthWithNilOrders(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add real orders
	bidID := ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})
	askID := ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	// Set orders to nil (simulates corruption)
	ob.mu.Lock()
	ob.orders[bidID] = nil
	ob.orders[askID] = nil
	ob.mu.Unlock()

	// GetDepth should handle nil orders gracefully
	depth := ob.GetDepth(10)
	assert.NotNil(t, depth)
	assert.Equal(t, 0, len(depth.Bids))
	assert.Equal(t, 0, len(depth.Asks))
}

// Test insertBid with non-existent order (covers lines 247-248)
func TestInsertBidNonExistent(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a real order first
	ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Try to insert a non-existent order ID
	ob.mu.Lock()
	initialBidCount := len(ob.bids)
	ob.insertBid(9999) // Non-existent order
	finalBidCount := len(ob.bids)
	ob.mu.Unlock()

	// Should not add anything
	assert.Equal(t, initialBidCount, finalBidCount)
}

// Test insertAsk with non-existent order (covers lines 271-272)
func TestInsertAskNonExistent(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a real order first
	ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	// Try to insert a non-existent order ID
	ob.mu.Lock()
	initialAskCount := len(ob.asks)
	ob.insertAsk(8888) // Non-existent order
	finalAskCount := len(ob.asks)
	ob.mu.Unlock()

	// Should not add anything
	assert.Equal(t, initialAskCount, finalAskCount)
}

// Test insertBid with orphaned IDs in existing bids list (covers lines 255-256)
func TestInsertBidWithOrphanedExisting(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a valid order
	validID := ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Insert orphaned ID into bids list
	ob.mu.Lock()
	ob.bids = append([]uint64{9999}, ob.bids...) // Orphan at front
	ob.mu.Unlock()

	// Add another valid order - should skip orphaned entry during insertion
	newID := ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Buy})

	ob.mu.RLock()
	// Both valid orders should be in the book
	_, exists1 := ob.orders[validID]
	_, exists2 := ob.orders[newID]
	ob.mu.RUnlock()

	assert.True(t, exists1)
	assert.True(t, exists2)
}

// Test insertAsk with orphaned IDs in existing asks list (covers lines 280-281)
func TestInsertAskWithOrphanedExisting(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a valid order
	validID := ob.AddOrder(&Order{Price: 52000.0, Quantity: 1.0, Side: Sell})

	// Insert orphaned ID into asks list
	ob.mu.Lock()
	ob.asks = append([]uint64{8888}, ob.asks...) // Orphan at front
	ob.mu.Unlock()

	// Add another valid order - should skip orphaned entry during insertion
	newID := ob.AddOrder(&Order{Price: 51000.0, Quantity: 1.0, Side: Sell})

	ob.mu.RLock()
	// Both valid orders should be in the book
	_, exists1 := ob.orders[validID]
	_, exists2 := ob.orders[newID]
	ob.mu.RUnlock()

	assert.True(t, exists1)
	assert.True(t, exists2)
}

// Test insertBid with nil order in map (covers lines 255-256)
func TestInsertBidWithNilExisting(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a valid order
	validID := ob.AddOrder(&Order{Price: 50000.0, Quantity: 1.0, Side: Buy})

	// Set the existing order to nil in map
	ob.mu.Lock()
	ob.orders[validID] = nil
	ob.mu.Unlock()

	// Add another order - insertion should handle nil existing order
	newID := ob.AddOrder(&Order{Price: 49000.0, Quantity: 1.0, Side: Buy})

	ob.mu.RLock()
	_, newExists := ob.orders[newID]
	ob.mu.RUnlock()

	assert.True(t, newExists)
}

// Test insertAsk with nil order in map (covers lines 280-281)
func TestInsertAskWithNilExisting(t *testing.T) {
	ob := NewGoOrderBook(Config{Symbol: "BTC/USDC"})

	// Add a valid order
	validID := ob.AddOrder(&Order{Price: 52000.0, Quantity: 1.0, Side: Sell})

	// Set the existing order to nil in map
	ob.mu.Lock()
	ob.orders[validID] = nil
	ob.mu.Unlock()

	// Add another order - insertion should handle nil existing order
	newID := ob.AddOrder(&Order{Price: 53000.0, Quantity: 1.0, Side: Sell})

	ob.mu.RLock()
	_, newExists := ob.orders[newID]
	ob.mu.RUnlock()

	assert.True(t, newExists)
}
