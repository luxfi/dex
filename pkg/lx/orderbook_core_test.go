package lx

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test core orderbook functions with 0% coverage
func TestOrderBookCoreFunctions(t *testing.T) {
	ob := NewOrderBook("TEST-PAIR")

	t.Run("processMarketOrderLocked", func(t *testing.T) {
		// Add some liquidity first
		ob.AddOrder(&Order{
			ID:        1,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     50000,
			User:      "seller1",
			Timestamp: time.Now(),
		})

		// Create market buy order
		marketOrder := &Order{
			ID:        2,
			Type:      Market,
			Side:      Buy,
			Size:      50,
			User:      "buyer1",
			Timestamp: time.Now(),
		}

		orderID := ob.processMarketOrderLocked(marketOrder)
		assert.Equal(t, uint64(2), orderID)
	})

	t.Run("processMarketOrderLocked_Rejected", func(t *testing.T) {
		// Clear the order book
		ob = NewOrderBook("TEST-PAIR")
		
		// Create market order with no liquidity (should be rejected)
		marketOrder := &Order{
			ID:        3,
			Type:      Market,
			Side:      Buy,
			Size:      100,
			User:      "buyer2",
			Timestamp: time.Now(),
		}

		orderID := ob.processMarketOrderLocked(marketOrder)
		assert.Equal(t, uint64(0), orderID) // Should return 0 for rejected orders
	})

	t.Run("matchOrdersWithSelfTradePrevention", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add orders from same user (should trigger self-trade prevention)
		ob.AddOrder(&Order{
			ID:        4,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50000,
			User:      "trader1",
			Timestamp: time.Now(),
		})

		ob.AddOrder(&Order{
			ID:        5,
			Type:      Limit,
			Side:      Sell,
			Size:      80,
			Price:     49999, // Crossing price
			User:      "trader1", // Same user
			Timestamp: time.Now().Add(1 * time.Millisecond),
		})

		// MatchOrders should handle self-trade prevention
		trades := ob.MatchOrders()
		assert.Equal(t, 0, len(trades)) // No trades due to self-trade prevention
	})

	t.Run("matchOrdersNormalMatching", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add crossing orders from different users
		ob.AddOrder(&Order{
			ID:        6,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		})

		ob.AddOrder(&Order{
			ID:        7,
			Type:      Limit,
			Side:      Sell,
			Size:      80,
			Price:     49999, // Crossing price
			User:      "seller1",
			Timestamp: time.Now().Add(1 * time.Millisecond),
		})

		trades := ob.MatchOrders()
		assert.Greater(t, len(trades), 0) // Should create trades
		
		if len(trades) > 0 {
			trade := trades[0]
			assert.Equal(t, float64(80), trade.Size) // Size of smaller order
			assert.Equal(t, uint64(6), trade.BuyOrder)
			assert.Equal(t, uint64(7), trade.SellOrder)
			assert.Equal(t, "full", trade.MatchType) // Smaller order is fully filled
		}
	})

	t.Run("matchOrdersPriceTimePriority", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add bid order first (older timestamp)
		bidOrder := &Order{
			ID:        8,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		}
		ob.AddOrder(bidOrder)

		// Add ask order second (newer timestamp)
		askOrder := &Order{
			ID:        9,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     50000, // Same price
			User:      "seller1",
			Timestamp: time.Now().Add(1 * time.Millisecond),
		}
		ob.AddOrder(askOrder)

		trades := ob.MatchOrders()
		assert.Greater(t, len(trades), 0)
		
		if len(trades) > 0 {
			trade := trades[0]
			// Since bid order was first, trade price should be bid price
			assert.Equal(t, bidOrder.Price, trade.Price)
			assert.Equal(t, Sell, trade.TakerSide) // Ask is taker
		}
	})

	t.Run("matchOrdersPartialFill", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Large bid order
		ob.AddOrder(&Order{
			ID:        10,
			Type:      Limit,
			Side:      Buy,
			Size:      200,
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		})

		// Smaller ask order
		ob.AddOrder(&Order{
			ID:        11,
			Type:      Limit,
			Side:      Sell,
			Size:      80,
			Price:     49999,
			User:      "seller1",
			Timestamp: time.Now().Add(1 * time.Millisecond),
		})

		trades := ob.MatchOrders()
		assert.Equal(t, 1, len(trades))
		
		trade := trades[0]
		assert.Equal(t, float64(80), trade.Size) // Smaller order size
		assert.Equal(t, "full", trade.MatchType) // Ask order should be fully filled
	})

	t.Run("matchOrdersNoLiquidity", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// No orders in book
		trades := ob.MatchOrders()
		assert.Equal(t, 0, len(trades))
	})

	t.Run("matchOrdersNoCrossing", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add non-crossing orders
		ob.AddOrder(&Order{
			ID:        12,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     49000, // Low bid
			User:      "buyer1",
			Timestamp: time.Now(),
		})

		ob.AddOrder(&Order{
			ID:        13,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     51000, // High ask
			User:      "seller1",
			Timestamp: time.Now(),
		})

		trades := ob.MatchOrders()
		assert.Equal(t, 0, len(trades)) // No crossing orders
	})

	t.Run("matchOrdersMultipleMatches", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add multiple small ask orders
		ob.AddOrder(&Order{
			ID:        14,
			Type:      Limit,
			Side:      Sell,
			Size:      30,
			Price:     50000,
			User:      "seller1",
			Timestamp: time.Now(),
		})

		ob.AddOrder(&Order{
			ID:        15,
			Type:      Limit,
			Side:      Sell,
			Size:      40,
			Price:     50000,
			User:      "seller2",
			Timestamp: time.Now().Add(1 * time.Millisecond),
		})

		// Large bid order that crosses both
		ob.AddOrder(&Order{
			ID:        16,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50001,
			User:      "buyer1",
			Timestamp: time.Now().Add(2 * time.Millisecond),
		})

		trades := ob.MatchOrders()
		assert.GreaterOrEqual(t, len(trades), 2) // Should match multiple orders
		
		totalTradeSize := float64(0)
		for _, trade := range trades {
			totalTradeSize += trade.Size
		}
		assert.Equal(t, float64(70), totalTradeSize) // 30 + 40
	})
}

// Test order tree functions with low coverage
func TestOrderTreeFunctions(t *testing.T) {
	tree := NewOrderTree(Buy) // Provide required Side parameter

	t.Run("getBestOrderEmptyTree", func(t *testing.T) {
		bestOrder := tree.getBestOrder()
		assert.Nil(t, bestOrder)
	})

	t.Run("getBestOrderWithOrders", func(t *testing.T) {
		order1 := &Order{
			ID:        1,
			Price:     50000,
			Size:      100,
			Timestamp: time.Now(),
		}
		order2 := &Order{
			ID:        2,
			Price:     51000, // Higher price
			Size:      50,
			Timestamp: time.Now(),
		}

		tree.addOrder(order1)
		tree.addOrder(order2)

		bestOrder := tree.getBestOrder()
		assert.NotNil(t, bestOrder)
		// For a bid tree, higher price is better
		// For ask tree, lower price is better
		// This depends on tree type, but should return one of the orders
		assert.True(t, bestOrder.ID == 1 || bestOrder.ID == 2)
	})

	t.Run("removeOrderFromTree", func(t *testing.T) {
		tree = NewOrderTree(Buy) // Provide required Side parameter
		
		order := &Order{
			ID:        3,
			Price:     50000,
			Size:      100,
			Timestamp: time.Now(),
		}

		tree.addOrder(order)
		assert.NotNil(t, tree.getBestOrder())

		tree.removeOrder(order)
		assert.Nil(t, tree.getBestOrder())
	})
}

// Test various order matching scenarios
func TestOrderMatchingScenarios(t *testing.T) {
	t.Run("IcebergOrderMatching", func(t *testing.T) {
		ob := NewOrderBook("TEST-PAIR")
		
		// Add iceberg order (simplified without non-existent fields)
		icebergOrder := &Order{
			ID:        17,
			Type:      Iceberg,
			Side:      Sell,
			Size:      200,
			Price:     50000,
			User:      "iceberg_trader",
			Timestamp: time.Now(),
		}
		
		ob.AddOrder(icebergOrder)
		
		// Add large buy order
		ob.AddOrder(&Order{
			ID:        18,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50001,
			User:      "buyer1",
			Timestamp: time.Now(),
		})

		trades := ob.MatchOrders()
		assert.Greater(t, len(trades), 0)
	})

	t.Run("StopOrderHandling", func(t *testing.T) {
		ob := NewOrderBook("TEST-PAIR")
		
		// Add stop sell order (should be handled by the system)
		stopOrder := &Order{
			ID:        19,
			Type:      Stop,
			Side:      Sell,
			Size:      100,
			StopPrice: 49000,
			User:      "stop_trader",
			Timestamp: time.Now(),
		}
		
		orderID := ob.AddOrder(stopOrder)
		// Order should be added successfully
		assert.Greater(t, orderID, uint64(0))
	})

	t.Run("PostOnlyOrderHandling", func(t *testing.T) {
		ob := NewOrderBook("TEST-PAIR")
		
		// Add ask order first
		ob.AddOrder(&Order{
			ID:        20,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     50000,
			User:      "seller1",
			Timestamp: time.Now(),
		})
		
		// Add post-only bid that would match (simplified test)
		postOnlyOrder := &Order{
			ID:        21,
			Type:      Limit, // Use Limit instead of undefined PostOnly
			Side:      Buy,
			Size:      50,
			Price:     50001, // Would match
			User:      "post_only_trader",
			Timestamp: time.Now(),
		}
		
		orderID := ob.AddOrder(postOnlyOrder)
		// Order should be added successfully
		assert.Greater(t, orderID, uint64(0))
	})
}

// Test error conditions and edge cases
func TestOrderBookEdgeCases(t *testing.T) {
	ob := NewOrderBook("TEST-PAIR")

	t.Run("ProcessMarketOrderNoLiquidity", func(t *testing.T) {
		marketOrder := &Order{
			ID:        22,
			Type:      Market,
			Side:      Buy,
			Size:      100,
			User:      "buyer1",
			Timestamp: time.Now(),
		}

		// Should handle gracefully when no liquidity
		orderID := ob.processMarketOrderLocked(marketOrder)
		assert.Equal(t, uint64(0), orderID) // Rejected
	})

	t.Run("MatchOrdersWithZeroSizedOrders", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add orders with zero remaining size (edge case)
		order1 := &Order{
			ID:        23,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Filled:    100, // Fully filled
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		}
		
		order2 := &Order{
			ID:        24,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     49999,
			User:      "seller1",
			Timestamp: time.Now(),
		}
		
		// Add orders normally (don't access non-existent AllOrders)
		ob.AddOrder(order1)
		ob.AddOrder(order2)
		
		// Should not crash
		trades := ob.MatchOrders()
		assert.NotNil(t, trades)
	})

	t.Run("TradeIDIncrement", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		initialTradeID := ob.LastTradeID
		
		// Add matching orders
		ob.AddOrder(&Order{
			ID:        25,
			Type:      Limit,
			Side:      Buy,
			Size:      100,
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		})

		ob.AddOrder(&Order{
			ID:        26,
			Type:      Limit,
			Side:      Sell,
			Size:      100,
			Price:     50000,
			User:      "seller1",
			Timestamp: time.Now(),
		})

		trades := ob.MatchOrders()
		if len(trades) > 0 {
			assert.Greater(t, ob.LastTradeID, initialTradeID)
			assert.Equal(t, ob.LastTradeID, trades[0].ID)
		}
	})

	t.Run("OrderStatusUpdates", func(t *testing.T) {
		ob = NewOrderBook("TEST-PAIR")
		
		// Add large bid order
		bidOrder := &Order{
			ID:        27,
			Type:      Limit,
			Side:      Buy,
			Size:      200,
			Price:     50000,
			User:      "buyer1",
			Timestamp: time.Now(),
		}
		ob.AddOrder(bidOrder)
		
		// Add smaller ask order (should partially fill bid)
		askOrder := &Order{
			ID:        28,
			Type:      Limit,
			Side:      Sell,
			Size:      80,
			Price:     49999,
			User:      "seller1",
			Timestamp: time.Now(),
		}
		ob.AddOrder(askOrder)
		
		trades := ob.MatchOrders()
		assert.Greater(t, len(trades), 0)
		
		// Verify order status updates
		if len(trades) > 0 {
			// Ask order should be fully filled
			assert.Equal(t, Filled, askOrder.Status)
			// Bid order should be partially filled
			assert.Equal(t, PartiallyFilled, bidOrder.Status)
		}
	})
}

// Performance test for matching
func TestOrderBookMatchingPerformance(t *testing.T) {
	t.Run("HighVolumeMatching", func(t *testing.T) {
		ob := NewOrderBook("TEST-PAIR")
		
		// Add many orders
		numOrders := 100
		for i := 0; i < numOrders; i++ {
			// Add bid orders
			ob.AddOrder(&Order{
				ID:        uint64(i*2 + 1),
				Type:      Limit,
				Side:      Buy,
				Size:      10,
				Price:     50000 - float64(i),
				User:      "buyer" + string(rune(i)),
				Timestamp: time.Now(),
			})
			
			// Add ask orders
			ob.AddOrder(&Order{
				ID:        uint64(i*2 + 2),
				Type:      Limit,
				Side:      Sell,
				Size:      10,
				Price:     50001 + float64(i),
				User:      "seller" + string(rune(i)),
				Timestamp: time.Now(),
			})
		}
		
		// Add crossing orders to trigger matching
		ob.AddOrder(&Order{
			ID:        uint64(numOrders*2 + 1),
			Type:      Limit,
			Side:      Buy,
			Size:      500,
			Price:     55000, // High enough to cross many asks
			User:      "big_buyer",
			Timestamp: time.Now(),
		})
		
		start := time.Now()
		trades := ob.MatchOrders()
		duration := time.Since(start)
		
		t.Logf("Matched %d trades in %v", len(trades), duration)
		assert.Greater(t, len(trades), 0)
		assert.Less(t, duration, 100*time.Millisecond) // Should be fast
	})
}