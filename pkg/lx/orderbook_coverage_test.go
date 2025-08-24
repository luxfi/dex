package lx

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

// Test market order processing
func TestMarketOrderProcessing(t *testing.T) {
	book := NewOrderBook("TEST")
	book.EnableImmediateMatching = true

	// Add limit orders to match against
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
		User:  "buyer1",
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Buy,
		Price: 99,
		Size:  15,
		User:  "buyer2",
	})

	// Process market sell order through AddOrder (which handles locking)
	numTrades := book.AddOrder(&Order{
		ID:   3,
		Type: Market,
		Side: Sell,
		Size: 20,
		User: "seller1",
	})

	if numTrades == 0 {
		t.Error("Market order should have generated trades")
	}

	// Check that trades were created
	trades := book.GetTrades()
	totalSize := 0.0
	for _, trade := range trades {
		totalSize += trade.Size
	}

	expectedSize := 20.0
	if totalSize != expectedSize {
		t.Errorf("Expected total trade size %f, got %f", expectedSize, totalSize)
	}
}

// Test MatchOrders function
func TestMatchOrders(t *testing.T) {
	book := NewOrderBook("TEST")
	book.EnableImmediateMatching = false // Disable immediate matching

	// Add multiple orders without immediate matching
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
		User:  "buyer1",
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 99,
		Size:  5,
		User:  "seller1",
	})

	book.AddOrder(&Order{
		ID:    3,
		Type:  Limit,
		Side:  Buy,
		Price: 101,
		Size:  8,
		User:  "buyer2",
	})

	// Now match all orders
	trades := book.MatchOrders()

	if len(trades) == 0 {
		t.Error("MatchOrders should have found crossing orders")
	}

	// Verify trades were executed at correct prices
	for _, trade := range trades {
		if trade.Price > 101 || trade.Price < 99 {
			t.Errorf("Trade price %f is outside expected range", trade.Price)
		}
	}
}

// Test wouldTakeLiquidity function
func TestWouldTakeLiquidity(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add a limit order
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
		User:  "buyer1",
	})

	// Test sell order that would take liquidity
	takerOrder := &Order{
		Type:  Limit,
		Side:  Sell,
		Price: 100,
		Size:  5,
	}

	if !book.wouldTakeLiquidity(takerOrder) {
		t.Error("Sell at 100 should take liquidity from buy at 100")
	}

	// Test sell order that wouldn't take liquidity
	makerOrder := &Order{
		Type:  Limit,
		Side:  Sell,
		Price: 101,
		Size:  5,
	}

	if book.wouldTakeLiquidity(makerOrder) {
		t.Error("Sell at 101 should not take liquidity from buy at 100")
	}
}

// Test order tree operations
func TestOrderTreeOperations(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add orders
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 101,
		Size:  10,
	})

	// Check that orders were added to trees
	if book.GetBestBid() != 100 {
		t.Errorf("Best bid should be 100")
	}

	if book.GetBestAsk() != 101 {
		t.Errorf("Best ask should be 101")
	}
}

// Test Reset function
func TestResetOrderBook(t *testing.T) {
	book := NewOrderBook("TEST")
	book.EnableImmediateMatching = true

	// Add orders and generate trades
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 100,
		Size:  5,
	})

	// Verify data exists
	if len(book.Orders) == 0 {
		t.Error("Orders should exist before reset")
	}
	if len(book.Trades) == 0 {
		t.Error("Trades should exist before reset")
	}

	// Reset
	book.Reset()

	// Verify everything is cleared
	if len(book.Orders) != 0 {
		t.Error("Orders should be empty after reset")
	}
	if len(book.Trades) != 0 {
		t.Error("Trades should be empty after reset")
	}
	if book.GetBestBid() != 0 || book.GetBestAsk() != 0 {
		t.Error("Best prices should be 0 after reset")
	}
}

// Test ModifyOrder function
func TestModifyOrderFunction(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add an order
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
		User:  "user1",
	})

	// Modify the order
	err := book.ModifyOrder(1, 105, 15)
	if err != nil {
		t.Errorf("ModifyOrder should succeed for existing order: %v", err)
	}

	// Verify modification
	if order, exists := book.Orders[1]; exists {
		if order.Price != 105 {
			t.Errorf("Expected price 105, got %f", order.Price)
		}
		if order.Size != 15 {
			t.Errorf("Expected size 15, got %f", order.Size)
		}
	} else {
		t.Error("Modified order should still exist")
	}

	// Try to modify non-existent order
	err = book.ModifyOrder(999, 110, 20)
	if err == nil {
		t.Error("ModifyOrder should fail for non-existent order")
	}
}

// Test multiple orders at same price
func TestMultipleOrdersAtSamePrice(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add multiple orders at same price
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  10,
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Buy,
		Price: 100,
		Size:  15,
	})

	book.AddOrder(&Order{
		ID:    3,
		Type:  Limit,
		Side:  Buy,
		Price: 99,
		Size:  5,
	})

	// Best bid should still be 100
	if book.GetBestBid() != 100 {
		t.Errorf("Expected best bid 100, got %f", book.GetBestBid())
	}

	// Sell order should match all at 100 first
	book.AddOrder(&Order{
		ID:   4,
		Type: Market,
		Side: Sell,
		Size: 20,
	})

	trades := book.GetTrades()
	if len(trades) == 0 {
		t.Error("Should have trades")
	}
}

// Test spread calculation
func TestSpreadCalculation(t *testing.T) {
	book := NewOrderBook("TEST")

	// Empty book - spread is 0 (best ask - best bid when both are 0)
	bestBid := book.GetBestBid()
	bestAsk := book.GetBestAsk()
	spread := bestAsk - bestBid
	if spread != 0 {
		t.Error("Spread should be 0 for empty book")
	}

	// Add bid
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 99,
		Size:  10,
	})

	// Add ask
	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 101,
		Size:  10,
	})

	// Calculate spread
	bestBid = book.GetBestBid()
	bestAsk = book.GetBestAsk()
	spread = bestAsk - bestBid
	if spread != 2 {
		t.Errorf("Expected spread 2, got %f", spread)
	}
}

// Test mid price calculation
func TestMidPriceCalculation(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add bid and ask
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 98,
		Size:  10,
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 102,
		Size:  10,
	})

	// Calculate mid price
	bestBid := book.GetBestBid()
	bestAsk := book.GetBestAsk()
	mid := (bestBid + bestAsk) / 2
	if mid != 100 {
		t.Errorf("Expected mid price 100, got %f", mid)
	}
}

// Test crossed book detection
func TestCrossedBookDetection(t *testing.T) {
	book := NewOrderBook("TEST")
	book.EnableImmediateMatching = false // Allow crossed book for testing

	// Normal book
	book.AddOrder(&Order{
		ID:    1,
		Type:  Limit,
		Side:  Buy,
		Price: 99,
		Size:  10,
	})

	book.AddOrder(&Order{
		ID:    2,
		Type:  Limit,
		Side:  Sell,
		Price: 101,
		Size:  10,
	})

	// Book is not crossed
	if book.GetBestBid() >= book.GetBestAsk() {
		t.Error("Normal book should not be crossed")
	}

	// Add crossing order
	book.AddOrder(&Order{
		ID:    3,
		Type:  Limit,
		Side:  Buy,
		Price: 102,
		Size:  10,
	})

	// Now best bid > best ask (crossed)
	if book.GetBestBid() <= book.GetBestAsk() {
		t.Error("Book should be crossed when best bid > best ask")
	}
}

// Test orders by user
func TestOrdersByUser(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add orders for different users
	book.AddOrder(&Order{
		ID:     1,
		Type:   Limit,
		Side:   Buy,
		Price:  100,
		Size:   10,
		UserID: "user1",
	})

	book.AddOrder(&Order{
		ID:     2,
		Type:   Limit,
		Side:   Sell,
		Price:  101,
		Size:   5,
		UserID: "user1",
	})

	book.AddOrder(&Order{
		ID:     3,
		Type:   Limit,
		Side:   Buy,
		Price:  99,
		Size:   15,
		UserID: "user2",
	})

	// Check that orders were added
	if len(book.Orders) != 3 {
		t.Errorf("Expected 3 orders, got %d", len(book.Orders))
	}

	// Count orders by user manually
	user1Count := 0
	user2Count := 0
	for _, order := range book.Orders {
		if order.UserID == "user1" {
			user1Count++
		} else if order.UserID == "user2" {
			user2Count++
		}
	}

	if user1Count != 2 {
		t.Errorf("Expected 2 orders for user1, got %d", user1Count)
	}

	if user2Count != 1 {
		t.Errorf("Expected 1 order for user2, got %d", user2Count)
	}
}

// Test concurrent operations
func TestConcurrentOperations(t *testing.T) {
	book := NewOrderBook("TEST")
	book.EnableImmediateMatching = true

	const numGoroutines = 10
	const ordersPerGoroutine = 100

	var wg sync.WaitGroup

	// Concurrent adds
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < ordersPerGoroutine; j++ {
				book.AddOrder(&Order{
					ID:    uint64(id*ordersPerGoroutine + j),
					Type:  Limit,
					Side:  Side((id + j) % 2),
					Price: 100 + float64(rand.Intn(10)-5),
					Size:  float64(rand.Intn(10) + 1),
					User:  "user",
				})
			}
		}(i)
	}

	// Concurrent cancels
	for i := 0; i < numGoroutines/2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			time.Sleep(10 * time.Millisecond) // Let some orders get added first
			for j := 0; j < ordersPerGoroutine/2; j++ {
				book.CancelOrder(uint64(rand.Intn(numGoroutines * ordersPerGoroutine)))
			}
		}(i)
	}

	// Concurrent reads
	for i := 0; i < numGoroutines/2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = book.GetBestBid()
				_ = book.GetBestAsk()
				_ = book.GetSnapshot()
				time.Sleep(time.Millisecond)
			}
		}()
	}

	wg.Wait()

	// Verify book is still consistent
	if book.GetBestBid() > book.GetBestAsk() && book.GetBestAsk() > 0 {
		t.Error("Book became crossed during concurrent operations")
	}
}

// Benchmark for MatchOrders
func BenchmarkMatchOrders(b *testing.B) {
	book := NewOrderBook("BENCH")
	book.EnableImmediateMatching = false

	// Pre-populate with orders
	for i := 0; i < 1000; i++ {
		book.AddOrder(&Order{
			ID:    uint64(i),
			Type:  Limit,
			Side:  Side(i % 2),
			Price: 100 + float64(rand.Intn(20)-10),
			Size:  float64(rand.Intn(10) + 1),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = book.MatchOrders()
	}
}

// Test detectBestBackend function
func TestDetectBestBackend(t *testing.T) {
	backend := detectBestBackend()

	// Should return a valid backend
	validBackends := []Backend{BackendGo, BackendCGO, BackendMLX, BackendCUDA}
	found := false
	for _, valid := range validBackends {
		if backend == valid {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("detectBestBackend returned invalid backend: %v", backend)
	}

	t.Logf("Detected backend: %v", backend)
}

// Test order book with many price levels
func TestManyPriceLevels(t *testing.T) {
	book := NewOrderBook("TEST")

	// Add orders at 1000 different price levels
	for i := 0; i < 1000; i++ {
		book.AddOrder(&Order{
			ID:    uint64(i),
			Type:  Limit,
			Side:  Buy,
			Price: 100 - float64(i)/10,
			Size:  1,
		})

		book.AddOrder(&Order{
			ID:    uint64(i + 1000),
			Type:  Limit,
			Side:  Sell,
			Price: 101 + float64(i)/10,
			Size:  1,
		})
	}

	// Should still get correct best prices
	bestBid := book.GetBestBid()
	bestAsk := book.GetBestAsk()

	// Best bid should be close to but less than 100
	if bestBid >= 100.1 {
		t.Errorf("Best bid should be around 100, got %f", bestBid)
	}

	// Best ask should be close to but greater than 101
	if bestAsk <= 100.9 {
		t.Errorf("Best ask should be around 101, got %f", bestAsk)
	}

	// Get depth should work
	depth := book.GetDepth(10)
	if len(depth.Bids) != 10 || len(depth.Asks) != 10 {
		t.Error("GetDepth should return correct number of levels")
	}
}
