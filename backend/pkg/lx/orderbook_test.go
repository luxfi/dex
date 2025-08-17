package lx

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestOrderBookCreation(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	if book == nil {
		t.Fatal("Failed to create order book")
	}

	if book.Symbol != "BTC-USD" {
		t.Errorf("Expected symbol BTC-USD, got %s", book.Symbol)
	}

	if book.Bids == nil || book.Asks == nil {
		t.Fatal("Order trees not initialized")
	}
}

func TestAddOrder(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "user1",
	}

	orderID := book.AddOrder(order)
	if orderID == 0 {
		t.Fatal("Failed to add order")
	}

	// Check order exists
	storedOrder, exists := book.Orders[orderID]
	if !exists {
		t.Fatal("Order not stored")
	}

	if storedOrder.Status != Open {
		t.Errorf("Expected order status Open, got %v", storedOrder.Status)
	}
}

func TestOrderMatching(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add buy order
	buyOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "buyer",
	}
	book.AddOrder(buyOrder)

	// Add sell order that should match
	sellOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  49999,
		Size:   0.5,
		User:   "seller",
	}
	book.AddOrder(sellOrder)

	// Match orders
	trades := book.MatchOrders()

	if len(trades) != 1 {
		t.Fatalf("Expected 1 trade, got %d", len(trades))
	}

	trade := trades[0]
	if trade.Size != 0.5 {
		t.Errorf("Expected trade size 0.5, got %f", trade.Size)
	}

	// Trade should execute at the maker's (first order's) price
	if trade.Price != 50000 {
		t.Errorf("Expected trade price 50000 (maker's price), got %f", trade.Price)
	}
}

func TestSelfTradePrevention(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add buy order
	buyOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "user1",
	}
	book.AddOrder(buyOrder)

	// Add sell order from same user - should be rejected
	sellOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  49999,
		Size:   1.0,
		User:   "user1",
	}
	orderID := book.AddOrder(sellOrder)

	if orderID != 0 {
		t.Error("Self-trade should have been prevented")
	}
}

func TestPostOnlyOrder(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add buy order
	buyOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "buyer",
	}
	book.AddOrder(buyOrder)

	// Add post-only sell order that would take liquidity
	sellOrder := &Order{
		Symbol:   "BTC-USD",
		Side:     Sell,
		Type:     Limit,
		Price:    49999,
		Size:     1.0,
		User:     "seller",
		PostOnly: true,
	}
	orderID := book.AddOrder(sellOrder)

	if orderID != 0 {
		t.Error("Post-only order should have been rejected")
	}
}

func TestCancelOrder(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "user1",
	}

	orderID := book.AddOrder(order)

	// Cancel order
	err := book.CancelOrder(orderID)
	if err != nil {
		t.Fatalf("Failed to cancel order: %v", err)
	}

	// Check order status
	cancelledOrder := book.Orders[orderID]
	if cancelledOrder.Status != Cancelled {
		t.Errorf("Expected order status Cancelled, got %v", cancelledOrder.Status)
	}
}

func TestModifyOrder(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	order := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "user1",
	}

	orderID := book.AddOrder(order)

	// Modify order
	err := book.ModifyOrder(orderID, 51000, 1.5)
	if err != nil {
		t.Fatalf("Failed to modify order: %v", err)
	}

	// Check modifications
	modifiedOrder := book.Orders[orderID]
	if modifiedOrder.Price != 51000 {
		t.Errorf("Expected price 51000, got %f", modifiedOrder.Price)
	}
	if modifiedOrder.Size != 1.5 {
		t.Errorf("Expected size 1.5, got %f", modifiedOrder.Size)
	}
}

func TestOrderBookSnapshot(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add multiple orders
	for i := 0; i < 5; i++ {
		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  float64(50000 - i*100),
			Size:   1.0,
			User:   "buyer",
		})

		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  float64(50100 + i*100),
			Size:   1.0,
			User:   "seller",
		})
	}

	snapshot := book.GetSnapshot()

	if snapshot == nil {
		t.Fatal("Failed to get snapshot")
	}

	if len(snapshot.Bids) == 0 || len(snapshot.Asks) == 0 {
		t.Error("Snapshot missing bid/ask levels")
	}

	// Check bid prices are descending
	for i := 1; i < len(snapshot.Bids); i++ {
		if snapshot.Bids[i].Price >= snapshot.Bids[i-1].Price {
			t.Error("Bid prices not in descending order")
		}
	}

	// Check ask prices are ascending
	for i := 1; i < len(snapshot.Asks); i++ {
		if snapshot.Asks[i].Price <= snapshot.Asks[i-1].Price {
			t.Error("Ask prices not in ascending order")
		}
	}
}

func TestL4BookSnapshot(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add orders with client IDs
	book.AddOrder(&Order{
		Symbol:   "BTC-USD",
		Side:     Buy,
		Type:     Limit,
		Price:    50000,
		Size:     1.0,
		User:     "user1",
		ClientID: "client-1",
	})

	book.AddOrder(&Order{
		Symbol:   "BTC-USD",
		Side:     Sell,
		Type:     Limit,
		Price:    50100,
		Size:     2.0,
		User:     "user2",
		ClientID: "client-2",
	})

	l4book := book.GetL4Book()

	if l4book.Symbol != "BTC-USD" {
		t.Errorf("Expected symbol BTC-USD, got %s", l4book.Symbol)
	}

	if len(l4book.Bids) != 1 || len(l4book.Asks) != 1 {
		t.Error("L4 book should have individual orders")
	}

	// Check individual order details
	if len(l4book.Bids) > 0 {
		bid := l4book.Bids[0]
		if bid.ClientID != "client-1" {
			t.Errorf("Expected client ID client-1, got %s", bid.ClientID)
		}
	}
}

func TestConcurrentOrderProcessing(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	var wg sync.WaitGroup
	numGoroutines := 10
	ordersPerGoroutine := 100

	wg.Add(numGoroutines)

	// Spawn multiple goroutines adding orders
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < ordersPerGoroutine; j++ {
				// Create non-crossing orders to avoid matching
				// Buys at 49000-49999, Sells at 51000-51999
				side := Buy
				price := 49000.0 + float64(j)
				if j%2 == 0 {
					side = Sell
					price = 51000.0 + float64(j)
				}

				book.AddOrder(&Order{
					Symbol: "BTC-USD",
					Side:   side,
					Type:   Limit,
					Price:  price,
					Size:   0.1,
					User:   fmt.Sprintf("user-%d", id),
				})
			}
		}(i)
	}

	wg.Wait()

	// Verify all orders were added
	totalOrders := len(book.Orders)
	expectedOrders := numGoroutines * ordersPerGoroutine

	if totalOrders != expectedOrders {
		t.Errorf("Expected %d orders, got %d", expectedOrders, totalOrders)
	}
}

func TestPartialFills(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add large buy order
	buyOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   10.0,
		User:   "buyer",
	}
	buyID := book.AddOrder(buyOrder)

	// Add small sell order
	sellOrder := &Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  50000,
		Size:   3.0,
		User:   "seller",
	}
	book.AddOrder(sellOrder)

	// Match orders
	trades := book.MatchOrders()

	if len(trades) != 1 {
		t.Fatalf("Expected 1 trade, got %d", len(trades))
	}

	// Check buy order is partially filled
	buyOrderAfter := book.Orders[buyID]
	if buyOrderAfter.Status != PartiallyFilled {
		t.Errorf("Expected PartiallyFilled status, got %v", buyOrderAfter.Status)
	}

	if buyOrderAfter.Filled != 3.0 {
		t.Errorf("Expected filled amount 3.0, got %f", buyOrderAfter.Filled)
	}
}

func TestPricePriority(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add multiple buy orders at different prices
	book.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  49900,
		Size:   1.0,
		User:   "buyer1",
	})

	book.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000, // Best price
		Size:   1.0,
		User:   "buyer2",
	})

	book.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  49800,
		Size:   1.0,
		User:   "buyer3",
	})

	// Add sell order
	book.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "seller",
	})

	// Match orders
	trades := book.MatchOrders()

	if len(trades) != 1 {
		t.Fatalf("Expected 1 trade, got %d", len(trades))
	}

	// Should match with buyer2 (best price)
	// Trade.BuyOrder is interface{}, need to assert type
	if buyOrder, ok := trades[0].BuyOrder.(*Order); ok {
		if buyOrder.User != "buyer2" {
			t.Errorf("Expected trade with buyer2, got %s", buyOrder.User)
		}
	} else {
		t.Errorf("BuyOrder is not *Order type")
	}
}

func TestTimePriority(t *testing.T) {
	book := NewOrderBook("BTC-USD")

	// Add multiple buy orders at same price
	order1 := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "buyer1",
	}
	book.AddOrder(order1)

	time.Sleep(10 * time.Millisecond)

	order2 := &Order{
		Symbol: "BTC-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "buyer2",
	}
	book.AddOrder(order2)

	// Add sell order
	book.AddOrder(&Order{
		Symbol: "BTC-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  50000,
		Size:   1.0,
		User:   "seller",
	})

	// Match orders
	trades := book.MatchOrders()

	if len(trades) != 1 {
		t.Fatalf("Expected 1 trade, got %d", len(trades))
	}

	// Should match with buyer1 (earlier order)
	// Type assertion for interface{} field
	if buyOrder, ok := trades[0].BuyOrder.(*Order); ok {
		if buyOrder.User != "buyer1" {
			t.Errorf("Expected trade with buyer1 (FIFO), got %s", buyOrder.User)
		}
	} else {
		t.Errorf("BuyOrder is not *Order type")
	}
}

func BenchmarkAddOrder(b *testing.B) {
	book := NewOrderBook("BTC-USD")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Side(i % 2),
			Type:   Limit,
			Price:  50000 + float64(i%1000),
			Size:   1.0,
			User:   "user",
		})
	}
}

func BenchmarkMatchOrders(b *testing.B) {
	book := NewOrderBook("BTC-USD")

	// Pre-populate order book
	for i := 0; i < 1000; i++ {
		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  49000 + float64(i),
			Size:   1.0,
			User:   "buyer",
		})

		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  51000 + float64(i),
			Size:   1.0,
			User:   "seller",
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Add crossing orders
		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  52000,
			Size:   0.1,
			User:   "buyer",
		})

		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  48000,
			Size:   0.1,
			User:   "seller",
		})

		book.MatchOrders()
	}
}

func BenchmarkGetSnapshot(b *testing.B) {
	book := NewOrderBook("BTC-USD")

	// Pre-populate order book
	for i := 0; i < 1000; i++ {
		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Buy,
			Type:   Limit,
			Price:  49000 + float64(i),
			Size:   1.0,
			User:   "buyer",
		})

		book.AddOrder(&Order{
			Symbol: "BTC-USD",
			Side:   Sell,
			Type:   Limit,
			Price:  51000 + float64(i),
			Size:   1.0,
			User:   "seller",
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = book.GetSnapshot()
	}
}

func BenchmarkConcurrentOrderProcessing(b *testing.B) {
	book := NewOrderBook("BTC-USD")

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			book.AddOrder(&Order{
				Symbol: "BTC-USD",
				Side:   Side(i % 2),
				Type:   Limit,
				Price:  50000 + float64(i%1000),
				Size:   1.0,
				User:   "user",
			})
			i++
		}
	})
}
