package orderbook

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestConcurrentOrders(t *testing.T) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "BTC/USD",
	})

	const goroutines = 5
	const ordersPerGoroutine = 100

	var wg sync.WaitGroup

	// Launch concurrent goroutines
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for i := 0; i < ordersPerGoroutine; i++ {
				orderID := uint64(id*ordersPerGoroutine + i + 1)
				order := &Order{
					ID:        orderID,
					Symbol:    "BTC/USD",
					Side:      OrderSide(i % 2),
					Price:     50000 + float64(i%100),
					Quantity:  1.0 + float64(i%10)*0.1,
					Timestamp: time.Now(),
				}
				
				// Add order
				resultID := ob.AddOrder(order)
				if resultID == 0 {
					t.Logf("Warning: Failed to add order %d", orderID)
				}
				
				// Randomly cancel some orders
				if i > 10 && i%10 == 0 {
					ob.CancelOrder(orderID - 5)
				}
				
				// Randomly modify some orders  
				if i > 20 && i%20 == 0 {
					ob.ModifyOrder(orderID-10, 50000+float64(i), 2.0)
				}
				
				// Match orders occasionally
				if i%50 == 0 {
					ob.MatchOrders()
				}
			}
		}(g)
	}

	wg.Wait()

	// Verify orderbook state
	volume := ob.GetVolume()
	t.Logf("Final volume: %d", volume)

	depth := ob.GetDepth(10)
	if depth != nil {
		t.Logf("Final depth: %d bids, %d asks", len(depth.Bids), len(depth.Asks))
	}
}

func TestLargeOrderBook(t *testing.T) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "ETH/USD",
	})

	const numOrders = 10000
	const priceRange = 100

	// Add many orders across price levels
	for i := 0; i < numOrders; i++ {
		order := &Order{
			ID:        uint64(i),
			Symbol:    "ETH/USD",
			Side:      OrderSide(i % 2),
			Price:     3000 + float64(i%priceRange),
			Quantity:  1.0 + float64(i%10)*0.1,
			Timestamp: time.Now(),
		}
		ob.AddOrder(order)
	}

	// Test matching performance with large book
	start := time.Now()
	trades := ob.MatchOrders()
	matchDuration := time.Since(start)

	if matchDuration > 100*time.Millisecond {
		t.Errorf("Matching took too long: %v", matchDuration)
	}

	// Test depth retrieval with large book
	start = time.Now()
	depth := ob.GetDepth(100)
	depthDuration := time.Since(start)

	if depthDuration > 10*time.Millisecond {
		t.Errorf("Depth retrieval took too long: %v", depthDuration)
	}

	if depth == nil || (len(depth.Bids) == 0 && len(depth.Asks) == 0) {
		t.Error("Expected non-empty depth for large orderbook")
	}

	t.Logf("Large orderbook test: %d orders, %d trades matched in %v, depth in %v",
		numOrders, len(trades), matchDuration, depthDuration)
}

func TestOrderBookEdgeCases(t *testing.T) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "SOL/USD",
	})

	// Test zero quantity order
	order := &Order{
		ID:        1,
		Symbol:    "SOL/USD",
		Side:      Buy,
		Price:     100,
		Quantity:  0,
		Timestamp: time.Now(),
	}
	if ob.AddOrder(order) != 0 {
		t.Error("Should not accept zero quantity order")
	}

	// Test negative price
	order.Quantity = 1
	order.Price = -100
	if ob.AddOrder(order) != 0 {
		t.Error("Should not accept negative price order")
	}

	// Test very large order
	order.ID = 2
	order.Price = 100
	order.Quantity = 1e10
	orderID := ob.AddOrder(order)
	if orderID == 0 {
		t.Error("Should accept large quantity order")
	}

	// Test cancelling non-existent order
	if ob.CancelOrder(999999) {
		t.Error("Should not be able to cancel non-existent order")
	}

	// Test modifying non-existent order
	if ob.ModifyOrder(999999, 200, 2) {
		t.Error("Should not be able to modify non-existent order")
	}

	// Test getting depth with no orders on one side
	ob = NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "SOL/USD",
	})

	// Add only buy orders
	for i := 0; i < 5; i++ {
		ob.AddOrder(&Order{
			ID:        uint64(i + 10),
			Symbol:    "SOL/USD",
			Side:      Buy,
			Price:     100 - float64(i),
			Quantity:  1,
			Timestamp: time.Now(),
		})
	}

	depth := ob.GetDepth(10)
	if depth == nil {
		t.Fatal("Should return depth even with one-sided book")
	}

	if len(depth.Bids) == 0 {
		t.Error("Expected bids in one-sided book")
	}
}

func TestMatchingAlgorithm(t *testing.T) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "AVAX/USD",
	})

	// Set up crossing orders that should match
	buyOrders := []struct {
		id    uint64
		price float64
		qty   float64
	}{
		{1, 35.5, 10},
		{2, 35.4, 5},
		{3, 35.3, 15},
	}

	sellOrders := []struct {
		id    uint64
		price float64
		qty   float64
	}{
		{4, 35.2, 8},  // Should match with buy at 35.5
		{5, 35.3, 12}, // Should match with remaining buy at 35.5 and 35.4
		{6, 35.6, 20}, // Should not match
	}

	// Add buy orders
	for _, o := range buyOrders {
		ob.AddOrder(&Order{
			ID:        o.id,
			Symbol:    "AVAX/USD",
			Side:      Buy,
			Price:     o.price,
			Quantity:  o.qty,
			Timestamp: time.Now(),
		})
	}

	// Add sell orders
	for _, o := range sellOrders {
		ob.AddOrder(&Order{
			ID:        o.id,
			Symbol:    "AVAX/USD",
			Side:      Sell,
			Price:     o.price,
			Quantity:  o.qty,
			Timestamp: time.Now(),
		})
	}

	// Match orders
	trades := ob.MatchOrders()
	
	if len(trades) == 0 {
		t.Fatal("Expected trades to be matched")
	}

	// Verify trades occurred at correct prices
	totalTraded := 0.0
	for _, trade := range trades {
		totalTraded += trade.Quantity
		if trade.Price < 35.2 || trade.Price > 35.5 {
			t.Errorf("Unexpected trade price: %f", trade.Price)
		}
	}

	expectedTraded := 20.0 // 8 + 12 from crossing orders
	if totalTraded != expectedTraded {
		t.Errorf("Expected %f total quantity traded, got %f", expectedTraded, totalTraded)
	}

	// Verify remaining book state
	bestBid := ob.GetBestBid()
	bestAsk := ob.GetBestAsk()

	if bestBid >= bestAsk && bestBid > 0 && bestAsk > 0 {
		t.Errorf("Book still has crossing orders: bid=%f, ask=%f", bestBid, bestAsk)
	}
}

func BenchmarkConcurrentOperations(b *testing.B) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "BTC/USD",
	})

	b.RunParallel(func(pb *testing.PB) {
		id := uint64(rand.Int63())
		for pb.Next() {
			id++
			
			// Mix of operations
			switch rand.Intn(4) {
			case 0: // Add order
				ob.AddOrder(&Order{
					ID:        id,
					Symbol:    "BTC/USD",
					Side:      OrderSide(rand.Intn(2)),
					Price:     50000 + rand.Float64()*1000,
					Quantity:  rand.Float64() * 10,
					Timestamp: time.Now(),
				})
			case 1: // Cancel order
				ob.CancelOrder(id - uint64(rand.Intn(100)))
			case 2: // Modify order
				ob.ModifyOrder(id-uint64(rand.Intn(100)),
					50000+rand.Float64()*1000,
					rand.Float64()*10)
			case 3: // Match orders
				ob.MatchOrders()
			}
		}
	})
}

func BenchmarkLargeDepth(b *testing.B) {
	ob := NewOrderBook(Config{
		Implementation: ImplGo,
		Symbol:         "ETH/USD",
	})

	// Pre-populate with many orders
	for i := 0; i < 10000; i++ {
		ob.AddOrder(&Order{
			ID:        uint64(i),
			Symbol:    "ETH/USD",
			Side:      OrderSide(i % 2),
			Price:     3000 + float64(i%1000)*0.01,
			Quantity:  rand.Float64() * 100,
			Timestamp: time.Now(),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		depth := ob.GetDepth(100)
		if depth == nil {
			b.Fatal("Failed to get depth")
		}
	}
}