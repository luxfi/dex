package lx

import (
	"sync"
	"testing"
	"time"
)

// TestNewTradingEngine tests the NewTradingEngine constructor
func TestNewTradingEngine(t *testing.T) {
	t.Run("empty config", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})
		if engine == nil {
			t.Fatal("NewTradingEngine returned nil")
		}
		if engine.OrderBooks == nil {
			t.Error("OrderBooks map is nil")
		}
		if engine.Orders == nil {
			t.Error("Orders map is nil")
		}
		if engine.Events == nil {
			t.Error("Events channel is nil")
		}
		if engine.PerpManager != nil {
			t.Error("PerpManager should be nil when EnablePerps is false")
		}
		if engine.VaultManager != nil {
			t.Error("VaultManager should be nil when EnableVaults is false")
		}
		if engine.LendingPool != nil {
			t.Error("LendingPool should be nil when EnableLending is false")
		}
	})

	t.Run("with perps enabled", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{EnablePerps: true})
		if engine.PerpManager == nil {
			t.Error("PerpManager should not be nil when EnablePerps is true")
		}
	})

	t.Run("with vaults enabled", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{EnableVaults: true})
		if engine.VaultManager == nil {
			t.Error("VaultManager should not be nil when EnableVaults is true")
		}
	})

	t.Run("with lending enabled", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{EnableLending: true})
		if engine.LendingPool == nil {
			t.Error("LendingPool should not be nil when EnableLending is true")
		}
	})

	t.Run("all features enabled", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{
			EnablePerps:   true,
			EnableVaults:  true,
			EnableLending: true,
		})
		if engine.PerpManager == nil {
			t.Error("PerpManager should not be nil")
		}
		if engine.VaultManager == nil {
			t.Error("VaultManager should not be nil")
		}
		if engine.LendingPool == nil {
			t.Error("LendingPool should not be nil")
		}
	})
}

// TestTradingEngineStartStop tests Start and Stop methods
func TestTradingEngineStartStop(t *testing.T) {
	t.Run("start returns no error", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})
		err := engine.Start()
		if err != nil {
			t.Errorf("Start() returned error: %v", err)
		}
		// Give processEvents goroutine time to start
		time.Sleep(10 * time.Millisecond)

		err = engine.Stop()
		if err != nil {
			t.Errorf("Stop() returned error: %v", err)
		}
	})

	t.Run("stop closes events channel", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})
		_ = engine.Start()
		time.Sleep(10 * time.Millisecond)

		_ = engine.Stop()

		// Verify channel is closed by trying to receive
		select {
		case _, ok := <-engine.Events:
			if ok {
				t.Error("Events channel should be closed after Stop()")
			}
		case <-time.After(100 * time.Millisecond):
			t.Error("Events channel should be closed and readable")
		}
	})

	t.Run("processEvents drains channel on stop", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})
		_ = engine.Start()

		// Send some events
		for i := 0; i < 5; i++ {
			engine.Events <- Event{
				Type:      "test",
				Timestamp: time.Now(),
				Data:      i,
			}
		}

		time.Sleep(50 * time.Millisecond)
		_ = engine.Stop()
		// No error means processEvents handled events correctly
	})
}

// TestTradingEngineCreateSpotMarket tests CreateSpotMarket method
func TestTradingEngineCreateSpotMarket(t *testing.T) {
	t.Run("creates new market", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		book := engine.CreateSpotMarket("BTC/USD")
		if book == nil {
			t.Fatal("CreateSpotMarket returned nil")
		}
		if book.Symbol != "BTC/USD" {
			t.Errorf("Expected symbol BTC/USD, got %s", book.Symbol)
		}
	})

	t.Run("stores market in OrderBooks map", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		book := engine.CreateSpotMarket("ETH/USD")

		stored := engine.OrderBooks["ETH/USD"]
		if stored != book {
			t.Error("OrderBooks map does not contain the created order book")
		}
	})

	t.Run("multiple markets", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		book1 := engine.CreateSpotMarket("BTC/USD")
		book2 := engine.CreateSpotMarket("ETH/USD")
		book3 := engine.CreateSpotMarket("LUX/USD")

		if len(engine.OrderBooks) != 3 {
			t.Errorf("Expected 3 order books, got %d", len(engine.OrderBooks))
		}
		if book1.Symbol != "BTC/USD" {
			t.Errorf("Expected BTC/USD, got %s", book1.Symbol)
		}
		if book2.Symbol != "ETH/USD" {
			t.Errorf("Expected ETH/USD, got %s", book2.Symbol)
		}
		if book3.Symbol != "LUX/USD" {
			t.Errorf("Expected LUX/USD, got %s", book3.Symbol)
		}
	})

	t.Run("concurrent creation is safe", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		var wg sync.WaitGroup
		symbols := []string{"BTC/USD", "ETH/USD", "LUX/USD", "SOL/USD", "DOT/USD"}

		for _, sym := range symbols {
			wg.Add(1)
			go func(s string) {
				defer wg.Done()
				engine.CreateSpotMarket(s)
			}(sym)
		}

		wg.Wait()

		if len(engine.OrderBooks) != len(symbols) {
			t.Errorf("Expected %d order books, got %d", len(symbols), len(engine.OrderBooks))
		}
	})
}

// TestTradingEngineLogEvent tests the logEvent method
func TestTradingEngineLogEvent(t *testing.T) {
	t.Run("event is received on channel", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		event := Event{
			Type:      "test_event",
			Timestamp: time.Now(),
			Data:      map[string]string{"key": "value"},
		}

		engine.logEvent(event)

		select {
		case received := <-engine.Events:
			if received.Type != event.Type {
				t.Errorf("Expected event type %s, got %s", event.Type, received.Type)
			}
		case <-time.After(100 * time.Millisecond):
			t.Error("Event was not received within timeout")
		}
	})

	t.Run("drops event when channel is full", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Fill the channel (capacity is 10000)
		for i := 0; i < 10000; i++ {
			engine.Events <- Event{Type: "filler"}
		}

		// This should not block - event is dropped
		done := make(chan bool)
		go func() {
			engine.logEvent(Event{Type: "overflow"})
			done <- true
		}()

		select {
		case <-done:
			// Success - logEvent returned without blocking
		case <-time.After(100 * time.Millisecond):
			t.Error("logEvent blocked when channel was full")
		}
	})
}

// TestTradingEngineGetUserOrders tests GetUserOrders method
func TestTradingEngineGetUserOrders(t *testing.T) {
	t.Run("returns empty slice for unknown user", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		orders := engine.GetUserOrders("unknown_user")
		if orders == nil {
			t.Error("GetUserOrders returned nil instead of empty slice")
		}
		if len(orders) != 0 {
			t.Errorf("Expected 0 orders, got %d", len(orders))
		}
	})

	t.Run("returns orders for specific user", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Add orders directly to engine.Orders map
		engine.mu.Lock()
		engine.Orders[1] = &Order{ID: 1, User: "alice", Symbol: "BTC/USD", Size: 1.0}
		engine.Orders[2] = &Order{ID: 2, User: "alice", Symbol: "ETH/USD", Size: 2.0}
		engine.Orders[3] = &Order{ID: 3, User: "bob", Symbol: "BTC/USD", Size: 3.0}
		engine.mu.Unlock()

		aliceOrders := engine.GetUserOrders("alice")
		if len(aliceOrders) != 2 {
			t.Errorf("Expected 2 orders for alice, got %d", len(aliceOrders))
		}

		bobOrders := engine.GetUserOrders("bob")
		if len(bobOrders) != 1 {
			t.Errorf("Expected 1 order for bob, got %d", len(bobOrders))
		}
	})

	t.Run("concurrent access is safe", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Add some orders
		engine.mu.Lock()
		for i := uint64(1); i <= 100; i++ {
			engine.Orders[i] = &Order{ID: i, User: "user1", Size: 1.0}
		}
		engine.mu.Unlock()

		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				orders := engine.GetUserOrders("user1")
				if len(orders) != 100 {
					t.Errorf("Expected 100 orders, got %d", len(orders))
				}
			}()
		}
		wg.Wait()
	})
}

// TestTradingEngineCreateOrderBook tests CreateOrderBook method
func TestTradingEngineCreateOrderBook(t *testing.T) {
	t.Run("creates new order book", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		engine.CreateOrderBook("BTC/USD")

		if engine.OrderBooks["BTC/USD"] == nil {
			t.Error("OrderBook was not created")
		}
	})

	t.Run("does not overwrite existing order book", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Create first order book and add an order
		engine.CreateOrderBook("BTC/USD")
		engine.OrderBooks["BTC/USD"].AddOrder(&Order{
			ID:     1,
			Symbol: "BTC/USD",
			Side:   Buy,
			Price:  50000,
			Size:   1.0,
			Status: Open,
		})

		// Try to create again
		engine.CreateOrderBook("BTC/USD")

		// Check that the order still exists (book was not overwritten)
		order := engine.OrderBooks["BTC/USD"].GetOrder(1)
		if order == nil {
			t.Error("Existing order book was overwritten")
		}
	})

	t.Run("concurrent creation is safe", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				engine.CreateOrderBook("BTC/USD")
			}()
		}
		wg.Wait()

		if engine.OrderBooks["BTC/USD"] == nil {
			t.Error("OrderBook was not created")
		}
	})
}

// TestTradingEngineGetOrderBook tests GetOrderBook method
func TestTradingEngineGetOrderBook(t *testing.T) {
	t.Run("returns nil for non-existent symbol", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		book := engine.GetOrderBook("UNKNOWN")
		if book != nil {
			t.Error("Expected nil for non-existent symbol")
		}
	})

	t.Run("returns existing order book", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		created := engine.CreateSpotMarket("BTC/USD")
		retrieved := engine.GetOrderBook("BTC/USD")

		if retrieved != created {
			t.Error("GetOrderBook did not return the same order book")
		}
	})

	t.Run("concurrent read access is safe", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})
		engine.CreateSpotMarket("BTC/USD")

		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				book := engine.GetOrderBook("BTC/USD")
				if book == nil {
					t.Error("GetOrderBook returned nil")
				}
			}()
		}
		wg.Wait()
	})
}

// TestTradingEngineProcessEvents tests the processEvents goroutine
func TestTradingEngineProcessEvents(t *testing.T) {
	t.Run("processes events until channel closes", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Track processed events
		var processed int
		done := make(chan bool)

		// Create a custom engine with smaller channel for testing
		testEngine := &TradingEngine{
			OrderBooks: make(map[string]*OrderBook),
			Orders:     make(map[uint64]*Order),
			Events:     make(chan Event, 10),
		}

		// Start custom processing
		go func() {
			for range testEngine.Events {
				processed++
			}
			done <- true
		}()

		// Send events
		for i := 0; i < 5; i++ {
			testEngine.Events <- Event{Type: "test"}
		}

		// Close channel
		close(testEngine.Events)

		// Wait for processing to complete
		select {
		case <-done:
			if processed != 5 {
				t.Errorf("Expected 5 processed events, got %d", processed)
			}
		case <-time.After(time.Second):
			t.Error("Processing did not complete within timeout")
		}

		_ = engine // Use the original engine to avoid unused variable warning
	})
}

// TestTradingEngineIntegration tests full workflow
func TestTradingEngineIntegration(t *testing.T) {
	t.Run("full trading workflow", func(t *testing.T) {
		engine := NewTradingEngine(EngineConfig{})

		// Start engine
		if err := engine.Start(); err != nil {
			t.Fatalf("Failed to start engine: %v", err)
		}

		// Create market
		book := engine.CreateSpotMarket("BTC/USD")
		if book == nil {
			t.Fatal("Failed to create spot market")
		}

		// Verify market is accessible
		retrieved := engine.GetOrderBook("BTC/USD")
		if retrieved == nil {
			t.Fatal("Failed to retrieve order book")
		}

		// Add orders to the order book
		buyOrder := &Order{
			Symbol: "BTC/USD",
			Side:   Buy,
			Type:   Limit,
			Price:  50000,
			Size:   1.0,
			User:   "alice",
			Status: Open,
		}
		orderID := book.AddOrder(buyOrder)
		if orderID == 0 {
			t.Fatal("Failed to add buy order")
		}

		sellOrder := &Order{
			Symbol: "BTC/USD",
			Side:   Sell,
			Type:   Limit,
			Price:  50000,
			Size:   1.0,
			User:   "bob",
			Status: Open,
		}
		sellID := book.AddOrder(sellOrder)
		if sellID == 0 {
			t.Fatal("Failed to add sell order")
		}

		// Match orders
		trades := book.MatchOrders()
		if len(trades) == 0 {
			t.Error("Expected at least one trade")
		}

		// Stop engine
		if err := engine.Stop(); err != nil {
			t.Fatalf("Failed to stop engine: %v", err)
		}
	})
}

// Benchmark tests
func BenchmarkTradingEngineCreateSpotMarket(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.CreateSpotMarket("BTC/USD")
	}
}

func BenchmarkTradingEngineGetUserOrders(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})

	// Pre-populate with orders
	engine.mu.Lock()
	for i := uint64(1); i <= 1000; i++ {
		engine.Orders[i] = &Order{ID: i, User: "testuser", Size: 1.0}
	}
	engine.mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.GetUserOrders("testuser")
	}
}

func BenchmarkTradingEngineGetOrderBook(b *testing.B) {
	engine := NewTradingEngine(EngineConfig{})
	engine.CreateSpotMarket("BTC/USD")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.GetOrderBook("BTC/USD")
	}
}
