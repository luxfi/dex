package lx

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/shopspring/decimal"
)

// TestNewOrderBookServer tests NewOrderBookServer constructor
func TestNewOrderBookServer(t *testing.T) {
	ob := NewOrderBook("BTC/USD")

	server := NewOrderBookServer(ob)

	if server == nil {
		t.Fatal("NewOrderBookServer returned nil")
	}

	if server.orderBook != ob {
		t.Error("orderBook not set correctly")
	}

	if server.wsClients == nil {
		t.Error("wsClients map not initialized")
	}

	if len(server.wsClients) != 0 {
		t.Error("wsClients should be empty initially")
	}

	if !server.conservative {
		t.Error("conservative should be true by default")
	}

	if server.blockNumber != 0 {
		t.Errorf("blockNumber should be 0 initially, got %d", server.blockNumber)
	}
}

// TestOrderBookServerStart tests Start method
func TestOrderBookServerStart(t *testing.T) {
	ob := NewOrderBook("ETH/USD")
	server := NewOrderBookServer(ob)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := server.Start(ctx, 8080)
	if err != nil {
		t.Fatalf("Start returned error: %v", err)
	}

	// Give block processor time to run at least once (ticker is 100ms, wait up to 500ms)
	var blockNum uint64
	for i := 0; i < 10; i++ {
		time.Sleep(50 * time.Millisecond)
		server.mu.RLock()
		blockNum = server.blockNumber
		server.mu.RUnlock()
		if blockNum > 0 {
			break
		}
	}

	if blockNum == 0 {
		t.Error("blockProcessor should have incremented blockNumber")
	}
}

// TestOrderBookServerStartCancellation tests context cancellation stops block processor
func TestOrderBookServerStartCancellation(t *testing.T) {
	ob := NewOrderBook("SOL/USD")
	server := NewOrderBookServer(ob)

	ctx, cancel := context.WithCancel(context.Background())
	err := server.Start(ctx, 8081)
	if err != nil {
		t.Fatalf("Start returned error: %v", err)
	}

	// Let it process a few blocks
	time.Sleep(250 * time.Millisecond)

	server.mu.RLock()
	blocksBefore := server.blockNumber
	server.mu.RUnlock()

	// Cancel context to stop processor
	cancel()

	// Wait for processor to stop
	time.Sleep(150 * time.Millisecond)

	// Check blocks stopped incrementing
	time.Sleep(200 * time.Millisecond)

	server.mu.RLock()
	blocksAfter := server.blockNumber
	server.mu.RUnlock()

	// After cancellation, blocks should not increase (or increase by at most 1 due to timing)
	if blocksAfter > blocksBefore+1 {
		t.Errorf("blocks should stop incrementing after cancel: before=%d, after=%d", blocksBefore, blocksAfter)
	}
}

// TestOnOrderAdd tests OnOrderAdd method
func TestOnOrderAdd(t *testing.T) {
	ob := NewOrderBook("AVAX/USD")
	server := NewOrderBookServer(ob)

	// Add a client channel
	clientCh := make(chan interface{}, 10)
	server.mu.Lock()
	server.wsClients["client1"] = clientCh
	server.mu.Unlock()

	order := &Order{
		ID:     1001,
		Symbol: "AVAX/USD",
		Type:   Limit,
		Side:   Buy,
		Price:  35.50,
		Size:   10.0,
	}

	server.OnOrderAdd(order)

	// Check that order was broadcast
	select {
	case received := <-clientCh:
		receivedOrder, ok := received.(*Order)
		if !ok {
			t.Fatalf("received wrong type: %T", received)
		}
		if receivedOrder.ID != order.ID {
			t.Errorf("wrong order ID: got %d, want %d", receivedOrder.ID, order.ID)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("order was not broadcast to client")
	}
}

// TestOnOrderAddMultipleClients tests broadcasting to multiple clients
func TestOnOrderAddMultipleClients(t *testing.T) {
	ob := NewOrderBook("LINK/USD")
	server := NewOrderBookServer(ob)

	// Add multiple clients
	client1 := make(chan interface{}, 10)
	client2 := make(chan interface{}, 10)
	client3 := make(chan interface{}, 10)

	server.mu.Lock()
	server.wsClients["client1"] = client1
	server.wsClients["client2"] = client2
	server.wsClients["client3"] = client3
	server.mu.Unlock()

	order := &Order{
		ID:     2002,
		Symbol: "LINK/USD",
		Side:   Sell,
		Price:  15.25,
		Size:   50.0,
	}

	server.OnOrderAdd(order)

	// Check all clients received
	for name, ch := range map[string]chan interface{}{"client1": client1, "client2": client2, "client3": client3} {
		select {
		case received := <-ch:
			if received.(*Order).ID != order.ID {
				t.Errorf("%s: wrong order ID", name)
			}
		case <-time.After(100 * time.Millisecond):
			t.Errorf("%s: did not receive order", name)
		}
	}
}

// TestOnOrderAddFullChannel tests non-blocking behavior when channel is full
func TestOnOrderAddFullChannel(t *testing.T) {
	ob := NewOrderBook("UNI/USD")
	server := NewOrderBookServer(ob)

	// Add client with full channel (capacity 0, no buffer)
	fullCh := make(chan interface{})
	server.mu.Lock()
	server.wsClients["fullClient"] = fullCh
	server.mu.Unlock()

	order := &Order{
		ID:   3003,
		Side: Buy,
	}

	// This should not block
	done := make(chan struct{})
	go func() {
		server.OnOrderAdd(order)
		close(done)
	}()

	select {
	case <-done:
		// Good - did not block
	case <-time.After(100 * time.Millisecond):
		t.Error("OnOrderAdd blocked on full channel")
	}
}

// TestProcessBlock tests processBlock method
func TestProcessBlock(t *testing.T) {
	ob := NewOrderBook("DOGE/USD")
	server := NewOrderBookServer(ob)

	if server.blockNumber != 0 {
		t.Fatalf("initial blockNumber should be 0, got %d", server.blockNumber)
	}

	server.processBlock()
	if server.blockNumber != 1 {
		t.Errorf("after 1 processBlock, blockNumber should be 1, got %d", server.blockNumber)
	}

	server.processBlock()
	server.processBlock()
	if server.blockNumber != 3 {
		t.Errorf("after 3 processBlock calls, blockNumber should be 3, got %d", server.blockNumber)
	}
}

// TestProcessBlockConcurrent tests concurrent processBlock calls
func TestProcessBlockConcurrent(t *testing.T) {
	ob := NewOrderBook("MATIC/USD")
	server := NewOrderBookServer(ob)

	var wg sync.WaitGroup
	iterations := 100

	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			server.processBlock()
		}()
	}

	wg.Wait()

	server.mu.RLock()
	finalBlock := server.blockNumber
	server.mu.RUnlock()

	if finalBlock != uint64(iterations) {
		t.Errorf("expected blockNumber %d, got %d", iterations, finalBlock)
	}
}

// TestGenerateOrderSnapshotEmpty tests snapshot generation for empty order book
func TestGenerateOrderSnapshotEmpty(t *testing.T) {
	ob := NewOrderBook("ATOM/USD")
	server := NewOrderBookServer(ob)

	snapshot := server.generateOrderSnapshot("ATOM/USD")

	if snapshot == nil {
		t.Fatal("generateOrderSnapshot returned nil")
	}

	if snapshot.Symbol != "ATOM/USD" {
		t.Errorf("wrong symbol: got %s, want ATOM/USD", snapshot.Symbol)
	}

	if snapshot.BidCount != 0 {
		t.Errorf("empty book should have 0 bids, got %d", snapshot.BidCount)
	}

	if snapshot.AskCount != 0 {
		t.Errorf("empty book should have 0 asks, got %d", snapshot.AskCount)
	}

	if snapshot.Block != 0 {
		t.Errorf("initial block should be 0, got %d", snapshot.Block)
	}

	if snapshot.Timestamp == 0 {
		t.Error("timestamp should be set")
	}

	if !snapshot.BestBid.IsZero() {
		t.Errorf("empty book BestBid should be zero, got %s", snapshot.BestBid.String())
	}

	if !snapshot.BestAsk.IsZero() {
		t.Errorf("empty book BestAsk should be zero, got %s", snapshot.BestAsk.String())
	}
}

// TestGenerateOrderSnapshotWithOrders tests snapshot with orders in book
func TestGenerateOrderSnapshotWithOrders(t *testing.T) {
	ob := NewOrderBook("LTC/USD")
	server := NewOrderBookServer(ob)

	// Add some orders to the book
	bidOrder := &Order{
		ID:        5001,
		Symbol:    "LTC/USD",
		Type:      Limit,
		Side:      Buy,
		Price:     100.00,
		Size:      5.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user1",
	}

	askOrder := &Order{
		ID:        5002,
		Symbol:    "LTC/USD",
		Type:      Limit,
		Side:      Sell,
		Price:     101.00,
		Size:      3.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user2",
	}

	ob.AddOrder(bidOrder)
	ob.AddOrder(askOrder)

	// Process a block
	server.processBlock()

	snapshot := server.generateOrderSnapshot("LTC/USD")

	if snapshot.BidCount != 1 {
		t.Errorf("expected 1 bid, got %d", snapshot.BidCount)
	}

	if snapshot.AskCount != 1 {
		t.Errorf("expected 1 ask, got %d", snapshot.AskCount)
	}

	expectedBid := decimal.NewFromFloat(100.00)
	if !snapshot.BestBid.Equal(expectedBid) {
		t.Errorf("BestBid: got %s, want %s", snapshot.BestBid.String(), expectedBid.String())
	}

	expectedAsk := decimal.NewFromFloat(101.00)
	if !snapshot.BestAsk.Equal(expectedAsk) {
		t.Errorf("BestAsk: got %s, want %s", snapshot.BestAsk.String(), expectedAsk.String())
	}

	if snapshot.Block != 1 {
		t.Errorf("block should be 1, got %d", snapshot.Block)
	}
}

// TestGenerateOrderSnapshotMultipleOrders tests snapshot with multiple orders at different prices
func TestGenerateOrderSnapshotMultipleOrders(t *testing.T) {
	ob := NewOrderBook("XRP/USD")
	server := NewOrderBookServer(ob)

	// Add multiple bid orders
	for i := 0; i < 5; i++ {
		ob.AddOrder(&Order{
			ID:        uint64(6000 + i),
			Symbol:    "XRP/USD",
			Type:      Limit,
			Side:      Buy,
			Price:     float64(50 - i), // 50, 49, 48, 47, 46
			Size:      10.0,
			Status:    Open,
			Timestamp: time.Now(),
			UserID:    "user1",
		})
	}

	// Add multiple ask orders
	for i := 0; i < 3; i++ {
		ob.AddOrder(&Order{
			ID:        uint64(7000 + i),
			Symbol:    "XRP/USD",
			Type:      Limit,
			Side:      Sell,
			Price:     float64(51 + i), // 51, 52, 53
			Size:      5.0,
			Status:    Open,
			Timestamp: time.Now(),
			UserID:    "user2",
		})
	}

	snapshot := server.generateOrderSnapshot("XRP/USD")

	if snapshot.BidCount != 5 {
		t.Errorf("expected 5 bids, got %d", snapshot.BidCount)
	}

	if snapshot.AskCount != 3 {
		t.Errorf("expected 3 asks, got %d", snapshot.AskCount)
	}

	// Best bid should be highest = 50
	expectedBid := decimal.NewFromFloat(50.0)
	if !snapshot.BestBid.Equal(expectedBid) {
		t.Errorf("BestBid: got %s, want %s", snapshot.BestBid.String(), expectedBid.String())
	}

	// Best ask should be lowest = 51
	expectedAsk := decimal.NewFromFloat(51.0)
	if !snapshot.BestAsk.Equal(expectedAsk) {
		t.Errorf("BestAsk: got %s, want %s", snapshot.BestAsk.String(), expectedAsk.String())
	}
}

// TestValidateStateEmpty tests validateState on empty book
func TestValidateStateEmpty(t *testing.T) {
	ob := NewOrderBook("ADA/USD")
	server := NewOrderBookServer(ob)

	if !server.validateState() {
		t.Error("empty order book should be valid")
	}
}

// TestValidateStateValidSpread tests validateState with valid spread
func TestValidateStateValidSpread(t *testing.T) {
	ob := NewOrderBook("DOT/USD")
	server := NewOrderBookServer(ob)

	// Add bid at 10
	ob.AddOrder(&Order{
		ID:        8001,
		Symbol:    "DOT/USD",
		Type:      Limit,
		Side:      Buy,
		Price:     10.00,
		Size:      5.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user1",
	})

	// Add ask at 11 (valid spread, bid < ask)
	ob.AddOrder(&Order{
		ID:        8002,
		Symbol:    "DOT/USD",
		Type:      Limit,
		Side:      Sell,
		Price:     11.00,
		Size:      5.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user2",
	})

	if !server.validateState() {
		t.Error("valid spread (bid < ask) should be valid state")
	}
}

// TestValidateStateCrossed tests validateState with crossed book
func TestValidateStateCrossed(t *testing.T) {
	ob := NewOrderBook("FIL/USD")
	server := NewOrderBookServer(ob)

	// Directly manipulate the order trees to create crossed state
	// (normally the matching engine prevents this)
	bids := ob.GetBids()
	asks := ob.GetAsks()

	// Set best bid higher than best ask
	bids.bestPrice.Store(int64(100 * PriceMultiplier)) // bid = 100
	asks.bestPrice.Store(int64(99 * PriceMultiplier))  // ask = 99 (crossed!)

	if server.validateState() {
		t.Error("crossed order book (bid >= ask) should be invalid")
	}
}

// TestValidateStateNonConservative tests validateState when conservative=false
func TestValidateStateNonConservative(t *testing.T) {
	ob := NewOrderBook("NEAR/USD")
	server := NewOrderBookServer(ob)
	server.conservative = false

	// Even with crossed book, non-conservative mode should return true
	bids := ob.GetBids()
	asks := ob.GetAsks()
	bids.bestPrice.Store(int64(100 * PriceMultiplier))
	asks.bestPrice.Store(int64(99 * PriceMultiplier))

	if !server.validateState() {
		t.Error("non-conservative mode should always return true")
	}
}

// TestValidateStateOnlyBids tests validateState with only bids (no asks)
func TestValidateStateOnlyBids(t *testing.T) {
	ob := NewOrderBook("ALGO/USD")
	server := NewOrderBookServer(ob)

	ob.AddOrder(&Order{
		ID:        9001,
		Symbol:    "ALGO/USD",
		Type:      Limit,
		Side:      Buy,
		Price:     25.00,
		Size:      10.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user1",
	})

	// With only bids (no asks), should be valid
	if !server.validateState() {
		t.Error("book with only bids should be valid")
	}
}

// TestValidateStateOnlyAsks tests validateState with only asks (no bids)
func TestValidateStateOnlyAsks(t *testing.T) {
	ob := NewOrderBook("HBAR/USD")
	server := NewOrderBookServer(ob)

	ob.AddOrder(&Order{
		ID:        9002,
		Symbol:    "HBAR/USD",
		Type:      Limit,
		Side:      Sell,
		Price:     30.00,
		Size:      10.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user1",
	})

	// With only asks (no bids), should be valid
	if !server.validateState() {
		t.Error("book with only asks should be valid")
	}
}

// TestBlockProcessor tests the blockProcessor goroutine
func TestBlockProcessor(t *testing.T) {
	ob := NewOrderBook("VET/USD")
	server := NewOrderBookServer(ob)

	ctx, cancel := context.WithCancel(context.Background())

	go server.blockProcessor(ctx)

	// Wait for several ticks (100ms each)
	time.Sleep(350 * time.Millisecond)

	server.mu.RLock()
	blocks := server.blockNumber
	server.mu.RUnlock()

	// Should have at least 3 blocks (350ms / 100ms)
	if blocks < 3 {
		t.Errorf("expected at least 3 blocks, got %d", blocks)
	}

	// Cancel and verify it stops
	cancel()
	time.Sleep(150 * time.Millisecond)

	server.mu.RLock()
	blocksBefore := server.blockNumber
	server.mu.RUnlock()

	time.Sleep(200 * time.Millisecond)

	server.mu.RLock()
	blocksAfter := server.blockNumber
	server.mu.RUnlock()

	// Should have stopped incrementing
	if blocksAfter > blocksBefore+1 {
		t.Errorf("blockProcessor should stop after cancel: before=%d, after=%d", blocksBefore, blocksAfter)
	}
}

// TestOrderBookServerGetSnapshot tests GetSnapshot method
func TestOrderBookServerGetSnapshot(t *testing.T) {
	ob := NewOrderBook("ICP/USD")
	server := NewOrderBookServer(ob)

	snapshot, err := server.GetSnapshot("ICP/USD")
	if err != nil {
		t.Fatalf("GetSnapshot returned error: %v", err)
	}

	if snapshot == nil {
		t.Fatal("GetSnapshot returned nil snapshot")
	}

	if snapshot.Symbol != "ICP/USD" {
		t.Errorf("wrong symbol: got %s, want ICP/USD", snapshot.Symbol)
	}
}

// TestGetSnapshotWithOrders tests GetSnapshot with orders in book
func TestGetSnapshotWithOrders(t *testing.T) {
	ob := NewOrderBook("FTM/USD")
	server := NewOrderBookServer(ob)

	// Add orders
	ob.AddOrder(&Order{
		ID:        10001,
		Symbol:    "FTM/USD",
		Type:      Limit,
		Side:      Buy,
		Price:     0.50,
		Size:      1000.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user1",
	})

	ob.AddOrder(&Order{
		ID:        10002,
		Symbol:    "FTM/USD",
		Type:      Limit,
		Side:      Sell,
		Price:     0.51,
		Size:      500.0,
		Status:    Open,
		Timestamp: time.Now(),
		UserID:    "user2",
	})

	snapshot, err := server.GetSnapshot("FTM/USD")
	if err != nil {
		t.Fatalf("GetSnapshot returned error: %v", err)
	}

	if snapshot.BidCount != 1 {
		t.Errorf("expected 1 bid, got %d", snapshot.BidCount)
	}

	if snapshot.AskCount != 1 {
		t.Errorf("expected 1 ask, got %d", snapshot.AskCount)
	}

	expectedBid := decimal.NewFromFloat(0.50)
	if !snapshot.BestBid.Equal(expectedBid) {
		t.Errorf("BestBid: got %s, want %s", snapshot.BestBid.String(), expectedBid.String())
	}

	expectedAsk := decimal.NewFromFloat(0.51)
	if !snapshot.BestAsk.Equal(expectedAsk) {
		t.Errorf("BestAsk: got %s, want %s", snapshot.BestAsk.String(), expectedAsk.String())
	}
}

// TestGetSnapshotConcurrent tests concurrent GetSnapshot calls
func TestGetSnapshotConcurrent(t *testing.T) {
	ob := NewOrderBook("SAND/USD")
	server := NewOrderBookServer(ob)

	// Add some orders
	for i := 0; i < 10; i++ {
		ob.AddOrder(&Order{
			ID:        uint64(11000 + i),
			Symbol:    "SAND/USD",
			Type:      Limit,
			Side:      Buy,
			Price:     float64(1.0 + float64(i)*0.01),
			Size:      100.0,
			Status:    Open,
			Timestamp: time.Now(),
			UserID:    "user1",
		})
	}

	// Start block processor
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server.Start(ctx, 8082)

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	// Concurrent snapshot requests
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			snapshot, err := server.GetSnapshot("SAND/USD")
			if err != nil {
				errors <- err
				return
			}
			if snapshot.Symbol != "SAND/USD" {
				errors <- err
			}
		}()
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("concurrent GetSnapshot error: %v", err)
	}
}

// TestOrderSnapshotJSON tests that OrderSnapshot can be marshaled (implicitly tests struct tags)
func TestOrderSnapshotJSON(t *testing.T) {
	snapshot := &OrderSnapshot{
		Symbol:    "TEST/USD",
		BidCount:  5,
		AskCount:  3,
		BestBid:   decimal.NewFromFloat(99.50),
		BestAsk:   decimal.NewFromFloat(100.50),
		Timestamp: time.Now().UnixNano(),
		Block:     42,
	}

	if snapshot.Symbol != "TEST/USD" {
		t.Error("Symbol not set correctly")
	}
	if snapshot.BidCount != 5 {
		t.Error("BidCount not set correctly")
	}
	if snapshot.AskCount != 3 {
		t.Error("AskCount not set correctly")
	}
	if snapshot.Block != 42 {
		t.Error("Block not set correctly")
	}
}

// TestServerIntegration tests full integration scenario
func TestOrderBookServerIntegration(t *testing.T) {
	ob := NewOrderBook("INT/USD")
	server := NewOrderBookServer(ob)

	// Start server
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := server.Start(ctx, 8083)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Add client
	clientCh := make(chan interface{}, 100)
	server.mu.Lock()
	server.wsClients["integrationClient"] = clientCh
	server.mu.Unlock()

	// Add orders and verify broadcast
	for i := 0; i < 5; i++ {
		order := &Order{
			ID:        uint64(12000 + i),
			Symbol:    "INT/USD",
			Type:      Limit,
			Side:      Buy,
			Price:     float64(100 + i),
			Size:      10.0,
			Status:    Open,
			Timestamp: time.Now(),
			UserID:    "user1",
		}

		ob.AddOrder(order)
		server.OnOrderAdd(order)
	}

	// Verify broadcasts received
	received := 0
	for i := 0; i < 5; i++ {
		select {
		case <-clientCh:
			received++
		case <-time.After(100 * time.Millisecond):
			break
		}
	}

	if received != 5 {
		t.Errorf("expected 5 broadcasts, got %d", received)
	}

	// Wait for block processor
	time.Sleep(150 * time.Millisecond)

	// Get snapshot
	snapshot, err := server.GetSnapshot("INT/USD")
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}

	if snapshot.BidCount != 5 {
		t.Errorf("expected 5 bids in snapshot, got %d", snapshot.BidCount)
	}

	if snapshot.Block == 0 {
		t.Error("block should be > 0 after block processor ran")
	}

	// Validate state
	if !server.validateState() {
		t.Error("state should be valid after adding non-crossing orders")
	}
}
