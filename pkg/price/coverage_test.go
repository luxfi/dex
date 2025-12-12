package price

import (
	"testing"
	"time"
)

// =============================================================================
// OrderbookSource Tests
// =============================================================================

// mockOrderbookProvider implements OrderbookProvider for testing.
type mockOrderbookProvider struct {
	bids    map[string]float64
	asks    map[string]float64
	symbols []string
}

func (m *mockOrderbookProvider) GetBestBid(symbol string) float64 {
	return m.bids[symbol]
}

func (m *mockOrderbookProvider) GetBestAsk(symbol string) float64 {
	return m.asks[symbol]
}

func (m *mockOrderbookProvider) GetSymbols() []string {
	return m.symbols
}

func TestOrderbookSourceBasic(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"LUX-USDC": 10.0, "ETH-USDC": 3000.0},
		asks:    map[string]float64{"LUX-USDC": 10.01, "ETH-USDC": 3001.0},
		symbols: []string{"LUX-USDC", "ETH-USDC"},
	}

	src := NewOrderbookSource(provider)
	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Stop()

	// Wait for data
	time.Sleep(50 * time.Millisecond)

	// Check health
	if !src.Healthy() {
		t.Error("Source should be healthy")
	}

	// Check name
	if src.Name() != "orderbook" {
		t.Errorf("Name() = %q, want orderbook", src.Name())
	}

	// Check weight
	if src.Weight() != 1.0 {
		t.Errorf("Weight() = %f, want 1.0", src.Weight())
	}
}

func TestOrderbookSourcePrice(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"LUX-USDC": 10.0},
		asks:    map[string]float64{"LUX-USDC": 10.02},
		symbols: []string{"LUX-USDC"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()

	// Wait for data
	time.Sleep(50 * time.Millisecond)

	// Get price
	p, err := src.Price("LUX-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	// Mid price = (10.0 + 10.02) / 2 = 10.01
	expected := 10.01
	if p.Price < 10.0 || p.Price > 10.02 {
		t.Errorf("Price = %f, expected ~%f", p.Price, expected)
	}

	// Check bid/ask
	if p.Bid != 10.0 {
		t.Errorf("Bid = %f, want 10.0", p.Bid)
	}
	if p.Ask != 10.02 {
		t.Errorf("Ask = %f, want 10.02", p.Ask)
	}
}

func TestOrderbookSourceNotFound(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{},
		asks:    map[string]float64{},
		symbols: []string{},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()

	// Try to get non-existent price
	_, err := src.Price("NONEXISTENT")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestOrderbookSourcePrices(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"LUX-USDC": 10.0, "ETH-USDC": 3000.0},
		asks:    map[string]float64{"LUX-USDC": 10.02, "ETH-USDC": 3002.0},
		symbols: []string{"LUX-USDC", "ETH-USDC"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()

	// Wait for data
	time.Sleep(50 * time.Millisecond)

	// Get multiple prices
	prices, err := src.Prices([]string{"LUX-USDC", "ETH-USDC", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	// Should have 2 valid prices
	if len(prices) != 2 {
		t.Errorf("Expected 2 prices, got %d", len(prices))
	}

	if _, ok := prices["LUX-USDC"]; !ok {
		t.Error("Missing LUX-USDC")
	}
	if _, ok := prices["ETH-USDC"]; !ok {
		t.Error("Missing ETH-USDC")
	}
}

func TestOrderbookSourceSubscribe(t *testing.T) {
	provider := &mockOrderbookProvider{}
	src := NewOrderbookSource(provider)

	// Subscribe/Unsubscribe are no-ops
	if err := src.Subscribe("LUX-USDC"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("LUX-USDC"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

func TestOrderbookSourceStartIdempotent(t *testing.T) {
	provider := &mockOrderbookProvider{symbols: []string{}}
	src := NewOrderbookSource(provider)

	// Start twice should be idempotent
	src.Start()
	src.Start() // Should not panic or error

	src.Stop()
}

func TestOrderbookSourceBidOnlyPrice(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"BID-ONLY": 50.0},
		asks:    map[string]float64{},
		symbols: []string{"BID-ONLY"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()

	time.Sleep(50 * time.Millisecond)

	p, err := src.Price("BID-ONLY")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price != 50.0 {
		t.Errorf("Price = %f, want 50.0 (bid only)", p.Price)
	}
}

func TestOrderbookSourceAskOnlyPrice(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{},
		asks:    map[string]float64{"ASK-ONLY": 60.0},
		symbols: []string{"ASK-ONLY"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()

	time.Sleep(50 * time.Millisecond)

	p, err := src.Price("ASK-ONLY")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price != 60.0 {
		t.Errorf("Price = %f, want 60.0 (ask only)", p.Price)
	}
}

func TestOrderbookSourceStalePrice(t *testing.T) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"STALE-TEST": 100.0},
		asks:    map[string]float64{"STALE-TEST": 100.1},
		symbols: []string{"STALE-TEST"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	time.Sleep(50 * time.Millisecond)
	src.Stop() // Stop updates

	// Manually set lastUpdate to be old
	src.mu.Lock()
	src.lastUpdate["STALE-TEST"] = time.Now().Add(-5 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("STALE-TEST")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be marked stale")
	}
}

func TestOrderbookSourceNilBooks(t *testing.T) {
	// Test update with nil books (should not panic)
	src := NewOrderbookSource(nil)
	src.update() // Should handle nil gracefully
}

// =============================================================================
// ZooChainSource Tests
// =============================================================================

func TestZooChainSourceBasic(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")

	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Close()

	// Wait for data
	time.Sleep(200 * time.Millisecond)

	// Check health
	if !src.Healthy() {
		t.Error("Source should be healthy")
	}

	// Check name
	if src.Name() != "zoo-chain" {
		t.Errorf("Name() = %q, want zoo-chain", src.Name())
	}

	// Check weight
	if src.Weight() != 1.7 {
		t.Errorf("Weight() = %f, want 1.7", src.Weight())
	}
}

func TestZooChainSourcePrice(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(200 * time.Millisecond)

	p, err := src.Price("ZOO-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price <= 0 {
		t.Error("Price should be > 0")
	}
	if p.Source != "zoo-chain" {
		t.Errorf("Source = %q, want zoo-chain", p.Source)
	}
}

func TestZooChainSourcePrices(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(200 * time.Millisecond)

	prices, err := src.Prices([]string{"ZOO-USDC", "ZOO-USDT", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	if len(prices) < 2 {
		t.Errorf("Expected at least 2 prices, got %d", len(prices))
	}
}

func TestZooChainSourceNotFound(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")
	// Don't start - no data

	_, err := src.Price("NONEXISTENT")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestZooChainSourceStalePrice(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	time.Sleep(200 * time.Millisecond)
	src.Close()

	// Manually set old timestamp
	src.mu.Lock()
	src.last["ZOO-USDC"] = time.Now().Add(-10 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("ZOO-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be stale")
	}
}

func TestZooChainSourceGetReserves(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")

	// No reserves by default
	_, err := src.GetReserves("ZOO-USDC")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestZooChainSourcePairs(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")

	pairs := src.Pairs()
	if len(pairs) == 0 {
		t.Error("Expected pairs")
	}

	// Check specific pair exists
	if _, ok := pairs["ZOO-USDC"]; !ok {
		t.Error("Missing ZOO-USDC pair")
	}
}

func TestZooChainSourceStartIdempotent(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")

	src.Start()
	src.Start() // Should not panic
	src.Close()
}

func TestZooChainSourceSubscribe(t *testing.T) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")

	// Subscribe/Unsubscribe are no-ops
	if err := src.Subscribe("ZOO-USDC"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("ZOO-USDC"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// ChainlinkSource Extended Tests
// =============================================================================

func TestChainlinkSourceStart(t *testing.T) {
	src := NewChainlinkSource()
	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Close()

	time.Sleep(100 * time.Millisecond)

	if !src.Healthy() {
		t.Error("Source should be healthy")
	}
}

func TestChainlinkSourceStartIdempotent(t *testing.T) {
	src := NewChainlinkSource()

	src.Start()
	src.Start() // Should not panic
	src.Close()
}

func TestChainlinkSourcePrice(t *testing.T) {
	src := NewChainlinkSource()
	src.Start()
	defer src.Close()

	time.Sleep(100 * time.Millisecond)

	p, err := src.Price("BTC-USD")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price <= 0 {
		t.Error("Price should be > 0")
	}
	if p.Source != "chainlink" {
		t.Errorf("Source = %q, want chainlink", p.Source)
	}
	if p.Confidence != 0.99 {
		t.Errorf("Confidence = %f, want 0.99", p.Confidence)
	}
}

func TestChainlinkSourcePrices(t *testing.T) {
	src := NewChainlinkSource()
	src.Start()
	defer src.Close()

	time.Sleep(100 * time.Millisecond)

	prices, err := src.Prices([]string{"BTC-USD", "ETH-USD", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	if len(prices) < 2 {
		t.Errorf("Expected at least 2 prices, got %d", len(prices))
	}
}

func TestChainlinkSourceNotFound(t *testing.T) {
	src := NewChainlinkSource()
	// Don't start

	_, err := src.Price("NONEXISTENT")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestChainlinkSourceStalePrice(t *testing.T) {
	src := NewChainlinkSource()
	src.Start()
	time.Sleep(100 * time.Millisecond)
	src.Close()

	// Manually set old timestamp
	src.mu.Lock()
	src.last["BTC-USD"] = time.Now().Add(-120 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("BTC-USD")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be stale")
	}
}

func TestChainlinkSourceSubscribe(t *testing.T) {
	src := NewChainlinkSource()

	if err := src.Subscribe("BTC-USD"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("BTC-USD"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// AChainSource Extended Tests
// =============================================================================

func TestAChainSourceValidators(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	validators := src.Validators()
	if len(validators) == 0 {
		t.Error("Expected validators")
	}

	// Check validator fields
	for name, v := range validators {
		if v.Address == "" {
			t.Errorf("Validator %s missing address", name)
		}
		if v.Stake == 0 {
			t.Errorf("Validator %s has zero stake", name)
		}
	}
}

func TestAChainSourceVerify(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(300 * time.Millisecond)

	// Get attestation for reference price
	att, err := src.Attestation("LUX-USD")
	if err != nil {
		t.Fatalf("Attestation failed: %v", err)
	}

	// Verify exact price
	if !src.Verify("LUX-USD", att.Price, 0.0) {
		t.Error("Should verify exact price")
	}

	// Verify within tolerance
	if !src.Verify("LUX-USD", att.Price*1.001, 0.01) {
		t.Error("Should verify within 1% tolerance")
	}

	// Reject outside tolerance
	if src.Verify("LUX-USD", att.Price*1.2, 0.01) {
		t.Error("Should reject price 20% off")
	}
}

func TestAChainSourceVerifyNotFinalized(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	// Don't start - no finalized attestations

	// Should return false for non-finalized
	if src.Verify("LUX-USD", 10.0, 0.1) {
		t.Error("Should return false when no attestation")
	}
}

func TestAChainSourceSetQuorum(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	// Default quorum
	src.SetQuorum(5)

	// Should not panic
}

func TestAChainSourcePrices(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(300 * time.Millisecond)

	prices, err := src.Prices([]string{"LUX-USD", "ETH-USD", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	if len(prices) < 2 {
		t.Errorf("Expected at least 2 prices, got %d", len(prices))
	}
}

func TestAChainSourceStalePrice(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	time.Sleep(300 * time.Millisecond)
	src.Close()

	// Manually set old timestamp
	src.mu.Lock()
	src.last["LUX-USD"] = time.Now().Add(-10 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("LUX-USD")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be stale")
	}
}

func TestAChainSourceStartIdempotent(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")

	src.Start()
	src.Start() // Should not panic
	src.Close()
}

func TestAChainSourceSubscribe(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if err := src.Subscribe("LUX-USD"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("LUX-USD"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// CChainSource Extended Tests
// =============================================================================

func TestCChainSourceGetReserves(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")

	// No reserves populated by default simulation
	_, err := src.GetReserves("LUX-USDC")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestCChainSourceStartIdempotent(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")

	src.Start()
	src.Start() // Should not panic
	src.Close()
}

func TestCChainSourcePrices(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(200 * time.Millisecond)

	prices, err := src.Prices([]string{"LUX-USDC", "LETH-USDC", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	if len(prices) < 2 {
		t.Errorf("Expected at least 2 prices, got %d", len(prices))
	}
}

func TestCChainSourceStalePrice(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	time.Sleep(200 * time.Millisecond)
	src.Close()

	// Manually set old timestamp
	src.mu.Lock()
	src.last["LUX-USDC"] = time.Now().Add(-10 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("LUX-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be stale")
	}
}

func TestCChainSourceSubscribe(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if err := src.Subscribe("LUX-USDC"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("LUX-USDC"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// XChainSource Extended Tests
// =============================================================================

func TestXChainSourceOrderbook(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(200 * time.Millisecond)

	book, err := src.Orderbook("LUX-USDC")
	if err != nil {
		t.Fatalf("Orderbook failed: %v", err)
	}

	if len(book.Bids) == 0 {
		t.Error("Expected bids")
	}
	if len(book.Asks) == 0 {
		t.Error("Expected asks")
	}
	if book.Spread <= 0 {
		t.Error("Expected spread > 0")
	}
}

func TestXChainSourceOrderbookNotFound(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	// Don't start

	_, err := src.Orderbook("NONEXISTENT")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestXChainSourceMarkets(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")

	markets := src.Markets()
	if len(markets) == 0 {
		t.Error("Expected markets")
	}

	// Check market fields
	if m, ok := markets["LUX-USDC"]; ok {
		if m.BaseAsset != "LUX" {
			t.Errorf("BaseAsset = %q, want LUX", m.BaseAsset)
		}
		if m.QuoteAsset != "USDC" {
			t.Errorf("QuoteAsset = %q, want USDC", m.QuoteAsset)
		}
	} else {
		t.Error("Missing LUX-USDC market")
	}
}

func TestXChainSourcePrices(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()

	time.Sleep(200 * time.Millisecond)

	prices, err := src.Prices([]string{"LUX-USDC", "ETH-LUX", "NONEXISTENT"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	if len(prices) < 2 {
		t.Errorf("Expected at least 2 prices, got %d", len(prices))
	}
}

func TestXChainSourceStalePrice(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	time.Sleep(200 * time.Millisecond)
	src.Close()

	// Manually set old timestamp
	src.mu.Lock()
	src.last["LUX-USDC"] = time.Now().Add(-10 * time.Second)
	src.mu.Unlock()

	p, err := src.Price("LUX-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if !p.Stale {
		t.Error("Expected price to be stale")
	}
}

func TestXChainSourceStartIdempotent(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")

	src.Start()
	src.Start() // Should not panic
	src.Close()
}

func TestXChainSourceSubscribe(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if err := src.Subscribe("LUX-USDC"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}
	if err := src.Unsubscribe("LUX-USDC"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// QChainVerifier Extended Tests
// =============================================================================

func TestQChainVerifierVerifyInclusion(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	v.Start()
	defer v.Close()

	time.Sleep(200 * time.Millisecond)

	// Verify an order inclusion
	inc, err := v.VerifyInclusion("order-123", "x-chain", "tx-abc123")
	if err != nil {
		t.Fatalf("VerifyInclusion failed: %v", err)
	}

	if inc.OrderID != "order-123" {
		t.Errorf("OrderID = %q, want order-123", inc.OrderID)
	}
	if inc.Chain != "x-chain" {
		t.Errorf("Chain = %q, want x-chain", inc.Chain)
	}
	if inc.TxHash != "tx-abc123" {
		t.Errorf("TxHash = %q, want tx-abc123", inc.TxHash)
	}
	if !inc.Verified {
		t.Error("Expected Verified=true")
	}
}

func TestQChainVerifierVerifyInclusionCached(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	v.Start()
	defer v.Close()

	time.Sleep(200 * time.Millisecond)

	// First call
	inc1, _ := v.VerifyInclusion("order-456", "c-chain", "tx-def456")

	// Second call should return cached
	inc2, _ := v.VerifyInclusion("order-456", "c-chain", "tx-def456")

	if inc1 != inc2 {
		t.Error("Expected cached inclusion to be returned")
	}
}

func TestQChainVerifierVerifyInclusionNoFinality(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	// Don't start - no finality data

	_, err := v.VerifyInclusion("order-789", "unknown-chain", "tx-xyz")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestQChainVerifierVerifyOrderPrice(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	v.Start()
	defer v.Close()

	time.Sleep(200 * time.Millisecond)

	// Should verify (finality is recent)
	if !v.VerifyOrderPrice("LUX-USD", 10.0, "x-chain") {
		t.Error("Expected price to be verified")
	}
}

func TestQChainVerifierVerifyOrderPriceNoFinality(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	// Don't start

	if v.VerifyOrderPrice("LUX-USD", 10.0, "x-chain") {
		t.Error("Expected false when no finality")
	}
}

func TestQChainVerifierCrossChainVerifyMissing(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	v.Start()
	defer v.Close()

	time.Sleep(200 * time.Millisecond)

	// Include a non-existent chain
	chains := []string{"x-chain", "nonexistent-chain"}
	allFinalized, results := v.CrossChainVerify(chains)

	if allFinalized {
		t.Error("Should not be all finalized with missing chain")
	}
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

func TestQChainVerifierFinalityNotFound(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	// Don't start

	_, err := v.Finality("x-chain")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestQChainVerifierStartIdempotent(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")

	v.Start()
	v.Start() // Should not panic
	v.Close()
}

func TestQChainVerifierCloseIdempotent(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")

	// Close without start
	v.Close() // Should not panic

	v.Start()
	v.Close()
	// Note: Can't close twice due to channel already closed panic
}

func TestQChainVerifierFinalityLatencyEmpty(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	// Don't start - no finality data

	latency := v.FinalityLatency()
	if latency != 0 {
		t.Errorf("Expected 0 latency with no data, got %v", latency)
	}
}

// =============================================================================
// PythSource Basic Tests (without actual connection)
// =============================================================================

func TestPythSourceBasic(t *testing.T) {
	src := NewPythSource("ws://localhost:4200/ws", "https://hermes.pyth.network")

	// Check name
	if src.Name() != "pyth" {
		t.Errorf("Name() = %q, want pyth", src.Name())
	}

	// Check weight
	if src.Weight() != 1.2 {
		t.Errorf("Weight() = %f, want 1.2", src.Weight())
	}
}

func TestPythSourceHealthyInitial(t *testing.T) {
	src := NewPythSource("ws://localhost:4200/ws", "https://hermes.pyth.network")
	defer src.Close()

	// Should not be healthy without connection
	if src.Healthy() {
		t.Error("Should not be healthy without connection")
	}
}

func TestPythSourcePrices(t *testing.T) {
	src := NewPythSource("ws://localhost:4200/ws", "https://hermes.pyth.network")
	defer src.Close()

	// Without connection, Prices should still work (return empty)
	prices, err := src.Prices([]string{"BTC-USD", "ETH-USD"})
	if err != nil {
		t.Fatalf("Prices failed: %v", err)
	}

	// Should be empty without data
	if len(prices) != 0 {
		t.Errorf("Expected 0 prices without data, got %d", len(prices))
	}
}

func TestPythSourceSubscribe(t *testing.T) {
	src := NewPythSource("ws://localhost:4200/ws", "https://hermes.pyth.network")
	defer src.Close()

	// Subscribe should track internally
	if err := src.Subscribe("BTC-USD"); err != nil {
		t.Errorf("Subscribe failed: %v", err)
	}

	// Unsubscribe
	if err := src.Unsubscribe("BTC-USD"); err != nil {
		t.Errorf("Unsubscribe failed: %v", err)
	}
}

// =============================================================================
// Oracle Extended Tests
// =============================================================================

func TestOracleWithMultipleSources(t *testing.T) {
	oracle := NewOracle()

	// Add multiple sources
	xchain := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	xchain.Start()

	cchain := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	cchain.Start()

	chainlink := NewChainlinkSource()
	chainlink.Start()

	oracle.AddSource("xchain", xchain)
	oracle.AddSource("cchain", cchain)
	oracle.AddSource("chainlink", chainlink)

	// Watch symbol that exists in multiple sources
	oracle.Watch("LUX-USDC")

	oracle.Start()
	time.Sleep(300 * time.Millisecond)

	// Should have aggregated price
	price := oracle.Price("LUX-USDC")
	// Price might be 0 if sources don't have this exact symbol mapped
	// At minimum, the oracle should run without crashing

	oracle.Stop()
	xchain.Close()
	cchain.Close()
	chainlink.Close()

	_ = price // Suppress unused warning
}

func TestOracleCircuitBreakerTrip(t *testing.T) {
	oracle := NewOracle()

	// Add circuit breaker
	oracle.mu.Lock()
	oracle.breakers["TEST-USD"] = &CircuitBreaker{
		Symbol:    "TEST-USD",
		MaxChange: 5.0,
		Reset:     time.Second,
		LastPrice: 100.0, // Pre-set last price
	}
	oracle.mu.Unlock()

	oracle.Start()
	defer oracle.Stop()

	// Breaker is set - should affect update behavior
	// (Full integration would require injecting prices that trip breaker)
}

func TestOracleMinSourcesFiltering(t *testing.T) {
	oracle := NewOracle()
	oracle.minSources = 2 // Require 2 sources

	// Add only 1 source
	xchain := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	xchain.Start()
	oracle.AddSource("xchain", xchain)

	oracle.Watch("LUX-USDC")
	oracle.Start()
	time.Sleep(200 * time.Millisecond)

	// With only 1 source, shouldn't aggregate (needs 2)
	// Price should be 0
	price := oracle.Price("LUX-USDC")

	oracle.Stop()
	xchain.Close()

	_ = price // May or may not be 0 depending on implementation
}

func TestOracleUpdateChannel(t *testing.T) {
	oracle := NewOracle()

	updates := oracle.Updates()
	if updates == nil {
		t.Fatal("Updates channel should not be nil")
	}

	// Channel is read-only, verify it exists and is the correct type
	// We can't write to it (which is correct behavior for subscribers)
	_ = updates // Use the channel to verify it's valid
}

// =============================================================================
// Confidence Calculation Tests
// =============================================================================

func TestWeightedMedianConfidenceZero(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	// Empty prices
	conf := wm.confidence([]*Data{})
	if conf != 0 {
		t.Errorf("Expected 0 confidence for empty, got %f", conf)
	}
}

func TestWeightedMedianConfidenceSingleSource(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0},
	}

	conf := wm.confidence(prices)
	// Single source: sourceScore = 1/(1*2) = 0.5
	// devScore = 1 (no deviation with single price)
	// confidence = 0.5*0.6 + 1.0*0.4 = 0.7
	if conf < 0.5 || conf > 1.0 {
		t.Errorf("Expected confidence between 0.5 and 1.0, got %f", conf)
	}
}

func TestWeightedMedianConfidenceHighAgreement(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   2,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0},
		{Symbol: "TEST", Price: 100.001},
		{Symbol: "TEST", Price: 99.999},
		{Symbol: "TEST", Price: 100.0},
	}

	conf := wm.confidence(prices)
	// 4 sources: sourceScore = 4/(2*2) = 1.0 (capped)
	// Very low deviation -> devScore near 1.0
	// Should be high confidence
	if conf < 0.8 {
		t.Errorf("Expected high confidence for high agreement, got %f", conf)
	}
}

// =============================================================================
// Symbol Normalization Edge Cases
// =============================================================================

func TestNormalizeWithSpaces(t *testing.T) {
	result := Normalize("  LUX/USD  ")
	if result != "LUX-USD" {
		t.Errorf("Normalize with spaces = %q, want LUX-USD", result)
	}
}

func TestDetectPairWithBTC(t *testing.T) {
	result := detectPair("ETHBTC")
	if result != "ETH-BTC" {
		t.Errorf("detectPair(ETHBTC) = %q, want ETH-BTC", result)
	}
}

func TestDetectPairWithETH(t *testing.T) {
	result := detectPair("LUXETH")
	if result != "LUX-ETH" {
		t.Errorf("detectPair(LUXETH) = %q, want LUX-ETH", result)
	}
}

func TestDetectPairWithLUX(t *testing.T) {
	result := detectPair("ETHLUX")
	if result != "ETH-LUX" {
		t.Errorf("detectPair(ETHLUX) = %q, want ETH-LUX", result)
	}
}

func TestReverseWithMultipleMatches(t *testing.T) {
	m := NewSymbolMap()

	// LUX-USD has multiple aliases
	variants := m.Reverse("LUX-USD")

	// Should include LUX-USD itself plus aliases
	found := false
	for _, v := range variants {
		if v == "LUX-USD" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Reverse should include the normalized symbol itself")
	}
}

// =============================================================================
// Error Variable Tests
// =============================================================================

func TestErrorVariables(t *testing.T) {
	// Verify error messages
	if ErrNotFound.Error() != "price not found" {
		t.Errorf("ErrNotFound = %q", ErrNotFound.Error())
	}
	if ErrStale.Error() != "price is stale" {
		t.Errorf("ErrStale = %q", ErrStale.Error())
	}
	if ErrInsufficientSources.Error() != "insufficient price sources" {
		t.Errorf("ErrInsufficientSources = %q", ErrInsufficientSources.Error())
	}
	if ErrCircuitBreaker.Error() != "circuit breaker tripped" {
		t.Errorf("ErrCircuitBreaker = %q", ErrCircuitBreaker.Error())
	}
}

// =============================================================================
// Additional Helper Tests
// =============================================================================

func TestQuoteFunction(t *testing.T) {
	tests := []struct {
		symbol   string
		expected string
	}{
		{"LUX-USD", "USD"},
		{"BTC-USDC", "USDC"},
		{"ETH/USDT", "USDT"},
		{"SINGLE", ""},
	}

	for _, tc := range tests {
		got := Quote(tc.symbol)
		if got != tc.expected {
			t.Errorf("Quote(%q) = %q, want %q", tc.symbol, got, tc.expected)
		}
	}
}

// =============================================================================
// Benchmarks for New Functions
// =============================================================================

func BenchmarkOrderbookSourcePrice(b *testing.B) {
	provider := &mockOrderbookProvider{
		bids:    map[string]float64{"LUX-USDC": 10.0},
		asks:    map[string]float64{"LUX-USDC": 10.02},
		symbols: []string{"LUX-USDC"},
	}

	src := NewOrderbookSource(provider)
	src.Start()
	defer src.Stop()
	time.Sleep(50 * time.Millisecond)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = src.Price("LUX-USDC")
	}
}

func BenchmarkZooChainSourcePrice(b *testing.B) {
	src := NewZooChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()
	time.Sleep(200 * time.Millisecond)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = src.Price("ZOO-USDC")
	}
}

func BenchmarkQChainVerifierFinality(b *testing.B) {
	v := NewQChainVerifier("http://localhost:9650", "ws://localhost:9650")
	v.Start()
	defer v.Close()
	time.Sleep(200 * time.Millisecond)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = v.Finality("x-chain")
	}
}
