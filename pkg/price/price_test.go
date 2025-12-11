package price

import (
	"testing"
	"time"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"LUX/USD", "LUX-USD"},
		{"LUX-USD", "LUX-USD"},
		{"LUXUSD", "LUX-USD"},
		{"ETH/USDC", "ETH-USDC"},
		{"btc-usd", "BTC-USD"},
		{"AVAX_USDT", "AVAX-USDT"},
	}

	for _, tc := range tests {
		got := Normalize(tc.input)
		if got != tc.expected {
			t.Errorf("Normalize(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

func TestSymbolMap(t *testing.T) {
	m := NewSymbolMap()

	// Test mapping
	if got := m.Map("LUX/USD"); got != "LUX-USD" {
		t.Errorf("Map(LUX/USD) = %q, want LUX-USD", got)
	}

	// Test reverse
	variants := m.Reverse("LUX-USD")
	if len(variants) == 0 {
		t.Error("Reverse(LUX-USD) returned empty")
	}
}

func TestBaseQuote(t *testing.T) {
	base, quote := BaseQuote("LUX-USDC")
	if base != "LUX" || quote != "USDC" {
		t.Errorf("BaseQuote(LUX-USDC) = %q, %q, want LUX, USDC", base, quote)
	}
}

func TestIsUSD(t *testing.T) {
	if !IsUSD("LUX-USD") {
		t.Error("IsUSD(LUX-USD) = false, want true")
	}
	if !IsUSD("ETH-USDC") {
		t.Error("IsUSD(ETH-USDC) = false, want true")
	}
	if IsUSD("ETH-BTC") {
		t.Error("IsUSD(ETH-BTC) = true, want false")
	}
}

func TestXChainSource(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Close()

	// Wait for data
	time.Sleep(200 * time.Millisecond)

	// Check health
	if !src.Healthy() {
		t.Error("Source not healthy")
	}

	// Check name
	if src.Name() != "x-chain" {
		t.Errorf("Name() = %q, want x-chain", src.Name())
	}

	// Get price
	p, err := src.Price("LUX-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price <= 0 {
		t.Error("Price <= 0")
	}
	if p.Symbol != "LUX-USDC" {
		t.Errorf("Symbol = %q, want LUX-USDC", p.Symbol)
	}
}

func TestAChainSource(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Close()

	// Wait for data
	time.Sleep(300 * time.Millisecond)

	// Check health
	if !src.Healthy() {
		t.Error("Source not healthy")
	}

	// Get attestation
	att, err := src.Attestation("LUX-USD")
	if err != nil {
		t.Fatalf("Attestation failed: %v", err)
	}

	if att.Price <= 0 {
		t.Error("Attestation price <= 0")
	}
	if !att.Finalized {
		t.Error("Attestation not finalized")
	}
}

func TestCChainSource(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	if err := src.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer src.Close()

	// Wait for data
	time.Sleep(200 * time.Millisecond)

	// Check health
	if !src.Healthy() {
		t.Error("Source not healthy")
	}

	// Get price
	p, err := src.Price("LUX-USDC")
	if err != nil {
		t.Fatalf("Price failed: %v", err)
	}

	if p.Price <= 0 {
		t.Error("Price <= 0")
	}
}

func TestOracle(t *testing.T) {
	oracle := NewOracle()

	// Add sources
	xchain := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	xchain.Start()

	chainlink := NewChainlinkSource()
	chainlink.Start()

	oracle.AddSource("xchain", xchain)
	oracle.AddSource("chainlink", chainlink)

	// Watch symbol
	oracle.Watch("LUX-USD")

	// Start
	if err := oracle.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Wait for aggregation
	time.Sleep(500 * time.Millisecond)

	// Get price
	price := oracle.Price("LUX-USD")
	if price <= 0 {
		t.Error("Aggregated price <= 0")
	}

	// Stop oracle first, then sources
	oracle.Stop()
	xchain.Close()
	chainlink.Close()
}

func TestCircuitBreaker(t *testing.T) {
	cb := &CircuitBreaker{
		Symbol:    "TEST",
		MaxChange: 10.0, // 10%
		Reset:     time.Second,
	}

	// First price
	if !cb.Check(100.0) {
		t.Error("First check should pass")
	}

	// Normal change
	if !cb.Check(105.0) {
		t.Error("5% change should pass")
	}

	// Excessive change
	if cb.Check(150.0) {
		t.Error("50% change should trip breaker")
	}

	// Should be tripped
	if !cb.Tripped {
		t.Error("Breaker should be tripped")
	}

	// Wait for reset
	time.Sleep(1100 * time.Millisecond)

	// Should reset
	if !cb.Check(100.0) {
		t.Error("Should pass after reset")
	}
}

func TestWeightedMedian(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a"},
		{Symbol: "TEST", Price: 101.0, Source: "b"},
		{Symbol: "TEST", Price: 99.0, Source: "c"},
	}

	agg, err := wm.Aggregate(prices)
	if err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}

	// Should be close to median
	if agg.Price < 99 || agg.Price > 101 {
		t.Errorf("Aggregated price %f outside expected range", agg.Price)
	}
}
