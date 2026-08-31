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

	// poll() drives the source's fetch path deterministically; calling it
	// directly avoids racing the background ticker started by Start().
	src.poll()

	if !src.Healthy() {
		t.Error("Source not healthy after poll")
	}

	// The source serves the pairs it is configured to poll (cchainTokens).
	// Pick the first configured pair and assert it is now populated.
	sym := firstPolledSymbol(t, src)
	p, err := src.Price(sym)
	if err != nil {
		t.Fatalf("Price(%q) after poll: %v", sym, err)
	}
	if p.Symbol != sym {
		t.Errorf("Price symbol = %q, want %q", p.Symbol, sym)
	}
	if p.Source != "c-chain" {
		t.Errorf("Price source = %q, want %q", p.Source, "c-chain")
	}

	// A symbol the source does not track must report not-found.
	if _, err := src.Price("NOT-A-PAIR"); err == nil {
		t.Error("Price for untracked symbol: want error, got nil")
	}
}

// firstPolledSymbol returns one symbol the source is configured to poll.
func firstPolledSymbol(t *testing.T, src *CChainSource) string {
	t.Helper()
	for sym := range src.tokens {
		return sym
	}
	t.Fatal("CChainSource has no configured token pairs")
	return ""
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

// === Additional Oracle Tests ===

func TestOracleData(t *testing.T) {
	oracle := NewOracle()

	// No data - should error
	_, err := oracle.Data("LUX-USD")
	if err == nil {
		t.Error("Expected error for non-existent symbol")
	}
}

func TestOracleTWAP(t *testing.T) {
	oracle := NewOracle()
	twap := oracle.TWAP("LUX-USD")
	if twap != 0 {
		t.Errorf("TWAP for unknown symbol should be 0, got %f", twap)
	}
}

func TestOracleVWAP(t *testing.T) {
	oracle := NewOracle()
	vwap := oracle.VWAP("LUX-USD")
	if vwap != 0 {
		t.Errorf("VWAP for unknown symbol should be 0, got %f", vwap)
	}
}

func TestOracleUpdates(t *testing.T) {
	oracle := NewOracle()
	updates := oracle.Updates()
	if updates == nil {
		t.Error("Updates channel should not be nil")
	}
}

func TestOracleAlerts(t *testing.T) {
	oracle := NewOracle()
	alerts := oracle.Alerts()
	if alerts == nil {
		t.Error("Alerts channel should not be nil")
	}
}

func TestOracleStartStop(t *testing.T) {
	oracle := NewOracle()

	// Start multiple times should be idempotent
	if err := oracle.Start(); err != nil {
		t.Fatalf("First start failed: %v", err)
	}
	if err := oracle.Start(); err != nil {
		t.Fatalf("Second start should be no-op: %v", err)
	}

	// Stop multiple times should be idempotent
	oracle.Stop()
	oracle.Stop() // Should not panic
}

func TestOracleWatch(t *testing.T) {
	oracle := NewOracle()

	// Watch symbol
	oracle.Watch("BTC-USD")
	oracle.Watch("ETH-USD")

	// Check internal state (via watching map)
	if oracle.Price("BTC-USD") != 0 {
		t.Error("Price should be 0 before data")
	}
}

// === WeightedMedian Extended Tests ===

func TestWeightedMedianInsufficientSources(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   3,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a"},
	}

	_, err := wm.Aggregate(prices)
	if err == nil {
		t.Error("Should fail with insufficient sources")
	}
}

func TestWeightedMedianOutliers(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   2,
		MaxDeviation: 0.05, // 5%
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a"},
		{Symbol: "TEST", Price: 101.0, Source: "b"},
		{Symbol: "TEST", Price: 200.0, Source: "c"}, // Outlier
	}

	agg, err := wm.Aggregate(prices)
	if err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}

	// Should exclude the outlier (200)
	if agg.Price > 105 {
		t.Errorf("Expected price near 100, got %f (outlier not filtered)", agg.Price)
	}
}

func TestWeightedMedianTooManyOutliers(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   3,
		MaxDeviation: 0.01, // Very strict 1%
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a"},
		{Symbol: "TEST", Price: 150.0, Source: "b"}, // Outlier
		{Symbol: "TEST", Price: 200.0, Source: "c"}, // Outlier
	}

	_, err := wm.Aggregate(prices)
	if err == nil {
		t.Error("Should fail with too many outliers")
	}
}

func TestWeightedMedianValidate(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	// Empty prices
	err := wm.Validate([]*Data{})
	if err == nil {
		t.Error("Validate should fail for empty prices")
	}

	// Valid prices
	err = wm.Validate([]*Data{{Symbol: "TEST", Price: 100.0}})
	if err != nil {
		t.Errorf("Validate should pass: %v", err)
	}
}

func TestWeightedMedianWithVolume(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Volume: 1000.0, Source: "a"},
		{Symbol: "TEST", Price: 101.0, Volume: 500.0, Source: "b"},
	}

	agg, err := wm.Aggregate(prices)
	if err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}

	// Total volume should be sum
	if agg.Volume != 1500.0 {
		t.Errorf("Expected volume 1500, got %f", agg.Volume)
	}
}

// === Circuit Breaker Extended Tests ===

func TestCircuitBreakerTrippedState(t *testing.T) {
	cb := &CircuitBreaker{
		Symbol:    "TEST",
		MaxChange: 10.0,
		Reset:     100 * time.Millisecond,
	}

	// Set initial price
	cb.Check(100.0)

	// Trip the breaker
	cb.Check(150.0)

	if !cb.Tripped {
		t.Error("Breaker should be tripped")
	}

	// Should reject prices while tripped
	if cb.Check(105.0) {
		t.Error("Should reject prices while tripped")
	}
}

func TestCircuitBreakerAutoReset(t *testing.T) {
	cb := &CircuitBreaker{
		Symbol:    "TEST",
		MaxChange: 10.0,
		Reset:     50 * time.Millisecond,
	}

	// Set initial and trip
	cb.Check(100.0)
	cb.Check(150.0)

	if !cb.Tripped {
		t.Error("Breaker should be tripped")
	}

	// Wait for auto-reset
	time.Sleep(100 * time.Millisecond)

	// Should accept now
	if !cb.Check(100.0) {
		t.Error("Breaker should auto-reset")
	}
}

func TestCircuitBreakerUpdateLastPrice(t *testing.T) {
	cb := &CircuitBreaker{
		Symbol:    "TEST",
		MaxChange: 10.0,
		Reset:     time.Second,
	}

	cb.Check(100.0)
	cb.Check(105.0)

	if cb.LastPrice != 105.0 {
		t.Errorf("LastPrice should be 105, got %f", cb.LastPrice)
	}
}

// === Helper Function Tests ===

func TestCalcTWAP(t *testing.T) {
	// Empty history
	result := calcTWAP([]*Data{}, 5*time.Minute)
	if result != 0 {
		t.Errorf("TWAP of empty history should be 0, got %f", result)
	}

	// Recent data
	now := time.Now()
	history := []*Data{
		{Price: 100.0, Timestamp: now.Add(-1 * time.Minute)},
		{Price: 102.0, Timestamp: now.Add(-2 * time.Minute)},
		{Price: 98.0, Timestamp: now.Add(-3 * time.Minute)},
	}

	result = calcTWAP(history, 5*time.Minute)
	expected := 100.0 // (100 + 102 + 98) / 3
	if result != expected {
		t.Errorf("TWAP should be %f, got %f", expected, result)
	}
}

func TestCalcTWAPWindowFiltering(t *testing.T) {
	now := time.Now()
	history := []*Data{
		{Price: 100.0, Timestamp: now.Add(-1 * time.Minute)},
		{Price: 200.0, Timestamp: now.Add(-10 * time.Minute)}, // Outside window
	}

	result := calcTWAP(history, 5*time.Minute)
	if result != 100.0 {
		t.Errorf("TWAP should only include recent data, got %f", result)
	}
}

func TestCalcVWAP(t *testing.T) {
	// Empty history
	result := calcVWAP([]*Data{}, 5*time.Minute)
	if result != 0 {
		t.Errorf("VWAP of empty history should be 0, got %f", result)
	}

	// With volume data
	now := time.Now()
	history := []*Data{
		{Price: 100.0, Volume: 1000.0, Timestamp: now.Add(-1 * time.Minute)},
		{Price: 110.0, Volume: 500.0, Timestamp: now.Add(-2 * time.Minute)},
	}

	result = calcVWAP(history, 5*time.Minute)
	// VWAP = (100*1000 + 110*500) / (1000 + 500) = 155000 / 1500 = 103.33
	expected := 103.33
	if result < 103.0 || result > 104.0 {
		t.Errorf("VWAP should be ~%f, got %f", expected, result)
	}
}

func TestCalcVWAPZeroVolume(t *testing.T) {
	now := time.Now()
	history := []*Data{
		{Price: 100.0, Volume: 0.0, Timestamp: now.Add(-1 * time.Minute)},
	}

	result := calcVWAP(history, 5*time.Minute)
	if result != 0 {
		t.Errorf("VWAP with zero volume should be 0, got %f", result)
	}
}

func TestAvg(t *testing.T) {
	// Empty
	if avg([]float64{}) != 0 {
		t.Error("avg of empty should be 0")
	}

	// Normal
	result := avg([]float64{1.0, 2.0, 3.0})
	if result != 2.0 {
		t.Errorf("avg should be 2.0, got %f", result)
	}
}

func TestStddev(t *testing.T) {
	// Empty
	if stddev([]float64{}, 0) != 0 {
		t.Error("stddev of empty should be 0")
	}

	// No variance
	values := []float64{5.0, 5.0, 5.0}
	result := stddev(values, 5.0)
	if result != 0 {
		t.Errorf("stddev of same values should be 0, got %f", result)
	}

	// With variance
	values = []float64{1.0, 2.0, 3.0}
	mean := 2.0
	result = stddev(values, mean)
	// sqrt((1 + 0 + 1) / 3) = sqrt(2/3) ≈ 0.816
	if result < 0.8 || result > 0.9 {
		t.Errorf("stddev should be ~0.816, got %f", result)
	}
}

// === Data Type Tests ===

func TestDataFields(t *testing.T) {
	now := time.Now()
	data := &Data{
		Symbol:     "BTC-USD",
		Price:      50000.0,
		Bid:        49999.0,
		Ask:        50001.0,
		Volume:     1000000.0,
		High24h:    51000.0,
		Low24h:     49000.0,
		Change24h:  2.5,
		Timestamp:  now,
		Source:     "exchange",
		Confidence: 0.95,
		Stale:      false,
	}

	if data.Symbol != "BTC-USD" {
		t.Error("Symbol mismatch")
	}
	if data.Price != 50000.0 {
		t.Error("Price mismatch")
	}
	if data.Confidence != 0.95 {
		t.Error("Confidence mismatch")
	}
}

func TestUpdateFields(t *testing.T) {
	now := time.Now()
	update := &Update{
		Symbol:    "ETH-USD",
		OldPrice:  3000.0,
		NewPrice:  3100.0,
		Source:    "aggregator",
		Timestamp: now,
		Change:    3.33,
	}

	if update.Symbol != "ETH-USD" {
		t.Error("Symbol mismatch")
	}
	if update.Change != 3.33 {
		t.Error("Change mismatch")
	}
}

func TestAlertFields(t *testing.T) {
	now := time.Now()
	alert := &Alert{
		ID:        "alert-1",
		Symbol:    "BTC-USD",
		Type:      AlertCircuitBreaker,
		Message:   "Price movement too large",
		Severity:  SeverityCrit,
		Price:     50000.0,
		Timestamp: now,
	}

	if alert.Type != AlertCircuitBreaker {
		t.Error("Type mismatch")
	}
	if alert.Severity != SeverityCrit {
		t.Error("Severity mismatch")
	}
}

func TestAlertTypes(t *testing.T) {
	if AlertStale != 0 {
		t.Error("AlertStale should be 0")
	}
	if AlertDeviation != 1 {
		t.Error("AlertDeviation should be 1")
	}
	if AlertSourceDown != 2 {
		t.Error("AlertSourceDown should be 2")
	}
	if AlertCircuitBreaker != 3 {
		t.Error("AlertCircuitBreaker should be 3")
	}
	if AlertLowSources != 4 {
		t.Error("AlertLowSources should be 4")
	}
}

func TestSeverityLevels(t *testing.T) {
	if SeverityInfo != 0 {
		t.Error("SeverityInfo should be 0")
	}
	if SeverityWarn != 1 {
		t.Error("SeverityWarn should be 1")
	}
	if SeverityCrit != 2 {
		t.Error("SeverityCrit should be 2")
	}
}

// === Normalize Extended Tests ===

func TestNormalizeEdgeCases(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"btcusd", "BTC-USD"},
		{"BTC_USD", "BTC-USD"},
		{"sol/usdt", "SOL-USDT"},
	}

	for _, tc := range tests {
		got := Normalize(tc.input)
		if got != tc.expected {
			t.Errorf("Normalize(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

// === Benchmarks ===

func BenchmarkWeightedMedianAggregate(b *testing.B) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a", Volume: 1000.0},
		{Symbol: "TEST", Price: 101.0, Source: "b", Volume: 800.0},
		{Symbol: "TEST", Price: 99.5, Source: "c", Volume: 1200.0},
		{Symbol: "TEST", Price: 100.5, Source: "d", Volume: 500.0},
		{Symbol: "TEST", Price: 99.8, Source: "e", Volume: 900.0},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = wm.Aggregate(prices)
	}
}

func BenchmarkCircuitBreakerCheck(b *testing.B) {
	cb := &CircuitBreaker{
		Symbol:    "BENCH",
		MaxChange: 10.0,
		Reset:     time.Second,
	}
	cb.Check(100.0) // Initialize

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Check(100.0 + float64(i%5))
	}
}

func BenchmarkCalcTWAP(b *testing.B) {
	now := time.Now()
	history := make([]*Data, 1000)
	for i := 0; i < 1000; i++ {
		history[i] = &Data{
			Price:     100.0 + float64(i%10),
			Timestamp: now.Add(-time.Duration(i) * time.Second),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calcTWAP(history, 5*time.Minute)
	}
}

func BenchmarkNormalize(b *testing.B) {
	symbols := []string{"LUX/USD", "ETH-USDC", "btcusd", "AVAX_USDT"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Normalize(symbols[i%len(symbols)])
	}
}

// === Additional Normalize Function Tests for 100% Coverage ===

func TestAddAlias(t *testing.T) {
	m := NewSymbolMap()

	// Add custom alias
	m.AddAlias("WLUX/USD", "LUX-USD")

	// Verify alias works
	result := m.Map("WLUX/USD")
	if result != "LUX-USD" {
		t.Errorf("AddAlias failed: got %q, want LUX-USD", result)
	}
}

func TestBase(t *testing.T) {
	tests := []struct {
		symbol   string
		expected string
	}{
		{"LUX-USD", "LUX"},
		{"ETH-USDC", "ETH"},
		{"BTC/USDT", "BTC"},
		{"SOL", "SOL"}, // No separator
	}

	for _, tc := range tests {
		got := Base(tc.symbol)
		if got != tc.expected {
			t.Errorf("Base(%q) = %q, want %q", tc.symbol, got, tc.expected)
		}
	}
}

func TestSameBase(t *testing.T) {
	tests := []struct {
		a, b     string
		expected bool
	}{
		{"LUX-USD", "LUX-USDC", true},
		{"ETH-USD", "ETH-BTC", true},
		{"BTC-USD", "ETH-USD", false},
		{"AVAX-USDT", "avax/usd", true}, // Case insensitive after normalize
	}

	for _, tc := range tests {
		got := SameBase(tc.a, tc.b)
		if got != tc.expected {
			t.Errorf("SameBase(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.expected)
		}
	}
}

func TestSameQuote(t *testing.T) {
	tests := []struct {
		a, b     string
		expected bool
	}{
		{"LUX-USD", "ETH-USD", true},
		{"BTC-USDC", "SOL-USDC", true},
		{"LUX-USD", "LUX-USDC", false},
		{"eth/usdt", "BTC-USDT", true}, // Case insensitive after normalize
	}

	for _, tc := range tests {
		got := SameQuote(tc.a, tc.b)
		if got != tc.expected {
			t.Errorf("SameQuote(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.expected)
		}
	}
}

func TestSymbolMapWithAlias(t *testing.T) {
	m := NewSymbolMap()

	// Test existing alias
	result := m.Map("LUX/USD")
	if result != "LUX-USD" {
		t.Errorf("Map(LUX/USD) = %q, want LUX-USD", result)
	}

	// Test non-aliased symbol goes through normalize
	result = m.Map("DOGE-BTC")
	if result != "DOGE-BTC" {
		t.Errorf("Map(DOGE-BTC) = %q, want DOGE-BTC", result)
	}
}

func TestOracleAlert(t *testing.T) {
	oracle := NewOracle()

	// Start to enable alert sending
	oracle.Start()
	defer oracle.Stop()

	// Add a circuit breaker
	oracle.breakers["TEST-USD"] = &CircuitBreaker{
		Symbol:    "TEST-USD",
		MaxChange: 5.0,
		Reset:     time.Second,
	}

	// The alert function is internal, but we can verify alerts channel works
	alerts := oracle.Alerts()
	if alerts == nil {
		t.Error("Alerts channel should not be nil")
	}
}

func TestXChainSourceName(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if src.Name() != "x-chain" {
		t.Errorf("Name() = %q, want x-chain", src.Name())
	}
}

func TestXChainSourceWeight(t *testing.T) {
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	weight := src.Weight()
	if weight <= 0 {
		t.Errorf("Weight() = %f, want > 0", weight)
	}
}

func TestAChainSourceName(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if src.Name() != "a-chain" {
		t.Errorf("Name() = %q, want a-chain", src.Name())
	}
}

func TestAChainSourceWeight(t *testing.T) {
	src := NewAChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	weight := src.Weight()
	if weight <= 0 {
		t.Errorf("Weight() = %f, want > 0", weight)
	}
}

func TestCChainSourceName(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	if src.Name() != "c-chain" {
		t.Errorf("Name() = %q, want c-chain", src.Name())
	}
}

func TestCChainSourceWeight(t *testing.T) {
	src := NewCChainSource("http://localhost:9650", "ws://localhost:9650")
	defer src.Close()

	weight := src.Weight()
	if weight <= 0 {
		t.Errorf("Weight() = %f, want > 0", weight)
	}
}

func TestChainlinkSourceName(t *testing.T) {
	src := NewChainlinkSource()
	defer src.Close()

	if src.Name() != "chainlink" {
		t.Errorf("Name() = %q, want chainlink", src.Name())
	}
}

func TestChainlinkSourceWeight(t *testing.T) {
	src := NewChainlinkSource()
	defer src.Close()

	weight := src.Weight()
	if weight <= 0 {
		t.Errorf("Weight() = %f, want > 0", weight)
	}
}

func TestOracleDataNotFound(t *testing.T) {
	oracle := NewOracle()

	_, err := oracle.Data("NONEXISTENT")
	if err == nil {
		t.Error("Expected error for non-existent symbol")
	}
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestOraclePriceStale(t *testing.T) {
	oracle := NewOracle()

	// Manually inject stale price
	oracle.mu.Lock()
	oracle.current["TEST-USD"] = &Data{
		Symbol:    "TEST-USD",
		Price:     100.0,
		Timestamp: time.Now().Add(-10 * time.Second), // Very stale
	}
	oracle.staleLimit = 1 * time.Second
	oracle.mu.Unlock()

	// Should return 0 for stale price
	price := oracle.Price("TEST-USD")
	if price != 0 {
		t.Errorf("Expected 0 for stale price, got %f", price)
	}
}

func TestWeightedMedianConfidence(t *testing.T) {
	wm := &WeightedMedian{
		MinSources:   1,
		MaxDeviation: 0.10,
	}

	// Test confidence with multiple sources (high agreement)
	prices := []*Data{
		{Symbol: "TEST", Price: 100.0, Source: "a"},
		{Symbol: "TEST", Price: 100.1, Source: "b"},
		{Symbol: "TEST", Price: 99.9, Source: "c"},
	}

	agg, err := wm.Aggregate(prices)
	if err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}

	if agg.Confidence <= 0 || agg.Confidence > 1.0 {
		t.Errorf("Confidence should be between 0 and 1, got %f", agg.Confidence)
	}
}

func TestDetectPairUnknown(t *testing.T) {
	// Test symbol that doesn't match any known quote
	result := detectPair("ABCXYZ")
	if result != "ABCXYZ" {
		t.Errorf("detectPair(ABCXYZ) = %q, want ABCXYZ", result)
	}
}

func TestDetectPairShort(t *testing.T) {
	// Test symbol too short to have base+quote
	result := detectPair("USD")
	if result != "USD" {
		t.Errorf("detectPair(USD) = %q, want USD", result)
	}
}

func TestBaseQuoteNoSeparator(t *testing.T) {
	// Test with symbol that can't be split
	base, quote := BaseQuote("INVALID")
	if quote != "" {
		t.Errorf("BaseQuote(INVALID) quote = %q, want empty", quote)
	}
	if base != "INVALID" {
		t.Errorf("BaseQuote(INVALID) base = %q, want INVALID", base)
	}
}

// === Q-Chain Verifier Integration Tests ===

func TestOracleSetVerifier(t *testing.T) {
	oracle := NewOracle()

	// Initially no verifier
	if oracle.Verifier() != nil {
		t.Error("Expected no verifier initially")
	}

	// Create and attach verifier
	verifier := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	oracle.SetVerifier(verifier)

	// Verify it's attached
	if oracle.Verifier() == nil {
		t.Error("Expected verifier to be attached")
	}
}

func TestOracleVerifiedPrice(t *testing.T) {
	oracle := NewOracle()

	// Add a source
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	defer src.Close()
	oracle.AddSource("x-chain", src)

	// Watch symbol
	oracle.Watch("LUX-USDC")

	// Start oracle
	oracle.Start()
	defer oracle.Stop()

	// Wait for data
	time.Sleep(200 * time.Millisecond)

	// Test without verifier
	vd, err := oracle.VerifiedPrice("LUX-USDC")
	if err != nil {
		t.Fatalf("VerifiedPrice failed: %v", err)
	}

	if vd.Finalized {
		t.Error("Expected Finalized=false without verifier")
	}
	if vd.Finality != nil {
		t.Error("Expected nil Finality without verifier")
	}
	if vd.Data == nil {
		t.Error("Expected Data to be set")
	}
}

func TestOracleVerifiedPriceWithVerifier(t *testing.T) {
	oracle := NewOracle()

	// Add a source
	src := NewXChainSource("http://localhost:9650", "ws://localhost:9650")
	src.Start()
	oracle.AddSource("x-chain", src)

	// Add verifier
	verifier := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	verifier.Start()
	oracle.SetVerifier(verifier)

	// Watch symbol
	oracle.Watch("LUX-USDC")

	// Start oracle
	oracle.Start()

	// Wait for data
	time.Sleep(300 * time.Millisecond)

	// Test with verifier
	vd, err := oracle.VerifiedPrice("LUX-USDC")
	if err != nil {
		t.Fatalf("VerifiedPrice failed: %v", err)
	}

	// Data should exist
	if vd.Data == nil {
		t.Fatal("Expected Data to be set")
	}

	// Verifier should have finality (simulated) if source is x-chain
	if vd.Data.Source == "x-chain" {
		if !vd.Finalized {
			t.Error("Expected Finalized=true for x-chain source with active verifier")
		}
		if vd.Finality == nil {
			t.Error("Expected Finality to be set for x-chain source with active verifier")
		}
	}

	// Cleanup - oracle.Stop() will close verifier, so don't double-close
	oracle.Stop()
	src.Close()
}

func TestOracleStopWithVerifier(t *testing.T) {
	oracle := NewOracle()

	// Add verifier
	verifier := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	verifier.Start()
	oracle.SetVerifier(verifier)

	// Start oracle
	oracle.Start()

	// Stop should close verifier too
	oracle.Stop()

	// Verify verifier was closed (no panic)
}

func TestSourceToChain(t *testing.T) {
	tests := []struct {
		source   string
		expected string
	}{
		{"x-chain", "x-chain"},
		{"c-chain", "c-chain"},
		{"zoo-chain", "zoo-chain"},
		{"a-chain", "a-chain"},
		{"pyth", ""},
		{"chainlink", ""},
		{"unknown", ""},
	}

	for _, tc := range tests {
		got := sourceToChain(tc.source)
		if got != tc.expected {
			t.Errorf("sourceToChain(%q) = %q, want %q", tc.source, got, tc.expected)
		}
	}
}

func TestVerifiedDataFields(t *testing.T) {
	now := time.Now()
	vd := &VerifiedData{
		Data: &Data{
			Symbol:     "LUX-USDC",
			Price:      15.50,
			Timestamp:  now,
			Source:     "x-chain",
			Confidence: 0.95,
		},
		Finalized: true,
		Finality: &QuantumFinality{
			BlockHash:   "abc123",
			BlockHeight: 12345,
			Finalized:   true,
			Latency:     50 * time.Millisecond,
		},
	}

	if vd.Symbol != "LUX-USDC" {
		t.Error("Symbol mismatch")
	}
	if !vd.Finalized {
		t.Error("Finalized should be true")
	}
	if vd.Finality == nil {
		t.Error("Finality should not be nil")
	}
	if vd.Finality.BlockHeight != 12345 {
		t.Error("BlockHeight mismatch")
	}
}

func TestQChainVerifierBasic(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")

	// Start verifier
	if err := v.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer v.Close()

	// Wait for finality data
	time.Sleep(200 * time.Millisecond)

	// Check health
	if !v.Healthy() {
		t.Error("Verifier should be healthy")
	}

	// Check supported chains
	chains := v.Chains()
	if len(chains) == 0 {
		t.Error("Expected supported chains")
	}

	// Check validators
	validators := v.Validators()
	if len(validators) == 0 {
		t.Error("Expected validators")
	}
}

func TestQChainVerifierFinality(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	v.Start()
	defer v.Close()

	// Wait for finality data
	time.Sleep(200 * time.Millisecond)

	// Get finality for x-chain
	fin, err := v.Finality("x-chain")
	if err != nil {
		t.Fatalf("Finality failed: %v", err)
	}

	if fin == nil {
		t.Fatal("Expected finality data")
	}

	if fin.BlockHash == "" {
		t.Error("Expected block hash")
	}
	if fin.Latency == 0 {
		t.Error("Expected latency > 0")
	}
}

func TestQChainVerifierCrossChain(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	v.Start()
	defer v.Close()

	// Wait for finality data
	time.Sleep(200 * time.Millisecond)

	// Cross-chain verify
	chains := []string{"x-chain", "c-chain", "zoo-chain"}
	allFinalized, results := v.CrossChainVerify(chains)

	if !allFinalized {
		t.Error("Expected all chains finalized (simulated)")
	}
	if len(results) != len(chains) {
		t.Errorf("Expected %d results, got %d", len(chains), len(results))
	}
}

func TestQChainVerifierSetQuorum(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")

	// Default quorum
	v.SetQuorum(5)

	// This shouldn't panic
	v.Close()
}

func TestQChainVerifierFinalityLatency(t *testing.T) {
	v := NewQChainVerifier("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
	v.Start()
	defer v.Close()

	// Wait for finality data
	time.Sleep(200 * time.Millisecond)

	latency := v.FinalityLatency()
	if latency == 0 {
		t.Error("Expected non-zero finality latency")
	}
}

func TestQChainBackwardsCompatibility(t *testing.T) {
	// Test that QChainSource alias works
	var _ *QChainSource = NewQChainSource("http://localhost:9650/v1/chain/Q", "ws://localhost:9650/v1/chain/Q/ws")
}
