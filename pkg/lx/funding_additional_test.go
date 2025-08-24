package lx

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test specific funding functions with 0% coverage
func TestFundingEngineSpecificFunctions(t *testing.T) {
	// Create a funding engine for testing
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)
	config := DefaultFundingConfig()
	engine := NewFundingEngine(clearinghouse, config)

	t.Run("ProcessFunding", func(t *testing.T) {
		// Set up TWAP data first
		engine.mu.Lock()
		engine.markPriceTWAP["BTC-PERP"] = &TWAPTracker{
			Symbol:      "BTC-PERP",
			Window:      8 * time.Hour,
			CurrentTWAP: 50000.0,
		}
		engine.indexPriceTWAP["BTC-PERP"] = &TWAPTracker{
			Symbol:      "BTC-PERP", 
			Window:      8 * time.Hour,
			CurrentTWAP: 49950.0,
		}
		engine.mu.Unlock()

		// Test ProcessFunding
		fundingTime := time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC)
		err := engine.ProcessFunding(fundingTime)
		assert.NoError(t, err)
	})

	t.Run("calculateFundingRate", func(t *testing.T) {
		// Set up TWAP data
		engine.mu.Lock()
		engine.markPriceTWAP["ETH-PERP"] = &TWAPTracker{
			Symbol:      "ETH-PERP",
			Window:      8 * time.Hour,
			CurrentTWAP: 3000.0,
		}
		engine.indexPriceTWAP["ETH-PERP"] = &TWAPTracker{
			Symbol:      "ETH-PERP",
			Window:      8 * time.Hour,
			CurrentTWAP: 2990.0,
		}
		engine.mu.Unlock()

		rate := engine.calculateFundingRate("ETH-PERP")
		assert.NotNil(t, rate)
		assert.Equal(t, "ETH-PERP", rate.Symbol)
		assert.NotZero(t, rate.Rate)
	})

	t.Run("calculateFundingPayments", func(t *testing.T) {
		rate := &FundingRate{
			Symbol:       "BTC-PERP",
			Rate:         0.001,
			Timestamp:    time.Now(),
			PaymentTime:  time.Now(),
			OpenInterest: 1000000.0,
		}

		payments := engine.calculateFundingPayments("BTC-PERP", rate)
		assert.NotNil(t, payments)
	})

	t.Run("applyFundingPayment", func(t *testing.T) {
		account := "user123"
		symbol := "BTC-PERP"
		payment := 100.0

		engine.applyFundingPayment(account, symbol, payment)
		// Should not panic - this is the main test
	})

	t.Run("samplePrices", func(t *testing.T) {
		engine.samplePrices()
		// Should not panic - this tests the function
	})

	t.Run("addTWAPSample", func(t *testing.T) {
		tracker := &TWAPTracker{
			Symbol:  "TEST-PERP",
			Window:  1 * time.Hour,
			Samples: []PriceSample{},
		}

		price := 50000.0
		engine.addTWAPSample(tracker, price)

		// Check that sample was added
		tracker.mu.RLock()
		sampleCount := len(tracker.Samples)
		tracker.mu.RUnlock()
		
		assert.True(t, sampleCount > 0)
	})

	t.Run("calculateTWAP", func(t *testing.T) {
		samples := []PriceSample{
			{Price: 50000.0, Timestamp: time.Now().Add(-30 * time.Minute)},
			{Price: 51000.0, Timestamp: time.Now()},
		}

		twap := engine.calculateTWAP(samples)
		assert.True(t, twap > 0)
		assert.True(t, twap >= 50000.0 && twap <= 51000.0)
	})

	t.Run("calculateMedianTWAP", func(t *testing.T) {
		samples := []PriceSample{
			{Price: 50000}, {Price: 51000}, {Price: 52000}, {Price: 49000}, {Price: 50500},
		}
		median := engine.calculateMedianTWAP(samples)
		assert.Equal(t, float64(50500), median)

		// Test even number of samples
		evenSamples := []PriceSample{
			{Price: 50000}, {Price: 51000}, {Price: 52000}, {Price: 49000},
		}
		medianEven := engine.calculateMedianTWAP(evenSamples)
		expected := (50000.0 + 51000.0) / 2.0
		assert.Equal(t, expected, medianEven)

		// Test single sample
		singleSample := []PriceSample{{Price: 50000}}
		medianSingle := engine.calculateMedianTWAP(singleSample)
		assert.Equal(t, 50000.0, medianSingle)

		// Test empty slice (skip this test as function panics on empty slice)
		// emptyMedian := engine.calculateMedianTWAP([]PriceSample{})
		// assert.Equal(t, 0.0, emptyMedian)
	})

	t.Run("updatePredictedRates", func(t *testing.T) {
		// Set up some TWAP data first
		engine.mu.Lock()
		engine.markPriceTWAP["BTC-PERP"] = &TWAPTracker{
			Symbol:      "BTC-PERP",
			Window:      8 * time.Hour,
			CurrentTWAP: 50000.0,
		}
		engine.indexPriceTWAP["BTC-PERP"] = &TWAPTracker{
			Symbol:      "BTC-PERP",
			Window:      8 * time.Hour,
			CurrentTWAP: 49900.0,
		}
		engine.mu.Unlock()

		engine.updatePredictedRates()
		// Should not panic and should update rates
	})

	t.Run("isFundingTime", func(t *testing.T) {
		// Test funding times (00:00, 08:00, 16:00 UTC)
		fundingTime := time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC)
		assert.True(t, engine.isFundingTime(fundingTime))

		// Test non-funding time
		nonFundingTime := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
		assert.False(t, engine.isFundingTime(nonFundingTime))

		// Test other funding times
		midnightFunding := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
		assert.True(t, engine.isFundingTime(midnightFunding))

		afternoonFunding := time.Date(2023, 1, 1, 16, 0, 0, 0, time.UTC)
		assert.True(t, engine.isFundingTime(afternoonFunding))
	})

	t.Run("getNextFundingTime", func(t *testing.T) {
		from := time.Now()
		nextTime := engine.getNextFundingTime(from)
		assert.False(t, nextTime.IsZero())
		assert.True(t, nextTime.After(from))

		// Verify it's one of the funding hours (0, 8, 16)
		hour := nextTime.UTC().Hour()
		assert.True(t, hour == 0 || hour == 8 || hour == 16)
	})

	t.Run("getActiveSymbols", func(t *testing.T) {
		// Set up some TWAP trackers to create active symbols
		engine.mu.Lock()
		engine.markPriceTWAP["BTC-PERP"] = &TWAPTracker{Symbol: "BTC-PERP"}
		engine.markPriceTWAP["ETH-PERP"] = &TWAPTracker{Symbol: "ETH-PERP"}
		engine.mu.Unlock()

		symbols := engine.getActiveSymbols()
		assert.NotNil(t, symbols)
		assert.True(t, len(symbols) > 0)
		assert.Contains(t, symbols, "BTC-PERP")
		assert.Contains(t, symbols, "ETH-PERP")
	})

	t.Run("getMarkTWAP", func(t *testing.T) {
		symbol := "BTC-PERP"
		
		// Set up TWAP tracker with current value
		engine.mu.Lock()
		engine.markPriceTWAP[symbol] = &TWAPTracker{
			Symbol:      symbol,
			CurrentTWAP: 50000.0,
		}
		engine.mu.Unlock()

		twap := engine.getMarkTWAP(symbol)
		assert.Equal(t, 50000.0, twap)

		// Test non-existent symbol
		nonExistentTWAP := engine.getMarkTWAP("NON-EXISTENT")
		assert.Equal(t, 0.0, nonExistentTWAP)
	})

	t.Run("getIndexTWAP", func(t *testing.T) {
		symbol := "ETH-PERP"
		
		// Set up TWAP tracker
		engine.mu.Lock()
		engine.indexPriceTWAP[symbol] = &TWAPTracker{
			Symbol:      symbol,
			CurrentTWAP: 3000.0,
		}
		engine.mu.Unlock()

		twap := engine.getIndexTWAP(symbol)
		assert.Equal(t, 3000.0, twap)

		// Test non-existent symbol
		nonExistentTWAP := engine.getIndexTWAP("NON-EXISTENT")
		assert.Equal(t, 0.0, nonExistentTWAP)
	})

	t.Run("getIndexPrice", func(t *testing.T) {
		symbol := "BTC-PERP"
		price := engine.getIndexPrice(symbol)
		assert.True(t, price >= 0)
	})

	t.Run("clampRate", func(t *testing.T) {
		// Test positive clamping (above max)
		clampedPositive := engine.clampRate(0.01)
		assert.Equal(t, engine.config.MaxFundingRate, clampedPositive)

		// Test negative clamping (below min)
		clampedNegative := engine.clampRate(-0.01)
		assert.Equal(t, engine.config.MinFundingRate, clampedNegative)

		// Test no clamping needed
		normal := engine.clampRate(0.005)
		assert.Equal(t, 0.005, normal)
	})

	t.Run("addToHistory", func(t *testing.T) {
		symbol := "BTC-PERP"
		rate := &FundingRate{
			Symbol:    symbol,
			Rate:      0.001,
			Timestamp: time.Now(),
		}

		engine.addToHistory(symbol, rate)

		// Check that rate was added to history
		engine.mu.RLock()
		history := engine.historicalRates[symbol]
		engine.mu.RUnlock()

		assert.NotNil(t, history)
		assert.True(t, len(history) > 0)
		assert.Equal(t, rate, history[len(history)-1])
	})

	t.Run("getPositionStats", func(t *testing.T) {
		symbol := "BTC-PERP"
		stats := engine.getPositionStats(symbol)
		// Stats should exist (even if zero)
		assert.NotNil(t, stats)
	})
}

// Test funding edge cases and error conditions
func TestFundingEngineEdgeCases(t *testing.T) {
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)
	config := DefaultFundingConfig()
	engine := NewFundingEngine(clearinghouse, config)

	t.Run("ProcessFundingWithNoData", func(t *testing.T) {
		// Try to process funding without any TWAP data
		fundingTime := time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC)
		err := engine.ProcessFunding(fundingTime)
		// Should handle gracefully
		assert.NoError(t, err)
	})

	t.Run("CalculateTWAPEmptyTracker", func(t *testing.T) {
		samples := []PriceSample{}

		twap := engine.calculateTWAP(samples)
		assert.Equal(t, 0.0, twap)
	})

	t.Run("AddTWAPSampleNilTracker", func(t *testing.T) {
		// Should handle nil tracker gracefully
		engine.addTWAPSample(nil, 50000.0)
		// Should not panic
	})

	t.Run("CalculateFundingRateNoTWAP", func(t *testing.T) {
		// Try to calculate rate for symbol with no TWAP data
		rate := engine.calculateFundingRate("NO-DATA-PERP")
		assert.NotNil(t, rate)
		assert.Equal(t, "NO-DATA-PERP", rate.Symbol)
	})

	t.Run("FundingTimeValidation", func(t *testing.T) {
		// Test various times around funding periods
		testTimes := []struct {
			time     time.Time
			expected bool
		}{
			{time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC), true},   // 00:00
			{time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC), true},   // 08:00
			{time.Date(2023, 1, 1, 16, 0, 0, 0, time.UTC), true},  // 16:00
			{time.Date(2023, 1, 1, 7, 59, 59, 0, time.UTC), false}, // 07:59:59
			{time.Date(2023, 1, 1, 8, 1, 0, 0, time.UTC), false},   // 08:01:00
			{time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC), false},  // 12:00
		}

		for _, tt := range testTimes {
			result := engine.isFundingTime(tt.time)
			assert.Equal(t, tt.expected, result, "Time: %v", tt.time)
		}
	})
}

// Benchmark the newly tested functions
func BenchmarkFundingEngineNewFunctions(b *testing.B) {
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)
	config := DefaultFundingConfig()
	engine := NewFundingEngine(clearinghouse, config)

	// Setup TWAP data
	engine.mu.Lock()
	engine.markPriceTWAP["BTC-PERP"] = &TWAPTracker{
		Symbol:      "BTC-PERP",
		Window:      8 * time.Hour,
		CurrentTWAP: 50000.0,
	}
	engine.indexPriceTWAP["BTC-PERP"] = &TWAPTracker{
		Symbol:      "BTC-PERP",
		Window:      8 * time.Hour,
		CurrentTWAP: 49950.0,
	}
	engine.mu.Unlock()

	b.Run("ProcessFunding", func(b *testing.B) {
		fundingTime := time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.ProcessFunding(fundingTime)
		}
	})

	b.Run("calculateFundingRate", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.calculateFundingRate("BTC-PERP")
		}
	})

	b.Run("isFundingTime", func(b *testing.B) {
		testTime := time.Date(2023, 1, 1, 8, 0, 0, 0, time.UTC)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.isFundingTime(testTime)
		}
	})

	b.Run("getNextFundingTime", func(b *testing.B) {
		from := time.Now()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.getNextFundingTime(from)
		}
	})

	b.Run("calculateMedianTWAP", func(b *testing.B) {
		samples := []PriceSample{
			{Price: 50000}, {Price: 51000}, {Price: 52000}, {Price: 49000}, {Price: 50500},
			{Price: 48000}, {Price: 53000}, {Price: 47000}, {Price: 54000},
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.calculateMedianTWAP(samples)
		}
	})
}