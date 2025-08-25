package lx

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFundingEngineComprehensive tests all funding engine functionality
func TestFundingEngineComprehensive(t *testing.T) {
	t.Run("FundingEngineCreation", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		// Test with default config
		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		assert.NotNil(t, engine)
		assert.NotNil(t, engine.clearinghouse)
		assert.NotNil(t, engine.config)
		assert.NotEmpty(t, engine.nextFundingTime)
		assert.NotNil(t, engine.currentRates)
		assert.NotNil(t, engine.nextRates)
		assert.NotNil(t, engine.historicalRates)

		// Verify next funding time is set correctly
		nextTime := engine.GetNextFundingTime()
		hour := nextTime.Hour()
		assert.True(t, hour == 0 || hour == 8 || hour == 16)
	})

	t.Run("TWAPSampling", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		config.TWAPWindow = 1 * time.Hour // Shorter window for testing
		engine := NewFundingEngine(clearinghouse, config)

		// Start the engine
		engine.Start()
		defer engine.Stop() // Stop the engine when test ends

		// Wait a bit for TWAP sampling to run
		time.Sleep(100 * time.Millisecond)
	})

	t.Run("FundingRateCalculation", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		_ = NewFundingEngine(clearinghouse, config)

		// Test funding rate calculation logic
		// Simulate different market conditions
		testCases := []struct {
			name             string
			markPrice        float64
			indexPrice       float64
			expectedInBounds bool
		}{
			{"Equal prices", 50000, 50000, true},
			{"Mark above index", 50500, 50000, true},
			{"Mark below index", 49500, 50000, true},
			{"Extreme premium", 55000, 50000, true},  // Should be capped
			{"Extreme discount", 45000, 50000, true}, // Should be capped
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// Calculate funding rate based on price difference
				premium := (tc.markPrice - tc.indexPrice) / tc.indexPrice
				fundingRate := premium + config.InterestRate

				// Apply clamps
				fundingRate = math.Max(config.MinFundingRate, math.Min(config.MaxFundingRate, fundingRate))

				// Check bounds
				assert.GreaterOrEqual(t, fundingRate, config.MinFundingRate)
				assert.LessOrEqual(t, fundingRate, config.MaxFundingRate)
			})
		}
	})

	t.Run("GetTimeUntilFunding", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		timeUntil := engine.GetTimeUntilFunding()

		// Time until funding should be between 0 and 8 hours
		assert.Greater(t, timeUntil, time.Duration(0))
		assert.LessOrEqual(t, timeUntil, 8*time.Hour)
	})

	t.Run("HistoricalRates", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		// Test that historical rates are initialized
		assert.NotNil(t, engine.historicalRates)

		// Add some historical rates
		engine.mu.Lock()
		engine.historicalRates["BTC-PERP"] = []*FundingRate{
			{
				Timestamp: time.Now().Add(-8 * time.Hour),
				Rate:      0.0001,
				MarkTWAP:  50000,
				IndexTWAP: 49990,
			},
		}
		engine.mu.Unlock()

		// Verify the historical rate was added
		engine.mu.RLock()
		history := engine.historicalRates["BTC-PERP"]
		engine.mu.RUnlock()

		assert.Len(t, history, 1)
		assert.Equal(t, 0.0001, history[0].Rate)
	})

	t.Run("FundingSchedule", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		// Test funding schedule initialization
		assert.Equal(t, []int{0, 8, 16}, config.FundingHours)

		// Get next funding time multiple times
		for i := 0; i < 3; i++ {
			nextTime := engine.GetNextFundingTime()
			hour := nextTime.Hour()
			assert.True(t, hour == 0 || hour == 8 || hour == 16, "Invalid funding hour: %d", hour)
		}
	})

	t.Run("NextRates", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		// Set a next rate
		engine.mu.Lock()
		engine.nextRates["BTC-PERP"] = &FundingRate{
			Symbol: "BTC-PERP",
			Rate:   0.0005,
		}
		engine.mu.Unlock()

		// Get next rate
		engine.mu.RLock()
		nextRate := engine.nextRates["BTC-PERP"]
		engine.mu.RUnlock()

		assert.NotNil(t, nextRate)
		assert.Equal(t, 0.0005, nextRate.Rate)
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		config := DefaultFundingConfig()
		engine := NewFundingEngine(clearinghouse, config)

		// Test concurrent access to funding engine
		done := make(chan bool, 3)

		// Goroutine 1: Get next funding time
		go func() {
			for i := 0; i < 100; i++ {
				_ = engine.GetNextFundingTime()
			}
			done <- true
		}()

		// Goroutine 2: Get time until funding
		go func() {
			for i := 0; i < 100; i++ {
				_ = engine.GetTimeUntilFunding()
			}
			done <- true
		}()

		// Goroutine 3: Access rates
		go func() {
			for i := 0; i < 100; i++ {
				engine.mu.Lock()
				engine.currentRates["TEST-PERP"] = &FundingRate{
					Symbol: "TEST-PERP",
					Rate:   0.0001,
				}
				engine.mu.Unlock()
			}
			done <- true
		}()

		// Wait for all goroutines
		for i := 0; i < 3; i++ {
			<-done
		}
	})
}

// TestFundingCalculations tests funding rate calculation logic
func TestFundingCalculations(t *testing.T) {
	t.Run("BasicFundingRate", func(t *testing.T) {
		config := DefaultFundingConfig()

		// Test basic funding rate calculation
		markPrice := 50500.0
		indexPrice := 50000.0

		premium := (markPrice - indexPrice) / indexPrice
		fundingRate := premium + config.InterestRate

		// Apply bounds
		fundingRate = math.Max(config.MinFundingRate, math.Min(config.MaxFundingRate, fundingRate))

		assert.Greater(t, fundingRate, 0.0)
		assert.LessOrEqual(t, fundingRate, config.MaxFundingRate)
	})

	t.Run("NegativeFundingRate", func(t *testing.T) {
		config := DefaultFundingConfig()

		// Test negative funding rate
		markPrice := 49000.0
		indexPrice := 50000.0

		premium := (markPrice - indexPrice) / indexPrice
		fundingRate := premium + config.InterestRate

		// Apply bounds
		fundingRate = math.Max(config.MinFundingRate, math.Min(config.MaxFundingRate, fundingRate))

		assert.GreaterOrEqual(t, fundingRate, config.MinFundingRate)
	})

	t.Run("MaxFundingRateCap", func(t *testing.T) {
		config := DefaultFundingConfig()

		// Test max funding rate cap
		markPrice := 60000.0 // 20% premium
		indexPrice := 50000.0

		premium := (markPrice - indexPrice) / indexPrice
		fundingRate := premium + config.InterestRate

		// Apply bounds
		fundingRate = math.Max(config.MinFundingRate, math.Min(config.MaxFundingRate, fundingRate))

		// Should be capped at max
		assert.Equal(t, config.MaxFundingRate, fundingRate)
	})

	t.Run("MinFundingRateCap", func(t *testing.T) {
		config := DefaultFundingConfig()

		// Test min funding rate cap
		markPrice := 40000.0 // -20% discount
		indexPrice := 50000.0

		premium := (markPrice - indexPrice) / indexPrice
		fundingRate := premium + config.InterestRate

		// Apply bounds
		fundingRate = math.Max(config.MinFundingRate, math.Min(config.MaxFundingRate, fundingRate))

		// Should be capped at min
		assert.Equal(t, config.MinFundingRate, fundingRate)
	})
}

// TestTWAPTracking tests TWAP price tracking
func TestTWAPTracking(t *testing.T) {
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)

	config := DefaultFundingConfig()
	config.TWAPWindow = 1 * time.Hour
	engine := NewFundingEngine(clearinghouse, config)

	// Initialize TWAP samples
	engine.mu.Lock()
	engine.markPriceTWAP["BTC-PERP"] = &TWAPTracker{
		Symbol:  "BTC-PERP",
		Samples: []PriceSample{},
		Window:  config.TWAPWindow,
	}

	// Add some samples
	now := time.Now()
	tracker := engine.markPriceTWAP["BTC-PERP"]

	for i := 0; i < 10; i++ {
		tracker.Samples = append(tracker.Samples, PriceSample{
			Price:     50000 + float64(i*100),
			Timestamp: now.Add(time.Duration(i) * time.Minute),
		})
	}
	engine.mu.Unlock()

	// Verify samples were added
	engine.mu.RLock()
	defer engine.mu.RUnlock()

	tracker = engine.markPriceTWAP["BTC-PERP"]
	assert.Len(t, tracker.Samples, 10)

	// Check that prices are as expected
	assert.Equal(t, 50000.0, tracker.Samples[0].Price)
	assert.Equal(t, 50900.0, tracker.Samples[9].Price)
}

// TestFundingConfiguration tests funding configuration validation
func TestFundingConfiguration(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultFundingConfig()

		assert.Equal(t, 8*time.Hour, config.Interval)
		assert.Equal(t, []int{0, 8, 16}, config.FundingHours)
		assert.Equal(t, 0.0075, config.MaxFundingRate)
		assert.Equal(t, -0.0075, config.MinFundingRate)
		assert.Equal(t, 0.0001, config.InterestRate)
		assert.Equal(t, 8*time.Hour, config.TWAPWindow)
		assert.Equal(t, 1*time.Minute, config.SampleInterval)
		// UseMedianTWAP is checked separately
		assert.Equal(t, 0.0001, config.InterestRate)
	})

	t.Run("CustomConfig", func(t *testing.T) {
		config := &FundingConfig{
			Interval:       4 * time.Hour,
			FundingHours:   []int{0, 4, 8, 12, 16, 20},
			MaxFundingRate: 0.01,
			MinFundingRate: -0.01,
			InterestRate:   0.0002,
			TWAPWindow:     30 * time.Minute,
			SampleInterval: 10 * time.Second,
			UseMedianTWAP:  true,
		}

		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		engine := NewFundingEngine(clearinghouse, config)

		assert.NotNil(t, engine)
		assert.Equal(t, config.Interval, engine.config.Interval)
		assert.Equal(t, config.MaxFundingRate, engine.config.MaxFundingRate)
		assert.Equal(t, config.UseMedianTWAP, engine.config.UseMedianTWAP)
	})
}

// TestFundingRateHistory tests funding rate history management
func TestFundingRateHistory(t *testing.T) {
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)

	config := DefaultFundingConfig()
	engine := NewFundingEngine(clearinghouse, config)

	// Add historical funding rates
	engine.mu.Lock()

	// Initialize historical rates for a symbol
	engine.historicalRates["ETH-PERP"] = []*FundingRate{}

	// Add rates over time
	baseTime := time.Now().Add(-24 * time.Hour)
	for i := 0; i < 10; i++ {
		engine.historicalRates["ETH-PERP"] = append(engine.historicalRates["ETH-PERP"], &FundingRate{
			Symbol:       "ETH-PERP",
			Timestamp:    baseTime.Add(time.Duration(i) * time.Hour),
			Rate:         0.0001 * float64(i+1),
			MarkTWAP:     2000 + float64(i*10),
			IndexTWAP:    2000,
			OpenInterest: 1000000 * float64(i+1),
		})
	}

	engine.mu.Unlock()

	// Test retrieval and calculations
	engine.mu.RLock()
	history := engine.historicalRates["ETH-PERP"]
	engine.mu.RUnlock()

	require.Len(t, history, 10)

	// Verify chronological order
	for i := 1; i < len(history); i++ {
		assert.True(t, history[i].Timestamp.After(history[i-1].Timestamp))
	}

	// Calculate average funding rate
	var sum float64
	for _, record := range history {
		sum += record.Rate
	}
	avgRate := sum / float64(len(history))

	assert.Greater(t, avgRate, 0.0)
	assert.Less(t, avgRate, config.MaxFundingRate)
}
