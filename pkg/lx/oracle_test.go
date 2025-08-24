package lx

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// MockOraclePriceSource is a simple mock implementation of OraclePriceSource
type MockOraclePriceSource struct {
	name      string
	healthy   bool
	weight    float64
	priceData map[string]*PriceData
	err       error
}

func (m *MockOraclePriceSource) GetPrice(symbol string) (*PriceData, error) {
	if m.err != nil {
		return nil, m.err
	}
	if data, ok := m.priceData[symbol]; ok {
		return data, nil
	}
	return nil, errors.New("price not found")
}

func (m *MockOraclePriceSource) GetPrices(symbols []string) (map[string]*PriceData, error) {
	if m.err != nil {
		return nil, m.err
	}
	result := make(map[string]*PriceData)
	for _, symbol := range symbols {
		if data, ok := m.priceData[symbol]; ok {
			result[symbol] = data
		}
	}
	return result, nil
}

func (m *MockOraclePriceSource) Subscribe(symbol string) error {
	return nil
}

func (m *MockOraclePriceSource) Unsubscribe(symbol string) error {
	return nil
}

func (m *MockOraclePriceSource) IsHealthy() bool {
	return m.healthy
}

func (m *MockOraclePriceSource) GetName() string {
	return m.name
}

func (m *MockOraclePriceSource) GetWeight() float64 {
	return m.weight
}

func TestNewPriceOracle(t *testing.T) {
	oracle := NewPriceOracle()
	assert.NotNil(t, oracle)
	assert.NotNil(t, oracle.PriceSources)
	assert.NotNil(t, oracle.CurrentPrices)
	assert.NotNil(t, oracle.PriceHistory)
	assert.NotNil(t, oracle.CircuitBreakers)
	assert.Equal(t, 50*time.Millisecond, oracle.UpdateInterval)
	assert.Equal(t, 2*time.Second, oracle.StaleThreshold)
	assert.Equal(t, 0.05, oracle.DeviationThreshold)
	assert.Equal(t, 2, oracle.MinimumSources)
}

func TestAddSource(t *testing.T) {
	oracle := NewPriceOracle()

	mockSource := &MockOraclePriceSource{
		name:    "test_source",
		healthy: true,
		weight:  1.0,
	}

	// Add source
	err := oracle.AddSource("test", mockSource)
	assert.NoError(t, err)
	assert.Equal(t, mockSource, oracle.PriceSources["test"])

	// Try to add duplicate
	err = oracle.AddSource("test", mockSource)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}

func TestGetPrice(t *testing.T) {
	oracle := NewPriceOracle()

	// Test with no price data
	price := oracle.GetPrice("BTC-USDT")
	assert.Equal(t, float64(0), price)

	// Add price data
	oracle.CurrentPrices["BTC-USDT"] = &PriceData{
		Symbol:    "BTC-USDT",
		Price:     50000.0,
		Timestamp: time.Now(),
	}

	price = oracle.GetPrice("BTC-USDT")
	assert.Equal(t, 50000.0, price)

	// Test stale price with emergency price
	oracle.CurrentPrices["BTC-USDT"].Timestamp = time.Now().Add(-5 * time.Second)
	oracle.EmergencyPrices["BTC-USDT"] = 49000.0

	price = oracle.GetPrice("BTC-USDT")
	assert.Equal(t, 49000.0, price)
}

func TestGetPriceData(t *testing.T) {
	oracle := NewPriceOracle()

	// Test with no data
	_, err := oracle.GetPriceData("BTC-USDT")
	assert.Error(t, err)

	// Add price data
	priceData := &PriceData{
		Symbol:    "BTC-USDT",
		Price:     50000.0,
		Volume:    1000.0,
		Timestamp: time.Now(),
	}
	oracle.CurrentPrices["BTC-USDT"] = priceData

	result, err := oracle.GetPriceData("BTC-USDT")
	assert.NoError(t, err)
	assert.Equal(t, priceData, result)
}

func TestMedianAggregation(t *testing.T) {
	agg := &MedianAggregation{
		MinSources:   2,
		MaxDeviation: 0.1,
	}

	t.Run("Successful aggregation", func(t *testing.T) {
		prices := []*PriceData{
			{Symbol: "BTC", Price: 50000, Volume: 100, Timestamp: time.Now()},
			{Symbol: "BTC", Price: 50100, Volume: 200, Timestamp: time.Now()},
			{Symbol: "BTC", Price: 50050, Volume: 150, Timestamp: time.Now()},
		}

		result, err := agg.Aggregate(prices)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, 50050.0, result.Price) // Median
		assert.Equal(t, 450.0, result.Volume)  // Total volume
	})

	t.Run("Insufficient sources", func(t *testing.T) {
		prices := []*PriceData{
			{Symbol: "BTC", Price: 50000, Volume: 100, Timestamp: time.Now()},
		}

		_, err := agg.Aggregate(prices)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient price sources")
	})

	t.Run("Outlier detection", func(t *testing.T) {
		agg.MaxDeviation = 0.05 // 5% max deviation
		prices := []*PriceData{
			{Symbol: "BTC", Price: 50000, Volume: 100, Timestamp: time.Now()},
			{Symbol: "BTC", Price: 50100, Volume: 200, Timestamp: time.Now()},
			{Symbol: "BTC", Price: 60000, Volume: 150, Timestamp: time.Now()}, // Outlier
		}

		result, err := agg.Aggregate(prices)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		// Median should be calculated without the outlier
	})
}

func TestWeightedAggregation(t *testing.T) {
	agg := &WeightedAggregation{
		SourceWeights: map[string]float64{
			"source1": 2.0,
			"source2": 1.0,
			"source3": 1.5,
		},
		VolumeWeighting: false,
	}

	t.Run("Weighted average", func(t *testing.T) {
		prices := []*PriceData{
			{Symbol: "BTC", Price: 50000, Volume: 100, Source: "source1"},
			{Symbol: "BTC", Price: 50200, Volume: 200, Source: "source2"},
			{Symbol: "BTC", Price: 50100, Volume: 150, Source: "source3"},
		}

		result, err := agg.Aggregate(prices)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		// Weighted: (50000*2 + 50200*1 + 50100*1.5) / (2+1+1.5)
		expectedPrice := (50000*2 + 50200*1 + 50100*1.5) / 4.5
		assert.InDelta(t, expectedPrice, result.Price, 0.01)
	})

	t.Run("No prices", func(t *testing.T) {
		prices := []*PriceData{}
		_, err := agg.Aggregate(prices)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no prices to aggregate")
	})

	t.Run("Volume weighting", func(t *testing.T) {
		agg.VolumeWeighting = true
		prices := []*PriceData{
			{Symbol: "BTC", Price: 50000, Volume: 1000, Source: "source1"},
			{Symbol: "BTC", Price: 50200, Volume: 100, Source: "source2"},
		}

		result, err := agg.Aggregate(prices)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		// Price should be weighted by both source weight and volume
	})
}

func TestPriceCircuitBreaker(t *testing.T) {
	cb := &PriceCircuitBreaker{
		Symbol:            "BTC-USDT",
		MaxChangePercent:  10,
		MaxChangeWindow:   1 * time.Minute,
		AutoResetDuration: 5 * time.Minute,
	}

	t.Run("First price", func(t *testing.T) {
		ok := cb.Check(50000)
		assert.True(t, ok)
		assert.Equal(t, 50000.0, cb.LastValidPrice)
	})

	t.Run("Normal price change", func(t *testing.T) {
		ok := cb.Check(51000) // 2% change
		assert.True(t, ok)
		assert.Equal(t, 51000.0, cb.LastValidPrice)
	})

	t.Run("Excessive price change", func(t *testing.T) {
		ok := cb.Check(60000) // ~18% change
		assert.False(t, ok)
		assert.True(t, cb.Tripped)
		assert.Equal(t, 51000.0, cb.LastValidPrice) // Unchanged
	})

	t.Run("Tripped breaker", func(t *testing.T) {
		ok := cb.Check(52000)
		assert.False(t, ok) // Still tripped
	})

	t.Run("Manual reset", func(t *testing.T) {
		cb.Reset()
		assert.False(t, cb.Tripped)
		ok := cb.Check(52000)
		assert.True(t, ok)
	})

	t.Run("Auto reset", func(t *testing.T) {
		cb.Trip()
		cb.TrippedAt = time.Now().Add(-6 * time.Minute) // Past auto-reset duration
		ok := cb.Check(52000)
		assert.True(t, ok)
		assert.False(t, cb.Tripped)
	})
}

func TestGetTWAP(t *testing.T) {
	oracle := NewPriceOracle()

	// Add price history
	now := time.Now()
	oracle.PriceHistory["BTC-USDT"] = []*PriceData{
		{Price: 50000, Timestamp: now.Add(-10 * time.Minute)},
		{Price: 50100, Timestamp: now.Add(-8 * time.Minute)},
		{Price: 50200, Timestamp: now.Add(-6 * time.Minute)},
		{Price: 50300, Timestamp: now.Add(-4 * time.Minute)},
		{Price: 50400, Timestamp: now.Add(-2 * time.Minute)},
		{Price: 50500, Timestamp: now},
	}

	// Calculate TWAP for 5 minutes
	twap := oracle.GetTWAP("BTC-USDT", 5*time.Minute)
	assert.Greater(t, twap, 50000.0)
	assert.Less(t, twap, 51000.0)

	// Test with no history
	twap = oracle.GetTWAP("ETH-USDT", 5*time.Minute)
	assert.Equal(t, 0.0, twap)
}

func TestGetVWAP(t *testing.T) {
	oracle := NewPriceOracle()

	// Add price history with volume
	now := time.Now()
	oracle.PriceHistory["BTC-USDT"] = []*PriceData{
		{Price: 50000, Volume: 100, Timestamp: now.Add(-4 * time.Minute)},
		{Price: 50100, Volume: 200, Timestamp: now.Add(-2 * time.Minute)},
		{Price: 50200, Volume: 150, Timestamp: now},
	}

	// Calculate VWAP for 5 minutes
	vwap := oracle.GetVWAP("BTC-USDT", 5*time.Minute)
	// VWAP = (50000*100 + 50100*200 + 50200*150) / (100+200+150)
	expectedVWAP := (50000*100 + 50100*200 + 50200*150) / 450.0
	assert.InDelta(t, expectedVWAP, vwap, 0.01)

	// Test with no history
	vwap = oracle.GetVWAP("ETH-USDT", 5*time.Minute)
	assert.Equal(t, 0.0, vwap)

	// Test with zero volume
	oracle.PriceHistory["XRP-USDT"] = []*PriceData{
		{Price: 1.0, Volume: 0, Timestamp: now},
	}
	vwap = oracle.GetVWAP("XRP-USDT", 5*time.Minute)
	assert.Equal(t, 0.0, vwap)
}

func TestOracleStartStop(t *testing.T) {
	oracle := NewPriceOracle()

	// Start oracle
	err := oracle.Start()
	assert.NoError(t, err)
	assert.True(t, oracle.Running)

	// Try to start again
	err = oracle.Start()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already running")

	// Stop oracle
	oracle.Stop()
	assert.False(t, oracle.Running)
}

func TestCalculateHelperFunctions(t *testing.T) {
	t.Run("calculateMedian", func(t *testing.T) {
		// Odd number of values
		median := calculateMedian([]float64{1, 2, 3, 4, 5})
		assert.Equal(t, 3.0, median)

		// Even number of values
		median = calculateMedian([]float64{1, 2, 3, 4})
		assert.Equal(t, 2.5, median)

		// Empty slice
		median = calculateMedian([]float64{})
		assert.Equal(t, 0.0, median)

		// Unsorted values
		median = calculateMedian([]float64{5, 1, 3, 2, 4})
		assert.Equal(t, 3.0, median)
	})

	t.Run("calculateMean", func(t *testing.T) {
		mean := calculateMean([]float64{1, 2, 3, 4, 5})
		assert.Equal(t, 3.0, mean)

		mean = calculateMean([]float64{})
		assert.Equal(t, 0.0, mean)
	})

	t.Run("calculateStdDev", func(t *testing.T) {
		values := []float64{2, 4, 4, 4, 5, 5, 7, 9}
		mean := calculateMean(values)
		stdDev := calculateStdDev(values, mean)
		assert.Greater(t, stdDev, 0.0)

		// All same values
		stdDev = calculateStdDev([]float64{5, 5, 5, 5}, 5)
		assert.Equal(t, 0.0, stdDev)

		// Empty slice
		stdDev = calculateStdDev([]float64{}, 0)
		assert.Equal(t, 0.0, stdDev)
	})

	t.Run("extractPrices", func(t *testing.T) {
		priceData := []*PriceData{
			{Price: 100},
			{Price: 200},
			{Price: 300},
		}
		prices := extractPrices(priceData)
		assert.Equal(t, []float64{100, 200, 300}, prices)

		// Empty input
		prices = extractPrices([]*PriceData{})
		assert.Equal(t, []float64{}, prices)
	})
}

func TestInitCircuitBreakers(t *testing.T) {
	breakers := initCircuitBreakers()
	assert.NotNil(t, breakers)
	assert.Len(t, breakers, 5) // BTC, ETH, BNB, SOL, AVAX

	// Check BTC circuit breaker
	btcBreaker := breakers["BTC-USDT"]
	assert.NotNil(t, btcBreaker)
	assert.Equal(t, "BTC-USDT", btcBreaker.Symbol)
	assert.Equal(t, 20.0, btcBreaker.MaxChangePercent)
	assert.Equal(t, 1*time.Minute, btcBreaker.MaxChangeWindow)
	assert.Equal(t, 5*time.Minute, btcBreaker.AutoResetDuration)
}

func TestNewOracleMetrics(t *testing.T) {
	metrics := NewOracleMetrics()
	assert.NotNil(t, metrics)
	assert.NotNil(t, metrics.SourceHealth)
	assert.Equal(t, uint64(0), metrics.TotalUpdates)
	assert.Equal(t, uint64(0), metrics.FailedUpdates)
}

// TestMedianAggregationHelpers tests the helper methods of MedianAggregation
func TestMedianAggregationHelpers(t *testing.T) {
	agg := &MedianAggregation{
		MinSources:   2,
		MaxDeviation: 0.1,
	}

	t.Run("ValidatePrices", func(t *testing.T) {
		// No prices
		err := agg.ValidatePrices([]*PriceData{})
		assert.Error(t, err)

		// Valid prices
		prices := []*PriceData{
			{Price: 100, Timestamp: time.Now()},
			{Price: 101, Timestamp: time.Now()},
			{Price: 99, Timestamp: time.Now()},
		}
		err = agg.ValidatePrices(prices)
		assert.NoError(t, err)

		// Stale price
		prices[0].Timestamp = time.Now().Add(-60 * time.Second)
		err = agg.ValidatePrices(prices)
		assert.NoError(t, err) // Should mark as stale but not error
		assert.True(t, prices[0].IsStale)

		// High deviation
		prices = []*PriceData{
			{Price: 100, Timestamp: time.Now()},
			{Price: 120, Timestamp: time.Now()}, // 20% deviation
			{Price: 99, Timestamp: time.Now()},
		}
		err = agg.ValidatePrices(prices)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "deviation too high")
	})

	t.Run("FilterOutliers", func(t *testing.T) {
		prices := []*PriceData{
			{Price: 100},
			{Price: 101},
			{Price: 99},
			{Price: 120}, // Outlier
		}

		filtered := agg.filterOutliers(prices, 100)
		assert.Len(t, filtered, 3) // Outlier removed
	})

	t.Run("CalculateConfidence", func(t *testing.T) {
		// High confidence - many sources, low deviation
		prices := []*PriceData{
			{Price: 100},
			{Price: 101},
			{Price: 99},
			{Price: 100},
		}
		confidence := agg.calculateConfidence(prices)
		assert.Greater(t, confidence, 0.5)
		assert.LessOrEqual(t, confidence, 1.0)

		// Low confidence - few sources
		prices = []*PriceData{{Price: 100}}
		confidence = agg.calculateConfidence(prices)
		assert.LessOrEqual(t, confidence, 0.6) // Should have lower confidence with just one source

		// No prices
		confidence = agg.calculateConfidence([]*PriceData{})
		assert.Equal(t, 0.0, confidence)
	})

	t.Run("GetBestBidAsk", func(t *testing.T) {
		prices := []*PriceData{
			{Bid: 99.5, Ask: 100.5},
			{Bid: 99.6, Ask: 100.4},
			{Bid: 99.4, Ask: 100.6},
		}

		bid, ask := agg.getBestBidAsk(prices)
		assert.Equal(t, 99.6, bid)  // Best bid
		assert.Equal(t, 100.4, ask) // Best ask

		// No prices
		bid, ask = agg.getBestBidAsk([]*PriceData{})
		assert.Equal(t, 0.0, bid)
		assert.Equal(t, 0.0, ask)

		// Invalid ask
		prices = []*PriceData{{Bid: 99, Ask: 0}}
		bid, ask = agg.getBestBidAsk(prices)
		assert.Equal(t, 99.0, bid)
		assert.Equal(t, 0.0, ask)
	})
}

// TestPriceUpdateNotifications tests the price update notification system
func TestPriceUpdateNotifications(t *testing.T) {
	oracle := NewPriceOracle()

	// Set up initial price
	oracle.CurrentPrices["BTC-USDT"] = &PriceData{
		Price:     50000,
		Timestamp: time.Now(),
		Source:    "test",
	}

	// Start listening for updates
	updateReceived := false
	go func() {
		select {
		case update := <-oracle.PriceUpdates:
			if update != nil {
				updateReceived = true
				assert.Equal(t, "BTC-USDT", update.Symbol)
				assert.Equal(t, 50000.0, update.OldPrice)
				assert.Equal(t, 51000.0, update.NewPrice)
			}
		case <-time.After(100 * time.Millisecond):
			// Timeout
		}
	}()

	// Update price
	oracle.updateCurrentPrice("BTC-USDT", &PriceData{
		Price:     51000,
		Timestamp: time.Now(),
		Source:    "test",
	})

	time.Sleep(50 * time.Millisecond)
	assert.True(t, updateReceived)
}

// TestPriceAlerts tests the alert system
func TestPriceAlerts(t *testing.T) {
	oracle := NewPriceOracle()

	// Test sending alert
	alert := &PriceAlert{
		Symbol:    "BTC-USDT",
		AlertType: PriceStale,
		Message:   "Price is stale",
		Severity:  WarningAlert,
		Price:     50000,
		Timestamp: time.Now(),
	}

	// Listen for alert
	alertReceived := false
	go func() {
		select {
		case a := <-oracle.AlertChannel:
			if a != nil {
				alertReceived = true
				assert.NotEmpty(t, a.AlertID)
				assert.Equal(t, "BTC-USDT", a.Symbol)
				assert.Equal(t, PriceStale, a.AlertType)
			}
		case <-time.After(100 * time.Millisecond):
			// Timeout
		}
	}()

	oracle.sendAlert(alert)
	time.Sleep(50 * time.Millisecond)
	assert.True(t, alertReceived)
}
