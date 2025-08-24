package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPerpetualBasics tests basic perpetual functionality
func TestPerpetualBasics(t *testing.T) {
	t.Run("PerpetualManager", func(t *testing.T) {
		engineConfig := EngineConfig{
			EnablePerps:   true,
			EnableVaults:  true,
			EnableLending: true,
		}
		engine := NewTradingEngine(engineConfig)
		pm := NewPerpetualManager(engine)

		assert.NotNil(t, pm)
		assert.NotNil(t, pm.engine)
	})

	t.Run("ClearingHouse", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		assert.NotNil(t, clearinghouse)

		// Test deposit
		err := clearinghouse.Deposit("user1", big.NewInt(10000000)) // $10,000
		require.NoError(t, err)

		// Test position opening
		position, err := clearinghouse.OpenPosition("user1", "BTC-PERP", Buy, 1, Market)
		require.NoError(t, err)
		assert.NotNil(t, position)
	})

	t.Run("FundingRate", func(t *testing.T) {
		marginEngine := NewMarginEngine(nil, nil)
		riskEngine := NewRiskEngine()
		clearinghouse := NewClearingHouse(marginEngine, riskEngine)

		// Get funding rate
		fundingRate := clearinghouse.GetFundingRate("BTC-PERP")

		// Should be within bounds (-0.75% to +0.75% per 8 hours)
		assert.GreaterOrEqual(t, fundingRate, -0.0075)
		assert.LessOrEqual(t, fundingRate, 0.0075)

		// Get next funding time
		nextTime := clearinghouse.GetNextFundingTime()
		assert.NotZero(t, nextTime)

		// Should be one of the standard times
		hour := nextTime.Hour()
		assert.True(t, hour == 0 || hour == 8 || hour == 16)
	})
}

// TestFundingEngine tests the funding engine directly
func TestFundingEngine(t *testing.T) {
	marginEngine := NewMarginEngine(nil, nil)
	riskEngine := NewRiskEngine()
	clearinghouse := NewClearingHouse(marginEngine, riskEngine)

	fundingConfig := DefaultFundingConfig()
	fundingEngine := NewFundingEngine(clearinghouse, fundingConfig)

	t.Run("Configuration", func(t *testing.T) {
		assert.Equal(t, 8*time.Hour, fundingConfig.Interval)
		assert.Equal(t, []int{0, 8, 16}, fundingConfig.FundingHours)
		assert.Equal(t, 0.0075, fundingConfig.MaxFundingRate)
		assert.Equal(t, -0.0075, fundingConfig.MinFundingRate)
		assert.Equal(t, 0.0001, fundingConfig.InterestRate) // 0.01% per 8 hours
	})

	t.Run("NextFundingTime", func(t *testing.T) {
		nextTime := fundingEngine.GetNextFundingTime()
		timeUntil := fundingEngine.GetTimeUntilFunding()

		assert.NotZero(t, nextTime)
		assert.Greater(t, timeUntil, time.Duration(0))
		assert.LessOrEqual(t, timeUntil, 8*time.Hour)
	})
}
