package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestModifyLeverage_Comprehensive tests ModifyLeverage with comprehensive coverage
func TestModifyLeverage_Comprehensive(t *testing.T) {
	// Helper to create a margin engine with position already created (bypassing borrowing)
	setupEngineWithPosition := func(leverage float64) (*MarginEngine, *MarginAccount, *MarginPosition) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		account, _ := engine.CreateMarginAccount("user1", CrossMargin)

		// Initialize account with sufficient margin
		account.Balance = big.NewInt(100000000)    // 100M units
		account.FreeMargin = big.NewInt(100000000) // 100M units free
		account.MarginUsed = big.NewInt(0)

		// Create position directly (bypassing OpenPosition which needs lending pool)
		positionValue := 50000.0 * 0.1 // EntryPrice * Size = 5000
		requiredMargin := int64(positionValue / leverage)

		position := &MarginPosition{
			ID:              "test_pos_1",
			Symbol:          "BTC-USDT",
			Side:            Buy,
			Size:            0.1,
			EntryPrice:      50000.0,
			MarkPrice:       50000.0,
			Leverage:        leverage,
			Margin:          big.NewInt(requiredMargin),
			UnrealizedPnL:   big.NewInt(0),
			RealizedPnL:     big.NewInt(0),
			Fees:            big.NewInt(0),
			OpenTime:        time.Now(),
			LastUpdate:      time.Now(),
			Isolated:        false,
			CollateralAsset: "USDT",
			FundingPaid:     big.NewInt(0),
		}

		account.Positions[position.ID] = position
		account.MarginUsed = new(big.Int).Set(position.Margin)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

		return engine, account, position
	}

	t.Run("AccountNotFound", func(t *testing.T) {
		oracle := NewPriceOracle()
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		err := engine.ModifyLeverage("nonexistent_user", "pos1", 10.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "account not found")
	})

	t.Run("PositionNotFound", func(t *testing.T) {
		oracle := NewPriceOracle()
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		engine.CreateMarginAccount("user1", CrossMargin)

		err := engine.ModifyLeverage("user1", "nonexistent_pos", 10.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "position not found")
	})

	t.Run("ExceedsMaxLeverage", func(t *testing.T) {
		engine, _, _ := setupEngineWithPosition(10.0)

		// Try to set leverage above max (100 for BTC-USDT in CrossMargin)
		err := engine.ModifyLeverage("user1", "test_pos_1", 150.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "leverage exceeds maximum")
	})

	t.Run("IncreaseLeverage_Success", func(t *testing.T) {
		engine, account, position := setupEngineWithPosition(5.0)

		// Initial state
		initialMargin := new(big.Int).Set(position.Margin)
		initialFreeMargin := new(big.Int).Set(account.FreeMargin)
		initialMarginUsed := new(big.Int).Set(account.MarginUsed)

		// Increase leverage from 5x to 10x (requires less margin)
		err := engine.ModifyLeverage("user1", "test_pos_1", 10.0)
		require.NoError(t, err)

		// Verify position leverage updated
		assert.Equal(t, 10.0, position.Leverage)

		// Margin should decrease (higher leverage = less margin needed)
		assert.True(t, position.Margin.Cmp(initialMargin) < 0, "Margin should decrease with higher leverage")

		// Free margin should increase (margin released)
		assert.True(t, account.FreeMargin.Cmp(initialFreeMargin) > 0, "Free margin should increase")

		// Margin used should decrease
		assert.True(t, account.MarginUsed.Cmp(initialMarginUsed) < 0, "Margin used should decrease")

		// Verify liquidation price was recalculated
		assert.NotEqual(t, 0.0, position.LiquidationPrice)
	})

	t.Run("DecreaseLeverage_Success", func(t *testing.T) {
		engine, account, position := setupEngineWithPosition(10.0)

		// Initial state
		initialMargin := new(big.Int).Set(position.Margin)
		initialFreeMargin := new(big.Int).Set(account.FreeMargin)
		initialMarginUsed := new(big.Int).Set(account.MarginUsed)

		// Decrease leverage from 10x to 5x (requires more margin)
		err := engine.ModifyLeverage("user1", "test_pos_1", 5.0)
		require.NoError(t, err)

		// Verify position leverage updated
		assert.Equal(t, 5.0, position.Leverage)

		// Margin should increase (lower leverage = more margin needed)
		assert.True(t, position.Margin.Cmp(initialMargin) > 0, "Margin should increase with lower leverage")

		// Free margin should decrease (margin locked)
		assert.True(t, account.FreeMargin.Cmp(initialFreeMargin) < 0, "Free margin should decrease")

		// Margin used should increase
		assert.True(t, account.MarginUsed.Cmp(initialMarginUsed) > 0, "Margin used should increase")
	})

	t.Run("DecreaseLeverage_InsufficientMargin", func(t *testing.T) {
		engine, account, position := setupEngineWithPosition(50.0)

		// Set free margin to a very small amount
		account.FreeMargin = big.NewInt(10)

		originalLeverage := position.Leverage
		originalMargin := new(big.Int).Set(position.Margin)

		// Try to decrease leverage significantly (would require much more margin)
		err := engine.ModifyLeverage("user1", "test_pos_1", 1.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient free margin")

		// Verify position was not modified
		assert.Equal(t, originalLeverage, position.Leverage)
		assert.Equal(t, 0, position.Margin.Cmp(originalMargin))
	})

	t.Run("SameLeverage_NoChange", func(t *testing.T) {
		engine, account, position := setupEngineWithPosition(10.0)

		initialMargin := new(big.Int).Set(position.Margin)
		initialFreeMargin := new(big.Int).Set(account.FreeMargin)

		// Set to same leverage
		err := engine.ModifyLeverage("user1", "test_pos_1", 10.0)
		require.NoError(t, err)

		// Values should remain the same (allowing for small rounding)
		assert.Equal(t, 10.0, position.Leverage)
		assert.Equal(t, 0, position.Margin.Cmp(initialMargin))
		assert.Equal(t, 0, account.FreeMargin.Cmp(initialFreeMargin))
	})

	t.Run("VerifyMarginCalculation", func(t *testing.T) {
		engine, _, position := setupEngineWithPosition(10.0)

		// Verify initial margin is correct: positionValue / leverage = 5000 / 10 = 500
		positionValue := position.EntryPrice * position.Size       // 50000 * 0.1 = 5000
		expectedMargin := int64(positionValue / position.Leverage) // 5000 / 10 = 500
		assert.Equal(t, expectedMargin, position.Margin.Int64())

		// Change to 20x leverage
		err := engine.ModifyLeverage("user1", "test_pos_1", 20.0)
		require.NoError(t, err)

		// Verify new margin: 5000 / 20 = 250
		newExpectedMargin := int64(positionValue / 20.0)
		assert.Equal(t, newExpectedMargin, position.Margin.Int64())
	})

	t.Run("LastUpdateTimestamp", func(t *testing.T) {
		engine, _, position := setupEngineWithPosition(10.0)

		initialTime := position.LastUpdate
		time.Sleep(time.Millisecond * 10) // Small delay

		err := engine.ModifyLeverage("user1", "test_pos_1", 5.0)
		require.NoError(t, err)

		assert.True(t, position.LastUpdate.After(initialTime), "LastUpdate should be updated")
	})

	t.Run("IsolatedMarginAccount", func(t *testing.T) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		// Create isolated margin account (higher max leverage)
		account, _ := engine.CreateMarginAccount("user1", IsolatedMargin)
		account.Balance = big.NewInt(100000000)
		account.FreeMargin = big.NewInt(100000000)

		position := &MarginPosition{
			ID:              "test_pos_1",
			Symbol:          "BTC-USDT",
			Side:            Buy,
			Size:            0.1,
			EntryPrice:      50000.0,
			MarkPrice:       50000.0,
			Leverage:        10.0,
			Margin:          big.NewInt(500),
			UnrealizedPnL:   big.NewInt(0),
			RealizedPnL:     big.NewInt(0),
			Fees:            big.NewInt(0),
			OpenTime:        time.Now(),
			LastUpdate:      time.Now(),
			Isolated:        true,
			CollateralAsset: "USDT",
			FundingPaid:     big.NewInt(0),
		}
		account.Positions[position.ID] = position
		account.MarginUsed = new(big.Int).Set(position.Margin)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

		// IsolatedMargin should allow up to 100x (same as BTC-USDT table)
		err := engine.ModifyLeverage("user1", "test_pos_1", 50.0)
		require.NoError(t, err)
		assert.Equal(t, 50.0, position.Leverage)
	})

	t.Run("PortfolioMarginAccount", func(t *testing.T) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		// Create portfolio margin account (doubled max leverage)
		account, _ := engine.CreateMarginAccount("user1", PortfolioMargin)
		account.Balance = big.NewInt(100000000)
		account.FreeMargin = big.NewInt(100000000)

		position := &MarginPosition{
			ID:              "test_pos_1",
			Symbol:          "BTC-USDT",
			Side:            Buy,
			Size:            0.1,
			EntryPrice:      50000.0,
			MarkPrice:       50000.0,
			Leverage:        10.0,
			Margin:          big.NewInt(500),
			UnrealizedPnL:   big.NewInt(0),
			RealizedPnL:     big.NewInt(0),
			Fees:            big.NewInt(0),
			OpenTime:        time.Now(),
			LastUpdate:      time.Now(),
			Isolated:        false,
			CollateralAsset: "USDT",
			FundingPaid:     big.NewInt(0),
		}
		account.Positions[position.ID] = position
		account.MarginUsed = new(big.Int).Set(position.Margin)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

		// PortfolioMargin allows 2x the normal max (200x for BTC-USDT)
		err := engine.ModifyLeverage("user1", "test_pos_1", 150.0)
		require.NoError(t, err)
		assert.Equal(t, 150.0, position.Leverage)

		// But should fail above 200x
		err = engine.ModifyLeverage("user1", "test_pos_1", 250.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "leverage exceeds maximum")
	})

	t.Run("ShortPositionLeverageModification", func(t *testing.T) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		account, _ := engine.CreateMarginAccount("user1", CrossMargin)
		account.Balance = big.NewInt(100000000)
		account.FreeMargin = big.NewInt(100000000)

		// Create short position
		position := &MarginPosition{
			ID:              "test_pos_1",
			Symbol:          "BTC-USDT",
			Side:            Sell, // Short position
			Size:            0.1,
			EntryPrice:      50000.0,
			MarkPrice:       50000.0,
			Leverage:        10.0,
			Margin:          big.NewInt(500),
			UnrealizedPnL:   big.NewInt(0),
			RealizedPnL:     big.NewInt(0),
			Fees:            big.NewInt(0),
			OpenTime:        time.Now(),
			LastUpdate:      time.Now(),
			Isolated:        false,
			CollateralAsset: "USDT",
			FundingPaid:     big.NewInt(0),
		}
		account.Positions[position.ID] = position
		account.MarginUsed = new(big.Int).Set(position.Margin)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

		// Modify leverage on short position
		err := engine.ModifyLeverage("user1", "test_pos_1", 20.0)
		require.NoError(t, err)
		assert.Equal(t, 20.0, position.Leverage)

		// Verify liquidation price was calculated for short position
		// For short, liquidation price should be above entry price
		assert.Greater(t, position.LiquidationPrice, position.EntryPrice)
	})

	t.Run("UnknownSymbolDefaultLeverage", func(t *testing.T) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["UNKNOWN-PAIR"] = &PriceData{Price: 100.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		account, _ := engine.CreateMarginAccount("user1", CrossMargin)
		account.Balance = big.NewInt(100000000)
		account.FreeMargin = big.NewInt(100000000)

		// Create position with unknown symbol (default max leverage is 10)
		position := &MarginPosition{
			ID:              "test_pos_1",
			Symbol:          "UNKNOWN-PAIR",
			Side:            Buy,
			Size:            1.0,
			EntryPrice:      100.0,
			MarkPrice:       100.0,
			Leverage:        5.0,
			Margin:          big.NewInt(20),
			UnrealizedPnL:   big.NewInt(0),
			RealizedPnL:     big.NewInt(0),
			Fees:            big.NewInt(0),
			OpenTime:        time.Now(),
			LastUpdate:      time.Now(),
			Isolated:        false,
			CollateralAsset: "USDT",
			FundingPaid:     big.NewInt(0),
		}
		account.Positions[position.ID] = position
		account.MarginUsed = new(big.Int).Set(position.Margin)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

		// Should succeed at 10x (default max)
		err := engine.ModifyLeverage("user1", "test_pos_1", 10.0)
		require.NoError(t, err)

		// Should fail above 10x (default max)
		err = engine.ModifyLeverage("user1", "test_pos_1", 15.0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "leverage exceeds maximum")
	})

	t.Run("MultiplePositions", func(t *testing.T) {
		oracle := NewPriceOracle()
		oracle.CurrentPrices = make(map[string]*PriceData)
		oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
		oracle.CurrentPrices["ETH-USDT"] = &PriceData{Price: 3000.0}
		riskEngine := NewRiskEngine()
		engine := NewMarginEngine(oracle, riskEngine)

		account, _ := engine.CreateMarginAccount("user1", CrossMargin)
		account.Balance = big.NewInt(100000000)
		account.FreeMargin = big.NewInt(100000000)

		// Create two positions
		pos1 := &MarginPosition{
			ID:            "pos1",
			Symbol:        "BTC-USDT",
			Side:          Buy,
			Size:          0.1,
			EntryPrice:    50000.0,
			Leverage:      10.0,
			Margin:        big.NewInt(500),
			UnrealizedPnL: big.NewInt(0),
			RealizedPnL:   big.NewInt(0),
			Fees:          big.NewInt(0),
			OpenTime:      time.Now(),
			LastUpdate:    time.Now(),
			FundingPaid:   big.NewInt(0),
		}
		pos2 := &MarginPosition{
			ID:            "pos2",
			Symbol:        "ETH-USDT",
			Side:          Sell,
			Size:          1.0,
			EntryPrice:    3000.0,
			Leverage:      10.0,
			Margin:        big.NewInt(300),
			UnrealizedPnL: big.NewInt(0),
			RealizedPnL:   big.NewInt(0),
			Fees:          big.NewInt(0),
			OpenTime:      time.Now(),
			LastUpdate:    time.Now(),
			FundingPaid:   big.NewInt(0),
		}
		account.Positions["pos1"] = pos1
		account.Positions["pos2"] = pos2
		account.MarginUsed = big.NewInt(800)
		account.FreeMargin = new(big.Int).Sub(account.FreeMargin, big.NewInt(800))

		// Modify only one position
		err := engine.ModifyLeverage("user1", "pos1", 20.0)
		require.NoError(t, err)

		// Verify only pos1 was modified
		assert.Equal(t, 20.0, pos1.Leverage)
		assert.Equal(t, 10.0, pos2.Leverage) // pos2 unchanged
	})

	t.Run("EdgeCase_MinimumLeverage", func(t *testing.T) {
		engine, _, position := setupEngineWithPosition(10.0)

		// Set to 1x leverage (minimum possible)
		err := engine.ModifyLeverage("user1", "test_pos_1", 1.0)
		require.NoError(t, err)

		assert.Equal(t, 1.0, position.Leverage)

		// Margin should equal full position value at 1x
		positionValue := int64(position.EntryPrice * position.Size) // 5000
		assert.Equal(t, positionValue, position.Margin.Int64())
	})

	t.Run("EdgeCase_VerySmallLeverageChange", func(t *testing.T) {
		engine, _, position := setupEngineWithPosition(10.0)

		// Small leverage change
		err := engine.ModifyLeverage("user1", "test_pos_1", 10.1)
		require.NoError(t, err)

		assert.Equal(t, 10.1, position.Leverage)
	})
}

// TestModifyLeverage_ConcurrentAccess tests thread safety
func TestModifyLeverage_ConcurrentAccess(t *testing.T) {
	oracle := NewPriceOracle()
	oracle.CurrentPrices = make(map[string]*PriceData)
	oracle.CurrentPrices["BTC-USDT"] = &PriceData{Price: 50000.0}
	riskEngine := NewRiskEngine()
	engine := NewMarginEngine(oracle, riskEngine)

	account, _ := engine.CreateMarginAccount("user1", CrossMargin)
	account.Balance = big.NewInt(1000000000)
	account.FreeMargin = big.NewInt(1000000000)

	position := &MarginPosition{
		ID:            "test_pos_1",
		Symbol:        "BTC-USDT",
		Side:          Buy,
		Size:          0.1,
		EntryPrice:    50000.0,
		MarkPrice:     50000.0,
		Leverage:      10.0,
		Margin:        big.NewInt(500),
		UnrealizedPnL: big.NewInt(0),
		RealizedPnL:   big.NewInt(0),
		Fees:          big.NewInt(0),
		OpenTime:      time.Now(),
		LastUpdate:    time.Now(),
		FundingPaid:   big.NewInt(0),
	}
	account.Positions[position.ID] = position
	account.MarginUsed = new(big.Int).Set(position.Margin)
	account.FreeMargin = new(big.Int).Sub(account.FreeMargin, position.Margin)

	// Run concurrent modifications
	done := make(chan bool)
	leverages := []float64{5.0, 10.0, 15.0, 20.0, 25.0}

	for _, lev := range leverages {
		go func(l float64) {
			_ = engine.ModifyLeverage("user1", "test_pos_1", l)
			done <- true
		}(lev)
	}

	// Wait for all goroutines
	for range leverages {
		<-done
	}

	// Verify position is in a valid state (leverage should be one of the values)
	validLeverages := map[float64]bool{5.0: true, 10.0: true, 15.0: true, 20.0: true, 25.0: true}
	assert.True(t, validLeverages[position.Leverage], "Position leverage should be one of the set values")
}
