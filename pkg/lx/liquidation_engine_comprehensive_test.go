package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test liquidation engine functions with 0% coverage
func TestLiquidationEngineFunctions(t *testing.T) {
	t.Run("NewLiquidationEngine", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		assert.NotNil(t, engine)
		assert.NotNil(t, engine.InsuranceFund)
		assert.NotNil(t, engine.LiquidationQueue)
		assert.NotNil(t, engine.AutoDeleveraging)
		assert.NotNil(t, engine.SocializedLoss)
		assert.NotNil(t, engine.MaintenanceMargin)
		assert.NotNil(t, engine.Liquidators)
		assert.NotNil(t, engine.LiquidatorRewards)
		assert.NotNil(t, engine.LiquidationMetrics)
		assert.NotNil(t, engine.LiquidationHistory)
		assert.NotNil(t, engine.RiskEngine)
		assert.NotNil(t, engine.CircuitBreaker)
		
		// Check default values
		assert.Equal(t, 0.005, engine.LiquidationFee)
		assert.Equal(t, 0.5, engine.InsuranceFundFee)
		assert.Equal(t, big.NewInt(0), engine.TotalLiquidated)
		assert.Equal(t, big.NewInt(0), engine.TotalLosses)
		assert.Equal(t, big.NewInt(0), engine.InsuranceFundUsed)
		assert.False(t, engine.EmergencyMode)
		
		// Check maintenance margins were initialized
		assert.True(t, len(engine.MaintenanceMargin) > 0)
		assert.Contains(t, engine.MaintenanceMargin, "BTC-USDT")
		assert.Equal(t, 0.005, engine.MaintenanceMargin["BTC-USDT"])
	})

	t.Run("ProcessLiquidation", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Create test position
		position := &MarginPosition{
			ID:               "pos1",
			Symbol:           "BTC-USDT",
			Side:             Buy,
			Size:             1.0,
			EntryPrice:       50000.0,
			MarkPrice:        48000.0,
			LiquidationPrice: 47000.0,
			Margin:           big.NewInt(5000),
			UnrealizedPnL:    big.NewInt(-2000),
		}
		
		userID := "user1"
		liquidationOrder := &Order{
			ID:   1,
			Type: Market,
			Side: Sell,
			Size: 1.0,
		}
		
		// Disable circuit breaker for test
		engine.CircuitBreaker.Enabled = false
		
		err := engine.ProcessLiquidation(userID, position, liquidationOrder)
		
		assert.NoError(t, err)
		assert.Equal(t, uint64(1), engine.TotalLiquidations)
		assert.Equal(t, uint64(1), engine.LiquidationMetrics.TotalLiquidations)
		
		// Check liquidation order was added to queue
		totalQueueSize := len(engine.LiquidationQueue.HighPriority) +
			len(engine.LiquidationQueue.MediumPriority) +
			len(engine.LiquidationQueue.LowPriority)
		assert.Equal(t, 1, totalQueueSize)
		
		// Test with circuit breaker enabled
		engine.CircuitBreaker.Enabled = true
		
		// Mock circuit breaker to trigger
		engine.CircuitBreaker.TriggerConditions["test"] = &TriggerCondition{
			Check: func() bool { return true },
			Action: func() error { return nil },
			Enabled: true,
		}
		
		err = engine.ProcessLiquidation(userID, position, liquidationOrder)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "circuit breaker triggered")
	})

	t.Run("calculatePriority", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// High priority - well below liquidation price (long position)
		position := &MarginPosition{
			Side:             Buy,
			MarkPrice:        45000.0,
			LiquidationPrice: 48000.0,
		}
		priority := engine.calculatePriority(position)
		assert.Equal(t, HighPriority, priority)
		
		// Medium priority - close to liquidation price
		position.MarkPrice = 47000.0
		priority = engine.calculatePriority(position)
		assert.Equal(t, MediumPriority, priority)
		
		// Low priority - just at liquidation price
		position.MarkPrice = 47500.0
		priority = engine.calculatePriority(position)
		assert.Equal(t, LowPriority, priority)
		
		// Test short position
		position = &MarginPosition{
			Side:             Sell,
			MarkPrice:        52000.0,
			LiquidationPrice: 48000.0,
		}
		priority = engine.calculatePriority(position)
		assert.Equal(t, HighPriority, priority)
	})

	t.Run("calculateLoss", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// No loss - positive PnL
		position := &MarginPosition{
			UnrealizedPnL: big.NewInt(1000),
			Margin:        big.NewInt(5000),
		}
		loss := engine.calculateLoss(position)
		assert.Equal(t, big.NewInt(0), loss)
		
		// No loss - negative PnL but covered by margin
		position.UnrealizedPnL = big.NewInt(-3000)
		loss = engine.calculateLoss(position)
		assert.Equal(t, big.NewInt(0), loss)
		
		// Loss exceeds margin
		position.UnrealizedPnL = big.NewInt(-8000)
		loss = engine.calculateLoss(position)
		assert.Equal(t, big.NewInt(3000), loss) // 8000 - 5000
	})

	t.Run("findBestLiquidator", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Add test liquidators
		liquidator1 := &Liquidator{
			LiquidatorID: "liq1",
			Balance:      big.NewInt(100000),
			Reputation:   0.8,
			SuccessRate:  0.9,
			Tier:         GoldLiquidator,
			LastActive:   time.Now().Add(-30 * time.Minute),
		}
		
		liquidator2 := &Liquidator{
			LiquidatorID: "liq2",
			Balance:      big.NewInt(200000),
			Reputation:   0.9,
			SuccessRate:  0.95,
			Tier:         PlatinumLiquidator,
			LastActive:   time.Now(),
		}
		
		engine.Liquidators["liq1"] = liquidator1
		engine.Liquidators["liq2"] = liquidator2
		
		order := &LiquidationOrder{
			Size:      1.0,
			MarkPrice: 50000.0,
		}
		
		bestLiquidator := engine.findBestLiquidator(order)
		assert.NotNil(t, bestLiquidator)
		assert.Equal(t, "liq2", bestLiquidator.LiquidatorID) // Should pick liquidator2 (higher score)
		
		// Test with insufficient balance
		liquidator1.Balance = big.NewInt(10000) // Too small
		liquidator2.Balance = big.NewInt(10000) // Too small
		
		bestLiquidator = engine.findBestLiquidator(order)
		assert.Nil(t, bestLiquidator)
	})

	t.Run("calculateLiquidatorScore", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		liquidator := &Liquidator{
			Reputation:  0.8,
			SuccessRate: 0.9,
			Tier:        GoldLiquidator,
			Balance:     big.NewInt(1000000),
			LastActive:  time.Now().Add(-25 * time.Hour), // More than 24 hours ago (gets 0 bonus)
		}
		
		order := &LiquidationOrder{}
		
		score := engine.calculateLiquidatorScore(liquidator, order)
		assert.True(t, score > 0)
		assert.True(t, score < 2.0) // Reasonable score range
		
		// Test with recent activity
		liquidator.LastActive = time.Now().Add(-30 * time.Second)
		scoreWithRecentActivity := engine.calculateLiquidatorScore(liquidator, order)
		assert.True(t, scoreWithRecentActivity > score)
		
		// Test with higher tier
		liquidator.Tier = DiamondLiquidator
		scoreWithHigherTier := engine.calculateLiquidatorScore(liquidator, order)
		assert.True(t, scoreWithHigherTier > scoreWithRecentActivity)
	})

	t.Run("executeTrade", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Test liquidating long position (need to sell)
		order := &LiquidationOrder{
			Side:      Buy,
			MarkPrice: 50000.0,
		}
		
		liquidator := &Liquidator{
			LiquidatorID: "test_liq",
		}
		
		executionPrice, err := engine.executeTrade(order, liquidator)
		assert.NoError(t, err)
		assert.True(t, executionPrice < order.MarkPrice) // Should be lower due to slippage
		
		// Test liquidating short position (need to buy)
		order.Side = Sell
		executionPrice, err = engine.executeTrade(order, liquidator)
		assert.NoError(t, err)
		assert.True(t, executionPrice > order.MarkPrice) // Should be higher due to slippage
	})

	t.Run("executeLiquidation", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Add test liquidator
		liquidator := &Liquidator{
			LiquidatorID: "test_liq",
			Balance:      big.NewInt(100000),
			TotalProfit:  big.NewInt(0),
		}
		engine.Liquidators["test_liq"] = liquidator
		
		order := &LiquidationOrder{
			OrderID:   "test_order",
			Size:      1.0,
			MarkPrice: 50000.0,
			Loss:      big.NewInt(0),
			Status:    LiquidationPending,
		}
		
		err := engine.executeLiquidation(order)
		assert.NoError(t, err)
		assert.Equal(t, LiquidationComplete, order.Status)
		assert.NotNil(t, order.ProcessedAt)
		assert.Equal(t, "test_liq", order.LiquidatorID)
		assert.True(t, order.ExecutionPrice > 0)
		assert.True(t, order.LiquidationFee.Sign() > 0)
		
		// Check liquidator stats updated
		assert.Equal(t, uint64(1), liquidator.CompletedCount)
		assert.True(t, liquidator.TotalProfit.Sign() > 0)
		assert.True(t, time.Since(liquidator.LastActive) < time.Second)
		
		// Check metrics updated
		assert.Equal(t, uint64(1), engine.LiquidationMetrics.SuccessfulLiquidations)
		assert.True(t, engine.LiquidationMetrics.TotalVolume.Sign() > 0)
		
		// Check order moved to completed
		assert.Equal(t, 1, len(engine.LiquidationQueue.CompletedOrders))
		assert.Equal(t, 0, len(engine.LiquidationQueue.ProcessingOrders))
	})
}

// Test InsuranceFund functions
func TestInsuranceFundFunctions(t *testing.T) {
	t.Run("NewInsuranceFund", func(t *testing.T) {
		fund := NewInsuranceFund()
		
		assert.NotNil(t, fund)
		assert.NotNil(t, fund.Balance)
		assert.Equal(t, big.NewInt(0), fund.TotalValueUSD)
		assert.Equal(t, big.NewInt(10000000), fund.TargetSize) // $10M target
		assert.Equal(t, big.NewInt(1000000), fund.MinimumSize) // $1M minimum
		assert.Equal(t, 0.5, fund.MaxDrawdown)
		assert.NotNil(t, fund.Contributions)
		assert.NotNil(t, fund.Withdrawals)
		assert.NotNil(t, fund.LossCoverage)
		assert.Equal(t, big.NewInt(0), fund.HighWaterMark)
		assert.NotNil(t, fund.FundManagers)
		assert.NotNil(t, fund.WithdrawalRules)
		assert.True(t, time.Since(fund.LastUpdate) < time.Second)
	})

	t.Run("AddContribution", func(t *testing.T) {
		fund := NewInsuranceFund()
		
		asset := "BTC"
		amount := big.NewInt(500000)
		
		fund.AddContribution(asset, amount)
		
		assert.Equal(t, amount, fund.Balance[asset])
		assert.Equal(t, amount, fund.TotalValueUSD)
		assert.Equal(t, amount, fund.HighWaterMark)
		assert.True(t, time.Since(fund.LastUpdate) < time.Second)
		
		// Add another contribution
		additionalAmount := big.NewInt(300000)
		fund.AddContribution(asset, additionalAmount)
		
		expectedTotal := new(big.Int).Add(amount, additionalAmount)
		assert.Equal(t, expectedTotal, fund.Balance[asset])
		assert.Equal(t, expectedTotal, fund.TotalValueUSD)
		assert.Equal(t, expectedTotal, fund.HighWaterMark)
		
		// Add contribution for different asset
		fund.AddContribution("ETH", big.NewInt(200000))
		assert.Equal(t, big.NewInt(200000), fund.Balance["ETH"])
		
		expectedTotalUSD := new(big.Int).Add(expectedTotal, big.NewInt(200000))
		assert.Equal(t, expectedTotalUSD, fund.TotalValueUSD)
	})

	t.Run("CanCoverLoss", func(t *testing.T) {
		fund := NewInsuranceFund()
		
		asset := "BTC"
		balance := big.NewInt(1000000)
		fund.AddContribution(asset, balance)
		
		// Can cover smaller loss
		loss := big.NewInt(500000)
		canCover := fund.CanCoverLoss(asset, loss)
		assert.True(t, canCover)
		
		// Can cover equal loss
		loss = big.NewInt(1000000)
		canCover = fund.CanCoverLoss(asset, loss)
		assert.True(t, canCover)
		
		// Cannot cover larger loss
		loss = big.NewInt(1500000)
		canCover = fund.CanCoverLoss(asset, loss)
		assert.False(t, canCover)
		
		// Cannot cover loss for non-existent asset
		canCover = fund.CanCoverLoss("ETH", big.NewInt(100000))
		assert.False(t, canCover)
	})

	t.Run("CoverLoss", func(t *testing.T) {
		fund := NewInsuranceFund()
		
		asset := "BTC"
		initialBalance := big.NewInt(1000000)
		fund.AddContribution(asset, initialBalance)
		
		loss := big.NewInt(300000)
		referenceID := "liq_123"
		
		err := fund.CoverLoss(asset, loss, referenceID)
		assert.NoError(t, err)
		
		// Check balance reduced
		expectedBalance := new(big.Int).Sub(initialBalance, loss)
		assert.Equal(t, expectedBalance, fund.Balance[asset])
		assert.Equal(t, expectedBalance, fund.TotalValueUSD)
		
		// Check loss coverage event recorded
		assert.Equal(t, 1, len(fund.LossCoverage))
		event := fund.LossCoverage[0]
		assert.Equal(t, asset, event.Asset)
		assert.Equal(t, loss, event.Amount)
		assert.Equal(t, referenceID, event.ReferenceID)
		assert.True(t, time.Since(event.Timestamp) < time.Second)
		
		// Check drawdown calculated
		assert.True(t, fund.CurrentDrawdown > 0)
		assert.True(t, fund.CurrentDrawdown < 1)
		
		// Test insufficient balance
		largeLoss := big.NewInt(1000000) // More than remaining balance
		err = fund.CoverLoss(asset, largeLoss, "ref2")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient insurance fund balance")
	})

	t.Run("GetCoverageRatio", func(t *testing.T) {
		fund := NewInsuranceFund()
		
		// Empty fund should have 0 ratio
		ratio := fund.GetCoverageRatio()
		assert.Equal(t, 0.0, ratio)
		
		// Add half of target
		halfTarget := new(big.Int).Div(fund.TargetSize, big.NewInt(2))
		fund.AddContribution("BTC", halfTarget)
		
		ratio = fund.GetCoverageRatio()
		assert.InDelta(t, 0.5, ratio, 0.01)
		
		// Add to reach target
		fund.AddContribution("ETH", halfTarget)
		
		ratio = fund.GetCoverageRatio()
		assert.InDelta(t, 1.0, ratio, 0.01)
		
		// Add beyond target
		fund.AddContribution("SOL", halfTarget)
		
		ratio = fund.GetCoverageRatio()
		assert.InDelta(t, 1.5, ratio, 0.01)
	})
}

// Test LiquidationQueue functions
func TestLiquidationQueueFunctions(t *testing.T) {
	t.Run("NewLiquidationQueue", func(t *testing.T) {
		queue := NewLiquidationQueue()
		
		assert.NotNil(t, queue)
		assert.NotNil(t, queue.HighPriority)
		assert.NotNil(t, queue.MediumPriority)
		assert.NotNil(t, queue.LowPriority)
		assert.NotNil(t, queue.Orders)
		assert.NotNil(t, queue.ProcessingOrders)
		assert.NotNil(t, queue.CompletedOrders)
		assert.NotNil(t, queue.FailedOrders)
		
		assert.Equal(t, 0, len(queue.HighPriority))
		assert.Equal(t, 0, len(queue.MediumPriority))
		assert.Equal(t, 0, len(queue.LowPriority))
		assert.Equal(t, 10000, queue.MaxQueueSize)
		assert.Equal(t, 100*time.Millisecond, queue.ProcessingInterval)
		assert.Equal(t, 3, queue.RetryAttempts)
	})

	t.Run("Add", func(t *testing.T) {
		queue := NewLiquidationQueue()
		
		// Add high priority order
		highPriorityOrder := &LiquidationOrder{
			OrderID:  "high1",
			Priority: HighPriority,
		}
		queue.Add(highPriorityOrder)
		assert.Equal(t, 1, len(queue.HighPriority))
		assert.Equal(t, highPriorityOrder, queue.HighPriority[0])
		
		// Add medium priority order
		mediumPriorityOrder := &LiquidationOrder{
			OrderID:  "medium1",
			Priority: MediumPriority,
		}
		queue.Add(mediumPriorityOrder)
		assert.Equal(t, 1, len(queue.MediumPriority))
		assert.Equal(t, mediumPriorityOrder, queue.MediumPriority[0])
		
		// Add low priority order
		lowPriorityOrder := &LiquidationOrder{
			OrderID:  "low1",
			Priority: LowPriority,
		}
		queue.Add(lowPriorityOrder)
		assert.Equal(t, 1, len(queue.LowPriority))
		assert.Equal(t, lowPriorityOrder, queue.LowPriority[0])
		
		// Check total counts
		assert.Equal(t, 1, len(queue.HighPriority))
		assert.Equal(t, 1, len(queue.MediumPriority))
		assert.Equal(t, 1, len(queue.LowPriority))
	})
}

// Test AutoDeleveraging functions
func TestAutoDeleveragingFunctions(t *testing.T) {
	t.Run("NewAutoDeleveragingEngine", func(t *testing.T) {
		adl := NewAutoDeleveragingEngine()
		
		assert.NotNil(t, adl)
		assert.NotNil(t, adl.ADLQueue)
		assert.Equal(t, 0.2, adl.ADLThreshold)
		assert.Equal(t, 0.5, adl.MaxADLPercentage)
		assert.NotNil(t, adl.ADLEvents)
		assert.Equal(t, big.NewInt(0), adl.TotalADLVolume)
	})

	t.Run("Execute", func(t *testing.T) {
		adl := NewAutoDeleveragingEngine()
		
		// Add ADL candidates
		candidates := []*ADLCandidate{
			{
				PositionID:    "pos1",
				UserID:        "user1",
				PnLRanking:    0.8,
				PositionSize:  10.0,
				UnrealizedPnL: big.NewInt(5000),
				Leverage:      2.0,
				ADLPriority:   1,
			},
			{
				PositionID:    "pos2",
				UserID:        "user2",
				PnLRanking:    0.6,
				PositionSize:  5.0,
				UnrealizedPnL: big.NewInt(2000),
				Leverage:      3.0,
				ADLPriority:   2,
			},
		}
		
		symbol := "BTC-USDT"
		adl.ADLQueue[symbol] = candidates
		
		order := &LiquidationOrder{
			OrderID: "liq1",
			Symbol:  symbol,
			Size:    3.0,
		}
		
		err := adl.Execute(order)
		assert.NoError(t, err)
		
		// Check ADL event was created
		assert.Equal(t, 1, len(adl.ADLEvents))
		event := adl.ADLEvents[0]
		assert.NotEmpty(t, event.EventID)
		assert.True(t, time.Since(event.Timestamp) < time.Second)
		assert.Contains(t, event.TriggerReason, order.OrderID)
		assert.True(t, len(event.AffectedPositions) > 0)
		
		// Check affected positions
		for _, affected := range event.AffectedPositions {
			assert.True(t, affected.ReducedSize <= affected.OriginalSize*adl.MaxADLPercentage)
			assert.True(t, affected.ReductionPercentage <= adl.MaxADLPercentage)
			assert.NotNil(t, affected.CompensationPaid)
		}
		
		// Test with no candidates
		err = adl.Execute(&LiquidationOrder{Symbol: "ETH-USDT", OrderID: "liq2"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no ADL candidates available")
	})
}

// Test SocializedLoss functions
func TestSocializedLossFunctions(t *testing.T) {
	t.Run("NewSocializedLossEngine", func(t *testing.T) {
		sl := NewSocializedLossEngine()
		
		assert.NotNil(t, sl)
		assert.NotNil(t, sl.PendingLosses)
		assert.NotNil(t, sl.DistributedLosses)
		assert.Equal(t, big.NewInt(100000), sl.LossThreshold)
		assert.Equal(t, 0.1, sl.MaxLossPerUser)
		assert.NotNil(t, sl.LossHistory)
		assert.Equal(t, big.NewInt(0), sl.TotalSocialized)
	})

	t.Run("DistributeLoss", func(t *testing.T) {
		sl := NewSocializedLossEngine()
		
		// Test loss above threshold
		order := &LiquidationOrder{
			OrderID: "liq1",
			Symbol:  "BTC-USDT",
			Loss:    big.NewInt(200000), // Above threshold
		}
		
		err := sl.DistributeLoss(order)
		assert.NoError(t, err)
		
		// Check socialized loss was created
		assert.Equal(t, 1, len(sl.PendingLosses))
		
		var loss *SocializedLoss
		for _, l := range sl.PendingLosses {
			loss = l
			break
		}
		assert.NotNil(t, loss)
		assert.NotEmpty(t, loss.LossID)
		assert.Equal(t, order.Loss, loss.TotalLoss)
		assert.Equal(t, order.Symbol, loss.AffectedAsset)
		assert.Equal(t, "proportional", loss.DistributionMethod)
		assert.NotNil(t, loss.AffectedUsers)
		assert.True(t, time.Since(loss.Timestamp) < time.Second)
		assert.Equal(t, LossPending, loss.Status)
		
		// Test loss below threshold
		smallOrder := &LiquidationOrder{
			OrderID: "liq2",
			Symbol:  "ETH-USDT",
			Loss:    big.NewInt(50000), // Below threshold
		}
		
		err = sl.DistributeLoss(smallOrder)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "loss below socialization threshold")
	})
}

// Test CircuitBreaker functions
func TestCircuitBreakerFunctions(t *testing.T) {
	t.Run("NewCircuitBreaker", func(t *testing.T) {
		cb := NewCircuitBreaker()
		
		assert.NotNil(t, cb)
		assert.True(t, cb.Enabled)
		assert.NotNil(t, cb.TriggerConditions)
		assert.Equal(t, 5*time.Minute, cb.CooldownPeriod)
		assert.Equal(t, uint64(0), cb.TriggeredCount)
		
		// Check default trigger conditions
		assert.Contains(t, cb.TriggerConditions, "high_liquidation_rate")
		assert.Contains(t, cb.TriggerConditions, "insurance_fund_depleted")
	})

	t.Run("ShouldTrigger", func(t *testing.T) {
		cb := NewCircuitBreaker()
		
		// Test disabled circuit breaker
		cb.Enabled = false
		result := cb.ShouldTrigger()
		assert.False(t, result)
		
		// Test enabled but no triggering conditions
		cb.Enabled = true
		result = cb.ShouldTrigger()
		assert.False(t, result) // Default conditions return false
		
		// Test with triggering condition
		cb.TriggerConditions["test_trigger"] = &TriggerCondition{
			ConditionID: "test",
			Check: func() bool { return true },
			Action: func() error { return nil },
			Enabled: true,
		}
		
		result = cb.ShouldTrigger()
		assert.True(t, result)
		assert.Equal(t, uint64(1), cb.TriggeredCount)
		assert.True(t, time.Since(cb.LastTriggered) < time.Second)
		
		// Test cooldown period
		result = cb.ShouldTrigger()
		assert.False(t, result) // Should be in cooldown
		
		// Test after cooldown (simulate)
		cb.LastTriggered = time.Now().Add(-10 * time.Minute)
		result = cb.ShouldTrigger()
		assert.True(t, result)
		assert.Equal(t, uint64(2), cb.TriggeredCount)
	})
}

// Test helper functions
func TestLiquidationHelperFunctions(t *testing.T) {
	t.Run("initLiquidationMaintenanceMargins", func(t *testing.T) {
		margins := initLiquidationMaintenanceMargins()
		
		assert.NotNil(t, margins)
		assert.True(t, len(margins) > 0)
		
		// Check specific margins
		assert.Contains(t, margins, "BTC-USDT")
		assert.Equal(t, 0.005, margins["BTC-USDT"])
		assert.Contains(t, margins, "ETH-USDT")
		assert.Equal(t, 0.01, margins["ETH-USDT"])
		assert.Contains(t, margins, "BNB-USDT")
		assert.Equal(t, 0.02, margins["BNB-USDT"])
	})

	t.Run("initTriggerConditions", func(t *testing.T) {
		conditions := initTriggerConditions()
		
		assert.NotNil(t, conditions)
		assert.True(t, len(conditions) > 0)
		
		// Check specific conditions
		assert.Contains(t, conditions, "high_liquidation_rate")
		highLiqCondition := conditions["high_liquidation_rate"]
		assert.Equal(t, "high_liq_rate", highLiqCondition.ConditionID)
		assert.Equal(t, "High Liquidation Rate", highLiqCondition.Name)
		assert.NotNil(t, highLiqCondition.Check)
		assert.NotNil(t, highLiqCondition.Action)
		assert.Equal(t, HighSeverity, highLiqCondition.Severity)
		assert.True(t, highLiqCondition.Enabled)
		
		assert.Contains(t, conditions, "insurance_fund_depleted")
		insFundCondition := conditions["insurance_fund_depleted"]
		assert.Equal(t, "ins_fund_depl", insFundCondition.ConditionID)
		assert.Equal(t, "Insurance Fund Depleted", insFundCondition.Name)
		assert.Equal(t, CriticalSeverity, insFundCondition.Severity)
	})

	t.Run("generateIDs", func(t *testing.T) {
		// Test liquidation order ID
		id1 := generateLiquidationOrderID()
		time.Sleep(time.Microsecond) // Use microsecond to ensure different timestamps
		id2 := generateLiquidationOrderID()
		assert.NotEmpty(t, id1)
		assert.NotEmpty(t, id2)
		assert.NotEqual(t, id1, id2)
		assert.Contains(t, id1, "liq_")
		
		// Test event ID
		eventID1 := generateEventID()
		time.Sleep(time.Microsecond) // Use microsecond to ensure different timestamps
		eventID2 := generateEventID()
		assert.NotEmpty(t, eventID1)
		assert.NotEmpty(t, eventID2)
		assert.NotEqual(t, eventID1, eventID2)
		assert.Contains(t, eventID1, "event_")
		
		// Test loss ID
		lossID1 := generateLossID()
		time.Sleep(time.Microsecond) // Use microsecond to ensure different timestamps
		lossID2 := generateLossID()
		assert.NotEmpty(t, lossID1)
		assert.NotEmpty(t, lossID2)
		assert.NotEqual(t, lossID1, lossID2)
		assert.Contains(t, lossID1, "loss_")
	})

	t.Run("NewLiquidationMetrics", func(t *testing.T) {
		metrics := NewLiquidationMetrics()
		
		assert.NotNil(t, metrics)
		assert.Equal(t, uint64(0), metrics.TotalLiquidations)
		assert.Equal(t, uint64(0), metrics.SuccessfulLiquidations)
		assert.Equal(t, uint64(0), metrics.FailedLiquidations)
		assert.Equal(t, uint64(0), metrics.PartialLiquidations)
		assert.Equal(t, big.NewInt(0), metrics.TotalVolume)
		assert.Equal(t, big.NewInt(0), metrics.AverageLoss)
		assert.Equal(t, big.NewInt(0), metrics.MaxLoss)
		assert.Equal(t, big.NewInt(0), metrics.InsuranceFundUsage)
		assert.Equal(t, uint64(0), metrics.ADLTriggers)
		assert.Equal(t, uint64(0), metrics.SocializedLosses)
		assert.True(t, time.Since(metrics.LastUpdate) < time.Second)
	})

	t.Run("NewWithdrawalRules", func(t *testing.T) {
		rules := NewWithdrawalRules()
		
		assert.NotNil(t, rules)
		assert.Equal(t, big.NewInt(1000000), rules.MinBalance)
		assert.Equal(t, big.NewInt(100000), rules.MaxWithdrawal)
		assert.Equal(t, 24*time.Hour, rules.CooldownPeriod)
		assert.Equal(t, 3, rules.RequiredVotes)
	})
}

// Test complex liquidation scenarios
func TestLiquidationScenarios(t *testing.T) {
	t.Run("CompleteLiquidationFlow", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// 1. Add insurance fund balance
		engine.InsuranceFund.AddContribution("BTC", big.NewInt(1000000))
		
		// 2. Add liquidator
		liquidator := &Liquidator{
			LiquidatorID: "test_liquidator",
			Balance:      big.NewInt(500000),
			Reputation:   0.9,
			SuccessRate:  0.95,
			Tier:         PlatinumLiquidator,
			TotalProfit:  big.NewInt(0),
			LastActive:   time.Now(),
		}
		engine.Liquidators["test_liquidator"] = liquidator
		
		// 3. Create position to liquidate
		position := &MarginPosition{
			ID:               "test_position",
			Symbol:           "BTC-USDT",
			Side:             Buy,
			Size:             1.0,
			EntryPrice:       50000.0,
			MarkPrice:        45000.0,
			LiquidationPrice: 46000.0,
			Margin:           big.NewInt(10000),
			UnrealizedPnL:    big.NewInt(-5000),
		}
		
		// 4. Disable circuit breaker for test
		engine.CircuitBreaker.Enabled = false
		
		// 5. Process liquidation
		err := engine.ProcessLiquidation("test_user", position, &Order{ID: 1, Type: Market, Side: Sell, Size: 1.0})
		assert.NoError(t, err)
		
		// 6. Wait briefly for async processing
		time.Sleep(10 * time.Millisecond)
		
		// 7. Verify metrics updated
		assert.Equal(t, uint64(1), engine.TotalLiquidations)
		assert.Equal(t, uint64(1), engine.LiquidationMetrics.TotalLiquidations)
		
		// 8. Check queue has order
		totalQueueSize := len(engine.LiquidationQueue.HighPriority) +
			len(engine.LiquidationQueue.MediumPriority) +
			len(engine.LiquidationQueue.LowPriority)
		assert.True(t, totalQueueSize > 0)
	})

	t.Run("InsuranceFundInsufficientFlow", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Small insurance fund
		engine.InsuranceFund.AddContribution("BTC", big.NewInt(50000))
		
		// Add ADL candidates
		candidates := []*ADLCandidate{
			{
				PositionID:   "profitable_pos",
				UserID:       "profitable_user",
				PositionSize: 5.0,
				PnLRanking:   0.9,
			},
		}
		engine.AutoDeleveraging.ADLQueue["BTC-USDT"] = candidates
		
		// Create order with large loss
		order := &LiquidationOrder{
			OrderID: "large_loss_order",
			Symbol:  "BTC-USDT",
			Loss:    big.NewInt(200000), // More than insurance fund
		}
		
		// Handle loss should trigger ADL
		err := engine.handleLoss(order)
		assert.NoError(t, err)
		
		// Should have created ADL event
		assert.Equal(t, 1, len(engine.AutoDeleveraging.ADLEvents))
	})

	t.Run("SocializedLossFlow", func(t *testing.T) {
		engine := NewLiquidationEngine()
		
		// Empty insurance fund and no ADL candidates
		engine.InsuranceFund.TotalValueUSD = big.NewInt(0)
		
		// Create order with large loss exceeding socialization threshold
		order := &LiquidationOrder{
			OrderID: "socialized_loss_order",
			Symbol:  "BTC-USDT",
			Loss:    big.NewInt(500000), // Above threshold
		}
		
		// Should trigger socialized loss
		err := engine.SocializedLoss.DistributeLoss(order)
		assert.NoError(t, err)
		
		// Check socialized loss was created
		assert.Equal(t, 1, len(engine.SocializedLoss.PendingLosses))
	})
}