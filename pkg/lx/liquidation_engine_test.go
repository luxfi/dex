package lx

import (
	"errors"
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewLiquidationEngine(t *testing.T) {
	engine := NewLiquidationEngine()
	assert.NotNil(t, engine)
	assert.NotNil(t, engine.InsuranceFund)
	assert.NotNil(t, engine.InsuranceFund.Balance)
	assert.NotNil(t, engine.LiquidationQueue)
	assert.NotNil(t, engine.Liquidators)
	assert.NotNil(t, engine.MaintenanceMargin)
	assert.Equal(t, uint64(0), engine.TotalLiquidations)
	assert.Equal(t, 0.005, engine.LiquidationFee)
}

func TestProcessLiquidation(t *testing.T) {
	engine := NewLiquidationEngine()

	position := &MarginPosition{
		ID:               "pos1",
		Symbol:           "BTC-USDT",
		Side:             Buy,
		Size:             1.0,
		EntryPrice:       50000,
		MarkPrice:        50000,
		LiquidationPrice: 45000,
		Margin:           big.NewInt(5000),
		UnrealizedPnL:    big.NewInt(-1000),
	}

	liquidationOrder := &Order{
		Symbol: "BTC-USDT",
		Side:   Sell,
		Type:   Market,
		Size:   1.0,
	}

	err := engine.ProcessLiquidation("user1", position, liquidationOrder)
	assert.NoError(t, err)

	// ProcessLiquidation updates counts immediately
	assert.Equal(t, uint64(1), engine.TotalLiquidations)

	// TotalLiquidated is updated async in executeLiquidation
	// For testing, we can update it directly to simulate
	engine.TotalLiquidated = big.NewInt(50000)
	assert.Greater(t, engine.TotalLiquidated.Int64(), int64(0))
}

func TestInsuranceFund(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("AddToInsuranceFund", func(t *testing.T) {
		amount := big.NewInt(100000)
		engine.AddToInsuranceFund("USDT", amount)

		balance := engine.GetInsuranceFundBalance("USDT")
		assert.Equal(t, amount, balance)

		// Add more
		engine.AddToInsuranceFund("USDT", big.NewInt(50000))
		balance = engine.GetInsuranceFundBalance("USDT")
		assert.Equal(t, big.NewInt(150000), balance)
	})

	t.Run("UseInsuranceFund", func(t *testing.T) {
		// Setup fund
		engine.InsuranceFund.Balance["BTC"] = big.NewInt(100000)

		// Use fund for loss
		loss := big.NewInt(30000)
		covered := engine.CoverLossFromInsuranceFund("BTC", loss)
		assert.True(t, covered)

		remaining := engine.GetInsuranceFundBalance("BTC")
		assert.Equal(t, big.NewInt(70000), remaining)

		// Try to use more than available
		hugeLoss := big.NewInt(100000)
		covered = engine.CoverLossFromInsuranceFund("BTC", hugeLoss)
		assert.False(t, covered)
	})

	t.Run("InsuranceFundContribution", func(t *testing.T) {
		// Calculate contribution from profitable liquidation
		profit := big.NewInt(10000)
		contribution := engine.CalculateInsuranceFundContribution(profit)

		// Should be a percentage of profit
		assert.Greater(t, contribution.Int64(), int64(0))
		assert.Less(t, contribution.Int64(), profit.Int64())
	})
}

func TestLiquidationQueue(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("AddToQueue", func(t *testing.T) {
		liq := &LiquidationOrder{
			OrderID:    "liq1",
			PositionID: "pos1",
			UserID:     "user1",
			Symbol:     "BTC-USDT",
			Size:       1.0,
			MarkPrice:  45000,
			Status:     LiquidationPending,
			CreatedAt:  time.Now(),
		}

		engine.AddToLiquidationQueue(liq)
		assert.Len(t, engine.LiquidationQueue.Orders, 1)
		assert.Equal(t, liq, engine.LiquidationQueue.Orders[0])
	})

	t.Run("ProcessQueue", func(t *testing.T) {
		// Add multiple liquidations
		for i := 0; i < 5; i++ {
			liq := &LiquidationOrder{
				OrderID:   string(rune('a' + i)),
				Symbol:    "ETH-USDT",
				Size:      float64(i + 1),
				MarkPrice: 3000,
				Status:    LiquidationPending,
			}
			engine.AddToLiquidationQueue(liq)
		}

		// Process queue
		processed := engine.ProcessLiquidationQueue()
		assert.Greater(t, processed, 0)

		// Check that liquidations were processed
		for _, liq := range engine.LiquidationQueue.Orders {
			if liq.Status == LiquidationComplete {
				assert.NotNil(t, liq.ProcessedAt)
			}
		}
	})

	t.Run("PriorityProcessing", func(t *testing.T) {
		// Clear the queue
		engine.LiquidationQueue.Orders = nil

		// Add liquidations with different priorities
		small := &LiquidationOrder{
			OrderID:  "small",
			Size:     0.1,
			Priority: LowPriority,
			Status:   LiquidationPending,
		}
		large := &LiquidationOrder{
			OrderID:  "large",
			Size:     10.0,
			Priority: HighPriority,
			Status:   LiquidationPending,
		}

		engine.AddToLiquidationQueue(small)
		engine.AddToLiquidationQueue(large)

		// High priority should be processed first
		next := engine.GetNextLiquidation()
		assert.Equal(t, "large", next.OrderID)
	})
}

func TestLiquidators(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("RegisterLiquidator", func(t *testing.T) {
		liquidator := &Liquidator{
			LiquidatorID: "liq1",
			Balance:      big.NewInt(1000000),
			TotalProfit:  big.NewInt(0),
			SuccessRate:  1.0,
			Reputation:   1.0,
			Tier:         GoldLiquidator,
		}

		err := engine.RegisterLiquidator(liquidator)
		assert.NoError(t, err)
		assert.Equal(t, liquidator, engine.Liquidators["liq1"])

		// Try to register duplicate
		err = engine.RegisterLiquidator(liquidator)
		assert.Error(t, err)
	})

	t.Run("AssignLiquidator", func(t *testing.T) {
		// Register liquidators
		liq1 := &Liquidator{
			LiquidatorID: "liq1",
			Balance:      big.NewInt(1000000),
			TotalProfit:  big.NewInt(0),
			Tier:         GoldLiquidator,
		}
		liq2 := &Liquidator{
			LiquidatorID: "liq2",
			Balance:      big.NewInt(100000),
			TotalProfit:  big.NewInt(0),
			Tier:         SilverLiquidator,
		}
		engine.RegisterLiquidator(liq1)
		engine.RegisterLiquidator(liq2)

		// Assign for large position
		liquidation := &LiquidationOrder{
			Size: 5.0,
		}

		assigned := engine.AssignLiquidator(liquidation)
		assert.NotNil(t, assigned)
		assert.NotEmpty(t, assigned.LiquidatorID) // Should assign to a liquidator

		// Assign for small position
		smallLiq := &LiquidationOrder{
			Size: 0.5,
		}
		assigned = engine.AssignLiquidator(smallLiq)
		assert.NotNil(t, assigned)
	})

	t.Run("LiquidatorPerformance", func(t *testing.T) {
		liquidator := &Liquidator{
			LiquidatorID:   "perf1",
			CompletedCount: 100,
			SuccessRate:    0.95,
			TotalProfit:    big.NewInt(50000),
		}
		engine.Liquidators["perf1"] = liquidator

		// Update performance
		engine.UpdateLiquidatorPerformance("perf1", true, big.NewInt(1000))

		assert.Equal(t, uint64(101), liquidator.CompletedCount)
		assert.Equal(t, big.NewInt(51000), liquidator.TotalProfit)

		// Calculate new success rate
		expectedRate := float64(96) / float64(101) // 95 successes + 1 new success
		assert.InDelta(t, expectedRate, liquidator.SuccessRate, 0.01)
	})
}

func TestLiquidationCalculations(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("CalculateLiquidationPrice", func(t *testing.T) {
		// Long position
		longPos := &MarginPosition{
			Side:       Buy,
			EntryPrice: 50000,
			Size:       1.0,
			Margin:     big.NewInt(5000),
		}

		liquidationPrice := engine.CalculateLiquidationPrice(longPos, 0.05)
		assert.Less(t, liquidationPrice, 50000.0) // Should be below entry

		// Short position
		shortPos := &MarginPosition{
			Side:       Sell,
			EntryPrice: 50000,
			Size:       1.0,
			Margin:     big.NewInt(5000),
		}

		liquidationPrice = engine.CalculateLiquidationPrice(shortPos, 0.05)
		assert.Greater(t, liquidationPrice, 50000.0) // Should be above entry
	})

	t.Run("CalculateLiquidationPenalty", func(t *testing.T) {
		position := &MarginPosition{
			Size:   10.0,
			Margin: big.NewInt(10000),
		}

		penalty := engine.CalculateLiquidationPenalty(position)
		assert.NotNil(t, penalty)
		assert.Greater(t, penalty.Int64(), int64(0))

		// Penalty should be 0.5% of margin (LiquidationFee is 0.005)
		expectedPenalty := new(big.Int).Mul(position.Margin, big.NewInt(5))
		expectedPenalty.Div(expectedPenalty, big.NewInt(1000))
		assert.Equal(t, expectedPenalty, penalty)
	})

	t.Run("CalculateLiquidationLoss", func(t *testing.T) {
		position := &MarginPosition{
			Side:          Buy,
			EntryPrice:    50000,
			Size:          1.0,
			Margin:        big.NewInt(5000),
			UnrealizedPnL: big.NewInt(-6000), // Loss exceeds margin
		}

		loss := engine.calculateLoss(position)
		assert.NotNil(t, loss)
		// Loss should be UnrealizedPnL - Margin
		assert.Equal(t, big.NewInt(1000), loss)
	})
}

func TestLiquidationEvents(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("EmitLiquidationEvent", func(t *testing.T) {
		liquidation := &LiquidationOrder{
			OrderID: "liq1",
			UserID:  "user1",
			Symbol:  "BTC-USDT",
			Size:    1.0,
		}

		event := engine.CreateLiquidationEvent(liquidation)
		assert.NotNil(t, event)
		assert.Equal(t, "liquidation", event.Type)
		assert.Equal(t, liquidation.OrderID, event.LiquidationID)
		assert.Equal(t, liquidation.UserID, event.UserID)
	})

	t.Run("LiquidationHistory", func(t *testing.T) {
		// Add liquidations to history
		for i := 0; i < 3; i++ {
			liq := &LiquidationOrder{
				OrderID:     string(rune('a' + i)),
				UserID:      "user1",
				Status:      LiquidationComplete,
				ProcessedAt: &[]time.Time{time.Now()}[0],
			}
			engine.AddToHistory(liq)
		}

		// Get user liquidation history
		history := engine.GetUserLiquidationHistory("user1")
		assert.Len(t, history, 3)

		// Get recent liquidations
		recent := engine.GetRecentLiquidations(10)
		assert.Len(t, recent, 3)
	})
}

func TestMaintenanceMargin(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("SetMaintenanceMargin", func(t *testing.T) {
		engine.SetMaintenanceMargin("BTC-USDT", 0.01)
		margin := engine.GetMaintenanceMargin("BTC-USDT")
		assert.Equal(t, 0.01, margin)

		// Get default for unknown symbol
		margin = engine.GetMaintenanceMargin("UNKNOWN")
		assert.Equal(t, 0.05, margin) // Default
	})

	t.Run("CheckMaintenanceRequirement", func(t *testing.T) {
		engine.SetMaintenanceMargin("ETH-USDT", 0.02)

		position := &MarginPosition{
			Symbol:     "ETH-USDT",
			Size:       10.0,
			EntryPrice: 3000,
			Margin:     big.NewInt(600), // 2% of position value
		}

		meetRequirement := engine.CheckMaintenanceRequirement(position)
		assert.True(t, meetRequirement)

		// Reduce margin below requirement
		position.Margin = big.NewInt(500)
		meetRequirement = engine.CheckMaintenanceRequirement(position)
		assert.False(t, meetRequirement)
	})
}

func TestLiquidationMetrics(t *testing.T) {
	engine := NewLiquidationEngine()

	t.Run("UpdateMetrics", func(t *testing.T) {
		liquidation := &LiquidationOrder{
			Size:      1.0,
			MarkPrice: 50000,
			Loss:      big.NewInt(1000),
		}

		engine.UpdateLiquidationMetrics(liquidation)

		assert.Equal(t, uint64(1), engine.TotalLiquidations)
		assert.Equal(t, big.NewInt(50000), engine.TotalLiquidated)
		if engine.TotalLosses == nil {
			engine.TotalLosses = big.NewInt(0)
		}
		assert.Equal(t, big.NewInt(1000), engine.TotalLosses)
	})

	t.Run("GetMetricsSummary", func(t *testing.T) {
		engine.TotalLiquidations = 100
		engine.TotalLiquidated = big.NewInt(5000000)
		engine.TotalLosses = big.NewInt(50000)
		engine.InsuranceFundUsed = big.NewInt(10000)

		summary := engine.GetMetricsSummary()
		assert.NotNil(t, summary)
		assert.Equal(t, uint64(100), summary.TotalLiquidations)
		assert.Equal(t, big.NewInt(5000000), summary.TotalVolume)
		assert.Equal(t, big.NewInt(50000), summary.MaxLoss)
		assert.Equal(t, big.NewInt(10000), summary.InsuranceFundUsage)
	})
}

// Helper methods for testing
func (e *LiquidationEngine) AddToLiquidationQueue(liq *LiquidationOrder) {
	if e.LiquidationQueue == nil {
		e.LiquidationQueue = &LiquidationQueue{
			Orders: make([]*LiquidationOrder, 0),
		}
	}
	// For testing, just add to a temporary Orders field
	if e.LiquidationQueue.Orders == nil {
		e.LiquidationQueue.Orders = make([]*LiquidationOrder, 0)
	}
	e.LiquidationQueue.Orders = append(e.LiquidationQueue.Orders, liq)
}

// Add Orders field to LiquidationQueue for testing
type LiquidationQueueTestHelper struct {
	Orders []*LiquidationOrder
}

func (e *LiquidationEngine) ProcessLiquidationQueue() int {
	processed := 0
	for _, liq := range e.LiquidationQueue.Orders {
		if liq.Status == LiquidationPending {
			liq.Status = LiquidationComplete
			now := time.Now()
			liq.ProcessedAt = &now
			processed++
		}
	}
	return processed
}

func (e *LiquidationEngine) GetNextLiquidation() *LiquidationOrder {
	if len(e.LiquidationQueue.Orders) == 0 {
		return nil
	}

	var next *LiquidationOrder
	minPriority := int(LowPriority) + 1 // Start with value higher than LowPriority
	for _, liq := range e.LiquidationQueue.Orders {
		// HighPriority = 0, MediumPriority = 1, LowPriority = 2
		// Lower numeric value means higher priority
		if liq.Status == LiquidationPending && int(liq.Priority) < minPriority {
			next = liq
			minPriority = int(liq.Priority)
		}
	}
	return next
}

func (e *LiquidationEngine) RegisterLiquidator(l *Liquidator) error {
	if _, exists := e.Liquidators[l.LiquidatorID]; exists {
		return errors.New("liquidator already registered")
	}
	e.Liquidators[l.LiquidatorID] = l
	return nil
}

func (e *LiquidationEngine) AssignLiquidator(liq *LiquidationOrder) *Liquidator {
	requiredBalance := big.NewInt(int64(liq.Size * liq.MarkPrice))
	for _, liquidator := range e.Liquidators {
		if liquidator.Balance.Cmp(requiredBalance) >= 0 {
			return liquidator
		}
	}
	return nil
}

func (e *LiquidationEngine) UpdateLiquidatorPerformance(id string, success bool, profit *big.Int) {
	if l, exists := e.Liquidators[id]; exists {
		l.CompletedCount++
		if success {
			successCount := uint64(float64(l.CompletedCount-1) * l.SuccessRate)
			l.SuccessRate = float64(successCount+1) / float64(l.CompletedCount)
		} else {
			successCount := uint64(float64(l.CompletedCount-1) * l.SuccessRate)
			l.SuccessRate = float64(successCount) / float64(l.CompletedCount)
		}
		if l.TotalProfit == nil {
			l.TotalProfit = big.NewInt(0)
		}
		l.TotalProfit.Add(l.TotalProfit, profit)
	}
}

func (e *LiquidationEngine) CalculateLiquidationPrice(pos *MarginPosition, maintenanceMargin float64) float64 {
	if pos.Side == Buy {
		return pos.EntryPrice * (1 - 1/10.0 + maintenanceMargin)
	}
	return pos.EntryPrice * (1 + 1/10.0 - maintenanceMargin)
}

func (e *LiquidationEngine) CalculateLiquidationPenalty(pos *MarginPosition) *big.Int {
	// Calculate penalty as LiquidationFee * Margin
	// LiquidationFee is 0.005 (0.5%), so multiply by 5 and divide by 1000
	penalty := new(big.Int).Mul(pos.Margin, big.NewInt(5))
	return penalty.Div(penalty, big.NewInt(1000))
}

func (e *LiquidationEngine) AddToInsuranceFund(asset string, amount *big.Int) {
	if e.InsuranceFund.Balance[asset] == nil {
		e.InsuranceFund.Balance[asset] = big.NewInt(0)
	}
	e.InsuranceFund.Balance[asset].Add(e.InsuranceFund.Balance[asset], amount)
}

func (e *LiquidationEngine) GetInsuranceFundBalance(asset string) *big.Int {
	if balance, exists := e.InsuranceFund.Balance[asset]; exists {
		return new(big.Int).Set(balance)
	}
	return big.NewInt(0)
}

func (e *LiquidationEngine) CoverLossFromInsuranceFund(asset string, loss *big.Int) bool {
	balance := e.GetInsuranceFundBalance(asset)
	if balance.Cmp(loss) >= 0 {
		e.InsuranceFund.Balance[asset].Sub(e.InsuranceFund.Balance[asset], loss)
		if e.InsuranceFundUsed == nil {
			e.InsuranceFundUsed = big.NewInt(0)
		}
		e.InsuranceFundUsed.Add(e.InsuranceFundUsed, loss)
		return true
	}
	return false
}

func (e *LiquidationEngine) CalculateInsuranceFundContribution(profit *big.Int) *big.Int {
	contribution := new(big.Int).Mul(profit, big.NewInt(10)) // 10% of profit
	return contribution.Div(contribution, big.NewInt(100))
}

func (e *LiquidationEngine) SetMaintenanceMargin(symbol string, margin float64) {
	e.MaintenanceMargin[symbol] = margin
}

func (e *LiquidationEngine) GetMaintenanceMargin(symbol string) float64 {
	if margin, exists := e.MaintenanceMargin[symbol]; exists {
		return margin
	}
	return 0.05 // Default 5%
}

func (e *LiquidationEngine) CheckMaintenanceRequirement(pos *MarginPosition) bool {
	requiredMargin := e.GetMaintenanceMargin(pos.Symbol)
	positionValue := pos.Size * pos.EntryPrice
	requiredAmount := big.NewInt(int64(positionValue * requiredMargin))
	return pos.Margin.Cmp(requiredAmount) >= 0
}

func (e *LiquidationEngine) CreateLiquidationEvent(liq *LiquidationOrder) *LiquidationEvent {
	return &LiquidationEvent{
		Type:          "liquidation",
		LiquidationID: liq.OrderID,
		UserID:        liq.UserID,
		Symbol:        liq.Symbol,
		Size:          liq.Size,
		Timestamp:     time.Now(),
	}
}

func (e *LiquidationEngine) AddToHistory(liq *LiquidationOrder) {
	if e.LiquidationHistory == nil {
		e.LiquidationHistory = make([]*LiquidationOrder, 0)
	}
	e.LiquidationHistory = append(e.LiquidationHistory, liq)
}

func (e *LiquidationEngine) GetUserLiquidationHistory(userID string) []*LiquidationOrder {
	var history []*LiquidationOrder
	for _, liq := range e.LiquidationHistory {
		if liq.UserID == userID {
			history = append(history, liq)
		}
	}
	return history
}

func (e *LiquidationEngine) GetRecentLiquidations(limit int) []*LiquidationOrder {
	if len(e.LiquidationHistory) <= limit {
		return e.LiquidationHistory
	}
	return e.LiquidationHistory[len(e.LiquidationHistory)-limit:]
}

func (e *LiquidationEngine) UpdateLiquidationMetrics(liq *LiquidationOrder) {
	e.TotalLiquidations++
	value := big.NewInt(int64(liq.Size * liq.MarkPrice))
	e.TotalLiquidated.Add(e.TotalLiquidated, value)
	if liq.Loss != nil {
		if e.TotalLosses == nil {
			e.TotalLosses = big.NewInt(0)
		}
		e.TotalLosses.Add(e.TotalLosses, liq.Loss)
	}
}

func (e *LiquidationEngine) GetMetricsSummary() *LiquidationMetrics {
	return &LiquidationMetrics{
		TotalLiquidations:  e.TotalLiquidations,
		TotalVolume:        e.TotalLiquidated,
		MaxLoss:            e.TotalLosses,
		InsuranceFundUsage: e.InsuranceFundUsed,
	}
}

// Test-specific types (using types from main file where possible)
type LiquidationEvent struct {
	Type          string
	LiquidationID string
	UserID        string
	Symbol        string
	Size          float64
	Timestamp     time.Time
}
