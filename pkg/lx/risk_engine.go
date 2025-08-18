package lx

import (
	"math"
	"sync"
	"time"
)

// RiskEngine manages risk for the trading platform
type RiskEngine struct {
	// Risk parameters
	MaxLeverage       map[string]float64 // Symbol -> max leverage
	MaintenanceMargin map[string]float64 // Symbol -> maintenance margin
	InitialMargin     map[string]float64 // Symbol -> initial margin
	MaxPositionSize   map[string]float64 // Symbol -> max position size

	// Risk metrics
	TotalExposure float64
	VaR           float64 // Value at Risk
	MaxDrawdown   float64

	// Risk limits
	MaxTotalExposure float64
	MaxVaR           float64
	MaxConcentration float64

	// State
	mu sync.RWMutex
}

// RiskMetrics contains risk metrics for monitoring
type RiskMetrics struct {
	TotalExposure float64
	ValueAtRisk   float64
	MaxDrawdown   float64
	Sharpe        float64
	Concentration map[string]float64
	UpdatedAt     time.Time
}

// NewRiskEngine creates a new risk engine
func NewRiskEngine() *RiskEngine {
	return &RiskEngine{
		MaxLeverage: map[string]float64{
			"BTC-USDT": 100,
			"ETH-USDT": 100,
			"BNB-USDT": 50,
			"default":  20,
		},
		MaintenanceMargin: map[string]float64{
			"BTC-USDT": 0.005, // 0.5%
			"ETH-USDT": 0.005,
			"BNB-USDT": 0.01,  // 1%
			"default":  0.025, // 2.5%
		},
		InitialMargin: map[string]float64{
			"BTC-USDT": 0.01, // 1%
			"ETH-USDT": 0.01,
			"BNB-USDT": 0.02, // 2%
			"default":  0.05, // 5%
		},
		MaxPositionSize: map[string]float64{
			"BTC-USDT": 100,
			"ETH-USDT": 1000,
			"BNB-USDT": 10000,
			"default":  1000000,
		},
		MaxTotalExposure: 100000000, // $100M
		MaxVaR:           10000000,  // $10M
		MaxConcentration: 0.3,       // 30% in single asset
	}
}

// CheckPositionRisk checks if a position meets risk requirements
func (re *RiskEngine) CheckPositionRisk(symbol string, size, leverage float64) bool {
	re.mu.RLock()
	defer re.mu.RUnlock()

	// Check leverage limit
	maxLev, exists := re.MaxLeverage[symbol]
	if !exists {
		maxLev = re.MaxLeverage["default"]
	}
	if leverage > maxLev {
		return false
	}

	// Check position size
	maxSize, exists := re.MaxPositionSize[symbol]
	if !exists {
		maxSize = re.MaxPositionSize["default"]
	}
	if size > maxSize {
		return false
	}

	// Check total exposure
	exposure := size * leverage
	if re.TotalExposure+exposure > re.MaxTotalExposure {
		return false
	}

	return true
}

// GetMaintenanceMargin returns maintenance margin for a symbol
func (re *RiskEngine) GetMaintenanceMargin(symbol string) float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()

	if margin, exists := re.MaintenanceMargin[symbol]; exists {
		return margin
	}
	return re.MaintenanceMargin["default"]
}

// GetInitialMargin returns initial margin for a symbol
func (re *RiskEngine) GetInitialMargin(symbol string) float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()

	if margin, exists := re.InitialMargin[symbol]; exists {
		return margin
	}
	return re.InitialMargin["default"]
}

// CalculateVaR calculates Value at Risk
func (re *RiskEngine) CalculateVaR(positions []*MarginPosition, confidence float64) float64 {
	re.mu.Lock()
	defer re.mu.Unlock()

	// Simplified VaR calculation
	totalValue := 0.0
	for _, pos := range positions {
		totalValue += pos.Size * pos.MarkPrice
	}

	// Assume 5% daily volatility
	volatility := 0.05
	zScore := 1.96 // 95% confidence

	re.VaR = totalValue * volatility * zScore
	return re.VaR
}

// UpdateExposure updates total exposure
func (re *RiskEngine) UpdateExposure(delta float64) {
	re.mu.Lock()
	defer re.mu.Unlock()

	re.TotalExposure += delta
	if re.TotalExposure < 0 {
		re.TotalExposure = 0
	}
}

// GetRiskMetrics returns current risk metrics
func (re *RiskEngine) GetRiskMetrics() *RiskMetrics {
	re.mu.RLock()
	defer re.mu.RUnlock()

	return &RiskMetrics{
		TotalExposure: re.TotalExposure,
		ValueAtRisk:   re.VaR,
		MaxDrawdown:   re.MaxDrawdown,
		UpdatedAt:     time.Now(),
	}
}

// CalculateLiquidationPrice calculates liquidation price for a position
func (re *RiskEngine) CalculateLiquidationPrice(position *MarginPosition) float64 {
	maintenanceMargin := re.GetMaintenanceMargin(position.Symbol)

	if position.Side == Buy {
		// Long position liquidation price
		// Liquidation when: (Price - Entry) / Entry <= -MaintenanceMargin
		return position.EntryPrice * (1 - maintenanceMargin)
	} else {
		// Short position liquidation price
		// Liquidation when: (Entry - Price) / Entry <= -MaintenanceMargin
		return position.EntryPrice * (1 + maintenanceMargin)
	}
}

// CheckMarginCall checks if account needs margin call
func (re *RiskEngine) CheckMarginCall(account *MarginAccount, markPrices map[string]float64) bool {
	totalValue := 0.0
	totalMarginRequired := 0.0

	for _, position := range account.Positions {
		markPrice := markPrices[position.Symbol]
		positionValue := position.Size * markPrice
		totalValue += positionValue

		maintenanceMargin := re.GetMaintenanceMargin(position.Symbol)
		totalMarginRequired += positionValue * maintenanceMargin
	}

	// Calculate available margin
	availableMargin := 0.0
	for asset, collateral := range account.CollateralAssets {
		if price, exists := markPrices[asset]; exists {
			assetValue := float64(collateral.Amount.Int64()) * price / 1e8
			availableMargin += assetValue * collateral.Haircut
		}
	}

	// Margin call if available margin < required margin * 1.2 (20% buffer)
	return availableMargin < totalMarginRequired*1.2
}

// ValidateLeverage validates leverage for an account type
func (re *RiskEngine) ValidateLeverage(accountType MarginAccountType, symbol string, leverage float64) bool {
	maxLeverage := re.MaxLeverage[symbol]
	if maxLeverage == 0 {
		maxLeverage = re.MaxLeverage["default"]
	}

	switch accountType {
	case CrossMargin:
		return leverage <= math.Min(maxLeverage, 10)
	case IsolatedMargin:
		return leverage <= math.Min(maxLeverage, 20)
	case PortfolioMargin:
		return leverage <= maxLeverage
	default:
		return false
	}
}
