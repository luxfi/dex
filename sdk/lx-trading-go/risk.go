// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"sync"

	"github.com/shopspring/decimal"
)

// RiskManager enforces trading risk limits.
type RiskManager struct {
	config     RiskConfig
	positions  map[string]decimal.Decimal
	dailyPnL   decimal.Decimal
	openOrders map[string]int // symbol -> count
	killSwitch bool

	mu sync.RWMutex
}

// NewRiskManager creates a new risk manager.
func NewRiskManager(config RiskConfig) *RiskManager {
	return &RiskManager{
		config:     config,
		positions:  make(map[string]decimal.Decimal),
		openOrders: make(map[string]int),
	}
}

// IsEnabled returns whether risk management is enabled.
func (r *RiskManager) IsEnabled() bool {
	return r.config.Enabled
}

// IsKilled returns whether the kill switch is active.
func (r *RiskManager) IsKilled() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.killSwitch
}

// Kill activates the kill switch, stopping all trading.
func (r *RiskManager) Kill() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.killSwitch = true
}

// Reset deactivates the kill switch.
func (r *RiskManager) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.killSwitch = false
}

// ValidateOrder validates an order against risk limits.
// Returns an error if the order violates any limit.
func (r *RiskManager) ValidateOrder(request OrderRequest) error {
	if !r.config.Enabled {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Check kill switch
	if r.killSwitch {
		return NewRiskError("kill_switch", "active", "inactive")
	}

	// Check order size
	if !r.config.MaxOrderSize.IsZero() && request.Quantity.GreaterThan(r.config.MaxOrderSize) {
		return NewRiskError("order_size", request.Quantity.String(), r.config.MaxOrderSize.String())
	}

	// Check position limit
	pair := ParseTradingPair(request.Symbol)
	if pair.Base != "" {
		current := r.positions[pair.Base]

		var newPosition decimal.Decimal
		if request.Side == SideBuy {
			newPosition = current.Add(request.Quantity)
		} else {
			newPosition = current.Sub(request.Quantity)
		}

		// Asset-specific limit
		if limit, ok := r.config.PositionLimits[pair.Base]; ok {
			if newPosition.Abs().GreaterThan(limit) {
				return NewRiskError(pair.Base+"_position", newPosition.Abs().String(), limit.String())
			}
		}

		// Global position limit
		if !r.config.MaxPositionSize.IsZero() && newPosition.Abs().GreaterThan(r.config.MaxPositionSize) {
			return NewRiskError("position_size", newPosition.Abs().String(), r.config.MaxPositionSize.String())
		}
	}

	// Check open orders count
	count := r.openOrders[request.Symbol]
	if count >= r.config.MaxOpenOrders {
		return NewRiskError("open_orders", itoa(count), itoa(r.config.MaxOpenOrders))
	}

	// Check daily loss
	if !r.config.MaxDailyLoss.IsZero() && r.dailyPnL.IsNegative() {
		if r.dailyPnL.Abs().GreaterThan(r.config.MaxDailyLoss) {
			return NewRiskError("daily_loss", r.dailyPnL.Abs().String(), r.config.MaxDailyLoss.String())
		}
	}

	return nil
}

// UpdatePosition updates position after a trade.
func (r *RiskManager) UpdatePosition(asset string, quantity decimal.Decimal, side Side) {
	r.mu.Lock()
	defer r.mu.Unlock()

	current := r.positions[asset]
	if side == SideBuy {
		r.positions[asset] = current.Add(quantity)
	} else {
		r.positions[asset] = current.Sub(quantity)
	}
}

// UpdatePnL updates daily PnL.
func (r *RiskManager) UpdatePnL(pnl decimal.Decimal) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.dailyPnL = r.dailyPnL.Add(pnl)

	// Auto kill switch
	if r.config.KillSwitchEnabled && !r.config.MaxDailyLoss.IsZero() {
		if r.dailyPnL.IsNegative() && r.dailyPnL.Abs().GreaterThan(r.config.MaxDailyLoss) {
			r.killSwitch = true
		}
	}
}

// OrderOpened increments open orders count.
func (r *RiskManager) OrderOpened(symbol string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.openOrders[symbol]++
}

// OrderClosed decrements open orders count.
func (r *RiskManager) OrderClosed(symbol string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.openOrders[symbol] > 0 {
		r.openOrders[symbol]--
	}
}

// Position returns current position for an asset.
func (r *RiskManager) Position(asset string) decimal.Decimal {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.positions[asset]
}

// Positions returns all positions.
func (r *RiskManager) Positions() map[string]decimal.Decimal {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]decimal.Decimal, len(r.positions))
	for k, v := range r.positions {
		result[k] = v
	}
	return result
}

// DailyPnL returns current daily PnL.
func (r *RiskManager) DailyPnL() decimal.Decimal {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.dailyPnL
}

// ResetDailyPnL resets daily PnL (call at start of trading day).
func (r *RiskManager) ResetDailyPnL() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.dailyPnL = decimal.Zero
}

// OpenOrderCount returns open orders count for a symbol.
func (r *RiskManager) OpenOrderCount(symbol string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.openOrders[symbol]
}

// SetPositionLimit sets position limit for an asset.
func (r *RiskManager) SetPositionLimit(asset string, limit decimal.Decimal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.config.PositionLimits == nil {
		r.config.PositionLimits = make(map[string]decimal.Decimal)
	}
	r.config.PositionLimits[asset] = limit
}

// SetMaxOrderSize sets maximum order size.
func (r *RiskManager) SetMaxOrderSize(max decimal.Decimal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.MaxOrderSize = max
}

// SetMaxDailyLoss sets maximum daily loss.
func (r *RiskManager) SetMaxDailyLoss(max decimal.Decimal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.MaxDailyLoss = max
}

// EnableKillSwitch enables automatic kill switch on max loss.
func (r *RiskManager) EnableKillSwitch(enabled bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.KillSwitchEnabled = enabled
}

// ClearPositions clears all position tracking.
func (r *RiskManager) ClearPositions() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.positions = make(map[string]decimal.Decimal)
}

// ClearOpenOrders clears open orders tracking.
func (r *RiskManager) ClearOpenOrders() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.openOrders = make(map[string]int)
}

// Summary returns a summary of current risk state.
func (r *RiskManager) Summary() RiskSummary {
	r.mu.RLock()
	defer r.mu.RUnlock()

	totalOpenOrders := 0
	for _, count := range r.openOrders {
		totalOpenOrders += count
	}

	return RiskSummary{
		Enabled:         r.config.Enabled,
		KillSwitch:      r.killSwitch,
		DailyPnL:        r.dailyPnL,
		MaxDailyLoss:    r.config.MaxDailyLoss,
		TotalOpenOrders: totalOpenOrders,
		MaxOpenOrders:   r.config.MaxOpenOrders,
		Positions:       r.Positions(),
	}
}

// RiskSummary provides a snapshot of risk state.
type RiskSummary struct {
	Enabled         bool
	KillSwitch      bool
	DailyPnL        decimal.Decimal
	MaxDailyLoss    decimal.Decimal
	TotalOpenOrders int
	MaxOpenOrders   int
	Positions       map[string]decimal.Decimal
}
