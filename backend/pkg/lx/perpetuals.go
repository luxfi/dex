package lx

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// PerpetualManager manages all perpetual markets
type PerpetualManager struct {
	markets   map[string]*PerpetualMarket
	positions map[string]map[string]*PerpPosition // user -> symbol -> position
	engine    *TradingEngine
	oracles   map[string]*PriceOracle
	mu        sync.RWMutex
}

// NewPerpetualManager creates a new perpetual manager
func NewPerpetualManager(engine *TradingEngine) *PerpetualManager {
	return &PerpetualManager{
		markets:   make(map[string]*PerpetualMarket),
		positions: make(map[string]map[string]*PerpPosition),
		engine:    engine,
		oracles:   make(map[string]*PriceOracle),
	}
}

// PerpPosition represents a perpetual position with enhanced tracking
type PerpPosition struct {
	Symbol            string
	User              string
	Size              float64 // Positive for long, negative for short
	EntryPrice        float64
	MarkPrice         float64
	IndexPrice        float64
	UnrealizedPnL     float64
	RealizedPnL       float64
	Margin            float64
	MaintenanceMargin float64
	InitialMargin     float64
	Leverage          float64
	LiquidationPrice  float64
	FundingPaid       float64
	LastFundingTime   time.Time
	OpenTime          time.Time
	UpdateTime        time.Time
	IsIsolated        bool
	CrossMargin       float64
	OrderMargin       float64 // Margin locked in open orders
	PositionValue     float64
	Notional          float64
}

// Enhanced PerpetualMarket with funding mechanism
type PerpetualMarket struct {
	Symbol            string
	UnderlyingAsset   string
	QuoteAsset        string
	MarkPrice         float64
	IndexPrice        float64
	FundingRate       float64
	FundingInterval   time.Duration
	NextFundingTime   time.Time
	LastFundingTime   time.Time
	OpenInterest      float64
	OpenInterestValue float64
	MaxLeverage       float64
	MaintenanceMargin float64
	InitialMargin     float64
	TakerFee          float64
	MakerFee          float64
	MaxPositionSize   float64
	PricePrecision    int
	SizePrecision     int
	TickSize          float64
	ContractSize      float64
	IsInverse         bool // Inverse perpetuals (e.g., BTC/USD inverse)
	State             MarketState
	Volume24h         float64
	Turnover24h       float64
	High24h           float64
	Low24h            float64
	PrevClose24h      float64
	FundingHistory    []FundingPayment
	mu                sync.RWMutex
}

// MarketState represents the state of a market
type MarketState int

const (
	MarketStateActive MarketState = iota
	MarketStateSuspended
	MarketStateSettlement
	MarketStateClosed
)

// FundingPayment represents a funding payment record
type FundingPayment struct {
	Time        time.Time
	Rate        float64
	MarkPrice   float64
	IndexPrice  float64
	PaymentSize float64
}

// CreateMarket creates a new perpetual market
func (pm *PerpetualManager) CreateMarket(config PerpMarketConfig) (*PerpetualMarket, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if _, exists := pm.markets[config.Symbol]; exists {
		return nil, fmt.Errorf("market %s already exists", config.Symbol)
	}

	market := &PerpetualMarket{
		Symbol:            config.Symbol,
		UnderlyingAsset:   config.UnderlyingAsset,
		QuoteAsset:        config.QuoteAsset,
		FundingInterval:   config.FundingInterval,
		NextFundingTime:   time.Now().Add(config.FundingInterval),
		LastFundingTime:   time.Now(),
		MaxLeverage:       config.MaxLeverage,
		MaintenanceMargin: config.MaintenanceMargin,
		InitialMargin:     config.InitialMargin,
		TakerFee:          config.TakerFee,
		MakerFee:          config.MakerFee,
		MaxPositionSize:   config.MaxPositionSize,
		PricePrecision:    config.PricePrecision,
		SizePrecision:     config.SizePrecision,
		TickSize:          config.TickSize,
		ContractSize:      config.ContractSize,
		IsInverse:         config.IsInverse,
		State:             MarketStateActive,
		FundingHistory:    make([]FundingPayment, 0, 1000),
	}

	// Initialize oracle
	oracle := NewPriceOracle()
	// Store the oracle for this market
	pm.oracles[config.Symbol] = oracle

	pm.markets[config.Symbol] = market
	return market, nil
}

// PerpMarketConfig configuration for a perpetual market
type PerpMarketConfig struct {
	Symbol            string
	UnderlyingAsset   string
	QuoteAsset        string
	FundingInterval   time.Duration
	MaxLeverage       float64
	MaintenanceMargin float64
	InitialMargin     float64
	TakerFee          float64
	MakerFee          float64
	MaxPositionSize   float64
	PricePrecision    int
	SizePrecision     int
	TickSize          float64
	ContractSize      float64
	IsInverse         bool
	OracleSource      string
}

// OpenPosition opens a new perpetual position
func (pm *PerpetualManager) OpenPosition(user string, symbol string, side Side, size float64, leverage float64, isIsolated bool) (*PerpPosition, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	market, exists := pm.markets[symbol]
	if !exists {
		return nil, ErrPerpNotFound
	}

	// Validate leverage
	if leverage > market.MaxLeverage {
		return nil, ErrExcessiveLeverage
	}

	// Check position size limits
	if size > market.MaxPositionSize {
		return nil, fmt.Errorf("position size exceeds limit: %f", market.MaxPositionSize)
	}

	market.mu.RLock()
	markPrice := market.MarkPrice
	if markPrice == 0 {
		markPrice = market.IndexPrice // Fallback to index price
	}
	market.mu.RUnlock()

	// Calculate required margin
	notional := size * markPrice * market.ContractSize
	requiredMargin := notional / leverage
	initialMargin := notional * market.InitialMargin
	maintenanceMargin := notional * market.MaintenanceMargin

	// Adjust size for side (negative for short)
	if side == Sell {
		size = -size
	}

	// Get or create user positions map
	if pm.positions[user] == nil {
		pm.positions[user] = make(map[string]*PerpPosition)
	}

	// Check for existing position
	existingPos, hasPosition := pm.positions[user][symbol]
	if hasPosition {
		// Merge positions
		return pm.modifyPosition(existingPos, size, markPrice, leverage)
	}

	// Create new position
	position := &PerpPosition{
		Symbol:            symbol,
		User:              user,
		Size:              size,
		EntryPrice:        markPrice,
		MarkPrice:         markPrice,
		IndexPrice:        market.IndexPrice,
		Margin:            requiredMargin,
		InitialMargin:     initialMargin,
		MaintenanceMargin: maintenanceMargin,
		Leverage:          leverage,
		OpenTime:          time.Now(),
		UpdateTime:        time.Now(),
		LastFundingTime:   market.LastFundingTime,
		IsIsolated:        isIsolated,
		PositionValue:     math.Abs(size) * markPrice * market.ContractSize,
		Notional:          notional,
	}

	// Calculate liquidation price
	position.LiquidationPrice = pm.calculateLiquidationPrice(market, position)

	// Store position
	pm.positions[user][symbol] = position

	// Update market open interest
	market.mu.Lock()
	market.OpenInterest += math.Abs(size)
	market.OpenInterestValue += notional
	market.mu.Unlock()

	return position, nil
}

// modifyPosition modifies an existing position
func (pm *PerpetualManager) modifyPosition(position *PerpPosition, deltaSize float64, price float64, leverage float64) (*PerpPosition, error) {
	market := pm.markets[position.Symbol]

	oldSize := position.Size
	newSize := oldSize + deltaSize

	// Check if closing or reversing position
	if oldSize*newSize < 0 {
		// Position reversal
		// First close existing position
		pm.closePositionInternal(position, math.Abs(oldSize), price)

		// Then open new position in opposite direction
		if newSize != 0 {
			side := Buy
			if newSize < 0 {
				side = Sell
			}
			return pm.OpenPosition(position.User, position.Symbol, side, math.Abs(newSize), leverage, position.IsIsolated)
		}
		return nil, nil
	}

	// Adding to position
	if math.Abs(newSize) > math.Abs(oldSize) {
		// Calculate new average entry price
		addSize := math.Abs(deltaSize)
		position.EntryPrice = (position.EntryPrice*math.Abs(oldSize) + price*addSize) / math.Abs(newSize)
	} else {
		// Reducing position - realize PnL
		closeSize := math.Abs(oldSize) - math.Abs(newSize)
		pnl := pm.calculatePnL(position, closeSize, price)
		position.RealizedPnL += pnl
	}

	// Update position
	position.Size = newSize
	position.MarkPrice = price
	position.UpdateTime = time.Now()

	// Recalculate margins
	notional := math.Abs(newSize) * price * market.ContractSize
	position.Notional = notional
	position.Margin = notional / leverage
	position.InitialMargin = notional * market.InitialMargin
	position.MaintenanceMargin = notional * market.MaintenanceMargin
	position.PositionValue = notional
	position.Leverage = leverage

	// Recalculate liquidation price
	position.LiquidationPrice = pm.calculateLiquidationPrice(market, position)

	// Update unrealized PnL
	position.UnrealizedPnL = pm.calculateUnrealizedPnL(position, price)

	// Update market open interest
	market.mu.Lock()
	market.OpenInterest += math.Abs(deltaSize)
	market.OpenInterestValue = market.OpenInterest * price * market.ContractSize
	market.mu.Unlock()

	return position, nil
}

// ClosePosition closes a perpetual position
func (pm *PerpetualManager) ClosePosition(user string, symbol string, size float64) (*PerpPosition, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	position, exists := pm.positions[user][symbol]
	if !exists {
		return nil, fmt.Errorf("position not found")
	}

	market := pm.markets[symbol]
	market.mu.RLock()
	closePrice := market.MarkPrice
	market.mu.RUnlock()

	// Close position
	return pm.closePositionInternal(position, size, closePrice)
}

// closePositionInternal handles position closing logic
func (pm *PerpetualManager) closePositionInternal(position *PerpPosition, size float64, price float64) (*PerpPosition, error) {
	if size > math.Abs(position.Size) {
		size = math.Abs(position.Size)
	}

	// Calculate PnL
	pnl := pm.calculatePnL(position, size, price)
	position.RealizedPnL += pnl

	// Update position
	if position.Size > 0 {
		position.Size -= size
	} else {
		position.Size += size
	}

	// If position fully closed
	if math.Abs(position.Size) < 1e-8 {
		// Remove position
		delete(pm.positions[position.User], position.Symbol)

		// Update market open interest
		market := pm.markets[position.Symbol]
		market.mu.Lock()
		market.OpenInterest -= size
		market.OpenInterestValue = market.OpenInterest * price * market.ContractSize
		market.mu.Unlock()

		return position, nil
	}

	// Update position metrics
	position.UpdateTime = time.Now()
	position.MarkPrice = price
	position.UnrealizedPnL = pm.calculateUnrealizedPnL(position, price)

	return position, nil
}

// calculatePnL calculates realized PnL for a position
func (pm *PerpetualManager) calculatePnL(position *PerpPosition, size float64, exitPrice float64) float64 {
	market := pm.markets[position.Symbol]

	if market.IsInverse {
		// Inverse perpetual PnL calculation
		if position.Size > 0 {
			// Long position
			return size * market.ContractSize * (1/position.EntryPrice - 1/exitPrice)
		} else {
			// Short position
			return size * market.ContractSize * (1/exitPrice - 1/position.EntryPrice)
		}
	} else {
		// Linear perpetual PnL calculation
		if position.Size > 0 {
			// Long position
			return size * market.ContractSize * (exitPrice - position.EntryPrice)
		} else {
			// Short position
			return size * market.ContractSize * (position.EntryPrice - exitPrice)
		}
	}
}

// calculateUnrealizedPnL calculates unrealized PnL
func (pm *PerpetualManager) calculateUnrealizedPnL(position *PerpPosition, markPrice float64) float64 {
	return pm.calculatePnL(position, math.Abs(position.Size), markPrice)
}

// calculateLiquidationPrice calculates the liquidation price for a position
func (pm *PerpetualManager) calculateLiquidationPrice(market *PerpetualMarket, position *PerpPosition) float64 {
	// Liquidation happens when margin ratio falls below maintenance margin
	// Margin Ratio = (Margin + Unrealized PnL) / Position Value

	maintenanceMarginRatio := market.MaintenanceMargin
	margin := position.Margin
	size := math.Abs(position.Size)
	entryPrice := position.EntryPrice
	contractSize := market.ContractSize

	if market.IsInverse {
		// Inverse perpetual liquidation price
		if position.Size > 0 {
			// Long position liquidates when price falls
			return entryPrice * margin / (margin + maintenanceMarginRatio*size*contractSize*entryPrice)
		} else {
			// Short position liquidates when price rises
			return entryPrice * margin / (margin - maintenanceMarginRatio*size*contractSize*entryPrice)
		}
	} else {
		// Linear perpetual liquidation price
		if position.Size > 0 {
			// Long position liquidates when price falls
			return entryPrice - (margin-maintenanceMarginRatio*size*contractSize*entryPrice)/(size*contractSize)
		} else {
			// Short position liquidates when price rises
			return entryPrice + (margin-maintenanceMarginRatio*size*contractSize*entryPrice)/(size*contractSize)
		}
	}
}

// ProcessFunding processes funding payments for all positions
func (pm *PerpetualManager) ProcessFunding() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	now := time.Now()

	for symbol, market := range pm.markets {
		market.mu.Lock()

		// Check if funding time
		if now.Before(market.NextFundingTime) {
			market.mu.Unlock()
			continue
		}

		// Calculate funding rate
		fundingRate := pm.calculateFundingRate(market)
		market.FundingRate = fundingRate

		// Process funding for all positions
		for user, positions := range pm.positions {
			if position, exists := positions[symbol]; exists {
				fundingPayment := pm.calculateFundingPayment(position, fundingRate, market)
				position.FundingPaid += fundingPayment
				position.LastFundingTime = now

				// Deduct/add funding from/to margin
				if position.IsIsolated {
					position.Margin -= fundingPayment
					// Check for liquidation
					if position.Margin < position.MaintenanceMargin {
						// Trigger liquidation
						pm.liquidatePosition(user, symbol)
					}
				}
			}
		}

		// Record funding payment
		market.FundingHistory = append(market.FundingHistory, FundingPayment{
			Time:       now,
			Rate:       fundingRate,
			MarkPrice:  market.MarkPrice,
			IndexPrice: market.IndexPrice,
		})

		// Limit history size
		if len(market.FundingHistory) > 1000 {
			market.FundingHistory = market.FundingHistory[len(market.FundingHistory)-500:]
		}

		// Update funding times
		market.LastFundingTime = now
		market.NextFundingTime = now.Add(market.FundingInterval)

		market.mu.Unlock()
	}

	return nil
}

// calculateFundingRate calculates the funding rate
func (pm *PerpetualManager) calculateFundingRate(market *PerpetualMarket) float64 {
	// Simplified funding rate calculation
	// Funding Rate = (Mark Price - Index Price) / Index Price / Funding Interval Count

	if market.IndexPrice == 0 {
		return 0
	}

	pricePremium := (market.MarkPrice - market.IndexPrice) / market.IndexPrice

	// Apply clamp to prevent extreme funding rates
	maxFundingRate := 0.001 // 0.1% max funding rate
	if pricePremium > maxFundingRate {
		pricePremium = maxFundingRate
	} else if pricePremium < -maxFundingRate {
		pricePremium = -maxFundingRate
	}

	// Adjust for funding interval (e.g., 8 hours = 3 times per day)
	intervalsPerDay := 24 * time.Hour / market.FundingInterval
	return pricePremium / float64(intervalsPerDay)
}

// calculateFundingPayment calculates funding payment for a position
func (pm *PerpetualManager) calculateFundingPayment(position *PerpPosition, fundingRate float64, market *PerpetualMarket) float64 {
	// Funding Payment = Position Size * Contract Size * Mark Price * Funding Rate
	// Long positions pay short positions when funding rate is positive
	// Short positions pay long positions when funding rate is negative

	payment := position.Size * market.ContractSize * market.MarkPrice * fundingRate
	return -payment // Negative because we deduct from position holder
}

// liquidatePosition liquidates a position
func (pm *PerpetualManager) liquidatePosition(user string, symbol string) error {
	position, exists := pm.positions[user][symbol]
	if !exists {
		return fmt.Errorf("position not found")
	}

	market := pm.markets[symbol]

	// Execute liquidation at mark price (in practice, would use liquidation engine)
	market.mu.RLock()
	liquidationPrice := market.MarkPrice
	market.mu.RUnlock()

	// Close position
	pm.closePositionInternal(position, math.Abs(position.Size), liquidationPrice)

	// Log liquidation event
	pm.engine.logEvent(Event{
		Type:      EventLiquidation,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user":             user,
			"symbol":           symbol,
			"size":             position.Size,
			"liquidationPrice": liquidationPrice,
			"loss":             position.RealizedPnL,
		},
	})

	return nil
}

// UpdateMarkPrice updates the mark price for a market
func (pm *PerpetualManager) UpdateMarkPrice(symbol string, price float64) error {
	pm.mu.RLock()
	market, exists := pm.markets[symbol]
	pm.mu.RUnlock()

	if !exists {
		return ErrPerpNotFound
	}

	market.mu.Lock()
	defer market.mu.Unlock()

	// Update price
	oldPrice := market.MarkPrice
	market.MarkPrice = price

	// Update 24h stats
	if market.High24h < price || market.High24h == 0 {
		market.High24h = price
	}
	if market.Low24h > price || market.Low24h == 0 {
		market.Low24h = price
	}

	// Check positions for liquidation
	go pm.checkLiquidations(symbol, price)

	// Log price update - oracle already manages its own price history internally
	// The PriceOracle.updateCurrentPrice method handles history management

	// Calculate price change percentage
	if oldPrice > 0 {
		_ = (price - oldPrice) / oldPrice * 100
	}

	return nil
}

// UpdateIndexPrice updates the index price for a market
func (pm *PerpetualManager) UpdateIndexPrice(symbol string, price float64) error {
	pm.mu.RLock()
	market, exists := pm.markets[symbol]
	pm.mu.RUnlock()

	if !exists {
		return ErrPerpNotFound
	}

	market.mu.Lock()
	market.IndexPrice = price
	market.mu.Unlock()

	return nil
}

// checkLiquidations checks all positions for liquidation
func (pm *PerpetualManager) checkLiquidations(symbol string, markPrice float64) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	market := pm.markets[symbol]
	if market == nil {
		return
	}

	// Check all positions for this symbol
	for user, positions := range pm.positions {
		if position, exists := positions[symbol]; exists {
			// Update position mark price
			position.MarkPrice = markPrice
			position.UnrealizedPnL = pm.calculateUnrealizedPnL(position, markPrice)

			// Check liquidation condition
			shouldLiquidate := false
			if position.Size > 0 {
				// Long position
				shouldLiquidate = markPrice <= position.LiquidationPrice
			} else {
				// Short position
				shouldLiquidate = markPrice >= position.LiquidationPrice
			}

			if shouldLiquidate {
				// Trigger liquidation
				go pm.liquidatePosition(user, symbol)
			}
		}
	}
}

// GetPosition returns a user's position for a symbol
func (pm *PerpetualManager) GetPosition(user string, symbol string) (*PerpPosition, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if pm.positions[user] == nil {
		return nil, fmt.Errorf("no positions for user")
	}

	position, exists := pm.positions[user][symbol]
	if !exists {
		return nil, fmt.Errorf("position not found")
	}

	return position, nil
}

// GetAllPositions returns all positions for a user
func (pm *PerpetualManager) GetAllPositions(user string) map[string]*PerpPosition {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return pm.positions[user]
}

// GetMarket returns market information
func (pm *PerpetualManager) GetMarket(symbol string) (*PerpetualMarket, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	market, exists := pm.markets[symbol]
	if !exists {
		return nil, ErrPerpNotFound
	}

	return market, nil
}

// GetAllMarkets returns all perpetual markets
func (pm *PerpetualManager) GetAllMarkets() map[string]*PerpetualMarket {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	markets := make(map[string]*PerpetualMarket)
	for k, v := range pm.markets {
		markets[k] = v
	}
	return markets
}

// AddMarginToPosition adds margin to an isolated position
func (pm *PerpetualManager) AddMarginToPosition(user string, symbol string, amount float64) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	position, exists := pm.positions[user][symbol]
	if !exists {
		return fmt.Errorf("position not found")
	}

	if !position.IsIsolated {
		return fmt.Errorf("can only add margin to isolated positions")
	}

	// Add margin
	position.Margin += amount
	position.UpdateTime = time.Now()

	// Recalculate liquidation price
	market := pm.markets[symbol]
	position.LiquidationPrice = pm.calculateLiquidationPrice(market, position)

	return nil
}

// RemoveMarginFromPosition removes margin from an isolated position
func (pm *PerpetualManager) RemoveMarginFromPosition(user string, symbol string, amount float64) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	position, exists := pm.positions[user][symbol]
	if !exists {
		return fmt.Errorf("position not found")
	}

	if !position.IsIsolated {
		return fmt.Errorf("can only remove margin from isolated positions")
	}

	// Check if removal would trigger liquidation
	newMargin := position.Margin - amount
	if newMargin < position.MaintenanceMargin {
		return fmt.Errorf("insufficient margin after removal")
	}

	// Remove margin
	position.Margin = newMargin
	position.UpdateTime = time.Now()

	// Recalculate liquidation price
	market := pm.markets[symbol]
	position.LiquidationPrice = pm.calculateLiquidationPrice(market, position)

	return nil
}

// SetLeverage sets the leverage for future positions
func (pm *PerpetualManager) SetLeverage(user string, symbol string, leverage float64) error {
	pm.mu.RLock()
	market, exists := pm.markets[symbol]
	pm.mu.RUnlock()

	if !exists {
		return ErrPerpNotFound
	}

	if leverage > market.MaxLeverage {
		return ErrExcessiveLeverage
	}

	// Store user leverage preference (would be stored in user settings)
	// This affects future positions, not existing ones

	return nil
}

// GetFundingHistory returns funding payment history for a market
func (pm *PerpetualManager) GetFundingHistory(symbol string, limit int) ([]FundingPayment, error) {
	pm.mu.RLock()
	market, exists := pm.markets[symbol]
	pm.mu.RUnlock()

	if !exists {
		return nil, ErrPerpNotFound
	}

	market.mu.RLock()
	defer market.mu.RUnlock()

	history := market.FundingHistory
	if limit > 0 && len(history) > limit {
		history = history[len(history)-limit:]
	}

	return history, nil
}
