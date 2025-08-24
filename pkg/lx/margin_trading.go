package lx

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync"
	"sync/atomic"
	"time"
)

// MarginAccount represents a user's margin trading account
type MarginAccount struct {
	UserID              string
	AccountType         MarginAccountType
	Balance             *big.Int // Base currency balance
	Equity              *big.Int // Total equity (balance + unrealized PnL)
	MarginUsed          *big.Int // Margin currently in use
	FreeMargin          *big.Int // Available margin
	MarginLevel         float64  // Equity / Margin Used (as percentage)
	Leverage            float64  // Current leverage
	MaxLeverage         float64  // Maximum allowed leverage
	Positions           map[string]*MarginPosition
	Orders              map[uint64]*Order
	CollateralAssets    map[string]*CollateralAsset
	BorrowedAmounts     map[string]*BorrowedAsset
	LiquidationPrice    float64
	MaintenanceMargin   float64
	InitialMargin       float64
	UnrealizedPnL       *big.Int
	RealizedPnL         *big.Int
	TotalVolume         *big.Int
	TradingFees         *big.Int
	BorrowingFees       *big.Int
	LastUpdate          time.Time
	MarginCallLevel     float64 // Margin level that triggers margin call
	LiquidationLevel    float64 // Margin level that triggers liquidation
	PortfolioMarginMode bool    // Use portfolio margining
	mu                  sync.RWMutex
}

// MarginAccountType represents the type of margin account
type MarginAccountType int

const (
	CrossMargin     MarginAccountType = iota // Share margin across all positions
	IsolatedMargin                           // Separate margin for each position
	PortfolioMargin                          // Advanced risk-based margining
)

// MarginPosition represents a leveraged position
type MarginPosition struct {
	ID               string
	Symbol           string
	Side             Side
	Size             float64
	EntryPrice       float64
	MarkPrice        float64
	LiquidationPrice float64
	Leverage         float64
	Margin           *big.Int
	UnrealizedPnL    *big.Int
	RealizedPnL      *big.Int
	Fees             *big.Int
	OpenTime         time.Time
	LastUpdate       time.Time
	StopLoss         float64
	TakeProfit       float64
	TrailingStop     float64
	ReduceOnly       bool
	Isolated         bool // Position-specific margin
	CollateralAsset  string
	FundingPaid      *big.Int
}

// CollateralAsset represents an asset used as collateral
type CollateralAsset struct {
	Asset       string
	Amount      *big.Int
	ValueUSD    *big.Int
	Haircut     float64 // Discount applied to collateral value
	LoanToValue float64 // Maximum borrowing against this collateral
	Locked      *big.Int
	Available   *big.Int
	LastUpdate  time.Time
}

// BorrowedAsset represents a borrowed asset
type BorrowedAsset struct {
	Asset           string
	Amount          *big.Int
	ValueUSD        *big.Int
	InterestRate    float64 // Annual interest rate
	AccruedInterest *big.Int
	BorrowTime      time.Time
	LastUpdate      time.Time
}

// MarginEngine manages all margin trading operations
type MarginEngine struct {
	Accounts          map[string]*MarginAccount
	LendingPool       *LendingPool
	RiskEngine        *RiskEngine
	Oracle            *PriceOracle
	LiquidationEngine *LiquidationEngine
	InsuranceFund     *big.Int
	TotalBorrowed     map[string]*big.Int
	TotalCollateral   map[string]*big.Int
	InterestRates     map[string]float64
	MaxLeverageTable  map[string]float64 // Per-asset max leverage
	MaintenanceMargin map[string]float64 // Per-asset maintenance margin
	InitialMargin     map[string]float64 // Per-asset initial margin
	mu                sync.RWMutex
}

// NewMarginEngine creates a new margin trading engine
func NewMarginEngine(oracle *PriceOracle, riskEngine *RiskEngine) *MarginEngine {
	return &MarginEngine{
		Accounts:          make(map[string]*MarginAccount),
		LendingPool:       NewLendingPool(),
		RiskEngine:        riskEngine,
		Oracle:            oracle,
		LiquidationEngine: NewLiquidationEngine(),
		InsuranceFund:     big.NewInt(0),
		TotalBorrowed:     make(map[string]*big.Int),
		TotalCollateral:   make(map[string]*big.Int),
		InterestRates:     make(map[string]float64),
		MaxLeverageTable:  initMaxLeverageTable(),
		MaintenanceMargin: initMaintenanceMarginTable(),
		InitialMargin:     initInitialMarginTable(),
	}
}

// CreateMarginAccount creates a new margin account
func (me *MarginEngine) CreateMarginAccount(userID string, accountType MarginAccountType) (*MarginAccount, error) {
	me.mu.Lock()
	defer me.mu.Unlock()

	if _, exists := me.Accounts[userID]; exists {
		return nil, errors.New("account already exists")
	}

	maxLeverage := 10.0 // Default max leverage
	if accountType == IsolatedMargin {
		maxLeverage = 20.0
	} else if accountType == PortfolioMargin {
		maxLeverage = 100.0
	}

	account := &MarginAccount{
		UserID:              userID,
		AccountType:         accountType,
		Balance:             big.NewInt(0),
		Equity:              big.NewInt(0),
		MarginUsed:          big.NewInt(0),
		FreeMargin:          big.NewInt(0),
		MaxLeverage:         maxLeverage,
		Positions:           make(map[string]*MarginPosition),
		Orders:              make(map[uint64]*Order),
		CollateralAssets:    make(map[string]*CollateralAsset),
		BorrowedAmounts:     make(map[string]*BorrowedAsset),
		UnrealizedPnL:       big.NewInt(0),
		RealizedPnL:         big.NewInt(0),
		TotalVolume:         big.NewInt(0),
		TradingFees:         big.NewInt(0),
		BorrowingFees:       big.NewInt(0),
		MaintenanceMargin:   0.05, // 5% maintenance margin
		InitialMargin:       0.10, // 10% initial margin
		MarginCallLevel:     120,  // Margin call at 120%
		LiquidationLevel:    100,  // Liquidation at 100%
		PortfolioMarginMode: accountType == PortfolioMargin,
		LastUpdate:          time.Now(),
	}

	me.Accounts[userID] = account
	return account, nil
}

// GetAccount returns an existing margin account
func (me *MarginEngine) GetAccount(userID string) *MarginAccount {
	me.mu.RLock()
	defer me.mu.RUnlock()

	return me.Accounts[userID]
}

// DepositCollateral deposits collateral into margin account
func (me *MarginEngine) DepositCollateral(userID, asset string, amount *big.Int) error {
	me.mu.Lock()
	defer me.mu.Unlock()

	account, exists := me.Accounts[userID]
	if !exists {
		return errors.New("account not found")
	}

	collateral, exists := account.CollateralAssets[asset]
	if !exists {
		collateral = &CollateralAsset{
			Asset:       asset,
			Amount:      big.NewInt(0),
			Available:   big.NewInt(0),
			Locked:      big.NewInt(0),
			Haircut:     me.getAssetHaircut(asset),
			LoanToValue: me.getAssetLTV(asset),
			LastUpdate:  time.Now(),
		}
		account.CollateralAssets[asset] = collateral
	}

	// Update collateral
	collateral.Amount.Add(collateral.Amount, amount)
	collateral.Available.Add(collateral.Available, amount)

	// Update collateral value
	price := me.Oracle.GetPrice(asset)
	collateral.ValueUSD = new(big.Int).Mul(collateral.Amount, big.NewInt(int64(price)))

	// Update total collateral
	if me.TotalCollateral[asset] == nil {
		me.TotalCollateral[asset] = big.NewInt(0)
	}
	me.TotalCollateral[asset].Add(me.TotalCollateral[asset], amount)

	// Recalculate account metrics
	me.updateAccountMetrics(account)

	return nil
}

// OpenPosition opens a new leveraged position
func (me *MarginEngine) OpenPosition(userID string, order *Order, leverage float64) (*MarginPosition, error) {
	me.mu.Lock()
	defer me.mu.Unlock()

	account, exists := me.Accounts[userID]
	if !exists {
		return nil, errors.New("account not found")
	}

	// Check leverage limits
	maxLeverage := me.getMaxLeverage(order.Symbol, account.AccountType)
	if leverage > maxLeverage {
		return nil, fmt.Errorf("leverage exceeds maximum: %f > %f", leverage, maxLeverage)
	}

	// Calculate required margin
	positionValue := order.Price * order.Size
	requiredMargin := big.NewInt(int64(positionValue / leverage))

	// Check available margin
	if account.FreeMargin.Cmp(requiredMargin) < 0 {
		return nil, errors.New("insufficient margin")
	}

	// Create position
	position := &MarginPosition{
		ID:              generatePositionID(),
		Symbol:          order.Symbol,
		Side:            order.Side,
		Size:            order.Size,
		EntryPrice:      order.Price,
		MarkPrice:       order.Price,
		Leverage:        leverage,
		Margin:          requiredMargin,
		UnrealizedPnL:   big.NewInt(0),
		RealizedPnL:     big.NewInt(0),
		Fees:            big.NewInt(0),
		OpenTime:        time.Now(),
		LastUpdate:      time.Now(),
		Isolated:        account.AccountType == IsolatedMargin,
		CollateralAsset: "USDT", // Default collateral
		FundingPaid:     big.NewInt(0),
	}

	// Calculate liquidation price
	position.LiquidationPrice = me.calculateLiquidationPrice(position, account)

	// Update account
	account.Positions[position.ID] = position
	account.MarginUsed.Add(account.MarginUsed, requiredMargin)
	account.FreeMargin.Sub(account.FreeMargin, requiredMargin)

	// If borrowing is needed
	borrowAmount := new(big.Int).Sub(big.NewInt(int64(positionValue)), requiredMargin)
	if borrowAmount.Cmp(big.NewInt(0)) > 0 {
		if err := me.borrowForPosition(account, order.Symbol, borrowAmount); err != nil {
			return nil, err
		}
	}

	// Update metrics
	me.updateAccountMetrics(account)

	return position, nil
}

// ClosePosition closes a leveraged position
func (me *MarginEngine) ClosePosition(userID, positionID string, size float64) error {
	me.mu.Lock()
	defer me.mu.Unlock()

	account, exists := me.Accounts[userID]
	if !exists {
		return errors.New("account not found")
	}

	position, exists := account.Positions[positionID]
	if !exists {
		return errors.New("position not found")
	}

	// Partial or full close
	closeSize := math.Min(size, position.Size)
	closeRatio := closeSize / position.Size

	// Calculate PnL
	currentPrice := me.Oracle.GetPrice(position.Symbol)
	var pnl float64
	if position.Side == Buy {
		pnl = (currentPrice - position.EntryPrice) * closeSize
	} else {
		pnl = (position.EntryPrice - currentPrice) * closeSize
	}

	pnlBig := big.NewInt(int64(pnl))
	position.RealizedPnL.Add(position.RealizedPnL, pnlBig)
	account.RealizedPnL.Add(account.RealizedPnL, pnlBig)

	// Release margin
	marginToRelease := new(big.Int).Mul(position.Margin, big.NewInt(int64(closeRatio*100)))
	marginToRelease.Div(marginToRelease, big.NewInt(100))

	account.MarginUsed.Sub(account.MarginUsed, marginToRelease)
	account.FreeMargin.Add(account.FreeMargin, marginToRelease)

	// Repay borrowed amounts if any
	if borrowed, exists := account.BorrowedAmounts[position.Symbol]; exists {
		repayAmount := new(big.Int).Mul(borrowed.Amount, big.NewInt(int64(closeRatio*100)))
		repayAmount.Div(repayAmount, big.NewInt(100))
		me.repayBorrowed(account, position.Symbol, repayAmount)
	}

	// Update or remove position
	if closeSize >= position.Size {
		delete(account.Positions, positionID)
	} else {
		position.Size -= closeSize
		position.Margin.Sub(position.Margin, marginToRelease)
		position.LastUpdate = time.Now()
	}

	// Update account equity
	account.Equity.Add(account.Balance, pnlBig)

	// Update metrics
	me.updateAccountMetrics(account)

	return nil
}

// ModifyLeverage adjusts the leverage of a position
func (me *MarginEngine) ModifyLeverage(userID, positionID string, newLeverage float64) error {
	me.mu.Lock()
	defer me.mu.Unlock()

	account, exists := me.Accounts[userID]
	if !exists {
		return errors.New("account not found")
	}

	position, exists := account.Positions[positionID]
	if !exists {
		return errors.New("position not found")
	}

	// Check new leverage limits
	maxLeverage := me.getMaxLeverage(position.Symbol, account.AccountType)
	if newLeverage > maxLeverage {
		return fmt.Errorf("leverage exceeds maximum: %f", maxLeverage)
	}

	// Calculate new margin requirement
	positionValue := position.EntryPrice * position.Size
	newMargin := big.NewInt(int64(positionValue / newLeverage))
	marginDelta := new(big.Int).Sub(newMargin, position.Margin)

	// Check if we need more margin
	if marginDelta.Cmp(big.NewInt(0)) > 0 {
		if account.FreeMargin.Cmp(marginDelta) < 0 {
			return errors.New("insufficient free margin")
		}
		account.FreeMargin.Sub(account.FreeMargin, marginDelta)
		account.MarginUsed.Add(account.MarginUsed, marginDelta)
	} else {
		// Release margin
		marginDelta.Neg(marginDelta)
		account.FreeMargin.Add(account.FreeMargin, marginDelta)
		account.MarginUsed.Sub(account.MarginUsed, marginDelta)
	}

	// Update position
	position.Leverage = newLeverage
	position.Margin = newMargin
	position.LiquidationPrice = me.calculateLiquidationPrice(position, account)
	position.LastUpdate = time.Now()

	// Update metrics
	me.updateAccountMetrics(account)

	return nil
}

// calculateLiquidationPrice calculates the liquidation price for a position
func (me *MarginEngine) calculateLiquidationPrice(position *MarginPosition, account *MarginAccount) float64 {
	maintenanceMarginRate := me.MaintenanceMargin[position.Symbol]
	if maintenanceMarginRate == 0 {
		maintenanceMarginRate = 0.05 // Default 5%
	}

	if position.Side == Buy {
		// Long position: Price at which losses equal available margin
		return position.EntryPrice * (1 - 1/position.Leverage + maintenanceMarginRate)
	} else {
		// Short position
		return position.EntryPrice * (1 + 1/position.Leverage - maintenanceMarginRate)
	}
}

// updateAccountMetrics updates account metrics
func (me *MarginEngine) updateAccountMetrics(account *MarginAccount) {
	account.mu.Lock()
	defer account.mu.Unlock()

	// Calculate total collateral value
	totalCollateralValue := big.NewInt(0)
	for _, collateral := range account.CollateralAssets {
		// Apply haircut
		adjustedValue := new(big.Float).SetInt(collateral.ValueUSD)
		adjustedValue.Mul(adjustedValue, big.NewFloat(1-collateral.Haircut))
		adjustedValueInt, _ := adjustedValue.Int(nil)
		totalCollateralValue.Add(totalCollateralValue, adjustedValueInt)
	}

	// Calculate unrealized PnL
	totalUnrealizedPnL := big.NewInt(0)
	for _, position := range account.Positions {
		currentPrice := me.Oracle.GetPrice(position.Symbol)
		var pnl float64
		if position.Side == Buy {
			pnl = (currentPrice - position.EntryPrice) * position.Size
		} else {
			pnl = (position.EntryPrice - currentPrice) * position.Size
		}
		position.UnrealizedPnL = big.NewInt(int64(pnl))
		totalUnrealizedPnL.Add(totalUnrealizedPnL, position.UnrealizedPnL)
	}
	account.UnrealizedPnL = totalUnrealizedPnL

	// Calculate equity
	account.Equity = new(big.Int).Add(totalCollateralValue, totalUnrealizedPnL)

	// Calculate margin level
	if account.MarginUsed.Cmp(big.NewInt(0)) > 0 {
		equityFloat := new(big.Float).SetInt(account.Equity)
		marginUsedFloat := new(big.Float).SetInt(account.MarginUsed)
		marginLevel := new(big.Float).Quo(equityFloat, marginUsedFloat)
		marginLevel.Mul(marginLevel, big.NewFloat(100))
		account.MarginLevel, _ = marginLevel.Float64()
	} else {
		account.MarginLevel = 0
	}

	// Calculate current leverage
	totalPositionValue := big.NewInt(0)
	for _, position := range account.Positions {
		posValue := big.NewInt(int64(position.EntryPrice * position.Size))
		totalPositionValue.Add(totalPositionValue, posValue)
	}

	if account.Equity.Cmp(big.NewInt(0)) > 0 {
		posValFloat := new(big.Float).SetInt(totalPositionValue)
		equityFloat := new(big.Float).SetInt(account.Equity)
		leverageFloat := new(big.Float).Quo(posValFloat, equityFloat)
		account.Leverage, _ = leverageFloat.Float64()
	}

	account.LastUpdate = time.Now()
}

// CheckLiquidations checks and processes liquidations
func (me *MarginEngine) CheckLiquidations() {
	me.mu.Lock()
	defer me.mu.Unlock()

	for userID, account := range me.Accounts {
		// Update metrics first
		me.updateAccountMetrics(account)

		// Check margin level
		if account.MarginLevel > 0 && account.MarginLevel <= account.LiquidationLevel {
			me.liquidateAccount(userID, account)
		}

		// Check individual positions for isolated margin
		if account.AccountType == IsolatedMargin {
			for posID, position := range account.Positions {
				currentPrice := me.Oracle.GetPrice(position.Symbol)
				if me.shouldLiquidatePosition(position, currentPrice) {
					me.liquidatePosition(userID, posID, position)
				}
			}
		}
	}
}

// shouldLiquidatePosition checks if a position should be liquidated
func (me *MarginEngine) shouldLiquidatePosition(position *MarginPosition, currentPrice float64) bool {
	if position.Side == Buy {
		return currentPrice <= position.LiquidationPrice
	}
	return currentPrice >= position.LiquidationPrice
}

// liquidatePosition liquidates a specific position
func (me *MarginEngine) liquidatePosition(userID, positionID string, position *MarginPosition) {
	// Execute liquidation order
	liquidationOrder := &Order{
		Symbol: position.Symbol,
		Side:   oppositeSide(position.Side),
		Type:   Market,
		Size:   position.Size,
		User:   "liquidation_engine",
	}

	// Process liquidation
	me.LiquidationEngine.ProcessLiquidation(userID, position, liquidationOrder)

	// Remove position
	account := me.Accounts[userID]
	delete(account.Positions, positionID)

	// Release margin to insurance fund if negative
	if position.UnrealizedPnL.Cmp(big.NewInt(0)) < 0 {
		loss := new(big.Int).Neg(position.UnrealizedPnL)
		if loss.Cmp(position.Margin) > 0 {
			insuranceClaim := new(big.Int).Sub(loss, position.Margin)
			me.InsuranceFund.Sub(me.InsuranceFund, insuranceClaim)
		}
	}
}

// liquidateAccount liquidates all positions in an account
func (me *MarginEngine) liquidateAccount(userID string, account *MarginAccount) {
	for posID, position := range account.Positions {
		me.liquidatePosition(userID, posID, position)
	}

	// Reset account
	account.MarginUsed = big.NewInt(0)
	account.FreeMargin = account.Equity
	account.Leverage = 0
}

// borrowForPosition handles borrowing for leveraged positions
func (me *MarginEngine) borrowForPosition(account *MarginAccount, asset string, amount *big.Int) error {
	// Check lending pool availability
	available := me.LendingPool.GetAvailable(asset)
	if available.Cmp(amount) < 0 {
		return errors.New("insufficient liquidity in lending pool")
	}

	// Get interest rate
	rate := me.LendingPool.GetBorrowRate(asset)

	// Create or update borrowed asset
	borrowed, exists := account.BorrowedAmounts[asset]
	if !exists {
		borrowed = &BorrowedAsset{
			Asset:           asset,
			Amount:          big.NewInt(0),
			AccruedInterest: big.NewInt(0),
			InterestRate:    rate,
			BorrowTime:      time.Now(),
			LastUpdate:      time.Now(),
		}
		account.BorrowedAmounts[asset] = borrowed
	}

	// Update borrowed amount
	borrowed.Amount.Add(borrowed.Amount, amount)

	// Update lending pool
	me.LendingPool.Borrow(asset, amount)

	// Update total borrowed
	if me.TotalBorrowed[asset] == nil {
		me.TotalBorrowed[asset] = big.NewInt(0)
	}
	me.TotalBorrowed[asset].Add(me.TotalBorrowed[asset], amount)

	return nil
}

// repayBorrowed handles repayment of borrowed assets
func (me *MarginEngine) repayBorrowed(account *MarginAccount, asset string, amount *big.Int) error {
	borrowed, exists := account.BorrowedAmounts[asset]
	if !exists {
		return errors.New("no borrowed amount found")
	}

	// Calculate accrued interest
	interest := me.calculateAccruedInterest(borrowed)
	totalRepay := new(big.Int).Add(amount, interest)

	// Update borrowed amount
	borrowed.Amount.Sub(borrowed.Amount, amount)
	if borrowed.Amount.Cmp(big.NewInt(0)) <= 0 {
		delete(account.BorrowedAmounts, asset)
	} else {
		borrowed.LastUpdate = time.Now()
	}

	// Update lending pool with total repay amount
	me.LendingPool.Repay(asset, totalRepay, interest)

	// Update total borrowed
	me.TotalBorrowed[asset].Sub(me.TotalBorrowed[asset], amount)

	// Add interest to borrowing fees
	account.BorrowingFees.Add(account.BorrowingFees, interest)

	return nil
}

// calculateAccruedInterest calculates interest on borrowed amount
func (me *MarginEngine) calculateAccruedInterest(borrowed *BorrowedAsset) *big.Int {
	// Simple interest calculation
	timeDiff := time.Since(borrowed.LastUpdate).Hours()
	dailyRate := borrowed.InterestRate / 365
	hourlyRate := dailyRate / 24

	interest := new(big.Float).SetInt(borrowed.Amount)
	interest.Mul(interest, big.NewFloat(hourlyRate*timeDiff))

	interestInt, _ := interest.Int(nil)
	return interestInt
}

// Helper functions
func (me *MarginEngine) getMaxLeverage(symbol string, accountType MarginAccountType) float64 {
	if leverage, exists := me.MaxLeverageTable[symbol]; exists {
		if accountType == PortfolioMargin {
			return leverage * 2 // Double leverage for portfolio margin
		}
		return leverage
	}
	return 10.0 // Default
}

func (me *MarginEngine) getAssetHaircut(asset string) float64 {
	// Haircut based on asset volatility
	haircuts := map[string]float64{
		"BTC":  0.10, // 10% haircut
		"ETH":  0.15, // 15% haircut
		"USDT": 0.00, // No haircut for stablecoins
		"USDC": 0.00,
	}

	if haircut, exists := haircuts[asset]; exists {
		return haircut
	}
	return 0.20 // Default 20% haircut
}

func (me *MarginEngine) getAssetLTV(asset string) float64 {
	// Loan-to-value ratios
	ltvs := map[string]float64{
		"BTC":  0.80, // 80% LTV
		"ETH":  0.75, // 75% LTV
		"USDT": 0.95, // 95% LTV for stablecoins
		"USDC": 0.95,
	}

	if ltv, exists := ltvs[asset]; exists {
		return ltv
	}
	return 0.50 // Default 50% LTV
}

func initMaxLeverageTable() map[string]float64 {
	return map[string]float64{
		"BTC-USDT":   100,
		"ETH-USDT":   100,
		"BNB-USDT":   50,
		"SOL-USDT":   50,
		"AVAX-USDT":  50,
		"MATIC-USDT": 20,
		"ARB-USDT":   20,
		"OP-USDT":    20,
	}
}

func initMaintenanceMarginTable() map[string]float64 {
	return map[string]float64{
		"BTC-USDT":   0.005, // 0.5%
		"ETH-USDT":   0.01,  // 1%
		"BNB-USDT":   0.02,  // 2%
		"SOL-USDT":   0.025, // 2.5%
		"AVAX-USDT":  0.025,
		"MATIC-USDT": 0.05, // 5%
		"ARB-USDT":   0.05,
		"OP-USDT":    0.05,
	}
}

func initInitialMarginTable() map[string]float64 {
	return map[string]float64{
		"BTC-USDT":   0.01, // 1%
		"ETH-USDT":   0.02, // 2%
		"BNB-USDT":   0.04, // 4%
		"SOL-USDT":   0.05, // 5%
		"AVAX-USDT":  0.05,
		"MATIC-USDT": 0.10, // 10%
		"ARB-USDT":   0.10,
		"OP-USDT":    0.10,
	}
}

var positionIDCounter uint64

func generatePositionID() string {
	atomic.AddUint64(&positionIDCounter, 1)
	return fmt.Sprintf("pos_%d_%d", time.Now().UnixNano(), positionIDCounter)
}

func oppositeSide(side Side) Side {
	if side == Buy {
		return Sell
	}
	return Buy
}
