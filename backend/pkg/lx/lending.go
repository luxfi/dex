package lx

import (
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// LendingManager manages all lending pools
type LendingManager struct {
	pools         map[string]*LendingPool
	loans         map[string]map[string]*Loan // user -> asset -> loan
	liquidations  []Liquidation
	engine        *TradingEngine
	oracles       map[string]*PriceOracle
	mu            sync.RWMutex
	interestAccum map[string]*InterestAccumulator
}

// NewLendingManager creates a new lending manager
func NewLendingManager(engine *TradingEngine) *LendingManager {
	return &LendingManager{
		pools:         make(map[string]*LendingPool),
		loans:         make(map[string]map[string]*Loan),
		liquidations:  make([]Liquidation, 0),
		engine:        engine,
		oracles:       make(map[string]*PriceOracle),
		interestAccum: make(map[string]*InterestAccumulator),
	}
}

// Enhanced LendingPool with advanced features
type LendingPool struct {
	Asset              string
	TotalSupply        *big.Int
	TotalBorrowed      *big.Int
	TotalReserves      *big.Int
	SupplyAPY          float64
	BorrowAPY          float64
	UtilizationRate    float64
	ReserveFactor      float64 // Protocol fee
	LiquidationBonus   float64 // Incentive for liquidators
	CollateralFactor   float64 // LTV ratio
	Suppliers          map[string]*Supply
	Borrowers          map[string]*Loan
	InterestRateModel  InterestRateModel
	LastUpdateTime     time.Time
	AccruedInterest    *big.Int
	SupplyIndex        *big.Int // For interest calculation
	BorrowIndex        *big.Int
	MinSupply          *big.Int
	MaxSupply          *big.Int
	MinBorrow          *big.Int
	MaxBorrow          *big.Int
	State              PoolState
	mu                 sync.RWMutex
}

// PoolState represents the state of a lending pool
type PoolState int

const (
	PoolStateActive PoolState = iota
	PoolStatePaused
	PoolStateFrozen // No new borrows
	PoolStateClosed
)

// Supply represents a supply position
type Supply struct {
	User              string
	Asset             string
	Amount            *big.Int
	Shares            *big.Int // For interest-bearing tokens
	SupplyTime        time.Time
	LastUpdateTime    time.Time
	AccruedInterest   *big.Int
	Index             *big.Int // Interest index at supply time
}

// Enhanced Loan with more details
type Loan struct {
	Borrower          string
	Asset             string
	Amount            *big.Int
	Collateral        map[string]*Collateral // Multiple collateral assets
	BorrowTime        time.Time
	LastUpdateTime    time.Time
	AccruedInterest   *big.Int
	HealthFactor      float64
	LiquidationPrice  map[string]float64 // Per collateral asset
	Index             *big.Int           // Interest index at borrow time
	IsIsolated        bool
	MaxLTV            float64
}

// Collateral represents collateral for a loan
type Collateral struct {
	Asset            string
	Amount           *big.Int
	Value            *big.Int // USD value
	CollateralFactor float64
	Price            float64
}

// Liquidation record
type Liquidation struct {
	User             string
	Asset            string
	DebtAmount       *big.Int
	CollateralAsset  string
	CollateralAmount *big.Int
	Liquidator       string
	Timestamp        time.Time
	Penalty          *big.Int
}

// InterestRateModel interface for different interest models
type InterestRateModel interface {
	CalculateRates(utilization float64) (supplyAPY, borrowAPY float64)
	GetBaseRate() float64
	GetSlope1() float64
	GetSlope2() float64
	GetOptimalUtilization() float64
}

// JumpRateModel implements a jump rate model (like Compound)
type JumpRateModel struct {
	BaseRate           float64
	Slope1             float64
	Slope2             float64
	OptimalUtilization float64
}

// InterestAccumulator tracks interest accumulation
type InterestAccumulator struct {
	LastAccrualTime time.Time
	TotalSupplyInterest *big.Int
	TotalBorrowInterest *big.Int
}

// CreatePool creates a new lending pool
func (lm *LendingManager) CreatePool(config LendingPoolConfig) (*LendingPool, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if _, exists := lm.pools[config.Asset]; exists {
		return nil, fmt.Errorf("pool for %s already exists", config.Asset)
	}

	pool := &LendingPool{
		Asset:             config.Asset,
		TotalSupply:       big.NewInt(0),
		TotalBorrowed:     big.NewInt(0),
		TotalReserves:     big.NewInt(0),
		ReserveFactor:     config.ReserveFactor,
		LiquidationBonus:  config.LiquidationBonus,
		CollateralFactor:  config.CollateralFactor,
		Suppliers:         make(map[string]*Supply),
		Borrowers:         make(map[string]*Loan),
		InterestRateModel: config.InterestRateModel,
		LastUpdateTime:    time.Now(),
		AccruedInterest:   big.NewInt(0),
		SupplyIndex:       big.NewInt(1e18), // Start at 1.0 (scaled)
		BorrowIndex:       big.NewInt(1e18),
		MinSupply:         config.MinSupply,
		MaxSupply:         config.MaxSupply,
		MinBorrow:         config.MinBorrow,
		MaxBorrow:         config.MaxBorrow,
		State:             PoolStateActive,
	}

	// Initialize interest accumulator
	lm.interestAccum[config.Asset] = &InterestAccumulator{
		LastAccrualTime:     time.Now(),
		TotalSupplyInterest: big.NewInt(0),
		TotalBorrowInterest: big.NewInt(0),
	}

	// Initialize oracle
	lm.oracles[config.Asset] = &PriceOracle{
		Symbol:       config.Asset,
		Source:       config.OracleSource,
		PriceHistory: make([]PricePoint, 0),
	}

	lm.pools[config.Asset] = pool
	return pool, nil
}

// LendingPoolConfig configuration for a lending pool
type LendingPoolConfig struct {
	Asset             string
	ReserveFactor     float64
	LiquidationBonus  float64
	CollateralFactor  float64
	InterestRateModel InterestRateModel
	MinSupply         *big.Int
	MaxSupply         *big.Int
	MinBorrow         *big.Int
	MaxBorrow         *big.Int
	OracleSource      string
}

// Supply adds assets to the lending pool
func (lm *LendingManager) Supply(user string, asset string, amount *big.Int) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	pool, exists := lm.pools[asset]
	if !exists {
		return fmt.Errorf("pool for %s not found", asset)
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()

	// Check pool state
	if pool.State != PoolStateActive {
		return fmt.Errorf("pool is not active")
	}

	// Check limits
	if amount.Cmp(pool.MinSupply) < 0 {
		return fmt.Errorf("supply below minimum")
	}

	newTotal := new(big.Int).Add(pool.TotalSupply, amount)
	if pool.MaxSupply != nil && newTotal.Cmp(pool.MaxSupply) > 0 {
		return fmt.Errorf("pool supply limit exceeded")
	}

	// Accrue interest first
	lm.accrueInterest(pool)

	// Calculate shares to mint (using exchange rate)
	shares := lm.calculateSupplyShares(pool, amount)

	// Get or create supply position
	supply, exists := pool.Suppliers[user]
	if !exists {
		supply = &Supply{
			User:            user,
			Asset:           asset,
			Amount:          big.NewInt(0),
			Shares:          big.NewInt(0),
			SupplyTime:      time.Now(),
			AccruedInterest: big.NewInt(0),
			Index:           new(big.Int).Set(pool.SupplyIndex),
		}
		pool.Suppliers[user] = supply
	}

	// Update supply position
	supply.Amount.Add(supply.Amount, amount)
	supply.Shares.Add(supply.Shares, shares)
	supply.LastUpdateTime = time.Now()

	// Update pool totals
	pool.TotalSupply.Add(pool.TotalSupply, amount)

	// Update interest rates
	pool.updateInterestRates()

	// Log event
	lm.engine.logEvent(Event{
		Type:      EventLending,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user":   user,
			"asset":  asset,
			"amount": amount.String(),
			"shares": shares.String(),
		},
	})

	return nil
}

// Withdraw removes assets from the lending pool
func (lm *LendingManager) Withdraw(user string, asset string, amount *big.Int) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	pool, exists := lm.pools[asset]
	if !exists {
		return fmt.Errorf("pool for %s not found", asset)
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()

	supply, exists := pool.Suppliers[user]
	if !exists {
		return fmt.Errorf("no supply position found")
	}

	// Accrue interest first
	lm.accrueInterest(pool)

	// Calculate current value of shares
	currentValue := lm.calculateSupplyValue(pool, supply.Shares)
	if amount.Cmp(currentValue) > 0 {
		return fmt.Errorf("insufficient balance")
	}

	// Check liquidity
	availableLiquidity := new(big.Int).Sub(pool.TotalSupply, pool.TotalBorrowed)
	if amount.Cmp(availableLiquidity) > 0 {
		return fmt.Errorf("insufficient liquidity in pool")
	}

	// Calculate shares to burn
	sharesToBurn := lm.calculateSharesToBurn(pool, amount)

	// Update supply position
	supply.Shares.Sub(supply.Shares, sharesToBurn)
	supply.Amount.Sub(supply.Amount, amount)
	supply.LastUpdateTime = time.Now()

	// Calculate and record interest earned
	interestEarned := new(big.Int).Sub(currentValue, supply.Amount)
	supply.AccruedInterest.Add(supply.AccruedInterest, interestEarned)

	// Remove if empty
	if supply.Shares.Cmp(big.NewInt(0)) == 0 {
		delete(pool.Suppliers, user)
	}

	// Update pool totals
	pool.TotalSupply.Sub(pool.TotalSupply, amount)

	// Update interest rates
	pool.updateInterestRates()

	return nil
}

// Borrow takes a loan from the pool
func (lm *LendingManager) Borrow(user string, asset string, amount *big.Int, collateralAssets map[string]*big.Int) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	pool, exists := lm.pools[asset]
	if !exists {
		return fmt.Errorf("pool for %s not found", asset)
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()

	// Check pool state
	if pool.State == PoolStateFrozen || pool.State == PoolStateClosed {
		return fmt.Errorf("pool does not allow new borrows")
	}

	// Check limits
	if amount.Cmp(pool.MinBorrow) < 0 {
		return fmt.Errorf("borrow below minimum")
	}

	if pool.MaxBorrow != nil && amount.Cmp(pool.MaxBorrow) > 0 {
		return fmt.Errorf("borrow exceeds maximum")
	}

	// Check available liquidity
	availableLiquidity := new(big.Int).Sub(pool.TotalSupply, pool.TotalBorrowed)
	if amount.Cmp(availableLiquidity) > 0 {
		return fmt.Errorf("insufficient liquidity")
	}

	// Validate collateral
	collateralValue := lm.calculateCollateralValue(collateralAssets)
	borrowValue := lm.getAssetValue(asset, amount)
	
	// Check collateral ratio
	requiredCollateral := new(big.Int).Mul(borrowValue, big.NewInt(int64(1/pool.CollateralFactor*100)))
	requiredCollateral.Div(requiredCollateral, big.NewInt(100))
	
	if collateralValue.Cmp(requiredCollateral) < 0 {
		return fmt.Errorf("insufficient collateral")
	}

	// Accrue interest
	lm.accrueInterest(pool)

	// Create or update loan
	if lm.loans[user] == nil {
		lm.loans[user] = make(map[string]*Loan)
	}

	loan, exists := lm.loans[user][asset]
	if !exists {
		loan = &Loan{
			Borrower:         user,
			Asset:            asset,
			Amount:           big.NewInt(0),
			Collateral:       make(map[string]*Collateral),
			BorrowTime:       time.Now(),
			AccruedInterest:  big.NewInt(0),
			Index:            new(big.Int).Set(pool.BorrowIndex),
			LiquidationPrice: make(map[string]float64),
		}
		lm.loans[user][asset] = loan
		pool.Borrowers[user] = loan
	}

	// Update loan
	loan.Amount.Add(loan.Amount, amount)
	loan.LastUpdateTime = time.Now()

	// Update collateral
	for collAsset, collAmount := range collateralAssets {
		if loan.Collateral[collAsset] == nil {
			loan.Collateral[collAsset] = &Collateral{
				Asset:  collAsset,
				Amount: big.NewInt(0),
			}
		}
		loan.Collateral[collAsset].Amount.Add(loan.Collateral[collAsset].Amount, collAmount)
		loan.Collateral[collAsset].CollateralFactor = lm.pools[collAsset].CollateralFactor
	}

	// Calculate health factor and liquidation prices
	loan.HealthFactor = lm.calculateHealthFactor(loan)
	loan.LiquidationPrice = lm.calculateLiquidationPrices(loan)

	// Update pool totals
	pool.TotalBorrowed.Add(pool.TotalBorrowed, amount)

	// Update interest rates
	pool.updateInterestRates()

	// Log event
	lm.engine.logEvent(Event{
		Type:      EventBorrowing,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user":         user,
			"asset":        asset,
			"amount":       amount.String(),
			"healthFactor": loan.HealthFactor,
		},
	})

	return nil
}

// Repay repays a loan
func (lm *LendingManager) Repay(user string, asset string, amount *big.Int) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	pool, exists := lm.pools[asset]
	if !exists {
		return fmt.Errorf("pool not found")
	}

	loan, exists := lm.loans[user][asset]
	if !exists {
		return fmt.Errorf("no loan found")
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()

	// Accrue interest
	lm.accrueInterest(pool)

	// Calculate total debt (principal + interest)
	totalDebt := lm.calculateTotalDebt(loan, pool)

	// Limit repayment to total debt
	if amount.Cmp(totalDebt) > 0 {
		amount = totalDebt
	}

	// Update loan
	loan.Amount.Sub(loan.Amount, amount)
	loan.LastUpdateTime = time.Now()

	// If fully repaid, remove loan
	if loan.Amount.Cmp(big.NewInt(0)) <= 0 {
		delete(lm.loans[user], asset)
		delete(pool.Borrowers, user)
		// Return collateral (simplified - would be a separate function)
	}

	// Update pool totals
	pool.TotalBorrowed.Sub(pool.TotalBorrowed, amount)

	// Update interest rates
	pool.updateInterestRates()

	return nil
}

// Liquidate liquidates an undercollateralized position
func (lm *LendingManager) Liquidate(liquidator string, borrower string, debtAsset string, collateralAsset string, debtAmount *big.Int) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	loan, exists := lm.loans[borrower][debtAsset]
	if !exists {
		return fmt.Errorf("loan not found")
	}

	// Check if position is liquidatable
	healthFactor := lm.calculateHealthFactor(loan)
	if healthFactor >= 1.0 {
		return fmt.Errorf("position is healthy, cannot liquidate")
	}

	pool := lm.pools[debtAsset]
	collateral := loan.Collateral[collateralAsset]
	if collateral == nil {
		return fmt.Errorf("collateral asset not found")
	}

	// Calculate liquidation amounts
	maxLiquidation := lm.calculateMaxLiquidation(loan, pool)
	if debtAmount.Cmp(maxLiquidation) > 0 {
		debtAmount = maxLiquidation
	}

	// Calculate collateral to seize (with bonus)
	collateralToSeize := lm.calculateCollateralToSeize(debtAmount, debtAsset, collateralAsset, pool.LiquidationBonus)

	// Check if enough collateral
	if collateralToSeize.Cmp(collateral.Amount) > 0 {
		collateralToSeize = collateral.Amount
	}

	// Update loan
	loan.Amount.Sub(loan.Amount, debtAmount)
	collateral.Amount.Sub(collateral.Amount, collateralToSeize)

	// Update pool
	pool.mu.Lock()
	pool.TotalBorrowed.Sub(pool.TotalBorrowed, debtAmount)
	pool.mu.Unlock()

	// Record liquidation
	lm.liquidations = append(lm.liquidations, Liquidation{
		User:             borrower,
		Asset:            debtAsset,
		DebtAmount:       debtAmount,
		CollateralAsset:  collateralAsset,
		CollateralAmount: collateralToSeize,
		Liquidator:       liquidator,
		Timestamp:        time.Now(),
		Penalty:          new(big.Int).Mul(collateralToSeize, big.NewInt(int64(pool.LiquidationBonus*100))),
	})

	// Log event
	lm.engine.logEvent(Event{
		Type:      EventLiquidation,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"liquidator":       liquidator,
			"borrower":         borrower,
			"debtAsset":        debtAsset,
			"debtAmount":       debtAmount.String(),
			"collateralAsset":  collateralAsset,
			"collateralAmount": collateralToSeize.String(),
		},
	})

	return nil
}

// Helper methods

// accrueInterest accrues interest for a pool
func (lm *LendingManager) accrueInterest(pool *LendingPool) {
	if pool.TotalBorrowed.Cmp(big.NewInt(0)) == 0 {
		return
	}

	timeDelta := time.Since(pool.LastUpdateTime).Seconds()
	if timeDelta == 0 {
		return
	}

	// Calculate interest
	borrowRate := pool.BorrowAPY / 365 / 24 / 3600 // Convert to per-second rate
	supplyRate := pool.SupplyAPY / 365 / 24 / 3600

	// Update indices
	borrowInterest := new(big.Float).SetInt(pool.TotalBorrowed)
	borrowInterest.Mul(borrowInterest, big.NewFloat(borrowRate*timeDelta))
	borrowInterestInt, _ := borrowInterest.Int(nil)

	supplyInterest := new(big.Float).SetInt(pool.TotalSupply)
	supplyInterest.Mul(supplyInterest, big.NewFloat(supplyRate*timeDelta))
	supplyInterestInt, _ := supplyInterest.Int(nil)

	// Update pool
	pool.AccruedInterest.Add(pool.AccruedInterest, borrowInterestInt)
	pool.TotalBorrowed.Add(pool.TotalBorrowed, borrowInterestInt)

	// Reserve factor (protocol fee)
	reserveAmount := new(big.Int).Mul(borrowInterestInt, big.NewInt(int64(pool.ReserveFactor*100)))
	reserveAmount.Div(reserveAmount, big.NewInt(100))
	pool.TotalReserves.Add(pool.TotalReserves, reserveAmount)

	// Update indices for compound interest
	pool.BorrowIndex.Add(pool.BorrowIndex, mulDiv(pool.BorrowIndex, borrowInterestInt, pool.TotalBorrowed))
	pool.SupplyIndex.Add(pool.SupplyIndex, mulDiv(pool.SupplyIndex, supplyInterestInt, pool.TotalSupply))

	pool.LastUpdateTime = time.Now()
}

// updateInterestRates updates the interest rates based on utilization
func (pool *LendingPool) updateInterestRates() {
	if pool.TotalSupply.Cmp(big.NewInt(0)) == 0 {
		pool.UtilizationRate = 0
		pool.SupplyAPY = 0
		pool.BorrowAPY = pool.InterestRateModel.GetBaseRate()
		return
	}

	// Calculate utilization rate
	utilization := new(big.Float).SetInt(pool.TotalBorrowed)
	totalSupply := new(big.Float).SetInt(pool.TotalSupply)
	utilization.Quo(utilization, totalSupply)
	pool.UtilizationRate, _ = utilization.Float64()

	// Get rates from model
	pool.SupplyAPY, pool.BorrowAPY = pool.InterestRateModel.CalculateRates(pool.UtilizationRate)
}

// calculateSupplyShares calculates shares for a supply amount
func (lm *LendingManager) calculateSupplyShares(pool *LendingPool, amount *big.Int) *big.Int {
	if pool.TotalSupply.Cmp(big.NewInt(0)) == 0 {
		return amount // 1:1 for first deposit
	}
	// shares = amount * totalShares / totalSupply
	totalShares := pool.TotalSupply // Simplified - would track actual shares
	return mulDiv(amount, totalShares, pool.TotalSupply)
}

// calculateSupplyValue calculates the value of supply shares
func (lm *LendingManager) calculateSupplyValue(pool *LendingPool, shares *big.Int) *big.Int {
	if shares.Cmp(big.NewInt(0)) == 0 {
		return big.NewInt(0)
	}
	// value = shares * totalSupply / totalShares
	totalShares := pool.TotalSupply // Simplified
	return mulDiv(shares, pool.TotalSupply, totalShares)
}

// calculateSharesToBurn calculates shares to burn for withdrawal
func (lm *LendingManager) calculateSharesToBurn(pool *LendingPool, amount *big.Int) *big.Int {
	// shares = amount * totalShares / totalSupply
	totalShares := pool.TotalSupply // Simplified
	return mulDiv(amount, totalShares, pool.TotalSupply)
}

// calculateCollateralValue calculates total collateral value in USD
func (lm *LendingManager) calculateCollateralValue(collateralAssets map[string]*big.Int) *big.Int {
	totalValue := big.NewInt(0)
	for asset, amount := range collateralAssets {
		value := lm.getAssetValue(asset, amount)
		totalValue.Add(totalValue, value)
	}
	return totalValue
}

// getAssetValue gets USD value of an asset amount
func (lm *LendingManager) getAssetValue(asset string, amount *big.Int) *big.Int {
	oracle := lm.oracles[asset]
	if oracle == nil {
		return big.NewInt(0)
	}
	
	price := big.NewFloat(oracle.Price)
	amountFloat := new(big.Float).SetInt(amount)
	value := new(big.Float).Mul(price, amountFloat)
	valueInt, _ := value.Int(nil)
	return valueInt
}

// calculateHealthFactor calculates the health factor of a loan
func (lm *LendingManager) calculateHealthFactor(loan *Loan) float64 {
	totalCollateralValue := big.NewInt(0)
	totalBorrowValue := big.NewInt(0)

	// Calculate weighted collateral value
	for asset, coll := range loan.Collateral {
		value := lm.getAssetValue(asset, coll.Amount)
		// Apply collateral factor
		weightedValue := mulDiv(value, big.NewInt(int64(coll.CollateralFactor*100)), big.NewInt(100))
		totalCollateralValue.Add(totalCollateralValue, weightedValue)
	}

	// Calculate borrow value
	totalBorrowValue = lm.getAssetValue(loan.Asset, loan.Amount)

	if totalBorrowValue.Cmp(big.NewInt(0)) == 0 {
		return math.MaxFloat64
	}

	// Health Factor = Collateral Value / Borrow Value
	healthFactor := new(big.Float).SetInt(totalCollateralValue)
	borrowFloat := new(big.Float).SetInt(totalBorrowValue)
	healthFactor.Quo(healthFactor, borrowFloat)
	
	result, _ := healthFactor.Float64()
	return result
}

// calculateLiquidationPrices calculates liquidation prices for each collateral
func (lm *LendingManager) calculateLiquidationPrices(loan *Loan) map[string]float64 {
	prices := make(map[string]float64)
	
	for asset, coll := range loan.Collateral {
		oracle := lm.oracles[asset]
		if oracle == nil {
			continue
		}
		
		// Liquidation occurs when health factor < 1
		// Price at which: (collateral * price * factor) / debt = 1
		currentPrice := oracle.Price
		liquidationPrice := currentPrice * coll.CollateralFactor
		prices[asset] = liquidationPrice
	}
	
	return prices
}

// calculateTotalDebt calculates total debt including interest
func (lm *LendingManager) calculateTotalDebt(loan *Loan, pool *LendingPool) *big.Int {
	// Calculate accrued interest based on index
	currentIndex := pool.BorrowIndex
	borrowIndex := loan.Index
	
	if borrowIndex.Cmp(big.NewInt(0)) == 0 {
		borrowIndex = big.NewInt(1e18)
	}
	
	// debt = principal * currentIndex / borrowIndex
	totalDebt := mulDiv(loan.Amount, currentIndex, borrowIndex)
	return totalDebt
}

// calculateMaxLiquidation calculates maximum liquidation amount
func (lm *LendingManager) calculateMaxLiquidation(loan *Loan, pool *LendingPool) *big.Int {
	// Typically 50% of the debt can be liquidated at once
	maxLiquidation := new(big.Int).Div(loan.Amount, big.NewInt(2))
	return maxLiquidation
}

// calculateCollateralToSeize calculates collateral amount to seize
func (lm *LendingManager) calculateCollateralToSeize(debtAmount *big.Int, debtAsset, collateralAsset string, liquidationBonus float64) *big.Int {
	debtValue := lm.getAssetValue(debtAsset, debtAmount)
	
	// Add liquidation bonus
	seizeValue := new(big.Int).Mul(debtValue, big.NewInt(int64((1+liquidationBonus)*100)))
	seizeValue.Div(seizeValue, big.NewInt(100))
	
	// Convert to collateral amount
	collateralPrice := lm.oracles[collateralAsset].Price
	if collateralPrice == 0 {
		return big.NewInt(0)
	}
	
	seizeAmount := new(big.Float).SetInt(seizeValue)
	seizeAmount.Quo(seizeAmount, big.NewFloat(collateralPrice))
	result, _ := seizeAmount.Int(nil)
	
	return result
}

// JumpRateModel implementation
func NewJumpRateModel(baseRate, slope1, slope2, optimalUtil float64) *JumpRateModel {
	return &JumpRateModel{
		BaseRate:           baseRate,
		Slope1:             slope1,
		Slope2:             slope2,
		OptimalUtilization: optimalUtil,
	}
}

func (m *JumpRateModel) CalculateRates(utilization float64) (supplyAPY, borrowAPY float64) {
	if utilization <= m.OptimalUtilization {
		// Before kink
		borrowAPY = m.BaseRate + utilization*m.Slope1
	} else {
		// After kink (jump)
		normalRate := m.BaseRate + m.OptimalUtilization*m.Slope1
		excessUtil := utilization - m.OptimalUtilization
		borrowAPY = normalRate + excessUtil*m.Slope2
	}
	
	// Supply rate = Borrow rate * Utilization * (1 - Reserve Factor)
	supplyAPY = borrowAPY * utilization * 0.9 // Assuming 10% reserve factor
	
	return supplyAPY, borrowAPY
}

func (m *JumpRateModel) GetBaseRate() float64           { return m.BaseRate }
func (m *JumpRateModel) GetSlope1() float64             { return m.Slope1 }
func (m *JumpRateModel) GetSlope2() float64             { return m.Slope2 }
func (m *JumpRateModel) GetOptimalUtilization() float64 { return m.OptimalUtilization }

// GetPool returns a lending pool
func (lm *LendingManager) GetPool(asset string) (*LendingPool, error) {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	
	pool, exists := lm.pools[asset]
	if !exists {
		return nil, fmt.Errorf("pool not found")
	}
	
	return pool, nil
}

// GetUserLoans returns all loans for a user
func (lm *LendingManager) GetUserLoans(user string) map[string]*Loan {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	
	return lm.loans[user]
}

// GetUserSupplies returns all supplies for a user
func (lm *LendingManager) GetUserSupplies(user string) map[string]*Supply {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	
	supplies := make(map[string]*Supply)
	for asset, pool := range lm.pools {
		if supply, exists := pool.Suppliers[user]; exists {
			supplies[asset] = supply
		}
	}
	
	return supplies
}