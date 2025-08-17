package lx

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// LendingPool manages lending and borrowing operations
type LendingPool struct {
	Pools              map[string]*AssetPool
	Suppliers          map[string]*LendingAccount
	Borrowers          map[string]*BorrowingAccount
	InterestModel      *InterestRateModel
	CollateralManager  *CollateralManager
	ReserveFactory     *ReserveFactory
	TotalSupplied      map[string]*big.Int
	TotalBorrowed      map[string]*big.Int
	ReserveFunds       map[string]*big.Int
	UtilizationRates   map[string]float64
	LastUpdateBlock    uint64
	AccruedFees        map[string]*big.Int
	ProtocolFeeRate    float64
	mu                 sync.RWMutex
}

// AssetPool represents a lending pool for a specific asset
type AssetPool struct {
	Asset                string
	TotalSupply          *big.Int
	TotalBorrow          *big.Int
	TotalReserves        *big.Int
	SupplyIndex          *big.Int
	BorrowIndex          *big.Int
	SupplyRate           float64
	BorrowRate           float64
	ExchangeRate         float64
	UtilizationRate      float64
	ReserveFactor        float64
	CollateralFactor     float64
	LiquidationThreshold float64
	LiquidationPenalty   float64
	MaxBorrowRate        float64
	MinBorrowRate        float64
	OptimalUtilization   float64
	LastUpdate           time.Time
	Suppliers            map[string]*SupplyPosition
	Borrowers            map[string]*BorrowPosition
	Paused               bool
	mu                   sync.RWMutex
}

// LendingAccount represents a supplier's account
type LendingAccount struct {
	UserID           string
	SuppliedAssets   map[string]*SupplyPosition
	CollateralAssets map[string]*CollateralPosition
	TotalSupplied    *big.Int
	TotalCollateral  *big.Int
	EarnedInterest   *big.Int
	LastUpdate       time.Time
}

// BorrowingAccount represents a borrower's account
type BorrowingAccount struct {
	UserID            string
	BorrowedAssets    map[string]*BorrowPosition
	CollateralAssets  map[string]*CollateralPosition
	TotalBorrowed     *big.Int
	TotalCollateral   *big.Int
	AccruedInterest   *big.Int
	HealthFactor      float64
	LiquidationPrice  map[string]float64
	LastUpdate        time.Time
}

// SupplyPosition represents a supply position in a pool
type SupplyPosition struct {
	Asset           string
	Amount          *big.Int
	Shares          *big.Int // LP tokens
	InterestEarned  *big.Int
	SupplyRate      float64
	StartTime       time.Time
	LastUpdate      time.Time
	IsCollateral    bool
}

// BorrowPosition represents a borrow position
type BorrowPosition struct {
	Asset            string
	Principal        *big.Int
	AccruedInterest  *big.Int
	TotalOwed        *big.Int
	BorrowRate       float64
	StartTime        time.Time
	LastUpdate       time.Time
	CollateralLocked *big.Int
}

// CollateralPosition represents collateral in the pool
type CollateralPosition struct {
	Asset            string
	Amount           *big.Int
	ValueUSD         *big.Int
	CollateralFactor float64
	Locked           bool
	LastUpdate       time.Time
}

// InterestRateModel calculates interest rates based on utilization
type InterestRateModel struct {
	BaseRate           float64
	MultiplierPerBlock float64
	JumpMultiplier     float64
	Kink               float64 // Utilization rate at which rates jump
	BlocksPerYear      uint64
}

// NewLendingPool creates a new lending pool
func NewLendingPool() *LendingPool {
	return &LendingPool{
		Pools:             make(map[string]*AssetPool),
		Suppliers:         make(map[string]*LendingAccount),
		Borrowers:         make(map[string]*BorrowingAccount),
		InterestModel:     NewDefaultInterestModel(),
		CollateralManager: NewCollateralManager(),
		ReserveFactory:    NewReserveFactory(),
		TotalSupplied:     make(map[string]*big.Int),
		TotalBorrowed:     make(map[string]*big.Int),
		ReserveFunds:      make(map[string]*big.Int),
		UtilizationRates:  make(map[string]float64),
		AccruedFees:       make(map[string]*big.Int),
		ProtocolFeeRate:   0.10, // 10% of interest goes to protocol
	}
}

// CreatePool creates a new lending pool for an asset
func (lp *LendingPool) CreatePool(asset string, config PoolConfig) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	
	if _, exists := lp.Pools[asset]; exists {
		return fmt.Errorf("pool already exists for %s", asset)
	}
	
	pool := &AssetPool{
		Asset:                asset,
		TotalSupply:          big.NewInt(0),
		TotalBorrow:          big.NewInt(0),
		TotalReserves:        big.NewInt(0),
		SupplyIndex:          big.NewInt(1e18), // Start at 1.0 (scaled)
		BorrowIndex:          big.NewInt(1e18),
		ExchangeRate:         1.0,
		ReserveFactor:        config.ReserveFactor,
		CollateralFactor:     config.CollateralFactor,
		LiquidationThreshold: config.LiquidationThreshold,
		LiquidationPenalty:   config.LiquidationPenalty,
		MaxBorrowRate:        config.MaxBorrowRate,
		MinBorrowRate:        config.MinBorrowRate,
		OptimalUtilization:   config.OptimalUtilization,
		Suppliers:            make(map[string]*SupplyPosition),
		Borrowers:            make(map[string]*BorrowPosition),
		LastUpdate:           time.Now(),
	}
	
	lp.Pools[asset] = pool
	lp.TotalSupplied[asset] = big.NewInt(0)
	lp.TotalBorrowed[asset] = big.NewInt(0)
	lp.ReserveFunds[asset] = big.NewInt(0)
	lp.AccruedFees[asset] = big.NewInt(0)
	
	return nil
}

// Supply adds liquidity to the pool
func (lp *LendingPool) Supply(userID, asset string, amount *big.Int) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	
	pool, exists := lp.Pools[asset]
	if !exists {
		return fmt.Errorf("pool not found for %s", asset)
	}
	
	if pool.Paused {
		return errors.New("pool is paused")
	}
	
	// Update interest rates
	lp.updateInterestRates(pool)
	
	// Get or create supplier account
	supplier, exists := lp.Suppliers[userID]
	if !exists {
		supplier = &LendingAccount{
			UserID:          userID,
			SuppliedAssets:  make(map[string]*SupplyPosition),
			CollateralAssets: make(map[string]*CollateralPosition),
			TotalSupplied:   big.NewInt(0),
			TotalCollateral: big.NewInt(0),
			EarnedInterest:  big.NewInt(0),
			LastUpdate:      time.Now(),
		}
		lp.Suppliers[userID] = supplier
	}
	
	// Calculate shares to mint (LP tokens)
	var shares *big.Int
	if pool.TotalSupply.Cmp(big.NewInt(0)) == 0 {
		shares = amount // First supplier gets 1:1
	} else {
		// shares = amount * totalShares / totalSupply
		totalShares := lp.getTotalShares(pool)
		shares = new(big.Int).Mul(amount, totalShares)
		shares.Div(shares, pool.TotalSupply)
	}
	
	// Update or create supply position
	position, exists := supplier.SuppliedAssets[asset]
	if !exists {
		position = &SupplyPosition{
			Asset:      asset,
			Amount:     big.NewInt(0),
			Shares:     big.NewInt(0),
			InterestEarned: big.NewInt(0),
			StartTime:  time.Now(),
			LastUpdate: time.Now(),
		}
		supplier.SuppliedAssets[asset] = position
	}
	
	// Update position
	position.Amount.Add(position.Amount, amount)
	position.Shares.Add(position.Shares, shares)
	position.SupplyRate = pool.SupplyRate
	position.LastUpdate = time.Now()
	
	// Update pool
	pool.TotalSupply.Add(pool.TotalSupply, amount)
	pool.Suppliers[userID] = position
	
	// Update totals
	supplier.TotalSupplied.Add(supplier.TotalSupplied, amount)
	lp.TotalSupplied[asset].Add(lp.TotalSupplied[asset], amount)
	
	// Recalculate rates
	lp.updateUtilizationRate(pool)
	lp.updateInterestRates(pool)
	
	return nil
}

// Withdraw removes liquidity from the pool
func (lp *LendingPool) Withdraw(userID, asset string, amount *big.Int) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	
	pool, exists := lp.Pools[asset]
	if !exists {
		return fmt.Errorf("pool not found for %s", asset)
	}
	
	supplier, exists := lp.Suppliers[userID]
	if !exists {
		return errors.New("no supply position found")
	}
	
	position, exists := supplier.SuppliedAssets[asset]
	if !exists {
		return errors.New("no supply position for asset")
	}
	
	// Check available liquidity
	availableLiquidity := new(big.Int).Sub(pool.TotalSupply, pool.TotalBorrow)
	if amount.Cmp(availableLiquidity) > 0 {
		return errors.New("insufficient liquidity")
	}
	
	// Check if amount exceeds position
	if amount.Cmp(position.Amount) > 0 {
		return errors.New("withdrawal amount exceeds supplied amount")
	}
	
	// Calculate shares to burn
	sharesToBurn := new(big.Int).Mul(position.Shares, amount)
	sharesToBurn.Div(sharesToBurn, position.Amount)
	
	// Calculate earned interest
	interest := lp.calculateSupplyInterest(position, pool)
	position.InterestEarned.Add(position.InterestEarned, interest)
	supplier.EarnedInterest.Add(supplier.EarnedInterest, interest)
	
	// Update position
	position.Amount.Sub(position.Amount, amount)
	position.Shares.Sub(position.Shares, sharesToBurn)
	position.LastUpdate = time.Now()
	
	// Remove position if empty
	if position.Amount.Cmp(big.NewInt(0)) == 0 {
		delete(supplier.SuppliedAssets, asset)
		delete(pool.Suppliers, userID)
	}
	
	// Update pool
	pool.TotalSupply.Sub(pool.TotalSupply, amount)
	
	// Update totals
	supplier.TotalSupplied.Sub(supplier.TotalSupplied, amount)
	lp.TotalSupplied[asset].Sub(lp.TotalSupplied[asset], amount)
	
	// Recalculate rates
	lp.updateUtilizationRate(pool)
	lp.updateInterestRates(pool)
	
	return nil
}

// Borrow allows users to borrow against collateral
func (lp *LendingPool) Borrow(asset string, amount *big.Int) error {
	// Implementation handled by MarginEngine
	pool, exists := lp.Pools[asset]
	if !exists {
		return fmt.Errorf("pool not found for %s", asset)
	}
	
	pool.mu.Lock()
	defer pool.mu.Unlock()
	
	// Check available liquidity
	available := new(big.Int).Sub(pool.TotalSupply, pool.TotalBorrow)
	if amount.Cmp(available) > 0 {
		return errors.New("insufficient liquidity")
	}
	
	// Update pool
	pool.TotalBorrow.Add(pool.TotalBorrow, amount)
	lp.TotalBorrowed[asset].Add(lp.TotalBorrowed[asset], amount)
	
	// Update rates
	lp.updateUtilizationRate(pool)
	lp.updateInterestRates(pool)
	
	return nil
}

// Repay repays borrowed amount with interest
func (lp *LendingPool) Repay(asset string, principal, interest *big.Int) error {
	pool, exists := lp.Pools[asset]
	if !exists {
		return fmt.Errorf("pool not found for %s", asset)
	}
	
	pool.mu.Lock()
	defer pool.mu.Unlock()
	
	// Update pool
	pool.TotalBorrow.Sub(pool.TotalBorrow, principal)
	lp.TotalBorrowed[asset].Sub(lp.TotalBorrowed[asset], principal)
	
	// Add interest to reserves
	protocolFee := new(big.Int).Mul(interest, big.NewInt(int64(lp.ProtocolFeeRate*1000)))
	protocolFee.Div(protocolFee, big.NewInt(1000))
	
	pool.TotalReserves.Add(pool.TotalReserves, protocolFee)
	lp.ReserveFunds[asset].Add(lp.ReserveFunds[asset], protocolFee)
	lp.AccruedFees[asset].Add(lp.AccruedFees[asset], protocolFee)
	
	// Remaining interest goes to suppliers
	supplierInterest := new(big.Int).Sub(interest, protocolFee)
	pool.TotalSupply.Add(pool.TotalSupply, supplierInterest)
	
	// Update rates
	lp.updateUtilizationRate(pool)
	lp.updateInterestRates(pool)
	
	return nil
}

// GetAvailable returns available liquidity for borrowing
func (lp *LendingPool) GetAvailable(asset string) *big.Int {
	lp.mu.RLock()
	defer lp.mu.RUnlock()
	
	pool, exists := lp.Pools[asset]
	if !exists {
		return big.NewInt(0)
	}
	
	return new(big.Int).Sub(pool.TotalSupply, pool.TotalBorrow)
}

// GetBorrowRate returns current borrow rate for an asset
func (lp *LendingPool) GetBorrowRate(asset string) float64 {
	lp.mu.RLock()
	defer lp.mu.RUnlock()
	
	pool, exists := lp.Pools[asset]
	if !exists {
		return 0
	}
	
	return pool.BorrowRate
}

// GetSupplyRate returns current supply rate for an asset
func (lp *LendingPool) GetSupplyRate(asset string) float64 {
	lp.mu.RLock()
	defer lp.mu.RUnlock()
	
	pool, exists := lp.Pools[asset]
	if !exists {
		return 0
	}
	
	return pool.SupplyRate
}

// updateUtilizationRate updates the utilization rate of a pool
func (lp *LendingPool) updateUtilizationRate(pool *AssetPool) {
	if pool.TotalSupply.Cmp(big.NewInt(0)) == 0 {
		pool.UtilizationRate = 0
		return
	}
	
	borrowed := new(big.Float).SetInt(pool.TotalBorrow)
	supplied := new(big.Float).SetInt(pool.TotalSupply)
	utilization := new(big.Float).Quo(borrowed, supplied)
	
	pool.UtilizationRate, _ = utilization.Float64()
	lp.UtilizationRates[pool.Asset] = pool.UtilizationRate
}

// updateInterestRates updates supply and borrow rates
func (lp *LendingPool) updateInterestRates(pool *AssetPool) {
	utilization := pool.UtilizationRate
	
	// Calculate borrow rate using interest model
	borrowRate := lp.InterestModel.calculateBorrowRate(utilization)
	pool.BorrowRate = math.Min(borrowRate, pool.MaxBorrowRate)
	pool.BorrowRate = math.Max(pool.BorrowRate, pool.MinBorrowRate)
	
	// Calculate supply rate
	// supplyRate = borrowRate * utilization * (1 - reserveFactor)
	supplyRate := pool.BorrowRate * utilization * (1 - pool.ReserveFactor)
	pool.SupplyRate = supplyRate
	
	pool.LastUpdate = time.Now()
}

// calculateSupplyInterest calculates accrued interest for a supply position
func (lp *LendingPool) calculateSupplyInterest(position *SupplyPosition, pool *AssetPool) *big.Int {
	timeDiff := time.Since(position.LastUpdate).Hours()
	hourlyRate := position.SupplyRate / (365 * 24)
	
	interest := new(big.Float).SetInt(position.Amount)
	interest.Mul(interest, big.NewFloat(hourlyRate*timeDiff))
	
	interestInt, _ := interest.Int(nil)
	return interestInt
}

// getTotalShares returns total LP token shares for a pool
func (lp *LendingPool) getTotalShares(pool *AssetPool) *big.Int {
	totalShares := big.NewInt(0)
	for _, position := range pool.Suppliers {
		totalShares.Add(totalShares, position.Shares)
	}
	return totalShares
}

// calculateBorrowRate calculates borrow rate based on utilization
func (model *InterestRateModel) calculateBorrowRate(utilization float64) float64 {
	if utilization <= model.Kink {
		// Below kink: linear increase
		return model.BaseRate + utilization*model.MultiplierPerBlock
	}
	
	// Above kink: jump rate
	normalRate := model.BaseRate + model.Kink*model.MultiplierPerBlock
	excessUtilization := utilization - model.Kink
	return normalRate + excessUtilization*model.JumpMultiplier
}

// NewDefaultInterestModel creates a default interest rate model
func NewDefaultInterestModel() *InterestRateModel {
	return &InterestRateModel{
		BaseRate:           0.02,  // 2% base rate
		MultiplierPerBlock: 0.15,  // 15% at 100% utilization (below kink)
		JumpMultiplier:     2.0,    // 200% above kink
		Kink:               0.80,   // 80% utilization kink
		BlocksPerYear:      2628000, // ~12 second blocks
	}
}

// CollateralManager manages collateral operations
type CollateralManager struct {
	CollateralFactors map[string]float64
	PriceOracle       *PriceOracle
	mu                sync.RWMutex
}

func NewCollateralManager() *CollateralManager {
	return &CollateralManager{
		CollateralFactors: initCollateralFactors(),
	}
}

func initCollateralFactors() map[string]float64 {
	return map[string]float64{
		"BTC":  0.80,
		"ETH":  0.75,
		"BNB":  0.70,
		"USDT": 0.95,
		"USDC": 0.95,
		"DAI":  0.90,
	}
}

// ReserveFactory creates and manages reserves
type ReserveFactory struct {
	Reserves map[string]*Reserve
	mu       sync.RWMutex
}

type Reserve struct {
	Asset         string
	Amount        *big.Int
	TargetRatio   float64
	CurrentRatio  float64
	LastRebalance time.Time
}

func NewReserveFactory() *ReserveFactory {
	return &ReserveFactory{
		Reserves: make(map[string]*Reserve),
	}
}

// PoolConfig configuration for creating a lending pool
type PoolConfig struct {
	ReserveFactor        float64
	CollateralFactor     float64
	LiquidationThreshold float64
	LiquidationPenalty   float64
	MaxBorrowRate        float64
	MinBorrowRate        float64
	OptimalUtilization   float64
}