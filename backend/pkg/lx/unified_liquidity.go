package lx

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// UnifiedLiquidityPool manages liquidity across spot, margin, and perpetuals
type UnifiedLiquidityPool struct {
	TotalLiquidity      map[string]*big.Int
	SpotLiquidity       map[string]*big.Int
	MarginLiquidity     map[string]*big.Int
	PerpetualLiquidity  map[string]*big.Int
	VaultLiquidity      map[string]*big.Int
	LiquidityProviders  map[string]*LiquidityProvider
	LiquidityAllocation *LiquidityAllocation
	CrossAssetNetting   bool
	AutoRebalancing     bool
	RebalanceThreshold  float64
	LastRebalance       time.Time
	RebalanceInterval   time.Duration
	FeeDistribution     *FeeDistribution
	YieldOptimizer      *YieldOptimizer
	RiskManager         *UnifiedRiskManager
	Settlements         map[string]*Settlement
	mu                  sync.RWMutex
}

// LiquidityProvider represents a liquidity provider in the unified pool
type LiquidityProvider struct {
	UserID             string
	ProvidedLiquidity  map[string]*big.Int
	ShareTokens        map[string]*big.Int
	EarnedFees         map[string]*big.Int
	EarnedYield        map[string]*big.Int
	ImpermanentLoss    map[string]*big.Int
	LockedUntil        map[string]time.Time
	Tier               LPTier
	FeeMultiplier      float64
	YieldBoost         float64
	LastUpdate         time.Time
}

// LPTier represents liquidity provider tiers
type LPTier int

const (
	BronzeTier LPTier = iota
	SilverTier
	GoldTier
	PlatinumTier
	DiamondTier
)

// LiquidityAllocation manages how liquidity is allocated across markets
type LiquidityAllocation struct {
	SpotAllocation       float64
	MarginAllocation     float64
	PerpetualAllocation  float64
	VaultAllocation      float64
	ReserveAllocation    float64
	DynamicAllocation    bool
	AllocationStrategy   AllocationStrategy
	LastUpdate           time.Time
}

// AllocationStrategy interface for different allocation strategies
type AllocationStrategy interface {
	CalculateAllocation(pool *UnifiedLiquidityPool) map[string]float64
	ShouldRebalance(pool *UnifiedLiquidityPool) bool
	GetRiskAdjustedAllocation(riskMetrics *RiskMetrics) map[string]float64
}

// VolumeWeightedAllocation allocates based on trading volume
type VolumeWeightedAllocation struct {
	VolumeWindow    time.Duration
	MinAllocation   float64
	MaxAllocation   float64
	VolumeHistory   map[string][]VolumeSnapshot
}

func (vwa *VolumeWeightedAllocation) CalculateAllocation(pool *UnifiedLiquidityPool) map[string]float64 {
	allocations := make(map[string]float64)
	totalVolume := 0.0
	
	// Calculate total volume across markets
	spotVolume := vwa.getVolumeForMarket("spot")
	marginVolume := vwa.getVolumeForMarket("margin")
	perpVolume := vwa.getVolumeForMarket("perpetual")
	vaultVolume := vwa.getVolumeForMarket("vault")
	
	totalVolume = spotVolume + marginVolume + perpVolume + vaultVolume
	
	if totalVolume == 0 {
		// Equal allocation if no volume
		allocations["spot"] = 0.25
		allocations["margin"] = 0.25
		allocations["perpetual"] = 0.25
		allocations["vault"] = 0.25
	} else {
		allocations["spot"] = math.Max(vwa.MinAllocation, math.Min(vwa.MaxAllocation, spotVolume/totalVolume))
		allocations["margin"] = math.Max(vwa.MinAllocation, math.Min(vwa.MaxAllocation, marginVolume/totalVolume))
		allocations["perpetual"] = math.Max(vwa.MinAllocation, math.Min(vwa.MaxAllocation, perpVolume/totalVolume))
		allocations["vault"] = math.Max(vwa.MinAllocation, math.Min(vwa.MaxAllocation, vaultVolume/totalVolume))
	}
	
	// Normalize to ensure sum = 1.0
	sum := allocations["spot"] + allocations["margin"] + allocations["perpetual"] + allocations["vault"]
	for k := range allocations {
		allocations[k] /= sum
	}
	
	return allocations
}

func (vwa *VolumeWeightedAllocation) ShouldRebalance(pool *UnifiedLiquidityPool) bool {
	return time.Since(pool.LastRebalance) > pool.RebalanceInterval
}

func (vwa *VolumeWeightedAllocation) GetRiskAdjustedAllocation(riskMetrics *RiskMetrics) map[string]float64 {
	baseAllocation := vwa.CalculateAllocation(nil)
	
	// Adjust based on risk metrics
	for market, allocation := range baseAllocation {
		if riskMetrics.MaxDrawdown > 0.10 {
			// Reduce risky allocations during high drawdown
			if market == "perpetual" || market == "margin" {
				baseAllocation[market] = allocation * 0.8
			}
		}
	}
	
	return baseAllocation
}

func (vwa *VolumeWeightedAllocation) getVolumeForMarket(market string) float64 {
	history, exists := vwa.VolumeHistory[market]
	if !exists {
		return 0
	}
	
	totalVolume := 0.0
	cutoff := time.Now().Add(-vwa.VolumeWindow)
	
	for _, snapshot := range history {
		if snapshot.Timestamp.After(cutoff) {
			totalVolume += snapshot.Volume
		}
	}
	
	return totalVolume
}

// VolumeSnapshot represents a volume measurement at a point in time
type VolumeSnapshot struct {
	Timestamp time.Time
	Volume    float64
	Market    string
}

// FeeDistribution manages fee distribution to LPs
type FeeDistribution struct {
	TotalFees          map[string]*big.Int
	LPShare            float64 // Percentage of fees going to LPs
	ProtocolShare      float64
	InsuranceShare     float64
	BuybackShare       float64
	DistributionQueue  []*FeeDistributionEvent
	LastDistribution   time.Time
	DistributionPeriod time.Duration
}

// FeeDistributionEvent represents a fee distribution event
type FeeDistributionEvent struct {
	Timestamp   time.Time
	Asset       string
	Amount      *big.Int
	Recipients  map[string]*big.Int
	Distributed bool
}

// YieldOptimizer optimizes yield generation strategies
type YieldOptimizer struct {
	Strategies       []YieldStrategy
	ActiveStrategies map[string]YieldStrategy
	TargetAPY        float64
	RiskTolerance    float64
	Rebalancer       *YieldRebalancer
}

// YieldStrategy interface for yield generation strategies
type YieldStrategy interface {
	GetAPY() float64
	GetRisk() float64
	Execute(capital *big.Int) error
	GetName() string
}

// StakingStrategy implements staking yield
type StakingStrategy struct {
	StakingPool     string
	CurrentAPY      float64
	LockupPeriod    time.Duration
	MinStake        *big.Int
	CompoundingFreq time.Duration
}

func (s *StakingStrategy) GetAPY() float64 {
	return s.CurrentAPY
}

func (s *StakingStrategy) GetRisk() float64 {
	// Lower risk for staking
	return 0.2
}

func (s *StakingStrategy) Execute(capital *big.Int) error {
	// Implementation for staking
	return nil
}

func (s *StakingStrategy) GetName() string {
	return "Staking"
}

// UnifiedRiskManager manages risk across unified liquidity
type UnifiedRiskManager struct {
	MaxDrawdown         float64
	VaR95              float64
	VaR99              float64
	StressTestScenarios []StressScenario
	RiskLimits         map[string]float64
	Monitoring         *RiskMonitoring
}

// StressScenario represents a stress test scenario
type StressScenario struct {
	Name        string
	PriceShocks map[string]float64
	VolumeShock float64
	Duration    time.Duration
}

// RiskMonitoring monitors risk metrics in real-time
type RiskMonitoring struct {
	Alerts        []RiskAlert
	Thresholds    map[string]float64
	LastCheck     time.Time
	CheckInterval time.Duration
}

// RiskAlert represents a risk alert
type RiskAlert struct {
	Timestamp   time.Time
	Severity    AlertSeverity
	Message     string
	Metric      string
	Value       float64
	Threshold   float64
}

// AlertSeverity represents alert severity levels
type AlertSeverity int

const (
	InfoAlert AlertSeverity = iota
	WarningAlert
	CriticalAlert
	EmergencyAlert
)

// NewUnifiedLiquidityPool creates a new unified liquidity pool
func NewUnifiedLiquidityPool() *UnifiedLiquidityPool {
	return &UnifiedLiquidityPool{
		TotalLiquidity:     make(map[string]*big.Int),
		SpotLiquidity:      make(map[string]*big.Int),
		MarginLiquidity:    make(map[string]*big.Int),
		PerpetualLiquidity: make(map[string]*big.Int),
		VaultLiquidity:     make(map[string]*big.Int),
		LiquidityProviders: make(map[string]*LiquidityProvider),
		LiquidityAllocation: &LiquidityAllocation{
			SpotAllocation:      0.30,
			MarginAllocation:    0.25,
			PerpetualAllocation: 0.35,
			VaultAllocation:     0.10,
			DynamicAllocation:   true,
			AllocationStrategy: &VolumeWeightedAllocation{
				VolumeWindow:  24 * time.Hour,
				MinAllocation: 0.05,
				MaxAllocation: 0.50,
				VolumeHistory: make(map[string][]VolumeSnapshot),
			},
		},
		CrossAssetNetting:  true,
		AutoRebalancing:    true,
		RebalanceThreshold: 0.05, // 5% threshold
		RebalanceInterval:  1 * time.Hour,
		FeeDistribution: &FeeDistribution{
			TotalFees:          make(map[string]*big.Int),
			LPShare:            0.70, // 70% to LPs
			ProtocolShare:      0.15,
			InsuranceShare:     0.10,
			BuybackShare:       0.05,
			DistributionPeriod: 24 * time.Hour,
		},
		YieldOptimizer: &YieldOptimizer{
			Strategies:       make([]YieldStrategy, 0),
			ActiveStrategies: make(map[string]YieldStrategy),
			TargetAPY:        0.20, // 20% target APY
			RiskTolerance:    0.5,
		},
		RiskManager: &UnifiedRiskManager{
			MaxDrawdown: 0.20,
			VaR95:       0.05,
			VaR99:       0.10,
			RiskLimits:  make(map[string]float64),
			Monitoring: &RiskMonitoring{
				Thresholds:    make(map[string]float64),
				CheckInterval: 5 * time.Minute,
			},
		},
		Settlements: make(map[string]*Settlement),
	}
}

// AddLiquidity adds liquidity to the unified pool
func (ulp *UnifiedLiquidityPool) AddLiquidity(userID, asset string, amount *big.Int) error {
	ulp.mu.Lock()
	defer ulp.mu.Unlock()
	
	// Get or create liquidity provider
	provider, exists := ulp.LiquidityProviders[userID]
	if !exists {
		provider = &LiquidityProvider{
			UserID:            userID,
			ProvidedLiquidity: make(map[string]*big.Int),
			ShareTokens:       make(map[string]*big.Int),
			EarnedFees:        make(map[string]*big.Int),
			EarnedYield:       make(map[string]*big.Int),
			ImpermanentLoss:   make(map[string]*big.Int),
			LockedUntil:       make(map[string]time.Time),
			Tier:              BronzeTier,
			FeeMultiplier:     1.0,
			YieldBoost:        1.0,
			LastUpdate:        time.Now(),
		}
		ulp.LiquidityProviders[userID] = provider
	}
	
	// Update provider liquidity
	if provider.ProvidedLiquidity[asset] == nil {
		provider.ProvidedLiquidity[asset] = big.NewInt(0)
	}
	provider.ProvidedLiquidity[asset].Add(provider.ProvidedLiquidity[asset], amount)
	
	// Calculate and mint share tokens
	shares := ulp.calculateShares(asset, amount)
	if provider.ShareTokens[asset] == nil {
		provider.ShareTokens[asset] = big.NewInt(0)
	}
	provider.ShareTokens[asset].Add(provider.ShareTokens[asset], shares)
	
	// Update total liquidity
	if ulp.TotalLiquidity[asset] == nil {
		ulp.TotalLiquidity[asset] = big.NewInt(0)
	}
	ulp.TotalLiquidity[asset].Add(ulp.TotalLiquidity[asset], amount)
	
	// Allocate liquidity across markets
	ulp.allocateLiquidity(asset, amount)
	
	// Update provider tier based on total liquidity
	ulp.updateProviderTier(provider)
	
	// Trigger rebalance if needed
	if ulp.AutoRebalancing && ulp.shouldRebalance() {
		go ulp.rebalance()
	}
	
	return nil
}

// RemoveLiquidity removes liquidity from the unified pool
func (ulp *UnifiedLiquidityPool) RemoveLiquidity(userID, asset string, shares *big.Int) (*big.Int, error) {
	ulp.mu.Lock()
	defer ulp.mu.Unlock()
	
	provider, exists := ulp.LiquidityProviders[userID]
	if !exists {
		return nil, errors.New("liquidity provider not found")
	}
	
	providerShares, exists := provider.ShareTokens[asset]
	if !exists || providerShares.Cmp(shares) < 0 {
		return nil, errors.New("insufficient shares")
	}
	
	// Check lockup period
	if lockup, exists := provider.LockedUntil[asset]; exists && time.Now().Before(lockup) {
		return nil, fmt.Errorf("liquidity locked until %s", lockup.Format(time.RFC3339))
	}
	
	// Calculate amount to return
	amount := ulp.calculateAmountFromShares(asset, shares)
	
	// Apply any impermanent loss
	if il, exists := provider.ImpermanentLoss[asset]; exists && il.Cmp(big.NewInt(0)) > 0 {
		amount.Sub(amount, il)
	}
	
	// Add earned fees and yield
	if fees, exists := provider.EarnedFees[asset]; exists {
		amount.Add(amount, fees)
	}
	if yield, exists := provider.EarnedYield[asset]; exists {
		amount.Add(amount, yield)
	}
	
	// Update provider
	provider.ShareTokens[asset].Sub(provider.ShareTokens[asset], shares)
	provider.ProvidedLiquidity[asset].Sub(provider.ProvidedLiquidity[asset], amount)
	
	// Update total liquidity
	ulp.TotalLiquidity[asset].Sub(ulp.TotalLiquidity[asset], amount)
	
	// Deallocate liquidity from markets
	ulp.deallocateLiquidity(asset, amount)
	
	return amount, nil
}

// allocateLiquidity allocates liquidity across different markets
func (ulp *UnifiedLiquidityPool) allocateLiquidity(asset string, amount *big.Int) {
	allocations := ulp.LiquidityAllocation.AllocationStrategy.CalculateAllocation(ulp)
	
	for market, allocation := range allocations {
		allocAmount := new(big.Int).Mul(amount, big.NewInt(int64(allocation*1000)))
		allocAmount.Div(allocAmount, big.NewInt(1000))
		
		switch market {
		case "spot":
			if ulp.SpotLiquidity[asset] == nil {
				ulp.SpotLiquidity[asset] = big.NewInt(0)
			}
			ulp.SpotLiquidity[asset].Add(ulp.SpotLiquidity[asset], allocAmount)
		case "margin":
			if ulp.MarginLiquidity[asset] == nil {
				ulp.MarginLiquidity[asset] = big.NewInt(0)
			}
			ulp.MarginLiquidity[asset].Add(ulp.MarginLiquidity[asset], allocAmount)
		case "perpetual":
			if ulp.PerpetualLiquidity[asset] == nil {
				ulp.PerpetualLiquidity[asset] = big.NewInt(0)
			}
			ulp.PerpetualLiquidity[asset].Add(ulp.PerpetualLiquidity[asset], allocAmount)
		case "vault":
			if ulp.VaultLiquidity[asset] == nil {
				ulp.VaultLiquidity[asset] = big.NewInt(0)
			}
			ulp.VaultLiquidity[asset].Add(ulp.VaultLiquidity[asset], allocAmount)
		}
	}
}

// deallocateLiquidity removes liquidity from markets
func (ulp *UnifiedLiquidityPool) deallocateLiquidity(asset string, amount *big.Int) {
	// Similar to allocateLiquidity but subtracts
	allocations := ulp.LiquidityAllocation.AllocationStrategy.CalculateAllocation(ulp)
	
	for market, allocation := range allocations {
		deallocAmount := new(big.Int).Mul(amount, big.NewInt(int64(allocation*1000)))
		deallocAmount.Div(deallocAmount, big.NewInt(1000))
		
		switch market {
		case "spot":
			if ulp.SpotLiquidity[asset] != nil {
				ulp.SpotLiquidity[asset].Sub(ulp.SpotLiquidity[asset], deallocAmount)
			}
		case "margin":
			if ulp.MarginLiquidity[asset] != nil {
				ulp.MarginLiquidity[asset].Sub(ulp.MarginLiquidity[asset], deallocAmount)
			}
		case "perpetual":
			if ulp.PerpetualLiquidity[asset] != nil {
				ulp.PerpetualLiquidity[asset].Sub(ulp.PerpetualLiquidity[asset], deallocAmount)
			}
		case "vault":
			if ulp.VaultLiquidity[asset] != nil {
				ulp.VaultLiquidity[asset].Sub(ulp.VaultLiquidity[asset], deallocAmount)
			}
		}
	}
}

// calculateShares calculates LP share tokens to mint
func (ulp *UnifiedLiquidityPool) calculateShares(asset string, amount *big.Int) *big.Int {
	totalLiquidity := ulp.TotalLiquidity[asset]
	if totalLiquidity == nil || totalLiquidity.Cmp(big.NewInt(0)) == 0 {
		return amount // First LP gets 1:1
	}
	
	totalShares := ulp.getTotalShares(asset)
	if totalShares.Cmp(big.NewInt(0)) == 0 {
		return amount
	}
	
	// shares = (amount * totalShares) / totalLiquidity
	shares := new(big.Int).Mul(amount, totalShares)
	shares.Div(shares, totalLiquidity)
	
	return shares
}

// calculateAmountFromShares calculates amount from LP shares
func (ulp *UnifiedLiquidityPool) calculateAmountFromShares(asset string, shares *big.Int) *big.Int {
	totalShares := ulp.getTotalShares(asset)
	if totalShares.Cmp(big.NewInt(0)) == 0 {
		return big.NewInt(0)
	}
	
	totalLiquidity := ulp.TotalLiquidity[asset]
	if totalLiquidity == nil {
		return big.NewInt(0)
	}
	
	// amount = (shares * totalLiquidity) / totalShares
	amount := new(big.Int).Mul(shares, totalLiquidity)
	amount.Div(amount, totalShares)
	
	return amount
}

// getTotalShares returns total LP shares for an asset
func (ulp *UnifiedLiquidityPool) getTotalShares(asset string) *big.Int {
	totalShares := big.NewInt(0)
	for _, provider := range ulp.LiquidityProviders {
		if shares, exists := provider.ShareTokens[asset]; exists {
			totalShares.Add(totalShares, shares)
		}
	}
	return totalShares
}

// shouldRebalance checks if rebalancing is needed
func (ulp *UnifiedLiquidityPool) shouldRebalance() bool {
	if time.Since(ulp.LastRebalance) < ulp.RebalanceInterval {
		return false
	}
	
	// Check if allocation deviates significantly
	for asset := range ulp.TotalLiquidity {
		currentAllocation := ulp.getCurrentAllocation(asset)
		targetAllocation := ulp.LiquidityAllocation.AllocationStrategy.CalculateAllocation(ulp)
		
		for market, target := range targetAllocation {
			current := currentAllocation[market]
			deviation := math.Abs(current - target)
			if deviation > ulp.RebalanceThreshold {
				return true
			}
		}
	}
	
	return false
}

// getCurrentAllocation returns current liquidity allocation
func (ulp *UnifiedLiquidityPool) getCurrentAllocation(asset string) map[string]float64 {
	total := ulp.TotalLiquidity[asset]
	if total == nil || total.Cmp(big.NewInt(0)) == 0 {
		return make(map[string]float64)
	}
	
	totalFloat := new(big.Float).SetInt(total)
	allocation := make(map[string]float64)
	
	if spot := ulp.SpotLiquidity[asset]; spot != nil {
		spotFloat := new(big.Float).SetInt(spot)
		spotFloat.Quo(spotFloat, totalFloat)
		allocation["spot"], _ = spotFloat.Float64()
	}
	
	if margin := ulp.MarginLiquidity[asset]; margin != nil {
		marginFloat := new(big.Float).SetInt(margin)
		marginFloat.Quo(marginFloat, totalFloat)
		allocation["margin"], _ = marginFloat.Float64()
	}
	
	if perp := ulp.PerpetualLiquidity[asset]; perp != nil {
		perpFloat := new(big.Float).SetInt(perp)
		perpFloat.Quo(perpFloat, totalFloat)
		allocation["perpetual"], _ = perpFloat.Float64()
	}
	
	if vault := ulp.VaultLiquidity[asset]; vault != nil {
		vaultFloat := new(big.Float).SetInt(vault)
		vaultFloat.Quo(vaultFloat, totalFloat)
		allocation["vault"], _ = vaultFloat.Float64()
	}
	
	return allocation
}

// rebalance rebalances liquidity across markets
func (ulp *UnifiedLiquidityPool) rebalance() {
	ulp.mu.Lock()
	defer ulp.mu.Unlock()
	
	for asset, totalLiquidity := range ulp.TotalLiquidity {
		if totalLiquidity.Cmp(big.NewInt(0)) == 0 {
			continue
		}
		
		targetAllocation := ulp.LiquidityAllocation.AllocationStrategy.CalculateAllocation(ulp)
		
		// Reallocate to match target
		for market, target := range targetAllocation {
			targetAmount := new(big.Int).Mul(totalLiquidity, big.NewInt(int64(target*1000)))
			targetAmount.Div(targetAmount, big.NewInt(1000))
			
			switch market {
			case "spot":
				ulp.SpotLiquidity[asset] = targetAmount
			case "margin":
				ulp.MarginLiquidity[asset] = targetAmount
			case "perpetual":
				ulp.PerpetualLiquidity[asset] = targetAmount
			case "vault":
				ulp.VaultLiquidity[asset] = targetAmount
			}
		}
	}
	
	ulp.LastRebalance = time.Now()
}

// updateProviderTier updates LP tier based on total liquidity
func (ulp *UnifiedLiquidityPool) updateProviderTier(provider *LiquidityProvider) {
	totalValue := big.NewInt(0)
	for _, amount := range provider.ProvidedLiquidity {
		totalValue.Add(totalValue, amount)
	}
	
	// Simple tier system based on total value
	valueFloat := new(big.Float).SetInt(totalValue)
	value, _ := valueFloat.Float64()
	
	switch {
	case value >= 1000000:
		provider.Tier = DiamondTier
		provider.FeeMultiplier = 1.5
		provider.YieldBoost = 1.3
	case value >= 500000:
		provider.Tier = PlatinumTier
		provider.FeeMultiplier = 1.3
		provider.YieldBoost = 1.2
	case value >= 100000:
		provider.Tier = GoldTier
		provider.FeeMultiplier = 1.2
		provider.YieldBoost = 1.15
	case value >= 50000:
		provider.Tier = SilverTier
		provider.FeeMultiplier = 1.1
		provider.YieldBoost = 1.05
	default:
		provider.Tier = BronzeTier
		provider.FeeMultiplier = 1.0
		provider.YieldBoost = 1.0
	}
}

// YieldRebalancer rebalances yield strategies
type YieldRebalancer struct {
	Strategies      []YieldStrategy
	CurrentYield    float64
	TargetYield     float64
	RebalancePeriod time.Duration
	LastRebalance   time.Time
}

// Settlement represents a settlement transaction
type Settlement struct {
	ID            string
	Asset         string
	Amount        *big.Int
	From          string
	To            string
	SettlementTime time.Time
	Status        SettlementStatus
	TxHash        string
}

// SettlementStatus moved to x_chain_types.go to avoid duplication