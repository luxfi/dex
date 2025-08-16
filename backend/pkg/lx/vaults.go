package lx

import (
	"fmt"
	"math/big"
	"sync"
	"time"
)

// VaultManager manages all vaults in the system
type VaultManager struct {
	vaults map[string]*Vault
	engine *TradingEngine
	mu     sync.RWMutex
}

// NewVaultManager creates a new vault manager
func NewVaultManager(engine *TradingEngine) *VaultManager {
	return &VaultManager{
		vaults: make(map[string]*Vault),
		engine: engine,
	}
}

// CreateVault creates a new trading vault
func (vm *VaultManager) CreateVault(config VaultConfig) (*Vault, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	if _, exists := vm.vaults[config.ID]; exists {
		return nil, fmt.Errorf("vault %s already exists", config.ID)
	}

	vault := &Vault{
		ID:                 config.ID,
		Name:               config.Name,
		Description:        config.Description,
		TotalDeposits:      big.NewInt(0),
		TotalShares:        big.NewInt(0),
		HighWaterMark:      big.NewInt(0),
		Strategies:         make([]TradingStrategy, 0),
		Performance:        NewPerformanceMetrics(),
		Depositors:         make(map[string]*VaultPosition),
		Config:             config,
		State:              VaultStateActive,
		CreatedAt:          time.Now(),
		LastRebalance:      time.Now(),
		PendingDeposits:    make(map[string]*PendingDeposit),
		PendingWithdrawals: make(map[string]*PendingWithdrawal),
	}

	// Initialize strategies
	for _, stratConfig := range config.Strategies {
		strategy := vm.createStrategy(stratConfig)
		if strategy != nil {
			vault.Strategies = append(vault.Strategies, strategy)
		}
	}

	vm.vaults[config.ID] = vault
	return vault, nil
}

// VaultConfig configuration for a vault
type VaultConfig struct {
	ID                string
	Name              string
	Description       string
	ManagementFee     float64 // Annual management fee (e.g., 0.02 for 2%)
	PerformanceFee    float64 // Performance fee (e.g., 0.20 for 20%)
	MinDeposit        *big.Int
	MaxCapacity       *big.Int
	LockupPeriod      time.Duration
	Strategies        []StrategyConfig
	RiskLimits        RiskLimits
	AllowedAssets     []string
	RebalanceInterval time.Duration
}

// VaultPosition represents a depositor's position in a vault
type VaultPosition struct {
	User          string
	Shares        *big.Int  // Number of vault shares owned
	DepositValue  *big.Int  // Original deposit value
	CurrentValue  *big.Int  // Current value of shares
	LockedUntil   time.Time // Lockup period end
	LastUpdate    time.Time
	RealizedPnL   *big.Int
	UnrealizedPnL *big.Int
}

// VaultState represents the state of a vault
type VaultState int

const (
	VaultStateActive VaultState = iota
	VaultStatePaused
	VaultStateClosing
	VaultStateClosed
)

// PendingDeposit represents a pending deposit
type PendingDeposit struct {
	User      string
	Amount    *big.Int
	Timestamp time.Time
}

// PendingWithdrawal represents a pending withdrawal
type PendingWithdrawal struct {
	User      string
	Shares    *big.Int
	Timestamp time.Time
}

// Enhanced Vault struct
type Vault struct {
	ID                 string
	Name               string
	Description        string
	TotalDeposits      *big.Int
	TotalShares        *big.Int
	HighWaterMark      *big.Int
	Strategies         []TradingStrategy
	Performance        *PerformanceMetrics
	Depositors         map[string]*VaultPosition
	Config             VaultConfig
	State              VaultState
	CreatedAt          time.Time
	LastRebalance      time.Time
	PendingDeposits    map[string]*PendingDeposit
	PendingWithdrawals map[string]*PendingWithdrawal
	mu                 sync.RWMutex
}

// Deposit adds funds to the vault
func (v *Vault) Deposit(user string, amount *big.Int) (*VaultPosition, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Check minimum deposit
	if amount.Cmp(v.Config.MinDeposit) < 0 {
		return nil, fmt.Errorf("deposit below minimum: %s", v.Config.MinDeposit.String())
	}

	// Check capacity
	newTotal := new(big.Int).Add(v.TotalDeposits, amount)
	if v.Config.MaxCapacity != nil && newTotal.Cmp(v.Config.MaxCapacity) > 0 {
		return nil, fmt.Errorf("vault at capacity")
	}

	// Calculate shares to mint
	var shares *big.Int
	if v.TotalShares.Cmp(big.NewInt(0)) == 0 {
		// First deposit, 1:1 ratio
		shares = new(big.Int).Set(amount)
	} else {
		// shares = (amount * totalShares) / totalDeposits
		shares = new(big.Int).Mul(amount, v.TotalShares)
		shares.Div(shares, v.TotalDeposits)
	}

	// Update or create position
	position, exists := v.Depositors[user]
	if !exists {
		position = &VaultPosition{
			User:          user,
			Shares:        big.NewInt(0),
			DepositValue:  big.NewInt(0),
			CurrentValue:  big.NewInt(0),
			RealizedPnL:   big.NewInt(0),
			UnrealizedPnL: big.NewInt(0),
		}
		v.Depositors[user] = position
	}

	// Update position
	position.Shares.Add(position.Shares, shares)
	position.DepositValue.Add(position.DepositValue, amount)
	position.CurrentValue.Add(position.CurrentValue, amount)
	position.LockedUntil = time.Now().Add(v.Config.LockupPeriod)
	position.LastUpdate = time.Now()

	// Update vault totals
	v.TotalDeposits.Add(v.TotalDeposits, amount)
	v.TotalShares.Add(v.TotalShares, shares)

	// Update high water mark if needed
	if v.TotalDeposits.Cmp(v.HighWaterMark) > 0 {
		v.HighWaterMark.Set(v.TotalDeposits)
	}

	return position, nil
}

// Withdraw removes funds from the vault
func (v *Vault) Withdraw(user string, shares *big.Int) (*big.Int, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	position, exists := v.Depositors[user]
	if !exists {
		return nil, fmt.Errorf("no position found")
	}

	// Check lockup period
	if time.Now().Before(position.LockedUntil) {
		return nil, fmt.Errorf("position locked until %s", position.LockedUntil.Format(time.RFC3339))
	}

	// Check available shares
	if shares.Cmp(position.Shares) > 0 {
		return nil, fmt.Errorf("insufficient shares")
	}

	// Calculate withdrawal amount
	// amount = (shares * totalDeposits) / totalShares
	amount := new(big.Int).Mul(shares, v.TotalDeposits)
	amount.Div(amount, v.TotalShares)

	// Apply fees
	amount = v.applyWithdrawalFees(amount, position)

	// Update position
	position.Shares.Sub(position.Shares, shares)
	position.CurrentValue.Sub(position.CurrentValue, amount)

	// Calculate realized PnL
	proportionalDeposit := new(big.Int).Mul(position.DepositValue, shares)
	proportionalDeposit.Div(proportionalDeposit, new(big.Int).Add(position.Shares, shares))
	realizedPnL := new(big.Int).Sub(amount, proportionalDeposit)
	position.RealizedPnL.Add(position.RealizedPnL, realizedPnL)
	position.DepositValue.Sub(position.DepositValue, proportionalDeposit)

	position.LastUpdate = time.Now()

	// Update vault totals
	v.TotalDeposits.Sub(v.TotalDeposits, amount)
	v.TotalShares.Sub(v.TotalShares, shares)

	// Remove position if empty
	if position.Shares.Cmp(big.NewInt(0)) == 0 {
		delete(v.Depositors, user)
	}

	return amount, nil
}

// applyWithdrawalFees applies management and performance fees
func (v *Vault) applyWithdrawalFees(amount *big.Int, position *VaultPosition) *big.Int {
	// Calculate performance
	profit := new(big.Int).Sub(position.CurrentValue, position.DepositValue)

	if profit.Cmp(big.NewInt(0)) > 0 {
		// Apply performance fee on profits
		perfFee := new(big.Int).Mul(profit, big.NewInt(int64(v.Config.PerformanceFee*10000)))
		perfFee.Div(perfFee, big.NewInt(10000))
		amount.Sub(amount, perfFee)
	}

	// Apply management fee (prorated)
	daysSinceDeposit := time.Since(position.LastUpdate).Hours() / 24
	annualFeeRate := v.Config.ManagementFee
	proratedFee := annualFeeRate * (daysSinceDeposit / 365)
	mgmtFee := new(big.Int).Mul(amount, big.NewInt(int64(proratedFee*10000)))
	mgmtFee.Div(mgmtFee, big.NewInt(10000))
	amount.Sub(amount, mgmtFee)

	return amount
}

// ExecuteStrategies runs all vault strategies
func (v *Vault) ExecuteStrategies(market *OrderBook) []Order {
	v.mu.RLock()
	defer v.mu.RUnlock()

	if v.State != VaultStateActive {
		return nil
	}

	allOrders := make([]Order, 0)
	availableCapital := v.getAvailableCapital()

	for _, strategy := range v.Strategies {
		// Allocate capital to strategy
		strategyCapital := v.allocateCapital(strategy, availableCapital)

		// Execute strategy
		orders := strategy.Execute(market, strategyCapital)

		// Apply risk limits
		orders = v.applyRiskLimits(orders)

		allOrders = append(allOrders, orders...)
	}

	return allOrders
}

// getAvailableCapital returns capital available for trading
func (v *Vault) getAvailableCapital() *big.Int {
	// Reserve some capital for withdrawals
	reserved := new(big.Int).Mul(v.TotalDeposits, big.NewInt(10))
	reserved.Div(reserved, big.NewInt(100)) // 10% reserve

	available := new(big.Int).Sub(v.TotalDeposits, reserved)
	if available.Cmp(big.NewInt(0)) < 0 {
		return big.NewInt(0)
	}
	return available
}

// allocateCapital allocates capital to a strategy
func (v *Vault) allocateCapital(strategy TradingStrategy, totalCapital *big.Int) *big.Int {
	// Simple equal allocation for now
	numStrategies := len(v.Strategies)
	if numStrategies == 0 {
		return big.NewInt(0)
	}

	allocation := new(big.Int).Div(totalCapital, big.NewInt(int64(numStrategies)))

	// Apply strategy-specific limits
	limits := strategy.GetRiskLimits()
	if limits.MaxPositionValue != nil && allocation.Cmp(limits.MaxPositionValue) > 0 {
		allocation = limits.MaxPositionValue
	}

	return allocation
}

// applyRiskLimits applies vault-level risk limits to orders
func (v *Vault) applyRiskLimits(orders []Order) []Order {
	filtered := make([]Order, 0)
	totalValue := float64(0)

	for _, order := range orders {
		orderValue := order.Price * order.Size

		// Check position limits
		if v.Config.RiskLimits.MaxPositionSize > 0 && order.Size > v.Config.RiskLimits.MaxPositionSize {
			continue
		}

		// Check daily loss limit (simplified)
		if totalValue+orderValue > float64(v.TotalDeposits.Int64())*v.Config.RiskLimits.MaxDrawdown {
			continue
		}

		filtered = append(filtered, order)
		totalValue += orderValue
	}

	return filtered
}

// Rebalance rebalances the vault portfolio
func (v *Vault) Rebalance() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if time.Since(v.LastRebalance) < v.Config.RebalanceInterval {
		return nil // Too soon to rebalance
	}

	// Process pending deposits
	for user, pending := range v.PendingDeposits {
		_, err := v.Deposit(user, pending.Amount)
		if err == nil {
			delete(v.PendingDeposits, user)
		}
	}

	// Process pending withdrawals
	for user, pending := range v.PendingWithdrawals {
		_, err := v.Withdraw(user, pending.Shares)
		if err == nil {
			delete(v.PendingWithdrawals, user)
		}
	}

	// Rebalance strategies
	for _, strategy := range v.Strategies {
		if rebalancer, ok := strategy.(Rebalancer); ok {
			rebalancer.Rebalance()
		}
	}

	v.LastRebalance = time.Now()
	return nil
}

// UpdatePerformance updates vault performance metrics
func (v *Vault) UpdatePerformance(currentValue *big.Int) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.Performance.UpdatedAt = time.Now()

	// Calculate returns
	if v.TotalDeposits.Cmp(big.NewInt(0)) > 0 {
		returns := new(big.Float).SetInt(currentValue)
		deposits := new(big.Float).SetInt(v.TotalDeposits)
		returns.Quo(returns, deposits)
		returns.Sub(returns, big.NewFloat(1))

		v.Performance.TotalReturn, _ = returns.Float64()

		// Update other metrics
		v.Performance.calculateSharpe()
		v.Performance.calculateMaxDrawdown()
	}
}

// TradingStrategy interface
type TradingStrategy interface {
	Execute(market *OrderBook, capital *big.Int) []Order
	GetRiskLimits() RiskLimits
	GetName() string
	GetPerformance() *StrategyPerformance
}

// Rebalancer interface for strategies that need rebalancing
type Rebalancer interface {
	Rebalance() error
}

// StrategyConfig configuration for a strategy
type StrategyConfig struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Parameters map[string]interface{} `json:"parameters"`
	RiskLimits RiskLimits             `json:"risk_limits"`
}

// StrategyPerformance tracks strategy performance
type StrategyPerformance struct {
	TotalTrades   int
	WinningTrades int
	LosingTrades  int
	TotalPnL      *big.Int
	MaxDrawdown   float64
	SharpeRatio   float64
	LastUpdate    time.Time
}

// Enhanced RiskLimits
type RiskLimits struct {
	MaxPositionSize  float64
	MaxPositionValue *big.Int
	MaxLeverage      float64
	MaxDrawdown      float64
	DailyLossLimit   float64
	PositionLimits   map[string]float64 // per symbol limits
	MaxOpenPositions int
	MaxOrdersPerMin  int
	RequiredMargin   float64
}

// createStrategy creates a strategy from configuration
func (vm *VaultManager) createStrategy(config StrategyConfig) TradingStrategy {
	switch config.Type {
	case "market_making":
		return NewMarketMakingStrategy(config)
	case "momentum":
		return NewMomentumStrategy(config)
	case "arbitrage":
		return NewArbitrageStrategy(config)
	case "mean_reversion":
		return NewMeanReversionStrategy(config)
	default:
		return nil
	}
}

// NewPerformanceMetrics creates new performance metrics
func NewPerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		UpdatedAt: time.Now(),
	}
}

// PerformanceMetrics methods
func (pm *PerformanceMetrics) calculateSharpe() {
	// Simplified Sharpe ratio calculation
	// In production, would use historical returns and risk-free rate
	if pm.TotalReturn > 0 && pm.MaxDrawdown > 0 {
		pm.SharpeRatio = pm.TotalReturn / pm.MaxDrawdown
	}
}

func (pm *PerformanceMetrics) calculateMaxDrawdown() {
	// Simplified max drawdown calculation
	// In production, would track peak-to-trough decline
	if pm.MaxDrawdown == 0 {
		pm.MaxDrawdown = 0.05 // Default 5%
	}
}

// Example Strategy Implementations

// MarketMakingStrategy implements a simple market making strategy
type MarketMakingStrategy struct {
	config      StrategyConfig
	performance *StrategyPerformance
	spread      float64
	depth       int
	orderSize   float64
}

func NewMarketMakingStrategy(config StrategyConfig) *MarketMakingStrategy {
	spread := 0.001 // Default 0.1% spread
	if s, ok := config.Parameters["spread"].(float64); ok {
		spread = s
	}

	depth := 5 // Default 5 levels
	if d, ok := config.Parameters["depth"].(int); ok {
		depth = d
	}

	return &MarketMakingStrategy{
		config:      config,
		performance: &StrategyPerformance{TotalPnL: big.NewInt(0)},
		spread:      spread,
		depth:       depth,
		orderSize:   0.1, // Default order size
	}
}

func (s *MarketMakingStrategy) Execute(market *OrderBook, capital *big.Int) []Order {
	orders := make([]Order, 0)

	// Get current mid price
	snapshot := market.GetSnapshot()
	if len(snapshot.Bids) == 0 || len(snapshot.Asks) == 0 {
		return orders
	}

	midPrice := (snapshot.Bids[0].Price + snapshot.Asks[0].Price) / 2

	// Place orders on both sides
	for i := 0; i < s.depth; i++ {
		spreadMultiplier := float64(i+1) * s.spread

		// Buy order
		buyPrice := midPrice * (1 - spreadMultiplier)
		orders = append(orders, Order{
			Symbol:   market.Symbol,
			Side:     Buy,
			Type:     Limit,
			Price:    buyPrice,
			Size:     s.orderSize,
			PostOnly: true,
		})

		// Sell order
		sellPrice := midPrice * (1 + spreadMultiplier)
		orders = append(orders, Order{
			Symbol:   market.Symbol,
			Side:     Sell,
			Type:     Limit,
			Price:    sellPrice,
			Size:     s.orderSize,
			PostOnly: true,
		})
	}

	return orders
}

func (s *MarketMakingStrategy) GetRiskLimits() RiskLimits {
	return s.config.RiskLimits
}

func (s *MarketMakingStrategy) GetName() string {
	return s.config.Name
}

func (s *MarketMakingStrategy) GetPerformance() *StrategyPerformance {
	return s.performance
}

// MomentumStrategy implements a momentum trading strategy
type MomentumStrategy struct {
	config      StrategyConfig
	performance *StrategyPerformance
	lookback    int
	threshold   float64
}

func NewMomentumStrategy(config StrategyConfig) *MomentumStrategy {
	return &MomentumStrategy{
		config:      config,
		performance: &StrategyPerformance{TotalPnL: big.NewInt(0)},
		lookback:    20,
		threshold:   0.02,
	}
}

func (s *MomentumStrategy) Execute(market *OrderBook, capital *big.Int) []Order {
	// Simplified momentum strategy
	return []Order{}
}

func (s *MomentumStrategy) GetRiskLimits() RiskLimits {
	return s.config.RiskLimits
}

func (s *MomentumStrategy) GetName() string {
	return s.config.Name
}

func (s *MomentumStrategy) GetPerformance() *StrategyPerformance {
	return s.performance
}

// ArbitrageStrategy implements cross-market arbitrage
type ArbitrageStrategy struct {
	config      StrategyConfig
	performance *StrategyPerformance
}

func NewArbitrageStrategy(config StrategyConfig) *ArbitrageStrategy {
	return &ArbitrageStrategy{
		config:      config,
		performance: &StrategyPerformance{TotalPnL: big.NewInt(0)},
	}
}

func (s *ArbitrageStrategy) Execute(market *OrderBook, capital *big.Int) []Order {
	// Simplified arbitrage strategy
	return []Order{}
}

func (s *ArbitrageStrategy) GetRiskLimits() RiskLimits {
	return s.config.RiskLimits
}

func (s *ArbitrageStrategy) GetName() string {
	return s.config.Name
}

func (s *ArbitrageStrategy) GetPerformance() *StrategyPerformance {
	return s.performance
}

// MeanReversionStrategy implements mean reversion trading
type MeanReversionStrategy struct {
	config      StrategyConfig
	performance *StrategyPerformance
	window      int
	zScore      float64
}

func NewMeanReversionStrategy(config StrategyConfig) *MeanReversionStrategy {
	return &MeanReversionStrategy{
		config:      config,
		performance: &StrategyPerformance{TotalPnL: big.NewInt(0)},
		window:      50,
		zScore:      2.0,
	}
}

func (s *MeanReversionStrategy) Execute(market *OrderBook, capital *big.Int) []Order {
	// Simplified mean reversion strategy
	return []Order{}
}

func (s *MeanReversionStrategy) GetRiskLimits() RiskLimits {
	return s.config.RiskLimits
}

func (s *MeanReversionStrategy) GetName() string {
	return s.config.Name
}

func (s *MeanReversionStrategy) GetPerformance() *StrategyPerformance {
	return s.performance
}

// GetVault returns a vault by ID
func (vm *VaultManager) GetVault(id string) (*Vault, error) {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	vault, exists := vm.vaults[id]
	if !exists {
		return nil, ErrVaultNotFound
	}
	return vault, nil
}

// ListVaults returns all vaults
func (vm *VaultManager) ListVaults() []*Vault {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	vaults := make([]*Vault, 0, len(vm.vaults))
	for _, vault := range vm.vaults {
		vaults = append(vaults, vault)
	}
	return vaults
}

// GetVaultPerformance returns vault performance metrics
func (vm *VaultManager) GetVaultPerformance(id string) (*PerformanceMetrics, error) {
	vault, err := vm.GetVault(id)
	if err != nil {
		return nil, err
	}

	vault.mu.RLock()
	defer vault.mu.RUnlock()

	return vault.Performance, nil
}

// Helper function for big.Int math
func mulDiv(a, b, c *big.Int) *big.Int {
	result := new(big.Int).Mul(a, b)
	return result.Div(result, c)
}
