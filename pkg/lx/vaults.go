package lx

import (
	"fmt"
	"math/big"
	"sync"
	"time"
)

// VaultManager manages all vaults in the system
type VaultManager struct {
	vaults       map[string]*Vault
	copyVaults   map[string]*CopyVault // Copy-trading vaults with 10% profit share
	userVaults   map[string][]string   // userAddress -> vaultIDs they're in
	leaderVaults map[string][]string   // leaderAddress -> vaultIDs they lead
	engine       *TradingEngine
	mu           sync.RWMutex
}

// NewVaultManager creates a new vault manager
func NewVaultManager(engine *TradingEngine) *VaultManager {
	return &VaultManager{
		vaults:       make(map[string]*Vault),
		copyVaults:   make(map[string]*CopyVault),
		userVaults:   make(map[string][]string),
		leaderVaults: make(map[string][]string),
		engine:       engine,
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
	InsuranceCoverage *big.Int
	RecoveryAddresses []string
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

// CopyVault represents a copy-trading vault
type CopyVault struct {
	ID            string
	Name          string
	Description   string
	Leader        string  // Leader trader address
	ProfitShare   float64 // Share of profits (default 10%)
	TotalDeposits *big.Int
	TotalShares   *big.Int
	Followers     map[string]*VaultPosition // follower -> position
	Performance   *PerformanceMetrics
	State         VaultState
	CreatedAt     time.Time
	mu            sync.RWMutex
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

// ============================================
// Copy-Trading Vault Implementation (10% Profit Share)
// ============================================

// CreateVaultWithProfitShare creates a new vault with 10% profit share for leaders
func (vm *VaultManager) CreateVaultWithProfitShare(leader string, name string, description string, initialDeposit *big.Int) (*CopyVault, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	// Validate minimum deposit (100 USDC)
	minDeposit := big.NewInt(100 * 1e6) // 100 USDC with 6 decimals
	if initialDeposit.Cmp(minDeposit) < 0 {
		return nil, fmt.Errorf("initial deposit must be at least 100 USDC")
	}

	// Generate vault ID (use up to 8 chars of leader)
	leaderPrefix := leader
	if len(leader) > 8 {
		leaderPrefix = leader[:8]
	}
	vaultID := fmt.Sprintf("copy_%s_%d", leaderPrefix, time.Now().Unix())

	// Check if vault already exists
	if _, exists := vm.copyVaults[vaultID]; exists {
		return nil, fmt.Errorf("vault %s already exists", vaultID)
	}

	// Calculate leader's initial shares (always starts with 100% of shares)
	initialShares := big.NewInt(1000000) // 1M shares initially

	vault := &CopyVault{
		ID:            vaultID,
		Name:          name,
		Description:   description,
		Leader:        leader,
		ProfitShare:   0.10, // 10% profit share for leader
		TotalDeposits: new(big.Int).Set(initialDeposit),
		TotalShares:   new(big.Int).Set(initialShares),
		Followers:     make(map[string]*VaultPosition),
		Performance:   NewPerformanceMetrics(),
		State:         VaultStateActive,
		CreatedAt:     time.Now(),
	}

	// Add leader as first member with initial position
	vault.Followers[leader] = &VaultPosition{
		User:          leader,
		Shares:        new(big.Int).Set(initialShares),
		DepositValue:  new(big.Int).Set(initialDeposit),
		CurrentValue:  new(big.Int).Set(initialDeposit),
		RealizedPnL:   big.NewInt(0),
		UnrealizedPnL: big.NewInt(0),
		LastUpdate:    time.Now(),
	}

	// Store vault
	vm.copyVaults[vaultID] = vault

	// Update mappings
	if vm.leaderVaults[leader] == nil {
		vm.leaderVaults[leader] = []string{}
	}
	vm.leaderVaults[leader] = append(vm.leaderVaults[leader], vaultID)

	if vm.userVaults[leader] == nil {
		vm.userVaults[leader] = []string{}
	}
	vm.userVaults[leader] = append(vm.userVaults[leader], vaultID)

	return vault, nil
}

// JoinVault allows a user to join a vault
func (vm *VaultManager) JoinVault(vaultID string, userAddress string, depositAmount *big.Int) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	vault, exists := vm.copyVaults[vaultID]
	if !exists {
		return fmt.Errorf("copy vault not found")
	}

	vault.mu.Lock()
	defer vault.mu.Unlock()

	// Check if vault is active
	if vault.State != VaultStateActive {
		return fmt.Errorf("vault is not accepting new members")
	}

	// Check minimum deposit (100 USDC)
	minDeposit := big.NewInt(100 * 1e6)
	if depositAmount.Cmp(minDeposit) < 0 {
		return fmt.Errorf("minimum deposit is 100 USDC")
	}

	// Check if already a member
	if _, isMember := vault.Followers[userAddress]; isMember {
		return fmt.Errorf("already a member of this vault")
	}

	// Calculate shares based on current vault value
	var shares *big.Int
	if vault.TotalShares.Cmp(big.NewInt(0)) == 0 {
		shares = new(big.Int).Set(depositAmount)
	} else {
		// shares = (deposit / total_deposits) * total_shares
		shares = new(big.Int).Mul(depositAmount, vault.TotalShares)
		shares.Div(shares, vault.TotalDeposits)
	}

	// Create follower position
	position := &VaultPosition{
		User:          userAddress,
		Shares:        shares,
		DepositValue:  new(big.Int).Set(depositAmount),
		CurrentValue:  new(big.Int).Set(depositAmount),
		RealizedPnL:   big.NewInt(0),
		UnrealizedPnL: big.NewInt(0),
		LastUpdate:    time.Now(),
	}

	// Update vault state
	vault.Followers[userAddress] = position
	vault.TotalShares = new(big.Int).Add(vault.TotalShares, shares)
	vault.TotalDeposits = new(big.Int).Add(vault.TotalDeposits, depositAmount)

	// Update user mappings
	if vm.userVaults[userAddress] == nil {
		vm.userVaults[userAddress] = []string{}
	}
	vm.userVaults[userAddress] = append(vm.userVaults[userAddress], vaultID)

	// Ensure leader maintains at least 5% share
	if !vm.checkLeaderMinimumShare(vault) {
		// Revert changes
		delete(vault.Followers, userAddress)
		vault.TotalShares = new(big.Int).Sub(vault.TotalShares, shares)
		vault.TotalDeposits = new(big.Int).Sub(vault.TotalDeposits, depositAmount)

		// Remove from user mappings
		userVaults := vm.userVaults[userAddress]
		for i, id := range userVaults {
			if id == vaultID {
				vm.userVaults[userAddress] = append(userVaults[:i], userVaults[i+1:]...)
				break
			}
		}

		return fmt.Errorf("accepting this deposit would dilute leader below 5%% minimum share")
	}

	return nil
}

// WithdrawFromVault allows withdrawal with 10% profit share to leader
func (vm *VaultManager) WithdrawFromVault(vaultID string, userAddress string, sharePercent float64) (*big.Int, error) {
	if sharePercent <= 0 || sharePercent > 1.0 {
		return nil, fmt.Errorf("share percent must be between 0 and 1")
	}

	vm.mu.Lock()
	defer vm.mu.Unlock()

	vault, exists := vm.copyVaults[vaultID]
	if !exists {
		return nil, fmt.Errorf("copy vault not found")
	}

	vault.mu.Lock()
	defer vault.mu.Unlock()

	position, isMember := vault.Followers[userAddress]
	if !isMember {
		return nil, fmt.Errorf("not a member of this vault")
	}

	// Calculate shares to withdraw
	sharesToWithdraw := new(big.Float).Mul(
		new(big.Float).SetInt(position.Shares),
		big.NewFloat(sharePercent),
	)
	sharesToWithdrawInt, _ := sharesToWithdraw.Int(nil)

	// Calculate value of shares
	shareValue := new(big.Int).Mul(sharesToWithdrawInt, vault.TotalDeposits)
	shareValue.Div(shareValue, vault.TotalShares)

	// Calculate profit and apply 10% profit share for leader (only for followers)
	withdrawalAmount := new(big.Int).Set(shareValue)
	if userAddress != vault.Leader {
		// Calculate proportional entry value
		proportionalEntry := new(big.Int).Mul(position.DepositValue, sharesToWithdrawInt)
		proportionalEntry.Div(proportionalEntry, position.Shares)

		// If withdrawing with profit, apply 10% profit share
		if shareValue.Cmp(proportionalEntry) > 0 {
			profit := new(big.Int).Sub(shareValue, proportionalEntry)
			profitShare := new(big.Int).Mul(profit, big.NewInt(10))
			profitShare.Div(profitShare, big.NewInt(100))

			// Deduct profit share from withdrawal
			withdrawalAmount.Sub(withdrawalAmount, profitShare)

			// Add profit share to leader's position
			if leaderPosition, exists := vault.Followers[vault.Leader]; exists {
				leaderPosition.CurrentValue.Add(leaderPosition.CurrentValue, profitShare)
				leaderPosition.RealizedPnL.Add(leaderPosition.RealizedPnL, profitShare)
			}
		}
	}

	// Check if leader withdrawal would go below 5% minimum
	if userAddress == vault.Leader {
		newLeaderShares := new(big.Int).Sub(position.Shares, sharesToWithdrawInt)
		newTotalShares := new(big.Int).Sub(vault.TotalShares, sharesToWithdrawInt)

		if newTotalShares.Sign() > 0 {
			leaderPercent := float64(newLeaderShares.Int64()) / float64(newTotalShares.Int64())
			if leaderPercent < 0.05 {
				return nil, fmt.Errorf("withdrawal would put leader below 5%% minimum share")
			}
		}
	}

	// Update position
	position.Shares.Sub(position.Shares, sharesToWithdrawInt)
	position.CurrentValue.Sub(position.CurrentValue, shareValue)

	// Calculate realized PnL
	proportionalDeposit := new(big.Int).Mul(position.DepositValue, sharesToWithdrawInt)
	proportionalDeposit.Div(proportionalDeposit, new(big.Int).Add(position.Shares, sharesToWithdrawInt))
	realizedPnL := new(big.Int).Sub(withdrawalAmount, proportionalDeposit)
	position.RealizedPnL.Add(position.RealizedPnL, realizedPnL)
	position.DepositValue.Sub(position.DepositValue, proportionalDeposit)
	position.LastUpdate = time.Now()

	// Update vault totals
	vault.TotalShares.Sub(vault.TotalShares, sharesToWithdrawInt)
	vault.TotalDeposits.Sub(vault.TotalDeposits, shareValue)

	// Remove member if fully withdrawn
	if position.Shares.Sign() == 0 {
		delete(vault.Followers, userAddress)

		// Remove from user mappings
		userVaults := vm.userVaults[userAddress]
		for i, id := range userVaults {
			if id == vaultID {
				vm.userVaults[userAddress] = append(userVaults[:i], userVaults[i+1:]...)
				break
			}
		}

		if len(vm.userVaults[userAddress]) == 0 {
			delete(vm.userVaults, userAddress)
		}
	}

	return withdrawalAmount, nil
}

// checkLeaderMinimumShare ensures leader maintains at least 5% of vault
func (vm *VaultManager) checkLeaderMinimumShare(vault *CopyVault) bool {
	leaderPosition, exists := vault.Followers[vault.Leader]
	if !exists || vault.TotalShares.Sign() == 0 {
		return false
	}

	leaderPercent := float64(leaderPosition.Shares.Int64()) / float64(vault.TotalShares.Int64())
	return leaderPercent >= 0.05
}

// GetVaultByID returns a vault by ID
func (vm *VaultManager) GetVaultByID(vaultID string) (*CopyVault, error) {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	vault, exists := vm.copyVaults[vaultID]
	if !exists {
		return nil, fmt.Errorf("vault not found")
	}

	return vault, nil
}

// GetUserVaults returns all vaults a user is part of
func (vm *VaultManager) GetUserVaults(userAddress string) ([]*CopyVault, error) {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	vaultIDs := vm.userVaults[userAddress]
	vaults := make([]*CopyVault, 0)

	for _, id := range vaultIDs {
		if vault, exists := vm.copyVaults[id]; exists {
			vaults = append(vaults, vault)
		}
	}

	return vaults, nil
}

// GetLeaderVaults returns all vaults led by a specific leader
func (vm *VaultManager) GetLeaderVaults(leaderAddress string) ([]*CopyVault, error) {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	vaultIDs := vm.leaderVaults[leaderAddress]
	vaults := make([]*CopyVault, 0)

	for _, id := range vaultIDs {
		if vault, exists := vm.copyVaults[id]; exists {
			vaults = append(vaults, vault)
		}
	}

	return vaults, nil
}

// UpdateVaultValue updates the vault's total value based on PnL from trading
// This is how Hyperliquid handles it - the vault trades as one entity and value changes affect all members proportionally
func (vm *VaultManager) UpdateVaultValue(vaultID string, newValue *big.Int) error {
	vm.mu.RLock()
	vault, exists := vm.copyVaults[vaultID]
	vm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("vault not found")
	}

	vault.mu.Lock()
	defer vault.mu.Unlock()

	// Update vault's total value
	vault.TotalDeposits = newValue

	// Update each member's current value proportionally
	for _, position := range vault.Followers {
		// Calculate member's share of the new value
		memberValue := new(big.Int).Mul(position.Shares, newValue)
		memberValue.Div(memberValue, vault.TotalShares)

		// Update unrealized PnL
		position.UnrealizedPnL = new(big.Int).Sub(memberValue, position.DepositValue)
		position.CurrentValue = memberValue
		position.LastUpdate = time.Now()
	}

	// Update high water mark if needed (for profit share calculation)
	returnRatio := float64(newValue.Int64()) / float64(vault.TotalDeposits.Int64())
	if returnRatio > vault.Performance.TotalReturn {
		vault.Performance.TotalReturn = returnRatio
	}

	vault.Performance.UpdatedAt = time.Now()

	return nil
}

// ExecuteVaultTrade places a trade using the vault's total capital
// The vault trades as a single entity, not individual user orders
func (vm *VaultManager) ExecuteVaultTrade(vaultID string, symbol string, side Side, size float64, orderType OrderType) error {
	vm.mu.RLock()
	vault, exists := vm.copyVaults[vaultID]
	vm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("vault not found")
	}

	vault.mu.Lock()
	defer vault.mu.Unlock()

	// Check if vault is active
	if vault.State != VaultStateActive {
		return fmt.Errorf("vault is not active")
	}

	// Create order for the entire vault (leader trades on behalf of all)
	_ = Order{
		Symbol:   symbol,
		Side:     side,
		Type:     orderType,
		Size:     size,
		User:     vault.Leader, // Leader executes on behalf of vault
		ClientID: vaultID,      // Track that this is a vault order
	}

	// This would integrate with the actual order book
	// The order is placed with the vault's total capital
	// PnL from this trade will update the vault's total value
	// which then affects all members proportionally

	// TODO: vm.engine.orderBook.AddOrder(&vaultOrder)

	return nil
}
