package lx

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"
)

// SimpleVault represents a simplified vault that trades as one entity
type SimpleVault struct {
	ID       string
	Leader   string
	Strategy VaultStrategy // The strategy this vault uses (can be AI, MM, etc)
	
	// Capital tracking
	TotalValue    *big.Int            // Current total value
	ShareSupply   *big.Int            // Total shares issued
	MemberShares  map[string]*big.Int // address -> shares owned
	
	// Performance
	HighWaterMark *big.Int // For profit share calculation
	ProfitShare   float64  // Leader's profit share (10%)
	
	// Integration
	OrderBookID string // Which orderbook this vault trades on
	SubaccountID string // Clearinghouse subaccount for this vault
	
	mu sync.RWMutex
}

// SimpleVaultManager manages vaults with strategies
type SimpleVaultManager struct {
	vaults         map[string]*SimpleVault
	clearinghouse  *ClearingHouse
	strategyEngine *StrategyEngine
	
	mu sync.RWMutex
}

// NewSimpleVaultManager creates a simplified vault manager
func NewSimpleVaultManager(ch *ClearingHouse) *SimpleVaultManager {
	return &SimpleVaultManager{
		vaults:         make(map[string]*SimpleVault),
		clearinghouse:  ch,
		strategyEngine: NewStrategyEngine(10 * time.Millisecond), // 10ms tick for HFT
	}
}

// CreateVaultWithStrategy creates a vault with a specific trading strategy
func (vm *SimpleVaultManager) CreateVaultWithStrategy(
	leader string,
	deposit *big.Int,
	strategyType string,
	strategyConfig map[string]interface{},
) (*SimpleVault, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	
	// Minimum 100 USDC
	minDeposit := big.NewInt(100 * 1e6)
	if deposit.Cmp(minDeposit) < 0 {
		return nil, fmt.Errorf("minimum deposit is 100 USDC")
	}
	
	vaultID := fmt.Sprintf("v_%s_%d", leader[:8], time.Now().Unix())
	
	// Create clearinghouse subaccount for this vault
	subaccountID := fmt.Sprintf("vault_%s", vaultID)
	if err := vm.clearinghouse.Deposit(subaccountID, deposit); err != nil {
		return nil, err
	}
	
	// Create the vault
	vault := &SimpleVault{
		ID:            vaultID,
		Leader:        leader,
		TotalValue:    new(big.Int).Set(deposit),
		ShareSupply:   big.NewInt(1000000), // 1M initial shares
		MemberShares:  make(map[string]*big.Int),
		HighWaterMark: new(big.Int).Set(deposit),
		ProfitShare:   0.10, // 10% profit share
		SubaccountID:  subaccountID,
		OrderBookID:   "BTC-USD", // Default, can be configured
	}
	
	// Leader gets all initial shares
	vault.MemberShares[leader] = new(big.Int).Set(vault.ShareSupply)
	
	// Create and register strategy
	strategy, err := vm.createStrategy(strategyType, strategyConfig)
	if err != nil {
		return nil, err
	}
	
	vault.Strategy = strategy
	
	// Register with strategy engine
	if err := vm.strategyEngine.RegisterStrategy(vaultID, strategy, deposit); err != nil {
		return nil, err
	}
	
	vm.vaults[vaultID] = vault
	
	return vault, nil
}

// createStrategy creates a strategy based on type
func (vm *SimpleVaultManager) createStrategy(strategyType string, config map[string]interface{}) (VaultStrategy, error) {
	switch strategyType {
	case "ai_hft":
		// AI-driven HFT strategy
		endpoint := config["model_endpoint"].(string)
		return NewAIStrategy(endpoint), nil
		
	case "simple_mm":
		// Simple market making
		spreadBps := config["spread_bps"].(int)
		orderSize := config["order_size"].(float64)
		return NewSimpleMMStrategy(spreadBps, orderSize), nil
		
	default:
		return nil, fmt.Errorf("unknown strategy type: %s", strategyType)
	}
}

// Deposit adds funds to a vault
func (vm *SimpleVaultManager) Deposit(vaultID string, user string, amount *big.Int) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	
	vault, exists := vm.vaults[vaultID]
	if !exists {
		return fmt.Errorf("vault not found")
	}
	
	vault.mu.Lock()
	defer vault.mu.Unlock()
	
	// Calculate shares to mint based on current value
	sharesToMint := new(big.Int).Mul(amount, vault.ShareSupply)
	sharesToMint.Div(sharesToMint, vault.TotalValue)
	
	// Update user's shares
	if vault.MemberShares[user] == nil {
		vault.MemberShares[user] = big.NewInt(0)
	}
	vault.MemberShares[user].Add(vault.MemberShares[user], sharesToMint)
	
	// Update vault totals
	vault.ShareSupply.Add(vault.ShareSupply, sharesToMint)
	vault.TotalValue.Add(vault.TotalValue, amount)
	
	// Deposit to clearinghouse
	return vm.clearinghouse.Deposit(vault.SubaccountID, amount)
}

// Withdraw removes funds from a vault
func (vm *SimpleVaultManager) Withdraw(vaultID string, user string, sharePercent float64) (*big.Int, error) {
	if sharePercent <= 0 || sharePercent > 1 {
		return nil, fmt.Errorf("invalid share percent")
	}
	
	vm.mu.Lock()
	defer vm.mu.Unlock()
	
	vault, exists := vm.vaults[vaultID]
	if !exists {
		return nil, fmt.Errorf("vault not found")
	}
	
	vault.mu.Lock()
	defer vault.mu.Unlock()
	
	userShares := vault.MemberShares[user]
	if userShares == nil || userShares.Sign() == 0 {
		return nil, fmt.Errorf("no shares owned")
	}
	
	// Calculate shares to burn
	sharesToBurn := new(big.Float).Mul(
		new(big.Float).SetInt(userShares),
		big.NewFloat(sharePercent),
	)
	sharesToBurnInt, _ := sharesToBurn.Int(nil)
	
	// Calculate value to withdraw
	withdrawValue := new(big.Int).Mul(sharesToBurnInt, vault.TotalValue)
	withdrawValue.Div(withdrawValue, vault.ShareSupply)
	
	// Apply profit share if withdrawing with profit (only for non-leaders)
	if user != vault.Leader && vault.TotalValue.Cmp(vault.HighWaterMark) > 0 {
		// Calculate user's share of profit above high water mark
		profit := new(big.Int).Sub(vault.TotalValue, vault.HighWaterMark)
		userProfitShare := new(big.Int).Mul(profit, sharesToBurnInt)
		userProfitShare.Div(userProfitShare, vault.ShareSupply)
		
		// Leader gets 10% of user's profit
		leaderFee := new(big.Int).Mul(userProfitShare, big.NewInt(int64(vault.ProfitShare * 100)))
		leaderFee.Div(leaderFee, big.NewInt(100))
		
		// Deduct from withdrawal
		withdrawValue.Sub(withdrawValue, leaderFee)
		
		// Add to leader's value (by minting shares)
		leaderShares := new(big.Int).Mul(leaderFee, vault.ShareSupply)
		leaderShares.Div(leaderShares, vault.TotalValue)
		vault.MemberShares[vault.Leader].Add(vault.MemberShares[vault.Leader], leaderShares)
		vault.ShareSupply.Add(vault.ShareSupply, leaderShares)
	}
	
	// Update shares
	vault.MemberShares[user].Sub(vault.MemberShares[user], sharesToBurnInt)
	vault.ShareSupply.Sub(vault.ShareSupply, sharesToBurnInt)
	vault.TotalValue.Sub(vault.TotalValue, withdrawValue)
	
	// Remove user if no shares left
	if vault.MemberShares[user].Sign() == 0 {
		delete(vault.MemberShares, user)
	}
	
	// Withdraw from clearinghouse
	// vm.clearinghouse.Withdraw(vault.SubaccountID, withdrawValue)
	
	return withdrawValue, nil
}

// UpdateVaultValue updates vault value based on PnL
func (vm *SimpleVaultManager) UpdateVaultValue(vaultID string) error {
	vm.mu.RLock()
	vault, exists := vm.vaults[vaultID]
	vm.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("vault not found")
	}
	
	vault.mu.Lock()
	defer vault.mu.Unlock()
	
	// Get current value from clearinghouse account
	// This includes all positions and unrealized PnL
	// currentValue := vm.clearinghouse.GetAccountValue(vault.SubaccountID)
	
	// For now, simulate with strategy performance
	if vault.Strategy != nil {
		metrics := vault.Strategy.GetPerformance()
		if metrics.PnL != nil {
			vault.TotalValue.Add(vault.TotalValue, metrics.PnL)
		}
	}
	
	// Update high water mark
	if vault.TotalValue.Cmp(vault.HighWaterMark) > 0 {
		vault.HighWaterMark.Set(vault.TotalValue)
	}
	
	return nil
}

// GetVaultInfo returns vault information
func (vm *SimpleVaultManager) GetVaultInfo(vaultID string) (map[string]interface{}, error) {
	vm.mu.RLock()
	vault, exists := vm.vaults[vaultID]
	vm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("vault not found")
	}
	
	vault.mu.RLock()
	defer vault.mu.RUnlock()
	
	info := map[string]interface{}{
		"id":             vault.ID,
		"leader":         vault.Leader,
		"total_value":    vault.TotalValue.String(),
		"share_supply":   vault.ShareSupply.String(),
		"high_water_mark": vault.HighWaterMark.String(),
		"member_count":   len(vault.MemberShares),
	}
	
	// Add strategy performance
	if vault.Strategy != nil {
		metrics := vault.Strategy.GetPerformance()
		info["strategy_pnl"] = metrics.PnL.String()
		info["trade_count"] = metrics.TradeCount
		info["win_rate"] = metrics.WinRate
		info["sharpe_ratio"] = metrics.SharpeRatio
	}
	
	return info, nil
}

// StartStrategyExecution starts executing all vault strategies
func (vm *SimpleVaultManager) StartStrategyExecution(ctx context.Context) {
	go vm.strategyEngine.Start(ctx)
}

// Integration with OrderBook and ClearingHouse

// ExecuteVaultOrder places an order for a vault
func (vm *SimpleVaultManager) ExecuteVaultOrder(vaultID string, order *Order) error {
	vault, exists := vm.vaults[vaultID]
	if !exists {
		return fmt.Errorf("vault not found")
	}
	
	// Set the order's user to the vault's subaccount
	order.User = vault.SubaccountID
	
	// The order goes through normal order book processing
	// The clearinghouse tracks the vault's positions
	// PnL updates the vault's total value
	
	return nil
}

// Example: Creating an AI-powered HFT vault
func ExampleCreateAIVault() {
	ch := &ClearingHouse{} // Would use real clearinghouse
	vm := NewSimpleVaultManager(ch)
	
	// Create vault with AI strategy
	config := map[string]interface{}{
		"model_endpoint": "localhost:8080", // Colocated AI model
		"max_position":   0.1,               // 10% max position
		"min_spread":     0.001,             // 0.1% min spread
	}
	
	leader := "leader_address"
	deposit := big.NewInt(10000 * 1e6) // 10,000 USDC
	
	vault, err := vm.CreateVaultWithStrategy(leader, deposit, "ai_hft", config)
	if err != nil {
		panic(err)
	}
	
	// Start strategy execution
	ctx := context.Background()
	vm.StartStrategyExecution(ctx)
	
	// The AI strategy now runs automatically:
	// 1. Every 10ms tick, it gets market data
	// 2. Calls colocated AI model for signals
	// 3. Places orders through the order book
	// 4. PnL updates vault value
	// 5. Members can deposit/withdraw based on shares
	
	_ = vault
}