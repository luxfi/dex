// Package arbitrage provides omnichain arbitrage execution
package arbitrage

import (
	"context"
	"crypto/ecdsa"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

// Executor executes arbitrage opportunities
type Executor struct {
	mu sync.RWMutex

	// Configuration
	config ExecutorConfig

	// Wallet for signing transactions
	privateKey *ecdsa.PrivateKey
	address    string

	// Contract addresses
	flashLoanPool     string
	arbitrageContract string

	// Chain clients
	chains map[string]ChainClient

	// Execution tracking
	pendingExecutions   map[string]*Execution
	completedExecutions []Execution

	// Metrics
	totalExecutions      int64
	successfulExecutions int64
	totalProfitUSD       decimal.Decimal
	totalGasSpent        decimal.Decimal

	// Running state
	ctx    context.Context
	cancel context.CancelFunc
}

// ExecutorConfig configures the arbitrage executor
type ExecutorConfig struct {
	// Maximum gas price willing to pay (gwei)
	MaxGasPrice decimal.Decimal

	// Slippage tolerance (basis points)
	MaxSlippageBps decimal.Decimal

	// Minimum confidence to execute
	MinConfidence float64

	// Maximum concurrent executions
	MaxConcurrent int

	// Use flash loans
	UseFlashLoans bool

	// MEV protection
	UseMEVProtection bool
	FlashbotsRPC     string

	// Execution timeout
	ExecutionTimeout time.Duration
}

// ChainClient interface for interacting with different chains
type ChainClient interface {
	// SendTransaction sends a transaction
	SendTransaction(ctx context.Context, tx *Transaction) (string, error)

	// GetBalance gets token balance
	GetBalance(ctx context.Context, token, address string) (decimal.Decimal, error)

	// EstimateGas estimates gas for a transaction
	EstimateGas(ctx context.Context, tx *Transaction) (uint64, error)

	// GetGasPrice gets current gas price
	GetGasPrice(ctx context.Context) (decimal.Decimal, error)

	// WaitForConfirmation waits for transaction confirmation
	WaitForConfirmation(ctx context.Context, txHash string) (*Receipt, error)
}

// Transaction represents a blockchain transaction
type Transaction struct {
	To       string
	Value    *big.Int
	Data     []byte
	GasLimit uint64
	GasPrice *big.Int
	Nonce    uint64
}

// Receipt represents a transaction receipt
type Receipt struct {
	TxHash      string
	BlockNumber uint64
	GasUsed     uint64
	Status      bool
	Logs        []Log
}

// Log represents a transaction log
type Log struct {
	Address string
	Topics  []string
	Data    []byte
}

// Execution represents an arbitrage execution
type Execution struct {
	ID           string
	Opportunity  ArbitrageOpportunity
	Status       ExecutionStatus
	StartTime    time.Time
	EndTime      time.Time
	Transactions []ExecutedTx
	ActualPnL    decimal.Decimal
	GasSpent     decimal.Decimal
	Error        error
}

// ExecutedTx represents an executed transaction
type ExecutedTx struct {
	ChainID   string
	TxHash    string
	GasUsed   uint64
	Status    bool
	Timestamp time.Time
}

// ExecutionStatus represents execution status
type ExecutionStatus string

const (
	StatusPending   ExecutionStatus = "pending"
	StatusExecuting ExecutionStatus = "executing"
	StatusCompleted ExecutionStatus = "completed"
	StatusFailed    ExecutionStatus = "failed"
	StatusReverted  ExecutionStatus = "reverted"
)

// NewExecutor creates a new arbitrage executor
func NewExecutor(config ExecutorConfig, privateKey *ecdsa.PrivateKey) *Executor {
	ctx, cancel := context.WithCancel(context.Background())

	return &Executor{
		config:            config,
		privateKey:        privateKey,
		chains:            make(map[string]ChainClient),
		pendingExecutions: make(map[string]*Execution),
		ctx:               ctx,
		cancel:            cancel,
	}
}

// AddChainClient adds a chain client
func (e *Executor) AddChainClient(chainID string, client ChainClient) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.chains[chainID] = client
}

// SetContracts sets contract addresses
func (e *Executor) SetContracts(flashLoanPool, arbitrageContract string) {
	e.flashLoanPool = flashLoanPool
	e.arbitrageContract = arbitrageContract
}

// Execute executes an arbitrage opportunity
func (e *Executor) Execute(ctx context.Context, opp ArbitrageOpportunity) (*Execution, error) {
	// Validate opportunity
	if err := e.validateOpportunity(opp); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Create execution record
	exec := &Execution{
		ID:          opp.ID,
		Opportunity: opp,
		Status:      StatusPending,
		StartTime:   time.Now(),
	}

	e.mu.Lock()
	e.pendingExecutions[opp.ID] = exec
	e.mu.Unlock()

	// Execute based on configuration
	var err error
	if e.config.UseFlashLoans {
		err = e.executeWithFlashLoan(ctx, exec)
	} else {
		err = e.executeDirectly(ctx, exec)
	}

	exec.EndTime = time.Now()

	if err != nil {
		exec.Status = StatusFailed
		exec.Error = err
		return exec, err
	}

	exec.Status = StatusCompleted

	// Update metrics
	e.mu.Lock()
	delete(e.pendingExecutions, opp.ID)
	e.completedExecutions = append(e.completedExecutions, *exec)
	e.totalExecutions++
	if exec.Status == StatusCompleted {
		e.successfulExecutions++
		e.totalProfitUSD = e.totalProfitUSD.Add(exec.ActualPnL)
	}
	e.totalGasSpent = e.totalGasSpent.Add(exec.GasSpent)
	e.mu.Unlock()

	return exec, nil
}

// validateOpportunity validates an arbitrage opportunity before execution
func (e *Executor) validateOpportunity(opp ArbitrageOpportunity) error {
	// Check expiry
	if time.Now().After(opp.ExpiresAt) {
		return fmt.Errorf("opportunity expired")
	}

	// Check confidence
	if opp.Confidence < e.config.MinConfidence {
		return fmt.Errorf("confidence too low: %.2f < %.2f", opp.Confidence, e.config.MinConfidence)
	}

	// Check profitability
	if opp.NetPnL.LessThanOrEqual(decimal.Zero) {
		return fmt.Errorf("negative PnL: %s", opp.NetPnL.String())
	}

	// Check we have clients for all chains
	for _, route := range opp.Routes {
		if _, ok := e.chains[route.ChainID]; !ok {
			return fmt.Errorf("no client for chain: %s", route.ChainID)
		}
	}

	return nil
}

// executeWithFlashLoan executes using flash loan for capital efficiency
func (e *Executor) executeWithFlashLoan(ctx context.Context, exec *Execution) error {
	opp := exec.Opportunity
	exec.Status = StatusExecuting

	// Get Lux chain client
	luxClient, ok := e.chains["lux"]
	if !ok {
		return fmt.Errorf("no Lux client configured")
	}

	// Build flash loan parameters
	params := e.buildFlashLoanParams(opp)

	// Encode the flash loan call
	callData := e.encodeFlashLoanCall(opp, params)

	// Estimate gas
	tx := &Transaction{
		To:   e.flashLoanPool,
		Data: callData,
	}

	gasLimit, err := luxClient.EstimateGas(ctx, tx)
	if err != nil {
		return fmt.Errorf("gas estimation failed: %w", err)
	}

	// Get gas price
	gasPrice, err := luxClient.GetGasPrice(ctx)
	if err != nil {
		return fmt.Errorf("failed to get gas price: %w", err)
	}

	// Check gas price limit
	if gasPrice.GreaterThan(e.config.MaxGasPrice) {
		return fmt.Errorf("gas price too high: %s > %s", gasPrice.String(), e.config.MaxGasPrice.String())
	}

	// Build final transaction with overflow protection
	const gasBuffer uint64 = 50000
	if gasLimit > ^uint64(0)-gasBuffer {
		return fmt.Errorf("gas limit overflow: %d", gasLimit)
	}
	tx.GasLimit = gasLimit + gasBuffer
	tx.GasPrice = gasPrice.BigInt()

	// Send transaction (with MEV protection if enabled)
	var txHash string
	if e.config.UseMEVProtection {
		txHash, err = e.sendMEVProtected(ctx, tx)
	} else {
		txHash, err = luxClient.SendTransaction(ctx, tx)
	}

	if err != nil {
		return fmt.Errorf("failed to send transaction: %w", err)
	}

	exec.Transactions = append(exec.Transactions, ExecutedTx{
		ChainID:   "lux",
		TxHash:    txHash,
		Timestamp: time.Now(),
	})

	// Wait for confirmation
	receipt, err := luxClient.WaitForConfirmation(ctx, txHash)
	if err != nil {
		return fmt.Errorf("confirmation failed: %w", err)
	}

	if !receipt.Status {
		exec.Status = StatusReverted
		return fmt.Errorf("transaction reverted")
	}

	// Update execution with actual results
	exec.Transactions[0].GasUsed = receipt.GasUsed
	exec.Transactions[0].Status = receipt.Status
	exec.GasSpent = gasPrice.Mul(decimal.NewFromInt(int64(receipt.GasUsed)))

	// Parse logs to get actual profit
	exec.ActualPnL = e.parseProfit(receipt.Logs)

	return nil
}

// executeDirectly executes without flash loan (requires capital)
func (e *Executor) executeDirectly(ctx context.Context, exec *Execution) error {
	opp := exec.Opportunity
	exec.Status = StatusExecuting

	// Execute each route sequentially
	for i, route := range opp.Routes {
		client, ok := e.chains[route.ChainID]
		if !ok {
			return fmt.Errorf("no client for chain: %s", route.ChainID)
		}

		// Build swap transaction
		tx := e.buildSwapTransaction(route)

		// Send transaction
		txHash, err := client.SendTransaction(ctx, tx)
		if err != nil {
			return fmt.Errorf("route %d failed: %w", i, err)
		}

		exec.Transactions = append(exec.Transactions, ExecutedTx{
			ChainID:   route.ChainID,
			TxHash:    txHash,
			Timestamp: time.Now(),
		})

		// Wait for confirmation
		receipt, err := client.WaitForConfirmation(ctx, txHash)
		if err != nil {
			return fmt.Errorf("route %d confirmation failed: %w", i, err)
		}

		if !receipt.Status {
			exec.Status = StatusFailed
			return fmt.Errorf("route %d reverted", i)
		}

		exec.Transactions[i].GasUsed = receipt.GasUsed
		exec.Transactions[i].Status = receipt.Status
	}

	return nil
}

// sendMEVProtected sends transaction via MEV-protected channel
func (e *Executor) sendMEVProtected(ctx context.Context, tx *Transaction) (string, error) {
	// Implement Flashbots-style MEV protection
	// This sends the transaction directly to block builders
	// to avoid frontrunning/sandwich attacks

	// For Lux, we can use:
	// 1. Private mempool submission
	// 2. Direct validator communication
	// 3. Encrypted mempool (future)

	// Placeholder - implement actual MEV protection
	return "", fmt.Errorf("MEV protection not implemented")
}

// buildFlashLoanParams builds parameters for flash loan arbitrage
func (e *Executor) buildFlashLoanParams(opp ArbitrageOpportunity) []byte {
	// ABI encode the arbitrage parameters
	// This would be passed to the flash loan callback

	// ArbParams struct encoding:
	// - arbType (uint8)
	// - routes (Route[])
	// - minProfitBps (uint256)
	// - maxSlippageBps (uint256)
	// - deadline (uint256)

	// Placeholder - implement actual ABI encoding
	return nil
}

// encodeFlashLoanCall encodes the flash loan function call
func (e *Executor) encodeFlashLoanCall(opp ArbitrageOpportunity, params []byte) []byte {
	// function flashLoan(address receiver, address asset, uint256 amount, bytes calldata params)

	// Placeholder - implement actual ABI encoding
	return nil
}

// buildSwapTransaction builds a swap transaction for a route
func (e *Executor) buildSwapTransaction(route Route) *Transaction {
	// Build DEX-specific swap call based on route.Venue

	// Placeholder - implement actual swap encoding
	return &Transaction{
		To:   route.Venue,
		Data: route.SwapData,
	}
}

// parseProfit parses execution logs to determine actual profit
func (e *Executor) parseProfit(logs []Log) decimal.Decimal {
	// Parse ArbitrageExecuted event to get actual profit

	// Placeholder - implement log parsing
	return decimal.Zero
}

// GetMetrics returns execution metrics
func (e *Executor) GetMetrics() ExecutorMetrics {
	e.mu.RLock()
	defer e.mu.RUnlock()

	successRate := float64(0)
	if e.totalExecutions > 0 {
		successRate = float64(e.successfulExecutions) / float64(e.totalExecutions)
	}

	return ExecutorMetrics{
		TotalExecutions:      e.totalExecutions,
		SuccessfulExecutions: e.successfulExecutions,
		SuccessRate:          successRate,
		TotalProfitUSD:       e.totalProfitUSD,
		TotalGasSpent:        e.totalGasSpent,
		PendingExecutions:    int64(len(e.pendingExecutions)),
	}
}

// ExecutorMetrics holds executor statistics
type ExecutorMetrics struct {
	TotalExecutions      int64
	SuccessfulExecutions int64
	SuccessRate          float64
	TotalProfitUSD       decimal.Decimal
	TotalGasSpent        decimal.Decimal
	PendingExecutions    int64
}

// DefaultExecutorConfig returns default configuration
func DefaultExecutorConfig() ExecutorConfig {
	return ExecutorConfig{
		MaxGasPrice:      decimal.NewFromInt(100), // 100 gwei max
		MaxSlippageBps:   decimal.NewFromInt(50),  // 0.5% max slippage
		MinConfidence:    0.7,                     // 70% minimum confidence
		MaxConcurrent:    10,
		UseFlashLoans:    true,
		UseMEVProtection: true,
		ExecutionTimeout: 30 * time.Second,
	}
}
