package lx

import (
	"context"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"
)

// BridgeAsset represents an asset that can be bridged
type BridgeAsset struct {
	Symbol          string
	Name            string
	Decimals        uint8
	SourceContract  string // Contract on source chain
	WrappedContract map[string]string // chainID -> wrapped contract
	MinTransfer     *big.Int
	MaxTransfer     *big.Int
	DailyLimit      *big.Int
	DailyVolume     *big.Int // Current daily volume
	LastReset       time.Time
	Paused          bool
}

// BridgeTransfer represents a cross-chain transfer
type BridgeTransfer struct {
	ID               string
	Asset            string
	Amount           *big.Int
	Fee              *big.Int
	SourceChain      string
	DestChain        string
	SourceAddress    string
	DestAddress      string
	SourceTxHash     string
	DestTxHash       string
	Status           BridgeStatus
	Confirmations    int
	RequiredConfirms int
	InitiatedAt      time.Time
	CompletedAt      time.Time
	Validators       map[string]*BridgeSignature // validator -> signature
	Nonce            uint64
	ExpiryTime       time.Time
}

// BridgeStatus represents the status of a bridge transfer
type BridgeStatus uint8

const (
	BridgeStatusPending BridgeStatus = iota
	BridgeStatusValidating
	BridgeStatusConfirmed
	BridgeStatusExecuting
	BridgeStatusCompleted
	BridgeStatusFailed
	BridgeStatusRefunded
)

// CrossChainBridge represents the base bridge structure
type CrossChainBridge struct {
	SupportedAssets       map[string]*BridgeAsset
	PendingTransfers      map[string]*BridgeTransfer
	CompletedTransfers    map[string]*BridgeTransfer
	FailedTransfers       map[string]*BridgeTransfer
	LiquidityPools        map[string]*BridgeLiquidityPool
	BridgeValidators      []*BridgeValidator
	RequiredConfirmations int
	ChallengePeriod       time.Duration
}

// BridgeValidator represents a bridge validator
type BridgeValidator struct {
	Address   string
	PublicKey *ecdsa.PublicKey
	Stake     *big.Int
	Active    bool
	Slashed   bool
	JoinedAt  time.Time
}

// BridgeSignature represents a validator's signature
type BridgeSignature struct {
	Validator string
	Signature []byte
	Timestamp time.Time
}

// BridgeLiquidityPool manages liquidity for a bridged asset
type BridgeLiquidityPool struct {
	Asset           string
	ChainID         string
	TotalLiquidity  *big.Int
	AvailableLiquidity *big.Int
	LockedLiquidity *big.Int
	Providers       map[string]*LiquidityProvider
	APY             float64
	Fees            *big.Int
}

// LiquidityProvider represents someone providing liquidity
type LiquidityProvider struct {
	Address     string
	Amount      *big.Int
	ShareTokens *big.Int
	JoinedAt    time.Time
	Rewards     *big.Int
}

// PoolRebalancer manages liquidity across chains
type PoolRebalancer struct {
	Pools              map[string]*BridgeLiquidityPool
	TargetRatios       map[string]float64 // chainID -> target ratio
	RebalanceThreshold float64            // Trigger rebalance if off by this %
	LastRebalance      time.Time
}

// Enhanced CrossChainBridge implementation
type EnhancedBridge struct {
	*CrossChainBridge
	
	// Multi-chain support
	Chains           map[string]*ChainConfig
	ActiveTransfers  map[string]*BridgeTransfer
	TransferHistory  []*BridgeTransfer
	
	// Security
	MultisigWallet   *MultisigBridge
	FraudProofs      map[string]*FraudProof
	EmergencyPaused  bool
	
	// Performance
	BatchProcessor   *BatchBridgeProcessor
	RelayerPool      []*Relayer
	
	// Monitoring
	Metrics          *BridgeMetrics
	AlertManager     *BridgeAlertManager
	
	mu sync.RWMutex
}

// ChainConfig represents configuration for a supported chain
type ChainConfig struct {
	ChainID         string
	Name            string
	Type            ChainType // EVM, Cosmos, Solana, etc
	RPCEndpoint     string
	WSEndpoint      string
	ContractAddress string // Bridge contract on this chain
	Confirmations   int
	BlockTime       time.Duration
	GasPrice        *big.Int
	Active          bool
}

// ChainType represents the type of blockchain
type ChainType uint8

const (
	ChainTypeEVM ChainType = iota
	ChainTypeCosmos
	ChainTypeSolana
	ChainTypeLux
	ChainTypeAvax
	ChainTypeBTC
)

// MultisigBridge handles multi-signature requirements
type MultisigBridge struct {
	RequiredSigs    int
	TotalSigners    int
	Signers         map[string]*BridgeSigner
	PendingTxs      map[string]*MultisigTx
	ExecutedTxs     map[string]*MultisigTx
	TimeoutDuration time.Duration
}

// BridgeSigner represents a multisig signer
type BridgeSigner struct {
	Address   string
	PublicKey *ecdsa.PublicKey
	Weight    int // For weighted multisig
	Active    bool
}

// MultisigTx represents a transaction requiring multiple signatures
type MultisigTx struct {
	ID          string
	Transfer    *BridgeTransfer
	Signatures  map[string][]byte
	CreatedAt   time.Time
	ExecutedAt  time.Time
	Status      string
}

// FraudProof represents a fraud proof for a bridge transfer
type FraudProof struct {
	TransferID      string
	ProofType       string
	Evidence        []byte
	Submitter       string
	SubmittedAt     time.Time
	Validated       bool
	SlashingAmount  *big.Int
}

// BatchBridgeProcessor handles batch processing of transfers
type BatchBridgeProcessor struct {
	BatchSize       int
	BatchInterval   time.Duration
	PendingBatch    []*BridgeTransfer
	ProcessingBatch []*BridgeTransfer
	LastBatch       time.Time
}

// Relayer represents a bridge relayer
type Relayer struct {
	Address         string
	Stake           *big.Int
	Performance     float64 // Success rate
	LastActive      time.Time
	AssignedChains  []string
}

// BridgeMetrics tracks bridge performance
type BridgeMetrics struct {
	TotalTransfers     uint64
	TotalVolume        *big.Int
	AverageTime        time.Duration
	SuccessRate        float64
	ActiveValidators   int
	TotalLiquidity     *big.Int
	DailyVolume        map[string]*big.Int // date -> volume
}

// BridgeAlertManager handles alerts
type BridgeAlertManager struct {
	Alerts          []*BridgeAlert
	AlertThresholds map[string]interface{}
	Subscribers     []string
}

// BridgeAlert represents an alert
type BridgeAlert struct {
	ID          string
	Type        string
	Severity    string
	Message     string
	Details     map[string]interface{}
	Timestamp   time.Time
	Resolved    bool
}

// NewEnhancedBridge creates a new enhanced bridge
func NewEnhancedBridge() *EnhancedBridge {
	return &EnhancedBridge{
		CrossChainBridge: &CrossChainBridge{
			SupportedAssets:       make(map[string]*BridgeAsset),
			PendingTransfers:      make(map[string]*BridgeTransfer),
			CompletedTransfers:    make(map[string]*BridgeTransfer),
			FailedTransfers:       make(map[string]*BridgeTransfer),
			LiquidityPools:        make(map[string]*BridgeLiquidityPool),
			RequiredConfirmations: 15,
			ChallengePeriod:       24 * time.Hour,
		},
		Chains:          make(map[string]*ChainConfig),
		ActiveTransfers: make(map[string]*BridgeTransfer),
		FraudProofs:     make(map[string]*FraudProof),
		Metrics: &BridgeMetrics{
			TotalVolume:  big.NewInt(0),
			DailyVolume:  make(map[string]*big.Int),
		},
	}
}

// InitiateTransfer starts a cross-chain transfer
func (b *EnhancedBridge) InitiateTransfer(
	ctx context.Context,
	asset string,
	amount *big.Int,
	sourceChain string,
	destChain string,
	sourceAddr string,
	destAddr string,
) (*BridgeTransfer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Validation
	if b.EmergencyPaused {
		return nil, errors.New("bridge is paused")
	}

	bridgeAsset, exists := b.SupportedAssets[asset]
	if !exists {
		return nil, fmt.Errorf("asset %s not supported", asset)
	}

	if bridgeAsset.Paused {
		return nil, fmt.Errorf("asset %s is paused", asset)
	}

	// Check limits
	if amount.Cmp(bridgeAsset.MinTransfer) < 0 {
		return nil, fmt.Errorf("amount below minimum: %s", bridgeAsset.MinTransfer)
	}

	if amount.Cmp(bridgeAsset.MaxTransfer) > 0 {
		return nil, fmt.Errorf("amount exceeds maximum: %s", bridgeAsset.MaxTransfer)
	}

	// Check daily limit
	if err := b.checkDailyLimit(bridgeAsset, amount); err != nil {
		return nil, err
	}

	// Check liquidity on destination chain
	destPool := b.LiquidityPools[destChain+":"+asset]
	if destPool == nil || destPool.AvailableLiquidity.Cmp(amount) < 0 {
		return nil, errors.New("insufficient liquidity on destination chain")
	}

	// Calculate fee (0.3% of transfer amount)
	fee := new(big.Int).Mul(amount, big.NewInt(3))
	fee.Div(fee, big.NewInt(1000))

	// Create transfer
	transfer := &BridgeTransfer{
		ID:               b.generateTransferID(),
		Asset:            asset,
		Amount:           amount,
		Fee:              fee,
		SourceChain:      sourceChain,
		DestChain:        destChain,
		SourceAddress:    sourceAddr,
		DestAddress:      destAddr,
		Status:           BridgeStatusPending,
		RequiredConfirms: b.RequiredConfirmations,
		InitiatedAt:      time.Now(),
		Validators:       make(map[string]*BridgeSignature),
		Nonce:            b.getNextNonce(),
		ExpiryTime:       time.Now().Add(24 * time.Hour),
	}

	// Lock liquidity
	destPool.AvailableLiquidity.Sub(destPool.AvailableLiquidity, amount)
	destPool.LockedLiquidity.Add(destPool.LockedLiquidity, amount)

	// Store transfer
	b.PendingTransfers[transfer.ID] = transfer
	b.ActiveTransfers[transfer.ID] = transfer

	// Update metrics
	b.updateMetrics(transfer)

	// Notify validators
	go b.notifyValidators(transfer)

	return transfer, nil
}

// ValidateTransfer validates a transfer (called by validators)
func (b *EnhancedBridge) ValidateTransfer(
	transferID string,
	validator string,
	signature []byte,
) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	transfer, exists := b.PendingTransfers[transferID]
	if !exists {
		return errors.New("transfer not found")
	}

	// Verify validator
	validatorInfo := b.findValidator(validator)
	if validatorInfo == nil || !validatorInfo.Active {
		return errors.New("invalid or inactive validator")
	}

	// Verify signature
	if !b.verifySignature(transfer, validatorInfo, signature) {
		return errors.New("invalid signature")
	}

	// Add signature
	transfer.Validators[validator] = &BridgeSignature{
		Validator: validator,
		Signature: signature,
		Timestamp: time.Now(),
	}

	// Check if we have enough signatures
	if len(transfer.Validators) >= b.RequiredConfirmations {
		transfer.Status = BridgeStatusConfirmed
		go b.executeTransfer(transfer)
	} else {
		transfer.Status = BridgeStatusValidating
	}

	return nil
}

// executeTransfer executes a confirmed transfer
func (b *EnhancedBridge) executeTransfer(transfer *BridgeTransfer) error {
	b.mu.Lock()
	transfer.Status = BridgeStatusExecuting
	b.mu.Unlock()

	// Execute on destination chain
	destChain := b.Chains[transfer.DestChain]
	if destChain == nil {
		return b.failTransfer(transfer, "destination chain not configured")
	}

	// Call destination chain contract to mint/release tokens
	txHash, err := b.executeOnChain(destChain, transfer)
	if err != nil {
		return b.failTransfer(transfer, err.Error())
	}

	// Wait for confirmation
	confirmed, err := b.waitForConfirmation(destChain, txHash)
	if err != nil || !confirmed {
		return b.failTransfer(transfer, "confirmation failed")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// Complete transfer
	transfer.Status = BridgeStatusCompleted
	transfer.CompletedAt = time.Now()
	transfer.DestTxHash = txHash

	// Move from pending to completed
	delete(b.PendingTransfers, transfer.ID)
	b.CompletedTransfers[transfer.ID] = transfer

	// Release locked liquidity
	destPool := b.LiquidityPools[transfer.DestChain+":"+transfer.Asset]
	if destPool != nil {
		destPool.LockedLiquidity.Sub(destPool.LockedLiquidity, transfer.Amount)
		
		// Distribute fees to liquidity providers
		b.distributeFees(destPool, transfer.Fee)
	}

	// Update metrics
	b.Metrics.TotalTransfers++
	b.Metrics.TotalVolume.Add(b.Metrics.TotalVolume, transfer.Amount)

	return nil
}

// failTransfer marks a transfer as failed
func (b *EnhancedBridge) failTransfer(transfer *BridgeTransfer, reason string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	transfer.Status = BridgeStatusFailed
	transfer.CompletedAt = time.Now()

	// Move to failed transfers
	delete(b.PendingTransfers, transfer.ID)
	b.FailedTransfers[transfer.ID] = transfer

	// Release locked liquidity
	destPool := b.LiquidityPools[transfer.DestChain+":"+transfer.Asset]
	if destPool != nil {
		destPool.LockedLiquidity.Sub(destPool.LockedLiquidity, transfer.Amount)
		destPool.AvailableLiquidity.Add(destPool.AvailableLiquidity, transfer.Amount)
	}

	// Create alert
	if b.AlertManager != nil {
		alert := &BridgeAlert{
			ID:        fmt.Sprintf("alert_%d", time.Now().Unix()),
			Type:      "TransferFailed",
			Severity:  "High",
			Message:   fmt.Sprintf("Transfer %s failed: %s", transfer.ID, reason),
			Timestamp: time.Now(),
		}
		b.AlertManager.Alerts = append(b.AlertManager.Alerts, alert)
	}

	return nil
}

// AddLiquidity adds liquidity to a pool
func (b *EnhancedBridge) AddLiquidity(
	chainID string,
	asset string,
	provider string,
	amount *big.Int,
) (*LiquidityProvider, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	poolKey := chainID + ":" + asset
	pool := b.LiquidityPools[poolKey]
	if pool == nil {
		// Create new pool
		pool = &BridgeLiquidityPool{
			Asset:              asset,
			ChainID:            chainID,
			TotalLiquidity:     big.NewInt(0),
			AvailableLiquidity: big.NewInt(0),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}
		b.LiquidityPools[poolKey] = pool
	}

	// Calculate share tokens (1:1 for first provider, proportional after)
	var shareTokens *big.Int
	if pool.TotalLiquidity.Sign() == 0 {
		shareTokens = new(big.Int).Set(amount)
	} else {
		// shares = (amount * totalShares) / totalLiquidity
		totalShares := big.NewInt(0)
		for _, p := range pool.Providers {
			totalShares.Add(totalShares, p.ShareTokens)
		}
		shareTokens = new(big.Int).Mul(amount, totalShares)
		shareTokens.Div(shareTokens, pool.TotalLiquidity)
	}

	// Add or update provider
	lp := pool.Providers[provider]
	if lp == nil {
		lp = &LiquidityProvider{
			Address:     provider,
			Amount:      big.NewInt(0),
			ShareTokens: big.NewInt(0),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}
		pool.Providers[provider] = lp
	}

	lp.Amount.Add(lp.Amount, amount)
	lp.ShareTokens.Add(lp.ShareTokens, shareTokens)

	// Update pool totals
	pool.TotalLiquidity.Add(pool.TotalLiquidity, amount)
	pool.AvailableLiquidity.Add(pool.AvailableLiquidity, amount)

	// Update metrics
	b.Metrics.TotalLiquidity.Add(b.Metrics.TotalLiquidity, amount)

	return lp, nil
}

// RemoveLiquidity removes liquidity from a pool
func (b *EnhancedBridge) RemoveLiquidity(
	chainID string,
	asset string,
	provider string,
	shareTokens *big.Int,
) (*big.Int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	poolKey := chainID + ":" + asset
	pool := b.LiquidityPools[poolKey]
	if pool == nil {
		return nil, errors.New("pool not found")
	}

	lp := pool.Providers[provider]
	if lp == nil {
		return nil, errors.New("provider not found")
	}

	if shareTokens.Cmp(lp.ShareTokens) > 0 {
		return nil, errors.New("insufficient shares")
	}

	// Calculate amount to return (including rewards)
	// amount = (shareTokens * totalLiquidity) / totalShares
	totalShares := big.NewInt(0)
	for _, p := range pool.Providers {
		totalShares.Add(totalShares, p.ShareTokens)
	}

	withdrawAmount := new(big.Int).Mul(shareTokens, pool.TotalLiquidity)
	withdrawAmount.Div(withdrawAmount, totalShares)

	// Check available liquidity
	if withdrawAmount.Cmp(pool.AvailableLiquidity) > 0 {
		return nil, errors.New("insufficient available liquidity")
	}

	// Add accumulated rewards
	rewardShare := new(big.Int).Mul(lp.Rewards, shareTokens)
	rewardShare.Div(rewardShare, lp.ShareTokens)
	withdrawAmount.Add(withdrawAmount, rewardShare)

	// Update provider
	lp.ShareTokens.Sub(lp.ShareTokens, shareTokens)
	lp.Amount.Sub(lp.Amount, withdrawAmount)
	lp.Rewards.Sub(lp.Rewards, rewardShare)

	// Remove provider if no shares left
	if lp.ShareTokens.Sign() == 0 {
		delete(pool.Providers, provider)
	}

	// Update pool
	pool.TotalLiquidity.Sub(pool.TotalLiquidity, withdrawAmount)
	pool.AvailableLiquidity.Sub(pool.AvailableLiquidity, withdrawAmount)

	return withdrawAmount, nil
}

// SubmitFraudProof submits a fraud proof
func (b *EnhancedBridge) SubmitFraudProof(
	transferID string,
	proofType string,
	evidence []byte,
	submitter string,
) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Verify transfer exists
	transfer := b.findTransfer(transferID)
	if transfer == nil {
		return errors.New("transfer not found")
	}

	// Create fraud proof
	proof := &FraudProof{
		TransferID:  transferID,
		ProofType:   proofType,
		Evidence:    evidence,
		Submitter:   submitter,
		SubmittedAt: time.Now(),
	}

	// Validate proof
	valid, slashAmount := b.validateFraudProof(proof, transfer)
	if !valid {
		return errors.New("invalid fraud proof")
	}

	proof.Validated = true
	proof.SlashingAmount = slashAmount

	// Store proof
	b.FraudProofs[transferID] = proof

	// Pause transfer if still pending
	if transfer.Status < BridgeStatusCompleted {
		transfer.Status = BridgeStatusFailed
		delete(b.PendingTransfers, transferID)
		b.FailedTransfers[transferID] = transfer
	}

	// Slash malicious validators
	b.slashValidators(transfer, slashAmount)

	// Reward fraud proof submitter (10% of slashed amount)
	_ = new(big.Int).Div(slashAmount, big.NewInt(10))
	// TODO: Transfer reward to submitter

	return nil
}

// Helper methods

func (b *EnhancedBridge) checkDailyLimit(asset *BridgeAsset, amount *big.Int) error {
	// Reset daily volume if needed
	if time.Since(asset.LastReset) > 24*time.Hour {
		asset.DailyVolume = big.NewInt(0)
		asset.LastReset = time.Now()
	}

	// Check if adding this amount would exceed daily limit
	newVolume := new(big.Int).Add(asset.DailyVolume, amount)
	if newVolume.Cmp(asset.DailyLimit) > 0 {
		return fmt.Errorf("exceeds daily limit: %s", asset.DailyLimit)
	}

	asset.DailyVolume = newVolume
	return nil
}

func (b *EnhancedBridge) generateTransferID() string {
	// Generate unique transfer ID
	data := fmt.Sprintf("%d_%d", time.Now().UnixNano(), b.getNextNonce())
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])[:16]
}

func (b *EnhancedBridge) getNextNonce() uint64 {
	// In production, this would be atomic
	return uint64(time.Now().UnixNano())
}

func (b *EnhancedBridge) findValidator(address string) *BridgeValidator {
	for _, v := range b.BridgeValidators {
		if v.Address == address {
			return v
		}
	}
	return nil
}

func (b *EnhancedBridge) verifySignature(
	transfer *BridgeTransfer,
	validator *BridgeValidator,
	signature []byte,
) bool {
	// Verify validator signature on transfer data
	// In production, use proper cryptographic verification
	return len(signature) > 0
}

func (b *EnhancedBridge) notifyValidators(transfer *BridgeTransfer) {
	// Notify all active validators about new transfer
	for _, v := range b.BridgeValidators {
		if v.Active && !v.Slashed {
			// Send notification
		}
	}
}

func (b *EnhancedBridge) executeOnChain(
	chain *ChainConfig,
	transfer *BridgeTransfer,
) (string, error) {
	// Execute transfer on destination chain
	// This would call the actual blockchain
	return fmt.Sprintf("0x%x", time.Now().Unix()), nil
}

func (b *EnhancedBridge) waitForConfirmation(
	chain *ChainConfig,
	txHash string,
) (bool, error) {
	// Wait for required confirmations
	// In production, monitor actual blockchain
	time.Sleep(chain.BlockTime * time.Duration(chain.Confirmations))
	return true, nil
}

func (b *EnhancedBridge) distributeFees(pool *BridgeLiquidityPool, fee *big.Int) {
	// Distribute fees proportionally to liquidity providers
	totalShares := big.NewInt(0)
	for _, p := range pool.Providers {
		totalShares.Add(totalShares, p.ShareTokens)
	}

	if totalShares.Sign() == 0 {
		return
	}

	for _, provider := range pool.Providers {
		// providerFee = (fee * providerShares) / totalShares
		providerFee := new(big.Int).Mul(fee, provider.ShareTokens)
		providerFee.Div(providerFee, totalShares)
		provider.Rewards.Add(provider.Rewards, providerFee)
	}

	pool.Fees.Add(pool.Fees, fee)
}

func (b *EnhancedBridge) findTransfer(id string) *BridgeTransfer {
	if t, exists := b.PendingTransfers[id]; exists {
		return t
	}
	if t, exists := b.CompletedTransfers[id]; exists {
		return t
	}
	if t, exists := b.FailedTransfers[id]; exists {
		return t
	}
	return nil
}

func (b *EnhancedBridge) validateFraudProof(
	proof *FraudProof,
	transfer *BridgeTransfer,
) (bool, *big.Int) {
	// Validate fraud proof
	// In production, implement actual validation logic
	return true, big.NewInt(1000000) // 1M USDC slash
}

func (b *EnhancedBridge) slashValidators(transfer *BridgeTransfer, amount *big.Int) {
	// Slash validators who signed invalid transfer
	numValidators := len(transfer.Validators)
	if numValidators == 0 {
		return
	}

	slashPerValidator := new(big.Int).Div(amount, big.NewInt(int64(numValidators)))

	for validatorAddr := range transfer.Validators {
		validator := b.findValidator(validatorAddr)
		if validator != nil {
			validator.Slashed = true
			validator.Active = false
			// Deduct from stake
			validator.Stake.Sub(validator.Stake, slashPerValidator)
		}
	}
}

func (b *EnhancedBridge) updateMetrics(transfer *BridgeTransfer) {
	// Update daily volume
	today := time.Now().Format("2006-01-02")
	if b.Metrics.DailyVolume[today] == nil {
		b.Metrics.DailyVolume[today] = big.NewInt(0)
	}
	b.Metrics.DailyVolume[today].Add(b.Metrics.DailyVolume[today], transfer.Amount)
}

// GetBridgeStatus returns current bridge status
func (b *EnhancedBridge) GetBridgeStatus() map[string]interface{} {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return map[string]interface{}{
		"active":            !b.EmergencyPaused,
		"pending_transfers": len(b.PendingTransfers),
		"total_transfers":   b.Metrics.TotalTransfers,
		"total_volume":      b.Metrics.TotalVolume.String(),
		"total_liquidity":   b.Metrics.TotalLiquidity.String(),
		"active_validators": b.Metrics.ActiveValidators,
		"chains":            len(b.Chains),
		"assets":            len(b.SupportedAssets),
	}
}