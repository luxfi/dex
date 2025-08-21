package lx

import (
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"
)

// MultisigWallet represents a multi-signature wallet
type MultisigWallet struct {
	// Wallet identification
	ID              string
	Name            string
	Description     string
	
	// Signers and thresholds
	Owners          map[string]*WalletOwner  // address -> owner
	RequiredSigs    int                      // Required signatures for execution
	TotalOwners     int                      // Total number of owners
	
	// Advanced multisig features
	ThresholdLevels map[TxType]int           // Different thresholds for different operations
	TimeLockedOps   map[string]*TimeLock     // Time-locked operations
	SpendingLimits  map[string]*SpendingLimit // Daily/weekly spending limits
	
	// Transactions
	PendingTxs      map[string]*MultisigTransaction
	ExecutedTxs     map[string]*MultisigTransaction
	RejectedTxs     map[string]*MultisigTransaction
	
	// Wallet state
	Balance         map[string]*big.Int      // token -> balance
	Nonce           uint64                   // Transaction nonce
	CreatedAt       time.Time
	LastActivity    time.Time
	
	// Security features
	Frozen          bool                     // Emergency freeze
	WhitelistedAddrs map[string]bool         // Whitelisted recipient addresses
	BlacklistedAddrs map[string]bool         // Blacklisted addresses
	RequireWhitelist bool                    // Only send to whitelisted addresses
	
	// Recovery mechanism
	RecoveryOwners  map[string]*WalletOwner  // Recovery signers
	RecoveryThreshold int                     // Signatures needed for recovery
	RecoveryDelay   time.Duration            // Delay before recovery takes effect
	
	mu sync.RWMutex
}

// WalletOwner represents an owner of the multisig wallet
type WalletOwner struct {
	Address         string
	Name            string
	PublicKey       *ecdsa.PublicKey
	Weight          int       // For weighted multisig
	AddedAt         time.Time
	LastActive      time.Time
	DailyLimit      *big.Int  // Individual daily spending limit
	SpentToday      *big.Int
	LastResetDate   time.Time
	CanAddOwner     bool      // Permission to add owners
	CanRemoveOwner  bool      // Permission to remove owners
	CanChangeThreshold bool   // Permission to change threshold
}

// MultisigTransaction represents a transaction requiring multiple signatures
type MultisigTransaction struct {
	ID              string
	WalletID        string
	Type            TxType
	Nonce           uint64
	
	// Transaction details
	To              string
	Value           *big.Int
	Token           string    // Token to transfer (empty for native)
	Data            []byte    // Additional data/calldata
	
	// Signatures
	Signatures      map[string]*TxSignature // owner -> signature
	Confirmations   int                     // Current confirmations
	RequiredSigs    int                     // Required signatures
	
	// Metadata
	Proposer        string
	Description     string
	CreatedAt       time.Time
	ExecutedAt      time.Time
	ExpiresAt       time.Time
	
	// Status
	Status          TxStatus
	ExecutionResult string
	GasUsed         uint64
	TxHash          string
}

// TxType represents the type of transaction
type TxType uint8

const (
	TxTypeTransfer TxType = iota
	TxTypeAddOwner
	TxTypeRemoveOwner
	TxTypeChangeThreshold
	TxTypeFreeze
	TxTypeUnfreeze
	TxTypeWhitelist
	TxTypeBlacklist
	TxTypeSetLimit
	TxTypeContractCall
	TxTypeRecovery
	TxTypeCancelTx
)

// TxStatus represents transaction status
type TxStatus uint8

const (
	TxStatusPending TxStatus = iota
	TxStatusApproved
	TxStatusExecuting
	TxStatusExecuted
	TxStatusRejected
	TxStatusExpired
	TxStatusCancelled
)

// TxSignature represents a signature on a transaction
type TxSignature struct {
	Signer          string
	Signature       []byte
	SignedAt        time.Time
	V               uint8
	R               []byte
	S               []byte
}

// TimeLock represents a time-locked operation
type TimeLock struct {
	OperationType   TxType
	UnlockTime      time.Time
	MinDelay        time.Duration
	MaxDelay        time.Duration
}

// SpendingLimit represents spending limits
type SpendingLimit struct {
	Period          LimitPeriod
	Amount          *big.Int
	Spent           *big.Int
	LastReset       time.Time
	Token           string    // Token this limit applies to
}

// LimitPeriod represents the period for spending limits
type LimitPeriod uint8

const (
	LimitPeriodDaily LimitPeriod = iota
	LimitPeriodWeekly
	LimitPeriodMonthly
)

// MultisigManager manages all multisig wallets
type MultisigManager struct {
	Wallets         map[string]*MultisigWallet
	UserWallets     map[string][]*MultisigWallet // user -> wallets they're part of
	
	// Global settings
	MaxOwnersPerWallet int
	MinThreshold       int
	MaxPendingTxs      int
	TxExpiryDuration   time.Duration
	
	// Security
	GlobalFreeze       bool
	EmergencyContacts  []string
	
	// Analytics
	TotalWallets       int
	TotalTransactions  uint64
	TotalVolume        map[string]*big.Int // token -> volume
	
	mu sync.RWMutex
}

// NewMultisigManager creates a new multisig manager
func NewMultisigManager() *MultisigManager {
	return &MultisigManager{
		Wallets:            make(map[string]*MultisigWallet),
		UserWallets:        make(map[string][]*MultisigWallet),
		MaxOwnersPerWallet: 20,
		MinThreshold:       1,
		MaxPendingTxs:      100,
		TxExpiryDuration:   7 * 24 * time.Hour, // 7 days
		TotalVolume:        make(map[string]*big.Int),
	}
}

// CreateWallet creates a new multisig wallet
func (mm *MultisigManager) CreateWallet(
	name string,
	owners []string,
	threshold int,
) (*MultisigWallet, error) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	// Validation
	if len(owners) == 0 {
		return nil, errors.New("at least one owner required")
	}

	if len(owners) > mm.MaxOwnersPerWallet {
		return nil, fmt.Errorf("maximum %d owners allowed", mm.MaxOwnersPerWallet)
	}

	if threshold < mm.MinThreshold || threshold > len(owners) {
		return nil, fmt.Errorf("threshold must be between %d and %d", mm.MinThreshold, len(owners))
	}

	// Check for duplicate owners
	ownerMap := make(map[string]bool)
	for _, owner := range owners {
		if ownerMap[owner] {
			return nil, errors.New("duplicate owner address")
		}
		ownerMap[owner] = true
	}

	walletID := mm.generateWalletID()

	// Create wallet
	wallet := &MultisigWallet{
		ID:               walletID,
		Name:             name,
		Owners:           make(map[string]*WalletOwner),
		RequiredSigs:     threshold,
		TotalOwners:      len(owners),
		ThresholdLevels:  make(map[TxType]int),
		TimeLockedOps:    make(map[string]*TimeLock),
		SpendingLimits:   make(map[string]*SpendingLimit),
		PendingTxs:       make(map[string]*MultisigTransaction),
		ExecutedTxs:      make(map[string]*MultisigTransaction),
		RejectedTxs:      make(map[string]*MultisigTransaction),
		Balance:          make(map[string]*big.Int),
		WhitelistedAddrs: make(map[string]bool),
		BlacklistedAddrs: make(map[string]bool),
		RecoveryOwners:   make(map[string]*WalletOwner),
		CreatedAt:        time.Now(),
		LastActivity:     time.Now(),
	}

	// Set default threshold levels for different operations
	wallet.ThresholdLevels[TxTypeTransfer] = threshold
	wallet.ThresholdLevels[TxTypeAddOwner] = min(threshold+1, len(owners))
	wallet.ThresholdLevels[TxTypeRemoveOwner] = min(threshold+1, len(owners))
	wallet.ThresholdLevels[TxTypeChangeThreshold] = min(threshold+1, len(owners))
	wallet.ThresholdLevels[TxTypeFreeze] = max(2, threshold/2)
	wallet.ThresholdLevels[TxTypeRecovery] = len(owners) // All owners for recovery

	// Add owners
	for _, ownerAddr := range owners {
		owner := &WalletOwner{
			Address:            ownerAddr,
			Weight:             1, // Equal weight by default
			AddedAt:            time.Now(),
			DailyLimit:         big.NewInt(0), // No individual limit by default
			SpentToday:         big.NewInt(0),
			LastResetDate:      time.Now(),
			CanAddOwner:        true,
			CanRemoveOwner:     true,
			CanChangeThreshold: true,
		}
		wallet.Owners[ownerAddr] = owner
		
		// Track user wallets
		mm.UserWallets[ownerAddr] = append(mm.UserWallets[ownerAddr], wallet)
	}

	mm.Wallets[walletID] = wallet
	mm.TotalWallets++

	return wallet, nil
}

// ProposeTransaction proposes a new transaction
func (mm *MultisigManager) ProposeTransaction(
	walletID string,
	proposer string,
	txType TxType,
	to string,
	value *big.Int,
	token string,
	data []byte,
	description string,
) (*MultisigTransaction, error) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	wallet, exists := mm.Wallets[walletID]
	if !exists {
		return nil, errors.New("wallet not found")
	}

	wallet.mu.Lock()
	defer wallet.mu.Unlock()

	// Check if proposer is owner
	if _, isOwner := wallet.Owners[proposer]; !isOwner {
		return nil, errors.New("proposer is not an owner")
	}

	// Check if wallet is frozen
	if wallet.Frozen && txType != TxTypeUnfreeze {
		return nil, errors.New("wallet is frozen")
	}

	// Check pending transaction limit
	if len(wallet.PendingTxs) >= mm.MaxPendingTxs {
		return nil, errors.New("too many pending transactions")
	}

	// Validate recipient for transfers
	if txType == TxTypeTransfer {
		if wallet.RequireWhitelist && !wallet.WhitelistedAddrs[to] {
			return nil, errors.New("recipient not whitelisted")
		}
		if wallet.BlacklistedAddrs[to] {
			return nil, errors.New("recipient is blacklisted")
		}
		
		// Check spending limits
		if err := mm.checkSpendingLimits(wallet, token, value); err != nil {
			return nil, err
		}
	}

	// Get required signatures for this transaction type
	requiredSigs := wallet.ThresholdLevels[txType]
	if requiredSigs == 0 {
		requiredSigs = wallet.RequiredSigs
	}

	// Create transaction
	tx := &MultisigTransaction{
		ID:            mm.generateTxID(),
		WalletID:      walletID,
		Type:          txType,
		Nonce:         wallet.Nonce,
		To:            to,
		Value:         value,
		Token:         token,
		Data:          data,
		Signatures:    make(map[string]*TxSignature),
		RequiredSigs:  requiredSigs,
		Proposer:      proposer,
		Description:   description,
		CreatedAt:     time.Now(),
		ExpiresAt:     time.Now().Add(mm.TxExpiryDuration),
		Status:        TxStatusPending,
	}

	// Automatically sign by proposer
	signature := mm.signTransaction(tx, proposer)
	tx.Signatures[proposer] = signature
	tx.Confirmations = 1

	// Store transaction
	wallet.PendingTxs[tx.ID] = tx
	wallet.Nonce++
	wallet.LastActivity = time.Now()

	// Check if immediately executable
	if tx.Confirmations >= tx.RequiredSigs {
		go mm.executeTransaction(tx)
	}

	return tx, nil
}

// ConfirmTransaction adds a signature to a pending transaction
func (mm *MultisigManager) ConfirmTransaction(
	walletID string,
	txID string,
	signer string,
) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	wallet, exists := mm.Wallets[walletID]
	if !exists {
		return errors.New("wallet not found")
	}

	wallet.mu.Lock()
	defer wallet.mu.Unlock()

	// Check if signer is owner
	owner, isOwner := wallet.Owners[signer]
	if !isOwner {
		return errors.New("signer is not an owner")
	}

	// Get transaction
	tx, exists := wallet.PendingTxs[txID]
	if !exists {
		return errors.New("transaction not found")
	}

	// Check if already signed
	if _, signed := tx.Signatures[signer]; signed {
		return errors.New("already signed by this owner")
	}

	// Check expiry
	if time.Now().After(tx.ExpiresAt) {
		tx.Status = TxStatusExpired
		delete(wallet.PendingTxs, txID)
		wallet.RejectedTxs[txID] = tx
		return errors.New("transaction expired")
	}

	// Add signature
	signature := mm.signTransaction(tx, signer)
	tx.Signatures[signer] = signature
	tx.Confirmations++

	// Update owner activity
	owner.LastActive = time.Now()

	// Check if ready to execute
	if tx.Confirmations >= tx.RequiredSigs {
		tx.Status = TxStatusApproved
		go mm.executeTransaction(tx)
	}

	return nil
}

// RevokeConfirmation revokes a signature from a transaction
func (mm *MultisigManager) RevokeConfirmation(
	walletID string,
	txID string,
	signer string,
) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	wallet, exists := mm.Wallets[walletID]
	if !exists {
		return errors.New("wallet not found")
	}

	wallet.mu.Lock()
	defer wallet.mu.Unlock()

	tx, exists := wallet.PendingTxs[txID]
	if !exists {
		return errors.New("transaction not found")
	}

	// Check if signed by this owner
	if _, signed := tx.Signatures[signer]; !signed {
		return errors.New("not signed by this owner")
	}

	// Remove signature
	delete(tx.Signatures, signer)
	tx.Confirmations--

	// Update status if needed
	if tx.Status == TxStatusApproved && tx.Confirmations < tx.RequiredSigs {
		tx.Status = TxStatusPending
	}

	return nil
}

// executeTransaction executes an approved transaction
func (mm *MultisigManager) executeTransaction(tx *MultisigTransaction) error {
	mm.mu.Lock()
	wallet := mm.Wallets[tx.WalletID]
	mm.mu.Unlock()

	if wallet == nil {
		return errors.New("wallet not found")
	}

	wallet.mu.Lock()
	defer wallet.mu.Unlock()

	// Double-check confirmations
	if tx.Confirmations < tx.RequiredSigs {
		return errors.New("insufficient confirmations")
	}

	tx.Status = TxStatusExecuting

	// Execute based on transaction type
	var err error
	switch tx.Type {
	case TxTypeTransfer:
		err = mm.executeTransfer(wallet, tx)
		
	case TxTypeAddOwner:
		err = mm.executeAddOwner(wallet, tx)
		
	case TxTypeRemoveOwner:
		err = mm.executeRemoveOwner(wallet, tx)
		
	case TxTypeChangeThreshold:
		err = mm.executeChangeThreshold(wallet, tx)
		
	case TxTypeFreeze:
		wallet.Frozen = true
		
	case TxTypeUnfreeze:
		wallet.Frozen = false
		
	case TxTypeWhitelist:
		wallet.WhitelistedAddrs[tx.To] = true
		
	case TxTypeBlacklist:
		wallet.BlacklistedAddrs[tx.To] = true
		
	default:
		err = errors.New("unsupported transaction type")
	}

	if err != nil {
		tx.Status = TxStatusRejected
		tx.ExecutionResult = err.Error()
		wallet.RejectedTxs[tx.ID] = tx
	} else {
		tx.Status = TxStatusExecuted
		tx.ExecutedAt = time.Now()
		tx.ExecutionResult = "Success"
		wallet.ExecutedTxs[tx.ID] = tx
		
		// Update metrics
		mm.TotalTransactions++
		if tx.Type == TxTypeTransfer && tx.Value != nil {
			if mm.TotalVolume[tx.Token] == nil {
				mm.TotalVolume[tx.Token] = big.NewInt(0)
			}
			mm.TotalVolume[tx.Token].Add(mm.TotalVolume[tx.Token], tx.Value)
		}
	}

	// Remove from pending
	delete(wallet.PendingTxs, tx.ID)
	wallet.LastActivity = time.Now()

	return err
}

// executeTransfer executes a transfer transaction
func (mm *MultisigManager) executeTransfer(wallet *MultisigWallet, tx *MultisigTransaction) error {
	// Check balance
	balance := wallet.Balance[tx.Token]
	if balance == nil || balance.Cmp(tx.Value) < 0 {
		return errors.New("insufficient balance")
	}

	// Update spending for daily limits
	for _, owner := range wallet.Owners {
		if owner.DailyLimit != nil && owner.DailyLimit.Sign() > 0 {
			// Reset daily spent if needed
			if time.Since(owner.LastResetDate) > 24*time.Hour {
				owner.SpentToday = big.NewInt(0)
				owner.LastResetDate = time.Now()
			}
			
			// Check individual limit
			newSpent := new(big.Int).Add(owner.SpentToday, tx.Value)
			if newSpent.Cmp(owner.DailyLimit) > 0 {
				// This owner has exceeded their limit
				// In a real implementation, we might handle this differently
			}
			
			owner.SpentToday = newSpent
		}
	}

	// Deduct balance
	wallet.Balance[tx.Token].Sub(wallet.Balance[tx.Token], tx.Value)

	// In a real implementation, this would interact with the blockchain
	tx.TxHash = fmt.Sprintf("0x%x", time.Now().Unix())

	return nil
}

// executeAddOwner adds a new owner to the wallet
func (mm *MultisigManager) executeAddOwner(wallet *MultisigWallet, tx *MultisigTransaction) error {
	if wallet.TotalOwners >= mm.MaxOwnersPerWallet {
		return errors.New("maximum owners reached")
	}

	if _, exists := wallet.Owners[tx.To]; exists {
		return errors.New("already an owner")
	}

	owner := &WalletOwner{
		Address:       tx.To,
		Weight:        1,
		AddedAt:       time.Now(),
		DailyLimit:    big.NewInt(0),
		SpentToday:    big.NewInt(0),
		LastResetDate: time.Now(),
	}

	wallet.Owners[tx.To] = owner
	wallet.TotalOwners++

	// Update user wallets
	mm.UserWallets[tx.To] = append(mm.UserWallets[tx.To], wallet)

	return nil
}

// executeRemoveOwner removes an owner from the wallet
func (mm *MultisigManager) executeRemoveOwner(wallet *MultisigWallet, tx *MultisigTransaction) error {
	if wallet.TotalOwners <= wallet.RequiredSigs {
		return errors.New("cannot remove owner: would break threshold")
	}

	if _, exists := wallet.Owners[tx.To]; !exists {
		return errors.New("not an owner")
	}

	delete(wallet.Owners, tx.To)
	wallet.TotalOwners--

	// Update user wallets
	userWallets := mm.UserWallets[tx.To]
	for i, w := range userWallets {
		if w.ID == wallet.ID {
			mm.UserWallets[tx.To] = append(userWallets[:i], userWallets[i+1:]...)
			break
		}
	}

	return nil
}

// executeChangeThreshold changes the signature threshold
func (mm *MultisigManager) executeChangeThreshold(wallet *MultisigWallet, tx *MultisigTransaction) error {
	newThreshold := int(tx.Value.Int64())
	
	if newThreshold < mm.MinThreshold || newThreshold > wallet.TotalOwners {
		return fmt.Errorf("invalid threshold: must be between %d and %d", 
			mm.MinThreshold, wallet.TotalOwners)
	}

	wallet.RequiredSigs = newThreshold
	
	// Update threshold levels proportionally
	for txType := range wallet.ThresholdLevels {
		if txType == TxTypeTransfer {
			wallet.ThresholdLevels[txType] = newThreshold
		}
	}

	return nil
}

// SetSpendingLimit sets a spending limit for the wallet
func (mm *MultisigManager) SetSpendingLimit(
	walletID string,
	token string,
	period LimitPeriod,
	amount *big.Int,
) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	wallet, exists := mm.Wallets[walletID]
	if !exists {
		return errors.New("wallet not found")
	}

	wallet.mu.Lock()
	defer wallet.mu.Unlock()

	limitKey := fmt.Sprintf("%s_%d", token, period)
	wallet.SpendingLimits[limitKey] = &SpendingLimit{
		Period:    period,
		Amount:    amount,
		Spent:     big.NewInt(0),
		LastReset: time.Now(),
		Token:     token,
	}

	return nil
}

// Helper methods

func (mm *MultisigManager) generateWalletID() string {
	data := fmt.Sprintf("wallet_%d_%d", time.Now().UnixNano(), mm.TotalWallets)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])[:16]
}

func (mm *MultisigManager) generateTxID() string {
	data := fmt.Sprintf("tx_%d_%d", time.Now().UnixNano(), mm.TotalTransactions)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])[:16]
}

func (mm *MultisigManager) signTransaction(tx *MultisigTransaction, signer string) *TxSignature {
	// In production, this would use proper cryptographic signing
	data := fmt.Sprintf("%s_%s_%s_%s", tx.ID, tx.To, tx.Value, signer)
	hash := sha256.Sum256([]byte(data))
	
	return &TxSignature{
		Signer:    signer,
		Signature: hash[:],
		SignedAt:  time.Now(),
	}
}

func (mm *MultisigManager) checkSpendingLimits(
	wallet *MultisigWallet,
	token string,
	amount *big.Int,
) error {
	now := time.Now()
	
	// Check daily limit
	dailyKey := fmt.Sprintf("%s_%d", token, LimitPeriodDaily)
	if limit, exists := wallet.SpendingLimits[dailyKey]; exists {
		// Reset if needed
		if now.Sub(limit.LastReset) > 24*time.Hour {
			limit.Spent = big.NewInt(0)
			limit.LastReset = now
		}
		
		newSpent := new(big.Int).Add(limit.Spent, amount)
		if newSpent.Cmp(limit.Amount) > 0 {
			return fmt.Errorf("exceeds daily limit: %s", limit.Amount)
		}
	}
	
	// Check weekly limit
	weeklyKey := fmt.Sprintf("%s_%d", token, LimitPeriodWeekly)
	if limit, exists := wallet.SpendingLimits[weeklyKey]; exists {
		if now.Sub(limit.LastReset) > 7*24*time.Hour {
			limit.Spent = big.NewInt(0)
			limit.LastReset = now
		}
		
		newSpent := new(big.Int).Add(limit.Spent, amount)
		if newSpent.Cmp(limit.Amount) > 0 {
			return fmt.Errorf("exceeds weekly limit: %s", limit.Amount)
		}
	}
	
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// GetWalletInfo returns wallet information
func (mm *MultisigManager) GetWalletInfo(walletID string) (map[string]interface{}, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	wallet, exists := mm.Wallets[walletID]
	if !exists {
		return nil, errors.New("wallet not found")
	}

	wallet.mu.RLock()
	defer wallet.mu.RUnlock()

	owners := make([]string, 0, len(wallet.Owners))
	for addr := range wallet.Owners {
		owners = append(owners, addr)
	}

	return map[string]interface{}{
		"id":               wallet.ID,
		"name":             wallet.Name,
		"owners":           owners,
		"required_sigs":    wallet.RequiredSigs,
		"total_owners":     wallet.TotalOwners,
		"pending_txs":      len(wallet.PendingTxs),
		"executed_txs":     len(wallet.ExecutedTxs),
		"balance":          wallet.Balance,
		"frozen":           wallet.Frozen,
		"created_at":       wallet.CreatedAt,
		"last_activity":    wallet.LastActivity,
	}, nil
}

// GetUserWallets returns all wallets a user is part of
func (mm *MultisigManager) GetUserWallets(user string) []*MultisigWallet {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	return mm.UserWallets[user]
}