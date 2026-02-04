package lx

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewEnhancedBridge tests bridge creation
func TestNewEnhancedBridge(t *testing.T) {
	bridge := NewEnhancedBridge()

	assert.NotNil(t, bridge)
	assert.NotNil(t, bridge.CrossChainBridge)
	assert.NotNil(t, bridge.Chains)
	assert.NotNil(t, bridge.ActiveTransfers)
	assert.NotNil(t, bridge.TransferHistory)
	assert.NotNil(t, bridge.MultisigWallet)
	assert.NotNil(t, bridge.FraudProofs)
	assert.NotNil(t, bridge.BatchProcessor)
	assert.NotNil(t, bridge.Metrics)
	assert.NotNil(t, bridge.AlertManager)

	// Check CrossChainBridge initialization
	assert.NotNil(t, bridge.SupportedAssets)
	assert.NotNil(t, bridge.PendingTransfers)
	assert.NotNil(t, bridge.CompletedTransfers)
	assert.NotNil(t, bridge.FailedTransfers)
	assert.NotNil(t, bridge.LiquidityPools)
	assert.Equal(t, 15, bridge.RequiredConfirmations)
	assert.Equal(t, 24*time.Hour, bridge.ChallengePeriod)
}

// TestInitiateTransfer tests transfer initiation
func TestInitiateTransfer(t *testing.T) {
	bridge := NewEnhancedBridge()

	// Add test asset
	asset := &BridgeAsset{
		Symbol:      "USDC",
		Name:        "USD Coin",
		Decimals:    6,
		MinTransfer: big.NewInt(1000000),      // 1 USDC
		MaxTransfer: big.NewInt(10000000000),  // 10000 USDC
		DailyLimit:  big.NewInt(100000000000), // 100000 USDC
		DailyVolume: big.NewInt(0),
		LastReset:   time.Now(),
	}
	bridge.SupportedAssets["USDC"] = asset

	// Add test chains
	bridge.Chains["1"] = &ChainConfig{
		ChainID:         "1",
		Name:            "Ethereum",
		Type:            ChainTypeEVM,
		RPCEndpoint:     "https://eth.example.com",
		ContractAddress: "0x123",
		Confirmations:   12,
		Active:          true,
	}

	bridge.Chains["2"] = &ChainConfig{
		ChainID:         "2",
		Name:            "Lux",
		Type:            ChainTypeLux,
		RPCEndpoint:     "https://lux.example.com",
		ContractAddress: "0x456",
		Confirmations:   6,
		Active:          true,
	}

	// Add liquidity to destination chain for transfers
	_, err := bridge.AddLiquidity("2", "USDC", "liquidityProvider", big.NewInt(100000000000))
	require.NoError(t, err)

	t.Run("ValidTransfer", func(t *testing.T) {
		ctx := context.Background()
		from := "0xFromAddress"
		to := "0xToAddress"
		amount := big.NewInt(5000000000) // 5000 USDC

		transfer, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", from, to)

		require.NoError(t, err)
		assert.NotNil(t, transfer)
		assert.NotEmpty(t, transfer.ID)

		// Verify transfer was created
		bridge.mu.RLock()
		activeTransfer, exists := bridge.ActiveTransfers[transfer.ID]
		bridge.mu.RUnlock()

		assert.True(t, exists)
		assert.Equal(t, "USDC", activeTransfer.Asset)
		assert.Equal(t, amount, activeTransfer.Amount)
		assert.Equal(t, "1", activeTransfer.SourceChain)
		assert.Equal(t, "2", activeTransfer.DestChain)
		assert.Equal(t, from, activeTransfer.SourceAddress)
		assert.Equal(t, to, activeTransfer.DestAddress)
		assert.Equal(t, BridgeStatusPending, activeTransfer.Status)
	})

	t.Run("AmountBelowMinimum", func(t *testing.T) {
		ctx := context.Background()
		amount := big.NewInt(100) // Below minimum

		_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", "from", "to")

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "below minimum")
	})

	t.Run("AmountAboveMaximum", func(t *testing.T) {
		ctx := context.Background()
		amount := big.NewInt(100000000000) // Above maximum

		_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", "from", "to")

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds maximum")
	})

	t.Run("UnsupportedAsset", func(t *testing.T) {
		ctx := context.Background()
		amount := big.NewInt(1000000)

		_, err := bridge.InitiateTransfer(ctx, "UNKNOWN", amount, "1", "2", "from", "to")

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not supported")
	})

	t.Run("InactiveChain", func(t *testing.T) {
		// Deactivate destination chain
		bridge.Chains["2"].Active = false

		ctx := context.Background()
		amount := big.NewInt(5000000000)

		_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", "from", "to")

		assert.Error(t, err)

		// Reactivate for other tests
		bridge.Chains["2"].Active = true
	})
}

// TestValidateTransfer tests transfer validation
func TestValidateTransfer(t *testing.T) {
	bridge := createTestBridgeWithAssets(t)

	// Add test validators BEFORE initiating transfer to avoid race with notifyValidators goroutine
	for i := 0; i < 3; i++ {
		validator := &BridgeValidator{
			Address:  "validator" + string(rune('1'+i)),
			Stake:    big.NewInt(1000000),
			Active:   true,
			JoinedAt: time.Now(),
		}
		// Generate a key pair for the validator
		privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		validator.PublicKey = &privateKey.PublicKey
		bridge.BridgeValidators = append(bridge.BridgeValidators, validator)
	}

	// Create a transfer after validators are set up
	ctx := context.Background()
	from := "0xFromAddress"
	to := "0xToAddress"
	amount := big.NewInt(5000000000)

	transfer, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", from, to)
	require.NoError(t, err)

	t.Run("ValidValidation", func(t *testing.T) {
		validatorAddr := "validator1"
		signature := []byte("mock-signature")

		err := bridge.ValidateTransfer(transfer.ID, validatorAddr, signature)

		assert.NoError(t, err)

		// Check validation was recorded
		bridge.mu.RLock()
		activeTransfer := bridge.ActiveTransfers[transfer.ID]
		bridge.mu.RUnlock()

		assert.NotNil(t, activeTransfer.Validators[validatorAddr])
		assert.Equal(t, signature, activeTransfer.Validators[validatorAddr].Signature)
	})

	t.Run("InvalidTransferID", func(t *testing.T) {
		err := bridge.ValidateTransfer("invalid-id", "validator1", []byte("sig"))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "transfer not found")
	})

	t.Run("DuplicateValidation", func(t *testing.T) {
		validatorAddr := "validator1"
		signature := []byte("mock-signature-2")

		err := bridge.ValidateTransfer(transfer.ID, validatorAddr, signature)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already validated this transfer")
	})
}

// TestAddLiquidity tests liquidity pool operations
func TestAddLiquidity(t *testing.T) {
	bridge := createTestBridgeWithAssets(t)

	t.Run("AddValidLiquidity", func(t *testing.T) {
		provider := "0xLiquidityProvider"
		asset := "USDC"
		chainID := "1"
		amount := big.NewInt(10000000000) // 10000 USDC

		lp, err := bridge.AddLiquidity(chainID, asset, provider, amount)

		assert.NoError(t, err)
		assert.NotNil(t, lp)

		// Check liquidity was added
		bridge.mu.RLock()
		poolKey := chainID + ":" + asset
		pool, exists := bridge.LiquidityPools[poolKey]
		bridge.mu.RUnlock()

		assert.True(t, exists)
		assert.Equal(t, amount, pool.TotalLiquidity)
		assert.Equal(t, amount, pool.Providers[provider].Amount)
	})

	t.Run("AddMoreLiquidity", func(t *testing.T) {
		provider := "0xLiquidityProvider"
		asset := "USDC"
		chainID := "1"
		additionalAmount := big.NewInt(5000000000) // 5000 USDC

		lp, err := bridge.AddLiquidity(chainID, asset, provider, additionalAmount)

		assert.NoError(t, err)
		assert.NotNil(t, lp)

		// Check liquidity was increased
		bridge.mu.RLock()
		poolKey := chainID + ":" + asset
		pool := bridge.LiquidityPools[poolKey]
		bridge.mu.RUnlock()

		expectedTotal := big.NewInt(15000000000)
		assert.Equal(t, expectedTotal, pool.TotalLiquidity)
		assert.Equal(t, expectedTotal, pool.Providers[provider].Amount)
	})
}

// TestRemoveLiquidity tests liquidity removal
func TestRemoveLiquidity(t *testing.T) {
	bridge := createTestBridgeWithAssets(t)

	// First add liquidity
	provider := "0xLiquidityProvider"
	asset := "USDC"
	chainID := "1"
	initialAmount := big.NewInt(10000000000) // 10000 USDC

	_, err := bridge.AddLiquidity(chainID, asset, provider, initialAmount)
	require.NoError(t, err)

	t.Run("RemovePartialLiquidity", func(t *testing.T) {
		removeAmount := big.NewInt(3000000000) // 3000 USDC

		withdrawAmount, err := bridge.RemoveLiquidity(chainID, asset, provider, removeAmount)

		assert.NoError(t, err)
		assert.NotNil(t, withdrawAmount)

		bridge.mu.RLock()
		poolKey := chainID + ":" + asset
		pool := bridge.LiquidityPools[poolKey]
		bridge.mu.RUnlock()

		expectedRemaining := big.NewInt(7000000000)
		assert.Equal(t, expectedRemaining, pool.TotalLiquidity)
		assert.Equal(t, expectedRemaining, pool.Providers[provider].Amount)
	})

	t.Run("RemoveExcessLiquidity", func(t *testing.T) {
		removeAmount := big.NewInt(20000000000) // More than available

		_, err := bridge.RemoveLiquidity(chainID, asset, provider, removeAmount)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient")
	})
}

// TestSubmitFraudProof tests fraud proof submission
func TestSubmitFraudProof(t *testing.T) {
	bridge := createTestBridgeWithAssets(t)
	ctx := context.Background()

	// Create a transfer
	from := "0xFromAddress"
	to := "0xToAddress"
	amount := big.NewInt(5000000000)

	transfer, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", from, to)
	require.NoError(t, err)

	t.Run("SubmitValidFraudProof", func(t *testing.T) {
		reporter := "0xReporter"
		evidence := []byte("fraud-evidence")

		err := bridge.SubmitFraudProof(transfer.ID, "InvalidSignature", evidence, reporter)

		assert.NoError(t, err)

		// Check fraud proof was recorded
		bridge.mu.RLock()
		proof, exists := bridge.FraudProofs[transfer.ID]
		bridge.mu.RUnlock()

		assert.True(t, exists)
		assert.Equal(t, transfer.ID, proof.TransferID)
		assert.Equal(t, "InvalidSignature", proof.ProofType)
		assert.Equal(t, evidence, proof.Evidence)
		assert.Equal(t, reporter, proof.Submitter)
	})

	t.Run("SubmitForInvalidTransfer", func(t *testing.T) {
		err := bridge.SubmitFraudProof("invalid-id", "InvalidSignature", []byte("evidence"), "reporter")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "transfer not found")
	})
}

// TestBridgeStructs tests bridge data structures
func TestBridgeStructs(t *testing.T) {
	t.Run("BridgeAsset", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			Name:        "USD Coin",
			Decimals:    6,
			MinTransfer: big.NewInt(1000000),
			MaxTransfer: big.NewInt(10000000000),
			DailyLimit:  big.NewInt(100000000000),
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
			Paused:      false,
		}

		assert.Equal(t, "USDC", asset.Symbol)
		assert.Equal(t, uint8(6), asset.Decimals)
		assert.False(t, asset.Paused)
	})

	t.Run("BridgeTransfer", func(t *testing.T) {
		transfer := &BridgeTransfer{
			ID:            "transfer-123",
			Asset:         "USDC",
			Amount:        big.NewInt(5000000000),
			Fee:           big.NewInt(5000000),
			SourceChain:   "1",
			DestChain:     "2",
			SourceAddress: "0xFrom",
			DestAddress:   "0xTo",
			Status:        BridgeStatusPending,
			Validators:    make(map[string]*BridgeSignature),
			Nonce:         1,
			InitiatedAt:   time.Now(),
		}

		assert.Equal(t, "transfer-123", transfer.ID)
		assert.Equal(t, BridgeStatusPending, transfer.Status)
		assert.Equal(t, uint64(1), transfer.Nonce)
	})

	t.Run("ChainConfig", func(t *testing.T) {
		config := &ChainConfig{
			ChainID:         "1",
			Name:            "Ethereum",
			Type:            ChainTypeEVM,
			RPCEndpoint:     "https://eth.example.com",
			ContractAddress: "0x123",
			Confirmations:   12,
			BlockTime:       15 * time.Second,
			GasPrice:        big.NewInt(30000000000),
			Active:          true,
		}

		assert.Equal(t, "1", config.ChainID)
		assert.Equal(t, ChainTypeEVM, config.Type)
		assert.True(t, config.Active)
	})

	t.Run("BridgeValidator", func(t *testing.T) {
		validator := &BridgeValidator{
			Address:  "0xValidator",
			Stake:    big.NewInt(1000000),
			Active:   true,
			Slashed:  false,
			JoinedAt: time.Now(),
		}

		assert.Equal(t, "0xValidator", validator.Address)
		assert.True(t, validator.Active)
		assert.False(t, validator.Slashed)
	})

	t.Run("MultisigBridge", func(t *testing.T) {
		multisig := &MultisigBridge{
			RequiredSigs:    2,
			TotalSigners:    3,
			Signers:         make(map[string]*BridgeSigner),
			PendingTxs:      make(map[string]*MultisigTx),
			ExecutedTxs:     make(map[string]*MultisigTx),
			TimeoutDuration: 24 * time.Hour,
		}

		assert.Equal(t, 2, multisig.RequiredSigs)
		assert.Equal(t, 3, multisig.TotalSigners)
		assert.Equal(t, 24*time.Hour, multisig.TimeoutDuration)
	})

	t.Run("BatchBridgeProcessor", func(t *testing.T) {
		batch := &BatchBridgeProcessor{
			BatchSize:     100,
			BatchInterval: 10 * time.Second,
			PendingBatch:  make([]*BridgeTransfer, 0),
		}

		assert.Equal(t, 100, batch.BatchSize)
		assert.Equal(t, 10*time.Second, batch.BatchInterval)
	})

	t.Run("BridgeMetrics", func(t *testing.T) {
		metrics := &BridgeMetrics{
			TotalTransfers:   1000,
			TotalVolume:      big.NewInt(1000000000000),
			AverageTime:      5 * time.Minute,
			SuccessRate:      0.99,
			ActiveValidators: 10,
			TotalLiquidity:   big.NewInt(50000000000000),
			DailyVolume:      make(map[string]*big.Int),
		}

		assert.Equal(t, uint64(1000), metrics.TotalTransfers)
		assert.Equal(t, 0.99, metrics.SuccessRate)
		assert.Equal(t, 10, metrics.ActiveValidators)
	})
}

// TestConcurrentBridgeOperations tests concurrent bridge operations
func TestConcurrentBridgeOperations(t *testing.T) {
	bridge := createTestBridgeWithAssets(t)
	ctx := context.Background()

	// Add initial liquidity to destination chain
	_, err := bridge.AddLiquidity("2", "USDC", "provider1", big.NewInt(100000000000))
	require.NoError(t, err)

	// Run concurrent transfers
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func(index int) {
			from := "0xFrom" + string(rune('0'+index))
			to := "0xTo" + string(rune('0'+index))
			amount := big.NewInt(int64(1000000000 + index*100000000))

			_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", from, to)
			assert.NoError(t, err)

			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}

	// Verify all transfers were created
	bridge.mu.RLock()
	transferCount := len(bridge.ActiveTransfers)
	bridge.mu.RUnlock()

	assert.Equal(t, 10, transferCount)
}

// Helper function to create a test bridge with assets
func createTestBridgeWithAssets(t testing.TB) *EnhancedBridge {
	bridge := NewEnhancedBridge()

	// Add test asset
	asset := &BridgeAsset{
		Symbol:      "USDC",
		Name:        "USD Coin",
		Decimals:    6,
		MinTransfer: big.NewInt(1000000),
		MaxTransfer: big.NewInt(100000000000),
		DailyLimit:  big.NewInt(1000000000000),
		DailyVolume: big.NewInt(0),
		LastReset:   time.Now(),
	}
	bridge.SupportedAssets["USDC"] = asset

	// Add test chains
	bridge.Chains["1"] = &ChainConfig{
		ChainID:         "1",
		Name:            "Ethereum",
		Type:            ChainTypeEVM,
		RPCEndpoint:     "https://eth.example.com",
		ContractAddress: "0x123",
		Confirmations:   12,
		Active:          true,
	}

	bridge.Chains["2"] = &ChainConfig{
		ChainID:         "2",
		Name:            "Lux",
		Type:            ChainTypeLux,
		RPCEndpoint:     "https://lux.example.com",
		ContractAddress: "0x456",
		Confirmations:   6,
		Active:          true,
	}

	// Add initial liquidity to destination chain
	bridge.AddLiquidity("2", "USDC", "initialProvider", big.NewInt(100000000000))

	return bridge
}

// TestFindValidator tests the findValidator function (0% coverage target)
func TestFindValidator(t *testing.T) {
	bridge := NewEnhancedBridge()

	// Add test validators
	privateKey1, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	privateKey2, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

	validator1 := &BridgeValidator{
		Address:   "0xValidator1",
		PublicKey: &privateKey1.PublicKey,
		Stake:     big.NewInt(1000000000000000000),
		Active:    true,
		Slashed:   false,
		JoinedAt:  time.Now(),
	}
	validator2 := &BridgeValidator{
		Address:   "0xValidator2",
		PublicKey: &privateKey2.PublicKey,
		Stake:     big.NewInt(2000000000000000000),
		Active:    true,
		Slashed:   false,
		JoinedAt:  time.Now(),
	}
	bridge.BridgeValidators = append(bridge.BridgeValidators, validator1, validator2)

	t.Run("FindExistingValidator", func(t *testing.T) {
		found := bridge.findValidator("0xValidator1")
		require.NotNil(t, found)
		assert.Equal(t, "0xValidator1", found.Address)
		assert.True(t, found.Active)
	})

	t.Run("FindSecondValidator", func(t *testing.T) {
		found := bridge.findValidator("0xValidator2")
		require.NotNil(t, found)
		assert.Equal(t, "0xValidator2", found.Address)
		assert.Equal(t, big.NewInt(2000000000000000000), found.Stake)
	})

	t.Run("FindNonExistentValidator", func(t *testing.T) {
		found := bridge.findValidator("0xNonExistent")
		assert.Nil(t, found)
	})

	t.Run("FindValidatorEmptyAddress", func(t *testing.T) {
		found := bridge.findValidator("")
		assert.Nil(t, found)
	})

	t.Run("FindValidatorEmptyList", func(t *testing.T) {
		emptyBridge := NewEnhancedBridge()
		found := emptyBridge.findValidator("0xAnyAddress")
		assert.Nil(t, found)
	})
}

// TestFindValidatorLocked tests the findValidatorLocked function
func TestFindValidatorLocked(t *testing.T) {
	bridge := NewEnhancedBridge()

	privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	validator := &BridgeValidator{
		Address:   "0xLockedValidator",
		PublicKey: &privateKey.PublicKey,
		Stake:     big.NewInt(500000000000000000),
		Active:    true,
		JoinedAt:  time.Now(),
	}
	bridge.BridgeValidators = append(bridge.BridgeValidators, validator)

	t.Run("FindWithManualLock", func(t *testing.T) {
		bridge.mu.Lock()
		found := bridge.findValidatorLocked("0xLockedValidator")
		bridge.mu.Unlock()

		require.NotNil(t, found)
		assert.Equal(t, "0xLockedValidator", found.Address)
	})
}

// TestFindTransfer tests the findTransfer function (0% coverage target)
func TestFindTransfer(t *testing.T) {
	bridge := NewEnhancedBridge()

	// Create transfers in different states
	pendingTransfer := &BridgeTransfer{
		ID:            "pending-001",
		Asset:         "USDC",
		Amount:        big.NewInt(1000000),
		SourceChain:   "1",
		DestChain:     "2",
		SourceAddress: "0xFrom",
		DestAddress:   "0xTo",
		Status:        BridgeStatusPending,
		InitiatedAt:   time.Now(),
		Validators:    make(map[string]*BridgeSignature),
	}

	completedTransfer := &BridgeTransfer{
		ID:            "completed-001",
		Asset:         "ETH",
		Amount:        big.NewInt(1000000000000000000),
		SourceChain:   "1",
		DestChain:     "2",
		SourceAddress: "0xFrom",
		DestAddress:   "0xTo",
		Status:        BridgeStatusCompleted,
		InitiatedAt:   time.Now().Add(-1 * time.Hour),
		CompletedAt:   time.Now(),
		Validators:    make(map[string]*BridgeSignature),
	}

	failedTransfer := &BridgeTransfer{
		ID:            "failed-001",
		Asset:         "DAI",
		Amount:        big.NewInt(5000000000000000000),
		SourceChain:   "1",
		DestChain:     "2",
		SourceAddress: "0xFrom",
		DestAddress:   "0xTo",
		Status:        BridgeStatusFailed,
		InitiatedAt:   time.Now().Add(-2 * time.Hour),
		CompletedAt:   time.Now().Add(-1 * time.Hour),
		Validators:    make(map[string]*BridgeSignature),
	}

	// Store transfers
	bridge.PendingTransfers[pendingTransfer.ID] = pendingTransfer
	bridge.CompletedTransfers[completedTransfer.ID] = completedTransfer
	bridge.FailedTransfers[failedTransfer.ID] = failedTransfer

	t.Run("FindPendingTransfer", func(t *testing.T) {
		found := bridge.findTransfer("pending-001")
		require.NotNil(t, found)
		assert.Equal(t, "pending-001", found.ID)
		assert.Equal(t, BridgeStatusPending, found.Status)
		assert.Equal(t, "USDC", found.Asset)
	})

	t.Run("FindCompletedTransfer", func(t *testing.T) {
		found := bridge.findTransfer("completed-001")
		require.NotNil(t, found)
		assert.Equal(t, "completed-001", found.ID)
		assert.Equal(t, BridgeStatusCompleted, found.Status)
		assert.Equal(t, "ETH", found.Asset)
	})

	t.Run("FindFailedTransfer", func(t *testing.T) {
		found := bridge.findTransfer("failed-001")
		require.NotNil(t, found)
		assert.Equal(t, "failed-001", found.ID)
		assert.Equal(t, BridgeStatusFailed, found.Status)
		assert.Equal(t, "DAI", found.Asset)
	})

	t.Run("FindNonExistentTransfer", func(t *testing.T) {
		found := bridge.findTransfer("non-existent-id")
		assert.Nil(t, found)
	})

	t.Run("FindTransferEmptyID", func(t *testing.T) {
		found := bridge.findTransfer("")
		assert.Nil(t, found)
	})

	t.Run("FindTransferEmptyBridge", func(t *testing.T) {
		emptyBridge := NewEnhancedBridge()
		found := emptyBridge.findTransfer("any-id")
		assert.Nil(t, found)
	})
}

// TestFindTransferLocked tests the findTransferLocked function
func TestFindTransferLocked(t *testing.T) {
	bridge := NewEnhancedBridge()

	transfer := &BridgeTransfer{
		ID:          "locked-transfer-001",
		Asset:       "USDC",
		Amount:      big.NewInt(1000000),
		Status:      BridgeStatusPending,
		InitiatedAt: time.Now(),
		Validators:  make(map[string]*BridgeSignature),
	}
	bridge.PendingTransfers[transfer.ID] = transfer

	t.Run("FindWithManualLock", func(t *testing.T) {
		bridge.mu.Lock()
		found := bridge.findTransferLocked("locked-transfer-001")
		bridge.mu.Unlock()

		require.NotNil(t, found)
		assert.Equal(t, "locked-transfer-001", found.ID)
	})

	t.Run("NotFoundWithLock", func(t *testing.T) {
		bridge.mu.Lock()
		found := bridge.findTransferLocked("non-existent")
		bridge.mu.Unlock()

		assert.Nil(t, found)
	})
}

// TestGetBridgeStatus tests the GetBridgeStatus function (0% coverage target)
func TestGetBridgeStatus(t *testing.T) {
	t.Run("DefaultBridgeStatus", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		status := bridge.GetBridgeStatus()

		require.NotNil(t, status)
		assert.True(t, status["active"].(bool))
		assert.Equal(t, 0, status["pending_transfers"].(int))
		assert.Equal(t, uint64(0), status["total_transfers"].(uint64))
		assert.Equal(t, "0", status["total_volume"].(string))
		assert.Equal(t, "0", status["total_liquidity"].(string))
		assert.Equal(t, 0, status["active_validators"].(int))
		assert.Equal(t, 0, status["chains"].(int))
		assert.Equal(t, 0, status["assets"].(int))
	})

	t.Run("BridgeStatusWithData", func(t *testing.T) {
		bridge := createTestBridgeWithAssets(t)

		// Add a pending transfer
		transfer := &BridgeTransfer{
			ID:          "status-test-001",
			Asset:       "USDC",
			Amount:      big.NewInt(5000000000),
			Status:      BridgeStatusPending,
			InitiatedAt: time.Now(),
			Validators:  make(map[string]*BridgeSignature),
		}
		bridge.PendingTransfers[transfer.ID] = transfer

		// Set metrics
		bridge.Metrics.TotalTransfers = 100
		bridge.Metrics.TotalVolume = big.NewInt(1000000000000)
		bridge.Metrics.TotalLiquidity = big.NewInt(500000000000)
		bridge.Metrics.ActiveValidators = 10

		status := bridge.GetBridgeStatus()

		require.NotNil(t, status)
		assert.True(t, status["active"].(bool))
		assert.Equal(t, 1, status["pending_transfers"].(int))
		assert.Equal(t, uint64(100), status["total_transfers"].(uint64))
		assert.Equal(t, "1000000000000", status["total_volume"].(string))
		assert.Equal(t, "500000000000", status["total_liquidity"].(string))
		assert.Equal(t, 10, status["active_validators"].(int))
		assert.Equal(t, 2, status["chains"].(int))
		assert.Equal(t, 1, status["assets"].(int))
	})

	t.Run("BridgeStatusPaused", func(t *testing.T) {
		bridge := NewEnhancedBridge()
		bridge.EmergencyPaused = true

		status := bridge.GetBridgeStatus()

		assert.False(t, status["active"].(bool))
	})

	t.Run("BridgeStatusConcurrent", func(t *testing.T) {
		bridge := NewEnhancedBridge()
		done := make(chan bool, 10)

		// Run concurrent GetBridgeStatus calls
		for i := 0; i < 10; i++ {
			go func() {
				status := bridge.GetBridgeStatus()
				assert.NotNil(t, status)
				done <- true
			}()
		}

		// Wait for all goroutines
		for i := 0; i < 10; i++ {
			<-done
		}
	})
}

// TestDistributeFees tests the distributeFees function (40% coverage target)
func TestDistributeFees(t *testing.T) {
	bridge := NewEnhancedBridge()

	t.Run("DistributeFeesEmptyPool", func(t *testing.T) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			ChainID:            "1",
			TotalLiquidity:     big.NewInt(0),
			AvailableLiquidity: big.NewInt(0),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}

		fee := big.NewInt(1000000) // 1 USDC fee

		// Should not panic with empty providers
		bridge.distributeFees(pool, fee)

		// Fees should not be added to pool with no providers
		assert.Equal(t, big.NewInt(0), pool.Fees)
	})

	t.Run("DistributeFeesSingleProvider", func(t *testing.T) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			ChainID:            "1",
			TotalLiquidity:     big.NewInt(10000000000),
			AvailableLiquidity: big.NewInt(10000000000),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}

		pool.Providers["provider1"] = &LiquidityProvider{
			Address:     "provider1",
			Amount:      big.NewInt(10000000000),
			ShareTokens: big.NewInt(10000000000),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}

		fee := big.NewInt(1000000) // 1 USDC fee

		bridge.distributeFees(pool, fee)

		// Single provider should get all fees
		assert.Equal(t, big.NewInt(1000000), pool.Providers["provider1"].Rewards)
		assert.Equal(t, big.NewInt(1000000), pool.Fees)
	})

	t.Run("DistributeFeesMultipleProviders", func(t *testing.T) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			ChainID:            "1",
			TotalLiquidity:     big.NewInt(10000000000),
			AvailableLiquidity: big.NewInt(10000000000),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}

		// Provider 1 has 60% shares
		pool.Providers["provider1"] = &LiquidityProvider{
			Address:     "provider1",
			Amount:      big.NewInt(6000000000),
			ShareTokens: big.NewInt(6000000000),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}

		// Provider 2 has 40% shares
		pool.Providers["provider2"] = &LiquidityProvider{
			Address:     "provider2",
			Amount:      big.NewInt(4000000000),
			ShareTokens: big.NewInt(4000000000),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}

		fee := big.NewInt(10000000) // 10 USDC fee

		bridge.distributeFees(pool, fee)

		// Provider 1 should get 60% of fees (6M)
		assert.Equal(t, big.NewInt(6000000), pool.Providers["provider1"].Rewards)
		// Provider 2 should get 40% of fees (4M)
		assert.Equal(t, big.NewInt(4000000), pool.Providers["provider2"].Rewards)
		// Total fees should be added to pool
		assert.Equal(t, big.NewInt(10000000), pool.Fees)
	})

	t.Run("DistributeFeesMultipleTimes", func(t *testing.T) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			ChainID:            "1",
			TotalLiquidity:     big.NewInt(10000000000),
			AvailableLiquidity: big.NewInt(10000000000),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}

		pool.Providers["provider1"] = &LiquidityProvider{
			Address:     "provider1",
			Amount:      big.NewInt(10000000000),
			ShareTokens: big.NewInt(10000000000),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}

		// Distribute fees multiple times
		bridge.distributeFees(pool, big.NewInt(1000000))
		bridge.distributeFees(pool, big.NewInt(2000000))
		bridge.distributeFees(pool, big.NewInt(3000000))

		// Rewards should accumulate
		assert.Equal(t, big.NewInt(6000000), pool.Providers["provider1"].Rewards)
		assert.Equal(t, big.NewInt(6000000), pool.Fees)
	})

	t.Run("DistributeZeroFees", func(t *testing.T) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			ChainID:            "1",
			TotalLiquidity:     big.NewInt(10000000000),
			AvailableLiquidity: big.NewInt(10000000000),
			LockedLiquidity:    big.NewInt(0),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}

		pool.Providers["provider1"] = &LiquidityProvider{
			Address:     "provider1",
			Amount:      big.NewInt(10000000000),
			ShareTokens: big.NewInt(10000000000),
			JoinedAt:    time.Now(),
			Rewards:     big.NewInt(0),
		}

		// Distribute zero fees
		bridge.distributeFees(pool, big.NewInt(0))

		// Rewards should still be zero
		assert.Equal(t, big.NewInt(0), pool.Providers["provider1"].Rewards)
	})
}

// TestSlashValidators tests the slashValidators function (30% coverage target)
func TestSlashValidators(t *testing.T) {
	t.Run("SlashNoValidators", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		transfer := &BridgeTransfer{
			ID:         "slash-test-001",
			Asset:      "USDC",
			Amount:     big.NewInt(1000000),
			Validators: make(map[string]*BridgeSignature),
		}

		// Should not panic with empty validators
		bridge.mu.Lock()
		bridge.slashValidators(transfer, big.NewInt(1000000))
		bridge.mu.Unlock()
	})

	t.Run("SlashSingleValidator", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		stake, _ := new(big.Int).SetString("10000000000000000000", 10) // 10 ETH
		validator := &BridgeValidator{
			Address:   "validator1",
			PublicKey: &privateKey.PublicKey,
			Stake:     stake,
			Active:    true,
			Slashed:   false,
			JoinedAt:  time.Now(),
		}
		bridge.BridgeValidators = append(bridge.BridgeValidators, validator)

		transfer := &BridgeTransfer{
			ID:         "slash-single-001",
			Asset:      "USDC",
			Amount:     big.NewInt(1000000),
			Validators: make(map[string]*BridgeSignature),
		}
		transfer.Validators["validator1"] = &BridgeSignature{
			Validator: "validator1",
			Signature: []byte("sig"),
			Timestamp: time.Now(),
		}

		slashAmount, _ := new(big.Int).SetString("1000000000000000000", 10) // 1 ETH

		bridge.mu.Lock()
		bridge.slashValidators(transfer, slashAmount)
		bridge.mu.Unlock()

		// Validator should be slashed
		assert.True(t, validator.Slashed)
		assert.False(t, validator.Active)
		// Stake should be reduced
		expectedStake, _ := new(big.Int).SetString("9000000000000000000", 10) // 9 ETH
		assert.Equal(t, expectedStake, validator.Stake)
	})

	t.Run("SlashMultipleValidators", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		// Add 3 validators
		for i := 1; i <= 3; i++ {
			privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
			stake, _ := new(big.Int).SetString("10000000000000000000", 10) // 10 ETH each
			validator := &BridgeValidator{
				Address:   "validator" + string(rune('0'+i)),
				PublicKey: &privateKey.PublicKey,
				Stake:     stake,
				Active:    true,
				Slashed:   false,
				JoinedAt:  time.Now(),
			}
			bridge.BridgeValidators = append(bridge.BridgeValidators, validator)
		}

		transfer := &BridgeTransfer{
			ID:         "slash-multi-001",
			Asset:      "USDC",
			Amount:     big.NewInt(1000000),
			Validators: make(map[string]*BridgeSignature),
		}

		// All 3 validators signed the fraudulent transfer
		for i := 1; i <= 3; i++ {
			addr := "validator" + string(rune('0'+i))
			transfer.Validators[addr] = &BridgeSignature{
				Validator: addr,
				Signature: []byte("sig"),
				Timestamp: time.Now(),
			}
		}

		slashAmount, _ := new(big.Int).SetString("3000000000000000000", 10) // 3 ETH total (1 ETH each)

		bridge.mu.Lock()
		bridge.slashValidators(transfer, slashAmount)
		bridge.mu.Unlock()

		// All validators should be slashed
		expectedStake, _ := new(big.Int).SetString("9000000000000000000", 10) // 9 ETH
		for _, v := range bridge.BridgeValidators {
			assert.True(t, v.Slashed)
			assert.False(t, v.Active)
			// Each validator should have 1 ETH deducted
			assert.Equal(t, expectedStake, v.Stake)
		}
	})

	t.Run("SlashValidatorNotFound", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		transfer := &BridgeTransfer{
			ID:         "slash-notfound-001",
			Asset:      "USDC",
			Amount:     big.NewInt(1000000),
			Validators: make(map[string]*BridgeSignature),
		}
		transfer.Validators["nonexistent"] = &BridgeSignature{
			Validator: "nonexistent",
			Signature: []byte("sig"),
			Timestamp: time.Now(),
		}

		// Should not panic when validator not found
		bridge.mu.Lock()
		bridge.slashValidators(transfer, big.NewInt(1000000))
		bridge.mu.Unlock()
	})

	t.Run("SlashZeroAmount", func(t *testing.T) {
		bridge := NewEnhancedBridge()

		privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		validator := &BridgeValidator{
			Address:   "validator1",
			PublicKey: &privateKey.PublicKey,
			Stake:     func() *big.Int { v, _ := new(big.Int).SetString("10000000000000000000", 10); return v }(),
			Active:    true,
			Slashed:   false,
			JoinedAt:  time.Now(),
		}
		bridge.BridgeValidators = append(bridge.BridgeValidators, validator)

		transfer := &BridgeTransfer{
			ID:         "slash-zero-001",
			Validators: make(map[string]*BridgeSignature),
		}
		transfer.Validators["validator1"] = &BridgeSignature{
			Validator: "validator1",
			Signature: []byte("sig"),
			Timestamp: time.Now(),
		}

		bridge.mu.Lock()
		bridge.slashValidators(transfer, big.NewInt(0))
		bridge.mu.Unlock()

		// Validator should be slashed but stake unchanged
		assert.True(t, validator.Slashed)
		assert.False(t, validator.Active)
	})
}

// TestCheckDailyLimit tests the checkDailyLimit function (62.5% coverage target)
func TestCheckDailyLimit(t *testing.T) {
	bridge := NewEnhancedBridge()

	t.Run("WithinDailyLimit", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000), // 100k USDC
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
		}

		amount := big.NewInt(1000000000) // 1k USDC

		err := bridge.checkDailyLimit(asset, amount)
		assert.NoError(t, err)

		// Volume should be updated
		assert.Equal(t, amount, asset.DailyVolume)
	})

	t.Run("ExceedsDailyLimit", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000), // 100k USDC
			DailyVolume: big.NewInt(99000000000),  // 99k already used
			LastReset:   time.Now(),
		}

		amount := big.NewInt(2000000000) // 2k USDC (would exceed limit)

		err := bridge.checkDailyLimit(asset, amount)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds daily limit")
	})

	t.Run("ExactlyAtDailyLimit", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000), // 100k USDC
			DailyVolume: big.NewInt(99000000000),  // 99k already used
			LastReset:   time.Now(),
		}

		amount := big.NewInt(1000000000) // 1k USDC (exactly at limit)

		err := bridge.checkDailyLimit(asset, amount)
		assert.NoError(t, err)
	})

	t.Run("DailyVolumeReset", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000),        // 100k USDC
			DailyVolume: big.NewInt(100000000000),        // At limit
			LastReset:   time.Now().Add(-25 * time.Hour), // More than 24h ago
		}

		amount := big.NewInt(1000000000) // 1k USDC

		err := bridge.checkDailyLimit(asset, amount)
		assert.NoError(t, err)

		// Volume should be reset and equal to new amount
		assert.Equal(t, amount, asset.DailyVolume)
		// LastReset should be updated
		assert.True(t, time.Since(asset.LastReset) < time.Second)
	})

	t.Run("MultipleTransfersWithinLimit", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000), // 100k USDC
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
		}

		// Multiple transfers
		amounts := []*big.Int{
			big.NewInt(10000000000), // 10k
			big.NewInt(20000000000), // 20k
			big.NewInt(30000000000), // 30k
		}

		for _, amount := range amounts {
			err := bridge.checkDailyLimit(asset, amount)
			assert.NoError(t, err)
		}

		// Total should be 60k
		assert.Equal(t, big.NewInt(60000000000), asset.DailyVolume)
	})

	t.Run("MultipleTransfersExceedLimit", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000), // 100k USDC
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
		}

		// First two should succeed
		err := bridge.checkDailyLimit(asset, big.NewInt(50000000000))
		assert.NoError(t, err)
		err = bridge.checkDailyLimit(asset, big.NewInt(40000000000))
		assert.NoError(t, err)

		// Third should fail (would exceed)
		err = bridge.checkDailyLimit(asset, big.NewInt(20000000000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds daily limit")

		// Volume should still be 90k
		assert.Equal(t, big.NewInt(90000000000), asset.DailyVolume)
	})

	t.Run("ZeroAmount", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000),
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
		}

		err := bridge.checkDailyLimit(asset, big.NewInt(0))
		assert.NoError(t, err)
		assert.Equal(t, big.NewInt(0), asset.DailyVolume)
	})

	t.Run("ResetAtJustUnder24Hours", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000),
			DailyVolume: big.NewInt(99000000000),
			LastReset:   time.Now().Add(-24*time.Hour + time.Minute), // Just under 24h (should NOT reset)
		}

		amount := big.NewInt(2000000000) // Would exceed

		err := bridge.checkDailyLimit(asset, amount)
		// At under 24h, it should NOT reset (> not >=), so error expected
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds daily limit")
	})

	t.Run("ResetJustPast24Hours", func(t *testing.T) {
		asset := &BridgeAsset{
			Symbol:      "USDC",
			DailyLimit:  big.NewInt(100000000000),
			DailyVolume: big.NewInt(99000000000),
			LastReset:   time.Now().Add(-24*time.Hour - time.Minute), // Just past 24h
		}

		amount := big.NewInt(2000000000) // Would exceed without reset

		err := bridge.checkDailyLimit(asset, amount)
		assert.NoError(t, err)
		// Volume should be reset
		assert.Equal(t, amount, asset.DailyVolume)
	})
}

// TestBridgePausedState tests bridge operations when paused
func TestBridgePausedState(t *testing.T) {
	t.Run("InitiateTransferWhenPaused", func(t *testing.T) {
		bridge := createTestBridgeWithAssets(t)
		bridge.EmergencyPaused = true

		ctx := context.Background()
		amount := big.NewInt(5000000000)

		_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", "from", "to")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "bridge is paused")
	})

	t.Run("AssetPaused", func(t *testing.T) {
		bridge := createTestBridgeWithAssets(t)
		bridge.SupportedAssets["USDC"].Paused = true

		ctx := context.Background()
		amount := big.NewInt(5000000000)

		_, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", "from", "to")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "is paused")
	})
}

// BenchmarkBridgeOperations benchmarks bridge operations
func BenchmarkBridgeOperations(b *testing.B) {
	bridge := createTestBridgeWithAssets(b)
	ctx := context.Background()

	// Add liquidity for benchmarks to destination chain
	bridge.AddLiquidity("2", "USDC", "provider", big.NewInt(1000000000000))

	b.Run("InitiateTransfer", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.InitiateTransfer(ctx, "USDC", big.NewInt(10000000), "1", "2", "from", "to")
		}
	})

	b.Run("AddLiquidity", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.AddLiquidity("1", "USDC", "provider", big.NewInt(1000000))
		}
	})

	b.Run("FindValidator", func(b *testing.B) {
		privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		bridge.BridgeValidators = append(bridge.BridgeValidators, &BridgeValidator{
			Address:   "benchValidator",
			PublicKey: &privateKey.PublicKey,
			Stake:     big.NewInt(1000000),
			Active:    true,
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.findValidator("benchValidator")
		}
	})

	b.Run("FindTransfer", func(b *testing.B) {
		bridge.PendingTransfers["benchTransfer"] = &BridgeTransfer{
			ID:         "benchTransfer",
			Asset:      "USDC",
			Amount:     big.NewInt(1000000),
			Validators: make(map[string]*BridgeSignature),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.findTransfer("benchTransfer")
		}
	})

	b.Run("GetBridgeStatus", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.GetBridgeStatus()
		}
	})

	b.Run("DistributeFees", func(b *testing.B) {
		pool := &BridgeLiquidityPool{
			Asset:              "USDC",
			TotalLiquidity:     big.NewInt(1000000000),
			AvailableLiquidity: big.NewInt(1000000000),
			Providers:          make(map[string]*LiquidityProvider),
			Fees:               big.NewInt(0),
		}
		pool.Providers["p1"] = &LiquidityProvider{
			ShareTokens: big.NewInt(500000000),
			Rewards:     big.NewInt(0),
		}
		pool.Providers["p2"] = &LiquidityProvider{
			ShareTokens: big.NewInt(500000000),
			Rewards:     big.NewInt(0),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bridge.distributeFees(pool, big.NewInt(10000))
		}
	})

	b.Run("CheckDailyLimit", func(b *testing.B) {
		asset := &BridgeAsset{
			DailyLimit:  big.NewInt(1000000000000),
			DailyVolume: big.NewInt(0),
			LastReset:   time.Now(),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			asset.DailyVolume = big.NewInt(0) // Reset for each iteration
			bridge.checkDailyLimit(asset, big.NewInt(1000000))
		}
	})
}
