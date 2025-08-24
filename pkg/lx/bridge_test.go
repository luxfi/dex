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

	// Create a transfer
	ctx := context.Background()
	from := "0xFromAddress"
	to := "0xToAddress"
	amount := big.NewInt(5000000000)

	transfer, err := bridge.InitiateTransfer(ctx, "USDC", amount, "1", "2", from, to)
	require.NoError(t, err)

	// Add test validators
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
}
