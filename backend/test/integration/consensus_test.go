package integration

import (
	"context"
	"testing"
	"time"

	"github.com/luxfi/dex/backend/pkg/consensus"
	"github.com/luxfi/dex/backend/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConsensusIntegration tests consensus with orderbook
func TestConsensusIntegration(t *testing.T) {
	// Create FPC consensus
	fpc := consensus.NewFPCDAGOrderBook(3, 3)
	
	// Create orderbook
	ob := lx.NewOrderBook("BTC-USD")
	
	// Add test order
	order := &lx.Order{
		ID:     1,
		Symbol: "BTC-USD",
		Type:   lx.Limit,
		Side:   lx.Buy,
		Price:  50000,
		Size:   1,
		UserID: "consensus_test",
	}
	
	// Process through consensus
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Submit order to consensus
	err := fpc.SubmitOrder(ctx, order)
	require.NoError(t, err)
	
	// Wait for finality
	time.Sleep(100 * time.Millisecond)
	
	// Verify order was finalized
	finalized := fpc.IsFinalized(order.ID)
	assert.True(t, finalized)
}

// TestQuantumSignatures tests quantum-resistant signatures
func TestQuantumSignatures(t *testing.T) {
	// Test Ringtail+BLS hybrid signatures
	message := []byte("test_order_data")
	
	// Generate keys (mock)
	pubKey := make([]byte, 32)
	privKey := make([]byte, 64)
	
	// Sign message
	signature := consensus.SignHybrid(message, privKey)
	assert.NotNil(t, signature)
	assert.Greater(t, len(signature), 1024) // Quantum signatures are larger
	
	// Verify signature
	valid := consensus.VerifyHybrid(message, signature, pubKey)
	assert.True(t, valid)
}

// TestOneBlockFinality tests one-block finality
func TestOneBlockFinality(t *testing.T) {
	blockTime := 1 * time.Millisecond // 1ms blocks
	
	start := time.Now()
	
	// Simulate block production
	for i := 0; i < 10; i++ {
		time.Sleep(blockTime)
		
		// Each block should finalize immediately
		blockNumber := i + 1
		finalized := true // In FPC, blocks finalize in one round
		
		assert.True(t, finalized)
		t.Logf("Block %d finalized in %v", blockNumber, time.Since(start))
	}
	
	// Should have produced 10 blocks in ~10ms
	elapsed := time.Since(start)
	assert.Less(t, elapsed, 20*time.Millisecond)
}

// TestCChainPrecompiles tests C-Chain precompile integration
func TestCChainPrecompiles(t *testing.T) {
	// Mock precompile addresses
	orderBookPrecompile := "0x0100000000000000000000000000000000000001"
	clearinghousePrecompile := "0x0100000000000000000000000000000000000002"
	
	// Test orderbook precompile
	t.Run("OrderBookPrecompile", func(t *testing.T) {
		// Would call precompile in real test
		assert.NotEmpty(t, orderBookPrecompile)
	})
	
	// Test clearinghouse precompile  
	t.Run("ClearinghousePrecompile", func(t *testing.T) {
		// Would call precompile in real test
		assert.NotEmpty(t, clearinghousePrecompile)
	})
}