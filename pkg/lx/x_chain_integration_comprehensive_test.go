package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test XChainIntegration functions with 0% coverage
func TestXChainIntegrationFunctions(t *testing.T) {
	t.Run("NewXChainIntegration", func(t *testing.T) {
		chainID := uint64(1337)
		contractAddr := "0x1234567890123456789012345678901234567890"
		
		integration := NewXChainIntegration(chainID, contractAddr)
		
		assert.NotNil(t, integration)
		assert.Equal(t, chainID, integration.chainID)
		assert.Equal(t, contractAddr, integration.contractAddr)
		assert.NotNil(t, integration.settledOrders)
		assert.NotNil(t, integration.pendingBatches)
		assert.Equal(t, 0, len(integration.settledOrders))
		assert.Equal(t, 0, len(integration.pendingBatches))
	})

	t.Run("SubmitBatch", func(t *testing.T) {
		integration := NewXChainIntegration(1337, "0x1234567890123456789012345678901234567890")
		
		orders := []*Order{
			{ID: 1, Type: Limit, Side: Buy, Size: 1.0, Price: 50000, User: "user1", Timestamp: time.Now()},
			{ID: 2, Type: Limit, Side: Sell, Size: 1.0, Price: 50100, User: "user2", Timestamp: time.Now()},
		}
		
		batch := &SettlementBatch{
			BatchID:   12345,
			Timestamp: time.Now(),
			Orders:    orders,
			Status:    SettlementPending,
			CreatedAt: time.Now(),
		}
		
		err := integration.SubmitBatch(batch)
		assert.NoError(t, err)
		assert.Equal(t, SettlementPending, batch.Status)
		
		// Check batch was added to pending
		assert.Equal(t, 1, len(integration.pendingBatches))
		assert.Equal(t, batch, integration.pendingBatches[12345])
		
		// Wait for processing to complete
		time.Sleep(400 * time.Millisecond)
		
		// Batch should be completed and moved to settled orders
		assert.Equal(t, 0, len(integration.pendingBatches))
		assert.Equal(t, 2, len(integration.settledOrders))
		
		// Check settled orders
		settled1 := integration.settledOrders[1]
		assert.NotNil(t, settled1)
		assert.Equal(t, uint64(1), settled1.OrderID)
		assert.NotEmpty(t, settled1.TxHash)
		
		settled2 := integration.settledOrders[2]
		assert.NotNil(t, settled2)
		assert.Equal(t, uint64(2), settled2.OrderID)
		assert.NotEmpty(t, settled2.TxHash)
	})

	t.Run("GetStats", func(t *testing.T) {
		integration := NewXChainIntegration(1337, "0x1234567890123456789012345678901234567890")
		
		// Initial stats should be empty
		stats := integration.GetStats()
		assert.NotNil(t, stats)
		assert.Equal(t, uint64(0), stats.TotalSettled)
		assert.Equal(t, uint64(0), stats.TotalPending)
		assert.Equal(t, uint64(0), stats.TotalFailed)
		assert.Equal(t, big.NewInt(0), stats.AvgGasUsed)
		
		// Add some settled orders manually
		integration.settledOrders[1] = &SettledOrder{
			OrderID:        1,
			TxHash:         "0xhash1",
			BlockNumber:    1000,
			SettlementTime: time.Now().Add(-1 * time.Hour),
			GasUsed:        big.NewInt(21000),
		}
		integration.settledOrders[2] = &SettledOrder{
			OrderID:        2,
			TxHash:         "0xhash2",
			BlockNumber:    1001,
			SettlementTime: time.Now(),
			GasUsed:        big.NewInt(25000),
		}
		
		// Add pending batch
		integration.pendingBatches[1] = &SettlementBatch{
			BatchID: 1,
			Orders:  []*Order{{ID: 3}},
			Status:  SettlementPending,
		}
		
		stats = integration.GetStats()
		assert.Equal(t, uint64(2), stats.TotalSettled)
		assert.Equal(t, uint64(1), stats.TotalPending)
		assert.Equal(t, big.NewInt(23000), stats.AvgGasUsed) // (21000+25000)/2
		assert.Equal(t, 300*time.Millisecond, stats.AvgSettlementTime)
		assert.True(t, time.Since(stats.LastSettlement) < time.Second)
	})

	t.Run("processBatch", func(t *testing.T) {
		integration := NewXChainIntegration(1337, "0x1234567890123456789012345678901234567890")
		
		batch := &SettlementBatch{
			BatchID:   98765,
			Orders:    []*Order{{ID: 100}},
			Status:    SettlementPending,
			CreatedAt: time.Now(),
		}
		
		// Process batch directly
		go integration.processBatch(batch)
		
		// Wait for initial processing state
		time.Sleep(150 * time.Millisecond)
		assert.Equal(t, SettlementProcessing, batch.Status)
		
		// Wait for completion
		time.Sleep(300 * time.Millisecond)
		assert.Equal(t, SettlementComplete, batch.Status)
		assert.NotEmpty(t, batch.TxHash)
		assert.NotNil(t, batch.GasUsed)
		
		// Check settled order was created
		settled := integration.settledOrders[100]
		assert.NotNil(t, settled)
		assert.Equal(t, uint64(100), settled.OrderID)
		assert.Equal(t, batch.TxHash, settled.TxHash)
	})
}

// Test SettlementEngine functions
func TestSettlementEngineFunctions(t *testing.T) {
	t.Run("NewSettlementEngine", func(t *testing.T) {
		batchSize := 50
		batchTimeout := 30 * time.Second
		
		engine := NewSettlementEngine(batchSize, batchTimeout)
		
		assert.NotNil(t, engine)
		assert.NotNil(t, engine.batches)
		assert.NotNil(t, engine.pendingOrders)
		assert.Equal(t, batchSize, engine.batchSize)
		assert.Equal(t, batchTimeout, engine.batchTimeout)
		assert.Equal(t, 0, len(engine.pendingOrders))
	})

	t.Run("AddOrder", func(t *testing.T) {
		engine := NewSettlementEngine(3, 30*time.Second)
		
		// Add orders one by one
		order1 := &Order{ID: 1, Type: Limit, Side: Buy, Size: 1.0, Price: 50000}
		order2 := &Order{ID: 2, Type: Limit, Side: Sell, Size: 1.0, Price: 50100}
		order3 := &Order{ID: 3, Type: Market, Side: Buy, Size: 0.5, Price: 50050}
		
		engine.AddOrder(order1)
		assert.Equal(t, 1, len(engine.pendingOrders))
		assert.Equal(t, 0, len(engine.batches))
		
		engine.AddOrder(order2)
		assert.Equal(t, 2, len(engine.pendingOrders))
		assert.Equal(t, 0, len(engine.batches))
		
		// Adding third order should trigger batch creation
		engine.AddOrder(order3)
		assert.Equal(t, 0, len(engine.pendingOrders)) // Orders moved to batch
		assert.Equal(t, 1, len(engine.batches))
		
		// Verify batch contains all orders
		var batch *SettlementBatch
		for _, b := range engine.batches {
			batch = b
			break
		}
		assert.NotNil(t, batch)
		assert.Equal(t, 3, len(batch.Orders))
		assert.Equal(t, SettlementPending, batch.Status)
	})

	t.Run("CreateBatch", func(t *testing.T) {
		engine := NewSettlementEngine(10, 30*time.Second)
		
		settledOrders := []*SettledOrder{
			{OrderID: 1, TxHash: "0xhash1", BlockNumber: 1000},
			{OrderID: 2, TxHash: "0xhash2", BlockNumber: 1001},
		}
		
		batch := engine.CreateBatch(settledOrders)
		
		assert.NotNil(t, batch)
		assert.Equal(t, uint64(1), batch.BatchID)
		assert.Equal(t, 2, len(batch.Orders))
		assert.Equal(t, SettlementPending, batch.Status)
		assert.True(t, time.Since(batch.CreatedAt) < time.Second)
		
		// Check batch was stored
		assert.Equal(t, 1, len(engine.batches))
		assert.Equal(t, batch, engine.batches[1])
		
		// Orders should match settled orders
		assert.Equal(t, uint64(1), batch.Orders[0].ID)
		assert.Equal(t, uint64(2), batch.Orders[1].ID)
	})

	t.Run("ExecuteBatch", func(t *testing.T) {
		engine := NewSettlementEngine(10, 30*time.Second)
		
		// Create a batch first
		settledOrders := []*SettledOrder{
			{OrderID: 100, TxHash: "0xhash100", BlockNumber: 2000},
		}
		batch := engine.CreateBatch(settledOrders)
		batchID := batch.BatchID
		
		// Execute the batch
		err := engine.ExecuteBatch(batchID)
		assert.NoError(t, err)
		
		// Check batch status was updated
		assert.Equal(t, SettlementCompleted, batch.Status)
		assert.True(t, time.Since(batch.CompletedAt) < time.Second)
		
		// Test non-existent batch
		err = engine.ExecuteBatch(999999)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "batch not found")
	})

	t.Run("createBatch", func(t *testing.T) {
		engine := NewSettlementEngine(5, 30*time.Second)
		
		// Add orders to pending list
		orders := []*Order{
			{ID: 10, Type: Limit, Side: Buy},
			{ID: 11, Type: Limit, Side: Sell},
		}
		engine.pendingOrders = orders
		
		batch := engine.createBatch()
		
		assert.NotNil(t, batch)
		assert.True(t, batch.BatchID > 0)
		assert.Equal(t, 2, len(batch.Orders))
		assert.Equal(t, SettlementPending, batch.Status)
		assert.True(t, time.Since(batch.Timestamp) < time.Second)
		
		// Pending orders should be cleared
		assert.Equal(t, 0, len(engine.pendingOrders))
		
		// Batch should be stored
		assert.Contains(t, engine.batches, batch.BatchID)
	})
}

// Test ClearingEngine functions
func TestClearingEngineFunctions(t *testing.T) {
	t.Run("NewClearingEngine", func(t *testing.T) {
		engine := NewClearingEngine()
		
		assert.NotNil(t, engine)
		assert.NotNil(t, engine.positions)
		assert.NotNil(t, engine.balances)
		assert.NotNil(t, engine.marginRequirements)
		assert.Equal(t, 0, len(engine.positions))
		assert.Equal(t, 0, len(engine.balances))
		assert.Equal(t, 0, len(engine.marginRequirements))
	})

	t.Run("UpdatePosition", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		symbol := "BTC-USDT"
		
		// First position update
		engine.UpdatePosition(user, symbol, 1.0, 50000.0)
		
		pos := engine.GetPosition(user, symbol)
		assert.NotNil(t, pos)
		assert.Equal(t, symbol, pos.Symbol)
		assert.Equal(t, user, pos.User)
		assert.Equal(t, 1.0, pos.Size)
		assert.Equal(t, 50000.0, pos.EntryPrice)
		assert.Equal(t, 50000.0, pos.MarkPrice)
		assert.Equal(t, 0.0, pos.UnrealizedPnL) // Same price
		
		// Update position with new size and price
		engine.UpdatePosition(user, symbol, 1.0, 51000.0)
		
		pos = engine.GetPosition(user, symbol)
		assert.Equal(t, 2.0, pos.Size)
		assert.Equal(t, 50500.0, pos.EntryPrice) // Average: (50000*1 + 51000*1) / 2
		assert.Equal(t, 51000.0, pos.MarkPrice)
		assert.Equal(t, 1000.0, pos.UnrealizedPnL) // (51000 - 50500) * 2
	})

	t.Run("GetPosition", func(t *testing.T) {
		engine := NewClearingEngine()
		
		// Non-existent position
		pos := engine.GetPosition("trader1", "BTC-USDT")
		assert.Nil(t, pos)
		
		// Create position
		engine.UpdatePosition("trader1", "BTC-USDT", 1.5, 48000.0)
		
		pos = engine.GetPosition("trader1", "BTC-USDT")
		assert.NotNil(t, pos)
		assert.Equal(t, "trader1", pos.User)
		assert.Equal(t, "BTC-USDT", pos.Symbol)
		assert.Equal(t, 1.5, pos.Size)
		assert.Equal(t, 48000.0, pos.EntryPrice)
		
		// Different symbol should return nil
		pos = engine.GetPosition("trader1", "ETH-USDT")
		assert.Nil(t, pos)
	})

	t.Run("UpdateBalanceWithLocked", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		available := big.NewInt(1000000)
		locked := big.NewInt(50000)
		
		engine.UpdateBalanceWithLocked(user, available, locked)
		
		balance := engine.GetBalance(user)
		assert.NotNil(t, balance)
		assert.Equal(t, user, balance.User)
		assert.Equal(t, available, balance.Available)
		assert.Equal(t, locked, balance.Locked)
		assert.Equal(t, big.NewInt(1050000), balance.Total) // available + locked
		assert.True(t, time.Since(balance.LastUpdate) < time.Second)
		
		// Update with new amounts
		newAvailable := big.NewInt(800000)
		newLocked := big.NewInt(100000)
		
		engine.UpdateBalanceWithLocked(user, newAvailable, newLocked)
		
		balance = engine.GetBalance(user)
		assert.Equal(t, newAvailable, balance.Available)
		assert.Equal(t, newLocked, balance.Locked)
		assert.Equal(t, big.NewInt(900000), balance.Total)
		
		// Update with nil values (should not change)
		engine.UpdateBalanceWithLocked(user, nil, nil)
		balance = engine.GetBalance(user)
		assert.Equal(t, newAvailable, balance.Available)
		assert.Equal(t, newLocked, balance.Locked)
	})

	t.Run("GetBalance", func(t *testing.T) {
		engine := NewClearingEngine()
		
		// Non-existent balance
		balance := engine.GetBalance("trader1")
		assert.Nil(t, balance)
		
		// Create balance
		engine.UpdateBalanceWithLocked("trader1", big.NewInt(500000), big.NewInt(25000))
		
		balance = engine.GetBalance("trader1")
		assert.NotNil(t, balance)
		assert.Equal(t, "trader1", balance.User)
		assert.Equal(t, big.NewInt(500000), balance.Available)
		assert.Equal(t, big.NewInt(25000), balance.Locked)
		assert.Equal(t, big.NewInt(525000), balance.Total)
	})

	t.Run("UpdateBalance", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// First update creates new balance
		amount := big.NewInt(100000)
		engine.UpdateBalance(user, amount)
		
		balance := engine.GetBalance(user)
		assert.NotNil(t, balance)
		assert.Equal(t, amount, balance.Available)
		assert.Equal(t, big.NewInt(0), balance.Locked)
		assert.Equal(t, amount, balance.Total)
		
		// Second update adds to existing balance
		additionalAmount := big.NewInt(50000)
		engine.UpdateBalance(user, additionalAmount)
		
		balance = engine.GetBalance(user)
		assert.Equal(t, big.NewInt(150000), balance.Available)
		assert.Equal(t, big.NewInt(0), balance.Locked)
		assert.Equal(t, big.NewInt(150000), balance.Total)
		
		// Negative amount (withdrawal)
		withdrawAmount := big.NewInt(-25000)
		engine.UpdateBalance(user, withdrawAmount)
		
		balance = engine.GetBalance(user)
		assert.Equal(t, big.NewInt(125000), balance.Available)
		assert.Equal(t, big.NewInt(125000), balance.Total)
	})

	t.Run("CalculateMarginRequirement", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// No positions - should have zero margin requirement
		req := engine.CalculateMarginRequirement(user)
		assert.NotNil(t, req)
		assert.Equal(t, user, req.User)
		assert.Equal(t, big.NewInt(0), req.InitialMargin)
		assert.Equal(t, big.NewInt(0), req.MaintenanceMargin)
		assert.True(t, time.Since(req.LastCalculation) < time.Second)
		
		// Add position
		engine.UpdatePosition(user, "BTC-USDT", 1.0, 50000.0)
		
		req = engine.CalculateMarginRequirement(user)
		assert.Equal(t, big.NewInt(5000), req.InitialMargin)     // 10% of 50000
		assert.Equal(t, big.NewInt(2500), req.MaintenanceMargin) // 5% of 50000
		
		// Add balance and check margin ratio
		engine.UpdateBalance(user, big.NewInt(10000))
		
		req = engine.CalculateMarginRequirement(user)
		assert.Equal(t, 2.0, req.MarginRatio) // 10000 / 5000
		
		// Multiple positions
		engine.UpdatePosition(user, "ETH-USDT", 10.0, 3000.0)
		
		req = engine.CalculateMarginRequirement(user)
		// BTC: 50000 * 1.0 = 50000 -> initial: 5000, maintenance: 2500
		// ETH: 3000 * 10.0 = 30000 -> initial: 3000, maintenance: 1500
		// Total: initial: 8000, maintenance: 4000
		assert.Equal(t, big.NewInt(8000), req.InitialMargin)
		assert.Equal(t, big.NewInt(4000), req.MaintenanceMargin)
	})

	t.Run("CheckMargin", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// No position or balance - should return false
		assert.False(t, engine.CheckMargin(user))
		
		// Add position but no balance
		engine.UpdatePosition(user, "BTC-USDT", 1.0, 50000.0)
		assert.False(t, engine.CheckMargin(user))
		
		// Add insufficient balance
		engine.UpdateBalance(user, big.NewInt(3000))
		assert.False(t, engine.CheckMargin(user)) // Need 5000 for initial margin
		
		// Add sufficient balance
		engine.UpdateBalance(user, big.NewInt(7000)) // Total: 10000
		assert.True(t, engine.CheckMargin(user)) // Have 10000, need 5000
	})

	t.Run("CheckLiquidation", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// No data - should return false
		assert.False(t, engine.CheckLiquidation(user))
		
		// Add position and balance
		engine.UpdatePosition(user, "BTC-USDT", 1.0, 50000.0)
		engine.UpdateBalance(user, big.NewInt(10000))
		
		// Calculate margin requirement to populate cache
		engine.CalculateMarginRequirement(user)
		
		// Should not need liquidation (have 10000, need 2500 maintenance)
		assert.False(t, engine.CheckLiquidation(user))
		
		// Reduce balance below maintenance margin
		engine.UpdateBalance(user, big.NewInt(-8000)) // Total: 2000, need 2500
		assert.True(t, engine.CheckLiquidation(user))
	})

	t.Run("Liquidate", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// No data - should return false
		assert.False(t, engine.Liquidate(user))
		
		// Add position and balance
		engine.UpdatePosition(user, "BTC-USDT", 1.0, 50000.0)
		engine.UpdateBalance(user, big.NewInt(10000))
		
		// Calculate margin requirement
		engine.CalculateMarginRequirement(user)
		
		// Should not liquidate (sufficient margin)
		assert.False(t, engine.Liquidate(user))
		
		// Reduce balance below maintenance margin
		engine.UpdateBalance(user, big.NewInt(-8000)) // Total: 2000, need 2500
		
		// Should liquidate
		assert.True(t, engine.Liquidate(user))
		
		// Positions should be cleared
		pos := engine.GetPosition(user, "BTC-USDT")
		assert.Nil(t, pos)
		
		// Margin requirements should be reset
		req := engine.marginRequirements[user]
		assert.NotNil(t, req)
		assert.Equal(t, big.NewInt(0), req.InitialMargin)
		assert.Equal(t, big.NewInt(0), req.MaintenanceMargin)
		assert.Equal(t, 0.0, req.MarginRatio)
	})
}

// Test utility functions
func TestUtilityFunctions(t *testing.T) {
	t.Run("NewGasEstimator", func(t *testing.T) {
		estimator := NewGasEstimator()
		
		assert.NotNil(t, estimator)
		assert.Equal(t, uint64(21000), estimator.baseGas)
		assert.Equal(t, uint64(50000), estimator.perOrderGas)
		assert.Equal(t, big.NewInt(100000000000), estimator.maxGasPrice) // 100 gwei
		assert.Equal(t, big.NewInt(2000000000), estimator.priorityFee)   // 2 gwei
	})

	t.Run("EstimateGas", func(t *testing.T) {
		estimator := NewGasEstimator()
		
		// Single order
		gas := estimator.EstimateGas(1)
		assert.Equal(t, uint64(71000), gas) // 21000 + 1*50000
		
		// Multiple orders
		gas = estimator.EstimateGas(5)
		assert.Equal(t, uint64(271000), gas) // 21000 + 5*50000
		
		// Zero orders
		gas = estimator.EstimateGas(0)
		assert.Equal(t, uint64(21000), gas) // just base gas
	})

	t.Run("NewNonceManager", func(t *testing.T) {
		startNonce := uint64(100)
		manager := NewNonceManager(startNonce)
		
		assert.NotNil(t, manager)
		assert.Equal(t, startNonce, manager.currentNonce)
		assert.NotNil(t, manager.pendingNonces)
		assert.Equal(t, 0, len(manager.pendingNonces))
	})

	t.Run("GetNextNonce", func(t *testing.T) {
		manager := NewNonceManager(200)
		
		// First nonce
		nonce1 := manager.GetNextNonce()
		assert.Equal(t, uint64(200), nonce1)
		assert.Equal(t, uint64(201), manager.currentNonce)
		assert.True(t, manager.pendingNonces[200])
		
		// Second nonce
		nonce2 := manager.GetNextNonce()
		assert.Equal(t, uint64(201), nonce2)
		assert.Equal(t, uint64(202), manager.currentNonce)
		assert.True(t, manager.pendingNonces[201])
		
		// Both nonces should be pending
		assert.Equal(t, 2, len(manager.pendingNonces))
	})

	t.Run("ConfirmNonce", func(t *testing.T) {
		manager := NewNonceManager(300)
		
		// Get some nonces
		nonce1 := manager.GetNextNonce()
		nonce2 := manager.GetNextNonce()
		
		assert.Equal(t, 2, len(manager.pendingNonces))
		
		// Confirm first nonce
		manager.ConfirmNonce(nonce1)
		assert.False(t, manager.pendingNonces[nonce1])
		assert.True(t, manager.pendingNonces[nonce2])
		assert.Equal(t, 1, len(manager.pendingNonces))
		
		// Confirm second nonce
		manager.ConfirmNonce(nonce2)
		assert.Equal(t, 0, len(manager.pendingNonces))
	})

	t.Run("NewAccessControl", func(t *testing.T) {
		ac := NewAccessControl()
		
		assert.NotNil(t, ac)
		assert.NotNil(t, ac.admins)
		assert.NotNil(t, ac.operators)
		assert.NotNil(t, ac.blacklist)
		assert.Equal(t, 0, len(ac.admins))
		assert.Equal(t, 0, len(ac.operators))
		assert.Equal(t, 0, len(ac.blacklist))
	})

	t.Run("AccessControlChecks", func(t *testing.T) {
		ac := NewAccessControl()
		address := "0x1234567890123456789012345678901234567890"
		
		// Initially should not be admin, operator, or blacklisted
		assert.False(t, ac.IsAdmin(address))
		assert.False(t, ac.IsOperator(address))
		assert.False(t, ac.IsBlacklisted(address))
		
		// Add to admin list manually for testing
		ac.admins[address] = true
		assert.True(t, ac.IsAdmin(address))
		
		// Add to operator list manually for testing
		ac.operators[address] = true
		assert.True(t, ac.IsOperator(address))
		
		// Add to blacklist manually for testing
		ac.blacklist[address] = true
		assert.True(t, ac.IsBlacklisted(address))
	})

	t.Run("NewRecoveryManager", func(t *testing.T) {
		maxRetries := 3
		retryDelay := 5 * time.Second
		
		manager := NewRecoveryManager(maxRetries, retryDelay)
		
		assert.NotNil(t, manager)
		assert.NotNil(t, manager.failedBatches)
		assert.NotNil(t, manager.retryAttempts)
		assert.Equal(t, maxRetries, manager.maxRetries)
		assert.Equal(t, retryDelay, manager.retryDelay)
		assert.Equal(t, 0, len(manager.failedBatches))
		assert.Equal(t, 0, len(manager.retryAttempts))
	})

	t.Run("AddFailedBatch", func(t *testing.T) {
		manager := NewRecoveryManager(2, 100*time.Millisecond)
		
		batch := &SettlementBatch{
			BatchID: 12345,
			Orders:  []*Order{{ID: 1}},
			Status:  SettlementFailed,
		}
		
		// First attempt should succeed
		result := manager.AddFailedBatch(batch)
		assert.True(t, result)
		assert.Equal(t, 1, manager.retryAttempts[batch.BatchID])
		
		// Second attempt should succeed
		result = manager.AddFailedBatch(batch)
		assert.True(t, result)
		assert.Equal(t, 2, manager.retryAttempts[batch.BatchID])
		
		// Third attempt should fail (exceeded max retries)
		result = manager.AddFailedBatch(batch)
		assert.False(t, result)
		assert.Equal(t, 2, manager.retryAttempts[batch.BatchID]) // Unchanged
		
		// Wait for retry to complete
		time.Sleep(200 * time.Millisecond)
		
		// Batch should be removed from failed batches after retry
		assert.Equal(t, 0, len(manager.failedBatches))
	})

	t.Run("scheduleRetry", func(t *testing.T) {
		manager := NewRecoveryManager(3, 50*time.Millisecond)
		
		batch := &SettlementBatch{
			BatchID: 67890,
			Orders:  []*Order{{ID: 2}},
			Status:  SettlementFailed,
		}
		
		// Add batch manually
		manager.failedBatches[batch.BatchID] = batch
		
		// Schedule retry
		go manager.scheduleRetry(batch.BatchID)
		
		// Should still exist initially
		assert.Equal(t, 1, len(manager.failedBatches))
		
		// Wait for retry delay
		time.Sleep(100 * time.Millisecond)
		
		// Batch should be removed after retry
		assert.Equal(t, 0, len(manager.failedBatches))
	})
}

// Test complex scenarios
func TestXChainIntegrationScenarios(t *testing.T) {
	t.Run("CompleteSettlementFlow", func(t *testing.T) {
		// Create integration and settlement engine
		integration := NewXChainIntegration(1, "0x123")
		engine := NewSettlementEngine(2, 10*time.Second)
		
		// Add orders to settlement engine
		orders := []*Order{
			{ID: 1, Type: Limit, Side: Buy, Size: 1.0, Price: 50000, User: "user1"},
			{ID: 2, Type: Limit, Side: Sell, Size: 1.0, Price: 50100, User: "user2"},
		}
		
		for _, order := range orders {
			engine.AddOrder(order)
		}
		
		// Should have created a batch
		assert.Equal(t, 1, len(engine.batches))
		assert.Equal(t, 0, len(engine.pendingOrders))
		
		// Get the batch and submit to integration
		var batch *SettlementBatch
		for _, b := range engine.batches {
			batch = b
			break
		}
		
		err := integration.SubmitBatch(batch)
		assert.NoError(t, err)
		
		// Wait for settlement
		time.Sleep(400 * time.Millisecond)
		
		// Check final state
		stats := integration.GetStats()
		assert.Equal(t, uint64(2), stats.TotalSettled)
		assert.Equal(t, uint64(0), stats.TotalPending)
	})

	t.Run("ClearingEngineMarginFlow", func(t *testing.T) {
		engine := NewClearingEngine()
		user := "trader1"
		
		// 1. User deposits funds
		engine.UpdateBalance(user, big.NewInt(100000)) // $1000
		
		// 2. User opens position
		engine.UpdatePosition(user, "BTC-USDT", 0.1, 50000.0) // 0.1 BTC at $50k
		
		// 3. Check initial margin
		assert.True(t, engine.CheckMargin(user))
		
		// 4. Market moves against user (simulate mark price change)
		engine.UpdatePosition(user, "BTC-USDT", 0.0, 45000.0) // Update mark price
		
		// 5. Check if liquidation needed
		shouldLiquidate := engine.CheckLiquidation(user)
		
		// 6. If needed, liquidate
		if shouldLiquidate {
			liquidated := engine.Liquidate(user)
			assert.True(t, liquidated)
			
			// Position should be cleared
			pos := engine.GetPosition(user, "BTC-USDT")
			assert.Nil(t, pos)
		}
		
		// Basic validation that the flow completed
		balance := engine.GetBalance(user)
		assert.NotNil(t, balance)
	})

	t.Run("RecoveryFlow", func(t *testing.T) {
		manager := NewRecoveryManager(2, 50*time.Millisecond)
		
		// Create multiple failed batches
		batches := []*SettlementBatch{
			{BatchID: 1, Orders: []*Order{{ID: 1}}, Status: SettlementFailed},
			{BatchID: 2, Orders: []*Order{{ID: 2}}, Status: SettlementFailed},
		}
		
		// Add first batch - should succeed
		result1 := manager.AddFailedBatch(batches[0])
		assert.True(t, result1)
		
		// Add second batch - should succeed
		result2 := manager.AddFailedBatch(batches[1])
		assert.True(t, result2)
		
		// Re-add first batch - should succeed (second attempt)
		result3 := manager.AddFailedBatch(batches[0])
		assert.True(t, result3)
		
		// Re-add first batch again - should fail (exceeded max retries)
		result4 := manager.AddFailedBatch(batches[0])
		assert.False(t, result4)
		
		// Wait for retries to complete
		time.Sleep(200 * time.Millisecond)
		
		// All retried batches should be removed
		assert.Equal(t, 0, len(manager.failedBatches))
	})
}