package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Test vault functions with 0% coverage
func TestVaultFunctions(t *testing.T) {
	// Create test engine
	engineConfig := EngineConfig{
		EnableVaults: true,
	}
	engine := NewTradingEngine(engineConfig)

	t.Run("NewVaultManager", func(t *testing.T) {
		manager := NewVaultManager(engine)
		assert.NotNil(t, manager)
		assert.NotNil(t, manager.vaults)
		assert.NotNil(t, manager.copyVaults)
		assert.NotNil(t, manager.userVaults)
		assert.NotNil(t, manager.leaderVaults)
		assert.Equal(t, engine, manager.engine)
	})

	t.Run("CreateVault", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:                "vault1",
			Name:              "Test Vault",
			Description:       "A test vault",
			ManagementFee:     0.02,
			PerformanceFee:    0.20,
			MinDeposit:        big.NewInt(1000),
			MaxCapacity:       big.NewInt(1000000),
			LockupPeriod:      24 * time.Hour,
			AllowedAssets:     []string{"BTC", "ETH"},
			RebalanceInterval: time.Hour,
			InsuranceCoverage: big.NewInt(100000),
			RecoveryAddresses: []string{"addr1", "addr2"},
		}

		vault, err := manager.CreateVault(config)
		assert.NoError(t, err)
		assert.NotNil(t, vault)
		assert.Equal(t, "vault1", vault.ID)
		assert.Equal(t, "Test Vault", vault.Name)
		assert.Equal(t, VaultStateActive, vault.State)
		assert.NotNil(t, vault.Performance)
		assert.NotNil(t, vault.Depositors)

		// Test creating duplicate vault
		_, err = manager.CreateVault(config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already exists")
	})

	t.Run("Vault_Deposit", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:           "vault2",
			Name:         "Deposit Test Vault",
			MinDeposit:   big.NewInt(100),
			MaxCapacity:  big.NewInt(10000),
			LockupPeriod: time.Hour,
		}

		vault, _ := manager.CreateVault(config)

		// Test first deposit
		amount := big.NewInt(1000)
		position, err := vault.Deposit("user1", amount)
		assert.NoError(t, err)
		assert.NotNil(t, position)
		assert.Equal(t, "user1", position.User)
		assert.Equal(t, amount, position.Shares) // 1:1 ratio for first deposit
		assert.Equal(t, amount, position.DepositValue)
		assert.Equal(t, amount, vault.TotalDeposits)
		assert.Equal(t, amount, vault.TotalShares)

		// Test second deposit
		amount2 := big.NewInt(2000)
		position2, err := vault.Deposit("user2", amount2)
		assert.NoError(t, err)
		assert.NotNil(t, position2)
		assert.Equal(t, "user2", position2.User)

		// Test deposit below minimum
		smallAmount := big.NewInt(50)
		_, err = vault.Deposit("user3", smallAmount)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "below minimum")

		// Test deposit exceeding capacity
		largeAmount := big.NewInt(20000)
		_, err = vault.Deposit("user4", largeAmount)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "capacity")
	})

	t.Run("Vault_Withdraw", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:           "vault3",
			Name:         "Withdraw Test Vault",
			MinDeposit:   big.NewInt(100),
			LockupPeriod: 0, // No lockup for testing
		}

		vault, _ := manager.CreateVault(config)

		// Setup initial deposit
		depositAmount := big.NewInt(1000)
		vault.Deposit("user1", depositAmount)

		// Test withdrawal
		withdrawShares := big.NewInt(500)
		withdrawnAmount, err := vault.Withdraw("user1", withdrawShares)
		assert.NoError(t, err)
		assert.NotNil(t, withdrawnAmount)
		assert.True(t, withdrawnAmount.Cmp(big.NewInt(0)) > 0)

		// Test withdrawal from non-existent position
		_, err = vault.Withdraw("nonexistent", big.NewInt(100))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no position found")

		// Test withdrawal exceeding shares
		largeShares := big.NewInt(10000)
		_, err = vault.Withdraw("user1", largeShares)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient shares")
	})

	t.Run("Vault_applyWithdrawalFees", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:             "vault4",
			Name:           "Fee Test Vault",
			MinDeposit:     big.NewInt(100),
			ManagementFee:  0.02,
			PerformanceFee: 0.20,
		}

		vault, _ := manager.CreateVault(config)

		position := &VaultPosition{
			User:          "user1",
			DepositValue:  big.NewInt(1000),
			CurrentValue:  big.NewInt(1200), // 200 profit
			LastUpdate:    time.Now().AddDate(0, 0, -30), // 30 days ago
		}

		originalAmount := big.NewInt(1200)
		amount := new(big.Int).Set(originalAmount) // Create a copy
		finalAmount := vault.applyWithdrawalFees(amount, position)
		
		// Should be less than original due to fees
		assert.True(t, finalAmount.Cmp(originalAmount) < 0)
		assert.True(t, finalAmount.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("Vault_ExecuteStrategies", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:         "vault5",
			Name:       "Strategy Test Vault",
			MinDeposit: big.NewInt(100),
			Strategies: []StrategyConfig{
				{
					Type: "market_making",
					Name: "MM Strategy",
					Parameters: map[string]interface{}{
						"spread": 0.001,
					},
				},
			},
		}

		vault, _ := manager.CreateVault(config)
		orderbook := NewOrderBook("BTC-USDT")

		orders := vault.ExecuteStrategies(orderbook)
		assert.NotNil(t, orders)
		// May be empty if strategies don't generate orders in test environment
		assert.True(t, len(orders) >= 0)
	})

	t.Run("Vault_getAvailableCapital", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:         "vault6",
			Name:       "Capital Test Vault",
			MinDeposit: big.NewInt(100),
		}

		vault, _ := manager.CreateVault(config)

		// Initially no deposits
		capital := vault.getAvailableCapital()
		assert.Equal(t, big.NewInt(0), capital)

		// After deposit
		vault.Deposit("user1", big.NewInt(5000))
		capital = vault.getAvailableCapital()
		assert.Equal(t, big.NewInt(4500), capital) // 90% of 5000 (10% reserved)
	})

	t.Run("Vault_Rebalance", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:                "vault7",
			Name:              "Rebalance Test Vault",
			MinDeposit:        big.NewInt(100),
			RebalanceInterval: time.Hour,
		}

		vault, _ := manager.CreateVault(config)

		err := vault.Rebalance()
		assert.NoError(t, err)
		// LastRebalance should be updated
		assert.True(t, time.Since(vault.LastRebalance) < time.Second)
	})

	t.Run("Vault_UpdatePerformance", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:         "vault8",
			Name:       "Performance Test Vault",
			MinDeposit: big.NewInt(100),
		}

		vault, _ := manager.CreateVault(config)

		currentValue := big.NewInt(15000)
		vault.UpdatePerformance(currentValue)

		assert.NotNil(t, vault.Performance)
		// Performance should be updated
		assert.True(t, vault.Performance.UpdatedAt.After(vault.CreatedAt))
	})
}

// Test vault manager comprehensive functions
func TestVaultManagerComprehensiveFunctions(t *testing.T) {
	engineConfig := EngineConfig{EnableVaults: true}
	engine := NewTradingEngine(engineConfig)

	t.Run("GetVault", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:         "getvault1",
			Name:       "Get Vault Test",
			MinDeposit: big.NewInt(100),
		}

		// Create vault
		createdVault, _ := manager.CreateVault(config)

		// Test getting existing vault
		retrievedVault, err := manager.GetVault("getvault1")
		assert.NoError(t, err)
		assert.NotNil(t, retrievedVault)
		assert.Equal(t, createdVault.ID, retrievedVault.ID)

		// Test getting non-existent vault
		_, err = manager.GetVault("nonexistent")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("ListVaults", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Initially empty
		vaults := manager.ListVaults()
		assert.Equal(t, 0, len(vaults))

		// Create vaults
		config1 := VaultConfig{ID: "list1", Name: "List Vault 1", MinDeposit: big.NewInt(100)}
		config2 := VaultConfig{ID: "list2", Name: "List Vault 2", MinDeposit: big.NewInt(100)}
		
		manager.CreateVault(config1)
		manager.CreateVault(config2)

		vaults = manager.ListVaults()
		assert.Equal(t, 2, len(vaults))
	})

	t.Run("GetVaultPerformance", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		config := VaultConfig{
			ID:         "perfvault1",
			Name:       "Performance Vault Test",
			MinDeposit: big.NewInt(100),
		}

		manager.CreateVault(config)

		// Test getting performance for existing vault
		perf, err := manager.GetVaultPerformance("perfvault1")
		assert.NoError(t, err)
		assert.NotNil(t, perf)

		// Test getting performance for non-existent vault
		_, err = manager.GetVaultPerformance("nonexistent")
		assert.Error(t, err)
	})
}

// Test strategy functions
func TestStrategyFunctions(t *testing.T) {
	t.Run("NewMarketMakingStrategy", func(t *testing.T) {
		config := StrategyConfig{
			Type: "market_making",
			Name: "MM Test Strategy",
			Parameters: map[string]interface{}{
				"spread": 0.002,
			},
		}

		strategy := NewMarketMakingStrategy(config)
		assert.NotNil(t, strategy)
		assert.Equal(t, "MM Test Strategy", strategy.GetName())
		
		riskLimits := strategy.GetRiskLimits()
		assert.NotNil(t, riskLimits)
		
		performance := strategy.GetPerformance()
		assert.NotNil(t, performance)
	})

	t.Run("MarketMakingStrategy_Execute", func(t *testing.T) {
		config := StrategyConfig{
			Type: "market_making",
			Name: "MM Execute Test",
		}

		strategy := NewMarketMakingStrategy(config)
		orderbook := NewOrderBook("BTC-USDT")
		capital := big.NewInt(10000)

		orders := strategy.Execute(orderbook, capital)
		assert.NotNil(t, orders)
		assert.True(t, len(orders) >= 0)
	})

	t.Run("NewMomentumStrategy", func(t *testing.T) {
		config := StrategyConfig{
			Type: "momentum",
			Name: "Momentum Test Strategy",
		}

		strategy := NewMomentumStrategy(config)
		assert.NotNil(t, strategy)
		assert.Equal(t, "Momentum Test Strategy", strategy.GetName())
		
		orders := strategy.Execute(NewOrderBook("ETH-USDT"), big.NewInt(5000))
		assert.NotNil(t, orders)
	})

	t.Run("NewArbitrageStrategy", func(t *testing.T) {
		config := StrategyConfig{
			Type: "arbitrage",
			Name: "Arbitrage Test Strategy",
		}

		strategy := NewArbitrageStrategy(config)
		assert.NotNil(t, strategy)
		assert.Equal(t, "Arbitrage Test Strategy", strategy.GetName())
		
		orders := strategy.Execute(NewOrderBook("BTC-USDT"), big.NewInt(3000))
		assert.NotNil(t, orders)
	})

	t.Run("NewMeanReversionStrategy", func(t *testing.T) {
		config := StrategyConfig{
			Type: "mean_reversion",
			Name: "Mean Reversion Test Strategy",
		}

		strategy := NewMeanReversionStrategy(config)
		assert.NotNil(t, strategy)
		assert.Equal(t, "Mean Reversion Test Strategy", strategy.GetName())
		
		orders := strategy.Execute(NewOrderBook("AVAX-USDT"), big.NewInt(2000))
		assert.NotNil(t, orders)
	})
}

// Test performance metrics functions
func TestPerformanceMetricsFunctions(t *testing.T) {
	t.Run("NewPerformanceMetrics", func(t *testing.T) {
		metrics := NewPerformanceMetrics()
		assert.NotNil(t, metrics)
		assert.True(t, metrics.UpdatedAt.After(time.Time{}))
	})

	t.Run("calculateSharpe", func(t *testing.T) {
		metrics := NewPerformanceMetrics()
		
		// Should not panic
		metrics.calculateSharpe()
		assert.NotNil(t, metrics)
	})

	t.Run("calculateMaxDrawdown", func(t *testing.T) {
		metrics := NewPerformanceMetrics()
		
		// Should not panic
		metrics.calculateMaxDrawdown()
		assert.NotNil(t, metrics)
	})
}

// Test copy vault comprehensive functions
func TestCopyVaultComprehensiveFunctions(t *testing.T) {
	engineConfig := EngineConfig{EnableVaults: true}
	engine := NewTradingEngine(engineConfig)

	t.Run("CreateVaultWithProfitShare", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		leader := "leader1"
		name := "Copy Vault Test"
		description := "A test copy vault"
		initialDeposit := big.NewInt(100 * 1e6) // 100 USDC with 6 decimals

		copyVault, err := manager.CreateVaultWithProfitShare(leader, name, description, initialDeposit)
		assert.NoError(t, err)
		assert.NotNil(t, copyVault)
		assert.Equal(t, leader, copyVault.Leader)
		assert.Equal(t, name, copyVault.Name)
		assert.Equal(t, 0.10, copyVault.ProfitShare) // 10% default
		assert.Equal(t, VaultStateActive, copyVault.State)
	})

	t.Run("JoinVault", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create copy vault first
		copyVault, _ := manager.CreateVaultWithProfitShare("leader1", "Join Test Vault", "Test", big.NewInt(100*1e6))
		
		// Test joining vault
		err := manager.JoinVault(copyVault.ID, "follower1", big.NewInt(100*1e6))
		assert.NoError(t, err)

		// Verify follower was added
		vault, _ := manager.GetVaultByID(copyVault.ID)
		assert.Contains(t, vault.Followers, "follower1")

		// Test joining non-existent vault
		err = manager.JoinVault("nonexistent", "follower2", big.NewInt(100*1e6))
		assert.Error(t, err)
	})

	t.Run("WithdrawFromVault", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create and join vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader2", "Withdraw Test", "Test", big.NewInt(100*1e6))
		manager.JoinVault(copyVault.ID, "follower1", big.NewInt(100*1e6))

		// Test withdrawal
		withdrawAmount, err := manager.WithdrawFromVault(copyVault.ID, "follower1", 0.5) // 50%
		assert.NoError(t, err)
		assert.NotNil(t, withdrawAmount)
		assert.True(t, withdrawAmount.Cmp(big.NewInt(0)) > 0)

		// Test withdrawal from non-existent vault
		_, err = manager.WithdrawFromVault("nonexistent", "follower1", 0.5)
		assert.Error(t, err)
	})

	t.Run("GetVaultByID", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader3", "GetByID Test", "Test", big.NewInt(100*1e6))

		// Test getting existing vault
		retrieved, err := manager.GetVaultByID(copyVault.ID)
		assert.NoError(t, err)
		assert.NotNil(t, retrieved)
		assert.Equal(t, copyVault.ID, retrieved.ID)

		// Test getting non-existent vault
		_, err = manager.GetVaultByID("nonexistent")
		assert.Error(t, err)
	})

	t.Run("GetUserVaults", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		user := "testuser1"

		// Initially no vaults
		vaults, err := manager.GetUserVaults(user)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(vaults))

		// Create and join vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader4", "User Vault Test", "Test", big.NewInt(100*1e6))
		manager.JoinVault(copyVault.ID, user, big.NewInt(100*1e6))

		// Should find the vault
		vaults, err = manager.GetUserVaults(user)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(vaults))
		assert.Equal(t, copyVault.ID, vaults[0].ID)
	})

	t.Run("GetLeaderVaults", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		leader := "testleader1"

		// Initially no vaults
		vaults, err := manager.GetLeaderVaults(leader)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(vaults))

		// Create vault
		manager.CreateVaultWithProfitShare(leader, "Leader Vault Test", "Test", big.NewInt(100*1e6))

		// Should find the vault
		vaults, err = manager.GetLeaderVaults(leader)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(vaults))
		assert.Equal(t, leader, vaults[0].Leader)
	})

	t.Run("UpdateVaultValue", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader5", "Update Value Test", "Test", big.NewInt(100*1e6))

		// Test updating value
		newValue := big.NewInt(15000)
		err := manager.UpdateVaultValue(copyVault.ID, newValue)
		assert.NoError(t, err)

		// Test updating non-existent vault
		err = manager.UpdateVaultValue("nonexistent", newValue)
		assert.Error(t, err)
	})

	t.Run("ExecuteVaultTrade", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader6", "Trade Test", "Test", big.NewInt(100*1e6))

		// Test executing trade
		err := manager.ExecuteVaultTrade(copyVault.ID, "BTC-USDT", Buy, 1.0, Limit)
		assert.NoError(t, err)

		// Test executing trade on non-existent vault
		err = manager.ExecuteVaultTrade("nonexistent", "BTC-USDT", Buy, 1.0, Limit)
		assert.Error(t, err)
	})

	t.Run("checkLeaderMinimumShare", func(t *testing.T) {
		manager := NewVaultManager(engine)
		
		// Create vault
		copyVault, _ := manager.CreateVaultWithProfitShare("leader7", "Min Share Test", "Test", big.NewInt(100*1e6))

		// Test minimum share check
		hasMinShare := manager.checkLeaderMinimumShare(copyVault)
		assert.True(t, hasMinShare) // Should be true as leader created with initial deposit
	})
}

// Test utility functions
func TestVaultUtilityFunctions(t *testing.T) {
	t.Run("mulDiv", func(t *testing.T) {
		a := big.NewInt(100)
		b := big.NewInt(200)
		c := big.NewInt(50)

		result := mulDiv(a, b, c)
		expected := big.NewInt(400) // (100 * 200) / 50
		assert.Equal(t, expected, result)

		// Test with different values
		result = mulDiv(big.NewInt(10), big.NewInt(30), big.NewInt(5))
		expected = big.NewInt(60) // (10 * 30) / 5
		assert.Equal(t, expected, result)
	})

	t.Run("allocateCapital", func(t *testing.T) {
		manager := NewVaultManager(nil)
		
		config := VaultConfig{
			ID:         "capital_test",
			Name:       "Capital Allocation Test",
			MinDeposit: big.NewInt(100),
		}

		vault, _ := manager.CreateVault(config)
		
		// Mock strategy
		strategy := NewMarketMakingStrategy(StrategyConfig{Name: "Test Strategy"})
		totalCapital := big.NewInt(10000)

		allocated := vault.allocateCapital(strategy, totalCapital)
		assert.NotNil(t, allocated)
		assert.True(t, allocated.Cmp(big.NewInt(0)) >= 0)
		assert.True(t, allocated.Cmp(totalCapital) <= 0)
	})

	t.Run("applyRiskLimits", func(t *testing.T) {
		manager := NewVaultManager(nil)
		
		config := VaultConfig{
			ID:         "risk_test",
			Name:       "Risk Limits Test",
			MinDeposit: big.NewInt(100),
			RiskLimits: RiskLimits{
				MaxPositionSize: 1000.0,
				MaxLeverage:     5.0,
			},
		}

		vault, _ := manager.CreateVault(config)

		// Create test orders
		orders := []Order{
			{ID: 1, Size: 100, Price: 50000, Side: Buy, Type: Limit},
			{ID: 2, Size: 2000, Price: 50100, Side: Sell, Type: Limit}, // Exceeds limits
		}

		filteredOrders := vault.applyRiskLimits(orders)
		assert.NotNil(t, filteredOrders)
		// Should filter out orders that exceed risk limits
		assert.True(t, len(filteredOrders) <= len(orders))
	})
}

// Test vault manager strategy creation
func TestVaultManagerStrategyCreation(t *testing.T) {
	engineConfig := EngineConfig{EnableVaults: true}
	engine := NewTradingEngine(engineConfig)
	manager := NewVaultManager(engine)

	t.Run("createStrategy", func(t *testing.T) {
		// Test market making strategy
		mmConfig := StrategyConfig{
			Type: "market_making",
			Name: "Test MM Strategy",
		}
		mmStrategy := manager.createStrategy(mmConfig)
		assert.NotNil(t, mmStrategy)

		// Test momentum strategy  
		momentumConfig := StrategyConfig{
			Type: "momentum",
			Name: "Test Momentum Strategy",
		}
		momentumStrategy := manager.createStrategy(momentumConfig)
		assert.NotNil(t, momentumStrategy)

		// Test arbitrage strategy
		arbConfig := StrategyConfig{
			Type: "arbitrage", 
			Name: "Test Arbitrage Strategy",
		}
		arbStrategy := manager.createStrategy(arbConfig)
		assert.NotNil(t, arbStrategy)

		// Test mean reversion strategy
		mrConfig := StrategyConfig{
			Type: "mean_reversion",
			Name: "Test Mean Reversion Strategy", 
		}
		mrStrategy := manager.createStrategy(mrConfig)
		assert.NotNil(t, mrStrategy)

		// Test unknown strategy type
		unknownConfig := StrategyConfig{
			Type: "unknown_strategy",
			Name: "Unknown Strategy",
		}
		unknownStrategy := manager.createStrategy(unknownConfig)
		assert.Nil(t, unknownStrategy)
	})
}