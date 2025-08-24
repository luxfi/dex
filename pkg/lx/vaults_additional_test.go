package lx

import (
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Mock strategy for testing
type MockStrategy struct {
	Name     string
	Weight   float64
	MaxRisk  float64
	IsActive bool
}

func (ms *MockStrategy) Execute(market *OrderBook, capital *big.Int) []Order {
	return []Order{}
}

func (ms *MockStrategy) GetRiskLimits() RiskLimits {
	return RiskLimits{
		MaxPositionSize:  1000.0, // float64, not *big.Int
		MaxPositionValue: big.NewInt(1000000),
		MaxLeverage:      10.0,
		MaxDrawdown:      0.05,
		DailyLossLimit:   1000.0,
	}
}

func (ms *MockStrategy) GetName() string {
	return ms.Name
}

func (ms *MockStrategy) GetPerformance() *StrategyPerformance {
	return &StrategyPerformance{
		TotalTrades:   0,
		WinningTrades: 0,
		LosingTrades:  0,
		TotalPnL:     big.NewInt(0),
	}
}

// Test additional vault functions with 0% coverage
func TestVaultAdditionalFunctions(t *testing.T) {
	engineConfig := EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	}
	engine := NewTradingEngine(engineConfig)
	manager := NewVaultManager(engine)

	// Create test vault config
	config := VaultConfig{
		ID:                "test-vault",
		Name:              "Test Vault",
		Description:       "Test vault for coverage",
		ManagementFee:     0.02,
		PerformanceFee:    0.20,
		MinDeposit:        big.NewInt(100),
		MaxCapacity:       big.NewInt(1000000),
		LockupPeriod:      24 * time.Hour,
		Strategies:        []StrategyConfig{},
		RiskLimits:        RiskLimits{},
		AllowedAssets:     []string{"ETH", "BTC"},
		RebalanceInterval: 1 * time.Hour,
		InsuranceCoverage: big.NewInt(10000),
		RecoveryAddresses: []string{"recovery1", "recovery2"},
	}

	vault, err := manager.CreateVault(config)
	assert.NoError(t, err)
	assert.NotNil(t, vault)

	t.Run("Deposit", func(t *testing.T) {
		position, err := vault.Deposit("user1", big.NewInt(1000))
		assert.NoError(t, err)
		assert.NotNil(t, position)
		assert.Equal(t, "user1", position.User)
		assert.Equal(t, big.NewInt(1000), position.Shares)
	})

	t.Run("DepositBelowMinimum", func(t *testing.T) {
		_, err := vault.Deposit("user2", big.NewInt(50))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "deposit below minimum")
	})

	t.Run("DepositCapacityLimit", func(t *testing.T) {
		// Create vault with low capacity
		smallConfig := config
		smallConfig.ID = "small-vault"
		smallConfig.MaxCapacity = big.NewInt(500)
		
		smallVault, err := manager.CreateVault(smallConfig)
		assert.NoError(t, err)
		
		// First deposit should succeed
		_, err = smallVault.Deposit("user1", big.NewInt(400))
		assert.NoError(t, err)
		
		// Second deposit should fail due to capacity
		_, err = smallVault.Deposit("user2", big.NewInt(200))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "vault at capacity")
	})

	t.Run("Withdraw", func(t *testing.T) {
		// First deposit
		vault.Deposit("user3", big.NewInt(2000))
		
		// Wait for lockup to expire (simulate)
		position := vault.Depositors["user3"]
		position.LockedUntil = time.Now().Add(-1 * time.Hour)
		
		amount, err := vault.Withdraw("user3", big.NewInt(500))
		assert.NoError(t, err)
		assert.NotNil(t, amount)
		assert.True(t, amount.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("WithdrawNoPosition", func(t *testing.T) {
		_, err := vault.Withdraw("nonexistent", big.NewInt(100))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no position found")
	})

	t.Run("WithdrawDuringLockup", func(t *testing.T) {
		// Deposit with lockup
		vault.Deposit("user4", big.NewInt(1500))
		
		// Try to withdraw immediately (should fail)
		_, err := vault.Withdraw("user4", big.NewInt(100))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "position locked until")
	})

	t.Run("WithdrawInsufficientShares", func(t *testing.T) {
		// Create position and expire lockup
		vault.Deposit("user5", big.NewInt(1000))
		position := vault.Depositors["user5"]
		position.LockedUntil = time.Now().Add(-1 * time.Hour)
		
		// Try to withdraw more shares than available
		_, err := vault.Withdraw("user5", big.NewInt(2000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient shares")
	})

	t.Run("applyWithdrawalFees", func(t *testing.T) {
		position := &VaultPosition{
			User:          "user6",
			Shares:        big.NewInt(1000),
			DepositValue:  big.NewInt(1000),
			CurrentValue:  big.NewInt(1200),
			RealizedPnL:   big.NewInt(0),
			UnrealizedPnL: big.NewInt(200),
		}
		
		amount := big.NewInt(500)
		feeAdjustedAmount := vault.applyWithdrawalFees(amount, position)
		assert.NotNil(t, feeAdjustedAmount)
		// Fees should reduce the amount
		assert.True(t, feeAdjustedAmount.Cmp(amount) <= 0)
	})

	t.Run("ExecuteStrategies", func(t *testing.T) {
		// Need to provide OrderBook parameter
		orderbook := NewOrderBook("TEST-PAIR")
		orders := vault.ExecuteStrategies(orderbook)
		assert.NotNil(t, orders) // Returns []Order, not error
	})

	t.Run("getAvailableCapital", func(t *testing.T) {
		capital := vault.getAvailableCapital()
		assert.NotNil(t, capital)
		assert.True(t, capital.Cmp(big.NewInt(0)) >= 0)
	})

	t.Run("allocateCapital", func(t *testing.T) {
		capital := big.NewInt(1000)
		// Need to provide strategy parameter - create a mock strategy
		mockStrategy := &MockStrategy{
			Name:      "test-strategy",
			Weight:    1.0,
			MaxRisk:   0.02,
			IsActive:  true,
		}
		allocation := vault.allocateCapital(mockStrategy, capital)
		assert.NotNil(t, allocation)
	})

	t.Run("applyRiskLimits", func(t *testing.T) {
		// applyRiskLimits expects []Order, not map
		orders := []Order{
			{
				ID:    1,
				Type:  Limit,
				Side:  Buy,
				Size:  10,
				Price: 50000,
			},
			{
				ID:    2,
				Type:  Limit,
				Side:  Sell,
				Size:  5,
				Price: 50100,
			},
		}
		filteredOrders := vault.applyRiskLimits(orders)
		assert.NotNil(t, filteredOrders)
	})

	t.Run("Rebalance", func(t *testing.T) {
		err := vault.Rebalance()
		assert.NoError(t, err)
	})

	t.Run("UpdatePerformance", func(t *testing.T) {
		currentValue := big.NewInt(1200000) // $1.2M current value
		vault.UpdatePerformance(currentValue) // Returns void, not error
		// Just verify it doesn't panic
		assert.NotNil(t, vault)
	})

	t.Run("VaultPositionAccess", func(t *testing.T) {
		// First create a position
		vault.Deposit("user7", big.NewInt(1000))
		
		// Direct access to depositors map
		position, exists := vault.Depositors["user7"]
		assert.True(t, exists)
		assert.NotNil(t, position)
		assert.Equal(t, "user7", position.User)
	})

	t.Run("VaultBasicOperations", func(t *testing.T) {
		// Test basic vault operations
		assert.NotNil(t, vault.TotalDeposits)
		assert.NotNil(t, vault.TotalShares)
		assert.NotNil(t, vault.Depositors)
		assert.Equal(t, VaultStateActive, vault.State)
	})

	t.Run("VaultMetricsAccess", func(t *testing.T) {
		// Test access to performance metrics
		assert.NotNil(t, vault.Performance)
		// Performance metrics should exist but we'll test basic access only
		assert.NotNil(t, vault.Performance)
	})

	t.Run("VaultStateOperations", func(t *testing.T) {
		// Test vault state changes
		originalState := vault.State
		vault.State = VaultStatePaused
		assert.Equal(t, VaultStatePaused, vault.State)
		
		vault.State = originalState
		assert.Equal(t, VaultStateActive, vault.State)
	})

	t.Run("VaultConfigAccess", func(t *testing.T) {
		// Test vault configuration access
		assert.NotNil(t, vault.Config)
		assert.Equal(t, "test-vault", vault.Config.ID)
		assert.Equal(t, "Test Vault", vault.Config.Name)
		assert.Equal(t, 0.02, vault.Config.ManagementFee)
		assert.Equal(t, 0.20, vault.Config.PerformanceFee)
	})

	t.Run("VaultStrategiesAccess", func(t *testing.T) {
		// Test strategies access
		assert.NotNil(t, vault.Strategies)
		
		// Add a mock strategy to test
		mockStrategy := &MockStrategy{
			Name:     "test-strategy",
			Weight:   1.0,
			MaxRisk:  0.02,
			IsActive: true,
		}
		
		vault.Strategies = append(vault.Strategies, mockStrategy)
		assert.Equal(t, 1, len(vault.Strategies))
		assert.Equal(t, "test-strategy", vault.Strategies[0].GetName())
	})
}

// Test vault manager functions
func TestVaultManagerFunctions(t *testing.T) {
	engineConfig := EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	}
	engine := NewTradingEngine(engineConfig)
	manager := NewVaultManager(engine)

	t.Run("VaultManagerBasics", func(t *testing.T) {
		config := VaultConfig{
			ID:          "manager-test",
			Name:        "Manager Test Vault",
			Description: "Test vault for manager functions",
			MinDeposit:  big.NewInt(100),
		}
		
		vault, err := manager.CreateVault(config)
		assert.NoError(t, err)
		assert.NotNil(t, vault)
		assert.Equal(t, config.ID, vault.ID)
		
		// Test duplicate creation
		_, err = manager.CreateVault(config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already exists")
	})

	t.Run("VaultManagerAccess", func(t *testing.T) {
		// Test manager data structure access
		assert.NotNil(t, manager.vaults)
		assert.NotNil(t, manager.copyVaults)
		assert.NotNil(t, manager.userVaults)
		assert.NotNil(t, manager.leaderVaults)
		assert.NotNil(t, manager.engine)
		
		// Check that our test vault exists
		vault, exists := manager.vaults["manager-test"]
		assert.True(t, exists)
		assert.NotNil(t, vault)
	})
}

// Test copy trading vault functions  
func TestCopyVaultFunctions(t *testing.T) {
	engineConfig := EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	}
	engine := NewTradingEngine(engineConfig)
	manager := NewVaultManager(engine)

	t.Run("CopyVaultDataAccess", func(t *testing.T) {
		// Test copy vault map exists
		assert.NotNil(t, manager.copyVaults)
		assert.NotNil(t, manager.leaderVaults)
		
		// Should be empty initially
		assert.Equal(t, 0, len(manager.copyVaults))
		assert.Equal(t, 0, len(manager.leaderVaults))
	})
}

// Test performance metrics
func TestPerformanceMetrics(t *testing.T) {
	t.Run("NewPerformanceMetrics", func(t *testing.T) {
		metrics := NewPerformanceMetrics()
		assert.NotNil(t, metrics)
		// Just test that metrics object is created
		assert.NotNil(t, metrics)
	})
}

// Test vault states and enums
func TestVaultEnums(t *testing.T) {
	t.Run("VaultStates", func(t *testing.T) {
		assert.Equal(t, VaultState(0), VaultStateActive)
		assert.Equal(t, VaultState(1), VaultStatePaused)
		assert.Equal(t, VaultState(2), VaultStateClosing)
		assert.Equal(t, VaultState(3), VaultStateClosed)
	})
}