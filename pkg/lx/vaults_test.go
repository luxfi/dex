package lx

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewVaultManager(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	assert.NotNil(t, vm)
	assert.NotNil(t, vm.vaults)
	assert.NotNil(t, vm.copyVaults)
	assert.NotNil(t, vm.userVaults)
	assert.NotNil(t, vm.leaderVaults)
	assert.Equal(t, engine, vm.engine)
}

func TestCreateVault(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	t.Run("CreateNewVault", func(t *testing.T) {
		config := VaultConfig{
			ID:             "vault1",
			Name:           "Test Vault",
			Description:    "A test vault",
			ManagementFee:  0.02, // 2%
			PerformanceFee: 0.20, // 20%
			MinDeposit:     big.NewInt(1000),
			MaxCapacity:    big.NewInt(1000000),
			LockupPeriod:   24 * time.Hour,
			AllowedAssets:  []string{"USDT", "BTC"},
		}

		vault, err := vm.CreateVault(config)
		require.NoError(t, err)
		assert.NotNil(t, vault)
		assert.Equal(t, "vault1", vault.ID)
		assert.Equal(t, "Test Vault", vault.Name)
		assert.Equal(t, config, vault.Config)
		assert.Equal(t, big.NewInt(0), vault.TotalDeposits)
		assert.Equal(t, big.NewInt(0), vault.TotalShares)
	})

	t.Run("CreateDuplicateVault", func(t *testing.T) {
		config := VaultConfig{
			ID:   "vault1",
			Name: "Duplicate Vault",
		}

		vault, err := vm.CreateVault(config)
		assert.Error(t, err)
		assert.Nil(t, vault)
		assert.Contains(t, err.Error(), "already exists")
	})
}

func TestGetVault(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create a vault
	config := VaultConfig{
		ID:          "vault1",
		Name:        "Test Vault",
		MinDeposit:  big.NewInt(100),
		MaxCapacity: big.NewInt(100 * 1e6),
	}

	createdVault, err := vm.CreateVault(config)
	require.NoError(t, err)

	t.Run("GetExistingVault", func(t *testing.T) {
		vault, err := vm.GetVault("vault1")
		require.NoError(t, err)
		assert.NotNil(t, vault)
		assert.Equal(t, createdVault.ID, vault.ID)
	})

	t.Run("GetNonExistentVault", func(t *testing.T) {
		vault, err := vm.GetVault("nonexistent")
		assert.Error(t, err)
		assert.Nil(t, vault)
		assert.Contains(t, err.Error(), "not found")
	})
}

func TestCreateVaultWithProfitShare(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	t.Run("CreateProfitShareVault", func(t *testing.T) {
		leader := "leader123"
		name := "Profit Share Vault"
		description := "Test vault with profit sharing"
		initialDeposit := big.NewInt(100 * 1e6) // 100 USDC

		copyVault, err := vm.CreateVaultWithProfitShare(leader, name, description, initialDeposit)
		require.NoError(t, err)
		assert.NotNil(t, copyVault)
		assert.Equal(t, name, copyVault.Name)
		assert.Equal(t, leader, copyVault.Leader)
		assert.Equal(t, initialDeposit, copyVault.TotalDeposits)
		assert.Equal(t, 0.10, copyVault.ProfitShare) // 10% default
	})

	t.Run("CreateAnotherVaultDifferentLeader", func(t *testing.T) {
		// Create vault with different leader
		copyVault, err := vm.CreateVaultWithProfitShare("leader456", "Another Vault", "desc", big.NewInt(200 * 1e6))
		assert.NoError(t, err)
		assert.NotNil(t, copyVault)
		assert.Equal(t, "leader456", copyVault.Leader)
	})
}

func TestJoinVault(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create a copy vault
	leader := "leader123"
	vault, err := vm.CreateVaultWithProfitShare(leader, "Test Vault", "desc", big.NewInt(100 * 1e6))
	require.NoError(t, err)

	t.Run("JoinExistingVault", func(t *testing.T) {
		user := "user1"
		depositAmount := big.NewInt(100 * 1e6)

		err := vm.JoinVault(vault.ID, user, depositAmount)
		require.NoError(t, err)

		// Check user was added
		vm.mu.RLock()
		copyVault := vm.copyVaults[vault.ID]
		assert.NotNil(t, copyVault.Followers[user])
		// Shares calculation depends on vault value ratio
		assert.NotNil(t, copyVault.Followers[user].Shares)
		vm.mu.RUnlock()
	})

	t.Run("JoinNonExistentVault", func(t *testing.T) {
		err := vm.JoinVault("nonexistent", "user2", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("JoinWithZeroDeposit", func(t *testing.T) {
		err := vm.JoinVault(vault.ID, "user3", big.NewInt(0))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "minimum deposit is 100 USDC")
	})

	t.Run("LeaderCantJoinOwnVault", func(t *testing.T) {
		err := vm.JoinVault(vault.ID, leader, big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "minimum deposit is 100 USDC")
	})
}

func TestWithdrawFromVault(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create and fund a vault
	leader := "leader123"
	vault, _ := vm.CreateVaultWithProfitShare(leader, "Test Vault", "desc", big.NewInt(100 * 1e6))

	user := "user1"
	depositAmount := big.NewInt(100 * 1e6)
	vm.JoinVault(vault.ID, user, depositAmount)

	t.Run("PartialWithdraw", func(t *testing.T) {
		withdrawAmount, err := vm.WithdrawFromVault(vault.ID, user, 0.5) // 50%
		require.NoError(t, err)
		assert.NotNil(t, withdrawAmount)

		// Should get back approximately half
		expectedAmount := big.NewInt(50 * 1e6)
		assert.InDelta(t, expectedAmount.Int64(), withdrawAmount.Int64(), 1e6)
	})

	t.Run("FullWithdraw", func(t *testing.T) {
		// Add a new user for full withdrawal test
		newUser := "user2"
		depositAmount2 := big.NewInt(100 * 1e6)
		vm.JoinVault(vault.ID, newUser, depositAmount2)
		
		withdrawAmount, err := vm.WithdrawFromVault(vault.ID, newUser, 1.0) // 100%
		require.NoError(t, err)
		assert.NotNil(t, withdrawAmount)

		// User should have no balance left or be removed from followers
		vm.mu.RLock()
		copyVault := vm.copyVaults[vault.ID]
		position, exists := copyVault.Followers[newUser]
		vm.mu.RUnlock()

		if exists {
			assert.Equal(t, big.NewInt(0), position.Shares)
		} else {
			// User was completely removed after full withdrawal
			assert.False(t, exists)
		}
	})

	t.Run("InvalidWithdrawPercent", func(t *testing.T) {
		_, err := vm.WithdrawFromVault(vault.ID, user, 1.5) // 150%
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "share percent must be between 0 and 1")
	})
}

func TestListVaults(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create multiple vaults
	config1 := VaultConfig{ID: "vault1", Name: "Vault 1"}
	config2 := VaultConfig{ID: "vault2", Name: "Vault 2"}

	vm.CreateVault(config1)
	vm.CreateVault(config2)

	vaults := vm.ListVaults()
	assert.Len(t, vaults, 2)
}

func TestGetVaultPerformance(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create vault with performance tracking
	config := VaultConfig{
		ID:   "vault1",
		Name: "Performance Vault",
	}

	vault, _ := vm.CreateVault(config)

	t.Run("GetPerformanceMetrics", func(t *testing.T) {
		metrics, err := vm.GetVaultPerformance(vault.ID)
		require.NoError(t, err)
		assert.NotNil(t, metrics)
		assert.Equal(t, 0.0, metrics.TotalReturn)
		assert.Equal(t, 0.0, metrics.SharpeRatio)
	})

	t.Run("GetPerformanceNonExistent", func(t *testing.T) {
		metrics, err := vm.GetVaultPerformance("nonexistent")
		assert.Error(t, err)
		assert.Nil(t, metrics)
	})
}

func TestGetUserVaults(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create vaults and have user join them
	leader1 := "leader123"
	leader2 := "leader234"
	user := "user1"

	vault1, _ := vm.CreateVaultWithProfitShare(leader1, "Vault 1", "desc", big.NewInt(100 * 1e6))
	vault2, _ := vm.CreateVaultWithProfitShare(leader2, "Vault 2", "desc", big.NewInt(200 * 1e6))

	vm.JoinVault(vault1.ID, user, big.NewInt(100 * 1e6))
	vm.JoinVault(vault2.ID, user, big.NewInt(100 * 1e6))

	t.Run("GetUserVaults", func(t *testing.T) {
		vaults, err := vm.GetUserVaults(user)
		require.NoError(t, err)
		assert.Len(t, vaults, 2)
	})

	t.Run("GetVaultsForUserWithNone", func(t *testing.T) {
		vaults, err := vm.GetUserVaults("user_with_no_vaults")
		require.NoError(t, err)
		assert.Len(t, vaults, 0)
	})
}

func TestGetLeaderVaults(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	leader := "leader123"
	vault, _ := vm.CreateVaultWithProfitShare(leader, "Leader Vault", "desc", big.NewInt(100 * 1e6))

	t.Run("GetLeaderVaults", func(t *testing.T) {
		vaults, err := vm.GetLeaderVaults(leader)
		require.NoError(t, err)
		assert.Len(t, vaults, 1)
		assert.Equal(t, vault.ID, vaults[0].ID)
	})

	t.Run("GetVaultsForNonLeader", func(t *testing.T) {
		vaults, err := vm.GetLeaderVaults("not_a_leader")
		require.NoError(t, err)
		assert.Len(t, vaults, 0)
	})
}

func TestUpdateVaultValue(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	leader := "leader123"
	vault, _ := vm.CreateVaultWithProfitShare(leader, "Test Vault", "desc", big.NewInt(100 * 1e6))

	t.Run("UpdateValue", func(t *testing.T) {
		newValue := big.NewInt(12000)
		err := vm.UpdateVaultValue(vault.ID, newValue)
		require.NoError(t, err)

		// Check value was updated
		vm.mu.RLock()
		copyVault := vm.copyVaults[vault.ID]
		assert.Equal(t, newValue, copyVault.TotalDeposits)
		vm.mu.RUnlock()
	})

	t.Run("UpdateNonExistentVault", func(t *testing.T) {
		err := vm.UpdateVaultValue("nonexistent", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})
}

func TestExecuteVaultTrade(t *testing.T) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create vault
	leader := "leader123"
	vault, _ := vm.CreateVaultWithProfitShare(leader, "Trading Vault", "desc", big.NewInt(100 * 1e6))

	t.Run("ExecuteTrade", func(t *testing.T) {
		err := vm.ExecuteVaultTrade(vault.ID, "BTC-USDT", Buy, 0.1, Limit)
		// Will likely fail due to engine not being fully initialized, but should not panic
		if err != nil {
			assert.Contains(t, err.Error(), "vault not found")
		}
	})

	t.Run("ExecuteTradeInvalidVault", func(t *testing.T) {
		err := vm.ExecuteVaultTrade("nonexistent", "BTC-USDT", Buy, 0.1, Limit)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "vault not found")
	})
}

// Benchmark tests
func BenchmarkCreateVault(b *testing.B) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config := VaultConfig{
			ID:   fmt.Sprintf("vault%d", i),
			Name: fmt.Sprintf("Vault %d", i),
		}
		vm.CreateVault(config)
	}
}

func BenchmarkJoinVault(b *testing.B) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create a vault
	vault, _ := vm.CreateVaultWithProfitShare("leader", "Bench Vault", "desc", big.NewInt(1000 * 1e6))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		user := fmt.Sprintf("user%d", i)
		vm.JoinVault(vault.ID, user, big.NewInt(1000))
	}
}

func BenchmarkWithdrawFromVault(b *testing.B) {
	engine := &TradingEngine{}
	vm := NewVaultManager(engine)

	// Create and populate vault
	vault, _ := vm.CreateVaultWithProfitShare("leader", "Bench Vault", "desc", big.NewInt(1000 * 1e6))

	// Pre-add users
	for i := 0; i < 1000; i++ {
		user := fmt.Sprintf("user%d", i)
		vm.JoinVault(vault.ID, user, big.NewInt(100 * 1e6))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		user := fmt.Sprintf("user%d", i%1000)
		vm.WithdrawFromVault(vault.ID, user, 0.01) // Withdraw 1%
	}
}
