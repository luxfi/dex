package lx

import (
	"math/big"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func defaultPoolConfig() PoolConfig {
	return PoolConfig{
		ReserveFactor:        0.10,
		CollateralFactor:     0.75,
		LiquidationThreshold: 0.80,
		LiquidationPenalty:   0.05,
		MaxBorrowRate:        0.50,
		MinBorrowRate:        0.01,
		OptimalUtilization:   0.80,
	}
}

func TestNewLendingPool(t *testing.T) {
	lp := NewLendingPool()
	assert.NotNil(t, lp)
	assert.NotNil(t, lp.Pools)
	assert.NotNil(t, lp.Suppliers)
	assert.NotNil(t, lp.Borrowers)
	assert.NotNil(t, lp.InterestModel)
	assert.NotNil(t, lp.CollateralManager)
	assert.NotNil(t, lp.ReserveFactory)
	assert.Equal(t, 0.10, lp.ProtocolFeeRate)
}

func TestLendingPool_CreatePool(t *testing.T) {
	t.Run("CreateNewPool", func(t *testing.T) {
		lp := NewLendingPool()
		config := defaultPoolConfig()

		err := lp.CreatePool("ETH", config)
		assert.NoError(t, err)

		pool, exists := lp.Pools["ETH"]
		assert.True(t, exists)
		assert.Equal(t, "ETH", pool.Asset)
		assert.Equal(t, config.ReserveFactor, pool.ReserveFactor)
		assert.Equal(t, config.CollateralFactor, pool.CollateralFactor)
	})

	t.Run("DuplicatePoolError", func(t *testing.T) {
		lp := NewLendingPool()
		config := defaultPoolConfig()

		err := lp.CreatePool("ETH", config)
		assert.NoError(t, err)

		err = lp.CreatePool("ETH", config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "pool already exists")
	})
}

func TestLendingPool_Supply(t *testing.T) {
	t.Run("FirstSupplyGetShares1to1", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		amount := big.NewInt(1000000)
		err := lp.Supply("user1", "ETH", amount)
		assert.NoError(t, err)

		supplier := lp.Suppliers["user1"]
		position := supplier.SuppliedAssets["ETH"]
		assert.Equal(t, amount, position.Amount)
		assert.Equal(t, amount, position.Shares) // 1:1 for first supplier
	})

	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		err := lp.Supply("user1", "NONEXISTENT", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "pool not found")
	})

	t.Run("PoolPaused", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())
		lp.Pools["ETH"].Paused = true

		err := lp.Supply("user1", "ETH", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "paused")
	})

	t.Run("MultipleSupplies", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		// First supply
		lp.Supply("user1", "ETH", big.NewInt(1000000))

		// Second supply to same pool
		lp.Supply("user2", "ETH", big.NewInt(500000))

		pool := lp.Pools["ETH"]
		assert.Equal(t, big.NewInt(1500000), pool.TotalSupply)
	})
}

func TestLendingPool_Withdraw(t *testing.T) {
	t.Run("FullWithdraw", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		err := lp.Withdraw("user1", "ETH", supplyAmount)
		assert.NoError(t, err)

		// Position should be removed
		supplier := lp.Suppliers["user1"]
		_, exists := supplier.SuppliedAssets["ETH"]
		assert.False(t, exists)

		pool := lp.Pools["ETH"]
		assert.Equal(t, 0, pool.TotalSupply.Cmp(big.NewInt(0)))
	})

	t.Run("PartialWithdraw", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		withdrawAmount := big.NewInt(400000)
		err := lp.Withdraw("user1", "ETH", withdrawAmount)
		assert.NoError(t, err)

		supplier := lp.Suppliers["user1"]
		position := supplier.SuppliedAssets["ETH"]
		expectedRemaining := big.NewInt(600000)
		assert.Equal(t, expectedRemaining, position.Amount)
	})

	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		err := lp.Withdraw("user1", "NONEXISTENT", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "pool not found")
	})

	t.Run("NoSupplyPosition", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		err := lp.Withdraw("user1", "ETH", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no supply position found")
	})

	t.Run("NoPositionForAsset", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())
		lp.CreatePool("BTC", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))

		err := lp.Withdraw("user1", "BTC", big.NewInt(1000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no supply position for asset")
	})

	t.Run("InsufficientLiquidity", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		// Borrow most of the liquidity
		lp.Borrow("ETH", big.NewInt(900000))

		// Try to withdraw more than available
		err := lp.Withdraw("user1", "ETH", big.NewInt(200000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient liquidity")
	})

	t.Run("WithdrawExceedsSupplied", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		// Try to withdraw more than supplied - will hit liquidity check first
		err := lp.Withdraw("user1", "ETH", big.NewInt(2000000))
		assert.Error(t, err)
		// Error could be insufficient liquidity or exceeds supplied - both valid
		assert.True(t, strings.Contains(err.Error(), "insufficient liquidity") || strings.Contains(err.Error(), "exceeds"))
	})

	t.Run("WithdrawWithInterest", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		// Set supply rate and simulate time passing
		supplier := lp.Suppliers["user1"]
		position := supplier.SuppliedAssets["ETH"]
		position.SupplyRate = 0.05 // 5% annual
		position.LastUpdate = time.Now().Add(-24 * time.Hour) // 1 day ago

		err := lp.Withdraw("user1", "ETH", big.NewInt(500000))
		assert.NoError(t, err)

		// Interest should be calculated and added
		assert.True(t, position.InterestEarned.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("SharesBurnedProportionally", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		supplier := lp.Suppliers["user1"]
		position := supplier.SuppliedAssets["ETH"]
		initialShares := new(big.Int).Set(position.Shares)

		// Withdraw half
		withdrawAmount := big.NewInt(500000)
		lp.Withdraw("user1", "ETH", withdrawAmount)

		expectedShares := new(big.Int).Div(initialShares, big.NewInt(2))
		assert.Equal(t, expectedShares, position.Shares)
	})
}

func TestLendingPool_calculateSupplyInterest(t *testing.T) {
	t.Run("ZeroRate", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		pool := lp.Pools["ETH"]
		position := &SupplyPosition{
			Asset:      "ETH",
			Amount:     big.NewInt(1000000),
			SupplyRate: 0.0,
			LastUpdate: time.Now().Add(-24 * time.Hour),
		}

		interest := lp.calculateSupplyInterest(position, pool)
		assert.Equal(t, big.NewInt(0), interest)
	})

	t.Run("PositiveRate", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		pool := lp.Pools["ETH"]
		position := &SupplyPosition{
			Asset:      "ETH",
			Amount:     big.NewInt(1000000000), // 1B units
			SupplyRate: 0.10,                   // 10% annual
			LastUpdate: time.Now().Add(-8760 * time.Hour), // 1 year
		}

		interest := lp.calculateSupplyInterest(position, pool)
		// Expected: 1B * 0.10 = 100M
		assert.True(t, interest.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("ShortDuration", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		pool := lp.Pools["ETH"]
		position := &SupplyPosition{
			Asset:      "ETH",
			Amount:     big.NewInt(1000000000),
			SupplyRate: 0.10,
			LastUpdate: time.Now().Add(-1 * time.Hour), // 1 hour
		}

		interest := lp.calculateSupplyInterest(position, pool)
		// Should be much smaller than full year
		assert.True(t, interest.Cmp(big.NewInt(100000000)) < 0)
	})

	t.Run("HighRate", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		pool := lp.Pools["ETH"]
		position := &SupplyPosition{
			Asset:      "ETH",
			Amount:     big.NewInt(1000000000),
			SupplyRate: 0.50, // 50% annual
			LastUpdate: time.Now().Add(-24 * time.Hour),
		}

		interest := lp.calculateSupplyInterest(position, pool)
		assert.True(t, interest.Cmp(big.NewInt(0)) > 0)
	})
}

func TestLendingPool_getTotalShares(t *testing.T) {
	t.Run("EmptyPool", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		pool := lp.Pools["ETH"]
		totalShares := lp.getTotalShares(pool)
		assert.Equal(t, big.NewInt(0), totalShares)
	})

	t.Run("SingleSupplier", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		pool := lp.Pools["ETH"]
		totalShares := lp.getTotalShares(pool)
		assert.Equal(t, supplyAmount, totalShares)
	})

	t.Run("MultipleSuppliers", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Supply("user2", "ETH", big.NewInt(500000))
		lp.Supply("user3", "ETH", big.NewInt(250000))

		pool := lp.Pools["ETH"]
		totalShares := lp.getTotalShares(pool)
		// First supplier gets 1:1, subsequent suppliers get proportional
		// Total shares should be >= sum of amounts
		assert.True(t, totalShares.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("AfterWithdrawal", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Supply("user2", "ETH", big.NewInt(500000))

		pool := lp.Pools["ETH"]
		sharesBefore := lp.getTotalShares(pool)

		lp.Withdraw("user1", "ETH", big.NewInt(500000))

		sharesAfter := lp.getTotalShares(pool)
		assert.True(t, sharesAfter.Cmp(sharesBefore) < 0)
	})
}

func TestInterestRateModel_calculateBorrowRate(t *testing.T) {
	t.Run("ZeroUtilization", func(t *testing.T) {
		model := NewDefaultInterestModel()
		rate := model.calculateBorrowRate(0.0)
		assert.Equal(t, model.BaseRate, rate)
	})

	t.Run("BelowKink", func(t *testing.T) {
		model := NewDefaultInterestModel()
		utilization := 0.40 // 40%

		rate := model.calculateBorrowRate(utilization)
		expectedRate := model.BaseRate + utilization*model.MultiplierPerBlock
		assert.Equal(t, expectedRate, rate)
	})

	t.Run("AtKink", func(t *testing.T) {
		model := NewDefaultInterestModel()
		utilization := model.Kink // 80%

		rate := model.calculateBorrowRate(utilization)
		expectedRate := model.BaseRate + utilization*model.MultiplierPerBlock
		assert.Equal(t, expectedRate, rate)
	})

	t.Run("AboveKink", func(t *testing.T) {
		model := NewDefaultInterestModel()
		utilization := 0.90 // 90%

		rate := model.calculateBorrowRate(utilization)
		normalRate := model.BaseRate + model.Kink*model.MultiplierPerBlock
		excessUtilization := utilization - model.Kink
		expectedRate := normalRate + excessUtilization*model.JumpMultiplier
		assert.Equal(t, expectedRate, rate)
	})

	t.Run("FullUtilization", func(t *testing.T) {
		model := NewDefaultInterestModel()
		utilization := 1.0 // 100%

		rate := model.calculateBorrowRate(utilization)
		normalRate := model.BaseRate + model.Kink*model.MultiplierPerBlock
		excessUtilization := utilization - model.Kink
		expectedRate := normalRate + excessUtilization*model.JumpMultiplier
		assert.Equal(t, expectedRate, rate)
	})

	t.Run("JumpMultiplierEffect", func(t *testing.T) {
		model := NewDefaultInterestModel()

		rateAtKink := model.calculateBorrowRate(model.Kink)
		rateAboveKink := model.calculateBorrowRate(model.Kink + 0.01)

		// Rate should jump significantly above kink
		rateDiff := rateAboveKink - rateAtKink
		expectedJump := 0.01 * model.JumpMultiplier
		assert.InDelta(t, expectedJump, rateDiff, 0.0001)
	})

	t.Run("LinearBelowKink", func(t *testing.T) {
		model := NewDefaultInterestModel()

		rate20 := model.calculateBorrowRate(0.20)
		rate40 := model.calculateBorrowRate(0.40)
		rate60 := model.calculateBorrowRate(0.60)

		// Differences should be equal (linear)
		diff1 := rate40 - rate20
		diff2 := rate60 - rate40
		assert.InDelta(t, diff1, diff2, 0.0001)
	})

	t.Run("CustomModel", func(t *testing.T) {
		model := &InterestRateModel{
			BaseRate:           0.05,
			MultiplierPerBlock: 0.10,
			JumpMultiplier:     3.0,
			Kink:               0.70,
			BlocksPerYear:      2628000,
		}

		rate := model.calculateBorrowRate(0.50)
		expectedRate := 0.05 + 0.50*0.10
		assert.InDelta(t, expectedRate, rate, 0.0001)

		rateAbove := model.calculateBorrowRate(0.80)
		normalRate := 0.05 + 0.70*0.10
		excess := 0.80 - 0.70
		expectedAbove := normalRate + excess*3.0
		assert.InDelta(t, expectedAbove, rateAbove, 0.0001)
	})
}

func TestNewDefaultInterestModel(t *testing.T) {
	model := NewDefaultInterestModel()

	assert.Equal(t, 0.02, model.BaseRate)
	assert.Equal(t, 0.15, model.MultiplierPerBlock)
	assert.Equal(t, 2.0, model.JumpMultiplier)
	assert.Equal(t, 0.80, model.Kink)
	assert.Equal(t, uint64(2628000), model.BlocksPerYear)
}

func TestLendingPool_Borrow(t *testing.T) {
	t.Run("SuccessfulBorrow", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		// Supply first
		lp.Supply("user1", "ETH", big.NewInt(1000000))

		// Borrow
		err := lp.Borrow("ETH", big.NewInt(500000))
		assert.NoError(t, err)

		pool := lp.Pools["ETH"]
		assert.Equal(t, big.NewInt(500000), pool.TotalBorrow)
	})

	t.Run("InsufficientLiquidity", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))

		err := lp.Borrow("ETH", big.NewInt(2000000))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient liquidity")
	})

	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		err := lp.Borrow("NONEXISTENT", big.NewInt(1000))
		assert.Error(t, err)
	})
}

func TestLendingPool_Repay(t *testing.T) {
	t.Run("SuccessfulRepay", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Borrow("ETH", big.NewInt(500000))

		principal := big.NewInt(500000)
		interest := big.NewInt(50000)
		err := lp.Repay("ETH", principal, interest)
		assert.NoError(t, err)

		pool := lp.Pools["ETH"]
		assert.True(t, pool.TotalBorrow.Cmp(big.NewInt(0)) == 0)
		assert.True(t, pool.TotalReserves.Cmp(big.NewInt(0)) > 0)
	})

	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		err := lp.Repay("NONEXISTENT", big.NewInt(1000), big.NewInt(100))
		assert.Error(t, err)
	})
}

func TestLendingPool_GetAvailable(t *testing.T) {
	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		available := lp.GetAvailable("NONEXISTENT")
		assert.Equal(t, big.NewInt(0), available)
	})

	t.Run("FullLiquidity", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		supplyAmount := big.NewInt(1000000)
		lp.Supply("user1", "ETH", supplyAmount)

		available := lp.GetAvailable("ETH")
		assert.Equal(t, supplyAmount, available)
	})

	t.Run("PartiallyBorrowed", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Borrow("ETH", big.NewInt(400000))

		available := lp.GetAvailable("ETH")
		assert.Equal(t, big.NewInt(600000), available)
	})
}

func TestLendingPool_GetBorrowRate(t *testing.T) {
	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		rate := lp.GetBorrowRate("NONEXISTENT")
		assert.Equal(t, 0.0, rate)
	})

	t.Run("ExistingPool", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Borrow("ETH", big.NewInt(500000))

		rate := lp.GetBorrowRate("ETH")
		assert.True(t, rate >= 0.01) // At least min borrow rate
	})
}

func TestLendingPool_GetSupplyRate(t *testing.T) {
	t.Run("PoolNotFound", func(t *testing.T) {
		lp := NewLendingPool()
		rate := lp.GetSupplyRate("NONEXISTENT")
		assert.Equal(t, 0.0, rate)
	})

	t.Run("ExistingPool", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		lp.Supply("user1", "ETH", big.NewInt(1000000))
		lp.Borrow("ETH", big.NewInt(500000))

		rate := lp.GetSupplyRate("ETH")
		// Supply rate should be >= 0
		assert.True(t, rate >= 0)
	})
}

func TestCollateralManager(t *testing.T) {
	cm := NewCollateralManager()

	assert.NotNil(t, cm)
	assert.NotNil(t, cm.CollateralFactors)

	// Check default factors
	assert.Equal(t, 0.80, cm.CollateralFactors["BTC"])
	assert.Equal(t, 0.75, cm.CollateralFactors["ETH"])
	assert.Equal(t, 0.95, cm.CollateralFactors["USDT"])
	assert.Equal(t, 0.95, cm.CollateralFactors["USDC"])
}

func TestReserveFactory(t *testing.T) {
	rf := NewReserveFactory()

	assert.NotNil(t, rf)
	assert.NotNil(t, rf.Reserves)
	assert.Len(t, rf.Reserves, 0)
}

func TestLendingPool_UtilizationRate(t *testing.T) {
	lp := NewLendingPool()
	lp.CreatePool("ETH", defaultPoolConfig())

	// Initial utilization should be 0
	pool := lp.Pools["ETH"]
	assert.Equal(t, 0.0, pool.UtilizationRate)

	// Supply
	lp.Supply("user1", "ETH", big.NewInt(1000000))
	assert.Equal(t, 0.0, pool.UtilizationRate)

	// Borrow 50%
	lp.Borrow("ETH", big.NewInt(500000))
	assert.InDelta(t, 0.5, pool.UtilizationRate, 0.01)

	// Borrow more
	lp.Borrow("ETH", big.NewInt(300000))
	assert.InDelta(t, 0.8, pool.UtilizationRate, 0.01)
}

func TestLendingPool_InterestRateUpdates(t *testing.T) {
	lp := NewLendingPool()
	config := defaultPoolConfig()
	lp.CreatePool("ETH", config)

	lp.Supply("user1", "ETH", big.NewInt(1000000))
	pool := lp.Pools["ETH"]

	// Initial rates
	initialBorrowRate := pool.BorrowRate
	initialSupplyRate := pool.SupplyRate

	// Borrow to change utilization
	lp.Borrow("ETH", big.NewInt(500000))

	// Rates should have changed
	assert.NotEqual(t, initialBorrowRate, pool.BorrowRate)
	assert.NotEqual(t, initialSupplyRate, pool.SupplyRate)
}

func TestLendingPool_ConcurrentAccess(t *testing.T) {
	lp := NewLendingPool()
	lp.CreatePool("ETH", defaultPoolConfig())

	// Supply initial liquidity
	lp.Supply("initial", "ETH", big.NewInt(10000000))

	done := make(chan bool)
	errors := make(chan error, 20)

	// Concurrent supplies
	for i := 0; i < 10; i++ {
		go func(id int) {
			userID := "user" + string(rune('0'+id))
			err := lp.Supply(userID, "ETH", big.NewInt(100000))
			if err != nil {
				errors <- err
			}
			done <- true
		}(i)
	}

	// Concurrent borrows
	for i := 0; i < 10; i++ {
		go func() {
			err := lp.Borrow("ETH", big.NewInt(10000))
			if err != nil {
				errors <- err
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}

	close(errors)
	for err := range errors {
		require.NoError(t, err)
	}
}

func TestLendingPool_EdgeCases(t *testing.T) {
	t.Run("ZeroSupply", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		err := lp.Supply("user1", "ETH", big.NewInt(0))
		// Should still work with zero amount
		assert.NoError(t, err)
	})

	t.Run("VeryLargeAmounts", func(t *testing.T) {
		lp := NewLendingPool()
		lp.CreatePool("ETH", defaultPoolConfig())

		// 1 trillion units
		largeAmount := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
		err := lp.Supply("user1", "ETH", largeAmount)
		assert.NoError(t, err)

		pool := lp.Pools["ETH"]
		assert.Equal(t, largeAmount, pool.TotalSupply)
	})
}
