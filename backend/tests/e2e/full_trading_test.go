//go:build integration
// +build integration

package e2e

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullTradingFlow tests the complete trading flow
func TestFullTradingFlow(t *testing.T) {
	// Initialize trading engine
	engine := lx.NewTradingEngine(lx.EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	})
	
	require.NoError(t, engine.Start())
	defer engine.Stop()

	// Create spot market
	spotMarket := engine.CreateSpotMarket("BTC-USDT")
	assert.NotNil(t, spotMarket)

	// Create perpetual market
	perpConfig := lx.PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		InitialMargin:     0.01,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	perpMarket, err := engine.PerpManager.CreateMarket(perpConfig)
	require.NoError(t, err)
	assert.NotNil(t, perpMarket)

	// Set initial prices
	perpMarket.MarkPrice = 50000
	perpMarket.IndexPrice = 50000

	// Test spot trading
	t.Run("SpotTrading", func(t *testing.T) {
		// Add liquidity
		for i := 0; i < 10; i++ {
			buyOrder := &lx.Order{
				Symbol: "BTC-USDT",
				Side:   lx.Buy,
				Type:   lx.Limit,
				Price:  49900 - float64(i*10),
				Size:   0.1,
				User:   fmt.Sprintf("maker%d", i),
			}
			spotMarket.AddOrder(buyOrder)

			sellOrder := &lx.Order{
				Symbol: "BTC-USDT",
				Side:   lx.Sell,
				Type:   lx.Limit,
				Price:  50100 + float64(i*10),
				Size:   0.1,
				User:   fmt.Sprintf("maker%d", i),
			}
			spotMarket.AddOrder(sellOrder)
		}

		// Place taker order
		takerOrder := &lx.Order{
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Market,
			Size:   0.5,
			User:   "taker1",
		}
		spotMarket.AddOrder(takerOrder)

		// Match orders
		trades := spotMarket.MatchOrders()
		assert.Greater(t, len(trades), 0)
	})

	// Test perpetual trading
	t.Run("PerpetualTrading", func(t *testing.T) {
		// Open long position
		longPos, err := engine.PerpManager.OpenPosition(
			"trader1", "BTC-USD-PERP", lx.Buy, 1.0, 10, false,
		)
		require.NoError(t, err)
		assert.Equal(t, 1.0, longPos.Size)
		assert.Equal(t, 10.0, longPos.Leverage)

		// Open short position
		shortPos, err := engine.PerpManager.OpenPosition(
			"trader2", "BTC-USD-PERP", lx.Sell, 0.5, 20, false,
		)
		require.NoError(t, err)
		assert.Equal(t, -0.5, shortPos.Size)

		// Update price and check PnL
		engine.PerpManager.UpdateMarkPrice("BTC-USD-PERP", 51000)
		
		// Check unrealized PnL
		longPos, _ = engine.PerpManager.GetPosition("trader1", "BTC-USD-PERP")
		assert.Greater(t, longPos.UnrealizedPnL, 0.0) // Long profits when price rises

		shortPos, _ = engine.PerpManager.GetPosition("trader2", "BTC-USD-PERP")
		assert.Less(t, shortPos.UnrealizedPnL, 0.0) // Short loses when price rises
	})

	// Test vault functionality
	t.Run("VaultOperations", func(t *testing.T) {
		vaultConfig := lx.VaultConfig{
			ID:                "btc-vault-1",
			Name:              "BTC Trading Vault",
			Description:       "Automated BTC trading strategies",
			ManagementFee:     0.02,
			PerformanceFee:    0.20,
			MinDeposit:        big.NewInt(1000),
			MaxCapacity:       big.NewInt(1000000),
			LockupPeriod:      24 * time.Hour,
			RebalanceInterval: 1 * time.Hour,
			Strategies: []lx.StrategyConfig{
				{
					Type: "market_making",
					Name: "BTC Market Maker",
					Parameters: map[string]interface{}{
						"spread": 0.001,
						"depth":  5,
					},
				},
			},
		}

		vault, err := engine.VaultManager.CreateVault(vaultConfig)
		require.NoError(t, err)
		assert.NotNil(t, vault)

		// Deposit to vault
		depositAmount := big.NewInt(10000)
		position, err := vault.Deposit("investor1", depositAmount)
		require.NoError(t, err)
		assert.Equal(t, depositAmount, position.DepositValue)

		// Execute vault strategies
		orders := vault.ExecuteStrategies(spotMarket)
		assert.Greater(t, len(orders), 0)
	})

	// Test lending/borrowing
	t.Run("LendingBorrowing", func(t *testing.T) {
		poolConfig := lx.LendingPoolConfig{
			Asset:             "USDT",
			ReserveFactor:     0.10,
			LiquidationBonus:  0.05,
			CollateralFactor:  0.75,
			InterestRateModel: lx.NewJumpRateModel(0.02, 0.10, 0.50, 0.80),
			MinSupply:         big.NewInt(100),
			MaxSupply:         big.NewInt(10000000),
			MinBorrow:         big.NewInt(100),
			MaxBorrow:         big.NewInt(1000000),
			OracleSource:      "aggregate",
		}

		pool, err := engine.LendingManager.CreatePool(poolConfig)
		require.NoError(t, err)
		assert.NotNil(t, pool)

		// Supply liquidity
		supplyAmount := big.NewInt(100000)
		err = engine.LendingManager.Supply("lender1", "USDT", supplyAmount)
		require.NoError(t, err)

		// Borrow with collateral
		collateral := map[string]*big.Int{
			"BTC": big.NewInt(2), // 2 BTC as collateral
		}
		borrowAmount := big.NewInt(50000)
		err = engine.LendingManager.Borrow("borrower1", "USDT", borrowAmount, collateral)
		require.NoError(t, err)

		// Check loan
		loans := engine.LendingManager.GetUserLoans("borrower1")
		assert.Len(t, loans, 1)
		assert.Equal(t, borrowAmount, loans["USDT"].Amount)
	})
}

// TestHighLoadConcurrency tests system under high concurrent load
func TestHighLoadConcurrency(t *testing.T) {
	engine := lx.NewTradingEngine(lx.EngineConfig{})
	require.NoError(t, engine.Start())
	defer engine.Stop()

	market := engine.CreateSpotMarket("ETH-USDT")
	
	numTraders := 100
	ordersPerTrader := 100
	var wg sync.WaitGroup

	// Spawn concurrent traders
	for i := 0; i < numTraders; i++ {
		wg.Add(1)
		go func(traderID int) {
			defer wg.Done()
			
			for j := 0; j < ordersPerTrader; j++ {
				side := lx.Buy
				if j%2 == 0 {
					side = lx.Sell
				}
				
				order := &lx.Order{
					Symbol: "ETH-USDT",
					Side:   side,
					Type:   lx.Limit,
					Price:  3000 + float64(j%100),
					Size:   0.01,
					User:   fmt.Sprintf("trader-%d", traderID),
				}
				
				market.AddOrder(order)
				
				// Occasionally match orders
				if j%10 == 0 {
					market.MatchOrders()
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify system integrity
	snapshot := market.GetSnapshot()
	assert.NotNil(t, snapshot)
	
	// Check that order book is consistent
	for i := 1; i < len(snapshot.Bids); i++ {
		assert.GreaterOrEqual(t, snapshot.Bids[i-1].Price, snapshot.Bids[i].Price)
	}
	for i := 1; i < len(snapshot.Asks); i++ {
		assert.LessOrEqual(t, snapshot.Asks[i-1].Price, snapshot.Asks[i].Price)
	}
}

// TestMultiMarketArbitrage tests cross-market arbitrage
func TestMultiMarketArbitrage(t *testing.T) {
	engine := lx.NewTradingEngine(lx.EngineConfig{})
	require.NoError(t, engine.Start())
	defer engine.Stop()

	// Create multiple markets
	markets := []string{"BTC-USDT", "ETH-USDT", "BTC-ETH"}
	orderBooks := make(map[string]*lx.OrderBook)
	
	for _, symbol := range markets {
		orderBooks[symbol] = engine.CreateSpotMarket(symbol)
	}

	// Set up price discrepancies
	// BTC = $50,000, ETH = $3,000, BTC/ETH should be 16.67
	
	// Add liquidity to BTC-USDT
	orderBooks["BTC-USDT"].AddOrder(&lx.Order{
		Symbol: "BTC-USDT",
		Side:   lx.Sell,
		Type:   lx.Limit,
		Price:  50000,
		Size:   1,
		User:   "mm1",
	})

	// Add liquidity to ETH-USDT
	orderBooks["ETH-USDT"].AddOrder(&lx.Order{
		Symbol: "ETH-USDT",
		Side:   lx.Sell,
		Type:   lx.Limit,
		Price:  3000,
		Size:   10,
		User:   "mm2",
	})

	// Add mispriced liquidity to BTC-ETH (arbitrage opportunity)
	orderBooks["BTC-ETH"].AddOrder(&lx.Order{
		Symbol: "BTC-ETH",
		Side:   lx.Sell,
		Type:   lx.Limit,
		Price:  15, // Should be 16.67, so BTC is cheap in ETH terms
		Size:   1,
		User:   "mm3",
	})

	// Execute arbitrage
	// Buy BTC with ETH, sell BTC for USDT, buy ETH with USDT
	
	// Step 1: Buy 1 BTC for 15 ETH
	orderBooks["BTC-ETH"].AddOrder(&lx.Order{
		Symbol: "BTC-ETH",
		Side:   lx.Buy,
		Type:   lx.Market,
		Size:   1,
		User:   "arbitrageur",
	})
	trades1 := orderBooks["BTC-ETH"].MatchOrders()
	assert.Len(t, trades1, 1)
	assert.Equal(t, 15.0, trades1[0].Price)

	// Step 2: Sell 1 BTC for USDT
	orderBooks["BTC-USDT"].AddOrder(&lx.Order{
		Symbol: "BTC-USDT",
		Side:   lx.Buy,
		Type:   lx.Market,
		Size:   1,
		User:   "arbitrageur",
	})
	trades2 := orderBooks["BTC-USDT"].MatchOrders()
	assert.Len(t, trades2, 1)
	assert.Equal(t, 50000.0, trades2[0].Price)

	// Step 3: Buy ETH with USDT
	orderBooks["ETH-USDT"].AddOrder(&lx.Order{
		Symbol: "ETH-USDT",
		Side:   lx.Buy,
		Type:   lx.Market,
		Size:   16.67,
		User:   "arbitrageur",
	})
	trades3 := orderBooks["ETH-USDT"].MatchOrders()
	assert.Greater(t, len(trades3), 0)

	// Arbitrageur made profit: Started with 15 ETH, ended with 16.67 ETH
}

// TestLiquidationCascade tests liquidation cascade scenario
func TestLiquidationCascade(t *testing.T) {
	engine := lx.NewTradingEngine(lx.EngineConfig{
		EnablePerps: true,
	})
	require.NoError(t, engine.Start())
	defer engine.Stop()

	// Create perp market
	perpConfig := lx.PerpMarketConfig{
		Symbol:            "SOL-USD-PERP",
		UnderlyingAsset:   "SOL",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       50,
		MaintenanceMargin: 0.01,
		InitialMargin:     0.02,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, err := engine.PerpManager.CreateMarket(perpConfig)
	require.NoError(t, err)
	market.MarkPrice = 100

	// Open multiple leveraged long positions
	positions := []string{"trader1", "trader2", "trader3", "trader4", "trader5"}
	
	for _, trader := range positions {
		_, err := engine.PerpManager.OpenPosition(
			trader, "SOL-USD-PERP", lx.Buy, 10, 40, true, // High leverage
		)
		require.NoError(t, err)
	}

	// Simulate price drop triggering liquidations
	priceDrops := []float64{98, 96, 94, 92, 90}
	
	for _, price := range priceDrops {
		engine.PerpManager.UpdateMarkPrice("SOL-USD-PERP", price)
		
		// Check for liquidations
		liquidatedCount := 0
		for _, trader := range positions {
			pos, err := engine.PerpManager.GetPosition(trader, "SOL-USD-PERP")
			if err != nil || pos == nil {
				liquidatedCount++
			}
		}
		
		t.Logf("Price: $%.2f, Liquidated: %d", price, liquidatedCount)
	}
}

// TestFundingPayments tests funding payment calculations
func TestFundingPayments(t *testing.T) {
	engine := lx.NewTradingEngine(lx.EngineConfig{
		EnablePerps: true,
	})
	require.NoError(t, engine.Start())
	defer engine.Stop()

	// Create perp market with short funding interval
	perpConfig := lx.PerpMarketConfig{
		Symbol:            "AVAX-USD-PERP",
		UnderlyingAsset:   "AVAX",
		QuoteAsset:        "USD",
		FundingInterval:   100 * time.Millisecond, // Short for testing
		MaxLeverage:       20,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, err := engine.PerpManager.CreateMarket(perpConfig)
	require.NoError(t, err)
	
	// Set prices (mark > index = longs pay shorts)
	market.MarkPrice = 105
	market.IndexPrice = 100

	// Open balanced positions
	engine.PerpManager.OpenPosition("long1", "AVAX-USD-PERP", lx.Buy, 100, 10, false)
	engine.PerpManager.OpenPosition("long2", "AVAX-USD-PERP", lx.Buy, 50, 10, false)
	engine.PerpManager.OpenPosition("short1", "AVAX-USD-PERP", lx.Sell, 100, 10, false)
	engine.PerpManager.OpenPosition("short2", "AVAX-USD-PERP", lx.Sell, 50, 10, false)

	// Wait for funding
	time.Sleep(150 * time.Millisecond)
	
	// Process funding
	err = engine.PerpManager.ProcessFunding()
	require.NoError(t, err)

	// Check funding payments
	longPos1, _ := engine.PerpManager.GetPosition("long1", "AVAX-USD-PERP")
	longPos2, _ := engine.PerpManager.GetPosition("long2", "AVAX-USD-PERP")
	shortPos1, _ := engine.PerpManager.GetPosition("short1", "AVAX-USD-PERP")
	shortPos2, _ := engine.PerpManager.GetPosition("short2", "AVAX-USD-PERP")

	// Longs should pay (negative funding)
	assert.Less(t, longPos1.FundingPaid, 0.0)
	assert.Less(t, longPos2.FundingPaid, 0.0)
	
	// Shorts should receive (positive funding)
	assert.Greater(t, shortPos1.FundingPaid, 0.0)
	assert.Greater(t, shortPos2.FundingPaid, 0.0)
	
	// Total funding should balance (zero-sum)
	totalFunding := longPos1.FundingPaid + longPos2.FundingPaid + 
	                shortPos1.FundingPaid + shortPos2.FundingPaid
	assert.InDelta(t, 0.0, totalFunding, 0.01)
}

// TestVaultRebalancing tests vault rebalancing mechanism
func TestVaultRebalancing(t *testing.T) {
	engine := lx.NewTradingEngine(lx.EngineConfig{
		EnableVaults: true,
	})
	require.NoError(t, engine.Start())
	defer engine.Stop()

	// Create multi-strategy vault
	vaultConfig := lx.VaultConfig{
		ID:                "multi-strat-vault",
		Name:              "Multi-Strategy Vault",
		ManagementFee:     0.02,
		PerformanceFee:    0.20,
		MinDeposit:        big.NewInt(1000),
		MaxCapacity:       big.NewInt(10000000),
		RebalanceInterval: 100 * time.Millisecond,
		Strategies: []lx.StrategyConfig{
			{
				Type: "market_making",
				Name: "MM Strategy",
				Parameters: map[string]interface{}{
					"spread": 0.001,
				},
			},
			{
				Type: "momentum",
				Name: "Momentum Strategy",
				Parameters: map[string]interface{}{
					"lookback": 20,
				},
			},
		},
	}

	vault, err := engine.VaultManager.CreateVault(vaultConfig)
	require.NoError(t, err)

	// Add deposits
	vault.Deposit("investor1", big.NewInt(100000))
	vault.Deposit("investor2", big.NewInt(200000))
	vault.Deposit("investor3", big.NewInt(150000))

	// Wait for rebalance
	time.Sleep(150 * time.Millisecond)
	
	// Trigger rebalance
	err = vault.Rebalance()
	require.NoError(t, err)

	// Check vault state
	assert.Equal(t, big.NewInt(450000), vault.TotalDeposits)
	assert.Greater(t, len(vault.Strategies), 0)
}

// BenchmarkOrderMatching benchmarks order matching performance
func BenchmarkOrderMatching(b *testing.B) {
	engine := lx.NewTradingEngine(lx.EngineConfig{})
	engine.Start()
	defer engine.Stop()

	market := engine.CreateSpotMarket("BTC-USDT")
	
	// Pre-populate order book
	for i := 0; i < 1000; i++ {
		market.AddOrder(&lx.Order{
			Symbol: "BTC-USDT",
			Side:   lx.Buy,
			Type:   lx.Limit,
			Price:  49000 + float64(i),
			Size:   0.1,
			User:   "maker",
		})
		market.AddOrder(&lx.Order{
			Symbol: "BTC-USDT",
			Side:   lx.Sell,
			Type:   lx.Limit,
			Price:  51000 + float64(i),
			Size:   0.1,
			User:   "maker",
		})
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Add crossing orders
			market.AddOrder(&lx.Order{
				Symbol: "BTC-USDT",
				Side:   lx.Buy,
				Type:   lx.Market,
				Size:   0.01,
				User:   "taker",
			})
			market.MatchOrders()
		}
	})
}

// BenchmarkPerpPositions benchmarks perpetual position management
func BenchmarkPerpPositions(b *testing.B) {
	engine := lx.NewTradingEngine(lx.EngineConfig{
		EnablePerps: true,
	})
	engine.Start()
	defer engine.Stop()

	perpConfig := lx.PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := engine.PerpManager.CreateMarket(perpConfig)
	market.MarkPrice = 50000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		user := fmt.Sprintf("user%d", i%1000)
		side := lx.Buy
		if i%2 == 0 {
			side = lx.Sell
		}
		
		engine.PerpManager.OpenPosition(user, "BTC-USD-PERP", side, 0.1, 10, false)
		
		if i%100 == 0 {
			// Periodically update price
			market.MarkPrice = 50000 + float64(i%1000)
		}
	}
}