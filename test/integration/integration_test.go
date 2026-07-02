//go:build integration

package integration

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

// IntegrationTestSuite tests the complete DEX functionality
type IntegrationTestSuite struct {
	suite.Suite
	engine *dex.TradingEngine
}

func (suite *IntegrationTestSuite) SetupTest() {
	config := dex.EngineConfig{
		EnableMargin:  true,
		EnableVaults:  true,
		EnableBridge:  true,
		EnableOracles: true,
	}
	suite.engine = dex.NewTradingEngine(config)
}

func (suite *IntegrationTestSuite) TearDownTest() {
	// Cleanup if needed
}

// Test complete order lifecycle
func (suite *IntegrationTestSuite) TestOrderLifecycle() {
	// Create orderbook
	ob := dex.NewOrderBook("BTC-USDT")

	// Place buy order
	buyOrder := &dex.Order{
		ID:        1,
		User:      "user1",
		Symbol:    "BTC-USDT",
		Side:      dex.Buy,
		Type:      dex.Limit,
		Price:     50000,
		Size:      1.0,
		Timestamp: time.Now(),
	}

	err := ob.PlaceOrder(buyOrder)
	assert.NoError(suite.T(), err)

	// Place sell order (should match)
	sellOrder := &dex.Order{
		ID:        2,
		User:      "user2",
		Symbol:    "BTC-USDT",
		Side:      dex.Sell,
		Type:      dex.Limit,
		Price:     50000,
		Size:      0.5,
		Timestamp: time.Now(),
	}

	trades, err := ob.PlaceOrder(sellOrder)
	assert.NoError(suite.T(), err)
	assert.Len(suite.T(), trades, 1)
	assert.Equal(suite.T(), 0.5, trades[0].Size)

	// Cancel remaining order
	err = ob.CancelOrder(1)
	assert.NoError(suite.T(), err)
}

// Test margin trading flow
func (suite *IntegrationTestSuite) TestMarginTradingFlow() {
	margin := dex.NewMarginEngine(suite.engine)

	// Open position
	position, err := margin.OpenPosition("user1", "BTC-USDT", dex.Buy, 0.1, 10)
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), position)
	assert.Equal(suite.T(), 10.0, position.Leverage)

	// Update position (price change)
	position.MarkPrice = 55000
	pnl := margin.CalculatePnL(position)
	assert.True(suite.T(), pnl > 0) // Profit since price went up

	// Close position
	err = margin.ClosePosition(position.ID)
	assert.NoError(suite.T(), err)
}

// Test vault operations
func (suite *IntegrationTestSuite) TestVaultOperations() {
	manager := dex.NewVaultManager(suite.engine)

	// Create vault
	config := dex.VaultConfig{
		ID:         "test-vault",
		Name:       "Test Vault",
		MinDeposit: big.NewInt(1000),
	}

	vault, err := manager.CreateVault(config)
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), vault)

	// Deposit
	position, err := vault.Deposit("user1", big.NewInt(5000))
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), position)

	// Withdraw
	amount, err := vault.Withdraw("user1", big.NewInt(2000))
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), amount)
}

// Test cross-chain bridge
func (suite *IntegrationTestSuite) TestCrossChainBridge() {
	bridge := dex.NewCrossChainBridge()

	// Lock tokens on source chain
	txHash, err := bridge.LockTokens("ETH", "user1", 1.5, "BSC")
	assert.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), txHash)

	// Verify on destination chain
	verified := bridge.VerifyTransaction(txHash)
	assert.True(suite.T(), verified)

	// Mint on destination
	err = bridge.MintTokens("BSC", "user1", 1.5, txHash)
	assert.NoError(suite.T(), err)
}

// Test liquidation engine
func (suite *IntegrationTestSuite) TestLiquidationEngine() {
	engine := dex.NewLiquidationEngine()
	margin := dex.NewMarginEngine(suite.engine)

	// Create underwater position
	position := &dex.MarginPosition{
		ID:                "pos1",
		User:              "user1",
		Symbol:            "BTC-USDT",
		Side:              dex.Buy,
		Size:              1.0,
		EntryPrice:        50000,
		MarkPrice:         45000, // Price dropped
		Leverage:          20,
		Margin:            2500,
		MaintenanceMargin: 0.005,
	}

	// Check if liquidatable
	shouldLiquidate := engine.ShouldLiquidate(position)
	assert.True(suite.T(), shouldLiquidate)

	// Execute liquidation
	err := engine.Liquidate(position)
	assert.NoError(suite.T(), err)
}

// Test oracle integration
func (suite *IntegrationTestSuite) TestOracleIntegration() {
	oracle := dex.NewPriceOracle()

	// Add price sources
	oracle.AddSource("chainlink", dex.NewChainlinkSource())
	oracle.AddSource("pyth", dex.NewPythSource())

	// Get aggregated price
	price, err := oracle.GetPrice("BTC-USDT")
	assert.NoError(suite.T(), err)
	assert.True(suite.T(), price > 0)

	// Test price deviation detection
	prices := map[string]float64{
		"chainlink": 50000,
		"pyth":      50100,
	}

	deviation := oracle.CalculateDeviation(prices)
	assert.True(suite.T(), deviation < 0.01) // Less than 1% deviation
}

// Test staking operations
func (suite *IntegrationTestSuite) TestStakingOperations() {
	staking := dex.NewStakingManager()

	// Stake tokens
	position, err := staking.Stake("pool1", "user1", big.NewInt(10000))
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), position)

	// Earn rewards (simulate time passing)
	time.Sleep(100 * time.Millisecond)

	// Claim rewards
	rewards, err := staking.ClaimRewards("user1", "pool1")
	assert.NoError(suite.T(), err)
	assert.True(suite.T(), rewards.Cmp(big.NewInt(0)) > 0)

	// Unstake
	amount, err := staking.Unstake("pool1", "user1", big.NewInt(5000))
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), big.NewInt(5000), amount)
}

// Test high load scenario
func (suite *IntegrationTestSuite) TestHighLoadScenario() {
	ob := dex.NewOrderBook("ETH-USDT")

	// Place many orders concurrently
	orderCount := 1000
	done := make(chan bool, orderCount)

	for i := 0; i < orderCount; i++ {
		go func(id int) {
			order := &dex.Order{
				ID:        uint64(id),
				User:      fmt.Sprintf("user%d", id%10),
				Symbol:    "ETH-USDT",
				Side:      dex.Side(id % 2),
				Type:      dex.Limit,
				Price:     3000 + float64(id%100),
				Size:      0.1,
				Timestamp: time.Now(),
			}
			ob.PlaceOrder(order)
			done <- true
		}(i)
	}

	// Wait for all orders
	for i := 0; i < orderCount; i++ {
		<-done
	}

	// Verify orderbook integrity
	bids, asks := ob.GetDepth(100)
	assert.True(suite.T(), len(bids) > 0)
	assert.True(suite.T(), len(asks) > 0)

	// Verify price ordering
	for i := 1; i < len(bids); i++ {
		assert.True(suite.T(), bids[i-1].Price >= bids[i].Price)
	}
	for i := 1; i < len(asks); i++ {
		assert.True(suite.T(), asks[i-1].Price <= asks[i].Price)
	}
}

// Run the test suite
func TestIntegrationSuite(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	suite.Run(t, new(IntegrationTestSuite))
}
