package lx

import (
	"math"
	"testing"
	"time"
)

func TestCreatePerpetualMarket(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		InitialMargin:     0.01,
		TakerFee:          0.0004,
		MakerFee:          0.0002,
		MaxPositionSize:   100,
		PricePrecision:    2,
		SizePrecision:     4,
		TickSize:          0.01,
		ContractSize:      1,
		IsInverse:         false,
		OracleSource:      "aggregate",
	}
	
	market, err := pm.CreateMarket(config)
	if err != nil {
		t.Fatalf("Failed to create market: %v", err)
	}
	
	if market.Symbol != "BTC-USD-PERP" {
		t.Errorf("Expected symbol BTC-USD-PERP, got %s", market.Symbol)
	}
	
	if market.MaxLeverage != 100 {
		t.Errorf("Expected max leverage 100, got %f", market.MaxLeverage)
	}
}

func TestOpenPosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Create market
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		InitialMargin:     0.01,
		TakerFee:          0.0004,
		MakerFee:          0.0002,
		MaxPositionSize:   100,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	market.IndexPrice = 50000
	
	// Open long position
	position, err := pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 10, false)
	if err != nil {
		t.Fatalf("Failed to open position: %v", err)
	}
	
	if position.Size != 1.0 {
		t.Errorf("Expected position size 1.0, got %f", position.Size)
	}
	
	if position.EntryPrice != 50000 {
		t.Errorf("Expected entry price 50000, got %f", position.EntryPrice)
	}
	
	if position.Leverage != 10 {
		t.Errorf("Expected leverage 10, got %f", position.Leverage)
	}
}

func TestOpenShortPosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Create market
	config := PerpMarketConfig{
		Symbol:            "ETH-USD-PERP",
		UnderlyingAsset:   "ETH",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       50,
		MaintenanceMargin: 0.005,
		InitialMargin:     0.01,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 3000
	market.IndexPrice = 3000
	
	// Open short position
	position, err := pm.OpenPosition("user1", "ETH-USD-PERP", Sell, 2.0, 5, false)
	if err != nil {
		t.Fatalf("Failed to open short position: %v", err)
	}
	
	if position.Size != -2.0 {
		t.Errorf("Expected position size -2.0, got %f", position.Size)
	}
}

func TestClosePosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup market and position
	config := PerpMarketConfig{
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
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	market.IndexPrice = 50000
	
	// Open position
	pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 10, false)
	
	// Change price
	market.MarkPrice = 51000
	
	// Close position
	closedPos, err := pm.ClosePosition("user1", "BTC-USD-PERP", 1.0)
	if err != nil {
		t.Fatalf("Failed to close position: %v", err)
	}
	
	// Check PnL
	expectedPnL := 1.0 * 1 * (51000 - 50000) // size * contractSize * price_diff
	if math.Abs(closedPos.RealizedPnL-expectedPnL) > 0.01 {
		t.Errorf("Expected PnL %f, got %f", expectedPnL, closedPos.RealizedPnL)
	}
}

func TestPartialClosePosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open position
	pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 2.0, 10, false)
	
	// Partially close
	pm.ClosePosition("user1", "BTC-USD-PERP", 1.0)
	
	// Check remaining position
	position, err := pm.GetPosition("user1", "BTC-USD-PERP")
	if err != nil {
		t.Fatalf("Failed to get position: %v", err)
	}
	
	if position.Size != 1.0 {
		t.Errorf("Expected remaining size 1.0, got %f", position.Size)
	}
}

func TestLiquidationPrice(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
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
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open high leverage position
	position, _ := pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 50, true)
	
	// Check liquidation price
	// For 50x leverage, liquidation should be around 2% below entry
	expectedLiqPrice := 50000 * (1 - 1.0/50 + 0.005) // entry * (1 - 1/leverage + maintenance_margin)
	
	if math.Abs(position.LiquidationPrice-expectedLiqPrice) > 100 {
		t.Errorf("Expected liquidation price around %f, got %f", expectedLiqPrice, position.LiquidationPrice)
	}
}

func TestFundingRate(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   1 * time.Second, // Short interval for testing
		MaxLeverage:       100,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 51000
	market.IndexPrice = 50000 // Premium
	
	// Open positions
	pm.OpenPosition("long1", "BTC-USD-PERP", Buy, 1.0, 10, false)
	pm.OpenPosition("short1", "BTC-USD-PERP", Sell, 1.0, 10, false)
	
	// Wait for funding time
	time.Sleep(1100 * time.Millisecond)
	
	// Process funding
	err := pm.ProcessFunding()
	if err != nil {
		t.Fatalf("Failed to process funding: %v", err)
	}
	
	// Check funding rate was calculated
	if market.FundingRate == 0 {
		t.Error("Funding rate should not be zero with price premium")
	}
	
	// Check positions were charged/paid funding
	longPos, _ := pm.GetPosition("long1", "BTC-USD-PERP")
	shortPos, _ := pm.GetPosition("short1", "BTC-USD-PERP")
	
	// Long pays, short receives when mark > index
	if longPos.FundingPaid >= 0 {
		t.Error("Long position should pay funding when mark > index")
	}
	if shortPos.FundingPaid <= 0 {
		t.Error("Short position should receive funding when mark > index")
	}
}

func TestAddMarginToPosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open isolated position
	position, _ := pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 20, true)
	
	initialMargin := position.Margin
	initialLiqPrice := position.LiquidationPrice
	
	// Add margin
	err := pm.AddMarginToPosition("user1", "BTC-USD-PERP", 1000)
	if err != nil {
		t.Fatalf("Failed to add margin: %v", err)
	}
	
	// Get updated position
	position, _ = pm.GetPosition("user1", "BTC-USD-PERP")
	
	if position.Margin != initialMargin+1000 {
		t.Errorf("Expected margin %f, got %f", initialMargin+1000, position.Margin)
	}
	
	// Liquidation price should improve (lower for long)
	if position.LiquidationPrice >= initialLiqPrice {
		t.Error("Liquidation price should improve after adding margin")
	}
}

func TestRemoveMarginFromPosition(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open isolated position with low leverage (more margin)
	position, _ := pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 5, true)
	
	initialMargin := position.Margin
	
	// Try to remove some margin
	removeAmount := initialMargin * 0.3 // Remove 30%
	err := pm.RemoveMarginFromPosition("user1", "BTC-USD-PERP", removeAmount)
	if err != nil {
		t.Fatalf("Failed to remove margin: %v", err)
	}
	
	// Get updated position
	position, _ = pm.GetPosition("user1", "BTC-USD-PERP")
	
	if position.Margin >= initialMargin {
		t.Error("Margin should decrease after removal")
	}
}

func TestExcessiveLeverage(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       20,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Try to open position with excessive leverage
	_, err := pm.OpenPosition("user1", "BTC-USD-PERP", Buy, 1.0, 50, false)
	
	if err != ErrExcessiveLeverage {
		t.Error("Should reject excessive leverage")
	}
}

func TestMarkPriceUpdate(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	
	// Update mark price
	err := pm.UpdateMarkPrice("BTC-USD-PERP", 55000)
	if err != nil {
		t.Fatalf("Failed to update mark price: %v", err)
	}
	
	if market.MarkPrice != 55000 {
		t.Errorf("Expected mark price 55000, got %f", market.MarkPrice)
	}
	
	// Check 24h high/low
	if market.High24h != 55000 {
		t.Errorf("Expected high24h 55000, got %f", market.High24h)
	}
}

func TestInversePerpetual(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup inverse perpetual
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-INV",
		UnderlyingAsset:   "USD",
		QuoteAsset:        "BTC",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		ContractSize:      100, // $100 per contract
		IsInverse:         true,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open position
	position, _ := pm.OpenPosition("user1", "BTC-USD-INV", Buy, 10, 10, false)
	
	// Change price
	market.MarkPrice = 55000
	
	// Calculate PnL for inverse contract
	// PnL (BTC) = contracts * contractSize * (1/entry - 1/exit)
	expectedPnL := 10 * 100 * (1.0/50000 - 1.0/55000)
	actualPnL := pm.calculatePnL(position, 10, 55000)
	
	if math.Abs(actualPnL-expectedPnL) > 0.00001 {
		t.Errorf("Expected inverse PnL %f, got %f", expectedPnL, actualPnL)
	}
}

func TestGetAllPositions(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup multiple markets
	markets := []string{"BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"}
	
	for _, symbol := range markets {
		config := PerpMarketConfig{
			Symbol:            symbol,
			UnderlyingAsset:   symbol[:3],
			QuoteAsset:        "USD",
			FundingInterval:   8 * time.Hour,
			MaxLeverage:       100,
			ContractSize:      1,
			OracleSource:      "aggregate",
		}
		market, _ := pm.CreateMarket(config)
		market.MarkPrice = 50000
	}
	
	// Open multiple positions
	for _, symbol := range markets {
		pm.OpenPosition("user1", symbol, Buy, 1.0, 10, false)
	}
	
	// Get all positions
	positions := pm.GetAllPositions("user1")
	
	if len(positions) != 3 {
		t.Errorf("Expected 3 positions, got %d", len(positions))
	}
}

func TestFundingHistory(t *testing.T) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	// Setup
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   1 * time.Millisecond,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 51000
	market.IndexPrice = 50000
	
	// Process funding multiple times
	for i := 0; i < 5; i++ {
		time.Sleep(2 * time.Millisecond)
		pm.ProcessFunding()
	}
	
	// Get funding history
	history, err := pm.GetFundingHistory("BTC-USD-PERP", 3)
	if err != nil {
		t.Fatalf("Failed to get funding history: %v", err)
	}
	
	if len(history) != 3 {
		t.Errorf("Expected 3 funding records, got %d", len(history))
	}
}

func BenchmarkOpenPosition(b *testing.B) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		user := fmt.Sprintf("user%d", i)
		pm.OpenPosition(user, "BTC-USD-PERP", Buy, 1.0, 10, false)
	}
}

func BenchmarkProcessFunding(b *testing.B) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   1 * time.Nanosecond, // Immediate funding
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 51000
	market.IndexPrice = 50000
	
	// Open many positions
	for i := 0; i < 1000; i++ {
		user := fmt.Sprintf("user%d", i)
		side := Buy
		if i%2 == 0 {
			side = Sell
		}
		pm.OpenPosition(user, "BTC-USD-PERP", side, 1.0, 10, false)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.ProcessFunding()
	}
}

func BenchmarkLiquidationCheck(b *testing.B) {
	engine := &TradingEngine{}
	pm := NewPerpetualManager(engine)
	
	config := PerpMarketConfig{
		Symbol:            "BTC-USD-PERP",
		UnderlyingAsset:   "BTC",
		QuoteAsset:        "USD",
		FundingInterval:   8 * time.Hour,
		MaxLeverage:       100,
		MaintenanceMargin: 0.005,
		ContractSize:      1,
		OracleSource:      "aggregate",
	}
	
	market, _ := pm.CreateMarket(config)
	market.MarkPrice = 50000
	
	// Open many high-leverage positions
	for i := 0; i < 1000; i++ {
		user := fmt.Sprintf("user%d", i)
		pm.OpenPosition(user, "BTC-USD-PERP", Buy, 1.0, 50, false)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate price movement that might trigger liquidations
		price := 49000 + float64(i%2000)
		pm.checkLiquidations("BTC-USD-PERP", price)
	}
}