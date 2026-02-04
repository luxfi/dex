package arbitrage

import (
	"context"
	"testing"
	"time"

	"github.com/shopspring/decimal"
)

func TestArbTypes(t *testing.T) {
	tests := []struct {
		name     string
		arbType  ArbType
		expected string
	}{
		{"Simple", ArbTypeSimple, "simple"},
		{"Triangular", ArbTypeTriangular, "triangular"},
		{"MultiHop", ArbTypeMultiHop, "multi_hop"},
		{"CEX-DEX", ArbTypeCEXDEX, "cex_dex"},
		{"FlashSwap", ArbTypeFlashSwap, "flash_swap"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.arbType) != tt.expected {
				t.Errorf("ArbType %s = %v, want %v", tt.name, tt.arbType, tt.expected)
			}
		})
	}
}

func TestPriceSource(t *testing.T) {
	source := PriceSource{
		ChainID:   "lux",
		Venue:     "lx_dex",
		Symbol:    "BTC-USDC",
		Bid:       decimal.NewFromInt(50000),
		Ask:       decimal.NewFromInt(50010),
		Liquidity: decimal.NewFromInt(100),
		Timestamp: time.Now(),
		Latency:   10 * time.Millisecond,
	}

	if source.ChainID != "lux" {
		t.Errorf("ChainID = %v, want lux", source.ChainID)
	}

	if source.Venue != "lx_dex" {
		t.Errorf("Venue = %v, want lx_dex", source.Venue)
	}

	spread := source.Ask.Sub(source.Bid)
	expectedSpread := decimal.NewFromInt(10)
	if !spread.Equal(expectedSpread) {
		t.Errorf("Spread = %v, want %v", spread, expectedSpread)
	}
}

func TestRoute(t *testing.T) {
	route := Route{
		ChainID:      "lux",
		Venue:        "lx_dex",
		Action:       "buy",
		TokenIn:      "USDC",
		TokenOut:     "BTC",
		AmountIn:     decimal.NewFromInt(50000),
		ExpectedOut:  decimal.NewFromFloat(1.0),
		MinAmountOut: decimal.NewFromFloat(0.99),
	}

	if route.Action != "buy" {
		t.Errorf("Action = %v, want buy", route.Action)
	}

	if !route.ExpectedOut.GreaterThan(route.MinAmountOut) {
		t.Error("ExpectedOut should be greater than MinAmountOut")
	}
}

func TestArbitrageOpportunity(t *testing.T) {
	now := time.Now()
	buySource := PriceSource{
		ChainID:   "lux",
		Venue:     "lx_dex",
		Symbol:    "BTC-USDC",
		Bid:       decimal.NewFromInt(49990),
		Ask:       decimal.NewFromInt(50000),
		Liquidity: decimal.NewFromInt(10),
		Timestamp: now,
	}

	sellSource := PriceSource{
		ChainID:   "ethereum",
		Venue:     "uniswap",
		Symbol:    "BTC-USDC",
		Bid:       decimal.NewFromInt(50200),
		Ask:       decimal.NewFromInt(50250),
		Liquidity: decimal.NewFromInt(10),
		Timestamp: now,
	}

	// Spread = sell bid - buy ask = 50200 - 50000 = 200
	spread := sellSource.Bid.Sub(buySource.Ask)
	spreadBps := spread.Div(buySource.Ask).Mul(decimal.NewFromInt(10000))

	opp := ArbitrageOpportunity{
		ID:            "test-opp-1",
		Type:          ArbTypeSimple,
		BuySource:     buySource,
		SellSource:    sellSource,
		SpreadBps:     spreadBps,
		EstimatedPnL:  decimal.NewFromInt(200),
		MaxSize:       decimal.NewFromInt(10),
		GasCostUSD:    decimal.NewFromFloat(0.50),
		BridgeCostUSD: decimal.NewFromInt(0),
		NetPnL:        decimal.NewFromFloat(199.50),
		Confidence:    0.9,
		ExpiresAt:     now.Add(5 * time.Second),
	}

	if opp.Type != ArbTypeSimple {
		t.Errorf("Type = %v, want simple", opp.Type)
	}

	if !opp.SpreadBps.GreaterThan(decimal.Zero) {
		t.Error("SpreadBps should be greater than 0")
	}

	// SpreadBps = 200/50000 * 10000 = 40 bps
	expectedBps := decimal.NewFromInt(40)
	if !opp.SpreadBps.Equal(expectedBps) {
		t.Errorf("SpreadBps = %v, want %v", opp.SpreadBps, expectedBps)
	}

	if !opp.NetPnL.GreaterThan(decimal.Zero) {
		t.Error("NetPnL should be positive")
	}

	if opp.Confidence <= 0 || opp.Confidence > 1 {
		t.Errorf("Confidence = %v, should be between 0 and 1", opp.Confidence)
	}
}

func TestDefaultScannerConfig(t *testing.T) {
	config := DefaultScannerConfig()

	if config.MinSpreadBps.LessThanOrEqual(decimal.Zero) {
		t.Error("MinSpreadBps should be positive")
	}

	if config.MinProfitUSD.LessThanOrEqual(decimal.Zero) {
		t.Error("MinProfitUSD should be positive")
	}

	if config.MaxPriceAge <= 0 {
		t.Error("MaxPriceAge should be positive")
	}

	if len(config.Symbols) == 0 {
		t.Error("Symbols should not be empty")
	}

	// Should include major tokens
	hasLux := false
	hasBTC := false
	for _, s := range config.Symbols {
		if s == "LUX" {
			hasLux = true
		}
		if s == "BTC" {
			hasBTC = true
		}
	}
	if !hasLux {
		t.Error("Symbols should include LUX")
	}
	if !hasBTC {
		t.Error("Symbols should include BTC")
	}
}

func TestDefaultUnifiedArbConfig(t *testing.T) {
	config := DefaultUnifiedArbConfig()

	if config.MinSpreadBps.LessThanOrEqual(decimal.Zero) {
		t.Error("MinSpreadBps should be positive")
	}

	if config.MinProfit.LessThanOrEqual(decimal.Zero) {
		t.Error("MinProfit should be positive")
	}

	if config.MaxPositionSize.LessThanOrEqual(decimal.Zero) {
		t.Error("MaxPositionSize should be positive")
	}

	if config.MaxTotalExposure.LessThanOrEqual(decimal.Zero) {
		t.Error("MaxTotalExposure should be positive")
	}

	if len(config.VenuePriority) == 0 {
		t.Error("VenuePriority should not be empty")
	}

	// LX DEX should be first priority
	if config.VenuePriority[0] != "lx_dex" {
		t.Errorf("First priority should be lx_dex, got %s", config.VenuePriority[0])
	}

	if config.ScanInterval <= 0 {
		t.Error("ScanInterval should be positive")
	}

	if config.ExecuteTimeout <= 0 {
		t.Error("ExecuteTimeout should be positive")
	}
}

func TestUnifiedArbitrageStats(t *testing.T) {
	client := &mockTradingClient{}
	config := DefaultUnifiedArbConfig()
	ua := NewUnifiedArbitrage(client, config)

	stats := ua.GetStats()

	if stats.TotalExecutions != 0 {
		t.Errorf("TotalExecutions = %d, want 0", stats.TotalExecutions)
	}

	if stats.SuccessfulExecutions != 0 {
		t.Errorf("SuccessfulExecutions = %d, want 0", stats.SuccessfulExecutions)
	}

	if !stats.TotalPnL.Equal(decimal.Zero) {
		t.Errorf("TotalPnL = %v, want 0", stats.TotalPnL)
	}

	if stats.WinRate != 0 {
		t.Errorf("WinRate = %v, want 0", stats.WinRate)
	}
}

func TestNewUnifiedArbitrage(t *testing.T) {
	client := &mockTradingClient{}
	config := DefaultUnifiedArbConfig()

	ua := NewUnifiedArbitrage(client, config)

	if ua == nil {
		t.Fatal("NewUnifiedArbitrage returned nil")
	}

	if ua.client == nil {
		t.Error("Client not set correctly")
	}
}

func TestUnifiedArbitrageStartStop(t *testing.T) {
	client := &mockTradingClient{}
	config := DefaultUnifiedArbConfig()
	config.ScanInterval = 100 * time.Millisecond

	ua := NewUnifiedArbitrage(client, config)

	err := ua.Start()
	if err != nil {
		t.Errorf("Start() error = %v", err)
	}

	// Let it run briefly
	time.Sleep(50 * time.Millisecond)

	ua.Stop()
}

func TestUnifiedArbitrageStartWithoutClient(t *testing.T) {
	config := DefaultUnifiedArbConfig()
	ua := &UnifiedArbitrage{
		config: config,
	}

	err := ua.Start()
	if err == nil {
		t.Error("Start() should error without client")
	}
}

// mockTradingClient implements TradingClient for testing
type mockTradingClient struct{}

func (m *mockTradingClient) AggregatedOrderbook(ctx context.Context, symbol string) (*AggregatedBook, error) {
	return &AggregatedBook{
		Symbol: symbol,
		Bids: []AggregatedLevel{
			{
				Price:     decimal.NewFromInt(50000),
				Quantity:  decimal.NewFromInt(10),
				Venue:     "lx_dex",
				Timestamp: time.Now(),
			},
		},
		Asks: []AggregatedLevel{
			{
				Price:     decimal.NewFromInt(50010),
				Quantity:  decimal.NewFromInt(10),
				Venue:     "binance",
				Timestamp: time.Now(),
			},
		},
	}, nil
}

func (m *mockTradingClient) PlaceOrder(ctx context.Context, req OrderRequest) (*Order, error) {
	return &Order{
		OrderID:        "mock-order-1",
		Symbol:         req.Symbol,
		Venue:          req.Venue,
		Side:           req.Side,
		Quantity:       req.Quantity,
		FilledQuantity: req.Quantity,
		AveragePrice:   *req.Price,
		Status:         "filled",
	}, nil
}

func (m *mockTradingClient) GetConnectedVenues() []VenueInfo {
	return []VenueInfo{
		{Name: "lx_dex", VenueType: "dex", Connected: true},
		{Name: "binance", VenueType: "cex", Connected: true},
	}
}
