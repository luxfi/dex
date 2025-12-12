package trading

import (
	"testing"

	"github.com/shopspring/decimal"
)

func TestParseTradingPair(t *testing.T) {
	tests := []struct {
		input string
		base  string
		quote string
	}{
		{"BTC-USDC", "BTC", "USDC"},
		{"ETH-BTC", "ETH", "BTC"},
		{"BTC/USDT", "BTC", "USDT"},
		{"ETH_USD", "ETH", "USD"},
		{"BTCUSDT", "BTC", "USDT"},
		{"ETHUSDC", "ETH", "USDC"},
		{"LUXETH", "LUX", "ETH"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			pair := ParseTradingPair(tt.input)
			if pair.Base != tt.base {
				t.Errorf("Base: got %q, want %q", pair.Base, tt.base)
			}
			if pair.Quote != tt.quote {
				t.Errorf("Quote: got %q, want %q", pair.Quote, tt.quote)
			}
		})
	}
}

func TestTradingPairString(t *testing.T) {
	pair := TradingPair{Base: "BTC", Quote: "USDC"}

	if got := pair.String(); got != "BTC-USDC" {
		t.Errorf("String(): got %q, want %q", got, "BTC-USDC")
	}

	if got := pair.ToCCXT(); got != "BTC/USDC" {
		t.Errorf("ToCCXT(): got %q, want %q", got, "BTC/USDC")
	}
}

func TestOrderbook(t *testing.T) {
	book := NewOrderbook("BTC-USDC", "test")

	// Add bids (descending order expected after sort)
	book.AddBid(decimal.NewFromFloat(100.00), decimal.NewFromFloat(1.0))
	book.AddBid(decimal.NewFromFloat(99.00), decimal.NewFromFloat(2.0))
	book.AddBid(decimal.NewFromFloat(101.00), decimal.NewFromFloat(0.5))

	// Add asks (ascending order expected after sort)
	book.AddAsk(decimal.NewFromFloat(102.00), decimal.NewFromFloat(1.0))
	book.AddAsk(decimal.NewFromFloat(103.00), decimal.NewFromFloat(2.0))
	book.AddAsk(decimal.NewFromFloat(101.50), decimal.NewFromFloat(0.5))

	book.Sort()

	// Check best bid
	bestBid := book.BestBid()
	if bestBid == nil || !bestBid.Equal(decimal.NewFromFloat(101.00)) {
		t.Errorf("BestBid: got %v, want 101.00", bestBid)
	}

	// Check best ask
	bestAsk := book.BestAsk()
	if bestAsk == nil || !bestAsk.Equal(decimal.NewFromFloat(101.50)) {
		t.Errorf("BestAsk: got %v, want 101.50", bestAsk)
	}

	// Check spread
	spread := book.Spread()
	expected := decimal.NewFromFloat(0.50)
	if spread == nil || !spread.Equal(expected) {
		t.Errorf("Spread: got %v, want %v", spread, expected)
	}

	// Check mid price
	mid := book.MidPrice()
	expectedMid := decimal.NewFromFloat(101.25)
	if mid == nil || !mid.Equal(expectedMid) {
		t.Errorf("MidPrice: got %v, want %v", mid, expectedMid)
	}
}

func TestOrderbookVWAP(t *testing.T) {
	book := NewOrderbook("BTC-USDC", "test")

	// Asks for buy VWAP: 100@1, 101@2, 102@3
	book.AddAsk(decimal.NewFromFloat(100.00), decimal.NewFromFloat(1.0))
	book.AddAsk(decimal.NewFromFloat(101.00), decimal.NewFromFloat(2.0))
	book.AddAsk(decimal.NewFromFloat(102.00), decimal.NewFromFloat(3.0))
	book.Sort()

	// VWAP for buying 3 units: (100*1 + 101*2) / 3 = 302/3 = 100.6667
	vwap := book.VwapBuy(decimal.NewFromFloat(3.0))
	if vwap == nil {
		t.Fatal("VwapBuy returned nil")
	}

	expected := decimal.NewFromFloat(100.6667)
	if vwap.Sub(expected).Abs().GreaterThan(decimal.NewFromFloat(0.001)) {
		t.Errorf("VwapBuy(3): got %v, want ~%v", vwap, expected)
	}
}

func TestRiskManager(t *testing.T) {
	config := RiskConfig{
		Enabled:       true,
		MaxOrderSize:  decimal.NewFromFloat(10.0),
		MaxOpenOrders: 5,
	}

	rm := NewRiskManager(config)

	// Valid order
	request := OrderRequest{
		Symbol:   "BTC-USDC",
		Side:     SideBuy,
		Quantity: decimal.NewFromFloat(5.0),
	}

	if err := rm.ValidateOrder(request); err != nil {
		t.Errorf("ValidateOrder should pass: %v", err)
	}

	// Order too large
	request.Quantity = decimal.NewFromFloat(15.0)
	if err := rm.ValidateOrder(request); err == nil {
		t.Error("ValidateOrder should fail for large order")
	}

	// Kill switch
	rm.Kill()
	request.Quantity = decimal.NewFromFloat(1.0)
	if err := rm.ValidateOrder(request); err == nil {
		t.Error("ValidateOrder should fail when killed")
	}

	rm.Reset()
	if err := rm.ValidateOrder(request); err != nil {
		t.Errorf("ValidateOrder should pass after reset: %v", err)
	}
}

func TestRiskManagerPositions(t *testing.T) {
	config := RiskConfig{
		Enabled:         true,
		MaxPositionSize: decimal.NewFromFloat(100.0),
	}

	rm := NewRiskManager(config)

	// Update position
	rm.UpdatePosition("BTC", decimal.NewFromFloat(50.0), SideBuy)
	pos := rm.Position("BTC")
	if !pos.Equal(decimal.NewFromFloat(50.0)) {
		t.Errorf("Position: got %v, want 50", pos)
	}

	// Update position sell
	rm.UpdatePosition("BTC", decimal.NewFromFloat(20.0), SideSell)
	pos = rm.Position("BTC")
	if !pos.Equal(decimal.NewFromFloat(30.0)) {
		t.Errorf("Position after sell: got %v, want 30", pos)
	}
}

func TestBlackScholes(t *testing.T) {
	// Test call option
	price := BlackScholes(100, 100, 1, 0.05, 0.2, true)
	if price < 10 || price > 15 {
		t.Errorf("Call price: got %.4f, expected ~10-15", price)
	}

	// Test put option
	price = BlackScholes(100, 100, 1, 0.05, 0.2, false)
	if price < 5 || price > 10 {
		t.Errorf("Put price: got %.4f, expected ~5-10", price)
	}

	// Test at expiry
	price = BlackScholes(110, 100, 0, 0.05, 0.2, true)
	if price != 10 {
		t.Errorf("Call at expiry: got %.4f, want 10", price)
	}

	price = BlackScholes(90, 100, 0, 0.05, 0.2, false)
	if price != 10 {
		t.Errorf("Put at expiry: got %.4f, want 10", price)
	}
}

func TestGreeks(t *testing.T) {
	greeks := CalculateGreeks(100, 100, 1, 0.05, 0.2, true)

	// Delta for ATM call should be around 0.5-0.6
	if greeks.Delta < 0.5 || greeks.Delta > 0.7 {
		t.Errorf("Delta: got %.4f, expected ~0.5-0.7", greeks.Delta)
	}

	// Gamma should be positive
	if greeks.Gamma <= 0 {
		t.Errorf("Gamma: got %.4f, expected > 0", greeks.Gamma)
	}

	// Vega should be positive
	if greeks.Vega <= 0 {
		t.Errorf("Vega: got %.4f, expected > 0", greeks.Vega)
	}
}

func TestConstantProductPrice(t *testing.T) {
	reserveX := decimal.NewFromFloat(1000)
	reserveY := decimal.NewFromFloat(1000)
	amountIn := decimal.NewFromFloat(10)
	feeRate := decimal.NewFromFloat(0.003)

	output, price := ConstantProductPrice(reserveX, reserveY, amountIn, feeRate, true)

	// With 0.3% fee: amountInWithFee = 10 * 0.997 = 9.97
	// output = 1000 * 9.97 / (1000 + 9.97) = 9970 / 1009.97 = ~9.87
	if output.LessThan(decimal.NewFromFloat(9.8)) || output.GreaterThan(decimal.NewFromFloat(9.9)) {
		t.Errorf("Output: got %v, expected ~9.87", output)
	}

	if price.LessThanOrEqual(decimal.Zero) {
		t.Error("Price should be positive")
	}
}

func TestVolatility(t *testing.T) {
	returns := []float64{0.01, -0.02, 0.015, -0.01, 0.02}

	vol := Volatility(returns, false, 252)
	if vol <= 0 {
		t.Errorf("Volatility should be positive, got %.4f", vol)
	}

	// Annualized should be larger
	volAnn := Volatility(returns, true, 252)
	if volAnn <= vol {
		t.Errorf("Annualized vol should be larger: got %.4f <= %.4f", volAnn, vol)
	}
}

func TestSharpeRatio(t *testing.T) {
	// Positive returns
	returns := []float64{0.01, 0.02, 0.015, 0.01, 0.02}
	sharpe := SharpeRatio(returns, 0.0, 252)

	if sharpe <= 0 {
		t.Errorf("Sharpe ratio should be positive for positive returns, got %.4f", sharpe)
	}

	// Negative returns
	returns = []float64{-0.01, -0.02, -0.015, -0.01, -0.02}
	sharpe = SharpeRatio(returns, 0.0, 252)

	if sharpe >= 0 {
		t.Errorf("Sharpe ratio should be negative for negative returns, got %.4f", sharpe)
	}
}

func TestMaxDrawdown(t *testing.T) {
	prices := []float64{100, 110, 105, 120, 90, 95, 100}

	dd, peakIdx, troughIdx := MaxDrawdown(prices)

	// Peak at 120, trough at 90, drawdown = (120-90)/120 = 0.25
	if dd < 0.24 || dd > 0.26 {
		t.Errorf("MaxDrawdown: got %.4f, expected ~0.25", dd)
	}

	if peakIdx != 3 { // 120 is at index 3
		t.Errorf("Peak index: got %d, expected 3", peakIdx)
	}

	if troughIdx != 4 { // 90 is at index 4
		t.Errorf("Trough index: got %d, expected 4", troughIdx)
	}
}

func TestVaR(t *testing.T) {
	// Generate some returns
	returns := make([]float64, 100)
	for i := range returns {
		returns[i] = float64(i-50) / 1000 // -0.05 to 0.05
	}

	var95 := VaR(returns, 0.95, "historical")
	if var95 <= 0 {
		t.Errorf("VaR should be positive, got %.4f", var95)
	}

	var99 := VaR(returns, 0.99, "historical")
	if var99 <= var95 {
		t.Errorf("99%% VaR should be larger than 95%% VaR: %.4f <= %.4f", var99, var95)
	}
}

func TestConfig(t *testing.T) {
	config := NewConfig().
		WithNative("lx_dex", NewLxDexConfig("https://api.lx.exchange")).
		WithCcxt("binance", NewCcxtConfig("binance").WithCredentials("key", "secret")).
		WithHummingbot("uniswap", NewHummingbotConfig("uniswap_v3")).
		WithSmartRouting(true).
		WithVenuePriority("lx_dex", "binance")

	if len(config.Native) != 1 {
		t.Errorf("Native venues: got %d, want 1", len(config.Native))
	}

	if len(config.Ccxt) != 1 {
		t.Errorf("CCXT venues: got %d, want 1", len(config.Ccxt))
	}

	if len(config.Hummingbot) != 1 {
		t.Errorf("Hummingbot venues: got %d, want 1", len(config.Hummingbot))
	}

	if !config.General.SmartRouting {
		t.Error("SmartRouting should be true")
	}

	if len(config.General.VenuePriority) != 2 {
		t.Errorf("VenuePriority: got %d, want 2", len(config.General.VenuePriority))
	}
}

func TestOrder(t *testing.T) {
	order := Order{
		OrderID:        "123",
		Symbol:         "BTC-USDC",
		Status:         OrderStatusOpen,
		Quantity:       decimal.NewFromFloat(10.0),
		FilledQuantity: decimal.NewFromFloat(5.0),
	}

	if !order.IsOpen() {
		t.Error("Order should be open")
	}

	if order.IsDone() {
		t.Error("Order should not be done")
	}

	fillPct := order.FillPercent()
	if !fillPct.Equal(decimal.NewFromFloat(50)) {
		t.Errorf("FillPercent: got %v, want 50", fillPct)
	}

	// Test filled order
	order.Status = OrderStatusFilled
	if order.IsOpen() {
		t.Error("Filled order should not be open")
	}
	if !order.IsDone() {
		t.Error("Filled order should be done")
	}
}

func TestNewMarketOrder(t *testing.T) {
	order := NewMarketOrder("BTC-USDC", SideBuy, decimal.NewFromFloat(1.0))

	if order.Symbol != "BTC-USDC" {
		t.Errorf("Symbol: got %s, want BTC-USDC", order.Symbol)
	}
	if order.Side != SideBuy {
		t.Errorf("Side: got %s, want buy", order.Side)
	}
	if order.OrderType != OrderTypeMarket {
		t.Errorf("OrderType: got %s, want market", order.OrderType)
	}
	if order.ClientOrderID == "" {
		t.Error("ClientOrderID should be set")
	}
}

func TestNewLimitOrder(t *testing.T) {
	price := decimal.NewFromFloat(50000)
	order := NewLimitOrder("BTC-USDC", SideSell, decimal.NewFromFloat(0.5), price)

	if order.OrderType != OrderTypeLimit {
		t.Errorf("OrderType: got %s, want limit", order.OrderType)
	}
	if order.Price == nil || !order.Price.Equal(price) {
		t.Errorf("Price: got %v, want %v", order.Price, price)
	}
	if order.TimeInForce != TimeInForceGTC {
		t.Errorf("TimeInForce: got %s, want GTC", order.TimeInForce)
	}
}

func TestAggregatedOrderbook(t *testing.T) {
	agg := NewAggregatedOrderbook("BTC-USDC")

	// Add first orderbook
	book1 := NewOrderbook("BTC-USDC", "venue1")
	book1.AddBid(decimal.NewFromFloat(100), decimal.NewFromFloat(1))
	book1.AddBid(decimal.NewFromFloat(99), decimal.NewFromFloat(2))
	book1.AddAsk(decimal.NewFromFloat(101), decimal.NewFromFloat(1))
	book1.AddAsk(decimal.NewFromFloat(102), decimal.NewFromFloat(2))
	book1.Sort()
	agg.AddOrderbook(book1)

	// Add second orderbook
	book2 := NewOrderbook("BTC-USDC", "venue2")
	book2.AddBid(decimal.NewFromFloat(100.5), decimal.NewFromFloat(0.5))
	book2.AddAsk(decimal.NewFromFloat(100.8), decimal.NewFromFloat(0.5))
	book2.Sort()
	agg.AddOrderbook(book2)

	// Best bid should be from venue2
	bestBid := agg.BestBid()
	if bestBid == nil {
		t.Fatal("BestBid returned nil")
	}
	if !bestBid.Price.Equal(decimal.NewFromFloat(100.5)) {
		t.Errorf("BestBid price: got %v, want 100.5", bestBid.Price)
	}
	if bestBid.Venue != "venue2" {
		t.Errorf("BestBid venue: got %s, want venue2", bestBid.Venue)
	}

	// Best ask should be from venue2
	bestAsk := agg.BestAsk()
	if bestAsk == nil {
		t.Fatal("BestAsk returned nil")
	}
	if !bestAsk.Price.Equal(decimal.NewFromFloat(100.8)) {
		t.Errorf("BestAsk price: got %v, want 100.8", bestAsk.Price)
	}
}
