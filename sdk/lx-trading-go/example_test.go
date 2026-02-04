// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trading_test

import (
	"fmt"

	trading "github.com/luxfi/trading"
	"github.com/shopspring/decimal"
)

// ExampleParseTradingPair demonstrates parsing various trading pair formats.
func ExampleParseTradingPair() {
	// The SDK supports multiple trading pair formats
	pairs := []string{
		"BTC-USDC", // Dash separator
		"ETH/USDT", // Slash separator (CCXT style)
		"LUX_USD",  // Underscore separator
		"BTCUSDT",  // No separator (exchange native)
	}

	for _, p := range pairs {
		pair := trading.ParseTradingPair(p)
		fmt.Printf("%s -> Base: %s, Quote: %s\n", p, pair.Base, pair.Quote)
	}
	// Output:
	// BTC-USDC -> Base: BTC, Quote: USDC
	// ETH/USDT -> Base: ETH, Quote: USDT
	// LUX_USD -> Base: LUX, Quote: USD
	// BTCUSDT -> Base: BTC, Quote: USDT
}

// ExampleTradingPair demonstrates TradingPair operations.
func ExampleTradingPair() {
	pair := trading.TradingPair{Base: "BTC", Quote: "USDC"}

	// Default string format uses dash separator
	fmt.Println("String:", pair.String())

	// CCXT format uses slash separator
	fmt.Println("CCXT:", pair.ToCCXT())
	// Output:
	// String: BTC-USDC
	// CCXT: BTC/USDC
}

// ExampleOrderbook demonstrates orderbook operations.
func ExampleOrderbook() {
	book := trading.NewOrderbook("BTC-USDC", "lx_dex")

	// Add bid levels (buy orders)
	book.AddBid(decimal.NewFromFloat(50000), decimal.NewFromFloat(1.0))
	book.AddBid(decimal.NewFromFloat(49900), decimal.NewFromFloat(2.0))
	book.AddBid(decimal.NewFromFloat(49800), decimal.NewFromFloat(3.0))

	// Add ask levels (sell orders)
	book.AddAsk(decimal.NewFromFloat(50100), decimal.NewFromFloat(0.5))
	book.AddAsk(decimal.NewFromFloat(50200), decimal.NewFromFloat(1.5))
	book.AddAsk(decimal.NewFromFloat(50300), decimal.NewFromFloat(2.5))

	// Sort to ensure proper ordering
	book.Sort()

	// Get market metrics
	fmt.Printf("Best Bid: %s\n", book.BestBid().StringFixed(2))
	fmt.Printf("Best Ask: %s\n", book.BestAsk().StringFixed(2))
	fmt.Printf("Mid Price: %s\n", book.MidPrice().StringFixed(2))
	fmt.Printf("Spread: %s\n", book.Spread().StringFixed(2))
	// Output:
	// Best Bid: 50000.00
	// Best Ask: 50100.00
	// Mid Price: 50050.00
	// Spread: 100.00
}

// ExampleOrderbook_VwapBuy demonstrates VWAP calculation for buying.
func ExampleOrderbook_VwapBuy() {
	book := trading.NewOrderbook("BTC-USDC", "lx_dex")

	// Add ask levels: we buy from asks
	book.AddAsk(decimal.NewFromFloat(50000), decimal.NewFromFloat(1.0))
	book.AddAsk(decimal.NewFromFloat(50100), decimal.NewFromFloat(2.0))
	book.AddAsk(decimal.NewFromFloat(50200), decimal.NewFromFloat(3.0))
	book.Sort()

	// Calculate VWAP for buying 2.5 BTC
	// Will sweep: 1.0 @ 50000, 1.5 @ 50100
	// VWAP = (1.0*50000 + 1.5*50100) / 2.5 = 125150 / 2.5 = 50060
	vwap := book.VwapBuy(decimal.NewFromFloat(2.5))
	fmt.Printf("VWAP for buying 2.5 BTC: %s\n", vwap.StringFixed(2))
	// Output:
	// VWAP for buying 2.5 BTC: 50060.00
}

// ExampleNewMarketOrder demonstrates creating a market order.
func ExampleNewMarketOrder() {
	order := trading.NewMarketOrder("BTC-USDC", trading.SideBuy, decimal.NewFromFloat(0.5))

	fmt.Printf("Symbol: %s\n", order.Symbol)
	fmt.Printf("Side: %s\n", order.Side)
	fmt.Printf("Type: %s\n", order.OrderType)
	fmt.Printf("Quantity: %s\n", order.Quantity.StringFixed(4))
	// Output:
	// Symbol: BTC-USDC
	// Side: buy
	// Type: market
	// Quantity: 0.5000
}

// ExampleNewLimitOrder demonstrates creating a limit order.
func ExampleNewLimitOrder() {
	price := decimal.NewFromFloat(50000)
	order := trading.NewLimitOrder("BTC-USDC", trading.SideSell, decimal.NewFromFloat(1.0), price)

	fmt.Printf("Symbol: %s\n", order.Symbol)
	fmt.Printf("Side: %s\n", order.Side)
	fmt.Printf("Type: %s\n", order.OrderType)
	fmt.Printf("Quantity: %s\n", order.Quantity.StringFixed(4))
	fmt.Printf("Price: %s\n", order.Price.StringFixed(2))
	fmt.Printf("TimeInForce: %s\n", order.TimeInForce)
	// Output:
	// Symbol: BTC-USDC
	// Side: sell
	// Type: limit
	// Quantity: 1.0000
	// Price: 50000.00
	// TimeInForce: GTC
}

// ExampleRiskManager demonstrates risk management operations.
func ExampleRiskManager() {
	config := trading.RiskConfig{
		Enabled:       true,
		MaxOrderSize:  decimal.NewFromFloat(10),
		MaxOpenOrders: 5,
	}

	rm := trading.NewRiskManager(config)

	// Validate a valid order
	order := trading.OrderRequest{
		Symbol:   "BTC-USDC",
		Side:     trading.SideBuy,
		Quantity: decimal.NewFromFloat(5),
	}

	if err := rm.ValidateOrder(order); err != nil {
		fmt.Println("Order rejected:", err)
	} else {
		fmt.Println("Order accepted")
	}

	// Try an oversized order
	order.Quantity = decimal.NewFromFloat(15)
	if err := rm.ValidateOrder(order); err != nil {
		fmt.Println("Large order rejected")
	}
	// Output:
	// Order accepted
	// Large order rejected
}

// Example_killSwitch demonstrates the kill switch functionality.
func Example_killSwitch() {
	config := trading.RiskConfig{
		Enabled: true,
	}

	rm := trading.NewRiskManager(config)

	fmt.Println("Kill switch active:", rm.IsKilled())

	// Activate kill switch
	rm.Kill()
	fmt.Println("After Kill(): active:", rm.IsKilled())

	// Any order will be rejected
	order := trading.OrderRequest{
		Symbol:   "BTC-USDC",
		Side:     trading.SideBuy,
		Quantity: decimal.NewFromFloat(1),
	}
	if err := rm.ValidateOrder(order); err != nil {
		fmt.Println("Order rejected due to kill switch")
	}

	// Reset to resume trading
	rm.Reset()
	fmt.Println("After Reset(): active:", rm.IsKilled())
	// Output:
	// Kill switch active: false
	// After Kill(): active: true
	// Order rejected due to kill switch
	// After Reset(): active: false
}

// ExampleConfig demonstrates SDK configuration.
func ExampleConfig() {
	config := trading.NewConfig().
		WithNative("lx_dex", trading.NewLxDexConfig("https://api.lx.exchange")).
		WithCcxt("binance", trading.NewCcxtConfig("binance").
			WithCredentials("api_key", "api_secret")).
		WithSmartRouting(true).
		WithVenuePriority("lx_dex", "binance")

	fmt.Printf("Native venues: %d\n", len(config.Native))
	fmt.Printf("CCXT venues: %d\n", len(config.Ccxt))
	fmt.Printf("Smart routing: %v\n", config.General.SmartRouting)
	// Output:
	// Native venues: 1
	// CCXT venues: 1
	// Smart routing: true
}

// ExampleBlackScholes demonstrates options pricing.
func ExampleBlackScholes() {
	// Price a call option:
	// Spot = $100, Strike = $100, Time = 1 year
	// Risk-free rate = 5%, Volatility = 20%
	callPrice := trading.BlackScholes(100, 100, 1, 0.05, 0.2, true)
	putPrice := trading.BlackScholes(100, 100, 1, 0.05, 0.2, false)

	fmt.Printf("ATM Call Price: $%.2f\n", callPrice)
	fmt.Printf("ATM Put Price: $%.2f\n", putPrice)
	// Output:
	// ATM Call Price: $10.45
	// ATM Put Price: $5.57
}

// ExampleCalculateGreeks demonstrates Greek calculations.
func ExampleCalculateGreeks() {
	// Calculate Greeks for an ATM call option
	greeks := trading.CalculateGreeks(100, 100, 1, 0.05, 0.2, true)

	fmt.Printf("Delta: %.4f\n", greeks.Delta)
	fmt.Printf("Gamma: %.4f\n", greeks.Gamma)
	fmt.Printf("Vega: %.4f\n", greeks.Vega)
	fmt.Printf("Theta: %.4f\n", greeks.Theta)
	// Output:
	// Delta: 0.6368
	// Gamma: 0.0188
	// Vega: 0.3752
	// Theta: -0.0176
}

// ExampleConstantProductPrice demonstrates AMM pricing.
func ExampleConstantProductPrice() {
	// Pool with 1000 ETH and 2,000,000 USDC
	reserveETH := decimal.NewFromFloat(1000)
	reserveUSDC := decimal.NewFromFloat(2000000)
	amountIn := decimal.NewFromFloat(10)   // Swap 10 ETH
	feeRate := decimal.NewFromFloat(0.003) // 0.3% fee

	// Swap ETH -> USDC
	output, price := trading.ConstantProductPrice(
		reserveETH, reserveUSDC, amountIn, feeRate, true)

	fmt.Printf("Swap 10 ETH for: %s USDC\n", output.StringFixed(2))
	fmt.Printf("Effective price: %s USDC/ETH\n", price.StringFixed(2))
	// Output:
	// Swap 10 ETH for: 19743.16 USDC
	// Effective price: 1974.32 USDC/ETH
}

// ExampleVolatility demonstrates volatility calculation.
func ExampleVolatility() {
	// Daily returns over 5 days
	returns := []float64{0.01, -0.02, 0.015, -0.01, 0.02}

	// Non-annualized volatility
	vol := trading.Volatility(returns, false, 252)
	fmt.Printf("Daily volatility: %.4f\n", vol)

	// Annualized volatility (252 trading days)
	volAnnual := trading.Volatility(returns, true, 252)
	fmt.Printf("Annual volatility: %.4f\n", volAnnual)
	// Output:
	// Daily volatility: 0.0172
	// Annual volatility: 0.2727
}

// ExampleMaxDrawdown demonstrates maximum drawdown calculation.
func ExampleMaxDrawdown() {
	// Price series with a drawdown
	prices := []float64{100, 110, 105, 120, 90, 95, 100}

	dd, peakIdx, troughIdx := trading.MaxDrawdown(prices)

	fmt.Printf("Maximum Drawdown: %.2f%%\n", dd*100)
	fmt.Printf("Peak at index %d (price: %.0f)\n", peakIdx, prices[peakIdx])
	fmt.Printf("Trough at index %d (price: %.0f)\n", troughIdx, prices[troughIdx])
	// Output:
	// Maximum Drawdown: 25.00%
	// Peak at index 3 (price: 120)
	// Trough at index 4 (price: 90)
}

// ExampleTradingError demonstrates error handling.
func ExampleTradingError() {
	err := trading.NewTradingError("RATE_LIMITED", "too many requests").
		WithVenue("binance").
		WithRetryable()

	fmt.Println("Error:", err.Error())
	fmt.Println("Retryable:", err.IsRetryable())
	// Output:
	// Error: [binance] RATE_LIMITED: too many requests
	// Retryable: true
}

// ExampleNewAggregatedOrderbook demonstrates multi-venue orderbook aggregation.
func ExampleNewAggregatedOrderbook() {
	agg := trading.NewAggregatedOrderbook("BTC-USDC")

	// Add orderbook from venue 1
	book1 := trading.NewOrderbook("BTC-USDC", "lx_dex")
	book1.AddBid(decimal.NewFromFloat(50000), decimal.NewFromFloat(1.0))
	book1.AddAsk(decimal.NewFromFloat(50100), decimal.NewFromFloat(1.0))
	book1.Sort()
	agg.AddOrderbook(book1)

	// Add orderbook from venue 2 with better prices
	book2 := trading.NewOrderbook("BTC-USDC", "binance")
	book2.AddBid(decimal.NewFromFloat(50050), decimal.NewFromFloat(0.5)) // Higher bid
	book2.AddAsk(decimal.NewFromFloat(50080), decimal.NewFromFloat(0.5)) // Lower ask
	book2.Sort()
	agg.AddOrderbook(book2)

	// Best prices across venues
	bestBid := agg.BestBid()
	bestAsk := agg.BestAsk()

	fmt.Printf("Best Bid: %s @ %s\n", bestBid.Price.StringFixed(2), bestBid.Venue)
	fmt.Printf("Best Ask: %s @ %s\n", bestAsk.Price.StringFixed(2), bestAsk.Venue)
	// Output:
	// Best Bid: 50050.00 @ binance
	// Best Ask: 50080.00 @ binance
}
