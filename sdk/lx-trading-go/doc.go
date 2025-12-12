// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package trading provides a unified high-frequency trading SDK with multi-venue support.

The SDK abstracts multiple trading venues behind a single interface:
  - Native LX DEX (Central Limit Order Book)
  - Native LX AMM (Automated Market Maker)
  - CCXT exchanges (Binance, MEXC, OKX, etc.) via HTTP gateway
  - Hummingbot Gateway connectors (Uniswap, PancakeSwap, etc.)

# Quick Start

Create a client with venue configurations:

	config := trading.NewConfig().
		WithNative("lx_dex", trading.NewLxDexConfig("https://api.lx.exchange")).
		WithCcxt("binance", trading.NewCcxtConfig("binance").
			WithCredentials(apiKey, apiSecret))

	client := trading.NewClient(config)
	ctx := context.Background()

	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Disconnect(ctx)

# Market Data

Get orderbooks from single or multiple venues:

	// Single venue orderbook
	book, err := client.Orderbook(ctx, "BTC-USDC", trading.WithVenue("lx_dex"))

	// Aggregated orderbook from all venues
	aggBook, err := client.AggregatedOrderbook(ctx, "BTC-USDC")

# Order Execution

Place orders with automatic smart routing or target specific venues:

	// Smart routed market buy
	order, err := client.Buy(ctx, "BTC-USDC", decimal.NewFromFloat(0.1))

	// Limit order on specific venue
	order, err := client.LimitBuy(ctx, "BTC-USDC",
		decimal.NewFromFloat(0.1),
		decimal.NewFromFloat(50000),
		trading.WithVenue("binance"))

# Execution Algorithms

Built-in execution algorithms for large orders:

	// TWAP execution over 10 minutes
	twap := trading.NewTwapExecutor(client, trading.TwapConfig{
		Symbol:          "BTC-USDC",
		Side:            trading.SideBuy,
		TotalQuantity:   decimal.NewFromFloat(10),
		DurationSeconds: 600,
		NumSlices:       20,
	})
	orders, err := twap.Execute(ctx)

# Risk Management

Built-in risk controls:

	config.Risk = trading.RiskConfig{
		Enabled:           true,
		MaxPositionSize:   decimal.NewFromFloat(100000),
		MaxOrderSize:      decimal.NewFromFloat(10000),
		MaxDailyLoss:      decimal.NewFromFloat(5000),
		KillSwitchEnabled: true,
	}

# AMM Operations

For AMM venues, swap and liquidity operations are available:

	// Get swap quote
	quote, err := client.Quote(ctx, "ETH", "USDC",
		decimal.NewFromFloat(1), true, "lx_amm")

	// Execute swap with slippage tolerance
	trade, err := client.Swap(ctx, "ETH", "USDC",
		decimal.NewFromFloat(1), true,
		decimal.NewFromFloat(0.005), "lx_amm")

# Thread Safety

The Client is safe for concurrent use. All methods properly synchronize
access to shared state using read-write mutexes.

# Error Handling

All errors are wrapped with context using fmt.Errorf with %w for proper
error chain inspection:

	if errors.Is(err, trading.ErrVenueNotFound) {
		// Handle venue not found
	}

	var tradingErr *trading.TradingError
	if errors.As(err, &tradingErr) {
		if tradingErr.IsRetryable() {
			// Retry the operation
		}
	}
*/
package trading
