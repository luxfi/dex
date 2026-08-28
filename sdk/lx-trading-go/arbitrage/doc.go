// Copyright 2024 Lux Industries Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package arbitrage provides unified liquidity arbitrage capabilities.

This package implements cross-venue arbitrage detection and execution
using the unified trading SDK. The core strategy is "LX First" - since
LX DEX has the fastest price updates (nanosecond updates, 200ms blocks),
it serves as the price oracle while other venues are inherently stale.

# Architecture

The arbitrage system consists of three main components:

1. Scanner: Continuously monitors orderbooks across all connected venues
2. Opportunity Detector: Identifies profitable cross-venue spreads
3. Executor: Simultaneously executes buy and sell legs

# Quick Start

	// Create trading client (see trading package documentation)
	client := trading.NewClient(config)
	client.Connect(ctx)

	// Create adapter for arbitrage system
	arbClient := &ClientAdapter{Client: client}

	// Configure arbitrage
	arbConfig := arbitrage.UnifiedArbConfig{
		MinSpreadBps:     decimal.NewFromInt(10),   // 10 bps minimum
		MinProfit:        decimal.NewFromInt(5),    // $5 minimum profit
		MaxPositionSize:  decimal.NewFromInt(10000),
		Symbols:          []string{"BTC-USDC", "ETH-USDC"},
		VenuePriority:    []string{"lx_dex", "binance"},
		ScanInterval:     100 * time.Millisecond,
		ExecuteTimeout:   5 * time.Second,
	}

	arb := arbitrage.NewUnifiedArbitrage(arbClient, arbConfig)
	arb.Start()
	defer arb.Stop()

# Opportunity Detection

An opportunity is detected when:
  - Best bid on venue A > Best ask on venue B
  - Spread exceeds MinSpreadBps threshold
  - Net profit (after fees) exceeds MinProfit threshold
  - Position size is within risk limits

# Execution Strategy

When an opportunity is found, the system:
1. Validates the opportunity hasn't expired
2. Executes buy and sell orders simultaneously
3. Tracks fill rates and actual profit
4. Updates position and PnL tracking

# Risk Management

Built-in risk controls include:
  - Maximum position size per asset
  - Maximum total exposure across all assets
  - Maximum daily loss limit
  - Maximum trades per day
  - Per-trade profit threshold

# Statistics

Monitor arbitrage performance:

	stats := arb.GetStats()
	fmt.Printf("Executions: %d, Win Rate: %.1f%%, PnL: $%s\n",
		stats.TotalExecutions,
		stats.WinRate * 100,
		stats.TotalPnL.StringFixed(2))

# Implementing TradingClient

To use the arbitrage system, implement the TradingClient interface:

	type TradingClient interface {
		AggregatedOrderbook(ctx context.Context, symbol string) (*AggregatedBook, error)
		PlaceOrder(ctx context.Context, req OrderRequest) (*Order, error)
		GetConnectedVenues() []VenueInfo
	}
*/
package arbitrage
