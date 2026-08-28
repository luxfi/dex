// Copyright 2024 Lux Industries Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package adapters provides venue adapter implementations for the trading SDK.

This package implements the trading.VenueAdapter interface for various
trading venues:

# Native Adapters

LxDexAdapter connects to the native LX DEX (Central Limit Order Book):

	adapter := adapters.NewLxDexAdapter("lx_dex", &trading.NativeVenueConfig{
		VenueType: "dex",
		APIURL:    "https://api.lx.exchange",
		APIKey:    apiKey,
		APISecret: apiSecret,
	})

LxAmmAdapter connects to the native LX AMM:

	adapter := adapters.NewLxAmmAdapter("lx_amm", &trading.NativeVenueConfig{
		VenueType: "amm",
		APIURL:    "https://api.amm.lux.network",
	})

# CCXT Adapter

CcxtAdapter connects to CCXT-supported exchanges via an HTTP gateway.
This requires running a CCXT gateway service:

	adapter := adapters.NewCcxtAdapter("binance", &trading.CcxtConfig{
		ExchangeID: "binance",
		APIKey:     apiKey,
		APISecret:  apiSecret,
		BaseURL:    "http://localhost:8080", // CCXT gateway
	})

Supported exchanges include: Binance, MEXC, OKX, Kraken, Coinbase, and
any other exchange supported by CCXT.

# Hummingbot Gateway Adapter

HummingbotAdapter connects to DEXs via Hummingbot Gateway:

	adapter := adapters.NewHummingbotAdapter("uniswap", &trading.HummingbotConfig{
		Host:      "localhost",
		Port:      15888,
		Connector: "uniswap",
		Chain:     "ethereum",
		Network:   "mainnet",
	})

# Adapter Registration

Adapters are automatically registered with the trading package via init():

	import _ "github.com/luxfi/trading/adapters"

This enables the trading.Client to create adapters based on configuration.

# Implementing Custom Adapters

To implement a custom adapter, implement the trading.VenueAdapter interface:

	type VenueAdapter interface {
		Name() string
		VenueType() VenueType
		Capabilities() VenueCapabilities
		IsConnected() bool
		LatencyMs() *int64
		Info() VenueInfo
		Connect(ctx context.Context) error
		Disconnect(ctx context.Context) error
		GetMarkets(ctx context.Context) ([]MarketInfo, error)
		GetTicker(ctx context.Context, symbol string) (Ticker, error)
		GetOrderbook(ctx context.Context, symbol string, depth int) (*Orderbook, error)
		GetTrades(ctx context.Context, symbol string, limit int) ([]Trade, error)
		GetBalances(ctx context.Context) ([]Balance, error)
		GetBalance(ctx context.Context, asset string) (Balance, error)
		GetOpenOrders(ctx context.Context, symbol string) ([]Order, error)
		PlaceOrder(ctx context.Context, request OrderRequest) (Order, error)
		CancelOrder(ctx context.Context, orderID, symbol string) (Order, error)
		CancelAllOrders(ctx context.Context, symbol string) ([]Order, error)
		// AMM methods (return ErrNotSupported if not applicable)
		GetSwapQuote(...) (SwapQuote, error)
		ExecuteSwap(...) (Trade, error)
		GetPoolInfo(...) (PoolInfo, error)
		AddLiquidity(...) (LiquidityResult, error)
		RemoveLiquidity(...) (LiquidityResult, error)
		GetLpPositions(ctx context.Context) ([]LpPosition, error)
	}

Use trading.BaseAdapter to get default implementations of common methods.
*/
package adapters
