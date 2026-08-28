// Copyright 2024 Lux Industries Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example: Basic Spot Trading
//
// This example demonstrates the fundamental trading operations:
// - Connecting to trading venues
// - Fetching market data (orderbooks, tickers)
// - Placing market and limit orders
// - Checking balances and open orders
//
// Run with: go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	trading "github.com/luxfi/trading"
	_ "github.com/luxfi/trading/adapters" // Register adapters
	"github.com/shopspring/decimal"
)

func main() {
	// Create configuration
	config := trading.NewConfig()

	// Add LX DEX (native venue)
	config.WithNative("lx_dex", trading.NewLxDexConfig(
		getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
	).WithCredentials(
		os.Getenv("LX_DEX_KEY"),
		os.Getenv("LX_DEX_SECRET"),
	))

	// Add Binance via CCXT (optional)
	if key := os.Getenv("BINANCE_KEY"); key != "" {
		config.WithCcxt("binance", trading.NewCcxtConfig("binance").
			WithCredentials(key, os.Getenv("BINANCE_SECRET")).
			WithBaseURL(getEnv("CCXT_GATEWAY", "http://localhost:8080")))
	}

	// Enable smart routing (will pick best venue automatically)
	config.WithSmartRouting(true)

	// Basic risk limits
	config.Risk = trading.RiskConfig{
		Enabled:      true,
		MaxOrderSize: decimal.NewFromFloat(1000), // Max $1000 per order
	}

	// Create client
	client := trading.NewClient(config)

	// Connect with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Println("Connecting to venues...")
	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(context.Background())

	// List connected venues
	venues := client.Venues()
	log.Printf("Connected to %d venues:", len(venues))
	for _, v := range venues {
		log.Printf("  - %s (%s)", v.Name, v.VenueType)
	}

	// Use a background context for trading operations
	ctx = context.Background()

	// Example 1: Get ticker
	ticker, err := client.Ticker(ctx, "BTC-USDC")
	if err != nil {
		log.Printf("Failed to get ticker: %v", err)
	} else {
		printTicker(ticker)
	}

	// Example 2: Get orderbook
	book, err := client.Orderbook(ctx, "BTC-USDC")
	if err != nil {
		log.Printf("Failed to get orderbook: %v", err)
	} else {
		printOrderbook(book)
	}

	// Example 3: Get aggregated orderbook from all venues
	aggBook, err := client.AggregatedOrderbook(ctx, "BTC-USDC")
	if err != nil {
		log.Printf("Failed to get aggregated orderbook: %v", err)
	} else {
		printAggregatedOrderbook(aggBook)
	}

	// Example 4: Get balances
	balances, err := client.Balances(ctx)
	if err != nil {
		log.Printf("Failed to get balances: %v", err)
	} else {
		printBalances(balances)
	}

	// Example 5: Place a limit order (uncomment to execute)
	// This places a limit buy order 5% below current price
	/*
		if ticker.Bid != nil {
			limitPrice := ticker.Bid.Mul(decimal.NewFromFloat(0.95))
			order, err := client.LimitBuy(ctx, "BTC-USDC",
				decimal.NewFromFloat(0.001), // 0.001 BTC
				limitPrice,
			)
			if err != nil {
				log.Printf("Failed to place order: %v", err)
			} else {
				printOrder(order)
			}
		}
	*/

	// Example 6: Place a market order (uncomment to execute)
	/*
		order, err := client.Buy(ctx, "BTC-USDC", decimal.NewFromFloat(0.001))
		if err != nil {
			log.Printf("Failed to place market order: %v", err)
		} else {
			printOrder(order)
		}
	*/

	// Example 7: Get open orders
	openOrders, err := client.OpenOrders(ctx, "BTC-USDC")
	if err != nil {
		log.Printf("Failed to get open orders: %v", err)
	} else {
		log.Printf("\nOpen Orders: %d", len(openOrders))
		for _, o := range openOrders {
			printOrder(o)
		}
	}

	log.Println("\nBasic trading example complete.")
}

func printTicker(t trading.Ticker) {
	fmt.Printf("\n=== Ticker: %s (%s) ===\n", t.Symbol, t.Venue)
	if t.Bid != nil {
		fmt.Printf("  Bid:      %s\n", t.Bid.StringFixed(2))
	}
	if t.Ask != nil {
		fmt.Printf("  Ask:      %s\n", t.Ask.StringFixed(2))
	}
	if t.Last != nil {
		fmt.Printf("  Last:     %s\n", t.Last.StringFixed(2))
	}
	if spread := t.Spread(); spread != nil {
		fmt.Printf("  Spread:   %s\n", spread.StringFixed(4))
	}
	if t.Volume24H != nil {
		fmt.Printf("  Volume:   %s\n", t.Volume24H.StringFixed(2))
	}
}

func printOrderbook(book *trading.Orderbook) {
	fmt.Printf("\n=== Orderbook: %s (%s) ===\n", book.Symbol, book.Venue)
	fmt.Println("Asks:")
	for i := min(4, len(book.Asks)) - 1; i >= 0; i-- {
		a := book.Asks[i]
		fmt.Printf("  %s @ %s\n", a.Quantity.StringFixed(4), a.Price.StringFixed(2))
	}
	fmt.Println("  ---")
	fmt.Println("Bids:")
	for i := 0; i < min(5, len(book.Bids)); i++ {
		b := book.Bids[i]
		fmt.Printf("  %s @ %s\n", b.Quantity.StringFixed(4), b.Price.StringFixed(2))
	}

	if mid := book.MidPrice(); mid != nil {
		fmt.Printf("\nMid Price: %s\n", mid.StringFixed(2))
	}
	if spread := book.SpreadPercent(); spread != nil {
		fmt.Printf("Spread: %s%%\n", spread.StringFixed(4))
	}
}

func printAggregatedOrderbook(book *trading.AggregatedOrderbook) {
	fmt.Printf("\n=== Aggregated Orderbook: %s ===\n", book.Symbol)

	bestBid := book.BestBid()
	bestAsk := book.BestAsk()

	if bestBid != nil {
		fmt.Printf("Best Bid: %s @ %s (%s)\n",
			bestBid.Quantity.StringFixed(4),
			bestBid.Price.StringFixed(2),
			bestBid.Venue)
	}
	if bestAsk != nil {
		fmt.Printf("Best Ask: %s @ %s (%s)\n",
			bestAsk.Quantity.StringFixed(4),
			bestAsk.Price.StringFixed(2),
			bestAsk.Venue)
	}

	// Show aggregated levels
	fmt.Println("\nAggregated Bids:")
	bids := book.AggregatedBids()
	for i := 0; i < min(3, len(bids)); i++ {
		fmt.Printf("  %s @ %s\n", bids[i].Quantity.StringFixed(4), bids[i].Price.StringFixed(2))
	}

	fmt.Println("Aggregated Asks:")
	asks := book.AggregatedAsks()
	for i := 0; i < min(3, len(asks)); i++ {
		fmt.Printf("  %s @ %s\n", asks[i].Quantity.StringFixed(4), asks[i].Price.StringFixed(2))
	}
}

func printBalances(balances []trading.AggregatedBalance) {
	fmt.Println("\n=== Balances ===")
	for _, b := range balances {
		if b.Total().GreaterThan(decimal.Zero) {
			fmt.Printf("  %s: %s (free: %s, locked: %s)\n",
				b.Asset,
				b.Total().StringFixed(8),
				b.TotalFree.StringFixed(8),
				b.TotalLocked.StringFixed(8))

			// Show per-venue breakdown
			for _, vb := range b.ByVenue {
				fmt.Printf("    [%s] %s\n", vb.Venue, vb.Total().StringFixed(8))
			}
		}
	}
}

func printOrder(o trading.Order) {
	fmt.Printf("\n=== Order: %s ===\n", o.OrderID)
	fmt.Printf("  Symbol:   %s\n", o.Symbol)
	fmt.Printf("  Venue:    %s\n", o.Venue)
	fmt.Printf("  Side:     %s\n", o.Side)
	fmt.Printf("  Type:     %s\n", o.OrderType)
	fmt.Printf("  Status:   %s\n", o.Status)
	fmt.Printf("  Quantity: %s\n", o.Quantity.StringFixed(8))
	fmt.Printf("  Filled:   %s (%s%%)\n",
		o.FilledQuantity.StringFixed(8),
		o.FillPercent().StringFixed(1))
	if o.Price != nil {
		fmt.Printf("  Price:    %s\n", o.Price.StringFixed(2))
	}
	if o.AveragePrice != nil {
		fmt.Printf("  Avg Fill: %s\n", o.AveragePrice.StringFixed(2))
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
