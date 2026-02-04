// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example: Market Making with Spreads
//
// This example demonstrates a simple market making strategy:
// - Placing bid/ask orders around the mid price
// - Managing order inventory
// - Adjusting spreads based on volatility
// - Position management and hedging
//
// WARNING: This is an educational example. Real market making
// requires much more sophisticated risk management.
//
// Run with: go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	trading "github.com/luxfi/trading"
	_ "github.com/luxfi/trading/adapters"
	"github.com/shopspring/decimal"
)

// MarketMaker implements a simple market making strategy.
type MarketMaker struct {
	client *trading.Client
	config MarketMakerConfig

	// State
	mu           sync.RWMutex
	position     decimal.Decimal
	activeOrders map[string]trading.Order
	totalPnL     decimal.Decimal
	tradesCount  int
	lastMidPrice decimal.Decimal
	volatility   decimal.Decimal

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// MarketMakerConfig configures the market maker.
type MarketMakerConfig struct {
	// Trading parameters
	Symbol          string
	Venue           string
	BaseSpreadBps   decimal.Decimal // Base spread in basis points
	OrderSize       decimal.Decimal // Size per order
	MaxPosition     decimal.Decimal // Maximum position (absolute)
	NumLevels       int             // Number of price levels
	LevelSpacingBps decimal.Decimal // Spacing between levels

	// Risk parameters
	MaxDailyLoss    decimal.Decimal
	PositionSkewBps decimal.Decimal // Skew spread based on position

	// Timing
	QuoteInterval    time.Duration
	CancelAgeSeconds int
}

// DefaultMarketMakerConfig returns sensible defaults.
func DefaultMarketMakerConfig() MarketMakerConfig {
	return MarketMakerConfig{
		Symbol:           "BTC-USDC",
		Venue:            "lx_dex",
		BaseSpreadBps:    decimal.NewFromInt(10),     // 0.10%
		OrderSize:        decimal.NewFromFloat(0.01), // 0.01 BTC
		MaxPosition:      decimal.NewFromFloat(0.1),  // 0.1 BTC
		NumLevels:        3,
		LevelSpacingBps:  decimal.NewFromInt(5),     // 0.05%
		MaxDailyLoss:     decimal.NewFromFloat(100), // $100
		PositionSkewBps:  decimal.NewFromInt(2),     // Skew per position unit
		QuoteInterval:    time.Second,
		CancelAgeSeconds: 30,
	}
}

// NewMarketMaker creates a new market maker.
func NewMarketMaker(client *trading.Client, config MarketMakerConfig) *MarketMaker {
	ctx, cancel := context.WithCancel(context.Background())
	return &MarketMaker{
		client:       client,
		config:       config,
		activeOrders: make(map[string]trading.Order),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// Start begins market making.
func (mm *MarketMaker) Start() {
	log.Printf("Starting market maker for %s on %s", mm.config.Symbol, mm.config.Venue)
	log.Printf("Config: spread=%sbps, size=%s, levels=%d",
		mm.config.BaseSpreadBps, mm.config.OrderSize, mm.config.NumLevels)

	mm.wg.Add(2)
	go mm.quoteLoop()
	go mm.monitorLoop()
}

// Stop stops market making and cancels all orders.
func (mm *MarketMaker) Stop() {
	log.Println("Stopping market maker...")
	mm.cancel()
	mm.wg.Wait()

	// Cancel all active orders
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := mm.client.CancelAllOrders(ctx, mm.config.Symbol, trading.WithVenue(mm.config.Venue))
	if err != nil {
		log.Printf("Error cancelling orders: %v", err)
	}

	mm.printStats()
}

// quoteLoop continuously updates quotes.
func (mm *MarketMaker) quoteLoop() {
	defer mm.wg.Done()

	ticker := time.NewTicker(mm.config.QuoteInterval)
	defer ticker.Stop()

	for {
		select {
		case <-mm.ctx.Done():
			return
		case <-ticker.C:
			mm.updateQuotes()
		}
	}
}

// updateQuotes places new bid/ask orders.
func (mm *MarketMaker) updateQuotes() {
	// Get current mid price
	ticker, err := mm.client.Ticker(mm.ctx, mm.config.Symbol, trading.WithVenue(mm.config.Venue))
	if err != nil {
		log.Printf("Failed to get ticker: %v", err)
		return
	}

	midPrice := ticker.MidPrice()
	if midPrice == nil {
		log.Println("No mid price available")
		return
	}

	mm.mu.Lock()
	mm.lastMidPrice = *midPrice
	position := mm.position
	mm.mu.Unlock()

	// Calculate spread with position skew
	// If long, widen ask (want to sell), tighten bid
	// If short, widen bid (want to buy), tighten ask
	halfSpread := mm.config.BaseSpreadBps.Div(decimal.NewFromInt(2))
	positionSkew := position.Mul(mm.config.PositionSkewBps)

	bidSpread := halfSpread.Add(positionSkew) // Wider if long
	askSpread := halfSpread.Sub(positionSkew) // Tighter if long

	// Ensure minimum spread
	minSpread := decimal.NewFromInt(1) // 0.01%
	if bidSpread.LessThan(minSpread) {
		bidSpread = minSpread
	}
	if askSpread.LessThan(minSpread) {
		askSpread = minSpread
	}

	// Cancel stale orders first
	mm.cancelStaleOrders()

	// Check position limits
	canBuy := position.LessThan(mm.config.MaxPosition)
	canSell := position.Neg().LessThan(mm.config.MaxPosition)

	// Place orders at multiple levels
	for level := 0; level < mm.config.NumLevels; level++ {
		levelOffset := mm.config.LevelSpacingBps.Mul(decimal.NewFromInt(int64(level)))

		// Bid price: mid * (1 - (bidSpread + levelOffset) / 10000)
		bidBps := bidSpread.Add(levelOffset)
		bidMultiplier := decimal.NewFromInt(1).Sub(bidBps.Div(decimal.NewFromInt(10000)))
		bidPrice := midPrice.Mul(bidMultiplier)

		// Ask price: mid * (1 + (askSpread + levelOffset) / 10000)
		askBps := askSpread.Add(levelOffset)
		askMultiplier := decimal.NewFromInt(1).Add(askBps.Div(decimal.NewFromInt(10000)))
		askPrice := midPrice.Mul(askMultiplier)

		// Place bid
		if canBuy {
			mm.placeOrder(trading.SideBuy, mm.config.OrderSize, bidPrice, level)
		}

		// Place ask
		if canSell {
			mm.placeOrder(trading.SideSell, mm.config.OrderSize, askPrice, level)
		}
	}
}

// placeOrder places a single order.
func (mm *MarketMaker) placeOrder(side trading.Side, quantity, price decimal.Decimal, level int) {
	order, err := mm.client.PlaceOrder(mm.ctx, trading.OrderRequest{
		ClientOrderID: fmt.Sprintf("mm-%s-%d-%d", side, level, time.Now().UnixNano()),
		Symbol:        mm.config.Symbol,
		Side:          side,
		OrderType:     trading.OrderTypeLimit,
		Quantity:      quantity,
		Price:         &price,
		TimeInForce:   trading.TimeInForceGTC,
		Venue:         mm.config.Venue,
	})
	if err != nil {
		log.Printf("Failed to place %s order at %s: %v", side, price.StringFixed(2), err)
		return
	}

	mm.mu.Lock()
	mm.activeOrders[order.OrderID] = order
	mm.mu.Unlock()

	log.Printf("Placed %s %s @ %s (order: %s)",
		side, quantity.StringFixed(4), price.StringFixed(2), order.OrderID)
}

// cancelStaleOrders cancels orders older than threshold.
func (mm *MarketMaker) cancelStaleOrders() {
	mm.mu.Lock()
	ordersToCancel := make([]string, 0)
	threshold := time.Now().Add(-time.Duration(mm.config.CancelAgeSeconds) * time.Second)

	for id, order := range mm.activeOrders {
		if order.CreatedAt.Before(threshold) {
			ordersToCancel = append(ordersToCancel, id)
		}
	}
	mm.mu.Unlock()

	for _, id := range ordersToCancel {
		_, err := mm.client.CancelOrder(mm.ctx, id, mm.config.Symbol, mm.config.Venue)
		if err != nil {
			log.Printf("Failed to cancel order %s: %v", id, err)
		} else {
			mm.mu.Lock()
			delete(mm.activeOrders, id)
			mm.mu.Unlock()
		}
	}
}

// monitorLoop monitors fills and updates position.
func (mm *MarketMaker) monitorLoop() {
	defer mm.wg.Done()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-mm.ctx.Done():
			return
		case <-ticker.C:
			mm.checkFills()
		}
	}
}

// checkFills checks for filled orders and updates position.
func (mm *MarketMaker) checkFills() {
	openOrders, err := mm.client.OpenOrders(mm.ctx, mm.config.Symbol)
	if err != nil {
		return
	}

	// Build map of current open orders
	openOrderIDs := make(map[string]bool)
	for _, o := range openOrders {
		openOrderIDs[o.OrderID] = true
	}

	// Check which active orders have been filled
	mm.mu.Lock()
	defer mm.mu.Unlock()

	for id, order := range mm.activeOrders {
		if !openOrderIDs[id] {
			// Order is no longer open - assume filled
			// In production, you would fetch the actual fill details
			delete(mm.activeOrders, id)
			mm.tradesCount++

			// Update position
			if order.Side == trading.SideBuy {
				mm.position = mm.position.Add(order.Quantity)
			} else {
				mm.position = mm.position.Sub(order.Quantity)
			}

			log.Printf("Order filled: %s %s %s (position: %s)",
				order.Side, order.Quantity.StringFixed(4), mm.config.Symbol,
				mm.position.StringFixed(4))
		}
	}
}

// printStats prints market maker statistics.
func (mm *MarketMaker) printStats() {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	fmt.Println("\n=== Market Maker Stats ===")
	fmt.Printf("Symbol:        %s\n", mm.config.Symbol)
	fmt.Printf("Venue:         %s\n", mm.config.Venue)
	fmt.Printf("Position:      %s\n", mm.position.StringFixed(8))
	fmt.Printf("Total Trades:  %d\n", mm.tradesCount)
	fmt.Printf("Active Orders: %d\n", len(mm.activeOrders))
	fmt.Printf("Last Mid:      %s\n", mm.lastMidPrice.StringFixed(2))
	fmt.Println("==========================")
}

func main() {
	// Create configuration
	config := trading.NewConfig()

	// Add venue
	venue := getEnv("MM_VENUE", "lx_dex")
	if venue == "lx_dex" {
		config.WithNative(venue, trading.NewLxDexConfig(
			getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
		).WithCredentials(
			os.Getenv("LX_DEX_KEY"),
			os.Getenv("LX_DEX_SECRET"),
		))
	} else {
		config.WithCcxt(venue, trading.NewCcxtConfig(venue).
			WithCredentials(
				os.Getenv(venue+"_KEY"),
				os.Getenv(venue+"_SECRET"),
			))
	}

	// Risk config
	config.Risk = trading.RiskConfig{
		Enabled:         true,
		MaxPositionSize: decimal.NewFromFloat(1),   // 1 BTC max position
		MaxOrderSize:    decimal.NewFromFloat(0.1), // 0.1 BTC max order
		MaxDailyLoss:    decimal.NewFromFloat(500), // $500 daily loss limit
		MaxOpenOrders:   50,
	}

	// Create client
	client := trading.NewClient(config)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(context.Background())

	// Create market maker
	mmConfig := DefaultMarketMakerConfig()
	mmConfig.Symbol = getEnv("MM_SYMBOL", "BTC-USDC")
	mmConfig.Venue = venue

	// Parse config from environment
	if spread := os.Getenv("MM_SPREAD_BPS"); spread != "" {
		mmConfig.BaseSpreadBps, _ = decimal.NewFromString(spread)
	}
	if size := os.Getenv("MM_ORDER_SIZE"); size != "" {
		mmConfig.OrderSize, _ = decimal.NewFromString(size)
	}

	mm := NewMarketMaker(client, mmConfig)
	mm.Start()

	// Wait for signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	<-sigCh
	mm.Stop()
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
