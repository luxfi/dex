// Copyright 2024 Lux Industries Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example: Real-time Price Feeds via Polling
//
// This example demonstrates real-time market data streaming:
// - Continuous ticker updates
// - Orderbook depth tracking
// - Trade feed monitoring
// - Multi-symbol price aggregation
//
// Note: This example uses polling. For true WebSocket streaming,
// implement the StreamingAdapter interface in your venue adapter.
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

// PriceFeed tracks real-time prices across venues.
type PriceFeed struct {
	client *trading.Client

	// Configuration
	symbols      []string
	pollInterval time.Duration
	venues       []string

	// State
	mu          sync.RWMutex
	prices      map[string]SymbolPrices // symbol -> prices
	orderbooks  map[string]*trading.AggregatedOrderbook
	lastUpdated map[string]time.Time
	callbacks   []PriceCallback

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// SymbolPrices holds prices for a symbol across venues.
type SymbolPrices struct {
	Symbol    string
	Prices    map[string]VenuePrice // venue -> price
	BestBid   decimal.Decimal
	BestAsk   decimal.Decimal
	MidPrice  decimal.Decimal
	Spread    decimal.Decimal
	SpreadBps decimal.Decimal
	UpdatedAt time.Time
}

// VenuePrice holds price data from a specific venue.
type VenuePrice struct {
	Venue     string
	Bid       decimal.Decimal
	Ask       decimal.Decimal
	Last      decimal.Decimal
	Volume24H decimal.Decimal
	UpdatedAt time.Time
}

// PriceCallback is called when prices update.
type PriceCallback func(symbol string, prices SymbolPrices)

// NewPriceFeed creates a new price feed.
func NewPriceFeed(client *trading.Client, symbols []string, pollInterval time.Duration) *PriceFeed {
	ctx, cancel := context.WithCancel(context.Background())

	// Get venue names from client
	var venues []string
	for _, v := range client.Venues() {
		venues = append(venues, v.Name)
	}

	return &PriceFeed{
		client:       client,
		symbols:      symbols,
		pollInterval: pollInterval,
		venues:       venues,
		prices:       make(map[string]SymbolPrices),
		orderbooks:   make(map[string]*trading.AggregatedOrderbook),
		lastUpdated:  make(map[string]time.Time),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// OnPriceUpdate registers a callback for price updates.
func (pf *PriceFeed) OnPriceUpdate(cb PriceCallback) {
	pf.mu.Lock()
	pf.callbacks = append(pf.callbacks, cb)
	pf.mu.Unlock()
}

// Start begins the price feed.
func (pf *PriceFeed) Start() {
	log.Printf("Starting price feed for %v", pf.symbols)
	log.Printf("Poll interval: %s", pf.pollInterval)

	pf.wg.Add(2)
	go pf.tickerLoop()
	go pf.orderbookLoop()
}

// Stop stops the price feed.
func (pf *PriceFeed) Stop() {
	pf.cancel()
	pf.wg.Wait()
}

// tickerLoop polls for ticker updates.
func (pf *PriceFeed) tickerLoop() {
	defer pf.wg.Done()

	ticker := time.NewTicker(pf.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-pf.ctx.Done():
			return
		case <-ticker.C:
			pf.updateTickers()
		}
	}
}

// orderbookLoop polls for orderbook updates.
func (pf *PriceFeed) orderbookLoop() {
	defer pf.wg.Done()

	// Orderbooks update slightly slower
	ticker := time.NewTicker(pf.pollInterval * 2)
	defer ticker.Stop()

	for {
		select {
		case <-pf.ctx.Done():
			return
		case <-ticker.C:
			pf.updateOrderbooks()
		}
	}
}

// updateTickers fetches latest tickers.
func (pf *PriceFeed) updateTickers() {
	for _, symbol := range pf.symbols {
		tickers, err := pf.client.Tickers(pf.ctx, symbol)
		if err != nil {
			continue
		}

		sp := SymbolPrices{
			Symbol:    symbol,
			Prices:    make(map[string]VenuePrice),
			UpdatedAt: time.Now(),
		}

		var bestBid, bestAsk *decimal.Decimal

		for _, t := range tickers {
			vp := VenuePrice{
				Venue:     t.Venue,
				UpdatedAt: t.Timestamp,
			}

			if t.Bid != nil {
				vp.Bid = *t.Bid
				if bestBid == nil || t.Bid.GreaterThan(*bestBid) {
					bestBid = t.Bid
				}
			}
			if t.Ask != nil {
				vp.Ask = *t.Ask
				if bestAsk == nil || t.Ask.LessThan(*bestAsk) {
					bestAsk = t.Ask
				}
			}
			if t.Last != nil {
				vp.Last = *t.Last
			}
			if t.Volume24H != nil {
				vp.Volume24H = *t.Volume24H
			}

			sp.Prices[t.Venue] = vp
		}

		if bestBid != nil {
			sp.BestBid = *bestBid
		}
		if bestAsk != nil {
			sp.BestAsk = *bestAsk
		}
		if bestBid != nil && bestAsk != nil {
			sp.MidPrice = bestBid.Add(*bestAsk).Div(decimal.NewFromInt(2))
			sp.Spread = bestAsk.Sub(*bestBid)
			if !bestAsk.IsZero() {
				sp.SpreadBps = sp.Spread.Div(*bestAsk).Mul(decimal.NewFromInt(10000))
			}
		}

		pf.mu.Lock()
		pf.prices[symbol] = sp
		pf.lastUpdated[symbol] = time.Now()
		callbacks := append([]PriceCallback{}, pf.callbacks...)
		pf.mu.Unlock()

		// Notify callbacks
		for _, cb := range callbacks {
			cb(symbol, sp)
		}
	}
}

// updateOrderbooks fetches latest orderbooks.
func (pf *PriceFeed) updateOrderbooks() {
	for _, symbol := range pf.symbols {
		book, err := pf.client.AggregatedOrderbook(pf.ctx, symbol)
		if err != nil {
			continue
		}

		pf.mu.Lock()
		pf.orderbooks[symbol] = book
		pf.mu.Unlock()
	}
}

// GetPrice returns current prices for a symbol.
func (pf *PriceFeed) GetPrice(symbol string) (SymbolPrices, bool) {
	pf.mu.RLock()
	defer pf.mu.RUnlock()
	sp, ok := pf.prices[symbol]
	return sp, ok
}

// GetOrderbook returns current aggregated orderbook.
func (pf *PriceFeed) GetOrderbook(symbol string) (*trading.AggregatedOrderbook, bool) {
	pf.mu.RLock()
	defer pf.mu.RUnlock()
	book, ok := pf.orderbooks[symbol]
	return book, ok
}

// GetAllPrices returns all current prices.
func (pf *PriceFeed) GetAllPrices() map[string]SymbolPrices {
	pf.mu.RLock()
	defer pf.mu.RUnlock()

	result := make(map[string]SymbolPrices, len(pf.prices))
	for k, v := range pf.prices {
		result[k] = v
	}
	return result
}

// PriceDisplay formats prices for display.
type PriceDisplay struct {
	feed      *PriceFeed
	lastPrint map[string]decimal.Decimal
}

// NewPriceDisplay creates a price display.
func NewPriceDisplay(feed *PriceFeed) *PriceDisplay {
	return &PriceDisplay{
		feed:      feed,
		lastPrint: make(map[string]decimal.Decimal),
	}
}

// Print prints current prices.
func (pd *PriceDisplay) Print() {
	prices := pd.feed.GetAllPrices()

	fmt.Printf("\033[2J\033[H") // Clear screen
	fmt.Println("=== Real-time Prices ===")
	fmt.Printf("Updated: %s\n\n", time.Now().Format("15:04:05.000"))

	for _, symbol := range pd.feed.symbols {
		sp, ok := prices[symbol]
		if !ok {
			continue
		}

		// Determine direction
		direction := " "
		if last, ok := pd.lastPrint[symbol]; ok {
			if sp.MidPrice.GreaterThan(last) {
				direction = "^" // Up arrow
			} else if sp.MidPrice.LessThan(last) {
				direction = "v" // Down arrow
			}
		}
		pd.lastPrint[symbol] = sp.MidPrice

		fmt.Printf("%s %s\n", symbol, direction)
		fmt.Printf("  Mid:    $%s\n", sp.MidPrice.StringFixed(2))
		fmt.Printf("  Bid:    $%s\n", sp.BestBid.StringFixed(2))
		fmt.Printf("  Ask:    $%s\n", sp.BestAsk.StringFixed(2))
		fmt.Printf("  Spread: %s bps\n", sp.SpreadBps.StringFixed(2))

		// Per-venue prices
		for venue, vp := range sp.Prices {
			fmt.Printf("    [%s] bid=%s ask=%s\n",
				venue,
				vp.Bid.StringFixed(2),
				vp.Ask.StringFixed(2))
		}
		fmt.Println()
	}
}

func main() {
	// Create configuration
	config := trading.NewConfig()

	// Add LX DEX
	config.WithNative("lx_dex", trading.NewLxDexConfig(
		getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
	))

	// Add CCXT exchanges
	if key := os.Getenv("BINANCE_KEY"); key != "" {
		config.WithCcxt("binance", trading.NewCcxtConfig("binance").
			WithCredentials(key, os.Getenv("BINANCE_SECRET")))
	}

	if key := os.Getenv("MEXC_KEY"); key != "" {
		config.WithCcxt("mexc", trading.NewCcxtConfig("mexc").
			WithCredentials(key, os.Getenv("MEXC_SECRET")))
	}

	// Create client
	client := trading.NewClient(config)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(context.Background())

	// Create price feed
	symbols := []string{"BTC-USDC", "ETH-USDC", "LUX-USDC"}
	pollInterval := 500 * time.Millisecond

	feed := NewPriceFeed(client, symbols, pollInterval)

	// Add callback for price updates
	feed.OnPriceUpdate(func(symbol string, prices SymbolPrices) {
		// This callback fires on every update
		// You can use it to trigger trading logic
		if prices.SpreadBps.GreaterThan(decimal.NewFromInt(50)) {
			log.Printf("Wide spread on %s: %s bps", symbol, prices.SpreadBps.StringFixed(2))
		}
	})

	feed.Start()
	defer feed.Stop()

	// Display
	display := NewPriceDisplay(feed)

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Display ticker
	displayTicker := time.NewTicker(time.Second)
	defer displayTicker.Stop()

	log.Println("Price feed started. Press Ctrl+C to exit.")

	for {
		select {
		case <-sigCh:
			return
		case <-displayTicker.C:
			display.Print()
		}
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
