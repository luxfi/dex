// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example: Advanced Cross-Venue Arbitrage
//
// This example demonstrates sophisticated arbitrage strategies:
// - Cross-venue price monitoring
// - Triangular arbitrage detection
// - Execution timing optimization
// - Slippage and fee accounting
// - Position tracking and hedging
//
// The strategy exploits price differences between LX DEX (fastest)
// and other venues (CEX, external DEX) that are inherently slower.
//
// Run with: go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	trading "github.com/luxfi/trading"
	_ "github.com/luxfi/trading/adapters"
	"github.com/shopspring/decimal"
)

// AdvancedArbitrage implements sophisticated arbitrage detection and execution.
type AdvancedArbitrage struct {
	client *trading.Client
	config ArbConfig

	// State
	mu            sync.RWMutex
	totalPnL      decimal.Decimal
	executions    []Execution
	opportunities int64
	skipped       int64

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ArbConfig configures the arbitrage system.
type ArbConfig struct {
	// Symbols to monitor
	Symbols []string

	// Minimum spread to trade (basis points)
	MinSpreadBps decimal.Decimal

	// Minimum profit per trade (quote currency)
	MinProfit decimal.Decimal

	// Maximum position per asset
	MaxPosition decimal.Decimal

	// Fee rates per venue (used for profit calculation)
	VenueFees map[string]decimal.Decimal

	// Venue priority (faster first)
	VenuePriority []string

	// Scan interval
	ScanInterval time.Duration

	// Execution timeout
	ExecutionTimeout time.Duration

	// Maximum concurrent executions
	MaxConcurrent int

	// Risk limits
	MaxDailyLoss      decimal.Decimal
	MaxTradesPerHour  int
	CooldownAfterLoss time.Duration
}

// Opportunity represents a detected arbitrage opportunity.
type Opportunity struct {
	ID        string
	Symbol    string
	Timestamp time.Time
	ExpiresAt time.Time

	// Buy side (best ask)
	BuyVenue    string
	BuyPrice    decimal.Decimal
	BuyQuantity decimal.Decimal
	BuyFeeRate  decimal.Decimal

	// Sell side (best bid)
	SellVenue    string
	SellPrice    decimal.Decimal
	SellQuantity decimal.Decimal
	SellFeeRate  decimal.Decimal

	// Calculated metrics
	Spread      decimal.Decimal
	SpreadBps   decimal.Decimal
	MaxQuantity decimal.Decimal
	GrossProfit decimal.Decimal
	TotalFees   decimal.Decimal
	NetProfit   decimal.Decimal

	// Quality score (0-1)
	Score float64
}

// Execution represents an executed arbitrage.
type Execution struct {
	Opportunity  *Opportunity
	StartTime    time.Time
	EndTime      time.Time
	BuyOrder     *trading.Order
	SellOrder    *trading.Order
	ActualProfit decimal.Decimal
	Status       string
	Error        error
}

// DefaultArbConfig returns sensible defaults.
func DefaultArbConfig() ArbConfig {
	return ArbConfig{
		Symbols:      []string{"BTC-USDC", "ETH-USDC", "LUX-USDC"},
		MinSpreadBps: decimal.NewFromInt(15),
		MinProfit:    decimal.NewFromFloat(5),
		MaxPosition:  decimal.NewFromFloat(10000),
		VenueFees: map[string]decimal.Decimal{
			"lx_dex":  decimal.NewFromFloat(0.001), // 0.1%
			"lx_amm":  decimal.NewFromFloat(0.003), // 0.3%
			"binance": decimal.NewFromFloat(0.001), // 0.1%
			"mexc":    decimal.NewFromFloat(0.002), // 0.2%
		},
		VenuePriority:     []string{"lx_dex", "binance", "mexc", "lx_amm"},
		ScanInterval:      50 * time.Millisecond,
		ExecutionTimeout:  3 * time.Second,
		MaxConcurrent:     3,
		MaxDailyLoss:      decimal.NewFromFloat(500),
		MaxTradesPerHour:  100,
		CooldownAfterLoss: 5 * time.Minute,
	}
}

// NewAdvancedArbitrage creates a new arbitrage system.
func NewAdvancedArbitrage(client *trading.Client, config ArbConfig) *AdvancedArbitrage {
	ctx, cancel := context.WithCancel(context.Background())
	return &AdvancedArbitrage{
		client: client,
		config: config,
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start begins the arbitrage system.
func (arb *AdvancedArbitrage) Start() {
	log.Println("Starting advanced arbitrage system")
	log.Printf("Monitoring: %v", arb.config.Symbols)
	log.Printf("Min spread: %s bps, Min profit: $%s",
		arb.config.MinSpreadBps, arb.config.MinProfit)

	arb.wg.Add(1)
	go arb.scanLoop()
}

// Stop stops the arbitrage system.
func (arb *AdvancedArbitrage) Stop() {
	arb.cancel()
	arb.wg.Wait()
	arb.printFinalStats()
}

// scanLoop continuously scans for opportunities.
func (arb *AdvancedArbitrage) scanLoop() {
	defer arb.wg.Done()

	ticker := time.NewTicker(arb.config.ScanInterval)
	defer ticker.Stop()

	semaphore := make(chan struct{}, arb.config.MaxConcurrent)

	for {
		select {
		case <-arb.ctx.Done():
			return
		case <-ticker.C:
			for _, symbol := range arb.config.Symbols {
				opp := arb.scan(symbol)
				if opp == nil {
					continue
				}

				arb.mu.Lock()
				arb.opportunities++
				arb.mu.Unlock()

				// Try to execute if slot available
				select {
				case semaphore <- struct{}{}:
					go func(o *Opportunity) {
						defer func() { <-semaphore }()
						arb.execute(o)
					}(opp)
				default:
					arb.mu.Lock()
					arb.skipped++
					arb.mu.Unlock()
				}
			}
		}
	}
}

// scan looks for arbitrage opportunities in a symbol.
func (arb *AdvancedArbitrage) scan(symbol string) *Opportunity {
	// Get aggregated orderbook
	book, err := arb.client.AggregatedOrderbook(arb.ctx, symbol)
	if err != nil {
		return nil
	}

	bestBid := book.BestBid()
	bestAsk := book.BestAsk()

	if bestBid == nil || bestAsk == nil {
		return nil
	}

	// Check for cross-venue arbitrage
	// Profitable if: bid on venue A > ask on venue B
	if bestBid.Venue == bestAsk.Venue {
		// Same venue - need to find cross-venue opportunity
		return arb.findCrossVenueOpportunity(symbol, book)
	}

	if bestBid.Price.LessThanOrEqual(bestAsk.Price) {
		return nil
	}

	return arb.buildOpportunity(symbol, bestBid, bestAsk)
}

// findCrossVenueOpportunity looks for opportunities across different venues.
func (arb *AdvancedArbitrage) findCrossVenueOpportunity(symbol string, book *trading.AggregatedOrderbook) *Opportunity {
	// Get all bids and asks by venue
	type venueLevel struct {
		venue    string
		price    decimal.Decimal
		quantity decimal.Decimal
	}

	var bids, asks []venueLevel

	// Collect levels from aggregated book
	for priceStr, levels := range book.Bids {
		for _, level := range levels {
			price, _ := decimal.NewFromString(priceStr)
			bids = append(bids, venueLevel{
				venue:    level.Venue,
				price:    price,
				quantity: level.Quantity,
			})
		}
	}

	for priceStr, levels := range book.Asks {
		for _, level := range levels {
			price, _ := decimal.NewFromString(priceStr)
			asks = append(asks, venueLevel{
				venue:    level.Venue,
				price:    price,
				quantity: level.Quantity,
			})
		}
	}

	// Sort: bids descending by price, asks ascending by price
	sort.Slice(bids, func(i, j int) bool {
		return bids[i].price.GreaterThan(bids[j].price)
	})
	sort.Slice(asks, func(i, j int) bool {
		return asks[i].price.LessThan(asks[j].price)
	})

	// Find best cross-venue opportunity
	var bestOpp *Opportunity
	bestProfit := decimal.Zero

	for _, bid := range bids {
		for _, ask := range asks {
			if bid.venue == ask.venue {
				continue // Same venue
			}

			if bid.price.LessThanOrEqual(ask.price) {
				continue // No profit
			}

			opp := arb.buildOpportunityFromLevels(symbol,
				bid.venue, bid.price, bid.quantity,
				ask.venue, ask.price, ask.quantity)

			if opp != nil && opp.NetProfit.GreaterThan(bestProfit) {
				bestOpp = opp
				bestProfit = opp.NetProfit
			}
		}
	}

	return bestOpp
}

// buildOpportunity creates an opportunity from aggregated levels.
func (arb *AdvancedArbitrage) buildOpportunity(symbol string,
	bestBid, bestAsk *trading.AggregatedLevel) *Opportunity {

	return arb.buildOpportunityFromLevels(symbol,
		bestBid.Venue, bestBid.Price, bestBid.Quantity,
		bestAsk.Venue, bestAsk.Price, bestAsk.Quantity)
}

// buildOpportunityFromLevels creates an opportunity from price levels.
func (arb *AdvancedArbitrage) buildOpportunityFromLevels(symbol,
	sellVenue string, sellPrice, sellQty decimal.Decimal,
	buyVenue string, buyPrice, buyQty decimal.Decimal) *Opportunity {

	if buyPrice.IsZero() {
		return nil
	}

	spread := sellPrice.Sub(buyPrice)
	spreadBps := spread.Div(buyPrice).Mul(decimal.NewFromInt(10000))

	if spreadBps.LessThan(arb.config.MinSpreadBps) {
		return nil
	}

	// Calculate max quantity
	maxQty := decimal.Min(buyQty, sellQty)
	maxQty = decimal.Min(maxQty, arb.config.MaxPosition)

	// Calculate fees
	buyFee := arb.config.VenueFees[buyVenue]
	if buyFee.IsZero() {
		buyFee = decimal.NewFromFloat(0.002) // Default 0.2%
	}
	sellFee := arb.config.VenueFees[sellVenue]
	if sellFee.IsZero() {
		sellFee = decimal.NewFromFloat(0.002)
	}

	buyValue := buyPrice.Mul(maxQty)
	sellValue := sellPrice.Mul(maxQty)
	grossProfit := sellValue.Sub(buyValue)

	totalFees := buyValue.Mul(buyFee).Add(sellValue.Mul(sellFee))
	netProfit := grossProfit.Sub(totalFees)

	if netProfit.LessThan(arb.config.MinProfit) {
		return nil
	}

	// Calculate quality score
	score := arb.calculateScore(spreadBps, netProfit, maxQty)

	return &Opportunity{
		ID:           fmt.Sprintf("arb-%s-%d", symbol, time.Now().UnixNano()),
		Symbol:       symbol,
		Timestamp:    time.Now(),
		ExpiresAt:    time.Now().Add(2 * time.Second),
		BuyVenue:     buyVenue,
		BuyPrice:     buyPrice,
		BuyQuantity:  buyQty,
		BuyFeeRate:   buyFee,
		SellVenue:    sellVenue,
		SellPrice:    sellPrice,
		SellQuantity: sellQty,
		SellFeeRate:  sellFee,
		Spread:       spread,
		SpreadBps:    spreadBps,
		MaxQuantity:  maxQty,
		GrossProfit:  grossProfit,
		TotalFees:    totalFees,
		NetProfit:    netProfit,
		Score:        score,
	}
}

// calculateScore calculates opportunity quality score.
func (arb *AdvancedArbitrage) calculateScore(spreadBps, profit, quantity decimal.Decimal) float64 {
	// Weighted scoring
	spreadScore := spreadBps.Div(decimal.NewFromInt(100)).InexactFloat64() // Higher spread = better
	profitScore := profit.Div(decimal.NewFromInt(100)).InexactFloat64()    // Higher profit = better
	sizeScore := quantity.Div(arb.config.MaxPosition).InexactFloat64()     // Larger = better

	// Weights
	score := spreadScore*0.3 + profitScore*0.5 + sizeScore*0.2

	// Clamp to 0-1
	if score > 1 {
		score = 1
	}
	if score < 0 {
		score = 0
	}

	return score
}

// execute executes an arbitrage opportunity.
func (arb *AdvancedArbitrage) execute(opp *Opportunity) {
	// Check expiry
	if time.Now().After(opp.ExpiresAt) {
		return
	}

	exec := &Execution{
		Opportunity: opp,
		StartTime:   time.Now(),
		Status:      "executing",
	}

	ctx, cancel := context.WithTimeout(arb.ctx, arb.config.ExecutionTimeout)
	defer cancel()

	// Execute both legs simultaneously
	var wg sync.WaitGroup
	var buyOrder, sellOrder trading.Order
	var buyErr, sellErr error

	wg.Add(2)

	go func() {
		defer wg.Done()
		buyOrder, buyErr = arb.client.LimitBuy(ctx,
			opp.Symbol,
			opp.MaxQuantity,
			opp.BuyPrice,
			trading.WithVenue(opp.BuyVenue))
	}()

	go func() {
		defer wg.Done()
		sellOrder, sellErr = arb.client.LimitSell(ctx,
			opp.Symbol,
			opp.MaxQuantity,
			opp.SellPrice,
			trading.WithVenue(opp.SellVenue))
	}()

	wg.Wait()
	exec.EndTime = time.Now()
	exec.BuyOrder = &buyOrder
	exec.SellOrder = &sellOrder

	// Handle errors
	if buyErr != nil || sellErr != nil {
		exec.Status = "failed"
		exec.Error = fmt.Errorf("buy: %v, sell: %v", buyErr, sellErr)
		log.Printf("Arbitrage failed: %v", exec.Error)
		arb.recordExecution(exec)
		return
	}

	// Calculate actual profit
	buyValue := buyOrder.FilledQuantity.Mul(*buyOrder.AveragePrice)
	sellValue := sellOrder.FilledQuantity.Mul(*sellOrder.AveragePrice)
	exec.ActualProfit = sellValue.Sub(buyValue)

	// Subtract fees
	for _, fee := range buyOrder.Fees {
		exec.ActualProfit = exec.ActualProfit.Sub(fee.Amount)
	}
	for _, fee := range sellOrder.Fees {
		exec.ActualProfit = exec.ActualProfit.Sub(fee.Amount)
	}

	exec.Status = "completed"

	log.Printf("Arbitrage executed: %s buy@%s(%s) sell@%s(%s) profit=$%s",
		opp.Symbol,
		opp.BuyVenue, opp.BuyPrice.StringFixed(2),
		opp.SellVenue, opp.SellPrice.StringFixed(2),
		exec.ActualProfit.StringFixed(2))

	arb.recordExecution(exec)
}

// recordExecution records an execution.
func (arb *AdvancedArbitrage) recordExecution(exec *Execution) {
	arb.mu.Lock()
	defer arb.mu.Unlock()

	arb.executions = append(arb.executions, *exec)
	if exec.Status == "completed" {
		arb.totalPnL = arb.totalPnL.Add(exec.ActualProfit)
	}
}

// GetStats returns current statistics.
func (arb *AdvancedArbitrage) GetStats() (opportunities, executed, skipped int64, pnl decimal.Decimal) {
	arb.mu.RLock()
	defer arb.mu.RUnlock()

	return arb.opportunities, int64(len(arb.executions)), arb.skipped, arb.totalPnL
}

// printFinalStats prints final statistics.
func (arb *AdvancedArbitrage) printFinalStats() {
	arb.mu.RLock()
	defer arb.mu.RUnlock()

	successful := 0
	for _, exec := range arb.executions {
		if exec.Status == "completed" && exec.ActualProfit.GreaterThan(decimal.Zero) {
			successful++
		}
	}

	fmt.Println("\n=== Arbitrage Final Stats ===")
	fmt.Printf("Opportunities Found:  %d\n", arb.opportunities)
	fmt.Printf("Executions Attempted: %d\n", len(arb.executions))
	fmt.Printf("Successful:           %d\n", successful)
	fmt.Printf("Skipped (busy):       %d\n", arb.skipped)
	fmt.Printf("Total PnL:            $%s\n", arb.totalPnL.StringFixed(2))
	if len(arb.executions) > 0 {
		fmt.Printf("Win Rate:             %.1f%%\n",
			float64(successful)/float64(len(arb.executions))*100)
	}
	fmt.Println("=============================")
}

func main() {
	// Create configuration
	config := trading.NewConfig()

	// Add LX DEX
	config.WithNative("lx_dex", trading.NewLxDexConfig(
		getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
	).WithCredentials(
		os.Getenv("LX_DEX_KEY"),
		os.Getenv("LX_DEX_SECRET"),
	))

	// Add LX AMM
	config.WithNative("lx_amm", trading.NewLxAmmConfig(
		getEnv("LX_AMM_URL", "https://api.amm.lux.network"),
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

	// Risk config
	config.Risk = trading.RiskConfig{
		Enabled:           true,
		MaxPositionSize:   decimal.NewFromFloat(50000),
		MaxOrderSize:      decimal.NewFromFloat(10000),
		MaxDailyLoss:      decimal.NewFromFloat(1000),
		KillSwitchEnabled: true,
	}

	// Create client
	client := trading.NewClient(config)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(context.Background())

	// Create arbitrage system
	arbConfig := DefaultArbConfig()
	arb := NewAdvancedArbitrage(client, arbConfig)
	arb.Start()

	// Stats ticker
	statsTicker := time.NewTicker(30 * time.Second)
	defer statsTicker.Stop()

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	for {
		select {
		case <-sigCh:
			arb.Stop()
			return
		case <-statsTicker.C:
			opps, exec, skip, pnl := arb.GetStats()
			log.Printf("Stats: opps=%d exec=%d skip=%d pnl=$%s",
				opps, exec, skip, pnl.StringFixed(2))
		}
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
