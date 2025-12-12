// Package arbitrage provides unified liquidity arbitrage through the SDK
package arbitrage

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

/*
UNIFIED LIQUIDITY ARBITRAGE - LX FIRST STRATEGY

Since LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks),
it becomes the price ORACLE. Other venues are always stale by comparison.

Architecture:
1. LX DEX prices are the TRUTH (most current)
2. Other venues (CEX, external DEX) are STALE
3. Arbitrage = exploiting stale venues before they catch up
4. LX always wins because it sees/moves prices first

NO SMART CONTRACTS - just coordinated trades through unified SDK.
*/

// TradingClient interface for the unified trading client
type TradingClient interface {
	// AggregatedOrderbook returns combined orderbook from all venues
	AggregatedOrderbook(ctx context.Context, symbol string) (*AggregatedBook, error)

	// PlaceOrder places an order on a specific venue
	PlaceOrder(ctx context.Context, req OrderRequest) (*Order, error)

	// GetConnectedVenues returns list of connected venues
	GetConnectedVenues() []VenueInfo
}

// OrderRequest represents an order to place
type OrderRequest struct {
	Symbol    string
	Side      Side
	OrderType OrderType
	Quantity  decimal.Decimal
	Price     *decimal.Decimal
	Venue     string
}

// Side represents buy or sell
type Side string

const (
	SideBuy  Side = "buy"
	SideSell Side = "sell"
)

// OrderType represents order type
type OrderType string

const (
	OrderTypeMarket OrderType = "market"
	OrderTypeLimit  OrderType = "limit"
)

// Order represents an executed order
type Order struct {
	OrderID        string
	Symbol         string
	Venue          string
	Side           Side
	Quantity       decimal.Decimal
	FilledQuantity decimal.Decimal
	AveragePrice   decimal.Decimal
	Fees           []Fee
	Status         string
}

// Fee represents a trading fee
type Fee struct {
	Asset  string
	Amount decimal.Decimal
}

// VenueInfo represents venue information
type VenueInfo struct {
	Name      string
	VenueType string
	Connected bool
}

// AggregatedLevel represents a price level from the aggregated orderbook
type AggregatedLevel struct {
	Price     decimal.Decimal
	Quantity  decimal.Decimal
	Venue     string
	Timestamp time.Time
}

// AggregatedBook represents an aggregated orderbook
type AggregatedBook struct {
	Symbol string
	Bids   []AggregatedLevel
	Asks   []AggregatedLevel
}

// BestBid returns the best bid (highest)
func (b *AggregatedBook) BestBid() *AggregatedLevel {
	if len(b.Bids) == 0 {
		return nil
	}
	return &b.Bids[0]
}

// BestAsk returns the best ask (lowest)
func (b *AggregatedBook) BestAsk() *AggregatedLevel {
	if len(b.Asks) == 0 {
		return nil
	}
	return &b.Asks[0]
}

// UnifiedArbitrage orchestrates arbitrage across all SDK-connected venues
type UnifiedArbitrage struct {
	mu sync.RWMutex

	// The unified trading client
	client TradingClient

	// Configuration
	config UnifiedArbConfig

	// State
	totalPnL      decimal.Decimal
	executions    []UnifiedExecution
	opportunities chan *UnifiedOpportunity

	// Running
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// UnifiedArbConfig configures the unified arbitrage system
type UnifiedArbConfig struct {
	// Minimum spread to trade (basis points)
	MinSpreadBps decimal.Decimal

	// Minimum profit per trade (in quote currency)
	MinProfit decimal.Decimal

	// Maximum position size per asset
	MaxPositionSize decimal.Decimal

	// Maximum total exposure
	MaxTotalExposure decimal.Decimal

	// Trading pairs to monitor
	Symbols []string

	// Venue priority for execution (faster venues first)
	VenuePriority []string

	// Scan interval
	ScanInterval time.Duration

	// Execute timeout
	ExecuteTimeout time.Duration

	// Risk limits
	MaxDailyLoss    decimal.Decimal
	MaxTradesPerDay int
}

// UnifiedOpportunity represents an arbitrage opportunity across venues
type UnifiedOpportunity struct {
	ID        string
	Symbol    string
	Timestamp time.Time
	ExpiresAt time.Time

	// Buy side (lowest ask)
	BuyVenue string
	BuyPrice decimal.Decimal
	BuySize  decimal.Decimal

	// Sell side (highest bid)
	SellVenue string
	SellPrice decimal.Decimal
	SellSize  decimal.Decimal

	// Calculated values
	Spread      decimal.Decimal
	SpreadBps   decimal.Decimal
	MaxSize     decimal.Decimal
	GrossProfit decimal.Decimal
	EstFees     decimal.Decimal
	NetProfit   decimal.Decimal

	// Quality metrics
	Confidence float64
	Latency    time.Duration
}

// UnifiedExecution represents an executed arbitrage
type UnifiedExecution struct {
	ID           string
	Opportunity  *UnifiedOpportunity
	StartTime    time.Time
	EndTime      time.Time
	Status       string
	BuyOrder     *Order
	SellOrder    *Order
	ActualProfit decimal.Decimal
	Fees         decimal.Decimal
	Error        error
}

// NewUnifiedArbitrage creates a new unified arbitrage system
func NewUnifiedArbitrage(client TradingClient, config UnifiedArbConfig) *UnifiedArbitrage {
	ctx, cancel := context.WithCancel(context.Background())

	return &UnifiedArbitrage{
		client:        client,
		config:        config,
		opportunities: make(chan *UnifiedOpportunity, 1000),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start begins the arbitrage system
func (ua *UnifiedArbitrage) Start() error {
	if ua.client == nil {
		return fmt.Errorf("client not configured")
	}

	ua.wg.Add(2)
	go ua.scanLoop()
	go ua.executeLoop()

	return nil
}

// Stop stops the arbitrage system
func (ua *UnifiedArbitrage) Stop() {
	ua.cancel()
	ua.wg.Wait()
}

// scanLoop continuously scans for opportunities
func (ua *UnifiedArbitrage) scanLoop() {
	defer ua.wg.Done()

	ticker := time.NewTicker(ua.config.ScanInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ua.ctx.Done():
			return
		case <-ticker.C:
			ua.scan()
		}
	}
}

// scan looks for arbitrage opportunities across all venues
func (ua *UnifiedArbitrage) scan() {
	for _, symbol := range ua.config.Symbols {
		opp := ua.findOpportunity(symbol)
		if opp != nil && opp.NetProfit.GreaterThan(ua.config.MinProfit) {
			select {
			case ua.opportunities <- opp:
			default:
			}
		}
	}
}

// findOpportunity finds the best arbitrage opportunity for a symbol
func (ua *UnifiedArbitrage) findOpportunity(symbol string) *UnifiedOpportunity {
	book, err := ua.client.AggregatedOrderbook(ua.ctx, symbol)
	if err != nil {
		return nil
	}

	bestBid := book.BestBid()
	bestAsk := book.BestAsk()

	if bestBid == nil || bestAsk == nil {
		return nil
	}

	// Cross-venue arbitrage: bid on one venue > ask on another
	if bestBid.Price.LessThanOrEqual(bestAsk.Price) {
		return nil
	}

	spread := bestBid.Price.Sub(bestAsk.Price)
	spreadBps := spread.Div(bestAsk.Price).Mul(decimal.NewFromInt(10000))

	if spreadBps.LessThan(ua.config.MinSpreadBps) {
		return nil
	}

	maxSize := decimal.Min(bestBid.Quantity, bestAsk.Quantity)
	maxSize = decimal.Min(maxSize, ua.config.MaxPositionSize)

	grossProfit := spread.Mul(maxSize)
	totalFees := bestAsk.Price.Mul(maxSize).Mul(decimal.NewFromFloat(0.002)) // ~0.2% total fees
	netProfit := grossProfit.Sub(totalFees)

	return &UnifiedOpportunity{
		ID:          fmt.Sprintf("arb-%s-%d", symbol, time.Now().UnixNano()),
		Symbol:      symbol,
		Timestamp:   time.Now(),
		ExpiresAt:   time.Now().Add(5 * time.Second),
		BuyVenue:    bestAsk.Venue,
		BuyPrice:    bestAsk.Price,
		BuySize:     bestAsk.Quantity,
		SellVenue:   bestBid.Venue,
		SellPrice:   bestBid.Price,
		SellSize:    bestBid.Quantity,
		Spread:      spread,
		SpreadBps:   spreadBps,
		MaxSize:     maxSize,
		GrossProfit: grossProfit,
		EstFees:     totalFees,
		NetProfit:   netProfit,
		Confidence:  0.8,
		Latency:     time.Since(bestAsk.Timestamp),
	}
}

// executeLoop processes opportunities
func (ua *UnifiedArbitrage) executeLoop() {
	defer ua.wg.Done()

	for {
		select {
		case <-ua.ctx.Done():
			return
		case opp := <-ua.opportunities:
			ua.execute(opp)
		}
	}
}

// execute executes an arbitrage opportunity
func (ua *UnifiedArbitrage) execute(opp *UnifiedOpportunity) {
	if time.Now().After(opp.ExpiresAt) {
		return
	}

	exec := &UnifiedExecution{
		ID:          opp.ID,
		Opportunity: opp,
		StartTime:   time.Now(),
		Status:      "executing",
	}

	ctx, cancel := context.WithTimeout(ua.ctx, ua.config.ExecuteTimeout)
	defer cancel()

	var wg sync.WaitGroup
	var buyErr, sellErr error
	var buyOrder, sellOrder *Order

	// Execute both legs simultaneously
	wg.Add(2)

	go func() {
		defer wg.Done()
		buyOrder, buyErr = ua.client.PlaceOrder(ctx, OrderRequest{
			Symbol:    opp.Symbol,
			Side:      SideBuy,
			OrderType: OrderTypeLimit,
			Quantity:  opp.MaxSize,
			Price:     &opp.BuyPrice,
			Venue:     opp.BuyVenue,
		})
	}()

	go func() {
		defer wg.Done()
		sellOrder, sellErr = ua.client.PlaceOrder(ctx, OrderRequest{
			Symbol:    opp.Symbol,
			Side:      SideSell,
			OrderType: OrderTypeLimit,
			Quantity:  opp.MaxSize,
			Price:     &opp.SellPrice,
			Venue:     opp.SellVenue,
		})
	}()

	wg.Wait()
	exec.EndTime = time.Now()
	exec.BuyOrder = buyOrder
	exec.SellOrder = sellOrder

	if buyErr != nil || sellErr != nil {
		exec.Status = "failed"
		exec.Error = fmt.Errorf("buy: %v, sell: %v", buyErr, sellErr)
		return
	}

	// Calculate actual profit
	if buyOrder != nil && sellOrder != nil {
		buyValue := buyOrder.AveragePrice.Mul(buyOrder.FilledQuantity)
		sellValue := sellOrder.AveragePrice.Mul(sellOrder.FilledQuantity)
		exec.ActualProfit = sellValue.Sub(buyValue)

		for _, fee := range buyOrder.Fees {
			exec.Fees = exec.Fees.Add(fee.Amount)
		}
		for _, fee := range sellOrder.Fees {
			exec.Fees = exec.Fees.Add(fee.Amount)
		}
		exec.ActualProfit = exec.ActualProfit.Sub(exec.Fees)
	}

	exec.Status = "completed"

	ua.mu.Lock()
	ua.totalPnL = ua.totalPnL.Add(exec.ActualProfit)
	ua.executions = append(ua.executions, *exec)
	ua.mu.Unlock()
}

// GetStats returns arbitrage statistics
func (ua *UnifiedArbitrage) GetStats() UnifiedArbStats {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	successful := 0
	for _, exec := range ua.executions {
		if exec.Status == "completed" && exec.ActualProfit.GreaterThan(decimal.Zero) {
			successful++
		}
	}

	winRate := float64(0)
	if len(ua.executions) > 0 {
		winRate = float64(successful) / float64(len(ua.executions))
	}

	return UnifiedArbStats{
		TotalExecutions:      len(ua.executions),
		SuccessfulExecutions: successful,
		TotalPnL:             ua.totalPnL,
		WinRate:              winRate,
	}
}

// UnifiedArbStats holds arbitrage statistics
type UnifiedArbStats struct {
	TotalExecutions      int
	SuccessfulExecutions int
	TotalPnL             decimal.Decimal
	WinRate              float64
}

// DefaultUnifiedArbConfig returns default configuration
func DefaultUnifiedArbConfig() UnifiedArbConfig {
	return UnifiedArbConfig{
		MinSpreadBps:     decimal.NewFromInt(10),
		MinProfit:        decimal.NewFromInt(5),
		MaxPositionSize:  decimal.NewFromInt(10000),
		MaxTotalExposure: decimal.NewFromInt(100000),
		Symbols:          []string{"BTC-USDC", "ETH-USDC", "LUX-USDC"},
		VenuePriority:    []string{"lx_dex", "binance", "mexc", "lx_amm"},
		ScanInterval:     100 * time.Millisecond,
		ExecuteTimeout:   5 * time.Second,
		MaxDailyLoss:     decimal.NewFromInt(1000),
		MaxTradesPerDay:  100,
	}
}
