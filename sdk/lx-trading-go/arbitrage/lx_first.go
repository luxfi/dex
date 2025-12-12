// Package arbitrage implements LX-first arbitrage strategy
package arbitrage

import (
	"context"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

/*
LX-FIRST ARBITRAGE STRATEGY

Key Insight: LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks).
By the time other venues update, LX has already moved.

This means:
1. LX DEX price is the "TRUE" price (most current)
2. Other venues are always STALE by comparison
3. Arbitrage = correcting stale venues to match LX
4. LX DEX is the ORACLE, not just another venue

Strategy:
1. Watch LX DEX prices (the reference)
2. Compare against "slow" venues (CEX, external DEX)
3. When slow venue diverges from LX, trade on SLOW venue
4. You're essentially front-running slow venues with LX information

Example:
- LX DEX BTC: $50,000 (current, true)
- Binance BTC: $49,990 (stale, 50ms behind)
- Uniswap BTC: $50,020 (stale, 12s behind)

Action:
- Buy on Binance at $49,990 (they haven't caught up yet)
- Sell on Uniswap at $50,020 (they haven't corrected yet)
- Net: $30 profit per BTC

Why LX wins: By the time Binance/Uniswap update, we've already executed.
*/

// LxFirstArbitrage uses LX DEX as the price oracle
type LxFirstArbitrage struct {
	mu sync.RWMutex

	// Price feeds
	lxPrices    map[string]LxPrice      // symbol -> LX DEX price (THE TRUTH)
	venuePrices map[string][]VenuePrice // symbol -> other venue prices (STALE)

	// Configuration
	config LxFirstConfig

	// Opportunities
	opportunities chan *LxFirstOpportunity

	// State
	ctx    context.Context
	cancel context.CancelFunc
}

// LxPrice represents the LX DEX price (the reference/oracle)
type LxPrice struct {
	Symbol    string
	Bid       decimal.Decimal
	Ask       decimal.Decimal
	Mid       decimal.Decimal
	Timestamp time.Time
	BlockNum  uint64
}

// VenuePrice represents a price from a "slow" venue
type VenuePrice struct {
	Venue     string
	Symbol    string
	Bid       decimal.Decimal
	Ask       decimal.Decimal
	Timestamp time.Time
	Latency   time.Duration // How far behind LX this venue typically is
	Stale     bool          // Is this price stale relative to LX?
}

// LxFirstConfig configures the LX-first strategy
type LxFirstConfig struct {
	// How stale is "too stale" to trade
	MaxStaleness time.Duration

	// Minimum divergence from LX price (bps)
	MinDivergenceBps decimal.Decimal

	// Minimum expected profit
	MinProfit decimal.Decimal

	// Venue latency estimates (how far behind LX each venue is)
	VenueLatencies map[string]time.Duration

	// Maximum position per trade
	MaxPositionSize decimal.Decimal

	// Symbols to monitor
	Symbols []string
}

// LxFirstOpportunity represents an arbitrage vs a stale venue
type LxFirstOpportunity struct {
	ID        string
	Symbol    string
	Timestamp time.Time

	// LX DEX price (the truth)
	LxPrice LxPrice

	// Stale venue to exploit
	StaleVenue  string
	StalePrice  VenuePrice
	Staleness   time.Duration

	// Trade direction
	Side       string // "buy" or "sell" on stale venue
	Divergence decimal.Decimal
	DivergenceBps decimal.Decimal

	// Expected profit
	ExpectedProfit decimal.Decimal
	MaxSize        decimal.Decimal

	// Confidence (higher = more stale = easier arbitrage)
	Confidence float64
}

// NewLxFirstArbitrage creates a new LX-first arbitrage system
func NewLxFirstArbitrage(config LxFirstConfig) *LxFirstArbitrage {
	ctx, cancel := context.WithCancel(context.Background())

	return &LxFirstArbitrage{
		lxPrices:      make(map[string]LxPrice),
		venuePrices:   make(map[string][]VenuePrice),
		config:        config,
		opportunities: make(chan *LxFirstOpportunity, 1000),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// UpdateLxPrice updates the LX DEX price (the oracle)
func (lf *LxFirstArbitrage) UpdateLxPrice(price LxPrice) {
	lf.mu.Lock()
	lf.lxPrices[price.Symbol] = price
	lf.mu.Unlock()

	// Immediately check for opportunities against stale venues
	lf.checkOpportunities(price.Symbol)
}

// UpdateVenuePrice updates a price from a "slow" venue
func (lf *LxFirstArbitrage) UpdateVenuePrice(price VenuePrice) {
	lf.mu.Lock()
	defer lf.mu.Unlock()

	prices := lf.venuePrices[price.Symbol]

	// Update or append
	found := false
	for i, p := range prices {
		if p.Venue == price.Venue {
			prices[i] = price
			found = true
			break
		}
	}
	if !found {
		prices = append(prices, price)
	}
	lf.venuePrices[price.Symbol] = prices
}

// checkOpportunities checks for arbitrage opportunities
func (lf *LxFirstArbitrage) checkOpportunities(symbol string) {
	lf.mu.RLock()
	lxPrice, hasLx := lf.lxPrices[symbol]
	venuePrices := lf.venuePrices[symbol]
	lf.mu.RUnlock()

	if !hasLx {
		return
	}

	// Guard against division by zero
	if lxPrice.Mid.IsZero() {
		return
	}

	now := time.Now()

	for _, vp := range venuePrices {
		// Calculate how stale the venue is
		staleness := now.Sub(vp.Timestamp)
		if staleness > lf.config.MaxStaleness {
			continue // Too stale, might have updated by now
		}

		// Check for BUY opportunity (venue ask < LX mid)
		// The slow venue hasn't caught up to LX's higher price
		if vp.Ask.LessThan(lxPrice.Mid) {
			divergence := lxPrice.Mid.Sub(vp.Ask)
			divergenceBps := divergence.Div(lxPrice.Mid).Mul(decimal.NewFromInt(10000))

			if divergenceBps.GreaterThanOrEqual(lf.config.MinDivergenceBps) {
				opp := &LxFirstOpportunity{
					ID:            generateOpportunityID(symbol, vp.Venue, "buy"),
					Symbol:        symbol,
					Timestamp:     now,
					LxPrice:       lxPrice,
					StaleVenue:    vp.Venue,
					StalePrice:    vp,
					Staleness:     staleness,
					Side:          "buy",
					Divergence:    divergence,
					DivergenceBps: divergenceBps,
					ExpectedProfit: divergence.Mul(lf.config.MaxPositionSize),
					MaxSize:       lf.config.MaxPositionSize,
					Confidence:    calculateConfidence(staleness, divergenceBps),
				}

				if opp.ExpectedProfit.GreaterThanOrEqual(lf.config.MinProfit) {
					select {
					case lf.opportunities <- opp:
					default:
					}
				}
			}
		}

		// Check for SELL opportunity (venue bid > LX mid)
		// The slow venue hasn't caught up to LX's lower price
		if vp.Bid.GreaterThan(lxPrice.Mid) {
			divergence := vp.Bid.Sub(lxPrice.Mid)
			divergenceBps := divergence.Div(lxPrice.Mid).Mul(decimal.NewFromInt(10000))

			if divergenceBps.GreaterThanOrEqual(lf.config.MinDivergenceBps) {
				opp := &LxFirstOpportunity{
					ID:            generateOpportunityID(symbol, vp.Venue, "sell"),
					Symbol:        symbol,
					Timestamp:     now,
					LxPrice:       lxPrice,
					StaleVenue:    vp.Venue,
					StalePrice:    vp,
					Staleness:     staleness,
					Side:          "sell",
					Divergence:    divergence,
					DivergenceBps: divergenceBps,
					ExpectedProfit: divergence.Mul(lf.config.MaxPositionSize),
					MaxSize:       lf.config.MaxPositionSize,
					Confidence:    calculateConfidence(staleness, divergenceBps),
				}

				if opp.ExpectedProfit.GreaterThanOrEqual(lf.config.MinProfit) {
					select {
					case lf.opportunities <- opp:
					default:
					}
				}
			}
		}
	}
}

// Opportunities returns the channel of opportunities
func (lf *LxFirstArbitrage) Opportunities() <-chan *LxFirstOpportunity {
	return lf.opportunities
}

// Stop stops the arbitrage system
func (lf *LxFirstArbitrage) Stop() {
	lf.cancel()
}

// Helper functions

func generateOpportunityID(symbol, venue, side string) string {
	return symbol + "-" + venue + "-" + side + "-" + time.Now().Format("150405.000")
}

func calculateConfidence(staleness time.Duration, divergenceBps decimal.Decimal) float64 {
	// Higher confidence when:
	// 1. Venue is more stale (hasn't had time to update)
	// 2. Divergence is larger (more room for profit)

	stalenessScore := 1.0 - (float64(staleness) / float64(5*time.Second))
	if stalenessScore < 0 {
		stalenessScore = 0
	}

	divergenceScore := divergenceBps.InexactFloat64() / 100 // 100bps = 1.0
	if divergenceScore > 1 {
		divergenceScore = 1
	}

	return 0.5*stalenessScore + 0.5*divergenceScore
}

// DefaultLxFirstConfig returns default configuration
func DefaultLxFirstConfig() LxFirstConfig {
	return LxFirstConfig{
		MaxStaleness:     2 * time.Second, // Only trade if venue is <2s stale
		MinDivergenceBps: decimal.NewFromInt(10), // 0.1% minimum divergence
		MinProfit:        decimal.NewFromInt(5),  // $5 minimum profit
		MaxPositionSize:  decimal.NewFromInt(1000), // $1k max per trade
		Symbols:          []string{"BTC-USDC", "ETH-USDC", "LUX-USDC"},
		VenueLatencies: map[string]time.Duration{
			"binance":     50 * time.Millisecond,
			"mexc":        100 * time.Millisecond,
			"okx":         80 * time.Millisecond,
			"uniswap":     12 * time.Second, // ETH block time
			"pancakeswap": 3 * time.Second,  // BSC block time
		},
	}
}

/*
TRADING EXECUTION STRATEGY

When an LxFirstOpportunity is detected:

1. DO NOT trade on LX DEX (it's the reference, not the opportunity)

2. Trade on the STALE venue:
   - If Side="buy": Buy on stale venue (their ask is behind LX)
   - If Side="sell": Sell on stale venue (their bid is behind LX)

3. Settlement options:
   a) Hold position until venues converge (market neutral)
   b) Immediately hedge on LX DEX (lock in profit)
   c) Bridge and sell on another venue (more complex)

4. The key insight:
   - You're NOT arbitraging between two venues
   - You're front-running the slow venue with LX information
   - LX price is where the slow venue WILL BE, you just got there first

Example execution:

  LX DEX shows BTC = $50,000 (current, true price)
  Binance shows BTC = $49,950 (50ms stale)

  Action: BUY on Binance at $49,950
  Why: Binance WILL update to ~$50,000, we bought before they did
  Profit: ~$50 per BTC (0.1%)

  Optional hedge: SELL on LX DEX at $50,000 to lock in profit immediately
*/
