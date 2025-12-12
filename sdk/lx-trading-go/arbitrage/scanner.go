// Package arbitrage provides omnichain arbitrage detection and execution
package arbitrage

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

// PriceSource represents a price feed from a specific venue/chain
type PriceSource struct {
	ChainID   string
	Venue     string
	Symbol    string
	Bid       decimal.Decimal
	Ask       decimal.Decimal
	Liquidity decimal.Decimal
	Timestamp time.Time
	Latency   time.Duration
}

// ArbitrageOpportunity represents a detected arbitrage opportunity
type ArbitrageOpportunity struct {
	ID            string
	Type          ArbType
	Routes        []Route
	BuySource     PriceSource
	SellSource    PriceSource
	SpreadBps     decimal.Decimal // Spread in basis points
	EstimatedPnL  decimal.Decimal
	MaxSize       decimal.Decimal // Limited by liquidity
	GasCostUSD    decimal.Decimal
	BridgeCostUSD decimal.Decimal
	NetPnL        decimal.Decimal
	Confidence    float64 // 0-1, based on price freshness and liquidity
	ExpiresAt     time.Time
}

// Route represents a single leg of an arbitrage
type Route struct {
	ChainID      string
	Venue        string
	Action       string // "buy" or "sell"
	TokenIn      string
	TokenOut     string
	AmountIn     decimal.Decimal
	ExpectedOut  decimal.Decimal
	MinAmountOut decimal.Decimal
	SwapData     []byte
}

// ArbType represents the type of arbitrage
type ArbType string

const (
	ArbTypeSimple     ArbType = "simple"      // Buy A, sell B
	ArbTypeTriangular ArbType = "triangular"  // A->B->C->A
	ArbTypeMultiHop   ArbType = "multi_hop"   // Complex routes
	ArbTypeCEXDEX     ArbType = "cex_dex"     // CEX<->DEX arb
	ArbTypeFlashSwap  ArbType = "flash_swap"  // DEX flash swap
)

// Scanner continuously scans for arbitrage opportunities
type Scanner struct {
	mu sync.RWMutex

	// Price feeds from all sources
	prices map[string][]PriceSource // symbol -> sources

	// Configuration
	config ScannerConfig

	// Detected opportunities
	opportunities chan ArbitrageOpportunity

	// Chain configurations
	chains map[string]ChainConfig

	// Running state
	ctx    context.Context
	cancel context.CancelFunc
}

// ScannerConfig configures the arbitrage scanner
type ScannerConfig struct {
	// Minimum spread to consider (basis points)
	MinSpreadBps decimal.Decimal

	// Minimum profit after fees (USD)
	MinProfitUSD decimal.Decimal

	// Maximum price age before stale
	MaxPriceAge time.Duration

	// Symbols to scan
	Symbols []string

	// Chains to scan
	ChainIDs []string

	// Scan interval
	ScanInterval time.Duration

	// Maximum concurrent scans
	MaxConcurrency int
}

// ChainConfig holds chain-specific configuration
type ChainConfig struct {
	ChainID        string
	Name           string
	GasPrice       decimal.Decimal
	BlockTime      time.Duration
	BridgeCost     decimal.Decimal
	BridgeLatency  time.Duration
	Venues         []string
	WarpSupported  bool // Native Lux Warp
	TeleportSupport bool // EVM Teleport bridge
}

// NewScanner creates a new arbitrage scanner
func NewScanner(config ScannerConfig) *Scanner {
	ctx, cancel := context.WithCancel(context.Background())

	return &Scanner{
		prices:        make(map[string][]PriceSource),
		config:        config,
		opportunities: make(chan ArbitrageOpportunity, 1000),
		chains:        make(map[string]ChainConfig),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// AddChain adds a chain configuration
func (s *Scanner) AddChain(config ChainConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chains[config.ChainID] = config
}

// UpdatePrice updates a price feed
func (s *Scanner) UpdatePrice(source PriceSource) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sources := s.prices[source.Symbol]

	// Update existing or append new
	found := false
	for i, existing := range sources {
		if existing.ChainID == source.ChainID && existing.Venue == source.Venue {
			sources[i] = source
			found = true
			break
		}
	}

	if !found {
		sources = append(sources, source)
	}

	s.prices[source.Symbol] = sources
}

// Start begins scanning for opportunities
func (s *Scanner) Start() {
	go s.scanLoop()
}

// Stop stops the scanner
func (s *Scanner) Stop() {
	s.cancel()
}

// Opportunities returns the channel of detected opportunities
func (s *Scanner) Opportunities() <-chan ArbitrageOpportunity {
	return s.opportunities
}

// scanLoop continuously scans for arbitrage opportunities
func (s *Scanner) scanLoop() {
	ticker := time.NewTicker(s.config.ScanInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.scan()
		}
	}
}

// scan performs a single scan across all symbols
func (s *Scanner) scan() {
	s.mu.RLock()
	symbols := make([]string, 0, len(s.prices))
	for symbol := range s.prices {
		symbols = append(symbols, symbol)
	}
	s.mu.RUnlock()

	// Scan each symbol concurrently
	var wg sync.WaitGroup
	sem := make(chan struct{}, s.config.MaxConcurrency)

	for _, symbol := range symbols {
		wg.Add(1)
		sem <- struct{}{}

		go func(sym string) {
			defer wg.Done()
			defer func() { <-sem }()

			opps := s.findOpportunities(sym)
			for _, opp := range opps {
				select {
				case s.opportunities <- opp:
				default:
					// Channel full, skip
				}
			}
		}(symbol)
	}

	wg.Wait()
}

// findOpportunities finds all arbitrage opportunities for a symbol
func (s *Scanner) findOpportunities(symbol string) []ArbitrageOpportunity {
	s.mu.RLock()
	sources := s.prices[symbol]
	s.mu.RUnlock()

	if len(sources) < 2 {
		return nil
	}

	var opportunities []ArbitrageOpportunity
	now := time.Now()

	// Filter stale prices
	validSources := make([]PriceSource, 0, len(sources))
	for _, src := range sources {
		if now.Sub(src.Timestamp) < s.config.MaxPriceAge {
			validSources = append(validSources, src)
		}
	}

	if len(validSources) < 2 {
		return nil
	}

	// Find simple arbitrage (buy low, sell high)
	opportunities = append(opportunities, s.findSimpleArb(symbol, validSources)...)

	// Find triangular arbitrage
	opportunities = append(opportunities, s.findTriangularArb(symbol, validSources)...)

	// Find CEX-DEX arbitrage
	opportunities = append(opportunities, s.findCEXDEXArb(symbol, validSources)...)

	return opportunities
}

// findSimpleArb finds simple buy-low-sell-high opportunities
func (s *Scanner) findSimpleArb(symbol string, sources []PriceSource) []ArbitrageOpportunity {
	var opportunities []ArbitrageOpportunity

	// Sort by ask price (lowest first for buying)
	buyOrder := make([]PriceSource, len(sources))
	copy(buyOrder, sources)
	sort.Slice(buyOrder, func(i, j int) bool {
		return buyOrder[i].Ask.LessThan(buyOrder[j].Ask)
	})

	// Sort by bid price (highest first for selling)
	sellOrder := make([]PriceSource, len(sources))
	copy(sellOrder, sources)
	sort.Slice(sellOrder, func(i, j int) bool {
		return sellOrder[i].Bid.GreaterThan(sellOrder[j].Bid)
	})

	// Check each buy/sell combination
	for _, buySrc := range buyOrder {
		for _, sellSrc := range sellOrder {
			// Skip same venue/chain
			if buySrc.ChainID == sellSrc.ChainID && buySrc.Venue == sellSrc.Venue {
				continue
			}

			// Calculate spread
			spread := sellSrc.Bid.Sub(buySrc.Ask)
			if spread.LessThanOrEqual(decimal.Zero) {
				continue
			}

			// Guard against division by zero
			if buySrc.Ask.IsZero() {
				continue
			}

			spreadBps := spread.Div(buySrc.Ask).Mul(decimal.NewFromInt(10000))
			if spreadBps.LessThan(s.config.MinSpreadBps) {
				continue
			}

			// Calculate costs
			gasCost, bridgeCost := s.calculateCosts(buySrc.ChainID, sellSrc.ChainID)

			// Maximum size limited by liquidity on both sides
			maxSize := decimal.Min(buySrc.Liquidity, sellSrc.Liquidity)

			// Calculate PnL
			grossPnL := spread.Mul(maxSize)
			netPnL := grossPnL.Sub(gasCost).Sub(bridgeCost)

			if netPnL.LessThan(s.config.MinProfitUSD) {
				continue
			}

			// Calculate confidence based on price freshness and liquidity
			confidence := s.calculateConfidence(buySrc, sellSrc)

			opp := ArbitrageOpportunity{
				ID:           fmt.Sprintf("simple-%s-%s-%s-%d", symbol, buySrc.Venue, sellSrc.Venue, time.Now().UnixNano()),
				Type:         ArbTypeSimple,
				BuySource:    buySrc,
				SellSource:   sellSrc,
				SpreadBps:    spreadBps,
				EstimatedPnL: grossPnL,
				MaxSize:      maxSize,
				GasCostUSD:   gasCost,
				BridgeCostUSD: bridgeCost,
				NetPnL:       netPnL,
				Confidence:   confidence,
				ExpiresAt:    time.Now().Add(5 * time.Second), // Short expiry for HFT
				Routes: []Route{
					{
						ChainID:  buySrc.ChainID,
						Venue:    buySrc.Venue,
						Action:   "buy",
						TokenIn:  "USDC", // Assuming USDC quote
						TokenOut: symbol,
						AmountIn: maxSize.Mul(buySrc.Ask),
					},
					{
						ChainID:  sellSrc.ChainID,
						Venue:    sellSrc.Venue,
						Action:   "sell",
						TokenIn:  symbol,
						TokenOut: "USDC",
						AmountIn: maxSize,
					},
				},
			}

			opportunities = append(opportunities, opp)
		}
	}

	return opportunities
}

// findTriangularArb finds A->B->C->A opportunities
func (s *Scanner) findTriangularArb(symbol string, sources []PriceSource) []ArbitrageOpportunity {
	// Triangular arbitrage within same chain/venue
	// Example: USDC -> BTC -> ETH -> USDC
	// If the product of exchange rates > 1, there's profit

	var opportunities []ArbitrageOpportunity

	// This is a simplified version - real implementation would:
	// 1. Build a graph of all trading pairs
	// 2. Find negative cycles using Bellman-Ford
	// 3. Calculate optimal trade sizes

	// For now, check common triangular routes
	triangles := [][]string{
		{"USDC", "BTC", "ETH"},
		{"USDC", "ETH", "LUX"},
		{"USDT", "BTC", "ETH"},
	}

	for _, triangle := range triangles {
		// Check if we have prices for all pairs
		// Calculate circular exchange rate
		// If > 1, there's an opportunity
		_ = triangle // Placeholder for full implementation
	}

	return opportunities
}

// findCEXDEXArb finds CEX<->DEX arbitrage opportunities
func (s *Scanner) findCEXDEXArb(symbol string, sources []PriceSource) []ArbitrageOpportunity {
	var opportunities []ArbitrageOpportunity

	// Separate CEX and DEX sources
	var cexSources, dexSources []PriceSource
	for _, src := range sources {
		if isCEX(src.Venue) {
			cexSources = append(cexSources, src)
		} else {
			dexSources = append(dexSources, src)
		}
	}

	// Find CEX buy -> DEX sell opportunities
	for _, cex := range cexSources {
		for _, dex := range dexSources {
			spread := dex.Bid.Sub(cex.Ask)
			if spread.LessThanOrEqual(decimal.Zero) {
				continue
			}

			// Guard against division by zero
			if cex.Ask.IsZero() {
				continue
			}

			spreadBps := spread.Div(cex.Ask).Mul(decimal.NewFromInt(10000))
			if spreadBps.LessThan(s.config.MinSpreadBps) {
				continue
			}

			// CEX-DEX arb requires considering:
			// - CEX withdrawal fees
			// - Blockchain confirmation times
			// - DEX slippage

			opp := ArbitrageOpportunity{
				ID:         fmt.Sprintf("cexdex-%s-%s-%s-%d", symbol, cex.Venue, dex.Venue, time.Now().UnixNano()),
				Type:       ArbTypeCEXDEX,
				BuySource:  cex,
				SellSource: dex,
				SpreadBps:  spreadBps,
				// Additional CEX-DEX specific calculations...
			}

			opportunities = append(opportunities, opp)
		}
	}

	return opportunities
}

// calculateCosts calculates gas and bridge costs between chains
func (s *Scanner) calculateCosts(sourceChain, destChain string) (gasCost, bridgeCost decimal.Decimal) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	srcConfig := s.chains[sourceChain]
	dstConfig := s.chains[destChain]

	// Estimate gas cost (simplified)
	gasCost = srcConfig.GasPrice.Mul(decimal.NewFromInt(200000)) // ~200k gas for swap

	// Bridge cost if crossing chains
	if sourceChain != destChain {
		// Check if Warp is available (Lux native, lowest cost)
		if srcConfig.WarpSupported && dstConfig.WarpSupported {
			bridgeCost = decimal.NewFromFloat(0.01) // Warp is nearly free
		} else if srcConfig.TeleportSupport && dstConfig.TeleportSupport {
			bridgeCost = decimal.NewFromFloat(0.10) // Teleport for EVM chains
		} else {
			bridgeCost = srcConfig.BridgeCost.Add(dstConfig.BridgeCost)
		}
	}

	return gasCost, bridgeCost
}

// calculateConfidence calculates confidence score for an opportunity
func (s *Scanner) calculateConfidence(buy, sell PriceSource) float64 {
	now := time.Now()

	// Freshness score (newer = better)
	buyAge := now.Sub(buy.Timestamp).Seconds()
	sellAge := now.Sub(sell.Timestamp).Seconds()
	maxAge := s.config.MaxPriceAge.Seconds()
	freshnessScore := 1.0
	if maxAge > 0 {
		freshnessScore = 1.0 - (buyAge+sellAge)/(2*maxAge)
	}
	if freshnessScore < 0 {
		freshnessScore = 0
	}

	// Liquidity score
	minLiq := decimal.Min(buy.Liquidity, sell.Liquidity)
	liquidityScore := 0.5 // Simplified

	if minLiq.GreaterThan(decimal.NewFromInt(100000)) {
		liquidityScore = 1.0
	} else if minLiq.GreaterThan(decimal.NewFromInt(10000)) {
		liquidityScore = 0.8
	}

	// Latency score
	latencyScore := 1.0 - float64(buy.Latency+sell.Latency)/(2*float64(time.Second))
	if latencyScore < 0 {
		latencyScore = 0
	}

	// Weighted average
	return 0.4*freshnessScore + 0.4*liquidityScore + 0.2*latencyScore
}

// Helper functions

func isCEX(venue string) bool {
	cexes := map[string]bool{
		"binance": true, "coinbase": true, "kraken": true,
		"okx": true, "bybit": true, "kucoin": true,
		"mexc": true, "gate": true, "huobi": true,
	}
	return cexes[venue]
}

// DefaultScannerConfig returns default configuration
func DefaultScannerConfig() ScannerConfig {
	return ScannerConfig{
		MinSpreadBps:   decimal.NewFromInt(10),                    // 0.1%
		MinProfitUSD:   decimal.NewFromInt(10),                    // $10 minimum
		MaxPriceAge:    5 * time.Second,                           // 5 second max age
		ScanInterval:   100 * time.Millisecond,                    // 10 scans/second
		MaxConcurrency: 50,
		Symbols:        []string{"BTC", "ETH", "LUX", "SOL", "AVAX"},
		ChainIDs:       []string{"lux", "ethereum", "bsc", "arbitrum", "polygon"},
	}
}

// LuxChainConfig returns Lux-optimized chain configuration
func LuxChainConfig() ChainConfig {
	return ChainConfig{
		ChainID:        "lux",
		Name:           "Lux Network",
		GasPrice:       decimal.NewFromFloat(0.000000025), // 25 gwei
		BlockTime:      400 * time.Millisecond,            // Sub-second finality
		BridgeCost:     decimal.NewFromFloat(0.01),        // Nearly free via Warp
		BridgeLatency:  500 * time.Millisecond,            // Fast Warp messaging
		Venues:         []string{"lx_dex", "lx_amm"},
		WarpSupported:  true,
		TeleportSupport: true,
	}
}
