package lx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// ChainlinkPriceSource connects to Chainlink oracles for price feeds
type ChainlinkPriceSource struct {
	// Price feed addresses for different assets
	feedAddresses   map[string]string
	
	// Cached prices
	prices          map[string]*PriceData
	lastUpdate      map[string]time.Time
	
	// Polling configuration
	pollInterval    time.Duration
	timeout         time.Duration
	
	// Health monitoring
	healthy         bool
	lastPoll        time.Time
	failureCount    int
	maxFailures     int
	
	// Control
	mu              sync.RWMutex
	done            chan struct{}
	polling         bool
}

// ChainlinkPriceFeed represents a Chainlink price feed response
type ChainlinkPriceFeed struct {
	RoundID         *big.Int
	Answer          *big.Int
	StartedAt       *big.Int
	UpdatedAt       *big.Int
	AnsweredInRound *big.Int
	Decimals        uint8
}

// NewChainlinkPriceSource creates a new Chainlink price source
func NewChainlinkPriceSource() *ChainlinkPriceSource {
	return &ChainlinkPriceSource{
		feedAddresses:  initChainlinkFeeds(),
		prices:         make(map[string]*PriceData),
		lastUpdate:     make(map[string]time.Time),
		pollInterval:   2 * time.Second,
		timeout:        5 * time.Second,
		healthy:        true,
		maxFailures:    3,
		done:          make(chan struct{}),
	}
}

// initChainlinkFeeds initializes Chainlink price feed addresses
func initChainlinkFeeds() map[string]string {
	// These would be actual Chainlink oracle addresses on-chain
	return map[string]string{
		"BTC-USDT":  "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c", // BTC/USD feed
		"ETH-USDT":  "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419", // ETH/USD feed
		"BNB-USDT":  "0x14e613AC84a31f709eadbdF89C6CC390fDc9540A", // BNB/USD feed
		"SOL-USDT":  "0x4ffC43a60e009B551865A93d232E33Fce9f01507", // SOL/USD feed
		"AVAX-USDT": "0xFF3EEb22B5E3dE6e705b44749C2559d704923FD7", // AVAX/USD feed
		"MATIC-USDT": "0x7bAC85A8a13A4BcD8abb3eB7d6b4d632c5a57676", // MATIC/USD feed
		"ARB-USDT":  "0x31697852a68433DbCc2Ff612c516d69E3D9bd08F", // ARB/USD feed
		"OP-USDT":   "0x0D276FC14719f9292D5C1eA2198673d1f4269246", // OP/USD feed
	}
}

// Start begins polling Chainlink feeds
func (cs *ChainlinkPriceSource) Start() error {
	cs.mu.Lock()
	if cs.polling {
		cs.mu.Unlock()
		return errors.New("already polling")
	}
	cs.polling = true
	cs.mu.Unlock()
	
	// Start polling loop
	go cs.pollLoop()
	
	// Initial poll
	cs.pollAllFeeds()
	
	return nil
}

// pollLoop continuously polls price feeds
func (cs *ChainlinkPriceSource) pollLoop() {
	ticker := time.NewTicker(cs.pollInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-cs.done:
			return
		case <-ticker.C:
			cs.pollAllFeeds()
		}
	}
}

// pollAllFeeds polls all configured price feeds
func (cs *ChainlinkPriceSource) pollAllFeeds() {
	ctx, cancel := context.WithTimeout(context.Background(), cs.timeout)
	defer cancel()
	
	var wg sync.WaitGroup
	for symbol, address := range cs.feedAddresses {
		wg.Add(1)
		go func(sym, addr string) {
			defer wg.Done()
			cs.pollFeed(ctx, sym, addr)
		}(symbol, address)
	}
	
	wg.Wait()
	
	cs.mu.Lock()
	cs.lastPoll = time.Now()
	cs.mu.Unlock()
}

// pollFeed polls a single price feed
func (cs *ChainlinkPriceSource) pollFeed(ctx context.Context, symbol, address string) {
	// In production, this would call the actual Chainlink contract
	// For now, we'll simulate with realistic prices
	price := cs.simulateChainlinkPrice(symbol)
	
	cs.mu.Lock()
	defer cs.mu.Unlock()
	
	cs.prices[symbol] = &PriceData{
		Symbol:     symbol,
		Price:      price,
		Confidence: 0.99, // Chainlink has high confidence
		Timestamp:  time.Now(),
		Source:     "chainlink",
	}
	cs.lastUpdate[symbol] = time.Now()
	cs.failureCount = 0 // Reset on success
	cs.healthy = true
}

// simulateChainlinkPrice simulates realistic Chainlink prices
func (cs *ChainlinkPriceSource) simulateChainlinkPrice(symbol string) float64 {
	basePrices := map[string]float64{
		"BTC-USDT":  50000.0,
		"ETH-USDT":  3000.0,
		"BNB-USDT":  350.0,
		"SOL-USDT":  100.0,
		"AVAX-USDT": 35.0,
		"MATIC-USDT": 0.85,
		"ARB-USDT":  1.20,
		"OP-USDT":   2.50,
	}
	
	base, exists := basePrices[symbol]
	if !exists {
		return 0
	}
	
	// Add small random variation (±0.1%)
	variation := (math.Sin(float64(time.Now().UnixNano())/1e9) * 0.001) + 1.0
	return base * variation
}

// GetPrice returns the latest price for a symbol
func (cs *ChainlinkPriceSource) GetPrice(symbol string) (*PriceData, error) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	
	price, exists := cs.prices[symbol]
	if !exists {
		return nil, fmt.Errorf("no price data for %s", symbol)
	}
	
	// Check staleness (Chainlink updates less frequently than Pyth)
	if time.Since(cs.lastUpdate[symbol]) > 60*time.Second {
		price.IsStale = true
	}
	
	// Create a copy to avoid race conditions
	priceCopy := *price
	return &priceCopy, nil
}

// GetPrices returns prices for multiple symbols
func (cs *ChainlinkPriceSource) GetPrices(symbols []string) (map[string]*PriceData, error) {
	prices := make(map[string]*PriceData)
	
	for _, symbol := range symbols {
		price, err := cs.GetPrice(symbol)
		if err == nil {
			prices[symbol] = price
		}
	}
	
	if len(prices) == 0 {
		return nil, errors.New("no prices available")
	}
	
	return prices, nil
}

// Subscribe is a no-op for Chainlink (uses polling)
func (cs *ChainlinkPriceSource) Subscribe(symbol string) error {
	// Chainlink uses polling, not subscriptions
	return nil
}

// Unsubscribe is a no-op for Chainlink
func (cs *ChainlinkPriceSource) Unsubscribe(symbol string) error {
	return nil
}

// IsHealthy returns the health status
func (cs *ChainlinkPriceSource) IsHealthy() bool {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	
	if !cs.healthy {
		return false
	}
	
	// Check if we've polled recently
	if time.Since(cs.lastPoll) > 30*time.Second {
		return false
	}
	
	// Check failure count
	if cs.failureCount >= cs.maxFailures {
		return false
	}
	
	return true
}

// GetName returns the source name
func (cs *ChainlinkPriceSource) GetName() string {
	return "chainlink"
}

// GetWeight returns the source weight for aggregation
func (cs *ChainlinkPriceSource) GetWeight() float64 {
	return 2.0 // Higher weight for Chainlink due to decentralization
}

// Close stops the price source
func (cs *ChainlinkPriceSource) Close() error {
	close(cs.done)
	
	cs.mu.Lock()
	cs.polling = false
	cs.mu.Unlock()
	
	return nil
}

// GetLatestRoundData would call the actual Chainlink aggregator in production
func (cs *ChainlinkPriceSource) GetLatestRoundData(feedAddress string) (*ChainlinkPriceFeed, error) {
	// In production, this would make an actual blockchain call
	// For now, return simulated data
	return &ChainlinkPriceFeed{
		RoundID:         big.NewInt(time.Now().Unix()),
		Answer:          big.NewInt(50000 * 1e8), // Price with 8 decimals
		StartedAt:       big.NewInt(time.Now().Unix()),
		UpdatedAt:       big.NewInt(time.Now().Unix()),
		AnsweredInRound: big.NewInt(time.Now().Unix()),
		Decimals:        8,
	}, nil
}