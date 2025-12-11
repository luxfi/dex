package price

import (
	"context"
	"math"
	"math/big"
	"sync"
	"time"
)

// ChainlinkSource provides prices from Chainlink oracles.
// Decentralized and highly reliable for reference prices.
type ChainlinkSource struct {
	feeds  map[string]string
	prices map[string]*Data
	last   map[string]time.Time

	interval time.Duration
	timeout  time.Duration

	healthy  bool
	failures int

	mu      sync.RWMutex
	done    chan struct{}
	polling bool
}

// NewChainlinkSource creates a Chainlink price source.
func NewChainlinkSource() *ChainlinkSource {
	return &ChainlinkSource{
		feeds:    chainlinkFeeds(),
		prices:   make(map[string]*Data),
		last:     make(map[string]time.Time),
		interval: 2 * time.Second,
		timeout:  5 * time.Second,
		healthy:  true,
		done:     make(chan struct{}),
	}
}

func chainlinkFeeds() map[string]string {
	return map[string]string{
		"BTC-USD":  "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c",
		"ETH-USD":  "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
		"SOL-USD":  "0x4ffC43a60e009B551865A93d232E33Fce9f01507",
		"AVAX-USD": "0xFF3EEb22B5E3dE6e705b44749C2559d704923FD7",
		"LUX-USD":  "0x0000000000000000000000000000000000000001", // Placeholder
	}
}

// Start begins polling Chainlink feeds.
func (s *ChainlinkSource) Start() error {
	s.mu.Lock()
	if s.polling {
		s.mu.Unlock()
		return nil
	}
	s.polling = true
	s.mu.Unlock()

	go s.loop()
	s.poll()
	return nil
}

func (s *ChainlinkSource) loop() {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.poll()
		}
	}
}

func (s *ChainlinkSource) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()

	var wg sync.WaitGroup
	for symbol, addr := range s.feeds {
		wg.Add(1)
		go func(sym, a string) {
			defer wg.Done()
			s.fetch(ctx, sym, a)
		}(symbol, addr)
	}
	wg.Wait()
}

func (s *ChainlinkSource) fetch(ctx context.Context, symbol, addr string) {
	// In production: call latestRoundData() on the aggregator contract
	// For now, simulate realistic prices
	price := s.simulate(symbol)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.prices[symbol] = &Data{
		Symbol:     symbol,
		Price:      price,
		Confidence: 0.99, // Chainlink is highly reliable
		Timestamp:  time.Now(),
		Source:     "chainlink",
	}
	s.last[symbol] = time.Now()
	s.failures = 0
	s.healthy = true
}

func (s *ChainlinkSource) simulate(symbol string) float64 {
	base := map[string]float64{
		"BTC-USD":  50000.0,
		"ETH-USD":  3000.0,
		"SOL-USD":  100.0,
		"AVAX-USD": 35.0,
		"LUX-USD":  10.0,
	}

	b, ok := base[symbol]
	if !ok {
		return 0
	}

	// Small variation
	v := (math.Sin(float64(time.Now().UnixNano())/1e9) * 0.001) + 1.0
	return b * v
}

// Price returns the latest price for a symbol.
func (s *ChainlinkSource) Price(symbol string) (*Data, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p, ok := s.prices[symbol]
	if !ok {
		return nil, ErrNotFound
	}

	if time.Since(s.last[symbol]) > 60*time.Second {
		copy := *p
		copy.Stale = true
		return &copy, nil
	}

	copy := *p
	return &copy, nil
}

// Prices returns prices for multiple symbols.
func (s *ChainlinkSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Subscribe is a no-op (uses polling).
func (s *ChainlinkSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *ChainlinkSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns health status.
func (s *ChainlinkSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy && s.failures < 3
}

// Name returns "chainlink".
func (s *ChainlinkSource) Name() string { return "chainlink" }

// Weight returns 2.0 (high trust due to decentralization).
func (s *ChainlinkSource) Weight() float64 { return 2.0 }

// Close stops the source.
func (s *ChainlinkSource) Close() error {
	close(s.done)
	s.mu.Lock()
	s.polling = false
	s.mu.Unlock()
	return nil
}

// ChainlinkRound represents a price feed response.
type ChainlinkRound struct {
	RoundID   *big.Int
	Answer    *big.Int
	StartedAt *big.Int
	UpdatedAt *big.Int
	Decimals  uint8
}
