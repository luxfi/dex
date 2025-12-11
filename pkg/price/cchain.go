package price

import (
	"context"
	"math/big"
	"sync"
	"time"
)

// CChainSource provides prices from Lux C-Chain AMMs.
// On-chain price discovery for cross-market arbitrage.
type CChainSource struct {
	rpcURL string
	wsURL  string

	routers map[string]string
	tokens  map[string]TokenPair
	prices  map[string]*Data
	last    map[string]time.Time

	reserves map[string]*Reserves

	interval time.Duration
	healthy  bool

	mu      sync.RWMutex
	done    chan struct{}
	polling bool
}

// TokenPair defines a trading pair.
type TokenPair struct {
	Token0   string
	Token1   string
	Decimals [2]uint8
	Pool     string
}

// Reserves tracks AMM pool state.
type Reserves struct {
	R0    *big.Int
	R1    *big.Int
	Block uint64
	Time  time.Time
}

// NewCChainSource creates a C-Chain price source.
func NewCChainSource(rpcURL, wsURL string) *CChainSource {
	return &CChainSource{
		rpcURL:   rpcURL,
		wsURL:    wsURL,
		routers:  cchainRouters(),
		tokens:   cchainTokens(),
		prices:   make(map[string]*Data),
		last:     make(map[string]time.Time),
		reserves: make(map[string]*Reserves),
		interval: 100 * time.Millisecond,
		healthy:  true,
		done:     make(chan struct{}),
	}
}

func cchainRouters() map[string]string {
	return map[string]string{
		"trader_joe": "0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
		"pangolin":   "0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106",
		"sushiswap":  "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
	}
}

func cchainTokens() map[string]TokenPair {
	return map[string]TokenPair{
		"AVAX-USDC": {
			Token0:   "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
			Token1:   "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
			Decimals: [2]uint8{18, 6},
		},
		"ETH-USDC": {
			Token0:   "0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB",
			Token1:   "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
			Decimals: [2]uint8{18, 6},
		},
		"LUX-USDC": {
			Token0:   "0x0000000000000000000000000000000000000000",
			Token1:   "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
			Decimals: [2]uint8{18, 6},
		},
	}
}

// Start begins polling C-Chain pools.
func (s *CChainSource) Start() error {
	s.mu.Lock()
	if s.polling {
		s.mu.Unlock()
		return nil
	}
	s.polling = true
	s.mu.Unlock()

	go s.loop()
	return nil
}

func (s *CChainSource) loop() {
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

func (s *CChainSource) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for symbol, pair := range s.tokens {
		wg.Add(1)
		go func(sym string, p TokenPair) {
			defer wg.Done()
			s.fetch(ctx, sym, p)
		}(symbol, pair)
	}
	wg.Wait()
}

func (s *CChainSource) fetch(ctx context.Context, symbol string, pair TokenPair) {
	// In production: call getReserves() on the pool
	// For now, simulate
	price := s.simulate(symbol)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.prices[symbol] = &Data{
		Symbol:     symbol,
		Price:      price,
		Confidence: 0.95,
		Timestamp:  time.Now(),
		Source:     "c-chain",
	}
	s.last[symbol] = time.Now()
	s.healthy = true
}

func (s *CChainSource) simulate(symbol string) float64 {
	base := map[string]float64{
		"AVAX-USDC": 35.0,
		"ETH-USDC":  3000.0,
		"LUX-USDC":  10.0,
	}

	b, ok := base[symbol]
	if !ok {
		return 0
	}

	// Slight AMM variation
	now := time.Now().UnixNano()
	v := float64(now%1000) / 100000.0
	return b * (1.0 + v)
}

// Price returns the latest price for a symbol.
func (s *CChainSource) Price(symbol string) (*Data, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p, ok := s.prices[symbol]
	if !ok {
		return nil, ErrNotFound
	}

	if time.Since(s.last[symbol]) > 5*time.Second {
		copy := *p
		copy.Stale = true
		return &copy, nil
	}

	copy := *p
	return &copy, nil
}

// Prices returns prices for multiple symbols.
func (s *CChainSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Subscribe is a no-op (uses polling).
func (s *CChainSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *CChainSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns health status.
func (s *CChainSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy
}

// Name returns "c-chain".
func (s *CChainSource) Name() string { return "c-chain" }

// Weight returns 1.2 (on-chain truth).
func (s *CChainSource) Weight() float64 { return 1.2 }

// Close stops the source.
func (s *CChainSource) Close() error {
	close(s.done)
	s.mu.Lock()
	s.polling = false
	s.mu.Unlock()
	return nil
}

// Reserves returns pool reserves for a symbol.
func (s *CChainSource) GetReserves(symbol string) (*Reserves, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	r, ok := s.reserves[symbol]
	if !ok {
		return nil, ErrNotFound
	}
	return r, nil
}
