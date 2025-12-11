package price

import (
	"context"
	"sync"
	"time"
)

// XChainSource provides prices from Lux X-Chain DEX orderbooks.
// Native exchange chain with on-chain matching and settlement.
type XChainSource struct {
	rpcURL string
	wsURL  string

	markets map[string]XMarket
	prices  map[string]*Data
	last    map[string]time.Time
	books   map[string]*XOrderbook

	interval time.Duration
	healthy  bool

	mu      sync.RWMutex
	done    chan struct{}
	polling bool
}

// XMarket defines an X-Chain trading pair.
type XMarket struct {
	BaseAsset  string
	QuoteAsset string
	MarketID   string
	MinSize    float64
	TickSize   float64
}

// XOrderbook tracks X-Chain orderbook state.
type XOrderbook struct {
	Bids   []XLevel
	Asks   []XLevel
	Block  uint64
	Time   time.Time
	Spread float64
}

// XLevel represents a price level.
type XLevel struct {
	Price float64
	Size  float64
	Count int
}

// NewXChainSource creates an X-Chain price source.
func NewXChainSource(rpcURL, wsURL string) *XChainSource {
	return &XChainSource{
		rpcURL:   rpcURL,
		wsURL:    wsURL,
		markets:  xchainMarkets(),
		prices:   make(map[string]*Data),
		last:     make(map[string]time.Time),
		books:    make(map[string]*XOrderbook),
		interval: 50 * time.Millisecond, // Faster polling - native chain
		healthy:  true,
		done:     make(chan struct{}),
	}
}

func xchainMarkets() map[string]XMarket {
	return map[string]XMarket{
		"LUX-USDC": {
			BaseAsset:  "LUX",
			QuoteAsset: "USDC",
			MarketID:   "lux-usdc-v1",
			MinSize:    0.01,
			TickSize:   0.0001,
		},
		"LUX-USDT": {
			BaseAsset:  "LUX",
			QuoteAsset: "USDT",
			MarketID:   "lux-usdt-v1",
			MinSize:    0.01,
			TickSize:   0.0001,
		},
		"ETH-LUX": {
			BaseAsset:  "ETH",
			QuoteAsset: "LUX",
			MarketID:   "eth-lux-v1",
			MinSize:    0.001,
			TickSize:   0.001,
		},
		"BTC-LUX": {
			BaseAsset:  "BTC",
			QuoteAsset: "LUX",
			MarketID:   "btc-lux-v1",
			MinSize:    0.0001,
			TickSize:   0.01,
		},
		"AVAX-LUX": {
			BaseAsset:  "AVAX",
			QuoteAsset: "LUX",
			MarketID:   "avax-lux-v1",
			MinSize:    0.1,
			TickSize:   0.0001,
		},
	}
}

// Start begins polling X-Chain orderbooks.
func (s *XChainSource) Start() error {
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

func (s *XChainSource) loop() {
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

func (s *XChainSource) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for symbol, market := range s.markets {
		wg.Add(1)
		go func(sym string, m XMarket) {
			defer wg.Done()
			s.fetch(ctx, sym, m)
		}(symbol, market)
	}
	wg.Wait()
}

func (s *XChainSource) fetch(ctx context.Context, symbol string, market XMarket) {
	// In production: query X-Chain VM for orderbook state
	// For now, simulate realistic orderbook
	book := s.simulate(symbol)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.books[symbol] = book

	mid := (book.Bids[0].Price + book.Asks[0].Price) / 2
	s.prices[symbol] = &Data{
		Symbol:     symbol,
		Price:      mid,
		Bid:        book.Bids[0].Price,
		Ask:        book.Asks[0].Price,
		Volume:     s.totalVolume(book),
		Confidence: 0.99, // Native chain - highest confidence
		Timestamp:  time.Now(),
		Source:     "x-chain",
	}
	s.last[symbol] = time.Now()
	s.healthy = true
}

func (s *XChainSource) simulate(symbol string) *XOrderbook {
	base := map[string]float64{
		"LUX-USDC": 10.0,
		"LUX-USDT": 10.0,
		"ETH-LUX":  300.0,
		"BTC-LUX":  6000.0,
		"AVAX-LUX": 3.5,
	}

	b, ok := base[symbol]
	if !ok {
		b = 1.0
	}

	// Slight variation
	now := time.Now().UnixNano()
	v := float64(now%500) / 100000.0
	mid := b * (1.0 + v)
	spread := mid * 0.0002 // 2 bps spread

	return &XOrderbook{
		Bids: []XLevel{
			{Price: mid - spread/2, Size: 1000.0, Count: 5},
			{Price: mid - spread, Size: 2000.0, Count: 8},
			{Price: mid - spread*2, Size: 5000.0, Count: 12},
		},
		Asks: []XLevel{
			{Price: mid + spread/2, Size: 1000.0, Count: 5},
			{Price: mid + spread, Size: 2000.0, Count: 8},
			{Price: mid + spread*2, Size: 5000.0, Count: 12},
		},
		Block:  uint64(time.Now().Unix()),
		Time:   time.Now(),
		Spread: spread,
	}
}

func (s *XChainSource) totalVolume(book *XOrderbook) float64 {
	var vol float64
	for _, l := range book.Bids {
		vol += l.Size
	}
	for _, l := range book.Asks {
		vol += l.Size
	}
	return vol
}

// Price returns the latest price for a symbol.
func (s *XChainSource) Price(symbol string) (*Data, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p, ok := s.prices[symbol]
	if !ok {
		return nil, ErrNotFound
	}

	if time.Since(s.last[symbol]) > 2*time.Second {
		copy := *p
		copy.Stale = true
		return &copy, nil
	}

	copy := *p
	return &copy, nil
}

// Prices returns prices for multiple symbols.
func (s *XChainSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Orderbook returns the current orderbook for a symbol.
func (s *XChainSource) Orderbook(symbol string) (*XOrderbook, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	book, ok := s.books[symbol]
	if !ok {
		return nil, ErrNotFound
	}
	return book, nil
}

// Subscribe is a no-op (uses polling).
func (s *XChainSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *XChainSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns health status.
func (s *XChainSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy
}

// Name returns "x-chain".
func (s *XChainSource) Name() string { return "x-chain" }

// Weight returns 2.0 (native DEX chain - highest weight).
func (s *XChainSource) Weight() float64 { return 2.0 }

// Close stops the source.
func (s *XChainSource) Close() error {
	close(s.done)
	s.mu.Lock()
	s.polling = false
	s.mu.Unlock()
	return nil
}

// Markets returns available markets.
func (s *XChainSource) Markets() map[string]XMarket {
	return s.markets
}
