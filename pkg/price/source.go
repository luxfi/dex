package price

import (
	"sync"
	"time"
)

// OrderbookSource provides prices from local orderbook data.
// This is the primary source for co-located low-latency clients.
// Target latency: <100ns for price lookup.
type OrderbookSource struct {
	books      OrderbookProvider
	prices     map[string]*Data
	lastUpdate map[string]time.Time
	interval   time.Duration

	mu sync.RWMutex
	lifecycle
	running bool
}

// OrderbookProvider abstracts orderbook access.
type OrderbookProvider interface {
	GetBestBid(symbol string) float64
	GetBestAsk(symbol string) float64
	GetSymbols() []string
}

// NewOrderbookSource creates a price source from orderbook data.
func NewOrderbookSource(books OrderbookProvider) *OrderbookSource {
	return &OrderbookSource{
		books:      books,
		prices:     make(map[string]*Data),
		lastUpdate: make(map[string]time.Time),
		interval:   10 * time.Millisecond,
		lifecycle:  newLifecycle(),
	}
}

// Start begins price updates.
func (s *OrderbookSource) Start() error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return nil
	}
	s.running = true
	s.mu.Unlock()

	s.run(s.loop)
	return nil
}

func (s *OrderbookSource) loop() {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.update()
		}
	}
}

func (s *OrderbookSource) update() {
	if s.books == nil {
		return
	}

	symbols := s.books.GetSymbols()
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, symbol := range symbols {
		bid := s.books.GetBestBid(symbol)
		ask := s.books.GetBestAsk(symbol)

		if bid == 0 && ask == 0 {
			continue
		}

		var price float64
		if bid > 0 && ask > 0 {
			price = (bid + ask) / 2
		} else if bid > 0 {
			price = bid
		} else {
			price = ask
		}

		// Spread-based confidence
		var confidence float64 = 1.0
		if bid > 0 && ask > 0 && price > 0 {
			spread := (ask - bid) / price
			confidence = 1.0 - spread
			if confidence < 0.5 {
				confidence = 0.5
			}
		}

		s.prices[symbol] = &Data{
			Symbol:     symbol,
			Price:      price,
			Bid:        bid,
			Ask:        ask,
			Confidence: confidence,
			Timestamp:  time.Now(),
			Source:     "orderbook",
		}
		s.lastUpdate[symbol] = time.Now()
	}
}

// Price returns the latest price for a symbol.
func (s *OrderbookSource) Price(symbol string) (*Data, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p, ok := s.prices[symbol]
	if !ok {
		return nil, ErrNotFound
	}

	if time.Since(s.lastUpdate[symbol]) > time.Second {
		copy := *p
		copy.Stale = true
		return &copy, nil
	}

	copy := *p
	return &copy, nil
}

// Prices returns prices for multiple symbols.
func (s *OrderbookSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Subscribe is a no-op (always tracking).
func (s *OrderbookSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *OrderbookSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns true if running.
func (s *OrderbookSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

// Name returns "orderbook".
func (s *OrderbookSource) Name() string { return "orderbook" }

// Weight returns 1.0 (base weight).
func (s *OrderbookSource) Weight() float64 { return 1.0 }

// Stop halts the source.
func (s *OrderbookSource) Stop() error {
	s.stop()
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
	return nil
}
