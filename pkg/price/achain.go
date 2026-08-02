package price

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// AChainSource provides price attestations from Lux A-Chain.
// Attestation chain for cross-chain price verification and consensus.
type AChainSource struct {
	rpcURL string
	wsURL  string

	attestations map[string]*Attestation
	prices       map[string]*Data
	last         map[string]time.Time
	validators   map[string]*Validator

	interval time.Duration
	healthy  bool
	quorum   int // Minimum validators for consensus

	mu sync.RWMutex
	lifecycle
	polling bool
}

// Attestation represents a price attestation from validators.
type Attestation struct {
	Symbol     string
	Price      float64
	Timestamp  time.Time
	Signatures []Signature
	Hash       string
	Block      uint64
	Finalized  bool
}

// Signature is a validator signature on a price attestation.
type Signature struct {
	Validator string
	Sig       []byte
	Timestamp time.Time
}

// Validator represents an A-Chain price validator.
type Validator struct {
	Address  string
	Stake    uint64
	Active   bool
	Uptime   float64
	LastSeen time.Time
}

// NewAChainSource creates an A-Chain attestation source.
func NewAChainSource(rpcURL, wsURL string) *AChainSource {
	return &AChainSource{
		rpcURL:       rpcURL,
		wsURL:        wsURL,
		attestations: make(map[string]*Attestation),
		prices:       make(map[string]*Data),
		last:         make(map[string]time.Time),
		validators:   achainValidators(),
		interval:     200 * time.Millisecond, // Attestation interval
		healthy:      true,
		quorum:       3, // 3 of N validators for consensus
		lifecycle:    newLifecycle(),
	}
}

func achainValidators() map[string]*Validator {
	return map[string]*Validator{
		"validator-1": {
			Address:  "A-lux1validator1...",
			Stake:    1000000,
			Active:   true,
			Uptime:   0.999,
			LastSeen: time.Now(),
		},
		"validator-2": {
			Address:  "A-lux1validator2...",
			Stake:    500000,
			Active:   true,
			Uptime:   0.998,
			LastSeen: time.Now(),
		},
		"validator-3": {
			Address:  "A-lux1validator3...",
			Stake:    750000,
			Active:   true,
			Uptime:   0.997,
			LastSeen: time.Now(),
		},
		"validator-4": {
			Address:  "A-lux1validator4...",
			Stake:    250000,
			Active:   true,
			Uptime:   0.995,
			LastSeen: time.Now(),
		},
	}
}

// Supported symbols for attestation.
func achainSymbols() []string {
	return []string{
		"LUX-USD",
		"ETH-USD",
		"BTC-USD",
		"AVAX-USD",
	}
}

// Start begins polling A-Chain attestations.
func (s *AChainSource) Start() error {
	s.mu.Lock()
	if s.polling {
		s.mu.Unlock()
		return nil
	}
	s.polling = true
	s.mu.Unlock()

	s.run(s.loop)
	return nil
}

func (s *AChainSource) loop() {
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

func (s *AChainSource) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for _, symbol := range achainSymbols() {
		wg.Add(1)
		go func(sym string) {
			defer wg.Done()
			s.fetch(ctx, sym)
		}(symbol)
	}
	wg.Wait()
}

func (s *AChainSource) fetch(ctx context.Context, symbol string) {
	// In production: query A-Chain for latest attestation
	// Simulate attestation with validator signatures
	att := s.simulate(symbol)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.attestations[symbol] = att

	// Only use attested price if quorum reached
	if len(att.Signatures) >= s.quorum {
		s.prices[symbol] = &Data{
			Symbol:     symbol,
			Price:      att.Price,
			Confidence: s.confidence(att),
			Timestamp:  att.Timestamp,
			Source:     "a-chain",
			Stale:      !att.Finalized,
		}
		s.last[symbol] = time.Now()
	}
	s.healthy = true
}

func (s *AChainSource) simulate(symbol string) *Attestation {
	base := map[string]float64{
		"LUX-USD":  10.0,
		"ETH-USD":  3000.0,
		"BTC-USD":  60000.0,
		"AVAX-USD": 35.0,
	}

	b, ok := base[symbol]
	if !ok {
		b = 1.0
	}

	// Slight variation
	now := time.Now().UnixNano()
	v := float64(now%300) / 100000.0
	price := b * (1.0 + v)

	// Generate attestation hash
	hashData := []byte(symbol + time.Now().String())
	hash := sha256.Sum256(hashData)

	// Simulate validator signatures
	sigs := make([]Signature, 0)
	for name, val := range s.validators {
		if val.Active {
			sigs = append(sigs, Signature{
				Validator: name,
				Sig:       hash[:16], // Simulated sig
				Timestamp: time.Now(),
			})
		}
	}

	return &Attestation{
		Symbol:     symbol,
		Price:      price,
		Timestamp:  time.Now(),
		Signatures: sigs,
		Hash:       hex.EncodeToString(hash[:]),
		Block:      uint64(time.Now().Unix()),
		Finalized:  len(sigs) >= s.quorum,
	}
}

func (s *AChainSource) confidence(att *Attestation) float64 {
	// Confidence based on validator coverage
	total := len(s.validators)
	signed := len(att.Signatures)
	if total == 0 {
		return 0
	}
	base := float64(signed) / float64(total)
	if att.Finalized {
		return 0.9 + base*0.1 // 0.9-1.0 range
	}
	return base * 0.9 // Max 0.9 if not finalized
}

// Price returns the latest attested price for a symbol.
func (s *AChainSource) Price(symbol string) (*Data, error) {
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
func (s *AChainSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Attestation returns the latest attestation for a symbol.
func (s *AChainSource) Attestation(symbol string) (*Attestation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	att, ok := s.attestations[symbol]
	if !ok {
		return nil, ErrNotFound
	}
	return att, nil
}

// Validators returns active validators.
func (s *AChainSource) Validators() map[string]*Validator {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]*Validator)
	for k, v := range s.validators {
		result[k] = v
	}
	return result
}

// Verify checks if a price has valid attestation.
func (s *AChainSource) Verify(symbol string, price float64, tolerance float64) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	att, ok := s.attestations[symbol]
	if !ok || !att.Finalized {
		return false
	}

	// Check price within tolerance
	diff := (price - att.Price) / att.Price
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance
}

// Subscribe is a no-op (uses polling).
func (s *AChainSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *AChainSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns health status.
func (s *AChainSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy
}

// Name returns "a-chain".
func (s *AChainSource) Name() string { return "a-chain" }

// Weight returns 1.5 (validator attestations).
func (s *AChainSource) Weight() float64 { return 1.5 }

// Close stops the source.
func (s *AChainSource) Close() error {
	s.stop()
	s.mu.Lock()
	s.polling = false
	s.mu.Unlock()
	return nil
}

// SetQuorum sets minimum validator count for consensus.
func (s *AChainSource) SetQuorum(n int) {
	s.mu.Lock()
	s.quorum = n
	s.mu.Unlock()
}
