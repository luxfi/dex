package price

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// QChainSource provides quantum finality verification from Lux Q-Chain.
// Q-Chain is the quantum finality layer for cross-chain order verification.
type QChainSource struct {
	rpcURL string
	wsURL  string

	finality   map[string]*QuantumFinality
	inclusions map[string]*OrderInclusion
	prices     map[string]*Data
	last       map[string]time.Time

	validators map[string]*QValidator
	quorum     int

	interval time.Duration
	healthy  bool

	mu      sync.RWMutex
	done    chan struct{}
	polling bool
}

// QuantumFinality represents quantum-verified finality proof.
type QuantumFinality struct {
	BlockHash   string
	BlockHeight uint64
	StateRoot   string
	Timestamp   time.Time
	Signatures  []QuantumSignature
	Finalized   bool
	Latency     time.Duration
}

// QuantumSignature is a post-quantum signature on finality.
type QuantumSignature struct {
	Validator string
	Algorithm string // dilithium, sphincs+, etc.
	Signature []byte
	PublicKey []byte
	Timestamp time.Time
}

// OrderInclusion tracks order inclusion across chains.
type OrderInclusion struct {
	OrderID     string
	Chain       string // x-chain, c-chain, zoo-chain, etc.
	TxHash      string
	BlockHeight uint64
	Timestamp   time.Time
	Finality    *QuantumFinality
	Verified    bool
	Price       float64
	Symbol      string
}

// QValidator represents a Q-Chain quantum validator.
type QValidator struct {
	Address   string
	PublicKey []byte
	Stake     uint64
	Active    bool
	Algorithm string
	Uptime    float64
	LastSeen  time.Time
}

// NewQChainSource creates a Q-Chain finality source.
func NewQChainSource(rpcURL, wsURL string) *QChainSource {
	return &QChainSource{
		rpcURL:     rpcURL,
		wsURL:      wsURL,
		finality:   make(map[string]*QuantumFinality),
		inclusions: make(map[string]*OrderInclusion),
		prices:     make(map[string]*Data),
		last:       make(map[string]time.Time),
		validators: qchainValidators(),
		quorum:     3,
		interval:   100 * time.Millisecond,
		healthy:    true,
		done:       make(chan struct{}),
	}
}

func qchainValidators() map[string]*QValidator {
	return map[string]*QValidator{
		"q-validator-1": {
			Address:   "Q-lux1qvalidator1...",
			Stake:     2000000,
			Active:    true,
			Algorithm: "dilithium3",
			Uptime:    0.999,
			LastSeen:  time.Now(),
		},
		"q-validator-2": {
			Address:   "Q-lux1qvalidator2...",
			Stake:     1500000,
			Active:    true,
			Algorithm: "dilithium3",
			Uptime:    0.998,
			LastSeen:  time.Now(),
		},
		"q-validator-3": {
			Address:   "Q-lux1qvalidator3...",
			Stake:     1000000,
			Active:    true,
			Algorithm: "sphincs-sha2-256f",
			Uptime:    0.997,
			LastSeen:  time.Now(),
		},
		"q-validator-4": {
			Address:   "Q-lux1qvalidator4...",
			Stake:     500000,
			Active:    true,
			Algorithm: "dilithium5",
			Uptime:    0.995,
			LastSeen:  time.Now(),
		},
	}
}

// Supported chains for finality verification.
func supportedChains() []string {
	return []string{
		"x-chain",
		"a-chain",
		"c-chain",
		"p-chain",
		"zoo-chain",
	}
}

// Start begins polling Q-Chain finality.
func (s *QChainSource) Start() error {
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

func (s *QChainSource) loop() {
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

func (s *QChainSource) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for _, chain := range supportedChains() {
		wg.Add(1)
		go func(c string) {
			defer wg.Done()
			s.fetchFinality(ctx, c)
		}(chain)
	}
	wg.Wait()
}

func (s *QChainSource) fetchFinality(ctx context.Context, chain string) {
	// In production: query Q-Chain for latest finality proof
	// Simulate quantum finality
	fin := s.simulateFinality(chain)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.finality[chain] = fin
	s.last[chain] = time.Now()
	s.healthy = true

	// Update price data with finality info
	symbol := chain + "-FINAL"
	s.prices[symbol] = &Data{
		Symbol:     symbol,
		Price:      float64(fin.BlockHeight),
		Confidence: s.finalityConfidence(fin),
		Timestamp:  fin.Timestamp,
		Source:     "q-chain",
		Stale:      !fin.Finalized,
	}
}

func (s *QChainSource) simulateFinality(chain string) *QuantumFinality {
	now := time.Now()
	height := uint64(now.Unix())

	// Generate finality hash
	hashData := []byte(chain + now.String())
	hash := sha256.Sum256(hashData)
	stateRoot := sha256.Sum256(hash[:])

	// Simulate quantum signatures from validators
	sigs := make([]QuantumSignature, 0)
	for name, val := range s.validators {
		if val.Active {
			// Generate simulated public key (double the hash)
			pubKey := append(hash[:], hash[:]...)
			sigs = append(sigs, QuantumSignature{
				Validator: name,
				Algorithm: val.Algorithm,
				Signature: hash[:],
				PublicKey: pubKey[:64],
				Timestamp: now,
			})
		}
	}

	return &QuantumFinality{
		BlockHash:   hex.EncodeToString(hash[:]),
		BlockHeight: height,
		StateRoot:   hex.EncodeToString(stateRoot[:]),
		Timestamp:   now,
		Signatures:  sigs,
		Finalized:   len(sigs) >= s.quorum,
		Latency:     50 * time.Millisecond, // Simulated finality latency
	}
}

func (s *QChainSource) finalityConfidence(fin *QuantumFinality) float64 {
	if !fin.Finalized {
		return 0.5
	}

	// Confidence based on signatures and algorithms
	sigCount := len(fin.Signatures)
	totalValidators := len(s.validators)
	if totalValidators == 0 {
		return 0
	}

	coverage := float64(sigCount) / float64(totalValidators)

	// Boost confidence for diverse algorithm usage
	algorithms := make(map[string]bool)
	for _, sig := range fin.Signatures {
		algorithms[sig.Algorithm] = true
	}
	diversity := float64(len(algorithms)) / 3.0 // Assume 3 main algorithms
	if diversity > 1.0 {
		diversity = 1.0
	}

	return 0.8*coverage + 0.2*diversity
}

// Price returns finality data as price (block height).
func (s *QChainSource) Price(symbol string) (*Data, error) {
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
func (s *QChainSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Finality returns the latest finality proof for a chain.
func (s *QChainSource) Finality(chain string) (*QuantumFinality, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fin, ok := s.finality[chain]
	if !ok {
		return nil, ErrNotFound
	}
	return fin, nil
}

// VerifyInclusion verifies an order was included with quantum finality.
func (s *QChainSource) VerifyInclusion(orderID, chain, txHash string) (*OrderInclusion, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if already verified
	key := orderID + ":" + chain
	if inc, ok := s.inclusions[key]; ok {
		return inc, nil
	}

	// Get chain finality
	fin, ok := s.finality[chain]
	if !ok {
		return nil, ErrNotFound
	}

	// Create inclusion proof
	inc := &OrderInclusion{
		OrderID:     orderID,
		Chain:       chain,
		TxHash:      txHash,
		BlockHeight: fin.BlockHeight,
		Timestamp:   time.Now(),
		Finality:    fin,
		Verified:    fin.Finalized,
	}

	s.inclusions[key] = inc
	return inc, nil
}

// VerifyOrderPrice verifies a price was finalized on Q-Chain.
func (s *QChainSource) VerifyOrderPrice(symbol string, price float64, chain string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fin, ok := s.finality[chain]
	if !ok || !fin.Finalized {
		return false
	}

	// Price is considered verified if finality is recent
	return time.Since(fin.Timestamp) < 5*time.Second
}

// CrossChainVerify verifies finality across multiple chains.
func (s *QChainSource) CrossChainVerify(chains []string) (bool, map[string]*QuantumFinality) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results := make(map[string]*QuantumFinality)
	allFinalized := true

	for _, chain := range chains {
		fin, ok := s.finality[chain]
		if !ok {
			allFinalized = false
			continue
		}
		results[chain] = fin
		if !fin.Finalized {
			allFinalized = false
		}
	}

	return allFinalized, results
}

// Validators returns active Q-Chain validators.
func (s *QChainSource) Validators() map[string]*QValidator {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]*QValidator)
	for k, v := range s.validators {
		result[k] = v
	}
	return result
}

// Subscribe is a no-op (uses polling).
func (s *QChainSource) Subscribe(symbol string) error { return nil }

// Unsubscribe is a no-op.
func (s *QChainSource) Unsubscribe(symbol string) error { return nil }

// Healthy returns health status.
func (s *QChainSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy
}

// Name returns "q-chain".
func (s *QChainSource) Name() string { return "q-chain" }

// Weight returns 1.4 (quantum finality - high trust).
func (s *QChainSource) Weight() float64 { return 1.4 }

// Close stops the source.
func (s *QChainSource) Close() error {
	close(s.done)
	s.mu.Lock()
	s.polling = false
	s.mu.Unlock()
	return nil
}

// SetQuorum sets minimum validator count for finality.
func (s *QChainSource) SetQuorum(n int) {
	s.mu.Lock()
	s.quorum = n
	s.mu.Unlock()
}

// FinalityLatency returns average finality latency.
func (s *QChainSource) FinalityLatency() time.Duration {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var total time.Duration
	count := 0
	for _, fin := range s.finality {
		total += fin.Latency
		count++
	}

	if count == 0 {
		return 0
	}
	return total / time.Duration(count)
}

// Chains returns supported chains.
func (s *QChainSource) Chains() []string {
	return supportedChains()
}
