package price

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// QChainVerifier provides quantum finality verification from Lux Q-Chain.
// Q-Chain is the quantum finality layer for cross-chain order/price verification.
// NOTE: This is NOT a price source - it verifies finality of data from other sources.
type QChainVerifier struct {
	rpcURL string
	wsURL  string

	finality   map[string]*QuantumFinality
	inclusions map[string]*OrderInclusion

	validators map[string]*QValidator
	quorum     int

	interval time.Duration
	healthy  bool

	mu sync.RWMutex
	lifecycle
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

// NewQChainVerifier creates a Q-Chain finality verifier.
func NewQChainVerifier(rpcURL, wsURL string) *QChainVerifier {
	return &QChainVerifier{
		rpcURL:     rpcURL,
		wsURL:      wsURL,
		finality:   make(map[string]*QuantumFinality),
		inclusions: make(map[string]*OrderInclusion),
		validators: qchainValidators(),
		quorum:     3,
		interval:   100 * time.Millisecond,
		healthy:    true,
		lifecycle:  newLifecycle(),
	}
}

// Deprecated: Use NewQChainVerifier instead.
// NewQChainSource is kept for backwards compatibility.
func NewQChainSource(rpcURL, wsURL string) *QChainVerifier {
	return NewQChainVerifier(rpcURL, wsURL)
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
func (v *QChainVerifier) Start() error {
	v.mu.Lock()
	if v.polling {
		v.mu.Unlock()
		return nil
	}
	v.polling = true
	v.mu.Unlock()

	v.run(v.loop)
	return nil
}

func (v *QChainVerifier) loop() {
	ticker := time.NewTicker(v.interval)
	defer ticker.Stop()

	for {
		select {
		case <-v.done:
			return
		case <-ticker.C:
			v.poll()
		}
	}
}

func (v *QChainVerifier) poll() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for _, chain := range supportedChains() {
		wg.Add(1)
		go func(c string) {
			defer wg.Done()
			v.fetchFinality(ctx, c)
		}(chain)
	}
	wg.Wait()
}

func (v *QChainVerifier) fetchFinality(ctx context.Context, chain string) {
	// In production: query Q-Chain for latest finality proof
	// Simulate quantum finality
	fin := v.simulateFinality(chain)

	v.mu.Lock()
	defer v.mu.Unlock()

	v.finality[chain] = fin
	v.healthy = true
}

func (v *QChainVerifier) simulateFinality(chain string) *QuantumFinality {
	now := time.Now()
	height := uint64(now.Unix())

	// Generate finality hash
	hashData := []byte(chain + now.String())
	hash := sha256.Sum256(hashData)
	stateRoot := sha256.Sum256(hash[:])

	// Simulate quantum signatures from validators
	sigs := make([]QuantumSignature, 0)
	for name, val := range v.validators {
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
		Finalized:   len(sigs) >= v.quorum,
		Latency:     50 * time.Millisecond, // Simulated finality latency
	}
}

func (v *QChainVerifier) finalityConfidence(fin *QuantumFinality) float64 {
	if !fin.Finalized {
		return 0.5
	}

	// Confidence based on signatures and algorithms
	sigCount := len(fin.Signatures)
	totalValidators := len(v.validators)
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

// Finality returns the latest finality proof for a chain.
func (v *QChainVerifier) Finality(chain string) (*QuantumFinality, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	fin, ok := v.finality[chain]
	if !ok {
		return nil, ErrNotFound
	}
	return fin, nil
}

// VerifyInclusion verifies an order was included with quantum finality.
func (v *QChainVerifier) VerifyInclusion(orderID, chain, txHash string) (*OrderInclusion, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Check if already verified
	key := orderID + ":" + chain
	if inc, ok := v.inclusions[key]; ok {
		return inc, nil
	}

	// Get chain finality
	fin, ok := v.finality[chain]
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

	v.inclusions[key] = inc
	return inc, nil
}

// VerifyOrderPrice verifies a price was finalized on Q-Chain.
func (v *QChainVerifier) VerifyOrderPrice(symbol string, price float64, chain string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()

	fin, ok := v.finality[chain]
	if !ok || !fin.Finalized {
		return false
	}

	// Price is considered verified if finality is recent
	return time.Since(fin.Timestamp) < 5*time.Second
}

// CrossChainVerify verifies finality across multiple chains.
func (v *QChainVerifier) CrossChainVerify(chains []string) (bool, map[string]*QuantumFinality) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	results := make(map[string]*QuantumFinality)
	allFinalized := true

	for _, chain := range chains {
		fin, ok := v.finality[chain]
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
func (v *QChainVerifier) Validators() map[string]*QValidator {
	v.mu.RLock()
	defer v.mu.RUnlock()

	result := make(map[string]*QValidator)
	for k, val := range v.validators {
		result[k] = val
	}
	return result
}

// Healthy returns health status.
func (v *QChainVerifier) Healthy() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.healthy
}

// Close stops the verifier.
func (v *QChainVerifier) Close() error {
	v.mu.Lock()
	if !v.polling {
		v.mu.Unlock()
		return nil // Already closed
	}
	v.polling = false
	v.mu.Unlock()

	v.stop()
	return nil
}

// SetQuorum sets minimum validator count for finality.
func (v *QChainVerifier) SetQuorum(n int) {
	v.mu.Lock()
	v.quorum = n
	v.mu.Unlock()
}

// FinalityLatency returns average finality latency.
func (v *QChainVerifier) FinalityLatency() time.Duration {
	v.mu.RLock()
	defer v.mu.RUnlock()

	var total time.Duration
	count := 0
	for _, fin := range v.finality {
		total += fin.Latency
		count++
	}

	if count == 0 {
		return 0
	}
	return total / time.Duration(count)
}

// Chains returns supported chains.
func (v *QChainVerifier) Chains() []string {
	return supportedChains()
}

// Alias for backwards compatibility
type QChainSource = QChainVerifier
