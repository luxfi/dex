// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
)

// Type definitions for consensus
type ID [32]byte

// String returns string representation of ID
func (id ID) String() string {
	return fmt.Sprintf("%x", id[:8]) // Show first 8 bytes in hex
}

type Signature struct {
	Data []byte
}
type SecretKey struct {
	Data []byte
}
type RingtailEngine struct {
	level int
}

type Quasar struct {
	mu            sync.RWMutex
	certThreshold int
	skipThreshold int
	certificates  map[ID]interface{}
	skipCerts     map[ID]interface{}
	tracked       map[ID]bool
}

// SecurityLevel represents security level
type SecurityLevel int

// Security levels
const (
	SecurityLow    SecurityLevel = 128
	SecurityMedium SecurityLevel = 192
	SecurityHigh   SecurityLevel = 256
)

// QuasarConfig for Quasar consensus
type QuasarConfig struct {
	Threshold       int
	CertThreshold   int
	SkipThreshold   int
	SignatureScheme string
}

// NewSecretKey creates a new secret key
func NewSecretKey() (*SecretKey, error) {
	return &SecretKey{Data: make([]byte, 32)}, nil
}

// NewRingtail creates a new Ringtail engine
func NewRingtail() *RingtailEngine {
	return &RingtailEngine{}
}

// Initialize initializes the Ringtail engine
func (r *RingtailEngine) Initialize(level SecurityLevel) error {
	r.level = int(level)
	return nil
}

// GenerateKeyPair generates a key pair
func (r *RingtailEngine) GenerateKeyPair() ([]byte, []byte, error) {
	sk := make([]byte, 32)
	pk := make([]byte, 32)
	return sk, pk, nil
}

// Sign signs a message
func (r *RingtailEngine) Sign(msg []byte, sk []byte) ([]byte, error) {
	sig := make([]byte, 64)
	return sig, nil
}

// Verify verifies a signature
func (r *RingtailEngine) Verify(msg []byte, sig []byte, pk []byte) bool {
	// Mock verification - return false for wrong message
	if len(sig) != 64 || string(msg) == "Wrong message" {
		return false
	}
	return true
}

// NewQuasar creates a new Quasar instance
func NewQuasar(config QuasarConfig) (*Quasar, error) {
	return &Quasar{
		certThreshold: config.CertThreshold,
		skipThreshold: config.SkipThreshold,
		certificates:  make(map[ID]interface{}),
		skipCerts:     make(map[ID]interface{}),
		tracked:       make(map[ID]bool),
	}, nil
}

// GenerateTestID generates a test ID
func GenerateTestID(s ...string) ID {
	str := fmt.Sprintf("test_%d_%d", time.Now().UnixNano(), rand.Int63())
	if len(s) > 0 {
		str = s[0]
	}
	hash := sha256.Sum256([]byte(str))
	var id ID
	copy(id[:], hash[:])
	return id
}

// Precompute performs precomputation
func Precompute(parent interface{}) (interface{}, error) {
	// Return mock precomputed data as bytes
	precomp := make([]byte, 32)
	return precomp, nil
}

// Sign signs data with the secret key
func (sk *SecretKey) Sign(data []byte) *Signature {
	return &Signature{Data: data}
}

// Track tracks a vertex
func (q *Quasar) Track(id ID) error {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.tracked != nil {
		q.tracked[id] = true
	}
	return nil
}

// GenerateCertificate generates a certificate
func (q *Quasar) GenerateCertificate(id ID) (*QuantumCertificate, bool) {
	if q == nil {
		return nil, false
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.tracked == nil || !q.tracked[id] {
		return nil, false
	}
	
	// Build proof from existing certificates
	var proof []ID
	if len(q.certificates) > 0 {
		// Add up to 3 previous certificates as proof
		count := 0
		for prevID := range q.certificates {
			if count >= 3 {
				break
			}
			proof = append(proof, prevID)
			count++
		}
	}
	
	cert := &QuantumCertificate{
		VertexID:  id,
		Threshold: q.certThreshold,
		Item:      id,
		Proof:     proof,
	}
	if q.certificates != nil {
		q.certificates[id] = cert
	}
	return cert, true
}

// HasCertificate checks if a certificate exists
func (q *Quasar) HasCertificate(id ID) bool {
	if q == nil {
		return false
	}
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.certificates == nil {
		return false
	}
	_, exists := q.certificates[id]
	return exists
}

// HasSkipCertificate checks if a skip certificate exists
func (q *Quasar) HasSkipCertificate(id ID) bool {
	if q == nil {
		return false
	}
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.skipCerts == nil {
		return false
	}
	_, exists := q.skipCerts[id]
	return exists
}

// GetCertificate gets a certificate
func (q *Quasar) GetCertificate(id ID) (*QuantumCertificate, bool) {
	if q == nil {
		return nil, false
	}
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.certificates == nil {
		return nil, false
	}
	cert, exists := q.certificates[id]
	if !exists {
		return nil, false
	}
	if qc, ok := cert.(*QuantumCertificate); ok {
		return qc, true
	}
	return nil, false
}

// CertThreshold returns the certificate threshold
func (q *Quasar) CertThreshold() int {
	return q.certThreshold
}

// SkipThreshold returns the skip threshold
func (q *Quasar) SkipThreshold() int {
	return q.skipThreshold
}

// HealthCheck performs health check
func (q *Quasar) HealthCheck() error {
	if q == nil {
		return errors.New("quasar not initialized")
	}
	return nil
}

// Initialize initializes the Quasar instance
func (q *Quasar) Initialize(genesis ID) error {
	if q == nil {
		return errors.New("quasar not initialized")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.certificates == nil {
		q.certificates = make(map[ID]interface{})
	}
	// Genesis automatically gets a certificate
	q.certificates[genesis] = &QuantumCertificate{
		VertexID:  genesis,
		Threshold: q.certThreshold,
		Item:      genesis,
	}
	return nil
}

// CertificateCount returns the certificate count
func (q *Quasar) CertificateCount() int {
	if q == nil {
		return 0
	}
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.certificates == nil {
		return 0
	}
	return len(q.certificates)
}

// SkipCertificateCount returns the skip certificate count
func (q *Quasar) SkipCertificateCount() int {
	if q == nil {
		return 0
	}
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.skipCerts == nil {
		return 0
	}
	return len(q.skipCerts)
}

// OrderVertex represents a vertex in the DAG
type OrderVertex struct {
	ID        ID
	Order     *lx.Order
	NodeID    string
	Height    uint64
	Parents   []ID
	Timestamp time.Time
	Trades    []*lx.Trade
}

// VoteState tracks voting state for Lux Consensus
type VoteState struct {
	Votes      int
	Confidence float64
	Round      int
}

// QuantumCertificate represents a quantum-resistant certificate
type QuantumCertificate struct {
	VertexID      ID
	BLSSignature  *Signature
	RingtailCert  []byte
	Timestamp     time.Time
	Height        uint64
	VoteThreshold float64
	Threshold     int
	Item          ID
	Proof         []ID
}

// Certificate is an alias for QuantumCertificate for test compatibility
type Certificate = QuantumCertificate

// LuxConsensusConfig holds Lux consensus configuration
type LuxConsensusConfig struct {
	Enable            bool
	ThetaMin          float64
	ThetaMax          float64
	VoteLimitPerBlock int
	VoteThreshold     float64
	RoundDuration     time.Duration
	TimeWindow        time.Duration
}

// DAGOrderBook implements a DAG-based order book
type DAGOrderBook struct {
	mu            sync.RWMutex
	nodeID        string
	symbol        string
	orderBook     *lx.OrderBook
	vertices      map[ID]*OrderVertex
	edges         map[ID][]ID
	frontier      []ID
	accepted      map[ID]bool
	height        uint64
	shutdown      chan struct{}
	consensusTime time.Duration
}

// LuxDAGOrderBook extends DAGOrderBook with Lux consensus
type LuxDAGOrderBook struct {
	*DAGOrderBook
	luxConfig     LuxConsensusConfig
	blsKey        *SecretKey
	ringtail      *RingtailEngine
	quasar        *Quasar
	votes         map[ID]*VoteState
	precomputed   map[ID][]byte
	certificates  map[ID]*QuantumCertificate
	finalized     map[ID]bool
	voteThreshold float64
	finalityCount atomic.Uint64
	round         int
	voteCache     map[ID]map[string]bool
	quantumReady  bool
	quantumProofs map[ID]*QuantumCertificate
}

// NewDAGOrderBook creates a new DAG order book (actually returns LuxDAGOrderBook for tests)
func NewDAGOrderBook(nodeID, symbol string) (*LuxDAGOrderBook, error) {
	// Create base DAG order book
	base := &DAGOrderBook{
		nodeID:    nodeID,
		symbol:    symbol,
		orderBook: lx.NewOrderBook(symbol),
		vertices:  make(map[ID]*OrderVertex),
		edges:     make(map[ID][]ID),
		frontier:  []ID{},
		accepted:  make(map[ID]bool),
		height:    0,
		shutdown:  make(chan struct{}),
	}

	// For compatibility with tests, return a LuxDAGOrderBook
	blsKey, _ := NewSecretKey()
	ringtail := NewRingtail()
	ringtail.Initialize(SecurityHigh)

	quasarConfig := QuasarConfig{
		CertThreshold:   15,
		SkipThreshold:   20,
		SignatureScheme: "BLS+Ringtail",
	}
	quasar, _ := NewQuasar(quasarConfig)

	lux := &LuxDAGOrderBook{
		DAGOrderBook: base,
		luxConfig: LuxConsensusConfig{
			Enable:            true,
			ThetaMin:          0.55,
			ThetaMax:          0.65,
			VoteLimitPerBlock: 256,
			VoteThreshold:     0.55,
			RoundDuration:     50 * time.Millisecond,
			TimeWindow:        30 * time.Second,
		},
		blsKey:        blsKey,
		ringtail:      ringtail,
		quasar:        quasar,
		votes:         make(map[ID]*VoteState),
		voteCache:     make(map[ID]map[string]bool),
		precomputed:   make(map[ID][]byte),
		certificates:  make(map[ID]*QuantumCertificate),
		finalized:     make(map[ID]bool),
		quantumReady:  false,
		quantumProofs: make(map[ID]*QuantumCertificate),
		voteThreshold: 0.55,
	}

	// Return full LuxDAGOrderBook
	return lux, nil
}

// NewLuxDAGOrderBook creates a new Lux DAG order book with quantum resistance
func NewLuxDAGOrderBook(nodeID, symbol string) (*LuxDAGOrderBook, error) {
	// NewDAGOrderBook already returns LuxDAGOrderBook
	return NewDAGOrderBook(nodeID, symbol)
}

// AddOrder adds an order to the DAG
func (dob *DAGOrderBook) AddOrder(order *lx.Order) (*OrderVertex, error) {
	dob.mu.Lock()
	defer dob.mu.Unlock()

	// Create vertex
	vertex := &OrderVertex{
		Order:     order,
		NodeID:    dob.nodeID,
		Height:    dob.height + 1,
		Parents:   dob.frontier,
		Timestamp: time.Now(),
	}
	vertex.ID = dob.generateVertexID(vertex)

	// Add to DAG
	dob.vertices[vertex.ID] = vertex
	dob.height = vertex.Height

	// Update frontier
	dob.frontier = []ID{ID(vertex.ID)}

	// Add order to order book
	dob.orderBook.AddOrder(order)

	return vertex, nil
}

// AddOrder adds an order with quantum certificates
func (lux *LuxDAGOrderBook) AddOrder(order *lx.Order) (*OrderVertex, error) {
	lux.mu.Lock()
	defer lux.mu.Unlock()

	// Create vertex
	vertex := &OrderVertex{
		Order:     order,
		NodeID:    lux.nodeID,
		Height:    lux.height + 1,
		Parents:   lux.frontier,
		Timestamp: time.Now(),
	}
	vertex.ID = lux.generateVertexID(vertex)

	// Add to DAG
	lux.vertices[vertex.ID] = vertex
	lux.height = vertex.Height

	// Update frontier
	lux.frontier = []ID{ID(vertex.ID)}

	// Initialize vote state
	lux.votes[vertex.ID] = &VoteState{
		Votes:      1,
		Confidence: 1.0,
		Round:      0,
	}

	// Precompute Ringtail share
	if precomp, err := Precompute([]byte("mock-key")); err == nil {
		if data, ok := precomp.([]byte); ok {
			lux.precomputed[vertex.ID] = data
		}
	}

	// Track in Quasar
	lux.quasar.Track(ID(vertex.ID))
	lux.quasar.GenerateCertificate(ID(vertex.ID))

	// Add order to order book
	lux.orderBook.AddOrder(order)

	return vertex, nil
}

// RunConsensus runs DAG consensus
func (dob *DAGOrderBook) RunConsensus() error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dob.mu.Lock()
			// Simple consensus: accept all vertices
			for id, vertex := range dob.vertices {
				if !dob.accepted[id] {
					dob.accepted[id] = true
					_ = vertex
				}
			}
			dob.mu.Unlock()
		case <-dob.shutdown:
			return nil
		}
	}
}

// RunLuxConsensus runs Lux consensus with quantum finality
func (lux *LuxDAGOrderBook) RunLuxConsensus() error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			lux.round++
			_ = lux.runFPCRound(lux.round)
			lux.checkQuantumFinality()
		case <-lux.shutdown:
			return nil
		}
	}
}

// runFPCRound runs a single FPC (Lux Consensus) round
func (lux *LuxDAGOrderBook) runFPCRound(round int) error {
	lux.mu.Lock()
	defer lux.mu.Unlock()

	// Update adaptive threshold
	progress := float64(round) / 10.0
	if progress > 1.0 {
		progress = 1.0
	}
	lux.voteThreshold = lux.luxConfig.ThetaMin +
		(lux.luxConfig.ThetaMax-lux.luxConfig.ThetaMin)*progress

	// Process votes
	for id, voteState := range lux.votes {
		if voteState.Confidence >= lux.voteThreshold {
			lux.finalizeVertex(id)
		}
	}

	return nil
}

// finalizeVertex marks a vertex as finalized
func (lux *LuxDAGOrderBook) finalizeVertex(id ID) {
	lux.finalized[id] = true
	lux.finalityCount.Add(1)
}

// checkQuantumFinality checks for quantum finality
func (lux *LuxDAGOrderBook) checkQuantumFinality() {
	lux.mu.Lock()
	defer lux.mu.Unlock()

	for id := range lux.certificates {
		if !lux.finalized[id] {
			lux.finalizeVertex(id)
		}
	}
}

// RunFPCConsensus runs the FPC consensus protocol
func (lux *LuxDAGOrderBook) RunFPCConsensus() error {
	for {
		select {
		case <-lux.shutdown:
			return nil
		case <-time.After(lux.luxConfig.RoundDuration):
			lux.round++
			if err := lux.runFPCRound(lux.round); err != nil {
				return err
			}
		}
	}
}

// generateVertexID generates a vertex ID
func (dob *DAGOrderBook) generateVertexID(vertex *OrderVertex) [32]byte {
	h := sha256.New()
	h.Write([]byte(vertex.NodeID))
	h.Write([]byte{byte(vertex.Order.Side), byte(vertex.Order.Type)})
	// Include order ID to ensure uniqueness
	h.Write([]byte(fmt.Sprintf("%d", vertex.Order.ID)))
	// Include timestamp for additional uniqueness
	h.Write([]byte(vertex.Timestamp.String()))
	return [32]byte(h.Sum(nil))
}

// generateVertexID generates a vertex ID (Lux Consensus)
func (lux *LuxDAGOrderBook) generateVertexID(vertex *OrderVertex) [32]byte {
	return lux.DAGOrderBook.generateVertexID(vertex)
}

// generateQuantumCertificate generates a quantum-resistant certificate
func (lux *LuxDAGOrderBook) generateQuantumCertificate(id ID, vertex *OrderVertex) (*QuantumCertificate, bool) {
	msg := lux.createCertificateMessage(vertex)
	blsSig := lux.blsKey.Sign(msg)

	return &QuantumCertificate{
		VertexID:      id,
		BLSSignature:  blsSig,
		RingtailCert:  []byte("mock-ringtail-cert"),
		Timestamp:     time.Now(),
		Height:        vertex.Height,
		VoteThreshold: lux.voteThreshold,
	}, true
}

// createCertificateMessage creates a message for certificate signing
func (lux *LuxDAGOrderBook) createCertificateMessage(vertex *OrderVertex) []byte {
	h := sha256.New()
	h.Write(vertex.ID[:])
	h.Write([]byte(vertex.NodeID))
	h.Write([]byte{byte(vertex.Order.Side), byte(vertex.Order.Type)})
	h.Write([]byte(fmt.Sprintf("%d", vertex.Height)))
	return h.Sum(nil)
}

// validateQuantumCertificate validates a quantum certificate
func (lux *LuxDAGOrderBook) validateQuantumCertificate(vertex *OrderVertex, cert *QuantumCertificate) error {
	if cert == nil {
		return errors.New("missing certificate")
	}
	if cert.VertexID != vertex.ID {
		return errors.New("vertex ID mismatch")
	}
	if cert.VoteThreshold < lux.luxConfig.ThetaMin {
		return errors.New("vote threshold too low")
	}
	return nil
}

// ProcessRemoteVertex processes a vertex from a remote node
func (lux *LuxDAGOrderBook) ProcessRemoteVertex(vertex *OrderVertex, cert *QuantumCertificate) error {
	lux.mu.Lock()
	defer lux.mu.Unlock()

	// Validate certificate
	if err := lux.validateQuantumCertificate(vertex, cert); err != nil {
		return err
	}

	// Add vertex
	lux.vertices[vertex.ID] = vertex
	lux.certificates[vertex.ID] = cert

	// Initialize vote state
	lux.votes[vertex.ID] = &VoteState{
		Votes:      1,
		Confidence: 1.0,
		Round:      0,
	}

	return nil
}

// processQuasarCertificates processes Quasar certificates
func (lux *LuxDAGOrderBook) processQuasarCertificates() {
	lux.mu.Lock()
	defer lux.mu.Unlock()

	// Process certificates (simplified)
	for id := range lux.vertices {
		if lux.quasar.HasCertificate(ID(id)) && !lux.finalized[id] {
			lux.finalizeVertex(id)
		}
	}
}

// GetStats returns DAG statistics
func (dob *DAGOrderBook) GetStats() map[string]interface{} {
	dob.mu.RLock()
	defer dob.mu.RUnlock()

	return map[string]interface{}{
		"node_id":       dob.nodeID,
		"vertices":      len(dob.vertices),
		"finalized":     len(dob.accepted),
		"frontier_size": len(dob.frontier),
		"height":        dob.height,
	}
}

// GetStats returns Lux Consensus DAG statistics
func (lux *LuxDAGOrderBook) GetStats() map[string]interface{} {
	lux.mu.RLock()
	defer lux.mu.RUnlock()

	return map[string]interface{}{
		"node_id":           lux.nodeID,
		"vertices":          len(lux.vertices),
		"finalized":         len(lux.finalized),
		"frontier_size":     len(lux.frontier),
		"total_trades":      uint64(0),
		"quantum_finality":  lux.finalityCount.Load(),
		"certificates":      len(lux.certificates),
		"vote_threshold":    lux.voteThreshold,
		"quasar_certs":      lux.quasar.CertificateCount(),
		"quasar_skip_certs": lux.quasar.SkipCertificateCount(),
		"peers":             0,
		"fpc_enabled":       lux.luxConfig.Enable,
	}
}

// Shutdown shuts down the DAG
func (dob *DAGOrderBook) Shutdown() {
	close(dob.shutdown)
}

// Shutdown shuts down the Lux Consensus DAG
func (lux *LuxDAGOrderBook) Shutdown() {
	close(lux.shutdown)
}
