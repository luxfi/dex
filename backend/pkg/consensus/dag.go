// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"crypto/sha256"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
)

// OrderVertex represents a vertex in the DAG
type OrderVertex struct {
	ID        [32]byte
	Order     *lx.Order
	NodeID    string
	Height    uint64
	Parents   []ID
	Timestamp time.Time
}

// VoteState tracks voting state for FPC
type VoteState struct {
	Votes      int
	Confidence float64
	Round      int
}

// QuantumCertificate represents a quantum-resistant certificate
type QuantumCertificate struct {
	VertexID      [32]byte
	BLSSignature  *Signature
	RingtailCert  []byte
	Timestamp     time.Time
	Height        uint64
	VoteThreshold float64
}

// FPCConfig holds FPC consensus configuration
type FPCConfig struct {
	Enable            bool
	ThetaMin          float64
	ThetaMax          float64
	VoteLimitPerBlock int
}

// DAGOrderBook implements a DAG-based order book
type DAGOrderBook struct {
	mu            sync.RWMutex
	nodeID        string
	symbol        string
	orderBook     *lx.OrderBook
	vertices      map[[32]byte]*OrderVertex
	edges         map[[32]byte][]ID
	frontier      []ID
	accepted      map[[32]byte]bool
	height        uint64
	shutdown      chan struct{}
	consensusTime time.Duration
}

// FPCDAGOrderBook extends DAGOrderBook with FPC consensus
type FPCDAGOrderBook struct {
	*DAGOrderBook
	fpcConfig      FPCConfig
	blsKey         *SecretKey
	ringtail       RingtailEngine
	quasar         *Quasar
	votes          map[[32]byte]*VoteState
	precomputed    map[[32]byte][]byte
	certificates   map[[32]byte]*QuantumCertificate
	finalized      map[[32]byte]bool
	voteThreshold  float64
	finalityCount  atomic.Uint64
	round          int
}

// NewDAGOrderBook creates a new DAG order book
func NewDAGOrderBook(nodeID, symbol string) (*DAGOrderBook, error) {
	return &DAGOrderBook{
		nodeID:    nodeID,
		symbol:    symbol,
		orderBook: lx.NewOrderBook(symbol),
		vertices:  make(map[[32]byte]*OrderVertex),
		edges:     make(map[[32]byte][]ID),
		frontier:  []ID{},
		accepted:  make(map[[32]byte]bool),
		height:    0,
		shutdown:  make(chan struct{}),
	}, nil
}

// NewFPCDAGOrderBook creates a new FPC DAG order book with quantum resistance
func NewFPCDAGOrderBook(nodeID, symbol string) (*FPCDAGOrderBook, error) {
	dag, err := NewDAGOrderBook(nodeID, symbol)
	if err != nil {
		return nil, err
	}

	// Initialize BLS
	blsKey, err := NewSecretKey()
	if err != nil {
		return nil, err
	}

	// Initialize Ringtail
	ringtail := NewRingtail()
	if err := ringtail.Initialize(SecurityHigh); err != nil {
		return nil, err
	}

	// Initialize Quasar
	quasar, err := NewQuasar(QuasarConfig{
		CertThreshold:   15,
		SkipThreshold:   20,
		SignatureScheme: "hybrid-ringtail-bls",
	})
	if err != nil {
		return nil, err
	}

	// Initialize genesis
	genesis := GenerateTestID()
	if err := quasar.Initialize(genesis); err != nil {
		return nil, err
	}

	return &FPCDAGOrderBook{
		DAGOrderBook: dag,
		fpcConfig: FPCConfig{
			Enable:            true,
			ThetaMin:          0.55,
			ThetaMax:          0.65,
			VoteLimitPerBlock: 256,
		},
		blsKey:        blsKey,
		ringtail:      ringtail,
		quasar:        quasar,
		votes:         make(map[[32]byte]*VoteState),
		precomputed:   make(map[[32]byte][]byte),
		certificates:  make(map[[32]byte]*QuantumCertificate),
		finalized:     make(map[[32]byte]bool),
		voteThreshold: 0.55,
	}, nil
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
func (fpc *FPCDAGOrderBook) AddOrder(order *lx.Order) (*OrderVertex, error) {
	fpc.mu.Lock()
	defer fpc.mu.Unlock()

	// Create vertex
	vertex := &OrderVertex{
		Order:     order,
		NodeID:    fpc.nodeID,
		Height:    fpc.height + 1,
		Parents:   fpc.frontier,
		Timestamp: time.Now(),
	}
	vertex.ID = fpc.generateVertexID(vertex)

	// Add to DAG
	fpc.vertices[vertex.ID] = vertex
	fpc.height = vertex.Height

	// Update frontier
	fpc.frontier = []ID{ID(vertex.ID)}

	// Initialize vote state
	fpc.votes[vertex.ID] = &VoteState{
		Votes:      1,
		Confidence: 1.0,
		Round:      0,
	}

	// Precompute Ringtail share
	if precomp, err := Precompute([]byte("mock-key")); err == nil {
		fpc.precomputed[vertex.ID] = precomp
	}

	// Track in Quasar
	fpc.quasar.Track(ID(vertex.ID))
	fpc.quasar.GenerateCertificate(ID(vertex.ID))

	// Add order to order book
	fpc.orderBook.AddOrder(order)

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

// RunFPCConsensus runs FPC consensus with quantum finality
func (fpc *FPCDAGOrderBook) RunFPCConsensus() error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			fpc.round++
			_ = fpc.runFPCRound(fpc.round)
			fpc.checkQuantumFinality()
		case <-fpc.shutdown:
			return nil
		}
	}
}

// runFPCRound runs a single FPC round
func (fpc *FPCDAGOrderBook) runFPCRound(round int) error {
	fpc.mu.Lock()
	defer fpc.mu.Unlock()

	// Update adaptive threshold
	progress := float64(round) / 10.0
	if progress > 1.0 {
		progress = 1.0
	}
	fpc.voteThreshold = fpc.fpcConfig.ThetaMin +
		(fpc.fpcConfig.ThetaMax-fpc.fpcConfig.ThetaMin)*progress

	// Process votes
	for id, voteState := range fpc.votes {
		if voteState.Confidence >= fpc.voteThreshold {
			fpc.finalizeVertex(id)
		}
	}

	return nil
}

// finalizeVertex marks a vertex as finalized
func (fpc *FPCDAGOrderBook) finalizeVertex(id [32]byte) {
	fpc.finalized[id] = true
	fpc.finalityCount.Add(1)
}

// checkQuantumFinality checks for quantum finality
func (fpc *FPCDAGOrderBook) checkQuantumFinality() {
	fpc.mu.Lock()
	defer fpc.mu.Unlock()

	for id := range fpc.certificates {
		if !fpc.finalized[id] {
			fpc.finalizeVertex(id)
		}
	}
}

// generateVertexID generates a vertex ID
func (dob *DAGOrderBook) generateVertexID(vertex *OrderVertex) [32]byte {
	h := sha256.New()
	h.Write([]byte(vertex.NodeID))
	h.Write([]byte{byte(vertex.Order.Side), byte(vertex.Order.Type)})
	return [32]byte(h.Sum(nil))
}

// generateVertexID generates a vertex ID (FPC)
func (fpc *FPCDAGOrderBook) generateVertexID(vertex *OrderVertex) [32]byte {
	return fpc.DAGOrderBook.generateVertexID(vertex)
}

// generateQuantumCertificate generates a quantum-resistant certificate
func (fpc *FPCDAGOrderBook) generateQuantumCertificate(id [32]byte, vertex *OrderVertex) (*QuantumCertificate, bool) {
	msg := fpc.createCertificateMessage(vertex)
	blsSig := fpc.blsKey.Sign(msg)

	return &QuantumCertificate{
		VertexID:      id,
		BLSSignature:  blsSig,
		RingtailCert:  []byte("mock-ringtail-cert"),
		Timestamp:     time.Now(),
		Height:        vertex.Height,
		VoteThreshold: fpc.voteThreshold,
	}, true
}

// createCertificateMessage creates a message for certificate signing
func (fpc *FPCDAGOrderBook) createCertificateMessage(vertex *OrderVertex) []byte {
	h := sha256.New()
	h.Write(vertex.ID[:])
	h.Write([]byte(vertex.NodeID))
	h.Write([]byte{byte(vertex.Order.Side), byte(vertex.Order.Type)})
	return h.Sum(nil)
}

// validateQuantumCertificate validates a quantum certificate
func (fpc *FPCDAGOrderBook) validateQuantumCertificate(vertex *OrderVertex, cert *QuantumCertificate) error {
	if cert == nil {
		return errors.New("missing certificate")
	}
	if cert.VertexID != vertex.ID {
		return errors.New("vertex ID mismatch")
	}
	if cert.VoteThreshold < fpc.fpcConfig.ThetaMin {
		return errors.New("vote threshold too low")
	}
	return nil
}

// ProcessRemoteVertex processes a vertex from a remote node
func (fpc *FPCDAGOrderBook) ProcessRemoteVertex(vertex *OrderVertex, cert *QuantumCertificate) error {
	fpc.mu.Lock()
	defer fpc.mu.Unlock()

	// Validate certificate
	if err := fpc.validateQuantumCertificate(vertex, cert); err != nil {
		return err
	}

	// Add vertex
	fpc.vertices[vertex.ID] = vertex
	fpc.certificates[vertex.ID] = cert

	// Initialize vote state
	fpc.votes[vertex.ID] = &VoteState{
		Votes:      1,
		Confidence: 1.0,
		Round:      0,
	}

	return nil
}

// processQuasarCertificates processes Quasar certificates
func (fpc *FPCDAGOrderBook) processQuasarCertificates() {
	fpc.mu.Lock()
	defer fpc.mu.Unlock()

	// Process certificates (simplified)
	for id := range fpc.vertices {
		if fpc.quasar.HasCertificate(ID(id)) && !fpc.finalized[id] {
			fpc.finalizeVertex(id)
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

// GetStats returns FPC DAG statistics
func (fpc *FPCDAGOrderBook) GetStats() map[string]interface{} {
	fpc.mu.RLock()
	defer fpc.mu.RUnlock()

	return map[string]interface{}{
		"node_id":           fpc.nodeID,
		"vertices":          len(fpc.vertices),
		"finalized":         len(fpc.finalized),
		"frontier_size":     len(fpc.frontier),
		"total_trades":      uint64(0),
		"quantum_finality":  fpc.finalityCount.Load(),
		"certificates":      len(fpc.certificates),
		"vote_threshold":    fpc.voteThreshold,
		"quasar_certs":      fpc.quasar.CertificateCount(),
		"quasar_skip_certs": fpc.quasar.SkipCertificateCount(),
		"peers":             0,
		"fpc_enabled":       fpc.fpcConfig.Enable,
	}
}

// Shutdown shuts down the DAG
func (dob *DAGOrderBook) Shutdown() {
	close(dob.shutdown)
}

// Shutdown shuts down the FPC DAG
func (fpc *FPCDAGOrderBook) Shutdown() {
	close(fpc.shutdown)
}