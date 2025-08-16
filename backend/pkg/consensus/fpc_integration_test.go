// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFPCDAGOrderBookCreation tests creating an FPC DAG order book
func TestFPCDAGOrderBookCreation(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	require.NotNil(t, dob)
	
	assert.Equal(t, "node1", dob.nodeID)
	assert.NotNil(t, dob.orderBook)
	assert.NotNil(t, dob.blsKey)
	assert.NotNil(t, dob.ringtail)
	assert.NotNil(t, dob.quasar)
	
	// Check FPC configuration
	assert.True(t, dob.fpcConfig.Enable)
	assert.Equal(t, 0.55, dob.fpcConfig.ThetaMin)
	assert.Equal(t, 0.65, dob.fpcConfig.ThetaMax)
	assert.Equal(t, 256, dob.fpcConfig.VoteLimitPerBlock)
	
	// Clean up
	dob.Shutdown()
}

// TestFPCOrderAddition tests adding orders with quantum certificates
func TestFPCOrderAddition(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add a buy order
	buyOrder := &lx.Order{
		ID:        1,
		Type:      lx.Market,
		Side:      lx.Buy,
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	
	vertex, err := dob.AddOrder(buyOrder)
	require.NoError(t, err)
	require.NotNil(t, vertex)
	
	// Verify vertex properties
	assert.Equal(t, buyOrder, vertex.Order)
	assert.Equal(t, "node1", vertex.NodeID)
	assert.Equal(t, uint64(1), vertex.Height)
	
	// Check that precomputed share was created
	dob.mu.RLock()
	_, exists := dob.precomputed[vertex.ID]
	dob.mu.RUnlock()
	assert.True(t, exists)
	
	// Check that vertex is in DAG
	dob.mu.RLock()
	storedVertex, exists := dob.vertices[vertex.ID]
	dob.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, vertex, storedVertex)
	
	// Check that vote state was initialized
	dob.mu.RLock()
	voteState, exists := dob.votes[vertex.ID]
	dob.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, 1, voteState.Votes)
	assert.Equal(t, 1.0, voteState.Confidence)
}

// TestFPCConsensusRounds tests FPC voting rounds
func TestFPCConsensusRounds(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add an order
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	vertex, err := dob.AddOrder(order)
	require.NoError(t, err)
	
	// Simulate voting rounds
	for round := 1; round <= 10; round++ {
		err := dob.runFPCRound(round)
		require.NoError(t, err)
		
		// Check adaptive threshold
		expectedThreshold := dob.fpcConfig.ThetaMin + 
			(dob.fpcConfig.ThetaMax-dob.fpcConfig.ThetaMin)*float64(round)/10.0
		if round > 10 {
			expectedThreshold = dob.fpcConfig.ThetaMax
		}
		assert.InDelta(t, expectedThreshold, dob.voteThreshold, 0.01)
	}
	
	// After 10 rounds, threshold should be at maximum
	err = dob.runFPCRound(11)
	require.NoError(t, err)
	assert.Equal(t, dob.fpcConfig.ThetaMax, dob.voteThreshold)
}

// TestQuantumCertificateGeneration tests generating quantum-resistant certificates
func TestQuantumCertificateGeneration(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add an order
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	vertex, err := dob.AddOrder(order)
	require.NoError(t, err)
	
	// Generate quantum certificate
	cert, ok := dob.generateQuantumCertificate(vertex.ID, vertex)
	assert.True(t, ok)
	require.NotNil(t, cert)
	
	// Verify certificate properties
	assert.Equal(t, vertex.ID, cert.VertexID)
	assert.NotNil(t, cert.BLSSignature)
	assert.NotNil(t, cert.RingtailCert)
	assert.Equal(t, vertex.Height, cert.Height)
	assert.InDelta(t, dob.voteThreshold, cert.VoteThreshold, 0.01)
}

// TestQuasarDualCertificates tests Quasar dual-certificate protocol
func TestQuasarDualCertificates(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add an order
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	vertex, err := dob.AddOrder(order)
	require.NoError(t, err)
	
	// Check that vertex is tracked in Quasar
	// (In production, this would involve actual certificate generation)
	dob.mu.Lock()
	// Simulate generating a Quasar certificate
	dob.quasar.GenerateCertificate(vertex.ID)
	dob.mu.Unlock()
	
	// Check certificate count
	assert.Equal(t, 1, dob.quasar.CertificateCount())
	
	// Verify certificate exists
	assert.True(t, dob.quasar.HasCertificate(vertex.ID))
}

// TestDAGFinalization tests vertex finalization with quantum finality
func TestDAGFinalization(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add multiple orders to create a DAG
	var vertices []*OrderVertex
	for i := 0; i < 5; i++ {
		order := &lx.Order{
			ID:    uint64(i + 1),
			Type:  lx.Limit,
			Side:  lx.Buy,
			Price: 50000.0 + float64(i),
			Size:  1.0,
		}
		
		vertex, err := dob.AddOrder(order)
		require.NoError(t, err)
		vertices = append(vertices, vertex)
	}
	
	// Simulate consensus and finalization
	dob.mu.Lock()
	
	// Mark first vertex as finalized
	dob.finalizeVertex(vertices[0].ID)
	assert.True(t, dob.finalized[vertices[0].ID])
	
	// Generate quantum certificates for all vertices
	for _, v := range vertices {
		cert, ok := dob.generateQuantumCertificate(v.ID, v)
		if ok {
			dob.certificates[v.ID] = cert
		}
	}
	
	dob.mu.Unlock()
	
	// Check quantum finality
	dob.checkQuantumFinality()
	
	// Verify finality count increased
	assert.Greater(t, dob.finalityCount.Load(), uint64(0))
}

// TestConcurrentFPCOperations tests concurrent operations on FPC DAG
func TestConcurrentFPCOperations(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Start consensus in background
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	go func() {
		select {
		case <-ctx.Done():
			return
		default:
			_ = dob.RunFPCConsensus()
		}
	}()
	
	// Concurrently add orders
	var wg sync.WaitGroup
	numOrders := 100
	
	for i := 0; i < numOrders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			order := &lx.Order{
				ID:    uint64(id),
				Type:  lx.Limit,
				Side:  lx.Buy,
				Price: 50000.0 + float64(id%10),
				Size:  1.0,
			}
			
			_, err := dob.AddOrder(order)
			assert.NoError(t, err)
		}(i)
	}
	
	wg.Wait()
	
	// Verify all orders were added
	dob.mu.RLock()
	vertexCount := len(dob.vertices)
	dob.mu.RUnlock()
	
	assert.Equal(t, numOrders, vertexCount)
	
	// Wait for some consensus rounds
	time.Sleep(500 * time.Millisecond)
	
	// Check that some vertices were finalized
	dob.mu.RLock()
	finalizedCount := len(dob.finalized)
	dob.mu.RUnlock()
	
	assert.Greater(t, finalizedCount, 0)
}

// TestRemoteVertexProcessing tests processing vertices from remote nodes
func TestRemoteVertexProcessing(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Create a remote vertex
	remoteOrder := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Sell,
		Price: 51000.0,
		Size:  0.5,
	}
	
	remoteVertex := &OrderVertex{
		Order:     remoteOrder,
		NodeID:    "node2",
		Height:    1,
		Timestamp: time.Now(),
	}
	remoteVertex.ID = dob.generateVertexID(remoteVertex)
	
	// Create a quantum certificate for the remote vertex
	msg := dob.createCertificateMessage(remoteVertex)
	blsSig := dob.blsKey.Sign(msg)
	
	remoteCert := &QuantumCertificate{
		VertexID:      remoteVertex.ID,
		BLSSignature:  (*BLSAggregateSignature)(blsSig),
		RingtailCert:  []byte("mock-ringtail-cert"),
		Timestamp:     time.Now(),
		Height:        1,
		VoteThreshold: 0.55,
	}
	
	// Process the remote vertex
	err = dob.ProcessRemoteVertex(remoteVertex, remoteCert)
	require.NoError(t, err)
	
	// Verify vertex was added
	dob.mu.RLock()
	storedVertex, exists := dob.vertices[remoteVertex.ID]
	storedCert, certExists := dob.certificates[remoteVertex.ID]
	dob.mu.RUnlock()
	
	assert.True(t, exists)
	assert.Equal(t, remoteVertex, storedVertex)
	assert.True(t, certExists)
	assert.Equal(t, remoteCert, storedCert)
	
	// Verify vote state was created
	dob.mu.RLock()
	voteState, voteExists := dob.votes[remoteVertex.ID]
	dob.mu.RUnlock()
	
	assert.True(t, voteExists)
	assert.Equal(t, 1, voteState.Votes)
}

// TestFPCStats tests statistics collection
func TestFPCStats(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add some orders
	for i := 0; i < 5; i++ {
		order := &lx.Order{
			ID:    uint64(i + 1),
			Type:  lx.Limit,
			Side:  lx.Buy,
			Price: 50000.0 + float64(i),
			Size:  1.0,
		}
		_, err := dob.AddOrder(order)
		require.NoError(t, err)
	}
	
	// Get stats
	stats := dob.GetStats()
	
	// Verify stats
	assert.Equal(t, "node1", stats["node_id"])
	assert.Equal(t, 5, stats["vertices"])
	assert.Equal(t, 0, stats["finalized"])
	assert.GreaterOrEqual(t, stats["frontier_size"], 1)
	assert.Equal(t, uint64(0), stats["total_trades"])
	assert.Equal(t, uint64(0), stats["quantum_finality"])
	assert.Equal(t, 0, stats["certificates"])
	assert.InDelta(t, 0.55, stats["vote_threshold"], 0.01)
	assert.Equal(t, 5, stats["quasar_certs"])
	assert.Equal(t, 0, stats["quasar_skip_certs"])
	assert.Equal(t, 0, stats["peers"])
	assert.True(t, stats["fpc_enabled"].(bool))
}

// TestQuantumCertificateValidation tests validation of quantum certificates
func TestQuantumCertificateValidation(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Create a vertex
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	vertex := &OrderVertex{
		Order:     order,
		NodeID:    "node1",
		Height:    1,
		Timestamp: time.Now(),
	}
	vertex.ID = dob.generateVertexID(vertex)
	
	// Test nil certificate
	err = dob.validateQuantumCertificate(vertex, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "missing certificate")
	
	// Test certificate with wrong vertex ID
	wrongCert := &QuantumCertificate{
		VertexID:      [32]byte{1, 2, 3}, // Wrong ID
		VoteThreshold: 0.55,
	}
	err = dob.validateQuantumCertificate(vertex, wrongCert)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "vertex ID mismatch")
	
	// Test certificate with low threshold
	lowThresholdCert := &QuantumCertificate{
		VertexID:      vertex.ID,
		VoteThreshold: 0.40, // Below minimum
	}
	err = dob.validateQuantumCertificate(vertex, lowThresholdCert)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "vote threshold too low")
	
	// Test valid certificate
	validCert := &QuantumCertificate{
		VertexID:      vertex.ID,
		VoteThreshold: 0.60,
		Timestamp:     time.Now(),
		Height:        1,
	}
	err = dob.validateQuantumCertificate(vertex, validCert)
	assert.NoError(t, err)
}

// BLSAggregateSignature is a type alias for testing
type BLSAggregateSignature = BLSSignature

// BLSSignature is a mock type for testing
type BLSSignature struct {
	data []byte
}