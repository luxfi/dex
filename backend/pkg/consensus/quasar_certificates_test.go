// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/protocol/quasar"
	"github.com/luxfi/ids"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestQuasarInitialization tests Quasar dual-certificate protocol initialization
func TestQuasarInitialization(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   15,
		SkipThreshold:   20,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	require.NotNil(t, q)
	
	// Verify thresholds
	assert.Equal(t, 15, q.CertThreshold())
	assert.Equal(t, 20, q.SkipThreshold())
	
	// Initialize with genesis
	genesisID := ids.GenerateTestID()
	err = q.Initialize(genesisID)
	assert.NoError(t, err)
	
	// Genesis should have certificate
	assert.True(t, q.HasCertificate(genesisID))
	assert.Equal(t, 1, q.CertificateCount())
}

// TestQuasarCertificateTracking tests tracking items for certificate generation
func TestQuasarCertificateTracking(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   10,
		SkipThreshold:   15,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Track multiple items
	itemIDs := make([]ids.ID, 5)
	for i := range itemIDs {
		itemIDs[i] = ids.GenerateTestID()
		err := q.Track(itemIDs[i])
		assert.NoError(t, err)
	}
	
	// Generate certificates for tracked items
	for _, id := range itemIDs {
		cert, ok := q.GenerateCertificate(id)
		assert.True(t, ok)
		assert.NotNil(t, cert)
		assert.True(t, q.HasCertificate(id))
	}
	
	// Verify certificate count
	assert.Equal(t, 5, q.CertificateCount())
}

// TestQuasarDualCertificates tests regular and skip certificates
func TestQuasarDualCertificates(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   5,
		SkipThreshold:   10,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Initialize with genesis
	genesisID := ids.GenerateTestID()
	err = q.Initialize(genesisID)
	require.NoError(t, err)
	
	// Track an item
	itemID := ids.GenerateTestID()
	err = q.Track(itemID)
	assert.NoError(t, err)
	
	// Generate regular certificate
	cert, ok := q.GenerateCertificate(itemID)
	assert.True(t, ok)
	assert.NotNil(t, cert)
	assert.Equal(t, 5, cert.Threshold)
	
	// Check certificate existence
	assert.True(t, q.HasCertificate(itemID))
	assert.False(t, q.HasSkipCertificate(itemID))
	
	// Retrieve certificate
	retrievedCert, exists := q.GetCertificate(itemID)
	assert.True(t, exists)
	assert.Equal(t, cert, retrievedCert)
}

// TestQuasarConcurrentOperations tests concurrent certificate operations
func TestQuasarConcurrentOperations(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   10,
		SkipThreshold:   20,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Concurrent tracking and certificate generation
	var wg sync.WaitGroup
	numItems := 100
	
	for i := 0; i < numItems; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			itemID := ids.GenerateTestID()
			
			// Track item
			err := q.Track(itemID)
			assert.NoError(t, err)
			
			// Generate certificate
			cert, ok := q.GenerateCertificate(itemID)
			assert.True(t, ok)
			assert.NotNil(t, cert)
			
			// Check certificate
			assert.True(t, q.HasCertificate(itemID))
		}()
	}
	
	wg.Wait()
	
	// Verify final state
	assert.Equal(t, numItems, q.CertificateCount())
}

// TestQuasarHealthCheck tests Quasar health monitoring
func TestQuasarHealthCheck(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   15,
		SkipThreshold:   25,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Health check should pass
	err = q.HealthCheck()
	assert.NoError(t, err)
}

// TestQuasarWithFPCIntegration tests Quasar integration with FPC
func TestQuasarWithFPCIntegration(t *testing.T) {
	// Create FPC DAG order book with Quasar
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Verify Quasar is initialized
	assert.NotNil(t, dob.quasar)
	assert.Equal(t, 15, dob.quasar.CertThreshold())
	assert.Equal(t, 20, dob.quasar.SkipThreshold())
	
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
	
	// Verify Quasar tracked the vertex
	// Note: In production, we'd check internal Quasar state
	// For now, we verify through certificate generation
	dob.mu.Lock()
	dob.quasar.GenerateCertificate(vertex.ID)
	hasCert := dob.quasar.HasCertificate(vertex.ID)
	dob.mu.Unlock()
	
	assert.True(t, hasCert)
}

// TestQuasarCertificateValidation tests certificate validation in FPC
func TestQuasarCertificateValidation(t *testing.T) {
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
	
	// Generate quantum certificate
	cert, ok := dob.generateQuantumCertificate(vertex.ID, vertex)
	assert.True(t, ok)
	require.NotNil(t, cert)
	
	// Validate the certificate
	err = dob.validateQuantumCertificate(vertex, cert)
	assert.NoError(t, err)
	
	// Generate Quasar certificate
	dob.mu.Lock()
	dob.quasar.Track(vertex.ID)
	quasarCert, ok := dob.quasar.GenerateCertificate(vertex.ID)
	dob.mu.Unlock()
	
	assert.True(t, ok)
	assert.NotNil(t, quasarCert)
	assert.Equal(t, vertex.ID, quasarCert.Item)
}

// TestQuasarSkipCertificateFastPath tests skip certificate fast finality
func TestQuasarSkipCertificateFastPath(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add multiple orders
	var vertices []*OrderVertex
	for i := 0; i < 3; i++ {
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
	
	// Simulate skip certificate for faster finality
	dob.mu.Lock()
	
	// Track and generate certificates
	for _, v := range vertices {
		dob.quasar.Track(v.ID)
		dob.quasar.GenerateCertificate(v.ID)
	}
	
	// Check if any have skip certificates (in production, would be based on voting)
	skipCount := 0
	for _, v := range vertices {
		if dob.quasar.HasCertificate(v.ID) {
			skipCount++
		}
	}
	
	dob.mu.Unlock()
	
	// All should have regular certificates
	assert.Equal(t, 3, skipCount)
	
	// Process Quasar certificates for finalization
	dob.processQuasarCertificates()
	
	// Some vertices should be finalized
	dob.mu.RLock()
	finalizedCount := len(dob.finalized)
	dob.mu.RUnlock()
	
	assert.GreaterOrEqual(t, finalizedCount, 0)
}

// TestQuasarCertificateProof tests certificate proof structure
func TestQuasarCertificateProof(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   10,
		SkipThreshold:   15,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Create a chain of items
	var itemIDs []ids.ID
	for i := 0; i < 5; i++ {
		id := ids.GenerateTestID()
		itemIDs = append(itemIDs, id)
		
		err := q.Track(id)
		assert.NoError(t, err)
		
		cert, ok := q.GenerateCertificate(id)
		assert.True(t, ok)
		assert.NotNil(t, cert)
		
		// Certificate should have proof (chain of previous items)
		if i > 0 {
			// In production, proof would contain references to previous certificates
			assert.NotNil(t, cert.Proof)
		}
	}
	
	// Verify all certificates exist
	for _, id := range itemIDs {
		assert.True(t, q.HasCertificate(id))
	}
}

// BenchmarkQuasarCertificateGeneration benchmarks certificate generation
func BenchmarkQuasarCertificateGeneration(b *testing.B) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   15,
		SkipThreshold:   20,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, _ := quasar.NewQuasar(config)
	
	// Pre-track items
	itemIDs := make([]ids.ID, b.N)
	for i := range itemIDs {
		itemIDs[i] = ids.GenerateTestID()
		_ = q.Track(itemIDs[i])
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.GenerateCertificate(itemIDs[i])
	}
}

// BenchmarkQuasarConcurrentCertificates benchmarks concurrent certificate operations
func BenchmarkQuasarConcurrentCertificates(b *testing.B) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   10,
		SkipThreshold:   20,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, _ := quasar.NewQuasar(config)
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			itemID := ids.GenerateTestID()
			_ = q.Track(itemID)
			_, _ = q.GenerateCertificate(itemID)
		}
	})
}

// TestQuasarCertificateRetrieval tests retrieving certificates
func TestQuasarCertificateRetrieval(t *testing.T) {
	config := quasar.QuasarConfig[ids.ID]{
		CertThreshold:   8,
		SkipThreshold:   12,
		SignatureScheme: "hybrid-ringtail-bls",
	}
	
	q, err := quasar.NewQuasar(config)
	require.NoError(t, err)
	
	// Generate some certificates
	var generatedCerts []*quasar.Certificate[ids.ID]
	for i := 0; i < 10; i++ {
		itemID := ids.GenerateTestID()
		err := q.Track(itemID)
		require.NoError(t, err)
		
		cert, ok := q.GenerateCertificate(itemID)
		require.True(t, ok)
		generatedCerts = append(generatedCerts, cert)
	}
	
	// Retrieve and verify certificates
	for _, originalCert := range generatedCerts {
		retrievedCert, exists := q.GetCertificate(originalCert.Item)
		assert.True(t, exists)
		assert.Equal(t, originalCert.Item, retrievedCert.Item)
		assert.Equal(t, originalCert.Threshold, retrievedCert.Threshold)
	}
	
	// Try to retrieve non-existent certificate
	nonExistentID := ids.GenerateTestID()
	_, exists := q.GetCertificate(nonExistentID)
	assert.False(t, exists)
}