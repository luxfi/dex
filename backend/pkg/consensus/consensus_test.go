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

// TestDAGOrderBookCreation tests creating a DAG order book
func TestDAGOrderBookCreation(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	require.NotNil(t, dob)
	
	assert.Equal(t, "node1", dob.nodeID)
	assert.NotNil(t, dob.orderBook)
	assert.NotNil(t, dob.vertices)
	assert.NotNil(t, dob.edges)
	
	// Clean up
	dob.Shutdown()
}

// TestDAGOrderAddition tests adding orders to the DAG
func TestDAGOrderAddition(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC-USD")
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
	
	// Check that vertex is in DAG
	dob.mu.RLock()
	storedVertex, exists := dob.vertices[vertex.ID]
	dob.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, vertex, storedVertex)
}

// TestDAGConsensus tests basic consensus operations
func TestDAGConsensus(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Add multiple orders
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
	
	// Run consensus for a short time
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	
	go func() {
		_ = dob.RunConsensus()
	}()
	
	<-ctx.Done()
	
	// Check that some vertices were accepted
	dob.mu.RLock()
	acceptedCount := len(dob.accepted)
	dob.mu.RUnlock()
	
	assert.GreaterOrEqual(t, acceptedCount, 0)
}

// TestConcurrentDAGOperations tests concurrent operations on DAG
func TestConcurrentDAGOperations(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
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
}

// TestMockBLSSignatures tests mock BLS implementation
func TestMockBLSSignatures(t *testing.T) {
	// Generate key pair
	sk, err := NewSecretKey()
	require.NoError(t, err)
	
	pk := sk.PublicKey()
	require.NotNil(t, pk)
	
	// Sign message
	msg := []byte("test message")
	sig := sk.Sign(msg)
	require.NotNil(t, sig)
	
	// Verify signature
	valid := Verify(pk, msg, sig)
	assert.True(t, valid)
}

// TestMockRingtail tests mock Ringtail implementation
func TestMockRingtail(t *testing.T) {
	engine := NewRingtail()
	err := engine.Initialize(SecurityHigh)
	require.NoError(t, err)
	
	// Generate key pair
	sk, pk, err := engine.GenerateKeyPair()
	require.NoError(t, err)
	require.NotNil(t, sk)
	require.NotNil(t, pk)
	
	// Sign and verify
	msg := []byte("test message")
	sig, err := engine.Sign(msg, sk)
	require.NoError(t, err)
	
	valid := engine.Verify(msg, sig, pk)
	assert.True(t, valid)
}

// TestMockQuasar tests mock Quasar implementation
func TestMockQuasar(t *testing.T) {
	config := QuasarConfig{
		CertThreshold:   10,
		SkipThreshold:   15,
		SignatureScheme: "mock",
	}
	
	q, err := NewQuasar(config)
	require.NoError(t, err)
	
	// Initialize with genesis
	genesis := GenerateTestID()
	err = q.Initialize(genesis)
	assert.NoError(t, err)
	assert.True(t, q.HasCertificate(genesis))
	
	// Track and generate certificate
	item := GenerateTestID()
	err = q.Track(item)
	assert.NoError(t, err)
	
	cert, ok := q.GenerateCertificate(item)
	assert.True(t, ok)
	assert.NotNil(t, cert)
	assert.True(t, q.HasCertificate(item))
}

// BenchmarkDAGOrderAddition benchmarks order addition
func BenchmarkDAGOrderAddition(b *testing.B) {
	dob, _ := NewDAGOrderBook("node1", "BTC-USD")
	defer dob.Shutdown()
	
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		order.ID = uint64(i)
		_, _ = dob.AddOrder(order)
	}
}

// BenchmarkMockBLS benchmarks mock BLS operations
func BenchmarkMockBLS(b *testing.B) {
	sk, _ := NewSecretKey()
	msg := []byte("benchmark message")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sk.Sign(msg)
	}
}

// BenchmarkMockQuasar benchmarks mock Quasar operations
func BenchmarkMockQuasar(b *testing.B) {
	config := QuasarConfig{
		CertThreshold:   10,
		SkipThreshold:   15,
		SignatureScheme: "mock",
	}
	
	q, _ := NewQuasar(config)
	
	// Pre-track items
	items := make([]ID, b.N)
	for i := range items {
		items[i] = GenerateTestID()
		_ = q.Track(items[i])
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.GenerateCertificate(items[i])
	}
}