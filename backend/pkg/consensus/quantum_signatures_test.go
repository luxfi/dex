// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"bytes"
	"crypto/rand"
	"testing"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBLSSignatureGeneration tests BLS signature generation and verification
func TestBLSSignatureGeneration(t *testing.T) {
	// Generate BLS key pair
	secretKey, err := NewSecretKey()
	require.NoError(t, err)
	require.NotNil(t, secretKey)
	
	publicKey := secretKey.PublicKey()
	require.NotNil(t, publicKey)
	
	// Create message to sign
	message := []byte("Test message for BLS signature")
	
	// Sign the message
	signature := secretKey.Sign(message)
	require.NotNil(t, signature)
	
	// Verify the signature
	valid := Verify(publicKey, message, signature)
	assert.True(t, valid)
	
	// Test with wrong message
	wrongMessage := []byte("Wrong message")
	invalidSig := Verify(publicKey, wrongMessage, signature)
	assert.False(t, invalidSig)
}

// TestBLSSignatureAggregation tests BLS signature aggregation
func TestBLSSignatureAggregation(t *testing.T) {
	numSigners := 5
	message := []byte("Consensus message")
	
	// Generate multiple key pairs
	var secretKeys []*SecretKey
	var publicKeys []*PublicKey
	var signatures []*Signature
	
	for i := 0; i < numSigners; i++ {
		sk, err := NewSecretKey()
		require.NoError(t, err)
		
		secretKeys = append(secretKeys, sk)
		publicKeys = append(publicKeys, sk.PublicKey())
		signatures = append(signatures, sk.Sign(message))
	}
	
	// Aggregate signatures
	aggregatedSig, err := AggregateSignatures(signatures)
	require.NoError(t, err)
	require.NotNil(t, aggregatedSig)
	
	// Aggregate public keys
	aggregatedPubKey, err := AggregatePublicKeys(publicKeys)
	require.NoError(t, err)
	require.NotNil(t, aggregatedPubKey)
	
	// Verify aggregated signature
	valid := Verify(aggregatedPubKey, message, aggregatedSig)
	assert.True(t, valid)
}

// TestRingtailSignatures tests Ringtail post-quantum signatures
func TestRingtailSignatures(t *testing.T) {
	// Initialize Ringtail engine
	engine := NewRingtail()
	err := engine.Initialize(SecurityHigh)
	require.NoError(t, err)
	
	// Generate key pair
	sk, pk, err := engine.GenerateKeyPair()
	require.NoError(t, err)
	require.NotNil(t, sk)
	require.NotNil(t, pk)
	
	// Sign a message
	message := []byte("Post-quantum secure message")
	signature, err := engine.Sign(message, sk)
	require.NoError(t, err)
	require.NotNil(t, signature)
	
	// Verify the signature
	valid := engine.Verify(message, signature, pk)
	assert.True(t, valid)
	
	// Test with wrong message
	wrongMessage := []byte("Wrong message")
	invalidSig := engine.Verify(wrongMessage, signature, pk)
	assert.False(t, invalidSig)
}

// TestRingtailPrecomputation tests Ringtail precomputed shares
func TestRingtailPrecomputation(t *testing.T) {
	// Generate secret key
	seed := make([]byte, 32)
	_, err := rand.Read(seed)
	require.NoError(t, err)
	
	sk, _, err := KeyGen(seed)
	require.NoError(t, err)
	
	// Precompute share
	precomp, err := Precompute(sk)
	require.NoError(t, err)
	require.NotNil(t, precomp)
	
	// Use precomputed share for quick signing
	message := []byte("Quick sign message")
	share, err := QuickSign(precomp, message)
	require.NoError(t, err)
	require.NotNil(t, share)
	
	// Verify share length
	assert.Equal(t, 32, len(share))
}

// TestRingtailShareAggregation tests aggregating Ringtail shares
func TestRingtailShareAggregation(t *testing.T) {
	numShares := 5
	var shares []Share
	
	// Generate multiple shares
	for i := 0; i < numShares; i++ {
		share := make([]byte, 32)
		_, err := rand.Read(share)
		require.NoError(t, err)
		shares = append(shares, share)
	}
	
	// Aggregate shares into certificate
	cert, err := Aggregate(shares)
	require.NoError(t, err)
	require.NotNil(t, cert)
	
	// Verify certificate length
	assert.Equal(t, 32, len(cert))
	
	// Test with empty shares
	emptyCert, err := Aggregate([]Share{})
	assert.Error(t, err)
	assert.Nil(t, emptyCert)
}

// TestHybridSignatures tests hybrid Ringtail+BLS signatures
func TestHybridSignatures(t *testing.T) {
	// Initialize both signature schemes
	blsKey, err := NewSecretKey()
	require.NoError(t, err)
	
	ringtailEngine := NewRingtail()
	err = ringtailEngine.Initialize(SecurityHigh)
	require.NoError(t, err)
	
	ringtailSK, ringtailPK, err := ringtailEngine.GenerateKeyPair()
	require.NoError(t, err)
	
	// Create message
	message := []byte("Hybrid quantum-resistant message")
	
	// Sign with both schemes
	blsSig := blsKey.Sign(message)
	ringtailSig, err := ringtailEngine.Sign(message, ringtailSK)
	require.NoError(t, err)
	
	// Create hybrid signature structure
	type HybridSignature struct {
		BLS      *Signature
		Ringtail []byte
	}
	
	hybridSig := HybridSignature{
		BLS:      blsSig,
		Ringtail: ringtailSig,
	}
	
	// Verify both signatures
	blsValid := Verify(blsKey.PublicKey(), message, hybridSig.BLS)
	assert.True(t, blsValid)
	
	ringtailValid := ringtailEngine.Verify(message, hybridSig.Ringtail, ringtailPK)
	assert.True(t, ringtailValid)
}

// TestQuantumCertificateCreation tests creating quantum-resistant certificates
func TestQuantumCertificateCreation(t *testing.T) {
	// Create FPC DAG order book
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Create an order vertex
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
	
	// Verify certificate components
	assert.Equal(t, vertex.ID, cert.VertexID)
	assert.NotNil(t, cert.BLSSignature)
	assert.NotNil(t, cert.RingtailCert)
	assert.Equal(t, vertex.Height, cert.Height)
	assert.WithinDuration(t, time.Now(), cert.Timestamp, 1*time.Second)
}

// TestCertificateMessageGeneration tests certificate message creation
func TestCertificateMessageGeneration(t *testing.T) {
	dob, err := NewFPCDAGOrderBook("node1", "BTC-USD")
	require.NoError(t, err)
	defer dob.Shutdown()
	
	// Create two identical vertices
	order := &lx.Order{
		ID:    1,
		Type:  lx.Limit,
		Side:  lx.Buy,
		Price: 50000.0,
		Size:  1.0,
	}
	
	timestamp := time.Now()
	vertex1 := &OrderVertex{
		Order:     order,
		NodeID:    "node1",
		Height:    1,
		Timestamp: timestamp,
		Parents:   []ID{{1, 2, 3}},
	}
	
	vertex2 := &OrderVertex{
		Order:     order,
		NodeID:    "node1",
		Height:    1,
		Timestamp: timestamp,
		Parents:   []ID{{1, 2, 3}},
	}
	
	// Generate messages
	msg1 := dob.createCertificateMessage(vertex1)
	msg2 := dob.createCertificateMessage(vertex2)
	
	// Messages should be identical for identical vertices
	assert.True(t, bytes.Equal(msg1, msg2))
	
	// Change a field and verify message changes
	vertex2.Height = 2
	msg3 := dob.createCertificateMessage(vertex2)
	assert.False(t, bytes.Equal(msg1, msg3))
}

// TestQuantumFinalityCheck tests quantum finality verification
func TestQuantumFinalityCheck(t *testing.T) {
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
	
	// Generate quantum certificate with both signatures
	cert, ok := dob.generateQuantumCertificate(vertex.ID, vertex)
	require.True(t, ok)
	
	// Add certificate to storage
	dob.mu.Lock()
	dob.certificates[vertex.ID] = cert
	dob.mu.Unlock()
	
	// Check quantum finality
	initialFinality := dob.finalityCount.Load()
	dob.checkQuantumFinality()
	
	// Verify finality was achieved
	newFinality := dob.finalityCount.Load()
	assert.Greater(t, newFinality, initialFinality)
	
	// Verify vertex was finalized
	dob.mu.RLock()
	isFinalized := dob.finalized[vertex.ID]
	dob.mu.RUnlock()
	assert.True(t, isFinalized)
}

// TestSecurityLevels tests different Ringtail security levels
func TestSecurityLevels(t *testing.T) {
	testCases := []struct {
		name  string
		level SecurityLevel
	}{
		{"Low Security", SecurityLow},
		{"Medium Security", SecurityMedium},
		{"High Security", SecurityHigh},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			engine := NewRingtail()
			err := engine.Initialize(tc.level)
			assert.NoError(t, err)
			
			// Generate key pair at this security level
			sk, pk, err := engine.GenerateKeyPair()
			assert.NoError(t, err)
			assert.NotNil(t, sk)
			assert.NotNil(t, pk)
			
			// Sign and verify
			message := []byte("Security level test")
			sig, err := engine.Sign(message, sk)
			assert.NoError(t, err)
			
			valid := engine.Verify(message, sig, pk)
			assert.True(t, valid)
		})
	}
}

// BenchmarkBLSSignature benchmarks BLS signature operations
func BenchmarkBLSSignature(b *testing.B) {
	sk, _ := NewSecretKey()
	message := []byte("Benchmark message")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sk.Sign(message)
	}
}

// BenchmarkRingtailSignature benchmarks Ringtail signature operations
func BenchmarkRingtailSignature(b *testing.B) {
	engine := NewRingtail()
	_ = engine.Initialize(SecurityHigh)
	sk, _, _ := engine.GenerateKeyPair()
	message := []byte("Benchmark message")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.Sign(message, sk)
	}
}

// BenchmarkQuantumCertificate benchmarks quantum certificate generation
func BenchmarkQuantumCertificate(b *testing.B) {
	dob, _ := NewFPCDAGOrderBook("node1", "BTC-USD")
	defer dob.Shutdown()
	
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
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dob.generateQuantumCertificate(vertex.ID, vertex)
	}
}