package consensus

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIDString(t *testing.T) {
	id := GenerateTestID("test")
	str := id.String()
	assert.NotEmpty(t, str)
	assert.Len(t, str, 16) // 8 bytes in hex = 16 chars
}

func TestGenerateTestID(t *testing.T) {
	// Without args
	id1 := GenerateTestID()
	id2 := GenerateTestID()
	assert.NotEqual(t, id1, id2) // Should be unique

	// With string arg
	id3 := GenerateTestID("specific_string")
	id4 := GenerateTestID("specific_string")
	assert.Equal(t, id3, id4) // Same input = same output
}

func TestNewSecretKey(t *testing.T) {
	sk, err := NewSecretKey()
	require.NoError(t, err)
	assert.NotNil(t, sk)
	assert.Len(t, sk.Data, 32)
}

func TestSecretKeySign(t *testing.T) {
	sk, err := NewSecretKey()
	require.NoError(t, err)

	data := []byte("test message")
	sig := sk.Sign(data)

	assert.NotNil(t, sig)
	assert.Equal(t, data, sig.Data)
}

func TestNewCorona(t *testing.T) {
	rt := NewCorona()
	assert.NotNil(t, rt)
}

func TestCoronaInitialize(t *testing.T) {
	rt := NewCorona()

	tests := []struct {
		name  string
		level SecurityLevel
	}{
		{"Low", SecurityLow},
		{"Medium", SecurityMedium},
		{"High", SecurityHigh},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := rt.Initialize(tt.level)
			assert.NoError(t, err)
			assert.Equal(t, int(tt.level), rt.level)
		})
	}
}

func TestCoronaGenerateKeyPair(t *testing.T) {
	rt := NewCorona()
	err := rt.Initialize(SecurityHigh)
	require.NoError(t, err)

	sk, pk, err := rt.GenerateKeyPair()
	require.NoError(t, err)
	assert.Len(t, sk, 32)
	assert.Len(t, pk, 32)
}

func TestCoronaSignAndVerify(t *testing.T) {
	rt := NewCorona()
	err := rt.Initialize(SecurityHigh)
	require.NoError(t, err)

	sk, pk, err := rt.GenerateKeyPair()
	require.NoError(t, err)

	msg := []byte("test message")
	sig, err := rt.Sign(msg, sk)
	require.NoError(t, err)
	assert.Len(t, sig, 64)

	// Valid verification
	valid := rt.Verify(msg, sig, pk)
	assert.True(t, valid)

	// Invalid message should fail
	valid = rt.Verify([]byte("Wrong message"), sig, pk)
	assert.False(t, valid)

	// Invalid signature length should fail
	valid = rt.Verify(msg, []byte("short"), pk)
	assert.False(t, valid)
}

func TestSecurityLevels(t *testing.T) {
	assert.Equal(t, SecurityLevel(128), SecurityLow)
	assert.Equal(t, SecurityLevel(192), SecurityMedium)
	assert.Equal(t, SecurityLevel(256), SecurityHigh)
}

func TestNewQuasar(t *testing.T) {
	config := QuasarConfig{
		Threshold:       3,
		CertThreshold:   5,
		SkipThreshold:   2,
		SignatureScheme: "corona",
	}

	q, err := NewQuasar(config)
	require.NoError(t, err)
	assert.NotNil(t, q)
	assert.Equal(t, 5, q.certThreshold)
	assert.Equal(t, 2, q.skipThreshold)
}

func TestQuasarTrack(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("vertex1")
	err = q.Track(id)
	assert.NoError(t, err)

	// Check it's tracked
	q.mu.RLock()
	tracked := q.tracked[id]
	q.mu.RUnlock()
	assert.True(t, tracked)
}

func TestQuasarTrackNil(t *testing.T) {
	var q *Quasar
	err := q.Track(GenerateTestID())
	assert.NoError(t, err) // Should handle nil gracefully
}

func TestQuasarGenerateCertificate(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("test_cert")

	// Track the ID first (required before generating certificate)
	err = q.Track(id)
	require.NoError(t, err)

	cert, ok := q.GenerateCertificate(id)

	// Should return a certificate after tracking
	assert.NotNil(t, cert)
	assert.True(t, ok)
	assert.Equal(t, id, cert.VertexID)
	assert.Equal(t, 5, cert.Threshold)
}

func TestPrecompute(t *testing.T) {
	result, err := Precompute(nil)
	require.NoError(t, err)
	assert.NotNil(t, result)

	// Result should be 32 bytes
	bytes, ok := result.([]byte)
	assert.True(t, ok)
	assert.Len(t, bytes, 32)
}

func TestSignature(t *testing.T) {
	sig := &Signature{Data: []byte("signature_data")}
	assert.Equal(t, []byte("signature_data"), sig.Data)
}

func TestQuasarConcurrency(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	done := make(chan bool, 100)

	// Concurrent tracking
	for i := 0; i < 100; i++ {
		go func(idx int) {
			id := GenerateTestID()
			_ = q.Track(id)
			done <- true
		}(i)
	}

	// Wait for all
	for i := 0; i < 100; i++ {
		<-done
	}

	// Should have 100 tracked items
	q.mu.RLock()
	count := len(q.tracked)
	q.mu.RUnlock()
	assert.Equal(t, 100, count)
}

func BenchmarkGenerateTestID(b *testing.B) {
	for i := 0; i < b.N; i++ {
		GenerateTestID()
	}
}

func BenchmarkCoronaSign(b *testing.B) {
	rt := NewCorona()
	_ = rt.Initialize(SecurityHigh)
	sk, _, _ := rt.GenerateKeyPair()
	msg := []byte("benchmark message")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = rt.Sign(msg, sk)
	}
}

func BenchmarkCoronaVerify(b *testing.B) {
	rt := NewCorona()
	_ = rt.Initialize(SecurityHigh)
	sk, pk, _ := rt.GenerateKeyPair()
	msg := []byte("benchmark message")
	sig, _ := rt.Sign(msg, sk)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt.Verify(msg, sig, pk)
	}
}

// === Quasar Extended Tests ===

func TestQuasarHasCertificate(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("cert_test")

	// Should not have certificate before tracking
	assert.False(t, q.HasCertificate(id))

	// Track and generate certificate
	err = q.Track(id)
	require.NoError(t, err)
	_, ok := q.GenerateCertificate(id)
	require.True(t, ok)

	// Should have certificate now
	assert.True(t, q.HasCertificate(id))
}

func TestQuasarHasCertificateNil(t *testing.T) {
	var q *Quasar
	assert.False(t, q.HasCertificate(GenerateTestID()))
}

func TestQuasarHasSkipCertificate(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("skip_cert")
	assert.False(t, q.HasSkipCertificate(id))
}

func TestQuasarHasSkipCertificateNil(t *testing.T) {
	var q *Quasar
	assert.False(t, q.HasSkipCertificate(GenerateTestID()))
}

func TestQuasarGetCertificate(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("get_cert")

	// Not found before tracking
	cert, ok := q.GetCertificate(id)
	assert.Nil(t, cert)
	assert.False(t, ok)

	// Track and generate
	err = q.Track(id)
	require.NoError(t, err)
	_, ok = q.GenerateCertificate(id)
	require.True(t, ok)

	// Should be found now
	cert, ok = q.GetCertificate(id)
	assert.NotNil(t, cert)
	assert.True(t, ok)
	assert.Equal(t, id, cert.VertexID)
}

func TestQuasarGetCertificateNil(t *testing.T) {
	var q *Quasar
	cert, ok := q.GetCertificate(GenerateTestID())
	assert.Nil(t, cert)
	assert.False(t, ok)
}

func TestQuasarCertThreshold(t *testing.T) {
	config := QuasarConfig{CertThreshold: 10, SkipThreshold: 3}
	q, err := NewQuasar(config)
	require.NoError(t, err)
	assert.Equal(t, 10, q.CertThreshold())
}

func TestQuasarSkipThreshold(t *testing.T) {
	config := QuasarConfig{CertThreshold: 10, SkipThreshold: 3}
	q, err := NewQuasar(config)
	require.NoError(t, err)
	assert.Equal(t, 3, q.SkipThreshold())
}

func TestQuasarHealthCheck(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	err = q.HealthCheck()
	assert.NoError(t, err)
}

func TestQuasarHealthCheckNil(t *testing.T) {
	var q *Quasar
	err := q.HealthCheck()
	assert.Error(t, err)
}

func TestQuasarInitialize(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	genesis := GenerateTestID("genesis")
	err = q.Initialize(genesis)
	assert.NoError(t, err)

	// Genesis should have a certificate
	assert.True(t, q.HasCertificate(genesis))
}

func TestQuasarInitializeNil(t *testing.T) {
	var q *Quasar
	err := q.Initialize(GenerateTestID())
	assert.Error(t, err)
}

func TestQuasarCertificateCount(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	assert.Equal(t, 0, q.CertificateCount())

	// Track and generate
	id := GenerateTestID("count_test")
	err = q.Track(id)
	require.NoError(t, err)
	_, _ = q.GenerateCertificate(id)

	assert.Equal(t, 1, q.CertificateCount())
}

func TestQuasarCertificateCountNil(t *testing.T) {
	var q *Quasar
	assert.Equal(t, 0, q.CertificateCount())
}

func TestQuasarSkipCertificateCount(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	assert.Equal(t, 0, q.SkipCertificateCount())
}

func TestQuasarSkipCertificateCountNil(t *testing.T) {
	var q *Quasar
	assert.Equal(t, 0, q.SkipCertificateCount())
}

func TestQuasarGenerateCertificateNotTracked(t *testing.T) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	id := GenerateTestID("untracked")

	// Should fail for untracked ID
	cert, ok := q.GenerateCertificate(id)
	assert.Nil(t, cert)
	assert.False(t, ok)
}

func TestQuasarGenerateCertificateNil(t *testing.T) {
	var q *Quasar
	cert, ok := q.GenerateCertificate(GenerateTestID())
	assert.Nil(t, cert)
	assert.False(t, ok)
}

func TestQuasarGenerateCertificateWithProof(t *testing.T) {
	config := QuasarConfig{CertThreshold: 3, SkipThreshold: 2}
	q, err := NewQuasar(config)
	require.NoError(t, err)

	// Generate multiple certificates
	for i := 0; i < 5; i++ {
		id := GenerateTestID()
		_ = q.Track(id)
		_, _ = q.GenerateCertificate(id)
	}

	// New certificate should have proofs from previous ones
	id := GenerateTestID()
	_ = q.Track(id)
	cert, ok := q.GenerateCertificate(id)
	assert.True(t, ok)
	assert.LessOrEqual(t, len(cert.Proof), 3) // Max 3 proofs
}

// === DAGOrderBook Tests ===

func TestNewDAGOrderBook(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)
	assert.NotNil(t, dob)
	assert.Equal(t, "node1", dob.nodeID)
	assert.Equal(t, "BTC/USDC", dob.symbol)
	assert.NotNil(t, dob.vertices)
	assert.NotNil(t, dob.quasar)
}

func TestNewLuxDAGOrderBook(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "ETH/USDC")
	require.NoError(t, err)
	assert.NotNil(t, lux)
	assert.NotNil(t, lux.blsKey)
	assert.NotNil(t, lux.corona)
	assert.NotNil(t, lux.quasar)
	assert.NotNil(t, lux.votes)
	assert.NotNil(t, lux.certificates)
}

func TestDAGOrderBookAddOrder(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	order := &dex.Order{
		ID:    1,
		Side:  dex.Buy,
		Type:  dex.Limit,
		Price: 50000.0,
		Size:  1.0,
	}

	vertex, err := dob.AddOrder(order)
	require.NoError(t, err)
	assert.NotNil(t, vertex)
	assert.Equal(t, order, vertex.Order)
	assert.Equal(t, uint64(1), vertex.Height)
	assert.Equal(t, "node1", vertex.NodeID)
}

func TestDAGOrderBookAddMultipleOrders(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	for i := 1; i <= 5; i++ {
		order := &dex.Order{
			ID:    uint64(i),
			Side:  dex.Buy,
			Type:  dex.Limit,
			Price: 50000.0 + float64(i*100),
			Size:  1.0,
		}
		vertex, err := dob.AddOrder(order)
		require.NoError(t, err)
		assert.Equal(t, uint64(i), vertex.Height)
	}

	stats := dob.GetStats()
	assert.Equal(t, 5, stats["vertices"])
	// Note: LuxDAGOrderBook.GetStats() doesn't return height
	assert.NotNil(t, stats["finalized"])
}

func TestDAGOrderBookGetStats(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	stats := dob.GetStats()
	assert.Equal(t, "node1", stats["node_id"])
	assert.Equal(t, 0, stats["vertices"])
	assert.Equal(t, 0, stats["finalized"])
	assert.Equal(t, 0, stats["frontier_size"])
	// Note: LuxDAGOrderBook.GetStats() uses different fields than base DAGOrderBook
	assert.NotNil(t, stats["fpc_enabled"])
}

func TestLuxDAGOrderBookAddOrder(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	order := &dex.Order{
		ID:    1,
		Side:  dex.Buy,
		Type:  dex.Limit,
		Price: 50000.0,
		Size:  1.0,
	}

	vertex, err := lux.AddOrder(order)
	require.NoError(t, err)
	assert.NotNil(t, vertex)

	// Check vote state was initialized
	lux.mu.RLock()
	voteState, exists := lux.votes[vertex.ID]
	lux.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, 1, voteState.Votes)
	assert.Equal(t, 1.0, voteState.Confidence)
}

func TestLuxDAGOrderBookGetStats(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	stats := lux.GetStats()
	assert.Equal(t, "node1", stats["node_id"])
	assert.Equal(t, 0, stats["vertices"])
	assert.Equal(t, 0, stats["finalized"])
	assert.Equal(t, true, stats["fpc_enabled"])
	assert.NotNil(t, stats["vote_threshold"])
}

func TestLuxDAGOrderBookRunFPCRound(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Add an order
	order := &dex.Order{
		ID:    1,
		Side:  dex.Buy,
		Type:  dex.Limit,
		Price: 50000.0,
		Size:  1.0,
	}
	_, err = lux.AddOrder(order)
	require.NoError(t, err)

	// Run FPC round
	err = lux.runFPCRound(1)
	assert.NoError(t, err)

	// Threshold should update
	assert.GreaterOrEqual(t, lux.voteThreshold, lux.luxConfig.ThetaMin)
}

func TestLuxDAGOrderBookProcessRemoteVertex(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:        GenerateTestID("remote_vertex"),
		Order:     &dex.Order{ID: 1, Side: dex.Sell, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID:    "node2",
		Height:    1,
		Timestamp: time.Now(),
	}

	cert := &QuantumCertificate{
		VertexID:      vertex.ID,
		BLSSignature:  &Signature{Data: []byte("test")},
		CoronaCert:    []byte("cert"),
		Height:        1,
		VoteThreshold: 0.55,
	}

	err = lux.ProcessRemoteVertex(vertex, cert)
	assert.NoError(t, err)

	// Verify vertex was added
	lux.mu.RLock()
	_, exists := lux.vertices[vertex.ID]
	lux.mu.RUnlock()
	assert.True(t, exists)
}

func TestLuxDAGOrderBookProcessRemoteVertexNilCert(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:     GenerateTestID("remote"),
		Order:  &dex.Order{ID: 1, Side: dex.Sell, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID: "node2",
		Height: 1,
	}

	err = lux.ProcessRemoteVertex(vertex, nil)
	assert.Error(t, err)
}

func TestLuxDAGOrderBookProcessRemoteVertexMismatchID(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:     GenerateTestID("v1"),
		Order:  &dex.Order{ID: 1, Side: dex.Sell, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID: "node2",
		Height: 1,
	}

	cert := &QuantumCertificate{
		VertexID:      GenerateTestID("v2"), // Different ID
		VoteThreshold: 0.55,
	}

	err = lux.ProcessRemoteVertex(vertex, cert)
	assert.Error(t, err)
}

func TestLuxDAGOrderBookProcessRemoteVertexLowThreshold(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:     GenerateTestID("low_thresh"),
		Order:  &dex.Order{ID: 1, Side: dex.Sell, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID: "node2",
		Height: 1,
	}

	cert := &QuantumCertificate{
		VertexID:      vertex.ID,
		VoteThreshold: 0.01, // Too low
	}

	err = lux.ProcessRemoteVertex(vertex, cert)
	assert.Error(t, err)
}

func TestLuxDAGOrderBookGenerateQuantumCertificate(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	order := &dex.Order{
		ID:    1,
		Side:  dex.Buy,
		Type:  dex.Limit,
		Price: 50000.0,
		Size:  1.0,
	}

	vertex, err := lux.AddOrder(order)
	require.NoError(t, err)

	cert, ok := lux.generateQuantumCertificate(vertex.ID, vertex)
	assert.True(t, ok)
	assert.NotNil(t, cert)
	assert.Equal(t, vertex.ID, cert.VertexID)
	assert.Equal(t, vertex.Height, cert.Height)
	assert.NotNil(t, cert.BLSSignature)
}

func TestLuxDAGOrderBookValidateQuantumCertificate(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:     GenerateTestID("validate_cert"),
		Order:  &dex.Order{ID: 1, Side: dex.Buy, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID: "node1",
		Height: 1,
	}

	cert := &QuantumCertificate{
		VertexID:      vertex.ID,
		VoteThreshold: 0.60,
	}

	err = lux.validateQuantumCertificate(vertex, cert)
	assert.NoError(t, err)
}

func TestLuxDAGOrderBookProcessQuasarCertificates(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Add an order which tracks in quasar
	order := &dex.Order{
		ID:    1,
		Side:  dex.Buy,
		Type:  dex.Limit,
		Price: 50000.0,
		Size:  1.0,
	}
	vertex, err := lux.AddOrder(order)
	require.NoError(t, err)

	// Process quasar certificates
	lux.processQuasarCertificates()

	// Vertex should be finalized
	lux.mu.RLock()
	finalized := lux.finalized[vertex.ID]
	lux.mu.RUnlock()
	assert.True(t, finalized)
}

func TestLuxDAGOrderBookCheckQuantumFinality(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Add certificate directly
	id := GenerateTestID("quantum_finality")
	lux.mu.Lock()
	lux.certificates[id] = &QuantumCertificate{VertexID: id}
	lux.mu.Unlock()

	// Check quantum finality
	lux.checkQuantumFinality()

	// Should be finalized now
	lux.mu.RLock()
	finalized := lux.finalized[id]
	lux.mu.RUnlock()
	assert.True(t, finalized)
}

func TestLuxDAGOrderBookFinalizeVertex(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	id := GenerateTestID("finalize_test")

	lux.mu.Lock()
	lux.finalizeVertex(id)
	lux.mu.Unlock()

	lux.mu.RLock()
	finalized := lux.finalized[id]
	lux.mu.RUnlock()
	assert.True(t, finalized)
	assert.Equal(t, uint64(1), lux.finalityCount.Load())
}

func TestDAGOrderBookShutdown(t *testing.T) {
	dob, err := NewDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Start consensus in background
	go func() {
		_ = dob.RunConsensus()
	}()

	// Allow some time for consensus to run
	time.Sleep(50 * time.Millisecond)

	// Shutdown should not block
	dob.Shutdown()
}

func TestLuxDAGOrderBookShutdown(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Start consensus in background
	go func() {
		_ = lux.RunLuxConsensus()
	}()

	// Allow some time for consensus to run
	time.Sleep(50 * time.Millisecond)

	// Shutdown should not block
	lux.Shutdown()
}

func TestLuxDAGOrderBookRunFPCConsensus(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	// Start FPC consensus in background
	go func() {
		_ = lux.RunFPCConsensus()
	}()

	// Allow some time for consensus to run
	time.Sleep(100 * time.Millisecond)

	// Shutdown
	lux.Shutdown()
}

func TestLuxDAGOrderBookCreateCertificateMessage(t *testing.T) {
	lux, err := NewLuxDAGOrderBook("node1", "BTC/USDC")
	require.NoError(t, err)

	vertex := &OrderVertex{
		ID:     GenerateTestID("cert_msg"),
		Order:  &dex.Order{ID: 1, Side: dex.Buy, Type: dex.Limit, Price: 50000.0, Size: 1.0},
		NodeID: "node1",
		Height: 1,
	}

	msg := lux.createCertificateMessage(vertex)
	assert.NotNil(t, msg)
	assert.Len(t, msg, 32) // SHA256 output
}

func TestOrderVertexFields(t *testing.T) {
	vertex := &OrderVertex{
		ID:        GenerateTestID("test"),
		Order:     &dex.Order{ID: 1, Side: dex.Buy},
		NodeID:    "node1",
		Height:    10,
		Parents:   []ID{GenerateTestID("parent1"), GenerateTestID("parent2")},
		Timestamp: time.Now(),
		Trades:    []*dex.Trade{{ID: 1, Price: 50000.0}},
	}

	assert.Equal(t, "node1", vertex.NodeID)
	assert.Equal(t, uint64(10), vertex.Height)
	assert.Len(t, vertex.Parents, 2)
	assert.Len(t, vertex.Trades, 1)
}

func TestVoteState(t *testing.T) {
	vs := &VoteState{
		Votes:      5,
		Confidence: 0.75,
		Round:      3,
	}

	assert.Equal(t, 5, vs.Votes)
	assert.Equal(t, 0.75, vs.Confidence)
	assert.Equal(t, 3, vs.Round)
}

func TestQuantumCertificateFields(t *testing.T) {
	id := GenerateTestID("qc")
	cert := &QuantumCertificate{
		VertexID:      id,
		BLSSignature:  &Signature{Data: []byte("sig")},
		CoronaCert:    []byte("corona"),
		Timestamp:     time.Now(),
		Height:        5,
		VoteThreshold: 0.67,
		Threshold:     15,
		Item:          id,
		Proof:         []ID{GenerateTestID("proof1")},
	}

	assert.Equal(t, id, cert.VertexID)
	assert.NotNil(t, cert.BLSSignature)
	assert.NotNil(t, cert.CoronaCert)
	assert.Equal(t, uint64(5), cert.Height)
	assert.Equal(t, 0.67, cert.VoteThreshold)
	assert.Equal(t, 15, cert.Threshold)
	assert.Len(t, cert.Proof, 1)
}

func TestLuxConsensusConfig(t *testing.T) {
	config := LuxConsensusConfig{
		Enable:            true,
		ThetaMin:          0.55,
		ThetaMax:          0.65,
		VoteLimitPerBlock: 256,
		VoteThreshold:     0.55,
		RoundDuration:     50 * time.Millisecond,
		TimeWindow:        30 * time.Second,
	}

	assert.True(t, config.Enable)
	assert.Equal(t, 0.55, config.ThetaMin)
	assert.Equal(t, 0.65, config.ThetaMax)
	assert.Equal(t, 256, config.VoteLimitPerBlock)
}

// === Benchmarks ===

func BenchmarkDAGOrderBookAddOrder(b *testing.B) {
	dob, _ := NewDAGOrderBook("node1", "BTC/USDC")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		order := &dex.Order{
			ID:    uint64(i),
			Side:  dex.Buy,
			Type:  dex.Limit,
			Price: 50000.0,
			Size:  1.0,
		}
		_, _ = dob.AddOrder(order)
	}
}

func BenchmarkLuxDAGOrderBookAddOrder(b *testing.B) {
	lux, _ := NewLuxDAGOrderBook("node1", "BTC/USDC")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		order := &dex.Order{
			ID:    uint64(i),
			Side:  dex.Buy,
			Type:  dex.Limit,
			Price: 50000.0,
			Size:  1.0,
		}
		_, _ = lux.AddOrder(order)
	}
}

func BenchmarkQuasarTrackAndCertificate(b *testing.B) {
	config := QuasarConfig{CertThreshold: 5, SkipThreshold: 2}
	q, _ := NewQuasar(config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := GenerateTestID()
		_ = q.Track(id)
		_, _ = q.GenerateCertificate(id)
	}
}
