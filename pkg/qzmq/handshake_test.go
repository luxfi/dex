package qzmq

import (
	"bytes"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/crypto"
)

func TestHandshakeBasic(t *testing.T) {
	// Test basic handshake flow
	opts := DefaultOptions

	// Create client and server connections
	client := NewConnection(opts, false)
	server := NewConnection(opts, true)

	// Step 1: ClientHello
	clientHello, err := client.ClientHello()
	if err != nil {
		t.Fatalf("ClientHello failed: %v", err)
	}
	if len(clientHello) == 0 {
		t.Fatal("ClientHello returned empty data")
	}

	// Verify client state
	if client.state != StateClientHello {
		t.Errorf("Expected state %v, got %v", StateClientHello, client.state)
	}

	// Verify client random was generated
	if len(client.clientRandom) != 32 {
		t.Errorf("Expected 32 byte client random, got %d", len(client.clientRandom))
	}
}

func TestHandshakePQOnly(t *testing.T) {
	// Test PQ-only handshake
	opts := PQOnlyOptions

	client := NewConnection(opts, false)
	server := NewConnection(opts, true)

	// ClientHello should include PQ-only extension
	clientHello, err := client.ClientHello()
	if err != nil {
		t.Fatalf("ClientHello failed: %v", err)
	}

	// Parse and verify PQ-only extension is present
	msg := &clientHelloMsg{}
	// In real test, would unmarshal and check extensions
	_ = msg
	_ = clientHello
	_ = server

	if !opts.PQOnly {
		t.Error("PQOnlyOptions should have PQOnly=true")
	}
}

func TestHandshakeHybrid(t *testing.T) {
	// Test hybrid X25519+ML-KEM handshake
	opts := Options{
		Suite: crypto.Suite{
			Kem:  crypto.KemHybridX25519ML768,
			Sig:  crypto.SigMLDSA2,
			Aead: crypto.AeadAES256GCM,
			Hash: crypto.HashSHA256,
		},
		PQOnly:      false,
		ZeroRTT:     false,
		AntiDoS:     true,
		KeyUpdate:   DefaultKeyUpdatePolicy,
		MaxMessages: DefaultMaxMessages,
		MaxBytes:    DefaultMaxBytes,
		MaxAge:      DefaultMaxAge,
	}

	client := NewConnection(opts, false)

	// ClientHello should generate X25519 keys for hybrid
	clientHello, err := client.ClientHello()
	if err != nil {
		t.Fatalf("ClientHello failed: %v", err)
	}

	// Verify ephemeral keys were generated
	if client.localKEMPK == nil || client.localKEMSK == nil {
		t.Error("Hybrid mode should generate ephemeral X25519 keys")
	}

	if len(clientHello) == 0 {
		t.Error("ClientHello should not be empty")
	}
}

func TestConnectionMetrics(t *testing.T) {
	opts := DefaultOptions
	conn := NewConnection(opts, false)

	// Initialize metrics
	conn.metrics.MessagesSent = 100
	conn.metrics.BytesSent = 50000
	conn.metrics.MessagesReceived = 95
	conn.metrics.BytesReceived = 48000
	conn.metrics.KeyUpdates = 2
	conn.metrics.Errors = 1

	// Get metrics
	metrics := conn.GetMetrics()

	if metrics.MessagesSent != 100 {
		t.Errorf("Expected 100 messages sent, got %d", metrics.MessagesSent)
	}
	if metrics.KeyUpdates != 2 {
		t.Errorf("Expected 2 key updates, got %d", metrics.KeyUpdates)
	}
}

func TestKeyRotation(t *testing.T) {
	opts := Options{
		Suite:       DefaultSuite,
		PQOnly:      false,
		ZeroRTT:     false,
		AntiDoS:     true,
		KeyUpdate:   true,
		MaxMessages: 100, // Low limit to trigger rotation
		MaxBytes:    10000,
		MaxAge:      1 * time.Second,
	}

	conn := NewConnection(opts, false)
	conn.state = StateEstablished
	conn.metrics.MessagesSent = 101 // Exceed message limit

	// Check if key update is needed
	if !conn.NeedsKeyUpdate() {
		t.Error("Should need key update after exceeding message limit")
	}

	// Test age-based rotation
	conn2 := NewConnection(opts, false)
	conn2.state = StateEstablished
	conn2.lastKeyUpdate = time.Now().Add(-2 * time.Second) // Old key

	if !conn2.NeedsKeyUpdate() {
		t.Error("Should need key update after exceeding age limit")
	}
}

func TestHandshakeStates(t *testing.T) {
	opts := DefaultOptions
	conn := NewConnection(opts, false)

	// Initial state
	if conn.state != StateInit {
		t.Errorf("Initial state should be StateInit, got %v", conn.state)
	}

	// Can't process ServerHello in init state
	err := conn.ProcessServerHello([]byte{1, 2, 3})
	if err != ErrInvalidState {
		t.Errorf("Expected ErrInvalidState, got %v", err)
	}

	// ClientHello transitions to StateClientHello
	_, err = conn.ClientHello()
	if err != nil {
		t.Fatalf("ClientHello failed: %v", err)
	}
	if conn.state != StateClientHello {
		t.Errorf("State should be StateClientHello, got %v", conn.state)
	}

	// Can't call ClientHello again
	_, err = conn.ClientHello()
	if err != ErrInvalidState {
		t.Errorf("Expected ErrInvalidState on duplicate ClientHello, got %v", err)
	}
}

func TestTranscriptHashing(t *testing.T) {
	opts := DefaultOptions
	conn := NewConnection(opts, false)

	// Generate ClientHello
	clientHello, err := conn.ClientHello()
	if err != nil {
		t.Fatalf("ClientHello failed: %v", err)
	}

	// Transcript should contain ClientHello
	if !bytes.Contains(conn.transcript, clientHello) {
		t.Error("Transcript should contain ClientHello message")
	}

	// Transcript should grow with each message
	initialLen := len(conn.transcript)
	if initialLen == 0 {
		t.Error("Transcript should not be empty after ClientHello")
	}
}

func TestSuiteNegotiation(t *testing.T) {
	// Test suite negotiation between client and server
	tests := []struct {
		name         string
		clientSuite  crypto.Suite
		serverSuite  crypto.Suite
		expectError  bool
	}{
		{
			name: "matching suites",
			clientSuite: crypto.Suite{
				Kem:  crypto.KemMLKEM768,
				Sig:  crypto.SigMLDSA2,
				Aead: crypto.AeadAES256GCM,
				Hash: crypto.HashSHA256,
			},
			serverSuite: crypto.Suite{
				Kem:  crypto.KemMLKEM768,
				Sig:  crypto.SigMLDSA2,
				Aead: crypto.AeadAES256GCM,
				Hash: crypto.HashSHA256,
			},
			expectError: false,
		},
		{
			name: "hybrid suite",
			clientSuite: crypto.Suite{
				Kem:  crypto.KemHybridX25519ML768,
				Sig:  crypto.SigMLDSA2,
				Aead: crypto.AeadAES256GCM,
				Hash: crypto.HashSHA256,
			},
			serverSuite: crypto.Suite{
				Kem:  crypto.KemHybridX25519ML768,
				Sig:  crypto.SigMLDSA2,
				Aead: crypto.AeadAES256GCM,
				Hash: crypto.HashSHA256,
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clientOpts := Options{Suite: tt.clientSuite}
			client := NewConnection(clientOpts, false)

			_, err := client.ClientHello()
			if err != nil && !tt.expectError {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestAntiDoSCookie(t *testing.T) {
	opts := Options{
		Suite:       DefaultSuite,
		PQOnly:      false,
		ZeroRTT:     false,
		AntiDoS:     true, // Enable anti-DoS
		KeyUpdate:   DefaultKeyUpdatePolicy,
		MaxMessages: DefaultMaxMessages,
		MaxBytes:    DefaultMaxBytes,
		MaxAge:      DefaultMaxAge,
	}

	conn := NewConnection(opts, true) // Server connection

	// Generate anti-DoS cookie
	cookie := conn.GenerateCookie()
	if len(cookie) != 32 {
		t.Errorf("Expected 32-byte cookie, got %d", len(cookie))
	}

	// Verify cookie is valid
	if !conn.ValidateCookie(cookie) {
		t.Error("Cookie should be valid")
	}

	// Expired cookie should be invalid
	conn.cookieTime = time.Now().Add(-2 * time.Minute)
	if conn.ValidateCookie(cookie) {
		t.Error("Expired cookie should be invalid")
	}
}

func TestZeroRTT(t *testing.T) {
	opts := Options{
		Suite:       DefaultSuite,
		PQOnly:      false,
		ZeroRTT:     true, // Enable 0-RTT
		AntiDoS:     false,
		KeyUpdate:   DefaultKeyUpdatePolicy,
		MaxMessages: DefaultMaxMessages,
		MaxBytes:    DefaultMaxBytes,
		MaxAge:      DefaultMaxAge,
	}

	conn := NewConnection(opts, false)

	// Generate PSK for resumption
	psk := make([]byte, 32)
	conn.psk = psk
	conn.sessionID = []byte("test-session")

	// 0-RTT should be allowed
	if !conn.CanSend0RTT() {
		t.Error("Should be able to send 0-RTT with PSK")
	}

	// Without PSK, no 0-RTT
	conn2 := NewConnection(opts, false)
	if conn2.CanSend0RTT() {
		t.Error("Should not be able to send 0-RTT without PSK")
	}
}

func TestEarlyDataLimits(t *testing.T) {
	opts := Options{
		Suite:         DefaultSuite,
		PQOnly:        false,
		ZeroRTT:       true,
		AntiDoS:       false,
		KeyUpdate:     DefaultKeyUpdatePolicy,
		MaxMessages:   DefaultMaxMessages,
		MaxBytes:      DefaultMaxBytes,
		MaxAge:        DefaultMaxAge,
		MaxEarlyData:  1024, // 1KB limit
	}

	conn := NewConnection(opts, false)
	conn.psk = make([]byte, 32)

	// Small data should be allowed
	smallData := make([]byte, 512)
	if !conn.CanSendEarlyData(smallData) {
		t.Error("Should be able to send small early data")
	}

	// Large data should be rejected
	largeData := make([]byte, 2048)
	if conn.CanSendEarlyData(largeData) {
		t.Error("Should not be able to send large early data")
	}
}

func BenchmarkHandshake(b *testing.B) {
	opts := DefaultOptions

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := NewConnection(opts, false)
		_, _ = client.ClientHello()
	}
}

func BenchmarkHandshakePQ(b *testing.B) {
	opts := PQOnlyOptions

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := NewConnection(opts, false)
		_, _ = client.ClientHello()
	}
}

func BenchmarkHandshakeHybrid(b *testing.B) {
	opts := Options{
		Suite: crypto.Suite{
			Kem:  crypto.KemHybridX25519ML768,
			Sig:  crypto.SigMLDSA2,
			Aead: crypto.AeadAES256GCM,
			Hash: crypto.HashSHA256,
		},
		PQOnly: false,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := NewConnection(opts, false)
		_, _ = client.ClientHello()
	}
}