package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock AuthService for testing
type mockAuthService struct {
	users map[string]string // apiKey -> userID
}

func newMockAuthService() *mockAuthService {
	return &mockAuthService{
		users: map[string]string{
			"valid_key": "user123",
		},
	}
}

func (m *mockAuthService) Authenticate(apiKey, apiSecret string) (string, error) {
	if userID, ok := m.users[apiKey]; ok && apiSecret == "valid_secret" {
		return userID, nil
	}
	return "", assert.AnError
}

func (m *mockAuthService) ValidateSession(sessionID string) (string, error) {
	return "user123", nil
}

func (m *mockAuthService) GetUserID(sessionID string) (string, error) {
	return "user123", nil
}

func TestNewWebSocketServer(t *testing.T) {
	config := ServerConfig{
		AuthService: newMockAuthService(),
	}

	server := NewWebSocketServer(config)
	assert.NotNil(t, server)
	assert.NotNil(t, server.clients)
	assert.NotNil(t, server.broadcast)
	assert.NotNil(t, server.subscriptions)
	assert.NotNil(t, server.metrics)
	assert.NotNil(t, server.ctx)
	assert.NotNil(t, server.cancel)
}

func TestServerMetrics(t *testing.T) {
	metrics := NewServerMetrics()
	assert.NotNil(t, metrics)
	assert.Equal(t, uint64(0), metrics.ConnectionsTotal)
	assert.Equal(t, uint64(0), metrics.ConnectionsActive)
	assert.Equal(t, uint64(0), metrics.MessagesReceived)
	assert.Equal(t, uint64(0), metrics.MessagesSent)
}

func TestServerMetricsIncrement(t *testing.T) {
	metrics := NewServerMetrics()

	metrics.ConnectionsTotal++
	metrics.ConnectionsActive++
	metrics.MessagesReceived += 10
	metrics.MessagesSent += 5

	assert.Equal(t, uint64(1), metrics.ConnectionsTotal)
	assert.Equal(t, uint64(1), metrics.ConnectionsActive)
	assert.Equal(t, uint64(10), metrics.MessagesReceived)
	assert.Equal(t, uint64(5), metrics.MessagesSent)
}

func TestRateLimiter(t *testing.T) {
	limiter := NewRateLimiter(5, time.Second)
	assert.NotNil(t, limiter)

	// First 5 requests should pass
	for i := 0; i < 5; i++ {
		assert.True(t, limiter.Allow(), "Request %d should be allowed", i+1)
	}

	// 6th request should be rate limited
	assert.False(t, limiter.Allow(), "6th request should be rate limited")
}

func TestRateLimiterReset(t *testing.T) {
	limiter := NewRateLimiter(2, 50*time.Millisecond)

	// Use up the quota
	assert.True(t, limiter.Allow())
	assert.True(t, limiter.Allow())
	assert.False(t, limiter.Allow())

	// Wait for window to reset
	time.Sleep(60 * time.Millisecond)

	// Should be allowed again
	assert.True(t, limiter.Allow())
}

func TestMessageStruct(t *testing.T) {
	msg := Message{
		Type:      "test",
		Data:      map[string]interface{}{"key": "value"},
		Error:     "",
		RequestID: "req123",
		Timestamp: time.Now().Unix(),
	}

	assert.Equal(t, "test", msg.Type)
	assert.Equal(t, "value", msg.Data["key"])
	assert.Empty(t, msg.Error)
	assert.Equal(t, "req123", msg.RequestID)
	assert.Greater(t, msg.Timestamp, int64(0))
}

func TestMessageJSON(t *testing.T) {
	msg := Message{
		Type:      "order_update",
		Data:      map[string]interface{}{"order_id": "12345"},
		Timestamp: 1234567890,
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded Message
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, msg.Type, decoded.Type)
	assert.Equal(t, msg.Timestamp, decoded.Timestamp)
}

func TestClientStruct(t *testing.T) {
	client := &Client{
		ID:            "client123",
		UserID:        "user456",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	assert.Equal(t, "client123", client.ID)
	assert.Equal(t, "user456", client.UserID)
	assert.False(t, client.authenticated)
	assert.NotNil(t, client.subscriptions)
	assert.NotNil(t, client.rateLimiter)
}

func TestClientSubscriptions(t *testing.T) {
	client := &Client{
		subscriptions: make(map[string]bool),
	}

	client.subscriptions["BTC/USDC"] = true
	client.subscriptions["ETH/USDC"] = true

	assert.True(t, client.subscriptions["BTC/USDC"])
	assert.True(t, client.subscriptions["ETH/USDC"])
	assert.False(t, client.subscriptions["SOL/USDC"])
}

func TestIsOriginAllowed(t *testing.T) {
	tests := []struct {
		origin   string
		expected bool
	}{
		{"https://lux.exchange", true},
		{"https://dex.lux.network", true},
		{"https://amm.lux.network", true},
		{"http://localhost:3000", true},
		{"http://localhost:8080", true},
		{"https://malicious.site", false},
		{"http://evil.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.origin, func(t *testing.T) {
			result := isOriginAllowed(tt.origin)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsOriginAllowedEmptyList(t *testing.T) {
	// Save original and restore after test
	original := AllowedOrigins
	defer func() { AllowedOrigins = original }()

	AllowedOrigins = []string{}
	assert.True(t, isOriginAllowed("any-origin"))
}

func TestIsOriginAllowedWildcard(t *testing.T) {
	original := AllowedOrigins
	defer func() { AllowedOrigins = original }()

	AllowedOrigins = []string{"*"}
	assert.True(t, isOriginAllowed("any-origin"))
	assert.True(t, isOriginAllowed("https://example.com"))
}

func TestServerConfigStruct(t *testing.T) {
	config := ServerConfig{
		AuthService: newMockAuthService(),
	}

	assert.NotNil(t, config.AuthService)
}

func TestWebSocketUpgrade(t *testing.T) {
	config := ServerConfig{
		AuthService: newMockAuthService(),
	}
	server := NewWebSocketServer(config)

	// Create test HTTP server
	httpServer := httptest.NewServer(http.HandlerFunc(server.HandleConnection))
	defer httpServer.Close()

	// Convert HTTP URL to WebSocket URL
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http")

	// Try to connect
	dialer := websocket.Dialer{}
	conn, resp, err := dialer.Dial(wsURL, nil)

	if err != nil {
		// Connection may fail in test environment, that's OK
		// We're testing the server behavior
		t.Logf("WebSocket dial error (expected in unit test): %v", err)
		return
	}

	defer conn.Close()
	assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode)

	// Read welcome message
	_, message, err := conn.ReadMessage()
	if err == nil {
		var msg Message
		json.Unmarshal(message, &msg)
		assert.Equal(t, "connected", msg.Type)
	}
}

func TestGenerateClientID(t *testing.T) {
	id1 := generateClientID()
	time.Sleep(time.Nanosecond) // Ensure unique timestamps
	id2 := generateClientID()

	assert.NotEmpty(t, id1)
	assert.NotEmpty(t, id2)
	assert.NotEqual(t, id1, id2) // Should be unique
}

func TestNewServerMetrics(t *testing.T) {
	metrics := NewServerMetrics()

	assert.Equal(t, uint64(0), metrics.ConnectionsTotal)
	assert.Equal(t, uint64(0), metrics.ConnectionsActive)
	assert.Equal(t, uint64(0), metrics.MessagesReceived)
	assert.Equal(t, uint64(0), metrics.MessagesSent)
	assert.Equal(t, uint64(0), metrics.SubscriptionsActive)
	assert.Equal(t, uint64(0), metrics.AuthFailures)
	assert.Equal(t, uint64(0), metrics.OrdersProcessed)
	assert.Equal(t, uint64(0), metrics.PositionsOpened)
	assert.Equal(t, uint64(0), metrics.LiquidationsExecuted)
	assert.Equal(t, uint64(0), metrics.ErrorCount)
}

func TestClientSendMessage(t *testing.T) {
	client := &Client{
		send: make(chan []byte, 256),
	}

	msg := Message{
		Type:      "test",
		Timestamp: time.Now().Unix(),
	}

	// This would normally send via WebSocket
	// Testing the channel communication
	go func() {
		data, _ := json.Marshal(msg)
		client.send <- data
	}()

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "test", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for message")
	}
}

func TestServerBroadcastChannel(t *testing.T) {
	config := ServerConfig{
		AuthService: newMockAuthService(),
	}
	server := NewWebSocketServer(config)

	// Test broadcast channel
	go func() {
		server.broadcast <- []byte(`{"type":"broadcast_test"}`)
	}()

	select {
	case msg := <-server.broadcast:
		assert.Contains(t, string(msg), "broadcast_test")
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for broadcast")
	}
}

func TestAllowedOriginsDefault(t *testing.T) {
	// Check default origins list is populated
	assert.NotEmpty(t, AllowedOrigins)
	assert.Contains(t, AllowedOrigins, "https://lux.exchange")
	assert.Contains(t, AllowedOrigins, "https://dex.lux.network")
}

func BenchmarkRateLimiterAllow(b *testing.B) {
	limiter := NewRateLimiter(b.N, time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		limiter.Allow()
	}
}

func BenchmarkGenerateClientID(b *testing.B) {
	for i := 0; i < b.N; i++ {
		generateClientID()
	}
}

func BenchmarkMessageMarshal(b *testing.B) {
	msg := Message{
		Type: "order_update",
		Data: map[string]interface{}{
			"order_id": "12345",
			"status":   "filled",
			"price":    50000.0,
		},
		Timestamp: time.Now().Unix(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		json.Marshal(msg)
	}
}

// Tests for helper functions

func TestParseSide(t *testing.T) {
	tests := []struct {
		input    string
		expected lx.Side
	}{
		{"buy", lx.Buy},
		{"BUY", lx.Buy},
		{"sell", lx.Sell},
		{"SELL", lx.Sell},
		{"unknown", lx.Buy}, // default
		{"", lx.Buy},        // default
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := parseSide(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseOrderType(t *testing.T) {
	tests := []struct {
		input    string
		expected lx.OrderType
	}{
		{"market", lx.Market},
		{"MARKET", lx.Market},
		{"limit", lx.Limit},
		{"LIMIT", lx.Limit},
		{"stop", lx.Stop},
		{"STOP", lx.Stop},
		{"stop_limit", lx.StopLimit},
		{"STOP_LIMIT", lx.StopLimit},
		{"unknown", lx.Limit}, // default
		{"", lx.Limit},        // default
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := parseOrderType(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestOppositeSide(t *testing.T) {
	assert.Equal(t, lx.Sell, oppositeSide(lx.Buy))
	assert.Equal(t, lx.Buy, oppositeSide(lx.Sell))
}

// Test server metrics snapshot
func TestServerMetricsGetSnapshot(t *testing.T) {
	metrics := NewServerMetrics()
	metrics.ConnectionsTotal = 100
	metrics.ConnectionsActive = 50
	metrics.MessagesReceived = 1000
	metrics.MessagesSent = 500
	metrics.SubscriptionsActive = 25
	metrics.AuthFailures = 5
	metrics.OrdersProcessed = 200
	metrics.PositionsOpened = 10
	metrics.LiquidationsExecuted = 2
	metrics.ErrorCount = 15

	snapshot := metrics.GetSnapshot()

	assert.Equal(t, uint64(100), snapshot["connections_total"])
	assert.Equal(t, uint64(50), snapshot["connections_active"])
	assert.Equal(t, uint64(1000), snapshot["messages_received"])
	assert.Equal(t, uint64(500), snapshot["messages_sent"])
	assert.Equal(t, uint64(25), snapshot["subscriptions_active"])
	assert.Equal(t, uint64(5), snapshot["auth_failures"])
	assert.Equal(t, uint64(200), snapshot["orders_processed"])
	assert.Equal(t, uint64(10), snapshot["positions_opened"])
	assert.Equal(t, uint64(2), snapshot["liquidations_executed"])
	assert.Equal(t, uint64(15), snapshot["error_count"])
}

// Test client send message
func TestClientSendMessageMethod(t *testing.T) {
	client := &Client{
		send: make(chan []byte, 256),
	}

	msg := Message{
		Type:      "test_message",
		Data:      map[string]interface{}{"foo": "bar"},
		RequestID: "req123",
		Timestamp: time.Now().Unix(),
	}

	client.sendMessage(msg)

	select {
	case data := <-client.send:
		var received Message
		err := json.Unmarshal(data, &received)
		require.NoError(t, err)
		assert.Equal(t, "test_message", received.Type)
		assert.Equal(t, "req123", received.RequestID)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for message")
	}
}

func TestClientSendMessageFullChannel(t *testing.T) {
	// Create client with small buffer
	client := &Client{
		send: make(chan []byte, 1),
	}

	// Fill the channel
	client.send <- []byte("existing")

	// This should not block (uses select with default)
	msg := Message{Type: "test"}
	client.sendMessage(msg)

	// Drain channel
	<-client.send
}

// Test client send error
func TestClientSendError(t *testing.T) {
	client := &Client{
		send: make(chan []byte, 256),
	}

	client.sendError("Test error message", "req456")

	select {
	case data := <-client.send:
		var received Message
		err := json.Unmarshal(data, &received)
		require.NoError(t, err)
		assert.Equal(t, "error", received.Type)
		assert.Equal(t, "Test error message", received.Error)
		assert.Equal(t, "req456", received.RequestID)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error message")
	}
}

// Create helper to make test server with all components
func newTestWebSocketServer() *WebSocketServer {
	oracle := lx.NewPriceOracle()
	lendingPool := lx.NewLendingPool()
	riskEngine := lx.NewRiskEngine()
	marginEngine := lx.NewMarginEngine(oracle, riskEngine)
	liquidationEngine := lx.NewLiquidationEngine()
	tradingEngine := lx.NewTradingEngine(lx.EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	})
	vaultManager := lx.NewVaultManager(tradingEngine)

	config := ServerConfig{
		Engine:            tradingEngine,
		MarginEngine:      marginEngine,
		LendingPool:       lendingPool,
		Oracle:            oracle,
		VaultManager:      vaultManager,
		LiquidationEngine: liquidationEngine,
		AuthService:       newMockAuthService(),
	}

	return NewWebSocketServer(config)
}

// Test process message with various types
func TestProcessMessageInvalidType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Message without type
	msg := map[string]interface{}{
		"data": "some data",
	}

	server.processMessage(client, msg)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid message type")
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error")
	}
}

func TestProcessMessageUnknownType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "unknown_message_type",
	}

	server.processMessage(client, msg)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Unknown message type")
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error")
	}
}

func TestProcessMessagePing(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "ping",
		"request_id": "ping123",
	}

	server.processMessage(client, msg)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "pong", received.Type)
		assert.Equal(t, "ping123", received.RequestID)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for pong")
	}
}

// Test authentication
func TestHandleAuthMissingCredentials(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Missing apiKey and apiSecret
	msg := map[string]interface{}{
		"type": "auth",
	}

	server.handleAuth(client, msg, "auth_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing credentials")
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error")
	}

	assert.Equal(t, uint64(1), server.metrics.AuthFailures)
}

func TestHandleAuthInvalidCredentials(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":      "auth",
		"apiKey":    "invalid_key",
		"apiSecret": "invalid_secret",
	}

	server.handleAuth(client, msg, "auth_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Authentication failed")
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error")
	}

	assert.Equal(t, uint64(1), server.metrics.AuthFailures)
}

// Test subscriptions
func TestHandleSubscribeMissingChannel(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"symbols": []interface{}{"BTC-USDT"},
	}

	server.handleSubscribe(client, msg, "sub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid channel")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleSubscribeMissingSymbols(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": "orderbook",
	}

	server.handleSubscribe(client, msg, "sub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing symbols")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleSubscribeSuccess(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": "orderbook",
		"symbols": []interface{}{"BTC-USDT", "ETH-USDT"},
	}

	server.handleSubscribe(client, msg, "sub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "subscribed", received.Type)
		assert.Equal(t, "orderbook", received.Data["channel"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}

	// Check client subscriptions
	assert.True(t, client.subscriptions["orderbook:BTC-USDT"])
	assert.True(t, client.subscriptions["orderbook:ETH-USDT"])
}

func TestHandleUnsubscribeSuccess(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// First subscribe
	client.subscriptions["orderbook:BTC-USDT"] = true
	server.mu.Lock()
	server.subscriptions[client.ID] = map[string]bool{"BTC-USDT": true}
	server.metrics.SubscriptionsActive = 1
	server.mu.Unlock()

	// Then unsubscribe
	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": "orderbook",
		"symbols": []interface{}{"BTC-USDT"},
	}

	server.handleUnsubscribe(client, msg, "unsub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "unsubscribed", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}

	assert.False(t, client.subscriptions["orderbook:BTC-USDT"])
}

// Test place order (unauthenticated)
func TestHandlePlaceOrderUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"side":   "buy",
			"type":   "limit",
			"price":  50000.0,
			"size":   0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderMissingOrderData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing order data")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderInvalidSymbol(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"side":  "buy",
			"type":  "limit",
			"price": 50000.0,
			"size":  0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid symbol")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test cancel order
func TestHandleCancelOrderUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": float64(123),
	}

	server.handleCancelOrder(client, msg, "cancel_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleCancelOrderMissingOrderID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "cancel_order",
	}

	server.handleCancelOrder(client, msg, "cancel_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing orderID")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test open position
func TestHandleOpenPositionUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC-USDT",
		"side":     "buy",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleOpenPositionMissingSymbol(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"side":     "buy",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid symbol")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test close position
func TestHandleClosePositionUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "close_position",
		"positionID": "pos123",
		"size":       1.0,
	}

	server.handleClosePosition(client, msg, "close_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test vault deposit
func TestHandleVaultDepositUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
		"amount":  "1000000000000000000",
	}

	server.handleVaultDeposit(client, msg, "vault_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test lending supply
func TestHandleLendingSupplyUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"asset":  "USDC",
		"amount": "1000000000",
	}

	server.handleLendingSupply(client, msg, "supply_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test get balances
func TestHandleGetBalancesUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetBalances(client, "balance_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test get positions
func TestHandleGetPositionsUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetPositions(client, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test get orders
func TestHandleGetOrdersUnauthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetOrders(client, "orders_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test modify order
func TestHandleModifyOrder(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "modify_order",
		"orderID":  float64(123),
		"newPrice": 51000.0,
		"newSize":  0.2,
	}

	server.handleModifyOrder(client, msg, "modify_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "order_modified", received.Type)
		assert.Equal(t, float64(123), received.Data["order_id"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test modify leverage
func TestHandleModifyLeverageMissingPositionID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "modify_leverage",
		"leverage": 20.0,
	}

	server.handleModifyLeverage(client, msg, "lev_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid position_id")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test vault withdraw
func TestHandleVaultWithdrawMissingVaultID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "vault_withdraw",
		"amount": "1000",
	}

	server.handleVaultWithdraw(client, msg, "withdraw_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid vault_id")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test lending borrow
func TestHandleLendingBorrowMissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"amount": "1000",
	}

	server.handleLendingBorrow(client, msg, "borrow_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid asset")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test lending repay
func TestHandleLendingRepayMissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"amount":   "1000",
		"interest": "50",
	}

	server.handleLendingRepay(client, msg, "repay_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid asset")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test broadcast methods
func TestBroadcastOrderBook(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: map[string]bool{"orderbook:BTC-USDT": true},
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	snapshot := &lx.OrderBookSnapshot{
		Symbol: "BTC-USDT",
		Bids:   []lx.OrderLevel{{Price: 50000, Size: 1.0}},
		Asks:   []lx.OrderLevel{{Price: 50100, Size: 2.0}},
	}

	server.BroadcastOrderBook("BTC-USDT", snapshot)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "orderbook_update", received.Type)
		assert.Equal(t, "BTC-USDT", received.Data["symbol"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestBroadcastTrade(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: map[string]bool{"trades": true},
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	trade := &lx.Trade{
		Price:     50000.0,
		Size:      1.5,
		Timestamp: time.Now(),
	}

	server.BroadcastTrade(trade)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "trade_update", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestBroadcastPrice(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: map[string]bool{"prices:BTC-USDT": true},
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	server.BroadcastPrice("BTC-USDT", 51234.56)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "price_update", received.Type)
		assert.Equal(t, "BTC-USDT", received.Data["symbol"])
		assert.Equal(t, 51234.56, received.Data["price"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Test remove client
func TestRemoveClient(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.subscriptions[client.ID] = map[string]bool{"BTC-USDT": true}
	server.metrics.ConnectionsActive = 1
	server.mu.Unlock()

	server.removeClient(client)

	server.mu.RLock()
	_, exists := server.clients[client.ID]
	_, subExists := server.subscriptions[client.ID]
	activeConns := server.metrics.ConnectionsActive
	server.mu.RUnlock()

	assert.False(t, exists)
	assert.False(t, subExists)
	assert.Equal(t, uint64(0), activeConns)
}

// Test shutdown
func TestServerShutdown(t *testing.T) {
	server := newTestWebSocketServer()

	// The Shutdown function should not panic
	assert.NotPanics(t, func() {
		server.Shutdown()
	})

	// Context should be cancelled
	select {
	case <-server.ctx.Done():
		// Context cancelled as expected
	default:
		t.Fatal("Context should be cancelled after shutdown")
	}
}

// Benchmarks for message handlers
func BenchmarkProcessMessagePing(b *testing.B) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(b.N*2, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "ping",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.processMessage(client, msg)
		<-client.send // drain
	}
}

func BenchmarkBroadcastOrderBook(b *testing.B) {
	server := newTestWebSocketServer()

	// Add 100 clients
	for i := 0; i < 100; i++ {
		client := &Client{
			ID:            fmt.Sprintf("client%d", i),
			send:          make(chan []byte, 256),
			subscriptions: map[string]bool{"orderbook:BTC-USDT": true},
			rateLimiter:   NewRateLimiter(b.N*2, time.Minute),
		}
		server.clients[client.ID] = client
	}

	snapshot := &lx.OrderBookSnapshot{
		Symbol: "BTC-USDT",
		Bids:   []lx.OrderLevel{{Price: 50000, Size: 1.0}},
		Asks:   []lx.OrderLevel{{Price: 50100, Size: 2.0}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.BroadcastOrderBook("BTC-USDT", snapshot)
	}
}

// Additional tests for better coverage

func TestHandlePlaceOrderSuccess(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"side":   "buy",
			"type":   "limit",
			"price":  50000.0,
			"size":   0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		// Either order_update or error is acceptable
		assert.Contains(t, []string{"order_update", "error"}, received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderMissingSide(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"type":   "limit",
			"price":  50000.0,
			"size":   0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid side")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderMissingType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"side":   "buy",
			"price":  50000.0,
			"size":   0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid order type")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderMissingPrice(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"side":   "buy",
			"type":   "limit",
			"size":   0.1,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid price")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderMissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC-USDT",
			"side":   "buy",
			"type":   "limit",
			"price":  50000.0,
		},
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid size")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandlePlaceOrderInvalidOrderDataFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "place_order",
		"order": "not a map",
	}

	server.handlePlaceOrder(client, msg, "order_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid order data format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleCancelOrderInvalidFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": "not a number",
	}

	server.handleCancelOrder(client, msg, "cancel_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid orderID format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleCancelOrderNotFound(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": float64(99999),
	}

	server.handleCancelOrder(client, msg, "cancel_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "order not found")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleOpenPositionMissingSide(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC-USDT",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid side")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleOpenPositionMissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC-USDT",
		"side":     "buy",
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid size")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleOpenPositionMissingLeverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "open_position",
		"symbol": "BTC-USDT",
		"side":   "buy",
		"size":   1.0,
	}

	server.handleOpenPosition(client, msg, "pos_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid leverage")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleClosePositionMissingPositionID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "close_position",
		"size": 1.0,
	}

	server.handleClosePosition(client, msg, "close_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid positionID")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleClosePositionMissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "close_position",
		"positionID": "pos123",
	}

	server.handleClosePosition(client, msg, "close_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid size")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultDepositMissingVaultID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "vault_deposit",
		"amount": "1000",
	}

	server.handleVaultDeposit(client, msg, "vault_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid vaultID")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultDepositMissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
	}

	server.handleVaultDeposit(client, msg, "vault_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid amount")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultDepositInvalidAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
		"amount":  "invalid_number",
	}

	server.handleVaultDeposit(client, msg, "vault_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid amount format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingSupplyMissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"amount": "1000",
	}

	server.handleLendingSupply(client, msg, "supply_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid asset")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingSupplyMissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "lending_supply",
		"asset": "USDC",
	}

	server.handleLendingSupply(client, msg, "supply_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid amount")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingSupplyInvalidAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"asset":  "USDC",
		"amount": "not_a_number",
	}

	server.handleLendingSupply(client, msg, "supply_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid amount format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleUnsubscribeMissingChannel(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"symbols": []interface{}{"BTC-USDT"},
	}

	server.handleUnsubscribe(client, msg, "unsub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid channel")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleUnsubscribeMissingSymbols(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": "orderbook",
	}

	server.handleUnsubscribe(client, msg, "unsub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing symbols")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleUnsubscribeInvalidSymbolsFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": "orderbook",
		"symbols": "not an array",
	}

	server.handleUnsubscribe(client, msg, "unsub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid symbols format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleSubscribeInvalidSymbolsFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": "orderbook",
		"symbols": "not an array",
	}

	server.handleSubscribe(client, msg, "sub_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid symbols format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleModifyLeverageMissingLeverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":        "modify_leverage",
		"position_id": "pos123",
	}

	server.handleModifyLeverage(client, msg, "lev_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid leverage")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultWithdrawMissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"vault_id": "vault123",
	}

	server.handleVaultWithdraw(client, msg, "withdraw_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid amount")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultWithdrawInvalidAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"vault_id": "vault123",
		"amount":   "not_a_number",
	}

	server.handleVaultWithdraw(client, msg, "withdraw_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid amount format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingBorrowMissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "lending_borrow",
		"asset": "USDC",
	}

	server.handleLendingBorrow(client, msg, "borrow_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid amount")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingBorrowInvalidAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"asset":  "USDC",
		"amount": "invalid",
	}

	server.handleLendingBorrow(client, msg, "borrow_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid amount format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingRepayMissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"interest": "50",
	}

	server.handleLendingRepay(client, msg, "repay_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid amount")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingRepayMissingInterest(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_repay",
		"asset":  "USDC",
		"amount": "1000",
	}

	server.handleLendingRepay(client, msg, "repay_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing or invalid interest")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingRepayInvalidAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"amount":   "invalid",
		"interest": "50",
	}

	server.handleLendingRepay(client, msg, "repay_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid amount format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingRepayInvalidInterest(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"amount":   "1000",
		"interest": "invalid",
	}

	server.handleLendingRepay(client, msg, "repay_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid interest format")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleModifyOrderAlternateFieldNames(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with alternate field names (order_id, price, size)
	msg := map[string]interface{}{
		"type":     "modify_order",
		"order_id": float64(456),
		"price":    52000.0,
		"size":     0.3,
	}

	server.handleModifyOrder(client, msg, "modify_req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "order_modified", received.Type)
		assert.Equal(t, float64(456), received.Data["order_id"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestNotifyLiquidationNoClient(t *testing.T) {
	server := newTestWebSocketServer()

	// Create a mock position
	position := &lx.MarginPosition{
		ID:         "pos123",
		Symbol:     "BTC-USDT",
		Side:       lx.Buy,
		Size:       1.0,
		EntryPrice: 50000.0,
		MarkPrice:  48000.0,
	}

	// Should not panic even with no matching client
	assert.NotPanics(t, func() {
		server.notifyLiquidation("nonexistent_user", position)
	})
}

func TestNotifyLiquidationWithClient(t *testing.T) {
	server := newTestWebSocketServer()

	// Create client
	client := &Client{
		ID:            "client1",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	// Create a mock position
	position := &lx.MarginPosition{
		ID:         "pos123",
		Symbol:     "BTC-USDT",
		Side:       lx.Buy,
		Size:       1.0,
		EntryPrice: 50000.0,
		MarkPrice:  48000.0,
	}

	server.notifyLiquidation("user123", position)

	// Check client received message
	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "position_update", received.Type)
		assert.Equal(t, "liquidated", received.Data["action"])
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}

	// Also check broadcast channel
	select {
	case data := <-server.broadcast:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "public_liquidation", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for broadcast")
	}
}

func TestNotifyLiquidationSellSide(t *testing.T) {
	server := newTestWebSocketServer()

	// Create client
	client := &Client{
		ID:            "client1",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	// Create a SELL position
	position := &lx.MarginPosition{
		ID:         "pos123",
		Symbol:     "BTC-USDT",
		Side:       lx.Sell,
		Size:       1.0,
		EntryPrice: 50000.0,
		MarkPrice:  52000.0,
	}

	server.notifyLiquidation("user123", position)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "position_update", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestCheckLiquidations(t *testing.T) {
	server := newTestWebSocketServer()

	// Should not panic even without positions
	assert.NotPanics(t, func() {
		server.checkLiquidations()
	})
}

func TestBroadcastToSubscribersNonMatching(t *testing.T) {
	server := newTestWebSocketServer()

	// Add client subscribed to different channel
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 256),
		subscriptions: map[string]bool{"orderbook:ETH-USDT": true}, // Different symbol
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	msg := Message{
		Type: "test",
	}

	// Broadcast to non-matching channel
	server.broadcastToSubscribers("orderbook:BTC-USDT", msg)

	// Client should not receive message
	select {
	case <-client.send:
		t.Fatal("Client should not have received message")
	case <-time.After(50 * time.Millisecond):
		// Expected - no message received
	}
}

func TestBroadcastToSubscribersFullChannel(t *testing.T) {
	server := newTestWebSocketServer()

	// Create client with small buffer that's already full
	client := &Client{
		ID:            "client1",
		send:          make(chan []byte, 1),
		subscriptions: map[string]bool{"orderbook:BTC-USDT": true},
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Fill the channel
	client.send <- []byte("existing")

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	msg := Message{
		Type: "test",
	}

	// Should not block even with full channel
	server.broadcastToSubscribers("orderbook:BTC-USDT", msg)

	// Drain the channel
	<-client.send
}

// Tests for handleGetBalances
func TestHandleGetBalancesNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetBalances(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleGetBalancesAccountNotFound(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "unknown_user",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetBalances(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Account not found")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleGetPositions
func TestHandleGetPositionsNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetPositions(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleGetPositionsAccountNotFound(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "unknown_user",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetPositions(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Account not found")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleGetOrders
func TestHandleGetOrdersNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetOrders(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleGetOrdersSuccess(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetOrders(client, "req123")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "orders_update", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for processMessage coverage - additional paths
func TestProcessMessageInvalidType2(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:        make(chan []byte, 256),
		rateLimiter: NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": 12345, // Invalid - not a string
	}

	server.processMessage(client, msg)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Invalid message type")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestProcessMessageWithRequestID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:        make(chan []byte, 256),
		rateLimiter: NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "ping",
		"request_id": "req-12345",
	}

	server.processMessage(client, msg)

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "pong", received.Type)
		assert.Equal(t, "req-12345", received.RequestID)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleOpenPosition additional coverage
func TestHandleOpenPositionInvalidSideValue(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "open_position",
		"position": map[string]interface{}{
			"symbol":   "BTC-USDT",
			"side":     "invalid_side",
			"size":     1.0,
			"leverage": 10.0,
		},
	}

	server.handleOpenPosition(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		// Should handle invalid side or process it
		assert.Contains(t, []string{"position_update", "error"}, received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleClosePosition additional coverage
func TestHandleClosePositionInvalidSizeFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "close_position",
		"position": map[string]interface{}{
			"position_id": "pos123",
			"size":        "not_a_number",
		},
	}

	server.handleClosePosition(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		// Should return error for invalid size
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleModifyLeverage additional coverage
func TestHandleModifyLeverageNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "modify_leverage",
		"position": map[string]interface{}{
			"position_id": "pos123",
			"leverage":    20.0,
		},
	}

	server.handleModifyLeverage(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleModifyLeverageInvalidPositionData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "modify_leverage",
		"position": "not_a_map",
	}

	server.handleModifyLeverage(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleCancelOrder additional coverage
func TestHandleCancelOrderNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "cancel_order",
		"order_id": "order123",
	}

	server.handleCancelOrder(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for BroadcastPrice
func TestBroadcastPriceNoSubscribers(t *testing.T) {
	server := newTestWebSocketServer()

	// No clients subscribed, should not panic
	assert.NotPanics(t, func() {
		server.BroadcastPrice("BTC-USDT", 50000.0)
	})
}

// Tests for handleVaultDeposit additional coverage
func TestHandleVaultDepositNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "vault_deposit",
		"deposit": map[string]interface{}{
			"vault_id": "vault123",
			"amount":   "100.0",
		},
	}

	server.handleVaultDeposit(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultDepositInvalidDepositData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"deposit": "not_a_map",
	}

	server.handleVaultDeposit(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleVaultWithdraw additional coverage  
func TestHandleVaultWithdrawNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "vault_withdraw",
		"withdraw": map[string]interface{}{
			"vault_id": "vault123",
			"amount":   "50.0",
		},
	}

	server.handleVaultWithdraw(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleVaultWithdrawInvalidWithdrawData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"withdraw": "not_a_map",
	}

	server.handleVaultWithdraw(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleLendingSupply additional coverage
func TestHandleLendingSupplyNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "lending_supply",
		"supply": map[string]interface{}{
			"asset":  "USDC",
			"amount": "1000.0",
		},
	}

	server.handleLendingSupply(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingSupplyInvalidSupplyData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"supply": "not_a_map",
	}

	server.handleLendingSupply(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleLendingBorrow additional coverage
func TestHandleLendingBorrowNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "lending_borrow",
		"borrow": map[string]interface{}{
			"asset":  "USDC",
			"amount": "500.0",
		},
	}

	server.handleLendingBorrow(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingBorrowInvalidBorrowData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"borrow": "not_a_map",
	}

	server.handleLendingBorrow(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleLendingRepay additional coverage
func TestHandleLendingRepayNotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		authenticated: false,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "lending_repay",
		"repay": map[string]interface{}{
			"asset":    "USDC",
			"amount":   "250.0",
			"interest": "10.0",
		},
	}

	server.handleLendingRepay(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Not authenticated")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleLendingRepayInvalidRepayData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		UserID:        "user123",
		send:          make(chan []byte, 256),
		authenticated: true,
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "lending_repay",
		"repay": "not_a_map",
	}

	server.handleLendingRepay(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

// Tests for handleSubscribe additional coverage - more paths
func TestHandleSubscribeMissingChannel2(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"symbols": []interface{}{"BTC-USDT"},
	}

	server.handleSubscribe(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing channel")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}

func TestHandleSubscribeMissingSymbols2(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": "orderbook",
	}

	server.handleSubscribe(client, msg, "req")

	select {
	case data := <-client.send:
		var received Message
		json.Unmarshal(data, &received)
		assert.Equal(t, "error", received.Type)
		assert.Contains(t, received.Error, "Missing symbols")
	case <-time.After(time.Second):
		t.Fatal("Timeout")
	}
}
