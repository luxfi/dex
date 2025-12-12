package api

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
	"github.com/stretchr/testify/assert"
)

// --- IPRateLimiter Tests ---

func TestIPRateLimiter_NewIPRateLimiter(t *testing.T) {
	rl := NewIPRateLimiter(100, time.Second)
	assert.NotNil(t, rl)
	assert.NotNil(t, rl.requests)
	assert.Equal(t, 100, rl.limit)
	assert.Equal(t, time.Second, rl.window)
}

func TestIPRateLimiter_Allow_NewEntry(t *testing.T) {
	rl := NewIPRateLimiter(10, time.Second)

	// First request from a new IP
	assert.True(t, rl.Allow("192.168.1.1"))

	// Entry should exist
	rl.mu.RLock()
	entry, exists := rl.requests["192.168.1.1"]
	rl.mu.RUnlock()
	assert.True(t, exists)
	assert.Equal(t, 1, entry.count)
}

func TestIPRateLimiter_Allow_ExistingEntry(t *testing.T) {
	rl := NewIPRateLimiter(10, time.Second)

	// Multiple requests from same IP
	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("192.168.1.1"))
	}

	rl.mu.RLock()
	entry := rl.requests["192.168.1.1"]
	rl.mu.RUnlock()
	assert.Equal(t, 5, entry.count)
}

func TestIPRateLimiter_Allow_RateLimitExceeded(t *testing.T) {
	rl := NewIPRateLimiter(3, time.Second)

	// First 3 requests should pass
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.True(t, rl.Allow("192.168.1.1"))

	// 4th should be rate limited
	assert.False(t, rl.Allow("192.168.1.1"))
}

func TestIPRateLimiter_Allow_WindowReset(t *testing.T) {
	rl := NewIPRateLimiter(2, 50*time.Millisecond)

	// Exhaust limit
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.False(t, rl.Allow("192.168.1.1"))

	// Wait for window to pass
	time.Sleep(60 * time.Millisecond)

	// Should be allowed again
	assert.True(t, rl.Allow("192.168.1.1"))
}

func TestIPRateLimiter_Allow_MultipleIPs(t *testing.T) {
	rl := NewIPRateLimiter(2, time.Second)

	// IP1 hits limit
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.True(t, rl.Allow("192.168.1.1"))
	assert.False(t, rl.Allow("192.168.1.1"))

	// IP2 should still work
	assert.True(t, rl.Allow("192.168.1.2"))
	assert.True(t, rl.Allow("192.168.1.2"))
	assert.False(t, rl.Allow("192.168.1.2"))
}

func TestIPRateLimiter_Cleanup_EntriesRemoved(t *testing.T) {
	rl := NewIPRateLimiter(10, 30*time.Millisecond)

	// Add some requests
	rl.Allow("192.168.1.1")
	rl.Allow("192.168.1.2")

	// Wait for cleanup to run
	time.Sleep(70 * time.Millisecond)

	// Entries should be cleaned up (expired)
	rl.mu.RLock()
	_, exists1 := rl.requests["192.168.1.1"]
	_, exists2 := rl.requests["192.168.1.2"]
	rl.mu.RUnlock()

	// After cleanup window, entries should be removed
	assert.False(t, exists1, "Entry 1 should be cleaned up")
	assert.False(t, exists2, "Entry 2 should be cleaned up")
}

// --- getClientIP Tests ---

func TestGetClientIP_XForwardedFor_SingleIP(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("X-Forwarded-For", "203.0.113.195")

	ip := getClientIP(req)
	assert.Equal(t, "203.0.113.195", ip)
}

func TestGetClientIP_XForwardedFor_MultipleIPs(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("X-Forwarded-For", "203.0.113.195, 70.41.3.18, 150.172.238.178")

	ip := getClientIP(req)
	assert.Equal(t, "203.0.113.195", ip) // First IP in chain
}

func TestGetClientIP_RemoteAddr_WithPort(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.RemoteAddr = "192.168.1.100:12345"

	ip := getClientIP(req)
	assert.Equal(t, "192.168.1.100", ip)
}

func TestGetClientIP_RemoteAddr_IPv6(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.RemoteAddr = "[::1]:12345"

	ip := getClientIP(req)
	assert.Equal(t, "::1", ip)
}

func TestGetClientIP_RemoteAddr_NoPort(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.RemoteAddr = "192.168.1.100" // Invalid format but should fallback

	ip := getClientIP(req)
	assert.Equal(t, "192.168.1.100", ip)
}

func TestGetClientIP_EmptyXForwardedFor(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("X-Forwarded-For", "")
	req.RemoteAddr = "192.168.1.100:12345"

	ip := getClientIP(req)
	assert.Equal(t, "192.168.1.100", ip)
}

// --- JSONRPCServer Rate Limiting Tests ---

func TestJSONRPCServer_RateLimiting_Exceeded(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)

	// Create server with low rate limit for testing
	server := &JSONRPCServer{
		orderBook:   orderBook,
		logger:      logger,
		rateLimiter: NewIPRateLimiter(2, time.Second),
	}

	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`

	// First two requests should pass
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
		req.RemoteAddr = "192.168.1.1:12345"
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code, "Request %d should succeed", i+1)
	}

	// Third request should be rate limited
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	req.RemoteAddr = "192.168.1.1:12345"
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)
	assert.Equal(t, "1", w.Header().Get("Retry-After"))
}

func TestJSONRPCServer_RateLimiting_DifferentIPs(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)

	server := &JSONRPCServer{
		orderBook:   orderBook,
		logger:      logger,
		rateLimiter: NewIPRateLimiter(1, time.Second),
	}

	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`

	// IP1 makes request
	req1 := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	req1.RemoteAddr = "192.168.1.1:12345"
	w1 := httptest.NewRecorder()
	server.ServeHTTP(w1, req1)
	assert.Equal(t, http.StatusOK, w1.Code)

	// IP1 is now limited
	req2 := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	req2.RemoteAddr = "192.168.1.1:12345"
	w2 := httptest.NewRecorder()
	server.ServeHTTP(w2, req2)
	assert.Equal(t, http.StatusTooManyRequests, w2.Code)

	// IP2 should still work
	req3 := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	req3.RemoteAddr = "192.168.1.2:12345"
	w3 := httptest.NewRecorder()
	server.ServeHTTP(w3, req3)
	assert.Equal(t, http.StatusOK, w3.Code)
}

// --- Additional processMessage coverage tests ---

func TestProcessMessage_WithRequestID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with request_id
	msg := map[string]interface{}{
		"type":       "ping",
		"request_id": "test-request-123",
	}

	server.processMessage(client, msg)

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- Additional handleAuth coverage ---

func TestHandleAuthSuccess(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"apiKey":    "valid_key",
		"apiSecret": "valid_secret",
	}

	server.handleAuth(client, msg, "auth123")

	// Should receive auth_success followed by initial data
	timeout := time.After(2 * time.Second)
	authSuccess := false
	for !authSuccess {
		select {
		case <-client.send:
			if client.authenticated {
				authSuccess = true
			}
		case <-timeout:
			break
		}
	}

	assert.True(t, client.authenticated)
	assert.Equal(t, "user123", client.UserID)
}

// --- handlePlaceOrder edge cases ---

func TestHandlePlaceOrder_OrderDataNotMap(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "place_order",
		"order": "not-a-map",
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandlePlaceOrder_MissingSymbol(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"side":  "buy",
			"type":  "limit",
			"price": 100.0,
			"size":  10.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandlePlaceOrder_MissingSide(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"type":   "limit",
			"price":  100.0,
			"size":   10.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandlePlaceOrder_MissingType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"side":   "buy",
			"price":  100.0,
			"size":   10.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandlePlaceOrder_MissingPrice(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"side":   "buy",
			"type":   "limit",
			"size":   10.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandlePlaceOrder_MissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"side":   "buy",
			"type":   "limit",
			"price":  100.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleCancelOrder edge cases ---

func TestHandleCancelOrder_InvalidOrderIDFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": "not-a-number",
	}

	server.handleCancelOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleCancelOrder_MissingOrderID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "cancel_order",
	}

	server.handleCancelOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleOpenPosition edge cases ---

func TestHandleOpenPosition_MissingSymbol(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"side":     "buy",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleOpenPosition_MissingSide(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC/USD",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleOpenPosition_MissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC/USD",
		"side":     "buy",
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleOpenPosition_MissingLeverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "open_position",
		"symbol": "BTC/USD",
		"side":   "buy",
		"size":   1.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleClosePosition edge cases ---

func TestHandleClosePosition_MissingPositionID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "close_position",
		"size": 1.0,
	}

	server.handleClosePosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleClosePosition_MissingSize(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "close_position",
		"positionID": "pos123",
	}

	server.handleClosePosition(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleVaultDeposit edge cases ---

func TestHandleVaultDeposit_MissingVaultID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "vault_deposit",
		"amount": "1000",
	}

	server.handleVaultDeposit(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleVaultDeposit_MissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
	}

	server.handleVaultDeposit(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleVaultDeposit_InvalidAmountFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
		"amount":  "not-a-number",
	}

	server.handleVaultDeposit(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleGetBalances edge cases ---

func TestHandleGetBalances_NotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetBalances(client, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleModifyLeverage edge cases ---

func TestHandleModifyLeverage_MissingPositionID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "modify_leverage",
		"leverage": 10.0,
	}

	server.handleModifyLeverage(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleModifyLeverage_MissingLeverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":        "modify_leverage",
		"position_id": "pos123",
	}

	server.handleModifyLeverage(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleVaultWithdraw edge cases ---

func TestHandleVaultWithdraw_MissingVaultID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "vault_withdraw",
		"amount": "1000",
	}

	server.handleVaultWithdraw(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleVaultWithdraw_MissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"vault_id": "vault123",
	}

	server.handleVaultWithdraw(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleVaultWithdraw_InvalidAmountFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"vault_id": "vault123",
		"amount":   "not-a-number",
	}

	server.handleVaultWithdraw(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleVaultWithdraw_ValidRequest(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "vault_withdraw",
		"vault_id": "vault123",
		"amount":   "1000000000",
	}

	server.handleVaultWithdraw(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleLendingSupply edge cases ---

func TestHandleLendingSupply_MissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"amount": "1000",
	}

	server.handleLendingSupply(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingSupply_MissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "lending_supply",
		"asset": "USDC",
	}

	server.handleLendingSupply(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingSupply_InvalidAmountFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"asset":  "USDC",
		"amount": "invalid",
	}

	server.handleLendingSupply(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleLendingBorrow edge cases ---

func TestHandleLendingBorrow_MissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"amount": "1000",
	}

	server.handleLendingBorrow(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingBorrow_MissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":  "lending_borrow",
		"asset": "USDC",
	}

	server.handleLendingBorrow(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingBorrow_InvalidAmountFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"asset":  "USDC",
		"amount": "invalid",
	}

	server.handleLendingBorrow(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleLendingRepay edge cases ---

func TestHandleLendingRepay_MissingAsset(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"amount":   "1000",
		"interest": "10",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingRepay_MissingAmount(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"interest": "10",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingRepay_MissingInterest(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_repay",
		"asset":  "USDC",
		"amount": "1000",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingRepay_InvalidAmountFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"amount":   "invalid",
		"interest": "10",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingRepay_InvalidInterestFormat(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"amount":   "1000",
		"interest": "invalid",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleLendingRepay_ValidRequest(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "lending_repay",
		"asset":    "USDC",
		"amount":   "1000000000",
		"interest": "10000000",
	}

	server.handleLendingRepay(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleModifyOrder edge cases ---

func TestHandleModifyOrderVariants(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	testCases := []struct {
		name string
		msg  map[string]interface{}
	}{
		{
			name: "with orderID and newPrice/newSize",
			msg: map[string]interface{}{
				"type":     "modify_order",
				"orderID":  float64(123),
				"newPrice": 50000.0,
				"newSize":  2.0,
			},
		},
		{
			name: "with order_id and price/size",
			msg: map[string]interface{}{
				"type":     "modify_order",
				"order_id": float64(456),
				"price":    51000.0,
				"size":     3.0,
			},
		},
		{
			name: "with only orderID",
			msg: map[string]interface{}{
				"type":    "modify_order",
				"orderID": float64(789),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server.handleModifyOrder(client, tc.msg, "req123")

			select {
			case <-client.send:
				// Response received
			case <-time.After(time.Second):
				t.Fatal("Expected response")
			}
		})
	}
}

// --- notifyLiquidation edge cases ---

func TestNotifyLiquidation_ClientFound_Coverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Register client
	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	position := &lx.MarginPosition{
		ID:         "pos123",
		Symbol:     "BTC/USD",
		Side:       lx.Buy,
		Size:       1.0,
		EntryPrice: 50000,
		MarkPrice:  45000,
		Leverage:   10,
	}

	server.notifyLiquidation("user123", position)

	// Should receive notification
	select {
	case <-client.send:
		// Received
	case <-time.After(time.Second):
		t.Fatal("Expected notification")
	}
}

func TestNotifyLiquidation_SellSide_Coverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Register client
	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	position := &lx.MarginPosition{
		ID:         "pos123",
		Symbol:     "BTC/USD",
		Side:       lx.Sell,
		Size:       1.0,
		EntryPrice: 45000,
		MarkPrice:  50000,
		Leverage:   10,
	}

	server.notifyLiquidation("user123", position)

	// Should receive notification
	select {
	case <-client.send:
		// Received
	case <-time.After(time.Second):
		t.Fatal("Expected notification")
	}
}

func TestNotifyLiquidation_ClientNotFound_Coverage(t *testing.T) {
	server := newTestWebSocketServer()

	position := &lx.MarginPosition{
		ID:     "pos123",
		Symbol: "BTC/USD",
		Side:   lx.Buy,
		Size:   1.0,
	}

	// Should not panic when client not found
	server.notifyLiquidation("unknown_user", position)
}

// --- broadcastToSubscribers edge cases ---

func TestBroadcastToSubscribers_NoClients(t *testing.T) {
	server := newTestWebSocketServer()

	msg := Message{Type: "test", Timestamp: time.Now().Unix()}

	// Should not panic
	server.broadcastToSubscribers("test:channel", msg)
}

func TestBroadcastToSubscribers_FullChannel(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 1), // Tiny buffer
		subscriptions: map[string]bool{"test:channel": true},
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.mu.Lock()
	server.clients[client.ID] = client
	server.mu.Unlock()

	// Fill the channel
	client.send <- []byte("existing")

	msg := Message{Type: "test", Timestamp: time.Now().Unix()}

	// Should not block
	server.broadcastToSubscribers("test:channel", msg)

	// Drain
	<-client.send
}

// --- Concurrent access tests ---

func TestIPRateLimiter_ConcurrentAccess(t *testing.T) {
	rl := NewIPRateLimiter(1000, time.Second)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(ip int) {
			defer wg.Done()
			ipStr := fmt.Sprintf("192.168.1.%d", ip%256)
			for j := 0; j < 10; j++ {
				rl.Allow(ipStr)
			}
		}(i)
	}

	wg.Wait()
}

func TestServerMetrics_ConcurrentSnapshot(t *testing.T) {
	metrics := NewServerMetrics()
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			metrics.mu.Lock()
			metrics.ConnectionsTotal++
			metrics.MessagesReceived++
			metrics.mu.Unlock()
		}()
	}

	// Concurrent reads
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = metrics.GetSnapshot()
		}()
	}

	wg.Wait()
}

// --- processMessage comprehensive coverage ---

func TestProcessMessage_InvalidType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Message without type string (number instead)
	msg := map[string]interface{}{
		"type": 123, // Not a string
	}

	server.processMessage(client, msg)

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestProcessMessage_UnknownType(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "unknown_message_type",
	}

	server.processMessage(client, msg)

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestProcessMessage_GetPositions(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "get_positions",
		"request_id": "req-get-pos",
	}

	server.processMessage(client, msg)

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

func TestProcessMessage_GetOrders(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "get_orders",
		"request_id": "req-get-orders",
	}

	server.processMessage(client, msg)

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleClosePosition valid path ---

func TestHandleClosePosition_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":       "close_position",
		"positionID": "position-123",
		"size":       0.5,
	}

	server.handleClosePosition(client, msg, "req123")

	select {
	case <-client.send:
		// Response received (could be error if position not found, but still tests the path)
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleVaultDeposit valid path ---

func TestHandleVaultDeposit_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": "vault123",
		"amount":  "1000000000",
	}

	server.handleVaultDeposit(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleGetBalances valid path ---

func TestHandleGetBalances_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetBalances(client, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleModifyLeverage valid path ---

func TestHandleModifyLeverage_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":        "modify_leverage",
		"position_id": "pos123",
		"leverage":    15.0,
	}

	server.handleModifyLeverage(client, msg, "req123")

	select {
	case <-client.send:
		// Response received (could be error if position not found, but still tests the path)
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleAuth edge cases ---

func TestHandleAuth_MissingCredentials(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	testCases := []struct {
		name string
		msg  map[string]interface{}
	}{
		{
			name: "missing_api_key",
			msg: map[string]interface{}{
				"apiSecret": "secret",
			},
		},
		{
			name: "missing_api_secret",
			msg: map[string]interface{}{
				"apiKey": "key",
			},
		},
		{
			name: "empty_credentials",
			msg: map[string]interface{}{
				"apiKey":    "",
				"apiSecret": "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server.handleAuth(client, tc.msg, "auth-req")

			select {
			case <-client.send:
				// Error received
			case <-time.After(time.Second):
				t.Fatal("Expected error response")
			}
		})
	}
}

// --- handleSubscribe/handleUnsubscribe tests ---

func TestHandleSubscribe_ValidChannel_Coverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": "orderbook:BTC/USD",
	}

	server.handleSubscribe(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

func TestHandleUnsubscribe_ValidChannel_Coverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: map[string]bool{"orderbook:BTC/USD": true},
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": "orderbook:BTC/USD",
	}

	server.handleUnsubscribe(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

func TestHandleSubscribe_MissingChannel(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "subscribe",
	}

	server.handleSubscribe(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

func TestHandleUnsubscribe_MissingChannel(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "unsubscribe",
	}

	server.handleUnsubscribe(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handlePing test ---

func TestHandlePing_Response(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handlePing(client, "ping-req")

	select {
	case <-client.send:
		// Pong received
	case <-time.After(time.Second):
		t.Fatal("Expected pong response")
	}
}

// --- handleOpenPosition valid path ---

func TestHandleOpenPosition_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC/USD",
		"side":     "buy",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handlePlaceOrder valid path ---

func TestHandlePlaceOrder_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"side":   "buy",
			"type":   "limit",
			"price":  50000.0,
			"size":   1.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleCancelOrder valid path ---

func TestHandleCancelOrder_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": float64(12345),
	}

	server.handleCancelOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Response received (could be error if order not found, but still tests the path)
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleLendingSupply valid path ---

func TestHandleLendingSupply_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_supply",
		"asset":  "USDC",
		"amount": "1000000000",
	}

	server.handleLendingSupply(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleLendingBorrow valid path ---

func TestHandleLendingBorrow_ValidPath(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"asset":  "USDC",
		"amount": "500000000",
	}

	server.handleLendingBorrow(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- sendInitialData test ---

func TestSendInitialData(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.sendInitialData(client)

	select {
	case <-client.send:
		// Initial data received
	case <-time.After(time.Second):
		t.Fatal("Expected initial data")
	}
}

// --- removeClient test ---

func TestRemoveClient_Coverage(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Add client first
	server.mu.Lock()
	server.clients[client.ID] = client
	server.metrics.ConnectionsActive = 1
	server.mu.Unlock()

	// Remove client
	server.removeClient(client)

	// Verify removed
	server.mu.RLock()
	_, exists := server.clients[client.ID]
	server.mu.RUnlock()

	assert.False(t, exists)
}

// --- RateLimiter tests ---

func TestRateLimiter_Allow(t *testing.T) {
	rl := NewRateLimiter(5, time.Second)

	// First 5 should pass
	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow())
	}

	// 6th should fail
	assert.False(t, rl.Allow())
}

func TestRateLimiter_WindowReset(t *testing.T) {
	rl := NewRateLimiter(2, 50*time.Millisecond)

	assert.True(t, rl.Allow())
	assert.True(t, rl.Allow())
	assert.False(t, rl.Allow())

	// Wait for window reset
	time.Sleep(60 * time.Millisecond)

	assert.True(t, rl.Allow())
}

// --- generateClientID test ---

func TestGenerateClientID_GeneratesID(t *testing.T) {
	id := generateClientID()
	assert.NotEmpty(t, id)
	assert.Contains(t, id, "client_")
}

// --- checkLiquidations test ---

func TestCheckLiquidations_NoPositions(t *testing.T) {
	server := newTestWebSocketServer()

	// Should not panic with no positions
	server.checkLiquidations()
}

// --- handleGetPositions tests ---

func TestHandleGetPositions_NotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	server.handleGetPositions(client, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleCancelOrder additional tests ---

func TestHandleCancelOrder_WithOrderIdField(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with order_id field (alternative field name)
	msg := map[string]interface{}{
		"type":     "cancel_order",
		"order_id": float64(99999),
	}

	server.handleCancelOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleVaultDeposit additional tests ---

func TestHandleVaultDeposit_WithVaultIdField(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with vault_id field (alternative field name)
	msg := map[string]interface{}{
		"type":     "vault_deposit",
		"vault_id": "vault123",
		"amount":   "1000000000",
	}

	server.handleVaultDeposit(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleOpenPosition additional tests ---

func TestHandleOpenPosition_InvalidSide(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   "BTC/USD",
		"side":     "invalid",
		"size":     1.0,
		"leverage": 10.0,
	}

	server.handleOpenPosition(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handlePlaceOrder additional tests ---

func TestHandlePlaceOrder_NotAuthenticated(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: false,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	msg := map[string]interface{}{
		"type": "place_order",
		"order": map[string]interface{}{
			"symbol": "BTC/USD",
			"side":   "buy",
			"type":   "limit",
			"price":  50000.0,
			"size":   1.0,
		},
	}

	server.handlePlaceOrder(client, msg, "req123")

	select {
	case <-client.send:
		// Error received
	case <-time.After(time.Second):
		t.Fatal("Expected error response")
	}
}

// --- handleClosePosition additional tests ---

func TestHandleClosePosition_WithPositionIdField(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with position_id field (alternative field name)
	msg := map[string]interface{}{
		"type":        "close_position",
		"position_id": "pos123",
		"size":        0.5,
	}

	server.handleClosePosition(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}

// --- handleModifyLeverage additional tests ---

func TestHandleModifyLeverage_WithPositionID(t *testing.T) {
	server := newTestWebSocketServer()
	client := &Client{
		ID:            "test-client",
		UserID:        "user123",
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		authenticated: true,
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}

	// Test with positionID field
	msg := map[string]interface{}{
		"type":       "modify_leverage",
		"positionID": "pos456",
		"leverage":   20.0,
	}

	server.handleModifyLeverage(client, msg, "req123")

	select {
	case <-client.send:
		// Response received
	case <-time.After(time.Second):
		t.Fatal("Expected response")
	}
}
