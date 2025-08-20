package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/luxfi/log"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONRPCServer_GetBestBid(t *testing.T) {
	// Create test orderbook
	orderBook := lx.NewOrderBook("TEST")
	orderBook.AddOrder(&lx.Order{
		ID:        1,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     100,
		Size:      10,
		User:      "test",
		Timestamp: time.Now(),
	})

	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.Equal(t, "2.0", resp["jsonrpc"])
	assert.Equal(t, float64(100), resp["result"])
	assert.Equal(t, float64(1), resp["id"])
}

func TestJSONRPCServer_GetBestAsk(t *testing.T) {
	// Create test orderbook
	orderBook := lx.NewOrderBook("TEST")
	orderBook.AddOrder(&lx.Order{
		ID:        1,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     101,
		Size:      10,
		User:      "test",
		Timestamp: time.Now(),
	})

	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestAsk","params":{},"id":2}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.Equal(t, "2.0", resp["jsonrpc"])
	assert.Equal(t, float64(101), resp["result"])
	assert.Equal(t, float64(2), resp["id"])
}

func TestJSONRPCServer_GetStats(t *testing.T) {
	// Create test orderbook with some data
	orderBook := lx.NewOrderBook("TEST")
	orderBook.AddOrder(&lx.Order{
		ID:        1,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     100,
		Size:      10,
		User:      "buyer",
		Timestamp: time.Now(),
	})
	orderBook.AddOrder(&lx.Order{
		ID:        2,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     100,
		Size:      10,
		User:      "seller",
		Timestamp: time.Now(),
	})

	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{},"id":3}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.Equal(t, "2.0", resp["jsonrpc"])
	assert.NotNil(t, resp["result"])

	result := resp["result"].(map[string]interface{})
	assert.Equal(t, "TEST", result["symbol"])
	assert.NotNil(t, result["orders"])
	assert.NotNil(t, result["trades"])
}

func TestJSONRPCServer_InvalidMethod(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request with invalid method
	reqBody := `{"jsonrpc":"2.0","method":"invalid.method","params":{},"id":4}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.Equal(t, "2.0", resp["jsonrpc"])
	assert.NotNil(t, resp["error"])
	assert.Nil(t, resp["result"])

	errorObj := resp["error"].(map[string]interface{})
	assert.Equal(t, float64(-32601), errorObj["code"])
	assert.Equal(t, "Method not found", errorObj["message"])
}

func TestJSONRPCServer_InvalidJSON(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request with invalid JSON
	reqBody := `{invalid json}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.NotNil(t, resp["error"])
	errorObj := resp["error"].(map[string]interface{})
	assert.Equal(t, float64(-32700), errorObj["code"])
	assert.Equal(t, "Parse error", errorObj["message"])
}

func TestJSONRPCServer_InvalidVersion(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create test request with wrong JSON-RPC version
	reqBody := `{"jsonrpc":"1.0","method":"orderbook.getBestBid","params":{},"id":5}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)

	assert.NotNil(t, resp["error"])
	errorObj := resp["error"].(map[string]interface{})
	assert.Equal(t, float64(-32600), errorObj["code"])
	assert.Equal(t, "Invalid Request", errorObj["message"])
}

func TestJSONRPCServer_GET_NotAllowed(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create GET request (should be rejected)
	req := httptest.NewRequest("GET", "/rpc", nil)
	w := httptest.NewRecorder()

	// Handle request
	server.ServeHTTP(w, req)

	// Check response
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func BenchmarkJSONRPCServer_GetBestBid(b *testing.B) {
	// Setup
	orderBook := lx.NewOrderBook("BENCH")
	for i := 0; i < 1000; i++ {
		orderBook.AddOrder(&lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     float64(100 - i%10),
			Size:      10,
			User:      "bench",
			Timestamp: time.Now(),
		})
	}

	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)
	
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
	}
}
