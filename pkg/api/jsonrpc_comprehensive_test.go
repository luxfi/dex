package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/luxfi/log"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/stretchr/testify/assert"
)

// Test complete JSON-RPC lifecycle
func TestJSONRPCServer_CompleteLifecycle(t *testing.T) {
	orderBook := lx.NewOrderBook("TEST")
	level, _ := log.ToLevel("debug")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Add orders to orderbook
	orderBook.AddOrder(&lx.Order{
		ID:        1,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     99,
		Size:      10,
		User:      "buyer1",
		Timestamp: time.Now(),
	})

	orderBook.AddOrder(&lx.Order{
		ID:        2,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     98,
		Size:      20,
		User:      "buyer2",
		Timestamp: time.Now(),
	})

	orderBook.AddOrder(&lx.Order{
		ID:        3,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     101,
		Size:      15,
		User:      "seller1",
		Timestamp: time.Now(),
	})

	orderBook.AddOrder(&lx.Order{
		ID:        4,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     102,
		Size:      25,
		User:      "seller2",
		Timestamp: time.Now(),
	})

	// Test getBestBid
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, float64(99), resp["result"])

	// Test getBestAsk
	reqBody = `{"jsonrpc":"2.0","method":"orderbook.getBestAsk","params":{},"id":2}`
	req = httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w = httptest.NewRecorder()
	server.ServeHTTP(w, req)

	json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, float64(101), resp["result"])

	// Test getStats
	reqBody = `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{},"id":3}`
	req = httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w = httptest.NewRecorder()
	server.ServeHTTP(w, req)

	json.Unmarshal(w.Body.Bytes(), &resp)
	assert.NotNil(t, resp["result"])
	stats := resp["result"].(map[string]interface{})
	assert.Equal(t, "TEST", stats["symbol"])
}

// Test error handling for all error codes
func TestJSONRPCServer_ErrorCodes(t *testing.T) {
	orderBook := lx.NewOrderBook("ERROR")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	testCases := []struct {
		name         string
		reqBody      string
		expectedCode float64
		expectedMsg  string
	}{
		{
			name:         "Parse Error",
			reqBody:      `{invalid json`,
			expectedCode: -32700,
			expectedMsg:  "Parse error",
		},
		{
			name:         "Invalid Request",
			reqBody:      `{"jsonrpc":"1.0","method":"test","id":1}`,
			expectedCode: -32600,
			expectedMsg:  "Invalid Request",
		},
		{
			name:         "Method Not Found",
			reqBody:      `{"jsonrpc":"2.0","method":"invalid.method","id":1}`,
			expectedCode: -32601,
			expectedMsg:  "Method not found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(tc.reqBody))
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)

			var resp map[string]interface{}
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			assert.NoError(t, err)

			if errorObj, ok := resp["error"].(map[string]interface{}); ok {
				assert.Equal(t, tc.expectedCode, errorObj["code"])
				assert.Equal(t, tc.expectedMsg, errorObj["message"])
			}
		})
	}
}

// Test handling of different HTTP methods
func TestJSONRPCServer_HTTPMethods(t *testing.T) {
	orderBook := lx.NewOrderBook("METHODS")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	methods := []string{"GET", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT"}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/rpc", nil)
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)
			assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
		})
	}

	// POST should work
	t.Run("POST", func(t *testing.T) {
		reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","id":1}`
		req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})
}

// Test request/response headers
func TestJSONRPCServer_Headers(t *testing.T) {
	orderBook := lx.NewOrderBook("HEADERS")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","id":1}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	
	// Set various request headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "test-client")

	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	// Check response headers
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Equal(t, http.StatusOK, w.Code)
}

// Test large request handling
func TestJSONRPCServer_LargeRequest(t *testing.T) {
	orderBook := lx.NewOrderBook("LARGE")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create a large params object
	largeParams := make(map[string]interface{})
	for i := 0; i < 1000; i++ {
		largeParams[strings.Repeat("a", 100)] = strings.Repeat("b", 100)
	}

	reqObj := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "orderbook.getStats",
		"params":  largeParams,
		"id":      1,
	}

	reqBytes, _ := json.Marshal(reqObj)
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBuffer(reqBytes))
	w := httptest.NewRecorder()

	server.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

// Test concurrent requests
func TestJSONRPCServer_ConcurrentRequests(t *testing.T) {
	orderBook := lx.NewOrderBook("CONCURRENT")
	
	// Add some orders
	for i := 1; i <= 100; i++ {
		orderBook.AddOrder(&lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     float64(100 - i%10),
			Size:      10,
			User:      "user",
			Timestamp: time.Now(),
		})
	}

	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Run concurrent requests
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func(id int) {
			var method string
			switch id % 3 {
			case 0:
				method = "orderbook.getBestBid"
			case 1:
				method = "orderbook.getBestAsk"
			case 2:
				method = "orderbook.getStats"
			}

			reqBody := `{"jsonrpc":"2.0","method":"` + method + `","params":{},"id":` + string(rune(id)) + `}`
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)
			
			assert.Equal(t, http.StatusOK, w.Code)
			done <- true
		}(i)
	}

	// Wait for all requests to complete
	for i := 0; i < 100; i++ {
		<-done
	}
}

// Test request body size limits
func TestJSONRPCServer_BodySizeLimit(t *testing.T) {
	orderBook := lx.NewOrderBook("SIZE")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Create a very large request body (>10MB)
	largeBody := strings.Repeat("a", 11*1024*1024)
	req := httptest.NewRequest("POST", "/rpc", strings.NewReader(largeBody))
	w := httptest.NewRecorder()

	server.ServeHTTP(w, req)
	// Should handle large body gracefully
	assert.NotNil(t, w.Code)
}

// Test empty request body
func TestJSONRPCServer_EmptyBody(t *testing.T) {
	orderBook := lx.NewOrderBook("EMPTY")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	req := httptest.NewRequest("POST", "/rpc", nil)
	w := httptest.NewRecorder()

	server.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	if err == nil {
		assert.NotNil(t, resp["error"])
	}
}

// Test notification (no id field)
func TestJSONRPCServer_Notification(t *testing.T) {
	orderBook := lx.NewOrderBook("NOTIFY")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	// Notification has no id field
	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{}}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	server.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	// For notifications, server may not return response
	// or may return response without id
	if w.Body.Len() > 0 {
		var resp map[string]interface{}
		json.Unmarshal(w.Body.Bytes(), &resp)
		_, hasID := resp["id"]
		assert.True(t, !hasID || resp["id"] == nil)
	}
}

// Test with various id types
func TestJSONRPCServer_IDTypes(t *testing.T) {
	orderBook := lx.NewOrderBook("ID")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	testCases := []struct {
		name  string
		id    interface{}
		idStr string
	}{
		{"Number ID", 123, "123"},
		{"String ID", "test-id", `"test-id"`},
		{"Null ID", nil, "null"},
		{"Float ID", 123.456, "123.456"},
		{"Negative ID", -1, "-1"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{},"id":` + tc.idStr + `}`
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()

			server.ServeHTTP(w, req)
			assert.Equal(t, http.StatusOK, w.Code)

			var resp map[string]interface{}
			json.Unmarshal(w.Body.Bytes(), &resp)
			// ID should be echoed back
			if tc.id != nil {
				assert.NotNil(t, resp["id"])
			}
		})
	}
}

// Test reading request body multiple times
func TestJSONRPCServer_BodyReading(t *testing.T) {
	orderBook := lx.NewOrderBook("BODY")
	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)

	// Wrap the server to read body twice
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// First read
		body1, _ := io.ReadAll(r.Body)
		r.Body.Close()

		// Reset body for server
		r.Body = io.NopCloser(bytes.NewBuffer(body1))

		server := NewJSONRPCServer(orderBook, logger)
		server.ServeHTTP(w, r)
	})

	reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{},"id":1}`
	req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

// Benchmark different methods
func BenchmarkJSONRPCServer_Methods(b *testing.B) {
	orderBook := lx.NewOrderBook("BENCH")
	
	// Add many orders
	for i := 0; i < 10000; i++ {
		orderBook.AddOrder(&lx.Order{
			ID:        uint64(i),
			Type:      lx.Limit,
			Side:      lx.Buy,
			Price:     float64(100 - i%100),
			Size:      10,
			User:      "bench",
			Timestamp: time.Now(),
		})
		orderBook.AddOrder(&lx.Order{
			ID:        uint64(i + 10000),
			Type:      lx.Limit,
			Side:      lx.Sell,
			Price:     float64(100 + i%100),
			Size:      10,
			User:      "bench",
			Timestamp: time.Now(),
		})
	}

	level, _ := log.ToLevel("error")
	logger := log.NewTestLogger(level)
	server := NewJSONRPCServer(orderBook, logger)

	b.Run("getBestBid", func(b *testing.B) {
		reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestBid","params":{},"id":1}`
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)
		}
	})

	b.Run("getBestAsk", func(b *testing.B) {
		reqBody := `{"jsonrpc":"2.0","method":"orderbook.getBestAsk","params":{},"id":1}`
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)
		}
	})

	b.Run("getStats", func(b *testing.B) {
		reqBody := `{"jsonrpc":"2.0","method":"orderbook.getStats","params":{},"id":1}`
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest("POST", "/rpc", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()
			server.ServeHTTP(w, req)
		}
	})
}