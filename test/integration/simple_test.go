//go:build integration
// +build integration

package integration

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSimpleJSONRPC tests basic JSON-RPC functionality
func TestSimpleJSONRPC(t *testing.T) {
	// Test placing an order via JSON-RPC
	reqBody := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "lx_placeOrder",
		"params": map[string]interface{}{
			"symbol": "BTC-USD",
			"type":   0, // Limit
			"side":   0, // Buy
			"price":  50000.0,
			"size":   0.1,
			"userID": "test-user",
		},
		"id": 1,
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", testJSONRPCURL+"/rpc", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Skip("Server not running, skipping integration test")
		return
	}
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&result)
	require.NoError(t, err)

	// Check for successful response
	if errObj, hasError := result["error"]; hasError {
		t.Logf("JSON-RPC error: %v", errObj)
	} else {
		assert.Contains(t, result, "result")
	}
}

// TestHealthEndpoint tests the health check endpoint
func TestHealthEndpoint(t *testing.T) {
	resp, err := http.Get(testJSONRPCURL + "/health")
	if err != nil {
		t.Skip("Server not running, skipping integration test")
		return
	}
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var health map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&health)
	require.NoError(t, err)

	assert.Equal(t, "healthy", health["status"])
}
