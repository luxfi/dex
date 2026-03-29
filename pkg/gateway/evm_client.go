package gateway

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// EVMClient sends eth_call JSON-RPC requests to an EVM endpoint.
// Used by all on-chain venues (V2, V3, V4/Native) for quoting.
type EVMClient struct {
	rpcURL     string
	httpClient *http.Client
	reqID      atomic.Uint64
}

// NewEVMClient creates an EVMClient targeting the given RPC URL.
func NewEVMClient(rpcURL string) *EVMClient {
	return &EVMClient{
		rpcURL: rpcURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// CallContract performs eth_call against the given contract address with the
// provided calldata. Returns the raw response bytes (hex-decoded).
func (c *EVMClient) CallContract(ctx context.Context, to string, data []byte) ([]byte, error) {
	id := c.reqID.Add(1)

	callObj := map[string]string{
		"to":   to,
		"data": "0x" + hex.EncodeToString(data),
	}

	rpcReq := map[string]any{
		"jsonrpc": "2.0",
		"method":  "eth_call",
		"params":  []any{callObj, "latest"},
		"id":      id,
	}

	body, err := json.Marshal(rpcReq)
	if err != nil {
		return nil, fmt.Errorf("marshal rpc request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.rpcURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("rpc call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read rpc response: %w", err)
	}

	var rpcResp struct {
		Result string `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(respBody, &rpcResp); err != nil {
		return nil, fmt.Errorf("decode rpc response: %w", err)
	}
	if rpcResp.Error != nil {
		return nil, fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}

	result := strings.TrimPrefix(rpcResp.Result, "0x")
	if result == "" || result == "0x" {
		return nil, fmt.Errorf("empty rpc response")
	}

	return hex.DecodeString(result)
}
