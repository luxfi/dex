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

// ask sends one JSON-RPC method and returns its result string.
//
// Every question this client asks a chain is one round trip with a hex string
// coming back, so the transport is written once and the methods sit on top.
func (c *EVMClient) ask(ctx context.Context, method string, params []any) (string, error) {
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  params,
		"id":      c.reqID.Add(1),
	})
	if err != nil {
		return "", fmt.Errorf("marshal rpc request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.rpcURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("rpc call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read rpc response: %w", err)
	}

	var rpcResp struct {
		Result string `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(respBody, &rpcResp); err != nil {
		return "", fmt.Errorf("decode rpc response: %w", err)
	}
	if rpcResp.Error != nil {
		return "", fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	return strings.TrimPrefix(rpcResp.Result, "0x"), nil
}

// CallContract performs eth_call against the given contract address with the
// provided calldata. Returns the raw response bytes (hex-decoded).
func (c *EVMClient) CallContract(ctx context.Context, to string, data []byte) ([]byte, error) {
	result, err := c.ask(ctx, "eth_call", []any{
		map[string]string{"to": to, "data": "0x" + hex.EncodeToString(data)},
		"latest",
	})
	if err != nil {
		return nil, err
	}
	if result == "" {
		// eth_call answers 0x two ways: the contract reverted, or there is no
		// contract at that address at all. "empty rpc response" named the
		// transport and left a reader looking at their own endpoint.
		return nil, fmt.Errorf("%s answered nothing: it reverted, or nothing is deployed there", to)
	}
	return hex.DecodeString(result)
}

// Code returns the bytecode at an address, empty where nothing is deployed.
func (c *EVMClient) Code(ctx context.Context, addr string) ([]byte, error) {
	result, err := c.ask(ctx, "eth_getCode", []any{addr, "latest"})
	if err != nil {
		return nil, err
	}
	return hex.DecodeString(result)
}
