// Package uniswap provides a Uniswap API provider implementation.
// This delegates to Uniswap's backend APIs as a fallback provider
// until native Lux infrastructure is deployed.
package uniswap

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// APIEndpoints contains the Uniswap API endpoints
type APIEndpoints struct {
	// Core API for quotes, tokens, and general data
	CoreAPI string
	
	// Liquidity backend for pool and position data
	LiquidityAPI string
	
	// Conversion tracking/entry gateway
	ConversionAPI string
}

// DefaultEndpoints returns the default Uniswap production endpoints
func DefaultEndpoints() APIEndpoints {
	return APIEndpoints{
		CoreAPI:       "https://api.uniswap.org",
		LiquidityAPI:  "https://liquidity.backend-prod.api.uniswap.org",
		ConversionAPI: "https://entry-gateway.backend-prod.api.uniswap.org",
	}
}

// Client is an HTTP client for Uniswap APIs
type Client struct {
	endpoints  APIEndpoints
	httpClient *http.Client
	apiKey     string
	userAgent  string
}

// ClientConfig holds client configuration
type ClientConfig struct {
	Endpoints  APIEndpoints
	APIKey     string
	Timeout    time.Duration
	UserAgent  string
}

// NewClient creates a new Uniswap API client
func NewClient(cfg ClientConfig) *Client {
	if cfg.Endpoints.CoreAPI == "" {
		cfg.Endpoints = DefaultEndpoints()
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.UserAgent == "" {
		cfg.UserAgent = "Lux-DEX-Gateway/1.0"
	}
	
	return &Client{
		endpoints: cfg.Endpoints,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
		apiKey:    cfg.APIKey,
		userAgent: cfg.UserAgent,
	}
}

// request makes an HTTP request to a Uniswap API
func (c *Client) request(ctx context.Context, method, url string, body interface{}, result interface{}) error {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}
	
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}
	
	if resp.StatusCode >= 400 {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(respBody),
		}
	}
	
	if result != nil {
		if err := json.Unmarshal(respBody, result); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}
	
	return nil
}

// APIError represents an API error response
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("uniswap API error (status %d): %s", e.StatusCode, e.Message)
}

// HealthCheck performs a health check on the APIs
func (c *Client) HealthCheck(ctx context.Context) error {
	// Simple health check - just verify the core API is reachable
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoints.CoreAPI+"/health", nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", c.userAgent)
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode >= 400 {
		return fmt.Errorf("health check failed with status %d", resp.StatusCode)
	}
	
	return nil
}

// Core API methods

// QuoteRequest for the Uniswap API
type QuoteAPIRequest struct {
	TokenIn          string `json:"tokenIn"`
	TokenInChainID   uint64 `json:"tokenInChainId"`
	TokenOut         string `json:"tokenOut"`
	TokenOutChainID  uint64 `json:"tokenOutChainId"`
	Amount           string `json:"amount"`
	Type             string `json:"type"` // "EXACT_INPUT" or "EXACT_OUTPUT"
	Slippage         int    `json:"slippageTolerance,omitempty"` // in bps
	Protocols        string `json:"protocols,omitempty"`
	Recipient        string `json:"recipient,omitempty"`
}

// QuoteAPIResponse from the Uniswap API
type QuoteAPIResponse struct {
	Quote            string          `json:"quote"`
	QuoteGasAdjusted string          `json:"quoteGasAdjusted"`
	GasUseEstimate   string          `json:"gasUseEstimate"`
	GasUseEstimateQuote string       `json:"gasUseEstimateQuote"`
	Route            []RouteAPIData  `json:"route"`
	PriceImpact      string          `json:"priceImpact,omitempty"`
	MethodParameters *MethodParams   `json:"methodParameters,omitempty"`
}

// RouteAPIData represents a route from the API
type RouteAPIData struct {
	Type       string           `json:"type"`
	Address    string           `json:"address"`
	TokenIn    TokenAPIData     `json:"tokenIn"`
	TokenOut   TokenAPIData     `json:"tokenOut"`
	Fee        string           `json:"fee,omitempty"`
	AmountIn   string           `json:"amountIn,omitempty"`
	AmountOut  string           `json:"amountOut,omitempty"`
}

// TokenAPIData represents token data from the API
type TokenAPIData struct {
	ChainID  uint64 `json:"chainId"`
	Decimals int    `json:"decimals"`
	Address  string `json:"address"`
	Symbol   string `json:"symbol"`
	Name     string `json:"name,omitempty"`
}

// MethodParams contains transaction method parameters
type MethodParams struct {
	Calldata string `json:"calldata"`
	Value    string `json:"value"`
	To       string `json:"to"`
}

// GetQuote gets a swap quote from the Core API
func (c *Client) GetQuote(ctx context.Context, req QuoteAPIRequest) (*QuoteAPIResponse, error) {
	url := fmt.Sprintf("%s/v2/quote", c.endpoints.CoreAPI)
	
	var resp QuoteAPIResponse
	if err := c.request(ctx, "POST", url, req, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// Liquidity API methods

// PoolsAPIRequest for querying pools
type PoolsAPIRequest struct {
	ChainID     uint64  `json:"chainId"`
	TokenA      string  `json:"tokenA,omitempty"`
	TokenB      string  `json:"tokenB,omitempty"`
	PoolType    string  `json:"poolType,omitempty"`
	MinTVL      float64 `json:"minTvl,omitempty"`
	Limit       int     `json:"limit,omitempty"`
	Offset      int     `json:"offset,omitempty"`
}

// PoolAPIData represents pool data from the API
type PoolAPIData struct {
	ID          string       `json:"id"`
	Address     string       `json:"address"`
	ChainID     uint64       `json:"chainId"`
	Protocol    string       `json:"protocol"`
	Token0      TokenAPIData `json:"token0"`
	Token1      TokenAPIData `json:"token1"`
	FeeTier     int          `json:"feeTier,omitempty"`
	TVL         string       `json:"tvl"`
	Volume24h   string       `json:"volume24h"`
	APR         float64      `json:"apr,omitempty"`
}

// GetPools queries pools from the Liquidity API
func (c *Client) GetPools(ctx context.Context, req PoolsAPIRequest) ([]PoolAPIData, error) {
	url := fmt.Sprintf("%s/v1/pools", c.endpoints.LiquidityAPI)
	
	var resp struct {
		Pools []PoolAPIData `json:"pools"`
	}
	if err := c.request(ctx, "POST", url, req, &resp); err != nil {
		return nil, err
	}
	
	return resp.Pools, nil
}

// PositionsAPIRequest for querying positions
type PositionsAPIRequest struct {
	ChainID uint64 `json:"chainId"`
	Owner   string `json:"owner"`
	PoolID  string `json:"poolId,omitempty"`
}

// PositionAPIData represents position data from the API
type PositionAPIData struct {
	ID         string       `json:"id"`
	Owner      string       `json:"owner"`
	Pool       PoolAPIData  `json:"pool"`
	Liquidity  string       `json:"liquidity"`
	Token0Owed string       `json:"token0Owed"`
	Token1Owed string       `json:"token1Owed"`
	TickLower  int          `json:"tickLower,omitempty"`
	TickUpper  int          `json:"tickUpper,omitempty"`
	Fees0      string       `json:"fees0,omitempty"`
	Fees1      string       `json:"fees1,omitempty"`
}

// GetPositions queries positions from the Liquidity API
func (c *Client) GetPositions(ctx context.Context, req PositionsAPIRequest) ([]PositionAPIData, error) {
	url := fmt.Sprintf("%s/v1/positions", c.endpoints.LiquidityAPI)
	
	var resp struct {
		Positions []PositionAPIData `json:"positions"`
	}
	if err := c.request(ctx, "POST", url, req, &resp); err != nil {
		return nil, err
	}
	
	return resp.Positions, nil
}

// Conversion API methods

// LeadAPIData represents a conversion lead
type LeadAPIData struct {
	ID         string            `json:"id"`
	Source     string            `json:"source"`
	Medium     string            `json:"medium,omitempty"`
	Campaign   string            `json:"campaign,omitempty"`
	WalletAddr string            `json:"walletAddr,omitempty"`
	CreatedAt  string            `json:"createdAt"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// CreateLead creates a conversion lead
func (c *Client) CreateLead(ctx context.Context, lead LeadAPIData) (*LeadAPIData, error) {
	url := fmt.Sprintf("%s/v1/leads", c.endpoints.ConversionAPI)
	
	var resp LeadAPIData
	if err := c.request(ctx, "POST", url, lead, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// EventAPIData represents a conversion event
type EventAPIData struct {
	LeadID    string            `json:"leadId"`
	EventType string            `json:"eventType"`
	ChainID   uint64            `json:"chainId"`
	TxHash    string            `json:"txHash,omitempty"`
	Value     string            `json:"value,omitempty"`
	Timestamp string            `json:"timestamp"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// TrackEvent tracks a conversion event
func (c *Client) TrackEvent(ctx context.Context, event EventAPIData) error {
	url := fmt.Sprintf("%s/v1/events", c.endpoints.ConversionAPI)
	return c.request(ctx, "POST", url, event, nil)
}

// GetLeadEvents gets events for a lead
func (c *Client) GetLeadEvents(ctx context.Context, leadID string) ([]EventAPIData, error) {
	url := fmt.Sprintf("%s/v1/leads/%s/events", c.endpoints.ConversionAPI, leadID)
	
	var resp struct {
		Events []EventAPIData `json:"events"`
	}
	if err := c.request(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	
	return resp.Events, nil
}

// Token list methods

// GetTokenList gets the token list for a chain
func (c *Client) GetTokenList(ctx context.Context, chainID uint64) ([]TokenAPIData, error) {
	url := fmt.Sprintf("%s/v1/tokens?chainId=%d", c.endpoints.CoreAPI, chainID)
	
	var resp struct {
		Tokens []TokenAPIData `json:"tokens"`
	}
	if err := c.request(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	
	return resp.Tokens, nil
}

// SearchTokens searches for tokens
func (c *Client) SearchTokens(ctx context.Context, chainID uint64, query string) ([]TokenAPIData, error) {
	url := fmt.Sprintf("%s/v1/tokens/search?chainId=%d&query=%s", c.endpoints.CoreAPI, chainID, query)
	
	var resp struct {
		Tokens []TokenAPIData `json:"tokens"`
	}
	if err := c.request(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	
	return resp.Tokens, nil
}

// GetTokenPrice gets token price
func (c *Client) GetTokenPrice(ctx context.Context, chainID uint64, address string) (*TokenPriceAPIData, error) {
	url := fmt.Sprintf("%s/v1/price?chainId=%d&address=%s", c.endpoints.CoreAPI, chainID, address)
	
	var resp TokenPriceAPIData
	if err := c.request(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// TokenPriceAPIData represents token price from API
type TokenPriceAPIData struct {
	ChainID        uint64  `json:"chainId"`
	Address        string  `json:"address"`
	PriceUSD       float64 `json:"priceUSD"`
	PriceChange24h float64 `json:"priceChange24h,omitempty"`
	Volume24h      string  `json:"volume24h,omitempty"`
	MarketCap      string  `json:"marketCap,omitempty"`
}
