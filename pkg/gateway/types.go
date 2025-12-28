// Package gateway provides an API gateway layer with pluggable providers.
// It supports delegation to external APIs (like Uniswap) and native Lux providers.
package gateway

import (
	"context"
	"math/big"
	"time"
)

// ChainID represents a blockchain chain ID
type ChainID uint64

// Common chain IDs
const (
	ChainIDEthereum ChainID = 1
	ChainIDArbitrum ChainID = 42161
	ChainIDOptimism ChainID = 10
	ChainIDPolygon  ChainID = 137
	ChainIDBase     ChainID = 8453
	ChainIDBNB      ChainID = 56
	ChainIDLux      ChainID = 96369
	ChainIDZoo      ChainID = 200200
)

// Token represents a token on a specific chain
type Token struct {
	Address  string  `json:"address"`
	ChainID  ChainID `json:"chainId"`
	Decimals int     `json:"decimals"`
	Symbol   string  `json:"symbol"`
	Name     string  `json:"name"`
	LogoURI  string  `json:"logoURI,omitempty"`
}

// TokenAmount represents a token with an amount
type TokenAmount struct {
	Token  Token    `json:"token"`
	Amount *big.Int `json:"amount"`
}

// SwapQuote represents a quote for a swap
type SwapQuote struct {
	TokenIn      TokenAmount `json:"tokenIn"`
	TokenOut     TokenAmount `json:"tokenOut"`
	Route        []PoolHop   `json:"route"`
	PriceImpact  float64     `json:"priceImpact"`
	GasEstimate  *big.Int    `json:"gasEstimate"`
	QuoteID      string      `json:"quoteId,omitempty"`
	ExpiresAt    time.Time   `json:"expiresAt,omitempty"`
	ProviderName string      `json:"providerName"`
}

// PoolHop represents a single hop in a swap route
type PoolHop struct {
	PoolAddress string `json:"poolAddress"`
	PoolType    string `json:"poolType"` // "v2", "v3", "v4"
	TokenIn     Token  `json:"tokenIn"`
	TokenOut    Token  `json:"tokenOut"`
	Fee         int    `json:"fee,omitempty"` // For V3/V4 pools
}

// SwapRequest represents a request to execute a swap
type SwapRequest struct {
	TokenIn         Token    `json:"tokenIn"`
	TokenOut        Token    `json:"tokenOut"`
	Amount          *big.Int `json:"amount"`
	IsExactIn       bool     `json:"isExactIn"`
	Slippage        float64  `json:"slippage"`
	Recipient       string   `json:"recipient"`
	Deadline        int64    `json:"deadline,omitempty"`
	PermitSignature string   `json:"permitSignature,omitempty"`
}

// SwapTransaction represents the transaction data for a swap
type SwapTransaction struct {
	To       string   `json:"to"`
	Data     string   `json:"data"`
	Value    *big.Int `json:"value"`
	GasLimit *big.Int `json:"gasLimit"`
	ChainID  ChainID  `json:"chainId"`
}

// Pool represents a liquidity pool
type Pool struct {
	Address    string      `json:"address"`
	ChainID    ChainID     `json:"chainId"`
	Protocol   string      `json:"protocol"` // "uniswap_v2", "uniswap_v3", "lux_amm"
	Token0     Token       `json:"token0"`
	Token1     Token       `json:"token1"`
	Fee        int         `json:"fee,omitempty"`
	TVL        *big.Int    `json:"tvl,omitempty"`
	Volume24h  *big.Int    `json:"volume24h,omitempty"`
	APR        float64     `json:"apr,omitempty"`
	TickData   []TickRange `json:"tickData,omitempty"` // For V3/V4 pools
	Reserves   *Reserves   `json:"reserves,omitempty"` // For V2 pools
}

// Reserves represents V2 pool reserves
type Reserves struct {
	Reserve0 *big.Int `json:"reserve0"`
	Reserve1 *big.Int `json:"reserve1"`
}

// TickRange represents a tick range with liquidity (V3/V4)
type TickRange struct {
	TickLower int      `json:"tickLower"`
	TickUpper int      `json:"tickUpper"`
	Liquidity *big.Int `json:"liquidity"`
}

// Position represents a liquidity position
type Position struct {
	ID          string      `json:"id"`
	Owner       string      `json:"owner"`
	Pool        Pool        `json:"pool"`
	Liquidity   *big.Int    `json:"liquidity"`
	Token0Owed  *big.Int    `json:"token0Owed"`
	Token1Owed  *big.Int    `json:"token1Owed"`
	TickLower   int         `json:"tickLower,omitempty"`
	TickUpper   int         `json:"tickUpper,omitempty"`
	FeesEarned  *FeesEarned `json:"feesEarned,omitempty"`
}

// FeesEarned represents fees earned from a position
type FeesEarned struct {
	Token0Fees *big.Int `json:"token0Fees"`
	Token1Fees *big.Int `json:"token1Fees"`
}

// ConversionLead represents a conversion tracking lead
type ConversionLead struct {
	ID           string            `json:"id"`
	Source       string            `json:"source"`
	Medium       string            `json:"medium,omitempty"`
	Campaign     string            `json:"campaign,omitempty"`
	WalletAddr   string            `json:"walletAddr,omitempty"`
	Timestamp    time.Time         `json:"timestamp"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// ConversionEvent represents a conversion event
type ConversionEvent struct {
	LeadID    string            `json:"leadId"`
	EventType string            `json:"eventType"` // "swap", "lp_add", "wallet_connect"
	ChainID   ChainID           `json:"chainId"`
	TxHash    string            `json:"txHash,omitempty"`
	Value     *big.Int          `json:"value,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// TokenPrice represents token price data
type TokenPrice struct {
	Token       Token     `json:"token"`
	PriceUSD    float64   `json:"priceUSD"`
	PriceChange float64   `json:"priceChange24h,omitempty"`
	Volume24h   *big.Int  `json:"volume24h,omitempty"`
	MarketCap   *big.Int  `json:"marketCap,omitempty"`
	UpdatedAt   time.Time `json:"updatedAt"`
}

// ProviderInfo contains information about a provider
type ProviderInfo struct {
	Name        string   `json:"name"`
	Version     string   `json:"version"`
	Description string   `json:"description"`
	SupportedChains []ChainID `json:"supportedChains"`
	Priority    int      `json:"priority"` // Lower = higher priority
	Healthy     bool     `json:"healthy"`
}

// HealthCheck represents health check result
type HealthCheck struct {
	Provider  string    `json:"provider"`
	Healthy   bool      `json:"healthy"`
	Latency   int64     `json:"latencyMs"`
	LastCheck time.Time `json:"lastCheck"`
	Error     string    `json:"error,omitempty"`
}

// QuoteRequest for getting swap quotes
type QuoteRequest struct {
	TokenIn   Token    `json:"tokenIn"`
	TokenOut  Token    `json:"tokenOut"`
	Amount    *big.Int `json:"amount"`
	IsExactIn bool     `json:"isExactIn"`
	ChainID   ChainID  `json:"chainId"`
	Slippage  float64  `json:"slippage,omitempty"`
}

// PoolsRequest for querying pools
type PoolsRequest struct {
	ChainID   ChainID `json:"chainId"`
	Token0    string  `json:"token0,omitempty"`
	Token1    string  `json:"token1,omitempty"`
	Protocol  string  `json:"protocol,omitempty"`
	MinTVL    *big.Int `json:"minTvl,omitempty"`
	Limit     int     `json:"limit,omitempty"`
	Offset    int     `json:"offset,omitempty"`
}

// PositionsRequest for querying positions
type PositionsRequest struct {
	ChainID ChainID `json:"chainId"`
	Owner   string  `json:"owner"`
	PoolID  string  `json:"poolId,omitempty"`
}

// Error types
type ProviderError struct {
	Provider string
	Err      error
}

func (e *ProviderError) Error() string {
	return e.Provider + ": " + e.Err.Error()
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// Context key type
type contextKey string

const (
	ContextKeyRequestID contextKey = "requestID"
	ContextKeyChainID   contextKey = "chainID"
)

// WithRequestID adds a request ID to the context
func WithRequestID(ctx context.Context, requestID string) context.Context {
	return context.WithValue(ctx, ContextKeyRequestID, requestID)
}

// GetRequestID gets the request ID from context
func GetRequestID(ctx context.Context) string {
	if v := ctx.Value(ContextKeyRequestID); v != nil {
		return v.(string)
	}
	return ""
}
