package gateway

import (
	"context"
)

// Provider is the base interface that all providers must implement
type Provider interface {
	// Info returns provider information
	Info() ProviderInfo

	// HealthCheck performs a health check
	HealthCheck(ctx context.Context) HealthCheck

	// Close cleans up provider resources
	Close() error
}

// QuoteProvider provides swap quotes
type QuoteProvider interface {
	Provider

	// GetQuote returns a swap quote for the given request
	GetQuote(ctx context.Context, req QuoteRequest) (*SwapQuote, error)

	// GetQuotes returns multiple quotes for comparison
	GetQuotes(ctx context.Context, req QuoteRequest) ([]SwapQuote, error)
}

// SwapProvider handles swap execution
type SwapProvider interface {
	QuoteProvider

	// BuildSwap builds a swap transaction from a quote
	BuildSwap(ctx context.Context, quote SwapQuote, recipient string, deadline int64) (*SwapTransaction, error)

	// ExecuteSwap executes a swap (for providers that support server-side execution)
	ExecuteSwap(ctx context.Context, req SwapRequest) (string, error)
}

// LiquidityProvider provides liquidity-related functionality
type LiquidityProvider interface {
	Provider

	// GetPools returns pools matching the request criteria
	GetPools(ctx context.Context, req PoolsRequest) ([]Pool, error)

	// GetPool returns a specific pool by address
	GetPool(ctx context.Context, chainID ChainID, address string) (*Pool, error)

	// GetPositions returns positions for an owner
	GetPositions(ctx context.Context, req PositionsRequest) ([]Position, error)

	// GetPosition returns a specific position
	GetPosition(ctx context.Context, chainID ChainID, positionID string) (*Position, error)
}

// LiquidityMutationProvider handles liquidity mutations
type LiquidityMutationProvider interface {
	LiquidityProvider

	// BuildAddLiquidity builds a transaction to add liquidity
	BuildAddLiquidity(ctx context.Context, pool Pool, amount0, amount1 string, tickLower, tickUpper int) (*SwapTransaction, error)

	// BuildRemoveLiquidity builds a transaction to remove liquidity
	BuildRemoveLiquidity(ctx context.Context, position Position, percentage float64) (*SwapTransaction, error)

	// BuildCollectFees builds a transaction to collect fees
	BuildCollectFees(ctx context.Context, position Position) (*SwapTransaction, error)
}

// PriceProvider provides token price data
type PriceProvider interface {
	Provider

	// GetTokenPrice returns the current price for a token
	GetTokenPrice(ctx context.Context, token Token) (*TokenPrice, error)

	// GetTokenPrices returns prices for multiple tokens
	GetTokenPrices(ctx context.Context, tokens []Token) ([]TokenPrice, error)
}

// ConversionProvider handles conversion tracking
type ConversionProvider interface {
	Provider

	// CreateLead creates a new conversion lead
	CreateLead(ctx context.Context, lead ConversionLead) (*ConversionLead, error)

	// GetLead retrieves a conversion lead
	GetLead(ctx context.Context, leadID string) (*ConversionLead, error)

	// TrackEvent tracks a conversion event
	TrackEvent(ctx context.Context, event ConversionEvent) error

	// GetLeadEvents gets events for a lead
	GetLeadEvents(ctx context.Context, leadID string) ([]ConversionEvent, error)
}

// TokenListProvider provides token list functionality
type TokenListProvider interface {
	Provider

	// GetTokenList returns the token list for a chain
	GetTokenList(ctx context.Context, chainID ChainID) ([]Token, error)

	// SearchTokens searches for tokens by symbol or address
	SearchTokens(ctx context.Context, chainID ChainID, query string) ([]Token, error)

	// GetToken returns a specific token
	GetToken(ctx context.Context, chainID ChainID, address string) (*Token, error)
}

// FullProvider combines all provider interfaces
type FullProvider interface {
	SwapProvider
	LiquidityMutationProvider
	PriceProvider
	ConversionProvider
	TokenListProvider
}

// ProviderRegistry manages provider registration and lookup
type ProviderRegistry interface {
	// RegisterProvider registers a provider
	RegisterProvider(provider Provider) error

	// UnregisterProvider removes a provider
	UnregisterProvider(name string) error

	// GetProvider returns a provider by name
	GetProvider(name string) (Provider, error)

	// ListProviders returns all registered providers
	ListProviders() []ProviderInfo

	// GetQuoteProviders returns all quote providers
	GetQuoteProviders() []QuoteProvider

	// GetLiquidityProviders returns all liquidity providers
	GetLiquidityProviders() []LiquidityProvider

	// GetPriceProviders returns all price providers
	GetPriceProviders() []PriceProvider

	// GetConversionProviders returns all conversion providers
	GetConversionProviders() []ConversionProvider
}

// ProviderConfig holds provider configuration
type ProviderConfig struct {
	Name     string            `json:"name"`
	Enabled  bool              `json:"enabled"`
	Priority int               `json:"priority"`
	Timeout  int               `json:"timeoutMs"`
	Options  map[string]string `json:"options"`
}

// GatewayConfig holds the gateway configuration
type GatewayConfig struct {
	Providers       []ProviderConfig `json:"providers"`
	DefaultChainID  ChainID          `json:"defaultChainId"`
	EnableFallback  bool             `json:"enableFallback"`
	CacheEnabled    bool             `json:"cacheEnabled"`
	CacheTTLSeconds int              `json:"cacheTtlSeconds"`
}
