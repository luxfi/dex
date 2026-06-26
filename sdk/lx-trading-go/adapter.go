// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"context"

	"github.com/shopspring/decimal"
)

// VenueCapabilities describes what a venue supports.
type VenueCapabilities struct {
	LimitOrders     bool
	MarketOrders    bool
	StopOrders      bool
	PostOnly        bool
	CancelOrders    bool
	BatchOrders     bool
	Streaming       bool
	Orderbook       bool
	Trades          bool
	AmmSwap         bool
	AddLiquidity    bool
	RemoveLiquidity bool
	LpPositions     bool
	MaxBatchSize    int
	SupportedPairs  map[string]struct{}
}

// OrderBookCapabilities returns capabilities for a DEX/orderbook venue.
func OrderBookCapabilities() VenueCapabilities {
	return VenueCapabilities{
		LimitOrders:    true,
		MarketOrders:   true,
		StopOrders:     true,
		PostOnly:       true,
		CancelOrders:   true,
		BatchOrders:    true,
		Streaming:      true,
		Orderbook:      true,
		Trades:         true,
		MaxBatchSize:   10,
		SupportedPairs: make(map[string]struct{}),
	}
}

// AMMCapabilities returns capabilities for an AMM venue.
func AMMCapabilities() VenueCapabilities {
	return VenueCapabilities{
		MarketOrders:    true,
		Streaming:       true,
		Trades:          true,
		AmmSwap:         true,
		AddLiquidity:    true,
		RemoveLiquidity: true,
		LpPositions:     true,
		MaxBatchSize:    1,
		SupportedPairs:  make(map[string]struct{}),
	}
}

// SupportsPair checks if a pair is supported.
func (c VenueCapabilities) SupportsPair(symbol string) bool {
	if len(c.SupportedPairs) == 0 {
		return true // No restrictions
	}
	_, ok := c.SupportedPairs[symbol]
	return ok
}

// VenueAdapter is the interface all venue adapters must implement.
type VenueAdapter interface {
	// Identity
	Name() string
	VenueType() VenueType
	Capabilities() VenueCapabilities
	IsConnected() bool
	LatencyMs() *int64

	// Info returns venue information.
	Info() VenueInfo

	// Connection
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error

	// Market data
	GetMarkets(ctx context.Context) ([]MarketInfo, error)
	GetTicker(ctx context.Context, symbol string) (Ticker, error)
	GetOrderbook(ctx context.Context, symbol string, depth int) (*Orderbook, error)
	GetTrades(ctx context.Context, symbol string, limit int) ([]Trade, error)

	// Account
	GetBalances(ctx context.Context) ([]Balance, error)
	GetBalance(ctx context.Context, asset string) (Balance, error)
	GetOpenOrders(ctx context.Context, symbol string) ([]Order, error)

	// Orders
	PlaceOrder(ctx context.Context, request OrderRequest) (Order, error)
	CancelOrder(ctx context.Context, orderID, symbol string) (Order, error)
	CancelAllOrders(ctx context.Context, symbol string) ([]Order, error)

	// AMM (optional - return ErrNotSupported if not implemented)
	GetSwapQuote(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool) (SwapQuote, error)
	ExecuteSwap(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, slippage decimal.Decimal) (Trade, error)
	GetPoolInfo(ctx context.Context, baseToken, quoteToken string) (PoolInfo, error)
	AddLiquidity(ctx context.Context, baseToken, quoteToken string, baseAmount, quoteAmount, slippage decimal.Decimal) (LiquidityResult, error)
	RemoveLiquidity(ctx context.Context, poolAddress string, liquidityAmount, slippage decimal.Decimal) (LiquidityResult, error)
	GetLpPositions(ctx context.Context) ([]LpPosition, error)
}

// ErrNotSupported is returned when an operation is not supported.
var ErrNotSupported = NewTradingError("NOT_SUPPORTED", "operation not supported by this venue")

// BaseAdapter provides common functionality for adapters.
type BaseAdapter struct {
	name         string
	venueType    VenueType
	capabilities VenueCapabilities
	connected    bool
	latency      *int64
	makerFee     decimal.Decimal
	takerFee     decimal.Decimal
}

// Name returns the venue name.
func (a *BaseAdapter) Name() string {
	return a.name
}

// VenueType returns the venue type.
func (a *BaseAdapter) VenueType() VenueType {
	return a.venueType
}

// Capabilities returns venue capabilities.
func (a *BaseAdapter) Capabilities() VenueCapabilities {
	return a.capabilities
}

// IsConnected returns connection status.
func (a *BaseAdapter) IsConnected() bool {
	return a.connected
}

// LatencyMs returns connection latency.
func (a *BaseAdapter) LatencyMs() *int64 {
	return a.latency
}

// SetLatency updates the latency measurement.
func (a *BaseAdapter) SetLatency(ms int64) {
	a.latency = &ms
}

// SetConnected updates connection status.
func (a *BaseAdapter) SetConnected(connected bool) {
	a.connected = connected
}

// Info returns venue information.
func (a *BaseAdapter) Info() VenueInfo {
	pairs := make([]string, 0, len(a.capabilities.SupportedPairs))
	for p := range a.capabilities.SupportedPairs {
		pairs = append(pairs, p)
	}
	return VenueInfo{
		Name:           a.name,
		VenueType:      a.venueType,
		Connected:      a.connected,
		LatencyMs:      a.latency,
		SupportedPairs: pairs,
		MakerFee:       a.makerFee,
		TakerFee:       a.takerFee,
	}
}

// Default AMM implementations that return ErrNotSupported
func (a *BaseAdapter) GetSwapQuote(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool) (SwapQuote, error) {
	return SwapQuote{}, ErrNotSupported
}

func (a *BaseAdapter) ExecuteSwap(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, slippage decimal.Decimal) (Trade, error) {
	return Trade{}, ErrNotSupported
}

func (a *BaseAdapter) GetPoolInfo(ctx context.Context, baseToken, quoteToken string) (PoolInfo, error) {
	return PoolInfo{}, ErrNotSupported
}

func (a *BaseAdapter) AddLiquidity(ctx context.Context, baseToken, quoteToken string, baseAmount, quoteAmount, slippage decimal.Decimal) (LiquidityResult, error) {
	return LiquidityResult{}, ErrNotSupported
}

func (a *BaseAdapter) RemoveLiquidity(ctx context.Context, poolAddress string, liquidityAmount, slippage decimal.Decimal) (LiquidityResult, error) {
	return LiquidityResult{}, ErrNotSupported
}

func (a *BaseAdapter) GetLpPositions(ctx context.Context) ([]LpPosition, error) {
	return nil, ErrNotSupported
}
