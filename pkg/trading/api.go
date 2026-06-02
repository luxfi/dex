// Package trading implements a Uniswap-style Trading API — unified swap
// routing across on-chain pools (V2, V3, V4/Native DEX) and off-chain
// venues (any Broker provider: Alpaca, Binance, Kraken, Coinbase, etc).
//
// Endpoints:
//
//	POST /v1/trade/quote              — Best-price quote across all venues
//	POST /v1/trade/swap               — Build unsigned tx for on-chain execution
//	POST /v1/trade/order              — Submit order (off-chain routing)
//	GET  /v1/trade/swaps/:txHash      — On-chain swap status
//	GET  /v1/trade/orders/:orderId    — Order status
//	POST /v1/trade/check-approval     — Check token approval for Permit2
//	GET  /v1/trade/venues             — List available venues and their status
package trading

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
)

// LXRouterAddress is the V4 router precompile.
const LXRouterAddress = "0x0000000000000000000000000000000000009012"

// PoolManagerAddress is the V4 pool manager precompile.
const PoolManagerAddress = "0x0000000000000000000000000000000000009010"

// QuoteType distinguishes exact-input from exact-output swaps.
const (
	QuoteTypeExactInput  = "EXACT_INPUT"
	QuoteTypeExactOutput = "EXACT_OUTPUT"
)

// OrderSide indicates buy or sell.
const (
	OrderSideBuy  = "BUY"
	OrderSideSell = "SELL"
)

// OrderType indicates market or limit.
const (
	OrderTypeMarket = "MARKET"
	OrderTypeLimit  = "LIMIT"
)

// Routing strategies returned in QuoteResponse.
const (
	RoutingV4Native = "V4_NATIVE"
	RoutingBroker   = "BROKER"
	RoutingMultiHop = "MULTI_HOP"
	RoutingSplit    = "SPLIT"
)

// Order statuses.
const (
	OrderStatusPending         = "PENDING"
	OrderStatusFilled          = "FILLED"
	OrderStatusPartiallyFilled = "PARTIALLY_FILLED"
	OrderStatusCancelled       = "CANCELLED"
)

// Venue types used in Hop.Venue to dispatch calldata building.
const (
	VenueTypeV4Native  = "v4_native"
	VenueTypeUniswapV2 = "uniswap_v2"
	VenueTypeUniswapV3 = "uniswap_v3"
	VenueTypeSushiswap = "sushiswap"
	VenueTypePancakeV2 = "pancakeswap_v2"
	VenueTypePancakeV3 = "pancakeswap_v3"
	VenueTypeTraderJoe = "traderjoe"
)

// --- Request types ---

// QuoteRequest is the input for POST /v1/trade/quote.
type QuoteRequest struct {
	TokenIn        string  `json:"tokenIn"`
	TokenOut       string  `json:"tokenOut"`
	Amount         string  `json:"amount"`
	Type           string  `json:"type"`
	ChainID        int     `json:"chainId"`
	Swapper        string  `json:"swapper"`
	Slippage       float64 `json:"slippageTolerance"`
	Deadline       int64   `json:"deadline,omitempty"`
	PreferredVenue string  `json:"preferredVenue,omitempty"`
}

// SwapRequest is the input for POST /v1/trade/swap.
type SwapRequest struct {
	Quote     *Quote `json:"quote"`
	Swapper   string `json:"swapper"`
	Signature string `json:"signature,omitempty"`
}

// OrderRequest is the input for POST /v1/trade/order.
type OrderRequest struct {
	TokenIn   string  `json:"tokenIn"`
	TokenOut  string  `json:"tokenOut"`
	Amount    string  `json:"amount"`
	Type      string  `json:"type"`
	Side      string  `json:"side"`
	OrderType string  `json:"orderType"`
	Price     string  `json:"price,omitempty"`
	Swapper   string  `json:"swapper"`
	ChainID   int     `json:"chainId"`
	Slippage  float64 `json:"slippageTolerance"`
}

// ApprovalCheckRequest is the input for POST /v1/trade/check-approval.
type ApprovalCheckRequest struct {
	TokenAddress string `json:"tokenAddress"`
	Owner        string `json:"owner"`
	ChainID      int    `json:"chainId"`
}

// --- Response types ---

// QuoteResponse is returned by POST /v1/trade/quote.
type QuoteResponse struct {
	Quote       *Quote       `json:"quote"`
	Routing     string       `json:"routing"`
	Venues      []VenueQuote `json:"venues"`
	GasEstimate string       `json:"gasEstimate"`
	PriceImpact string       `json:"priceImpact"`
	PermitData  interface{}  `json:"permitData"`
}

// Quote describes the best execution path for a swap.
type Quote struct {
	AmountIn       string `json:"amountIn"`
	AmountOut      string `json:"amountOut"`
	Route          []Hop  `json:"route"`
	Fee            string `json:"fee"`
	ExecutionPrice string `json:"executionPrice"`
}

// Hop is a single step in a swap route.
type Hop struct {
	TokenIn   string `json:"tokenIn"`
	TokenOut  string `json:"tokenOut"`
	Venue     string `json:"venue"`
	Fee       int    `json:"fee"`
	PoolID    string `json:"poolId,omitempty"`
	AmountIn  string `json:"amountIn"`
	AmountOut string `json:"amountOut"`
}

// VenueQuote is a quote from a single venue, returned for transparency.
type VenueQuote struct {
	Venue       string `json:"venue"`
	AmountOut   string `json:"amountOut"`
	Fee         string `json:"fee"`
	GasEstimate string `json:"gasEstimate"`
	Executable  bool   `json:"executable"`
}

// SwapResponse is returned by POST /v1/trade/swap.
type SwapResponse struct {
	Swap TransactionRequest `json:"swap"`
}

// TransactionRequest is an unsigned EVM transaction.
type TransactionRequest struct {
	To       string `json:"to"`
	From     string `json:"from"`
	Data     string `json:"data"`
	Value    string `json:"value"`
	ChainID  int    `json:"chainId"`
	GasLimit string `json:"gasLimit"`
}

// OrderResponse is returned by POST /v1/trade/order.
type OrderResponse struct {
	OrderID   string `json:"orderId"`
	Status    string `json:"status"`
	CreatedAt string `json:"createdAt"`
	Route     string `json:"route"`
}

// SwapStatusResponse is returned by GET /v1/trade/swaps/:txHash.
type SwapStatusResponse struct {
	TxHash      string `json:"txHash"`
	Status      string `json:"status"`
	BlockNumber string `json:"blockNumber,omitempty"`
	GasUsed     string `json:"gasUsed,omitempty"`
}

// OrderStatusResponse is returned by GET /v1/trade/orders/:orderId.
type OrderStatusResponse struct {
	OrderID   string `json:"orderId"`
	Status    string `json:"status"`
	FilledQty string `json:"filledQty"`
	AvgPrice  string `json:"avgPrice"`
	CreatedAt string `json:"createdAt"`
	UpdatedAt string `json:"updatedAt"`
	Venue     string `json:"venue"`
}

// ApprovalCheckResponse is returned by POST /v1/trade/check-approval.
type ApprovalCheckResponse struct {
	Approved      bool   `json:"approved"`
	Allowance     string `json:"allowance"`
	NeedsApproval bool   `json:"needsApproval"`
}

// VenueInfo describes an available liquidity venue.
type VenueInfo struct {
	Name       string `json:"name"`
	Status     string `json:"status"`
	Executable bool   `json:"executable"`
	Type       string `json:"type"`
}

// --- Config ---

// Config configures the Trading API for a specific chain deployment.
type Config struct {
	// ChainIDs is the set of accepted chain IDs for this deployment.
	ChainIDs []int

	// DefaultChainID is used when building swap transactions.
	DefaultChainID int

	// VenueRouters maps venue names to their on-chain router addresses.
	// Used by the swap handler to build calldata for the correct router.
	// Example: {"uniswap_v2": "0x7a250d...", "uniswap_v3": "0xE592427A..."}
	// V4 native routes always go to LXRouterAddress (0x9012).
	VenueRouters map[string]string
}

// --- API ---

// TradingAPI holds the handler state: registered venues and an in-memory order store.
type TradingAPI struct {
	venues           []Venue
	mu               sync.RWMutex
	orders           map[string]*OrderStatusResponse
	validChainIDs    map[int]bool
	defaultChain     int
	venueRouters     map[string]string
	crossChainRouter *CrossChainRouter
	arbScanner       *ArbScanner
}

// New creates a TradingAPI with the given config and venues.
func New(cfg Config, venues ...Venue) *TradingAPI {
	valid := make(map[int]bool, len(cfg.ChainIDs))
	for _, id := range cfg.ChainIDs {
		valid[id] = true
	}
	routers := cfg.VenueRouters
	if routers == nil {
		routers = make(map[string]string)
	}
	return &TradingAPI{
		venues:        venues,
		orders:        make(map[string]*OrderStatusResponse),
		validChainIDs: valid,
		defaultChain:  cfg.DefaultChainID,
		venueRouters:  routers,
	}
}

// RegisterRoutes mounts all trading endpoints on the given mux under /v1/trade/*.
// Requires Go 1.22+ ServeMux (method-based routing).
func (api *TradingAPI) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/trade/quote", api.handleQuote)
	mux.HandleFunc("POST /v1/trade/swap", api.handleSwap)
	mux.HandleFunc("POST /v1/trade/order", api.handleOrder)
	mux.HandleFunc("GET /v1/trade/swaps/{txHash}", api.handleSwapStatus)
	mux.HandleFunc("GET /v1/trade/orders/{orderId}", api.handleOrderStatus)
	mux.HandleFunc("POST /v1/trade/check-approval", api.handleCheckApproval)
	mux.HandleFunc("GET /v1/trade/venues", api.handleVenues)
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// Error ignored: response is already partially written at this point.
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func isValidAddress(s string) bool {
	s = strings.TrimPrefix(s, "0x")
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

func isPositiveDecimal(s string) bool {
	if s == "" || s == "0" {
		return false
	}
	// Cap at 78 digits (uint256 max is 78 decimal digits: 2^256-1).
	// Longer strings are either invalid or would cause computational overhead
	// in big.Int arithmetic with no valid EVM use case.
	if len(s) > 78 {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
