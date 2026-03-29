package trading

import (
	"context"
	"time"
)

// Supported chain IDs.
const (
	ChainEthereum = 1
	ChainOptimism = 10
	ChainBSC      = 56
	ChainPolygon  = 137
	ChainBase     = 8453
	ChainArbitrum = 42161
	ChainLux      = 96369
	ChainZoo      = 200200
)

// BridgeType identifies the bridging mechanism.
type BridgeType string

const (
	BridgeWarp     BridgeType = "warp"       // Lux AWM native cross-subnet
	BridgeTeleport BridgeType = "teleport"   // EIP-712 bridge with MPC oracles
	BridgeCCTP     BridgeType = "cctp"       // Circle cross-chain USDC
	BridgeLux      BridgeType = "lux_bridge" // Lux native bridge precompiles
)

// CrossChainHop step types.
const (
	HopTypeSwap   = "swap"
	HopTypeBridge = "bridge"
)

// Chain represents a supported blockchain with its local venues.
type Chain struct {
	ChainID     int
	Name        string
	RPCURL      string
	Venues      []Venue
	NativeToken string // wrapped native (e.g. WETH, WLUX)
}

// CrossChainHop is a single step in a cross-chain route.
type CrossChainHop struct {
	Type        string     `json:"type"` // "swap" or "bridge"
	ChainID     int        `json:"chainId"`
	TokenIn     string     `json:"tokenIn"`
	TokenOut    string     `json:"tokenOut"`
	Venue       string     `json:"venue,omitempty"`
	Bridge      BridgeType `json:"bridge,omitempty"`
	DestChainID int        `json:"destChainId,omitempty"`
	AmountIn    string     `json:"amountIn"`
	AmountOut   string     `json:"amountOut"`
	Fee         string     `json:"fee"`
	GasEstimate string     `json:"gasEstimate"`
}

// CrossChainRoute is a complete path across one or more chains.
type CrossChainRoute struct {
	Hops             []CrossChainHop `json:"hops"`
	TotalAmountIn    string          `json:"totalAmountIn"`
	TotalAmountOut   string          `json:"totalAmountOut"`
	TotalGasEstimate string          `json:"totalGasEstimate"`
	TotalBridgeFee   string          `json:"totalBridgeFee"`
	EstimatedTime    time.Duration   `json:"estimatedTimeMs"`
	Chains           []int           `json:"chains"`
}

// CrossChainQuoteRequest is the input for POST /v1/trade/cross-chain/quote.
type CrossChainQuoteRequest struct {
	TokenIn      string     `json:"tokenIn"`
	TokenOut     string     `json:"tokenOut"`
	Amount       string     `json:"amount"`
	Type         string     `json:"type"`
	SourceChain  int        `json:"sourceChain"`
	DestChain    int        `json:"destChain"`
	Swapper      string     `json:"swapper"`
	Slippage     float64    `json:"slippageTolerance"`
	MaxHops      int        `json:"maxHops,omitempty"`
	PreferBridge BridgeType `json:"preferBridge,omitempty"`
}

// CrossChainQuoteResponse is returned by POST /v1/trade/cross-chain/quote.
type CrossChainQuoteResponse struct {
	Routes    []CrossChainRoute `json:"routes"`
	BestRoute *CrossChainRoute  `json:"bestRoute"`
	Arbitrage []ArbOpportunity  `json:"arbitrage,omitempty"`
}

// BridgeQuoteRequest is input for bridge venue quoting.
type BridgeQuoteRequest struct {
	Token       string `json:"token"`
	Amount      string `json:"amount"`
	SourceChain int    `json:"sourceChain"`
	DestChain   int    `json:"destChain"`
	Recipient   string `json:"recipient"`
}

// BridgeQuote is a price quote from a bridge venue.
type BridgeQuote struct {
	Bridge      BridgeType    `json:"bridge"`
	AmountOut   string        `json:"amountOut"`
	Fee         string        `json:"fee"`
	GasEstimate string        `json:"gasEstimate"`
	Duration    time.Duration `json:"durationMs"`
}

// BridgeInfo describes an available bridge for GET /v1/trade/bridges.
type BridgeInfo struct {
	Type        BridgeType `json:"type"`
	SourceChain int        `json:"sourceChain"`
	DestChain   int        `json:"destChain"`
	Status      string     `json:"status"`
	Tokens      []string   `json:"tokens"`
}

// BridgeVenue bridges tokens between chains.
type BridgeVenue interface {
	BridgeType() BridgeType
	SupportedRoutes() [][2]int
	QuoteBridge(ctx context.Context, req BridgeQuoteRequest) (*BridgeQuote, error)
	SupportedTokens(sourceChain, destChain int) []string
}
