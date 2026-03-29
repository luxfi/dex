package gateway

import "encoding/json"

// Uniswap Trading API-compatible types.
// These match the request/response shapes of https://trade-api.gateway.uniswap.org/v1
// so the Liquidity Exchange frontend can talk to the LXD gateway using the same
// client code it uses for Uniswap.

// UniQuoteRequest is the Uniswap-compatible quote request body.
type UniQuoteRequest struct {
	Type              string   `json:"type"`                        // EXACT_INPUT or EXACT_OUTPUT
	Amount            string   `json:"amount"`                      // wei string
	TokenIn           string   `json:"tokenIn"`                     // address
	TokenOut          string   `json:"tokenOut"`                    // address
	TokenInChainId    int      `json:"tokenInChainId"`
	TokenOutChainId   int      `json:"tokenOutChainId"`
	Swapper           string   `json:"swapper"`                     // wallet address
	SlippageTolerance float64  `json:"slippageTolerance,omitempty"` // percentage (0.5 = 0.5%)
	Protocols         []string `json:"protocols,omitempty"`         // V2, V3, V4, DUTCH_V2, etc.
	RoutingPreference string   `json:"routingPreference,omitempty"` // BEST_PRICE, FASTEST
	Urgency           string   `json:"urgency,omitempty"`           // normal, fast, urgent
	AutoSlippage      string   `json:"autoSlippage,omitempty"`      // DEFAULT
	Deadline          int64    `json:"deadline,omitempty"`
	PermitAmount      string   `json:"permitAmount,omitempty"` // FULL or EXACT
	// Fee fields (API-key associated)
	PortionBips      int    `json:"portionBips,omitempty"`
	PortionRecipient string `json:"portionRecipient,omitempty"`
}

// UniQuoteResponse is the Uniswap-compatible quote response.
type UniQuoteResponse struct {
	RequestId  string          `json:"requestId"`
	Routing    string          `json:"routing"` // CLASSIC, DUTCH_V2, DUTCH_V3, PRIORITY, WRAP, UNWRAP, BRIDGE
	Quote      json.RawMessage `json:"quote"`
	PermitData *UniPermitData  `json:"permitData"` // null when no permit needed
}

// UniClassicQuote is the classic (AMM) quote response body.
type UniClassicQuote struct {
	Input             UniTokenAmount    `json:"input"`
	Output            UniTokenAmount    `json:"output"`
	Swapper           string            `json:"swapper"`
	Route             [][]UniRouteHop   `json:"route"`
	GasFee            string            `json:"gasFee"`
	GasFeeUSD         string            `json:"gasFeeUSD,omitempty"`
	GasFeeQuote       string            `json:"gasFeeQuote,omitempty"`
	GasUseEstimate    string            `json:"gasUseEstimate"`
	RouteString       string            `json:"routeString"`
	QuoteId           string            `json:"quoteId"`
	BlockNumber       string            `json:"blockNumber,omitempty"`
	PriceImpact       float64           `json:"priceImpact,omitempty"`
	SlippageTolerance float64           `json:"slippageTolerance,omitempty"`
	TxFailureReasons  []string          `json:"txFailureReasons,omitempty"`
	PortionBips       int               `json:"portionBips,omitempty"`
	PortionAmount     string            `json:"portionAmount,omitempty"`
	PortionRecipient  string            `json:"portionRecipient,omitempty"`
}

// UniTokenAmount represents a token address + amount pair with chain context.
type UniTokenAmount struct {
	Token   string `json:"token"`   // address
	Amount  string `json:"amount"`  // wei string
	ChainId int    `json:"chainId"`
}

// UniRouteHop represents one pool hop in a swap route.
type UniRouteHop struct {
	Type         string `json:"type"`                   // v2-pool, v3-pool, v4-pool
	Address      string `json:"address"`
	TokenIn      string `json:"tokenIn"`
	TokenOut     string `json:"tokenOut"`
	Fee          string `json:"fee,omitempty"`
	Liquidity    string `json:"liquidity,omitempty"`
	SqrtRatioX96 string `json:"sqrtRatioX96,omitempty"`
	TickCurrent  string `json:"tickCurrent,omitempty"`
	AmountIn     string `json:"amountIn,omitempty"`
	AmountOut    string `json:"amountOut,omitempty"`
}

// UniPermitData is the EIP-712 typed data for Permit2 approval.
type UniPermitData struct {
	Domain UniPermitDomain            `json:"domain"`
	Types  map[string][]UniTypedField `json:"types"`
	Values map[string]interface{}     `json:"values"`
}

// UniPermitDomain is the EIP-712 domain for Permit2.
type UniPermitDomain struct {
	Name              string `json:"name"`
	ChainId           int    `json:"chainId"`
	VerifyingContract string `json:"verifyingContract"`
}

// UniTypedField is a single field in an EIP-712 struct type.
type UniTypedField struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// UniCreateSwapRequest is the Uniswap-compatible swap creation request.
type UniCreateSwapRequest struct {
	Quote               json.RawMessage `json:"quote"`
	Signature           string          `json:"signature,omitempty"`
	PermitData          *UniPermitData  `json:"permitData,omitempty"`
	SimulateTransaction bool            `json:"simulateTransaction,omitempty"`
	Deadline            int64           `json:"deadline,omitempty"`
	Urgency             string          `json:"urgency,omitempty"`
	RefreshGasPrice     bool            `json:"refreshGasPrice,omitempty"`
}

// UniCreateSwapResponse is the Uniswap-compatible swap creation response.
type UniCreateSwapResponse struct {
	Type string              `json:"type"` // TRANSACTION or SWAP_SUCCESS
	Swap *UniTransactionData `json:"swap,omitempty"`
}

// UniTransactionData is an unsigned EVM transaction envelope.
type UniTransactionData struct {
	To                   string `json:"to"`
	From                 string `json:"from"`
	Data                 string `json:"data"`
	Value                string `json:"value"`
	ChainId              int    `json:"chainId"`
	GasLimit             string `json:"gasLimit,omitempty"`
	MaxFeePerGas         string `json:"maxFeePerGas,omitempty"`
	MaxPriorityFeePerGas string `json:"maxPriorityFeePerGas,omitempty"`
	GasPrice             string `json:"gasPrice,omitempty"`
}

// UniApprovalRequest is the Uniswap-compatible token approval check request.
type UniApprovalRequest struct {
	WalletAddress  string `json:"walletAddress"`
	Token          string `json:"token"`
	Amount         string `json:"amount"`
	ChainId        int    `json:"chainId"`
	IncludeGasInfo bool   `json:"includeGasInfo,omitempty"`
	Urgency        string `json:"urgency,omitempty"`
}

// UniApprovalResponse is the Uniswap-compatible token approval response.
type UniApprovalResponse struct {
	Approval *UniTransactionData `json:"approval"`
	Cancel   *UniTransactionData `json:"cancel,omitempty"`
	GasFee   string              `json:"gasFee,omitempty"`
}

// UniOrderRequest submits a signed UniswapX order.
type UniOrderRequest struct {
	Signature string          `json:"signature"`
	Quote     json.RawMessage `json:"quote"`
	Routing   string          `json:"routing"`
}

// UniOrderResponse is the result of submitting an order.
type UniOrderResponse struct {
	EncodedOrder string `json:"encodedOrder"`
	OrderId      string `json:"orderId"`
}

// UniGetOrdersResponse wraps a paginated list of orders.
type UniGetOrdersResponse struct {
	Orders []UniOrderStatus `json:"orders"`
	Cursor string           `json:"cursor,omitempty"`
}

// UniOrderStatus describes an order's current state.
type UniOrderStatus struct {
	OrderId        string `json:"orderId"`
	OrderStatus    string `json:"orderStatus"` // open, filled, expired, cancelled, error
	SwapperAddress string `json:"swapper"`
	EncodedOrder   string `json:"encodedOrder,omitempty"`
	TxHash         string `json:"txHash,omitempty"`
	ChainId        int    `json:"chainId"`
	CreatedAt      int64  `json:"createdAt"`
}

// UniGetSwapsResponse wraps swap status queries.
type UniGetSwapsResponse struct {
	Swaps []UniSwapStatus `json:"swaps"`
}

// UniSwapStatus tracks the lifecycle of a swap transaction.
type UniSwapStatus struct {
	TxHash  string `json:"txHash"`
	ChainId int    `json:"chainId"`
	Status  string `json:"status"` // pending, confirmed, failed
}

// UniLPApproveRequest checks whether LP token approvals are needed.
type UniLPApproveRequest struct {
	WalletAddress string `json:"walletAddress"`
	Token         string `json:"token"`
	Amount        string `json:"amount"`
	ChainId       int    `json:"chainId"`
	PositionToken string `json:"positionToken,omitempty"` // V4 position manager
}

// UniCreateLPRequest creates a new LP position.
type UniCreateLPRequest struct {
	WalletAddress string `json:"walletAddress"`
	ChainId       int    `json:"chainId"`
	Protocol      string `json:"protocol"` // V2, V3, V4
	Token0        string `json:"token0"`
	Token1        string `json:"token1"`
	Fee           int    `json:"fee,omitempty"` // V3/V4
	TickLower     int    `json:"tickLower,omitempty"`
	TickUpper     int    `json:"tickUpper,omitempty"`
	Amount0       string `json:"amount0"`
	Amount1       string `json:"amount1"`
	Amount0Min    string `json:"amount0Min,omitempty"`
	Amount1Min    string `json:"amount1Min,omitempty"`
	Deadline      int64  `json:"deadline,omitempty"`
	SqrtPriceX96  string `json:"sqrtPriceX96,omitempty"` // for initializing new pool
	CreatePool    bool   `json:"createPool,omitempty"`
	Hooks         string `json:"hooks,omitempty"` // V4 hooks address
	Salt          string `json:"salt,omitempty"`
}

// UniModifyLPRequest is used for increase, decrease, and claim operations.
type UniModifyLPRequest struct {
	WalletAddress string `json:"walletAddress"`
	ChainId       int    `json:"chainId"`
	Protocol      string `json:"protocol"` // V2, V3, V4
	Token0        string `json:"token0"`
	Token1        string `json:"token1"`
	Fee           int    `json:"fee,omitempty"`
	TickLower     int    `json:"tickLower,omitempty"`
	TickUpper     int    `json:"tickUpper,omitempty"`
	Liquidity     string `json:"liquidity,omitempty"` // for decrease
	Amount0       string `json:"amount0,omitempty"`   // for increase
	Amount1       string `json:"amount1,omitempty"`   // for increase
	Amount0Min    string `json:"amount0Min,omitempty"`
	Amount1Min    string `json:"amount1Min,omitempty"`
	Deadline      int64  `json:"deadline,omitempty"`
	Hooks         string `json:"hooks,omitempty"`
	Salt          string `json:"salt,omitempty"`
}

// UniLPResponse wraps LP operation results.
type UniLPResponse struct {
	Type  string          `json:"type"` // TRANSACTION
	Tx    *UniTransactionData `json:"transaction,omitempty"`
	Steps []UniLPStep     `json:"steps,omitempty"` // for multi-step (approve + create)
}

// UniLPStep is one step in a multi-step LP operation.
type UniLPStep struct {
	Type string              `json:"type"` // APPROVAL, CREATE, INCREASE, DECREASE, CLAIM
	Tx   *UniTransactionData `json:"transaction"`
}

// UniPoolInfoRequest queries pool metadata.
type UniPoolInfoRequest struct {
	ChainId  int    `json:"chainId"`
	Protocol string `json:"protocol"` // V2, V3, V4
	Token0   string `json:"token0"`
	Token1   string `json:"token1"`
	Fee      int    `json:"fee,omitempty"`
	Hooks    string `json:"hooks,omitempty"`
}

// UniPoolInfoResponse returns pool metadata.
type UniPoolInfoResponse struct {
	PoolAddress  string `json:"poolAddress"`
	Token0       string `json:"token0"`
	Token1       string `json:"token1"`
	Fee          int    `json:"fee"`
	Liquidity    string `json:"liquidity"`
	SqrtPriceX96 string `json:"sqrtPriceX96"`
	Tick         int    `json:"tick"`
	TVL          string `json:"tvl,omitempty"`
	Volume24h    string `json:"volume24h,omitempty"`
}

// UniSendRequest builds an ERC20 transfer transaction.
type UniSendRequest struct {
	WalletAddress string `json:"walletAddress"`
	Token         string `json:"token"`
	Amount        string `json:"amount"`
	Recipient     string `json:"recipient"`
	ChainId       int    `json:"chainId"`
}

// UniSendResponse returns the built transfer transaction.
type UniSendResponse struct {
	Type string              `json:"type"` // TRANSACTION
	Tx   *UniTransactionData `json:"transaction"`
}

// UniSwappableTokensResponse lists bridgeable destination chains for a token.
type UniSwappableTokensResponse struct {
	Tokens []UniSwappableToken `json:"tokens"`
}

// UniSwappableToken is a token available for cross-chain swapping.
type UniSwappableToken struct {
	Address string `json:"address"`
	ChainId int    `json:"chainId"`
	Symbol  string `json:"symbol"`
	Name    string `json:"name"`
}

// UniErrorResponse is the Uniswap-compatible error body.
type UniErrorResponse struct {
	ErrorCode string `json:"errorCode,omitempty"`
	Detail    string `json:"detail"`
}
