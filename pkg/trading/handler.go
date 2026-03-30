package trading

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// handleQuote queries all venues and returns the best price.
func (api *TradingAPI) handleQuote(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req QuoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if err := api.validateQuoteRequest(req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	quotes, routing := api.queryAllVenues(r.Context(), req)
	if len(quotes) == 0 {
		writeError(w, http.StatusNotFound, "no liquidity available for this pair")
		return
	}

	sort.Slice(quotes, func(i, j int) bool {
		ai, _ := new(big.Int).SetString(quotes[i].AmountOut, 10)
		aj, _ := new(big.Int).SetString(quotes[j].AmountOut, 10)
		if ai == nil {
			return false
		}
		if aj == nil {
			return true
		}
		return ai.Cmp(aj) > 0
	})

	best := quotes[0]

	route := []Hop{{
		TokenIn:   req.TokenIn,
		TokenOut:  req.TokenOut,
		Venue:     best.Venue,
		Fee:       parseBPS(best.Fee),
		AmountIn:  req.Amount,
		AmountOut: best.AmountOut,
	}}

	priceImpact := computePriceImpact(quotes)
	execPrice := computeExecutionPrice(req.Amount, best.AmountOut)

	resp := QuoteResponse{
		Quote: &Quote{
			AmountIn:       req.Amount,
			AmountOut:      best.AmountOut,
			Route:          route,
			Fee:            best.Fee,
			ExecutionPrice: execPrice,
		},
		Routing:     routing,
		Venues:      quotes,
		GasEstimate: best.GasEstimate,
		PriceImpact: priceImpact,
		PermitData:  nil,
	}

	writeJSON(w, http.StatusOK, resp)
}

// handleSwap converts a quote into unsigned transaction calldata.
// Dispatches to the correct router based on the venue type in the route.
func (api *TradingAPI) handleSwap(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req SwapRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Quote == nil || len(req.Quote.Route) == 0 {
		writeError(w, http.StatusBadRequest, "quote with at least one route hop is required")
		return
	}

	quote := req.Quote
	route := quote.Route

	amountIn, ok := new(big.Int).SetString(quote.AmountIn, 10)
	if !ok || amountIn.Sign() <= 0 {
		writeError(w, http.StatusBadRequest, "invalid amountIn")
		return
	}
	amountOut, ok := new(big.Int).SetString(quote.AmountOut, 10)
	if !ok || amountOut.Sign() <= 0 {
		writeError(w, http.StatusBadRequest, "invalid amountOut")
		return
	}

	from := req.Swapper
	if from == "" {
		from = route[0].TokenIn
	}
	if from == "" {
		writeError(w, http.StatusBadRequest, "cannot determine swapper address; set swapper field")
		return
	}

	// Validate all addresses in the route.
	for i, hop := range route {
		if !isValidAddress(hop.TokenIn) || !isValidAddress(hop.TokenOut) {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid token address in route hop %d", i))
			return
		}
	}

	// Dispatch calldata building based on venue type.
	to, calldata, err := api.buildSwapCalldata(route, amountIn, amountOut)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to build calldata: "+err.Error())
		return
	}

	resp := SwapResponse{
		Swap: TransactionRequest{
			To:       to,
			From:     from,
			Data:     calldata,
			Value:    "0",
			ChainID:  api.defaultChain,
			GasLimit: "250000",
		},
	}

	writeJSON(w, http.StatusOK, resp)
}

// buildSwapCalldata dispatches calldata building based on the venue type in the route.
func (api *TradingAPI) buildSwapCalldata(route []Hop, amountIn, amountOut *big.Int) (string, string, error) {
	venue := route[0].Venue

	// Check for off-chain venues (broker, SOR) — these can't produce calldata.
	if strings.HasPrefix(venue, "broker") || strings.HasPrefix(venue, "alpaca") ||
		strings.HasPrefix(venue, "binance") || strings.HasPrefix(venue, "kraken") ||
		strings.HasPrefix(venue, "coinbase") || strings.HasPrefix(venue, "gemini") ||
		strings.HasPrefix(venue, "hyperliquid") {
		return "", "", fmt.Errorf("venue %q is off-chain only — use /v1/trade/order for broker execution", venue)
	}

	switch {
	case isV4Venue(venue):
		return api.buildV4Calldata(route, amountIn, amountOut)

	case isV2Venue(venue):
		return api.buildV2Calldata(route, amountIn, amountOut)

	case isV3Venue(venue):
		return api.buildV3Calldata(route, amountIn, amountOut)

	default:
		// Default to V4 native for unknown on-chain venues.
		return api.buildV4Calldata(route, amountIn, amountOut)
	}
}

func isV4Venue(name string) bool {
	return name == VenueTypeV4Native || name == "v4_native_clob" || strings.HasPrefix(name, "lux_")
}

func isV2Venue(name string) bool {
	switch name {
	case VenueTypeUniswapV2, VenueTypeSushiswap, VenueTypePancakeV2, VenueTypeTraderJoe:
		return true
	}
	return strings.HasSuffix(name, "_v2")
}

func isV3Venue(name string) bool {
	switch name {
	case VenueTypeUniswapV3, VenueTypePancakeV3:
		return true
	}
	return strings.HasSuffix(name, "_v3")
}

// buildV4Calldata builds calldata for the V4 LXRouter (0x9012).
func (api *TradingAPI) buildV4Calldata(route []Hop, amountIn, amountOut *big.Int) (string, string, error) {
	if len(route) == 1 {
		hop := route[0]
		calldata, err := BuildExactInputSingleCalldata(
			hop.TokenIn, hop.TokenOut, amountIn, amountOut,
			big.NewInt(0), hop.PoolID,
		)
		return LXRouterAddress, calldata, err
	}

	path := make([]string, 0, len(route)+1)
	path = append(path, route[0].TokenIn)
	for _, hop := range route {
		path = append(path, hop.TokenOut)
	}
	calldata, err := BuildExactInputCalldata(path, amountIn, amountOut, big.NewInt(0))
	return LXRouterAddress, calldata, err
}

// buildV2Calldata builds calldata for a Uniswap V2-compatible Router02.
func (api *TradingAPI) buildV2Calldata(route []Hop, amountIn, amountOut *big.Int) (string, string, error) {
	venue := route[0].Venue
	routerAddr, ok := api.venueRouters[venue]
	if !ok {
		return "", "", fmt.Errorf("no router address configured for V2 venue %q", venue)
	}

	path := make([]string, 0, len(route)+1)
	path = append(path, route[0].TokenIn)
	for _, hop := range route {
		path = append(path, hop.TokenOut)
	}

	// Recipient is the swapper (route[0].TokenIn is actually the from address for V4,
	// but for V2 we need the actual recipient). Use zero address as placeholder —
	// the ATS fills in the real MPC wallet address before signing.
	recipient := "0x0000000000000000000000000000000000000000"

	calldata, err := BuildV2SwapCalldata(routerAddr, path, amountIn, amountOut, recipient, nil)
	return routerAddr, calldata, err
}

// buildV3Calldata builds calldata for a Uniswap V3-compatible SwapRouter.
func (api *TradingAPI) buildV3Calldata(route []Hop, amountIn, amountOut *big.Int) (string, string, error) {
	if len(route) != 1 {
		return "", "", fmt.Errorf("V3 multi-hop not yet implemented — use single hop")
	}

	venue := route[0].Venue
	routerAddr, ok := api.venueRouters[venue]
	if !ok {
		return "", "", fmt.Errorf("no router address configured for V3 venue %q", venue)
	}

	hop := route[0]
	// Hop.Fee is in BPS. V3 fee = BPS * 100 (hundredths of bip). 30 BPS -> 3000.
	fee := uint32(hop.Fee * 100)
	if fee == 0 {
		fee = 3000 // default V3 fee tier
	}

	recipient := "0x0000000000000000000000000000000000000000"

	calldata, err := BuildV3SwapCalldata(routerAddr, hop.TokenIn, hop.TokenOut, fee, recipient, amountIn, amountOut, nil)
	return routerAddr, calldata, err
}

// handleOrder submits an order for off-chain routing via Broker.
func (api *TradingAPI) handleOrder(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req OrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if err := api.validateOrderRequest(req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	routing := api.determineOrderRouting(req)
	orderID := generateOrderID()
	now := time.Now().UTC().Format(time.RFC3339)

	order := &OrderStatusResponse{
		OrderID: orderID, Status: OrderStatusPending,
		FilledQty: "0", AvgPrice: "0",
		CreatedAt: now, UpdatedAt: now, Venue: routing,
	}

	api.mu.Lock()
	api.orders[orderID] = order
	api.mu.Unlock()

	writeJSON(w, http.StatusCreated, OrderResponse{
		OrderID: orderID, Status: OrderStatusPending,
		CreatedAt: now, Route: routing,
	})
}

// Stub — returns mock pending status. Production: query EVMClient for tx receipt.
func (api *TradingAPI) handleSwapStatus(w http.ResponseWriter, r *http.Request) {
	txHash := r.PathValue("txHash")
	if txHash == "" {
		writeError(w, http.StatusBadRequest, "txHash is required")
		return
	}
	if len(txHash) != 66 || !strings.HasPrefix(txHash, "0x") {
		writeError(w, http.StatusBadRequest, "invalid transaction hash format")
		return
	}
	writeJSON(w, http.StatusOK, SwapStatusResponse{TxHash: txHash, Status: "pending"})
}

func (api *TradingAPI) handleOrderStatus(w http.ResponseWriter, r *http.Request) {
	orderID := r.PathValue("orderId")
	if orderID == "" {
		writeError(w, http.StatusBadRequest, "orderId is required")
		return
	}

	api.mu.RLock()
	order, ok := api.orders[orderID]
	api.mu.RUnlock()

	if !ok {
		writeError(w, http.StatusNotFound, "order not found")
		return
	}
	writeJSON(w, http.StatusOK, order)
}

// Stub — always approves. Production: call ERC20.allowance(owner, permit2) via EVMClient.
func (api *TradingAPI) handleCheckApproval(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req ApprovalCheckRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if !isValidAddress(req.TokenAddress) {
		writeError(w, http.StatusBadRequest, "invalid token address")
		return
	}
	if !isValidAddress(req.Owner) {
		writeError(w, http.StatusBadRequest, "invalid owner address")
		return
	}
	if !api.validChainIDs[req.ChainID] {
		writeError(w, http.StatusBadRequest, "unsupported chainId")
		return
	}
	writeJSON(w, http.StatusOK, ApprovalCheckResponse{
		Approved: true, Allowance: "115792089237316195423570985008687907853269984665640564039457584007913129639935",
		NeedsApproval: false,
	})
}

func (api *TradingAPI) handleVenues(w http.ResponseWriter, r *http.Request) {
	venues := make([]VenueInfo, len(api.venues))
	for i, v := range api.venues {
		vType := "off_chain"
		if v.IsExecutable() {
			vType = "on_chain"
		}
		venues[i] = VenueInfo{
			Name: v.Name(), Status: "active",
			Executable: v.IsExecutable(), Type: vType,
		}
	}
	writeJSON(w, http.StatusOK, venues)
}

// --- Internal routing logic ---

func (api *TradingAPI) queryAllVenues(ctx context.Context, req QuoteRequest) ([]VenueQuote, string) {
	type result struct {
		quote *VenueQuote
		err   error
	}

	results := make([]result, len(api.venues))
	var wg sync.WaitGroup

	for i, v := range api.venues {
		wg.Add(1)
		go func(idx int, venue Venue) {
			defer wg.Done()
			if req.PreferredVenue != "" && venue.Name() != req.PreferredVenue {
				return
			}
			q, err := venue.Quote(ctx, req)
			results[idx] = result{quote: q, err: err}
		}(i, v)
	}
	wg.Wait()

	var quotes []VenueQuote
	hasOnChain := false

	for _, res := range results {
		if res.err != nil || res.quote == nil {
			continue
		}
		quotes = append(quotes, *res.quote)
		if res.quote.Executable {
			hasOnChain = true
		}
	}

	routing := RoutingBroker
	if hasOnChain && len(quotes) > 0 {
		routing = RoutingV4Native
	}

	return quotes, routing
}

func (api *TradingAPI) determineOrderRouting(req OrderRequest) string {
	for _, v := range api.venues {
		if v.Name() == "v4_native" && v.IsExecutable() {
			return "v4_native"
		}
	}
	for _, v := range api.venues {
		if !v.IsExecutable() {
			return v.Name()
		}
	}
	return "none"
}

// --- Validation ---

func (api *TradingAPI) validateQuoteRequest(req QuoteRequest) error {
	if !isValidAddress(req.TokenIn) {
		return fmt.Errorf("invalid tokenIn address")
	}
	if !isValidAddress(req.TokenOut) {
		return fmt.Errorf("invalid tokenOut address")
	}
	if req.TokenIn == req.TokenOut {
		return fmt.Errorf("tokenIn and tokenOut must be different")
	}
	if !isPositiveDecimal(req.Amount) {
		return fmt.Errorf("amount must be a positive integer")
	}
	if req.Type != QuoteTypeExactInput && req.Type != QuoteTypeExactOutput {
		return fmt.Errorf("type must be EXACT_INPUT or EXACT_OUTPUT")
	}
	if !api.validChainIDs[req.ChainID] {
		return fmt.Errorf("unsupported chainId: %d", req.ChainID)
	}
	if !isValidAddress(req.Swapper) {
		return fmt.Errorf("invalid swapper address")
	}
	if req.Slippage < 0 || req.Slippage > 50 {
		return fmt.Errorf("slippageTolerance must be between 0 and 50")
	}
	return nil
}

func (api *TradingAPI) validateOrderRequest(req OrderRequest) error {
	if !isValidAddress(req.TokenIn) {
		return fmt.Errorf("invalid tokenIn address")
	}
	if !isValidAddress(req.TokenOut) {
		return fmt.Errorf("invalid tokenOut address")
	}
	if req.TokenIn == req.TokenOut {
		return fmt.Errorf("tokenIn and tokenOut must be different")
	}
	if !isPositiveDecimal(req.Amount) {
		return fmt.Errorf("amount must be a positive integer")
	}
	if req.Type != QuoteTypeExactInput && req.Type != QuoteTypeExactOutput {
		return fmt.Errorf("type must be EXACT_INPUT or EXACT_OUTPUT")
	}
	if req.Side != OrderSideBuy && req.Side != OrderSideSell {
		return fmt.Errorf("side must be BUY or SELL")
	}
	if req.OrderType != OrderTypeMarket && req.OrderType != OrderTypeLimit {
		return fmt.Errorf("orderType must be MARKET or LIMIT")
	}
	if req.OrderType == OrderTypeLimit && !isPositiveDecimal(req.Price) {
		return fmt.Errorf("price is required for LIMIT orders")
	}
	if !isValidAddress(req.Swapper) {
		return fmt.Errorf("invalid swapper address")
	}
	if !api.validChainIDs[req.ChainID] {
		return fmt.Errorf("unsupported chainId: %d", req.ChainID)
	}
	return nil
}

// --- Helpers ---

func computePriceImpact(quotes []VenueQuote) string {
	if len(quotes) < 2 {
		return "0.00"
	}
	best, _ := new(big.Int).SetString(quotes[0].AmountOut, 10)
	second, _ := new(big.Int).SetString(quotes[1].AmountOut, 10)
	if best == nil || second == nil || best.Sign() == 0 {
		return "0.00"
	}
	diff := new(big.Int).Sub(best, second)
	diff.Mul(diff, big.NewInt(10000))
	diff.Div(diff, best)
	whole := diff.Int64() / 100
	frac := diff.Int64() % 100
	return fmt.Sprintf("%d.%02d", whole, frac)
}

func computeExecutionPrice(amountInStr, amountOutStr string) string {
	amountIn, ok := new(big.Int).SetString(amountInStr, 10)
	if !ok || amountIn.Sign() == 0 {
		return "0"
	}
	amountOut, ok := new(big.Int).SetString(amountOutStr, 10)
	if !ok {
		return "0"
	}
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	price := new(big.Int).Mul(amountOut, e18)
	price.Div(price, amountIn)
	return price.String()
}

var (
	orderCounter   uint64
	orderCounterMu sync.Mutex
)

func generateOrderID() string {
	orderCounterMu.Lock()
	orderCounter++
	n := orderCounter
	orderCounterMu.Unlock()
	return fmt.Sprintf("ord_%d_%d", time.Now().UnixMilli(), n)
}

func parseBPS(s string) int {
	n := 0
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		} else {
			return 0
		}
	}
	return n
}
