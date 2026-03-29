package gateway

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// registerTradingRoutes registers Uniswap Trading API-compatible endpoints
// under the /trading/ prefix. These return response objects directly (no
// apiResponse wrapper) to match the Uniswap spec.
func (s *Server) registerTradingRoutes() {
	s.mux.HandleFunc("/trading/quote", s.handleTradingQuote)
	s.mux.HandleFunc("/trading/swap", s.handleTradingSwap)
	s.mux.HandleFunc("/trading/check_approval", s.handleTradingCheckApproval)
	s.mux.HandleFunc("/trading/order", s.handleTradingOrder)
	s.mux.HandleFunc("/trading/orders", s.handleTradingGetOrders)
	s.mux.HandleFunc("/trading/swaps", s.handleTradingGetSwaps)
	s.mux.HandleFunc("/trading/send", s.handleTradingSend)
	s.mux.HandleFunc("/trading/swappable_tokens", s.handleTradingSwappableTokens)
	s.mux.HandleFunc("/trading/lp/approve", s.handleTradingLPApprove)
	s.mux.HandleFunc("/trading/lp/create", s.handleTradingLPCreate)
	s.mux.HandleFunc("/trading/lp/increase", s.handleTradingLPIncrease)
	s.mux.HandleFunc("/trading/lp/decrease", s.handleTradingLPDecrease)
	s.mux.HandleFunc("/trading/lp/claim", s.handleTradingLPClaim)
	s.mux.HandleFunc("/trading/lp/pool_info", s.handleTradingLPPoolInfo)
}

// writeTradingJSON writes a JSON response without the apiResponse wrapper.
func (s *Server) writeTradingJSON(w http.ResponseWriter, r *http.Request, status int, data interface{}) {
	reqID := r.Header.Get("X-Request-ID")
	if reqID == "" {
		reqID = uuid.New().String()
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-Id", reqID)
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeTradingError writes a Uniswap-compatible error response.
func (s *Server) writeTradingError(w http.ResponseWriter, r *http.Request, status int, code string, detail string) {
	reqID := r.Header.Get("X-Request-ID")
	if reqID == "" {
		reqID = uuid.New().String()
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-Id", reqID)
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(UniErrorResponse{
		ErrorCode: code,
		Detail:    detail,
	})
}

// POST /trading/quote
func (s *Server) handleTradingQuote(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniQuoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	if req.TokenIn == "" || req.TokenOut == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "tokenIn and tokenOut are required")
		return
	}
	if req.Amount == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "amount is required")
		return
	}

	isExactIn := strings.ToUpper(req.Type) != "EXACT_OUTPUT"
	chainID := req.TokenInChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	slippage := req.SlippageTolerance
	if slippage == 0 {
		slippage = 0.5 // default 0.5%
	}

	ctx := s.requestContext(r)
	quote, err := s.router.GetBestQuote(ctx, QuoteRequest{
		TokenIn:   Token{Address: req.TokenIn, ChainID: ChainID(chainID)},
		TokenOut:  Token{Address: req.TokenOut, ChainID: ChainID(chainID)},
		Amount:    parseBigIntStr(req.Amount),
		IsExactIn: isExactIn,
		ChainID:   ChainID(chainID),
		Slippage:  slippage,
	})
	if err != nil {
		s.writeTradingError(w, r, http.StatusInternalServerError, "QUOTE_ERROR", "failed to get quote: "+err.Error())
		return
	}

	requestID := uuid.New().String()
	quoteID := quote.QuoteID
	if quoteID == "" {
		quoteID = uuid.New().String()
	}

	// Build route segments from the internal quote
	route := make([][]UniRouteHop, 0, 1)
	hops := make([]UniRouteHop, 0, len(quote.Route))
	for _, hop := range quote.Route {
		hops = append(hops, UniRouteHop{
			Type:    poolTypeToUni(hop.PoolType),
			Address: hop.PoolAddress,
			TokenIn: hop.TokenIn.Address,
			TokenOut: hop.TokenOut.Address,
			Fee:     strconv.Itoa(hop.Fee),
		})
	}
	if len(hops) == 0 {
		// Synthesize a single v4-pool hop for the precompile path
		hops = append(hops, UniRouteHop{
			Type:     "v4-pool",
			Address:  LXPoolPrecompile,
			TokenIn:  req.TokenIn,
			TokenOut: req.TokenOut,
			Fee:      "3000",
		})
	}
	route = append(route, hops)

	// Build route string
	routeParts := make([]string, 0, len(hops)+1)
	if len(hops) > 0 {
		routeParts = append(routeParts, hops[0].TokenIn)
		for _, h := range hops {
			routeParts = append(routeParts, h.TokenOut)
		}
	}
	routeString := strings.Join(routeParts, " -> ")

	gasEstimate := "150000"
	gasFee := "0"
	if quote.GasEstimate != nil && quote.GasEstimate.Sign() > 0 {
		gasEstimate = quote.GasEstimate.String()
	}

	outChainID := req.TokenOutChainId
	if outChainID == 0 {
		outChainID = chainID
	}

	classic := UniClassicQuote{
		Input: UniTokenAmount{
			Token:   req.TokenIn,
			Amount:  bigIntStr(quote.TokenIn.Amount),
			ChainId: chainID,
		},
		Output: UniTokenAmount{
			Token:   req.TokenOut,
			Amount:  bigIntStr(quote.TokenOut.Amount),
			ChainId: outChainID,
		},
		Swapper:           req.Swapper,
		Route:             route,
		GasFee:            gasFee,
		GasUseEstimate:    gasEstimate,
		RouteString:       routeString,
		QuoteId:           quoteID,
		PriceImpact:       quote.PriceImpact,
		SlippageTolerance: slippage,
		PortionBips:       req.PortionBips,
		PortionRecipient:  req.PortionRecipient,
	}

	classicBytes, _ := json.Marshal(classic)

	s.writeTradingJSON(w, r, http.StatusOK, UniQuoteResponse{
		RequestId:  requestID,
		Routing:    "CLASSIC",
		Quote:      json.RawMessage(classicBytes),
		PermitData: nil, // No Permit2 on Lux chains
	})
}

// POST /trading/swap
func (s *Server) handleTradingSwap(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniCreateSwapRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	// Extract the classic quote from the request
	var classic UniClassicQuote
	if err := json.Unmarshal(req.Quote, &classic); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid quote in request: "+err.Error())
		return
	}

	if classic.Input.Token == "" || classic.Output.Token == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "quote must contain input and output tokens")
		return
	}

	chainID := classic.Input.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// Determine fee from route
	var fee uint32 = 3000
	if len(classic.Route) > 0 && len(classic.Route[0]) > 0 {
		if f, err := strconv.ParseUint(classic.Route[0][0].Fee, 10, 32); err == nil && f > 0 {
			fee = uint32(f)
		}
	}

	// Build swap calldata using existing swap_builder
	swReq := swapRequest{
		TokenIn:   classic.Input.Token,
		TokenOut:  classic.Output.Token,
		ChainID:   uint64(chainID),
		Amount:    classic.Input.Amount,
		IsExactIn: true, // Classic quotes are always exact-in from the quote step
		Slippage:  classic.SlippageTolerance,
		Recipient: classic.Swapper,
		Deadline:  req.Deadline,
		Fee:       fee,
	}

	ctx := s.requestContext(r)
	quote, err := s.router.GetBestQuote(ctx, QuoteRequest{
		TokenIn:   Token{Address: swReq.TokenIn, ChainID: ChainID(swReq.ChainID)},
		TokenOut:  Token{Address: swReq.TokenOut, ChainID: ChainID(swReq.ChainID)},
		Amount:    parseBigIntStr(swReq.Amount),
		IsExactIn: swReq.IsExactIn,
		ChainID:   ChainID(swReq.ChainID),
		Slippage:  swReq.Slippage,
	})
	if err != nil {
		s.writeTradingError(w, r, http.StatusInternalServerError, "SWAP_ERROR", "failed to build swap: "+err.Error())
		return
	}

	resp := buildSwapFromQuote(swReq, quote)

	gasStr := strconv.FormatUint(resp.Swap.GasLimit, 10)

	s.writeTradingJSON(w, r, http.StatusOK, UniCreateSwapResponse{
		Type: "TRANSACTION",
		Swap: &UniTransactionData{
			To:       resp.Swap.To,
			From:     classic.Swapper,
			Data:     resp.Swap.Data,
			Value:    resp.Swap.Value,
			ChainId:  chainID,
			GasLimit: gasStr,
		},
	})
}

// POST /trading/check_approval
func (s *Server) handleTradingCheckApproval(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniApprovalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	if req.Token == "" || req.WalletAddress == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "token and walletAddress are required")
		return
	}

	amount := parseBigIntStr(req.Amount)
	if amount == nil {
		// MaxUint256 approval
		amount, _ = new(big.Int).SetString("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10)
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// Build ERC20 approve(spender, amount) calldata
	// Function selector: approve(address,uint256) = 0x095ea7b3
	calldata := buildApproveCalldata(LXPoolPrecompile, amount)

	s.writeTradingJSON(w, r, http.StatusOK, UniApprovalResponse{
		Approval: &UniTransactionData{
			To:       req.Token,
			From:     req.WalletAddress,
			Data:     "0x" + hex.EncodeToString(calldata),
			Value:    "0x0",
			ChainId:  chainID,
			GasLimit: "50000",
		},
	})
}

// POST /trading/order
func (s *Server) handleTradingOrder(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniOrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	if req.Signature == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "signature is required")
		return
	}

	// Extract quote info to build an internal order
	var quoteInfo struct {
		Input   UniTokenAmount `json:"input"`
		Output  UniTokenAmount `json:"output"`
		Swapper string         `json:"swapper"`
		QuoteId string         `json:"quoteId"`
	}
	if err := json.Unmarshal(req.Quote, &quoteInfo); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid quote: "+err.Error())
		return
	}

	chainID := quoteInfo.Input.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// Create an internal limit order from the UniswapX order
	deadline := time.Now().Add(24 * time.Hour).Unix()
	createReq := CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    quoteInfo.Input.Token,
		TokenOut:   quoteInfo.Output.Token,
		AmountIn:   quoteInfo.Input.Amount,
		LimitPrice: quoteInfo.Output.Amount,
		ChainID:    uint64(chainID),
		Deadline:   deadline,
		Recipient:  quoteInfo.Swapper,
	}

	order := s.orders.store.Create(&createReq)

	s.writeTradingJSON(w, r, http.StatusOK, UniOrderResponse{
		EncodedOrder: req.Signature, // Echo signature as encoded order
		OrderId:      order.ID,
	})
}

// GET /trading/orders
func (s *Server) handleTradingGetOrders(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	swapper := r.URL.Query().Get("swapper")
	if swapper == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "swapper query parameter is required")
		return
	}

	statusFilter := OrderStatus(r.URL.Query().Get("orderStatus"))
	orders := s.orders.store.ListByOwner(swapper, statusFilter)

	uniOrders := make([]UniOrderStatus, 0, len(orders))
	for _, o := range orders {
		uniOrders = append(uniOrders, gatewayOrderToUniStatus(&o))
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniGetOrdersResponse{
		Orders: uniOrders,
	})
}

// GET /trading/swaps
func (s *Server) handleTradingGetSwaps(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	txHash := r.URL.Query().Get("txHash")
	if txHash == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "txHash query parameter is required")
		return
	}

	chainID, _ := strconv.Atoi(r.URL.Query().Get("chainId"))
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// We don't have on-chain indexing yet, so return a pending status.
	// This is sufficient for the frontend to show a "pending" state and
	// retry until it sees confirmed.
	s.writeTradingJSON(w, r, http.StatusOK, UniGetSwapsResponse{
		Swaps: []UniSwapStatus{{
			TxHash:  txHash,
			ChainId: chainID,
			Status:  "pending",
		}},
	})
}

// POST /trading/send
func (s *Server) handleTradingSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniSendRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	if req.Token == "" || req.Recipient == "" || req.Amount == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "token, recipient, and amount are required")
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	amount := parseBigIntStr(req.Amount)
	if amount == nil || amount.Sign() <= 0 {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "amount must be a positive integer")
		return
	}

	// Check if this is a native token transfer (address(0))
	isNative := req.Token == "0x0000000000000000000000000000000000000000" ||
		strings.ToLower(req.Token) == "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

	if isNative {
		s.writeTradingJSON(w, r, http.StatusOK, UniSendResponse{
			Type: "TRANSACTION",
			Tx: &UniTransactionData{
				To:       req.Recipient,
				From:     req.WalletAddress,
				Data:     "0x",
				Value:    "0x" + amount.Text(16),
				ChainId:  chainID,
				GasLimit: "21000",
			},
		})
		return
	}

	// Build ERC20 transfer(address,uint256) calldata
	// Function selector: 0xa9059cbb
	calldata := buildTransferCalldata(req.Recipient, amount)

	s.writeTradingJSON(w, r, http.StatusOK, UniSendResponse{
		Type: "TRANSACTION",
		Tx: &UniTransactionData{
			To:       req.Token,
			From:     req.WalletAddress,
			Data:     "0x" + hex.EncodeToString(calldata),
			Value:    "0x0",
			ChainId:  chainID,
			GasLimit: "65000",
		},
	})
}

// GET /trading/swappable_tokens
func (s *Server) handleTradingSwappableTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	tokenAddr := r.URL.Query().Get("tokenAddress")
	chainID, _ := strconv.Atoi(r.URL.Query().Get("chainId"))
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// Return the token on supported Lux ecosystem chains
	tokens := make([]UniSwappableToken, 0, 2)

	if ChainID(chainID) != ChainIDLux {
		tokens = append(tokens, UniSwappableToken{
			Address: tokenAddr,
			ChainId: int(ChainIDLux),
			Symbol:  "",
			Name:    "",
		})
	}
	if ChainID(chainID) != ChainIDZoo {
		tokens = append(tokens, UniSwappableToken{
			Address: tokenAddr,
			ChainId: int(ChainIDZoo),
			Symbol:  "",
			Name:    "",
		})
	}

	// Enrich with token info if available
	ctx := s.requestContext(r)
	for i := range tokens {
		tl, err := s.router.GetTokenList(ctx, ChainID(tokens[i].ChainId))
		if err != nil {
			continue
		}
		for _, t := range tl {
			if strings.EqualFold(t.Address, tokens[i].Address) {
				tokens[i].Symbol = t.Symbol
				tokens[i].Name = t.Name
				break
			}
		}
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniSwappableTokensResponse{Tokens: tokens})
}

// POST /trading/lp/approve
func (s *Server) handleTradingLPApprove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniLPApproveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	if req.Token == "" || req.WalletAddress == "" {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "token and walletAddress are required")
		return
	}

	amount := parseBigIntStr(req.Amount)
	if amount == nil {
		amount, _ = new(big.Int).SetString("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10)
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	spender := lxPoolAddress
	if req.PositionToken != "" {
		spender = req.PositionToken
	}

	calldata := buildApproveCalldata(spender, amount)

	s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
		Type: "TRANSACTION",
		Tx: &UniTransactionData{
			To:       req.Token,
			From:     req.WalletAddress,
			Data:     "0x" + hex.EncodeToString(calldata),
			Value:    "0x0",
			ChainId:  chainID,
			GasLimit: "50000",
		},
	})
}

// POST /trading/lp/create
func (s *Server) handleTradingLPCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniCreateLPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	internal := CreatePositionRequest{
		Token0:         req.Token0,
		Token1:         req.Token1,
		Fee:            uint32(req.Fee),
		TickLower:      int32(req.TickLower),
		TickUpper:      int32(req.TickUpper),
		Amount0Desired: req.Amount0,
		Amount1Desired: req.Amount1,
		Amount0Min:     req.Amount0Min,
		Amount1Min:     req.Amount1Min,
		Recipient:      req.WalletAddress,
		Deadline:       req.Deadline,
		SqrtPriceX96:   req.SqrtPriceX96,
		Salt:           req.Salt,
		Hooks:          req.Hooks,
		ChainID:        uint64(chainID),
	}

	steps := make([]UniLPStep, 0, 2)

	// If creating a new pool, include the initialize step
	if req.CreatePool && req.SqrtPriceX96 != "" {
		initTx, err := BuildInitializePoolTx(internal)
		if err != nil {
			s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
			return
		}
		steps = append(steps, UniLPStep{
			Type: "CREATE",
			Tx:   unsignedToUniTx(initTx, req.WalletAddress, chainID),
		})
	}

	addTx, err := BuildCreatePositionTx(internal)
	if err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	if len(steps) > 0 {
		steps = append(steps, UniLPStep{
			Type: "INCREASE",
			Tx:   unsignedToUniTx(addTx, req.WalletAddress, chainID),
		})
		s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
			Type:  "TRANSACTION",
			Steps: steps,
		})
	} else {
		s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
			Type: "TRANSACTION",
			Tx:   unsignedToUniTx(addTx, req.WalletAddress, chainID),
		})
	}
}

// POST /trading/lp/increase
func (s *Server) handleTradingLPIncrease(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniModifyLPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	internal := IncreasePositionRequest{
		Token0:         req.Token0,
		Token1:         req.Token1,
		Fee:            uint32(req.Fee),
		TickLower:      int32(req.TickLower),
		TickUpper:      int32(req.TickUpper),
		Amount0Desired: req.Amount0,
		Amount1Desired: req.Amount1,
		Amount0Min:     req.Amount0Min,
		Amount1Min:     req.Amount1Min,
		Deadline:       req.Deadline,
		Salt:           req.Salt,
		Hooks:          req.Hooks,
		ChainID:        uint64(chainID),
	}

	tx, err := BuildIncreasePositionTx(internal)
	if err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
		Type: "TRANSACTION",
		Tx:   unsignedToUniTx(tx, req.WalletAddress, chainID),
	})
}

// POST /trading/lp/decrease
func (s *Server) handleTradingLPDecrease(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniModifyLPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	internal := DecreasePositionRequest{
		Token0:     req.Token0,
		Token1:     req.Token1,
		Fee:        uint32(req.Fee),
		TickLower:  int32(req.TickLower),
		TickUpper:  int32(req.TickUpper),
		Liquidity:  req.Liquidity,
		Amount0Min: req.Amount0Min,
		Amount1Min: req.Amount1Min,
		Deadline:   req.Deadline,
		Salt:       req.Salt,
		Hooks:      req.Hooks,
		ChainID:    uint64(chainID),
	}

	tx, err := BuildDecreasePositionTx(internal)
	if err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
		Type: "TRANSACTION",
		Tx:   unsignedToUniTx(tx, req.WalletAddress, chainID),
	})
}

// POST /trading/lp/claim
func (s *Server) handleTradingLPClaim(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniModifyLPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	internal := ClaimFeesRequest{
		Token0:    req.Token0,
		Token1:    req.Token1,
		Fee:       uint32(req.Fee),
		TickLower: int32(req.TickLower),
		TickUpper: int32(req.TickUpper),
		Recipient: req.WalletAddress,
		Salt:      req.Salt,
		Hooks:     req.Hooks,
		ChainID:   uint64(chainID),
	}

	tx, err := BuildClaimFeesTx(internal)
	if err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniLPResponse{
		Type: "TRANSACTION",
		Tx:   unsignedToUniTx(tx, req.WalletAddress, chainID),
	})
}

// POST /trading/lp/pool_info
func (s *Server) handleTradingLPPoolInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeTradingError(w, r, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	var req UniPoolInfoRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeTradingError(w, r, http.StatusBadRequest, "INVALID_REQUEST", "invalid request body: "+err.Error())
		return
	}

	chainID := req.ChainId
	if chainID == 0 {
		chainID = int(ChainIDLux)
	}

	// Try to find the pool via the router
	ctx := s.requestContext(r)
	pools, err := s.router.GetPools(ctx, PoolsRequest{
		ChainID:  ChainID(chainID),
		Token0:   req.Token0,
		Token1:   req.Token1,
		Protocol: strings.ToLower(req.Protocol),
	})

	if err != nil || len(pools) == 0 {
		// Return a synthetic pool response based on the precompile address
		fee := req.Fee
		if fee == 0 {
			fee = 3000
		}
		s.writeTradingJSON(w, r, http.StatusOK, UniPoolInfoResponse{
			PoolAddress:  lxPoolAddress,
			Token0:       req.Token0,
			Token1:       req.Token1,
			Fee:          fee,
			Liquidity:    "0",
			SqrtPriceX96: "0",
			Tick:         0,
		})
		return
	}

	pool := pools[0]
	tvlStr := ""
	if pool.TVL != nil {
		tvlStr = pool.TVL.String()
	}
	volStr := ""
	if pool.Volume24h != nil {
		volStr = pool.Volume24h.String()
	}

	s.writeTradingJSON(w, r, http.StatusOK, UniPoolInfoResponse{
		PoolAddress:  pool.Address,
		Token0:       pool.Token0.Address,
		Token1:       pool.Token1.Address,
		Fee:          pool.Fee,
		Liquidity:    "0", // On-chain query needed for real value
		SqrtPriceX96: "0",
		Tick:         0,
		TVL:          tvlStr,
		Volume24h:    volStr,
	})
}

// --- helpers ---

// buildApproveCalldata builds ERC20 approve(address spender, uint256 amount) calldata.
// Selector: 0x095ea7b3
func buildApproveCalldata(spender string, amount *big.Int) []byte {
	data := make([]byte, 68)
	// selector
	data[0] = 0x09
	data[1] = 0x5e
	data[2] = 0xa7
	data[3] = 0xb3
	// spender (address, left-padded to 32 bytes)
	copy(data[4+12:4+32], addrBytes(spender))
	// amount (uint256)
	ab := amount.Bytes()
	copy(data[36+32-len(ab):36+32], ab)
	return data
}

// buildTransferCalldata builds ERC20 transfer(address to, uint256 amount) calldata.
// Selector: 0xa9059cbb
func buildTransferCalldata(to string, amount *big.Int) []byte {
	data := make([]byte, 68)
	// selector
	data[0] = 0xa9
	data[1] = 0x05
	data[2] = 0x9c
	data[3] = 0xbb
	// to (address, left-padded to 32 bytes)
	copy(data[4+12:4+32], addrBytes(to))
	// amount (uint256)
	ab := amount.Bytes()
	copy(data[36+32-len(ab):36+32], ab)
	return data
}

// bigIntStr converts a *big.Int to a decimal string, defaulting to "0".
func bigIntStr(n *big.Int) string {
	if n == nil {
		return "0"
	}
	return n.String()
}

// poolTypeToUni maps internal pool type strings to Uniswap route segment types.
func poolTypeToUni(poolType string) string {
	switch strings.ToLower(poolType) {
	case "v2", "uniswap_v2":
		return "v2-pool"
	case "v3", "uniswap_v3":
		return "v3-pool"
	case "v4", "lux_amm", "lux_v4":
		return "v4-pool"
	default:
		return "v4-pool"
	}
}

// gatewayOrderToUniStatus maps an internal GatewayOrder to a UniOrderStatus.
func gatewayOrderToUniStatus(o *GatewayOrder) UniOrderStatus {
	status := "open"
	switch o.Status {
	case OrderStatusFilled:
		status = "filled"
	case OrderStatusExpired:
		status = "expired"
	case OrderStatusCancelled:
		status = "cancelled"
	case OrderStatusPartiallyFilled:
		status = "open"
	}

	var txHash string
	if len(o.Fills) > 0 {
		txHash = o.Fills[len(o.Fills)-1].TxHash
	}

	return UniOrderStatus{
		OrderId:        o.ID,
		OrderStatus:    status,
		SwapperAddress: o.Owner,
		TxHash:         txHash,
		ChainId:        int(o.ChainID),
		CreatedAt:      o.CreatedAt.Unix(),
	}
}

// unsignedToUniTx converts an internal UnsignedTxResponse to a UniTransactionData.
func unsignedToUniTx(tx *UnsignedTxResponse, from string, chainID int) *UniTransactionData {
	return &UniTransactionData{
		To:       tx.To,
		From:     from,
		Data:     tx.Data,
		Value:    tx.Value,
		ChainId:  chainID,
		GasLimit: fmt.Sprintf("%d", tx.GasLimit),
	}
}
