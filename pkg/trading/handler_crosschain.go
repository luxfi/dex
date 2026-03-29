package trading

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// RegisterCrossChainRoutes mounts cross-chain and arbitrage endpoints.
func (api *TradingAPI) RegisterCrossChainRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/trade/cross-chain/quote", api.handleCrossChainQuote)
	mux.HandleFunc("GET /v1/trade/bridges", api.handleBridges)
	mux.HandleFunc("POST /v1/trade/arbitrage/scan", api.handleArbScan)
}

// SetCrossChainRouter configures cross-chain routing.
func (api *TradingAPI) SetCrossChainRouter(r *CrossChainRouter) {
	api.crossChainRouter = r
}

// SetArbScanner configures the arbitrage scanner.
func (api *TradingAPI) SetArbScanner(s *ArbScanner) {
	api.arbScanner = s
}

func (api *TradingAPI) handleCrossChainQuote(w http.ResponseWriter, r *http.Request) {
	if api.crossChainRouter == nil {
		writeError(w, http.StatusServiceUnavailable, "cross-chain routing not configured")
		return
	}

	var req CrossChainQuoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if err := validateCrossChainRequest(req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Type == "" {
		req.Type = QuoteTypeExactInput
	}

	routes, err := api.crossChainRouter.FindRoutes(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if len(routes) == 0 {
		writeError(w, http.StatusNotFound, "no cross-chain route found")
		return
	}

	writeJSON(w, http.StatusOK, CrossChainQuoteResponse{
		Routes: routes, BestRoute: &routes[0],
	})
}

func (api *TradingAPI) handleBridges(w http.ResponseWriter, _ *http.Request) {
	if api.crossChainRouter == nil {
		writeJSON(w, http.StatusOK, []BridgeInfo{})
		return
	}

	api.crossChainRouter.mu.RLock()
	bridges := api.crossChainRouter.bridges
	api.crossChainRouter.mu.RUnlock()

	var infos []BridgeInfo
	for _, b := range bridges {
		for _, route := range b.SupportedRoutes() {
			tokens := b.SupportedTokens(route[0], route[1])
			infos = append(infos, BridgeInfo{
				Type: b.BridgeType(), SourceChain: route[0], DestChain: route[1],
				Status: "active", Tokens: tokens,
			})
		}
	}
	writeJSON(w, http.StatusOK, infos)
}

func (api *TradingAPI) handleArbScan(w http.ResponseWriter, r *http.Request) {
	if api.arbScanner == nil {
		writeError(w, http.StatusServiceUnavailable, "arbitrage scanner not configured")
		return
	}

	var req ArbScanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if len(req.Tokens) < 3 {
		writeError(w, http.StatusBadRequest, "need at least 3 tokens")
		return
	}
	if req.InputAmount == "" || !isPositiveDecimal(req.InputAmount) {
		writeError(w, http.StatusBadRequest, "inputAmount must be a positive integer")
		return
	}
	if req.MinProfitBPS <= 0 {
		req.MinProfitBPS = 10
	}

	start := time.Now()
	var allOpps []ArbOpportunity
	scannedPairs := 0

	// Triangular scan per chain.
	for _, chainID := range req.ChainIDs {
		opps, err := api.arbScanner.ScanTriangular(r.Context(), chainID, req.Tokens, req.InputAmount)
		if err != nil {
			continue
		}
		allOpps = append(allOpps, opps...)
		n := len(req.Tokens)
		scannedPairs += n * (n - 1) * (n - 2)
	}

	// Cross-chain scan for each pair.
	if len(req.ChainIDs) >= 2 && len(req.Tokens) >= 2 {
		for i := 0; i < len(req.Tokens); i++ {
			for j := i + 1; j < len(req.Tokens); j++ {
				opps, _ := api.arbScanner.ScanCrossChain(r.Context(), req.Tokens[i], req.Tokens[j], req.ChainIDs, req.InputAmount)
				allOpps = append(allOpps, opps...)
				scannedPairs += len(req.ChainIDs) * (len(req.ChainIDs) - 1) / 2
			}
		}
	}

	writeJSON(w, http.StatusOK, ArbScanResponse{
		Opportunities: allOpps, ScannedPairs: scannedPairs,
		ElapsedMs: time.Since(start).Milliseconds(),
	})
}

func validateCrossChainRequest(req CrossChainQuoteRequest) error {
	if !isValidAddress(req.TokenIn) {
		return fmt.Errorf("invalid tokenIn address")
	}
	if !isValidAddress(req.TokenOut) {
		return fmt.Errorf("invalid tokenOut address")
	}
	if !isPositiveDecimal(req.Amount) {
		return fmt.Errorf("amount must be a positive integer")
	}
	if req.SourceChain <= 0 {
		return fmt.Errorf("invalid sourceChain")
	}
	if req.DestChain <= 0 {
		return fmt.Errorf("invalid destChain")
	}
	return nil
}
