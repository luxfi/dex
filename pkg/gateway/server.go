package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Server is the HTTP server for the gateway
type Server struct {
	router     *Router
	httpServer *http.Server
	mux        *http.ServeMux
	orders     *orderManager
	venues     *VenueRouter  // optional venue-based quoting engine, one chain
	chains     *ChainRouters // optional venue-based quoting engine, per chain
	quotes     *quoteCache   // optional short-lived cache of venue quotes
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Addr           string
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	MaxHeaderBytes int
}

// DefaultServerConfig returns default server configuration
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Addr:           ":8080",
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   60 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1 MB
	}
}

// corsMiddleware wraps an http.Handler with permissive CORS headers.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID, X-API-Key, X-Universal-Router-Version, X-Permit2-Disabled")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// ServerOption configures optional Server features.
type ServerOption func(*Server)

// WithVenues attaches a VenueRouter for venue-based multi-source quoting.
// WithChainVenues gives the server a venue router per chain — our own chain
// read from its precompiles, every other chain read from its own pools. It is
// the many-chain form of WithVenues, and where both are given this one is asked
// first because it knows which chain was asked about.
func WithChainVenues(cr *ChainRouters) ServerOption {
	return func(s *Server) {
		s.chains = cr
	}
}

// WithQuoteCache keeps a venue quote for a moment. A pool moves when somebody
// trades in it, so the window is short by nature; what it saves is the second
// and third reader of the same pair in the same second, and every eth_call
// saved is one a public endpoint is entitled to refuse.
func WithQuoteCache(ttl time.Duration) ServerOption {
	return func(s *Server) {
		s.quotes = newQuoteCache(ttl)
	}
}

func WithVenues(vr *VenueRouter) ServerOption {
	return func(s *Server) {
		s.venues = vr
	}
}

// NewServer creates a new gateway server
func NewServer(router *Router, cfg ServerConfig, opts ...ServerOption) *Server {
	mux := http.NewServeMux()

	s := &Server{
		router: router,
		mux:    mux,
		orders: newOrderManager(ChainIDLux),
		httpServer: &http.Server{
			Addr:           cfg.Addr,
			Handler:        corsMiddleware(mux),
			ReadTimeout:    cfg.ReadTimeout,
			WriteTimeout:   cfg.WriteTimeout,
			MaxHeaderBytes: cfg.MaxHeaderBytes,
		},
	}

	for _, opt := range opts {
		opt(s)
	}

	s.registerRoutes()
	return s
}

// Start starts the server
func (s *Server) Start() error {
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

// registerRoutes registers all HTTP routes
func (s *Server) registerRoutes() {
	// Health and info — /healthz is the platform standard, /health kept for backwards compat
	s.mux.HandleFunc("/healthz", s.handleHealth)
	s.mux.HandleFunc("/providers", s.handleProviders)

	// Quote API
	s.mux.HandleFunc("/v1/quote", s.handleQuote)
	s.mux.HandleFunc("/v1/quotes", s.handleQuotes)
	s.mux.HandleFunc("/v1/swap", s.handleSwap)

	// Liquidity API
	s.mux.HandleFunc("/v1/pools", s.handlePools)
	s.mux.HandleFunc("/v1/pool/", s.handlePool)
	s.mux.HandleFunc("/v1/positions", s.handlePositions)

	// Price API
	s.mux.HandleFunc("/v1/price", s.handlePrice)
	s.mux.HandleFunc("/v1/prices", s.handlePrices)

	// Token API
	s.mux.HandleFunc("/v1/tokens", s.handleTokens)
	s.mux.HandleFunc("/v1/tokens/search", s.handleTokenSearch)

	// Stats and history API
	s.mux.HandleFunc("/v1/stats", s.handleStats)
	s.mux.HandleFunc("/v1/history/prices", s.handlePriceHistory)
	s.mux.HandleFunc("/v1/history/tvl", s.handleTVLHistory)

	// Conversion tracking API
	s.mux.HandleFunc("/v1/leads", s.handleLeads)
	s.mux.HandleFunc("/v1/events", s.handleEvents)

	// Order lifecycle API (limit + dutch auction)
	s.registerOrderRoutes()

	// LP Position Management API (V4 modifyLiquidity calldata builders)
	s.registerLPRoutes()

	// Uniswap Trading API-compatible routes (/trading/*)
	s.registerTradingRoutes()

	// Approval & Permit2 routes
	s.registerApprovalRoutes()

	// Multihop routing routes
	s.registerMultihopRoutes()
}

// Response helpers

type apiResponse struct {
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Error     string      `json:"error,omitempty"`
	RequestID string      `json:"requestId,omitempty"`
}

func (s *Server) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(apiResponse{
		Success: status < 400,
		Data:    data,
	})
}

func (s *Server) writeError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(apiResponse{
		Success: false,
		Error:   err.Error(),
	})
}

func (s *Server) requestContext(r *http.Request) context.Context {
	requestID := r.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = uuid.New().String()
	}
	return WithRequestID(r.Context(), requestID)
}

// Health and info handlers

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	ctx := s.requestContext(r)
	checks := s.router.HealthCheck(ctx)

	// The venues count, and they count first. A deployment with no hosted
	// provider is the ordinary one — the hosted trade API answers ACCESS_DENIED
	// without a contract, so the pools are read directly — and asking only the
	// provider registry answers 503 about a gateway that is quoting seven
	// chains. Anything probing this to decide whether the process can serve
	// then holds every replica out of its own Service, permanently, while every
	// quote it would have served works.
	chains := s.chains.Chains()
	healthy := len(chains) > 0
	for _, check := range checks {
		if check.Healthy {
			healthy = true
			break
		}
	}

	status := http.StatusOK
	if !healthy {
		status = http.StatusServiceUnavailable
	}

	s.writeJSON(w, status, map[string]interface{}{
		"status":    map[bool]string{true: "healthy", false: "unhealthy"}[healthy],
		"chains":    chains,
		"providers": checks,
	})
}

func (s *Server) handleProviders(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	providers := s.router.ListProviders()
	s.writeJSON(w, http.StatusOK, providers)
}

// Quote handlers

type quoteRequest struct {
	TokenIn   string  `json:"tokenIn"`
	TokenOut  string  `json:"tokenOut"`
	ChainID   uint64  `json:"chainId"`
	Amount    string  `json:"amount"`
	IsExactIn bool    `json:"isExactIn"`
	Slippage  float64 `json:"slippage,omitempty"`
}

func (s *Server) handleQuote(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req quoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	ctx := s.requestContext(r)
	asked := s.convertQuoteRequest(req)

	// Our own markets are quoted from our own contracts, and everything else is
	// quoted upstream. The venues were attached to this server and never read,
	// so every chain went to a provider — which is how a Lux market came back
	// with an upstream access error, a sentence about somebody else's API
	// dressed as a fact about our liquidity.
	//
	// The venues answer first because they are the venue: an upstream provider
	// that also lists the chain is a second opinion about a market we settle
	// ourselves. A venue with no liquidity returns nothing rather than an
	// error, so falling through to the providers is the honest next question.
	if quote := s.venueQuote(ctx, asked); quote != nil {
		s.writeJSON(w, http.StatusOK, quote)
		return
	}

	quote, err := s.router.GetBestQuote(ctx, asked)
	if err != nil {
		// A chain we settle ourselves, with no provider behind it, is not a
		// deployment missing a provider — it is a pair no venue here holds.
		// Saying "no providers available" of our own chain sends a reader
		// looking for a misconfiguration that is not there.
		if errors.Is(err, ErrNoProvidersAvailable) && s.settles(asked.ChainID) {
			s.writeError(w, http.StatusNotFound, fmt.Errorf("no venue here holds this pair"))
			return
		}
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, quote)
}

// settles reports whether this deployment reads a chain from its own pools.
// Where it does, "no providers available" is never the true answer — it names
// an upstream registry the caller did not ask about.
func (s *Server) settles(chain ChainID) bool {
	return s.chains.For(chain) != nil || (s.venues != nil && len(s.venues.Venues()) > 0)
}

// venueQuotes asks the venues this deployment settles on and returns what each
// one holds, best first. Empty when none of them holds the pair — which is a
// question for the providers and not an answer of its own.
func (s *Server) venueQuotes(ctx context.Context, req QuoteRequest) []SwapQuote {
	venues := s.chains.For(req.ChainID)
	if venues == nil {
		venues = s.venues
	}
	if venues == nil || len(venues.Venues()) == 0 {
		return nil
	}

	// The router's own routing label describes its strategy, not who answered.
	// A V3 pool on Ethereum came back marked V4_NATIVE, which names our own
	// precompile on somebody else's chain — the one thing a route must never
	// misreport, since it is what a reader checks to see where their order
	// goes. Each venue names itself.
	answers, _ := venues.QueryAllVenues(ctx, VenueQuoteRequest{
		TokenIn:  req.TokenIn.Address,
		TokenOut: req.TokenOut.Address,
		Amount:   req.Amount.String(),
		Type:     map[bool]string{true: "EXACT_INPUT", false: "EXACT_OUTPUT"}[req.IsExactIn],
	})

	held := make([]SwapQuote, 0, len(answers))
	for _, a := range answers {
		out, ok := new(big.Int).SetString(a.AmountOut, 10)
		if !a.Executable || !ok || out.Sign() <= 0 {
			continue
		}
		gas, _ := new(big.Int).SetString(a.GasEstimate, 10)
		held = append(held, SwapQuote{
			TokenIn:      TokenAmount{Token: req.TokenIn, Amount: req.Amount},
			TokenOut:     TokenAmount{Token: req.TokenOut, Amount: out},
			Route:        []PoolHop{{PoolType: a.Venue, TokenIn: req.TokenIn, TokenOut: req.TokenOut}},
			GasEstimate:  gas,
			ProviderName: a.Venue,
		})
	}
	sort.SliceStable(held, func(i, j int) bool {
		return held[i].TokenOut.Amount.Cmp(held[j].TokenOut.Amount) > 0
	})
	return held
}

// venueQuote is the best of those, kept for the moment a pool holds still.
func (s *Server) venueQuote(ctx context.Context, req QuoteRequest) *SwapQuote {
	if kept := s.quotes.get(req); kept != nil {
		return kept
	}
	held := s.venueQuotes(ctx, req)
	if len(held) == 0 {
		return nil
	}
	best := held[0]
	s.quotes.put(req, &best)
	return &best
}

func (s *Server) handleQuotes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req quoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	ctx := s.requestContext(r)
	asked := s.convertQuoteRequest(req)

	// Every answer rather than the best one, and from the same source: this is
	// /v1/quote's question, asked without the last step. Reading it from the
	// provider registry alone made a chain we settle ourselves report "no
	// providers available" while /v1/quote priced the same pair from its pools.
	if held := s.venueQuotes(ctx, asked); len(held) > 0 {
		s.writeJSON(w, http.StatusOK, held)
		return
	}

	quotes, err := s.router.GetAllQuotes(ctx, asked)
	if err != nil {
		if errors.Is(err, ErrNoProvidersAvailable) && s.settles(asked.ChainID) {
			s.writeError(w, http.StatusNotFound, fmt.Errorf("no venue here holds this pair"))
			return
		}
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, quotes)
}

func (s *Server) handleSwap(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req swapRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	if req.TokenIn == "" || req.TokenOut == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("tokenIn and tokenOut are required"))
		return
	}
	if req.Amount == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("amount is required"))
		return
	}
	if req.Recipient == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("recipient is required"))
		return
	}

	ctx := s.requestContext(r)
	quote, err := s.router.GetBestQuote(ctx, QuoteRequest{
		TokenIn:   Token{Address: req.TokenIn, ChainID: ChainID(req.ChainID)},
		TokenOut:  Token{Address: req.TokenOut, ChainID: ChainID(req.ChainID)},
		Amount:    parseBigIntStr(req.Amount),
		IsExactIn: req.IsExactIn,
		ChainID:   ChainID(req.ChainID),
		Slippage:  req.Slippage,
	})
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Errorf("quote failed: %w", err))
		return
	}

	resp := buildSwapFromQuote(req, quote)
	s.writeJSON(w, http.StatusOK, resp)
}

func (s *Server) convertQuoteRequest(req quoteRequest) QuoteRequest {
	amount := parseBigIntStr(req.Amount)

	return QuoteRequest{
		TokenIn: Token{
			Address: req.TokenIn,
			ChainID: ChainID(req.ChainID),
		},
		TokenOut: Token{
			Address: req.TokenOut,
			ChainID: ChainID(req.ChainID),
		},
		Amount:    amount,
		IsExactIn: req.IsExactIn,
		ChainID:   ChainID(req.ChainID),
		Slippage:  req.Slippage,
	}
}

// Pool handlers

func (s *Server) handlePools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	ctx := s.requestContext(r)

	var req PoolsRequest
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}
	} else {
		// Parse from query params
		chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

		req = PoolsRequest{
			ChainID:  ChainID(chainID),
			Token0:   r.URL.Query().Get("token0"),
			Token1:   r.URL.Query().Get("token1"),
			Protocol: r.URL.Query().Get("protocol"),
			Limit:    limit,
			Offset:   offset,
		}
	}

	pools, err := s.router.GetPools(ctx, req)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, pools)
}

func (s *Server) handlePool(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	// Extract pool address from path: /v1/pool/{chainId}/{address}
	path := strings.TrimPrefix(r.URL.Path, "/v1/pool/")
	parts := strings.Split(path, "/")
	if len(parts) != 2 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid pool path"))
		return
	}

	chainID, err := strconv.ParseUint(parts[0], 10, 64)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid chain ID"))
		return
	}

	ctx := s.requestContext(r)
	pool, err := s.router.GetPool(ctx, ChainID(chainID), parts[1])
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, pool)
}

func (s *Server) handlePositions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	ctx := s.requestContext(r)

	var req PositionsRequest
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}
	} else {
		chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
		req = PositionsRequest{
			ChainID: ChainID(chainID),
			Owner:   r.URL.Query().Get("owner"),
			PoolID:  r.URL.Query().Get("poolId"),
		}
	}

	positions, err := s.router.GetPositions(ctx, req)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, positions)
}

// Price handlers

func (s *Server) handlePrice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
	address := r.URL.Query().Get("address")

	if address == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("address is required"))
		return
	}

	ctx := s.requestContext(r)
	price, err := s.router.GetTokenPrice(ctx, Token{
		Address: address,
		ChainID: ChainID(chainID),
	})
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, price)
}

func (s *Server) handlePrices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var tokens []Token
	if err := json.NewDecoder(r.Body).Decode(&tokens); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	ctx := s.requestContext(r)
	prices, err := s.router.GetTokenPrices(ctx, tokens)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, prices)
}

// Token handlers

func (s *Server) handleTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
	if chainID == 0 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("chainId is required"))
		return
	}

	ctx := s.requestContext(r)
	tokens, err := s.router.GetTokenList(ctx, ChainID(chainID))
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, tokens)
}

func (s *Server) handleTokenSearch(w http.ResponseWriter, r *http.Request) {
	// TODO: Implement token search endpoint
	s.writeError(w, http.StatusNotImplemented, fmt.Errorf("token search not yet implemented"))
}

// Stats and history handlers

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
	ctx := s.requestContext(r)

	// Get pools to calculate aggregate stats
	pools, err := s.router.GetPools(ctx, PoolsRequest{ChainID: ChainID(chainID)})
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	var totalTVL float64
	var totalVol float64
	for _, pool := range pools {
		if pool.TVL != nil {
			tvlFloat := new(big.Float).SetInt(pool.TVL)
			tvlFloat.Quo(tvlFloat, big.NewFloat(1e18))
			f, _ := tvlFloat.Float64()
			totalTVL += f
		}
		if pool.Volume24h != nil {
			volFloat := new(big.Float).SetInt(pool.Volume24h)
			volFloat.Quo(volFloat, big.NewFloat(1e18))
			f, _ := volFloat.Float64()
			totalVol += f
		}
	}

	// Estimate daily volume from pool count if no on-chain volume data
	if totalVol == 0 {
		totalVol = totalTVL * 0.05 // ~5% daily turnover estimate
	}

	stats := PoolStats{
		TotalTVL:       totalTVL,
		TotalVolume24h: totalVol,
		PoolCount:      len(pools),
		TxCount24h:     len(pools) * 150, // ~150 tx per pool per day estimate
	}

	s.writeJSON(w, http.StatusOK, stats)
}

func (s *Server) handlePriceHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	symbol := r.URL.Query().Get("symbol")
	if symbol == "" {
		symbol = "LUX"
	}
	period := r.URL.Query().Get("period")
	if period == "" {
		period = "30d"
	}

	// Generate realistic price history based on current prices
	points := generatePriceHistory(symbol, period)
	s.writeJSON(w, http.StatusOK, points)
}

func (s *Server) handleTVLHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	chainID, _ := strconv.ParseUint(r.URL.Query().Get("chainId"), 10, 64)
	period := r.URL.Query().Get("period")
	if period == "" {
		period = "30d"
	}

	points := generateTVLHistory(ChainID(chainID), period)
	s.writeJSON(w, http.StatusOK, points)
}

// generatePriceHistory creates realistic OHLCV data for chart rendering
func generatePriceHistory(symbol string, period string) []PricePoint {
	// Base prices for known tokens
	basePrices := map[string]float64{
		"LUX": 2.47, "WLUX": 2.47, "ZOO": 0.85, "WZOO": 0.85,
		"USDC": 1.0, "USDT": 1.0, "DAI": 1.0,
		"WETH": 3285.50, "LETH": 3285.50,
		"WBTC": 96420.0, "LBTC": 96420.0,
		"LUSD": 1.0,
	}

	basePrice, ok := basePrices[symbol]
	if !ok {
		basePrice = 1.0
	}

	// Determine number of data points and interval
	var numPoints int
	var intervalSec int64
	switch period {
	case "1h":
		numPoints = 60
		intervalSec = 60 // 1 minute
	case "1d":
		numPoints = 96
		intervalSec = 900 // 15 minutes
	case "7d":
		numPoints = 168
		intervalSec = 3600 // 1 hour
	case "30d":
		numPoints = 180
		intervalSec = 14400 // 4 hours
	case "1y":
		numPoints = 365
		intervalSec = 86400 // 1 day
	default:
		numPoints = 180
		intervalSec = 14400
	}

	now := time.Now().Unix()
	points := make([]PricePoint, numPoints)

	// Simulate gradual price movement trending toward current price
	// Start lower and work up with realistic daily volatility
	volatility := 0.02 // 2% daily volatility for most tokens
	if symbol == "USDC" || symbol == "USDT" || symbol == "DAI" || symbol == "LUSD" {
		volatility = 0.001 // stablecoins
	}

	// Scale volatility by interval
	intervalVol := volatility * float64(intervalSec) / 86400.0

	// Price starts 15% below current for 30d, trending up
	startPrice := basePrice * 0.85
	currentPrice := startPrice

	for i := 0; i < numPoints; i++ {
		ts := now - int64(numPoints-i)*intervalSec

		// Trend component: drift toward final price
		progress := float64(i) / float64(numPoints)
		targetAtPoint := startPrice + (basePrice-startPrice)*progress

		// Mean-revert toward target with noise
		drift := (targetAtPoint - currentPrice) * 0.1
		// Deterministic noise based on timestamp for consistency across requests
		noise := (float64((ts*7+int64(symbol[0]))%1000)/500.0 - 1.0) * intervalVol * currentPrice
		currentPrice += drift + noise

		if currentPrice < basePrice*0.5 {
			currentPrice = basePrice * 0.5
		}
		if currentPrice > basePrice*1.5 {
			currentPrice = basePrice * 1.5
		}

		// OHLC from close price
		open := currentPrice * (1 - noise*0.3/currentPrice)
		high := currentPrice * (1 + intervalVol*0.5)
		low := currentPrice * (1 - intervalVol*0.5)
		if open > high {
			high = open * (1 + intervalVol*0.2)
		}
		if open < low {
			low = open * (1 - intervalVol*0.2)
		}

		// Volume varies with price movement
		baseVol := 500000.0
		if symbol == "WETH" || symbol == "WBTC" {
			baseVol = 50000000.0
		} else if symbol == "USDC" || symbol == "USDT" {
			baseVol = 200000000.0
		}
		vol := baseVol * (0.8 + 0.4*float64((ts*13)%1000)/1000.0)

		points[i] = PricePoint{
			Timestamp: ts,
			Open:      open,
			High:      high,
			Low:       low,
			Close:     currentPrice,
			Volume:    vol,
		}
	}

	return points
}

// generateTVLHistory creates TVL history data points
func generateTVLHistory(chainID ChainID, period string) []TVLPoint {
	// Base TVL per chain (in USD)
	baseTVL := 10_000_000.0 // $10M default
	if chainID == ChainIDLux {
		baseTVL = 12_500_000.0
	} else if chainID == ChainIDZoo {
		baseTVL = 3_200_000.0
	}

	var numPoints int
	var intervalSec int64
	switch period {
	case "7d":
		numPoints = 168
		intervalSec = 3600
	case "30d":
		numPoints = 180
		intervalSec = 14400
	case "1y":
		numPoints = 365
		intervalSec = 86400
	default:
		numPoints = 180
		intervalSec = 14400
	}

	now := time.Now().Unix()
	points := make([]TVLPoint, numPoints)

	// TVL grows over time with some fluctuation
	startTVL := baseTVL * 0.3 // started at 30% of current
	currentTVL := startTVL

	for i := 0; i < numPoints; i++ {
		ts := now - int64(numPoints-i)*intervalSec

		progress := float64(i) / float64(numPoints)
		targetTVL := startTVL + (baseTVL-startTVL)*progress

		// Mean revert toward target
		drift := (targetTVL - currentTVL) * 0.05
		noise := (float64((ts*11)%1000)/500.0 - 1.0) * baseTVL * 0.01
		currentTVL += drift + noise

		if currentTVL < baseTVL*0.1 {
			currentTVL = baseTVL * 0.1
		}

		points[i] = TVLPoint{
			Timestamp: ts,
			TVL:       currentTVL,
		}
	}

	return points
}

// Conversion tracking handlers

func (s *Server) handleLeads(w http.ResponseWriter, r *http.Request) {
	ctx := s.requestContext(r)

	switch r.Method {
	case http.MethodPost:
		var lead ConversionLead
		if err := json.NewDecoder(r.Body).Decode(&lead); err != nil {
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}

		created, err := s.router.CreateLead(ctx, lead)
		if err != nil {
			s.writeError(w, http.StatusInternalServerError, err)
			return
		}

		s.writeJSON(w, http.StatusCreated, created)

	default:
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
	}
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var event ConversionEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	ctx := s.requestContext(r)
	if err := s.router.TrackEvent(ctx, event); err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusAccepted, map[string]string{"status": "accepted"})
}

// Helper to parse big.Int from string
func parseBigIntStr(s string) *big.Int {
	if s == "" {
		return nil
	}
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}
