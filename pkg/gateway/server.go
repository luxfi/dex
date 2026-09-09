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

// The whole surface lives under one prefix, and this is it.
//
// api.lux.cloud carries the chain at /v1/chain, the C-chain at the bare root
// and several services besides, so /v1/quote is a generic name claimed on a
// shared host. Worse, the same operations were answering twice: once here and
// once at /trading/* in Uniswap's shapes, while a published api/openapi.yaml
// described a third set of paths no binary served. Three contracts, one API.
//
// One namespace, one owner, one document.
const tradePrefix = "/v1/trade"

// tail is what a subtree route was asked about — the pool id, the order id.
// It reads the same constant the route was mounted from, because a route
// registered from the prefix and parsed from a copy of it is one rename away
// from every id arriving with a path stuck to the front of it.
func tail(r *http.Request, route string) string {
	return strings.TrimPrefix(r.URL.Path, tradePrefix+route)
}

// routes is the surface, once. api/openapi.yaml describes exactly this, and a
// test holds the two to each other — a route added here and not written down
// is a route no client can be told about, and a path written down and not
// served is a promise the binary breaks.
//
// Paths ending in "/" are subtrees: what follows is read with tail.
func (s *Server) routes() map[string]http.HandlerFunc {
	return map[string]http.HandlerFunc{
		// What this deployment quotes, and from where.
		"/venues": s.handleVenues,

		// Quotes and swaps
		"/quote":  s.handleQuote,
		"/quotes": s.handleQuotes,
		"/swap":   s.handleSwap,
		"/route":  s.handleRoute,

		// Orders — limit and dutch auction
		"/order":  s.handleOrder,
		"/order/": s.handleOrderByID,

		// Allowances
		"/approval/check": s.handleApprovalCheck,
		"/approval/build": s.handleApprovalBuild,
		"/permit2/check":  s.handlePermit2Check,
		"/permit2/build":  s.handlePermit2Build,

		// Pools and positions
		"/pools":             s.handlePools,
		"/pool/":             s.handlePool,
		"/positions":         s.handlePositions,
		"/position":          s.handlePosition,
		"/position/increase": s.handlePositionIncrease,
		"/position/decrease": s.handlePositionDecrease,
		"/position/claim":    s.handlePositionClaim,

		// Reference
		"/tokens": s.handleTokens,
		"/price":  s.handlePrice,
		"/prices": s.handlePrices,
		"/stats":  s.handleStats,
	}
}

// registerRoutes registers all HTTP routes
func (s *Server) registerRoutes() {
	// The pod's own questions, and no part of the trading contract: probes,
	// answered in-cluster, never routed at the edge.
	s.mux.HandleFunc("/healthz", s.handleHealth)
	s.mux.HandleFunc("/providers", s.handleProviders)

	for path, h := range s.routes() {
		s.mux.HandleFunc(tradePrefix+path, h)
	}
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
	path := tail(r, "/pool/")
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
