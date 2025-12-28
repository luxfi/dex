package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
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

// NewServer creates a new gateway server
func NewServer(router *Router, cfg ServerConfig) *Server {
	mux := http.NewServeMux()
	
	s := &Server{
		router: router,
		mux:    mux,
		httpServer: &http.Server{
			Addr:           cfg.Addr,
			Handler:        mux,
			ReadTimeout:    cfg.ReadTimeout,
			WriteTimeout:   cfg.WriteTimeout,
			MaxHeaderBytes: cfg.MaxHeaderBytes,
		},
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
	// Health and info
	s.mux.HandleFunc("/health", s.handleHealth)
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
	
	// Conversion tracking API
	s.mux.HandleFunc("/v1/leads", s.handleLeads)
	s.mux.HandleFunc("/v1/events", s.handleEvents)
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
	
	healthy := true
	for _, check := range checks {
		if !check.Healthy {
			healthy = false
			break
		}
	}
	
	status := http.StatusOK
	if !healthy {
		status = http.StatusServiceUnavailable
	}
	
	s.writeJSON(w, status, map[string]interface{}{
		"status":    map[bool]string{true: "healthy", false: "unhealthy"}[healthy],
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
	quote, err := s.router.GetBestQuote(ctx, s.convertQuoteRequest(req))
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}
	
	s.writeJSON(w, http.StatusOK, quote)
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
	quotes, err := s.router.GetAllQuotes(ctx, s.convertQuoteRequest(req))
	if err != nil {
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
	
	// TODO: Implement swap transaction building
	s.writeError(w, http.StatusNotImplemented, fmt.Errorf("swap building not yet implemented"))
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
