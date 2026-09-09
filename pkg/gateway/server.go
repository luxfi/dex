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
	"time"

	"github.com/google/uuid"
)

// Server is the HTTP server for the gateway
type Server struct {
	router     *Router
	httpServer *http.Server
	mux        *http.ServeMux
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
// Every path here is an unauthenticated read of public pools. No cookie, no
// bearer, no credential of any kind rides these requests, which is why the
// origin is `*`: there is nothing one origin could be trusted with that another
// could not, and naming one would break the docs site, a wallet and any partner
// reading a price, each silently.
//
// The header list is the headers this surface reads. It used to also permit
// Authorization, X-API-Key, X-Universal-Router-Version and X-Permit2-Disabled —
// the last two from the Uniswap-shaped /trading/* compat routes, which are
// gone, and the first two from an authentication this has never had. A
// permitted header nobody sends is an invitation to send it.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
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

		// A price, and the transaction that takes it.
		"/quote":  s.handleQuote,
		"/quotes": s.handleQuotes,
		"/swap":   s.handleSwap,

		// What a spender may already move.
		"/approval/check": s.handleApprovalCheck,
		"/approval/build": s.handleApprovalBuild,
		"/permit2/check":  s.handlePermit2Check,
		"/permit2/build":  s.handlePermit2Build,
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
			s.writeError(w, http.StatusNotFound, s.whyNothingHeld(asked))
			return
		}
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.writeJSON(w, http.StatusOK, quote)
}

// whyNothingHeld says what was asked as well as that nothing answered.
//
// The V2 and V3 arms read the quoter each protocol published, and those price a
// known INPUT only — `getAmountsOut` and `quoteExactInputSingle` have no other
// direction. So an exact-output request gets nothing back from them however
// deep the pool is, and answering it with "no venue here holds this pair" sends
// someone looking for liquidity that is sitting right there. The direction is
// named as the likely cause, not asserted as the fact, because a deployment
// carrying an arm that prices both ways would reach this line only when the
// pair really is unheld.
func (s *Server) whyNothingHeld(asked QuoteRequest) error {
	if !asked.IsExactIn {
		return fmt.Errorf("nothing here answered for a known output — the v2 and v3 arms read quoters that price a known input only; ask with isExactIn and the amount of %s you will spend", asked.TokenIn.Address)
	}
	return fmt.Errorf("no venue here holds this pair")
}

// arms is the venues that read one chain: the per-chain set where there is
// one, and otherwise the single set a one-chain deployment was handed.
//
// Every path that needs a venue asks this. The swap used to look only at the
// per-chain set, so a deployment configured the other way could price a pair
// and then report the venue that had just priced it as unregistered.
func (s *Server) arms(chain ChainID) *VenueRouter {
	if v := s.chains.For(chain); v != nil {
		return v
	}
	return s.venues
}

// arm is one of them, by the name it calls itself.
func (s *Server) arm(chain ChainID, name string) Venue {
	venues := s.arms(chain)
	if venues == nil {
		return nil
	}
	for _, v := range venues.Venues() {
		if v.Name() == name {
			return v
		}
	}
	return nil
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
	venues := s.arms(req.ChainID)
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
		// The tier rides on the hop it belongs to. A V3 quote is a reading of
		// ONE pool, and a swap that does not name the same tier executes
		// against a different one at a different price.
		fee, _ := strconv.Atoi(a.Fee)
		held = append(held, SwapQuote{
			TokenIn:      TokenAmount{Token: req.TokenIn, Amount: req.Amount},
			TokenOut:     TokenAmount{Token: req.TokenOut, Amount: out},
			Route:        []PoolHop{{PoolType: a.Venue, TokenIn: req.TokenIn, TokenOut: req.TokenOut, Fee: fee}},
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
			s.writeError(w, http.StatusNotFound, s.whyNothingHeld(asked))
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
	if req.Recipient == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("recipient is required"))
		return
	}
	amount, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amount.Sign() <= 0 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("amount %q is not a whole number of the token's smallest unit", req.Amount))
		return
	}

	// The price and the transaction come from ONE venue. Asking the provider
	// registry for the price and then encoding it for a fixed precompile —
	// which is what this did — quotes a router on one chain and addresses a
	// contract on another, in one answer.
	//
	// The same conversion /v1/trade/quote uses, on the same fields, so a swap
	// can never be priced by a question shaped differently from the one a
	// caller could have asked directly.
	ctx := s.requestContext(r)
	asked := s.convertQuoteRequest(req.quoteRequest)
	held := s.venueQuotes(ctx, asked)
	if len(held) == 0 {
		s.writeError(w, http.StatusNotFound, s.whyNothingHeld(asked))
		return
	}
	best := held[0]
	if len(best.Route) == 0 {
		s.writeError(w, http.StatusInternalServerError, fmt.Errorf("%s priced this pair without naming the pool", best.ProviderName))
		return
	}

	venue := s.arm(asked.ChainID, best.ProviderName)
	if venue == nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Errorf("venue %q priced this and is not registered", best.ProviderName))
		return
	}

	tx, err := venue.Swap(SwapOrder{
		TokenIn:   req.TokenIn,
		TokenOut:  req.TokenOut,
		AmountIn:  amount,
		MinOut:    leastAccepted(best.TokenOut.Amount, req.Slippage),
		Recipient: req.Recipient,
		Deadline:  req.Deadline,
		Fee:       uint32(best.Route[0].Fee),
	})
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	if tx == nil {
		// A venue that prices a market it does not settle. Naming it is the
		// difference between "we cannot" and "there is no price".
		s.writeError(w, http.StatusNotFound, fmt.Errorf("%s priced this pair and cannot build the transaction for it", best.ProviderName))
		return
	}
	tx.ChainID = req.ChainID

	s.writeJSON(w, http.StatusOK, swapResponse{Swap: *tx, Quote: best})
}

// leastAccepted is the amountOutMinimum for a quote taken at a tolerance.
//
// A router given the quoted amount as its floor reverts on any movement at
// all, including the movement the caller's own trade causes, so a tolerance is
// not a nicety. Zero or nonsense means half a percent — what a wallet offers by
// default — because a swap submitted with no floor is one anybody can stand in
// front of.
//
// The arithmetic is in hundredths of a basis point, which is the unit a V3 fee
// tier is already stated in. In whole basis points a tolerance finer than 0.01%
// truncated to nothing, so 0.005 produced a floor EQUAL to the quote and the
// swap reverted on the first wei of movement — a request for very tight
// protection answered with none at all.
func leastAccepted(quoted *big.Int, tolerancePercent float64) *big.Int {
	if quoted == nil || quoted.Sign() <= 0 {
		return big.NewInt(0)
	}
	if tolerancePercent <= 0 || tolerancePercent >= 100 {
		tolerancePercent = 0.5
	}
	keep := big.NewInt(1_000_000 - int64(tolerancePercent*10_000))
	least := new(big.Int).Mul(quoted, keep)
	return least.Div(least, big.NewInt(1_000_000))
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

// Price handlers

// Token handlers

// Stats and history handlers

// Helper to parse big.Int from string
func parseBigIntStr(s string) *big.Int {
	if s == "" {
		return nil
	}
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}
