package gateway

import (
	"fmt"
	"math/big"
	"sort"
	"strings"
)

// Routing constants
const (
	DefaultMaxHops    = 3
	AbsoluteMaxHops   = 4
	MaxRoutesReturned = 5
)

// LXRouter precompile address (LP-9012)
const LXRouterPrecompile = "0x0000000000000000000000000000000000009012"

// --- Types ---

// RouteHop represents a single hop in a multihop route.
type RouteHop struct {
	TokenIn      string `json:"tokenIn"`
	TokenOut     string `json:"tokenOut"`
	PoolID       string `json:"poolId"`
	Fee          uint32 `json:"fee"`
	TickSpacing  int32  `json:"tickSpacing"`
	AmountIn     string `json:"amountIn"`
	AmountOut    string `json:"amountOut"`
	SqrtPriceX96 string `json:"sqrtPriceX96,omitempty"`
}

// MultihopRoute represents a complete multihop route with gas and impact estimates.
type MultihopRoute struct {
	Hops             []RouteHop `json:"hops"`
	TotalGasEstimate uint64     `json:"totalGasEstimate"`
	PriceImpact      float64    `json:"priceImpact"`
	TotalAmountIn    string     `json:"totalAmountIn"`
	TotalAmountOut   string     `json:"totalAmountOut"`
}

// RoutingRequest is the JSON body for POST /v1/route.
type RoutingRequest struct {
	TokenIn   string `json:"tokenIn"`
	TokenOut  string `json:"tokenOut"`
	Amount    string `json:"amount"`
	IsExactIn bool   `json:"isExactIn"`
	MaxHops   int    `json:"maxHops,omitempty"`
	ChainID   uint64 `json:"chainId"`
}

// RoutingResponse is returned from POST /v1/route.
type RoutingResponse struct {
	Routes  []MultihopRoute `json:"routes"`
	ChainID uint64          `json:"chainId"`
}

// PoolSummary is a lightweight pool descriptor for the GET /v1/pools listing.
// The full Pool type from types.go is used by the provider layer; this adds
// fields specific to the routing response.
type PoolSummary struct {
	PoolID      string `json:"poolId"`
	Token0      string `json:"token0"`
	Token1      string `json:"token1"`
	Fee         uint32 `json:"fee"`
	TickSpacing int32  `json:"tickSpacing"`
	TVL         string `json:"tvl,omitempty"`
	Volume24h   string `json:"volume24h,omitempty"`
	Liquidity   string `json:"liquidity,omitempty"`
	SqrtPrice   string `json:"sqrtPrice,omitempty"`
	Tick        int32  `json:"tick,omitempty"`
}

// --- Validation ---

func (r *RoutingRequest) Validate() error {
	if r.TokenIn == "" {
		return fmt.Errorf("tokenIn is required")
	}
	if !isValidHexAddress(r.TokenIn) {
		return fmt.Errorf("invalid tokenIn address")
	}
	if r.TokenOut == "" {
		return fmt.Errorf("tokenOut is required")
	}
	if !isValidHexAddress(r.TokenOut) {
		return fmt.Errorf("invalid tokenOut address")
	}
	if strings.EqualFold(r.TokenIn, r.TokenOut) {
		return fmt.Errorf("tokenIn and tokenOut must be different")
	}
	if r.Amount == "" {
		return fmt.Errorf("amount is required")
	}
	amt := parseBigIntStr(r.Amount)
	if amt == nil || amt.Sign() <= 0 {
		return fmt.Errorf("amount must be a positive integer")
	}
	if r.MaxHops < 0 {
		return fmt.Errorf("maxHops must be non-negative")
	}
	if r.MaxHops > AbsoluteMaxHops {
		return fmt.Errorf("maxHops must be at most %d", AbsoluteMaxHops)
	}
	return nil
}

// --- Token graph for route discovery ---

// tokenGraph is an adjacency list representing which tokens can be swapped
// through known pools. Each edge stores the pool metadata needed for routing.
type tokenGraph struct {
	// edges[tokenA][tokenB] = list of pools connecting A <-> B
	edges map[string]map[string][]poolEdge
}

// poolEdge represents a single pool connecting two tokens.
type poolEdge struct {
	poolID      string
	token0      string
	token1      string
	fee         uint32
	tickSpacing int32
	liquidity   *big.Int
}

func newTokenGraph() *tokenGraph {
	return &tokenGraph{
		edges: make(map[string]map[string][]poolEdge),
	}
}

// addPool adds a pool to the graph (bidirectional).
func (g *tokenGraph) addPool(pool poolEdge) {
	t0 := strings.ToLower(pool.token0)
	t1 := strings.ToLower(pool.token1)

	if g.edges[t0] == nil {
		g.edges[t0] = make(map[string][]poolEdge)
	}
	g.edges[t0][t1] = append(g.edges[t0][t1], pool)

	if g.edges[t1] == nil {
		g.edges[t1] = make(map[string][]poolEdge)
	}
	g.edges[t1][t0] = append(g.edges[t1][t0], pool)
}

// neighbors returns all tokens directly reachable from the given token.
func (g *tokenGraph) neighbors(token string) map[string][]poolEdge {
	return g.edges[strings.ToLower(token)]
}

// --- Route discovery via BFS ---

// candidateRoute is an intermediate structure during BFS.
type candidateRoute struct {
	path  []string   // token addresses visited
	edges []poolEdge // pool edges taken
}

// FindRoutes discovers all routes from tokenIn to tokenOut up to maxHops using BFS.
// Returns raw candidate routes; the caller simulates execution for each.
func (g *tokenGraph) FindRoutes(tokenIn, tokenOut string, maxHops int) []candidateRoute {
	if maxHops <= 0 {
		maxHops = DefaultMaxHops
	}
	if maxHops > AbsoluteMaxHops {
		maxHops = AbsoluteMaxHops
	}

	tokenInLower := strings.ToLower(tokenIn)
	tokenOutLower := strings.ToLower(tokenOut)

	var results []candidateRoute

	// BFS queue
	queue := []candidateRoute{{
		path:  []string{tokenInLower},
		edges: nil,
	}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		lastToken := current.path[len(current.path)-1]

		// Exceeded hop limit
		if len(current.edges) >= maxHops {
			continue
		}

		neighbors := g.neighbors(lastToken)
		for nextToken, pools := range neighbors {
			// Avoid cycles: don't revisit a token
			if containsToken(current.path, nextToken) {
				continue
			}

			for _, pool := range pools {
				newPath := make([]string, len(current.path), len(current.path)+1)
				copy(newPath, current.path)
				newPath = append(newPath, nextToken)

				newEdges := make([]poolEdge, len(current.edges), len(current.edges)+1)
				copy(newEdges, current.edges)
				newEdges = append(newEdges, pool)

				candidate := candidateRoute{path: newPath, edges: newEdges}

				if nextToken == tokenOutLower {
					results = append(results, candidate)
				} else if len(newEdges) < maxHops {
					queue = append(queue, candidate)
				}
			}
		}
	}

	return results
}

// containsToken checks if a token is already in the path.
func containsToken(path []string, token string) bool {
	for _, t := range path {
		if t == token {
			return true
		}
	}
	return false
}

// --- Route simulation ---

// SimulateRoute simulates execution of a candidate route given an input amount.
// Uses constant-product approximation for gas estimation. The actual V4 math
// runs in the DEX engine via the precompile -- the gateway never does V4 math.
//
// For each hop:
//   - amountOut = amountIn * (1 - fee/1e6) * 997/1000 (simplified model)
//   - Gas: base 150k + 50k per additional hop
func SimulateRoute(route candidateRoute, amountIn *big.Int, isExactIn bool) *MultihopRoute {
	if len(route.edges) == 0 || amountIn == nil || amountIn.Sign() <= 0 {
		return nil
	}

	hops := make([]RouteHop, len(route.edges))
	currentAmount := new(big.Int).Set(amountIn)

	for i, edge := range route.edges {
		// Determine tokenIn/tokenOut for this hop direction
		hopTokenIn := route.path[i]
		hopTokenOut := route.path[i+1]

		// Simplified output estimation:
		// amountOut ~= amountIn * (1_000_000 - fee) / 1_000_000
		// This is conservative; real V4 math runs on-chain.
		feeMultiplier := new(big.Int).SetUint64(uint64(1_000_000 - edge.fee))
		amountOut := new(big.Int).Mul(currentAmount, feeMultiplier)
		amountOut.Div(amountOut, big.NewInt(1_000_000))

		hops[i] = RouteHop{
			TokenIn:     hopTokenIn,
			TokenOut:    hopTokenOut,
			PoolID:      edge.poolID,
			Fee:         edge.fee,
			TickSpacing: edge.tickSpacing,
			AmountIn:    currentAmount.String(),
			AmountOut:   amountOut.String(),
		}

		currentAmount = amountOut
	}

	// Gas estimate: base + per-hop overhead
	gasEstimate := uint64(150_000) + uint64(len(route.edges)-1)*50_000

	// Price impact: rough estimate based on hop count and fees
	totalFee := float64(0)
	for _, edge := range route.edges {
		totalFee += float64(edge.fee) / 1_000_000
	}

	return &MultihopRoute{
		Hops:             hops,
		TotalGasEstimate: gasEstimate,
		PriceImpact:      totalFee,
		TotalAmountIn:    amountIn.String(),
		TotalAmountOut:   currentAmount.String(),
	}
}

// --- Top-level routing ---

// BuildRoutesFromPools builds a token graph from known pools and finds the
// best routes between two tokens.
func BuildRoutesFromPools(pools []Pool, req RoutingRequest) (*RoutingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	maxHops := req.MaxHops
	if maxHops == 0 {
		maxHops = DefaultMaxHops
	}

	graph := newTokenGraph()
	for _, pool := range pools {
		ts := int32(pool.Fee)
		if mapped, ok := feeToTS[uint32(pool.Fee)]; ok {
			ts = mapped
		}

		liq := big.NewInt(0)
		if pool.TVL != nil {
			liq = pool.TVL
		}

		graph.addPool(poolEdge{
			poolID:      pool.Address,
			token0:      pool.Token0.Address,
			token1:      pool.Token1.Address,
			fee:         uint32(pool.Fee),
			tickSpacing: ts,
			liquidity:   liq,
		})
	}

	candidates := graph.FindRoutes(req.TokenIn, req.TokenOut, maxHops)
	if len(candidates) == 0 {
		return &RoutingResponse{
			Routes:  []MultihopRoute{},
			ChainID: req.ChainID,
		}, nil
	}

	amount := parseBigIntStr(req.Amount)
	var routes []MultihopRoute
	for _, c := range candidates {
		route := SimulateRoute(c, amount, req.IsExactIn)
		if route != nil {
			routes = append(routes, *route)
		}
	}

	// Sort by output amount descending (best route first)
	sort.Slice(routes, func(i, j int) bool {
		outI := parseBigIntStr(routes[i].TotalAmountOut)
		outJ := parseBigIntStr(routes[j].TotalAmountOut)
		if outI == nil || outJ == nil {
			return false
		}
		return outI.Cmp(outJ) > 0
	})

	// Limit to top N
	if len(routes) > MaxRoutesReturned {
		routes = routes[:MaxRoutesReturned]
	}

	return &RoutingResponse{
		Routes:  routes,
		ChainID: req.ChainID,
	}, nil
}
