package gateway

import (
	"bytes"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ==========================================================================
// Token Graph Tests
// ==========================================================================

func TestTokenGraphBasic(t *testing.T) {
	g := newTokenGraph()
	g.addPool(poolEdge{
		poolID: "pool-AB",
		token0: "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		token1: "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
		fee:    3000,
	})

	neighbors := g.neighbors("0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
	if len(neighbors) != 1 {
		t.Fatalf("expected 1 neighbor, got %d", len(neighbors))
	}

	// Bidirectional
	neighbors2 := g.neighbors("0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
	if len(neighbors2) != 1 {
		t.Fatalf("expected 1 reverse neighbor, got %d", len(neighbors2))
	}
}

// ==========================================================================
// Route Finding Tests
// ==========================================================================

func TestFindRoutes_SingleHop(t *testing.T) {
	g := newTokenGraph()
	g.addPool(poolEdge{
		poolID: "pool-AB",
		token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		fee:    3000,
	})

	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		3,
	)

	if len(routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(routes))
	}
	if len(routes[0].edges) != 1 {
		t.Fatalf("expected 1 hop, got %d", len(routes[0].edges))
	}
}

func TestFindRoutes_MultiHop(t *testing.T) {
	g := newTokenGraph()

	// A -> B pool
	g.addPool(poolEdge{
		poolID: "pool-AB",
		token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		fee:    3000,
	})
	// B -> C pool
	g.addPool(poolEdge{
		poolID: "pool-BC",
		token0: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		token1: "0xcccccccccccccccccccccccccccccccccccccccc",
		fee:    500,
	})

	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xcccccccccccccccccccccccccccccccccccccccc",
		3,
	)

	if len(routes) != 1 {
		t.Fatalf("expected 1 route (A->B->C), got %d", len(routes))
	}
	if len(routes[0].edges) != 2 {
		t.Fatalf("expected 2 hops, got %d", len(routes[0].edges))
	}
	if routes[0].edges[0].poolID != "pool-AB" {
		t.Fatalf("first hop should use pool-AB, got %s", routes[0].edges[0].poolID)
	}
	if routes[0].edges[1].poolID != "pool-BC" {
		t.Fatalf("second hop should use pool-BC, got %s", routes[0].edges[1].poolID)
	}
}

func TestFindRoutes_NoRoute(t *testing.T) {
	g := newTokenGraph()
	g.addPool(poolEdge{
		poolID: "pool-AB",
		token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		fee:    3000,
	})

	// C is not connected
	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xcccccccccccccccccccccccccccccccccccccccc",
		3,
	)

	if len(routes) != 0 {
		t.Fatalf("expected 0 routes, got %d", len(routes))
	}
}

func TestFindRoutes_MaxHopsLimit(t *testing.T) {
	g := newTokenGraph()

	// Chain: A -> B -> C -> D
	g.addPool(poolEdge{poolID: "AB", token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", fee: 3000})
	g.addPool(poolEdge{poolID: "BC", token0: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", token1: "0xcccccccccccccccccccccccccccccccccccccccc", fee: 3000})
	g.addPool(poolEdge{poolID: "CD", token0: "0xcccccccccccccccccccccccccccccccccccccccc", token1: "0xdddddddddddddddddddddddddddddddddddddddd", fee: 3000})

	// With maxHops=2, should NOT find A->B->C->D (3 hops)
	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xdddddddddddddddddddddddddddddddddddddddd",
		2,
	)
	if len(routes) != 0 {
		t.Fatalf("expected 0 routes with maxHops=2, got %d", len(routes))
	}

	// With maxHops=3, should find it
	routes = g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xdddddddddddddddddddddddddddddddddddddddd",
		3,
	)
	if len(routes) != 1 {
		t.Fatalf("expected 1 route with maxHops=3, got %d", len(routes))
	}
	if len(routes[0].edges) != 3 {
		t.Fatalf("expected 3 hops, got %d", len(routes[0].edges))
	}
}

func TestFindRoutes_NoCycles(t *testing.T) {
	g := newTokenGraph()

	// Triangle: A-B, B-C, A-C
	g.addPool(poolEdge{poolID: "AB", token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", fee: 3000})
	g.addPool(poolEdge{poolID: "BC", token0: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", token1: "0xcccccccccccccccccccccccccccccccccccccccc", fee: 500})
	g.addPool(poolEdge{poolID: "AC", token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token1: "0xcccccccccccccccccccccccccccccccccccccccc", fee: 100})

	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xcccccccccccccccccccccccccccccccccccccccc",
		4,
	)

	// Should find 2 routes: A->C (direct) and A->B->C (2-hop)
	// Should NOT find any route that revisits a token
	if len(routes) != 2 {
		t.Fatalf("expected 2 routes, got %d", len(routes))
	}

	for _, route := range routes {
		seen := make(map[string]bool)
		for _, token := range route.path {
			if seen[token] {
				t.Fatalf("cycle detected in route: token %s visited twice", token)
			}
			seen[token] = true
		}
	}
}

func TestFindRoutes_MultiplePoolsSamePair(t *testing.T) {
	g := newTokenGraph()

	// Two pools between A and B with different fees
	g.addPool(poolEdge{poolID: "AB-low", token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", fee: 500})
	g.addPool(poolEdge{poolID: "AB-high", token0: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token1: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", fee: 3000})

	routes := g.FindRoutes(
		"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		3,
	)

	// Should find 2 routes (one through each fee tier)
	if len(routes) != 2 {
		t.Fatalf("expected 2 routes, got %d", len(routes))
	}
}

// ==========================================================================
// Route Simulation Tests
// ==========================================================================

func TestSimulateRoute_SingleHop(t *testing.T) {
	route := candidateRoute{
		path: []string{
			"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		},
		edges: []poolEdge{
			{poolID: "pool-AB", fee: 3000, tickSpacing: 60},
		},
	}

	amountIn := big.NewInt(1_000_000_000) // 1e9
	result := SimulateRoute(route, amountIn, true)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if len(result.Hops) != 1 {
		t.Fatalf("expected 1 hop, got %d", len(result.Hops))
	}

	// Output should be amountIn * (1_000_000 - 3000) / 1_000_000 = 997_000 * 1000 = 997_000_000
	expectedOut := big.NewInt(997_000_000)
	actualOut := parseBigIntStr(result.Hops[0].AmountOut)
	if actualOut.Cmp(expectedOut) != 0 {
		t.Fatalf("expected amountOut %s, got %s", expectedOut, actualOut)
	}

	if result.TotalGasEstimate != 150_000 {
		t.Fatalf("expected gas 150000, got %d", result.TotalGasEstimate)
	}
}

func TestSimulateRoute_MultiHop(t *testing.T) {
	route := candidateRoute{
		path: []string{
			"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			"0xcccccccccccccccccccccccccccccccccccccccc",
		},
		edges: []poolEdge{
			{poolID: "AB", fee: 3000, tickSpacing: 60},
			{poolID: "BC", fee: 500, tickSpacing: 10},
		},
	}

	amountIn := big.NewInt(1_000_000_000)
	result := SimulateRoute(route, amountIn, true)

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if len(result.Hops) != 2 {
		t.Fatalf("expected 2 hops, got %d", len(result.Hops))
	}

	// Hop 1: 1e9 * 997000/1000000 = 997_000_000
	// Hop 2: 997_000_000 * 999500/1000000 = 996_501_500
	expectedFinalOut := big.NewInt(996_501_500)
	actualFinalOut := parseBigIntStr(result.TotalAmountOut)
	if actualFinalOut.Cmp(expectedFinalOut) != 0 {
		t.Fatalf("expected total amountOut %s, got %s", expectedFinalOut, actualFinalOut)
	}

	// Gas: 150k + 50k for second hop = 200k
	if result.TotalGasEstimate != 200_000 {
		t.Fatalf("expected gas 200000, got %d", result.TotalGasEstimate)
	}
}

func TestSimulateRoute_Empty(t *testing.T) {
	route := candidateRoute{
		path:  []string{},
		edges: []poolEdge{},
	}

	result := SimulateRoute(route, big.NewInt(1000), true)
	if result != nil {
		t.Fatal("expected nil for empty route")
	}
}

func TestSimulateRoute_ZeroAmount(t *testing.T) {
	route := candidateRoute{
		path: []string{"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
		edges: []poolEdge{
			{poolID: "AB", fee: 3000},
		},
	}

	result := SimulateRoute(route, big.NewInt(0), true)
	if result != nil {
		t.Fatal("expected nil for zero amount")
	}
}

// ==========================================================================
// BuildRoutesFromPools Tests
// ==========================================================================

func TestBuildRoutesFromPools_SortedByOutput(t *testing.T) {
	pools := []Pool{
		{
			Address: "0x0000000000000000000000000000000000001001",
			Token0:  Token{Address: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
			Token1:  Token{Address: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
			Fee:     500, // Lower fee = more output
			TVL:     big.NewInt(1_000_000),
		},
		{
			Address: "0x0000000000000000000000000000000000001002",
			Token0:  Token{Address: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
			Token1:  Token{Address: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
			Fee:     3000, // Higher fee = less output
			TVL:     big.NewInt(1_000_000),
		},
	}

	req := RoutingRequest{
		TokenIn:   "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		TokenOut:  "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Amount:    "1000000000",
		IsExactIn: true,
		ChainID:   96369,
	}

	resp, err := BuildRoutesFromPools(pools, req)
	if err != nil {
		t.Fatalf("BuildRoutesFromPools failed: %v", err)
	}

	if len(resp.Routes) != 2 {
		t.Fatalf("expected 2 routes, got %d", len(resp.Routes))
	}

	// First route should have higher output (lower fee)
	out0 := parseBigIntStr(resp.Routes[0].TotalAmountOut)
	out1 := parseBigIntStr(resp.Routes[1].TotalAmountOut)
	if out0.Cmp(out1) <= 0 {
		t.Fatalf("routes not sorted by output: first=%s, second=%s", out0, out1)
	}
}

func TestBuildRoutesFromPools_EmptyPools(t *testing.T) {
	req := RoutingRequest{
		TokenIn:   "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		TokenOut:  "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Amount:    "1000000000",
		IsExactIn: true,
		ChainID:   96369,
	}

	resp, err := BuildRoutesFromPools(nil, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Routes) != 0 {
		t.Fatalf("expected 0 routes, got %d", len(resp.Routes))
	}
}

// ==========================================================================
// Routing Validation Tests
// ==========================================================================

func TestRoutingRequestValidation(t *testing.T) {
	valid := RoutingRequest{
		TokenIn:   "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		TokenOut:  "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Amount:    "1000000000",
		IsExactIn: true,
		ChainID:   96369,
	}

	t.Run("valid", func(t *testing.T) {
		if err := valid.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("missing tokenIn", func(t *testing.T) {
		req := valid
		req.TokenIn = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("invalid tokenIn", func(t *testing.T) {
		req := valid
		req.TokenIn = "not_hex"
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("missing tokenOut", func(t *testing.T) {
		req := valid
		req.TokenOut = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("same tokens", func(t *testing.T) {
		req := valid
		req.TokenOut = req.TokenIn
		if err := req.Validate(); err == nil {
			t.Fatal("expected error for same tokens")
		}
	})

	t.Run("missing amount", func(t *testing.T) {
		req := valid
		req.Amount = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("zero amount", func(t *testing.T) {
		req := valid
		req.Amount = "0"
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("maxHops exceeds limit", func(t *testing.T) {
		req := valid
		req.MaxHops = 5
		if err := req.Validate(); err == nil {
			t.Fatal("expected error for maxHops > 4")
		}
	})

	t.Run("negative maxHops", func(t *testing.T) {
		req := valid
		req.MaxHops = -1
		if err := req.Validate(); err == nil {
			t.Fatal("expected error for negative maxHops")
		}
	})
}

// ==========================================================================
// HTTP Handler Tests
// ==========================================================================

func newTestMultihopServer(t *testing.T) *Server {
	t.Helper()
	registry := NewRegistry()
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	if err := registry.RegisterProvider(provider); err != nil {
		t.Fatal(err)
	}
	router := NewRouter(registry, true)
	cfg := DefaultServerConfig()
	return NewServer(router, cfg)
}

func TestHandleRoute(t *testing.T) {
	s := newTestMultihopServer(t)

	body := `{
		"tokenIn": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"tokenOut": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		"amount": "1000000000",
		"isExactIn": true,
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/route", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleRoute(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp apiResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if !resp.Success {
		t.Fatalf("expected success: %s", resp.Error)
	}
}

func TestHandleRoute_InvalidMethod(t *testing.T) {
	s := newTestMultihopServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/route", nil)
	w := httptest.NewRecorder()
	s.handleRoute(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleRoute_InvalidBody(t *testing.T) {
	s := newTestMultihopServer(t)

	body := `{"tokenIn": ""}`
	req := httptest.NewRequest(http.MethodPost, "/v1/route", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleRoute(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}
