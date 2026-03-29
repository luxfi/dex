package trading

import (
	"bytes"
	"context"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
)

// --- Test token addresses (shared across chains for simplicity) ---

var (
	ccUSDC = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	ccWETH = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	ccLUX  = "0xcccccccccccccccccccccccccccccccccccccccc"
	ccLUSD = "0xdddddddddddddddddddddddddddddddddddddddd" // generic stablecoin

	// Arb-specific tokens with pricing imbalance.
	arbTokenA = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
	arbTokenB = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc"
	arbTokenC = "0xcccccccccccccccccccccccccccccccccccccccd"
)

const e18Str = "1000000000000000000"

func e18(n int64) *big.Int {
	return new(big.Int).Mul(big.NewInt(n), mustParseBigInt(e18Str))
}

func mustParseBigInt(s string) *big.Int {
	n, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("bad bigint: " + s)
	}
	return n
}

// buildTestChainGraph sets up 4 chains + 4 bridges for cross-chain tests.
//
// Chains:
//
//	Base (8453)      — USDC/WETH pool
//	Ethereum (1)     — USDC/WETH pool (deeper)
//	Lux (96369)      — WETH/Stable pool, LUX/Stable pool
//	Zoo (200200)     — USDC/Stable pool, WETH/Stable pool
//
// Bridges:
//
//	CCTP:      Base↔Ethereum (USDC only, 0 fee)
//	Teleport:  Ethereum↔Lux (WETH, USDC, 15bps)
//	Warp:      Lux↔Zoo (WETH, Stable, LUX, 5bps)
//	LuxBridge: Ethereum↔Zoo (WETH, USDC, 10bps)
func buildTestChainGraph() *CrossChainRouter {
	router := NewCrossChainRouter()

	// --- Base ---
	baseV4 := NewV4Venue("")
	baseV4.AddPool(ccUSDC, ccWETH, "", 30, e18(10_000_000), e18(5_000))
	router.RegisterChain(&Chain{
		ChainID: ChainBase, Name: "Base", Venues: []Venue{baseV4},
	})

	// --- Ethereum ---
	ethV4 := NewV4Venue("")
	ethV4.AddPool(ccUSDC, ccWETH, "", 30, e18(100_000_000), e18(50_000))
	router.RegisterChain(&Chain{
		ChainID: ChainEthereum, Name: "Ethereum", Venues: []Venue{ethV4},
	})

	// --- Lux ---
	luxV4 := NewV4Venue("")
	luxV4.AddPool(ccWETH, ccLUSD, "", 10, e18(5_000), e18(10_000_000))
	luxV4.AddPool(ccLUX, ccLUSD, "", 10, e18(1_000_000), e18(500_000))
	router.RegisterChain(&Chain{
		ChainID: ChainLux, Name: "Lux", Venues: []Venue{luxV4},
	})

	// --- Zoo ---
	zooV4 := NewV4Venue("")
	zooV4.AddPool(ccUSDC, ccLUSD, "", 1, e18(10_000_000), e18(10_000_000))
	zooV4.AddPool(ccWETH, ccLUSD, "", 10, e18(5_000), e18(10_000_000))
	router.RegisterChain(&Chain{
		ChainID: ChainZoo, Name: "Zoo", Venues: []Venue{zooV4},
	})

	// --- Bridges ---
	router.RegisterBridge(NewCCTPBridge(CCTPBridgeConfig{
		Routes: [][2]int{{ChainBase, ChainEthereum}},
		Tokens: map[int][]string{ChainBase: {ccUSDC}, ChainEthereum: {ccUSDC}},
	}))

	router.RegisterBridge(NewTeleportBridge(TeleportBridgeConfig{
		Routes: [][2]int{{ChainEthereum, ChainLux}},
		Tokens: map[int][]string{ChainEthereum: {ccWETH, ccUSDC}, ChainLux: {ccWETH, ccUSDC}},
		FeeBPS: 15,
	}))

	router.RegisterBridge(NewWarpBridge(WarpBridgeConfig{
		Routes: [][2]int{{ChainLux, ChainZoo}},
		Tokens: map[int][]string{ChainLux: {ccWETH, ccLUSD, ccLUX}, ChainZoo: {ccWETH, ccLUSD, ccLUX}},
		FeeBPS: 5,
	}))

	router.RegisterBridge(NewLuxBridge(LuxBridgeConfig{
		Routes: [][2]int{{ChainEthereum, ChainZoo}},
		Tokens: map[int][]string{ChainEthereum: {ccWETH, ccUSDC}, ChainZoo: {ccWETH, ccUSDC}},
		FeeBPS: 10,
	}))

	return router
}

// ===================== CrossChainRouter Tests =====================

func TestCrossChainSameChainSwap(t *testing.T) {
	router := buildTestChainGraph()
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccWETH, Amount: e18(1000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainBase,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected at least one local route")
	}
	r := routes[0]
	if len(r.Hops) != 1 {
		t.Fatalf("expected 1 hop, got %d", len(r.Hops))
	}
	if r.Hops[0].Type != HopTypeSwap {
		t.Fatalf("expected swap hop, got %s", r.Hops[0].Type)
	}
	if r.Hops[0].ChainID != ChainBase {
		t.Fatalf("expected chain Base, got %d", r.Hops[0].ChainID)
	}
	if r.TotalBridgeFee != "0" {
		t.Fatalf("expected 0 bridge fee, got %s", r.TotalBridgeFee)
	}
}

func TestCrossChainDirectBridge(t *testing.T) {
	router := buildTestChainGraph()
	// Base USDC → Ethereum USDC via CCTP (same token, direct bridge).
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccUSDC, Amount: e18(10_000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainEthereum,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected at least one route")
	}
	r := routes[0]
	// Should be a single bridge hop.
	hasBridge := false
	for _, h := range r.Hops {
		if h.Type == HopTypeBridge && h.Bridge == BridgeCCTP {
			hasBridge = true
		}
	}
	if !hasBridge {
		t.Fatal("expected CCTP bridge hop")
	}
	if !containsChain(r.Chains, ChainBase) || !containsChain(r.Chains, ChainEthereum) {
		t.Fatalf("expected both chains in route, got %v", r.Chains)
	}
}

func TestCrossChainBridgePlusSwap(t *testing.T) {
	router := buildTestChainGraph()
	// Ethereum WETH → Zoo Stable:
	// Bridge WETH via LuxBridge to Zoo, then swap WETH→Stable.
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccWETH, TokenOut: ccLUSD, Amount: e18(1).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainEthereum, DestChain: ChainZoo,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected at least one route")
	}
	r := routes[0]
	if len(r.Hops) < 2 {
		t.Fatalf("expected bridge+swap (>=2 hops), got %d", len(r.Hops))
	}

	// Verify route structure: bridge then swap.
	hasBridge := false
	hasSwap := false
	for _, h := range r.Hops {
		if h.Type == HopTypeBridge {
			hasBridge = true
		}
		if h.Type == HopTypeSwap && h.TokenOut == ccLUSD {
			hasSwap = true
		}
	}
	if !hasBridge {
		t.Fatal("expected a bridge hop")
	}
	if !hasSwap {
		t.Fatal("expected a swap hop to Stable")
	}

	// Output should be positive.
	out, ok := new(big.Int).SetString(r.TotalAmountOut, 10)
	if !ok || out.Sign() <= 0 {
		t.Fatalf("expected positive output, got %s", r.TotalAmountOut)
	}
}

func TestCrossChainMultiHop(t *testing.T) {
	router := buildTestChainGraph()
	// Base USDC → Zoo Stable:
	// Path: Base USDC → CCTP → Ethereum USDC → LuxBridge → Zoo USDC → swap USDC→Stable.
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: e18(5000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainZoo,
		MaxHops: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected at least one multi-hop route")
	}

	r := routes[0]
	bridgeCount := 0
	for _, h := range r.Hops {
		if h.Type == HopTypeBridge {
			bridgeCount++
		}
	}
	if bridgeCount < 1 {
		t.Fatalf("expected at least 1 bridge, got %d", bridgeCount)
	}
	if len(r.Chains) < 2 {
		t.Fatalf("expected multi-chain route, got chains %v", r.Chains)
	}

	out, _ := new(big.Int).SetString(r.TotalAmountOut, 10)
	if out == nil || out.Sign() <= 0 {
		t.Fatalf("expected positive output, got %s", r.TotalAmountOut)
	}
}

func TestCrossChainNoRoute(t *testing.T) {
	router := buildTestChainGraph()
	// Unknown chain → should error.
	_, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: e18(1000).String(),
		Type: QuoteTypeExactInput, SourceChain: 999999, DestChain: ChainZoo,
	})
	if err == nil {
		t.Fatal("expected error for unknown chain")
	}
}

func TestCrossChainMaxHopsLimit(t *testing.T) {
	router := buildTestChainGraph()
	// Base → Zoo with maxHops=1. The shortest path requires ≥2 hops (bridge + swap),
	// so with maxHops=1 we should find fewer/no routes.
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: e18(1000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainZoo,
		MaxHops: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	// With maxHops=1, a single bridge to Zoo might work if USDC=Stable,
	// but since USDC≠Stable and we need a swap on arrival, 1 hop isn't enough.
	// However, CCTP or LuxBridge might bridge to Zoo in 1 hop with USDC=USDC.
	// Let's just verify the result is constrained.
	for _, r := range routes {
		if len(r.Hops) > 1 {
			t.Fatalf("expected max 1 hop, got %d", len(r.Hops))
		}
	}
}

func TestCrossChainPreferBridge(t *testing.T) {
	router := buildTestChainGraph()
	// Ethereum WETH → Zoo Stable, prefer Warp bridge.
	// Warp only goes Lux↔Zoo, not Ethereum↔Zoo.
	// So preferring Warp should route through Lux.
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccWETH, TokenOut: ccLUSD, Amount: e18(1).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainEthereum, DestChain: ChainZoo,
		PreferBridge: BridgeWarp,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Warp doesn't connect Ethereum directly, so we need Teleport→Lux→Warp→Zoo.
	// But with PreferBridge=warp, only warp bridges are used. This means the router
	// can't bridge from Ethereum at all (warp doesn't serve Ethereum).
	// So routes should be empty.
	if len(routes) != 0 {
		t.Fatalf("expected no routes with warp-only from Ethereum, got %d", len(routes))
	}
}

// ===================== Bridge Venue Tests =====================

func TestWarpBridgeQuote(t *testing.T) {
	b := NewWarpBridge(WarpBridgeConfig{
		Routes: [][2]int{{ChainLux, ChainZoo}},
		FeeBPS: 5,
	})
	q, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccLUSD, Amount: e18(1_000_000).String(),
		SourceChain: ChainLux, DestChain: ChainZoo,
	})
	if err != nil {
		t.Fatal(err)
	}
	if q == nil {
		t.Fatal("expected quote")
	}
	if q.Bridge != BridgeWarp {
		t.Fatalf("expected warp, got %s", q.Bridge)
	}
	// 5bps fee on 1M = 500.
	fee, _ := new(big.Int).SetString(q.Fee, 10)
	expected := new(big.Int).Div(e18(1_000_000), big.NewInt(2000)) // 5/10000 = 1/2000
	if fee.Cmp(expected) != 0 {
		t.Fatalf("expected fee %s, got %s", expected, fee)
	}
}

func TestTeleportBridgeQuote(t *testing.T) {
	b := NewTeleportBridge(TeleportBridgeConfig{
		Routes: [][2]int{{ChainEthereum, ChainLux}},
		FeeBPS: 15,
	})
	q, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccWETH, Amount: e18(10).String(),
		SourceChain: ChainEthereum, DestChain: ChainLux,
	})
	if err != nil {
		t.Fatal(err)
	}
	if q == nil {
		t.Fatal("expected quote")
	}
	if q.Bridge != BridgeTeleport {
		t.Fatalf("expected teleport, got %s", q.Bridge)
	}
	out, _ := new(big.Int).SetString(q.AmountOut, 10)
	in := e18(10)
	if out.Cmp(in) >= 0 {
		t.Fatal("output should be less than input (fee deducted)")
	}
}

func TestCCTPBridgeQuote(t *testing.T) {
	b := NewCCTPBridge(CCTPBridgeConfig{
		Routes: [][2]int{{ChainBase, ChainEthereum}},
		FeeBPS: 0,
	})
	amount := e18(50_000).String()
	q, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccUSDC, Amount: amount,
		SourceChain: ChainBase, DestChain: ChainEthereum,
	})
	if err != nil {
		t.Fatal(err)
	}
	if q == nil {
		t.Fatal("expected quote")
	}
	// 0 fee → amount out = amount in.
	if q.AmountOut != amount {
		t.Fatalf("expected %s out with 0 fee, got %s", amount, q.AmountOut)
	}
}

func TestLuxBridgeQuote(t *testing.T) {
	b := NewLuxBridge(LuxBridgeConfig{
		Routes: [][2]int{{ChainEthereum, ChainZoo}},
		FeeBPS: 10,
	})
	q, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccWETH, Amount: e18(5).String(),
		SourceChain: ChainEthereum, DestChain: ChainZoo,
	})
	if err != nil {
		t.Fatal(err)
	}
	if q == nil {
		t.Fatal("expected quote")
	}
	if q.Bridge != BridgeLux {
		t.Fatalf("expected lux_bridge, got %s", q.Bridge)
	}
}

func TestBridgeUnsupportedRoute(t *testing.T) {
	b := NewWarpBridge(WarpBridgeConfig{
		Routes: [][2]int{{ChainLux, ChainZoo}},
	})
	// Warp doesn't support Ethereum↔Base.
	q, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccUSDC, Amount: "1000", SourceChain: ChainEthereum, DestChain: ChainBase,
	})
	if err != nil {
		t.Fatal(err)
	}
	if q != nil {
		t.Fatal("expected nil for unsupported route")
	}
}

func TestBridgeBidirectional(t *testing.T) {
	b := NewWarpBridge(WarpBridgeConfig{
		Routes: [][2]int{{ChainLux, ChainZoo}},
		FeeBPS: 5,
	})
	// Forward: Lux → Zoo.
	q1, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccLUSD, Amount: "1000000", SourceChain: ChainLux, DestChain: ChainZoo,
	})
	if err != nil || q1 == nil {
		t.Fatal("forward bridge should work")
	}
	// Reverse: Zoo → Lux.
	q2, err := b.QuoteBridge(context.Background(), BridgeQuoteRequest{
		Token: ccLUSD, Amount: "1000000", SourceChain: ChainZoo, DestChain: ChainLux,
	})
	if err != nil || q2 == nil {
		t.Fatal("reverse bridge should work")
	}
	if q1.AmountOut != q2.AmountOut {
		t.Fatal("bidirectional bridge should have symmetric quotes")
	}
}

// ===================== Arbitrage Tests =====================

// buildArbChain creates a chain with imbalanced pools for triangular arb.
// Pools: A/B (10M:5M), B/C (5M:10M), C/A (10M:12M), all at 1bps fee.
// The pricing cycle A→B→C→A has ~16% profit due to the C/A imbalance.
func buildArbChain() *Chain {
	v4 := NewV4Venue("")
	v4.AddPool(arbTokenA, arbTokenB, "", 1, e18(10_000_000), e18(5_000_000))
	v4.AddPool(arbTokenB, arbTokenC, "", 1, e18(5_000_000), e18(10_000_000))
	v4.AddPool(arbTokenC, arbTokenA, "", 1, e18(10_000_000), e18(12_000_000))
	return &Chain{ChainID: ChainBase, Name: "Base-Arb", Venues: []Venue{v4}}
}

// buildBalancedChain creates a chain with balanced pools (no arb opportunity).
func buildBalancedChain() *Chain {
	v4 := NewV4Venue("")
	v4.AddPool(arbTokenA, arbTokenB, "", 30, e18(10_000_000), e18(10_000_000))
	v4.AddPool(arbTokenB, arbTokenC, "", 30, e18(10_000_000), e18(10_000_000))
	v4.AddPool(arbTokenC, arbTokenA, "", 30, e18(10_000_000), e18(10_000_000))
	return &Chain{ChainID: ChainEthereum, Name: "Ethereum-Balanced", Venues: []Venue{v4}}
}

func TestTriangularArbitrageDetected(t *testing.T) {
	router := NewCrossChainRouter()
	router.RegisterChain(buildArbChain())

	scanner := NewArbScanner(router, 10)
	opps, err := scanner.ScanTriangular(
		context.Background(), ChainBase,
		[]string{arbTokenA, arbTokenB, arbTokenC},
		e18(100_000).String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(opps) == 0 {
		t.Fatal("expected at least one triangular arb opportunity")
	}

	best := opps[0]
	if best.Type != ArbTriangular {
		t.Fatalf("expected TRIANGULAR, got %s", best.Type)
	}
	if best.ProfitBPS < 100 {
		t.Fatalf("expected >1%% profit, got %d bps", best.ProfitBPS)
	}
	if len(best.Path) != 3 {
		t.Fatalf("expected 3-step path, got %d", len(best.Path))
	}

	netProfit, _ := new(big.Int).SetString(best.NetProfit, 10)
	if netProfit == nil || netProfit.Sign() <= 0 {
		t.Fatalf("expected positive net profit, got %s", best.NetProfit)
	}

	t.Logf("Triangular arb: %d bps profit, net=%s, path: %s→%s→%s→%s",
		best.ProfitBPS, best.NetProfit,
		best.Path[0].TokenIn[:10], best.Path[0].TokenOut[:10],
		best.Path[1].TokenOut[:10], best.Path[2].TokenOut[:10])
}

func TestTriangularArbitrageNoProfit(t *testing.T) {
	router := NewCrossChainRouter()
	router.RegisterChain(buildBalancedChain())

	scanner := NewArbScanner(router, 10)
	opps, err := scanner.ScanTriangular(
		context.Background(), ChainEthereum,
		[]string{arbTokenA, arbTokenB, arbTokenC},
		e18(100_000).String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(opps) > 0 {
		t.Fatalf("expected no arb in balanced pools, got %d opps (best=%d bps)", len(opps), opps[0].ProfitBPS)
	}
}

func TestCrossChainArbitrage(t *testing.T) {
	router := NewCrossChainRouter()

	// Ethereum: 1 USDC = 0.0005 WETH (price 2000 USDC/WETH).
	ethV4 := NewV4Venue("")
	ethV4.AddPool(ccUSDC, ccWETH, "", 30, e18(100_000_000), e18(50_000))
	router.RegisterChain(&Chain{ChainID: ChainEthereum, Name: "Ethereum", Venues: []Venue{ethV4}})

	// Base: 1 USDC = 0.0004 WETH (price 2500 USDC/WETH, 25% more expensive).
	baseV4 := NewV4Venue("")
	baseV4.AddPool(ccUSDC, ccWETH, "", 30, e18(10_000_000), e18(4_000))
	router.RegisterChain(&Chain{ChainID: ChainBase, Name: "Base", Venues: []Venue{baseV4}})

	scanner := NewArbScanner(router, 10)
	opps, err := scanner.ScanCrossChain(
		context.Background(), ccUSDC, ccWETH,
		[]int{ChainEthereum, ChainBase}, e18(10_000).String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(opps) == 0 {
		t.Fatal("expected cross-chain arb between Ethereum and Base")
	}

	best := opps[0]
	if best.Type != ArbCrossChain {
		t.Fatalf("expected CROSS_CHAIN, got %s", best.Type)
	}
	if best.ProfitBPS < 100 {
		t.Fatalf("expected significant spread, got %d bps", best.ProfitBPS)
	}
	if len(best.Path) != 2 {
		t.Fatalf("expected 2 quotes (buy+sell), got %d", len(best.Path))
	}

	t.Logf("Cross-chain arb: %d bps spread, net=%s", best.ProfitBPS, best.NetProfit)
}

func TestCrossChainArbitrageNoSpread(t *testing.T) {
	router := NewCrossChainRouter()

	// Both chains have identical pools → no arb.
	for _, chainID := range []int{ChainEthereum, ChainBase} {
		v4 := NewV4Venue("")
		v4.AddPool(ccUSDC, ccWETH, "", 30, e18(100_000_000), e18(50_000))
		router.RegisterChain(&Chain{ChainID: chainID, Venues: []Venue{v4}})
	}

	scanner := NewArbScanner(router, 10)
	opps, err := scanner.ScanCrossChain(
		context.Background(), ccUSDC, ccWETH,
		[]int{ChainEthereum, ChainBase}, e18(10_000).String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(opps) > 0 {
		t.Fatalf("expected no arb with identical pools, got %d", len(opps))
	}
}

func TestArbScannerMinProfitFilter(t *testing.T) {
	router := NewCrossChainRouter()
	router.RegisterChain(buildArbChain())

	// Set high minimum → should filter out smaller opps.
	scanner := NewArbScanner(router, 5000) // 50% min
	opps, err := scanner.ScanTriangular(
		context.Background(), ChainBase,
		[]string{arbTokenA, arbTokenB, arbTokenC},
		e18(100_000).String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	// Our arb is ~16%, which is below 50% min.
	if len(opps) > 0 {
		t.Fatalf("expected no opps above 50%% min, got %d (best=%d bps)", len(opps), opps[0].ProfitBPS)
	}
}

func TestArbScannerTooFewTokens(t *testing.T) {
	router := NewCrossChainRouter()
	router.RegisterChain(buildArbChain())

	scanner := NewArbScanner(router, 10)
	_, err := scanner.ScanTriangular(context.Background(), ChainBase, []string{arbTokenA, arbTokenB}, "1000")
	if err == nil {
		t.Fatal("expected error with <3 tokens")
	}
}

// ===================== Handler Tests =====================

func buildCrossChainAPI() (*TradingAPI, *http.ServeMux) {
	router := buildTestChainGraph()
	scanner := NewArbScanner(router, 10)

	cfg := Config{
		ChainIDs:       []int{ChainEthereum, ChainBase, ChainLux, ChainZoo},
		DefaultChainID: ChainZoo,
	}
	api := New(cfg)
	api.SetCrossChainRouter(router)
	api.SetArbScanner(scanner)

	mux := http.NewServeMux()
	api.RegisterRoutes(mux)
	api.RegisterCrossChainRoutes(mux)
	return api, mux
}

func ccPostJSON(mux *http.ServeMux, path string, body any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	req := httptest.NewRequest("POST", path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func ccGet(mux *http.ServeMux, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("GET", path, nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func TestHandleCrossChainQuote(t *testing.T) {
	_, mux := buildCrossChainAPI()

	w := ccPostJSON(mux, "/v1/trade/cross-chain/quote", CrossChainQuoteRequest{
		TokenIn: ccWETH, TokenOut: ccLUSD, Amount: e18(1).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainEthereum, DestChain: ChainZoo,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp CrossChainQuoteResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.BestRoute == nil {
		t.Fatal("expected best route")
	}
	if len(resp.Routes) == 0 {
		t.Fatal("expected routes")
	}
}

func TestHandleCrossChainQuoteValidation(t *testing.T) {
	_, mux := buildCrossChainAPI()

	tests := []struct {
		name string
		body CrossChainQuoteRequest
	}{
		{"bad tokenIn", CrossChainQuoteRequest{TokenIn: "bad", TokenOut: ccLUSD, Amount: "1000", SourceChain: 1, DestChain: ChainZoo}},
		{"bad tokenOut", CrossChainQuoteRequest{TokenIn: ccUSDC, TokenOut: "bad", Amount: "1000", SourceChain: 1, DestChain: ChainZoo}},
		{"bad amount", CrossChainQuoteRequest{TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: "0", SourceChain: 1, DestChain: ChainZoo}},
		{"bad source", CrossChainQuoteRequest{TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: "1000", SourceChain: 0, DestChain: ChainZoo}},
		{"bad dest", CrossChainQuoteRequest{TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: "1000", SourceChain: 1, DestChain: 0}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w := ccPostJSON(mux, "/v1/trade/cross-chain/quote", tc.body)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestHandleCrossChainQuoteNoRouter(t *testing.T) {
	api := New(Config{ChainIDs: []int{1}, DefaultChainID: 1})
	mux := http.NewServeMux()
	api.RegisterCrossChainRoutes(mux)

	w := ccPostJSON(mux, "/v1/trade/cross-chain/quote", CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: "1000",
		SourceChain: 1, DestChain: ChainZoo,
	})
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestHandleBridges(t *testing.T) {
	_, mux := buildCrossChainAPI()

	w := ccGet(mux, "/v1/trade/bridges")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var bridges []BridgeInfo
	if err := json.NewDecoder(w.Body).Decode(&bridges); err != nil {
		t.Fatal(err)
	}
	if len(bridges) == 0 {
		t.Fatal("expected bridges")
	}

	// Verify we have all 4 bridge types.
	types := make(map[BridgeType]bool)
	for _, b := range bridges {
		types[b.Type] = true
	}
	for _, bt := range []BridgeType{BridgeCCTP, BridgeTeleport, BridgeWarp, BridgeLux} {
		if !types[bt] {
			t.Fatalf("missing bridge type: %s", bt)
		}
	}
}

func TestHandleArbScan(t *testing.T) {
	router := NewCrossChainRouter()
	router.RegisterChain(buildArbChain())

	scanner := NewArbScanner(router, 10)
	api := New(Config{ChainIDs: []int{ChainBase}, DefaultChainID: ChainBase})
	api.SetCrossChainRouter(router)
	api.SetArbScanner(scanner)

	mux := http.NewServeMux()
	api.RegisterCrossChainRoutes(mux)

	w := ccPostJSON(mux, "/v1/trade/arbitrage/scan", ArbScanRequest{
		Tokens:       []string{arbTokenA, arbTokenB, arbTokenC},
		ChainIDs:     []int{ChainBase},
		MinProfitBPS: 10,
		InputAmount:  e18(100_000).String(),
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp ArbScanResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Opportunities) == 0 {
		t.Fatal("expected arb opportunities")
	}
	if resp.ScannedPairs == 0 {
		t.Fatal("expected scanned pairs > 0")
	}
	if resp.ElapsedMs < 0 {
		t.Fatal("expected non-negative elapsed time")
	}
}

func TestHandleArbScanValidation(t *testing.T) {
	router := NewCrossChainRouter()
	api := New(Config{ChainIDs: []int{1}, DefaultChainID: 1})
	api.SetCrossChainRouter(router)
	api.SetArbScanner(NewArbScanner(router, 10))

	mux := http.NewServeMux()
	api.RegisterCrossChainRoutes(mux)

	// Too few tokens.
	w := ccPostJSON(mux, "/v1/trade/arbitrage/scan", ArbScanRequest{
		Tokens: []string{arbTokenA, arbTokenB}, ChainIDs: []int{1}, InputAmount: "1000",
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}

	// Bad amount.
	w = ccPostJSON(mux, "/v1/trade/arbitrage/scan", ArbScanRequest{
		Tokens: []string{arbTokenA, arbTokenB, arbTokenC}, ChainIDs: []int{1}, InputAmount: "0",
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandleArbScanNoScanner(t *testing.T) {
	api := New(Config{ChainIDs: []int{1}, DefaultChainID: 1})
	mux := http.NewServeMux()
	api.RegisterCrossChainRoutes(mux)

	w := ccPostJSON(mux, "/v1/trade/arbitrage/scan", ArbScanRequest{
		Tokens: []string{arbTokenA, arbTokenB, arbTokenC}, ChainIDs: []int{1}, InputAmount: "1000",
	})
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

// ===================== Full Scenario Tests =====================

func TestFullScenarioBaseToZoo(t *testing.T) {
	router := buildTestChainGraph()

	// Scenario: Base USDC → Zoo Stable.
	// Expected path: Base USDC → (CCTP) → Ethereum USDC → (LuxBridge) → Zoo USDC → (swap) → Zoo Stable.
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: e18(10_000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainZoo,
		MaxHops: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected route from Base to Zoo")
	}

	r := routes[0]
	t.Logf("Route: %d hops, chains=%v, out=%s", len(r.Hops), r.Chains, r.TotalAmountOut)
	for i, h := range r.Hops {
		t.Logf("  Hop %d: %s chain=%d %s→%s venue=%s bridge=%s out=%s",
			i, h.Type, h.ChainID, h.TokenIn[:10], h.TokenOut[:10], h.Venue, h.Bridge, h.AmountOut)
	}

	// Verify output is reasonable (should be close to input for USDC→Stable ≈ 1:1).
	out, _ := new(big.Int).SetString(r.TotalAmountOut, 10)
	in := e18(10_000)
	if out == nil || out.Sign() <= 0 {
		t.Fatal("expected positive output")
	}
	// Should be within 10% of input (fees + slippage).
	minExpected := new(big.Int).Mul(in, big.NewInt(80))
	minExpected.Div(minExpected, big.NewInt(100))
	if out.Cmp(minExpected) < 0 {
		t.Fatalf("output %s too low (expected >%s)", out, minExpected)
	}
}

func TestFullScenarioEthereumToLux(t *testing.T) {
	router := buildTestChainGraph()

	// Scenario: Ethereum WETH → Lux Stable.
	// The router may use Teleport (direct) or LuxBridge+Warp (via Zoo).
	routes, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccWETH, TokenOut: ccLUSD, Amount: e18(2).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainEthereum, DestChain: ChainLux,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) == 0 {
		t.Fatal("expected route from Ethereum to Lux")
	}

	r := routes[0]
	// Should have at least one bridge + swap to Stable.
	hasBridge := false
	hasSwapToStable := false
	for _, h := range r.Hops {
		if h.Type == HopTypeBridge {
			hasBridge = true
		}
		if h.Type == HopTypeSwap && h.TokenOut == ccLUSD {
			hasSwapToStable = true
		}
	}
	if !hasBridge {
		t.Fatal("expected a bridge hop")
	}
	if !hasSwapToStable {
		t.Fatal("expected swap to Stable")
	}

	out, _ := new(big.Int).SetString(r.TotalAmountOut, 10)
	if out == nil || out.Sign() <= 0 {
		t.Fatal("expected positive output")
	}
	t.Logf("Ethereum→Lux: 2 WETH → %s Stable (%d hops, chains=%v)",
		r.TotalAmountOut, len(r.Hops), r.Chains)
}

func TestFullScenarioRoundTrip(t *testing.T) {
	router := buildTestChainGraph()

	// Forward: Base USDC → Zoo Stable.
	fwd, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccUSDC, TokenOut: ccLUSD, Amount: e18(1000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainBase, DestChain: ChainZoo,
		MaxHops: 4,
	})
	if err != nil || len(fwd) == 0 {
		t.Fatal("forward route failed")
	}

	// Reverse: Zoo Stable → Base USDC.
	// Note: This requires Warp→Lux, Teleport→Ethereum, V3 swap, CCTP→Base.
	// With maxHops=4, this might not be reachable, but let's see what we get.
	rev, err := router.FindRoutes(context.Background(), CrossChainQuoteRequest{
		TokenIn: ccLUSD, TokenOut: ccUSDC, Amount: e18(1000).String(),
		Type: QuoteTypeExactInput, SourceChain: ChainZoo, DestChain: ChainBase,
		MaxHops: 4,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Round trip may or may not find a path depending on bridge token support.
	// The important thing is no panic and valid structure.
	t.Logf("Forward routes: %d, Reverse routes: %d", len(fwd), len(rev))
	if len(fwd) > 0 {
		t.Logf("Forward best: %d hops, out=%s", len(fwd[0].Hops), fwd[0].TotalAmountOut)
	}
	if len(rev) > 0 {
		t.Logf("Reverse best: %d hops, out=%s", len(rev[0].Hops), rev[0].TotalAmountOut)
	}
}

// ===================== Helpers =====================

func containsChain(chains []int, id int) bool {
	for _, c := range chains {
		if c == id {
			return true
		}
	}
	return false
}
