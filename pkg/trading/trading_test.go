package trading

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Test addresses — generic, not chain-specific.
const (
	testLUSD   = "0xc65ea8882020Af7CDa7854d590C6Fcd34BF364ec"
	testWBTC  = "0x1234567890abcdef1234567890abcdef12345678"
	testWETH  = "0xabcdef1234567890abcdef1234567890abcdef12"
	testUser  = "0x9011E888251AB053B7bD1cdB598Db4f9DEd94714"
	testOther = "0x0000000000000000000000000000000000000001"

	testChainID  = 100001
	testChainID2 = 100002
	testChainID3 = 100003
)

func testConfig() Config {
	return Config{
		ChainIDs:       []int{testChainID, testChainID2, testChainID3},
		DefaultChainID: testChainID,
	}
}

func newTestAPI() (*TradingAPI, *http.ServeMux) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	v4 := NewV4Venue("")
	lusdReserve := new(big.Int).Mul(big.NewInt(10_000_000), e18)
	wbtcReserve := new(big.Int).Mul(big.NewInt(166), e18)
	v4.AddPool(testLUSD, testWBTC,
		"0x0000000000000000000000000000000000000000000000000000000000000001",
		30, lusdReserve, wbtcReserve)

	wethReserve := new(big.Int).Mul(big.NewInt(1666), e18)
	lusdReserve2 := new(big.Int).Mul(big.NewInt(5_000_000), e18)
	v4.AddPool(testLUSD, testWETH,
		"0x0000000000000000000000000000000000000000000000000000000000000002",
		30, lusdReserve2, wethReserve)

	wbtcReserve2 := new(big.Int).Mul(big.NewInt(100), e18)
	wethReserve2 := new(big.Int).Mul(big.NewInt(2000), e18)
	v4.AddPool(testWBTC, testWETH,
		"0x0000000000000000000000000000000000000000000000000000000000000003",
		5, wbtcReserve2, wethReserve2)

	alpaca := NewBrokerVenue("http://broker:8090", "alpaca")
	alpacaPrice := new(big.Int).Mul(big.NewInt(59800), e18)
	alpaca.SetMockPrice(alpacaPrice)
	alpaca.SetFees(10, 25)

	hyper := NewBrokerVenue("http://broker:8090", "hyperliquid")
	hyperPrice := new(big.Int).Mul(big.NewInt(59500), e18)
	hyper.SetMockPrice(hyperPrice)
	hyper.SetFees(15, 20)

	api := New(testConfig(), v4, alpaca, hyper)
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)
	return api, mux
}

func postJSON(mux *http.ServeMux, path string, body any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	req := httptest.NewRequest("POST", path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func getPath(mux *http.ServeMux, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("GET", path, nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

// ==========================================================================
// Quote tests
// ==========================================================================

func TestQuoteExactInput(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	amount := new(big.Int).Mul(big.NewInt(1000), e18)
	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: amount.String(),
		Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser, Slippage: 0.5,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp QuoteResponse
	json.NewDecoder(w.Body).Decode(&resp)

	if resp.Quote == nil {
		t.Fatal("expected non-nil quote")
	}
	if resp.Quote.AmountIn != amount.String() {
		t.Errorf("amountIn mismatch: got %s, want %s", resp.Quote.AmountIn, amount.String())
	}

	amountOut, _ := new(big.Int).SetString(resp.Quote.AmountOut, 10)
	if amountOut == nil || amountOut.Sign() <= 0 {
		t.Fatalf("expected positive amountOut, got %s", resp.Quote.AmountOut)
	}

	if len(resp.Quote.Route) == 0 {
		t.Fatal("expected at least one route hop")
	}
	if resp.Venues == nil || len(resp.Venues) != 3 {
		t.Errorf("expected 3 venue quotes, got %d", len(resp.Venues))
	}
	if resp.Routing != RoutingV4Native {
		t.Errorf("expected routing %s, got %s", RoutingV4Native, resp.Routing)
	}
}

func TestQuoteExactOutput(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	wantOut := new(big.Int).Div(e18, big.NewInt(100))
	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: wantOut.String(),
		Type: QuoteTypeExactOutput, ChainID: testChainID, Swapper: testUser, Slippage: 1.0,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp QuoteResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Quote == nil {
		t.Fatal("expected non-nil quote")
	}
}

func TestQuoteNoLiquidity(t *testing.T) {
	v4 := NewV4Venue("")
	api := New(testConfig(), v4)
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testOther, TokenOut: testWBTC, Amount: "1000000000000000000",
		Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser, Slippage: 0.5,
	})
	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", w.Code)
	}
}

func TestQuoteValidation(t *testing.T) {
	_, mux := newTestAPI()

	tests := []struct {
		name string
		req  QuoteRequest
	}{
		{"invalid tokenIn", QuoteRequest{
			TokenIn: "bad", TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser}},
		{"same tokens", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testLUSD, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser}},
		{"zero amount", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "0",
			Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser}},
		{"invalid type", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: "INVALID", ChainID: testChainID, Swapper: testUser}},
		{"bad chainId", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: 999, Swapper: testUser}},
		{"invalid swapper", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: "bad"}},
		{"slippage too high", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser, Slippage: 51}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := postJSON(mux, "/v1/trade/quote", tt.req)
			if w.Code != http.StatusBadRequest {
				t.Errorf("expected 400, got %d: %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestQuotePreferredVenue(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amount := new(big.Int).Mul(big.NewInt(1000), e18)

	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: amount.String(),
		Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser,
		Slippage: 0.5, PreferredVenue: "alpaca",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp QuoteResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Venues) != 1 || resp.Venues[0].Venue != "alpaca" {
		t.Errorf("expected single alpaca venue, got %v", resp.Venues)
	}
}

// ==========================================================================
// Swap tests — V4 calldata
// ==========================================================================

func TestSwapSingleHopV4(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	// Use preferredVenue to get a V4 quote (otherwise broker may win on price).
	amount := new(big.Int).Mul(big.NewInt(1000), e18)
	qw := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: amount.String(),
		Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser,
		Slippage: 0.5, PreferredVenue: "v4_native",
	})

	var qr QuoteResponse
	json.NewDecoder(qw.Body).Decode(&qr)
	if qr.Quote == nil {
		t.Fatal("expected quote")
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: qr.Quote, Swapper: testUser})
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", sw.Code, sw.Body.String())
	}

	var resp SwapResponse
	json.NewDecoder(sw.Body).Decode(&resp)

	if resp.Swap.To != LXRouterAddress {
		t.Errorf("to = %s, want %s", resp.Swap.To, LXRouterAddress)
	}
	if resp.Swap.ChainID != testChainID {
		t.Errorf("chainId = %d, want %d", resp.Swap.ChainID, testChainID)
	}

	selector, params, err := DecodeCalldata(resp.Swap.Data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(selector, SelectorExactInputSingle) {
		t.Errorf("selector = %x, want ExactInputSingle %x", selector, SelectorExactInputSingle)
	}

	tokenIn, _ := DecodeAddress(params, 0)
	tokenOut, _ := DecodeAddress(params, 1)
	if !strings.EqualFold(tokenIn, testLUSD) {
		t.Errorf("tokenIn = %s", tokenIn)
	}
	if !strings.EqualFold(tokenOut, testWBTC) {
		t.Errorf("tokenOut = %s", tokenOut)
	}
}

func TestSwapMultiHopV4(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amount := new(big.Int).Mul(big.NewInt(1000), e18)

	quote := &Quote{
		AmountIn: amount.String(), AmountOut: "16000000000000000000",
		Route: []Hop{
			{TokenIn: testLUSD, TokenOut: testWBTC, Venue: "v4_native", Fee: 30,
				AmountIn: amount.String(), AmountOut: "16600000000000000"},
			{TokenIn: testWBTC, TokenOut: testWETH, Venue: "v4_native", Fee: 5,
				AmountIn: "16600000000000000", AmountOut: "16000000000000000000"},
		},
		Fee: "35", ExecutionPrice: "16000000000000000000",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", sw.Code, sw.Body.String())
	}

	var resp SwapResponse
	json.NewDecoder(sw.Body).Decode(&resp)

	selector, params, _ := DecodeCalldata(resp.Swap.Data)
	if !bytes.Equal(selector, SelectorExactInput) {
		t.Errorf("selector = %x, want ExactInput %x", selector, SelectorExactInput)
	}

	pathLen, _ := DecodeUint256(params, 4)
	if pathLen.Int64() != 60 {
		t.Errorf("path length = %d, want 60 (3 addresses * 20 bytes)", pathLen.Int64())
	}
}

// ==========================================================================
// Swap tests — V2 calldata dispatch
// ==========================================================================

func TestSwapV2Dispatch(t *testing.T) {
	v2Router := "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D" // Uniswap V2 Router02
	cfg := Config{
		ChainIDs:       []int{testChainID},
		DefaultChainID: testChainID,
		VenueRouters:   map[string]string{"uniswap_v2": v2Router},
	}
	api := New(cfg, NewV4Venue(""))
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	quote := &Quote{
		AmountIn: "1000000000000000000", AmountOut: "500000000000000000",
		Route: []Hop{{
			TokenIn: testLUSD, TokenOut: testWBTC, Venue: "uniswap_v2",
			Fee: 30, AmountIn: "1000000000000000000", AmountOut: "500000000000000000",
		}},
		Fee: "30", ExecutionPrice: "500000000000000000",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", sw.Code, sw.Body.String())
	}

	var resp SwapResponse
	json.NewDecoder(sw.Body).Decode(&resp)

	// V2 swap should target the V2 router, not LXRouter.
	if !strings.EqualFold(resp.Swap.To, v2Router) {
		t.Errorf("to = %s, want V2 router %s", resp.Swap.To, v2Router)
	}

	// Verify V2 swapExactTokensForTokens selector.
	selector, _, err := DecodeCalldata(resp.Swap.Data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(selector, selectorV2SwapExact) {
		t.Errorf("selector = %x, want V2 swapExact %x", selector, selectorV2SwapExact)
	}
}

func TestSwapV2MissingRouter(t *testing.T) {
	// No VenueRouters configured — V2 swap should fail.
	cfg := Config{ChainIDs: []int{testChainID}, DefaultChainID: testChainID}
	api := New(cfg, NewV4Venue(""))
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	quote := &Quote{
		AmountIn: "1000", AmountOut: "500",
		Route: []Hop{{TokenIn: testLUSD, TokenOut: testWBTC, Venue: "uniswap_v2",
			Fee: 30, AmountIn: "1000", AmountOut: "500"}},
		Fee: "30", ExecutionPrice: "500",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 for missing V2 router, got %d", sw.Code)
	}
}

// ==========================================================================
// Swap tests — V3 calldata dispatch
// ==========================================================================

func TestSwapV3Dispatch(t *testing.T) {
	v3Router := "0xE592427A0AEce92De3Edee1F18E0157C05861564" // Uniswap V3 SwapRouter
	cfg := Config{
		ChainIDs:       []int{testChainID},
		DefaultChainID: testChainID,
		VenueRouters:   map[string]string{"uniswap_v3": v3Router},
	}
	api := New(cfg, NewV4Venue(""))
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	quote := &Quote{
		AmountIn: "1000000000000000000", AmountOut: "500000000000000000",
		Route: []Hop{{
			TokenIn: testLUSD, TokenOut: testWBTC, Venue: "uniswap_v3",
			Fee: 30, AmountIn: "1000000000000000000", AmountOut: "500000000000000000",
		}},
		Fee: "30", ExecutionPrice: "500000000000000000",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", sw.Code, sw.Body.String())
	}

	var resp SwapResponse
	json.NewDecoder(sw.Body).Decode(&resp)

	// V3 swap should target the V3 router.
	if !strings.EqualFold(resp.Swap.To, v3Router) {
		t.Errorf("to = %s, want V3 router %s", resp.Swap.To, v3Router)
	}

	// Verify V3 exactInputSingle selector.
	selector, _, err := DecodeCalldata(resp.Swap.Data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(selector, selectorV3ExactInputSingle) {
		t.Errorf("selector = %x, want V3 exactInputSingle %x", selector, selectorV3ExactInputSingle)
	}
}

// ==========================================================================
// Swap — off-chain venue rejected
// ==========================================================================

func TestSwapBrokerVenueRejected(t *testing.T) {
	_, mux := newTestAPI()

	quote := &Quote{
		AmountIn: "1000", AmountOut: "500",
		Route: []Hop{{TokenIn: testLUSD, TokenOut: testWBTC, Venue: "alpaca",
			Fee: 25, AmountIn: "1000", AmountOut: "500"}},
		Fee: "25", ExecutionPrice: "500",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 for off-chain venue swap, got %d", sw.Code)
	}
}

func TestSwapInvalidQuote(t *testing.T) {
	_, mux := newTestAPI()

	w := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: nil})
	if w.Code != http.StatusBadRequest {
		t.Errorf("nil quote: expected 400, got %d", w.Code)
	}

	w = postJSON(mux, "/v1/trade/swap", SwapRequest{
		Quote: &Quote{AmountIn: "1000", AmountOut: "500", Route: []Hop{}},
	})
	if w.Code != http.StatusBadRequest {
		t.Errorf("empty route: expected 400, got %d", w.Code)
	}
}

// ==========================================================================
// V2 calldata builder
// ==========================================================================

func TestBuildV2SwapCalldata(t *testing.T) {
	router := "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	calldata, err := BuildV2SwapCalldata(
		router,
		[]string{testLUSD, testWBTC},
		new(big.Int).Mul(big.NewInt(1000), e18),
		new(big.Int).Mul(big.NewInt(500), e18),
		testUser, nil,
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	sel, params, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, selectorV2SwapExact) {
		t.Errorf("selector = %x, want %x", sel, selectorV2SwapExact)
	}

	// Word 0 = amountIn, Word 1 = amountOutMin, Word 2 = offset to path.
	amountIn, _ := DecodeUint256(params, 0)
	if amountIn.Cmp(new(big.Int).Mul(big.NewInt(1000), e18)) != 0 {
		t.Errorf("amountIn = %s", amountIn)
	}
}

func TestBuildV2SwapCalldataMultiHop(t *testing.T) {
	router := "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

	calldata, err := BuildV2SwapCalldata(
		router,
		[]string{testLUSD, testWBTC, testWETH}, // 3-hop
		big.NewInt(1000), big.NewInt(500), testUser, nil,
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	// Verify selector.
	sel, _, _ := DecodeCalldata(calldata)
	if !bytes.Equal(sel, selectorV2SwapExact) {
		t.Errorf("selector mismatch")
	}
}

func TestBuildV2SwapCalldataPathTooShort(t *testing.T) {
	_, err := BuildV2SwapCalldata(
		"0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
		[]string{testLUSD}, big.NewInt(1000), big.NewInt(500), testUser, nil,
	)
	if err == nil {
		t.Error("expected error for single-element path")
	}
}

// ==========================================================================
// V3 calldata builder
// ==========================================================================

func TestBuildV3SwapCalldata(t *testing.T) {
	router := "0xE592427A0AEce92De3Edee1F18E0157C05861564"
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	calldata, err := BuildV3SwapCalldata(
		router, testLUSD, testWBTC, 3000, testUser,
		new(big.Int).Mul(big.NewInt(1000), e18),
		new(big.Int).Mul(big.NewInt(500), e18),
		nil,
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	sel, params, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, selectorV3ExactInputSingle) {
		t.Errorf("selector = %x, want %x", sel, selectorV3ExactInputSingle)
	}

	// V3 layout: tokenIn(0), tokenOut(1), fee(2), recipient(3), deadline(4),
	//            amountIn(5), amountOutMin(6), sqrtPriceLimit(7)
	tokenIn, _ := DecodeAddress(params, 0)
	tokenOut, _ := DecodeAddress(params, 1)
	fee, _ := DecodeUint256(params, 2)

	if !strings.EqualFold(tokenIn, testLUSD) {
		t.Errorf("tokenIn = %s", tokenIn)
	}
	if !strings.EqualFold(tokenOut, testWBTC) {
		t.Errorf("tokenOut = %s", tokenOut)
	}
	if fee.Int64() != 3000 {
		t.Errorf("fee = %d, want 3000", fee.Int64())
	}
}

// ==========================================================================
// V4 calldata builder (existing)
// ==========================================================================

func TestBuildV4ExactInputSingleCalldata(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amountIn := new(big.Int).Mul(big.NewInt(1000), e18)
	amountOutMin := new(big.Int).Mul(big.NewInt(500), e18)

	calldata, err := BuildExactInputSingleCalldata(
		testLUSD, testWBTC, amountIn, amountOutMin, big.NewInt(0),
		"0x0000000000000000000000000000000000000000000000000000000000000001",
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	// 4 + 6*32 = 196 bytes = 392 hex + "0x" = 394 chars.
	if len(calldata) != 394 {
		t.Errorf("calldata length = %d, want 394", len(calldata))
	}

	sel, params, _ := DecodeCalldata(calldata)
	if !bytes.Equal(sel, SelectorExactInputSingle) {
		t.Errorf("selector mismatch")
	}

	decoded, _ := DecodeAddress(params, 0)
	if !strings.EqualFold(decoded, testLUSD) {
		t.Errorf("tokenIn = %s", decoded)
	}
}

func TestBuildV4ExactInputCalldata(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	path := []string{testLUSD, testWBTC, testWETH}
	calldata, err := BuildExactInputCalldata(path,
		new(big.Int).Mul(big.NewInt(1000), e18),
		new(big.Int).Mul(big.NewInt(16), e18),
		big.NewInt(0),
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	sel, params, _ := DecodeCalldata(calldata)
	if !bytes.Equal(sel, SelectorExactInput) {
		t.Errorf("selector mismatch")
	}
	pathLen, _ := DecodeUint256(params, 4)
	if pathLen.Int64() != 60 {
		t.Errorf("path length = %d, want 60", pathLen.Int64())
	}
}

func TestBuildExactInputCalldataPathTooShort(t *testing.T) {
	_, err := BuildExactInputCalldata([]string{testLUSD}, big.NewInt(1000), big.NewInt(500), big.NewInt(0))
	if err == nil {
		t.Error("expected error for single-element path")
	}
}

func TestBuildExactOutputSingleCalldata(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	calldata, err := BuildExactOutputSingleCalldata(
		testLUSD, testWBTC,
		new(big.Int).Div(e18, big.NewInt(100)),
		new(big.Int).Mul(big.NewInt(700), e18),
		big.NewInt(0), "",
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	sel, _, _ := DecodeCalldata(calldata)
	if !bytes.Equal(sel, SelectorExactOutputSingle) {
		t.Errorf("selector = %x, want %x", sel, SelectorExactOutputSingle)
	}
}

// ==========================================================================
// Order tests
// ==========================================================================

func TestOrderCreation(t *testing.T) {
	_, mux := newTestAPI()

	w := postJSON(mux, "/v1/trade/order", OrderRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000000000000000000000",
		Type: QuoteTypeExactInput, Side: OrderSideBuy, OrderType: OrderTypeMarket,
		Swapper: testUser, ChainID: testChainID, Slippage: 0.5,
	})
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}

	var resp OrderResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.OrderID == "" {
		t.Fatal("expected non-empty orderId")
	}
	if resp.Status != OrderStatusPending {
		t.Errorf("status = %s, want PENDING", resp.Status)
	}
}

func TestOrderStatusLookup(t *testing.T) {
	_, mux := newTestAPI()

	w := postJSON(mux, "/v1/trade/order", OrderRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
		Type: QuoteTypeExactInput, Side: OrderSideBuy, OrderType: OrderTypeMarket,
		Swapper: testUser, ChainID: testChainID,
	})
	var or OrderResponse
	json.NewDecoder(w.Body).Decode(&or)

	sw := getPath(mux, "/v1/trade/orders/"+or.OrderID)
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", sw.Code)
	}

	var status OrderStatusResponse
	json.NewDecoder(sw.Body).Decode(&status)
	if status.OrderID != or.OrderID {
		t.Errorf("orderId mismatch")
	}
}

func TestOrderNotFound(t *testing.T) {
	_, mux := newTestAPI()
	w := getPath(mux, "/v1/trade/orders/nonexistent")
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestOrderValidation(t *testing.T) {
	_, mux := newTestAPI()

	tests := []struct {
		name string
		req  OrderRequest
	}{
		{"invalid side", OrderRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, Side: "INVALID", OrderType: OrderTypeMarket,
			Swapper: testUser, ChainID: testChainID}},
		{"limit without price", OrderRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, Side: OrderSideBuy, OrderType: OrderTypeLimit,
			Swapper: testUser, ChainID: testChainID}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := postJSON(mux, "/v1/trade/order", tt.req)
			if w.Code != http.StatusBadRequest {
				t.Errorf("expected 400, got %d", w.Code)
			}
		})
	}
}

// ==========================================================================
// Venue endpoints
// ==========================================================================

func TestVenuesEndpoint(t *testing.T) {
	_, mux := newTestAPI()
	w := getPath(mux, "/v1/trade/venues")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var venues []VenueInfo
	json.NewDecoder(w.Body).Decode(&venues)
	if len(venues) != 3 {
		t.Fatalf("expected 3 venues, got %d", len(venues))
	}

	names := map[string]bool{}
	for _, v := range venues {
		names[v.Name] = true
	}
	for _, n := range []string{"v4_native", "alpaca", "hyperliquid"} {
		if !names[n] {
			t.Errorf("missing venue: %s", n)
		}
	}
}

func TestVenueComparison(t *testing.T) {
	_, mux := newTestAPI()
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amount := new(big.Int).Mul(big.NewInt(100_000), e18)

	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: amount.String(),
		Type: QuoteTypeExactInput, ChainID: testChainID, Swapper: testUser, Slippage: 1.0,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp QuoteResponse
	json.NewDecoder(w.Body).Decode(&resp)

	// Venues should be sorted descending by amountOut.
	for i := 1; i < len(resp.Venues); i++ {
		prev, _ := new(big.Int).SetString(resp.Venues[i-1].AmountOut, 10)
		curr, _ := new(big.Int).SetString(resp.Venues[i].AmountOut, 10)
		if prev != nil && curr != nil && prev.Cmp(curr) < 0 {
			t.Errorf("venues not sorted: [%d]=%s < [%d]=%s",
				i-1, prev, i, curr)
		}
	}
}

// ==========================================================================
// Swap/approval status
// ==========================================================================

func TestSwapStatus(t *testing.T) {
	_, mux := newTestAPI()
	txHash := "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
	w := getPath(mux, "/v1/trade/swaps/"+txHash)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp SwapStatusResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.TxHash != txHash || resp.Status != "pending" {
		t.Errorf("unexpected response: %+v", resp)
	}
}

func TestSwapStatusInvalidHash(t *testing.T) {
	_, mux := newTestAPI()
	w := getPath(mux, "/v1/trade/swaps/not-a-hash")
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestCheckApproval(t *testing.T) {
	_, mux := newTestAPI()
	w := postJSON(mux, "/v1/trade/check-approval", ApprovalCheckRequest{
		TokenAddress: testLUSD, Owner: testUser, ChainID: testChainID,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp ApprovalCheckResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if !resp.Approved || resp.NeedsApproval {
		t.Errorf("unexpected: %+v", resp)
	}
}

// ==========================================================================
// Config tests
// ==========================================================================

func TestConfigMultipleChainIDs(t *testing.T) {
	cfg := Config{
		ChainIDs:       []int{testChainID, testChainID2, testChainID3},
		DefaultChainID: testChainID,
	}
	v4 := NewV4Venue("")
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	reserve := new(big.Int).Mul(big.NewInt(1_000_000), e18)
	v4.AddPool(testLUSD, testWBTC, "", 30, reserve, reserve)

	api := New(cfg, v4)
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	for _, id := range []int{testChainID, testChainID2, testChainID3} {
		w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
			Type: QuoteTypeExactInput, ChainID: id, Swapper: testUser,
		})
		if w.Code != http.StatusOK {
			t.Errorf("chainID %d: expected 200, got %d", id, w.Code)
		}
	}

	// Unknown chain rejected.
	w := postJSON(mux, "/v1/trade/quote", QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
		Type: QuoteTypeExactInput, ChainID: 1, Swapper: testUser,
	})
	if w.Code != http.StatusBadRequest {
		t.Errorf("unknown chain: expected 400, got %d", w.Code)
	}
}

func TestSwapUsesDefaultChainID(t *testing.T) {
	cfg := Config{
		ChainIDs:       []int{testChainID, testChainID2},
		DefaultChainID: testChainID2,
	}
	v4 := NewV4Venue("")
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	reserve := new(big.Int).Mul(big.NewInt(1_000_000), e18)
	v4.AddPool(testLUSD, testWBTC, "", 30, reserve, reserve)

	api := New(cfg, v4)
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	quote := &Quote{
		AmountIn: "1000", AmountOut: "500",
		Route: []Hop{{TokenIn: testLUSD, TokenOut: testWBTC, Venue: "v4_native",
			Fee: 30, AmountIn: "1000", AmountOut: "500"}},
		Fee: "30", ExecutionPrice: "500",
	}

	sw := postJSON(mux, "/v1/trade/swap", SwapRequest{Quote: quote, Swapper: testUser})
	if sw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", sw.Code)
	}

	var resp SwapResponse
	json.NewDecoder(sw.Body).Decode(&resp)
	if resp.Swap.ChainID != testChainID2 {
		t.Errorf("chainId = %d, want %d", resp.Swap.ChainID, testChainID2)
	}
}

// ==========================================================================
// NativeDEXVenue — mock RPC server
// ==========================================================================

func TestNativeDEXVenueQuote(t *testing.T) {
	// Mock EVM RPC that returns a fixed amountOut for any quoteExactInputSingle call.
	expectedOut := new(big.Int).Mul(big.NewInt(42), new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var result [32]byte
		outBytes := expectedOut.Bytes()
		copy(result[32-len(outBytes):], outBytes)

		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"result":  "0x" + hex.EncodeToString(result[:]),
		})
	}))
	defer srv.Close()

	venue := NewNativeDEXVenue(NativeDEXConfig{RPCURL: srv.URL, FeeBPS: 30})
	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000000000000000000",
		Type: QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote: %v", err)
	}
	if quote == nil {
		t.Fatal("expected non-nil quote")
	}
	if quote.AmountOut != expectedOut.String() {
		t.Errorf("amountOut = %s, want %s", quote.AmountOut, expectedOut)
	}
	if !quote.Executable {
		t.Error("native venue should be executable")
	}
	if quote.Venue != "v4_native" {
		t.Errorf("venue = %s, want v4_native", quote.Venue)
	}
}

func TestNativeDEXVenueNoLiquidity(t *testing.T) {
	// RPC returns error → no liquidity.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"error":   map[string]any{"code": -32000, "message": "execution reverted"},
		})
	}))
	defer srv.Close()

	venue := NewNativeDEXVenue(NativeDEXConfig{RPCURL: srv.URL})
	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000",
		Type: QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if quote != nil {
		t.Errorf("expected nil quote for no liquidity, got %+v", quote)
	}
}

// ==========================================================================
// UniswapV2Venue — mock RPC server
// ==========================================================================

func TestUniswapV2VenueQuote(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	expectedOut := new(big.Int).Mul(big.NewInt(997), e18)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// getAmountsOut returns: offset(32) + length(32) + amounts[0](32) + amounts[1](32)
		var resp [128]byte

		// offset = 32
		resp[31] = 32
		// length = 2
		resp[63] = 2
		// amounts[0] = amountIn (doesn't matter for this test)
		copy(resp[64:96], padUint256(new(big.Int).Mul(big.NewInt(1000), e18)))
		// amounts[1] = amountOut
		copy(resp[96:128], padUint256(expectedOut))

		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"result":  "0x" + hex.EncodeToString(resp[:]),
		})
	}))
	defer srv.Close()

	venue := NewUniswapV2Venue(UniswapV2Config{
		RPCURL:        srv.URL,
		RouterAddress: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
		Name:          "uniswap_v2",
	})

	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC,
		Amount: new(big.Int).Mul(big.NewInt(1000), e18).String(),
		Type:   QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote: %v", err)
	}
	if quote == nil {
		t.Fatal("expected non-nil quote")
	}
	if quote.AmountOut != expectedOut.String() {
		t.Errorf("amountOut = %s, want %s", quote.AmountOut, expectedOut)
	}
	if quote.Venue != "uniswap_v2" {
		t.Errorf("venue = %s", quote.Venue)
	}
	if !quote.Executable {
		t.Error("V2 venue should be executable")
	}
}

// ==========================================================================
// UniswapV3Venue — mock RPC server
// ==========================================================================

func TestUniswapV3VenueQuote(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	expectedOut := new(big.Int).Mul(big.NewInt(995), e18)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// quoteExactInputSingle returns single uint256.
		var resp [32]byte
		outBytes := expectedOut.Bytes()
		copy(resp[32-len(outBytes):], outBytes)

		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"result":  "0x" + hex.EncodeToString(resp[:]),
		})
	}))
	defer srv.Close()

	venue := NewUniswapV3Venue(UniswapV3Config{
		RPCURL:        srv.URL,
		QuoterAddress: "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
		RouterAddress: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
		Name:          "uniswap_v3",
		FeeTier:       3000,
	})

	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC,
		Amount: new(big.Int).Mul(big.NewInt(1000), e18).String(),
		Type:   QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote: %v", err)
	}
	if quote == nil {
		t.Fatal("expected non-nil quote")
	}
	if quote.AmountOut != expectedOut.String() {
		t.Errorf("amountOut = %s, want %s", quote.AmountOut, expectedOut)
	}
	if quote.Venue != "uniswap_v3" {
		t.Errorf("venue = %s", quote.Venue)
	}
	if quote.Fee != "30" { // 3000 / 100 = 30 bps
		t.Errorf("fee = %s, want 30", quote.Fee)
	}
}

// ==========================================================================
// BrokerHTTPVenue — mock HTTP server
// ==========================================================================

func TestBrokerHTTPVenueQuote(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/market/alpaca/snapshot/BTC-USD" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(brokerSnapshot{
			Symbol:   "BTC-USD",
			BidPrice: 59800.0,
			AskPrice: 60000.0,
		})
	}))
	defer srv.Close()

	// Register symbol mapping.
	RegisterSymbol(testLUSD, testWBTC, "BTC-USD")
	defer DefaultRegistry.Delete(testLUSD, testWBTC)

	venue := NewBrokerHTTPVenue(BrokerHTTPConfig{
		BrokerURL: srv.URL,
		Provider:  "alpaca",
	})

	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: testLUSD, TokenOut: testWBTC,
		Amount: new(big.Int).Mul(big.NewInt(1), e18).String(), // 1 LUSD
		Type:   QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote: %v", err)
	}
	if quote == nil {
		t.Fatal("expected non-nil quote")
	}

	// 1 LUSD at ask price 60000 = 60000 * 1e18 output
	amountOut, _ := new(big.Int).SetString(quote.AmountOut, 10)
	if amountOut == nil || amountOut.Sign() <= 0 {
		t.Fatalf("expected positive amountOut, got %s", quote.AmountOut)
	}
	if quote.Venue != "alpaca" {
		t.Errorf("venue = %s", quote.Venue)
	}
	if quote.Executable {
		t.Error("broker venue should NOT be executable")
	}
}

func TestBrokerHTTPVenueNoSymbol(t *testing.T) {
	venue := NewBrokerHTTPVenue(BrokerHTTPConfig{
		BrokerURL: "http://localhost:9999",
		Provider:  "kraken",
	})

	// No symbol registered for this pair → nil quote.
	quote, err := venue.Quote(context.Background(), QuoteRequest{
		TokenIn: "0x0000000000000000000000000000000000000099",
		TokenOut: "0x0000000000000000000000000000000000000098",
		Amount:  "1000", Type: QuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if quote != nil {
		t.Errorf("expected nil quote for unknown symbol")
	}
}

// ==========================================================================
// Venue type classification
// ==========================================================================

func TestVenueTypeClassification(t *testing.T) {
	tests := []struct {
		name   string
		isV2   bool
		isV3   bool
		isV4   bool
	}{
		{"v4_native", false, false, true},
		{"v4_native_clob", false, false, true},
		{"lux_amm", false, false, true},
		{"uniswap_v2", true, false, false},
		{"sushiswap", true, false, false},
		{"pancakeswap_v2", true, false, false},
		{"traderjoe", true, false, false},
		{"custom_v2", true, false, false},
		{"uniswap_v3", false, true, false},
		{"pancakeswap_v3", false, true, false},
		{"custom_v3", false, true, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isV2Venue(tt.name); got != tt.isV2 {
				t.Errorf("isV2Venue(%q) = %v, want %v", tt.name, got, tt.isV2)
			}
			if got := isV3Venue(tt.name); got != tt.isV3 {
				t.Errorf("isV3Venue(%q) = %v, want %v", tt.name, got, tt.isV3)
			}
			if got := isV4Venue(tt.name); got != tt.isV4 {
				t.Errorf("isV4Venue(%q) = %v, want %v", tt.name, got, tt.isV4)
			}
		})
	}
}

// ==========================================================================
// Math helpers
// ==========================================================================

func TestConstantProductQuote(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	reserveIn := new(big.Int).Mul(big.NewInt(1_000_000), e18)
	reserveOut := new(big.Int).Mul(big.NewInt(1_000_000), e18)
	amountIn := new(big.Int).Mul(big.NewInt(1000), e18)

	out := constantProductQuote(amountIn, reserveIn, reserveOut, 30)
	expected := new(big.Int).Mul(big.NewInt(996), e18)
	if out.Cmp(expected) < 0 {
		t.Errorf("output too low: got %s, expected >= %s", out, expected)
	}
}

func TestConstantProductInverse(t *testing.T) {
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	reserveIn := new(big.Int).Mul(big.NewInt(1_000_000), e18)
	reserveOut := new(big.Int).Mul(big.NewInt(1_000_000), e18)

	amountOut := new(big.Int).Mul(big.NewInt(996), e18)
	amountIn := constantProductInverse(amountOut, reserveIn, reserveOut, 30)

	maxIn := new(big.Int).Mul(big.NewInt(1000), e18)
	if amountIn.Cmp(maxIn) > 0 {
		t.Errorf("amountIn too high: %s > %s", amountIn, maxIn)
	}
	if amountIn.Sign() <= 0 {
		t.Error("amountIn should be positive")
	}
}

func TestIsValidAddress(t *testing.T) {
	tests := []struct {
		addr string
		want bool
	}{
		{testLUSD, true},
		{"0x0000000000000000000000000000000000000001", true},
		{"not-an-address", false},
		{"0x123", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isValidAddress(tt.addr); got != tt.want {
			t.Errorf("isValidAddress(%q) = %v, want %v", tt.addr, got, tt.want)
		}
	}
}

func TestIsPositiveDecimal(t *testing.T) {
	tests := []struct {
		s    string
		want bool
	}{
		{"1000", true}, {"1", true}, {"0", false}, {"", false}, {"abc", false},
	}
	for _, tt := range tests {
		if got := isPositiveDecimal(tt.s); got != tt.want {
			t.Errorf("isPositiveDecimal(%q) = %v, want %v", tt.s, got, tt.want)
		}
	}
}
