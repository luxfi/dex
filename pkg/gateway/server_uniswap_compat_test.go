package gateway

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTradingTestServer creates a Server wired to a mock provider for testing.
func newTradingTestServer() *Server {
	registry := NewRegistry()
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	registry.RegisterProvider(provider)
	router := NewRouter(registry, true)
	cfg := DefaultServerConfig()
	return NewServer(router, cfg)
}

func tradingPost(s *Server, path string, body interface{}) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.mux.ServeHTTP(w, req)
	return w
}

func tradingGet(s *Server, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	s.mux.ServeHTTP(w, req)
	return w
}

func TestTradingQuote(t *testing.T) {
	s := newTradingTestServer()

	w := tradingPost(s, "/trading/quote", UniQuoteRequest{
		Type:           "EXACT_INPUT",
		Amount:         "1000000000000000000",
		TokenIn:        "0x1111111111111111111111111111111111111111",
		TokenOut:       "0x2222222222222222222222222222222222222222",
		TokenInChainId: int(ChainIDLux),
		Swapper:        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify X-Request-Id header is present
	if w.Header().Get("X-Request-Id") == "" {
		t.Error("expected X-Request-Id header")
	}

	var resp UniQuoteResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.RequestId == "" {
		t.Error("expected non-empty requestId")
	}
	if resp.Routing != "CLASSIC" {
		t.Errorf("expected routing CLASSIC, got %s", resp.Routing)
	}
	if resp.PermitData != nil {
		t.Error("expected nil permitData on Lux chain")
	}

	// Verify the classic quote
	var classic UniClassicQuote
	if err := json.Unmarshal(resp.Quote, &classic); err != nil {
		t.Fatalf("failed to decode classic quote: %v", err)
	}
	if classic.Input.Token != "0x1111111111111111111111111111111111111111" {
		t.Errorf("unexpected input token: %s", classic.Input.Token)
	}
	if classic.Output.Amount == "0" || classic.Output.Amount == "" {
		t.Error("expected non-zero output amount")
	}
	if classic.QuoteId == "" {
		t.Error("expected non-empty quoteId")
	}
	if classic.Swapper != "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		t.Errorf("expected swapper to be echoed, got %s", classic.Swapper)
	}
	if len(classic.Route) == 0 {
		t.Error("expected at least one route")
	}
}

func TestTradingQuoteMissingFields(t *testing.T) {
	s := newTradingTestServer()

	// Missing tokenIn
	w := tradingPost(s, "/trading/quote", UniQuoteRequest{
		Amount:   "1000",
		TokenOut: "0x2222222222222222222222222222222222222222",
	})
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing tokenIn, got %d", w.Code)
	}

	// Missing amount
	w = tradingPost(s, "/trading/quote", UniQuoteRequest{
		TokenIn:  "0x1111111111111111111111111111111111111111",
		TokenOut: "0x2222222222222222222222222222222222222222",
	})
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing amount, got %d", w.Code)
	}
}

func TestTradingQuoteMethodNotAllowed(t *testing.T) {
	s := newTradingTestServer()
	w := tradingGet(s, "/trading/quote")
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestTradingSwap(t *testing.T) {
	s := newTradingTestServer()

	// First get a quote
	qw := tradingPost(s, "/trading/quote", UniQuoteRequest{
		Type:           "EXACT_INPUT",
		Amount:         "1000000000000000000",
		TokenIn:        "0x1111111111111111111111111111111111111111",
		TokenOut:       "0x2222222222222222222222222222222222222222",
		TokenInChainId: int(ChainIDLux),
		Swapper:        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if qw.Code != http.StatusOK {
		t.Fatalf("quote failed: %d", qw.Code)
	}

	var quoteResp UniQuoteResponse
	json.Unmarshal(qw.Body.Bytes(), &quoteResp)

	// Now build swap
	w := tradingPost(s, "/trading/swap", UniCreateSwapRequest{
		Quote: quoteResp.Quote,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniCreateSwapResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Type != "TRANSACTION" {
		t.Errorf("expected type TRANSACTION, got %s", resp.Type)
	}
	if resp.Swap == nil {
		t.Fatal("expected swap transaction data")
	}
	if resp.Swap.To != LXPoolPrecompile {
		t.Errorf("expected to=%s, got %s", LXPoolPrecompile, resp.Swap.To)
	}
	if resp.Swap.Data == "" || resp.Swap.Data == "0x" {
		t.Error("expected non-empty calldata")
	}
}

func TestTradingCheckApproval(t *testing.T) {
	s := newTradingTestServer()

	w := tradingPost(s, "/trading/check_approval", UniApprovalRequest{
		WalletAddress: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Token:         "0x1111111111111111111111111111111111111111",
		Amount:        "1000000000000000000",
		ChainId:       int(ChainIDLux),
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniApprovalResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.Approval == nil {
		t.Fatal("expected approval transaction")
	}
	if resp.Approval.To != "0x1111111111111111111111111111111111111111" {
		t.Errorf("expected approval to target the token contract, got %s", resp.Approval.To)
	}
	// Verify the calldata starts with approve selector 0x095ea7b3
	data := resp.Approval.Data
	if len(data) < 10 || data[:10] != "0x095ea7b3" {
		t.Errorf("expected approve selector 0x095ea7b3, got %s", data[:10])
	}
}

func TestTradingCheckApprovalMaxUint(t *testing.T) {
	s := newTradingTestServer()

	// No amount means max uint256 approval
	w := tradingPost(s, "/trading/check_approval", UniApprovalRequest{
		WalletAddress: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Token:         "0x1111111111111111111111111111111111111111",
		ChainId:       int(ChainIDLux),
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp UniApprovalResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	// Decode the amount from calldata (bytes 36-68)
	dataHex := resp.Approval.Data[2:] // strip 0x
	dataBytes, _ := hex.DecodeString(dataHex)
	amountBytes := dataBytes[36:68]
	amount := new(big.Int).SetBytes(amountBytes)
	maxUint256, _ := new(big.Int).SetString("115792089237316195423570985008687907853269984665640564039457584007913129639935", 10)
	if amount.Cmp(maxUint256) != 0 {
		t.Errorf("expected max uint256 amount, got %s", amount.String())
	}
}

func TestTradingSend(t *testing.T) {
	s := newTradingTestServer()

	w := tradingPost(s, "/trading/send", UniSendRequest{
		WalletAddress: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Token:         "0x1111111111111111111111111111111111111111",
		Amount:        "1000000",
		Recipient:     "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		ChainId:       int(ChainIDLux),
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniSendResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.Type != "TRANSACTION" {
		t.Errorf("expected type TRANSACTION, got %s", resp.Type)
	}
	if resp.Tx == nil {
		t.Fatal("expected transaction data")
	}
	// ERC20 transfer targets the token contract
	if resp.Tx.To != "0x1111111111111111111111111111111111111111" {
		t.Errorf("expected to=token, got %s", resp.Tx.To)
	}
	// Verify transfer selector 0xa9059cbb
	if len(resp.Tx.Data) < 10 || resp.Tx.Data[:10] != "0xa9059cbb" {
		t.Errorf("expected transfer selector 0xa9059cbb, got %s", resp.Tx.Data[:10])
	}
}

func TestTradingSendNative(t *testing.T) {
	s := newTradingTestServer()

	w := tradingPost(s, "/trading/send", UniSendRequest{
		WalletAddress: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Token:         "0x0000000000000000000000000000000000000000",
		Amount:        "1000000000000000000",
		Recipient:     "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		ChainId:       int(ChainIDLux),
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniSendResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	// Native send goes directly to recipient, not a token contract
	if resp.Tx.To != "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" {
		t.Errorf("expected to=recipient for native send, got %s", resp.Tx.To)
	}
	// Native send has no ERC20 calldata
	if resp.Tx.Data != "0x" {
		t.Errorf("expected empty calldata for native send, got %s", resp.Tx.Data)
	}
	// Value should be the amount
	if resp.Tx.Value == "0x0" || resp.Tx.Value == "" {
		t.Error("expected non-zero value for native send")
	}
	if resp.Tx.GasLimit != "21000" {
		t.Errorf("expected gasLimit 21000 for native send, got %s", resp.Tx.GasLimit)
	}
}

func TestTradingGetOrders(t *testing.T) {
	s := newTradingTestServer()

	w := tradingGet(s, "/trading/orders?swapper=0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniGetOrdersResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.Orders == nil {
		t.Error("expected non-nil orders array")
	}
}

func TestTradingGetOrdersMissingSwapper(t *testing.T) {
	s := newTradingTestServer()
	w := tradingGet(s, "/trading/orders")
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing swapper, got %d", w.Code)
	}
}

func TestTradingGetSwaps(t *testing.T) {
	s := newTradingTestServer()

	w := tradingGet(s, "/trading/swaps?txHash=0xabcdef&chainId=96369")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniGetSwapsResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if len(resp.Swaps) != 1 {
		t.Fatalf("expected 1 swap, got %d", len(resp.Swaps))
	}
	if resp.Swaps[0].TxHash != "0xabcdef" {
		t.Errorf("expected txHash 0xabcdef, got %s", resp.Swaps[0].TxHash)
	}
	if resp.Swaps[0].Status != "pending" {
		t.Errorf("expected status pending, got %s", resp.Swaps[0].Status)
	}
}

func TestTradingGetSwapsMissingTxHash(t *testing.T) {
	s := newTradingTestServer()
	w := tradingGet(s, "/trading/swaps")
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing txHash, got %d", w.Code)
	}
}

func TestTradingSwappableTokens(t *testing.T) {
	s := newTradingTestServer()

	w := tradingGet(s, "/trading/swappable_tokens?tokenAddress=0x1111111111111111111111111111111111111111&chainId=96369")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniSwappableTokensResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	// When queried from Lux chain, should include Zoo chain
	found := false
	for _, tok := range resp.Tokens {
		if tok.ChainId == int(ChainIDZoo) {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected Zoo chain in swappable tokens")
	}
}

func TestTradingLPPoolInfo(t *testing.T) {
	s := newTradingTestServer()

	w := tradingPost(s, "/trading/lp/pool_info", UniPoolInfoRequest{
		ChainId:  int(ChainIDLux),
		Protocol: "V4",
		Token0:   "0x1111111111111111111111111111111111111111",
		Token1:   "0x2222222222222222222222222222222222222222",
		Fee:      3000,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniPoolInfoResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.PoolAddress == "" {
		t.Error("expected non-empty pool address")
	}
	if resp.Fee != 3000 {
		t.Errorf("expected fee 3000, got %d", resp.Fee)
	}
}

func TestTradingOrder(t *testing.T) {
	s := newTradingTestServer()

	quote := UniClassicQuote{
		Input:   UniTokenAmount{Token: "0x1111111111111111111111111111111111111111", Amount: "1000000", ChainId: int(ChainIDLux)},
		Output:  UniTokenAmount{Token: "0x2222222222222222222222222222222222222222", Amount: "2000000", ChainId: int(ChainIDLux)},
		Swapper: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		QuoteId: "test-quote-123",
	}
	quoteBytes, _ := json.Marshal(quote)

	w := tradingPost(s, "/trading/order", UniOrderRequest{
		Signature: "0xdeadbeef",
		Quote:     json.RawMessage(quoteBytes),
		Routing:   "DUTCH_V2",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp UniOrderResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.OrderId == "" {
		t.Error("expected non-empty orderId")
	}
}

func TestTradingErrorResponseFormat(t *testing.T) {
	s := newTradingTestServer()

	// Send invalid JSON
	req := httptest.NewRequest(http.MethodPost, "/trading/quote", bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}

	var errResp UniErrorResponse
	body, _ := io.ReadAll(w.Body)
	if err := json.Unmarshal(body, &errResp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	if errResp.Detail == "" {
		t.Error("expected non-empty error detail")
	}
	if errResp.ErrorCode != "INVALID_REQUEST" {
		t.Errorf("expected errorCode INVALID_REQUEST, got %s", errResp.ErrorCode)
	}
}

func TestBuildApproveCalldata(t *testing.T) {
	amount := new(big.Int).SetUint64(1000000)
	spender := "0x0000000000000000000000000000000000009010"
	data := buildApproveCalldata(spender, amount)

	// Check selector
	if data[0] != 0x09 || data[1] != 0x5e || data[2] != 0xa7 || data[3] != 0xb3 {
		t.Errorf("wrong selector: %x", data[:4])
	}

	// Check length: 4 + 32 + 32 = 68
	if len(data) != 68 {
		t.Errorf("expected 68 bytes, got %d", len(data))
	}

	// Verify spender is in the right position (bytes 4-36, right-aligned 20 bytes)
	spenderBytes := data[4+12 : 4+32]
	expectedAddr := addrBytes(spender)
	for i := range expectedAddr {
		if spenderBytes[i] != expectedAddr[i] {
			t.Errorf("spender mismatch at byte %d", i)
			break
		}
	}
}

func TestBuildTransferCalldata(t *testing.T) {
	amount := new(big.Int).SetUint64(500000)
	to := "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	data := buildTransferCalldata(to, amount)

	// Check selector
	if data[0] != 0xa9 || data[1] != 0x05 || data[2] != 0x9c || data[3] != 0xbb {
		t.Errorf("wrong selector: %x", data[:4])
	}

	if len(data) != 68 {
		t.Errorf("expected 68 bytes, got %d", len(data))
	}
}

func TestPoolTypeToUni(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"v2", "v2-pool"},
		{"V2", "v2-pool"},
		{"uniswap_v2", "v2-pool"},
		{"v3", "v3-pool"},
		{"uniswap_v3", "v3-pool"},
		{"v4", "v4-pool"},
		{"lux_amm", "v4-pool"},
		{"lux_v4", "v4-pool"},
		{"unknown", "v4-pool"},
	}

	for _, tt := range tests {
		got := poolTypeToUni(tt.input)
		if got != tt.want {
			t.Errorf("poolTypeToUni(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestBigIntStr(t *testing.T) {
	if bigIntStr(nil) != "0" {
		t.Error("expected nil -> 0")
	}
	if bigIntStr(big.NewInt(42)) != "42" {
		t.Errorf("expected 42, got %s", bigIntStr(big.NewInt(42)))
	}
}

func TestGatewayOrderToUniStatus(t *testing.T) {
	order := &GatewayOrder{
		ID:      "test-123",
		Status:  OrderStatusFilled,
		Owner:   "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		ChainID: ChainIDLux,
	}

	uni := gatewayOrderToUniStatus(order)
	if uni.OrderStatus != "filled" {
		t.Errorf("expected filled, got %s", uni.OrderStatus)
	}
	if uni.OrderId != "test-123" {
		t.Errorf("expected test-123, got %s", uni.OrderId)
	}
}

func TestNoApiResponseWrapper(t *testing.T) {
	s := newTradingTestServer()

	w := tradingGet(s, "/trading/swaps?txHash=0xabc&chainId=96369")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	// The response should NOT have the apiResponse wrapper (no "success" field)
	var raw map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &raw)

	if _, ok := raw["success"]; ok {
		t.Error("trading endpoints must NOT use the apiResponse wrapper")
	}
	if _, ok := raw["swaps"]; !ok {
		t.Error("expected 'swaps' field in response")
	}
}
