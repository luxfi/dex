package gateway

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

// ==========================================================================
// Approval Calldata Tests
// ==========================================================================

func TestBuildERC20ApproveCalldata(t *testing.T) {
	spender := "0x000000000022D473030F116dDEE9F6B43aC78BA3"
	amount := parseBigIntStr("1000000000000000000") // 1e18

	data := BuildERC20ApproveCalldata(spender, amount)

	// Check selector: 0x095ea7b3
	if hex.EncodeToString(data[:4]) != "095ea7b3" {
		t.Fatalf("wrong selector: got %s, want 095ea7b3", hex.EncodeToString(data[:4]))
	}

	// Check total length: 4 + 2*32 = 68
	if len(data) != 68 {
		t.Fatalf("wrong length: got %d, want 68", len(data))
	}

	// Spender should be in bytes [16:36] (left-padded address in first param slot)
	spenderHex := hex.EncodeToString(data[4:36])
	if spenderHex[:24] != "000000000000000000000000" {
		t.Fatalf("spender not left-padded: %s", spenderHex)
	}
	// Last 20 bytes should match permit2 address
	spenderAddr := hex.EncodeToString(data[16:36])
	expectedAddr := "000000000022d473030f116ddee9f6b43ac78ba3"
	if spenderAddr != expectedAddr {
		t.Fatalf("spender mismatch: got %s, want %s", spenderAddr, expectedAddr)
	}

	// Amount should be 1e18 = 0xde0b6b3a7640000
	amountSlot := data[36:68]
	amountHex := hex.EncodeToString(amountSlot)
	// Amount is right-aligned in 32 bytes
	if amountHex[len(amountHex)-16:] != "0de0b6b3a7640000" {
		t.Fatalf("amount mismatch: got %s", amountHex)
	}
}

func TestBuildERC20AllowanceCalldata(t *testing.T) {
	owner := "0x1111111111111111111111111111111111111111"
	spender := "0x2222222222222222222222222222222222222222"

	data := BuildERC20AllowanceCalldata(owner, spender)

	// Check selector: 0xdd62ed3e
	if hex.EncodeToString(data[:4]) != "dd62ed3e" {
		t.Fatalf("wrong selector: got %s, want dd62ed3e", hex.EncodeToString(data[:4]))
	}

	// Length: 4 + 2*32 = 68
	if len(data) != 68 {
		t.Fatalf("wrong length: got %d, want 68", len(data))
	}
}

func TestBuildPermit2ApproveCalldata(t *testing.T) {
	token := "0x1111111111111111111111111111111111111111"
	spender := "0x2222222222222222222222222222222222222222"
	amount := parseBigIntStr("1000000000000000000")
	expiration := int64(1893456000) // 2030-01-01

	data := BuildPermit2ApproveCalldata(token, spender, amount, expiration)

	// Check selector: 0x87517c45
	if hex.EncodeToString(data[:4]) != "87517c45" {
		t.Fatalf("wrong selector: got %s, want 87517c45", hex.EncodeToString(data[:4]))
	}

	// Length: 4 + 4*32 = 132
	if len(data) != 132 {
		t.Fatalf("wrong length: got %d, want 132", len(data))
	}
}

func TestBuildPermit2AllowanceCalldata(t *testing.T) {
	owner := "0x1111111111111111111111111111111111111111"
	token := "0x2222222222222222222222222222222222222222"
	spender := "0x3333333333333333333333333333333333333333"

	data := BuildPermit2AllowanceCalldata(owner, token, spender)

	// Check selector: 0x927da105
	if hex.EncodeToString(data[:4]) != "927da105" {
		t.Fatalf("wrong selector: got %s, want 927da105", hex.EncodeToString(data[:4]))
	}

	// Length: 4 + 3*32 = 100
	if len(data) != 100 {
		t.Fatalf("wrong length: got %d, want 100", len(data))
	}
}

// ==========================================================================
// Approval Tx Builder Tests
// ==========================================================================

func TestBuildApproveTx(t *testing.T) {
	req := ApprovalRequest{
		TokenAddress: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		Owner:        "0x1111111111111111111111111111111111111111",
		Spender:      "0x2222222222222222222222222222222222222222",
		Amount:       "1000000",
		ChainID:      96369,
	}

	tx, err := BuildApproveTx(req)
	if err != nil {
		t.Fatalf("BuildApproveTx failed: %v", err)
	}

	// Target is the token contract
	if tx.To != "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" {
		t.Fatalf("wrong to: got %s", tx.To)
	}

	// ChainID
	if tx.ChainID != 96369 {
		t.Fatalf("wrong chainId: got %d", tx.ChainID)
	}

	// Gas limit
	if tx.GasLimit != gasERC20Approve {
		t.Fatalf("wrong gasLimit: got %d, want %d", tx.GasLimit, gasERC20Approve)
	}

	// Value should be 0
	if tx.Value != "0" {
		t.Fatalf("wrong value: got %s, want 0", tx.Value)
	}

	// Data should start with 0x095ea7b3
	if len(tx.Data) < 10 {
		t.Fatalf("data too short: %s", tx.Data)
	}
	if tx.Data[:10] != "0x095ea7b3" {
		t.Fatalf("wrong selector in data: got %s", tx.Data[:10])
	}
}

func TestBuildPermit2ApproveTx(t *testing.T) {
	req := Permit2Request{
		TokenAddress: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		Owner:        "0x1111111111111111111111111111111111111111",
		Spender:      "0x2222222222222222222222222222222222222222",
		Amount:       "1000000",
		Deadline:     1893456000,
		ChainID:      96369,
	}

	tx, err := BuildPermit2ApproveTx(req)
	if err != nil {
		t.Fatalf("BuildPermit2ApproveTx failed: %v", err)
	}

	// Target is the Permit2 contract
	expectedTo := "0x000000000022d473030f116ddee9f6b43ac78ba3"
	if tx.To != expectedTo {
		t.Fatalf("wrong to: got %s, want %s", tx.To, expectedTo)
	}

	// Data should start with 0x87517c45
	if len(tx.Data) < 10 {
		t.Fatalf("data too short: %s", tx.Data)
	}
	if tx.Data[:10] != "0x87517c45" {
		t.Fatalf("wrong selector in data: got %s", tx.Data[:10])
	}
}

func TestBuildPermit2SignatureRequest(t *testing.T) {
	req := Permit2Request{
		TokenAddress: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		Owner:        "0x1111111111111111111111111111111111111111",
		Spender:      "0x2222222222222222222222222222222222222222",
		Amount:       "1000000",
		Deadline:     1893456000,
		ChainID:      96369,
	}

	sigReq, err := BuildPermit2SignatureRequest(req)
	if err != nil {
		t.Fatalf("BuildPermit2SignatureRequest failed: %v", err)
	}

	// EIP-712 domain
	if sigReq.Domain.Name != "Permit2" {
		t.Fatalf("wrong domain name: %s", sigReq.Domain.Name)
	}
	if sigReq.Domain.ChainID != "96369" {
		t.Fatalf("wrong domain chainId: %s", sigReq.Domain.ChainID)
	}
	if sigReq.Domain.VerifyingContract != Permit2Address {
		t.Fatalf("wrong domain verifyingContract: %s", sigReq.Domain.VerifyingContract)
	}

	// Primary type
	if sigReq.PrimaryType != "PermitSingle" {
		t.Fatalf("wrong primaryType: %s", sigReq.PrimaryType)
	}

	// Types must contain EIP712Domain, PermitSingle, PermitDetails
	if _, ok := sigReq.Types["EIP712Domain"]; !ok {
		t.Fatal("missing EIP712Domain type")
	}
	if _, ok := sigReq.Types["PermitSingle"]; !ok {
		t.Fatal("missing PermitSingle type")
	}
	if _, ok := sigReq.Types["PermitDetails"]; !ok {
		t.Fatal("missing PermitDetails type")
	}

	// Message: check spender is lowercased
	spender, ok := sigReq.Message["spender"].(string)
	if !ok || spender != "0x2222222222222222222222222222222222222222" {
		t.Fatalf("wrong spender in message: %v", sigReq.Message["spender"])
	}

	// Message: check details has token
	details, ok := sigReq.Message["details"].(map[string]interface{})
	if !ok {
		t.Fatal("details not a map")
	}
	token, ok := details["token"].(string)
	if !ok || token != "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" {
		t.Fatalf("wrong token in details: %v", details["token"])
	}
}

// ==========================================================================
// Approval Validation Tests
// ==========================================================================

func TestApprovalRequestValidation(t *testing.T) {
	valid := ApprovalRequest{
		TokenAddress: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		Owner:        "0x1111111111111111111111111111111111111111",
		Spender:      "0x2222222222222222222222222222222222222222",
		Amount:       "1000000",
		ChainID:      96369,
	}

	t.Run("valid", func(t *testing.T) {
		if err := valid.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("missing tokenAddress", func(t *testing.T) {
		req := valid
		req.TokenAddress = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("invalid tokenAddress", func(t *testing.T) {
		req := valid
		req.TokenAddress = "not_hex"
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("missing owner", func(t *testing.T) {
		req := valid
		req.Owner = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("missing spender", func(t *testing.T) {
		req := valid
		req.Spender = ""
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
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

	t.Run("negative amount", func(t *testing.T) {
		req := valid
		req.Amount = "-100"
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})
}

func TestPermit2RequestValidation(t *testing.T) {
	valid := Permit2Request{
		TokenAddress: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		Owner:        "0x1111111111111111111111111111111111111111",
		Spender:      "0x2222222222222222222222222222222222222222",
		Amount:       "1000000",
		Deadline:     1893456000,
		ChainID:      96369,
	}

	t.Run("valid", func(t *testing.T) {
		if err := valid.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("missing deadline", func(t *testing.T) {
		req := valid
		req.Deadline = 0
		if err := req.Validate(); err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("amount exceeds uint160", func(t *testing.T) {
		req := valid
		// 2^160 is too big for Permit2
		req.Amount = "1461501637330902918203684832716283019655932542976"
		if err := req.Validate(); err == nil {
			t.Fatal("expected error for amount exceeding uint160")
		}
	})
}

// ==========================================================================
// HTTP Handler Tests
// ==========================================================================

func newTestApprovalServer(t *testing.T) *Server {
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

func TestHandleApprovalBuild(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/trade/approval/build", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleApprovalBuild(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp apiResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if !resp.Success {
		t.Fatalf("expected success, got error: %s", resp.Error)
	}
}

func TestHandleApprovalBuild_InvalidMethod(t *testing.T) {
	s := newTestApprovalServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/trade/approval/build", nil)
	w := httptest.NewRecorder()
	s.handleApprovalBuild(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleApprovalBuild_InvalidBody(t *testing.T) {
	s := newTestApprovalServer(t)

	// Missing required fields
	body := `{"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/trade/approval/build", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleApprovalBuild(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

// The check answered allowance 0 and needsApproval on every request, without
// ever asking a chain. A wallet believing it sends an approve for a token it
// already approved, and the user pays gas for a state change that changes
// nothing, before every swap. So the test is what the chain says.
func TestApprovalCheckReadsTheAllowance(t *testing.T) {
	for _, c := range []struct {
		name    string
		holds   string // what allowance() returns
		wants   string // what the caller asked to move
		granted bool
	}{
		{"already approved for more than it needs", "5000000", "1000000", true},
		{"approved for exactly it", "1000000", "1000000", true},
		{"approved for less", "999999", "1000000", false},
		{"never approved", "0", "1000000", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newChainReadingServer(t, words(c.holds))

			w := ask(s, http.MethodPost, "/v1/trade/approval/check", map[string]any{
				"tokenAddress": testLUSD,
				"owner":        "0x1111111111111111111111111111111111111111",
				"spender":      "0x2222222222222222222222222222222222222222",
				"amount":       c.wants,
				"chainId":      uint64(ChainIDLux),
			})
			if w.Code != http.StatusOK {
				t.Fatalf("got %d: %s", w.Code, w.Body.String())
			}

			var got struct {
				Data ApprovalResponse `json:"data"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Fatalf("body: %v", err)
			}
			if got.Data.Allowance != c.holds {
				t.Errorf("allowance = %s, chain says %s", got.Data.Allowance, c.holds)
			}
			if got.Data.Approved != c.granted {
				t.Errorf("approved = %v, want %v on %s held against %s asked",
					got.Data.Approved, c.granted, c.holds, c.wants)
			}
			if got.Data.NeedsApproval == c.granted {
				t.Errorf("needsApproval = %v beside approved = %v", got.Data.NeedsApproval, got.Data.Approved)
			}
			// The tx rides along only when it is needed. Handed back beside
			// "you already approved" it is a tx somebody sends.
			if c.granted && got.Data.ApproveTx != nil {
				t.Error("an approve transaction returned to a wallet that has already approved")
			}
			if !c.granted && got.Data.ApproveTx == nil {
				t.Error("no approve transaction returned to a wallet that needs one")
			}
		})
	}
}

// A chain this deployment does not read is said so, rather than answered with
// a number nobody looked up.
func TestApprovalCheckWillNotGuessAtAChainItDoesNotRead(t *testing.T) {
	w := ask(newChainReadingServer(t, words("0")), http.MethodPost, "/v1/trade/approval/check", map[string]any{
		"tokenAddress": testLUSD,
		"owner":        "0x1111111111111111111111111111111111111111",
		"spender":      "0x2222222222222222222222222222222222222222",
		"amount":       "1",
		"chainId":      uint64(ChainIDEthereum),
	})
	if w.Code != http.StatusBadGateway {
		t.Fatalf("got %d, want 502: %s", w.Code, w.Body.String())
	}
}

func TestHandlePermit2Build_TxMode(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"deadline": 1893456000,
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/trade/permit2/build?mode=tx", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handlePermit2Build(w, req)

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

	// Should return a tx object
	data, ok := resp.Data.(map[string]interface{})
	if !ok {
		t.Fatal("data is not a map")
	}
	if data["to"] == nil {
		t.Fatal("expected 'to' in tx response")
	}
}

func TestHandlePermit2Build_SigMode(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"deadline": 1893456000,
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/trade/permit2/build?mode=sig", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handlePermit2Build(w, req)

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

	// Should return EIP-712 typed data
	data, ok := resp.Data.(map[string]interface{})
	if !ok {
		t.Fatal("data is not a map")
	}
	if data["primaryType"] == nil {
		t.Fatal("expected 'primaryType' in sig response")
	}
}

func TestHandlePermit2Build_InvalidMode(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"deadline": 1893456000,
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/trade/permit2/build?mode=invalid", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handlePermit2Build(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

// Permit2's allowance is (amount, expiration, nonce), and the expiration is
// half the answer: a grant that has run out is not a grant. Reading only the
// amount reports a wallet as ready to swap on a permit that expired.
func TestPermit2CheckReadsAmountAndExpiry(t *testing.T) {
	future := strconv.FormatInt(time.Now().Add(24*time.Hour).Unix(), 10)
	past := strconv.FormatInt(time.Now().Add(-time.Minute).Unix(), 10)

	for _, c := range []struct {
		name    string
		holds   string
		expires string
		granted bool
	}{
		{"enough and live", "5000000", future, true},
		{"enough but expired", "5000000", past, false},
		{"live but not enough", "1", future, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newChainReadingServer(t, words(c.holds, c.expires, "0"))

			w := ask(s, http.MethodPost, "/v1/trade/permit2/check", map[string]any{
				"tokenAddress": testLUSD,
				"owner":        "0x1111111111111111111111111111111111111111",
				"spender":      "0x2222222222222222222222222222222222222222",
				"amount":       "1000000",
				"deadline":     1893456000,
				"chainId":      uint64(ChainIDLux),
			})
			if w.Code != http.StatusOK {
				t.Fatalf("got %d: %s", w.Code, w.Body.String())
			}

			var got struct {
				Data Permit2Response `json:"data"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Fatalf("body: %v", err)
			}
			if got.Data.Permit2Allowance != c.holds {
				t.Errorf("allowance = %s, chain says %s", got.Data.Permit2Allowance, c.holds)
			}
			if got.Data.Permit2Approved != c.granted {
				t.Errorf("approved = %v, want %v", got.Data.Permit2Approved, c.granted)
			}
			if c.granted && got.Data.SignatureRequest != nil {
				t.Error("a signature request returned to a wallet whose permit is live")
			}
			if !c.granted && got.Data.SignatureRequest == nil {
				t.Error("no signature request returned to a wallet that needs one")
			}
		})
	}
}

// A chain that answers. `words` is what its eth_call returns, one 32-byte word
// per value, which is how every read in this file comes back.
func newChainReadingServer(t *testing.T, answer string) *Server {
	t.Helper()

	chain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     uint64 `json:"id"`
			Method string `json:"method"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Method != "eth_call" {
			t.Errorf("chain asked %q, want eth_call", req.Method)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0", "id": req.ID, "result": answer,
		})
	}))
	t.Cleanup(chain.Close)

	return NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(NewChainRouters(map[ChainID]ChainVenues{
			ChainIDLux: {RPC: chain.URL, Native: true},
		})),
	)
}

// words is an eth_call result: each value right-aligned in its own 32 bytes.
func words(values ...string) string {
	out := "0x"
	for _, v := range values {
		n, ok := new(big.Int).SetString(v, 10)
		if !ok {
			panic("not a number: " + v)
		}
		out += fmt.Sprintf("%064x", n)
	}
	return out
}
