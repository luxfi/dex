package gateway

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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

	req := httptest.NewRequest(http.MethodPost, "/v1/approval/build", bytes.NewBufferString(body))
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

	req := httptest.NewRequest(http.MethodGet, "/v1/approval/build", nil)
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
	req := httptest.NewRequest(http.MethodPost, "/v1/approval/build", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleApprovalBuild(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleApprovalCheck(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/approval/check", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleApprovalCheck(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp apiResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Should always indicate needs approval (no RPC access)
	data, ok := resp.Data.(map[string]interface{})
	if !ok {
		t.Fatal("data is not a map")
	}
	if needsApproval, _ := data["needsApproval"].(bool); !needsApproval {
		t.Fatal("expected needsApproval=true")
	}
	if data["approveTx"] == nil {
		t.Fatal("expected approveTx in response")
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

	req := httptest.NewRequest(http.MethodPost, "/v1/permit2/build?mode=tx", bytes.NewBufferString(body))
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

	req := httptest.NewRequest(http.MethodPost, "/v1/permit2/build?mode=sig", bytes.NewBufferString(body))
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

	req := httptest.NewRequest(http.MethodPost, "/v1/permit2/build?mode=invalid", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handlePermit2Build(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandlePermit2Check(t *testing.T) {
	s := newTestApprovalServer(t)

	body := `{
		"tokenAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
		"owner": "0x1111111111111111111111111111111111111111",
		"spender": "0x2222222222222222222222222222222222222222",
		"amount": "1000000",
		"deadline": 1893456000,
		"chainId": 96369
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/permit2/check", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handlePermit2Check(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp apiResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	data, ok := resp.Data.(map[string]interface{})
	if !ok {
		t.Fatal("data is not a map")
	}
	if data["signatureRequest"] == nil {
		t.Fatal("expected signatureRequest in response")
	}
}
