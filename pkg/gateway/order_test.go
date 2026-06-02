package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// ==========================================================================
// Order Types & Validation Tests
// ==========================================================================

func TestOrderKindValidation(t *testing.T) {
	base := CreateOrderRequest{
		TokenIn:   "0x0000000000000000000000000000000000000001",
		TokenOut:  "0x0000000000000000000000000000000000000002",
		AmountIn:  "1000000",
		Recipient: "0x0000000000000000000000000000000000000003",
		Deadline:  time.Now().Add(1 * time.Hour).Unix(),
	}

	t.Run("invalid type", func(t *testing.T) {
		req := base
		req.Type = "market"
		if err := req.Validate(); err != ErrInvalidOrderKind {
			t.Fatalf("expected ErrInvalidOrderKind, got %v", err)
		}
	})

	t.Run("empty type", func(t *testing.T) {
		req := base
		req.Type = ""
		if err := req.Validate(); err != ErrInvalidOrderKind {
			t.Fatalf("expected ErrInvalidOrderKind, got %v", err)
		}
	})
}

func TestLimitOrderValidation(t *testing.T) {
	valid := CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000000000000000",
		LimitPrice: "2500000000000000000000",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	}

	t.Run("valid limit order", func(t *testing.T) {
		if err := valid.Validate(); err != nil {
			t.Fatalf("expected nil, got %v", err)
		}
	})

	t.Run("missing tokenIn", func(t *testing.T) {
		req := valid
		req.TokenIn = ""
		if err := req.Validate(); err != ErrMissingTokenIn {
			t.Fatalf("expected ErrMissingTokenIn, got %v", err)
		}
	})

	t.Run("missing tokenOut", func(t *testing.T) {
		req := valid
		req.TokenOut = ""
		if err := req.Validate(); err != ErrMissingTokenOut {
			t.Fatalf("expected ErrMissingTokenOut, got %v", err)
		}
	})

	t.Run("invalid address format", func(t *testing.T) {
		req := valid
		req.TokenIn = "not_an_address"
		if err := req.Validate(); err != ErrInvalidAddress {
			t.Fatalf("expected ErrInvalidAddress, got %v", err)
		}
	})

	t.Run("zero amountIn", func(t *testing.T) {
		req := valid
		req.AmountIn = "0"
		if err := req.Validate(); err != ErrMissingAmountIn {
			t.Fatalf("expected ErrMissingAmountIn, got %v", err)
		}
	})

	t.Run("negative amountIn", func(t *testing.T) {
		req := valid
		req.AmountIn = "-100"
		if err := req.Validate(); err != ErrMissingAmountIn {
			t.Fatalf("expected ErrMissingAmountIn, got %v", err)
		}
	})

	t.Run("expired deadline", func(t *testing.T) {
		req := valid
		req.Deadline = time.Now().Add(-1 * time.Minute).Unix()
		if err := req.Validate(); err != ErrInvalidDeadline {
			t.Fatalf("expected ErrInvalidDeadline, got %v", err)
		}
	})

	t.Run("missing limit price", func(t *testing.T) {
		req := valid
		req.LimitPrice = ""
		if err := req.Validate(); err != ErrMissingLimitPrice {
			t.Fatalf("expected ErrMissingLimitPrice, got %v", err)
		}
	})

	t.Run("zero limit price", func(t *testing.T) {
		req := valid
		req.LimitPrice = "0"
		if err := req.Validate(); err != ErrInvalidLimitPrice {
			t.Fatalf("expected ErrInvalidLimitPrice, got %v", err)
		}
	})

	t.Run("missing recipient", func(t *testing.T) {
		req := valid
		req.Recipient = ""
		if err := req.Validate(); err != ErrMissingRecipient {
			t.Fatalf("expected ErrMissingRecipient, got %v", err)
		}
	})
}

func TestDutchOrderValidation(t *testing.T) {
	now := time.Now()
	valid := CreateOrderRequest{
		Type:           OrderKindDutch,
		TokenIn:        "0x0000000000000000000000000000000000000001",
		TokenOut:       "0x0000000000000000000000000000000000000002",
		AmountIn:       "1000000000000000000",
		StartAmount:    "2500000000000000000000",
		EndAmount:      "2400000000000000000000",
		DecayStartTime: now.Add(1 * time.Minute).Unix(),
		DecayEndTime:   now.Add(10 * time.Minute).Unix(),
		Recipient:      "0x0000000000000000000000000000000000000003",
		Deadline:       now.Add(1 * time.Hour).Unix(),
	}

	t.Run("valid dutch order", func(t *testing.T) {
		if err := valid.Validate(); err != nil {
			t.Fatalf("expected nil, got %v", err)
		}
	})

	t.Run("missing startAmount", func(t *testing.T) {
		req := valid
		req.StartAmount = ""
		if err := req.Validate(); err != ErrMissingStartAmount {
			t.Fatalf("expected ErrMissingStartAmount, got %v", err)
		}
	})

	t.Run("missing endAmount", func(t *testing.T) {
		req := valid
		req.EndAmount = ""
		if err := req.Validate(); err != ErrMissingEndAmount {
			t.Fatalf("expected ErrMissingEndAmount, got %v", err)
		}
	})

	t.Run("missing decayStartTime", func(t *testing.T) {
		req := valid
		req.DecayStartTime = 0
		if err := req.Validate(); err != ErrMissingDecayStart {
			t.Fatalf("expected ErrMissingDecayStart, got %v", err)
		}
	})

	t.Run("missing decayEndTime", func(t *testing.T) {
		req := valid
		req.DecayEndTime = 0
		if err := req.Validate(); err != ErrMissingDecayEnd {
			t.Fatalf("expected ErrMissingDecayEnd, got %v", err)
		}
	})

	t.Run("decayEnd before decayStart", func(t *testing.T) {
		req := valid
		req.DecayEndTime = req.DecayStartTime - 1
		if err := req.Validate(); err != ErrInvalidDecayWindow {
			t.Fatalf("expected ErrInvalidDecayWindow, got %v", err)
		}
	})

	t.Run("startAmount less than endAmount", func(t *testing.T) {
		req := valid
		req.StartAmount = "100"
		req.EndAmount = "200"
		if err := req.Validate(); err != ErrInvalidDutchAmounts {
			t.Fatalf("expected ErrInvalidDutchAmounts, got %v", err)
		}
	})

	t.Run("equal start and end amounts allowed", func(t *testing.T) {
		req := valid
		req.StartAmount = "2500000000000000000000"
		req.EndAmount = "2500000000000000000000"
		if err := req.Validate(); err != nil {
			t.Fatalf("expected nil for equal start/end amounts, got %v", err)
		}
	})
}

func TestHexAddressValidation(t *testing.T) {
	tests := []struct {
		addr  string
		valid bool
	}{
		{"0x0000000000000000000000000000000000000001", true},
		{"0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF", true},
		{"0X0000000000000000000000000000000000000001", true},
		{"", false},
		{"0x", false},
		{"0x000000000000000000000000000000000000000", false},   // 39 chars
		{"0x00000000000000000000000000000000000000001", false}, // 41 chars
		{"0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", false},  // invalid hex
		{"000000000000000000000000000000000000000001", false},  // no 0x
	}

	for _, tt := range tests {
		t.Run(tt.addr, func(t *testing.T) {
			got := isValidHexAddress(tt.addr)
			if got != tt.valid {
				t.Fatalf("isValidHexAddress(%q) = %v, want %v", tt.addr, got, tt.valid)
			}
		})
	}
}

// ==========================================================================
// Order State Machine Tests
// ==========================================================================

func TestOrderStatusTransitions(t *testing.T) {
	tests := []struct {
		from  OrderStatus
		to    OrderStatus
		valid bool
	}{
		// Valid transitions from open
		{OrderStatusOpen, OrderStatusPartiallyFilled, true},
		{OrderStatusOpen, OrderStatusFilled, true},
		{OrderStatusOpen, OrderStatusExpired, true},
		{OrderStatusOpen, OrderStatusCancelled, true},

		// Valid transitions from partially_filled
		{OrderStatusPartiallyFilled, OrderStatusPartiallyFilled, true},
		{OrderStatusPartiallyFilled, OrderStatusFilled, true},
		{OrderStatusPartiallyFilled, OrderStatusExpired, true},
		{OrderStatusPartiallyFilled, OrderStatusCancelled, true},

		// Invalid: terminal states cannot transition
		{OrderStatusFilled, OrderStatusOpen, false},
		{OrderStatusFilled, OrderStatusCancelled, false},
		{OrderStatusExpired, OrderStatusOpen, false},
		{OrderStatusExpired, OrderStatusFilled, false},
		{OrderStatusCancelled, OrderStatusOpen, false},
		{OrderStatusCancelled, OrderStatusFilled, false},

		// Invalid: cannot go backwards
		{OrderStatusOpen, OrderStatusOpen, false},
		{OrderStatusPartiallyFilled, OrderStatusOpen, false},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s->%s", tt.from, tt.to)
		t.Run(name, func(t *testing.T) {
			err := ValidateTransition(tt.from, tt.to)
			if tt.valid && err != nil {
				t.Fatalf("expected valid transition, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Fatal("expected invalid transition, got nil")
			}
		})
	}
}

func TestIsTerminal(t *testing.T) {
	terminal := []OrderStatus{OrderStatusFilled, OrderStatusExpired, OrderStatusCancelled}
	nonTerminal := []OrderStatus{OrderStatusOpen, OrderStatusPartiallyFilled}

	for _, s := range terminal {
		if !s.IsTerminal() {
			t.Fatalf("expected %s to be terminal", s)
		}
	}
	for _, s := range nonTerminal {
		if s.IsTerminal() {
			t.Fatalf("expected %s to be non-terminal", s)
		}
	}
}

// ==========================================================================
// Order Store Tests
// ==========================================================================

func TestOrderStoreCreate(t *testing.T) {
	store := NewOrderStore()

	req := &CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000000000000000",
		LimitPrice: "2500000000000000000000",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
		ChainID:    96369,
	}

	order := store.Create(req)

	if order.ID == "" {
		t.Fatal("order ID should not be empty")
	}
	if order.Status != OrderStatusOpen {
		t.Fatalf("expected status open, got %s", order.Status)
	}
	if order.Kind != OrderKindLimit {
		t.Fatalf("expected kind limit, got %s", order.Kind)
	}
	if order.AmountIn.Cmp(big.NewInt(1000000000000000000)) != 0 {
		t.Fatalf("unexpected amountIn: %s", order.AmountIn)
	}
	if order.FilledAmount.Sign() != 0 {
		t.Fatal("filled amount should be 0")
	}
	if store.Count() != 1 {
		t.Fatalf("expected count 1, got %d", store.Count())
	}
}

func TestOrderStoreGet(t *testing.T) {
	store := NewOrderStore()

	req := &CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	}

	created := store.Create(req)
	got := store.Get(created.ID)

	if got == nil {
		t.Fatal("expected order, got nil")
	}
	if got.ID != created.ID {
		t.Fatalf("expected ID %s, got %s", created.ID, got.ID)
	}

	// Non-existent
	if store.Get("nonexistent") != nil {
		t.Fatal("expected nil for nonexistent order")
	}
}

func TestOrderStoreListByOwner(t *testing.T) {
	store := NewOrderStore()
	addr := "0x0000000000000000000000000000000000000003"

	// Create 3 orders for the same owner
	for i := 0; i < 3; i++ {
		store.Create(&CreateOrderRequest{
			Type:       OrderKindLimit,
			TokenIn:    "0x0000000000000000000000000000000000000001",
			TokenOut:   "0x0000000000000000000000000000000000000002",
			AmountIn:   "1000000",
			LimitPrice: "2500",
			Recipient:  addr,
			Deadline:   time.Now().Add(1 * time.Hour).Unix(),
		})
	}

	// Create 1 order for a different owner
	store.Create(&CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000099",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	})

	orders := store.ListByOwner(addr, "")
	if len(orders) != 3 {
		t.Fatalf("expected 3 orders, got %d", len(orders))
	}

	// Filter by status
	orders = store.ListByOwner(addr, OrderStatusOpen)
	if len(orders) != 3 {
		t.Fatalf("expected 3 open orders, got %d", len(orders))
	}

	orders = store.ListByOwner(addr, OrderStatusFilled)
	if len(orders) != 0 {
		t.Fatalf("expected 0 filled orders, got %d", len(orders))
	}

	// Case insensitivity
	orders = store.ListByOwner("0x0000000000000000000000000000000000000003", "")
	if len(orders) != 3 {
		t.Fatalf("expected 3 orders with uppercase addr, got %d", len(orders))
	}
}

func TestOrderStoreCancel(t *testing.T) {
	store := NewOrderStore()

	order := store.Create(&CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	})

	cancelled, err := store.Cancel(order.ID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cancelled.Status != OrderStatusCancelled {
		t.Fatalf("expected cancelled status, got %s", cancelled.Status)
	}

	// Cannot cancel again
	_, err = store.Cancel(order.ID)
	if err != ErrInvalidTransition {
		t.Fatalf("expected ErrInvalidTransition, got %v", err)
	}

	// Cannot cancel nonexistent
	_, err = store.Cancel("nonexistent")
	if err != ErrOrderNotFoundGateway {
		t.Fatalf("expected ErrOrderNotFoundGateway, got %v", err)
	}
}

func TestOrderStoreRecordFill(t *testing.T) {
	store := NewOrderStore()

	order := store.Create(&CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	})

	t.Run("partial fill", func(t *testing.T) {
		filled, err := store.RecordFill(order.ID, big.NewInt(300), big.NewInt(2500), "0xabc")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if filled.Status != OrderStatusPartiallyFilled {
			t.Fatalf("expected partially_filled, got %s", filled.Status)
		}
		if filled.FilledAmount.Cmp(big.NewInt(300)) != 0 {
			t.Fatalf("expected filled 300, got %s", filled.FilledAmount)
		}
		if len(filled.Fills) != 1 {
			t.Fatalf("expected 1 fill, got %d", len(filled.Fills))
		}
	})

	t.Run("complete fill", func(t *testing.T) {
		filled, err := store.RecordFill(order.ID, big.NewInt(700), big.NewInt(2500), "0xdef")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if filled.Status != OrderStatusFilled {
			t.Fatalf("expected filled, got %s", filled.Status)
		}
		if filled.FilledAmount.Cmp(big.NewInt(1000)) != 0 {
			t.Fatalf("expected filled 1000, got %s", filled.FilledAmount)
		}
		if len(filled.Fills) != 2 {
			t.Fatalf("expected 2 fills, got %d", len(filled.Fills))
		}
	})

	t.Run("fill after filled", func(t *testing.T) {
		_, err := store.RecordFill(order.ID, big.NewInt(1), big.NewInt(2500), "0xghi")
		if err != ErrOrderTerminal {
			t.Fatalf("expected ErrOrderTerminal, got %v", err)
		}
	})

	t.Run("fill exceeds remaining", func(t *testing.T) {
		order2 := store.Create(&CreateOrderRequest{
			Type:       OrderKindLimit,
			TokenIn:    "0x0000000000000000000000000000000000000001",
			TokenOut:   "0x0000000000000000000000000000000000000002",
			AmountIn:   "100",
			LimitPrice: "2500",
			Recipient:  "0x0000000000000000000000000000000000000003",
			Deadline:   time.Now().Add(1 * time.Hour).Unix(),
		})
		_, err := store.RecordFill(order2.ID, big.NewInt(101), big.NewInt(2500), "0x")
		if err != ErrFillExceedsRemaining {
			t.Fatalf("expected ErrFillExceedsRemaining, got %v", err)
		}
	})

	t.Run("fill nonexistent", func(t *testing.T) {
		_, err := store.RecordFill("nonexistent", big.NewInt(1), nil, "")
		if err != ErrOrderNotFoundGateway {
			t.Fatalf("expected ErrOrderNotFoundGateway, got %v", err)
		}
	})

	t.Run("fill with nil amount", func(t *testing.T) {
		_, err := store.RecordFill(order.ID, nil, nil, "")
		if err != ErrMissingAmountIn {
			t.Fatalf("expected ErrMissingAmountIn, got %v", err)
		}
	})
}

func TestOrderStoreExpireStale(t *testing.T) {
	store := NewOrderStore()

	// Create order with past deadline
	store.Create(&CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	})

	// Manually set a past deadline on the order
	store.mu.Lock()
	for _, order := range store.orders {
		order.Deadline = time.Now().Add(-1 * time.Minute).Unix()
	}
	store.mu.Unlock()

	expired := store.ExpireStale()
	if expired != 1 {
		t.Fatalf("expected 1 expired, got %d", expired)
	}

	// Running again should expire 0 (already terminal)
	expired = store.ExpireStale()
	if expired != 0 {
		t.Fatalf("expected 0 expired on second run, got %d", expired)
	}
}

// ==========================================================================
// HTTP Handler Tests
// ==========================================================================

func newTestServer(t *testing.T) *Server {
	t.Helper()
	registry := NewRegistry()
	router := NewRouter(registry, true)
	cfg := DefaultServerConfig()
	s := NewServer(router, cfg)
	return s
}

func TestCreateLimitOrderHandler(t *testing.T) {
	s := newTestServer(t)

	body := CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000000000000000000",
		LimitPrice: "2500000000000000000000",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
		ChainID:    96369,
	}

	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleOrder(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp apiResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if !resp.Success {
		t.Fatalf("expected success, got error: %s", resp.Error)
	}

	// Decode the data field
	dataBytes, _ := json.Marshal(resp.Data)
	var orderResp CreateOrderResponse
	json.Unmarshal(dataBytes, &orderResp)

	if orderResp.OrderID == "" {
		t.Fatal("expected order ID in response")
	}
	if orderResp.Status != OrderStatusOpen {
		t.Fatalf("expected status open, got %s", orderResp.Status)
	}
	if orderResp.Tx == nil {
		t.Fatal("expected unsigned tx for limit order")
	}
	if orderResp.EIP712 != nil {
		t.Fatal("did not expect EIP712 data for limit order")
	}
	if orderResp.Tx.To != "0x0000000000000000000000000000000000009020" {
		t.Fatalf("expected CLOB address, got %s", orderResp.Tx.To)
	}
	if orderResp.Tx.ChainID != 96369 {
		t.Fatalf("expected chainId 96369, got %d", orderResp.Tx.ChainID)
	}
}

func TestCreateDutchOrderHandler(t *testing.T) {
	s := newTestServer(t)
	now := time.Now()

	body := CreateOrderRequest{
		Type:           OrderKindDutch,
		TokenIn:        "0x0000000000000000000000000000000000000001",
		TokenOut:       "0x0000000000000000000000000000000000000002",
		AmountIn:       "1000000000000000000",
		StartAmount:    "2500000000000000000000",
		EndAmount:      "2400000000000000000000",
		DecayStartTime: now.Add(1 * time.Minute).Unix(),
		DecayEndTime:   now.Add(10 * time.Minute).Unix(),
		Recipient:      "0x0000000000000000000000000000000000000003",
		Deadline:       now.Add(1 * time.Hour).Unix(),
		ChainID:        96369,
	}

	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleOrder(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp apiResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)

	dataBytes, _ := json.Marshal(resp.Data)
	var orderResp CreateOrderResponse
	json.Unmarshal(dataBytes, &orderResp)

	if orderResp.Tx != nil {
		t.Fatal("did not expect unsigned tx for dutch order")
	}
	if orderResp.EIP712 == nil {
		t.Fatal("expected EIP712 data for dutch order")
	}
	if orderResp.EIP712.PrimaryType != "DutchOrder" {
		t.Fatalf("expected primary type DutchOrder, got %s", orderResp.EIP712.PrimaryType)
	}
	if orderResp.EIP712.Domain.Name != "LuxDutchOrderReactor" {
		t.Fatalf("expected domain name LuxDutchOrderReactor, got %s", orderResp.EIP712.Domain.Name)
	}
	if orderResp.EIP712.Domain.ChainId != "96369" {
		t.Fatalf("expected chainId 96369, got %s", orderResp.EIP712.Domain.ChainId)
	}

	// Verify message fields
	msg := orderResp.EIP712.Message
	if msg["inputToken"] != "0x0000000000000000000000000000000000000001" {
		t.Fatalf("unexpected inputToken: %v", msg["inputToken"])
	}
	if msg["outputToken"] != "0x0000000000000000000000000000000000000002" {
		t.Fatalf("unexpected outputToken: %v", msg["outputToken"])
	}
	if msg["nonce"] == "" || msg["nonce"] == nil {
		t.Fatal("expected nonce in message")
	}
}

func TestCreateOrderValidationError(t *testing.T) {
	s := newTestServer(t)

	// Invalid JSON
	req := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader([]byte("{")))
	rr := httptest.NewRecorder()
	s.handleOrder(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid JSON, got %d", rr.Code)
	}

	// Missing required fields
	b, _ := json.Marshal(CreateOrderRequest{Type: OrderKindLimit})
	req = httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	rr = httptest.NewRecorder()
	s.handleOrder(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing fields, got %d", rr.Code)
	}
}

func TestListOrdersHandler(t *testing.T) {
	s := newTestServer(t)
	addr := "0x0000000000000000000000000000000000000003"

	// Create 2 orders
	for i := 0; i < 2; i++ {
		body := CreateOrderRequest{
			Type:       OrderKindLimit,
			TokenIn:    "0x0000000000000000000000000000000000000001",
			TokenOut:   "0x0000000000000000000000000000000000000002",
			AmountIn:   "1000",
			LimitPrice: "2500",
			Recipient:  addr,
			Deadline:   time.Now().Add(1 * time.Hour).Unix(),
		}
		b, _ := json.Marshal(body)
		req := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
		rr := httptest.NewRecorder()
		s.handleOrder(rr, req)
		if rr.Code != http.StatusCreated {
			t.Fatalf("failed to create order: %d", rr.Code)
		}
	}

	// List orders
	req := httptest.NewRequest(http.MethodGet, "/v1/order?address="+addr, nil)
	rr := httptest.NewRecorder()
	s.handleOrder(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp apiResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)

	dataBytes, _ := json.Marshal(resp.Data)
	var listResp OrderListResponse
	json.Unmarshal(dataBytes, &listResp)

	if listResp.Total != 2 {
		t.Fatalf("expected 2 orders, got %d", listResp.Total)
	}

	// List with status filter
	req = httptest.NewRequest(http.MethodGet, "/v1/order?address="+addr+"&status=filled", nil)
	rr = httptest.NewRecorder()
	s.handleOrder(rr, req)
	json.Unmarshal(rr.Body.Bytes(), &resp)
	dataBytes, _ = json.Marshal(resp.Data)
	json.Unmarshal(dataBytes, &listResp)

	if listResp.Total != 0 {
		t.Fatalf("expected 0 filled orders, got %d", listResp.Total)
	}
}

func TestListOrdersMissingAddress(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/order", nil)
	rr := httptest.NewRecorder()
	s.handleOrder(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestListOrdersInvalidStatusFilter(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/order?address=0x0000000000000000000000000000000000000003&status=bogus", nil)
	rr := httptest.NewRecorder()
	s.handleOrder(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestGetOrderHandler(t *testing.T) {
	s := newTestServer(t)

	// Create an order
	body := CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	}
	b, _ := json.Marshal(body)
	createReq := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	createRR := httptest.NewRecorder()
	s.handleOrder(createRR, createReq)

	var createResp apiResponse
	json.Unmarshal(createRR.Body.Bytes(), &createResp)
	dataBytes, _ := json.Marshal(createResp.Data)
	var orderResp CreateOrderResponse
	json.Unmarshal(dataBytes, &orderResp)

	// Get the order
	req := httptest.NewRequest(http.MethodGet, "/v1/order/"+orderResp.OrderID, nil)
	rr := httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	// Get nonexistent
	req = httptest.NewRequest(http.MethodGet, "/v1/order/nonexistent", nil)
	rr = httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rr.Code)
	}
}

func TestCancelOrderHandler(t *testing.T) {
	s := newTestServer(t)

	// Create a limit order
	body := CreateOrderRequest{
		Type:       OrderKindLimit,
		TokenIn:    "0x0000000000000000000000000000000000000001",
		TokenOut:   "0x0000000000000000000000000000000000000002",
		AmountIn:   "1000",
		LimitPrice: "2500",
		Recipient:  "0x0000000000000000000000000000000000000003",
		Deadline:   time.Now().Add(1 * time.Hour).Unix(),
	}
	b, _ := json.Marshal(body)
	createReq := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	createRR := httptest.NewRecorder()
	s.handleOrder(createRR, createReq)

	var createResp apiResponse
	json.Unmarshal(createRR.Body.Bytes(), &createResp)
	dataBytes, _ := json.Marshal(createResp.Data)
	var orderResp CreateOrderResponse
	json.Unmarshal(dataBytes, &orderResp)

	// Cancel the order
	req := httptest.NewRequest(http.MethodDelete, "/v1/order/"+orderResp.OrderID, nil)
	rr := httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var cancelResp apiResponse
	json.Unmarshal(rr.Body.Bytes(), &cancelResp)
	cancelData, _ := json.Marshal(cancelResp.Data)
	var cancelOrder CancelOrderResponse
	json.Unmarshal(cancelData, &cancelOrder)

	if cancelOrder.Status != OrderStatusCancelled {
		t.Fatalf("expected cancelled, got %s", cancelOrder.Status)
	}
	if cancelOrder.Tx == nil {
		t.Fatal("expected cancel tx for limit order")
	}

	// Cancel again should fail
	req = httptest.NewRequest(http.MethodDelete, "/v1/order/"+orderResp.OrderID, nil)
	rr = httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("expected 409 for double cancel, got %d", rr.Code)
	}
}

func TestCancelDutchOrderHandler(t *testing.T) {
	s := newTestServer(t)
	now := time.Now()

	// Create a dutch order
	body := CreateOrderRequest{
		Type:           OrderKindDutch,
		TokenIn:        "0x0000000000000000000000000000000000000001",
		TokenOut:       "0x0000000000000000000000000000000000000002",
		AmountIn:       "1000",
		StartAmount:    "2500",
		EndAmount:      "2400",
		DecayStartTime: now.Add(1 * time.Minute).Unix(),
		DecayEndTime:   now.Add(10 * time.Minute).Unix(),
		Recipient:      "0x0000000000000000000000000000000000000003",
		Deadline:       now.Add(1 * time.Hour).Unix(),
	}
	b, _ := json.Marshal(body)
	createReq := httptest.NewRequest(http.MethodPost, "/v1/order", bytes.NewReader(b))
	createRR := httptest.NewRecorder()
	s.handleOrder(createRR, createReq)

	var createResp apiResponse
	json.Unmarshal(createRR.Body.Bytes(), &createResp)
	dataBytes, _ := json.Marshal(createResp.Data)
	var orderResp CreateOrderResponse
	json.Unmarshal(dataBytes, &orderResp)

	// Cancel the dutch order — should NOT return an unsigned tx
	req := httptest.NewRequest(http.MethodDelete, "/v1/order/"+orderResp.OrderID, nil)
	rr := httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var cancelResp apiResponse
	json.Unmarshal(rr.Body.Bytes(), &cancelResp)
	cancelData, _ := json.Marshal(cancelResp.Data)
	var cancelOrder CancelOrderResponse
	json.Unmarshal(cancelData, &cancelOrder)

	if cancelOrder.Tx != nil {
		t.Fatal("did not expect cancel tx for dutch order (off-chain cancellation)")
	}
}

func TestCancelNonexistentOrder(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodDelete, "/v1/order/nonexistent", nil)
	rr := httptest.NewRecorder()
	s.handleOrderByID(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rr.Code)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	s := newTestServer(t)

	// PUT on /v1/order
	req := httptest.NewRequest(http.MethodPut, "/v1/order", nil)
	rr := httptest.NewRecorder()
	s.handleOrder(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rr.Code)
	}

	// POST on /v1/order/{id}
	req = httptest.NewRequest(http.MethodPost, "/v1/order/someid", nil)
	rr = httptest.NewRecorder()
	s.handleOrderByID(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rr.Code)
	}
}
