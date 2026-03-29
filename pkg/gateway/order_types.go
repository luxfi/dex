package gateway

import (
	"errors"
	"math/big"
	"time"
)

// OrderKind distinguishes order execution strategies.
type OrderKind string

const (
	OrderKindLimit OrderKind = "limit"
	OrderKindDutch OrderKind = "dutch"
)

// OrderStatus tracks the lifecycle of an order.
type OrderStatus string

const (
	OrderStatusOpen            OrderStatus = "open"
	OrderStatusPartiallyFilled OrderStatus = "partially_filled"
	OrderStatusFilled          OrderStatus = "filled"
	OrderStatusExpired         OrderStatus = "expired"
	OrderStatusCancelled       OrderStatus = "cancelled"
)

// validTransitions defines the allowed state machine transitions.
// Any transition not in this map is illegal and will be rejected.
var validTransitions = map[OrderStatus][]OrderStatus{
	OrderStatusOpen:            {OrderStatusPartiallyFilled, OrderStatusFilled, OrderStatusExpired, OrderStatusCancelled},
	OrderStatusPartiallyFilled: {OrderStatusPartiallyFilled, OrderStatusFilled, OrderStatusExpired, OrderStatusCancelled},
	// Terminal states: no transitions out.
	OrderStatusFilled:    {},
	OrderStatusExpired:   {},
	OrderStatusCancelled: {},
}

// Order errors
var (
	ErrInvalidOrderKind     = errors.New("invalid order kind: must be 'limit' or 'dutch'")
	ErrMissingTokenIn       = errors.New("tokenIn is required")
	ErrMissingTokenOut      = errors.New("tokenOut is required")
	ErrMissingAmountIn      = errors.New("amountIn is required and must be positive")
	ErrMissingRecipient     = errors.New("recipient address is required")
	ErrInvalidDeadline      = errors.New("deadline must be in the future")
	ErrMissingLimitPrice    = errors.New("limitPrice is required for limit orders")
	ErrInvalidLimitPrice    = errors.New("limitPrice must be positive")
	ErrMissingStartAmount   = errors.New("startAmount is required for dutch orders")
	ErrMissingEndAmount     = errors.New("endAmount is required for dutch orders")
	ErrMissingDecayStart    = errors.New("decayStartTime is required for dutch orders")
	ErrMissingDecayEnd      = errors.New("decayEndTime is required for dutch orders")
	ErrInvalidDecayWindow   = errors.New("decayEndTime must be after decayStartTime")
	ErrInvalidDutchAmounts  = errors.New("startAmount must be >= endAmount (price decays)")
	ErrOrderNotFoundGateway = errors.New("order not found")
	ErrInvalidTransition    = errors.New("invalid order status transition")
	ErrOrderTerminal        = errors.New("order is in terminal state")
	ErrFillExceedsRemaining = errors.New("fill amount exceeds remaining order amount")
	ErrInvalidAddress       = errors.New("invalid address format")
)

// CreateOrderRequest is the JSON body for POST /v1/order.
type CreateOrderRequest struct {
	Type     OrderKind `json:"type"`
	TokenIn  string    `json:"tokenIn"`
	TokenOut string    `json:"tokenOut"`
	AmountIn string    `json:"amountIn"`
	ChainID  uint64    `json:"chainId"`

	// Limit order fields
	LimitPrice string `json:"limitPrice,omitempty"`

	// Dutch auction fields
	StartAmount    string `json:"startAmount,omitempty"`
	EndAmount      string `json:"endAmount,omitempty"`
	DecayStartTime int64  `json:"decayStartTime,omitempty"`
	DecayEndTime   int64  `json:"decayEndTime,omitempty"`

	// Common fields
	Deadline  int64  `json:"deadline"`
	Recipient string `json:"recipient"`
}

// CreateOrderResponse is returned from POST /v1/order.
type CreateOrderResponse struct {
	OrderID   string             `json:"orderId"`
	Status    OrderStatus        `json:"status"`
	Tx        *TxData            `json:"tx,omitempty"`        // unsigned tx for limit orders
	EIP712    *DutchOrderEIP712  `json:"eip712,omitempty"`    // typed data for dutch orders
	CreatedAt time.Time          `json:"createdAt"`
}

// TxData is an unsigned transaction envelope.
type TxData struct {
	To       string `json:"to"`
	Data     string `json:"data"`
	Value    string `json:"value"`
	ChainID  uint64 `json:"chainId"`
	GasLimit uint64 `json:"gasLimit"`
}

// EIP712Domain is the EIP-712 domain separator, shared across Permit2 and order signing.
type EIP712Domain struct {
	Name              string `json:"name"`
	ChainID           string `json:"chainId"`
	VerifyingContract string `json:"verifyingContract"`
}

// EIP712Field describes a single field in an EIP-712 struct.
type EIP712Field struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// DutchOrderEIP712 contains the typed data structure for off-chain Dutch order signing.
// Uses DutchOrderEIP712Domain which adds a version field to the domain separator,
// matching the UniswapX reactor ABI.
type DutchOrderEIP712 struct {
	Types       map[string][]EIP712Field   `json:"types"`
	PrimaryType string                     `json:"primaryType"`
	Domain      DutchOrderEIP712Domain     `json:"domain"`
	Message     map[string]interface{}     `json:"message"`
}

// DutchOrderEIP712Domain is the domain separator for Dutch auction orders.
// Extends EIP712Domain with a version field, matching the UniswapX reactor contract.
type DutchOrderEIP712Domain struct {
	Name              string `json:"name"`
	Version           string `json:"version"`
	ChainId           string `json:"chainId"`
	VerifyingContract string `json:"verifyingContract"`
}

// GatewayOrder is the internal representation of an order.
type GatewayOrder struct {
	ID        string      `json:"orderId"`
	Kind      OrderKind   `json:"type"`
	Status    OrderStatus `json:"status"`
	Owner     string      `json:"owner"`
	TokenIn   string      `json:"tokenIn"`
	TokenOut  string      `json:"tokenOut"`
	AmountIn  *big.Int    `json:"amountIn"`
	ChainID   ChainID     `json:"chainId"`
	Deadline  int64       `json:"deadline"`
	Recipient string      `json:"recipient"`
	CreatedAt time.Time   `json:"createdAt"`
	UpdatedAt time.Time   `json:"updatedAt"`

	// Limit-specific
	LimitPrice *big.Int `json:"limitPrice,omitempty"`

	// Dutch-specific
	StartAmount    *big.Int `json:"startAmount,omitempty"`
	EndAmount      *big.Int `json:"endAmount,omitempty"`
	DecayStartTime int64    `json:"decayStartTime,omitempty"`
	DecayEndTime   int64    `json:"decayEndTime,omitempty"`

	// Fill tracking
	FilledAmount *big.Int    `json:"filledAmount"`
	Fills        []OrderFill `json:"fills"`
}

// OrderFill records a single partial or full fill event.
type OrderFill struct {
	Amount    *big.Int  `json:"amount"`
	Price     *big.Int  `json:"price,omitempty"`
	TxHash    string    `json:"txHash,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// CancelOrderResponse is returned from DELETE /v1/order/{orderId}.
type CancelOrderResponse struct {
	OrderID string      `json:"orderId"`
	Status  OrderStatus `json:"status"`
	Tx      *TxData     `json:"tx,omitempty"` // unsigned cancel tx for limit orders
}

// OrderListResponse wraps a list of orders for the API.
type OrderListResponse struct {
	Orders []GatewayOrder `json:"orders"`
	Total  int            `json:"total"`
}

// IsTerminal returns true if the order is in a terminal state.
func (s OrderStatus) IsTerminal() bool {
	return s == OrderStatusFilled || s == OrderStatusExpired || s == OrderStatusCancelled
}

// ValidateTransition checks whether transitioning from current to next is allowed.
func ValidateTransition(current, next OrderStatus) error {
	allowed, ok := validTransitions[current]
	if !ok {
		return ErrInvalidTransition
	}
	for _, s := range allowed {
		if s == next {
			return nil
		}
	}
	return ErrInvalidTransition
}

// Validate performs server-side validation of a CreateOrderRequest.
// All amounts are validated as positive big.Int strings.
// Deadline is checked against the current time.
func (r *CreateOrderRequest) Validate() error {
	// Kind
	if r.Type != OrderKindLimit && r.Type != OrderKindDutch {
		return ErrInvalidOrderKind
	}

	// Required fields
	if r.TokenIn == "" {
		return ErrMissingTokenIn
	}
	if r.TokenOut == "" {
		return ErrMissingTokenOut
	}
	if r.Recipient == "" {
		return ErrMissingRecipient
	}

	// Validate recipient is hex address (0x + 40 hex chars)
	if !isValidHexAddress(r.Recipient) {
		return ErrInvalidAddress
	}
	if !isValidHexAddress(r.TokenIn) {
		return ErrInvalidAddress
	}
	if !isValidHexAddress(r.TokenOut) {
		return ErrInvalidAddress
	}

	// AmountIn: must parse to positive big.Int
	amountIn := parseBigIntStr(r.AmountIn)
	if amountIn == nil || amountIn.Sign() <= 0 {
		return ErrMissingAmountIn
	}

	// Deadline: must be in the future
	if r.Deadline <= time.Now().Unix() {
		return ErrInvalidDeadline
	}

	// Kind-specific validation
	switch r.Type {
	case OrderKindLimit:
		return r.validateLimit()
	case OrderKindDutch:
		return r.validateDutch()
	}

	return nil
}

func (r *CreateOrderRequest) validateLimit() error {
	price := parseBigIntStr(r.LimitPrice)
	if price == nil {
		return ErrMissingLimitPrice
	}
	if price.Sign() <= 0 {
		return ErrInvalidLimitPrice
	}
	return nil
}

func (r *CreateOrderRequest) validateDutch() error {
	startAmt := parseBigIntStr(r.StartAmount)
	if startAmt == nil || startAmt.Sign() <= 0 {
		return ErrMissingStartAmount
	}

	endAmt := parseBigIntStr(r.EndAmount)
	if endAmt == nil || endAmt.Sign() <= 0 {
		return ErrMissingEndAmount
	}

	if r.DecayStartTime <= 0 {
		return ErrMissingDecayStart
	}
	if r.DecayEndTime <= 0 {
		return ErrMissingDecayEnd
	}
	if r.DecayEndTime <= r.DecayStartTime {
		return ErrInvalidDecayWindow
	}

	// startAmount must be >= endAmount (output decays for the swapper)
	if startAmt.Cmp(endAmt) < 0 {
		return ErrInvalidDutchAmounts
	}

	return nil
}

// isValidHexAddress checks for 0x-prefixed 40-char hex string.
func isValidHexAddress(addr string) bool {
	if len(addr) != 42 {
		return false
	}
	if addr[:2] != "0x" && addr[:2] != "0X" {
		return false
	}
	for _, c := range addr[2:] {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}
