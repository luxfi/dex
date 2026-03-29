package gateway

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strings"

	"github.com/luxfi/crypto"
)

// orderManager holds the order store and configuration for the order endpoints.
// Attached to Server so handlers can access it.
type orderManager struct {
	store *OrderStore

	// Precompile addresses for building unsigned transactions
	clobAddress string // LXBook precompile for limit orders

	// Dutch auction reactor address for EIP-712 domain
	reactorAddress string

	// Default chain ID when not specified
	defaultChainID ChainID
}

// newOrderManager creates an orderManager with production defaults.
func newOrderManager(defaultChainID ChainID) *orderManager {
	return &orderManager{
		store:          NewOrderStore(),
		clobAddress:    "0x0000000000000000000000000000000000009020", // LXBook
		reactorAddress: "0x0000000000000000000000000000000000009021", // DutchOrderReactor
		defaultChainID: defaultChainID,
	}
}

// registerOrderRoutes registers order lifecycle routes on the mux.
func (s *Server) registerOrderRoutes() {
	s.mux.HandleFunc("/v1/order", s.handleOrder)
	s.mux.HandleFunc("/v1/order/", s.handleOrderByID)
}

// handleOrder dispatches POST (create) and GET (list) on /v1/order.
func (s *Server) handleOrder(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handleCreateOrder(w, r)
	case http.MethodGet:
		s.handleListOrders(w, r)
	case http.MethodOptions:
		w.WriteHeader(http.StatusNoContent)
	default:
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
	}
}

// handleOrderByID dispatches GET and DELETE on /v1/order/{orderId}.
func (s *Server) handleOrderByID(w http.ResponseWriter, r *http.Request) {
	orderID := strings.TrimPrefix(r.URL.Path, "/v1/order/")
	if orderID == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("order ID is required"))
		return
	}

	switch r.Method {
	case http.MethodGet:
		s.handleGetOrder(w, r, orderID)
	case http.MethodDelete:
		s.handleCancelOrder(w, r, orderID)
	case http.MethodOptions:
		w.WriteHeader(http.StatusNoContent)
	default:
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
	}
}

// handleCreateOrder handles POST /v1/order.
func (s *Server) handleCreateOrder(w http.ResponseWriter, r *http.Request) {
	// Limit request body size to 1MB
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req CreateOrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	if err := req.Validate(); err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	// Default chain ID if not specified
	if req.ChainID == 0 {
		req.ChainID = uint64(s.orders.defaultChainID)
	}

	order := s.orders.store.Create(&req)

	resp := CreateOrderResponse{
		OrderID:   order.ID,
		Status:    order.Status,
		CreatedAt: order.CreatedAt,
	}

	switch order.Kind {
	case OrderKindLimit:
		resp.Tx = s.buildLimitOrderTx(order)
	case OrderKindDutch:
		resp.EIP712 = s.buildDutchOrderEIP712(order)
	}

	s.writeJSON(w, http.StatusCreated, resp)
}

// handleListOrders handles GET /v1/order?address=...&status=...
func (s *Server) handleListOrders(w http.ResponseWriter, r *http.Request) {
	address := r.URL.Query().Get("address")
	if address == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("address query parameter is required"))
		return
	}
	if !isValidHexAddress(address) {
		s.writeError(w, http.StatusBadRequest, ErrInvalidAddress)
		return
	}

	statusFilter := OrderStatus(r.URL.Query().Get("status"))
	// Validate status filter if provided
	if statusFilter != "" {
		switch statusFilter {
		case OrderStatusOpen, OrderStatusPartiallyFilled, OrderStatusFilled, OrderStatusExpired, OrderStatusCancelled:
			// valid
		default:
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid status filter: must be open, partially_filled, filled, expired, or cancelled"))
			return
		}
	}

	orders := s.orders.store.ListByOwner(address, statusFilter)
	if orders == nil {
		orders = make([]GatewayOrder, 0)
	}

	s.writeJSON(w, http.StatusOK, OrderListResponse{
		Orders: orders,
		Total:  len(orders),
	})
}

// handleGetOrder handles GET /v1/order/{orderId}.
func (s *Server) handleGetOrder(w http.ResponseWriter, _ *http.Request, orderID string) {
	order := s.orders.store.Get(orderID)
	if order == nil {
		s.writeError(w, http.StatusNotFound, ErrOrderNotFoundGateway)
		return
	}

	s.writeJSON(w, http.StatusOK, order)
}

// handleCancelOrder handles DELETE /v1/order/{orderId}.
func (s *Server) handleCancelOrder(w http.ResponseWriter, _ *http.Request, orderID string) {
	order := s.orders.store.Get(orderID)
	if order == nil {
		s.writeError(w, http.StatusNotFound, ErrOrderNotFoundGateway)
		return
	}

	cancelled, err := s.orders.store.Cancel(orderID)
	if err != nil {
		s.writeError(w, http.StatusConflict, err)
		return
	}

	resp := CancelOrderResponse{
		OrderID: cancelled.ID,
		Status:  cancelled.Status,
	}

	// For limit orders, return an unsigned cancel transaction
	if cancelled.Kind == OrderKindLimit {
		resp.Tx = s.buildCancelLimitOrderTx(cancelled)
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// buildLimitOrderTx constructs an unsigned transaction to place a limit order
// via the LXBook (CLOB) precompile.
func (s *Server) buildLimitOrderTx(order *GatewayOrder) *TxData {
	// Function selector: placeLimitOrder(address tokenIn, address tokenOut, uint256 amountIn, uint256 limitPrice, uint64 deadline, address recipient)
	selector := crypto.Keccak256([]byte("placeLimitOrder(address,address,uint256,uint256,uint64,address)"))[:4]

	data := make([]byte, 0, 4+6*32)
	data = append(data, selector...)

	// ABI encode parameters — each padded to 32 bytes
	data = append(data, padAddress(order.TokenIn)...)
	data = append(data, padAddress(order.TokenOut)...)
	data = append(data, padBigInt(order.AmountIn)...)
	data = append(data, padBigInt(order.LimitPrice)...)
	data = append(data, padUint64(uint64(order.Deadline))...)
	data = append(data, padAddress(order.Recipient)...)

	return &TxData{
		To:       s.orders.clobAddress,
		Data:     "0x" + hex.EncodeToString(data),
		Value:    "0",
		ChainID:  uint64(order.ChainID),
		GasLimit: 200000,
	}
}

// buildCancelLimitOrderTx constructs an unsigned transaction to cancel a limit order
// via the LXBook precompile.
func (s *Server) buildCancelLimitOrderTx(order *GatewayOrder) *TxData {
	// Function selector: cancelOrder(bytes32 orderId)
	selector := crypto.Keccak256([]byte("cancelOrder(bytes32)"))[:4]

	data := make([]byte, 0, 4+32)
	data = append(data, selector...)
	data = append(data, padBytes32FromString(order.ID)...)

	return &TxData{
		To:       s.orders.clobAddress,
		Data:     "0x" + hex.EncodeToString(data),
		Value:    "0",
		ChainID:  uint64(order.ChainID),
		GasLimit: 100000,
	}
}

// buildDutchOrderEIP712 constructs the EIP-712 typed data for a Dutch auction order.
// This follows the UniswapX DutchOrder format: the user signs this off-chain,
// and a solver submits the fill transaction when profitable.
func (s *Server) buildDutchOrderEIP712(order *GatewayOrder) *DutchOrderEIP712 {
	return &DutchOrderEIP712{
		Types: map[string][]EIP712Field{
			"EIP712Domain": {
				{Name: "name", Type: "string"},
				{Name: "version", Type: "string"},
				{Name: "chainId", Type: "uint256"},
				{Name: "verifyingContract", Type: "address"},
			},
			"DutchOrder": {
				{Name: "reactor", Type: "address"},
				{Name: "swapper", Type: "address"},
				{Name: "nonce", Type: "uint256"},
				{Name: "deadline", Type: "uint256"},
				{Name: "inputToken", Type: "address"},
				{Name: "inputAmount", Type: "uint256"},
				{Name: "outputToken", Type: "address"},
				{Name: "outputStartAmount", Type: "uint256"},
				{Name: "outputEndAmount", Type: "uint256"},
				{Name: "decayStartTime", Type: "uint256"},
				{Name: "decayEndTime", Type: "uint256"},
			},
		},
		PrimaryType: "DutchOrder",
		Domain: DutchOrderEIP712Domain{
			Name:              "LuxDutchOrderReactor",
			Version:           "1",
			ChainId:           fmt.Sprintf("%d", order.ChainID),
			VerifyingContract: s.orders.reactorAddress,
		},
		Message: map[string]interface{}{
			"reactor":           s.orders.reactorAddress,
			"swapper":           order.Recipient,
			"nonce":             orderIDToNonce(order.ID),
			"deadline":          fmt.Sprintf("%d", order.Deadline),
			"inputToken":        order.TokenIn,
			"inputAmount":       order.AmountIn.String(),
			"outputToken":       order.TokenOut,
			"outputStartAmount": order.StartAmount.String(),
			"outputEndAmount":   order.EndAmount.String(),
			"decayStartTime":    fmt.Sprintf("%d", order.DecayStartTime),
			"decayEndTime":      fmt.Sprintf("%d", order.DecayEndTime),
		},
	}
}

// orderIDToNonce deterministically converts a UUID string to a uint256 nonce.
// This ensures each order has a unique nonce for replay protection.
func orderIDToNonce(orderID string) string {
	h := crypto.Keccak256([]byte(orderID))
	n := new(big.Int).SetBytes(h)
	return n.String()
}

// ABI encoding helpers

// padAddress pads a hex address string to 32 bytes (left-padded with zeros).
func padAddress(addr string) []byte {
	addr = strings.TrimPrefix(strings.ToLower(addr), "0x")
	addrBytes, _ := hex.DecodeString(addr)
	padded := make([]byte, 32)
	copy(padded[32-len(addrBytes):], addrBytes)
	return padded
}

// padBigInt pads a big.Int to 32 bytes (left-padded with zeros).
func padBigInt(n *big.Int) []byte {
	if n == nil {
		return make([]byte, 32)
	}
	b := n.Bytes()
	padded := make([]byte, 32)
	copy(padded[32-len(b):], b)
	return padded
}

// padUint64 encodes a uint64 as 32 bytes (left-padded with zeros).
func padUint64(v uint64) []byte {
	n := new(big.Int).SetUint64(v)
	return padBigInt(n)
}

// padBytes32FromString hashes a string to produce a bytes32 value.
func padBytes32FromString(s string) []byte {
	h := crypto.Keccak256([]byte(s))
	padded := make([]byte, 32)
	copy(padded, h)
	return padded
}
