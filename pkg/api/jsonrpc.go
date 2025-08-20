package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
)

// JSONRPCServer handles JSON-RPC 2.0 requests
type JSONRPCServer struct {
	orderBook *lx.OrderBook
	logger    log.Logger
}

// NewJSONRPCServer creates a new JSON-RPC server
func NewJSONRPCServer(orderBook *lx.OrderBook, logger log.Logger) *JSONRPCServer {
	return &JSONRPCServer{
		orderBook: orderBook,
		logger:    logger,
	}
}

// JSONRPCRequest represents a JSON-RPC 2.0 request
type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
	ID      interface{}     `json:"id"`
}

// JSONRPCResponse represents a JSON-RPC 2.0 response
type JSONRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
	ID      interface{} `json:"id"`
}

// RPCError represents a JSON-RPC error
type RPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Error implements error interface
func (e *RPCError) Error() string {
	return fmt.Sprintf("RPC Error %d: %s", e.Code, e.Message)
}

// Standard JSON-RPC error codes
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// ServeHTTP implements http.Handler
func (s *JSONRPCServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.sendError(w, nil, ParseError, "Parse error")
		return
	}

	if req.JSONRPC != "2.0" {
		s.sendError(w, req.ID, InvalidRequest, "Invalid Request")
		return
	}

	// Route to method handler
	result, err := s.handleMethod(req.Method, req.Params)
	if err != nil {
		s.sendError(w, req.ID, err.(*RPCError).Code, err.(*RPCError).Message)
		return
	}

	// Send success response
	resp := JSONRPCResponse{
		JSONRPC: "2.0",
		Result:  result,
		ID:      req.ID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *JSONRPCServer) handleMethod(method string, params json.RawMessage) (interface{}, error) {
	switch method {
	// Order methods
	case "lx_placeOrder":
		return s.placeOrder(params)
	case "lx_cancelOrder":
		return s.cancelOrder(params)
	case "lx_getOrder":
		return s.getOrder(params)
	
	// Market data methods
	case "lx_getOrderBook":
		return s.getOrderBook(params)
	case "lx_getBestBid":
		return s.getBestBid(params)
	case "lx_getBestAsk":
		return s.getBestAsk(params)
	case "lx_getTrades":
		return s.getTrades(params)
	
	// Info methods
	case "lx_getInfo":
		return s.getInfo(params)
	case "lx_ping":
		return "pong", nil
	
	default:
		return nil, &RPCError{Code: MethodNotFound, Message: "Method not found"}
	}
}

// Order placement
func (s *JSONRPCServer) placeOrder(params json.RawMessage) (interface{}, error) {
	var order lx.Order
	if err := json.Unmarshal(params, &order); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	orderID := s.orderBook.AddOrder(&order)
	if orderID == 0 {
		return nil, &RPCError{Code: InternalError, Message: "Order rejected"}
	}

	return map[string]interface{}{
		"orderId": orderID,
		"status":  "accepted",
	}, nil
}

// Order cancellation
func (s *JSONRPCServer) cancelOrder(params json.RawMessage) (interface{}, error) {
	var p struct {
		OrderID uint64 `json:"orderId"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if err := s.orderBook.CancelOrder(p.OrderID); err != nil {
		return nil, &RPCError{Code: InternalError, Message: err.Error()}
	}

	return map[string]interface{}{
		"orderId": p.OrderID,
		"status":  "cancelled",
	}, nil
}

// Get order by ID
func (s *JSONRPCServer) getOrder(params json.RawMessage) (interface{}, error) {
	var p struct {
		OrderID uint64 `json:"orderId"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	order := s.orderBook.GetOrder(p.OrderID)
	if order == nil {
		return nil, &RPCError{Code: InternalError, Message: "Order not found"}
	}

	return order, nil
}

// Get order book snapshot
func (s *JSONRPCServer) getOrderBook(params json.RawMessage) (interface{}, error) {
	var p struct {
		Depth int `json:"depth"`
	}
	// Default depth
	p.Depth = 10
	json.Unmarshal(params, &p)

	snapshot := s.orderBook.GetSnapshot()
	
	// Limit depth
	if p.Depth > 0 {
		if len(snapshot.Bids) > p.Depth {
			snapshot.Bids = snapshot.Bids[:p.Depth]
		}
		if len(snapshot.Asks) > p.Depth {
			snapshot.Asks = snapshot.Asks[:p.Depth]
		}
	}

	return snapshot, nil
}

// Get best bid
func (s *JSONRPCServer) getBestBid(params json.RawMessage) (interface{}, error) {
	price := s.orderBook.GetBestBid()
	return map[string]interface{}{
		"price": price,
	}, nil
}

// Get best ask
func (s *JSONRPCServer) getBestAsk(params json.RawMessage) (interface{}, error) {
	price := s.orderBook.GetBestAsk()
	return map[string]interface{}{
		"price": price,
	}, nil
}

// Get recent trades
func (s *JSONRPCServer) getTrades(params json.RawMessage) (interface{}, error) {
	var p struct {
		Limit int `json:"limit"`
	}
	p.Limit = 100
	json.Unmarshal(params, &p)

	trades := s.orderBook.GetTrades()
	
	// Limit results
	if p.Limit > 0 && len(trades) > p.Limit {
		trades = trades[len(trades)-p.Limit:]
	}

	return trades, nil
}

// Get node info
func (s *JSONRPCServer) getInfo(params json.RawMessage) (interface{}, error) {
	return map[string]interface{}{
		"version":    "1.0.0",
		"network":    "lux-mainnet",
		"symbol":     s.orderBook.Symbol,
		"timestamp":  time.Now().Unix(),
		"orderCount": len(s.orderBook.Orders),
		"tradeCount": len(s.orderBook.Trades),
	}, nil
}

func (s *JSONRPCServer) sendError(w http.ResponseWriter, id interface{}, code int, message string) {
	resp := JSONRPCResponse{
		JSONRPC: "2.0",
		Error: &RPCError{
			Code:    code,
			Message: message,
		},
		ID: id,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// StartJSONRPCServer starts the JSON-RPC server
func StartJSONRPCServer(ctx context.Context, port int, orderBook *lx.OrderBook, logger log.Logger) error {
	server := NewJSONRPCServer(orderBook, logger)
	
	mux := http.NewServeMux()
	mux.Handle("/", server)
	
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}
	
	go func() {
		<-ctx.Done()
		httpServer.Shutdown(context.Background())
	}()
	
	logger.Info("JSON-RPC server started", "port", port)
	return httpServer.ListenAndServe()
}