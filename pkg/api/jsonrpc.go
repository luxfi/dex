package api

import (
	"encoding/json"
	"net/http"
	
	"github.com/luxfi/log"
	"github.com/luxfi/dex/pkg/lx"
)

// JSONRPCServer handles JSON-RPC requests
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

// ServeHTTP handles JSON-RPC requests
func (s *JSONRPCServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Jsonrpc string          `json:"jsonrpc"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params"`
		ID      interface{}     `json:"id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, req.ID, -32700, "Parse error")
		return
	}

	if req.Jsonrpc != "2.0" {
		s.writeError(w, req.ID, -32600, "Invalid Request")
		return
	}

	switch req.Method {
	case "orderbook.getBestBid":
		bestBid := s.orderBook.GetBestBid()
		s.writeResult(w, req.ID, bestBid)
		
	case "orderbook.getBestAsk":
		bestAsk := s.orderBook.GetBestAsk()
		s.writeResult(w, req.ID, bestAsk)
		
	case "orderbook.getStats":
		stats := map[string]interface{}{
			"symbol": s.orderBook.Symbol,
			"orders": len(s.orderBook.Orders),
			"trades": len(s.orderBook.Trades),
		}
		s.writeResult(w, req.ID, stats)
		
	default:
		s.writeError(w, req.ID, -32601, "Method not found")
	}
}

func (s *JSONRPCServer) writeResult(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := map[string]interface{}{
		"jsonrpc": "2.0",
		"result":  result,
		"id":      id,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *JSONRPCServer) writeError(w http.ResponseWriter, id interface{}, code int, message string) {
	resp := map[string]interface{}{
		"jsonrpc": "2.0",
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
		},
		"id": id,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
