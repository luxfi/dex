package api

import (
	"encoding/json"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/log"
)

// IPRateLimiter tracks requests per IP
type IPRateLimiter struct {
	requests map[string]*rateLimitEntry
	mu       sync.RWMutex
	limit    int           // max requests per window
	window   time.Duration // time window
}

type rateLimitEntry struct {
	count     int
	resetTime time.Time
}

// NewIPRateLimiter creates a new rate limiter
func NewIPRateLimiter(limit int, window time.Duration) *IPRateLimiter {
	rl := &IPRateLimiter{
		requests: make(map[string]*rateLimitEntry),
		limit:    limit,
		window:   window,
	}
	// Cleanup expired entries periodically
	go rl.cleanup()
	return rl
}

// Allow checks if a request from IP is allowed
func (rl *IPRateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	entry, exists := rl.requests[ip]

	if !exists || now.After(entry.resetTime) {
		rl.requests[ip] = &rateLimitEntry{
			count:     1,
			resetTime: now.Add(rl.window),
		}
		return true
	}

	if entry.count >= rl.limit {
		return false
	}

	entry.count++
	return true
}

func (rl *IPRateLimiter) cleanup() {
	ticker := time.NewTicker(rl.window)
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		now := time.Now()
		for ip, entry := range rl.requests {
			if now.After(entry.resetTime) {
				delete(rl.requests, ip)
			}
		}
		rl.mu.Unlock()
	}
}

// JSONRPCServer handles JSON-RPC requests
type JSONRPCServer struct {
	orderBook   *dex.OrderBook
	logger      log.Logger
	rateLimiter *IPRateLimiter
}

// NewJSONRPCServer creates a new JSON-RPC server with rate limiting
// Default: 100 requests per second per IP
func NewJSONRPCServer(orderBook *dex.OrderBook, logger log.Logger) *JSONRPCServer {
	return &JSONRPCServer{
		orderBook:   orderBook,
		logger:      logger,
		rateLimiter: NewIPRateLimiter(100, time.Second),
	}
}

// getClientIP extracts client IP from request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxied requests)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// Take first IP in chain
		if idx := len(xff); idx > 0 {
			for i, c := range xff {
				if c == ',' {
					return xff[:i]
				}
			}
			return xff
		}
	}
	// Fall back to remote address
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// ServeHTTP handles JSON-RPC requests
func (s *JSONRPCServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Rate limit check
	clientIP := getClientIP(r)
	if !s.rateLimiter.Allow(clientIP) {
		w.Header().Set("Retry-After", "1")
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		return
	}

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
