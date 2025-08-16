package matching

// #cgo CFLAGS: -I. -std=c++17 -O3 -march=native
// #cgo darwin CFLAGS: -DHAS_METAL=1
// #cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -lc++
// #cgo linux LDFLAGS: -lstdc++
// #cgo linux,cuda CFLAGS: -DHAS_CUDA=1
// #cgo linux,cuda LDFLAGS: -lcudart
/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint64_t order_id;
    float price;
    float quantity;
    uint32_t timestamp;
    uint8_t side;
    uint8_t status;
    uint16_t user_id;
} Order;

typedef struct {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    float price;
    float quantity;
    uint32_t timestamp;
} Trade;

typedef struct {
    uint64_t orders_processed;
    uint64_t trades_executed;
    uint64_t total_latency_ns;
    double throughput_orders_per_sec;
    double throughput_trades_per_sec;
} MatchingStats;

// Functions from matching_engine.cpp
void* create_matching_engine();
void* create_matching_engine_with_backend(int backend);
void destroy_matching_engine(void* engine);
int match_orders(void* engine, Order* bids, int bid_count, 
                Order* asks, int ask_count, Trade* trades, int max_trades);
void get_matching_stats(void* engine, MatchingStats* stats);
const char* get_backend_name(void* engine);
int detect_available_backends();
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Backend represents the matching engine backend
type Backend int

const (
	BackendAuto Backend = iota
	BackendCPU
	BackendMLXMetal
	BackendCUDA
)

// Order represents a trading order
type Order struct {
	OrderID   uint64
	Price     float32
	Quantity  float32
	Timestamp uint32
	Side      uint8 // 0=buy, 1=sell
	Status    uint8 // 0=active, 1=filled
	UserID    uint16
}

// Trade represents a matched trade
type Trade struct {
	TradeID     uint64
	BuyOrderID  uint64
	SellOrderID uint64
	Price       float32
	Quantity    float32
	Timestamp   uint32
}

// Stats contains matching engine statistics
type Stats struct {
	OrdersProcessed        uint64
	TradesExecuted         uint64
	TotalLatencyNs         uint64
	ThroughputOrdersPerSec float64
	ThroughputTradesPerSec float64
}

// Engine is a high-performance matching engine with auto-backend selection
type Engine struct {
	handle  unsafe.Pointer
	backend string
	mu      sync.Mutex
}

// NewEngine creates a new matching engine with automatic backend selection
// It will use GPU (Metal on Mac, CUDA on Linux) if available, otherwise optimized CPU
func NewEngine() (*Engine, error) {
	handle := C.create_matching_engine()
	if handle == nil {
		return nil, fmt.Errorf("failed to create matching engine")
	}
	
	backend := C.GoString(C.get_backend_name(handle))
	
	engine := &Engine{
		handle:  handle,
		backend: backend,
	}
	
	runtime.SetFinalizer(engine, (*Engine).Close)
	
	fmt.Printf("Matching engine initialized with backend: %s\n", backend)
	return engine, nil
}

// NewEngineWithBackend creates a matching engine with specific backend
func NewEngineWithBackend(backend Backend) (*Engine, error) {
	handle := C.create_matching_engine_with_backend(C.int(backend))
	if handle == nil {
		return nil, fmt.Errorf("failed to create matching engine with backend %d", backend)
	}
	
	backendName := C.GoString(C.get_backend_name(handle))
	
	engine := &Engine{
		handle:  handle,
		backend: backendName,
	}
	
	runtime.SetFinalizer(engine, (*Engine).Close)
	
	fmt.Printf("Matching engine initialized with backend: %s\n", backendName)
	return engine, nil
}

// MatchOrders performs high-performance order matching
func (e *Engine) MatchOrders(bids, asks []Order) ([]Trade, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle == nil {
		return nil, fmt.Errorf("engine is closed")
	}
	
	if len(bids) == 0 || len(asks) == 0 {
		return nil, nil
	}
	
	// Convert to C structures
	cBids := make([]C.Order, len(bids))
	for i, bid := range bids {
		cBids[i] = C.Order{
			order_id:  C.uint64_t(bid.OrderID),
			price:     C.float(bid.Price),
			quantity:  C.float(bid.Quantity),
			timestamp: C.uint32_t(bid.Timestamp),
			side:      C.uint8_t(bid.Side),
			status:    C.uint8_t(bid.Status),
			user_id:   C.uint16_t(bid.UserID),
		}
	}
	
	cAsks := make([]C.Order, len(asks))
	for i, ask := range asks {
		cAsks[i] = C.Order{
			order_id:  C.uint64_t(ask.OrderID),
			price:     C.float(ask.Price),
			quantity:  C.float(ask.Quantity),
			timestamp: C.uint32_t(ask.Timestamp),
			side:      C.uint8_t(ask.Side),
			status:    C.uint8_t(ask.Status),
			user_id:   C.uint16_t(ask.UserID),
		}
	}
	
	// Allocate space for trades
	maxTrades := len(bids) * len(asks)
	if maxTrades > 500000 {
		maxTrades = 500000
	}
	cTrades := make([]C.Trade, maxTrades)
	
	// Call matching engine
	tradeCount := int(C.match_orders(
		e.handle,
		(*C.Order)(unsafe.Pointer(&cBids[0])), C.int(len(bids)),
		(*C.Order)(unsafe.Pointer(&cAsks[0])), C.int(len(asks)),
		(*C.Trade)(unsafe.Pointer(&cTrades[0])), C.int(maxTrades),
	))
	
	if tradeCount == 0 {
		return nil, nil
	}
	
	// Convert trades back to Go
	trades := make([]Trade, tradeCount)
	for i := 0; i < tradeCount; i++ {
		trades[i] = Trade{
			TradeID:     uint64(cTrades[i].trade_id),
			BuyOrderID:  uint64(cTrades[i].buy_order_id),
			SellOrderID: uint64(cTrades[i].sell_order_id),
			Price:       float32(cTrades[i].price),
			Quantity:    float32(cTrades[i].quantity),
			Timestamp:   uint32(cTrades[i].timestamp),
		}
	}
	
	return trades, nil
}

// GetStats returns matching engine statistics
func (e *Engine) GetStats() (*Stats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle == nil {
		return nil, fmt.Errorf("engine is closed")
	}
	
	var cStats C.MatchingStats
	C.get_matching_stats(e.handle, &cStats)
	
	return &Stats{
		OrdersProcessed:        uint64(cStats.orders_processed),
		TradesExecuted:         uint64(cStats.trades_executed),
		TotalLatencyNs:         uint64(cStats.total_latency_ns),
		ThroughputOrdersPerSec: float64(cStats.throughput_orders_per_sec),
		ThroughputTradesPerSec: float64(cStats.throughput_trades_per_sec),
	}, nil
}

// GetBackend returns the backend name
func (e *Engine) GetBackend() string {
	return e.backend
}

// Close releases resources
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle != nil {
		C.destroy_matching_engine(e.handle)
		e.handle = nil
	}
	return nil
}

// DetectAvailableBackends returns available backends on this system
func DetectAvailableBackends() []string {
	backends := []string{"CPU (Always Available)"}
	
	flags := int(C.detect_available_backends())
	
	if flags&(1<<1) != 0 {
		backends = append(backends, "MLX (Metal GPU)")
	}
	
	if flags&(1<<2) != 0 {
		backends = append(backends, "CUDA GPU")
	}
	
	return backends
}