package gpu

// #cgo CFLAGS: -I. -std=c++17
// #cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -lc++
// #cgo linux LDFLAGS: -lcudart -lcuda -lstdc++
// #cgo CXXFLAGS: -std=c++17
/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint64_t order_id;
    uint32_t price;
    uint32_t quantity;
    uint32_t timestamp;
    uint8_t side;
    uint8_t status;
    uint16_t pad;
} GPUOrder;

typedef struct {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint32_t price;
    uint32_t quantity;
    uint32_t timestamp;
} GPUTrade;

typedef struct {
    uint64_t orders_processed;
    uint64_t trades_executed;
    uint64_t total_volume;
    uint32_t min_latency_ns;
    uint32_t max_latency_ns;
    uint32_t avg_latency_ns;
} GPUStats;

void* create_gpu_engine();
void destroy_gpu_engine(void* engine);
int gpu_match_orders(void* engine, GPUOrder* bids, int bid_count, 
                    GPUOrder* asks, int ask_count, 
                    GPUTrade* trades, int max_trades);
void gpu_get_stats(void* engine, GPUStats* stats);
const char* gpu_get_device_info(void* engine);
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Order represents an order for GPU processing
type Order struct {
	OrderID   uint64
	Price     uint32 // Fixed point, 7 decimals
	Quantity  uint32 // Fixed point, 7 decimals
	Timestamp uint32
	Side      uint8 // 0=buy, 1=sell
	Status    uint8 // 0=active, 1=filled, 2=cancelled
}

// Trade represents a matched trade
type Trade struct {
	TradeID     uint64
	BuyOrderID  uint64
	SellOrderID uint64
	Price       uint32
	Quantity    uint32
	Timestamp   uint32
}

// Stats contains GPU performance statistics
type Stats struct {
	OrdersProcessed uint64
	TradesExecuted  uint64
	TotalVolume     uint64
	MinLatencyNs    uint32
	MaxLatencyNs    uint32
	AvgLatencyNs    uint32
}

// Engine is a GPU-accelerated matching engine
type Engine struct {
	handle unsafe.Pointer
	mu     sync.Mutex
}

// NewEngine creates a new GPU matching engine
func NewEngine() (*Engine, error) {
	handle := C.create_gpu_engine()
	if handle == nil {
		return nil, fmt.Errorf("failed to initialize GPU engine")
	}
	
	engine := &Engine{
		handle: handle,
	}
	
	// Set finalizer to clean up GPU resources
	runtime.SetFinalizer(engine, (*Engine).Close)
	
	return engine, nil
}

// Close releases GPU resources
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle != nil {
		C.destroy_gpu_engine(e.handle)
		e.handle = nil
	}
	return nil
}

// MatchOrders performs GPU-accelerated order matching
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
	cBids := make([]C.GPUOrder, len(bids))
	for i, bid := range bids {
		cBids[i] = C.GPUOrder{
			order_id:  C.uint64_t(bid.OrderID),
			price:     C.uint32_t(bid.Price),
			quantity:  C.uint32_t(bid.Quantity),
			timestamp: C.uint32_t(bid.Timestamp),
			side:      C.uint8_t(bid.Side),
			status:    C.uint8_t(bid.Status),
		}
	}
	
	cAsks := make([]C.GPUOrder, len(asks))
	for i, ask := range asks {
		cAsks[i] = C.GPUOrder{
			order_id:  C.uint64_t(ask.OrderID),
			price:     C.uint32_t(ask.Price),
			quantity:  C.uint32_t(ask.Quantity),
			timestamp: C.uint32_t(ask.Timestamp),
			side:      C.uint8_t(ask.Side),
			status:    C.uint8_t(ask.Status),
		}
	}
	
	// Allocate space for trades
	maxTrades := len(bids) * len(asks) / 2
	if maxTrades > 500000 {
		maxTrades = 500000
	}
	cTrades := make([]C.GPUTrade, maxTrades)
	
	// Call GPU matching
	tradeCount := int(C.gpu_match_orders(
		e.handle,
		(*C.GPUOrder)(unsafe.Pointer(&cBids[0])), C.int(len(bids)),
		(*C.GPUOrder)(unsafe.Pointer(&cAsks[0])), C.int(len(asks)),
		(*C.GPUTrade)(unsafe.Pointer(&cTrades[0])), C.int(maxTrades),
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
			Price:       uint32(cTrades[i].price),
			Quantity:    uint32(cTrades[i].quantity),
			Timestamp:   uint32(cTrades[i].timestamp),
		}
	}
	
	return trades, nil
}

// GetStats returns GPU performance statistics
func (e *Engine) GetStats() (*Stats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle == nil {
		return nil, fmt.Errorf("engine is closed")
	}
	
	var cStats C.GPUStats
	C.gpu_get_stats(e.handle, &cStats)
	
	return &Stats{
		OrdersProcessed: uint64(cStats.orders_processed),
		TradesExecuted:  uint64(cStats.trades_executed),
		TotalVolume:     uint64(cStats.total_volume),
		MinLatencyNs:    uint32(cStats.min_latency_ns),
		MaxLatencyNs:    uint32(cStats.max_latency_ns),
		AvgLatencyNs:    uint32(cStats.avg_latency_ns),
	}, nil
}

// GetDeviceInfo returns GPU device information
func (e *Engine) GetDeviceInfo() (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.handle == nil {
		return "", fmt.Errorf("engine is closed")
	}
	
	info := C.gpu_get_device_info(e.handle)
	return C.GoString(info), nil
}

// IsAvailable checks if GPU acceleration is available
func IsAvailable() bool {
	engine := C.create_gpu_engine()
	if engine == nil {
		return false
	}
	C.destroy_gpu_engine(engine)
	return true
}