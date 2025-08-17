// +build cgo

package lx

// #cgo CFLAGS: -I../../bridge -I../../cpp -O3 -march=native -mtune=native
// #cgo CXXFLAGS: -std=c++20 -O3 -march=native -mtune=native -ffast-math
// #cgo LDFLAGS: -L../../build -lultra_matcher -lstdc++ -lm
// #include <stdint.h>
// #include <stdlib.h>
//
// typedef void* matcher_handle_t;
//
// typedef struct {
//     uint64_t id;
//     uint64_t price;
//     uint64_t quantity;
//     uint64_t timestamp;
//     uint32_t trader_id;
//     uint8_t  side;
//     uint8_t  order_type;
//     uint8_t  flags;
// } COrder;
//
// typedef struct {
//     uint64_t buy_order_id;
//     uint64_t sell_order_id;
//     uint64_t price;
//     uint64_t quantity;
//     uint64_t timestamp;
// } CTrade;
//
// matcher_handle_t ultra_matcher_create();
// void ultra_matcher_destroy(matcher_handle_t handle);
// uint64_t ultra_matcher_add_order(matcher_handle_t handle, const COrder* order);
// int ultra_matcher_match(matcher_handle_t handle, const COrder* order, CTrade* trades, int max_trades);
// int ultra_matcher_cancel(matcher_handle_t handle, uint64_t order_id);
// uint64_t ultra_matcher_best_bid(matcher_handle_t handle);
// uint64_t ultra_matcher_best_ask(matcher_handle_t handle);
// void ultra_matcher_stats(matcher_handle_t handle, uint64_t* total_orders, uint64_t* total_trades, uint64_t* total_volume);
import "C"
import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// UltraFastMatcher is a Go wrapper for the C++ ultra-fast matching engine
type UltraFastMatcher struct {
	handle C.matcher_handle_t
	mu     sync.RWMutex
	
	// Performance metrics
	totalLatency  atomic.Uint64
	orderCount    atomic.Uint64
	minLatency    atomic.Uint64
	maxLatency    atomic.Uint64
}

// NewUltraFastMatcher creates a new ultra-fast matching engine
func NewUltraFastMatcher() *UltraFastMatcher {
	matcher := &UltraFastMatcher{
		handle: C.ultra_matcher_create(),
	}
	matcher.minLatency.Store(^uint64(0))
	return matcher
}

// Destroy cleans up the C++ matcher
func (m *UltraFastMatcher) Destroy() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.handle != nil {
		C.ultra_matcher_destroy(m.handle)
		m.handle = nil
	}
}

// AddOrder adds an order to the book with sub-microsecond latency
func (m *UltraFastMatcher) AddOrder(order *Order) (time.Duration, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.handle == nil {
		return 0, fmt.Errorf("matcher destroyed")
	}
	
	// Convert Go order to C order
	cOrder := C.COrder{
		id:        C.uint64_t(order.ID),
		price:     C.uint64_t(order.Price * 100000000), // Convert to fixed point
		quantity:  C.uint64_t(order.Size * 100000000),
		timestamp:  C.uint64_t(time.Now().UnixNano()),
		trader_id:  C.uint32_t(0), // Would map from order.UserID
		side:       C.uint8_t(0),  // 0 for buy
		order_type: C.uint8_t(0),  // 0 for limit
		flags:      C.uint8_t(0),
	}
	
	if order.Side == "sell" {
		cOrder.side = 1
	}
	
	if order.Type == "market" {
		cOrder.order_type = 1
	}
	
	// Add order and measure latency
	latencyNs := uint64(C.ultra_matcher_add_order(m.handle, &cOrder))
	
	// Update metrics
	m.orderCount.Add(1)
	m.totalLatency.Add(latencyNs)
	
	// Update min/max latency
	for {
		min := m.minLatency.Load()
		if latencyNs >= min || m.minLatency.CompareAndSwap(min, latencyNs) {
			break
		}
	}
	
	for {
		max := m.maxLatency.Load()
		if latencyNs <= max || m.maxLatency.CompareAndSwap(max, latencyNs) {
			break
		}
	}
	
	return time.Duration(latencyNs) * time.Nanosecond, nil
}

// MatchOrder matches an order against the book
func (m *UltraFastMatcher) MatchOrder(order *Order) ([]*Trade, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.handle == nil {
		return nil, fmt.Errorf("matcher destroyed")
	}
	
	// Convert Go order to C order
	cOrder := C.COrder{
		id:         C.uint64_t(order.ID),
		price:      C.uint64_t(order.Price * 100000000),
		quantity:   C.uint64_t(order.Size * 100000000),
		timestamp:  C.uint64_t(time.Now().UnixNano()),
		trader_id:  C.uint32_t(0),
		side:       C.uint8_t(0),
		order_type: C.uint8_t(0),
		flags:      C.uint8_t(0),
	}
	
	if order.Side == "sell" {
		cOrder.side = 1
	}
	
	// Allocate space for trades
	maxTrades := 100
	cTrades := make([]C.CTrade, maxTrades)
	
	// Match order
	numTrades := int(C.ultra_matcher_match(
		m.handle,
		&cOrder,
		(*C.CTrade)(unsafe.Pointer(&cTrades[0])),
		C.int(maxTrades),
	))
	
	// Convert C trades to Go trades
	trades := make([]*Trade, numTrades)
	for i := 0; i < numTrades; i++ {
		trades[i] = &Trade{
			ID:        fmt.Sprintf("t_%d", time.Now().UnixNano()),
			Symbol:    order.Symbol,
			Price:     float64(cTrades[i].price) / 100000000,
			Quantity:  float64(cTrades[i].quantity) / 100000000,
			BuyOrderID:  fmt.Sprintf("%d", cTrades[i].buy_order_id),
			SellOrderID: fmt.Sprintf("%d", cTrades[i].sell_order_id),
			Timestamp:   time.Unix(0, int64(cTrades[i].timestamp)),
		}
	}
	
	return trades, nil
}

// CancelOrder cancels an order
func (m *UltraFastMatcher) CancelOrder(orderID string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.handle == nil {
		return fmt.Errorf("matcher destroyed")
	}
	
	// Parse order ID to uint64
	var id uint64
	fmt.Sscanf(orderID, "%d", &id)
	
	result := C.ultra_matcher_cancel(m.handle, C.uint64_t(id))
	if result == 0 {
		return fmt.Errorf("order not found")
	}
	
	return nil
}

// GetBestBidAsk returns the best bid and ask prices
func (m *UltraFastMatcher) GetBestBidAsk() (float64, float64) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.handle == nil {
		return 0, 0
	}
	
	bid := uint64(C.ultra_matcher_best_bid(m.handle))
	ask := uint64(C.ultra_matcher_best_ask(m.handle))
	
	return float64(bid) / 100000000, float64(ask) / 100000000
}

// GetStats returns performance statistics
func (m *UltraFastMatcher) GetStats() UltraMatcherStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	var totalOrders, totalTrades, totalVolume C.uint64_t
	
	if m.handle != nil {
		C.ultra_matcher_stats(m.handle, &totalOrders, &totalTrades, &totalVolume)
	}
	
	orderCount := m.orderCount.Load()
	avgLatency := time.Duration(0)
	if orderCount > 0 {
		avgLatency = time.Duration(m.totalLatency.Load()/orderCount) * time.Nanosecond
	}
	
	return UltraMatcherStats{
		TotalOrders:    uint64(totalOrders),
		TotalTrades:    uint64(totalTrades),
		TotalVolume:    uint64(totalVolume),
		AverageLatency: avgLatency,
		MinLatency:     time.Duration(m.minLatency.Load()) * time.Nanosecond,
		MaxLatency:     time.Duration(m.maxLatency.Load()) * time.Nanosecond,
		OrdersPerSec:   float64(orderCount) / time.Since(time.Now()).Seconds(),
	}
}

// UltraMatcherStats contains performance statistics
type UltraMatcherStats struct {
	TotalOrders    uint64
	TotalTrades    uint64
	TotalVolume    uint64
	AverageLatency time.Duration
	MinLatency     time.Duration
	MaxLatency     time.Duration
	OrdersPerSec   float64
}