//go:build cgo
// +build cgo

package orderbook

/*
#cgo CXXFLAGS: -std=c++17 -O3 -march=native -I../../bridge
#cgo LDFLAGS: -L../../bridge -lorderbook -lstdc++

#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint64_t id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    double quantity;
} TradeC;

void* orderbook_create();
void orderbook_destroy(void* ob);
void orderbook_add_order(void* ob, uint64_t id, double price, double quantity, uint8_t side);
int orderbook_cancel_order(void* ob, uint64_t order_id);
int orderbook_match_orders(void* ob, TradeC* trades_out, int max_trades);
double orderbook_get_best_bid(void* ob);
double orderbook_get_best_ask(void* ob);
int orderbook_get_depth(void* ob, uint8_t side);
*/
import "C"
import (
	"sync"
	"unsafe"
)

// CGOOrderBook is a high-performance C++ orderbook wrapped for Go
type CGOOrderBook struct {
	handle unsafe.Pointer
	mu     sync.RWMutex
	symbol string
	nextID uint64
}

// NewCGOOrderBook creates a new C++ backed orderbook
func NewCGOOrderBook(config Config) *CGOOrderBook {
	return &CGOOrderBook{
		handle: C.orderbook_create(),
		symbol: config.Symbol,
		nextID: 1,
	}
}

// AddOrder adds an order to the book
func (ob *CGOOrderBook) AddOrder(order *Order) uint64 {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	if order.ID == 0 {
		order.ID = ob.nextID
		ob.nextID++
	}

	side := uint8(0)
	if order.Side == Sell {
		side = 1
	}

	C.orderbook_add_order(
		ob.handle,
		C.uint64_t(order.ID),
		C.double(order.Price),
		C.double(order.Quantity),
		C.uint8_t(side),
	)

	return order.ID
}

// CancelOrder cancels an order by ID
func (ob *CGOOrderBook) CancelOrder(orderID uint64) bool {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	result := C.orderbook_cancel_order(ob.handle, C.uint64_t(orderID))
	return result == 1
}

// MatchOrders executes matching orders and returns trades
func (ob *CGOOrderBook) MatchOrders() []Trade {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	// Allocate space for up to 1000 trades
	maxTrades := 1000
	tradesC := make([]C.TradeC, maxTrades)

	count := int(C.orderbook_match_orders(
		ob.handle,
		&tradesC[0],
		C.int(maxTrades),
	))

	trades := make([]Trade, count)
	for i := 0; i < count; i++ {
		trades[i] = Trade{
			ID:          uint64(tradesC[i].id),
			BuyOrderID:  uint64(tradesC[i].buy_order_id),
			SellOrderID: uint64(tradesC[i].sell_order_id),
			Price:       float64(tradesC[i].price),
			Quantity:    float64(tradesC[i].quantity),
		}
	}

	return trades
}

// GetBestBid returns the best bid price
func (ob *CGOOrderBook) GetBestBid() float64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	return float64(C.orderbook_get_best_bid(ob.handle))
}

// GetBestAsk returns the best ask price
func (ob *CGOOrderBook) GetBestAsk() float64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	return float64(C.orderbook_get_best_ask(ob.handle))
}

// GetDepth returns the order book depth
func (ob *CGOOrderBook) GetDepth(levels int) *Depth {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	// For simplicity, just return bid/ask counts
	bidDepth := int(C.orderbook_get_depth(ob.handle, 0))
	askDepth := int(C.orderbook_get_depth(ob.handle, 1))

	return &Depth{
		Bids: make([]PriceLevel, 0, bidDepth),
		Asks: make([]PriceLevel, 0, askDepth),
	}
}

// GetSpread returns the bid-ask spread
func (ob *CGOOrderBook) GetSpread() float64 {
	return ob.GetBestAsk() - ob.GetBestBid()
}

// GetMidPrice returns the mid price
func (ob *CGOOrderBook) GetMidPrice() float64 {
	bid := ob.GetBestBid()
	ask := ob.GetBestAsk()
	if bid == 0 || ask == 0 {
		return 0
	}
	return (bid + ask) / 2
}

// GetOrderCount returns the number of orders
func (ob *CGOOrderBook) GetOrderCount() int {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	bidCount := int(C.orderbook_get_depth(ob.handle, 0))
	askCount := int(C.orderbook_get_depth(ob.handle, 1))
	return bidCount + askCount
}

// GetOrder returns an order by ID (not implemented in C++)
func (ob *CGOOrderBook) GetOrder(orderID uint64) *Order {
	return nil
}

// ModifyOrder modifies an existing order (cancel and re-add)
func (ob *CGOOrderBook) ModifyOrder(orderID uint64, price, quantity float64) bool {
	if !ob.CancelOrder(orderID) {
		return false
	}
	// Re-add with same ID
	ob.AddOrder(&Order{
		ID:       orderID,
		Price:    price,
		Quantity: quantity,
	})
	return true
}

// Clear removes all orders
func (ob *CGOOrderBook) Clear() {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	// Recreate the C++ orderbook
	C.orderbook_destroy(ob.handle)
	ob.handle = C.orderbook_create()
}

// GetVolume returns the total volume
func (ob *CGOOrderBook) GetVolume() uint64 {
	// Not tracked in simple C++ implementation
	return 0
}

// Destroy cleans up the C++ orderbook
func (ob *CGOOrderBook) Destroy() {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	if ob.handle != nil {
		C.orderbook_destroy(ob.handle)
		ob.handle = nil
	}
}
