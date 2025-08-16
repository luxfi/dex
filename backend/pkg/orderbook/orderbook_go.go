package orderbook

import (
	"sync"
	"time"
)

// GoOrderBook is a pure Go implementation of the OrderBook interface
type GoOrderBook struct {
	mu     sync.RWMutex
	symbol string
	orders map[uint64]*Order
	nextID uint64
	bids   []uint64 // Order IDs sorted by price (descending)
	asks   []uint64 // Order IDs sorted by price (ascending)
	volume uint64
}

// NewGoOrderBook creates a new Go orderbook
func NewGoOrderBook(config Config) *GoOrderBook {
	return &GoOrderBook{
		symbol: config.Symbol,
		orders: make(map[uint64]*Order),
		nextID: 1,
	}
}

// AddOrder adds an order to the book
func (ob *GoOrderBook) AddOrder(order *Order) uint64 {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	if order.ID == 0 {
		order.ID = ob.nextID
		ob.nextID++
	}

	order.Timestamp = time.Now()
	ob.orders[order.ID] = order

	// Add to sorted lists
	if order.Side == Buy {
		ob.insertBid(order.ID)
	} else {
		ob.insertAsk(order.ID)
	}

	return order.ID
}

// CancelOrder cancels an order by ID
func (ob *GoOrderBook) CancelOrder(orderID uint64) bool {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	order, exists := ob.orders[orderID]
	if !exists {
		return false
	}

	delete(ob.orders, orderID)

	// Remove from sorted lists
	if order.Side == Buy {
		ob.removeBid(orderID)
	} else {
		ob.removeAsk(orderID)
	}

	return true
}

// ModifyOrder modifies an existing order
func (ob *GoOrderBook) ModifyOrder(orderID uint64, newPrice, newQuantity float64) bool {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	order, exists := ob.orders[orderID]
	if !exists {
		return false
	}

	// Remove from old position
	if order.Side == Buy {
		ob.removeBid(orderID)
	} else {
		ob.removeAsk(orderID)
	}

	// Update order
	order.Price = newPrice
	order.Quantity = newQuantity

	// Re-insert at new position
	if order.Side == Buy {
		ob.insertBid(orderID)
	} else {
		ob.insertAsk(orderID)
	}

	return true
}

// MatchOrders executes matching orders
func (ob *GoOrderBook) MatchOrders() []Trade {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	var trades []Trade

	for len(ob.bids) > 0 && len(ob.asks) > 0 {
		bidID := ob.bids[0]
		askID := ob.asks[0]

		bid := ob.orders[bidID]
		ask := ob.orders[askID]

		if bid.Price >= ask.Price {
			// Execute trade
			tradeQty := bid.Quantity
			if ask.Quantity < tradeQty {
				tradeQty = ask.Quantity
			}

			trade := Trade{
				ID:          uint64(len(trades) + 1),
				BuyOrderID:  bidID,
				SellOrderID: askID,
				Price:       ask.Price,
				Quantity:    tradeQty,
				Timestamp:   time.Now(),
			}
			trades = append(trades, trade)

			// Update quantities
			bid.Quantity -= tradeQty
			bid.FilledQuantity += tradeQty
			ask.Quantity -= tradeQty
			ask.FilledQuantity += tradeQty

			// Update volume
			ob.volume += uint64(tradeQty)

			// Remove filled orders
			if bid.Quantity == 0 {
				delete(ob.orders, bidID)
				ob.bids = ob.bids[1:]
			}
			if ask.Quantity == 0 {
				delete(ob.orders, askID)
				ob.asks = ob.asks[1:]
			}
		} else {
			break
		}
	}

	return trades
}

// GetBestBid returns the best bid price
func (ob *GoOrderBook) GetBestBid() float64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	if len(ob.bids) == 0 {
		return 0
	}

	return ob.orders[ob.bids[0]].Price
}

// GetBestAsk returns the best ask price
func (ob *GoOrderBook) GetBestAsk() float64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	if len(ob.asks) == 0 {
		return 0
	}

	return ob.orders[ob.asks[0]].Price
}

// GetDepth returns the order book depth
func (ob *GoOrderBook) GetDepth(levels int) *Depth {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	depth := &Depth{
		Bids: make([]PriceLevel, 0, levels),
		Asks: make([]PriceLevel, 0, levels),
	}

	// Aggregate bids by price
	bidLevels := make(map[float64]float64)
	for _, id := range ob.bids {
		order := ob.orders[id]
		bidLevels[order.Price] += order.Quantity
	}

	// Aggregate asks by price
	askLevels := make(map[float64]float64)
	for _, id := range ob.asks {
		order := ob.orders[id]
		askLevels[order.Price] += order.Quantity
	}

	// Convert to price levels
	for price, size := range bidLevels {
		depth.Bids = append(depth.Bids, PriceLevel{Price: price, Size: size})
		if len(depth.Bids) >= levels {
			break
		}
	}

	for price, size := range askLevels {
		depth.Asks = append(depth.Asks, PriceLevel{Price: price, Size: size})
		if len(depth.Asks) >= levels {
			break
		}
	}

	return depth
}

// GetVolume returns the total volume
func (ob *GoOrderBook) GetVolume() uint64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	return ob.volume
}

// Helper methods for maintaining sorted order
func (ob *GoOrderBook) insertBid(orderID uint64) {
	order := ob.orders[orderID]

	// Find insertion point (descending price order)
	pos := 0
	for pos < len(ob.bids) {
		if ob.orders[ob.bids[pos]].Price < order.Price {
			break
		}
		pos++
	}

	// Insert at position
	ob.bids = append(ob.bids, 0)
	copy(ob.bids[pos+1:], ob.bids[pos:])
	ob.bids[pos] = orderID
}

func (ob *GoOrderBook) insertAsk(orderID uint64) {
	order := ob.orders[orderID]

	// Find insertion point (ascending price order)
	pos := 0
	for pos < len(ob.asks) {
		if ob.orders[ob.asks[pos]].Price > order.Price {
			break
		}
		pos++
	}

	// Insert at position
	ob.asks = append(ob.asks, 0)
	copy(ob.asks[pos+1:], ob.asks[pos:])
	ob.asks[pos] = orderID
}

func (ob *GoOrderBook) removeBid(orderID uint64) {
	for i, id := range ob.bids {
		if id == orderID {
			ob.bids = append(ob.bids[:i], ob.bids[i+1:]...)
			break
		}
	}
}

func (ob *GoOrderBook) removeAsk(orderID uint64) {
	for i, id := range ob.asks {
		if id == orderID {
			ob.asks = append(ob.asks[:i], ob.asks[i+1:]...)
			break
		}
	}
}
