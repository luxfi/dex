package lx

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Extended order types - defined in types_common.go

// Time in Force - defined in types_common.go

// Update types - defined in types_common.go

// TimeInForce and UpdateType - defined in types_common.go

// MarketDataUpdate - defined in types_common.go

// PriceLevel - defined in types_common.go

// OrderBookDepth - defined in types_common.go

// OrderBookSnapshot - defined in types_common.go

// ExtendedOrderBook adds advanced features to the basic order book
type ExtendedOrderBook struct {
	*OrderBook
	
	// Market data subscribers
	subscribers     []chan MarketDataUpdate
	subscriberMutex sync.RWMutex
	
	// Stop orders tracking
	stopBuyOrders  map[uint64]*Order
	stopSellOrders map[uint64]*Order
	
	// Iceberg orders tracking
	icebergOrders map[uint64]*IcebergState
	
	// Advanced features
	EnableSelfTradePrevention bool
	EnablePostOnly            bool
	EnableHiddenOrders        bool
	
	// Performance metrics
	totalVolume      float64
	totalTrades      uint64
	lastUpdateID     uint64
	lastSnapshotTime time.Time
}

// IcebergState - defined in types_common.go

// NewExtendedOrderBook creates an order book with advanced features
func NewExtendedOrderBook(symbol string) *ExtendedOrderBook {
	return &ExtendedOrderBook{
		OrderBook:      NewOrderBook(symbol),
		subscribers:    make([]chan MarketDataUpdate, 0),
		stopBuyOrders:  make(map[uint64]*Order),
		stopSellOrders: make(map[uint64]*Order),
		icebergOrders:  make(map[uint64]*IcebergState),
	}
}

// Subscribe adds a market data subscriber
func (book *ExtendedOrderBook) Subscribe(ch chan MarketDataUpdate) {
	book.subscriberMutex.Lock()
	defer book.subscriberMutex.Unlock()
	book.subscribers = append(book.subscribers, ch)
}

// Unsubscribe removes a market data subscriber
func (book *ExtendedOrderBook) Unsubscribe(ch chan MarketDataUpdate) {
	book.subscriberMutex.Lock()
	defer book.subscriberMutex.Unlock()
	
	for i, subscriber := range book.subscribers {
		if subscriber == ch {
			book.subscribers = append(book.subscribers[:i], book.subscribers[i+1:]...)
			break
		}
	}
}

// publishUpdate sends market data updates to all subscribers
func (book *ExtendedOrderBook) publishUpdate(update MarketDataUpdate) {
	book.subscriberMutex.RLock()
	defer book.subscriberMutex.RUnlock()
	
	for _, ch := range book.subscribers {
		select {
		case ch <- update:
		default:
			// Non-blocking send, skip if channel is full
		}
	}
}

// AddOrderExtended adds an order with extended features
func (book *ExtendedOrderBook) AddOrderExtended(order *Order) (uint64, error) {
	book.mu.Lock()
	defer book.mu.Unlock()
	
	// Validate order
	if err := book.validateOrder(order); err != nil {
		return 0, err
	}
	
	// Handle different order types
	switch order.Type {
	case Stop, StopLimit:
		return book.addStopOrder(order)
	case Iceberg:
		return book.addIcebergOrder(order)
	case Hidden:
		return book.addHiddenOrder(order)
	default:
		return book.addRegularOrder(order)
	}
}

// validateOrder performs order validation
func (book *ExtendedOrderBook) validateOrder(order *Order) error {
	if order == nil {
		return errors.New("order cannot be nil")
	}
	
	if order.Size <= 0 {
		return errors.New("order size must be positive")
	}
	
	if order.Type == Limit && order.Price <= 0 {
		return errors.New("limit order price must be positive")
	}
	
	// Self-trade prevention
	if book.EnableSelfTradePrevention && order.UserID != "" {
		if book.wouldSelfTrade(order) {
			return errors.New("order would result in self-trade")
		}
	}
	
	// Post-only validation
	if order.PostOnly && book.wouldCrossSpread(order) {
		return errors.New("post-only order would cross the spread")
	}
	
	return nil
}

// wouldSelfTrade checks if order would match with user's own orders
func (book *ExtendedOrderBook) wouldSelfTrade(order *Order) bool {
	userOrders := book.UserOrders[order.UserID]
	
	for _, orderID := range userOrders {
		existingOrder := book.Orders[orderID]
		if existingOrder != nil && existingOrder.Side != order.Side {
			// Check if prices would match
			if order.Type == Market {
				return true
			}
			if order.Side == Buy && order.Price >= existingOrder.Price {
				return true
			}
			if order.Side == Sell && order.Price <= existingOrder.Price {
				return true
			}
		}
	}
	
	return false
}

// wouldCrossSpread checks if order would immediately match
func (book *ExtendedOrderBook) wouldCrossSpread(order *Order) bool {
	// Simplified implementation - would need access to OrderTree internals
	return false
}

// addStopOrder adds a stop or stop-limit order
func (book *ExtendedOrderBook) addStopOrder(order *Order) (uint64, error) {
	if order.Side == Buy {
		book.stopBuyOrders[order.ID] = order
	} else {
		book.stopSellOrders[order.ID] = order
	}
	
	book.Orders[order.ID] = order
	
	// Track user orders
	if order.UserID != "" {
		book.UserOrders[order.UserID] = append(book.UserOrders[order.UserID], order.ID)
	}
	
	book.publishUpdate(MarketDataUpdate{
		Type:      OrderAdded,
		OrderID:   order.ID,
		Price:     order.Price,
		Size:      order.Size,
		Side:      order.Side,
		Timestamp: time.Now(),
		Symbol:    book.Symbol,
	})
	
	return 0, nil
}

// addIcebergOrder adds an iceberg order
func (book *ExtendedOrderBook) addIcebergOrder(order *Order) (uint64, error) {
	// Create iceberg state
	icebergState := &IcebergState{
		TotalSize:      order.Size,
		RemainingSize:  order.Size,
		DisplaySize:    order.DisplaySize,
		CurrentOrderID: order.ID,
		RefillCount:    0,
	}
	
	book.icebergOrders[order.ID] = icebergState
	
	// Add visible portion as regular order
	visibleOrder := *order
	visibleOrder.Size = order.DisplaySize
	
	return book.addRegularOrder(&visibleOrder)
}

// addHiddenOrder adds a hidden order
func (book *ExtendedOrderBook) addHiddenOrder(order *Order) (uint64, error) {
	// Hidden orders are added but not displayed in market data
	book.Orders[order.ID] = order
	
	// Add to appropriate tree but mark as hidden
	// Simplified - would need proper AddOrder method
	book.Orders[order.ID] = order
	
	return 0, nil
}

// addRegularOrder adds a regular limit or market order
func (book *ExtendedOrderBook) addRegularOrder(order *Order) (uint64, error) {
	// Handle time-in-force
	switch order.TimeInForce {
	case FillOrKill:
		if !book.canFullyFill(order) {
			return 0, errors.New("FOK order cannot be fully filled")
		}
	case ImmediateOrCancel:
		// Will be handled after matching
	}
	
	// Process the order
	trades := book.AddOrder(order)
	
	// Handle IOC orders
	if order.TimeInForce == ImmediateOrCancel {
		if remainingOrder := book.Orders[order.ID]; remainingOrder != nil {
			book.CancelOrder(order.ID)
		}
	}
	
	// Publish update
	book.publishUpdate(MarketDataUpdate{
		Type:      OrderAdded,
		OrderID:   order.ID,
		Price:     order.Price,
		Size:      order.Size,
		Side:      order.Side,
		Timestamp: time.Now(),
		Symbol:    book.Symbol,
	})
	
	return trades, nil
}

// canFullyFill checks if an order can be fully filled
func (book *ExtendedOrderBook) canFullyFill(order *Order) bool {
	// Simplified implementation - needs OrderTree methods
	return true // Assume it can be filled for now
}

// GetDepth returns the order book depth up to specified levels
func (book *ExtendedOrderBook) GetDepth(levels int) OrderBookDepth {
	book.mu.RLock()
	defer book.mu.RUnlock()
	
	depth := OrderBookDepth{
		Bids:         book.aggregateLevels(book.Bids, levels, false),
		Asks:         book.aggregateLevels(book.Asks, levels, true),
		LastUpdateID: book.lastUpdateID,
		Timestamp:    time.Now(),
	}
	
	return depth
}

// aggregateLevels aggregates orders into price levels
func (book *ExtendedOrderBook) aggregateLevels(tree *OrderTree, maxLevels int, ascending bool) []PriceLevel {
	// Simplified implementation - needs OrderTree traversal methods
	return make([]PriceLevel, 0)
}

// GetSnapshot returns a full order book snapshot
func (book *ExtendedOrderBook) GetSnapshot() OrderBookSnapshot {
	book.mu.RLock()
	defer book.mu.RUnlock()
	
	snapshot := OrderBookSnapshot{
		Symbol:       book.Symbol,
		Bids:         book.aggregateLevels(book.Bids, 0, false),
		Asks:         book.aggregateLevels(book.Asks, 0, true),
		LastTradeID:  book.LastTradeID,
		LastUpdateID: book.lastUpdateID,
		Timestamp:    time.Now(),
	}
	
	book.lastSnapshotTime = snapshot.Timestamp
	
	return snapshot
}

// Reset clears the order book
func (book *ExtendedOrderBook) Reset() {
	book.mu.Lock()
	defer book.mu.Unlock()
	
	// Clear all data structures
	book.Orders = make(map[uint64]*Order)
	book.UserOrders = make(map[string][]uint64)
	book.Trades = []Trade{}
	book.Bids = NewOrderTree(Buy)
	book.Asks = NewOrderTree(Sell)
	book.stopBuyOrders = make(map[uint64]*Order)
	book.stopSellOrders = make(map[uint64]*Order)
	book.icebergOrders = make(map[uint64]*IcebergState)
	
	// Reset counters
	book.LastTradeID = 0
	book.LastOrderID = 0
	book.lastUpdateID = 0
	book.totalVolume = 0
	book.totalTrades = 0
	
	// Notify subscribers
	book.publishUpdate(MarketDataUpdate{
		Type:      BookReset,
		Timestamp: time.Now(),
		Symbol:    book.Symbol,
	})
}

// CheckStopOrders checks and triggers stop orders based on last trade price
func (book *ExtendedOrderBook) CheckStopOrders(lastTradePrice float64) {
	book.mu.Lock()
	defer book.mu.Unlock()
	
	// Check stop buy orders (triggered when price rises above stop price)
	for orderID, stopOrder := range book.stopBuyOrders {
		if lastTradePrice >= stopOrder.Price {
			// Convert to market or limit order
			if stopOrder.Type == Stop {
				stopOrder.Type = Market
			} else {
				stopOrder.Type = Limit
				stopOrder.Price = stopOrder.LimitPrice
			}
			
			// Add to order book
			book.AddOrder(stopOrder)
			
			// Remove from stop orders
			delete(book.stopBuyOrders, orderID)
		}
	}
	
	// Check stop sell orders (triggered when price falls below stop price)
	for orderID, stopOrder := range book.stopSellOrders {
		if lastTradePrice <= stopOrder.Price {
			// Convert to market or limit order
			if stopOrder.Type == Stop {
				stopOrder.Type = Market
			} else {
				stopOrder.Type = Limit
				stopOrder.Price = stopOrder.LimitPrice
			}
			
			// Add to order book
			book.AddOrder(stopOrder)
			
			// Remove from stop orders
			delete(book.stopSellOrders, orderID)
		}
	}
}

// RefillIcebergOrder refills the visible portion of an iceberg order
func (book *ExtendedOrderBook) RefillIcebergOrder(orderID uint64) {
	icebergState, exists := book.icebergOrders[orderID]
	if !exists || icebergState.RemainingSize <= 0 {
		return
	}
	
	// Calculate refill size
	refillSize := icebergState.DisplaySize
	if refillSize > icebergState.RemainingSize {
		refillSize = icebergState.RemainingSize
	}
	
	// Get original order details
	originalOrder := book.Orders[icebergState.CurrentOrderID]
	if originalOrder == nil {
		return
	}
	
	// Create new visible order
	newOrder := &Order{
		ID:        book.LastOrderID + 1,
		Type:      originalOrder.Type,
		Side:      originalOrder.Side,
		Price:     originalOrder.Price,
		Size:      refillSize,
		UserID:    originalOrder.UserID,
		Timestamp: time.Now(),
	}
	
	book.LastOrderID++
	icebergState.CurrentOrderID = newOrder.ID
	icebergState.RemainingSize -= refillSize
	icebergState.RefillCount++
	
	// Add new visible portion
	book.AddOrder(newOrder)
}

// GetStatistics returns order book statistics
func (book *ExtendedOrderBook) GetStatistics() map[string]interface{} {
	book.mu.RLock()
	defer book.mu.RUnlock()
	
	stats := make(map[string]interface{})
	
	// Basic stats
	stats["symbol"] = book.Symbol
	stats["total_orders"] = len(book.Orders)
	stats["total_trades"] = book.totalTrades
	stats["total_volume"] = book.totalVolume
	stats["last_trade_id"] = book.LastTradeID
	stats["last_update_id"] = book.lastUpdateID
	
	// Order counts by type
	stats["bid_orders"] = 0 // Simplified
	stats["ask_orders"] = 0 // Simplified
	stats["stop_orders"] = len(book.stopBuyOrders) + len(book.stopSellOrders)
	stats["iceberg_orders"] = len(book.icebergOrders)
	
	// Price stats - simplified (needs OrderTree methods)
	stats["best_bid"] = 0.0
	stats["best_ask"] = 0.0
	
	// Performance metrics
	stats["subscribers"] = len(book.subscribers)
	stats["last_snapshot_time"] = book.lastSnapshotTime
	
	return stats
}

// GetBestPrices returns the best bid and ask prices
func (book *ExtendedOrderBook) GetBestPrices() (bestBid, bestAsk float64) {
	// Simplified - needs OrderTree methods
	return 0.0, 0.0
}

// GetSpread returns the bid-ask spread
func (book *ExtendedOrderBook) GetSpread() float64 {
	bestBid, bestAsk := book.GetBestPrices()
	if bestBid > 0 && bestAsk > 0 {
		return bestAsk - bestBid
	}
	return 0
}

// GetMidPrice returns the mid-market price
func (book *ExtendedOrderBook) GetMidPrice() float64 {
	bestBid, bestAsk := book.GetBestPrices()
	if bestBid > 0 && bestAsk > 0 {
		return (bestBid + bestAsk) / 2
	}
	return 0
}

// GetVWAP calculates volume-weighted average price for a given size
func (book *ExtendedOrderBook) GetVWAP(side Side, size float64) (float64, error) {
	// Simplified - needs OrderTree traversal
	return 0, fmt.Errorf("not implemented")
}

// GetMarketImpact estimates the market impact of a large order
func (book *ExtendedOrderBook) GetMarketImpact(side Side, size float64) (float64, error) {
	vwap, err := book.GetVWAP(side, size)
	if err != nil {
		return 0, err
	}
	
	midPrice := book.GetMidPrice()
	if midPrice == 0 {
		return 0, errors.New("cannot calculate mid price")
	}
	
	// Market impact as percentage from mid price
	impact := ((vwap - midPrice) / midPrice) * 100
	if side == Sell {
		impact = -impact
	}
	
	return impact, nil
}