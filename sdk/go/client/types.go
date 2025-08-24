package client

import "time"

// OrderType represents the type of order
type OrderType int32

const (
	OrderTypeLimit     OrderType = 0
	OrderTypeMarket    OrderType = 1
	OrderTypeStop      OrderType = 2
	OrderTypeStopLimit OrderType = 3
	OrderTypeIceberg   OrderType = 4
	OrderTypePeg       OrderType = 5
)

// OrderSide represents the side of an order
type OrderSide int32

const (
	OrderSideBuy  OrderSide = 0
	OrderSideSell OrderSide = 1
)

// OrderStatus represents the status of an order
type OrderStatus string

const (
	OrderStatusOpen      OrderStatus = "open"
	OrderStatusPartial   OrderStatus = "partial"
	OrderStatusFilled    OrderStatus = "filled"
	OrderStatusCancelled OrderStatus = "cancelled"
	OrderStatusRejected  OrderStatus = "rejected"
)

// TimeInForce represents order time in force
type TimeInForce string

const (
	TimeInForceGTC TimeInForce = "GTC" // Good Till Cancelled
	TimeInForceIOC TimeInForce = "IOC" // Immediate Or Cancel
	TimeInForceFOK TimeInForce = "FOK" // Fill Or Kill
	TimeInForceDAY TimeInForce = "DAY" // Day Order
)

// Order represents a trading order
type Order struct {
	OrderID     uint64      `json:"orderId,omitempty"`
	Symbol      string      `json:"symbol"`
	Type        OrderType   `json:"type"`
	Side        OrderSide   `json:"side"`
	Price       float64     `json:"price"`
	Size        float64     `json:"size"`
	Filled      float64     `json:"filled,omitempty"`
	Remaining   float64     `json:"remaining,omitempty"`
	Status      OrderStatus `json:"status,omitempty"`
	UserID      string      `json:"userId,omitempty"`
	ClientID    string      `json:"clientId,omitempty"`
	Timestamp   int64       `json:"timestamp,omitempty"`
	TimeInForce TimeInForce `json:"timeInForce,omitempty"`
	PostOnly    bool        `json:"postOnly,omitempty"`
	ReduceOnly  bool        `json:"reduceOnly,omitempty"`
}

// OrderResponse represents the response to an order request
type OrderResponse struct {
	OrderID uint64 `json:"orderId"`
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// Trade represents a completed trade
type Trade struct {
	TradeID     uint64    `json:"tradeId"`
	Symbol      string    `json:"symbol"`
	Price       float64   `json:"price"`
	Size        float64   `json:"size"`
	Side        OrderSide `json:"side"`
	BuyOrderID  uint64    `json:"buyOrderId"`
	SellOrderID uint64    `json:"sellOrderId"`
	BuyerID     string    `json:"buyerId"`
	SellerID    string    `json:"sellerId"`
	Timestamp   int64     `json:"timestamp"`
}

// PriceLevel represents a price level in the order book
type PriceLevel struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
	Count int32   `json:"count,omitempty"`
}

// OrderBook represents the order book for a symbol
type OrderBook struct {
	Symbol    string       `json:"symbol"`
	Bids      []PriceLevel `json:"bids"`
	Asks      []PriceLevel `json:"asks"`
	Timestamp int64        `json:"timestamp"`
}

// NodeInfo represents node information
type NodeInfo struct {
	Version     string `json:"version"`
	Network     string `json:"network"`
	OrderCount  int64  `json:"orderCount"`
	TradeCount  int64  `json:"tradeCount"`
	Timestamp   int64  `json:"timestamp"`
	BlockHeight int64  `json:"blockHeight,omitempty"`
	Syncing     bool   `json:"syncing"`
	Uptime      int64  `json:"uptime,omitempty"`
}

// Balance represents account balance
type Balance struct {
	Asset     string  `json:"asset"`
	Available float64 `json:"available"`
	Locked    float64 `json:"locked"`
	Total     float64 `json:"total"`
}

// Position represents a trading position
type Position struct {
	Symbol     string  `json:"symbol"`
	Size       float64 `json:"size"`
	EntryPrice float64 `json:"entryPrice"`
	MarkPrice  float64 `json:"markPrice"`
	PnL        float64 `json:"pnl"`
	Margin     float64 `json:"margin"`
}

// Helper methods

// IsOpen returns true if the order is open
func (o *Order) IsOpen() bool {
	return o.Status == OrderStatusOpen || o.Status == OrderStatusPartial
}

// IsClosed returns true if the order is closed
func (o *Order) IsClosed() bool {
	return o.Status == OrderStatusFilled || o.Status == OrderStatusCancelled || o.Status == OrderStatusRejected
}

// FillRate returns the fill rate of the order
func (o *Order) FillRate() float64 {
	if o.Size > 0 {
		return o.Filled / o.Size
	}
	return 0
}

// TotalValue returns the total value of the trade
func (t *Trade) TotalValue() float64 {
	return t.Price * t.Size
}

// TimestampTime returns the timestamp as a time.Time
func (t *Trade) TimestampTime() time.Time {
	return time.Unix(t.Timestamp, 0)
}

// BestBid returns the best bid price
func (ob *OrderBook) BestBid() float64 {
	if len(ob.Bids) > 0 {
		return ob.Bids[0].Price
	}
	return 0
}

// BestAsk returns the best ask price
func (ob *OrderBook) BestAsk() float64 {
	if len(ob.Asks) > 0 {
		return ob.Asks[0].Price
	}
	return 0
}

// Spread returns the bid-ask spread
func (ob *OrderBook) Spread() float64 {
	bid := ob.BestBid()
	ask := ob.BestAsk()
	if bid > 0 && ask > 0 {
		return ask - bid
	}
	return 0
}

// MidPrice returns the mid price
func (ob *OrderBook) MidPrice() float64 {
	bid := ob.BestBid()
	ask := ob.BestAsk()
	if bid > 0 && ask > 0 {
		return (bid + ask) / 2
	}
	return 0
}

// SpreadPercentage returns the spread as a percentage
func (ob *OrderBook) SpreadPercentage() float64 {
	spread := ob.Spread()
	mid := ob.MidPrice()
	if mid > 0 {
		return (spread / mid) * 100
	}
	return 0
}

// UnrealizedPnL calculates unrealized P&L
func (p *Position) UnrealizedPnL() float64 {
	return (p.MarkPrice - p.EntryPrice) * p.Size
}

// PnLPercentage calculates P&L as a percentage
func (p *Position) PnLPercentage() float64 {
	if p.EntryPrice > 0 {
		return ((p.MarkPrice - p.EntryPrice) / p.EntryPrice) * 100
	}
	return 0
}

// Utilization calculates balance utilization
func (b *Balance) Utilization() float64 {
	if b.Total > 0 {
		return b.Locked / b.Total
	}
	return 0
}
