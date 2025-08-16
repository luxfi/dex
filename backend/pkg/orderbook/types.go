package orderbook

import (
	"time"
)

// OrderSide represents buy or sell
type OrderSide uint8

const (
	Buy OrderSide = iota
	Sell
)

// OrderType represents market or limit order
type OrderType uint8

const (
	Market OrderType = iota
	Limit
)

// OrderStatus represents the current state of an order
type OrderStatus uint8

const (
	Pending OrderStatus = iota
	PartiallyFilled
	Filled
	Cancelled
)

// Order represents a trading order
type Order struct {
	ID             uint64      `json:"id"`
	UserID         uint64      `json:"user_id"`
	Symbol         string      `json:"symbol"`
	Price          float64     `json:"price"`
	Quantity       float64     `json:"quantity"`
	FilledQuantity float64     `json:"filled_quantity"`
	Side           OrderSide   `json:"side"`
	Type           OrderType   `json:"type"`
	Status         OrderStatus `json:"status"`
	Timestamp      time.Time   `json:"timestamp"`
}

// Trade represents an executed trade
type Trade struct {
	ID          uint64    `json:"id"`
	BuyOrderID  uint64    `json:"buy_order_id"`
	SellOrderID uint64    `json:"sell_order_id"`
	Price       float64   `json:"price"`
	Quantity    float64   `json:"quantity"`
	Timestamp   time.Time `json:"timestamp"`
}

// PriceLevel represents a price and size at that level
type PriceLevel struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
}

// Depth represents the order book depth
type Depth struct {
	Bids []PriceLevel `json:"bids"`
	Asks []PriceLevel `json:"asks"`
}

// Config holds orderbook configuration
type Config struct {
	Implementation    Implementation
	Symbol           string
	MaxOrdersPerLevel int
	PricePrecision    int
}

// OrderBook interface defines the operations for an order book
type OrderBook interface {
	AddOrder(order *Order) uint64
	CancelOrder(orderID uint64) bool
	ModifyOrder(orderID uint64, newPrice, newQuantity float64) bool
	MatchOrders() []Trade
	GetBestBid() float64
	GetBestAsk() float64
	GetDepth(level int) *Depth
	GetVolume() uint64
}