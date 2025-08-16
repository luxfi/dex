package types

import (
	"time"
)

// Order represents a trading order
type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`      // "buy" or "sell"
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Type      string    `json:"type"`      // "market", "limit", etc.
	Status    string    `json:"status"`    // "pending", "filled", "cancelled"
	Timestamp time.Time `json:"timestamp"`
	User      string    `json:"user,omitempty"`
	Filled    float64   `json:"filled,omitempty"`
}

// Trade represents an executed trade
type Trade struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Size      float64   `json:"size"`
	BuyerID   string    `json:"buyer_id"`
	SellerID  string    `json:"seller_id"`
	Timestamp time.Time `json:"timestamp"`
}

// OrderBook represents the order book state
type OrderBook struct {
	Symbol string    `json:"symbol"`
	Bids   [][]float64 `json:"bids"` // [price, quantity]
	Asks   [][]float64 `json:"asks"` // [price, quantity]
	Time   time.Time `json:"time"`
}

// Response represents a generic API response
type Response struct {
	Success bool        `json:"success"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}