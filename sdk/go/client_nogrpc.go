//go:build !grpc

// Package lxsdk is the gRPC-backed Go SDK for the LX trading platform.
//
// In the default build (no `grpc` build tag), gRPC support is excluded
// so consumers compile zero gRPC code. NewClient and NewHighFrequencyClient
// return an explanatory error. Rebuild with `-tags=grpc` to get the
// real client.
//
// External trading clients that do not need the binary gRPC transport
// should use the JSON-RPC + WebSocket SDK at
// github.com/luxfi/dex/sdk/go/client instead.
package lxsdk

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// ErrGRPCNotBuilt is returned when the SDK is used without the `grpc`
// build tag. Rebuild with `go build -tags=grpc` to enable.
var ErrGRPCNotBuilt = errors.New("lxsdk: gRPC client requires the `grpc` build tag (rebuild with -tags=grpc)")

// Side represents order side.
type Side int

const (
	Buy Side = iota
	Sell
)

// OrderType represents order type.
type OrderType int

const (
	Market OrderType = iota
	Limit
	Stop
	StopLimit
)

// Order represents a trading order.
type Order struct {
	ID        string
	Symbol    string
	Price     float64
	Quantity  float64
	Side      Side
	Type      OrderType
	Timestamp time.Time
}

// Trade represents an executed trade.
type Trade struct {
	ID        string
	OrderID   string
	Symbol    string
	Price     float64
	Quantity  float64
	Side      Side
	Timestamp time.Time
}

// Client is the no-op placeholder for the gRPC client. All operations
// return ErrGRPCNotBuilt.
type Client struct {
	address string

	// Match the gRPC build's exported observability surface so consumers
	// compile against both builds.
	ordersSent   atomic.Int64
	tradesRecv   atomic.Int64
	latencyNanos atomic.Int64

	mu sync.RWMutex
}

// NewClient returns an error in the default build. Rebuild with `-tags=grpc`.
func NewClient(address string) (*Client, error) {
	return nil, ErrGRPCNotBuilt
}

// EnableFIX is a no-op stub.
func (c *Client) EnableFIX(senderID, targetID string) {}

// SendOrder returns ErrGRPCNotBuilt.
func (c *Client) SendOrder(symbol string, price, quantity float64, side Side) (*Order, error) {
	return nil, ErrGRPCNotBuilt
}

// SendBulkOrders returns ErrGRPCNotBuilt.
func (c *Client) SendBulkOrders(orders []*Order) error { return ErrGRPCNotBuilt }

// CancelOrder returns ErrGRPCNotBuilt.
func (c *Client) CancelOrder(orderID string) error { return ErrGRPCNotBuilt }

// Subscribe returns ErrGRPCNotBuilt.
func (c *Client) Subscribe(symbols []string, callback func(*Trade)) error { return ErrGRPCNotBuilt }

// GetOrderBook returns ErrGRPCNotBuilt.
func (c *Client) GetOrderBook(symbol string) (bids, asks []Order, err error) {
	return nil, nil, ErrGRPCNotBuilt
}

// GetMetrics returns zeroed metrics.
func (c *Client) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"orders_sent":     c.ordersSent.Load(),
		"trades_received": c.tradesRecv.Load(),
		"latency_ns":      c.latencyNanos.Load(),
		"latency_ms":      float64(c.latencyNanos.Load()) / 1e6,
	}
}

// GetThroughput returns 0.
func (c *Client) GetThroughput() float64 { return 0 }

// Close is a no-op.
func (c *Client) Close() error { return nil }

// HighFrequencyClient placeholder.
type HighFrequencyClient struct{ *Client }

// NewHighFrequencyClient returns ErrGRPCNotBuilt.
func NewHighFrequencyClient(address string) (*HighFrequencyClient, error) {
	return nil, ErrGRPCNotBuilt
}

// SendOrderAsync is a no-op.
func (hft *HighFrequencyClient) SendOrderAsync(symbol string, price, quantity float64, side Side) {
}
