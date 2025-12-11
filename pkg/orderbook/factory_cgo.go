//go:build cgo
// +build cgo

package orderbook

import "os"

// NewOrderBook creates an order book.
// With CGO enabled, uses C++ for performance.
// Set LX_ORDERBOOK_IMPL=go to use pure Go instead.
func NewOrderBook(cfg Config) OrderBook {
	if os.Getenv("LX_ORDERBOOK_IMPL") == "go" {
		return NewGoOrderBook(cfg)
	}
	return NewCppOrderBook(cfg)
}
