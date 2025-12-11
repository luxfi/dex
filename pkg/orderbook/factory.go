//go:build !cgo
// +build !cgo

package orderbook

// NewOrderBook creates a pure Go order book.
func NewOrderBook(cfg Config) OrderBook {
	return NewGoOrderBook(cfg)
}
