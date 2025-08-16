// +build !cgo

package orderbook

// NewOrderBookImpl creates a pure Go orderbook when CGO is disabled
func NewOrderBookImpl(cfg Config) OrderBook {
	return &GoOrderBook{
		symbol: cfg.Symbol,
		orders: make(map[uint64]*Order),
		nextID: 1,
	}
}
