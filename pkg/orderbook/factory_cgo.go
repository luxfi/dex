//go:build cgo && never
// +build cgo,never

package orderbook

// NewOrderBookImpl creates a CGO-backed orderbook when CGO is enabled
func NewOrderBookImpl(cfg Config) OrderBook {
	if cfg.Implementation == ImplCpp {
		return NewCGOOrderBook(cfg)
	}
	return &GoOrderBook{
		symbol: cfg.Symbol,
		orders: make(map[uint64]*Order),
		nextID: 1,
	}
}
