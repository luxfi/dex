package engine

import (
	"github.com/luxfi/dex/backend/pkg/orderbook"
)

// EngineImplementation represents the underlying engine type
type EngineImplementation string

const (
	ImplGo     EngineImplementation = "go"
	ImplCpp    EngineImplementation = "cpp"
	ImplHybrid EngineImplementation = "hybrid"
)

// GetDefaultImplementation returns the implementation based on build tags
func GetDefaultImplementation() EngineImplementation {
	if isUsingCGO() {
		return ImplCpp
	}
	return ImplGo
}

// NewOrderBook creates an order book with the appropriate implementation
func NewOrderBook(symbol string) orderbook.OrderBook {
	impl := GetDefaultImplementation()
	
	config := orderbook.Config{
		Symbol:     symbol,
		MaxDepth:   1000,
		UseCache:   true,
		UseCpp:     impl == ImplCpp,
	}
	
	switch impl {
	case ImplCpp:
		// When CGO is enabled, use C++ order book
		return orderbook.NewCGOOrderBook(config)
	default:
		// Pure Go implementation
		return orderbook.New(config)
	}
}