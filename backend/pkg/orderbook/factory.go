package orderbook

import (
	"os"
)

// Implementation type
type Implementation string

const (
	ImplGo  Implementation = "go"
	ImplCpp Implementation = "cpp"
)

// NewOrderBook creates a new order book with the specified implementation
func NewOrderBook(cfg Config) OrderBook {
	// Check environment variable override
	if impl := os.Getenv("LX_ORDERBOOK_IMPL"); impl != "" {
		cfg.Implementation = Implementation(impl)
	}

	// The actual implementation is selected via build tags
	// CGO builds will use NewOrderBookImpl from cpp_orderbook.go
	// Non-CGO builds will use NewOrderBookImpl from go_orderbook.go
	return NewOrderBookImpl(cfg)
}

// GetImplementation returns the current implementation being used
func GetImplementation() Implementation {
	if os.Getenv("CGO_ENABLED") == "1" {
		return ImplCpp
	}
	return ImplGo
}
