package engine

import "errors"

// Common engine errors
var (
	ErrSessionNotFound     = errors.New("session not found")
	ErrOrderNotFound       = errors.New("order not found")
	ErrInstrumentNotFound  = errors.New("instrument not found")
	ErrInsufficientBalance = errors.New("insufficient balance")
	ErrInvalidPrice        = errors.New("invalid price")
	ErrInvalidQuantity     = errors.New("invalid quantity")
	ErrOrderBookFull       = errors.New("order book full")
	ErrEngineShutdown      = errors.New("engine is shutting down")
)

// MarketData represents market data for an instrument
type MarketData struct {
	Symbol     string
	BestBid    float64
	BestAsk    float64
	Volume     uint64
	Depth      interface{}
	UpdateTime interface{}
}

// EngineStats represents engine statistics
type EngineStats struct {
	OrdersProcessed  uint64
	TradesExecuted   uint64
	MessagesReceived uint64
	ActiveOrderBooks int
	ActiveSessions   int
}
