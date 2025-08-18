package lx

import (
	"sync"
	"time"
)

// TradingEngine is the main trading engine
type TradingEngine struct {
	OrderBooks   map[string]*OrderBook
	PerpManager  *PerpetualManager
	VaultManager *VaultManager
	LendingPool  *LendingPool
	Events       chan Event
	Orders       map[uint64]*Order // Track all orders
	mu           sync.RWMutex
}

// EngineConfig configuration for the trading engine
type EngineConfig struct {
	EnablePerps   bool
	EnableVaults  bool
	EnableLending bool
}

// Order is defined in orderbook.go

// OrderType is defined in types_common.go

// OrderBook is defined in orderbook.go

// Event - defined in types_common.go

// EventType - defined in types_common.go

// PriceOracleData represents price oracle data
type PriceOracleData struct {
	Symbol       string
	Source       string
	Price        float64
	Confidence   float64
	LastUpdate   time.Time
	PriceHistory []PricePoint
	mu           sync.RWMutex
}

// PricePoint represents a historical price point
type PricePoint struct {
	Price     float64
	Timestamp time.Time
	Volume    float64
}

// PerformanceMetrics tracks performance metrics
type PerformanceMetrics struct {
	TotalReturn  float64
	SharpeRatio  float64
	MaxDrawdown  float64
	WinRate      float64
	ProfitFactor float64
	UpdatedAt    time.Time
}

// L4Level represents an individual order in L4 data
type L4Level struct {
	Price    float64
	Size     float64
	OrderID  uint64
	UserID   string
	ClientID string
}

// L4BookSnapshot represents the full L4 order book
type L4BookSnapshot struct {
	Symbol    string
	Timestamp time.Time
	Bids      []L4Level
	Asks      []L4Level
	Sequence  uint64
}

// NewTradingEngine creates a new trading engine
func NewTradingEngine(config EngineConfig) *TradingEngine {
	engine := &TradingEngine{
		OrderBooks: make(map[string]*OrderBook),
		Orders:     make(map[uint64]*Order),
		Events:     make(chan Event, 10000),
	}

	if config.EnablePerps {
		engine.PerpManager = NewPerpetualManager(engine)
	}
	if config.EnableVaults {
		engine.VaultManager = NewVaultManager(engine)
	}
	if config.EnableLending {
		engine.LendingPool = NewLendingPool()
	}

	return engine
}

// Start starts the trading engine
func (engine *TradingEngine) Start() error {
	// Start event processing
	go engine.processEvents()
	return nil
}

// Stop stops the trading engine
func (engine *TradingEngine) Stop() error {
	close(engine.Events)
	return nil
}

// CreateSpotMarket creates a new spot market
func (engine *TradingEngine) CreateSpotMarket(symbol string) *OrderBook {
	engine.mu.Lock()
	defer engine.mu.Unlock()

	book := NewOrderBook(symbol)
	engine.OrderBooks[symbol] = book
	return book
}

// processEvents processes system events
func (engine *TradingEngine) processEvents() {
	for event := range engine.Events {
		// Process events (logging, notifications, etc.)
		_ = event
	}
}

// logEvent logs a system event
func (engine *TradingEngine) logEvent(event Event) {
	select {
	case engine.Events <- event:
	default:
		// Event channel full, drop event
	}
}

// GetUserOrders returns all orders for a user
func (e *TradingEngine) GetUserOrders(userID string) []*Order {
	e.mu.RLock()
	defer e.mu.RUnlock()

	orders := make([]*Order, 0)
	for _, order := range e.Orders {
		if order.User == userID {
			orders = append(orders, order)
		}
	}
	return orders
}

// CreateOrderBook creates a new order book for a symbol
func (e *TradingEngine) CreateOrderBook(symbol string) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.OrderBooks[symbol]; !exists {
		e.OrderBooks[symbol] = NewOrderBook(symbol)
	}
}

// GetOrderBook returns the order book for a symbol
func (e *TradingEngine) GetOrderBook(symbol string) *OrderBook {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.OrderBooks[symbol]
}
