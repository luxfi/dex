package lx

import (
	"sync"
	"time"
)

// TradingEngine is the main trading engine
type TradingEngine struct {
	OrderBooks     map[string]*OrderBook
	PerpManager    *PerpetualManager
	VaultManager   *VaultManager
	LendingManager *LendingManager
	Events         chan Event
	mu             sync.RWMutex
}

// EngineConfig configuration for the trading engine
type EngineConfig struct {
	EnablePerps   bool
	EnableVaults  bool
	EnableLending bool
}

// Order represents a trading order
type Order struct {
	ID         uint64
	Symbol     string
	Side       Side
	Type       OrderType
	Price      float64
	Size       float64
	Filled     float64
	Status     OrderStatus
	User       string
	ClientID   string
	Timestamp  time.Time
	PostOnly   bool
	ReduceOnly bool
	IOC        bool // Immediate or Cancel
	FOK        bool // Fill or Kill
}

// OrderType represents the type of order
type OrderType int

const (
	Market OrderType = iota
	Limit
	Stop
	StopLimit
	TrailingStop
)

// OrderBook represents a trading pair's order book
type OrderBook struct {
	Symbol       string
	Bids         *OrderTree
	Asks         *OrderTree
	Trades       []Trade
	Orders       map[uint64]*Order
	UserOrders   map[string][]uint64
	LastTradeID  uint64
	LastOrderID  uint64
	mu           sync.RWMutex
}

// Event represents a system event
type Event struct {
	Type      EventType
	Timestamp time.Time
	Data      map[string]interface{}
}

// EventType represents the type of event
type EventType int

const (
	EventTrade EventType = iota
	EventOrderPlaced
	EventOrderCancelled
	EventOrderModified
	EventLiquidation
	EventFunding
	EventLending
	EventBorrowing
)

// PriceOracle represents a price oracle
type PriceOracle struct {
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
		Events:     make(chan Event, 10000),
	}

	if config.EnablePerps {
		engine.PerpManager = NewPerpetualManager(engine)
	}
	if config.EnableVaults {
		engine.VaultManager = NewVaultManager(engine)
	}
	if config.EnableLending {
		engine.LendingManager = NewLendingManager(engine)
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
