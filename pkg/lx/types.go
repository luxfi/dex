package lx

import (
	"sync"
	"time"
)

// Side represents order side (buy/sell)
type Side int

const (
	Buy Side = iota
	Sell
)

// OrderType represents the type of order
type OrderType int

const (
	Limit OrderType = iota
	Market
	Stop
	StopLimit
	Iceberg
	Peg
	Bracket
)

// Order represents a trading order
type Order struct {
	ID        uint64
	Symbol    string
	Type      OrderType
	Side      Side
	Price     float64
	Size      float64
	Filled    float64
	Timestamp time.Time
	UserID    string
	User      string // Alias for UserID
	ClientID  string
	
	// Advanced order fields
	StopPrice    float64
	DisplaySize  float64 // For iceberg orders
	PegOffset    float64
	TakeProfit   float64
	StopLoss     float64
	TimeInForce  string
	PostOnly     bool
	ReduceOnly   bool
	
	// Internal fields
	RemainingSize float64
	Status        string
	Flags         OrderFlags
}

// Trade represents an executed trade
type Trade struct {
	ID        uint64
	Price     float64
	Size      float64
	BuyOrder  uint64
	SellOrder uint64
	Buyer     string
	Seller    string
	Timestamp time.Time
	IsMaker   bool
}

// MarketDataUpdate represents market data updates
type MarketDataUpdate struct {
	Type      string
	Symbol    string
	Timestamp time.Time
	Data      interface{}
}

// PriceLevel represents a price level in the order book
type PriceLevel struct {
	Price float64
	Size  float64
	Count int
}

// IcebergData represents iceberg order data
type IcebergData struct {
	TotalSize   float64
	DisplaySize float64
	Remaining   float64
}

// ConditionalOrder represents a conditional order
type ConditionalOrder struct {
	Condition string
	Order     *Order
}

// MarketUpdate represents a market update
type MarketUpdate struct {
	Type   string
	Symbol string
	Data   interface{}
}

// TimeInForce represents time in force options
type TimeInForce uint8

const (
	TIF_DAY TimeInForce = iota
	TIF_IOC
	TIF_FOK
	TIF_GTC
)

// OrderStatus represents order status
type OrderStatus uint8

const (
	StatusNew OrderStatus = iota
	StatusPartiallyFilled
	StatusFilled
	StatusCanceled
	StatusRejected
)

// Status string constants
const (
	Open              = "open"
	PartiallyFilled   = "partially_filled"
	Filled            = "filled"
	Canceled          = "canceled"
	Rejected          = "rejected"
	ImmediateOrCancel = "IOC"
	FillOrKill        = "FOK"
	OrderAdded        = "order_added"
)

// IcebergState represents iceberg order state
type IcebergState struct {
	TotalSize     float64
	DisplaySize   float64
	RemainingSize float64
	RefillCount   int
}

// Event represents a trading event
type Event struct {
	Type      string
	Timestamp time.Time
	Data      interface{}
}

// OrderFlags represents order flags
type OrderFlags uint32

const (
	FlagPostOnly   OrderFlags = 1 << iota
	FlagReduceOnly
	FlagImmediate
	FlagHidden
)

// OrderBookDepth represents order book depth
type OrderBookDepth struct {
	Symbol    string
	Timestamp time.Time
	Bids      []PriceLevel
	Asks      []PriceLevel
	Sequence  uint64
}

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

// OrderLevel represents an individual order at a price level
type OrderLevel struct {
	Price    float64
	Size     float64
	OrderID  uint64
	UserID   string
	ClientID string
}

// OrderBookSnapshot represents the full order book state
type OrderBookSnapshot struct {
	Symbol    string
	Timestamp time.Time
	Bids      []OrderLevel
	Asks      []OrderLevel
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
