package lx

import (
	"sync"
	"time"
)

// Side represents order side
type Side int

const (
	Buy Side = iota
	Sell
)

// OrderType represents the type of order
type OrderType int

// Basic order types
const (
	Market OrderType = iota
	Limit
)

// Extended order types (start from 10 to avoid conflicts)
const (
	Stop OrderType = iota + 10
	StopLimit
	Iceberg
	Hidden
	Pegged
	TrailingStop
	OneCancelsOther
	AllOrNone
	StopOrder            = Stop // Alias for compatibility
	StopLimitOrder       = StopLimit
	IcebergOrder         = Iceberg
	HiddenOrder          = Hidden
	PeggedOrder          = Pegged
	TrailingStopOrder    = TrailingStop
	OneCancelsOtherOrder = OneCancelsOther
	AllOrNoneOrder       = AllOrNone
)

// OrderStatus represents the status of an order
type OrderStatus int

const (
	// Basic statuses (matching orderbook.go)
	Open OrderStatus = iota
	PartiallyFilled
	Filled
	Cancelled
	Rejected
	Expired

	// Additional statuses for advanced order book
	StatusNew                         = Open // Alias for compatibility
	StatusPartiallyFilled             = PartiallyFilled
	StatusFilled                      = Filled
	StatusCancelled                   = Cancelled
	StatusRejected                    = Rejected
	StatusExpired                     = Expired
	StatusPending         OrderStatus = 100 // For pending orders like stops
)

// TimeInForce represents order time-in-force options
type TimeInForce int

const (
	GTC               TimeInForce = iota // Good Till Cancelled
	IOC                                  // Immediate Or Cancel
	FOK                                  // Fill Or Kill
	GTD                                  // Good Till Date
	GTT                                  // Good Till Time
	ATO                                  // At The Open
	ATC                                  // At The Close
	GoodTillCancelled = GTC
	ImmediateOrCancel = IOC
	FillOrKill        = FOK
	GoodTillDate      = GTD
	GoodTillTime      = GTT
	AtTheOpen         = ATO
	AtTheClose        = ATC
)

// UpdateType represents market data update types
type UpdateType int

const (
	OrderAdded UpdateType = iota
	OrderModified
	OrderCancelled
	TradeExecuted
	BookReset
	SnapshotUpdate
)

// PriceLevel represents an aggregated price level in the order book
type PriceLevel struct {
	Price      float64
	Size       float64
	Count      int
	OrderIDs   []uint64
	Orders     []*Order // For compatibility
	TotalSize  float64
	OrderCount int
	UpdateTime time.Time
	mu         sync.RWMutex
}

// OrderBookSnapshot represents a full order book snapshot
type OrderBookSnapshot struct {
	Symbol       string
	Bids         []PriceLevel
	Asks         []PriceLevel
	LastTradeID  uint64
	LastUpdateID uint64
	Timestamp    time.Time
	Sequence     uint64    // For compatibility
	SeqNum       uint64    // Alias for Sequence
	Time         time.Time // Alias for Timestamp
}

// OrderBookDepth represents the order book depth
type OrderBookDepth struct {
	Bids         []PriceLevel
	Asks         []PriceLevel
	LastUpdateID uint64
	Timestamp    time.Time
}

// MarketDataUpdate represents a market data event
type MarketDataUpdate struct {
	Type      UpdateType
	OrderID   uint64
	Price     float64
	Size      float64
	Side      Side
	Timestamp time.Time
	TradeID   uint64
	Symbol    string
}

// MarketUpdate represents a market update event (alias for compatibility)
type MarketUpdate struct {
	Type      string
	Timestamp time.Time
	Data      interface{}
}

// Trade represents a completed trade
type Trade struct {
	ID         uint64
	Price      float64
	Size       float64
	BuyOrder   interface{} // Can be uint64 or *Order
	SellOrder  interface{} // Can be uint64 or *Order
	Timestamp  time.Time
	Symbol     string
	BuyUserID  string
	SellUserID string
	TakerSide  Side
	MatchType  string // "full", "partial"
	Fee        float64
}

// IcebergState tracks the state of an iceberg order
type IcebergState struct {
	TotalSize      float64
	RemainingSize  float64
	DisplaySize    float64
	CurrentOrderID uint64
	RefillCount    int
}

// IcebergData is an alias for IcebergState (compatibility)
type IcebergData = IcebergState

// ConditionalOrder represents a conditional order
type ConditionalOrder struct {
	TriggerPrice     float64
	TriggerCondition string
	LinkedOrderID    uint64
}

// OrderFlags for order behavior (bitwise flags)
type OrderFlags uint32

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

// Event represents a system event
type Event struct {
	Type      EventType
	Timestamp time.Time
	Data      map[string]interface{}
}

// Order represents a trading order
type Order struct {
	ID            uint64
	Symbol        string
	Side          Side
	Type          OrderType
	Price         float64
	Size          float64
	ExecutedSize  float64
	RemainingSize float64
	Filled        float64 // Alias for ExecutedSize
	Status        OrderStatus
	User          string // Alias for UserID
	UserID        string
	ClientID      string
	ClientOrderID string
	Timestamp     time.Time
	UpdatedAt     time.Time

	// Extended fields for advanced orders
	StopPrice      float64 // For stop orders
	LimitPrice     float64 // For stop-limit orders
	DisplaySize    float64 // For iceberg orders
	TimeInForce    TimeInForce
	ExpireTime     time.Time
	PostOnly       bool
	ReduceOnly     bool
	Hidden         bool
	MinExecuteSize float64
	AllOrNone      bool
	PegOffset      float64    // For pegged orders
	TrailAmount    float64    // For trailing stops
	Flags          OrderFlags // Bitwise flags for order behavior

	// Fees
	MakerFee float64
	TakerFee float64
	FeesPaid float64
}

// AdvancedOrder - defined in orderbook_advanced.go (has its own structure)

// OrderBook - actual implementation in orderbook.go

// OrderTree and OrderNode - actual implementations in orderbook.go
