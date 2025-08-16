package engine

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
	
	"github.com/luxexchange/engine/backend/pkg/orderbook"
	pb "github.com/luxexchange/engine/backend/pkg/proto/engine"
)

// LXEngine is the high-performance matching engine for LX
// It combines the best features from go-trader and cpp-trader:
// - Fixed-point arithmetic for precision (from go-trader)
// - Lock-free atomic operations where possible
// - Hybrid Go/C++ orderbook via CGO (optional)
// - Zero-copy networking with buffer pools
type LXEngine struct {
	pb.UnimplementedEngineServiceServer
	// Core components
	orderBooks   sync.Map // map[string]*orderbook.OrderBook
	sessions     sync.Map // map[string]*Session
	instruments  sync.Map // map[string]*Instrument
	
	// Performance counters
	ordersProcessed  uint64
	tradesExecuted   uint64
	messagesReceived uint64
	
	// Configuration
	config Config
	
	// Buffer pools for zero-copy
	orderPool  sync.Pool
	tradePool  sync.Pool
	bufferPool sync.Pool
}

// getImplementation returns the orderbook implementation based on mode
func getImplementation(mode string) orderbook.Implementation {
	switch mode {
	case "cpp":
		return orderbook.ImplCpp
	case "hybrid":
		// Hybrid mode will use CGO at build time
		return orderbook.ImplCpp
	default:
		return orderbook.ImplGo
	}
}

// Config holds engine configuration
type Config struct {
	Mode      string           // "go", "hybrid", or "cpp"
	OrderBook orderbook.Config // OrderBook configuration
	
	// Performance tuning
	MaxOrdersPerBook     int
	MaxSessionsPerEngine int
	BufferPoolSize       int
	
	// Features
	EnableMarketData bool
	EnableFIXGateway bool
	EnableWebSocket  bool
	
	// Network settings
	TCPNoDelay bool
	SocketBufferSize int
}

// Instrument represents a tradable instrument
type Instrument struct {
	Symbol       string
	Exchange     string
	TickSize     float64
	LotSize      float64
	MaxOrderSize float64
	MinOrderSize float64
}

// Session represents a trading session
type Session struct {
	mu          sync.RWMutex
	ID          string
	UserID      uint64
	Orders      map[uint64]*orderbook.Order
	LastActivity time.Time
	
	// Performance metrics
	OrdersSent   uint64
	TradesReceived uint64
	
	// Callbacks
	OnOrderStatus func(*orderbook.Order)
	OnTrade       func(*orderbook.Trade)
}

// NewLXEngine creates a new high-performance matching engine
func NewLXEngine(config Config) *LXEngine {
	engine := &LXEngine{
		config: config,
	}
	
	// Initialize buffer pools for zero-copy operations
	engine.orderPool = sync.Pool{
		New: func() interface{} {
			return &orderbook.Order{}
		},
	}
	
	engine.tradePool = sync.Pool{
		New: func() interface{} {
			return &orderbook.Trade{}
		},
	}
	
	engine.bufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, config.SocketBufferSize)
		},
	}
	
	return engine
}

// CreateOrder processes a new order
func (e *LXEngine) CreateOrder(sessionID string, order *orderbook.Order) (uint64, error) {
	// Atomic increment for performance metrics
	atomic.AddUint64(&e.ordersProcessed, 1)
	
	// Get or create orderbook for instrument
	ob := e.getOrCreateOrderBook(order.Symbol)
	
	// Get session
	session := e.getSession(sessionID)
	if session == nil {
		return 0, ErrSessionNotFound
	}
	
	// Add order to book
	orderID := ob.AddOrder(order)
	
	// Store order in session
	session.mu.Lock()
	session.Orders[orderID] = order
	session.LastActivity = time.Now()
	atomic.AddUint64(&session.OrdersSent, 1)
	session.mu.Unlock()
	
	// Execute matching
	trades := ob.MatchOrders()
	
	// Process trades
	for _, trade := range trades {
		atomic.AddUint64(&e.tradesExecuted, 1)
		e.processTrade(trade)
	}
	
	// Send order status
	if session.OnOrderStatus != nil {
		session.OnOrderStatus(order)
	}
	
	return orderID, nil
}

// CancelOrder cancels an existing order
func (e *LXEngine) CancelOrderByID(sessionID string, orderID uint64) error {
	session := e.getSession(sessionID)
	if session == nil {
		return ErrSessionNotFound
	}
	
	session.mu.RLock()
	order, exists := session.Orders[orderID]
	session.mu.RUnlock()
	
	if !exists {
		return ErrOrderNotFound
	}
	
	ob := e.getOrderBook(order.Symbol)
	if ob == nil {
		return ErrInstrumentNotFound
	}
	
	if ob.CancelOrder(orderID) {
		order.Status = orderbook.Cancelled
		
		session.mu.Lock()
		delete(session.Orders, orderID)
		session.mu.Unlock()
		
		if session.OnOrderStatus != nil {
			session.OnOrderStatus(order)
		}
	}
	
	return nil
}

// ModifyOrder modifies an existing order
func (e *LXEngine) ModifyOrder(sessionID string, orderID uint64, newPrice, newQuantity float64) error {
	session := e.getSession(sessionID)
	if session == nil {
		return ErrSessionNotFound
	}
	
	session.mu.RLock()
	order, exists := session.Orders[orderID]
	session.mu.RUnlock()
	
	if !exists {
		return ErrOrderNotFound
	}
	
	ob := e.getOrderBook(order.Symbol)
	if ob == nil {
		return ErrInstrumentNotFound
	}
	
	if ob.ModifyOrder(orderID, newPrice, newQuantity) {
		order.Price = newPrice
		order.Quantity = newQuantity
		
		if session.OnOrderStatus != nil {
			session.OnOrderStatus(order)
		}
		
		// Re-run matching after modification
		trades := ob.MatchOrders()
		for _, trade := range trades {
			atomic.AddUint64(&e.tradesExecuted, 1)
			e.processTrade(trade)
		}
	}
	
	return nil
}

// getOrCreateOrderBook gets or creates an orderbook for an instrument
func (e *LXEngine) getOrCreateOrderBook(symbol string) orderbook.OrderBook {
	if ob, ok := e.orderBooks.Load(symbol); ok {
		return ob.(orderbook.OrderBook)
	}
	
	// Create new orderbook with configured implementation
	ob := orderbook.NewOrderBook(orderbook.Config{
		Implementation: getImplementation(e.config.Mode),
		Symbol:         symbol,
		MaxOrdersPerLevel: e.config.OrderBook.MaxOrdersPerLevel,
		PricePrecision: e.config.OrderBook.PricePrecision,
	})
	
	actual, _ := e.orderBooks.LoadOrStore(symbol, ob)
	return actual.(orderbook.OrderBook)
}

// getOrderBook retrieves an existing orderbook
func (e *LXEngine) getOrderBook(symbol string) orderbook.OrderBook {
	if ob, ok := e.orderBooks.Load(symbol); ok {
		return ob.(orderbook.OrderBook)
	}
	return nil
}

// getSession retrieves a session by ID
func (e *LXEngine) getSession(sessionID string) *Session {
	if s, ok := e.sessions.Load(sessionID); ok {
		return s.(*Session)
	}
	return nil
}

// RegisterSession registers a new trading session
func (e *LXEngine) RegisterSession(session *Session) {
	e.sessions.Store(session.ID, session)
}

// UnregisterSession removes a trading session
func (e *LXEngine) UnregisterSession(sessionID string) {
	if session := e.getSession(sessionID); session != nil {
		// Cancel all open orders
		session.mu.RLock()
		orders := make([]*orderbook.Order, 0, len(session.Orders))
		for _, order := range session.Orders {
			orders = append(orders, order)
		}
		session.mu.RUnlock()
		
		for _, order := range orders {
			e.CancelOrderByID(sessionID, order.ID)
		}
		
		e.sessions.Delete(sessionID)
	}
}

// processTrade handles trade execution
func (e *LXEngine) processTrade(trade orderbook.Trade) {
	// Find sessions for both sides of the trade
	e.sessions.Range(func(key, value interface{}) bool {
		session := value.(*Session)
		session.mu.RLock()
		_, hasBuy := session.Orders[trade.BuyOrderID]
		_, hasSell := session.Orders[trade.SellOrderID]
		session.mu.RUnlock()
		
		if hasBuy || hasSell {
			atomic.AddUint64(&session.TradesReceived, 1)
			if session.OnTrade != nil {
				session.OnTrade(&trade)
			}
		}
		
		return true
	})
}

// GetMarketData returns current market data for an instrument
func (e *LXEngine) GetMarketData(symbol string) *MarketData {
	ob := e.getOrderBook(symbol)
	if ob == nil {
		return nil
	}
	
	return &MarketData{
		Symbol:   symbol,
		BestBid:  ob.GetBestBid(),
		BestAsk:  ob.GetBestAsk(),
		Volume:   ob.GetVolume(),
		Depth:    ob.GetDepth(10),
		UpdateTime: time.Now(),
	}
}

// GetStats returns engine statistics
func (e *LXEngine) GetStats() *EngineStats {
	return &EngineStats{
		OrdersProcessed:  atomic.LoadUint64(&e.ordersProcessed),
		TradesExecuted:   atomic.LoadUint64(&e.tradesExecuted),
		MessagesReceived: atomic.LoadUint64(&e.messagesReceived),
		ActiveOrderBooks: e.countOrderBooks(),
		ActiveSessions:   e.countSessions(),
	}
}

// New creates a new LX engine
func New(config Config) (*LXEngine, error) {
	e := &LXEngine{
		config: config,
	}
	
	// Initialize buffer pools
	e.orderPool = sync.Pool{
		New: func() interface{} {
			return &orderbook.Order{}
		},
	}
	
	e.tradePool = sync.Pool{
		New: func() interface{} {
			return &orderbook.Trade{}
		},
	}
	
	e.bufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 4096)
		},
	}
	
	return e, nil
}

// Implement gRPC EngineService interface
func (e *LXEngine) SubmitOrder(ctx context.Context, req *pb.SubmitOrderRequest) (*pb.SubmitOrderResponse, error) {
	atomic.AddUint64(&e.ordersProcessed, 1)
	
	// TODO: Implement order submission
	return &pb.SubmitOrderResponse{
		OrderId: "order-" + req.Symbol,
		Status:  pb.OrderStatus_ORDER_STATUS_NEW,
	}, nil
}

func (e *LXEngine) CancelOrder(ctx context.Context, req *pb.CancelOrderRequest) (*pb.CancelOrderResponse, error) {
	// TODO: Implement order cancellation
	return &pb.CancelOrderResponse{
		Success: true,
	}, nil
}

func (e *LXEngine) GetOrderBook(ctx context.Context, req *pb.GetOrderBookRequest) (*pb.GetOrderBookResponse, error) {
	// TODO: Implement orderbook retrieval
	return &pb.GetOrderBookResponse{
		Symbol: req.Symbol,
		Bids:   []*pb.PriceLevel{},
		Asks:   []*pb.PriceLevel{},
	}, nil
}

func (e *LXEngine) StreamOrderBook(req *pb.StreamOrderBookRequest, stream pb.EngineService_StreamOrderBookServer) error {
	// TODO: Implement orderbook streaming
	return nil
}

// countOrderBooks returns the number of active orderbooks
func (e *LXEngine) countOrderBooks() int {
	count := 0
	e.orderBooks.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

// countSessions returns the number of active sessions
func (e *LXEngine) countSessions() int {
	count := 0
	e.sessions.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

// Shutdown gracefully shuts down the engine
func (e *LXEngine) Shutdown() {
	// Cancel all open orders
	e.sessions.Range(func(key, value interface{}) bool {
		sessionID := key.(string)
		e.UnregisterSession(sessionID)
		return true
	})
	
	// Clean up C++ resources if using CGO
	if e.config.Mode == "cpp" || e.config.Mode == "hybrid" {
		e.orderBooks.Range(func(key, value interface{}) bool {
			// Cleanup will be handled by orderbook implementation
			return true
		})
	}
}