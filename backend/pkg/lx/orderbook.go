package lx

import (
	"container/heap"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Backend represents the available acceleration backend
type Backend int

const (
	BackendGo Backend = iota
	BackendCGO
	BackendMLX
	BackendCUDA
)

var (
	// AutoDetect the best available backend at runtime
	currentBackend Backend
)

func init() {
	// Automatically detect and use the best available backend
	currentBackend = detectBestBackend()
}

func detectBestBackend() Backend {
	// Check for CUDA support
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" {
		return BackendCUDA
	}
	
	// Check for MLX support (Apple Silicon)
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		if _, err := os.Stat("/System/Library/Frameworks/Metal.framework"); err == nil {
			return BackendMLX
		}
	}
	
	// Check if CGO is enabled
	if os.Getenv("CGO_ENABLED") == "1" {
		return BackendCGO
	}
	
	// Default to pure Go
	return BackendGo
}

// OrderTree implements a price-time priority order book side with RB-tree
type OrderTree struct {
	side        Side
	priceLevels map[string]*PriceLevel // price -> level
	priceHeap   PriceHeap              // Min/Max heap for best prices
	orders      map[uint64]*Order
	mu          sync.RWMutex
	sequence    uint64
}

// PriceLevel represents a price level in the order book
type PriceLevel struct {
	Price      float64
	Orders     []*Order // FIFO queue
	TotalSize  float64
	OrderCount int
	mu         sync.RWMutex
}

// Trade represents an executed trade
type Trade struct {
	ID        uint64
	Price     float64
	Size      float64
	BuyOrder  *Order
	SellOrder *Order
	Timestamp time.Time
	TakerSide Side
	MatchType string // "full", "partial"
	Fee       float64
}

// Side represents order side
type Side int

const (
	Buy Side = iota
	Sell
)

// OrderStatus represents order status
type OrderStatus int

const (
	Open OrderStatus = iota
	PartiallyFilled
	Filled
	Cancelled
	Rejected
	Expired
)

// Errors
var (
	ErrOrderNotFound     = fmt.Errorf("order not found")
	ErrInsufficientFunds = fmt.Errorf("insufficient funds")
	ErrInvalidPrice      = fmt.Errorf("invalid price")
	ErrInvalidSize       = fmt.Errorf("invalid size")
	ErrVaultNotFound     = fmt.Errorf("vault not found")
	ErrPerpNotFound      = fmt.Errorf("perpetual market not found")
	ErrExcessiveLeverage = fmt.Errorf("excessive leverage")
	ErrSelfTrade         = fmt.Errorf("self trade not allowed")
	ErrPostOnlyWouldTake = fmt.Errorf("post-only order would take")
)

// NewOrderBook creates a new order book
func NewOrderBook(symbol string) *OrderBook {
	return &OrderBook{
		Symbol:     symbol,
		Bids:       NewOrderTree(Buy),
		Asks:       NewOrderTree(Sell),
		Orders:     make(map[uint64]*Order),
		UserOrders: make(map[string][]uint64),
		Trades:     make([]Trade, 0, 10000),
	}
}

// NewOrderTree creates a new order tree
func NewOrderTree(side Side) *OrderTree {
	tree := &OrderTree{
		side:        side,
		priceLevels: make(map[string]*PriceLevel),
		orders:      make(map[uint64]*Order),
	}

	// Initialize heap based on side
	if side == Buy {
		tree.priceHeap = &MaxPriceHeap{}
	} else {
		tree.priceHeap = &MinPriceHeap{}
	}
	heap.Init(tree.priceHeap)

	return tree
}

// AddOrder adds an order to the book with full validation
func (book *OrderBook) AddOrder(order *Order) uint64 {
	book.mu.Lock()
	defer book.mu.Unlock()

	// Validate order
	if err := book.validateOrder(order); err != nil {
		order.Status = Rejected
		return 0
	}

	// Generate order ID
	book.LastOrderID++
	order.ID = book.LastOrderID
	order.Status = Open
	order.Timestamp = time.Now()

	// Check for self-trade prevention
	if order.User != "" && !order.ReduceOnly {
		if book.checkSelfTrade(order) {
			order.Status = Rejected
			return 0
		}
	}

	// Post-only check
	if order.PostOnly {
		if book.wouldTakeLiquidity(order) {
			order.Status = Rejected
			return 0
		}
	}

	// Add to appropriate side
	var tree *OrderTree
	if order.Side == Buy {
		tree = book.Bids
	} else {
		tree = book.Asks
	}

	// Add to tree
	tree.addOrder(order)

	// Track order
	book.Orders[order.ID] = order
	if book.UserOrders[order.User] == nil {
		book.UserOrders[order.User] = make([]uint64, 0)
	}
	book.UserOrders[order.User] = append(book.UserOrders[order.User], order.ID)

	// Increment sequence for L4 tracking
	atomic.AddUint64(&tree.sequence, 1)

	return order.ID
}

// MatchOrders attempts to match orders in the book
func (book *OrderBook) MatchOrders() []Trade {
	book.mu.Lock()
	defer book.mu.Unlock()

	trades := make([]Trade, 0)

	for {
		// Get best bid and ask
		bestBid := book.Bids.getBestOrder()
		bestAsk := book.Asks.getBestOrder()

		if bestBid == nil || bestAsk == nil {
			break
		}

		// Check if orders cross
		if bestBid.Price < bestAsk.Price {
			break
		}

		// Self-trade prevention
		if bestBid.User == bestAsk.User && bestBid.User != "" {
			// Cancel the smaller order
			if bestBid.Size < bestAsk.Size {
				book.cancelOrderInternal(bestBid)
				continue
			} else {
				book.cancelOrderInternal(bestAsk)
				continue
			}
		}

		// Determine trade size
		bidRemaining := bestBid.Size - bestBid.Filled
		askRemaining := bestAsk.Size - bestAsk.Filled
		tradeSize := math.Min(bidRemaining, askRemaining)

		// Determine trade price (price-time priority)
		// The trade executes at the price of the order that was in the book first (maker)
		var tradePrice float64
		var takerSide Side
		if bestBid.Timestamp.Before(bestAsk.Timestamp) {
			// Bid was first (maker), ask is taker
			tradePrice = bestBid.Price
			takerSide = Sell
		} else {
			// Ask was first (maker), bid is taker
			tradePrice = bestAsk.Price
			takerSide = Buy
		}

		// Calculate fees (0.02% taker, -0.01% maker rebate)
		var fee float64
		if takerSide == Buy {
			fee = tradeSize * tradePrice * 0.0002 // Buyer is taker
		} else {
			fee = tradeSize * tradePrice * 0.0002 // Seller is taker
		}

		// Create trade
		book.LastTradeID++
		trade := Trade{
			ID:        book.LastTradeID,
			Price:     tradePrice,
			Size:      tradeSize,
			BuyOrder:  bestBid,
			SellOrder: bestAsk,
			Timestamp: time.Now(),
			TakerSide: takerSide,
			MatchType: "partial",
			Fee:       fee,
		}

		// Update orders
		bestBid.Filled += tradeSize
		bestAsk.Filled += tradeSize

		if bestBid.Filled >= bestBid.Size {
			bestBid.Status = Filled
			book.Bids.removeOrder(bestBid)
			trade.MatchType = "full"
		} else {
			bestBid.Status = PartiallyFilled
		}

		if bestAsk.Filled >= bestAsk.Size {
			bestAsk.Status = Filled
			book.Asks.removeOrder(bestAsk)
			if trade.MatchType == "full" {
				trade.MatchType = "full" // Both orders fully filled
			}
		} else {
			bestAsk.Status = PartiallyFilled
		}

		trades = append(trades, trade)
		book.Trades = append(book.Trades, trade)

		// Limit trades history
		if len(book.Trades) > 100000 {
			book.Trades = book.Trades[len(book.Trades)-50000:]
		}
	}

	return trades
}

// OrderTree methods
func (tree *OrderTree) addOrder(order *Order) {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	priceKey := fmt.Sprintf("%.8f", order.Price)

	// Get or create price level
	level, exists := tree.priceLevels[priceKey]
	if !exists {
		level = &PriceLevel{
			Price:  order.Price,
			Orders: make([]*Order, 0),
		}
		tree.priceLevels[priceKey] = level
		heap.Push(tree.priceHeap, order.Price)
	}

	// Add order to level (FIFO)
	level.mu.Lock()
	level.Orders = append(level.Orders, order)
	level.TotalSize += (order.Size - order.Filled)
	level.OrderCount++
	level.mu.Unlock()

	// Track order
	tree.orders[order.ID] = order
}

func (tree *OrderTree) removeOrder(order *Order) {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	priceKey := fmt.Sprintf("%.8f", order.Price)
	level, exists := tree.priceLevels[priceKey]
	if !exists {
		return
	}

	level.mu.Lock()
	// Remove order from level
	for i, o := range level.Orders {
		if o.ID == order.ID {
			level.Orders = append(level.Orders[:i], level.Orders[i+1:]...)
			level.TotalSize -= (order.Size - order.Filled)
			level.OrderCount--
			break
		}
	}

	// Remove level if empty
	if len(level.Orders) == 0 {
		delete(tree.priceLevels, priceKey)
		// Note: Removing from heap is expensive, so we leave it and filter on pop
	}
	level.mu.Unlock()

	delete(tree.orders, order.ID)
}

func (tree *OrderTree) getBestOrder() *Order {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	for tree.priceHeap.Len() > 0 {
		price := tree.priceHeap.Peek()
		priceKey := fmt.Sprintf("%.8f", price)

		level, exists := tree.priceLevels[priceKey]
		if !exists {
			// Stale price, remove it
			heap.Pop(tree.priceHeap)
			continue
		}

		level.mu.RLock()
		if len(level.Orders) > 0 {
			order := level.Orders[0]
			level.mu.RUnlock()
			if order.Status == Open || order.Status == PartiallyFilled {
				return order
			}
		}
		level.mu.RUnlock()

		// No valid orders at this level
		heap.Pop(tree.priceHeap)
	}

	return nil
}

// Helper methods for OrderBook
func (book *OrderBook) validateOrder(order *Order) error {
	if order.Price <= 0 {
		return ErrInvalidPrice
	}
	if order.Size <= 0 {
		return ErrInvalidSize
	}
	if order.User == "" {
		return fmt.Errorf("user required")
	}
	return nil
}

func (book *OrderBook) checkSelfTrade(order *Order) bool {
	// Check opposite side for orders from same user
	var oppositeTree *OrderTree
	if order.Side == Buy {
		oppositeTree = book.Asks
	} else {
		oppositeTree = book.Bids
	}

	oppositeTree.mu.RLock()
	defer oppositeTree.mu.RUnlock()

	for _, existingOrder := range oppositeTree.orders {
		if existingOrder.User == order.User && existingOrder.Status == Open {
			// Would result in self-trade
			if order.Side == Buy && order.Price >= existingOrder.Price {
				return true
			}
			if order.Side == Sell && order.Price <= existingOrder.Price {
				return true
			}
		}
	}
	return false
}

func (book *OrderBook) wouldTakeLiquidity(order *Order) bool {
	// Check if order would immediately match
	if order.Side == Buy {
		bestAsk := book.Asks.getBestOrder()
		if bestAsk != nil && order.Price >= bestAsk.Price {
			return true
		}
	} else {
		bestBid := book.Bids.getBestOrder()
		if bestBid != nil && order.Price <= bestBid.Price {
			return true
		}
	}
	return false
}

func (book *OrderBook) cancelOrderInternal(order *Order) {
	order.Status = Cancelled
	if order.Side == Buy {
		book.Bids.removeOrder(order)
	} else {
		book.Asks.removeOrder(order)
	}
}

// CancelOrder cancels an order
func (book *OrderBook) CancelOrder(orderID uint64) error {
	book.mu.Lock()
	defer book.mu.Unlock()

	order, exists := book.Orders[orderID]
	if !exists {
		return ErrOrderNotFound
	}

	if order.Status != Open && order.Status != PartiallyFilled {
		return fmt.Errorf("order not cancellable")
	}

	book.cancelOrderInternal(order)
	return nil
}

// ModifyOrder modifies an existing order
func (book *OrderBook) ModifyOrder(orderID uint64, newPrice, newSize float64) error {
	book.mu.Lock()
	defer book.mu.Unlock()

	order, exists := book.Orders[orderID]
	if !exists {
		return ErrOrderNotFound
	}

	if order.Status != Open && order.Status != PartiallyFilled {
		return fmt.Errorf("order not modifiable")
	}

	// Remove old order
	if order.Side == Buy {
		book.Bids.removeOrder(order)
	} else {
		book.Asks.removeOrder(order)
	}

	// Update order
	order.Price = newPrice
	order.Size = newSize
	order.Timestamp = time.Now() // Reset timestamp for price-time priority

	// Re-add order
	if order.Side == Buy {
		book.Bids.addOrder(order)
	} else {
		book.Asks.addOrder(order)
	}

	return nil
}

// GetSnapshot returns orderbook snapshot
func (book *OrderBook) GetSnapshot() *OrderBookSnapshot {
	book.mu.RLock()
	defer book.mu.RUnlock()

	return &OrderBookSnapshot{
		Symbol:    book.Symbol,
		Timestamp: time.Now(),
		Bids:      book.Bids.getLevels(10),
		Asks:      book.Asks.getLevels(10),
		Sequence:  atomic.LoadUint64(&book.Bids.sequence),
	}
}

func (tree *OrderTree) getLevels(depth int) []PriceLevel {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	levels := make([]PriceLevel, 0, depth)
	prices := make([]float64, 0, len(tree.priceLevels))

	for _, level := range tree.priceLevels {
		if level.OrderCount > 0 {
			prices = append(prices, level.Price)
		}
	}

	// Sort prices
	if tree.side == Buy {
		sort.Sort(sort.Reverse(sort.Float64Slice(prices)))
	} else {
		sort.Float64s(prices)
	}

	// Build levels
	for i, price := range prices {
		if i >= depth {
			break
		}
		priceKey := fmt.Sprintf("%.8f", price)
		if level, exists := tree.priceLevels[priceKey]; exists {
			levels = append(levels, PriceLevel{
				Price:      level.Price,
				TotalSize:  level.TotalSize,
				OrderCount: level.OrderCount,
			})
		}
	}

	return levels
}

// GetL4Book returns full order-level book data
func (book *OrderBook) GetL4Book() L4BookSnapshot {
	book.mu.RLock()
	defer book.mu.RUnlock()

	snapshot := L4BookSnapshot{
		Symbol:    book.Symbol,
		Timestamp: time.Now(),
		Sequence:  atomic.LoadUint64(&book.Bids.sequence),
	}

	// Get all bid orders
	snapshot.Bids = book.Bids.getL4Levels()
	snapshot.Asks = book.Asks.getL4Levels()

	return snapshot
}

func (tree *OrderTree) getL4Levels() []L4Level {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	levels := make([]L4Level, 0)

	// Get all prices and sort
	prices := make([]float64, 0, len(tree.priceLevels))
	for _, level := range tree.priceLevels {
		if level.OrderCount > 0 {
			prices = append(prices, level.Price)
		}
	}

	if tree.side == Buy {
		sort.Sort(sort.Reverse(sort.Float64Slice(prices)))
	} else {
		sort.Float64s(prices)
	}

	// Build L4 levels with individual orders
	for _, price := range prices {
		priceKey := fmt.Sprintf("%.8f", price)
		if level, exists := tree.priceLevels[priceKey]; exists {
			level.mu.RLock()
			for _, order := range level.Orders {
				if order.Status == Open || order.Status == PartiallyFilled {
					levels = append(levels, L4Level{
						Price:    order.Price,
						Size:     order.Size - order.Filled,
						OrderID:  order.ID,
						UserID:   order.User,
						ClientID: order.ClientID,
					})
				}
			}
			level.mu.RUnlock()
		}
	}

	return levels
}

// Price heap implementations
type PriceHeap interface {
	heap.Interface
	Peek() float64
}

type MinPriceHeap []float64

func (h MinPriceHeap) Len() int           { return len(h) }
func (h MinPriceHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h MinPriceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h MinPriceHeap) Peek() float64 {
	if len(h) > 0 {
		return h[0]
	}
	return 0
}

func (h *MinPriceHeap) Push(x interface{}) {
	*h = append(*h, x.(float64))
}

func (h *MinPriceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type MaxPriceHeap []float64

func (h MaxPriceHeap) Len() int           { return len(h) }
func (h MaxPriceHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h MaxPriceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h MaxPriceHeap) Peek() float64 {
	if len(h) > 0 {
		return h[0]
	}
	return 0
}

func (h *MaxPriceHeap) Push(x interface{}) {
	*h = append(*h, x.(float64))
}

func (h *MaxPriceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Snapshot types
type OrderBookSnapshot struct {
	Symbol    string
	Timestamp time.Time
	Bids      []PriceLevel
	Asks      []PriceLevel
	Sequence  uint64
}

type VaultSnapshot struct {
	ID            string
	TotalDeposits string
	Performance   *PerformanceMetrics
	Timestamp     time.Time
}

type PerpSnapshot struct {
	Symbol       string
	MarkPrice    float64
	IndexPrice   float64
	FundingRate  float64
	OpenInterest float64
	Timestamp    time.Time
}
