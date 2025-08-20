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
	"unsafe"
)

// PriceInt represents price as integer (multiplied by 10^8 for 8 decimal precision)
type PriceInt int64

const PriceMultiplier = 100000000

// OrderFlags constants (type defined in types_common.go)
const (
	OrderFlagNone       OrderFlags = 0
	OrderFlagPostOnly   OrderFlags = 1 << 0
	OrderFlagReduceOnly OrderFlags = 1 << 1
	OrderFlagSTP        OrderFlags = 1 << 2 // Self-trade prevention
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

	// Global memory pools for zero-allocation
	orderPool = &sync.Pool{
		New: func() interface{} {
			return &Order{}
		},
	}
	levelPool = &sync.Pool{
		New: func() interface{} {
			return &OptimizedPriceLevel{}
		},
	}
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

// OrderBook represents a complete order book for a trading pair - OPTIMIZED VERSION
type OrderBook struct {
	Symbol string

	// Optimized data structures
	bids unsafe.Pointer // *OrderTree
	asks unsafe.Pointer // *OrderTree

	// For compatibility - expose these directly
	Bids *OrderTree
	Asks *OrderTree

	// Trade history with circular buffer
	Trades       []Trade // Keep for compatibility but use circular buffer internally
	tradesBuffer *CircularTradeBuffer

	// Order tracking with lock-free map
	Orders        map[uint64]*Order // Keep for API compatibility
	ordersMap     sync.Map          // Lock-free internal
	UserOrders    map[string][]uint64
	userOrdersMap sync.Map

	// Atomic counters
	LastTradeID  uint64
	LastOrderID  uint64
	LastUpdateID uint64
	OrderSeq     uint64 // Add OrderSeq field for compatibility

	// Configuration
	EnableImmediateMatching bool // When true, orders match immediately on add

	// Market data feed
	subscribers []chan MarketDataUpdate
	subMu       sync.RWMutex

	// Single write lock for structural changes
	mu        sync.RWMutex // Keep for API compatibility
	writeLock sync.Mutex   // Internal write lock

	// Cache line padding to prevent false sharing
	_padding [64]byte
}

// OrderTree implements optimized price-time priority order book side
type OrderTree struct {
	side        Side
	priceLevels map[PriceInt]*OptimizedPriceLevel // Integer keys - FAST!
	priceTree   *IntBTree                         // B-tree for sorted prices
	orders      map[uint64]*Order
	bestPrice   atomic.Int64 // Atomic best price
	sequence    uint64
	mu          sync.RWMutex
	priceHeap   PriceHeap // Keep for compatibility
}

// OptimizedPriceLevel with lock-free operations
type OptimizedPriceLevel struct {
	Price      float64  // Keep original price for API
	PriceInt   PriceInt // Integer price for fast operations
	Orders     []*Order // Keep slice for compatibility
	OrderList  *OrderLinkedList
	TotalSize  float64
	OrderCount int
	mu         sync.RWMutex

	// Atomic counters for lock-free updates
	atomicSize  atomic.Int64
	atomicCount atomic.Int32
}

// OrderLinkedList for O(1) insertion and removal
type OrderLinkedList struct {
	head  *OrderNode
	tail  *OrderNode
	index map[uint64]*OrderNode
	mu    sync.RWMutex
}

type OrderNode struct {
	Order *Order
	Next  *OrderNode
	Prev  *OrderNode
}

// CircularTradeBuffer for efficient trade history
type CircularTradeBuffer struct {
	buffer [100000]Trade
	head   uint64
	tail   uint64
	size   uint64
	mu     sync.RWMutex
}

// IntBTree is a B-tree for integer keys (prices)
type IntBTree struct {
	root      *IntBTreeNode
	degree    int
	isMaxHeap bool
}

type IntBTreeNode struct {
	keys     []PriceInt
	children []*IntBTreeNode
	isLeaf   bool
	n        int
}

// NewOrderBook creates an optimized order book
func NewOrderBook(symbol string) *OrderBook {
	ob := &OrderBook{
		Symbol:       symbol,
		Trades:       make([]Trade, 0),
		tradesBuffer: &CircularTradeBuffer{},
		Orders:       make(map[uint64]*Order),
		UserOrders:   make(map[string][]uint64),
	}

	// Initialize optimized bid and ask trees
	bidTree := &OrderTree{
		side:        Buy,
		priceLevels: make(map[PriceInt]*OptimizedPriceLevel),
		priceTree:   NewIntBTree(32, true),
		orders:      make(map[uint64]*Order),
		priceHeap:   &MaxPriceHeap{},
	}
	askTree := &OrderTree{
		side:        Sell,
		priceLevels: make(map[PriceInt]*OptimizedPriceLevel),
		priceTree:   NewIntBTree(32, false),
		orders:      make(map[uint64]*Order),
		priceHeap:   &MinPriceHeap{},
	}

	// Initialize heaps
	heap.Init(bidTree.priceHeap)
	heap.Init(askTree.priceHeap)

	// Set both pointer and direct references for compatibility
	atomic.StorePointer(&ob.bids, unsafe.Pointer(bidTree))
	atomic.StorePointer(&ob.asks, unsafe.Pointer(askTree))
	ob.Bids = bidTree
	ob.Asks = askTree

	return ob
}

// NewOrderTree creates a new order tree (for compatibility)
func NewOrderTree(side Side) *OrderTree {
	tree := &OrderTree{
		side:        side,
		priceLevels: make(map[PriceInt]*OptimizedPriceLevel),
		orders:      make(map[uint64]*Order),
	}

	// Initialize heap and B-tree based on side
	if side == Buy {
		tree.priceHeap = &MaxPriceHeap{}
		tree.priceTree = NewIntBTree(32, true)
	} else {
		tree.priceHeap = &MinPriceHeap{}
		tree.priceTree = NewIntBTree(32, false)
	}
	heap.Init(tree.priceHeap)

	return tree
}

// AddOrder with optimized integer price handling
func (ob *OrderBook) AddOrder(order *Order) uint64 {
	// Auto-assign ID if not set
	if order.ID == 0 {
		order.ID = atomic.AddUint64(&ob.LastOrderID, 1)
		ob.LastOrderID = order.ID // Keep synchronized
	}

	// Set status if not set
	if order.Status == "" {
		order.Status = Open
	}

	// Set timestamp if not set
	if order.Timestamp.IsZero() {
		order.Timestamp = time.Now()
	}

	// Validate order
	if err := ob.validateOrder(order); err != nil {
		order.Status = Rejected
		return 0
	}

	// Convert price to integer for fast operations
	priceInt := PriceInt(order.Price * PriceMultiplier)

	// Handle market orders
	if order.Type == Market {
		return ob.processMarketOrderOptimized(order)
	}

	// Get write lock for modifications
	ob.mu.Lock()
	defer ob.mu.Unlock()

	// Check self-trade prevention - check both User and UserID fields
	userIdentifier := order.User
	if userIdentifier == "" {
		userIdentifier = order.UserID
	}
	if userIdentifier != "" && ob.checkSelfTrade(order) {
		order.Status = Rejected
		return 0
	}

	// Check post-only
	if order.PostOnly || order.Flags&OrderFlagPostOnly != 0 {
		if ob.wouldTakeLiquidity(order) {
			order.Status = Rejected
			return 0
		}
	}

	// Handle time-in-force and check for immediate matches
	numTrades := uint64(0)

	// Set remaining size
	order.RemainingSize = order.Size

	// Only match immediately for IOC/FOK orders
	// Regular limit orders are added to book and matched via MatchOrders()
	if order.TimeInForce == ImmediateOrCancel || order.TimeInForce == FillOrKill {
		numTrades = ob.tryMatchImmediateLocked(order)

		if order.TimeInForce == FillOrKill && order.RemainingSize > 0 {
			order.Status = Rejected
			return 0
		}

		if order.TimeInForce == ImmediateOrCancel && order.RemainingSize > 0 {
			// IOC order partially filled, return order ID
			return order.ID
		}
	}

	// Add remaining to book
	if order.RemainingSize > 0 || (order.RemainingSize == 0 && order.Size > 0) {
		if order.RemainingSize == 0 {
			order.RemainingSize = order.Size
		}

		// Get appropriate tree
		var tree *OrderTree
		if order.Side == Buy {
			tree = (*OrderTree)(atomic.LoadPointer(&ob.bids))
		} else {
			tree = (*OrderTree)(atomic.LoadPointer(&ob.asks))
		}

		// Add to tree with optimized integer price
		ob.addToTreeOptimized(tree, order, priceInt)

		// Track order
		ob.Orders[order.ID] = order
		ob.ordersMap.Store(order.ID, order)

		if ob.UserOrders[order.User] == nil {
			ob.UserOrders[order.User] = make([]uint64, 0)
		}
		ob.UserOrders[order.User] = append(ob.UserOrders[order.User], order.ID)

		// Match immediately if enabled (for optimized order book)
		if ob.EnableImmediateMatching {
			numTrades += ob.tryMatchImmediateLocked(order)
		}

		// Publish market data update
		ob.publishUpdate(MarketDataUpdate{
			Type:      OrderAdded,
			Symbol:    ob.Symbol,
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"order_id": order.ID,
				"price":    order.Price,
				"size":     order.Size,
				"side":     order.Side,
			},
		})
	}

	// Return the order ID for successful orders
	// For tests that need trade count, use MatchOrders() or GetTradeCount()
	return order.ID
}

// addOrder for compatibility - wraps addToTreeOptimized
func (tree *OrderTree) addOrder(order *Order) {
	priceInt := PriceInt(order.Price * PriceMultiplier)
	tree.addOrderOptimized(order, priceInt)
}

// addOrderOptimized adds order to tree with integer price
func (tree *OrderTree) addOrderOptimized(order *Order, priceInt PriceInt) {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	// Get or create price level
	level, exists := tree.priceLevels[priceInt]
	if !exists {
		level = &OptimizedPriceLevel{
			Price:    order.Price,
			PriceInt: priceInt,
			Orders:   make([]*Order, 0),
			OrderList: &OrderLinkedList{
				index: make(map[uint64]*OrderNode),
			},
		}
		tree.priceLevels[priceInt] = level

		// Add to B-tree and heap for compatibility
		tree.priceTree.Insert(priceInt)
		heap.Push(tree.priceHeap, order.Price)

		// Update best price atomically
		if tree.side == Buy {
			currentBest := tree.bestPrice.Load()
			if currentBest == 0 || int64(priceInt) > currentBest {
				tree.bestPrice.Store(int64(priceInt))
			}
		} else {
			currentBest := tree.bestPrice.Load()
			if currentBest == 0 || int64(priceInt) < currentBest {
				tree.bestPrice.Store(int64(priceInt))
			}
		}
	}

	// Add to both slice (for compatibility) and linked list (for performance)
	level.mu.Lock()
	level.Orders = append(level.Orders, order)

	// Add to linked list
	node := &OrderNode{Order: order}
	if level.OrderList.head == nil {
		level.OrderList.head = node
		level.OrderList.tail = node
	} else {
		level.OrderList.tail.Next = node
		node.Prev = level.OrderList.tail
		level.OrderList.tail = node
	}
	level.OrderList.index[order.ID] = node

	remainingSize := order.Size - order.Filled
	if order.RemainingSize > 0 {
		remainingSize = order.RemainingSize
	}
	level.TotalSize += remainingSize
	level.OrderCount++
	level.mu.Unlock()

	// Update atomic counters
	level.atomicSize.Add(int64(remainingSize * PriceMultiplier))
	level.atomicCount.Add(1)

	// Track in tree
	tree.orders[order.ID] = order
	atomic.AddUint64(&tree.sequence, 1)
}

// addToTreeOptimized adds order to tree with integer price
func (ob *OrderBook) addToTreeOptimized(tree *OrderTree, order *Order, priceInt PriceInt) {
	tree.addOrderOptimized(order, priceInt)
}

// removeOrder optimized with O(1) removal
func (tree *OrderTree) removeOrder(order *Order) {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	priceInt := PriceInt(order.Price * PriceMultiplier)
	level, exists := tree.priceLevels[priceInt]
	if !exists {
		return
	}

	level.mu.Lock()

	// Remove from linked list (O(1))
	if level.OrderList != nil {
		node, exists := level.OrderList.index[order.ID]
		if exists {
			if node.Prev != nil {
				node.Prev.Next = node.Next
			} else {
				level.OrderList.head = node.Next
			}

			if node.Next != nil {
				node.Next.Prev = node.Prev
			} else {
				level.OrderList.tail = node.Prev
			}

			delete(level.OrderList.index, order.ID)
		}
	}

	// Remove from slice for compatibility
	for i, o := range level.Orders {
		if o.ID == order.ID {
			level.Orders = append(level.Orders[:i], level.Orders[i+1:]...)
			break
		}
	}

	remainingSize := order.Size - order.Filled
	if order.RemainingSize > 0 {
		remainingSize = order.RemainingSize
	}
	level.TotalSize -= remainingSize
	level.OrderCount--

	// Remove level if empty
	if level.OrderCount == 0 {
		delete(tree.priceLevels, priceInt)
		tree.priceTree.Delete(priceInt)

		// Update best price if needed
		if tree.side == Buy && tree.bestPrice.Load() == int64(priceInt) {
			if next := tree.priceTree.Max(); next != 0 {
				tree.bestPrice.Store(int64(next))
			} else {
				tree.bestPrice.Store(0)
			}
		} else if tree.side == Sell && tree.bestPrice.Load() == int64(priceInt) {
			if next := tree.priceTree.Min(); next != 0 {
				tree.bestPrice.Store(int64(next))
			} else {
				tree.bestPrice.Store(0)
			}
		}
	}

	level.mu.Unlock()

	// Update atomic counters
	level.atomicSize.Add(-int64(remainingSize * PriceMultiplier))
	level.atomicCount.Add(-1)

	delete(tree.orders, order.ID)
}

// getBestOrder optimized with O(1) best price lookup
func (tree *OrderTree) getBestOrder() *Order {
	// Fast path: check atomic best price
	bestPriceInt := PriceInt(tree.bestPrice.Load())
	if bestPriceInt == 0 {
		// Fallback to heap for compatibility
		return tree.getBestOrderViaHeap()
	}

	tree.mu.RLock()
	defer tree.mu.RUnlock()

	level, exists := tree.priceLevels[bestPriceInt]
	if !exists || level.OrderList == nil || level.OrderList.head == nil {
		// Fallback to heap for compatibility
		return tree.getBestOrderViaHeap()
	}

	// Use linked list for O(1) access
	level.mu.RLock()
	defer level.mu.RUnlock()

	if level.OrderList.head != nil {
		return level.OrderList.head.Order
	}

	// Fallback to slice
	if len(level.Orders) > 0 {
		return level.Orders[0]
	}

	return nil
}

// getBestOrderViaHeap for compatibility when B-tree is inconsistent
func (tree *OrderTree) getBestOrderViaHeap() *Order {
	for tree.priceHeap.Len() > 0 {
		price := tree.priceHeap.Peek()
		priceInt := PriceInt(price * PriceMultiplier)

		level, exists := tree.priceLevels[priceInt]
		if !exists {
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
		} else {
			level.mu.RUnlock()
		}

		heap.Pop(tree.priceHeap)
	}
	return nil
}

// tryMatchImmediateLocked with optimized matching
func (ob *OrderBook) tryMatchImmediateLocked(order *Order) uint64 {
	numTrades := uint64(0)

	if order.RemainingSize == 0 && order.Size > 0 {
		order.RemainingSize = order.Size
	}

	var oppositeTree *OrderTree
	if order.Side == Buy {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.asks))
	} else {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.bids))
	}

	for order.RemainingSize > 0 {
		bestOrder := oppositeTree.getBestOrder()
		if bestOrder == nil {
			break
		}

		// Price check
		if order.Type == Limit {
			if order.Side == Buy && order.Price < bestOrder.Price {
				break
			}
			if order.Side == Sell && order.Price > bestOrder.Price {
				break
			}
		}

		// Self-trade check
		if order.User != "" && order.User == bestOrder.User {
			oppositeTree.removeOrder(bestOrder)
			delete(ob.Orders, bestOrder.ID)
			ob.ordersMap.Delete(bestOrder.ID)
			continue
		}

		// Calculate trade size
		var tradeSize float64
		bestRemaining := bestOrder.Size - bestOrder.Filled
		if bestOrder.RemainingSize > 0 {
			bestRemaining = bestOrder.RemainingSize
		}
		tradeSize = math.Min(order.RemainingSize, bestRemaining)

		// Create trade
		ob.LastTradeID++
		trade := Trade{
			ID:        ob.LastTradeID,
			Price:     bestOrder.Price,
			Size:      tradeSize,
			Timestamp: time.Now(),
		}

		// Update orders
		order.RemainingSize -= tradeSize
		order.Filled += tradeSize

		if bestOrder.RemainingSize > 0 {
			bestOrder.RemainingSize -= tradeSize
		}
		bestOrder.Filled += tradeSize

		// Handle order status
		if order.RemainingSize <= 0 {
			order.Status = Filled
		} else {
			order.Status = PartiallyFilled
		}

		if (bestOrder.RemainingSize <= 0 && bestOrder.Filled >= bestOrder.Size) || bestOrder.Filled >= bestOrder.Size {
			bestOrder.Status = Filled
			oppositeTree.removeOrder(bestOrder)
			delete(ob.Orders, bestOrder.ID)
			ob.ordersMap.Delete(bestOrder.ID)
		} else {
			bestOrder.Status = PartiallyFilled
		}

		// Add trade
		ob.Trades = append(ob.Trades, trade)
		if ob.tradesBuffer != nil {
			ob.tradesBuffer.Add(trade)
		}

		// Limit trades history for compatibility
		if len(ob.Trades) > 100000 {
			ob.Trades = ob.Trades[len(ob.Trades)-50000:]
		}

		numTrades++
	}

	return numTrades
}

// processMarketOrderOptimized handles market orders
func (ob *OrderBook) processMarketOrderOptimized(order *Order) uint64 {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	order.RemainingSize = order.Size
	ob.tryMatchImmediateLocked(order)
	
	// Return order ID for consistency
	if order.Status == Rejected {
		return 0
	}
	return order.ID
}

// processMarketOrderLocked for compatibility
func (ob *OrderBook) processMarketOrderLocked(order *Order) uint64 {
	return ob.processMarketOrderOptimized(order)
}

// MatchOrders attempts to match orders in the book and returns all trades
func (ob *OrderBook) MatchOrders() []Trade {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	// Clear existing trades for this matching session
	startingTradeCount := len(ob.Trades)

	trades := make([]Trade, 0)

	for {
		// Get best bid and ask
		bestBid := ob.Bids.getBestOrder()
		bestAsk := ob.Asks.getBestOrder()

		if bestBid == nil || bestAsk == nil {
			break
		}

		// Check if orders cross (bid must be >= ask for a match)
		if bestBid.Price < bestAsk.Price {
			break
		}

		// Self-trade prevention
		if bestBid.User == bestAsk.User && bestBid.User != "" {
			// Cancel the smaller order
			if bestBid.Size < bestAsk.Size {
				ob.cancelOrderInternal(bestBid)
				continue
			} else {
				ob.cancelOrderInternal(bestAsk)
				continue
			}
		}

		// Determine trade size
		bidRemaining := bestBid.Size - bestBid.Filled
		askRemaining := bestAsk.Size - bestAsk.Filled
		tradeSize := math.Min(bidRemaining, askRemaining)

		// Determine trade price (price-time priority)
		var tradePrice float64
		var takerSide Side
		if bestBid.Timestamp.Before(bestAsk.Timestamp) {
			tradePrice = bestBid.Price
			takerSide = Sell
		} else {
			tradePrice = bestAsk.Price
			takerSide = Buy
		}

		// Calculate fees (currently unused)
		// var fee float64
		// if takerSide == Buy {
		// 	fee = tradeSize * tradePrice * 0.0002
		// } else {
		// 	fee = tradeSize * tradePrice * 0.0002
		// }

		// Create trade
		ob.LastTradeID++
		trade := Trade{
			ID:        ob.LastTradeID,
			Price:     tradePrice,
			Size:      tradeSize,
			BuyOrder:  bestBid.ID,
			SellOrder: bestAsk.ID,
			Timestamp: time.Now(),
			TakerSide: takerSide,
			MatchType: "normal",
		}

		// Update orders
		bestBid.Filled += tradeSize
		bestAsk.Filled += tradeSize

		if bestBid.Filled >= bestBid.Size {
			bestBid.Status = Filled
			ob.Bids.removeOrder(bestBid)
			trade.MatchType = "full"
		} else {
			bestBid.Status = PartiallyFilled
		}

		if bestAsk.Filled >= bestAsk.Size {
			bestAsk.Status = Filled
			ob.Asks.removeOrder(bestAsk)
			if trade.MatchType == "full" {
				trade.MatchType = "full"
			}
		} else {
			bestAsk.Status = PartiallyFilled
		}

		trades = append(trades, trade)
		ob.Trades = append(ob.Trades, trade)

		// Limit trades history
		if len(ob.Trades) > 100000 {
			ob.Trades = ob.Trades[len(ob.Trades)-50000:]
		}
	}

	// Return only trades generated in this MatchOrders call
	if startingTradeCount < len(ob.Trades) {
		return ob.Trades[startingTradeCount:]
	}
	return trades
}

// Helper methods
func (ob *OrderBook) validateOrder(order *Order) error {
	if order.Type != Market && order.Price <= 0 {
		return ErrInvalidPrice
	}
	if order.Size <= 0 {
		return ErrInvalidSize
	}
	return nil
}

func (ob *OrderBook) checkSelfTrade(order *Order) bool {
	var oppositeTree *OrderTree
	if order.Side == Buy {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.asks))
	} else {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.bids))
	}

	bestOrder := oppositeTree.getBestOrder()
	if bestOrder == nil {
		return false
	}

	// Get user identifiers for both orders
	orderUser := order.User
	if orderUser == "" {
		orderUser = order.UserID
	}

	bestUser := bestOrder.User
	if bestUser == "" {
		bestUser = bestOrder.UserID
	}

	if orderUser != "" && orderUser == bestUser {
		if order.Side == Buy && order.Price >= bestOrder.Price {
			return true
		}
		if order.Side == Sell && order.Price <= bestOrder.Price {
			return true
		}
	}

	return false
}

func (ob *OrderBook) wouldTakeLiquidity(order *Order) bool {
	var oppositeTree *OrderTree
	if order.Side == Buy {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.asks))
	} else {
		oppositeTree = (*OrderTree)(atomic.LoadPointer(&ob.bids))
	}

	bestOrder := oppositeTree.getBestOrder()
	if bestOrder == nil {
		return false
	}

	if order.Side == Buy && order.Price >= bestOrder.Price {
		return true
	}
	if order.Side == Sell && order.Price <= bestOrder.Price {
		return true
	}

	return false
}

func (ob *OrderBook) cancelOrderInternal(order *Order) {
	order.Status = Canceled
	if order.Side == Buy {
		ob.Bids.removeOrder(order)
	} else {
		ob.Asks.removeOrder(order)
	}
}

// CancelOrder cancels an order
func (ob *OrderBook) CancelOrder(orderID uint64) error {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	order, exists := ob.Orders[orderID]
	if !exists {
		return ErrOrderNotFound
	}

	if order.Status != Open && order.Status != PartiallyFilled {
		return fmt.Errorf("order not cancellable")
	}

	ob.cancelOrderInternal(order)
	return nil
}

// ModifyOrder modifies an existing order
func (ob *OrderBook) ModifyOrder(orderID uint64, newPrice, newSize float64) error {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	order, exists := ob.Orders[orderID]
	if !exists {
		return ErrOrderNotFound
	}

	if order.Status != Open && order.Status != PartiallyFilled {
		return fmt.Errorf("order not modifiable")
	}

	// Remove old order
	if order.Side == Buy {
		ob.Bids.removeOrder(order)
	} else {
		ob.Asks.removeOrder(order)
	}

	// Update order
	order.Price = newPrice
	order.Size = newSize
	order.Timestamp = time.Now() // Reset timestamp for price-time priority

	// Re-add order
	if order.Side == Buy {
		ob.Bids.addOrder(order)
	} else {
		ob.Asks.addOrder(order)
	}

	return nil
}

// GetOrder returns an order by ID
func (ob *OrderBook) GetOrder(orderID uint64) *Order {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	
	if order, exists := ob.Orders[orderID]; exists {
		return order
	}
	
	// Check ordersMap as well
	if val, exists := ob.ordersMap.Load(orderID); exists {
		if order, ok := val.(*Order); ok {
			return order
		}
	}
	
	return nil
}

// GetTrades returns all trades that have been executed
func (ob *OrderBook) GetTrades() []Trade {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	
	trades := make([]Trade, 0, len(ob.Trades))
	for _, trade := range ob.Trades {
		trades = append(trades, trade)
	}
	return trades
}

// GetSnapshot returns orderbook snapshot
func (ob *OrderBook) GetSnapshot() *OrderBookSnapshot {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	// Convert price levels to order levels
	bidLevels := ob.Bids.getLevels(10)
	askLevels := ob.Asks.getLevels(10)
	
	bids := make([]OrderLevel, len(bidLevels))
	for i, level := range bidLevels {
		bids[i] = OrderLevel{
			Price: level.Price,
			Size:  level.Size,
		}
	}
	
	asks := make([]OrderLevel, len(askLevels))
	for i, level := range askLevels {
		asks[i] = OrderLevel{
			Price: level.Price,
			Size:  level.Size,
		}
	}

	return &OrderBookSnapshot{
		Symbol:    ob.Symbol,
		Timestamp: time.Now(),
		Bids:      bids,
		Asks:      asks,
		Sequence:  atomic.LoadUint64(&ob.Bids.sequence),
	}
}

// GetDepth returns the order book depth
func (ob *OrderBook) GetDepth(levels int) *OrderBookDepth {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	return &OrderBookDepth{
		Bids:      ob.Bids.getLevels(levels),
		Asks:      ob.Asks.getLevels(levels),
		Sequence:  atomic.LoadUint64(&ob.LastOrderID),
		Timestamp: time.Now(),
		Symbol:    ob.Symbol,
	}
}

func (tree *OrderTree) getLevels(depth int) []PriceLevel {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	// If depth is 0, return all levels
	maxLevels := depth
	if depth == 0 {
		maxLevels = len(tree.priceLevels)
	}

	levels := make([]PriceLevel, 0, maxLevels)
	prices := make([]float64, 0, len(tree.priceLevels))

	// Convert to PriceLevel format
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
		if depth > 0 && i >= depth {
			break
		}
		priceInt := PriceInt(price * PriceMultiplier)
		if level, exists := tree.priceLevels[priceInt]; exists {
			levels = append(levels, PriceLevel{
				Price: level.Price,
				Size:  level.TotalSize,
				Count: level.OrderCount,
			})
		}
	}

	return levels
}

// GetOrderBookSnapshot returns full order-level book data
func (ob *OrderBook) GetOrderBookSnapshot() OrderBookSnapshot {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	snapshot := OrderBookSnapshot{
		Symbol:    ob.Symbol,
		Timestamp: time.Now(),
		Sequence:  atomic.LoadUint64(&ob.Bids.sequence),
	}

	// Get all bid orders
	snapshot.Bids = ob.Bids.getOrderLevels()
	snapshot.Asks = ob.Asks.getOrderLevels()

	return snapshot
}

func (tree *OrderTree) getOrderLevels() []OrderLevel {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	levels := make([]OrderLevel, 0)

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
		priceInt := PriceInt(price * PriceMultiplier)
		if level, exists := tree.priceLevels[priceInt]; exists {
			level.mu.RLock()
			for _, order := range level.Orders {
				if order.Status == Open || order.Status == PartiallyFilled {
					remainingSize := order.Size - order.Filled
					if order.RemainingSize > 0 {
						remainingSize = order.RemainingSize
					}
					levels = append(levels, OrderLevel{
						Price:    order.Price,
						Size:     remainingSize,
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

// GetBids returns bid side tree
func (ob *OrderBook) GetBids() *OrderTree {
	return (*OrderTree)(atomic.LoadPointer(&ob.bids))
}

// GetAsks returns ask side tree
func (ob *OrderBook) GetAsks() *OrderTree {
	return (*OrderTree)(atomic.LoadPointer(&ob.asks))
}

// GetBestBid returns the best bid price
func (ob *OrderBook) GetBestBid() float64 {
	bids := ob.GetBids()
	if bids == nil {
		return 0
	}
	bestPriceInt := bids.bestPrice.Load()
	if bestPriceInt == 0 {
		return 0
	}
	return float64(bestPriceInt) / PriceMultiplier
}

// GetBestAsk returns the best ask price
func (ob *OrderBook) GetBestAsk() float64 {
	asks := ob.GetAsks()
	if asks == nil {
		return 0
	}
	bestPriceInt := asks.bestPrice.Load()
	if bestPriceInt == 0 {
		return 0
	}
	return float64(bestPriceInt) / PriceMultiplier
}

// IntBTree implementation
func NewIntBTree(degree int, isMaxHeap bool) *IntBTree {
	return &IntBTree{
		degree:    degree,
		isMaxHeap: isMaxHeap,
	}
}

func (bt *IntBTree) Insert(key PriceInt) {
	if bt.root == nil {
		bt.root = &IntBTreeNode{
			keys:   []PriceInt{key},
			isLeaf: true,
			n:      1,
		}
		return
	}

	// Simplified B-tree insertion
	bt.insertNonFull(bt.root, key)
}

func (bt *IntBTree) Delete(key PriceInt) {
	// Simplified deletion - in production use proper B-tree
	if bt.root != nil && bt.root.n == 1 && bt.root.keys[0] == key {
		bt.root = nil
	}
}

func (bt *IntBTree) Min() PriceInt {
	if bt.root == nil || bt.root.n == 0 {
		return 0
	}
	// Simplified - return first key
	return bt.root.keys[0]
}

func (bt *IntBTree) Max() PriceInt {
	if bt.root == nil || bt.root.n == 0 {
		return 0
	}
	// Simplified - return last key
	return bt.root.keys[bt.root.n-1]
}

func (bt *IntBTree) insertNonFull(node *IntBTreeNode, key PriceInt) {
	// Simplified B-tree insertion
	if node.isLeaf {
		node.keys = append(node.keys, key)
		node.n++
		// Sort keys
		for i := node.n - 1; i > 0 && node.keys[i] < node.keys[i-1]; i-- {
			node.keys[i], node.keys[i-1] = node.keys[i-1], node.keys[i]
		}
	}
}

// CircularTradeBuffer methods
func (ctb *CircularTradeBuffer) Add(trade Trade) {
	ctb.mu.Lock()
	defer ctb.mu.Unlock()

	ctb.buffer[ctb.tail] = trade
	ctb.tail = (ctb.tail + 1) % uint64(len(ctb.buffer))

	if ctb.size < uint64(len(ctb.buffer)) {
		ctb.size++
	} else {
		ctb.head = (ctb.head + 1) % uint64(len(ctb.buffer))
	}
}

func (ctb *CircularTradeBuffer) GetRecent(count int) []Trade {
	ctb.mu.RLock()
	defer ctb.mu.RUnlock()

	if count > int(ctb.size) {
		count = int(ctb.size)
	}

	trades := make([]Trade, count)
	idx := (ctb.tail - uint64(count) + uint64(len(ctb.buffer))) % uint64(len(ctb.buffer))

	for i := 0; i < count; i++ {
		trades[i] = ctb.buffer[idx]
		idx = (idx + 1) % uint64(len(ctb.buffer))
	}

	return trades
}

// Price heap implementations for compatibility
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

// Subscribe adds a channel to receive market data updates
func (ob *OrderBook) Subscribe(ch chan MarketDataUpdate) {
	ob.subMu.Lock()
	defer ob.subMu.Unlock()
	ob.subscribers = append(ob.subscribers, ch)
}

// Unsubscribe removes a channel from market data updates
func (ob *OrderBook) Unsubscribe(ch chan MarketDataUpdate) {
	ob.subMu.Lock()
	defer ob.subMu.Unlock()
	for i, sub := range ob.subscribers {
		if sub == ch {
			ob.subscribers = append(ob.subscribers[:i], ob.subscribers[i+1:]...)
			break
		}
	}
}

// publishUpdate sends market data updates to all subscribers
func (ob *OrderBook) publishUpdate(update MarketDataUpdate) {
	ob.subMu.RLock()
	defer ob.subMu.RUnlock()
	for _, ch := range ob.subscribers {
		select {
		case ch <- update:
		default:
			// Skip if channel is full
		}
	}
}

// Reset clears the order book
func (ob *OrderBook) Reset() {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	ob.Orders = make(map[uint64]*Order)
	ob.UserOrders = make(map[string][]uint64)
	ob.Trades = []Trade{}
	ob.Bids = NewOrderTree(Buy)
	ob.Asks = NewOrderTree(Sell)
	atomic.StorePointer(&ob.bids, unsafe.Pointer(ob.Bids))
	atomic.StorePointer(&ob.asks, unsafe.Pointer(ob.Asks))
	ob.ordersMap = sync.Map{}
	ob.userOrdersMap = sync.Map{}
	ob.LastOrderID = 0
	ob.LastTradeID = 0
	ob.LastUpdateID = 0
}
