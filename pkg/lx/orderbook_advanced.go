package lx

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Advanced order types - defined in types_common.go

// TimeInForce - defined in types_common.go

// AdvancedOrderBook implements full-featured order book with all order types
type AdvancedOrderBook struct {
	mu sync.RWMutex

	// Core data structures
	symbol    string
	bidLevels map[float64]*PriceLevel
	askLevels map[float64]*PriceLevel
	bidHeap   *OrderHeap
	askHeap   *OrderHeap

	// Order tracking
	orders            map[uint64]*AdvancedOrder
	stopOrders        map[uint64]*AdvancedOrder
	icebergOrders     map[uint64]*IcebergData
	hiddenOrders      map[uint64]*AdvancedOrder
	conditionalOrders map[uint64]*ConditionalOrder

	// Market data
	lastPrice       float64
	lastTradeTime   time.Time
	volume24h       float64
	trades          []Trade
	marketDataFeeds []chan MarketUpdate

	// Statistics
	totalOrders uint64
	totalTrades uint64
	totalVolume float64
	bestBid     float64
	bestAsk     float64

	// Configuration
	enableSelfTrade bool
	enableHidden    bool
	tickSize        float64
	lotSize         float64

	// Performance metrics
	lastUpdateID   uint64
	sequenceNumber uint64
}

// AdvancedOrder extends basic order with all features
type AdvancedOrder struct {
	// Basic fields
	ID            uint64
	ClientID      string
	UserID        string
	Symbol        string
	Side          Side
	Type          OrderType
	Price         float64
	StopPrice     float64
	Size          float64
	ExecutedSize  float64
	RemainingSize float64

	// Advanced fields
	TimeInForce    TimeInForce
	ExpireTime     time.Time
	DisplaySize    float64 // For iceberg
	PegOffset      float64 // For pegged orders
	TrailAmount    float64 // For trailing stop
	MinExecuteSize float64 // Minimum execution size
	AllOrNone      bool
	PostOnly       bool
	ReduceOnly     bool
	Hidden         bool

	// Status
	Status        OrderStatus
	CreateTime    time.Time
	UpdateTime    time.Time
	LastFillTime  time.Time
	LastFillPrice float64
	LastFillSize  float64

	// Fees
	MakerFee float64
	TakerFee float64
	FeesPaid float64
}

// OrderStatus - defined in types_common.go

// IcebergData - defined in types_common.go (alias for IcebergState)

// ConditionalOrder - defined in types_common.go

// MarketUpdate - defined in types_common.go

type OrderHeap []*AdvancedOrder

func (h OrderHeap) Len() int { return len(h) }
func (h OrderHeap) Less(i, j int) bool {
	// For bids: higher price = higher priority
	// For asks: lower price = higher priority
	if h[i].Side == Buy {
		if h[i].Price != h[j].Price {
			return h[i].Price > h[j].Price
		}
	} else {
		if h[i].Price != h[j].Price {
			return h[i].Price < h[j].Price
		}
	}
	// Same price: earlier order has priority (FIFO)
	return h[i].CreateTime.Before(h[j].CreateTime)
}
func (h OrderHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *OrderHeap) Push(x interface{}) {
	*h = append(*h, x.(*AdvancedOrder))
}
func (h *OrderHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// NewAdvancedOrderBook creates a fully-featured order book
func NewAdvancedOrderBook(symbol string) *AdvancedOrderBook {
	bidHeap := &OrderHeap{}
	askHeap := &OrderHeap{}
	heap.Init(bidHeap)
	heap.Init(askHeap)

	return &AdvancedOrderBook{
		symbol:            symbol,
		bidLevels:         make(map[float64]*PriceLevel),
		askLevels:         make(map[float64]*PriceLevel),
		bidHeap:           bidHeap,
		askHeap:           askHeap,
		orders:            make(map[uint64]*AdvancedOrder),
		stopOrders:        make(map[uint64]*AdvancedOrder),
		icebergOrders:     make(map[uint64]*IcebergData),
		hiddenOrders:      make(map[uint64]*AdvancedOrder),
		conditionalOrders: make(map[uint64]*ConditionalOrder),
		trades:            make([]Trade, 0),
		marketDataFeeds:   make([]chan MarketUpdate, 0),
		tickSize:          0.01,
		lotSize:           0.001,
	}
}

// AddOrder processes any type of order
func (book *AdvancedOrderBook) AddOrder(order *AdvancedOrder) ([]Trade, error) {
	book.mu.Lock()
	defer book.mu.Unlock()

	// Validate order
	if err := book.validateOrder(order); err != nil {
		order.Status = StatusRejected
		return nil, err
	}

	// Set initial values
	order.Status = StatusNew
	order.CreateTime = time.Now()
	order.UpdateTime = order.CreateTime
	order.RemainingSize = order.Size
	atomic.AddUint64(&book.totalOrders, 1)

	// Route to appropriate handler
	switch order.Type {
	case Market:
		return book.processMarketOrder(order)
	case Limit:
		return book.processLimitOrder(order)
	case Stop, StopLimit:
		return book.processStop(order)
	case Iceberg:
		return book.processIcebergOrder(order)
	case Peg:
		return book.processPeggedOrder(order)
	case Bracket:
		return book.processTrailingStop(order)
	default:
		return nil, fmt.Errorf("unsupported order type: %v", order.Type)
	}
}

// processMarketOrder executes market order immediately
func (book *AdvancedOrderBook) processMarketOrder(order *AdvancedOrder) ([]Trade, error) {
	trades := make([]Trade, 0)

	var oppositeHeap *OrderHeap
	if order.Side == Buy {
		oppositeHeap = book.askHeap
	} else {
		oppositeHeap = book.bidHeap
	}

	// Match against opposite side
	for oppositeHeap.Len() > 0 && order.RemainingSize > 0 {
		bestOrder := (*oppositeHeap)[0]

		// Skip hidden orders for market orders
		if bestOrder.Hidden {
			heap.Pop(oppositeHeap)
			continue
		}

		// Execute trade
		trade := book.executeTrade(order, bestOrder)
		trades = append(trades, trade)

		// Remove filled orders
		if bestOrder.RemainingSize == 0 {
			heap.Pop(oppositeHeap)
			bestOrder.Status = StatusFilled
			delete(book.orders, bestOrder.ID)
		}

		if order.RemainingSize == 0 {
			order.Status = StatusFilled
			break
		}
	}

	// Cancel any remaining market order quantity
	if order.RemainingSize > 0 {
		order.Status = StatusCanceled
	}

	book.publishMarketData("trade", trades)
	return trades, nil
}

// processLimitOrder adds limit order to book or matches immediately
func (book *AdvancedOrderBook) processLimitOrder(order *AdvancedOrder) ([]Trade, error) {
	trades := make([]Trade, 0)

	// Check if order crosses the spread
	canMatch := false
	if order.Side == Buy && book.bestAsk > 0 && order.Price >= book.bestAsk {
		canMatch = true
	} else if order.Side == Sell && book.bestBid > 0 && order.Price <= book.bestBid {
		canMatch = true
	}

	// Post-only check
	if order.PostOnly && canMatch {
		order.Status = StatusRejected
		return nil, errors.New("post-only order would cross spread")
	}

	// Match if possible
	if canMatch {
		trades = book.matchOrder(order)
	}

	// Add remaining to book
	if order.RemainingSize > 0 {
		if order.TimeInForce == TIF_IOC {
			order.Status = StatusCanceled
		} else if order.TimeInForce == TIF_FOK && order.RemainingSize < order.Size {
			// FOK not fully filled, cancel entire order
			order.Status = StatusCanceled
			order.RemainingSize = order.Size
			order.ExecutedSize = 0
			return nil, errors.New("FOK order could not be fully filled")
		} else {
			book.addToBook(order)
		}
	}

	book.updateBestPrices()
	book.publishMarketData("orderbook", nil)

	return trades, nil
}

// processStop handles stop and stop-limit orders
func (book *AdvancedOrderBook) processStop(order *AdvancedOrder) ([]Trade, error) {
	order.Status = StatusPending
	book.stopOrders[order.ID] = order
	book.orders[order.ID] = order

	// Check if stop should trigger immediately
	if book.shouldTriggerStop(order) {
		return book.triggerStop(order)
	}

	return nil, nil
}

// processIcebergOrder handles iceberg orders
func (book *AdvancedOrderBook) processIcebergOrder(order *AdvancedOrder) ([]Trade, error) {
	// Store iceberg data
	book.icebergOrders[order.ID] = &IcebergData{
		TotalSize:     order.Size,
		DisplaySize:   order.DisplaySize,
		RemainingSize: order.Size,
		RefillCount:   0,
	}

	// Create visible portion
	visibleOrder := *order
	visibleOrder.Size = order.DisplaySize
	visibleOrder.Type = Limit

	return book.processLimitOrder(&visibleOrder)
}

// processHiddenOrder handles hidden orders
func (book *AdvancedOrderBook) processHiddenOrder(order *AdvancedOrder) ([]Trade, error) {
	order.Hidden = true
	book.hiddenOrders[order.ID] = order

	// Hidden orders still match but aren't displayed
	trades := book.matchOrder(order)

	if order.RemainingSize > 0 {
		book.addToBook(order)
	}

	return trades, nil
}

// processPeggedOrder handles pegged orders
func (book *AdvancedOrderBook) processPeggedOrder(order *AdvancedOrder) ([]Trade, error) {
	// Update price based on peg
	if order.Side == Buy {
		order.Price = book.bestBid + order.PegOffset
	} else {
		order.Price = book.bestAsk + order.PegOffset
	}

	return book.processLimitOrder(order)
}

// processTrailingStop handles trailing stop orders
func (book *AdvancedOrderBook) processTrailingStop(order *AdvancedOrder) ([]Trade, error) {
	// Set initial stop price
	if order.Side == Sell {
		order.StopPrice = book.lastPrice - order.TrailAmount
	} else {
		order.StopPrice = book.lastPrice + order.TrailAmount
	}

	order.Type = Stop
	return book.processStop(order)
}

// matchOrder matches an order against the book
func (book *AdvancedOrderBook) matchOrder(order *AdvancedOrder) []Trade {
	trades := make([]Trade, 0)

	var oppositeHeap *OrderHeap
	if order.Side == Buy {
		oppositeHeap = book.askHeap
	} else {
		oppositeHeap = book.bidHeap
	}

	for oppositeHeap.Len() > 0 && order.RemainingSize > 0 {
		bestOrder := (*oppositeHeap)[0]

		// Price check for limit orders
		if order.Type == Limit {
			if order.Side == Buy && order.Price < bestOrder.Price {
				break
			}
			if order.Side == Sell && order.Price > bestOrder.Price {
				break
			}
		}

		// Self-trade prevention
		if book.enableSelfTrade && order.UserID == bestOrder.UserID {
			heap.Pop(oppositeHeap)
			continue
		}

		// Execute trade
		trade := book.executeTrade(order, bestOrder)
		trades = append(trades, trade)

		// Update or remove matched order
		if bestOrder.RemainingSize == 0 {
			heap.Pop(oppositeHeap)
			bestOrder.Status = StatusFilled
			delete(book.orders, bestOrder.ID)

			// Check for iceberg refill
			if iceberg, exists := book.icebergOrders[bestOrder.ID]; exists {
				book.refillIceberg(bestOrder, iceberg)
			}
		}
	}

	return trades
}

// executeTrade executes a trade between two orders
func (book *AdvancedOrderBook) executeTrade(taker, maker *AdvancedOrder) Trade {
	// Determine trade size
	tradeSize := math.Min(taker.RemainingSize, maker.RemainingSize)
	tradePrice := maker.Price

	// Update orders
	taker.RemainingSize -= tradeSize
	taker.ExecutedSize += tradeSize
	taker.LastFillTime = time.Now()
	taker.LastFillPrice = tradePrice
	taker.LastFillSize = tradeSize

	maker.RemainingSize -= tradeSize
	maker.ExecutedSize += tradeSize
	maker.LastFillTime = time.Now()
	maker.LastFillPrice = tradePrice
	maker.LastFillSize = tradeSize

	// Update status
	if taker.RemainingSize == 0 {
		taker.Status = StatusFilled
	} else {
		taker.Status = StatusPartiallyFilled
	}

	if maker.RemainingSize == 0 {
		maker.Status = StatusFilled
	} else {
		maker.Status = StatusPartiallyFilled
	}

	// Create trade record
	trade := Trade{
		ID:        atomic.AddUint64(&book.totalTrades, 1),
		Price:     tradePrice,
		Size:      tradeSize,
		BuyOrder:  taker.ID,
		SellOrder: maker.ID,
		Timestamp: time.Now(),
	}

	if taker.Side == Sell {
		trade.BuyOrder = maker.ID
		trade.SellOrder = taker.ID
	}

	// Update book statistics
	book.lastPrice = tradePrice
	book.lastTradeTime = trade.Timestamp
	book.volume24h += tradeSize * tradePrice
	book.totalVolume += tradeSize * tradePrice
	book.trades = append(book.trades, trade)

	// Check stop orders
	book.checkStops(tradePrice)

	// Update trailing stops
	book.updateTrailingStops(tradePrice)

	return trade
}

// addToBook adds order to the order book
func (book *AdvancedOrderBook) addToBook(order *AdvancedOrder) {
	book.orders[order.ID] = order

	if order.Side == Buy {
		heap.Push(book.bidHeap, order)
		level := book.bidLevels[order.Price]
		if level == nil {
			level = &PriceLevel{
				Price:  order.Price,
				Size:   0,
				Orders: make([]*Order, 0),
			}
			book.bidLevels[order.Price] = level
		}
		level.Size += order.RemainingSize
		// Convert AdvancedOrder to Order for storage
		basicOrder := &Order{
			ID:     order.ID,
			Symbol: order.Symbol,
			Side:   order.Side,
			Type:   order.Type,
			Price:  order.Price,
			Size:   order.Size,
		}
		level.Orders = append(level.Orders, basicOrder)
	} else {
		heap.Push(book.askHeap, order)
		level := book.askLevels[order.Price]
		if level == nil {
			level = &PriceLevel{
				Price:  order.Price,
				Size:   0,
				Orders: make([]*Order, 0),
			}
			book.askLevels[order.Price] = level
		}
		level.Size += order.RemainingSize
		// Convert AdvancedOrder to Order for storage
		basicOrder := &Order{
			ID:     order.ID,
			Symbol: order.Symbol,
			Side:   order.Side,
			Type:   order.Type,
			Price:  order.Price,
			Size:   order.Size,
		}
		level.Orders = append(level.Orders, basicOrder)
	}

	order.Status = StatusNew
}

// CancelOrder cancels an order
func (book *AdvancedOrderBook) CancelOrder(orderID uint64) error {
	book.mu.Lock()
	defer book.mu.Unlock()

	order, exists := book.orders[orderID]
	if !exists {
		// Check stop orders
		if stopOrder, exists := book.stopOrders[orderID]; exists {
			stopOrder.Status = StatusCanceled
			delete(book.stopOrders, orderID)
			return nil
		}
		return errors.New("order not found")
	}

	order.Status = StatusCanceled
	order.UpdateTime = time.Now()

	// Remove from book
	book.removeFromBook(order)
	delete(book.orders, orderID)

	// Clean up special order types
	delete(book.icebergOrders, orderID)
	delete(book.hiddenOrders, orderID)
	delete(book.conditionalOrders, orderID)

	book.updateBestPrices()
	book.publishMarketData("cancel", order)

	return nil
}

// ModifyOrder modifies an existing order
func (book *AdvancedOrderBook) ModifyOrder(orderID uint64, newPrice, newSize float64) error {
	book.mu.Lock()
	defer book.mu.Unlock()

	order, exists := book.orders[orderID]
	if !exists {
		return errors.New("order not found")
	}

	// Remove from current position
	book.removeFromBook(order)

	// Update order
	order.Price = newPrice
	order.Size = newSize
	order.RemainingSize = newSize - order.ExecutedSize
	order.UpdateTime = time.Now()

	// Re-add to book
	book.addToBook(order)

	book.updateBestPrices()
	book.publishMarketData("modify", order)

	return nil
}

// removeFromBook removes order from book structures
func (book *AdvancedOrderBook) removeFromBook(order *AdvancedOrder) {
	// Remove from heap
	if order.Side == Buy {
		for i, o := range *book.bidHeap {
			if o.ID == order.ID {
				heap.Remove(book.bidHeap, i)
				break
			}
		}
		// Remove from level
		if level := book.bidLevels[order.Price]; level != nil {
			level.Size -= order.RemainingSize
			for i, o := range level.Orders {
				if o.ID == order.ID {
					level.Orders = append(level.Orders[:i], level.Orders[i+1:]...)
					break
				}
			}
			if len(level.Orders) == 0 {
				delete(book.bidLevels, order.Price)
			}
		}
	} else {
		for i, o := range *book.askHeap {
			if o.ID == order.ID {
				heap.Remove(book.askHeap, i)
				break
			}
		}
		// Remove from level
		if level := book.askLevels[order.Price]; level != nil {
			level.Size -= order.RemainingSize
			for i, o := range level.Orders {
				if o.ID == order.ID {
					level.Orders = append(level.Orders[:i], level.Orders[i+1:]...)
					break
				}
			}
			if len(level.Orders) == 0 {
				delete(book.askLevels, order.Price)
			}
		}
	}
}

// checkStops checks and triggers stop orders
func (book *AdvancedOrderBook) checkStops(lastPrice float64) {
	for _, order := range book.stopOrders {
		if book.shouldTriggerStop(order) {
			book.triggerStop(order)
		}
	}
}

// shouldTriggerStop checks if stop order should trigger
func (book *AdvancedOrderBook) shouldTriggerStop(order *AdvancedOrder) bool {
	if order.Side == Buy {
		return book.lastPrice >= order.StopPrice
	}
	return book.lastPrice <= order.StopPrice
}

// triggerStop converts stop order to market/limit
func (book *AdvancedOrderBook) triggerStop(order *AdvancedOrder) ([]Trade, error) {
	delete(book.stopOrders, order.ID)

	if order.Type == Stop {
		order.Type = Market
	} else {
		order.Type = Limit
	}

	return book.AddOrder(order)
}

// updateTrailingStops updates trailing stop orders
func (book *AdvancedOrderBook) updateTrailingStops(lastPrice float64) {
	for _, order := range book.stopOrders {
		if order.Type != Bracket {
			continue
		}

		if order.Side == Sell {
			newStop := lastPrice - order.TrailAmount
			if newStop > order.StopPrice {
				order.StopPrice = newStop
			}
		} else {
			newStop := lastPrice + order.TrailAmount
			if newStop < order.StopPrice {
				order.StopPrice = newStop
			}
		}
	}
}

// refillIceberg refills iceberg order
func (book *AdvancedOrderBook) refillIceberg(order *AdvancedOrder, iceberg *IcebergData) {
	if iceberg.RemainingSize <= 0 {
		return
	}

	refillSize := math.Min(iceberg.DisplaySize, iceberg.RemainingSize)
	iceberg.RemainingSize -= refillSize
	iceberg.RefillCount++

	// Create new visible order
	newOrder := *order
	newOrder.ID = atomic.AddUint64(&book.sequenceNumber, 1)
	newOrder.Size = refillSize
	newOrder.RemainingSize = refillSize
	newOrder.ExecutedSize = 0
	newOrder.Status = StatusNew

	book.addToBook(&newOrder)
}

// updateBestPrices updates best bid/ask prices
func (book *AdvancedOrderBook) updateBestPrices() {
	if book.bidHeap.Len() > 0 {
		book.bestBid = (*book.bidHeap)[0].Price
	} else {
		book.bestBid = 0
	}

	if book.askHeap.Len() > 0 {
		book.bestAsk = (*book.askHeap)[0].Price
	} else {
		book.bestAsk = 0
	}
}

// validateOrder validates order parameters
func (book *AdvancedOrderBook) validateOrder(order *AdvancedOrder) error {
	if order.Size <= 0 {
		return errors.New("order size must be positive")
	}

	if order.Type == Limit && order.Price <= 0 {
		return errors.New("limit order must have positive price")
	}

	// Check tick size with tolerance for floating point precision
	const epsilon = 1e-9
	remainder := math.Mod(order.Price, book.tickSize)
	if remainder > epsilon && remainder < (book.tickSize - epsilon) {
		return fmt.Errorf("price must be multiple of tick size %.2f", book.tickSize)
	}

	// Check lot size with tolerance for floating point precision
	sizeRemainder := math.Mod(order.Size, book.lotSize)
	if sizeRemainder > epsilon && sizeRemainder < (book.lotSize - epsilon) {
		return fmt.Errorf("size must be multiple of lot size %.3f", book.lotSize)
	}

	// FOK validation
	if order.TimeInForce == TIF_FOK && !book.canFillCompletely(order) {
		return errors.New("FOK order cannot be filled completely")
	}

	return nil
}

// canFillCompletely checks if order can be filled completely
func (book *AdvancedOrderBook) canFillCompletely(order *AdvancedOrder) bool {
	availableSize := 0.0

	if order.Side == Buy {
		for _, o := range *book.askHeap {
			if order.Type == Market || order.Price >= o.Price {
				availableSize += o.RemainingSize
				if availableSize >= order.Size {
					return true
				}
			}
		}
	} else {
		for _, o := range *book.bidHeap {
			if order.Type == Market || order.Price <= o.Price {
				availableSize += o.RemainingSize
				if availableSize >= order.Size {
					return true
				}
			}
		}
	}

	return false
}

// GetDepth returns order book depth
func (book *AdvancedOrderBook) GetDepth(levels int) map[string]interface{} {
	book.mu.RLock()
	defer book.mu.RUnlock()

	bids := make([][]float64, 0)
	asks := make([][]float64, 0)

	// Aggregate bid levels
	bidPrices := make([]float64, 0, len(book.bidLevels))
	for price := range book.bidLevels {
		bidPrices = append(bidPrices, price)
	}
	// Sort descending
	for i := 0; i < len(bidPrices); i++ {
		for j := i + 1; j < len(bidPrices); j++ {
			if bidPrices[i] < bidPrices[j] {
				bidPrices[i], bidPrices[j] = bidPrices[j], bidPrices[i]
			}
		}
	}

	count := 0
	for _, price := range bidPrices {
		if levels > 0 && count >= levels {
			break
		}
		level := book.bidLevels[price]
		if !book.shouldShowLevel(level) {
			continue
		}
		bids = append(bids, []float64{price, level.Size})
		count++
	}

	// Aggregate ask levels
	askPrices := make([]float64, 0, len(book.askLevels))
	for price := range book.askLevels {
		askPrices = append(askPrices, price)
	}
	// Sort ascending
	for i := 0; i < len(askPrices); i++ {
		for j := i + 1; j < len(askPrices); j++ {
			if askPrices[i] > askPrices[j] {
				askPrices[i], askPrices[j] = askPrices[j], askPrices[i]
			}
		}
	}

	count = 0
	for _, price := range askPrices {
		if levels > 0 && count >= levels {
			break
		}
		level := book.askLevels[price]
		if !book.shouldShowLevel(level) {
			continue
		}
		asks = append(asks, []float64{price, level.Size})
		count++
	}

	return map[string]interface{}{
		"bids":      bids,
		"asks":      asks,
		"timestamp": time.Now().Unix(),
		"sequence":  atomic.LoadUint64(&book.sequenceNumber),
	}
}

// shouldShowLevel checks if level should be displayed
func (book *AdvancedOrderBook) shouldShowLevel(level *PriceLevel) bool {
	// Don't show levels with only hidden orders
	for _, order := range level.Orders {
		if !order.Hidden {
			return true
		}
	}
	return false
}

// GetSnapshot returns full order book snapshot
func (book *AdvancedOrderBook) GetSnapshot() map[string]interface{} {
	book.mu.RLock()
	defer book.mu.RUnlock()

	return map[string]interface{}{
		"symbol":    book.symbol,
		"bids":      book.GetDepth(0)["bids"],
		"asks":      book.GetDepth(0)["asks"],
		"lastPrice": book.lastPrice,
		"volume24h": book.volume24h,
		"bestBid":   book.bestBid,
		"bestAsk":   book.bestAsk,
		"timestamp": time.Now().Unix(),
		"sequence":  atomic.LoadUint64(&book.sequenceNumber),
	}
}

// SubscribeMarketData subscribes to market data updates
func (book *AdvancedOrderBook) SubscribeMarketData() chan MarketUpdate {
	book.mu.Lock()
	defer book.mu.Unlock()

	feed := make(chan MarketUpdate, 1000)
	book.marketDataFeeds = append(book.marketDataFeeds, feed)
	return feed
}

// publishMarketData publishes updates to subscribers
func (book *AdvancedOrderBook) publishMarketData(updateType string, data interface{}) {
	update := MarketUpdate{
		Type:      updateType,
		Timestamp: time.Now(),
		Data:      data,
	}

	for _, feed := range book.marketDataFeeds {
		select {
		case feed <- update:
		default:
			// Channel full, skip
		}
	}
}

// GetStatistics returns order book statistics
func (book *AdvancedOrderBook) GetStatistics() map[string]interface{} {
	book.mu.RLock()
	defer book.mu.RUnlock()

	return map[string]interface{}{
		"symbol":        book.symbol,
		"totalOrders":   atomic.LoadUint64(&book.totalOrders),
		"totalTrades":   atomic.LoadUint64(&book.totalTrades),
		"totalVolume":   book.totalVolume,
		"volume24h":     book.volume24h,
		"bestBid":       book.bestBid,
		"bestAsk":       book.bestAsk,
		"spread":        book.bestAsk - book.bestBid,
		"lastPrice":     book.lastPrice,
		"lastTradeTime": book.lastTradeTime,
		"bidLevels":     len(book.bidLevels),
		"askLevels":     len(book.askLevels),
		"activeOrders":  len(book.orders),
		"stopOrders":    len(book.stopOrders),
		"hiddenOrders":  len(book.hiddenOrders),
		"icebergOrders": len(book.icebergOrders),
	}
}
