package lx

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"
)

// VaultStrategy represents an automated trading strategy for a vault
type VaultStrategy interface {
	// Initialize sets up the strategy with vault capital
	Initialize(vaultID string, capital *big.Int) error

	// Execute runs one iteration of the strategy
	Execute(ctx context.Context, book *OrderBook) ([]*Order, error)

	// OnOrderUpdate handles order fills/cancels
	OnOrderUpdate(order *Order, status OrderStatus)

	// OnMarketData handles market data updates
	OnMarketData(snapshot *OrderBookSnapshot)

	// GetPerformance returns strategy metrics
	GetPerformance() *StrategyMetrics

	// Shutdown cleanly stops the strategy
	Shutdown() error
}

// StrategyMetrics tracks strategy performance
type StrategyMetrics struct {
	PnL         *big.Int
	TradeCount  int64
	WinRate     float64
	SharpeRatio float64
	MaxDrawdown float64
	LastUpdate  time.Time
}

// AIStrategy uses an AI model for trading decisions
type AIStrategy struct {
	vaultID       string
	capital       *big.Int
	modelEndpoint string // Colocated AI model endpoint

	// HFT parameters
	maxPositionSize float64
	minSpread       float64
	maxLatency      time.Duration

	// State
	openOrders map[string]*Order
	position   *PerpPosition
	lastSignal time.Time

	mu sync.RWMutex
}

// NewAIStrategy creates a new AI-driven strategy
func NewAIStrategy(modelEndpoint string) *AIStrategy {
	return &AIStrategy{
		modelEndpoint:   modelEndpoint,
		maxPositionSize: 0.1,                    // 10% of capital max per position
		minSpread:       0.001,                  // 0.1% minimum spread
		maxLatency:      100 * time.Microsecond, // 100μs max decision time
		openOrders:      make(map[string]*Order),
	}
}

// Initialize sets up the AI strategy
func (s *AIStrategy) Initialize(vaultID string, capital *big.Int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.vaultID = vaultID
	s.capital = capital

	// Connect to colocated AI model
	// This could be a local gRPC endpoint for ultra-low latency
	// The model runs on the same machine/rack as the matching engine

	return nil
}

// Execute runs the AI strategy - this is called on every tick
func (s *AIStrategy) Execute(ctx context.Context, book *OrderBook) ([]*Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check latency budget
	start := time.Now()
	defer func() {
		if time.Since(start) > s.maxLatency {
			// Log slow execution for monitoring
		}
	}()

	// Get market snapshot
	snapshot := book.GetSnapshot()

	// Call colocated AI model for signal
	// In production, this would be a local RPC call to the AI model
	signal := s.getAISignal(snapshot)

	// Generate orders based on signal
	orders := make([]*Order, 0)

	switch signal.Action {
	case "BUY":
		if s.canBuy(signal.Size) {
			order := &Order{
				Symbol:   book.Symbol,
				Side:     Buy,
				Type:     Limit,
				Price:    signal.Price,
				Size:     signal.Size,
				User:     s.vaultID,
				PostOnly: true, // Maker only for lower fees
			}
			orders = append(orders, order)
		}

	case "SELL":
		if s.canSell(signal.Size) {
			order := &Order{
				Symbol:   book.Symbol,
				Side:     Sell,
				Type:     Limit,
				Price:    signal.Price,
				Size:     signal.Size,
				User:     s.vaultID,
				PostOnly: true,
			}
			orders = append(orders, order)
		}

	case "MARKET_MAKE":
		// Place orders on both sides of the spread
		if len(snapshot.Bids) > 0 && len(snapshot.Asks) > 0 {
			spread := snapshot.Asks[0].Price - snapshot.Bids[0].Price
			if spread > s.minSpread*snapshot.Bids[0].Price {
				// Place bid
				bidOrder := &Order{
					Symbol:   book.Symbol,
					Side:     Buy,
					Type:     Limit,
					Price:    snapshot.Bids[0].Price + 0.01,
					Size:     signal.Size,
					User:     s.vaultID,
					PostOnly: true,
				}
				orders = append(orders, bidOrder)

				// Place ask
				askOrder := &Order{
					Symbol:   book.Symbol,
					Side:     Sell,
					Type:     Limit,
					Price:    snapshot.Asks[0].Price - 0.01,
					Size:     signal.Size,
					User:     s.vaultID,
					PostOnly: true,
				}
				orders = append(orders, askOrder)
			}
		}
	}

	s.lastSignal = time.Now()
	return orders, nil
}

// getAISignal calls the colocated AI model
func (s *AIStrategy) getAISignal(snapshot *OrderBookSnapshot) *AISignal {
	// In production, this would call a colocated AI service
	// For HFT, the model would be:
	// 1. Running on same machine or adjacent rack
	// 2. Connected via shared memory or Unix socket
	// 3. Using optimized inference (ONNX, TensorRT, etc)
	// 4. Pre-loaded in GPU memory for <1ms inference

	// Simplified signal for now
	return &AISignal{
		Action:     "MARKET_MAKE",
		Price:      0,
		Size:       0.1,
		Confidence: 0.8,
	}
}

// AISignal represents a trading signal from the AI model
type AISignal struct {
	Action     string // BUY, SELL, MARKET_MAKE, HOLD
	Price      float64
	Size       float64
	Confidence float64
	Timestamp  time.Time
}

func (s *AIStrategy) canBuy(size float64) bool {
	// Check position limits
	if s.position != nil {
		currentExposure := s.position.Size * s.position.EntryPrice
		maxExposure := float64(s.capital.Int64()) * s.maxPositionSize
		return currentExposure+(size*s.position.EntryPrice) <= maxExposure
	}
	return true
}

func (s *AIStrategy) canSell(size float64) bool {
	// Check if we have position to sell
	if s.position != nil && s.position.Size >= size {
		return true
	}
	return false
}

func (s *AIStrategy) OnOrderUpdate(order *Order, status OrderStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch status {
	case StatusFilled:
		// Update position
		if s.position == nil {
			s.position = &PerpPosition{
				Symbol:     order.Symbol,
				Size:       0,
				EntryPrice: order.Price,
			}
		}

		if order.Side == Buy {
			// Update average entry price
			totalCost := s.position.Size*s.position.EntryPrice + order.Size*order.Price
			s.position.Size += order.Size
			s.position.EntryPrice = totalCost / s.position.Size
		} else {
			s.position.Size -= order.Size
		}

		delete(s.openOrders, fmt.Sprintf("%d", order.ID))

	case StatusCanceled:
		delete(s.openOrders, fmt.Sprintf("%d", order.ID))
	}
}

func (s *AIStrategy) OnMarketData(snapshot *OrderBookSnapshot) {
	// Update internal state with latest market data
	// This is used by the AI model for next decision
}

func (s *AIStrategy) GetPerformance() *StrategyMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return &StrategyMetrics{
		PnL:        big.NewInt(0), // Calculate from position
		TradeCount: int64(len(s.openOrders)),
		LastUpdate: time.Now(),
	}
}

func (s *AIStrategy) Shutdown() error {
	// Cancel all open orders
	// Close positions
	// Disconnect from AI model
	return nil
}

// ============================================
// Simple HFT Market Making Strategy
// ============================================

// SimpleMMStrategy is a basic market making strategy
type SimpleMMStrategy struct {
	vaultID string
	capital *big.Int

	// Parameters
	spreadBps  int     // Spread in basis points
	orderSize  float64 // Size per order
	numLevels  int     // Number of price levels
	skewFactor float64 // Skew based on inventory

	// State
	bidOrders    []*Order
	askOrders    []*Order
	netPosition  float64
	lastMidPrice float64

	mu sync.RWMutex
}

// NewSimpleMMStrategy creates a simple market maker
func NewSimpleMMStrategy(spreadBps int, orderSize float64) *SimpleMMStrategy {
	return &SimpleMMStrategy{
		spreadBps:  spreadBps,
		orderSize:  orderSize,
		numLevels:  5,
		skewFactor: 0.1,
	}
}

func (s *SimpleMMStrategy) Initialize(vaultID string, capital *big.Int) error {
	s.vaultID = vaultID
	s.capital = capital
	return nil
}

func (s *SimpleMMStrategy) Execute(ctx context.Context, book *OrderBook) ([]*Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Cancel existing orders
	s.cancelAllOrders()

	// Get current mid price
	snapshot := book.GetSnapshot()
	if len(snapshot.Bids) == 0 || len(snapshot.Asks) == 0 {
		return nil, nil
	}

	midPrice := (snapshot.Bids[0].Price + snapshot.Asks[0].Price) / 2
	s.lastMidPrice = midPrice

	// Calculate spread with inventory skew
	halfSpread := midPrice * float64(s.spreadBps) / 10000 / 2
	skew := s.netPosition * s.skewFactor * midPrice / 100

	orders := make([]*Order, 0)

	// Place bid orders
	for i := 0; i < s.numLevels; i++ {
		bidPrice := midPrice - halfSpread - skew - float64(i)*halfSpread/2
		bidOrder := &Order{
			Symbol:   book.Symbol,
			Side:     Buy,
			Type:     Limit,
			Price:    bidPrice,
			Size:     s.orderSize,
			User:     s.vaultID,
			PostOnly: true,
		}
		orders = append(orders, bidOrder)
		s.bidOrders = append(s.bidOrders, bidOrder)
	}

	// Place ask orders
	for i := 0; i < s.numLevels; i++ {
		askPrice := midPrice + halfSpread - skew + float64(i)*halfSpread/2
		askOrder := &Order{
			Symbol:   book.Symbol,
			Side:     Sell,
			Type:     Limit,
			Price:    askPrice,
			Size:     s.orderSize,
			User:     s.vaultID,
			PostOnly: true,
		}
		orders = append(orders, askOrder)
		s.askOrders = append(s.askOrders, askOrder)
	}

	return orders, nil
}

func (s *SimpleMMStrategy) cancelAllOrders() {
	s.bidOrders = s.bidOrders[:0]
	s.askOrders = s.askOrders[:0]
}

func (s *SimpleMMStrategy) OnOrderUpdate(order *Order, status OrderStatus) {
	if status == StatusFilled {
		if order.Side == Buy {
			s.netPosition += order.Size
		} else {
			s.netPosition -= order.Size
		}
	}
}

func (s *SimpleMMStrategy) OnMarketData(snapshot *OrderBookSnapshot) {
	// React to market data changes
}

func (s *SimpleMMStrategy) GetPerformance() *StrategyMetrics {
	return &StrategyMetrics{
		PnL:        big.NewInt(0),
		TradeCount: 0,
		LastUpdate: time.Now(),
	}
}

func (s *SimpleMMStrategy) Shutdown() error {
	s.cancelAllOrders()
	return nil
}

// ============================================
// Strategy Execution Engine
// ============================================

// StrategyEngine manages strategy execution for vaults
type StrategyEngine struct {
	strategies map[string]VaultStrategy // vaultID -> strategy
	orderBooks map[string]*OrderBook    // symbol -> orderbook

	// Execution control
	tickInterval time.Duration
	stopCh       chan struct{}

	mu sync.RWMutex
}

// NewStrategyEngine creates a new strategy execution engine
func NewStrategyEngine(tickInterval time.Duration) *StrategyEngine {
	return &StrategyEngine{
		strategies:   make(map[string]VaultStrategy),
		orderBooks:   make(map[string]*OrderBook),
		tickInterval: tickInterval,
		stopCh:       make(chan struct{}),
	}
}

// RegisterStrategy registers a vault's strategy
func (se *StrategyEngine) RegisterStrategy(vaultID string, strategy VaultStrategy, capital *big.Int) error {
	se.mu.Lock()
	defer se.mu.Unlock()

	// Initialize strategy
	if err := strategy.Initialize(vaultID, capital); err != nil {
		return err
	}

	se.strategies[vaultID] = strategy
	return nil
}

// Start begins strategy execution
func (se *StrategyEngine) Start(ctx context.Context) {
	ticker := time.NewTicker(se.tickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-se.stopCh:
			return
		case <-ticker.C:
			se.executeTick(ctx)
		}
	}
}

// executeTick runs one iteration of all strategies
func (se *StrategyEngine) executeTick(ctx context.Context) {
	se.mu.RLock()
	strategies := make(map[string]VaultStrategy)
	for k, v := range se.strategies {
		strategies[k] = v
	}
	se.mu.RUnlock()

	// Execute all strategies in parallel for speed
	var wg sync.WaitGroup
	for vaultID, strategy := range strategies {
		wg.Add(1)
		go func(vid string, s VaultStrategy) {
			defer wg.Done()

			// Get relevant order book (simplified - would match vault's trading pairs)
			book := se.orderBooks["BTC-USD"]
			if book == nil {
				return
			}

			// Execute strategy
			orders, err := s.Execute(ctx, book)
			if err != nil {
				// Log error
				return
			}

			// Submit orders to order book
			for range orders {
				// This would integrate with actual order submission
				// book.AddOrder(order)
			}
		}(vaultID, strategy)
	}
	wg.Wait()
}

// Stop halts strategy execution
func (se *StrategyEngine) Stop() {
	close(se.stopCh)

	// Shutdown all strategies
	se.mu.Lock()
	defer se.mu.Unlock()

	for _, strategy := range se.strategies {
		strategy.Shutdown()
	}
}
