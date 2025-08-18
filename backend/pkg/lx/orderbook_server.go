package lx

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

// SimpleOrderBookServer serves order book data
type SimpleOrderBookServer struct {
	orderBook    *OrderBook
	mu           sync.RWMutex
	blockNumber  uint64
	wsClients    map[string]chan interface{}
	conservative bool
}

// OrderSnapshot represents a simple order book snapshot
type OrderSnapshot struct {
	Symbol    string          `json:"symbol"`
	BidCount  int             `json:"bid_count"`
	AskCount  int             `json:"ask_count"`
	BestBid   decimal.Decimal `json:"best_bid,omitempty"`
	BestAsk   decimal.Decimal `json:"best_ask,omitempty"`
	Timestamp int64           `json:"timestamp"`
	Block     uint64          `json:"block"`
}

// NewOrderBookServer creates a new order book server
func NewOrderBookServer(ob *OrderBook) *SimpleOrderBookServer {
	return &SimpleOrderBookServer{
		orderBook:    ob,
		wsClients:    make(map[string]chan interface{}),
		conservative: true,
	}
}

// Start starts the server
func (s *SimpleOrderBookServer) Start(ctx context.Context, port int) error {
	// Start block processor
	go s.blockProcessor(ctx)
	return nil
}

// OnOrderAdd handles order add events
func (s *SimpleOrderBookServer) OnOrderAdd(order *Order) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Broadcast to clients
	for _, ch := range s.wsClients {
		select {
		case ch <- order:
		default:
		}
	}
}

// processBlock processes a block
func (s *SimpleOrderBookServer) processBlock() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.blockNumber++
}

// generateOrderSnapshot generates a snapshot
func (s *SimpleOrderBookServer) generateOrderSnapshot(symbol string) *OrderSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	snapshot := &OrderSnapshot{
		Symbol:    symbol,
		Timestamp: time.Now().UnixNano(),
		Block:     s.blockNumber,
	}
	
	// Get order counts from the trees
	bids := s.orderBook.GetBids()
	asks := s.orderBook.GetAsks()
	
	if bids != nil {
		bids.mu.RLock()
		snapshot.BidCount = len(bids.orders)
		if bids.bestPrice.Load() != 0 {
			snapshot.BestBid = decimal.NewFromFloat(float64(bids.bestPrice.Load()) / PriceMultiplier)
		}
		bids.mu.RUnlock()
	}
	
	if asks != nil {
		asks.mu.RLock()
		snapshot.AskCount = len(asks.orders)
		if asks.bestPrice.Load() != 0 {
			snapshot.BestAsk = decimal.NewFromFloat(float64(asks.bestPrice.Load()) / PriceMultiplier)
		}
		asks.mu.RUnlock()
	}
	
	return snapshot
}

// validateState validates the order book state
func (s *SimpleOrderBookServer) validateState() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	if !s.conservative {
		return true
	}
	
	// Check bid/ask crossing
	bids := s.orderBook.GetBids()
	asks := s.orderBook.GetAsks()
	
	if bids != nil && asks != nil {
		bidBest := bids.bestPrice.Load()
		askBest := asks.bestPrice.Load()
		
		if bidBest != 0 && askBest != 0 && bidBest >= askBest {
			// Order book is crossed
			return false
		}
	}
	
	return true
}

// blockProcessor processes blocks periodically
func (s *SimpleOrderBookServer) blockProcessor(ctx context.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.processBlock()
		}
	}
}

// GetSnapshot returns the current snapshot
func (s *SimpleOrderBookServer) GetSnapshot(symbol string) (*OrderSnapshot, error) {
	snapshot := s.generateOrderSnapshot(symbol)
	if snapshot == nil {
		return nil, fmt.Errorf("failed to generate snapshot")
	}
	return snapshot, nil
}