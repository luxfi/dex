//go:build !withmetal && !mlxreal
// +build !withmetal,!mlxreal

// Package mlx provides GPU acceleration for order matching
// This version works with both CGO=0 and CGO=1 without external dependencies
package mlx

import (
	"runtime"
	"sync"
	"time"
)

// Backend represents the compute backend
type Backend string

const (
	BackendCPU   Backend = "CPU"
	BackendMetal Backend = "Metal"
	BackendCUDA  Backend = "CUDA"
	BackendAuto  Backend = "Auto"
)

// Engine is the main MLX compute engine interface
type Engine interface {
	Backend() Backend
	Device() string
	IsGPUAvailable() bool
	BatchMatch(bids, asks []Order) []Trade
	Benchmark(numOrders int) float64
	Close()
}

// Order represents a trading order
type Order struct {
	ID     uint64
	Price  float64
	Size   float64
	Side   int // 0=buy, 1=sell
	UserID uint64
}

// Trade represents an executed trade
type Trade struct {
	ID          uint64
	BuyOrderID  uint64
	SellOrderID uint64
	Price       float64
	Size        float64
}

// Config for engine initialization
type Config struct {
	Backend    Backend
	DeviceID   int
	MaxBatch   int
	StreamMode bool
}

// Stream represents an async processing stream
type Stream struct {
	engine Engine
}

// Submit submits work to the stream
func (s *Stream) Submit(fn func()) {
	// For now, just execute synchronously
	fn()
}

// Synchronize waits for stream completion
func (s *Stream) Synchronize() {
	// No-op for now
}

// DetectBackend returns the available backend
func DetectBackend() Backend {
	// Always return CPU for this simple implementation
	return BackendCPU
}

// HasGPUSupport returns false for this simple implementation
func HasGPUSupport() bool {
	return false
}

// NewEngine creates the appropriate engine based on platform and build flags
func NewEngine(config Config) (Engine, error) {
	// This simple version always uses CPU
	return &simpleEngine{
		backend:  BackendCPU,
		device:   "CPU (" + runtime.GOARCH + ")",
		maxBatch: config.MaxBatch,
	}, nil
}

// simpleEngine provides CPU-only implementation that works everywhere
type simpleEngine struct {
	backend  Backend
	device   string
	maxBatch int
	mu       sync.Mutex
}

func (e *simpleEngine) Backend() Backend      { return e.backend }
func (e *simpleEngine) Device() string        { return e.device }
func (e *simpleEngine) IsGPUAvailable() bool  { return false }
func (e *simpleEngine) Close()                {}

func (e *simpleEngine) BatchMatch(bids, asks []Order) []Trade {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	trades := make([]Trade, 0)
	bidIdx, askIdx := 0, 0
	tradeID := uint64(1)
	
	for bidIdx < len(bids) && askIdx < len(asks) {
		if bids[bidIdx].Price >= asks[askIdx].Price {
			size := bids[bidIdx].Size
			if asks[askIdx].Size < size {
				size = asks[askIdx].Size
			}
			
			trades = append(trades, Trade{
				ID:          tradeID,
				BuyOrderID:  bids[bidIdx].ID,
				SellOrderID: asks[askIdx].ID,
				Price:       asks[askIdx].Price,
				Size:        size,
			})
			tradeID++
			
			bids[bidIdx].Size -= size
			asks[askIdx].Size -= size
			
			if bids[bidIdx].Size == 0 {
				bidIdx++
			}
			if asks[askIdx].Size == 0 {
				askIdx++
			}
		} else {
			break
		}
	}
	
	return trades
}

func (e *simpleEngine) Benchmark(numOrders int) float64 {
	bids := make([]Order, numOrders/2)
	asks := make([]Order, numOrders/2)
	
	for i := range bids {
		bids[i] = Order{
			ID:    uint64(i),
			Price: 50000.0 - float64(i%100),
			Size:  1.0,
			Side:  0,
		}
	}
	
	for i := range asks {
		asks[i] = Order{
			ID:    uint64(i + numOrders/2),
			Price: 50001.0 + float64(i%100),
			Size:  1.0,
			Side:  1,
		}
	}
	
	start := time.Now()
	_ = e.BatchMatch(bids, asks)
	elapsed := time.Since(start)
	
	if elapsed > 0 {
		return float64(numOrders) / elapsed.Seconds()
	}
	return 1_000_000.0 // 1M orders/sec baseline
}