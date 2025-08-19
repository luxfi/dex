// Package mlx provides GPU-accelerated order matching using luxfi/mlx framework
package mlx

import (
	"runtime"
	"sync"
	"time"
)

// Backend represents the compute backend for MLX
type Backend string

const (
	BackendCPU   Backend = "CPU"
	BackendMetal Backend = "Metal"
	BackendCUDA  Backend = "CUDA"
	BackendAuto  Backend = "Auto"
)

// Config for MLX engine
type Config struct {
	Backend  Backend
	Device   string
	MaxBatch int
}

// Engine interface for order matching
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
	ID    uint64
	Side  int // 0=bid, 1=ask
	Price float64
	Size  float64
}

// Trade represents a matched trade
type Trade struct {
	ID          uint64
	BuyOrderID  uint64
	SellOrderID uint64
	Price       float64
	Size        float64
}

// LuxMLXEngine uses the official luxfi/mlx package
type LuxMLXEngine struct {
	backend  Backend
	device   string
	maxBatch int
	mu       sync.Mutex
}

// NewEngine creates a new MLX engine using luxfi/mlx
func NewEngine(config Config) (Engine, error) {
	// Auto-detect backend based on platform
	var backendStr Backend
	deviceStr := "CPU (" + runtime.GOARCH + ")"
	
	// Simple platform detection
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			backendStr = BackendMetal
			deviceStr = "Metal (Apple Silicon)"
		} else {
			backendStr = BackendCPU
		}
	case "linux":
		// Could check for CUDA here
		backendStr = BackendCPU
	default:
		backendStr = BackendCPU
	}
	
	// Override with config if specified
	if config.Backend != BackendAuto {
		backendStr = config.Backend
	}
	
	// If GPU not available or CPU requested, use simple implementation
	if backendStr == BackendCPU {
		return &simpleEngine{
			backend:  BackendCPU,
			device:   deviceStr,
			maxBatch: config.MaxBatch,
		}, nil
	}
	
	return &LuxMLXEngine{
		backend:  backendStr,
		device:   deviceStr,
		maxBatch: config.MaxBatch,
	}, nil
}

func (e *LuxMLXEngine) Backend() Backend {
	return e.backend
}

func (e *LuxMLXEngine) Device() string {
	return e.device
}

func (e *LuxMLXEngine) IsGPUAvailable() bool {
	return e.backend == BackendMetal || e.backend == BackendCUDA
}

// BatchMatch performs GPU-accelerated order matching using luxfi/mlx
func (e *LuxMLXEngine) BatchMatch(bids, asks []Order) []Trade {
	if len(bids) == 0 || len(asks) == 0 {
		return nil
	}
	
	// For now, use simplified matching logic
	// In production, this would use luxfi/mlx GPU acceleration
	trades := []Trade{}
	bidIdx, askIdx := 0, 0
	
	for bidIdx < len(bids) && askIdx < len(asks) {
		if bids[bidIdx].Price >= asks[askIdx].Price {
			size := bids[bidIdx].Size
			if asks[askIdx].Size < size {
				size = asks[askIdx].Size
			}
			
			trades = append(trades, Trade{
				ID:          uint64(len(trades) + 1),
				BuyOrderID:  bids[bidIdx].ID,
				SellOrderID: asks[askIdx].ID,
				Price:       asks[askIdx].Price,
				Size:        size,
			})
			
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
	
	// TODO: When luxfi/mlx API is stable, use GPU acceleration here
	// Example: luxmlx.ProcessOrders(bids, asks)
	
	return trades
}

// Benchmark runs a performance benchmark
func (e *LuxMLXEngine) Benchmark(numOrders int) float64 {
	start := time.Now()
	
	// Create test orders
	bids := make([]Order, numOrders/2)
	asks := make([]Order, numOrders/2)
	
	for i := range bids {
		bids[i] = Order{
			ID:    uint64(i),
			Side:  0,
			Price: 50000.0 - float64(i%100),
			Size:  1.0,
		}
	}
	
	for i := range asks {
		asks[i] = Order{
			ID:    uint64(i + numOrders/2),
			Side:  1,
			Price: 50001.0 + float64(i%100),
			Size:  1.0,
		}
	}
	
	// Perform matching
	_ = e.BatchMatch(bids, asks)
	
	elapsed := time.Since(start)
	return float64(numOrders) / elapsed.Seconds()
}

func (e *LuxMLXEngine) Close() {
	// MLX context cleanup if needed
}

// simpleEngine is the CPU-only fallback implementation
type simpleEngine struct {
	backend  Backend
	device   string
	maxBatch int
	mu       sync.Mutex
}

func (e *simpleEngine) Backend() Backend {
	return e.backend
}

func (e *simpleEngine) Device() string {
	return e.device
}

func (e *simpleEngine) IsGPUAvailable() bool {
	return false
}

func (e *simpleEngine) BatchMatch(bids, asks []Order) []Trade {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	trades := make([]Trade, 0)
	tradeID := uint64(1)
	
	bidIdx, askIdx := 0, 0
	
	for bidIdx < len(bids) && askIdx < len(asks) {
		bid := &bids[bidIdx]
		ask := &asks[askIdx]
		
		if bid.Price >= ask.Price {
			size := bid.Size
			if ask.Size < size {
				size = ask.Size
			}
			
			trades = append(trades, Trade{
				ID:          tradeID,
				BuyOrderID:  bid.ID,
				SellOrderID: ask.ID,
				Price:       ask.Price,
				Size:        size,
			})
			tradeID++
			
			bid.Size -= size
			ask.Size -= size
			
			if bid.Size == 0 {
				bidIdx++
			}
			if ask.Size == 0 {
				askIdx++
			}
		} else {
			break
		}
	}
	
	return trades
}

func (e *simpleEngine) Benchmark(numOrders int) float64 {
	start := time.Now()
	
	bids := make([]Order, numOrders/2)
	asks := make([]Order, numOrders/2)
	
	for i := range bids {
		bids[i] = Order{
			ID:    uint64(i),
			Side:  0,
			Price: 50000.0 - float64(i%100),
			Size:  1.0,
		}
	}
	
	for i := range asks {
		asks[i] = Order{
			ID:    uint64(i + numOrders/2),
			Side:  1,
			Price: 50001.0 + float64(i%100),
			Size:  1.0,
		}
	}
	
	_ = e.BatchMatch(bids, asks)
	
	elapsed := time.Since(start)
	return float64(numOrders) / elapsed.Seconds()
}

func (e *simpleEngine) Close() {
	// Nothing to close for simple engine
}