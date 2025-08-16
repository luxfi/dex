package mlx

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"sync"
)

// MLXEngine wraps the Python MLX matching engine
type MLXEngine struct {
	pythonPath string
	scriptPath string
	mu         sync.Mutex
	stats      *Stats
}

// Stats contains MLX engine statistics
type Stats struct {
	OrdersProcessed         uint64  `json:"orders_processed"`
	TradesExecuted          uint64  `json:"trades_executed"`
	AvgLatencyNs            float64 `json:"avg_latency_ns"`
	ThroughputOrdersPerSec  float64 `json:"throughput_orders_per_sec"`
}

// Order represents an order for MLX processing
type Order struct {
	OrderID   uint64  `json:"order_id"`
	Price     float64 `json:"price"`
	Quantity  float64 `json:"quantity"`
	Timestamp int64   `json:"timestamp"`
	Side      int     `json:"side"`      // 0=buy, 1=sell
	Status    int     `json:"status"`    // 0=active, 1=filled
	UserID    int     `json:"user_id"`
}

// Trade represents a matched trade
type Trade struct {
	TradeID     uint64  `json:"trade_id"`
	BuyOrderID  uint64  `json:"buy_order_id"`
	SellOrderID uint64  `json:"sell_order_id"`
	Price       float64 `json:"price"`
	Quantity    float64 `json:"quantity"`
	Timestamp   int64   `json:"timestamp"`
}

// NewMLXEngine creates a new MLX matching engine wrapper
func NewMLXEngine() (*MLXEngine, error) {
	// Check if MLX is available
	cmd := exec.Command("python3", "-c", "import mlx.core as mx; print(mx.default_device())")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("MLX not available: %v", err)
	}
	
	fmt.Printf("MLX available on device: %s", output)
	
	return &MLXEngine{
		pythonPath: "python3",
		scriptPath: "pkg/mlx/mlx_matching.py",
		stats:      &Stats{},
	}, nil
}

// MatchOrders performs GPU-accelerated matching via MLX
func (e *MLXEngine) MatchOrders(bids, asks []Order) ([]Trade, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	// Prepare input data
	input := map[string]interface{}{
		"bids": bids,
		"asks": asks,
	}
	
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	
	// Call Python MLX engine
	cmd := exec.Command(e.pythonPath, "-c", fmt.Sprintf(`
import sys
import json
import numpy as np
sys.path.insert(0, '.')
from pkg.mlx.mlx_matching import MLXMatchingEngine

# Parse input
data = json.loads('%s')
bids = np.array([[o['order_id'], o['price'], o['quantity'], o['timestamp'], o['side'], o['status'], o['user_id']] for o in data['bids']])
asks = np.array([[o['order_id'], o['price'], o['quantity'], o['timestamp'], o['side'], o['status'], o['user_id']] for o in data['asks']])

# Create engine and match
engine = MLXMatchingEngine()
trades = engine.match_orders_batch(bids, asks)

# Convert to JSON
result = []
for t in trades:
    result.append({
        'trade_id': t.trade_id,
        'buy_order_id': t.buy_order_id,
        'sell_order_id': t.sell_order_id,
        'price': t.price,
        'quantity': t.quantity,
        'timestamp': t.timestamp
    })

print(json.dumps(result))
`, string(inputJSON)))
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("MLX matching failed: %v", err)
	}
	
	// Parse results
	var trades []Trade
	if err := json.Unmarshal(output, &trades); err != nil {
		return nil, err
	}
	
	// Update stats
	e.stats.OrdersProcessed += uint64(len(bids) + len(asks))
	e.stats.TradesExecuted += uint64(len(trades))
	
	return trades, nil
}

// RunBenchmark runs the MLX benchmark
func (e *MLXEngine) RunBenchmark() error {
	cmd := exec.Command(e.pythonPath, e.scriptPath)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("benchmark failed: %v", err)
	}
	
	fmt.Printf("%s\n", output)
	return nil
}

// GetStats returns engine statistics
func (e *MLXEngine) GetStats() *Stats {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stats
}

// IsAvailable checks if MLX is available on the system
func IsAvailable() bool {
	cmd := exec.Command("python3", "-c", "import mlx.core")
	err := cmd.Run()
	return err == nil
}