// +build cgo

package mlx

// #cgo CPPFLAGS: -I../../bridge
// #cgo CXXFLAGS: -std=c++17 -O3
// #cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
// #cgo linux LDFLAGS: -lcudart -lcuda
// #cgo LDFLAGS: -L../../bridge -lmlx_engine -lstdc++
/*
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Backend types
enum Backend {
    BACKEND_CPU,
    BACKEND_METAL,
    BACKEND_CUDA
};

// Order structure for C side
typedef struct {
    uint64_t id;
    double price;
    double size;
    int side;
} MLXOrder;

// Trade result
typedef struct {
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    double size;
} MLXTrade;

// C interface functions
void* mlx_engine_create();
void mlx_engine_destroy(void* engine);
int mlx_engine_get_backend(void* engine);
const char* mlx_engine_get_device_name(void* engine);
int mlx_engine_is_gpu_available(void* engine);
double mlx_engine_benchmark(void* engine, int num_orders);
int mlx_engine_match_orders(
    void* engine,
    MLXOrder* bids, int num_bids,
    MLXOrder* asks, int num_asks,
    MLXTrade* trades, int max_trades
);

#ifdef __cplusplus
}
#endif
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/luxfi/dex/pkg/lx"
)

// CGOMLXMatcher uses C++ MLX engine with CUDA/Metal support
type CGOMLXMatcher struct {
	engine   unsafe.Pointer
	backend  string
	device   string
	hasGPU   bool
}

// NewCGOMLXMatcher creates a new CGO-based MLX matcher
func NewCGOMLXMatcher() (*CGOMLXMatcher, error) {
	engine := C.mlx_engine_create()
	if engine == nil {
		return nil, fmt.Errorf("failed to create MLX engine")
	}
	
	m := &CGOMLXMatcher{
		engine: engine,
	}
	
	// Get backend info
	backend := C.mlx_engine_get_backend(engine)
	switch backend {
	case 0:
		m.backend = "CPU"
	case 1:
		m.backend = "Metal"
	case 2:
		m.backend = "CUDA"
	}
	
	// Get device name
	deviceName := C.mlx_engine_get_device_name(engine)
	m.device = C.GoString(deviceName)
	
	// Check GPU availability
	m.hasGPU = C.mlx_engine_is_gpu_available(engine) != 0
	
	return m, nil
}

// Close releases the MLX engine
func (m *CGOMLXMatcher) Close() {
	if m.engine != nil {
		C.mlx_engine_destroy(m.engine)
		m.engine = nil
	}
}

// GetBackend returns the active backend (CPU, Metal, or CUDA)
func (m *CGOMLXMatcher) GetBackend() string {
	return m.backend
}

// GetDevice returns the device name
func (m *CGOMLXMatcher) GetDevice() string {
	return m.device
}

// HasGPU returns true if GPU acceleration is available
func (m *CGOMLXMatcher) HasGPU() bool {
	return m.hasGPU
}

// MatchOrders performs GPU-accelerated order matching
func (m *CGOMLXMatcher) MatchOrders(bids, asks []*lx.Order) ([]*lx.Trade, error) {
	if len(bids) == 0 || len(asks) == 0 {
		return nil, nil
	}
	
	// Convert Go orders to C++ format
	cBids := make([]C.MLXOrder, len(bids))
	for i, order := range bids {
		cBids[i] = C.MLXOrder{
			id:    C.uint64_t(order.ID),
			price: C.double(order.Price),
			size:  C.double(order.Size),
			side:  0, // Buy
		}
	}
	
	cAsks := make([]C.MLXOrder, len(asks))
	for i, order := range asks {
		cAsks[i] = C.MLXOrder{
			id:    C.uint64_t(order.ID),
			price: C.double(order.Price),
			size:  C.double(order.Size),
			side:  1, // Sell
		}
	}
	
	// Prepare trade buffer
	maxTrades := len(bids)
	if len(asks) < maxTrades {
		maxTrades = len(asks)
	}
	cTrades := make([]C.MLXTrade, maxTrades)
	
	// Call C++ matching engine
	numTrades := C.mlx_engine_match_orders(
		m.engine,
		(*C.MLXOrder)(unsafe.Pointer(&cBids[0])), C.int(len(bids)),
		(*C.MLXOrder)(unsafe.Pointer(&cAsks[0])), C.int(len(asks)),
		(*C.MLXTrade)(unsafe.Pointer(&cTrades[0])), C.int(maxTrades),
	)
	
	// Convert results back to Go
	trades := make([]*lx.Trade, numTrades)
	for i := 0; i < int(numTrades); i++ {
		trades[i] = &lx.Trade{
			BuyOrder:  uint64(cTrades[i].buy_order_id),
			SellOrder: uint64(cTrades[i].sell_order_id),
			Price:     float64(cTrades[i].price),
			Size:      float64(cTrades[i].size),
		}
	}
	
	return trades, nil
}

// Benchmark runs a performance benchmark
func (m *CGOMLXMatcher) Benchmark(numOrders int) float64 {
	throughput := C.mlx_engine_benchmark(m.engine, C.int(numOrders))
	return float64(throughput)
}

// GetMLXInfo returns information about the MLX backend
func GetMLXInfo() string {
	// Temporary engine just to get info
	engine := C.mlx_engine_create()
	if engine == nil {
		return "MLX engine not available"
	}
	defer C.mlx_engine_destroy(engine)
	
	deviceName := C.GoString(C.mlx_engine_get_device_name(engine))
	hasGPU := C.mlx_engine_is_gpu_available(engine) != 0
	
	info := fmt.Sprintf("MLX Engine: %s", deviceName)
	if hasGPU {
		info += " (GPU Accelerated)"
		if runtime.GOOS == "linux" {
			info += " - CUDA Backend"
		} else if runtime.GOOS == "darwin" {
			info += " - Metal Backend"
		}
	} else {
		info += " (CPU Only)"
	}
	
	return info
}