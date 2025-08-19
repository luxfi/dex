package mlx

import (
	"fmt"
	"testing"
)

func TestMLXEngine(t *testing.T) {
	// Create MLX engine
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		t.Fatalf("Failed to create MLX engine: %v", err)
	}
	defer engine.Close()

	// Check backend detection
	backend := engine.Backend()
	device := engine.Device()
	hasGPU := engine.IsGPUAvailable()

	t.Logf("MLX Backend: %s", backend)
	t.Logf("Device: %s", device)
	t.Logf("GPU Available: %v", hasGPU)

	// Ensure we got a valid backend
	if backend != BackendCPU && backend != BackendMetal && backend != BackendCUDA {
		t.Errorf("Invalid backend: %s", backend)
	}

	// Device should not be empty
	if device == "" {
		t.Error("Device name is empty")
	}
}

func TestMLXInfo(t *testing.T) {
	// Test engine creation
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
		MaxBatch: 1000,
	})
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	backend := engine.Backend()
	t.Logf("Detected Backend: %s", backend)
	
	hasGPU := engine.IsGPUAvailable()
	t.Logf("GPU Support: %v", hasGPU)
	
	device := engine.Device()
	t.Logf("Device: %s", device)
	
	// Should always have at least CPU backend
	if backend == "" {
		t.Error("Backend should not be empty")
	}
}

func TestMLXBenchmark(t *testing.T) {
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		t.Fatalf("Failed to create MLX engine: %v", err)
	}
	defer engine.Close()

	// Run a small benchmark
	throughput := engine.Benchmark(1000)
	t.Logf("Throughput: %.2f orders/sec", throughput)

	if throughput <= 0 {
		t.Error("Benchmark throughput should be positive")
	}
}

func BenchmarkMLXMatching(b *testing.B) {
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		b.Fatalf("Failed to create MLX engine: %v", err)
	}
	defer engine.Close()

	// Create test orders
	bids := make([]Order, 100)
	asks := make([]Order, 100)

	for i := 0; i < 100; i++ {
		bids[i] = Order{
			ID:    uint64(i),
			Price: 50000.0 - float64(i)*0.1,
			Size:  1.0,
			Side:  0,
		}
		asks[i] = Order{
			ID:    uint64(i + 100),
			Price: 50001.0 + float64(i)*0.1,
			Size:  1.0,
			Side:  1,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trades := engine.BatchMatch(bids, asks)
		_ = trades
	}
}

func ExampleNewEngine() {
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		fmt.Printf("Failed to create engine: %v\n", err)
		return
	}
	defer engine.Close()
	
	fmt.Printf("Backend: %s\n", engine.Backend())
	fmt.Printf("Device: %s\n", engine.Device())
	// Output will vary based on hardware:
	// Backend: Metal
	// Device: Apple Silicon GPU
	// or
	// Backend: CPU
	// Device: CPU (arm64)
}

// Tests for the Engine interface
func TestEngineInterface(t *testing.T) {
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		t.Logf("Failed to create engine (CGO may be disabled): %v", err)
		// This should never fail with pure Go implementation
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	// Test Backend
	backend := engine.Backend()
	t.Logf("Engine Backend: %s", backend)
	
	// Test Device
	device := engine.Device()
	t.Logf("Engine Device: %s", device)
	
	// Test GPU availability
	hasGPU := engine.IsGPUAvailable()
	t.Logf("GPU Available: %v", hasGPU)
	
	// Test BatchMatch
	bids := []Order{
		{ID: 1, Price: 100.0, Size: 10.0, Side: 0},
		{ID: 2, Price: 99.0, Size: 20.0, Side: 0},
	}
	asks := []Order{
		{ID: 3, Price: 100.0, Size: 15.0, Side: 1},
		{ID: 4, Price: 101.0, Size: 25.0, Side: 1},
	}
	
	trades := engine.BatchMatch(bids, asks)
	t.Logf("Matched %d trades", len(trades))
	
	// Test Benchmark
	throughput := engine.Benchmark(1000)
	t.Logf("Benchmark Throughput: %.2f orders/sec", throughput)
	
	if throughput <= 0 {
		t.Error("Throughput should be positive")
	}
}

func BenchmarkEngineMatching(b *testing.B) {
	engine, err := NewEngine(Config{
		Backend: BackendAuto,
	})
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	// Create test orders
	bids := make([]Order, 100)
	asks := make([]Order, 100)
	
	for i := 0; i < 100; i++ {
		bids[i] = Order{
			ID:    uint64(i),
			Price: 100.0 - float64(i)*0.01,
			Size:  10.0,
			Side:  0,
		}
		asks[i] = Order{
			ID:    uint64(i + 100),
			Price: 100.0 + float64(i)*0.01,
			Size:  10.0,
			Side:  1,
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.BatchMatch(bids, asks)
	}
}