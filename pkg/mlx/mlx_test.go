// +build cgo

package mlx

import (
	"fmt"
	"testing"

	"github.com/luxfi/dex/pkg/lx"
)

func TestMLXEngine(t *testing.T) {
	// Create MLX engine
	matcher, err := NewCGOMLXMatcher()
	if err != nil {
		t.Fatalf("Failed to create MLX matcher: %v", err)
	}
	defer matcher.Close()

	// Check backend detection
	backend := matcher.GetBackend()
	device := matcher.GetDevice()
	hasGPU := matcher.HasGPU()

	t.Logf("MLX Backend: %s", backend)
	t.Logf("Device: %s", device)
	t.Logf("GPU Available: %v", hasGPU)

	// Ensure we got a valid backend
	if backend != "CPU" && backend != "Metal" && backend != "CUDA" {
		t.Errorf("Invalid backend: %s", backend)
	}

	// Device should not be empty
	if device == "" {
		t.Error("Device name is empty")
	}
}

func TestMLXInfo(t *testing.T) {
	info := GetMLXInfo()
	t.Logf("MLX Info: %s", info)

	if info == "MLX engine not available" {
		t.Error("MLX engine should be available")
	}
}

func TestMLXBenchmark(t *testing.T) {
	matcher, err := NewCGOMLXMatcher()
	if err != nil {
		t.Fatalf("Failed to create MLX matcher: %v", err)
	}
	defer matcher.Close()

	// Run a small benchmark
	throughput := matcher.Benchmark(1000)
	t.Logf("Throughput: %.2f orders/sec", throughput)

	if throughput <= 0 {
		t.Error("Benchmark throughput should be positive")
	}
}

func BenchmarkMLXMatching(b *testing.B) {
	matcher, err := NewCGOMLXMatcher()
	if err != nil {
		b.Fatalf("Failed to create MLX matcher: %v", err)
	}
	defer matcher.Close()

	// Create test orders
	bids := make([]*lx.Order, 100)
	asks := make([]*lx.Order, 100)

	for i := 0; i < 100; i++ {
		bids[i] = &lx.Order{
			ID:    uint64(i),
			Price: 50000.0 - float64(i)*0.1,
			Size:  1.0,
		}
		asks[i] = &lx.Order{
			ID:    uint64(i + 100),
			Price: 50001.0 + float64(i)*0.1,
			Size:  1.0,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trades, err := matcher.MatchOrders(bids, asks)
		if err != nil {
			b.Fatal(err)
		}
		_ = trades
	}
}

func ExampleMLXInfo() {
	info := GetMLXInfo()
	fmt.Println(info)
	// Output will vary based on hardware:
	// MLX Engine: Apple Silicon GPU (Metal) (GPU Accelerated) - Metal Backend
	// or
	// MLX Engine: NVIDIA RTX 4090 (CUDA) (GPU Accelerated) - CUDA Backend
	// or
	// MLX Engine: CPU (No GPU acceleration) (CPU Only)
}