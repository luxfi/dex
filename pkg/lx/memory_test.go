package lx

import (
	"runtime"
	"testing"
	"time"
)

// TestMemoryUsage tracks memory usage during tests
func TestMemoryUsage(t *testing.T) {
	t.Run("BaselineMemory", func(t *testing.T) {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		t.Logf("Baseline - Alloc: %v MB, TotalAlloc: %v MB, Sys: %v MB, NumGC: %v",
			m.Alloc/1024/1024, m.TotalAlloc/1024/1024, m.Sys/1024/1024, m.NumGC)
	})

	t.Run("OrderBookMemory", func(t *testing.T) {
		var m runtime.MemStats

		// Before creating orderbooks
		runtime.GC()
		runtime.ReadMemStats(&m)
		allocBefore := m.Alloc

		// Create multiple orderbooks
		books := make([]*OrderBook, 100)
		for i := 0; i < 100; i++ {
			books[i] = NewOrderBook("TEST")
			// Add some orders
			for j := 0; j < 100; j++ {
				books[i].AddOrder(&Order{
					ID:        uint64(j),
					Type:      Limit,
					Side:      Buy,
					Price:     100.0 + float64(j),
					Size:      10.0,
					User:      "test",
					Timestamp: time.Now(),
				})
			}
		}

		// After creating orderbooks
		runtime.ReadMemStats(&m)
		allocAfter := m.Alloc

		memUsed := (allocAfter - allocBefore) / 1024 / 1024
		t.Logf("OrderBooks (100x100 orders) - Memory used: %v MB", memUsed)

		// Check for reasonable memory usage (< 50MB for 10k orders)
		if memUsed > 50 {
			t.Errorf("Excessive memory usage: %v MB for 10,000 orders", memUsed)
		}
	})

	t.Run("MLXMemoryCheck", func(t *testing.T) {
		// Skip if MLX not enabled
		if !IsMLXEnabled() {
			t.Skip("MLX not enabled")
		}

		var m runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m)

		t.Logf("MLX Check - Alloc: %v MB, Sys: %v MB",
			m.Alloc/1024/1024, m.Sys/1024/1024)
	})
}

// BenchmarkMemoryAllocation benchmarks memory allocation
func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("OrderCreation", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = &Order{
				ID:        uint64(i),
				Type:      Limit,
				Side:      Buy,
				Price:     100.0,
				Size:      10.0,
				User:      "test",
				Timestamp: time.Now(),
			}
		}
	})

	b.Run("OrderBookAddOrder", func(b *testing.B) {
		book := NewOrderBook("TEST")
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			book.AddOrder(&Order{
				ID:        uint64(i),
				Type:      Limit,
				Side:      Buy,
				Price:     100.0 + float64(i%100),
				Size:      10.0,
				User:      "test",
				Timestamp: time.Now(),
			})
		}
	})
}

// Helper to check if MLX is enabled
func IsMLXEnabled() bool {
	// Check if MLX environment variable or build tag is set
	return false // Simplified for now
}
