package mlx

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleEngine(t *testing.T) {
	// Test simple engine (CPU fallback)
	engine := &simpleEngine{
		backend:  BackendCPU,
		device:   "CPU (test)",
		maxBatch: 100,
	}

	assert.Equal(t, BackendCPU, engine.Backend())
	assert.Equal(t, "CPU (test)", engine.Device())
	assert.False(t, engine.IsGPUAvailable())

	// Test batch matching
	bids := []Order{
		{ID: 1, Side: 0, Price: 100, Size: 10},
		{ID: 2, Side: 0, Price: 99, Size: 5},
	}
	asks := []Order{
		{ID: 3, Side: 1, Price: 99, Size: 8},
		{ID: 4, Side: 1, Price: 101, Size: 10},
	}

	trades := engine.BatchMatch(bids, asks)
	assert.NotEmpty(t, trades)
	assert.Equal(t, uint64(1), trades[0].ID)

	// Test benchmark
	throughput := engine.Benchmark(100)
	assert.Greater(t, throughput, float64(0))

	// Test close
	engine.Close()
}

func TestLuxMLXEngine(t *testing.T) {
	// Test LuxMLX engine
	engine := &LuxMLXEngine{
		backend:  BackendMetal,
		device:   "Metal (Test)",
		maxBatch: 1000,
	}

	assert.Equal(t, BackendMetal, engine.Backend())
	assert.Equal(t, "Metal (Test)", engine.Device())
	assert.True(t, engine.IsGPUAvailable())

	// Test batch matching
	bids := []Order{
		{ID: 1, Side: 0, Price: 100, Size: 10},
		{ID: 2, Side: 0, Price: 99, Size: 5},
	}
	asks := []Order{
		{ID: 3, Side: 1, Price: 99, Size: 8},
		{ID: 4, Side: 1, Price: 101, Size: 10},
	}

	trades := engine.BatchMatch(bids, asks)
	assert.NotEmpty(t, trades)

	// Test benchmark
	throughput := engine.Benchmark(100)
	assert.Greater(t, throughput, float64(0))

	// Test close
	engine.Close()
}

func TestEngineCreation(t *testing.T) {
	testCases := []struct {
		name    string
		config  Config
		wantGPU bool
		wantErr bool
	}{
		{
			name: "Auto backend",
			config: Config{
				Backend:  BackendAuto,
				MaxBatch: 100,
			},
			wantGPU: false, // Depends on platform
			wantErr: false,
		},
		{
			name: "CPU backend",
			config: Config{
				Backend:  BackendCPU,
				MaxBatch: 100,
			},
			wantGPU: false,
			wantErr: false,
		},
		{
			name: "Metal backend",
			config: Config{
				Backend:  BackendMetal,
				MaxBatch: 100,
			},
			wantGPU: false, // Will fall back to CPU if not available
			wantErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			engine, err := NewEngine(tc.config)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, engine)

				// Test engine methods
				assert.NotEmpty(t, engine.Backend())
				assert.NotEmpty(t, engine.Device())

				// Test empty batch
				trades := engine.BatchMatch([]Order{}, []Order{})
				assert.Empty(t, trades)

				// Test with orders
				bids := []Order{{ID: 1, Side: 0, Price: 100, Size: 10}}
				asks := []Order{{ID: 2, Side: 1, Price: 100, Size: 10}}
				trades = engine.BatchMatch(bids, asks)
				assert.NotEmpty(t, trades)

				engine.Close()
			}
		})
	}
}

func TestBatchMatchEdgeCases(t *testing.T) {
	engine := &simpleEngine{
		backend:  BackendCPU,
		device:   "CPU",
		maxBatch: 100,
	}

	// Empty bids
	trades := engine.BatchMatch([]Order{}, []Order{{ID: 1, Side: 1, Price: 100, Size: 10}})
	assert.Empty(t, trades)

	// Empty asks
	trades = engine.BatchMatch([]Order{{ID: 1, Side: 0, Price: 100, Size: 10}}, []Order{})
	assert.Empty(t, trades)

	// No matching prices
	bids := []Order{{ID: 1, Side: 0, Price: 90, Size: 10}}
	asks := []Order{{ID: 2, Side: 1, Price: 100, Size: 10}}
	trades = engine.BatchMatch(bids, asks)
	assert.Empty(t, trades)

	// Partial fill
	bids = []Order{{ID: 1, Side: 0, Price: 100, Size: 15}}
	asks = []Order{{ID: 2, Side: 1, Price: 100, Size: 10}}
	trades = engine.BatchMatch(bids, asks)
	assert.Len(t, trades, 1)
	assert.Equal(t, float64(10), trades[0].Size)

	// Multiple matches
	bids = []Order{
		{ID: 1, Side: 0, Price: 100, Size: 10},
		{ID: 2, Side: 0, Price: 99, Size: 5},
	}
	asks = []Order{
		{ID: 3, Side: 1, Price: 99, Size: 8},
		{ID: 4, Side: 1, Price: 98, Size: 10},
	}
	trades = engine.BatchMatch(bids, asks)
	assert.NotEmpty(t, trades)
}

func BenchmarkSimpleEngine(b *testing.B) {
	engine := &simpleEngine{
		backend:  BackendCPU,
		device:   "CPU",
		maxBatch: 1000,
	}

	bids := make([]Order, 100)
	asks := make([]Order, 100)
	for i := 0; i < 100; i++ {
		bids[i] = Order{
			ID:    uint64(i),
			Side:  0,
			Price: 100 - float64(i%10),
			Size:  10,
		}
		asks[i] = Order{
			ID:    uint64(i + 100),
			Side:  1,
			Price: 100 + float64(i%10),
			Size:  10,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.BatchMatch(bids, asks)
	}
}

func BenchmarkLuxMLXEngine(b *testing.B) {
	engine := &LuxMLXEngine{
		backend:  BackendMetal,
		device:   "Metal",
		maxBatch: 1000,
	}

	bids := make([]Order, 100)
	asks := make([]Order, 100)
	for i := 0; i < 100; i++ {
		bids[i] = Order{
			ID:    uint64(i),
			Side:  0,
			Price: 100 - float64(i%10),
			Size:  10,
		}
		asks[i] = Order{
			ID:    uint64(i + 100),
			Side:  1,
			Price: 100 + float64(i%10),
			Size:  10,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.BatchMatch(bids, asks)
	}
}
