package engine

import (
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewMLXEngine(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	config := MLXConfig{
		MaxMarkets:    100,
		MarketDepth:   1000,
		UnifiedMemory: true,
		CacheGB:       8,
		BufferGB:      4,
	}

	engine, err := NewMLXEngine(config)
	require.NoError(t, err)
	assert.NotNil(t, engine)
	assert.True(t, engine.initialized)
	assert.Equal(t, uint32(100), engine.maxMarkets)
	assert.True(t, engine.unifiedMemory)
}

func TestNewMLXEngineNonApple(t *testing.T) {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Skip("This test is for non-Apple Silicon systems")
	}

	config := MLXConfig{MaxMarkets: 100}
	engine, err := NewMLXEngine(config)

	assert.Error(t, err)
	assert.Nil(t, engine)
	assert.Contains(t, err.Error(), "Apple Silicon")
}

func TestMLXEngineProcessBatch(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	orders := []Order{
		{ID: 1, Symbol: "BTC/USDC", Price: 50000, Size: 1.0, Side: 0, Type: 0},
		{ID: 2, Symbol: "BTC/USDC", Price: 50100, Size: 0.5, Side: 1, Type: 0},
		{ID: 3, Symbol: "ETH/USDC", Price: 3000, Size: 2.0, Side: 0, Type: 1},
	}

	result, err := engine.ProcessBatch(orders)
	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, uint64(3), result.OrdersProcessed)
	assert.GreaterOrEqual(t, result.TradesExecuted, uint64(0))
	assert.Greater(t, result.LatencyNanos, uint64(0))
}

func TestMLXEngineProcessBatchUninitialized(t *testing.T) {
	engine := &MLXEngine{initialized: false}

	orders := []Order{{ID: 1}}
	result, err := engine.ProcessBatch(orders)

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "not initialized")
}

func TestMLXEngineProcessBatchEmpty(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	result, err := engine.ProcessBatch([]Order{})
	require.NoError(t, err)
	assert.Equal(t, uint64(0), result.OrdersProcessed)
}

func TestMLXEngineGetStats(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10, UnifiedMemory: true})
	require.NoError(t, err)

	stats := engine.GetStats()
	assert.NotNil(t, stats)
	assert.Equal(t, uint64(0), stats.OrdersTotal)
	assert.Equal(t, uint64(0), stats.TradesTotal)
	assert.True(t, stats.UnifiedMemory)
	assert.Greater(t, stats.GPUCores, 0)
	assert.Greater(t, stats.NeuralCores, 0)
}

func TestMLXEngineLoadMarkets(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 5})
	require.NoError(t, err)

	markets := []Market{
		{Symbol: "BTC/USDC", BaseAsset: "BTC", QuoteAsset: "USDC", TickSize: 0.01, LotSize: 0.001, MaxDepth: 100},
		{Symbol: "ETH/USDC", BaseAsset: "ETH", QuoteAsset: "USDC", TickSize: 0.01, LotSize: 0.01, MaxDepth: 100},
	}

	err = engine.LoadMarkets(markets)
	assert.NoError(t, err)
	assert.Equal(t, uint32(2), engine.markets)
}

func TestMLXEngineLoadMarketsTooMany(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 2})
	require.NoError(t, err)

	markets := []Market{
		{Symbol: "BTC/USDC"},
		{Symbol: "ETH/USDC"},
		{Symbol: "SOL/USDC"}, // Exceeds limit
	}

	err = engine.LoadMarkets(markets)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too many markets")
}

func TestMLXEngineStatsAfterProcessing(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	// Process some orders
	orders := make([]Order, 100)
	for i := range orders {
		orders[i] = Order{ID: uint64(i), Symbol: "BTC/USDC", Price: 50000, Size: 1.0}
	}

	_, err = engine.ProcessBatch(orders)
	require.NoError(t, err)

	stats := engine.GetStats()
	assert.Equal(t, uint64(100), stats.OrdersTotal)
}

func TestOrderStruct(t *testing.T) {
	order := Order{
		ID:     123,
		Symbol: "BTC/USDC",
		Price:  50000.0,
		Size:   1.5,
		Side:   0, // Buy
		Type:   1, // Limit
	}

	assert.Equal(t, uint64(123), order.ID)
	assert.Equal(t, "BTC/USDC", order.Symbol)
	assert.Equal(t, 50000.0, order.Price)
	assert.Equal(t, 1.5, order.Size)
	assert.Equal(t, uint8(0), order.Side)
	assert.Equal(t, uint8(1), order.Type)
}

func TestMarketStruct(t *testing.T) {
	market := Market{
		Symbol:     "ETH/USDC",
		BaseAsset:  "ETH",
		QuoteAsset: "USDC",
		TickSize:   0.01,
		LotSize:    0.001,
		MaxDepth:   1000,
	}

	assert.Equal(t, "ETH/USDC", market.Symbol)
	assert.Equal(t, "ETH", market.BaseAsset)
	assert.Equal(t, "USDC", market.QuoteAsset)
	assert.Equal(t, 0.01, market.TickSize)
	assert.Equal(t, 0.001, market.LotSize)
	assert.Equal(t, uint32(1000), market.MaxDepth)
}

func TestBatchResultStruct(t *testing.T) {
	result := BatchResult{
		OrdersProcessed: 1000,
		TradesExecuted:  50,
		LatencyNanos:    597000,
		ThroughputMPS:   1.67,
	}

	assert.Equal(t, uint64(1000), result.OrdersProcessed)
	assert.Equal(t, uint64(50), result.TradesExecuted)
	assert.Equal(t, uint64(597000), result.LatencyNanos)
	assert.InDelta(t, 1.67, result.ThroughputMPS, 0.01)
}

func TestEngineStatsStruct(t *testing.T) {
	stats := EngineStats{
		OrdersTotal:   10000,
		TradesTotal:   500,
		Markets:       20,
		GPUCores:      40,
		NeuralCores:   16,
		UnifiedMemory: true,
	}

	assert.Equal(t, uint64(10000), stats.OrdersTotal)
	assert.Equal(t, uint64(500), stats.TradesTotal)
	assert.Equal(t, uint32(20), stats.Markets)
	assert.Equal(t, 40, stats.GPUCores)
	assert.Equal(t, 16, stats.NeuralCores)
	assert.True(t, stats.UnifiedMemory)
}

func TestDetectGPUCores(t *testing.T) {
	cores := detectGPUCores()
	assert.Equal(t, 76, cores) // Default M2 Ultra
}

func TestDetectNeuralCores(t *testing.T) {
	cores := detectNeuralCores()
	assert.Equal(t, 32, cores) // M2 Ultra
}

func TestHashSymbol(t *testing.T) {
	tests := []struct {
		symbol   string
		expected bool // Just verify it's deterministic and non-zero
	}{
		{"BTC/USDC", true},
		{"ETH/USDC", true},
		{"SOL/USDC", true},
		{"", true}, // Empty string should still hash
	}

	for _, tt := range tests {
		t.Run(tt.symbol, func(t *testing.T) {
			hash1 := hashSymbol(tt.symbol)
			hash2 := hashSymbol(tt.symbol)
			assert.Equal(t, hash1, hash2, "hash should be deterministic")
			if tt.symbol != "" {
				assert.NotEqual(t, uint64(0), hash1, "non-empty symbol should have non-zero hash")
			}
		})
	}

	// Different symbols should produce different hashes
	hash1 := hashSymbol("BTC/USDC")
	hash2 := hashSymbol("ETH/USDC")
	assert.NotEqual(t, hash1, hash2, "different symbols should have different hashes")
}

func TestHashSymbolEmptyString(t *testing.T) {
	hash := hashSymbol("")
	assert.Equal(t, uint64(5381), hash) // DJB2 hash initial value
}

func TestMLXEngineMemoryInfo(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	// Load some markets to affect memory calculation
	markets := []Market{
		{Symbol: "BTC/USDC"},
		{Symbol: "ETH/USDC"},
	}
	err = engine.LoadMarkets(markets)
	require.NoError(t, err)

	memInfo := engine.MemoryInfo()
	assert.NotNil(t, memInfo)
	assert.Greater(t, memInfo.Allocated, uint64(0))
	assert.Greater(t, memInfo.TotalAllocated, uint64(0))
	assert.Greater(t, memInfo.System, uint64(0))
	assert.Equal(t, uint64(512*1024*1024*1024), memInfo.UnifiedTotal) // 512GB
	assert.Equal(t, uint64(2*160*1024), memInfo.UnifiedUsed)          // 2 markets * 160KB
}

func TestMemoryStatsStruct(t *testing.T) {
	stats := MemoryStats{
		Allocated:      1024 * 1024,
		TotalAllocated: 10 * 1024 * 1024,
		System:         50 * 1024 * 1024,
		NumGC:          5,
		UnifiedTotal:   512 * 1024 * 1024 * 1024,
		UnifiedUsed:    320 * 1024,
	}

	assert.Equal(t, uint64(1024*1024), stats.Allocated)
	assert.Equal(t, uint64(10*1024*1024), stats.TotalAllocated)
	assert.Equal(t, uint64(50*1024*1024), stats.System)
	assert.Equal(t, uint32(5), stats.NumGC)
	assert.Equal(t, uint64(512*1024*1024*1024), stats.UnifiedTotal)
	assert.Equal(t, uint64(320*1024), stats.UnifiedUsed)
}

func TestMLXEngineBenchmark(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	// Run a short benchmark (10ms)
	result := engine.Benchmark(10 * time.Millisecond)

	assert.NotNil(t, result)
	assert.Greater(t, result.Duration, time.Duration(0))
	assert.Greater(t, result.OrdersProcessed, uint64(0))
	assert.Greater(t, result.OrdersPerSecond, float64(0))
	assert.Greater(t, result.Batches, 0)
}

func TestBenchmarkResultStruct(t *testing.T) {
	result := BenchmarkResult{
		Duration:        time.Second,
		OrdersProcessed: 1000000,
		OrdersPerSecond: 1000000.0,
		AvgLatencyNanos: 597,
		Batches:         100,
	}

	assert.Equal(t, time.Second, result.Duration)
	assert.Equal(t, uint64(1000000), result.OrdersProcessed)
	assert.Equal(t, 1000000.0, result.OrdersPerSecond)
	assert.Equal(t, uint64(597), result.AvgLatencyNanos)
	assert.Equal(t, 100, result.Batches)
}

func TestMLXEngineLoadMarketsEmpty(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	err = engine.LoadMarkets([]Market{})
	assert.NoError(t, err)
	assert.Equal(t, uint32(0), engine.markets)
}

func TestMLXEngineProcessBatchLarge(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	// Process a large batch
	orders := make([]Order, 10000)
	for i := range orders {
		orders[i] = Order{
			ID:     uint64(i),
			Symbol: "BTC/USDC",
			Price:  50000 + float64(i%100),
			Size:   float64(1 + i%10),
			Side:   uint8(i % 2),
			Type:   uint8(i % 2),
		}
	}

	result, err := engine.ProcessBatch(orders)
	require.NoError(t, err)
	assert.Equal(t, uint64(10000), result.OrdersProcessed)
	assert.Equal(t, uint64(1000), result.TradesExecuted) // 10% fill rate
	assert.Greater(t, result.ThroughputMPS, float64(0))
}

func TestMLXEngineMultipleBatches(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	orders := make([]Order, 100)
	for i := range orders {
		orders[i] = Order{ID: uint64(i), Symbol: "BTC/USDC", Price: 50000, Size: 1.0}
	}

	// Process multiple batches
	for batch := 0; batch < 5; batch++ {
		_, err = engine.ProcessBatch(orders)
		require.NoError(t, err)
	}

	stats := engine.GetStats()
	assert.Equal(t, uint64(500), stats.OrdersTotal) // 5 batches * 100 orders
}

func TestMLXConfigDefaults(t *testing.T) {
	config := MLXConfig{}

	assert.Equal(t, uint32(0), config.MaxMarkets)
	assert.Equal(t, uint32(0), config.MarketDepth)
	assert.False(t, config.UnifiedMemory)
	assert.Equal(t, uint32(0), config.CacheGB)
	assert.Equal(t, uint32(0), config.BufferGB)
}

func TestMLXEngineInitialState(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{
		MaxMarkets:    50,
		MarketDepth:   500,
		UnifiedMemory: true,
		CacheGB:       16,
		BufferGB:      8,
	})
	require.NoError(t, err)

	assert.True(t, engine.initialized)
	assert.Equal(t, uint32(50), engine.maxMarkets)
	assert.True(t, engine.unifiedMemory)
	assert.Equal(t, uint64(0), engine.ordersTotal)
	assert.Equal(t, uint64(0), engine.tradesTotal)
	assert.Equal(t, uint32(0), engine.markets)
	assert.Greater(t, engine.gpuCores, 0)
	assert.Greater(t, engine.neuralCores, 0)
}

func TestOrderSide(t *testing.T) {
	buyOrder := Order{ID: 1, Side: 0}
	sellOrder := Order{ID: 2, Side: 1}

	assert.Equal(t, uint8(0), buyOrder.Side)
	assert.Equal(t, uint8(1), sellOrder.Side)
}

func TestOrderType(t *testing.T) {
	limitOrder := Order{ID: 1, Type: 0}
	marketOrder := Order{ID: 2, Type: 1}

	assert.Equal(t, uint8(0), limitOrder.Type)
	assert.Equal(t, uint8(1), marketOrder.Type)
}

func TestMLXEngineMemoryInfoNoMarkets(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 10})
	require.NoError(t, err)

	memInfo := engine.MemoryInfo()
	assert.Equal(t, uint64(0), memInfo.UnifiedUsed) // No markets loaded
}

func TestHashSymbolDifferentLengths(t *testing.T) {
	short := hashSymbol("A")
	medium := hashSymbol("ABC")
	long := hashSymbol("ABCDEFGHIJ")

	// All should be different
	assert.NotEqual(t, short, medium)
	assert.NotEqual(t, medium, long)
	assert.NotEqual(t, short, long)
}

func TestMLXEngineLoadMarketsExactLimit(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, err := NewMLXEngine(MLXConfig{MaxMarkets: 3})
	require.NoError(t, err)

	markets := []Market{
		{Symbol: "BTC/USDC"},
		{Symbol: "ETH/USDC"},
		{Symbol: "SOL/USDC"},
	}

	err = engine.LoadMarkets(markets)
	assert.NoError(t, err) // Exactly at limit should work
	assert.Equal(t, uint32(3), engine.markets)
}

func BenchmarkProcessBatch(b *testing.B) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		b.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, _ := NewMLXEngine(MLXConfig{MaxMarkets: 10})

	orders := make([]Order, 1000)
	for i := range orders {
		orders[i] = Order{ID: uint64(i), Symbol: "BTC/USDC", Price: 50000, Size: 1.0}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.ProcessBatch(orders)
	}
}

func BenchmarkHashSymbol(b *testing.B) {
	symbol := "BTC/USDC"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hashSymbol(symbol)
	}
}

func BenchmarkLoadMarkets(b *testing.B) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		b.Skip("MLX engine requires Apple Silicon Mac")
	}

	engine, _ := NewMLXEngine(MLXConfig{MaxMarkets: 100})

	markets := make([]Market, 50)
	for i := range markets {
		markets[i] = Market{Symbol: "TEST" + string(rune('A'+i))}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.LoadMarkets(markets)
	}
}
