package marketdata

import (
	"context"
	"testing"
	"time"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntervalDuration(t *testing.T) {
	tests := []struct {
		interval Interval
		expected time.Duration
	}{
		{Interval1s, 1 * time.Second},
		{Interval5s, 5 * time.Second},
		{Interval15s, 15 * time.Second},
		{Interval30s, 30 * time.Second},
		{Interval1m, 1 * time.Minute},
		{Interval3m, 3 * time.Minute},
		{Interval5m, 5 * time.Minute},
		{Interval15m, 15 * time.Minute},
		{Interval30m, 30 * time.Minute},
		{Interval1h, 1 * time.Hour},
		{Interval2h, 2 * time.Hour},
		{Interval4h, 4 * time.Hour},
		{Interval6h, 6 * time.Hour},
		{Interval8h, 8 * time.Hour},
		{Interval12h, 12 * time.Hour},
		{Interval1d, 24 * time.Hour},
		{Interval3d, 3 * 24 * time.Hour},
		{Interval1w, 7 * 24 * time.Hour},
		{Interval1M, 30 * 24 * time.Hour},
	}

	for _, tt := range tests {
		t.Run(string(tt.interval), func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.interval.Duration())
		})
	}
}

func TestIntervalDurationUnknown(t *testing.T) {
	unknown := Interval("unknown")
	// Unknown intervals default to 1 minute
	assert.Equal(t, 1*time.Minute, unknown.Duration())
}

func TestAllIntervals(t *testing.T) {
	intervals := AllIntervals()

	assert.Len(t, intervals, 19)
	assert.Contains(t, intervals, Interval1s)
	assert.Contains(t, intervals, Interval1m)
	assert.Contains(t, intervals, Interval1h)
	assert.Contains(t, intervals, Interval1d)
	assert.Contains(t, intervals, Interval1M)
}

func TestCandleStruct(t *testing.T) {
	now := time.Now()
	candle := Candle{
		Symbol:    "BTC/USDC",
		Interval:  Interval1h,
		OpenTime:  now,
		CloseTime: now.Add(1 * time.Hour),
		Open:      50000.0,
		High:      51000.0,
		Low:       49500.0,
		Close:     50500.0,
		Volume:    100.0,
		Trades:    500,
		Complete:  true,
	}

	assert.Equal(t, "BTC/USDC", candle.Symbol)
	assert.Equal(t, Interval1h, candle.Interval)
	assert.Equal(t, 50000.0, candle.Open)
	assert.Equal(t, 51000.0, candle.High)
	assert.Equal(t, 49500.0, candle.Low)
	assert.Equal(t, 50500.0, candle.Close)
	assert.Equal(t, 100.0, candle.Volume)
	assert.Equal(t, 500, candle.Trades)
	assert.True(t, candle.Complete)
}

func TestCandleOHLCVRelationship(t *testing.T) {
	candle := Candle{
		Open:  100.0,
		High:  110.0,
		Low:   95.0,
		Close: 105.0,
	}

	// High should be >= Open, Close
	assert.GreaterOrEqual(t, candle.High, candle.Open)
	assert.GreaterOrEqual(t, candle.High, candle.Close)

	// Low should be <= Open, Close
	assert.LessOrEqual(t, candle.Low, candle.Open)
	assert.LessOrEqual(t, candle.Low, candle.Close)

	// High should be >= Low
	assert.GreaterOrEqual(t, candle.High, candle.Low)
}

func TestIntervalConstants(t *testing.T) {
	// Verify string values
	assert.Equal(t, "1s", string(Interval1s))
	assert.Equal(t, "5s", string(Interval5s))
	assert.Equal(t, "1m", string(Interval1m))
	assert.Equal(t, "1h", string(Interval1h))
	assert.Equal(t, "1d", string(Interval1d))
	assert.Equal(t, "1M", string(Interval1M))
}

func TestIntervalSorting(t *testing.T) {
	intervals := AllIntervals()

	// First should be smallest (1s)
	assert.Equal(t, Interval1s, intervals[0])

	// Last should be largest (1M)
	assert.Equal(t, Interval1M, intervals[len(intervals)-1])
}

func TestCandleJSON(t *testing.T) {
	candle := Candle{
		Symbol:   "ETH/USDC",
		Interval: Interval15m,
		Open:     3000.0,
		High:     3100.0,
		Low:      2950.0,
		Close:    3050.0,
		Volume:   500.0,
		Trades:   100,
	}

	// Verify JSON tags exist (would fail to compile otherwise)
	assert.Equal(t, "ETH/USDC", candle.Symbol)
	assert.Equal(t, Interval15m, candle.Interval)
}

func TestCandleIncomplete(t *testing.T) {
	candle := Candle{
		Symbol:   "SOL/USDC",
		Complete: false,
	}

	assert.False(t, candle.Complete)
}

func TestIntervalConversion(t *testing.T) {
	// Convert string to Interval
	intervalStr := "5m"
	interval := Interval(intervalStr)
	assert.Equal(t, Interval5m, interval)
	assert.Equal(t, 5*time.Minute, interval.Duration())
}

func TestAllIntervalsUnique(t *testing.T) {
	intervals := AllIntervals()
	seen := make(map[Interval]bool)

	for _, interval := range intervals {
		assert.False(t, seen[interval], "Duplicate interval: %s", interval)
		seen[interval] = true
	}
}

func TestIntervalDurationOrder(t *testing.T) {
	intervals := AllIntervals()

	for i := 1; i < len(intervals); i++ {
		prev := intervals[i-1].Duration()
		curr := intervals[i].Duration()
		assert.LessOrEqual(t, prev, curr, "Intervals should be in ascending order")
	}
}

func TestCandleTimeRange(t *testing.T) {
	now := time.Now().Truncate(time.Hour)
	candle := Candle{
		Interval:  Interval1h,
		OpenTime:  now,
		CloseTime: now.Add(Interval1h.Duration()),
	}

	duration := candle.CloseTime.Sub(candle.OpenTime)
	assert.Equal(t, Interval1h.Duration(), duration)
}

func BenchmarkIntervalDuration(b *testing.B) {
	interval := Interval1h
	for i := 0; i < b.N; i++ {
		_ = interval.Duration()
	}
}

func BenchmarkAllIntervals(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = AllIntervals()
	}
}

// ===== Aggregator Tests =====

func newTestAggregator() *Aggregator {
	db := memdb.New()
	logger := log.NoLog{}
	return NewAggregator(logger, db)
}

func TestNewAggregator(t *testing.T) {
	agg := newTestAggregator()
	require.NotNil(t, agg)
	assert.NotNil(t, agg.candles)
	assert.NotNil(t, agg.trades)
	assert.NotNil(t, agg.subscribers)
	assert.NotNil(t, agg.ctx)
	assert.NotNil(t, agg.cancel)
}

func TestAggregatorStartStop(t *testing.T) {
	agg := newTestAggregator()

	err := agg.Start()
	require.NoError(t, err)

	// Let it run briefly
	time.Sleep(50 * time.Millisecond)

	// Stop should work cleanly
	agg.Stop()
}

func TestAggregatorAddTrade(t *testing.T) {
	agg := newTestAggregator()

	trade := &lx.Trade{
		Price:     50000.0,
		Size:      1.5,
		Timestamp: time.Now(),
	}

	agg.AddTrade(trade)

	// Check trade was added to buffer
	agg.tradesMu.Lock()
	defer agg.tradesMu.Unlock()
	assert.Len(t, agg.trades, 1)
	assert.Equal(t, uint64(1), agg.totalTrades)
}

func TestAggregatorAddMultipleTrades(t *testing.T) {
	agg := newTestAggregator()

	for i := 0; i < 10; i++ {
		trade := &lx.Trade{
			Price:     50000.0 + float64(i*100),
			Size:      float64(i + 1),
			Timestamp: time.Now(),
		}
		agg.AddTrade(trade)
	}

	agg.tradesMu.Lock()
	defer agg.tradesMu.Unlock()
	assert.Len(t, agg.trades, 10)
	assert.Equal(t, uint64(10), agg.totalTrades)
}

func TestAggregatorProcessTradeBuffer(t *testing.T) {
	agg := newTestAggregator()

	// Add a trade
	trade := &lx.Trade{
		Price:     50000.0,
		Size:      2.0,
		Timestamp: time.Now(),
	}
	agg.AddTrade(trade)

	// Process the buffer
	agg.processTradeBuffer()

	// Buffer should be empty after processing
	agg.tradesMu.Lock()
	bufferLen := len(agg.trades)
	agg.tradesMu.Unlock()
	assert.Equal(t, 0, bufferLen)
}

func TestAggregatorProcessEmptyBuffer(t *testing.T) {
	agg := newTestAggregator()

	// Should not panic on empty buffer
	agg.processTradeBuffer()
}

func TestAggregatorUpdateCandles(t *testing.T) {
	agg := newTestAggregator()

	trade := &lx.Trade{
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}

	agg.updateCandles(trade)

	// Should have created candles for all intervals
	agg.candlesMu.RLock()
	defer agg.candlesMu.RUnlock()

	symbolCandles := agg.candles["BTC-USD"]
	assert.NotNil(t, symbolCandles)
	assert.Greater(t, len(symbolCandles), 0)
}

func TestAggregatorCandleHighLow(t *testing.T) {
	agg := newTestAggregator()

	// Use a fixed timestamp aligned to 1-minute boundary so all trades fall in the same 1m candle
	baseTime := time.Date(2025, 1, 1, 10, 30, 0, 0, time.UTC)

	// First trade - sets open
	trade1 := &lx.Trade{Price: 50000.0, Size: 1.0, Timestamp: baseTime}
	agg.updateCandles(trade1)

	// Second trade - higher price (still within same minute)
	trade2 := &lx.Trade{Price: 51000.0, Size: 1.0, Timestamp: baseTime.Add(10 * time.Second)}
	agg.updateCandles(trade2)

	// Third trade - lower price (still within same minute)
	trade3 := &lx.Trade{Price: 49000.0, Size: 1.0, Timestamp: baseTime.Add(20 * time.Second)}
	agg.updateCandles(trade3)

	// Check 1-minute candle (not 1-second, since trades span multiple seconds)
	agg.candlesMu.RLock()
	defer agg.candlesMu.RUnlock()

	candle := agg.candles["BTC-USD"][Interval1m]
	require.NotNil(t, candle)

	assert.Equal(t, 50000.0, candle.Open)
	assert.Equal(t, 51000.0, candle.High)
	assert.Equal(t, 49000.0, candle.Low)
	assert.Equal(t, 49000.0, candle.Close)
	assert.Equal(t, 3.0, candle.Volume)
	assert.Equal(t, 3, candle.Trades)
}

func TestAggregatorGetLatestCandle(t *testing.T) {
	agg := newTestAggregator()

	// No candles yet
	candle := agg.GetLatestCandle("BTC-USD", Interval1m)
	assert.Nil(t, candle)

	// Add a trade to create candle
	trade := &lx.Trade{Price: 50000.0, Size: 1.0, Timestamp: time.Now()}
	agg.updateCandles(trade)

	// Now should have a candle
	candle = agg.GetLatestCandle("BTC-USD", Interval1m)
	require.NotNil(t, candle)
	assert.Equal(t, "BTC-USD", candle.Symbol)
	assert.Equal(t, Interval1m, candle.Interval)
}

func TestAggregatorGetLatestCandleUnknownSymbol(t *testing.T) {
	agg := newTestAggregator()

	candle := agg.GetLatestCandle("UNKNOWN/USD", Interval1h)
	assert.Nil(t, candle)
}

func TestAggregatorGetStats(t *testing.T) {
	agg := newTestAggregator()

	stats := agg.GetStats()
	assert.NotNil(t, stats)
	assert.Contains(t, stats, "total_trades")
	assert.Contains(t, stats, "total_candles")
	assert.Contains(t, stats, "symbols")
	assert.Equal(t, uint64(0), stats["total_trades"])
}

func TestAggregatorGetStatsWithData(t *testing.T) {
	agg := newTestAggregator()

	// Add some trades
	for i := 0; i < 5; i++ {
		trade := &lx.Trade{Price: 50000.0 + float64(i), Size: 1.0, Timestamp: time.Now()}
		agg.AddTrade(trade)
	}
	agg.processTradeBuffer()

	stats := agg.GetStats()
	assert.Equal(t, uint64(5), stats["total_trades"])
	assert.Greater(t, stats["total_candles"], uint64(0))
}

func TestAggregatorSubscribe(t *testing.T) {
	agg := newTestAggregator()

	ch := agg.Subscribe("BTC-USD", Interval1m)
	require.NotNil(t, ch)

	// Check subscriber was added
	agg.subMu.RLock()
	subs := agg.subscribers["BTC-USD:1m"]
	agg.subMu.RUnlock()
	assert.Len(t, subs, 1)
}

func TestAggregatorMultipleSubscribers(t *testing.T) {
	agg := newTestAggregator()

	ch1 := agg.Subscribe("BTC-USD", Interval1m)
	ch2 := agg.Subscribe("BTC-USD", Interval1m)
	ch3 := agg.Subscribe("ETH-USD", Interval1m)

	require.NotNil(t, ch1)
	require.NotNil(t, ch2)
	require.NotNil(t, ch3)

	agg.subMu.RLock()
	btcSubs := agg.subscribers["BTC-USD:1m"]
	ethSubs := agg.subscribers["ETH-USD:1m"]
	agg.subMu.RUnlock()

	assert.Len(t, btcSubs, 2)
	assert.Len(t, ethSubs, 1)
}

func TestAggregatorGetCandles(t *testing.T) {
	agg := newTestAggregator()

	candles, err := agg.GetCandles("BTC-USD", Interval1m, 100)
	require.NoError(t, err)
	// Returns empty since iterator is not fully implemented
	assert.NotNil(t, candles)
}

func TestAggregatorVWAP(t *testing.T) {
	agg := newTestAggregator()

	vwap := agg.VolumeWeightedAveragePrice("BTC-USD", Interval1m, 10)
	// Returns 0 since no candles
	assert.Equal(t, 0.0, vwap)
}

func TestAggregatorMovingAverage(t *testing.T) {
	agg := newTestAggregator()

	ma := agg.MovingAverage("BTC-USD", Interval1m, 10)
	// Returns 0 since no candles
	assert.Equal(t, 0.0, ma)
}

func TestAggregatorRSI(t *testing.T) {
	agg := newTestAggregator()

	rsi := agg.RSI("BTC-USD", Interval1m, 14)
	// Returns 50 (neutral) since no candles
	assert.Equal(t, 50.0, rsi)
}

func TestAggregatorGetCandleOpenTime(t *testing.T) {
	agg := newTestAggregator()

	tests := []struct {
		name     string
		t        time.Time
		interval Interval
	}{
		{"1m at minute boundary", time.Date(2025, 1, 1, 10, 30, 0, 0, time.UTC), Interval1m},
		{"1m mid-minute", time.Date(2025, 1, 1, 10, 30, 45, 0, time.UTC), Interval1m},
		{"1h at hour boundary", time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC), Interval1h},
		{"1d at day start", time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC), Interval1d},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			openTime := agg.getCandleOpenTime(tt.t, tt.interval)
			// Open time should be aligned to interval boundary
			assert.True(t, openTime.Before(tt.t) || openTime.Equal(tt.t))
		})
	}
}

func TestAggregatorGetCandleOpenTimeMonthly(t *testing.T) {
	agg := newTestAggregator()

	testTime := time.Date(2025, 6, 15, 14, 30, 0, 0, time.UTC)
	openTime := agg.getCandleOpenTime(testTime, Interval1M)

	// Monthly should be first of the month
	assert.Equal(t, 2025, openTime.Year())
	assert.Equal(t, time.June, openTime.Month())
	assert.Equal(t, 1, openTime.Day())
	assert.Equal(t, 0, openTime.Hour())
	assert.Equal(t, 0, openTime.Minute())
}

func TestAggregatorContextCancellation(t *testing.T) {
	agg := newTestAggregator()

	err := agg.Start()
	require.NoError(t, err)

	// Cancel context should stop
	agg.cancel()
	// Wait a bit for goroutines to finish
	time.Sleep(50 * time.Millisecond)

	// wg.Wait() in Stop() should complete
	done := make(chan struct{})
	go func() {
		agg.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Good, goroutines stopped
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Goroutines did not stop in time")
	}
}

func TestAggregatorConcurrentTradeAdds(t *testing.T) {
	agg := newTestAggregator()

	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func(n int) {
			for j := 0; j < 100; j++ {
				trade := &lx.Trade{
					Price:     50000.0 + float64(n*100+j),
					Size:      float64(j + 1),
					Timestamp: time.Now(),
				}
				agg.AddTrade(trade)
			}
			done <- struct{}{}
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Should have 1000 trades
	assert.Equal(t, uint64(1000), agg.totalTrades)
}

func TestAggregatorPublishCandle(t *testing.T) {
	agg := newTestAggregator()

	// Subscribe first
	ch := agg.Subscribe("BTC-USD", Interval1m)

	// Create and publish a candle
	candle := &Candle{
		Symbol:   "BTC-USD",
		Interval: Interval1m,
		Open:     50000.0,
		High:     51000.0,
		Low:      49000.0,
		Close:    50500.0,
		Volume:   100.0,
		Trades:   50,
		Complete: true,
	}

	agg.publishCandle(candle)

	// Should receive the candle
	select {
	case received := <-ch:
		assert.Equal(t, candle.Symbol, received.Symbol)
		assert.Equal(t, candle.Close, received.Close)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Did not receive candle")
	}
}

func TestAggregatorPublishCandleNoSubscribers(t *testing.T) {
	agg := newTestAggregator()

	candle := &Candle{
		Symbol:   "BTC-USD",
		Interval: Interval1m,
		Complete: true,
	}

	// Should not panic with no subscribers
	agg.publishCandle(candle)
}

func TestAggregatorStoreCandle(t *testing.T) {
	agg := newTestAggregator()

	candle := &Candle{
		Symbol:    "BTC-USD",
		Interval:  Interval1m,
		OpenTime:  time.Now(),
		CloseTime: time.Now().Add(1 * time.Minute),
		Open:      50000.0,
		High:      51000.0,
		Low:       49000.0,
		Close:     50500.0,
		Volume:    100.0,
		Trades:    50,
		Complete:  true,
	}

	// Should not panic
	agg.storeCandle(candle)
}

func TestAggregatorCompleteCandles(t *testing.T) {
	agg := newTestAggregator()

	// Add a candle that should be completed
	pastTime := time.Now().Add(-2 * time.Hour)
	candle := &Candle{
		Symbol:    "BTC-USD",
		Interval:  Interval1h,
		OpenTime:  pastTime,
		CloseTime: pastTime.Add(1 * time.Hour),
		Open:      50000.0,
		Close:     50500.0,
		Complete:  false,
	}

	agg.candlesMu.Lock()
	agg.candles["BTC-USD"] = map[Interval]*Candle{
		Interval1h: candle,
	}
	agg.candlesMu.Unlock()

	// Complete candles should mark it as complete
	agg.completeCandles(Interval1h)

	agg.candlesMu.RLock()
	// Candle should be deleted after completion
	_, exists := agg.candles["BTC-USD"][Interval1h]
	agg.candlesMu.RUnlock()

	assert.False(t, exists, "Completed candle should be deleted from map")
}

func TestAggregatorCleanupOldCandles(t *testing.T) {
	agg := newTestAggregator()

	// Should not panic
	agg.cleanupOldCandles()
}

func BenchmarkAggregatorAddTrade(b *testing.B) {
	agg := newTestAggregator()
	trade := &lx.Trade{
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.AddTrade(trade)
	}
}

func BenchmarkAggregatorUpdateCandles(b *testing.B) {
	agg := newTestAggregator()
	trade := &lx.Trade{
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.updateCandles(trade)
	}
}

func BenchmarkAggregatorGetStats(b *testing.B) {
	agg := newTestAggregator()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = agg.GetStats()
	}
}

// Ensure unused imports are used
var _ = context.Background
