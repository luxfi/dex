// Package marketdata provides market data aggregation and OHLCV generation
package marketdata

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
)

// Aggregator collects trades and generates OHLCV data
type Aggregator struct {
	logger log.Logger
	db     database.Database

	// Candle storage by symbol and interval
	candles   map[string]map[Interval]*Candle
	candlesMu sync.RWMutex

	// Trade buffer
	trades   []*lx.Trade
	tradesMu sync.Mutex

	// Subscribers
	subscribers map[string][]chan *Candle
	subMu       sync.RWMutex

	// Stats
	totalTrades  uint64
	totalCandles uint64

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Candle represents OHLCV data
type Candle struct {
	Symbol    string    `json:"symbol"`
	Interval  Interval  `json:"interval"`
	OpenTime  time.Time `json:"openTime"`
	CloseTime time.Time `json:"closeTime"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Trades    int       `json:"trades"`
	Complete  bool      `json:"complete"`
}

// Interval represents a time interval for candles
type Interval string

const (
	Interval1s  Interval = "1s"
	Interval5s  Interval = "5s"
	Interval15s Interval = "15s"
	Interval30s Interval = "30s"
	Interval1m  Interval = "1m"
	Interval3m  Interval = "3m"
	Interval5m  Interval = "5m"
	Interval15m Interval = "15m"
	Interval30m Interval = "30m"
	Interval1h  Interval = "1h"
	Interval2h  Interval = "2h"
	Interval4h  Interval = "4h"
	Interval6h  Interval = "6h"
	Interval8h  Interval = "8h"
	Interval12h Interval = "12h"
	Interval1d  Interval = "1d"
	Interval3d  Interval = "3d"
	Interval1w  Interval = "1w"
	Interval1M  Interval = "1M"
)

// Duration returns the time.Duration for an interval
func (i Interval) Duration() time.Duration {
	switch i {
	case Interval1s:
		return 1 * time.Second
	case Interval5s:
		return 5 * time.Second
	case Interval15s:
		return 15 * time.Second
	case Interval30s:
		return 30 * time.Second
	case Interval1m:
		return 1 * time.Minute
	case Interval3m:
		return 3 * time.Minute
	case Interval5m:
		return 5 * time.Minute
	case Interval15m:
		return 15 * time.Minute
	case Interval30m:
		return 30 * time.Minute
	case Interval1h:
		return 1 * time.Hour
	case Interval2h:
		return 2 * time.Hour
	case Interval4h:
		return 4 * time.Hour
	case Interval6h:
		return 6 * time.Hour
	case Interval8h:
		return 8 * time.Hour
	case Interval12h:
		return 12 * time.Hour
	case Interval1d:
		return 24 * time.Hour
	case Interval3d:
		return 3 * 24 * time.Hour
	case Interval1w:
		return 7 * 24 * time.Hour
	case Interval1M:
		return 30 * 24 * time.Hour
	default:
		return 1 * time.Minute
	}
}

// AllIntervals returns all supported intervals
func AllIntervals() []Interval {
	return []Interval{
		Interval1s, Interval5s, Interval15s, Interval30s,
		Interval1m, Interval3m, Interval5m, Interval15m, Interval30m,
		Interval1h, Interval2h, Interval4h, Interval6h, Interval8h, Interval12h,
		Interval1d, Interval3d, Interval1w, Interval1M,
	}
}

// NewAggregator creates a new market data aggregator
func NewAggregator(logger log.Logger, db database.Database) *Aggregator {
	ctx, cancel := context.WithCancel(context.Background())

	return &Aggregator{
		logger:      logger,
		db:          db,
		candles:     make(map[string]map[Interval]*Candle),
		trades:      make([]*lx.Trade, 0, 1000),
		subscribers: make(map[string][]chan *Candle),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start begins the aggregator
func (a *Aggregator) Start() error {
	// Start candle generator for each interval
	for _, interval := range AllIntervals() {
		a.wg.Add(1)
		go a.generateCandles(interval)
	}

	// Start trade processor
	a.wg.Add(1)
	go a.processTrades()

	// Start cleanup routine
	a.wg.Add(1)
	go a.cleanup()

	a.logger.Info("Market data aggregator started")
	return nil
}

// Stop shuts down the aggregator
func (a *Aggregator) Stop() {
	a.logger.Info("Stopping market data aggregator")
	a.cancel()
	a.wg.Wait()
}

// AddTrade adds a trade to be processed
func (a *Aggregator) AddTrade(trade *lx.Trade) {
	a.tradesMu.Lock()
	a.trades = append(a.trades, trade)
	a.totalTrades++
	a.tradesMu.Unlock()
}

// processTrades processes buffered trades
func (a *Aggregator) processTrades() {
	defer a.wg.Done()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.processTradeBuffer()
		}
	}
}

// processTradeBuffer processes the trade buffer
func (a *Aggregator) processTradeBuffer() {
	a.tradesMu.Lock()
	if len(a.trades) == 0 {
		a.tradesMu.Unlock()
		return
	}

	trades := a.trades
	a.trades = make([]*lx.Trade, 0, 1000)
	a.tradesMu.Unlock()

	// Update candles for each trade
	for _, trade := range trades {
		a.updateCandles(trade)
	}
}

// updateCandles updates all interval candles with a trade
func (a *Aggregator) updateCandles(trade *lx.Trade) {
	symbol := "BTC-USD" // TODO: Get from trade
	tradeTime := trade.Timestamp

	a.candlesMu.Lock()
	defer a.candlesMu.Unlock()

	// Ensure symbol map exists
	if a.candles[symbol] == nil {
		a.candles[symbol] = make(map[Interval]*Candle)
	}

	// Update candle for each interval
	for _, interval := range AllIntervals() {
		candle := a.candles[symbol][interval]

		// Get candle period
		openTime := a.getCandleOpenTime(tradeTime, interval)
		closeTime := openTime.Add(interval.Duration())

		// Create new candle if needed
		if candle == nil || candle.OpenTime != openTime {
			// Complete previous candle if exists
			if candle != nil && !candle.Complete {
				candle.Complete = true
				a.publishCandle(candle)
				a.storeCandle(candle)
			}

			// Create new candle
			candle = &Candle{
				Symbol:    symbol,
				Interval:  interval,
				OpenTime:  openTime,
				CloseTime: closeTime,
				Open:      trade.Price,
				High:      trade.Price,
				Low:       trade.Price,
				Close:     trade.Price,
				Volume:    trade.Size,
				Trades:    1,
				Complete:  false,
			}
			a.candles[symbol][interval] = candle
			a.totalCandles++
		} else {
			// Update existing candle
			candle.High = math.Max(candle.High, trade.Price)
			candle.Low = math.Min(candle.Low, trade.Price)
			candle.Close = trade.Price
			candle.Volume += trade.Size
			candle.Trades++
		}
	}
}

// generateCandles generates candles at regular intervals
func (a *Aggregator) generateCandles(interval Interval) {
	defer a.wg.Done()

	ticker := time.NewTicker(interval.Duration())
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.completeCandles(interval)
		}
	}
}

// completeCandles marks candles as complete for an interval
func (a *Aggregator) completeCandles(interval Interval) {
	a.candlesMu.Lock()
	defer a.candlesMu.Unlock()

	now := time.Now()

	for _, intervalCandles := range a.candles {
		candle := intervalCandles[interval]
		if candle != nil && !candle.Complete && now.After(candle.CloseTime) {
			candle.Complete = true
			a.publishCandle(candle)
			a.storeCandle(candle)

			// Clear the candle
			delete(intervalCandles, interval)
		}
	}
}

// getCandleOpenTime returns the open time for a candle
func (a *Aggregator) getCandleOpenTime(t time.Time, interval Interval) time.Time {
	duration := interval.Duration()

	// Special handling for monthly interval
	if interval == Interval1M {
		return time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, t.Location())
	}

	// Align to interval boundary
	unix := t.Unix()
	intervalSeconds := int64(duration.Seconds())
	aligned := (unix / intervalSeconds) * intervalSeconds

	return time.Unix(aligned, 0)
}

// publishCandle publishes a completed candle to subscribers
func (a *Aggregator) publishCandle(candle *Candle) {
	key := fmt.Sprintf("%s:%s", candle.Symbol, candle.Interval)

	a.subMu.RLock()
	subscribers := a.subscribers[key]
	a.subMu.RUnlock()

	if len(subscribers) == 0 {
		return
	}

	// Send to all subscribers
	for _, ch := range subscribers {
		select {
		case ch <- candle:
		default:
			// Subscriber is not ready, skip
		}
	}
}

// storeCandle stores a candle in the database
func (a *Aggregator) storeCandle(candle *Candle) {
	key := fmt.Sprintf("candle:%s:%s:%d", candle.Symbol, candle.Interval, candle.OpenTime.Unix())

	value, err := json.Marshal(candle)
	if err != nil {
		a.logger.Error("Failed to marshal candle", "error", err)
		return
	}

	if err := a.db.Put([]byte(key), value); err != nil {
		a.logger.Error("Failed to store candle", "error", err)
	}
}

// Subscribe subscribes to candle updates
func (a *Aggregator) Subscribe(symbol string, interval Interval) <-chan *Candle {
	key := fmt.Sprintf("%s:%s", symbol, interval)
	ch := make(chan *Candle, 100)

	a.subMu.Lock()
	a.subscribers[key] = append(a.subscribers[key], ch)
	a.subMu.Unlock()

	return ch
}

// GetCandles retrieves historical candles
func (a *Aggregator) GetCandles(symbol string, interval Interval, limit int) ([]*Candle, error) {
	candles := make([]*Candle, 0, limit)

	// Calculate time range
	now := time.Now()
	duration := interval.Duration()
	startTime := now.Add(-duration * time.Duration(limit))

	// Iterate through database
	prefix := fmt.Sprintf("candle:%s:%s:", symbol, interval)
	// TODO: Implement proper iterator for database.Database interface
	// For now, return empty candles
	// TODO: Implement when database iterator is available
	_ = prefix
	_ = startTime
	return candles, nil

	/*
		iter := a.db.NewIterator([]byte(prefix), nil)
		defer iter.Release()

		for iter.Next() {
			var candle Candle
			if err := json.Unmarshal(iter.Value(), &candle); err != nil {
				continue
			}

			if candle.OpenTime.After(startTime) {
				candles = append(candles, &candle)
			}

			if len(candles) >= limit {
				break
			}
		}

		return candles, nil
	*/
}

// GetLatestCandle returns the most recent candle
func (a *Aggregator) GetLatestCandle(symbol string, interval Interval) *Candle {
	a.candlesMu.RLock()
	defer a.candlesMu.RUnlock()

	if symbolCandles, ok := a.candles[symbol]; ok {
		return symbolCandles[interval]
	}

	return nil
}

// GetStats returns aggregator statistics
func (a *Aggregator) GetStats() map[string]interface{} {
	a.candlesMu.RLock()
	numSymbols := len(a.candles)
	a.candlesMu.RUnlock()

	return map[string]interface{}{
		"total_trades":  a.totalTrades,
		"total_candles": a.totalCandles,
		"symbols":       numSymbols,
	}
}

// cleanup removes old data periodically
func (a *Aggregator) cleanup() {
	defer a.wg.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.cleanupOldCandles()
		}
	}
}

// cleanupOldCandles removes old candles from database
func (a *Aggregator) cleanupOldCandles() {
	// Keep different retention periods for different intervals
	retentionMap := map[Interval]time.Duration{
		Interval1s:  24 * time.Hour,
		Interval5s:  3 * 24 * time.Hour,
		Interval15s: 7 * 24 * time.Hour,
		Interval30s: 7 * 24 * time.Hour,
		Interval1m:  30 * 24 * time.Hour,
		Interval5m:  90 * 24 * time.Hour,
		Interval15m: 180 * 24 * time.Hour,
		Interval30m: 365 * 24 * time.Hour,
		Interval1h:  365 * 24 * time.Hour,
		Interval1d:  10 * 365 * 24 * time.Hour,
	}

	now := time.Now()
	batch := a.db.NewBatch()
	defer batch.Reset()

	// TODO: Implement when database iterator is available
	for interval, retention := range retentionMap {
		cutoff := now.Add(-retention)
		prefix := fmt.Sprintf("candle:*:%s:", interval)

		_ = cutoff
		_ = prefix

		/*
			iter := a.db.NewIterator([]byte(prefix), nil)
			for iter.Next() {
				var candle Candle
				if err := json.Unmarshal(iter.Value(), &candle); err != nil {
					continue
				}

				if candle.OpenTime.Before(cutoff) {
					batch.Delete(iter.Key())
				}
			}
			iter.Release()
		*/
	}

	if err := batch.Write(); err != nil {
		a.logger.Error("Failed to cleanup old candles", "error", err)
	}
}

// VolumeWeightedAveragePrice calculates VWAP for a period
func (a *Aggregator) VolumeWeightedAveragePrice(symbol string, interval Interval, periods int) float64 {
	candles, err := a.GetCandles(symbol, interval, periods)
	if err != nil || len(candles) == 0 {
		return 0
	}

	var totalVolume float64
	var volumePrice float64

	for _, candle := range candles {
		avgPrice := (candle.High + candle.Low + candle.Close) / 3
		volumePrice += avgPrice * candle.Volume
		totalVolume += candle.Volume
	}

	if totalVolume == 0 {
		return 0
	}

	return volumePrice / totalVolume
}

// MovingAverage calculates simple moving average
func (a *Aggregator) MovingAverage(symbol string, interval Interval, periods int) float64 {
	candles, err := a.GetCandles(symbol, interval, periods)
	if err != nil || len(candles) == 0 {
		return 0
	}

	var sum float64
	for _, candle := range candles {
		sum += candle.Close
	}

	return sum / float64(len(candles))
}

// RSI calculates Relative Strength Index
func (a *Aggregator) RSI(symbol string, interval Interval, periods int) float64 {
	candles, err := a.GetCandles(symbol, interval, periods+1)
	if err != nil || len(candles) < 2 {
		return 50 // Neutral RSI
	}

	var gains, losses float64
	for i := 1; i < len(candles); i++ {
		change := candles[i].Close - candles[i-1].Close
		if change > 0 {
			gains += change
		} else {
			losses += -change
		}
	}

	if losses == 0 {
		return 100 // Overbought
	}

	avgGain := gains / float64(periods)
	avgLoss := losses / float64(periods)
	rs := avgGain / avgLoss

	return 100 - (100 / (1 + rs))
}
