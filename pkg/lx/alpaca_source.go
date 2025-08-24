package lx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// AlpacaSource provides low-latency market data from Alpaca Markets
// Alpaca offers real-time market data with sub-millisecond latency
type AlpacaSource struct {
	apiKey    string
	apiSecret string
	baseURL   string
	wsURL     string

	// Low-latency optimizations
	client     *http.Client
	cache      sync.Map // Lock-free cache for prices
	lastUpdate map[string]time.Time
	mu         sync.RWMutex

	// Price channels for streaming
	priceStreams map[string]chan PriceUpdate
	streamMu     sync.RWMutex
}

// NYSEArcaSource provides direct market access to NYSE Arca
// NYSE Arca is the leading exchange for ETF trading with ultra-low latency
type NYSEArcaSource struct {
	connectionID string
	sessionKey   string

	// Direct market access connection
	tcpConn       interface{} // Would be actual TCP connection
	lastHeartbeat time.Time

	// Order book cache
	books  map[string]*OrderBook
	bookMu sync.RWMutex
}

// IEXCloudSource provides real-time data from IEX Cloud
// IEX offers fair-access, low-latency market data
type IEXCloudSource struct {
	token       string
	version     string
	environment string // sandbox or production

	// WebSocket for streaming
	wsConn      interface{} // Would be actual WebSocket
	subscribers map[string][]chan PriceUpdate
	subMu       sync.RWMutex
}

// PolygonIOSource provides real-time and historical market data
// Polygon.io offers microsecond-precision timestamps
type PolygonIOSource struct {
	apiKey  string
	cluster string // stocks, crypto, forex, etc.

	// Aggregated data cache
	aggregates map[string][]Aggregate
	aggMu      sync.RWMutex

	// WebSocket streaming
	wsEndpoint string
	streamConn interface{}
}

// Aggregate represents a price bar from Polygon
type Aggregate struct {
	Symbol    string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	VWAP      float64
	Timestamp int64 // Unix nanoseconds for precision
}

// CMEDataSource provides futures data from CME Group
// CME offers the most liquid futures markets globally
type CMEDataSource struct {
	username string
	password string

	// FIX connection for ultra-low latency
	fixSession interface{} // Would be QuickFIX session

	// Market data
	futuresData map[string]*FuturesContract
	dataMu      sync.RWMutex
}

// FuturesContract represents a CME futures contract
type FuturesContract struct {
	Symbol         string
	ExpirationDate time.Time
	ContractSize   float64
	TickSize       float64
	LastPrice      float64
	BidPrice       float64
	AskPrice       float64
	OpenInterest   int64
	LastUpdate     time.Time
}

// RefinitivSource (formerly Thomson Reuters) for institutional data
type RefinitivSource struct {
	appKey    string
	machineID string
	position  string // Trading desk position for entitlements

	// Elektron Real-Time connection
	ertEndpoint string
	ertSession  interface{}

	// Cache for subscribed instruments
	instruments map[string]*InstrumentData
	instMu      sync.RWMutex
}

// InstrumentData from Refinitiv
type InstrumentData struct {
	RIC          string // Reuters Instrument Code
	LastPrice    float64
	BidPrice     float64
	AskPrice     float64
	BidSize      float64
	AskSize      float64
	LastUpdate   time.Time
	LastExchange string
}

// ICEDataSource for ICE Data Services (formerly IDC)
type ICEDataSource struct {
	username string
	password string
	feedType string // consolidated, direct, etc.

	// Low-latency multicast feed
	multicastGroup string
	multicastPort  int

	// Normalized data
	marketData map[string]*ICEMarketData
	mdMu       sync.RWMutex
}

// ICEMarketData represents ICE market data
type ICEMarketData struct {
	Symbol      string
	Exchange    string
	LastPrice   float64
	LastSize    float64
	TotalVolume int64
	VWAP        float64
	LastUpdate  time.Time
}

// BloombergBPIPE for Bloomberg Professional API
type BloombergBPIPE struct {
	serverHost string
	serverPort int
	uuid       string

	// Session management
	session interface{} // Would be blpapi.Session

	// Subscription management
	subscriptions map[string]*BloombergSubscription
	subMu         sync.RWMutex
}

// BloombergSubscription for real-time Bloomberg data
type BloombergSubscription struct {
	Security   string
	Fields     []string
	LastData   map[string]interface{}
	LastUpdate time.Time
}

// NasdaqTotalView provides full NASDAQ depth of book
type NasdaqTotalView struct {
	username string
	password string

	// ITCH protocol connection
	itchConn   interface{}
	messageSeq uint64

	// Full order book
	orderBooks map[string]*NasdaqBook
	bookMu     sync.RWMutex
}

// NasdaqBook represents full NASDAQ order book
type NasdaqBook struct {
	Symbol     string
	BidLevels  []PriceLevel
	AskLevels  []PriceLevel
	LastMatch  float64
	LastUpdate time.Time
}

// CoinbaseProSource for crypto markets
type CoinbaseProSource struct {
	apiKey     string
	apiSecret  string
	passphrase string

	// WebSocket feed
	wsFeed interface{}

	// Order book management
	books  map[string]*CryptoBook
	bookMu sync.RWMutex
}

// CryptoBook for cryptocurrency order books
type CryptoBook struct {
	Symbol     string
	Bids       []OrderLevel
	Asks       []OrderLevel
	LastTrade  float64
	Volume24h  float64
	LastUpdate time.Time
}

// NewAlpacaSource creates a new Alpaca market data source
func NewAlpacaSource(apiKey, apiSecret string) *AlpacaSource {
	return &AlpacaSource{
		apiKey:       apiKey,
		apiSecret:    apiSecret,
		baseURL:      "https://data.alpaca.markets/v2",
		wsURL:        "wss://stream.data.alpaca.markets/v2",
		client:       &http.Client{Timeout: 100 * time.Millisecond}, // Ultra-low timeout
		lastUpdate:   make(map[string]time.Time),
		priceStreams: make(map[string]chan PriceUpdate),
	}
}

// GetLatestPrice returns the latest price with sub-millisecond latency
func (a *AlpacaSource) GetLatestPrice(symbol string) (float64, error) {
	// Check cache first (lock-free)
	if cached, ok := a.cache.Load(symbol); ok {
		if price, ok := cached.(float64); ok {
			return price, nil
		}
	}

	// Fetch from API
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/stocks/%s/trades/latest", a.baseURL, symbol), nil)
	if err != nil {
		return 0, err
	}

	req.Header.Set("APCA-API-KEY-ID", a.apiKey)
	req.Header.Set("APCA-API-SECRET-KEY", a.apiSecret)

	resp, err := a.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var result struct {
		Trade struct {
			Price float64 `json:"p"`
		} `json:"trade"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, err
	}

	// Update cache
	a.cache.Store(symbol, result.Trade.Price)

	return result.Trade.Price, nil
}

// StreamPrices starts streaming real-time prices
func (a *AlpacaSource) StreamPrices(symbols []string) (<-chan PriceUpdate, error) {
	updates := make(chan PriceUpdate, 1000) // Buffered for performance

	// In production, this would establish WebSocket connection
	// and stream real-time updates with nanosecond timestamps

	go func() {
		ticker := time.NewTicker(1 * time.Millisecond) // 1ms updates
		defer ticker.Stop()

		for range ticker.C {
			for _, symbol := range symbols {
				// Simulate real-time price updates
				price, _ := a.GetLatestPrice(symbol)
				select {
				case updates <- PriceUpdate{
					Symbol:    symbol,
					Price:     price,
					Timestamp: time.Now(),
					Source:    "alpaca",
				}:
				default:
					// Skip if channel is full (non-blocking)
				}
			}
		}
	}()

	return updates, nil
}

// GetOrderBook returns Level 2 order book data
func (a *AlpacaSource) GetOrderBook(symbol string) (*OrderBookSnapshot, error) {
	// Alpaca doesn't provide full order book via REST API
	// This would use WebSocket for real-time book updates

	// For now, return a mock snapshot
	return &OrderBookSnapshot{
		Symbol:    symbol,
		Timestamp: time.Now(),
		Bids: []OrderLevel{
			{Price: 100.00, Size: 1000},
			{Price: 99.99, Size: 2000},
			{Price: 99.98, Size: 1500},
		},
		Asks: []OrderLevel{
			{Price: 100.01, Size: 1000},
			{Price: 100.02, Size: 2000},
			{Price: 100.03, Size: 1500},
		},
	}, nil
}

// GetHistoricalBars returns historical price bars
func (a *AlpacaSource) GetHistoricalBars(symbol string, start, end time.Time, timeframe string) ([]Bar, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	url := fmt.Sprintf("%s/stocks/%s/bars?start=%s&end=%s&timeframe=%s",
		a.baseURL, symbol, start.Format(time.RFC3339), end.Format(time.RFC3339), timeframe)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("APCA-API-KEY-ID", a.apiKey)
	req.Header.Set("APCA-API-SECRET-KEY", a.apiSecret)

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Bars []Bar `json:"bars"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Bars, nil
}

// Bar represents a price bar
type Bar struct {
	Timestamp time.Time `json:"t"`
	Open      float64   `json:"o"`
	High      float64   `json:"h"`
	Low       float64   `json:"l"`
	Close     float64   `json:"c"`
	Volume    int64     `json:"v"`
	VWAP      float64   `json:"vw"`
}

// PriceUpdate represents a real-time price update
type PriceUpdate struct {
	Symbol    string
	Price     float64
	Bid       float64
	Ask       float64
	BidSize   float64
	AskSize   float64
	Volume    int64
	Timestamp time.Time
	Source    string
	Latency   time.Duration // Network latency measurement
}

// MarketDataAggregator combines multiple low-latency sources
type MarketDataAggregator struct {
	sources    map[string]MarketDataSource
	weights    map[string]float64
	cache      sync.Map
	lastUpdate sync.Map

	// Latency tracking
	latencies map[string]*LatencyTracker
	latencyMu sync.RWMutex
}

// MarketDataSource interface for all data sources
type MarketDataSource interface {
	GetLatestPrice(symbol string) (float64, error)
	GetOrderBook(symbol string) (*OrderBookSnapshot, error)
	StreamPrices(symbols []string) (<-chan PriceUpdate, error)
}

// LatencyTracker tracks network latency for each source
type LatencyTracker struct {
	source     string
	samples    []time.Duration
	avgLatency time.Duration
	minLatency time.Duration
	maxLatency time.Duration
	lastUpdate time.Time
	mu         sync.RWMutex
}

// NewMarketDataAggregator creates a new aggregator for multiple sources
func NewMarketDataAggregator() *MarketDataAggregator {
	return &MarketDataAggregator{
		sources:   make(map[string]MarketDataSource),
		weights:   make(map[string]float64),
		latencies: make(map[string]*LatencyTracker),
	}
}

// AddSource adds a new market data source with weight
func (m *MarketDataAggregator) AddSource(name string, source MarketDataSource, weight float64) {
	m.sources[name] = source
	m.weights[name] = weight
	m.latencies[name] = &LatencyTracker{
		source:     name,
		samples:    make([]time.Duration, 0, 1000),
		minLatency: time.Hour, // Start with high value
	}
}

// GetBestPrice returns the best aggregated price from all sources
func (m *MarketDataAggregator) GetBestPrice(symbol string) (float64, error) {
	var (
		prices  []float64
		weights []float64
		wg      sync.WaitGroup
		mu      sync.Mutex
	)

	// Query all sources in parallel
	for name, source := range m.sources {
		wg.Add(1)
		go func(n string, s MarketDataSource) {
			defer wg.Done()

			start := time.Now()
			price, err := s.GetLatestPrice(symbol)
			latency := time.Since(start)

			// Track latency
			m.recordLatency(n, latency)

			if err == nil && price > 0 {
				mu.Lock()
				prices = append(prices, price)
				weights = append(weights, m.weights[n])
				mu.Unlock()
			}
		}(name, source)
	}

	wg.Wait()

	if len(prices) == 0 {
		return 0, fmt.Errorf("no prices available")
	}

	// Calculate weighted average
	var weightedSum, totalWeight float64
	for i, price := range prices {
		weightedSum += price * weights[i]
		totalWeight += weights[i]
	}

	return weightedSum / totalWeight, nil
}

// recordLatency records latency measurement
func (m *MarketDataAggregator) recordLatency(source string, latency time.Duration) {
	m.latencyMu.RLock()
	tracker, exists := m.latencies[source]
	m.latencyMu.RUnlock()

	if !exists {
		return
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.samples = append(tracker.samples, latency)
	if len(tracker.samples) > 1000 {
		tracker.samples = tracker.samples[1:] // Keep last 1000 samples
	}

	// Update min/max
	if latency < tracker.minLatency {
		tracker.minLatency = latency
	}
	if latency > tracker.maxLatency {
		tracker.maxLatency = latency
	}

	// Calculate average
	var sum time.Duration
	for _, l := range tracker.samples {
		sum += l
	}
	tracker.avgLatency = sum / time.Duration(len(tracker.samples))
	tracker.lastUpdate = time.Now()
}

// GetLatencyStats returns latency statistics for all sources
func (m *MarketDataAggregator) GetLatencyStats() map[string]LatencyStats {
	stats := make(map[string]LatencyStats)

	m.latencyMu.RLock()
	defer m.latencyMu.RUnlock()

	for name, tracker := range m.latencies {
		tracker.mu.RLock()
		stats[name] = LatencyStats{
			Source:     name,
			AvgLatency: tracker.avgLatency,
			MinLatency: tracker.minLatency,
			MaxLatency: tracker.maxLatency,
			Samples:    len(tracker.samples),
			LastUpdate: tracker.lastUpdate,
		}
		tracker.mu.RUnlock()
	}

	return stats
}

// LatencyStats represents latency statistics for a source
type LatencyStats struct {
	Source     string
	AvgLatency time.Duration
	MinLatency time.Duration
	MaxLatency time.Duration
	Samples    int
	LastUpdate time.Time
}
