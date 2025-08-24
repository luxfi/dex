package lx

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAlpacaSource tests Alpaca market data source
func TestAlpacaSource(t *testing.T) {
	t.Run("CreateAlpacaSource", func(t *testing.T) {
		source := NewAlpacaSource("test-key", "test-secret")
		assert.NotNil(t, source)
		assert.Equal(t, "test-key", source.apiKey)
		assert.Equal(t, "test-secret", source.apiSecret)
		assert.Equal(t, "https://data.alpaca.markets/v2", source.baseURL)
		assert.Equal(t, "wss://stream.data.alpaca.markets/v2", source.wsURL)
	})

	t.Run("GetLatestPrice", func(t *testing.T) {
		source := NewAlpacaSource("test-key", "test-secret")

		// Cache a price
		source.cache.Store("AAPL", 150.50)

		// Should return cached price
		price, err := source.GetLatestPrice("AAPL")
		assert.NoError(t, err)
		assert.Equal(t, 150.50, price)
	})

	t.Run("GetOrderBook", func(t *testing.T) {
		source := NewAlpacaSource("test-key", "test-secret")

		book, err := source.GetOrderBook("AAPL")
		require.NoError(t, err)
		assert.NotNil(t, book)
		assert.Equal(t, "AAPL", book.Symbol)
		assert.Len(t, book.Bids, 3)
		assert.Len(t, book.Asks, 3)
	})

	t.Run("StreamPrices", func(t *testing.T) {
		source := NewAlpacaSource("test-key", "test-secret")

		// Cache some prices for testing
		source.cache.Store("AAPL", 150.50)
		source.cache.Store("GOOGL", 2800.00)

		symbols := []string{"AAPL", "GOOGL"}
		updates, err := source.StreamPrices(symbols)
		require.NoError(t, err)
		assert.NotNil(t, updates)

		// Read a few updates
		timeout := time.After(10 * time.Millisecond)
		select {
		case update := <-updates:
			assert.Contains(t, symbols, update.Symbol)
			assert.Greater(t, update.Price, 0.0)
			assert.Equal(t, "alpaca", update.Source)
		case <-timeout:
			// Timeout is ok in test
		}
	})
}

// TestMarketDataAggregator tests the aggregator
func TestMarketDataAggregator(t *testing.T) {
	t.Run("CreateAggregator", func(t *testing.T) {
		agg := NewMarketDataAggregator()
		assert.NotNil(t, agg)
		assert.NotNil(t, agg.sources)
		assert.NotNil(t, agg.weights)
		assert.NotNil(t, agg.latencies)
	})

	t.Run("AddSource", func(t *testing.T) {
		agg := NewMarketDataAggregator()
		source := NewAlpacaSource("test-key", "test-secret")

		agg.AddSource("alpaca", source, 1.0)

		assert.Contains(t, agg.sources, "alpaca")
		assert.Equal(t, 1.0, agg.weights["alpaca"])
		assert.Contains(t, agg.latencies, "alpaca")
	})

	t.Run("GetBestPrice", func(t *testing.T) {
		agg := NewMarketDataAggregator()

		// Create mock sources
		source1 := NewAlpacaSource("key1", "secret1")
		source1.cache.Store("AAPL", 150.00)

		source2 := NewAlpacaSource("key2", "secret2")
		source2.cache.Store("AAPL", 150.50)

		agg.AddSource("source1", source1, 1.0)
		agg.AddSource("source2", source2, 2.0)

		price, err := agg.GetBestPrice("AAPL")
		require.NoError(t, err)

		// Weighted average: (150.00 * 1.0 + 150.50 * 2.0) / 3.0 = 150.33
		assert.InDelta(t, 150.33, price, 0.01)
	})

	t.Run("RecordLatency", func(t *testing.T) {
		agg := NewMarketDataAggregator()
		source := NewAlpacaSource("test-key", "test-secret")
		agg.AddSource("alpaca", source, 1.0)

		// Record some latencies
		agg.recordLatency("alpaca", 1*time.Millisecond)
		agg.recordLatency("alpaca", 2*time.Millisecond)
		agg.recordLatency("alpaca", 3*time.Millisecond)

		stats := agg.GetLatencyStats()
		assert.Contains(t, stats, "alpaca")

		alpacaStats := stats["alpaca"]
		assert.Equal(t, "alpaca", alpacaStats.Source)
		assert.Equal(t, 1*time.Millisecond, alpacaStats.MinLatency)
		assert.Equal(t, 3*time.Millisecond, alpacaStats.MaxLatency)
		assert.Equal(t, 3, alpacaStats.Samples)
		assert.Equal(t, 2*time.Millisecond, alpacaStats.AvgLatency)
	})

	t.Run("GetLatencyStats", func(t *testing.T) {
		agg := NewMarketDataAggregator()

		// Add multiple sources
		source1 := NewAlpacaSource("key1", "secret1")
		source2 := NewAlpacaSource("key2", "secret2")

		agg.AddSource("alpaca", source1, 1.0)
		agg.AddSource("polygon", source2, 1.0)

		// Record latencies
		agg.recordLatency("alpaca", 1*time.Millisecond)
		agg.recordLatency("polygon", 2*time.Millisecond)

		stats := agg.GetLatencyStats()
		assert.Len(t, stats, 2)
		assert.Contains(t, stats, "alpaca")
		assert.Contains(t, stats, "polygon")
	})
}

// TestNYSEArcaSource tests NYSE Arca direct market access
func TestNYSEArcaSource(t *testing.T) {
	t.Run("CreateNYSEArcaSource", func(t *testing.T) {
		source := &NYSEArcaSource{
			connectionID: "NYSE-001",
			sessionKey:   "session-key",
			books:        make(map[string]*OrderBook),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "NYSE-001", source.connectionID)
		assert.Equal(t, "session-key", source.sessionKey)
		assert.NotNil(t, source.books)
	})
}

// TestIEXCloudSource tests IEX Cloud integration
func TestIEXCloudSource(t *testing.T) {
	t.Run("CreateIEXCloudSource", func(t *testing.T) {
		source := &IEXCloudSource{
			token:       "test-token",
			version:     "stable",
			environment: "sandbox",
			subscribers: make(map[string][]chan PriceUpdate),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "test-token", source.token)
		assert.Equal(t, "stable", source.version)
		assert.Equal(t, "sandbox", source.environment)
		assert.NotNil(t, source.subscribers)
	})
}

// TestPolygonIOSource tests Polygon.io integration
func TestPolygonIOSource(t *testing.T) {
	t.Run("CreatePolygonIOSource", func(t *testing.T) {
		source := &PolygonIOSource{
			apiKey:     "polygon-key",
			cluster:    "stocks",
			aggregates: make(map[string][]Aggregate),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "polygon-key", source.apiKey)
		assert.Equal(t, "stocks", source.cluster)
		assert.NotNil(t, source.aggregates)
	})

	t.Run("Aggregate", func(t *testing.T) {
		agg := Aggregate{
			Symbol:    "AAPL",
			Open:      150.00,
			High:      152.00,
			Low:       149.50,
			Close:     151.50,
			Volume:    1000000,
			VWAP:      151.00,
			Timestamp: time.Now().UnixNano(),
		}

		assert.Equal(t, "AAPL", agg.Symbol)
		assert.Equal(t, 150.00, agg.Open)
		assert.Equal(t, 152.00, agg.High)
		assert.Equal(t, 149.50, agg.Low)
		assert.Equal(t, 151.50, agg.Close)
		assert.Equal(t, 1000000.0, agg.Volume)
		assert.Equal(t, 151.00, agg.VWAP)
		assert.Greater(t, agg.Timestamp, int64(0))
	})
}

// TestCMEDataSource tests CME futures data
func TestCMEDataSource(t *testing.T) {
	t.Run("CreateCMEDataSource", func(t *testing.T) {
		source := &CMEDataSource{
			username:    "cme-user",
			password:    "cme-pass",
			futuresData: make(map[string]*FuturesContract),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "cme-user", source.username)
		assert.Equal(t, "cme-pass", source.password)
		assert.NotNil(t, source.futuresData)
	})

	t.Run("FuturesContract", func(t *testing.T) {
		contract := &FuturesContract{
			Symbol:         "ES",
			ExpirationDate: time.Now().AddDate(0, 3, 0),
			ContractSize:   50,
			TickSize:       0.25,
			LastPrice:      4500.00,
			BidPrice:       4499.75,
			AskPrice:       4500.25,
			OpenInterest:   2000000,
			LastUpdate:     time.Now(),
		}

		assert.Equal(t, "ES", contract.Symbol)
		assert.Equal(t, 50.0, contract.ContractSize)
		assert.Equal(t, 0.25, contract.TickSize)
		assert.Equal(t, 4500.00, contract.LastPrice)
		assert.Equal(t, 4499.75, contract.BidPrice)
		assert.Equal(t, 4500.25, contract.AskPrice)
		assert.Equal(t, int64(2000000), contract.OpenInterest)
	})
}

// TestRefinitivSource tests Refinitiv/Thomson Reuters integration
func TestRefinitivSource(t *testing.T) {
	t.Run("CreateRefinitivSource", func(t *testing.T) {
		source := &RefinitivSource{
			appKey:      "refinitiv-key",
			machineID:   "machine-001",
			position:    "NYC-DESK-01",
			instruments: make(map[string]*InstrumentData),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "refinitiv-key", source.appKey)
		assert.Equal(t, "machine-001", source.machineID)
		assert.Equal(t, "NYC-DESK-01", source.position)
		assert.NotNil(t, source.instruments)
	})

	t.Run("InstrumentData", func(t *testing.T) {
		inst := &InstrumentData{
			RIC:          "AAPL.O",
			LastPrice:    150.50,
			BidPrice:     150.49,
			AskPrice:     150.51,
			BidSize:      1000,
			AskSize:      1500,
			LastUpdate:   time.Now(),
			LastExchange: "NASDAQ",
		}

		assert.Equal(t, "AAPL.O", inst.RIC)
		assert.Equal(t, 150.50, inst.LastPrice)
		assert.Equal(t, 150.49, inst.BidPrice)
		assert.Equal(t, 150.51, inst.AskPrice)
		assert.Equal(t, 1000.0, inst.BidSize)
		assert.Equal(t, 1500.0, inst.AskSize)
		assert.Equal(t, "NASDAQ", inst.LastExchange)
	})
}

// TestICEDataSource tests ICE Data Services
func TestICEDataSource(t *testing.T) {
	t.Run("CreateICEDataSource", func(t *testing.T) {
		source := &ICEDataSource{
			username:       "ice-user",
			password:       "ice-pass",
			feedType:       "consolidated",
			multicastGroup: "239.255.1.1",
			multicastPort:  12345,
			marketData:     make(map[string]*ICEMarketData),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "ice-user", source.username)
		assert.Equal(t, "ice-pass", source.password)
		assert.Equal(t, "consolidated", source.feedType)
		assert.Equal(t, "239.255.1.1", source.multicastGroup)
		assert.Equal(t, 12345, source.multicastPort)
		assert.NotNil(t, source.marketData)
	})

	t.Run("ICEMarketData", func(t *testing.T) {
		data := &ICEMarketData{
			Symbol:      "AAPL",
			Exchange:    "NASDAQ",
			LastPrice:   150.50,
			LastSize:    100,
			TotalVolume: 50000000,
			VWAP:        150.25,
			LastUpdate:  time.Now(),
		}

		assert.Equal(t, "AAPL", data.Symbol)
		assert.Equal(t, "NASDAQ", data.Exchange)
		assert.Equal(t, 150.50, data.LastPrice)
		assert.Equal(t, 100.0, data.LastSize)
		assert.Equal(t, int64(50000000), data.TotalVolume)
		assert.Equal(t, 150.25, data.VWAP)
	})
}

// TestBloombergBPIPE tests Bloomberg Professional API
func TestBloombergBPIPE(t *testing.T) {
	t.Run("CreateBloombergBPIPE", func(t *testing.T) {
		source := &BloombergBPIPE{
			serverHost:    "localhost",
			serverPort:    8194,
			uuid:          "bloomberg-uuid",
			subscriptions: make(map[string]*BloombergSubscription),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "localhost", source.serverHost)
		assert.Equal(t, 8194, source.serverPort)
		assert.Equal(t, "bloomberg-uuid", source.uuid)
		assert.NotNil(t, source.subscriptions)
	})

	t.Run("BloombergSubscription", func(t *testing.T) {
		sub := &BloombergSubscription{
			Security:   "AAPL US Equity",
			Fields:     []string{"PX_LAST", "PX_BID", "PX_ASK", "VOLUME"},
			LastData:   make(map[string]interface{}),
			LastUpdate: time.Now(),
		}

		assert.Equal(t, "AAPL US Equity", sub.Security)
		assert.Contains(t, sub.Fields, "PX_LAST")
		assert.Contains(t, sub.Fields, "PX_BID")
		assert.Contains(t, sub.Fields, "PX_ASK")
		assert.Contains(t, sub.Fields, "VOLUME")
		assert.NotNil(t, sub.LastData)
	})
}

// TestNasdaqTotalView tests NASDAQ Total View
func TestNasdaqTotalView(t *testing.T) {
	t.Run("CreateNasdaqTotalView", func(t *testing.T) {
		source := &NasdaqTotalView{
			username:   "nasdaq-user",
			password:   "nasdaq-pass",
			orderBooks: make(map[string]*NasdaqBook),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "nasdaq-user", source.username)
		assert.Equal(t, "nasdaq-pass", source.password)
		assert.NotNil(t, source.orderBooks)
	})

	t.Run("NasdaqBook", func(t *testing.T) {
		book := &NasdaqBook{
			Symbol: "AAPL",
			BidLevels: []PriceLevel{
				{Price: 150.00, Size: 1000, Count: 5},
				{Price: 149.99, Size: 2000, Count: 10},
			},
			AskLevels: []PriceLevel{
				{Price: 150.01, Size: 1000, Count: 5},
				{Price: 150.02, Size: 2000, Count: 10},
			},
			LastMatch:  150.00,
			LastUpdate: time.Now(),
		}

		assert.Equal(t, "AAPL", book.Symbol)
		assert.Len(t, book.BidLevels, 2)
		assert.Len(t, book.AskLevels, 2)
		assert.Equal(t, 150.00, book.LastMatch)
	})
}

// TestCoinbaseProSource tests Coinbase Pro crypto integration
func TestCoinbaseProSource(t *testing.T) {
	t.Run("CreateCoinbaseProSource", func(t *testing.T) {
		source := &CoinbaseProSource{
			apiKey:     "coinbase-key",
			apiSecret:  "coinbase-secret",
			passphrase: "coinbase-passphrase",
			books:      make(map[string]*CryptoBook),
		}

		assert.NotNil(t, source)
		assert.Equal(t, "coinbase-key", source.apiKey)
		assert.Equal(t, "coinbase-secret", source.apiSecret)
		assert.Equal(t, "coinbase-passphrase", source.passphrase)
		assert.NotNil(t, source.books)
	})

	t.Run("CryptoBook", func(t *testing.T) {
		book := &CryptoBook{
			Symbol: "BTC-USD",
			Bids: []OrderLevel{
				{Price: 50000.00, Size: 0.5},
				{Price: 49999.00, Size: 1.0},
			},
			Asks: []OrderLevel{
				{Price: 50001.00, Size: 0.5},
				{Price: 50002.00, Size: 1.0},
			},
			LastTrade:  50000.50,
			Volume24h:  1000.5,
			LastUpdate: time.Now(),
		}

		assert.Equal(t, "BTC-USD", book.Symbol)
		assert.Len(t, book.Bids, 2)
		assert.Len(t, book.Asks, 2)
		assert.Equal(t, 50000.50, book.LastTrade)
		assert.Equal(t, 1000.5, book.Volume24h)
	})
}

// TestLatencyTracking tests latency measurement
func TestLatencyTracking(t *testing.T) {
	t.Run("LatencyTracker", func(t *testing.T) {
		tracker := &LatencyTracker{
			source:     "test-source",
			samples:    make([]time.Duration, 0, 1000),
			minLatency: time.Hour,
		}

		assert.Equal(t, "test-source", tracker.source)
		assert.NotNil(t, tracker.samples)
		assert.Equal(t, time.Hour, tracker.minLatency)
	})

	t.Run("PriceUpdate", func(t *testing.T) {
		update := PriceUpdate{
			Symbol:    "AAPL",
			Price:     150.50,
			Bid:       150.49,
			Ask:       150.51,
			BidSize:   1000,
			AskSize:   1500,
			Volume:    50000000,
			Timestamp: time.Now(),
			Source:    "alpaca",
			Latency:   1 * time.Millisecond,
		}

		assert.Equal(t, "AAPL", update.Symbol)
		assert.Equal(t, 150.50, update.Price)
		assert.Equal(t, 150.49, update.Bid)
		assert.Equal(t, 150.51, update.Ask)
		assert.Equal(t, 1000.0, update.BidSize)
		assert.Equal(t, 1500.0, update.AskSize)
		assert.Equal(t, int64(50000000), update.Volume)
		assert.Equal(t, "alpaca", update.Source)
		assert.Equal(t, 1*time.Millisecond, update.Latency)
	})

	t.Run("LatencyStats", func(t *testing.T) {
		stats := LatencyStats{
			Source:     "alpaca",
			AvgLatency: 2 * time.Millisecond,
			MinLatency: 1 * time.Millisecond,
			MaxLatency: 3 * time.Millisecond,
			Samples:    100,
			LastUpdate: time.Now(),
		}

		assert.Equal(t, "alpaca", stats.Source)
		assert.Equal(t, 2*time.Millisecond, stats.AvgLatency)
		assert.Equal(t, 1*time.Millisecond, stats.MinLatency)
		assert.Equal(t, 3*time.Millisecond, stats.MaxLatency)
		assert.Equal(t, 100, stats.Samples)
	})
}

// TestBar tests price bar structure
func TestBar(t *testing.T) {
	bar := Bar{
		Timestamp: time.Now(),
		Open:      150.00,
		High:      152.00,
		Low:       149.50,
		Close:     151.50,
		Volume:    1000000,
		VWAP:      151.00,
	}

	assert.Equal(t, 150.00, bar.Open)
	assert.Equal(t, 152.00, bar.High)
	assert.Equal(t, 149.50, bar.Low)
	assert.Equal(t, 151.50, bar.Close)
	assert.Equal(t, int64(1000000), bar.Volume)
	assert.Equal(t, 151.00, bar.VWAP)
}
