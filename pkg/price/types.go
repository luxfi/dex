// Package price provides multi-source price aggregation with sub-millisecond latency.
//
// Architecture:
//
//	┌─────────────────────────────────────────────────────────────┐
//	│                    PRICE AGGREGATOR                          │
//	│                                                               │
//	│  Sources:                       Aggregation:                 │
//	│  ┌──────────┐                  ┌────────────────────┐       │
//	│  │ Orderbook │──┐              │  Weighted Average   │       │
//	│  │ (local)   │  │              │  TWAP / VWAP        │       │
//	│  └──────────┘  │              │  Median             │       │
//	│  ┌──────────┐  │  ┌────────┐  │  Circuit Breakers   │       │
//	│  │  Pyth    │──┼──│ Merge  │──│  Outlier Detection  │──→ Price
//	│  │ Network  │  │  └────────┘  └────────────────────┘       │
//	│  └──────────┘  │                                             │
//	│  ┌──────────┐  │                                             │
//	│  │Chainlink │──┤                                             │
//	│  │ Oracle   │  │                                             │
//	│  └──────────┘  │                                             │
//	│  ┌──────────┐  │                                             │
//	│  │ C-Chain  │──┘                                             │
//	│  │  AMMs    │                                                 │
//	│  └──────────┘                                                 │
//	└─────────────────────────────────────────────────────────────┘
//
// Performance targets for co-located clients:
//   - Local orderbook: <100ns
//   - Aggregated price: <1μs
//   - External source: <10ms
package price

import (
	"time"
)

// Data represents a price quote from any source.
type Data struct {
	Symbol     string    `json:"symbol"`
	Price      float64   `json:"price"`
	Bid        float64   `json:"bid"`
	Ask        float64   `json:"ask"`
	Volume     float64   `json:"volume"`
	High24h    float64   `json:"high_24h"`
	Low24h     float64   `json:"low_24h"`
	Change24h  float64   `json:"change_24h"`
	Timestamp  time.Time `json:"timestamp"`
	Source     string    `json:"source"`
	Confidence float64   `json:"confidence"` // 0.0-1.0
	Stale      bool      `json:"stale"`
}

// Source provides prices from a single data source.
type Source interface {
	// Price returns the latest price for a symbol.
	Price(symbol string) (*Data, error)

	// Prices returns prices for multiple symbols.
	Prices(symbols []string) (map[string]*Data, error)

	// Subscribe registers for real-time updates.
	Subscribe(symbol string) error

	// Unsubscribe removes subscription.
	Unsubscribe(symbol string) error

	// Healthy returns true if the source is operational.
	Healthy() bool

	// Name returns the source identifier.
	Name() string

	// Weight returns the aggregation weight (higher = more trusted).
	Weight() float64
}

// Aggregator combines prices from multiple sources.
type Aggregator interface {
	// Aggregate combines prices into a single quote.
	Aggregate(prices []*Data) (*Data, error)

	// Validate checks price sanity.
	Validate(prices []*Data) error
}

// Update represents a price change event.
type Update struct {
	Symbol    string
	OldPrice  float64
	NewPrice  float64
	Source    string
	Timestamp time.Time
	Change    float64 // Percentage
}

// Alert represents a price anomaly.
type Alert struct {
	ID        string
	Symbol    string
	Type      AlertType
	Message   string
	Severity  Severity
	Price     float64
	Timestamp time.Time
}

// AlertType categorizes alerts.
type AlertType int

const (
	AlertStale AlertType = iota
	AlertDeviation
	AlertSourceDown
	AlertCircuitBreaker
	AlertLowSources
)

// Severity indicates alert importance.
type Severity int

const (
	SeverityInfo Severity = iota
	SeverityWarn
	SeverityCrit
)
