package price

import (
	"errors"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Oracle aggregates prices from multiple sources with configurable strategies.
type Oracle struct {
	sources    map[string]Source
	strategy   Aggregator
	symbolMap  *SymbolMap
	watching   map[string]bool
	current    map[string]*Data
	history    map[string][]*Data
	twap       map[string]float64
	vwap       map[string]float64
	breakers   map[string]*CircuitBreaker
	updates    chan *Update
	alerts     chan *Alert
	interval   time.Duration
	staleLimit time.Duration
	minSources int
	running    atomic.Bool
	mu         sync.RWMutex

	// Q-Chain quantum finality verifier (optional)
	verifier *QChainVerifier
}

// NewOracle creates a price oracle with default configuration.
func NewOracle() *Oracle {
	return &Oracle{
		sources: make(map[string]Source),
		strategy: &WeightedMedian{
			MinSources:   1,
			MaxDeviation: 0.10,
		},
		symbolMap:  NewSymbolMap(),
		watching:   make(map[string]bool),
		current:    make(map[string]*Data),
		history:    make(map[string][]*Data),
		twap:       make(map[string]float64),
		vwap:       make(map[string]float64),
		breakers:   make(map[string]*CircuitBreaker),
		updates:    make(chan *Update, 10000),
		alerts:     make(chan *Alert, 1000),
		interval:   50 * time.Millisecond,
		staleLimit: 5 * time.Second,
		minSources: 1,
	}
}

// Watch adds a symbol to the watchlist.
func (o *Oracle) Watch(symbol string) {
	o.mu.Lock()
	o.watching[o.symbolMap.Map(symbol)] = true
	o.mu.Unlock()
}

// AddSource registers a price source.
func (o *Oracle) AddSource(name string, src Source) {
	o.mu.Lock()
	o.sources[name] = src
	o.mu.Unlock()
}

// Start begins price aggregation.
func (o *Oracle) Start() error {
	// Use atomic compare-and-swap to avoid race
	if !o.running.CompareAndSwap(false, true) {
		return nil // Already running
	}

	go o.loop()
	go o.calcAverages()
	return nil
}

func (o *Oracle) loop() {
	ticker := time.NewTicker(o.interval)
	defer ticker.Stop()

	for o.running.Load() {
		<-ticker.C
		o.update()
	}
}

func (o *Oracle) update() {
	o.mu.RLock()
	syms := make([]string, 0, len(o.watching))
	for sym := range o.watching {
		syms = append(syms, sym)
	}
	o.mu.RUnlock()

	for _, normalized := range syms {
		prices := make([]*Data, 0)

		// Get all variants that map to this normalized symbol
		variants := o.symbolMap.Reverse(normalized)

		for _, src := range o.sources {
			if !src.Healthy() {
				continue
			}

			// Try each variant
			for _, sym := range variants {
				p, err := src.Price(sym)
				if err != nil {
					continue
				}

				// Normalize the symbol in the result
				p.Symbol = normalized

				// Check circuit breaker
				if cb, ok := o.breakers[normalized]; ok {
					if !cb.Check(p.Price) {
						o.alert(normalized, AlertCircuitBreaker, SeverityCrit, "circuit breaker tripped")
						continue
					}
				}

				prices = append(prices, p)
				break // Got price from this source
			}
		}

		if len(prices) < o.minSources {
			continue
		}

		agg, err := o.strategy.Aggregate(prices)
		if err != nil {
			continue
		}

		agg.Symbol = normalized
		o.store(normalized, agg)
	}
}

func (o *Oracle) store(symbol string, price *Data) {
	o.mu.Lock()
	defer o.mu.Unlock()

	old := o.current[symbol]
	o.current[symbol] = price

	// History
	if o.history[symbol] == nil {
		o.history[symbol] = make([]*Data, 0)
	}
	o.history[symbol] = append(o.history[symbol], price)
	if len(o.history[symbol]) > 10000 {
		o.history[symbol] = o.history[symbol][1:]
	}

	// Update notification
	if old != nil {
		change := (price.Price - old.Price) / old.Price * 100
		select {
		case o.updates <- &Update{
			Symbol:    symbol,
			OldPrice:  old.Price,
			NewPrice:  price.Price,
			Source:    price.Source,
			Timestamp: price.Timestamp,
			Change:    change,
		}:
		default:
		}
	}
}

func (o *Oracle) calcAverages() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for o.running.Load() {
		<-ticker.C

		o.mu.Lock()
		for symbol, hist := range o.history {
			if len(hist) == 0 {
				continue
			}

			// TWAP (5 min)
			o.twap[symbol] = calcTWAP(hist, 5*time.Minute)

			// VWAP (5 min)
			o.vwap[symbol] = calcVWAP(hist, 5*time.Minute)
		}
		o.mu.Unlock()
	}
}

// Price returns the aggregated price for a symbol.
func (o *Oracle) Price(symbol string) float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()

	if p, ok := o.current[symbol]; ok {
		if time.Since(p.Timestamp) > o.staleLimit {
			return 0
		}
		return p.Price
	}
	return 0
}

// Data returns full price data for a symbol.
func (o *Oracle) Data(symbol string) (*Data, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	p, ok := o.current[symbol]
	if !ok {
		return nil, ErrNotFound
	}
	copy := *p
	return &copy, nil
}

// TWAP returns time-weighted average price.
func (o *Oracle) TWAP(symbol string) float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.twap[symbol]
}

// VWAP returns volume-weighted average price.
func (o *Oracle) VWAP(symbol string) float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.vwap[symbol]
}

// Updates returns the update channel.
func (o *Oracle) Updates() <-chan *Update {
	return o.updates
}

// Alerts returns the alert channel.
func (o *Oracle) Alerts() <-chan *Alert {
	return o.alerts
}

// Stop halts the oracle.
func (o *Oracle) Stop() {
	// Use atomic compare-and-swap to avoid race
	if !o.running.CompareAndSwap(true, false) {
		return // Already stopped
	}

	// Stop verifier if running
	if o.verifier != nil {
		o.verifier.Close()
	}

	// Give time for goroutines to exit
	time.Sleep(100 * time.Millisecond)
}

// SetVerifier attaches a Q-Chain quantum finality verifier.
// When set, the Oracle automatically verifies prices against Q-Chain finality.
func (o *Oracle) SetVerifier(v *QChainVerifier) {
	o.mu.Lock()
	o.verifier = v
	o.mu.Unlock()
}

// Verifier returns the attached Q-Chain verifier (or nil if not set).
func (o *Oracle) Verifier() *QChainVerifier {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.verifier
}

// VerifiedPrice returns price data with quantum finality status.
// If a verifier is attached, includes finality verification.
func (o *Oracle) VerifiedPrice(symbol string) (*VerifiedData, error) {
	data, err := o.Data(symbol)
	if err != nil {
		return nil, err
	}

	vd := &VerifiedData{
		Data:      data,
		Finalized: false,
	}

	// If verifier attached, check finality
	if o.verifier != nil {
		// Determine which chain this price came from
		chain := sourceToChain(data.Source)
		if chain != "" {
			if fin, err := o.verifier.Finality(chain); err == nil && fin.Finalized {
				vd.Finalized = true
				vd.Finality = fin
			}
		}
	}

	return vd, nil
}

// sourceToChain maps source names to chain identifiers for finality verification.
func sourceToChain(source string) string {
	switch source {
	case "x-chain":
		return "x-chain"
	case "c-chain":
		return "c-chain"
	case "zoo-chain":
		return "zoo-chain"
	case "a-chain":
		return "a-chain"
	default:
		return "" // External oracles don't have on-chain finality
	}
}


func (o *Oracle) alert(symbol string, t AlertType, sev Severity, msg string) {
	select {
	case o.alerts <- &Alert{
		Symbol:    symbol,
		Type:      t,
		Severity:  sev,
		Message:   msg,
		Timestamp: time.Now(),
	}:
	default:
	}
}

// WeightedMedian aggregates using weighted median.
type WeightedMedian struct {
	MinSources   int
	MaxDeviation float64
}

// Aggregate combines prices using weighted median.
func (w *WeightedMedian) Aggregate(prices []*Data) (*Data, error) {
	if len(prices) < w.MinSources {
		return nil, errors.New("insufficient sources")
	}

	// Sort by price
	sort.Slice(prices, func(i, j int) bool {
		return prices[i].Price < prices[j].Price
	})

	// Calculate median
	median := prices[len(prices)/2].Price

	// Filter outliers
	filtered := make([]*Data, 0)
	for _, p := range prices {
		dev := math.Abs(p.Price-median) / median
		if dev <= w.MaxDeviation {
			filtered = append(filtered, p)
		}
	}

	if len(filtered) < w.MinSources {
		return nil, errors.New("too many outliers")
	}

	// Weighted average
	totalWeight := 0.0
	weightedSum := 0.0
	totalVolume := 0.0

	for _, p := range filtered {
		weight := 1.0 // Could use source weight
		weightedSum += p.Price * weight
		totalWeight += weight
		totalVolume += p.Volume
	}

	return &Data{
		Symbol:     prices[0].Symbol,
		Price:      weightedSum / totalWeight,
		Volume:     totalVolume,
		Confidence: w.confidence(filtered),
		Timestamp:  time.Now(),
		Source:     "aggregated",
	}, nil
}

// Validate checks price sanity.
func (w *WeightedMedian) Validate(prices []*Data) error {
	if len(prices) == 0 {
		return errors.New("no prices")
	}
	return nil
}

func (w *WeightedMedian) confidence(prices []*Data) float64 {
	if len(prices) == 0 {
		return 0
	}

	sourceScore := float64(len(prices)) / float64(w.MinSources*2)
	if sourceScore > 1.0 {
		sourceScore = 1.0
	}

	// Calculate agreement
	values := make([]float64, len(prices))
	for i, p := range prices {
		values[i] = p.Price
	}

	mean := avg(values)
	stdDev := stddev(values, mean)
	devScore := 1.0 - (stdDev / mean)
	if devScore < 0 {
		devScore = 0
	}

	return sourceScore*0.6 + devScore*0.4
}

// CircuitBreaker prevents erroneous prices.
type CircuitBreaker struct {
	Symbol    string
	MaxChange float64
	LastPrice float64
	LastTime  time.Time
	Tripped   bool
	TripTime  time.Time
	Reset     time.Duration
}

// Check validates a new price.
func (cb *CircuitBreaker) Check(price float64) bool {
	if cb.LastPrice == 0 {
		cb.LastPrice = price
		cb.LastTime = time.Now()
		return true
	}

	// Auto-reset
	if cb.Tripped && time.Since(cb.TripTime) > cb.Reset {
		cb.Tripped = false
	}

	if cb.Tripped {
		return false
	}

	change := math.Abs(price-cb.LastPrice) / cb.LastPrice * 100
	if change > cb.MaxChange {
		cb.Tripped = true
		cb.TripTime = time.Now()
		return false
	}

	cb.LastPrice = price
	cb.LastTime = time.Now()
	return true
}

// Helpers

func calcTWAP(history []*Data, window time.Duration) float64 {
	cutoff := time.Now().Add(-window)
	var sum float64
	var count int

	for _, p := range history {
		if p.Timestamp.After(cutoff) {
			sum += p.Price
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func calcVWAP(history []*Data, window time.Duration) float64 {
	cutoff := time.Now().Add(-window)
	var totalValue, totalVolume float64

	for _, p := range history {
		if p.Timestamp.After(cutoff) {
			totalValue += p.Price * p.Volume
			totalVolume += p.Volume
		}
	}

	if totalVolume == 0 {
		return 0
	}
	return totalValue / totalVolume
}

func avg(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func stddev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		d := v - mean
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(values)))
}
