package price

import (
	"errors"
	"math"
	"sort"
	"sync"
	"time"
)

// Oracle aggregates prices from multiple sources with configurable strategies.
type Oracle struct {
	sources    map[string]Source
	strategy   Aggregator
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
	running    bool
	mu         sync.RWMutex
}

// NewOracle creates a price oracle with default configuration.
func NewOracle() *Oracle {
	return &Oracle{
		sources: make(map[string]Source),
		strategy: &WeightedMedian{
			MinSources:   2,
			MaxDeviation: 0.05,
		},
		current:    make(map[string]*Data),
		history:    make(map[string][]*Data),
		twap:       make(map[string]float64),
		vwap:       make(map[string]float64),
		breakers:   make(map[string]*CircuitBreaker),
		updates:    make(chan *Update, 10000),
		alerts:     make(chan *Alert, 1000),
		interval:   50 * time.Millisecond,
		staleLimit: 2 * time.Second,
		minSources: 2,
	}
}

// AddSource registers a price source.
func (o *Oracle) AddSource(name string, src Source) {
	o.mu.Lock()
	o.sources[name] = src
	o.mu.Unlock()
}

// Start begins price aggregation.
func (o *Oracle) Start() error {
	o.mu.Lock()
	if o.running {
		o.mu.Unlock()
		return nil
	}
	o.running = true
	o.mu.Unlock()

	go o.loop()
	go o.calcAverages()
	return nil
}

func (o *Oracle) loop() {
	ticker := time.NewTicker(o.interval)
	defer ticker.Stop()

	for o.running {
		<-ticker.C
		o.update()
	}
}

func (o *Oracle) update() {
	symbols := o.symbols()

	for _, symbol := range symbols {
		prices := make([]*Data, 0)

		for name, src := range o.sources {
			if !src.Healthy() {
				o.alert(symbol, AlertSourceDown, SeverityWarn, "source unhealthy: "+name)
				continue
			}

			p, err := src.Price(symbol)
			if err != nil {
				continue
			}

			// Check circuit breaker
			if cb, ok := o.breakers[symbol]; ok {
				if !cb.Check(p.Price) {
					o.alert(symbol, AlertCircuitBreaker, SeverityCrit, "circuit breaker tripped")
					continue
				}
			}

			prices = append(prices, p)
		}

		if len(prices) < o.minSources {
			o.alert(symbol, AlertLowSources, SeverityCrit, "insufficient sources")
			continue
		}

		agg, err := o.strategy.Aggregate(prices)
		if err != nil {
			continue
		}

		o.store(symbol, agg)
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

	for o.running {
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
	o.mu.Lock()
	o.running = false
	o.mu.Unlock()
	close(o.updates)
	close(o.alerts)
}

func (o *Oracle) symbols() []string {
	o.mu.RLock()
	defer o.mu.RUnlock()

	seen := make(map[string]bool)
	for _, src := range o.sources {
		if prices, _ := src.Prices(nil); prices != nil {
			for sym := range prices {
				seen[sym] = true
			}
		}
	}

	result := make([]string, 0, len(seen))
	for sym := range seen {
		result = append(result, sym)
	}
	return result
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
