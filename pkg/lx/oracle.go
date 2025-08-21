package lx

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// AlertSeverity represents the severity level of an alert
type AlertSeverity int

const (
	InfoAlert AlertSeverity = iota
	WarningAlert
	CriticalAlert
)

// PriceOracle aggregates prices from multiple sources
type PriceOracle struct {
	// Price sources
	PriceSources        map[string]OraclePriceSource
	AggregationStrategy AggregationStrategy

	// Price data
	CurrentPrices map[string]*PriceData
	PriceHistory  map[string][]*PriceData
	TWAP          map[string]*TWAPData
	VWAP          map[string]*VWAPData

	// Oracle configuration
	UpdateInterval     time.Duration
	StaleThreshold     time.Duration
	DeviationThreshold float64
	MinimumSources     int

	// Circuit breakers
	CircuitBreakers map[string]*PriceCircuitBreaker
	EmergencyPrices map[string]float64

	// Monitoring
	PriceUpdates chan *PriceUpdate
	AlertChannel chan *PriceAlert
	Metrics      *OracleMetrics

	// State
	Running    bool
	LastUpdate time.Time
	mu         sync.RWMutex
}

// OraclePriceSource represents a price data source
type OraclePriceSource interface {
	GetPrice(symbol string) (*PriceData, error)
	GetPrices(symbols []string) (map[string]*PriceData, error)
	Subscribe(symbol string) error
	Unsubscribe(symbol string) error
	IsHealthy() bool
	GetName() string
	GetWeight() float64
}

// PriceData represents price information
type PriceData struct {
	Symbol     string
	Price      float64
	Volume     float64
	Bid        float64
	Ask        float64
	High24h    float64
	Low24h     float64
	Change24h  float64
	Timestamp  time.Time
	Source     string
	Confidence float64
	IsStale    bool
}

// TWAPData represents time-weighted average price
type TWAPData struct {
	Symbol      string
	Price       float64
	Window      time.Duration
	SampleCount int
	StartTime   time.Time
	EndTime     time.Time
	Prices      []float64
	Timestamps  []time.Time
}

// VWAPData represents volume-weighted average price
type VWAPData struct {
	Symbol      string
	Price       float64
	TotalVolume float64
	TotalValue  float64
	Window      time.Duration
	StartTime   time.Time
	EndTime     time.Time
}

// AggregationStrategy defines how to aggregate prices from multiple sources
type AggregationStrategy interface {
	Aggregate(prices []*PriceData) (*PriceData, error)
	ValidatePrices(prices []*PriceData) error
}

// MedianAggregation uses median price from sources
type MedianAggregation struct {
	MinSources      int
	MaxDeviation    float64
	OutlierHandling string
}

func (ma *MedianAggregation) Aggregate(prices []*PriceData) (*PriceData, error) {
	if len(prices) < ma.MinSources {
		return nil, fmt.Errorf("insufficient price sources: %d < %d", len(prices), ma.MinSources)
	}

	// Extract price values
	values := make([]float64, len(prices))
	totalVolume := 0.0
	for i, p := range prices {
		values[i] = p.Price
		totalVolume += p.Volume
	}

	// Calculate median
	median := calculateMedian(values)

	// Check for outliers
	filtered := ma.filterOutliers(prices, median)
	if len(filtered) < ma.MinSources {
		return nil, errors.New("too many outliers detected")
	}

	// Create aggregated price
	aggregated := &PriceData{
		Symbol:     prices[0].Symbol,
		Price:      median,
		Volume:     totalVolume,
		Timestamp:  time.Now(),
		Source:     "aggregated",
		Confidence: ma.calculateConfidence(filtered),
	}

	// Set bid/ask from best prices
	aggregated.Bid, aggregated.Ask = ma.getBestBidAsk(filtered)

	return aggregated, nil
}

func (ma *MedianAggregation) ValidatePrices(prices []*PriceData) error {
	if len(prices) == 0 {
		return errors.New("no prices to validate")
	}

	// Check for stale prices
	now := time.Now()
	for _, p := range prices {
		if now.Sub(p.Timestamp) > 30*time.Second {
			p.IsStale = true
		}
	}

	// Check price deviation
	median := calculateMedian(extractPrices(prices))
	for _, p := range prices {
		deviation := math.Abs(p.Price-median) / median
		if deviation > ma.MaxDeviation {
			return fmt.Errorf("price deviation too high: %.2f%%", deviation*100)
		}
	}

	return nil
}

func (ma *MedianAggregation) filterOutliers(prices []*PriceData, median float64) []*PriceData {
	filtered := make([]*PriceData, 0)

	for _, p := range prices {
		deviation := math.Abs(p.Price-median) / median
		if deviation <= ma.MaxDeviation {
			filtered = append(filtered, p)
		}
	}

	return filtered
}

func (ma *MedianAggregation) calculateConfidence(prices []*PriceData) float64 {
	if len(prices) == 0 {
		return 0
	}

	// Base confidence on number of sources and deviation
	sourceScore := float64(len(prices)) / float64(ma.MinSources*2)
	sourceScore = math.Min(sourceScore, 1.0)

	// Calculate standard deviation
	mean := calculateMean(extractPrices(prices))
	stdDev := calculateStdDev(extractPrices(prices), mean)
	deviationScore := 1.0 - (stdDev / mean)
	deviationScore = math.Max(deviationScore, 0)

	// Weight scores
	confidence := sourceScore*0.6 + deviationScore*0.4
	return math.Min(confidence, 1.0)
}

func (ma *MedianAggregation) getBestBidAsk(prices []*PriceData) (float64, float64) {
	if len(prices) == 0 {
		return 0, 0
	}

	bestBid := 0.0
	bestAsk := math.MaxFloat64

	for _, p := range prices {
		if p.Bid > bestBid {
			bestBid = p.Bid
		}
		if p.Ask < bestAsk && p.Ask > 0 {
			bestAsk = p.Ask
		}
	}

	if bestAsk == math.MaxFloat64 {
		bestAsk = 0
	}

	return bestBid, bestAsk
}

// WeightedAggregation uses weighted average based on source reliability
type WeightedAggregation struct {
	SourceWeights   map[string]float64
	VolumeWeighting bool
}

func (wa *WeightedAggregation) Aggregate(prices []*PriceData) (*PriceData, error) {
	if len(prices) == 0 {
		return nil, errors.New("no prices to aggregate")
	}

	totalWeight := 0.0
	weightedSum := 0.0
	totalVolume := 0.0

	for _, p := range prices {
		weight := wa.SourceWeights[p.Source]
		if weight == 0 {
			weight = 1.0
		}

		// Apply volume weighting if enabled
		if wa.VolumeWeighting && p.Volume > 0 {
			weight *= math.Log10(p.Volume + 1)
		}

		weightedSum += p.Price * weight
		totalWeight += weight
		totalVolume += p.Volume
	}

	if totalWeight == 0 {
		return nil, errors.New("total weight is zero")
	}

	aggregated := &PriceData{
		Symbol:     prices[0].Symbol,
		Price:      weightedSum / totalWeight,
		Volume:     totalVolume,
		Timestamp:  time.Now(),
		Source:     "weighted_aggregate",
		Confidence: wa.calculateConfidence(prices),
	}

	return aggregated, nil
}

func (wa *WeightedAggregation) ValidatePrices(prices []*PriceData) error {
	// Similar to MedianAggregation validation
	return nil
}

func (wa *WeightedAggregation) calculateConfidence(prices []*PriceData) float64 {
	// Calculate based on source weights and agreement
	return 0.9 // Placeholder
}

// PriceCircuitBreaker prevents erroneous prices
type PriceCircuitBreaker struct {
	Symbol            string
	MaxChangePercent  float64
	MaxChangeWindow   time.Duration
	LastValidPrice    float64
	LastValidTime     time.Time
	TripCount         int
	Tripped           bool
	AutoResetDuration time.Duration
	TrippedAt         time.Time
}

func (pcb *PriceCircuitBreaker) Check(newPrice float64) bool {
	if pcb.LastValidPrice == 0 {
		pcb.LastValidPrice = newPrice
		pcb.LastValidTime = time.Now()
		return true
	}

	// Check if circuit breaker should auto-reset
	if pcb.Tripped && time.Since(pcb.TrippedAt) > pcb.AutoResetDuration {
		pcb.Reset()
	}

	if pcb.Tripped {
		return false
	}

	// Check price change
	changePercent := math.Abs(newPrice-pcb.LastValidPrice) / pcb.LastValidPrice * 100

	if changePercent > pcb.MaxChangePercent {
		pcb.Trip()
		return false
	}

	pcb.LastValidPrice = newPrice
	pcb.LastValidTime = time.Now()
	return true
}

func (pcb *PriceCircuitBreaker) Trip() {
	pcb.Tripped = true
	pcb.TrippedAt = time.Now()
	pcb.TripCount++
}

func (pcb *PriceCircuitBreaker) Reset() {
	pcb.Tripped = false
}

// PriceUpdate represents a price update event
type PriceUpdate struct {
	Symbol        string
	OldPrice      float64
	NewPrice      float64
	Source        string
	Timestamp     time.Time
	ChangePercent float64
}

// PriceAlert represents a price alert
type PriceAlert struct {
	AlertID   string
	Symbol    string
	AlertType PriceAlertType
	Message   string
	Severity  AlertSeverity
	Price     float64
	Timestamp time.Time
}

type PriceAlertType int

const (
	PriceStale PriceAlertType = iota
	PriceDeviation
	SourceFailure
	CircuitBreakerTripped
	InsufficientSources
)

// OracleMetrics tracks oracle performance
type OracleMetrics struct {
	TotalUpdates        uint64
	FailedUpdates       uint64
	StaleDetections     uint64
	CircuitBreakerTrips uint64
	AverageLatency      time.Duration
	SourceHealth        map[string]bool
	LastUpdate          time.Time
}

// NewPriceOracle creates a new price oracle with Pyth and Chainlink integration
func NewPriceOracle() *PriceOracle {
	oracle := &PriceOracle{
		PriceSources: make(map[string]OraclePriceSource),
		AggregationStrategy: &WeightedAggregation{
			SourceWeights: map[string]float64{
				"pyth":      1.5, // Higher weight for real-time updates
				"chainlink": 2.0, // Higher weight for decentralization
				"internal":  1.0, // Internal DEX price
			},
			VolumeWeighting: true,
		},
		CurrentPrices:      make(map[string]*PriceData),
		PriceHistory:       make(map[string][]*PriceData),
		TWAP:               make(map[string]*TWAPData),
		VWAP:               make(map[string]*VWAPData),
		UpdateInterval:     50 * time.Millisecond, // Fast updates for HFT
		StaleThreshold:     2 * time.Second,
		DeviationThreshold: 0.05, // 5% deviation threshold
		MinimumSources:     2,
		CircuitBreakers:    initCircuitBreakers(),
		EmergencyPrices:    make(map[string]float64),
		PriceUpdates:       make(chan *PriceUpdate, 10000),
		AlertChannel:       make(chan *PriceAlert, 1000),
		Metrics:            NewOracleMetrics(),
	}

	// Initialize Pyth and Chainlink sources
	oracle.initializeDefaultSources()

	return oracle
}

// initializeDefaultSources sets up Pyth and Chainlink price sources
func (po *PriceOracle) initializeDefaultSources() {
	// TODO: Implement Pyth and Chainlink sources
	// Add Pyth Network source
	// pythSource := NewPythPriceSource(
	// 	"wss://hermes.pyth.network/ws",
	// 	"https://hermes.pyth.network",
	// )
	// po.AddSource("pyth", pythSource)

	// Add Chainlink source
	// chainlinkSource := NewChainlinkPriceSource()
	// po.AddSource("chainlink", chainlinkSource)
}

// AddSource adds a price source to the oracle
func (po *PriceOracle) AddSource(name string, source OraclePriceSource) error {
	po.mu.Lock()
	defer po.mu.Unlock()

	if _, exists := po.PriceSources[name]; exists {
		return fmt.Errorf("source %s already exists", name)
	}

	po.PriceSources[name] = source
	return nil
}

// Start starts the price oracle
func (po *PriceOracle) Start() error {
	po.mu.Lock()
	if po.Running {
		po.mu.Unlock()
		return errors.New("oracle already running")
	}
	po.Running = true
	po.mu.Unlock()

	// Start price update loop
	go po.updateLoop()

	// Start monitoring
	go po.monitorSources()

	// Start TWAP/VWAP calculation
	go po.calculateAverages()

	return nil
}

// Stop stops the price oracle
func (po *PriceOracle) Stop() {
	po.mu.Lock()
	defer po.mu.Unlock()

	po.Running = false
	close(po.PriceUpdates)
	close(po.AlertChannel)
}

// GetPrice returns the current price for a symbol
func (po *PriceOracle) GetPrice(symbol string) float64 {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if price, exists := po.CurrentPrices[symbol]; exists {
		// Check if price is stale
		if time.Since(price.Timestamp) > po.StaleThreshold {
			// Return emergency price if available
			if emergencyPrice, ok := po.EmergencyPrices[symbol]; ok {
				return emergencyPrice
			}
		}
		return price.Price
	}

	// Return emergency price if no current price
	if emergencyPrice, ok := po.EmergencyPrices[symbol]; ok {
		return emergencyPrice
	}

	return 0
}

// GetPriceData returns detailed price data for a symbol
func (po *PriceOracle) GetPriceData(symbol string) (*PriceData, error) {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if price, exists := po.CurrentPrices[symbol]; exists {
		return price, nil
	}

	return nil, fmt.Errorf("no price data for %s", symbol)
}

// GetTWAP returns the time-weighted average price
func (po *PriceOracle) GetTWAP(symbol string, window time.Duration) float64 {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if twap, exists := po.TWAP[symbol]; exists {
		if twap.Window == window {
			return twap.Price
		}
	}

	// Calculate TWAP from history
	return po.calculateTWAP(symbol, window)
}

// GetVWAP returns the volume-weighted average price
func (po *PriceOracle) GetVWAP(symbol string, window time.Duration) float64 {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if vwap, exists := po.VWAP[symbol]; exists {
		if vwap.Window == window {
			return vwap.Price
		}
	}

	// Calculate VWAP from history
	return po.calculateVWAP(symbol, window)
}

// updateLoop continuously updates prices
func (po *PriceOracle) updateLoop() {
	ticker := time.NewTicker(po.UpdateInterval)
	defer ticker.Stop()

	for po.Running {
		select {
		case <-ticker.C:
			po.updatePrices()
		}
	}
}

// updatePrices fetches and aggregates prices from all sources
func (po *PriceOracle) updatePrices() {
	symbols := po.getTrackedSymbols()

	for _, symbol := range symbols {
		prices := make([]*PriceData, 0)

		// Fetch from all sources
		for name, source := range po.PriceSources {
			if !source.IsHealthy() {
				po.sendAlert(&PriceAlert{
					Symbol:    symbol,
					AlertType: SourceFailure,
					Message:   fmt.Sprintf("Source %s is unhealthy", name),
					Severity:  WarningAlert,
					Timestamp: time.Now(),
				})
				continue
			}

			price, err := source.GetPrice(symbol)
			if err != nil {
				continue
			}

			// Check circuit breaker
			if cb, exists := po.CircuitBreakers[symbol]; exists {
				if !cb.Check(price.Price) {
					po.sendAlert(&PriceAlert{
						Symbol:    symbol,
						AlertType: CircuitBreakerTripped,
						Message:   fmt.Sprintf("Circuit breaker tripped for %s", symbol),
						Severity:  CriticalAlert,
						Price:     price.Price,
						Timestamp: time.Now(),
					})
					continue
				}
			}

			prices = append(prices, price)
		}

		// Check minimum sources
		if len(prices) < po.MinimumSources {
			po.sendAlert(&PriceAlert{
				Symbol:    symbol,
				AlertType: InsufficientSources,
				Message:   fmt.Sprintf("Only %d sources available for %s", len(prices), symbol),
				Severity:  CriticalAlert,
				Timestamp: time.Now(),
			})
			continue
		}

		// Aggregate prices
		aggregated, err := po.AggregationStrategy.Aggregate(prices)
		if err != nil {
			po.Metrics.FailedUpdates++
			continue
		}

		// Update current price
		po.updateCurrentPrice(symbol, aggregated)

		// Update metrics
		po.Metrics.TotalUpdates++
	}

	po.LastUpdate = time.Now()
}

// updateCurrentPrice updates the current price for a symbol
func (po *PriceOracle) updateCurrentPrice(symbol string, price *PriceData) {
	po.mu.Lock()
	defer po.mu.Unlock()

	oldPrice := po.CurrentPrices[symbol]
	po.CurrentPrices[symbol] = price

	// Add to history
	if po.PriceHistory[symbol] == nil {
		po.PriceHistory[symbol] = make([]*PriceData, 0)
	}
	po.PriceHistory[symbol] = append(po.PriceHistory[symbol], price)

	// Limit history size
	maxHistory := 10000
	if len(po.PriceHistory[symbol]) > maxHistory {
		po.PriceHistory[symbol] = po.PriceHistory[symbol][1:]
	}

	// Send update notification
	if oldPrice != nil {
		changePercent := (price.Price - oldPrice.Price) / oldPrice.Price * 100
		update := &PriceUpdate{
			Symbol:        symbol,
			OldPrice:      oldPrice.Price,
			NewPrice:      price.Price,
			Source:        price.Source,
			Timestamp:     price.Timestamp,
			ChangePercent: changePercent,
		}

		select {
		case po.PriceUpdates <- update:
		default:
			// Channel full, drop update
		}
	}
}

// monitorSources monitors health of price sources
func (po *PriceOracle) monitorSources() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for po.Running {
		select {
		case <-ticker.C:
			po.checkSourceHealth()
		}
	}
}

// checkSourceHealth checks health of all price sources
func (po *PriceOracle) checkSourceHealth() {
	po.mu.Lock()
	defer po.mu.Unlock()

	for name, source := range po.PriceSources {
		healthy := source.IsHealthy()
		po.Metrics.SourceHealth[name] = healthy

		if !healthy {
			po.sendAlert(&PriceAlert{
				AlertType: SourceFailure,
				Message:   fmt.Sprintf("Source %s is unhealthy", name),
				Severity:  WarningAlert,
				Timestamp: time.Now(),
			})
		}
	}
}

// calculateAverages calculates TWAP and VWAP
func (po *PriceOracle) calculateAverages() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for po.Running {
		select {
		case <-ticker.C:
			po.updateAverages()
		}
	}
}

// updateAverages updates TWAP and VWAP calculations
func (po *PriceOracle) updateAverages() {
	symbols := po.getTrackedSymbols()

	for _, symbol := range symbols {
		// Update TWAP
		twap := po.calculateTWAP(symbol, 5*time.Minute)
		po.mu.Lock()
		po.TWAP[symbol] = &TWAPData{
			Symbol:    symbol,
			Price:     twap,
			Window:    5 * time.Minute,
			StartTime: time.Now().Add(-5 * time.Minute),
			EndTime:   time.Now(),
		}
		po.mu.Unlock()

		// Update VWAP
		vwap := po.calculateVWAP(symbol, 5*time.Minute)
		po.mu.Lock()
		po.VWAP[symbol] = &VWAPData{
			Symbol:    symbol,
			Price:     vwap,
			Window:    5 * time.Minute,
			StartTime: time.Now().Add(-5 * time.Minute),
			EndTime:   time.Now(),
		}
		po.mu.Unlock()
	}
}

// calculateTWAP calculates time-weighted average price
func (po *PriceOracle) calculateTWAP(symbol string, window time.Duration) float64 {
	history := po.PriceHistory[symbol]
	if len(history) == 0 {
		return 0
	}

	cutoff := time.Now().Add(-window)
	relevantPrices := make([]float64, 0)

	for _, p := range history {
		if p.Timestamp.After(cutoff) {
			relevantPrices = append(relevantPrices, p.Price)
		}
	}

	if len(relevantPrices) == 0 {
		return 0
	}

	return calculateMean(relevantPrices)
}

// calculateVWAP calculates volume-weighted average price
func (po *PriceOracle) calculateVWAP(symbol string, window time.Duration) float64 {
	history := po.PriceHistory[symbol]
	if len(history) == 0 {
		return 0
	}

	cutoff := time.Now().Add(-window)
	totalValue := 0.0
	totalVolume := 0.0

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

// getTrackedSymbols returns all symbols being tracked
func (po *PriceOracle) getTrackedSymbols() []string {
	po.mu.RLock()
	defer po.mu.RUnlock()

	symbols := make([]string, 0, len(po.CurrentPrices))
	for symbol := range po.CurrentPrices {
		symbols = append(symbols, symbol)
	}

	return symbols
}

// sendAlert sends a price alert
func (po *PriceOracle) sendAlert(alert *PriceAlert) {
	alert.AlertID = fmt.Sprintf("alert_%d", time.Now().UnixNano())

	select {
	case po.AlertChannel <- alert:
	default:
		// Channel full, drop alert
	}
}

// Helper functions

func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort values
	sorted := make([]float64, len(values))
	copy(sorted, values)
	// Simple bubble sort for small arrays
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	if len(sorted)%2 == 0 {
		return (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	return sorted[len(sorted)/2]
}

func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}

	variance := sumSquaredDiff / float64(len(values))
	return math.Sqrt(variance)
}

func extractPrices(priceData []*PriceData) []float64 {
	prices := make([]float64, len(priceData))
	for i, p := range priceData {
		prices[i] = p.Price
	}
	return prices
}

func initCircuitBreakers() map[string]*PriceCircuitBreaker {
	symbols := []string{"BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "AVAX-USDT"}
	breakers := make(map[string]*PriceCircuitBreaker)

	for _, symbol := range symbols {
		breakers[symbol] = &PriceCircuitBreaker{
			Symbol:            symbol,
			MaxChangePercent:  20, // 20% max change
			MaxChangeWindow:   1 * time.Minute,
			AutoResetDuration: 5 * time.Minute,
		}
	}

	return breakers
}

func NewOracleMetrics() *OracleMetrics {
	return &OracleMetrics{
		SourceHealth: make(map[string]bool),
		LastUpdate:   time.Now(),
	}
}
