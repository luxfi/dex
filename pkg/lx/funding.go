package lx

import (
	"math"
	"sync"
	"time"
)

// FundingEngine manages the 8-hour funding mechanism for perpetual contracts
type FundingEngine struct {
	// Funding intervals (00:00, 08:00, 16:00 UTC)
	fundingTimes    []time.Time
	fundingInterval time.Duration // 8 hours

	// Current and next funding rates
	currentRates    map[string]*FundingRate   // symbol -> current rate
	nextRates       map[string]*FundingRate   // symbol -> predicted next rate
	historicalRates map[string][]*FundingRate // symbol -> historical rates

	// TWAP tracking for funding calculation
	markPriceTWAP  map[string]*TWAPTracker
	indexPriceTWAP map[string]*TWAPTracker
	premiumTWAP    map[string]*TWAPTracker

	// Funding parameters
	config *FundingConfig

	// Integration
	clearinghouse *ClearingHouse

	// State
	lastFundingTime time.Time
	nextFundingTime time.Time
	isProcessing    bool
	stopCh          chan struct{}
	stopped         bool

	mu sync.RWMutex
}

// FundingRate represents the funding rate for a symbol at a specific time
type FundingRate struct {
	Symbol         string
	Rate           float64 // Actual funding rate (can be positive or negative)
	PremiumIndex   float64 // Premium component
	InterestRate   float64 // Interest rate component (usually 0.01% for 8 hours)
	MarkTWAP       float64 // Mark price TWAP
	IndexTWAP      float64 // Index price TWAP
	Timestamp      time.Time
	PaymentTime    time.Time // When funding was/will be exchanged
	OpenInterest   float64   // Total open interest at funding time
	LongPositions  float64   // Total long positions
	ShortPositions float64   // Total short positions
}

// FundingConfig configures the funding mechanism
type FundingConfig struct {
	// Standard 8-hour funding times (UTC)
	FundingHours []int         // [0, 8, 16]
	Interval     time.Duration // 8 * time.Hour

	// Rate limits
	MaxFundingRate float64 // Maximum funding rate per period (e.g., 0.75%)
	MinFundingRate float64 // Minimum funding rate per period (e.g., -0.75%)

	// TWAP parameters
	TWAPWindow     time.Duration // Window for TWAP calculation (8 hours)
	SampleInterval time.Duration // How often to sample prices (1 minute)

	// Premium/Interest split
	InterestRate    float64 // Fixed interest rate component (0.01% = 0.0001)
	PremiumDampener float64 // Dampener for premium component (0.1 to 1.0)

	// Special handling
	ClampPremium  bool // Whether to clamp premium to max/min
	UseMedianTWAP bool // Use median instead of mean for TWAP
}

// TWAPTracker tracks time-weighted average price
type TWAPTracker struct {
	Symbol      string
	Samples     []PriceSample
	Window      time.Duration
	LastUpdate  time.Time
	CurrentTWAP float64
	mu          sync.RWMutex
}

// PriceSample represents a price at a specific time
type PriceSample struct {
	Price     float64
	Volume    float64 // Optional: volume weight
	Timestamp time.Time
}

// NewFundingEngine creates a new funding engine
func NewFundingEngine(clearinghouse *ClearingHouse, config *FundingConfig) *FundingEngine {
	if config == nil {
		config = DefaultFundingConfig()
	}

	fe := &FundingEngine{
		fundingInterval: config.Interval,
		currentRates:    make(map[string]*FundingRate),
		nextRates:       make(map[string]*FundingRate),
		historicalRates: make(map[string][]*FundingRate),
		markPriceTWAP:   make(map[string]*TWAPTracker),
		indexPriceTWAP:  make(map[string]*TWAPTracker),
		premiumTWAP:     make(map[string]*TWAPTracker),
		config:          config,
		clearinghouse:   clearinghouse,
		stopCh:          make(chan struct{}),
	}

	// Initialize funding times
	fe.initializeFundingSchedule()

	// Initialize TWAP trackers for each symbol
	fe.initializeTWAPTrackers()

	return fe
}

// DefaultFundingConfig returns standard 8-hour funding configuration
func DefaultFundingConfig() *FundingConfig {
	return &FundingConfig{
		FundingHours:    []int{0, 8, 16}, // 00:00, 08:00, 16:00 UTC
		Interval:        8 * time.Hour,
		MaxFundingRate:  0.0075,  // 0.75% max per 8 hours
		MinFundingRate:  -0.0075, // -0.75% min per 8 hours
		TWAPWindow:      8 * time.Hour,
		SampleInterval:  1 * time.Minute,
		InterestRate:    0.0001, // 0.01% per 8 hours
		PremiumDampener: 1.0,    // No dampening by default
		ClampPremium:    true,
		UseMedianTWAP:   false,
	}
}

// Start begins the funding engine's continuous operation
func (fe *FundingEngine) Start() {
	fe.mu.Lock()
	if fe.stopped {
		fe.stopCh = make(chan struct{})
		fe.stopped = false
	}
	fe.mu.Unlock()
	
	go fe.runFundingLoop()
	go fe.runTWAPSampling()
}

// Stop stops the funding engine's operation
func (fe *FundingEngine) Stop() {
	fe.mu.Lock()
	defer fe.mu.Unlock()
	
	if !fe.stopped {
		close(fe.stopCh)
		fe.stopped = true
	}
}

// runFundingLoop processes funding at scheduled times
func (fe *FundingEngine) runFundingLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-fe.stopCh:
			return
		case <-ticker.C:
			now := time.Now().UTC()

			// Check if it's funding time
			if fe.isFundingTime(now) && !fe.isProcessing {
				fe.isProcessing = true
				go func() {
					defer func() { fe.isProcessing = false }()
					fe.ProcessFunding(now)
				}()
			}

			// Update predicted funding rates
			fe.updatePredictedRates()
		}
	}
}

// runTWAPSampling continuously samples prices for TWAP calculation
func (fe *FundingEngine) runTWAPSampling() {
	ticker := time.NewTicker(fe.config.SampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-fe.stopCh:
			return
		case <-ticker.C:
			fe.samplePrices()
		}
	}
}

// ProcessFunding executes funding payments for all positions
func (fe *FundingEngine) ProcessFunding(fundingTime time.Time) error {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	// Calculate final funding rates for all symbols
	symbols := fe.getActiveSymbols()
	fundingPayments := make(map[string]map[string]float64) // symbol -> account -> payment

	for _, symbol := range symbols {
		// Calculate final funding rate
		rate := fe.calculateFundingRate(symbol)

		// Store the rate
		fe.currentRates[symbol] = rate
		fe.addToHistory(symbol, rate)

		// Process payments for this symbol
		payments := fe.calculateFundingPayments(symbol, rate)
		fundingPayments[symbol] = payments
	}

	// Apply all funding payments atomically
	for symbol, payments := range fundingPayments {
		for account, payment := range payments {
			fe.applyFundingPayment(account, symbol, payment)
		}
	}

	// Update next funding time
	fe.lastFundingTime = fundingTime
	fe.nextFundingTime = fe.getNextFundingTime(fundingTime)

	return nil
}

// calculateFundingRate calculates the funding rate for a symbol
func (fe *FundingEngine) calculateFundingRate(symbol string) *FundingRate {
	markTWAP := fe.getMarkTWAP(symbol)
	indexTWAP := fe.getIndexTWAP(symbol)

	// Calculate premium index
	premiumIndex := (markTWAP - indexTWAP) / indexTWAP

	// Apply dampener
	premiumIndex *= fe.config.PremiumDampener

	// Calculate funding rate = Premium Index + Interest Rate
	fundingRate := premiumIndex + fe.config.InterestRate

	// Clamp to max/min if configured
	if fe.config.ClampPremium {
		fundingRate = fe.clampRate(fundingRate)
	}

	// Get position statistics
	stats := fe.getPositionStats(symbol)

	return &FundingRate{
		Symbol:         symbol,
		Rate:           fundingRate,
		PremiumIndex:   premiumIndex,
		InterestRate:   fe.config.InterestRate,
		MarkTWAP:       markTWAP,
		IndexTWAP:      indexTWAP,
		Timestamp:      time.Now(),
		PaymentTime:    fe.nextFundingTime,
		OpenInterest:   stats.openInterest,
		LongPositions:  stats.longPositions,
		ShortPositions: stats.shortPositions,
	}
}

// calculateFundingPayments calculates funding payments for all positions
func (fe *FundingEngine) calculateFundingPayments(symbol string, rate *FundingRate) map[string]float64 {
	payments := make(map[string]float64)

	// Get all positions for this symbol from clearinghouse
	positions := fe.clearinghouse.GetAllPositions(symbol)

	for _, position := range positions {
		if position.Size == 0 {
			continue
		}

		// Funding Payment = Position Size × Mark TWAP × Funding Rate
		// Positive rate: Longs pay shorts
		// Negative rate: Shorts pay longs
		notionalValue := math.Abs(position.Size) * rate.MarkTWAP
		payment := notionalValue * rate.Rate

		if position.Size > 0 {
			// Long position pays when rate is positive
			payment = -payment
		}
		// Short position receives when rate is positive (payment stays positive)

		payments[position.User] = payment
	}

	return payments
}

// applyFundingPayment applies a funding payment to an account
func (fe *FundingEngine) applyFundingPayment(account string, symbol string, payment float64) {
	// This integrates with the clearinghouse
	fe.clearinghouse.ApplyFundingPayment(account, symbol, payment)
}

// samplePrices samples current prices for TWAP calculation
func (fe *FundingEngine) samplePrices() {
	symbols := fe.getActiveSymbols()

	for _, symbol := range symbols {
		// Get current mark and index prices
		markPrice := fe.clearinghouse.getMarkPrice(symbol)
		indexPrice := fe.getIndexPrice(symbol)

		// Add samples to TWAP trackers
		fe.addTWAPSample(fe.markPriceTWAP[symbol], markPrice)
		fe.addTWAPSample(fe.indexPriceTWAP[symbol], indexPrice)

		// Calculate and track premium
		premium := (markPrice - indexPrice) / indexPrice
		fe.addTWAPSample(fe.premiumTWAP[symbol], premium)
	}
}

// addTWAPSample adds a price sample to a TWAP tracker
func (fe *FundingEngine) addTWAPSample(tracker *TWAPTracker, price float64) {
	if tracker == nil {
		return
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	now := time.Now()

	// Add new sample
	tracker.Samples = append(tracker.Samples, PriceSample{
		Price:     price,
		Timestamp: now,
	})

	// Remove old samples outside the window
	cutoff := now.Add(-tracker.Window)
	i := 0
	for i < len(tracker.Samples) && tracker.Samples[i].Timestamp.Before(cutoff) {
		i++
	}
	tracker.Samples = tracker.Samples[i:]

	// Recalculate TWAP
	tracker.CurrentTWAP = fe.calculateTWAP(tracker.Samples)
	tracker.LastUpdate = now
}

// calculateTWAP calculates time-weighted average price from samples
func (fe *FundingEngine) calculateTWAP(samples []PriceSample) float64 {
	if len(samples) == 0 {
		return 0
	}

	if len(samples) == 1 {
		return samples[0].Price
	}

	if fe.config.UseMedianTWAP {
		return fe.calculateMedianTWAP(samples)
	}

	// Calculate time-weighted average
	var weightedSum float64
	var totalWeight float64

	for i := 0; i < len(samples)-1; i++ {
		price := samples[i].Price
		duration := samples[i+1].Timestamp.Sub(samples[i].Timestamp).Seconds()

		weightedSum += price * duration
		totalWeight += duration
	}

	// Add the last sample with weight to current time
	if len(samples) > 0 {
		lastSample := samples[len(samples)-1]
		duration := time.Since(lastSample.Timestamp).Seconds()
		if duration > 0 && duration < fe.config.SampleInterval.Seconds() {
			weightedSum += lastSample.Price * duration
			totalWeight += duration
		}
	}

	if totalWeight == 0 {
		return samples[len(samples)-1].Price
	}

	return weightedSum / totalWeight
}

// calculateMedianTWAP calculates median price from samples
func (fe *FundingEngine) calculateMedianTWAP(samples []PriceSample) float64 {
	prices := make([]float64, len(samples))
	for i, s := range samples {
		prices[i] = s.Price
	}

	// Sort prices
	for i := 0; i < len(prices); i++ {
		for j := i + 1; j < len(prices); j++ {
			if prices[i] > prices[j] {
				prices[i], prices[j] = prices[j], prices[i]
			}
		}
	}

	// Return median
	n := len(prices)
	if n%2 == 0 {
		return (prices[n/2-1] + prices[n/2]) / 2
	}
	return prices[n/2]
}

// updatePredictedRates updates the predicted funding rates
func (fe *FundingEngine) updatePredictedRates() {
	symbols := fe.getActiveSymbols()

	for _, symbol := range symbols {
		// Get current premium
		markPrice := fe.clearinghouse.getMarkPrice(symbol)
		indexPrice := fe.getIndexPrice(symbol)

		if indexPrice == 0 {
			continue
		}

		currentPremium := (markPrice - indexPrice) / indexPrice

		// Get TWAP premium
		var twapPremium float64
		if tracker, exists := fe.premiumTWAP[symbol]; exists {
			twapPremium = tracker.CurrentTWAP
		}

		// Weighted average of current and TWAP
		predictedPremium := 0.3*currentPremium + 0.7*twapPremium
		predictedRate := predictedPremium*fe.config.PremiumDampener + fe.config.InterestRate

		// Clamp predicted rate
		predictedRate = fe.clampRate(predictedRate)

		fe.nextRates[symbol] = &FundingRate{
			Symbol:       symbol,
			Rate:         predictedRate,
			PremiumIndex: predictedPremium,
			InterestRate: fe.config.InterestRate,
			Timestamp:    time.Now(),
			PaymentTime:  fe.nextFundingTime,
		}
	}
}

// Helper functions

func (fe *FundingEngine) initializeFundingSchedule() {
	now := time.Now().UTC()

	// Find next funding time
	hour := now.Hour()
	var nextHour int

	for _, fh := range fe.config.FundingHours {
		if fh > hour {
			nextHour = fh
			break
		}
	}

	// If no funding hour found after current hour, use first one tomorrow
	if nextHour == 0 && hour >= fe.config.FundingHours[len(fe.config.FundingHours)-1] {
		nextHour = fe.config.FundingHours[0]
		now = now.AddDate(0, 0, 1)
	}

	fe.nextFundingTime = time.Date(now.Year(), now.Month(), now.Day(), nextHour, 0, 0, 0, time.UTC)
	fe.lastFundingTime = fe.nextFundingTime.Add(-fe.config.Interval)
}

func (fe *FundingEngine) initializeTWAPTrackers() {
	symbols := []string{"BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "AVAX-PERP"}

	for _, symbol := range symbols {
		fe.markPriceTWAP[symbol] = &TWAPTracker{
			Symbol:  symbol,
			Window:  fe.config.TWAPWindow,
			Samples: make([]PriceSample, 0),
		}

		fe.indexPriceTWAP[symbol] = &TWAPTracker{
			Symbol:  symbol,
			Window:  fe.config.TWAPWindow,
			Samples: make([]PriceSample, 0),
		}

		fe.premiumTWAP[symbol] = &TWAPTracker{
			Symbol:  symbol,
			Window:  fe.config.TWAPWindow,
			Samples: make([]PriceSample, 0),
		}
	}
}

func (fe *FundingEngine) isFundingTime(t time.Time) bool {
	for _, hour := range fe.config.FundingHours {
		if t.Hour() == hour && t.Minute() == 0 {
			return true
		}
	}
	return false
}

func (fe *FundingEngine) getNextFundingTime(from time.Time) time.Time {
	next := from.Add(fe.config.Interval)

	// Ensure it aligns with funding hours
	for _, hour := range fe.config.FundingHours {
		candidate := time.Date(next.Year(), next.Month(), next.Day(), hour, 0, 0, 0, time.UTC)
		if candidate.After(from) {
			return candidate
		}
	}

	// Next day, first funding hour
	tomorrow := from.AddDate(0, 0, 1)
	return time.Date(tomorrow.Year(), tomorrow.Month(), tomorrow.Day(), fe.config.FundingHours[0], 0, 0, 0, time.UTC)
}

func (fe *FundingEngine) getActiveSymbols() []string {
	// Get from clearinghouse
	return []string{"BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "AVAX-PERP"}
}

func (fe *FundingEngine) getMarkTWAP(symbol string) float64 {
	if tracker, exists := fe.markPriceTWAP[symbol]; exists {
		return tracker.CurrentTWAP
	}
	return 0
}

func (fe *FundingEngine) getIndexTWAP(symbol string) float64 {
	if tracker, exists := fe.indexPriceTWAP[symbol]; exists {
		return tracker.CurrentTWAP
	}
	return 0
}

func (fe *FundingEngine) getIndexPrice(symbol string) float64 {
	// Get from oracle/clearinghouse
	if oracle, exists := fe.clearinghouse.oracles[symbol]; exists {
		return oracle.IndexPrice
	}
	return 0
}

func (fe *FundingEngine) clampRate(rate float64) float64 {
	if rate > fe.config.MaxFundingRate {
		return fe.config.MaxFundingRate
	}
	if rate < fe.config.MinFundingRate {
		return fe.config.MinFundingRate
	}
	return rate
}

func (fe *FundingEngine) addToHistory(symbol string, rate *FundingRate) {
	if _, exists := fe.historicalRates[symbol]; !exists {
		fe.historicalRates[symbol] = make([]*FundingRate, 0)
	}

	fe.historicalRates[symbol] = append(fe.historicalRates[symbol], rate)

	// Keep only last 30 days of history
	maxHistory := 30 * 24 / 8 // 90 funding periods
	if len(fe.historicalRates[symbol]) > maxHistory {
		fe.historicalRates[symbol] = fe.historicalRates[symbol][1:]
	}
}

type positionStats struct {
	openInterest   float64
	longPositions  float64
	shortPositions float64
}

func (fe *FundingEngine) getPositionStats(symbol string) positionStats {
	var stats positionStats

	// Get from clearinghouse
	positions := fe.clearinghouse.GetAllPositions(symbol)

	for _, pos := range positions {
		notional := math.Abs(pos.Size) * pos.MarkPrice
		stats.openInterest += notional

		if pos.Size > 0 {
			stats.longPositions += notional
		} else {
			stats.shortPositions += notional
		}
	}

	return stats
}

// GetCurrentFundingRate returns the current funding rate for a symbol
func (fe *FundingEngine) GetCurrentFundingRate(symbol string) *FundingRate {
	fe.mu.RLock()
	defer fe.mu.RUnlock()

	if rate, exists := fe.currentRates[symbol]; exists {
		return rate
	}
	return nil
}

// GetPredictedFundingRate returns the predicted next funding rate
func (fe *FundingEngine) GetPredictedFundingRate(symbol string) *FundingRate {
	fe.mu.RLock()
	defer fe.mu.RUnlock()

	if rate, exists := fe.nextRates[symbol]; exists {
		return rate
	}
	return nil
}

// GetFundingHistory returns historical funding rates for a symbol
func (fe *FundingEngine) GetFundingHistory(symbol string, limit int) []*FundingRate {
	fe.mu.RLock()
	defer fe.mu.RUnlock()

	history, exists := fe.historicalRates[symbol]
	if !exists {
		return nil
	}

	if limit <= 0 || limit > len(history) {
		return history
	}

	// Return last 'limit' rates
	return history[len(history)-limit:]
}

// GetNextFundingTime returns the next funding time
func (fe *FundingEngine) GetNextFundingTime() time.Time {
	fe.mu.RLock()
	defer fe.mu.RUnlock()

	return fe.nextFundingTime
}

// GetTimeUntilFunding returns duration until next funding
func (fe *FundingEngine) GetTimeUntilFunding() time.Duration {
	return time.Until(fe.GetNextFundingTime())
}
