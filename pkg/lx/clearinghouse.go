package lx

import (
	"errors"
	"math"
	"math/big"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ClearingHouse manages perps margin state and spot balances for each address
// Implements HyperCore-style clearing with cross and isolated margin
type ClearingHouse struct {
	// Perps margin state
	perpAccounts map[string]*PerpAccountState // address -> account state

	// Spot clearinghouse state
	spotBalances map[string]map[string]*big.Int // address -> token -> balance
	spotHolds    map[string]map[string]*big.Int // address -> token -> holds

	// Oracle management - weighted median from multiple sources
	oracles    map[string]*MultiSourceOracle // asset -> oracle
	validators []*ValidatorOracle

	// Risk parameters
	maintenanceMargin map[string]float64 // symbol -> maintenance margin ratio
	initialMargin     map[string]float64 // symbol -> initial margin ratio
	maxLeverage       map[string]float64 // symbol -> max leverage

	// Funding mechanism
	fundingRates    map[string]float64   // symbol -> current funding rate
	fundingInterval time.Duration        // 8 hours standard
	nextFundingTime map[string]time.Time // symbol -> next funding time
	fundingEngine   *FundingEngine       // 8-hour funding engine

	// Performance metrics
	totalVolume atomic.Uint64
	totalOrders atomic.Uint64
	totalTrades atomic.Uint64

	// Lock-free operations
	mu           sync.RWMutex
	accountLocks map[string]*sync.RWMutex // Per-account locks for parallel processing

	// Integration with orderbooks
	orderBooks   map[string]*OrderBook
	marginEngine *MarginEngine
	riskEngine   *RiskEngine

	// FPGA acceleration flags
	fpgaEnabled   bool
	fpgaOrderPath bool // Use FPGA for order processing
	fpgaRiskPath  bool // Use FPGA for risk checks
}

// PerpAccountState represents a user's perpetual trading state
type PerpAccountState struct {
	Address string

	// Cross margin mode (default)
	CrossBalance   *big.Int                 // USDC balance for cross margin
	CrossPositions map[string]*PerpPosition // symbol -> position
	CrossPnL       *big.Int

	// Isolated margin positions
	IsolatedPositions map[string]*IsolatedPosition // symbol -> isolated position

	// Risk metrics
	AccountValue *big.Int
	MarginUsed   *big.Int
	FreeMargin   *big.Int
	MarginLevel  float64 // AccountValue / MarginUsed

	// Order margin
	OrderMargin map[string]*big.Int // symbol -> margin locked in orders

	// Liquidation tracking
	LiquidationPrice map[string]float64 // symbol -> liquidation price
	IsLiquidating    bool

	// Activity tracking
	TotalVolume  *big.Int
	TotalFees    *big.Int
	LastActivity time.Time

	mu sync.RWMutex
}

// IsolatedPosition represents an isolated margin position
type IsolatedPosition struct {
	Symbol           string
	Size             float64
	EntryPrice       float64
	MarkPrice        float64
	Margin           *big.Int // Allocated margin for this position
	UnrealizedPnL    *big.Int
	RealizedPnL      *big.Int
	LiquidationPrice float64
	MaxSize          float64 // Max position size based on allocated margin
	LastUpdate       time.Time
}

// MultiSourceOracle aggregates prices from multiple exchanges
type MultiSourceOracle struct {
	Asset          string
	Sources        map[string]*PriceSource // exchange -> price source
	Weights        map[string]int          // exchange -> weight
	LastPrice      float64
	LastUpdate     time.Time
	UpdateInterval time.Duration

	// Price components
	SpotPrice  float64
	MarkPrice  float64 // Used for margining
	IndexPrice float64 // Used for funding

	mu sync.RWMutex
}

// PriceSource represents a price feed from an exchange
type PriceSource struct {
	Exchange   string
	Price      float64
	Volume     float64
	LastUpdate time.Time
	IsActive   bool
	Confidence float64 // 0-1 confidence score
}

// ValidatorOracle represents a validator's oracle submission
type ValidatorOracle struct {
	ValidatorID    string
	Stake          *big.Int
	Prices         map[string]float64 // asset -> price
	LastSubmission time.Time
	IsActive       bool
	Reliability    float64 // Historical reliability score
}

// NewClearingHouse creates a new clearing house instance
func NewClearingHouse(marginEngine *MarginEngine, riskEngine *RiskEngine) *ClearingHouse {
	ch := &ClearingHouse{
		perpAccounts:      make(map[string]*PerpAccountState),
		spotBalances:      make(map[string]map[string]*big.Int),
		spotHolds:         make(map[string]map[string]*big.Int),
		oracles:           make(map[string]*MultiSourceOracle),
		validators:        make([]*ValidatorOracle, 0),
		maintenanceMargin: initMaintenanceMargins(),
		initialMargin:     initInitialMargins(),
		maxLeverage:       initMaxLeverages(),
		fundingRates:      make(map[string]float64),
		fundingInterval:   8 * time.Hour,
		nextFundingTime:   make(map[string]time.Time),
		accountLocks:      make(map[string]*sync.RWMutex),
		orderBooks:        make(map[string]*OrderBook),
		marginEngine:      marginEngine,
		riskEngine:        riskEngine,
	}

	// Initialize oracle sources with exchange weights
	ch.initializeOracleSources()

	// Initialize 8-hour funding engine
	ch.fundingEngine = NewFundingEngine(ch, DefaultFundingConfig())
	ch.fundingEngine.Start()

	// Check for FPGA acceleration
	ch.detectFPGACapabilities()

	return ch
}

// initializeOracleSources sets up weighted oracle sources
// Weights: Binance(3), OKX(2), Bybit(2), Kraken(1), Kucoin(1), Gate(1), MEXC(1), Hyperliquid(1)
func (ch *ClearingHouse) initializeOracleSources() {
	exchanges := map[string]int{
		"Binance":     3,
		"OKX":         2,
		"Bybit":       2,
		"Kraken":      1,
		"Kucoin":      1,
		"Gate":        1,
		"MEXC":        1,
		"Hyperliquid": 1,
	}

	// Initialize for major assets
	assets := []string{"BTC", "ETH", "SOL", "ARB", "MATIC", "AVAX"}

	for _, asset := range assets {
		oracle := &MultiSourceOracle{
			Asset:          asset,
			Sources:        make(map[string]*PriceSource),
			Weights:        exchanges,
			UpdateInterval: 3 * time.Second, // Update every 3 seconds
		}

		// Initialize price sources
		for exchange, weight := range exchanges {
			oracle.Sources[exchange] = &PriceSource{
				Exchange:   exchange,
				IsActive:   true,
				Confidence: 1.0,
			}
			oracle.Weights[exchange] = weight
		}

		ch.oracles[asset] = oracle
	}
}

// Deposit credits funds to an address's cross margin balance
func (ch *ClearingHouse) Deposit(address string, amount *big.Int) error {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	if amount.Sign() <= 0 {
		return errors.New("deposit amount must be positive")
	}

	account := ch.getOrCreateAccount(address)
	account.mu.Lock()
	defer account.mu.Unlock()

	// Credit to cross margin balance
	account.CrossBalance = new(big.Int).Add(account.CrossBalance, amount)
	account.LastActivity = time.Now()

	// Update account metrics
	ch.updateAccountMetrics(account)

	return nil
}

// OpenPosition opens a new perp position (cross margin by default)
func (ch *ClearingHouse) OpenPosition(address string, symbol string, side Side, size float64, orderType OrderType) (*PerpPosition, error) {
	// Get account lock for concurrent safety
	lock := ch.getAccountLock(address)
	lock.Lock()
	defer lock.Unlock()

	account := ch.getOrCreateAccount(address)

	// Perform margin check with FPGA acceleration if available
	if ch.fpgaRiskPath {
		if !ch.fpgaMarginCheck(account, symbol, size) {
			return nil, errors.New("insufficient margin")
		}
	} else {
		if !ch.performMarginCheck(account, symbol, size) {
			return nil, errors.New("insufficient margin")
		}
	}

	// Get mark price from oracle
	markPrice := ch.getMarkPrice(symbol)

	// Create or update position
	position, isNew := ch.updatePosition(account, symbol, side, size, markPrice)

	// Update margin usage
	ch.updateMarginUsage(account, symbol, position)

	// Increment metrics
	ch.totalOrders.Add(1)
	if isNew {
		ch.totalTrades.Add(1)
	}

	notional := size * markPrice
	ch.totalVolume.Add(uint64(notional))

	return position, nil
}

// AllocateIsolatedMargin allocates margin to an isolated position
func (ch *ClearingHouse) AllocateIsolatedMargin(address string, symbol string, margin *big.Int) error {
	lock := ch.getAccountLock(address)
	lock.Lock()
	defer lock.Unlock()

	account := ch.getOrCreateAccount(address)

	// Check available balance
	if account.CrossBalance.Cmp(margin) < 0 {
		return errors.New("insufficient balance for isolated margin")
	}

	// Transfer from cross to isolated
	account.CrossBalance = new(big.Int).Sub(account.CrossBalance, margin)

	// Create or update isolated position
	if account.IsolatedPositions == nil {
		account.IsolatedPositions = make(map[string]*IsolatedPosition)
	}

	isolated, exists := account.IsolatedPositions[symbol]
	if !exists {
		isolated = &IsolatedPosition{
			Symbol:     symbol,
			Margin:     big.NewInt(0),
			LastUpdate: time.Now(),
		}
		account.IsolatedPositions[symbol] = isolated
	}

	isolated.Margin = new(big.Int).Add(isolated.Margin, margin)

	// Calculate max position size based on margin and leverage
	maxLeverage := ch.maxLeverage[symbol]
	markPrice := ch.getMarkPrice(symbol)
	isolated.MaxSize = float64(isolated.Margin.Int64()) * maxLeverage / markPrice

	return nil
}

// ProcessFunding processes funding payments for all positions
func (ch *ClearingHouse) ProcessFunding() {
	ch.mu.RLock()
	symbols := make([]string, 0, len(ch.fundingRates))
	for symbol := range ch.fundingRates {
		symbols = append(symbols, symbol)
	}
	ch.mu.RUnlock()

	for _, symbol := range symbols {
		ch.processFundingForSymbol(symbol)
	}
}

// processFundingForSymbol processes funding for a specific symbol
func (ch *ClearingHouse) processFundingForSymbol(symbol string) {
	fundingRate := ch.fundingRates[symbol]
	markPrice := ch.getMarkPrice(symbol)

	ch.mu.RLock()
	accounts := make([]*PerpAccountState, 0, len(ch.perpAccounts))
	for _, account := range ch.perpAccounts {
		accounts = append(accounts, account)
	}
	ch.mu.RUnlock()

	// Process each account's positions
	for _, account := range accounts {
		account.mu.Lock()

		// Process cross margin positions
		if position, exists := account.CrossPositions[symbol]; exists {
			fundingPayment := position.Size * markPrice * fundingRate

			if position.Size > 0 { // Long pays short
				fundingPayment = -fundingPayment
			}

			position.FundingPaid += fundingPayment
			account.CrossBalance = new(big.Int).Add(
				account.CrossBalance,
				big.NewInt(int64(fundingPayment)),
			)
		}

		// Process isolated positions
		if isolated, exists := account.IsolatedPositions[symbol]; exists && isolated.Size != 0 {
			fundingPayment := isolated.Size * markPrice * fundingRate

			if isolated.Size > 0 { // Long pays short
				fundingPayment = -fundingPayment
			}

			// Deduct from isolated margin
			isolated.Margin = new(big.Int).Add(
				isolated.Margin,
				big.NewInt(int64(fundingPayment)),
			)
		}

		account.mu.Unlock()
	}
}

// UpdateOraclePrice updates the oracle price from validators
func (ch *ClearingHouse) UpdateOraclePrice(asset string, validatorPrices map[string]float64) {
	// Extract base asset from symbol (e.g., "BTC-USD" -> "BTC")
	baseAsset := asset
	if idx := strings.Index(asset, "-"); idx > 0 {
		baseAsset = asset[:idx]
	}

	ch.mu.Lock()
	oracle, exists := ch.oracles[baseAsset]
	if !exists {
		// Create oracle if it doesn't exist
		oracle = &MultiSourceOracle{
			Asset:          baseAsset,
			Sources:        make(map[string]*PriceSource),
			Weights:        make(map[string]int),
			UpdateInterval: 3 * time.Second,
		}
		ch.oracles[baseAsset] = oracle
	}
	ch.mu.Unlock()

	oracle.mu.Lock()
	defer oracle.mu.Unlock()

	// Calculate weighted median of validator prices
	prices := make([]float64, 0)
	weights := make([]float64, 0)

	for validatorID, price := range validatorPrices {
		if price > 0 {
			prices = append(prices, price)
			// Use equal weights if no validators configured
			weights = append(weights, 1.0)

			// Update source
			if oracle.Sources[validatorID] == nil {
				oracle.Sources[validatorID] = &PriceSource{
					Exchange:   validatorID,
					IsActive:   true,
					Confidence: 1.0,
				}
			}
			oracle.Sources[validatorID].Price = price
			oracle.Sources[validatorID].LastUpdate = time.Now()
		}
	}

	if len(prices) > 0 {
		oracle.IndexPrice = calculateWeightedMedian(prices, weights)
		oracle.LastUpdate = time.Now()

		// Update mark price (index + premium/discount)
		premium := ch.calculatePremium(baseAsset)
		oracle.MarkPrice = oracle.IndexPrice * (1 + premium)
	}
}

// Helper functions

func (ch *ClearingHouse) getOrCreateAccount(address string) *PerpAccountState {
	if account, exists := ch.perpAccounts[address]; exists {
		return account
	}

	account := &PerpAccountState{
		Address:           address,
		CrossBalance:      big.NewInt(0),
		CrossPositions:    make(map[string]*PerpPosition),
		CrossPnL:          big.NewInt(0),
		IsolatedPositions: make(map[string]*IsolatedPosition),
		AccountValue:      big.NewInt(0),
		MarginUsed:        big.NewInt(0),
		FreeMargin:        big.NewInt(0),
		OrderMargin:       make(map[string]*big.Int),
		LiquidationPrice:  make(map[string]float64),
		TotalVolume:       big.NewInt(0),
		TotalFees:         big.NewInt(0),
		LastActivity:      time.Now(),
	}

	ch.perpAccounts[address] = account
	ch.accountLocks[address] = &sync.RWMutex{}

	return account
}

func (ch *ClearingHouse) getAccountLock(address string) *sync.RWMutex {
	ch.mu.RLock()
	lock, exists := ch.accountLocks[address]
	ch.mu.RUnlock()

	if !exists {
		ch.mu.Lock()
		lock = &sync.RWMutex{}
		ch.accountLocks[address] = lock
		ch.mu.Unlock()
	}

	return lock
}

func (ch *ClearingHouse) performMarginCheck(account *PerpAccountState, symbol string, size float64) bool {
	markPrice := ch.getMarkPrice(symbol)
	notional := size * markPrice

	// Initial margin requirement
	initialMarginReq := notional * ch.initialMargin[symbol]

	// Check free margin
	freeMargin := new(big.Int).Sub(account.CrossBalance, account.MarginUsed)

	return freeMargin.Cmp(big.NewInt(int64(initialMarginReq))) >= 0
}

func (ch *ClearingHouse) fpgaMarginCheck(account *PerpAccountState, symbol string, size float64) bool {
	// FPGA-accelerated margin check would go here
	// For now, fallback to software implementation
	return ch.performMarginCheck(account, symbol, size)
}

func (ch *ClearingHouse) updatePosition(account *PerpAccountState, symbol string, side Side, size float64, markPrice float64) (*PerpPosition, bool) {
	position, exists := account.CrossPositions[symbol]
	isNew := false

	if !exists {
		position = &PerpPosition{
			Symbol:     symbol,
			User:       account.Address,
			OpenTime:   time.Now(),
			UpdateTime: time.Now(),
		}
		account.CrossPositions[symbol] = position
		isNew = true
	}

	// Update position
	if side == Buy {
		if position.Size < 0 {
			// Closing short
			closeSize := math.Min(size, -position.Size)
			position.RealizedPnL += (position.EntryPrice - markPrice) * closeSize
			size -= closeSize
			position.Size += closeSize
		}
		if size > 0 {
			// Opening or adding to long
			totalCost := position.Size*position.EntryPrice + size*markPrice
			position.Size += size
			position.EntryPrice = totalCost / position.Size
		}
	} else {
		if position.Size > 0 {
			// Closing long
			closeSize := math.Min(size, position.Size)
			position.RealizedPnL += (markPrice - position.EntryPrice) * closeSize
			size -= closeSize
			position.Size -= closeSize
		}
		if size > 0 {
			// Opening or adding to short
			totalCost := math.Abs(position.Size)*position.EntryPrice + size*markPrice
			position.Size -= size
			position.EntryPrice = totalCost / math.Abs(position.Size)
		}
	}

	position.MarkPrice = markPrice
	position.UpdateTime = time.Now()

	// Update unrealized PnL
	if position.Size > 0 {
		position.UnrealizedPnL = (markPrice - position.EntryPrice) * position.Size
	} else if position.Size < 0 {
		position.UnrealizedPnL = (position.EntryPrice - markPrice) * math.Abs(position.Size)
	} else {
		position.UnrealizedPnL = 0
	}

	return position, isNew
}

func (ch *ClearingHouse) updateMarginUsage(account *PerpAccountState, symbol string, position *PerpPosition) {
	markPrice := ch.getMarkPrice(symbol)
	_ = math.Abs(position.Size) * markPrice // notional value

	// Update total margin used
	totalMargin := big.NewInt(0)
	for sym, pos := range account.CrossPositions {
		price := ch.getMarkPrice(sym)
		posNotional := math.Abs(pos.Size) * price
		posMargin := posNotional * ch.maintenanceMargin[sym]
		totalMargin = new(big.Int).Add(totalMargin, big.NewInt(int64(posMargin)))
	}

	account.MarginUsed = totalMargin

	// Calculate liquidation price
	if position.Size != 0 {
		// Simplified liquidation price calculation
		// Actual implementation would be more complex
		if position.Size > 0 {
			position.LiquidationPrice = position.EntryPrice * (1 - ch.maintenanceMargin[symbol])
		} else {
			position.LiquidationPrice = position.EntryPrice * (1 + ch.maintenanceMargin[symbol])
		}
		account.LiquidationPrice[symbol] = position.LiquidationPrice
	}
}

func (ch *ClearingHouse) updateAccountMetrics(account *PerpAccountState) {
	// Calculate total account value
	accountValue := new(big.Int).Set(account.CrossBalance)

	// Add unrealized PnL from all positions
	for _, position := range account.CrossPositions {
		pnl := big.NewInt(int64(position.UnrealizedPnL))
		accountValue = new(big.Int).Add(accountValue, pnl)
	}

	account.AccountValue = accountValue

	// Calculate free margin
	account.FreeMargin = new(big.Int).Sub(accountValue, account.MarginUsed)

	// Calculate margin level
	if account.MarginUsed.Sign() > 0 {
		account.MarginLevel = float64(accountValue.Int64()) / float64(account.MarginUsed.Int64()) * 100
	} else {
		account.MarginLevel = math.Inf(1)
	}
}

func (ch *ClearingHouse) getMarkPrice(symbol string) float64 {
	// Extract base asset from symbol (e.g., "BTC-USD" -> "BTC", "BTC-PERP" -> "BTC")
	baseAsset := symbol
	if idx := strings.Index(symbol, "-"); idx > 0 {
		baseAsset = symbol[:idx]
	}

	ch.mu.RLock()
	oracle, exists := ch.oracles[baseAsset]
	ch.mu.RUnlock()

	if exists && oracle != nil {
		oracle.mu.RLock()
		price := oracle.MarkPrice
		oracle.mu.RUnlock()
		if price > 0 {
			return price
		}
	}

	// Default price if oracle not found
	return 50000 // Default BTC price for testing
}

func (ch *ClearingHouse) getValidator(validatorID string) *ValidatorOracle {
	for _, validator := range ch.validators {
		if validator.ValidatorID == validatorID {
			return validator
		}
	}
	return nil
}

func (ch *ClearingHouse) calculatePremium(asset string) float64 {
	// Calculate funding premium based on perp vs spot
	// Simplified implementation
	return 0.0001 // 0.01% premium
}

func (ch *ClearingHouse) detectFPGACapabilities() {
	// Check for FPGA acceleration availability
	// This would interface with actual FPGA drivers
	ch.fpgaEnabled = false
	ch.fpgaOrderPath = false
	ch.fpgaRiskPath = false
}

// Helper function to calculate weighted median
func calculateWeightedMedian(values []float64, weights []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort values and weights together
	type pair struct {
		value  float64
		weight float64
	}

	pairs := make([]pair, len(values))
	for i := range values {
		pairs[i] = pair{values[i], weights[i]}
	}

	// Sort by value
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].value > pairs[j].value {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Find weighted median
	totalWeight := 0.0
	for _, p := range pairs {
		totalWeight += p.weight
	}

	halfWeight := totalWeight / 2
	cumWeight := 0.0

	for _, p := range pairs {
		cumWeight += p.weight
		if cumWeight >= halfWeight {
			return p.value
		}
	}

	return pairs[len(pairs)-1].value
}

// GetAllPositions returns all positions for a symbol across all accounts
func (ch *ClearingHouse) GetAllPositions(symbol string) []*PerpPosition {
	ch.mu.RLock()
	defer ch.mu.RUnlock()

	positions := make([]*PerpPosition, 0)

	for _, account := range ch.perpAccounts {
		account.mu.RLock()

		// Check cross margin positions
		if pos, exists := account.CrossPositions[symbol]; exists && pos.Size != 0 {
			positions = append(positions, pos)
		}

		// Check isolated positions
		if isolated, exists := account.IsolatedPositions[symbol]; exists && isolated.Size != 0 {
			// Convert isolated to PerpPosition for funding calculation
			positions = append(positions, &PerpPosition{
				Symbol:     symbol,
				User:       account.Address,
				Size:       isolated.Size,
				EntryPrice: isolated.EntryPrice,
				MarkPrice:  isolated.MarkPrice,
			})
		}

		account.mu.RUnlock()
	}

	return positions
}

// ApplyFundingPayment applies a funding payment to an account
func (ch *ClearingHouse) ApplyFundingPayment(address string, symbol string, payment float64) {
	lock := ch.getAccountLock(address)
	lock.Lock()
	defer lock.Unlock()

	account := ch.getOrCreateAccount(address)

	// Apply to cross margin balance
	paymentInt := big.NewInt(int64(payment * 1e6)) // Convert to 6 decimal precision
	account.CrossBalance = new(big.Int).Add(account.CrossBalance, paymentInt)

	// Track funding paid in position
	if pos, exists := account.CrossPositions[symbol]; exists {
		pos.FundingPaid += payment
	}

	// Update account metrics
	ch.updateAccountMetrics(account)

	// Check for liquidation if balance is negative
	if account.CrossBalance.Sign() < 0 || account.MarginLevel < 100 {
		ch.triggerLiquidation(account)
	}
}

// triggerLiquidation initiates liquidation for an account
func (ch *ClearingHouse) triggerLiquidation(account *PerpAccountState) {
	if account.IsLiquidating {
		return
	}

	account.IsLiquidating = true

	// Liquidation logic would go here
	// This would close all positions and distribute to insurance fund
}

// GetFundingRate returns the current funding rate for a symbol
func (ch *ClearingHouse) GetFundingRate(symbol string) float64 {
	if ch.fundingEngine != nil {
		rate := ch.fundingEngine.GetCurrentFundingRate(symbol)
		if rate != nil {
			return rate.Rate
		}
	}

	// Fallback to stored rate
	if rate, exists := ch.fundingRates[symbol]; exists {
		return rate
	}

	return 0
}

// GetPredictedFundingRate returns the predicted next funding rate
func (ch *ClearingHouse) GetPredictedFundingRate(symbol string) float64 {
	if ch.fundingEngine != nil {
		rate := ch.fundingEngine.GetPredictedFundingRate(symbol)
		if rate != nil {
			return rate.Rate
		}
	}

	return 0
}

// GetNextFundingTime returns the next funding time
func (ch *ClearingHouse) GetNextFundingTime() time.Time {
	if ch.fundingEngine != nil {
		return ch.fundingEngine.GetNextFundingTime()
	}

	// Fallback calculation
	now := time.Now().UTC()
	hour := now.Hour()

	// Next funding time is at 00:00, 08:00, or 16:00 UTC
	fundingHours := []int{0, 8, 16}

	for _, fh := range fundingHours {
		if fh > hour {
			return time.Date(now.Year(), now.Month(), now.Day(), fh, 0, 0, 0, time.UTC)
		}
	}

	// Next day at 00:00
	tomorrow := now.AddDate(0, 0, 1)
	return time.Date(tomorrow.Year(), tomorrow.Month(), tomorrow.Day(), 0, 0, 0, 0, time.UTC)
}

// Initialize functions for margin parameters
func initMaintenanceMargins() map[string]float64 {
	return map[string]float64{
		"BTC-PERP":  0.005, // 0.5%
		"ETH-PERP":  0.01,  // 1%
		"SOL-PERP":  0.02,  // 2%
		"ARB-PERP":  0.03,  // 3%
		"AVAX-PERP": 0.03,  // 3%
		"HYPE-PERP": 0.05,  // 5% for newer assets
	}
}

func initInitialMargins() map[string]float64 {
	return map[string]float64{
		"BTC-PERP":  0.01, // 1%
		"ETH-PERP":  0.02, // 2%
		"SOL-PERP":  0.04, // 4%
		"ARB-PERP":  0.06, // 6%
		"AVAX-PERP": 0.06, // 6%
		"HYPE-PERP": 0.10, // 10% for newer assets
	}
}

func initMaxLeverages() map[string]float64 {
	return map[string]float64{
		"BTC-PERP":  100.0,
		"ETH-PERP":  50.0,
		"SOL-PERP":  25.0,
		"ARB-PERP":  20.0,
		"AVAX-PERP": 20.0,
		"HYPE-PERP": 10.0,
	}
}
