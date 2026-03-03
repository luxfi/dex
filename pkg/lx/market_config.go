package lx

// MarketConfig defines the trading rules and regulatory constraints for a market.
// For unregulated crypto markets, use DefaultMarketConfig().
type MarketConfig struct {
	Symbol string

	// Asset classification
	AssetClass AssetClass
	Exemption  ExemptionType
	ShareClass string // e.g. "Series A Preferred", "Class B Common"

	// Transfer restrictions (nil = no restrictions)
	Restrictions *TransferRestriction

	// Trading constraints
	MinNotional float64 // Minimum trade notional value (price * size)
	MinSize     float64 // Minimum order size in base units
	MaxSize     float64 // Maximum order size (0 = unlimited)
	TickSize    float64 // Price tick size
	LotSize     float64 // Size lot size

	// Matching behavior
	SingleMatchMode bool // If true, remove both sides after first match (private secondaries)

	// Compliance hooks (nil = no checks)
	EligibilityChecker EligibilityChecker
	SettlementHandler  SettlementHandler

	// Offering / Issuer metadata — populated for exempt securities
	OfferingType string // reg_cf, reg_d_506c, reg_a, reg_s, public
	IssuerID     string
	IssuerName   string

	// Structured product metadata — credit, real estate, private offerings
	ProductType       string  // abs, mbs, cdo, clo, reit, cdp, pe_fund, vc_fund
	CreditRating      string  // AAA, AA, A, BBB, BB, B, CCC, NR
	CollateralType    string  // residential, commercial, auto, student, mixed
	MaturityDate      string  // ISO 8601 date
	CouponRate        float64 // Annual coupon rate as decimal
	YieldToMaturity   float64 // Current YTM
	Leverage          float64 // Max leverage for the product
	CollateralRatio   float64 // Min collateral ratio (e.g. 1.50 = 150%)
	LiquidationRatio  float64 // Ratio below which liquidation triggers
	MinInvestment     float64 // Minimum investment amount
	LockupPeriod      string  // e.g. "180d", "1y"
	AccreditedOnly    bool    // Requires accredited/professional investor
	InstitutionalOnly bool    // Institutions only (QIB, etc.)

	// CEX reconciliation — links this DEX market to a CEX venue
	CEXSymbol         string // Corresponding CEX symbol (if reconciled)
	CEXVenue          string // CEX venue identifier (e.g. "lux-cex")
	ReconciliationMode ReconciliationMode // How cross-venue settlement works
}

// ReconciliationMode determines how assets are settled across CEX and DEX.
type ReconciliationMode uint8

const (
	ReconciliationNone       ReconciliationMode = iota // No cross-venue reconciliation
	ReconciliationAtomicSwap                           // On-chain atomic swap between venues
	ReconciliationBridge                               // Bridge-based transfer (lock/mint)
	ReconciliationCustodial                            // Custodial settlement via broker
	ReconciliationHybrid                               // On-chain proof + off-chain settlement
)

// DefaultMarketConfig returns a config for an unregulated crypto market.
func DefaultMarketConfig() MarketConfig {
	return MarketConfig{
		AssetClass: AssetClassCrypto,
		Exemption:  ExemptionNone,
		TickSize:   0.01,
		LotSize:    0.001,
	}
}

// FixedIncomeMarketConfig returns a config for bond/note markets.
func FixedIncomeMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:     symbol,
		AssetClass: AssetClassFixedIncome,
		TickSize:   0.001,
		LotSize:    1000, // Standard bond lot
		MinNotional: 1000,
	}
}

// StructuredProductMarketConfig returns a config for ABS/MBS/CDO markets.
func StructuredProductMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:         symbol,
		AssetClass:     AssetClassStructured,
		TickSize:       0.01,
		LotSize:        10000,
		MinNotional:    25000,
		AccreditedOnly: true,
		Restrictions: &TransferRestriction{
			MinInvestorStatus: InvestorAccredited,
		},
	}
}

// PreciousMetalsMarketConfig returns a config for XAU/XAG/XPT/XPD spot.
func PreciousMetalsMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:     symbol,
		AssetClass: AssetClassPreciousMetals,
		TickSize:   0.01,
		LotSize:    0.001,
	}
}

// RealEstateMarketConfig returns a config for tokenized real estate.
func RealEstateMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:         symbol,
		AssetClass:     AssetClassRealEstate,
		TickSize:       0.01,
		LotSize:        1,
		AccreditedOnly: true,
		Restrictions: &TransferRestriction{
			MinInvestorStatus: InvestorAccredited,
			HoldingPeriod:     365 * 24 * 3600000000000, // 1 year in nanoseconds
		},
	}
}

// PrivateEquityMarketConfig returns a config for PE/VC fund tokens.
func PrivateEquityMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:            symbol,
		AssetClass:        AssetClassPrivateEquity,
		TickSize:          0.01,
		LotSize:           1,
		AccreditedOnly:    true,
		InstitutionalOnly: false,
		SingleMatchMode:   true, // Private secondaries
		Restrictions: &TransferRestriction{
			MinInvestorStatus: InvestorQualifiedPurchaser,
			ROFRRequired:      true,
			BoardApprovalRequired: true,
		},
	}
}

// CDPMarketConfig returns a config for collateralized debt positions.
func CDPMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:     symbol,
		AssetClass: AssetClassCDP,
		TickSize:   0.01,
		LotSize:    0.001,
	}
}

// LPTokenMarketConfig returns a config for liquidity pool tokens.
func LPTokenMarketConfig(symbol string) MarketConfig {
	return MarketConfig{
		Symbol:     symbol,
		AssetClass: AssetClassLP,
		TickSize:   0.000001,
		LotSize:    0.000001,
	}
}

// IsRegulated returns true if this market trades a regulated security.
func (mc *MarketConfig) IsRegulated() bool {
	return mc.Exemption != ExemptionNone
}

// HasRestrictions returns true if transfer restrictions are configured.
func (mc *MarketConfig) HasRestrictions() bool {
	return mc.Restrictions != nil
}

// SecuritiesMarket is an AdvancedOrderBook with regulatory configuration.
type SecuritiesMarket struct {
	*AdvancedOrderBook
	Config MarketConfig
}

// NewSecuritiesMarket creates a market with full securities compliance support.
func NewSecuritiesMarket(config MarketConfig) *SecuritiesMarket {
	book := NewAdvancedOrderBook(config.Symbol)
	book.tickSize = config.TickSize
	book.lotSize = config.LotSize

	if config.LotSize == 0 {
		book.lotSize = 0.001
	}
	if config.TickSize == 0 {
		book.tickSize = 0.01
	}

	// Always enable self-trade prevention for regulated markets
	if config.IsRegulated() {
		book.enableSelfTrade = true
	}

	return &SecuritiesMarket{
		AdvancedOrderBook: book,
		Config:            config,
	}
}

// AddOrder validates compliance before placing an order.
func (sm *SecuritiesMarket) AddOrder(order *AdvancedOrder) ([]Trade, error) {
	// Pre-trade compliance check
	if sm.Config.EligibilityChecker != nil {
		if err := sm.Config.EligibilityChecker.CanTrade(
			order.UserID,
			sm.Config.Symbol,
			order.Side,
			order.Size,
			order.Price,
		); err != nil {
			order.Status = StatusRejected
			return nil, err
		}
	}

	// Min notional check
	if sm.Config.MinNotional > 0 && order.Type == Limit {
		notional := order.Price * order.Size
		if notional < sm.Config.MinNotional {
			order.Status = StatusRejected
			return nil, &ComplianceError{
				Code:    ErrMinNotional,
				Message: "order notional below minimum",
				Detail:  sm.Config.MinNotional,
			}
		}
	}

	// Max size check
	if sm.Config.MaxSize > 0 && order.Size > sm.Config.MaxSize {
		order.Status = StatusRejected
		return nil, &ComplianceError{
			Code:    ErrMaxSize,
			Message: "order size exceeds maximum",
			Detail:  sm.Config.MaxSize,
		}
	}

	// Min size check
	if sm.Config.MinSize > 0 && order.Size < sm.Config.MinSize {
		order.Status = StatusRejected
		return nil, &ComplianceError{
			Code:    ErrMinSize,
			Message: "order size below minimum",
			Detail:  sm.Config.MinSize,
		}
	}

	// Single-match mode: use IOC so unmatched qty is auto-cancelled
	if sm.Config.SingleMatchMode && order.TimeInForce != TIF_FOK {
		order.TimeInForce = TIF_IOC
	}

	// Delegate to the underlying order book
	trades, err := sm.AdvancedOrderBook.AddOrder(order)
	if err != nil {
		return nil, err
	}

	// Post-trade settlement
	if sm.Config.SettlementHandler != nil && len(trades) > 0 {
		for i := range trades {
			if sErr := sm.Config.SettlementHandler.OnTrade(&trades[i], &sm.Config); sErr != nil {
				// Log but don't fail the match — trade already happened.
				// Settlement errors are handled asynchronously.
				_ = sErr
			}
		}
	}

	// Single-match mode: IOC already cancelled the remainder via the engine.
	// Ensure status reflects cancellation if there was a partial fill.
	if sm.Config.SingleMatchMode && len(trades) > 0 && order.ExecutedSize < order.Size {
		order.Status = StatusCanceled
		order.RemainingSize = 0
	}

	return trades, nil
}
