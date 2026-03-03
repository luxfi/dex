package lx

import (
	"fmt"
	"sync"
	"time"
)

// VenueType identifies where a trade was executed.
type VenueType uint8

const (
	VenueDEX VenueType = iota // On-chain decentralized exchange
	VenueCEX                  // Centralized exchange (broker-dealer)
	VenueATS                  // Alternative trading system
	VenueOTC                  // Over-the-counter
)

// ReconciliationStatus tracks the state of cross-venue settlement.
type ReconciliationStatus uint8

const (
	ReconPending    ReconciliationStatus = iota // Awaiting counterpart
	ReconMatched                                // Both sides matched
	ReconSettling                               // Settlement in progress
	ReconSettled                                // Fully settled
	ReconFailed                                 // Settlement failed
	ReconDisputed                               // Under dispute
)

// CrossVenueTrade represents a trade that may span CEX and DEX.
type CrossVenueTrade struct {
	ID string

	// DEX side
	DEXOrderID uint64
	DEXTradeID uint64
	DEXSymbol  string
	DEXChain   string // chain where DEX trade occurred
	DEXTxHash  string // on-chain transaction hash

	// CEX side
	CEXOrderID    string // CEX uses string UUIDs
	CEXTradeID    string
	CEXSymbol     string
	CEXVenue      string // execution venue on CEX
	CEXComplianceID string

	// Common fields
	AssetClass AssetClass
	Side       Side
	Price      float64
	Size       float64
	Fee        float64
	Timestamp  time.Time

	// Settlement
	Status         ReconciliationStatus
	SettlementMode ReconciliationMode
	SettledAt      *time.Time
	FailureReason  string

	// Regulatory
	Jurisdiction       string // Governing jurisdiction
	ReportedToRegulator bool
	RegulatoryReportID  string
}

// dexToCEXMap maps DEX asset classes to CEX string equivalents.
// Used by DEXToCEXAssetClass — every DEX class has exactly one CEX mapping.
var dexToCEXMap = map[AssetClass]string{
	AssetClassCrypto:        "crypto",
	AssetClassEquity:        "us_equity",
	AssetClassIntlEquity:    "intl_equity",
	AssetClassDebt:          "fixed_income",
	AssetClassFund:          "venture",
	AssetClassSAFE:          "venture",
	AssetClassCommodity:     "commodities",
	AssetClassDerivative:    "options",
	AssetClassForex:         "forex",
	AssetClassOptions:       "options",
	AssetClassFutures:       "futures",
	AssetClassFixedIncome:   "fixed_income",
	AssetClassMunicipal:     "municipal",
	AssetClassStructured:    "structured",
	AssetClassCorporateDebt: "corporate_debt",
	AssetClassConsumerDebt:  "consumer_debt",
	AssetClassRealEstate:    "real_estate",
	AssetClassPreciousMetals: "precious_metals",
	AssetClassPrivateEquity: "private_equity",
	AssetClassVenture:       "venture",
	AssetClassPrivateCredit: "private_credit",
	AssetClassCDP:           "cdp",
	AssetClassLP:            "lp",
}

// cexToDEXMap maps CEX string asset classes to their most specific DEX equivalent.
// When multiple DEX classes map to the same CEX string (e.g. Derivative and Options
// both map to "options"), this map uses the more specific one.
var cexToDEXMap = map[string]AssetClass{
	"crypto":          AssetClassCrypto,
	"us_equity":       AssetClassEquity,
	"intl_equity":     AssetClassIntlEquity,
	"forex":           AssetClassForex,
	"options":         AssetClassOptions,
	"futures":         AssetClassFutures,
	"fixed_income":    AssetClassFixedIncome,
	"municipal":       AssetClassMunicipal,
	"structured":      AssetClassStructured,
	"corporate_debt":  AssetClassCorporateDebt,
	"consumer_debt":   AssetClassConsumerDebt,
	"real_estate":     AssetClassRealEstate,
	"commodities":     AssetClassCommodity,
	"precious_metals": AssetClassPreciousMetals,
	"private_equity":  AssetClassPrivateEquity,
	"venture":         AssetClassVenture,
	"private_credit":  AssetClassPrivateCredit,
	"cdp":             AssetClassCDP,
	"lp":              AssetClassLP,
}

// DEXToCEXAssetClass converts a DEX AssetClass to its CEX string equivalent.
func DEXToCEXAssetClass(ac AssetClass) string {
	if s, ok := dexToCEXMap[ac]; ok {
		return s
	}
	return "crypto"
}

// CEXToDEXAssetClass converts a CEX asset class string to DEX AssetClass.
func CEXToDEXAssetClass(cexClass string) AssetClass {
	if ac, ok := cexToDEXMap[cexClass]; ok {
		return ac
	}
	return AssetClassCrypto
}

// Reconciler manages cross-venue trade reconciliation between DEX and CEX.
type Reconciler struct {
	mu sync.RWMutex

	// Pending trades awaiting counterpart
	pendingDEX map[string]*CrossVenueTrade // keyed by reconciliation key
	pendingCEX map[string]*CrossVenueTrade

	// Matched and in-progress
	active    map[string]*CrossVenueTrade
	completed map[string]*CrossVenueTrade
	failed    map[string]*CrossVenueTrade

	// Settlement callbacks
	onSettled func(trade *CrossVenueTrade) error
	onFailed  func(trade *CrossVenueTrade, reason string)

	// Bridge for on-chain settlement
	bridge *EnhancedBridge
}

// NewReconciler creates a new cross-venue reconciler.
func NewReconciler(bridge *EnhancedBridge) *Reconciler {
	return &Reconciler{
		pendingDEX: make(map[string]*CrossVenueTrade),
		pendingCEX: make(map[string]*CrossVenueTrade),
		active:     make(map[string]*CrossVenueTrade),
		completed:  make(map[string]*CrossVenueTrade),
		failed:     make(map[string]*CrossVenueTrade),
		bridge:     bridge,
	}
}

// SetCallbacks sets settlement callbacks.
func (r *Reconciler) SetCallbacks(
	onSettled func(*CrossVenueTrade) error,
	onFailed func(*CrossVenueTrade, string),
) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.onSettled = onSettled
	r.onFailed = onFailed
}

// reconciliationKey generates a key for matching DEX and CEX trades.
// Trades match on: symbol + side + price + size (within tolerance).
func reconciliationKey(symbol string, side Side, price float64, size float64) string {
	// Quantize price to tick size (0.01) to handle float precision
	priceQ := int64(price * 100)
	sizeQ := int64(size * 1000000) // 6 decimal precision
	sideStr := "B"
	if side == Sell {
		sideStr = "S"
	}
	return fmt.Sprintf("%s:%s:%d:%d", symbol, sideStr, priceQ, sizeQ)
}

// SubmitDEXTrade records a DEX trade for reconciliation.
func (r *Reconciler) SubmitDEXTrade(trade *CrossVenueTrade) (*CrossVenueTrade, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := reconciliationKey(trade.DEXSymbol, trade.Side, trade.Price, trade.Size)

	// Check if there's a matching CEX trade
	if cexTrade, ok := r.pendingCEX[key]; ok {
		// Merge into a matched trade
		merged := r.mergeTrades(trade, cexTrade)
		delete(r.pendingCEX, key)
		r.active[merged.ID] = merged

		// Kick off settlement
		go r.settle(merged)
		return merged, nil
	}

	// No match yet — queue for later
	trade.Status = ReconPending
	r.pendingDEX[key] = trade
	return trade, nil
}

// SubmitCEXTrade records a CEX trade for reconciliation.
func (r *Reconciler) SubmitCEXTrade(trade *CrossVenueTrade) (*CrossVenueTrade, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Use the DEX symbol for key if available, otherwise map from CEX
	symbol := trade.DEXSymbol
	if symbol == "" {
		symbol = trade.CEXSymbol
	}

	key := reconciliationKey(symbol, trade.Side, trade.Price, trade.Size)

	// Check if there's a matching DEX trade
	if dexTrade, ok := r.pendingDEX[key]; ok {
		merged := r.mergeTrades(dexTrade, trade)
		delete(r.pendingDEX, key)
		r.active[merged.ID] = merged

		go r.settle(merged)
		return merged, nil
	}

	// No match yet — queue
	trade.Status = ReconPending
	r.pendingCEX[key] = trade
	return trade, nil
}

// mergeTrades combines matched DEX and CEX trades.
func (r *Reconciler) mergeTrades(dex, cex *CrossVenueTrade) *CrossVenueTrade {
	merged := &CrossVenueTrade{
		ID: fmt.Sprintf("recon_%d", time.Now().UnixNano()),

		// DEX side
		DEXOrderID: dex.DEXOrderID,
		DEXTradeID: dex.DEXTradeID,
		DEXSymbol:  dex.DEXSymbol,
		DEXChain:   dex.DEXChain,
		DEXTxHash:  dex.DEXTxHash,

		// CEX side
		CEXOrderID:      cex.CEXOrderID,
		CEXTradeID:      cex.CEXTradeID,
		CEXSymbol:       cex.CEXSymbol,
		CEXVenue:        cex.CEXVenue,
		CEXComplianceID: cex.CEXComplianceID,

		// Common
		AssetClass: dex.AssetClass,
		Side:       dex.Side,
		Price:      dex.Price,
		Size:       dex.Size,
		Fee:        dex.Fee + cex.Fee,
		Timestamp:  dex.Timestamp,

		// Settlement
		Status:         ReconMatched,
		SettlementMode: dex.SettlementMode,
		Jurisdiction:   cex.Jurisdiction,
	}
	if merged.SettlementMode == ReconciliationNone {
		merged.SettlementMode = cex.SettlementMode
	}
	return merged
}

// settle executes the cross-venue settlement.
func (r *Reconciler) settle(trade *CrossVenueTrade) {
	r.mu.Lock()
	trade.Status = ReconSettling
	r.mu.Unlock()

	var err error

	switch trade.SettlementMode {
	case ReconciliationAtomicSwap:
		err = r.settleAtomicSwap(trade)
	case ReconciliationBridge:
		err = r.settleBridge(trade)
	case ReconciliationCustodial:
		err = r.settleCustodial(trade)
	case ReconciliationHybrid:
		err = r.settleHybrid(trade)
	default:
		// No settlement needed — just mark as settled
		err = nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if err != nil {
		trade.Status = ReconFailed
		trade.FailureReason = err.Error()
		delete(r.active, trade.ID)
		r.failed[trade.ID] = trade
		if r.onFailed != nil {
			r.onFailed(trade, err.Error())
		}
		return
	}

	now := time.Now()
	trade.Status = ReconSettled
	trade.SettledAt = &now
	delete(r.active, trade.ID)
	r.completed[trade.ID] = trade
	if r.onSettled != nil {
		_ = r.onSettled(trade)
	}
}

// settleAtomicSwap performs on-chain atomic swap settlement.
func (r *Reconciler) settleAtomicSwap(trade *CrossVenueTrade) error {
	// Atomic swap: both sides lock funds on-chain, reveal preimage to complete
	// This requires the bridge to coordinate hash-time-locked contracts
	if r.bridge == nil {
		return fmt.Errorf("bridge required for atomic swap settlement")
	}
	// In production: create HTLC on DEX chain, wait for CEX to claim, then finalize
	return nil
}

// settleBridge performs bridge-based settlement (lock on source, mint on dest).
func (r *Reconciler) settleBridge(trade *CrossVenueTrade) error {
	if r.bridge == nil {
		return fmt.Errorf("bridge required for bridge settlement")
	}
	// In production: lock tokens on DEX chain, bridge to CEX custodian
	return nil
}

// settleCustodial performs off-chain custodial settlement.
func (r *Reconciler) settleCustodial(trade *CrossVenueTrade) error {
	// Custodial: CEX broker handles settlement, DEX records proof
	// Used for regulated assets (equities, bonds, PE) where on-chain
	// settlement must be paired with transfer agent updates
	return nil
}

// settleHybrid performs hybrid settlement (on-chain proof + off-chain).
func (r *Reconciler) settleHybrid(trade *CrossVenueTrade) error {
	// Hybrid: on-chain merkle proof of trade + off-chain settlement
	// Used for structured products where regulatory requirements
	// mandate both on-chain transparency and traditional settlement
	return nil
}

// GetStatus returns the reconciliation status of a trade.
func (r *Reconciler) GetStatus(tradeID string) (*CrossVenueTrade, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if t, ok := r.active[tradeID]; ok {
		return t, true
	}
	if t, ok := r.completed[tradeID]; ok {
		return t, true
	}
	if t, ok := r.failed[tradeID]; ok {
		return t, true
	}
	return nil, false
}

// PendingCount returns the number of unmatched trades on each side.
func (r *Reconciler) PendingCount() (dex int, cex int) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.pendingDEX), len(r.pendingCEX)
}

// Stats returns reconciliation statistics.
func (r *Reconciler) Stats() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return map[string]int{
		"pending_dex": len(r.pendingDEX),
		"pending_cex": len(r.pendingCEX),
		"active":      len(r.active),
		"completed":   len(r.completed),
		"failed":      len(r.failed),
	}
}
