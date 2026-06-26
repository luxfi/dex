package lx

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Asset class mapping tests ---

func TestDEXToCEXAssetClassMapping(t *testing.T) {
	tests := []struct {
		dex AssetClass
		cex string
	}{
		{AssetClassCrypto, "crypto"},
		{AssetClassEquity, "us_equity"},
		{AssetClassIntlEquity, "intl_equity"},
		{AssetClassForex, "forex"},
		{AssetClassOptions, "options"},
		{AssetClassFutures, "futures"},
		{AssetClassFixedIncome, "fixed_income"},
		{AssetClassMunicipal, "municipal"},
		{AssetClassStructured, "structured"},
		{AssetClassCorporateDebt, "corporate_debt"},
		{AssetClassConsumerDebt, "consumer_debt"},
		{AssetClassRealEstate, "real_estate"},
		{AssetClassPreciousMetals, "precious_metals"},
		{AssetClassPrivateEquity, "private_equity"},
		{AssetClassVenture, "venture"},
		{AssetClassPrivateCredit, "private_credit"},
		{AssetClassCDP, "cdp"},
		{AssetClassLP, "lp"},
	}
	for _, tt := range tests {
		got := DEXToCEXAssetClass(tt.dex)
		assert.Equal(t, tt.cex, got, "DEX class %d should map to CEX %q", tt.dex, tt.cex)
	}
}

func TestCEXToDEXAssetClassMapping(t *testing.T) {
	tests := []struct {
		cex string
		dex AssetClass
	}{
		{"crypto", AssetClassCrypto},
		{"us_equity", AssetClassEquity},
		{"intl_equity", AssetClassIntlEquity},
		{"forex", AssetClassForex},
		{"options", AssetClassOptions},
		{"futures", AssetClassFutures},
		{"fixed_income", AssetClassFixedIncome},
		{"municipal", AssetClassMunicipal},
		{"structured", AssetClassStructured},
		{"corporate_debt", AssetClassCorporateDebt},
		{"consumer_debt", AssetClassConsumerDebt},
		{"real_estate", AssetClassRealEstate},
		{"precious_metals", AssetClassPreciousMetals},
		{"private_equity", AssetClassPrivateEquity},
		{"venture", AssetClassVenture},
		{"private_credit", AssetClassPrivateCredit},
		{"cdp", AssetClassCDP},
		{"lp", AssetClassLP},
	}
	for _, tt := range tests {
		got := CEXToDEXAssetClass(tt.cex)
		assert.Equal(t, tt.dex, got, "CEX %q should map to DEX class %d", tt.cex, tt.dex)
	}
}

func TestUnknownCEXClassFallsToCrypto(t *testing.T) {
	got := CEXToDEXAssetClass("unknown_class")
	assert.Equal(t, AssetClassCrypto, got)
}

func TestRoundTripAssetClassMapping(t *testing.T) {
	// Verify that DEX→CEX→DEX round-trips preserve the class
	classes := []AssetClass{
		AssetClassCrypto, AssetClassForex, AssetClassOptions,
		AssetClassFutures, AssetClassFixedIncome, AssetClassMunicipal,
		AssetClassStructured, AssetClassCorporateDebt, AssetClassConsumerDebt,
		AssetClassRealEstate, AssetClassPreciousMetals, AssetClassPrivateEquity,
		AssetClassPrivateCredit, AssetClassCDP, AssetClassLP,
	}
	for _, ac := range classes {
		cex := DEXToCEXAssetClass(ac)
		dex := CEXToDEXAssetClass(cex)
		assert.Equal(t, ac, dex, "round-trip failed for DEX class %d (CEX: %q)", ac, cex)
	}
}

// --- Reconciler tests ---

func TestReconcilerMatchDEXFirst(t *testing.T) {
	recon := NewReconciler(nil)

	dexTrade := &CrossVenueTrade{
		DEXOrderID: 1001,
		DEXTradeID: 2001,
		DEXSymbol:  "BTC-USD",
		DEXChain:   "lux-x",
		DEXTxHash:  "0xabc",
		Side:       Buy,
		Price:      50000.0,
		Size:       1.0,
		Timestamp:  time.Now(),
	}

	// Submit DEX trade first — no match yet
	result, err := recon.SubmitDEXTrade(dexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconPending, result.Status)

	dexPending, cexPending := recon.PendingCount()
	assert.Equal(t, 1, dexPending)
	assert.Equal(t, 0, cexPending)

	// Now submit matching CEX trade
	cexTrade := &CrossVenueTrade{
		CEXOrderID:   "uuid-1",
		CEXTradeID:   "uuid-2",
		CEXSymbol:    "BTC-USD",
		CEXVenue:     "lux-cex",
		DEXSymbol:    "BTC-USD",
		Side:         Buy,
		Price:        50000.0,
		Size:         1.0,
		Jurisdiction: "US",
		Timestamp:    time.Now(),
	}

	result, err = recon.SubmitCEXTrade(cexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconMatched, result.Status)

	// Both pending queues should be empty
	dexPending, cexPending = recon.PendingCount()
	assert.Equal(t, 0, dexPending)
	assert.Equal(t, 0, cexPending)

	// Merged trade should have both sides
	assert.Equal(t, uint64(1001), result.DEXOrderID)
	assert.Equal(t, "uuid-1", result.CEXOrderID)
	assert.Equal(t, "lux-cex", result.CEXVenue)
	assert.Equal(t, "0xabc", result.DEXTxHash)
	assert.Equal(t, "US", result.Jurisdiction)

	// Wait for settlement goroutine
	time.Sleep(50 * time.Millisecond)

	// Should be settled (no bridge needed for default mode)
	stats := recon.Stats()
	assert.Equal(t, 1, stats["completed"])
	assert.Equal(t, 0, stats["failed"])
}

func TestReconcilerMatchCEXFirst(t *testing.T) {
	recon := NewReconciler(nil)

	// Submit CEX trade first
	cexTrade := &CrossVenueTrade{
		CEXOrderID: "uuid-1",
		CEXSymbol:  "ETH-USD",
		DEXSymbol:  "ETH-USD",
		Side:       Sell,
		Price:      4000.0,
		Size:       10.0,
		Timestamp:  time.Now(),
	}

	result, err := recon.SubmitCEXTrade(cexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconPending, result.Status)

	// Now submit matching DEX trade
	dexTrade := &CrossVenueTrade{
		DEXOrderID: 500,
		DEXSymbol:  "ETH-USD",
		Side:       Sell,
		Price:      4000.0,
		Size:       10.0,
		Timestamp:  time.Now(),
	}

	result, err = recon.SubmitDEXTrade(dexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconMatched, result.Status)
}

func TestReconcilerNoMatchDifferentPrice(t *testing.T) {
	recon := NewReconciler(nil)

	dexTrade := &CrossVenueTrade{
		DEXSymbol: "BTC-USD",
		Side:      Buy,
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	recon.SubmitDEXTrade(dexTrade)

	cexTrade := &CrossVenueTrade{
		CEXSymbol: "BTC-USD",
		DEXSymbol: "BTC-USD",
		Side:      Buy,
		Price:     50001.0, // Different price
		Size:      1.0,
		Timestamp: time.Now(),
	}
	result, err := recon.SubmitCEXTrade(cexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconPending, result.Status)

	dexPending, cexPending := recon.PendingCount()
	assert.Equal(t, 1, dexPending)
	assert.Equal(t, 1, cexPending)
}

func TestReconcilerNoMatchDifferentSide(t *testing.T) {
	recon := NewReconciler(nil)

	dexTrade := &CrossVenueTrade{
		DEXSymbol: "BTC-USD",
		Side:      Buy,
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	recon.SubmitDEXTrade(dexTrade)

	cexTrade := &CrossVenueTrade{
		DEXSymbol: "BTC-USD",
		Side:      Sell, // Different side
		Price:     50000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	result, err := recon.SubmitCEXTrade(cexTrade)
	require.NoError(t, err)
	assert.Equal(t, ReconPending, result.Status)
}

func TestReconcilerSettlementCallbacks(t *testing.T) {
	recon := NewReconciler(nil)

	var settledTrade atomic.Pointer[CrossVenueTrade]
	recon.SetCallbacks(
		func(trade *CrossVenueTrade) error {
			settledTrade.Store(trade)
			return nil
		},
		nil,
	)

	dexTrade := &CrossVenueTrade{
		DEXSymbol: "XAU-USD",
		Side:      Buy,
		Price:     2000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	recon.SubmitDEXTrade(dexTrade)

	cexTrade := &CrossVenueTrade{
		CEXSymbol: "XAU-USD",
		DEXSymbol: "XAU-USD",
		Side:      Buy,
		Price:     2000.0,
		Size:      1.0,
		Timestamp: time.Now(),
	}
	recon.SubmitCEXTrade(cexTrade)

	// Wait for settlement
	time.Sleep(50 * time.Millisecond)

	got := settledTrade.Load()
	require.NotNil(t, got)
	// Read trade state through the reconciler's mutex.
	st, ok := recon.GetStatus(got.ID)
	require.True(t, ok)
	assert.Equal(t, ReconSettled, st.Status)
	assert.NotNil(t, st.SettledAt)
}

func TestReconcilerBridgeSettlementFails(t *testing.T) {
	// Bridge settlement without a bridge should fail
	recon := NewReconciler(nil) // nil bridge

	var failedReason atomic.Value // string, written from settle goroutine
	recon.SetCallbacks(
		nil,
		func(trade *CrossVenueTrade, reason string) {
			failedReason.Store(reason)
		},
	)

	dexTrade := &CrossVenueTrade{
		DEXSymbol:      "REAL-USD",
		Side:           Buy,
		Price:          100.0,
		Size:           10.0,
		SettlementMode: ReconciliationBridge,
		Timestamp:      time.Now(),
	}
	recon.SubmitDEXTrade(dexTrade)

	cexTrade := &CrossVenueTrade{
		DEXSymbol:      "REAL-USD",
		Side:           Buy,
		Price:          100.0,
		Size:           10.0,
		SettlementMode: ReconciliationBridge,
		Timestamp:      time.Now(),
	}
	recon.SubmitCEXTrade(cexTrade)

	time.Sleep(50 * time.Millisecond)

	got, _ := failedReason.Load().(string)
	assert.Contains(t, got, "bridge required")
	stats := recon.Stats()
	assert.Equal(t, 1, stats["failed"])
}

func TestReconcilerGetStatus(t *testing.T) {
	recon := NewReconciler(nil)

	dexTrade := &CrossVenueTrade{
		DEXSymbol: "SOL-USD",
		Side:      Buy,
		Price:     100.0,
		Size:      50.0,
		Timestamp: time.Now(),
	}
	recon.SubmitDEXTrade(dexTrade)

	cexTrade := &CrossVenueTrade{
		DEXSymbol: "SOL-USD",
		Side:      Buy,
		Price:     100.0,
		Size:      50.0,
		Timestamp: time.Now(),
	}
	result, _ := recon.SubmitCEXTrade(cexTrade)

	time.Sleep(50 * time.Millisecond)

	trade, ok := recon.GetStatus(result.ID)
	require.True(t, ok)
	assert.Equal(t, ReconSettled, trade.Status)

	// Non-existent trade
	_, ok = recon.GetStatus("nonexistent")
	assert.False(t, ok)
}

// --- Expanded asset class tests ---

func TestNewAssetClassesAreDefined(t *testing.T) {
	// Verify all 23 asset classes are distinct values
	classes := []AssetClass{
		AssetClassCrypto, AssetClassEquity, AssetClassDebt,
		AssetClassFund, AssetClassSAFE, AssetClassCommodity,
		AssetClassDerivative, AssetClassIntlEquity, AssetClassForex,
		AssetClassOptions, AssetClassFutures, AssetClassFixedIncome,
		AssetClassMunicipal, AssetClassStructured, AssetClassCorporateDebt,
		AssetClassConsumerDebt, AssetClassRealEstate, AssetClassPreciousMetals,
		AssetClassPrivateEquity, AssetClassVenture, AssetClassPrivateCredit,
		AssetClassCDP, AssetClassLP,
	}

	seen := make(map[AssetClass]bool)
	for _, ac := range classes {
		assert.False(t, seen[ac], "duplicate asset class value: %d", ac)
		seen[ac] = true
	}
	assert.Equal(t, 23, len(seen), "expected 23 unique asset classes")
}

// --- Reconciliation mode tests ---

func TestReconciliationModeValues(t *testing.T) {
	assert.Equal(t, ReconciliationMode(0), ReconciliationNone)
	assert.Equal(t, ReconciliationMode(1), ReconciliationAtomicSwap)
	assert.Equal(t, ReconciliationMode(2), ReconciliationBridge)
	assert.Equal(t, ReconciliationMode(3), ReconciliationCustodial)
	assert.Equal(t, ReconciliationMode(4), ReconciliationHybrid)
}
