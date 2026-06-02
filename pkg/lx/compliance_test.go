package lx

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- helpers ---

func accreditedProfile(userID string) *InvestorProfile {
	return &InvestorProfile{
		UserID:       userID,
		Status:       InvestorAccredited,
		Jurisdiction: "US",
		AccreditedAt: time.Now().Add(-365 * 24 * time.Hour),
		KYCVerified:  true,
		AMLCleared:   true,
		Holdings:     map[string]float64{},
	}
}

func retailProfile(userID string) *InvestorProfile {
	return &InvestorProfile{
		UserID:       userID,
		Status:       InvestorRetail,
		Jurisdiction: "US",
		KYCVerified:  true,
		AMLCleared:   true,
		Holdings:     map[string]float64{},
	}
}

func regD506cMarket() MarketConfig {
	return MarketConfig{
		Symbol:      "ACME-USD",
		AssetClass:  AssetClassEquity,
		Exemption:   ExemptionRegD506c,
		ShareClass:  "Series A Preferred",
		TickSize:    0.01,
		LotSize:     1.0,
		MinNotional: 25000.0,
		Restrictions: &TransferRestriction{
			HoldingPeriod:     365 * 24 * time.Hour,
			MaxHolders:        2000,
			MinInvestorStatus: InvestorAccredited,
		},
	}
}

// --- StandardEligibilityChecker tests ---

func TestAccreditedInvestorCanTrade(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 10, nil },
		Restrictions: &TransferRestriction{
			MinInvestorStatus: InvestorAccredited,
			MaxHolders:        2000,
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 100, 50.0)
	assert.NoError(t, err)
}

func TestRetailInvestorRejectedForRegD(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return retailProfile(userID), nil
		},
		Restrictions: &TransferRestriction{
			MinInvestorStatus: InvestorAccredited,
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 100, 50.0)
	require.Error(t, err)

	ce, ok := err.(*ComplianceError)
	require.True(t, ok)
	assert.Equal(t, ErrNotAccredited, ce.Code)
}

func TestKYCRequired(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			p := accreditedProfile(userID)
			p.KYCVerified = false
			return p, nil
		},
		Restrictions: &TransferRestriction{},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrKYCRequired, err.(*ComplianceError).Code)
}

func TestAMLRequired(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			p := accreditedProfile(userID)
			p.AMLCleared = false
			return p, nil
		},
		Restrictions: &TransferRestriction{},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrAMLRequired, err.(*ComplianceError).Code)
}

func TestJurisdictionWhitelist(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			p := accreditedProfile(userID)
			p.Jurisdiction = "CN"
			return p, nil
		},
		Restrictions: &TransferRestriction{
			JurisdictionWhitelist: []string{"US", "CA", "GB"},
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrJurisdiction, err.(*ComplianceError).Code)
}

func TestJurisdictionWhitelistAllowed(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		Restrictions: &TransferRestriction{
			JurisdictionWhitelist: []string{"US", "CA", "GB"},
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	assert.NoError(t, err)
}

func TestJurisdictionBlacklist(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			p := accreditedProfile(userID)
			p.Jurisdiction = "KP"
			return p, nil
		},
		Restrictions: &TransferRestriction{
			JurisdictionBlacklist: []string{"KP", "IR", "CU"},
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrJurisdiction, err.(*ComplianceError).Code)
}

func TestMaxHoldersBuyRejected(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil // Holdings empty = new holder
		},
		GetHolderCount: func(symbol string) (int, error) { return 2000, nil },
		Restrictions: &TransferRestriction{
			MaxHolders: 2000,
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrMaxHolders, err.(*ComplianceError).Code)
}

func TestMaxHoldersExistingHolderOK(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			p := accreditedProfile(userID)
			p.Holdings["ACME-USD"] = 50.0 // Already a holder
			return p, nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 2000, nil },
		Restrictions: &TransferRestriction{
			MaxHolders: 2000,
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	assert.NoError(t, err)
}

func TestMaxHoldersSellAlwaysOK(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 2000, nil },
		Restrictions: &TransferRestriction{
			MaxHolders: 2000,
		},
	}

	// Sell side should never be blocked by max holders
	err := checker.CanTrade("user1", "ACME-USD", Sell, 10, 100.0)
	assert.NoError(t, err)
}

func TestROFRRequired(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		Restrictions: &TransferRestriction{
			ROFRRequired: true,
		},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrROFRPending, err.(*ComplianceError).Code)

	// Sell side should also work (ROFR only blocks buy)
	// Actually, ROFR is on buy side in our impl
	err = checker.CanTrade("user1", "ACME-USD", Sell, 10, 100.0)
	assert.NoError(t, err)
}

func TestNoRestrictionsPassesAll(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		Restrictions: nil,
	}

	err := checker.CanTrade("user1", "BTC-USD", Buy, 1, 50000.0)
	assert.NoError(t, err)
}

func TestProfileLookupError(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return nil, fmt.Errorf("database unavailable")
		},
		Restrictions: &TransferRestriction{},
	}

	err := checker.CanTrade("user1", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrTransferRestricted, err.(*ComplianceError).Code)
}

func TestProfileNotFound(t *testing.T) {
	checker := &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return nil, nil
		},
		Restrictions: &TransferRestriction{},
	}

	err := checker.CanTrade("unknown", "ACME-USD", Buy, 10, 100.0)
	require.Error(t, err)
	assert.Equal(t, ErrKYCRequired, err.(*ComplianceError).Code)
}

// --- SecuritiesMarket tests ---

func TestSecuritiesMarketMinNotional(t *testing.T) {
	config := regD506cMarket()
	config.EligibilityChecker = &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 10, nil },
		Restrictions:   config.Restrictions,
	}

	sm := NewSecuritiesMarket(config)

	// Order notional = 10 * 100 = $1,000 < $25,000 minimum
	order := &AdvancedOrder{
		ID:     1,
		UserID: "user1",
		Symbol: "ACME-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  100.0,
		Size:   10.0,
	}

	_, err := sm.AddOrder(order)
	require.Error(t, err)
	ce, ok := err.(*ComplianceError)
	require.True(t, ok)
	assert.Equal(t, ErrMinNotional, ce.Code)
}

func TestSecuritiesMarketMinNotionalPasses(t *testing.T) {
	config := regD506cMarket()
	config.EligibilityChecker = &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 10, nil },
		Restrictions:   config.Restrictions,
	}

	sm := NewSecuritiesMarket(config)

	// Order notional = 500 * 100 = $50,000 >= $25,000 minimum
	order := &AdvancedOrder{
		ID:     1,
		UserID: "user1",
		Symbol: "ACME-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  100.0,
		Size:   500.0,
	}

	_, err := sm.AddOrder(order)
	assert.NoError(t, err)
}

func TestSecuritiesMarketRejectsRetailInvestor(t *testing.T) {
	config := regD506cMarket()
	config.EligibilityChecker = &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return retailProfile(userID), nil
		},
		Restrictions: config.Restrictions,
	}

	sm := NewSecuritiesMarket(config)

	order := &AdvancedOrder{
		ID:     1,
		UserID: "retail_user",
		Symbol: "ACME-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  100.0,
		Size:   500.0,
	}

	_, err := sm.AddOrder(order)
	require.Error(t, err)
	assert.Equal(t, ErrNotAccredited, err.(*ComplianceError).Code)
}

func TestSecuritiesMarketMatchWithSettlement(t *testing.T) {
	var settledTrades []Trade

	config := regD506cMarket()
	config.EligibilityChecker = &StandardEligibilityChecker{
		GetProfile: func(userID string) (*InvestorProfile, error) {
			return accreditedProfile(userID), nil
		},
		GetHolderCount: func(symbol string) (int, error) { return 10, nil },
		Restrictions:   config.Restrictions,
	}
	config.SettlementHandler = &testSettlementHandler{
		onTrade: func(trade *Trade, mc *MarketConfig) error {
			settledTrades = append(settledTrades, *trade)
			return nil
		},
	}

	sm := NewSecuritiesMarket(config)

	// Place a sell order
	sell := &AdvancedOrder{
		ID:     1,
		UserID: "seller1",
		Symbol: "ACME-USD",
		Side:   Sell,
		Type:   Limit,
		Price:  100.0,
		Size:   500.0,
	}
	_, err := sm.AddOrder(sell)
	require.NoError(t, err)

	// Place a matching buy order
	buy := &AdvancedOrder{
		ID:     2,
		UserID: "buyer1",
		Symbol: "ACME-USD",
		Side:   Buy,
		Type:   Limit,
		Price:  100.0,
		Size:   500.0,
	}
	trades, err := sm.AddOrder(buy)
	require.NoError(t, err)
	require.Len(t, trades, 1)

	// Verify settlement was called
	require.Len(t, settledTrades, 1)
	assert.Equal(t, 100.0, settledTrades[0].Price)
	assert.Equal(t, 500.0, settledTrades[0].Size)
}

func TestSecuritiesMarketSingleMatchMode(t *testing.T) {
	config := regD506cMarket()
	config.SingleMatchMode = true
	config.MinNotional = 0 // Disable for this test
	config.EligibilityChecker = nil
	config.Restrictions = nil

	sm := NewSecuritiesMarket(config)

	// Place one sell order for 50 shares only
	sell1 := &AdvancedOrder{
		ID: 1, UserID: "s1", Symbol: "ACME-USD",
		Side: Sell, Type: Limit, Price: 100.0, Size: 50.0,
	}
	sm.AdvancedOrderBook.AddOrder(sell1)

	// Buy 200 shares — only 50 available, so 150 remain after match.
	// In single-match mode, remainder should be canceled.
	buy := &AdvancedOrder{
		ID: 2, UserID: "b1", Symbol: "ACME-USD",
		Side: Buy, Type: Limit, Price: 100.0, Size: 200.0,
	}
	trades, err := sm.AddOrder(buy)
	require.NoError(t, err)
	assert.Len(t, trades, 1)
	assert.Equal(t, 50.0, trades[0].Size)

	// Single-match mode: remaining 150 should be canceled
	assert.Equal(t, StatusCanceled, buy.Status)
	assert.Equal(t, 0.0, buy.RemainingSize)
}

func TestSecuritiesMarketMaxSize(t *testing.T) {
	config := DefaultMarketConfig()
	config.Symbol = "TEST-USD"
	config.MaxSize = 1000.0

	sm := NewSecuritiesMarket(config)

	order := &AdvancedOrder{
		ID: 1, UserID: "user1", Symbol: "TEST-USD",
		Side: Buy, Type: Limit, Price: 10.0, Size: 2000.0,
	}
	_, err := sm.AddOrder(order)
	require.Error(t, err)
	assert.Equal(t, ErrMaxSize, err.(*ComplianceError).Code)
}

func TestSecuritiesMarketMinSize(t *testing.T) {
	config := DefaultMarketConfig()
	config.Symbol = "TEST-USD"
	config.MinSize = 100.0
	config.LotSize = 1.0

	sm := NewSecuritiesMarket(config)

	order := &AdvancedOrder{
		ID: 1, UserID: "user1", Symbol: "TEST-USD",
		Side: Buy, Type: Limit, Price: 10.0, Size: 50.0,
	}
	_, err := sm.AddOrder(order)
	require.Error(t, err)
	assert.Equal(t, ErrMinSize, err.(*ComplianceError).Code)
}

func TestDefaultMarketConfigIsUnregulated(t *testing.T) {
	config := DefaultMarketConfig()
	assert.False(t, config.IsRegulated())
	assert.False(t, config.HasRestrictions())
}

func TestRegDMarketConfigIsRegulated(t *testing.T) {
	config := regD506cMarket()
	assert.True(t, config.IsRegulated())
	assert.True(t, config.HasRestrictions())
}

func TestComplianceErrorMessage(t *testing.T) {
	e := &ComplianceError{
		Code:    ErrNotAccredited,
		Message: "investor not accredited",
		Detail:  InvestorAccredited,
	}
	assert.Contains(t, e.Error(), "investor not accredited")
	assert.Contains(t, e.Error(), "compliance:")

	e2 := &ComplianceError{
		Code:    ErrKYCRequired,
		Message: "KYC required",
	}
	assert.Contains(t, e2.Error(), "KYC required")
	assert.NotContains(t, e2.Error(), "detail")
}

// --- test helpers ---

type testSettlementHandler struct {
	onTrade func(trade *Trade, mc *MarketConfig) error
}

func (h *testSettlementHandler) OnTrade(trade *Trade, mc *MarketConfig) error {
	return h.onTrade(trade, mc)
}
