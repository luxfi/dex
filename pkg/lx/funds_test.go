package lx

import (
	"math/big"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Route risk profile tests ---

func TestRiskProfileDEXInternal(t *testing.T) {
	rp := GetRiskProfile(RouteDEXInternal)
	assert.Equal(t, "none", rp.CounterpartyRisk)
	assert.True(t, rp.SmartContractRisk)
	assert.False(t, rp.CustodialRisk)
	assert.False(t, rp.BridgeRisk)
	assert.Equal(t, "instant", rp.SettlementTime)
	assert.Empty(t, rp.ComplianceGates)
}

func TestRiskProfileCEXInternal(t *testing.T) {
	rp := GetRiskProfile(RouteCEXInternal)
	assert.Equal(t, "custodial", rp.CounterpartyRisk)
	assert.True(t, rp.CustodialRisk)
	assert.False(t, rp.SmartContractRisk)
	assert.Equal(t, "instant", rp.SettlementTime)
}

func TestRiskProfileDEXToDEX(t *testing.T) {
	rp := GetRiskProfile(RouteDEXToDEX)
	assert.True(t, rp.SmartContractRisk)
	assert.True(t, rp.BridgeRisk)
	assert.Contains(t, rp.SettlementTime, "15min")
}

func TestRiskProfileCEXToCEX(t *testing.T) {
	rp := GetRiskProfile(RouteCEXToCEX)
	assert.True(t, rp.CustodialRisk)
	assert.Contains(t, rp.SettlementTime, "T+1")
	assert.Contains(t, rp.ComplianceGates, "kyc")
	assert.Contains(t, rp.ComplianceGates, "aml")
}

func TestRiskProfileDEXToCEX(t *testing.T) {
	rp := GetRiskProfile(RouteDEXToCEX)
	assert.True(t, rp.SmartContractRisk)
	assert.True(t, rp.CustodialRisk)
	assert.Contains(t, rp.ComplianceGates, "source_verification")
	assert.Contains(t, rp.ComplianceGates, "address_whitelist")
}

func TestRiskProfileCEXToDEX(t *testing.T) {
	rp := GetRiskProfile(RouteCEXToDEX)
	assert.True(t, rp.CustodialRisk)
	assert.True(t, rp.SmartContractRisk)
	assert.Contains(t, rp.ComplianceGates, "2fa")
	assert.Contains(t, rp.ComplianceGates, "withdrawal_limit")
}

func TestRiskProfileExternalIn(t *testing.T) {
	rp := GetRiskProfile(RouteExternalIn)
	assert.Equal(t, "banking", rp.CounterpartyRisk)
	assert.Contains(t, rp.ComplianceGates, "source_of_funds")
	assert.Contains(t, rp.ComplianceGates, "bank_verification")
}

func TestRiskProfileExternalOut(t *testing.T) {
	rp := GetRiskProfile(RouteExternalOut)
	assert.Equal(t, "banking", rp.CounterpartyRisk)
	assert.Contains(t, rp.ComplianceGates, "travel_rule")
}

// --- FundsManager tests ---

func TestFundsManagerInitiateDeposit(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction:   FundsDeposit,
		Route:       RouteDEXInternal,
		SourceVenue: VenueKindDEX,
		SourceID:    "0xabc",
		Asset:       "USDC",
		Amount:      big.NewInt(1000000), // 1 USDC (6 decimals)
	}

	result, err := fm.Initiate(m)
	require.NoError(t, err)
	assert.Equal(t, FundsStatusApproved, result.Status)
	assert.NotEmpty(t, result.ID)
	assert.Equal(t, RiskLow, result.Risk)
}

func TestFundsManagerInitiateWithdrawal(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction:   FundsWithdraw,
		Route:       RouteCEXToDEX,
		SourceVenue: VenueKindCEX,
		SourceID:    "acct-123",
		DestVenue:   VenueKindDEX,
		DestID:      "0xdef",
		DestChain:   "lux-x",
		Asset:       "BTC",
		Amount:      big.NewInt(100000000), // 1 BTC (8 decimals)
	}

	result, err := fm.Initiate(m)
	require.NoError(t, err)
	assert.Equal(t, FundsStatusApproved, result.Status)
	assert.Equal(t, RiskMedium, result.Risk) // Cross-venue = medium
}

func TestFundsManagerRejectsZeroAmount(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(0),
	}

	_, err := fm.Initiate(m)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "positive")
}

func TestFundsManagerRejectsMissingAsset(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Amount:    big.NewInt(1000),
	}

	_, err := fm.Initiate(m)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "asset")
}

func TestFundsManagerDailyWithdrawalLimit(t *testing.T) {
	fm := NewFundsManager(nil)
	fm.SetDailyLimit("acct-1", big.NewInt(5000))

	// First withdrawal: 3000 — OK
	m1 := &FundsMovement{
		Direction: FundsWithdraw,
		Route:     RouteCEXToDEX,
		SourceID:  "acct-1",
		DestID:    "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(3000),
	}
	_, err := fm.Initiate(m1)
	require.NoError(t, err)

	// Second withdrawal: 3000 — exceeds limit (3000 + 3000 > 5000)
	m2 := &FundsMovement{
		Direction: FundsWithdraw,
		Route:     RouteCEXToDEX,
		SourceID:  "acct-1",
		DestID:    "0xdef",
		Asset:     "USDC",
		Amount:    big.NewInt(3000),
	}
	_, err = fm.Initiate(m2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "daily withdrawal limit")
}

func TestFundsManagerNoLimitNoRestriction(t *testing.T) {
	fm := NewFundsManager(nil)
	// No limit set — any amount should work

	m := &FundsMovement{
		Direction: FundsWithdraw,
		Route:     RouteCEXToDEX,
		SourceID:  "acct-1",
		DestID:    "0xabc",
		Asset:     "BTC",
		Amount:    big.NewInt(999999999),
	}
	_, err := fm.Initiate(m)
	require.NoError(t, err)
}

func TestFundsManagerPreCheckHook(t *testing.T) {
	fm := NewFundsManager(nil)

	complianceChecked := false
	fm.SetHooks(
		func(m *FundsMovement) error {
			complianceChecked = true
			return nil
		},
		nil, nil,
	)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "ETH",
		Amount:    big.NewInt(1000),
	}
	_, err := fm.Initiate(m)
	require.NoError(t, err)
	assert.True(t, complianceChecked)
}

func TestFundsManagerPreCheckRejection(t *testing.T) {
	fm := NewFundsManager(nil)

	fm.SetHooks(
		func(m *FundsMovement) error {
			if m.AMLFlag {
				return assert.AnError
			}
			return nil
		},
		nil, nil,
	)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "ETH",
		Amount:    big.NewInt(1000),
		AMLFlag:   true,
	}
	_, err := fm.Initiate(m)
	require.Error(t, err)
}

func TestFundsManagerExecute(t *testing.T) {
	fm := NewFundsManager(nil)

	var completed bool
	fm.SetHooks(nil,
		func(m *FundsMovement) { completed = true },
		nil,
	)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(5000),
	}
	result, err := fm.Initiate(m)
	require.NoError(t, err)

	err = fm.Execute(result.ID)
	require.NoError(t, err)
	assert.True(t, completed)

	// Verify status
	final, ok := fm.GetMovement(result.ID)
	require.True(t, ok)
	assert.Equal(t, FundsStatusCompleted, final.Status)
	assert.NotNil(t, final.CompletedAt)
}

func TestFundsManagerExecuteNotFound(t *testing.T) {
	fm := NewFundsManager(nil)
	err := fm.Execute("nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestFundsManagerExecuteNotApproved(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(1000),
	}
	result, _ := fm.Initiate(m)

	// Execute once
	fm.Execute(result.ID)

	// Try to execute again — not in approved state
	err := fm.Execute(result.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not in approved state")
}

func TestFundsManagerDEXToDEXRequiresBridge(t *testing.T) {
	fm := NewFundsManager(nil) // nil bridge

	var failReason string
	fm.SetHooks(nil, nil,
		func(m *FundsMovement, reason string) { failReason = reason },
	)

	m := &FundsMovement{
		Direction:   FundsWithdraw,
		Route:       RouteDEXToDEX,
		SourceVenue: VenueKindDEX,
		SourceID:    "0xabc",
		SourceChain: "lux-x",
		DestVenue:   VenueKindDEX,
		DestID:      "0xdef",
		DestChain:   "ethereum",
		Asset:       "USDC",
		Amount:      big.NewInt(1000),
	}
	result, _ := fm.Initiate(m)
	err := fm.Execute(result.ID)
	require.Error(t, err)
	assert.Contains(t, failReason, "bridge required")
}

func TestFundsManagerGetAccountMovements(t *testing.T) {
	fm := NewFundsManager(nil)

	for i := 0; i < 3; i++ {
		m := &FundsMovement{
			Direction: FundsDeposit,
			Route:     RouteDEXInternal,
			SourceID:  "acct-A",
			Asset:     "USDC",
			Amount:    big.NewInt(int64(1000 * (i + 1))),
		}
		fm.Initiate(m)
	}

	movements := fm.GetAccountMovements("acct-A")
	assert.Len(t, movements, 3)
}

// --- Risk classification tests ---

func TestRiskClassificationSanctionsIsCritical(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction:     FundsDeposit,
		Route:         RouteDEXInternal,
		SourceID:      "0xabc",
		Asset:         "USDC",
		Amount:        big.NewInt(100),
		SanctionsFlag: true,
	}
	result, _ := fm.Initiate(m)
	assert.Equal(t, RiskCritical, result.Risk)
}

func TestRiskClassificationAMLIsCritical(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(100),
		AMLFlag:   true,
	}
	result, _ := fm.Initiate(m)
	assert.Equal(t, RiskCritical, result.Risk)
}

func TestRiskClassificationLargeAmountCrossVenue(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsWithdraw,
		Route:     RouteCEXToDEX,
		SourceID:  "acct-1",
		DestID:    "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(50000_000000), // 50K USDC
	}
	result, _ := fm.Initiate(m)
	assert.Equal(t, RiskHigh, result.Risk)
}

func TestRiskClassificationSmallInternalIsLow(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsDeposit,
		Route:     RouteDEXInternal,
		SourceID:  "0xabc",
		Asset:     "USDC",
		Amount:    big.NewInt(100),
	}
	result, _ := fm.Initiate(m)
	assert.Equal(t, RiskLow, result.Risk)
}

func TestRiskClassificationSmallCrossVenueIsMedium(t *testing.T) {
	fm := NewFundsManager(nil)

	m := &FundsMovement{
		Direction: FundsWithdraw,
		Route:     RouteDEXToCEX,
		SourceID:  "0xabc",
		DestID:    "acct-1",
		Asset:     "USDC",
		Amount:    big.NewInt(500), // Small amount but cross-venue
	}
	result, _ := fm.Initiate(m)
	assert.Equal(t, RiskMedium, result.Risk)
}

// --- Venue kind / route constants ---

func TestVenueKindValues(t *testing.T) {
	assert.Equal(t, VenueKind(0), VenueKindDEX)
	assert.Equal(t, VenueKind(1), VenueKindCEX)
}

func TestAllRoutesHaveRiskProfiles(t *testing.T) {
	routes := []FundsRoute{
		RouteDEXInternal, RouteCEXInternal,
		RouteDEXToDEX, RouteCEXToCEX,
		RouteDEXToCEX, RouteCEXToDEX,
		RouteExternalIn, RouteExternalOut,
	}
	for _, r := range routes {
		rp := GetRiskProfile(r)
		assert.NotEqual(t, "unknown", rp.CounterpartyRisk,
			"route %d should have a defined risk profile", r)
	}
}
