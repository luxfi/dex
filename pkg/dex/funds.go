package dex

import (
	"fmt"
	"math/big"
	"sync"
	"time"
)

// VenueKind distinguishes the fundamental type of trading venue.
// CEX and DEX have completely different deposit/withdrawal mechanics,
// risk profiles, and settlement times.
type VenueKind uint8

const (
	VenueKindDEX VenueKind = iota // On-chain, non-custodial
	VenueKindCEX                  // Custodial, regulated
)

// FundsDirection is the direction of a funds movement.
type FundsDirection uint8

const (
	FundsDeposit  FundsDirection = iota // Into a venue
	FundsWithdraw                       // Out of a venue
)

// FundsRoute describes the source→destination pair for a funds movement.
type FundsRoute uint8

const (
	RouteDEXInternal FundsRoute = iota // DEX → same DEX (internal: margin, vault, lending)
	RouteCEXInternal                   // CEX → same CEX (account transfer, margin)
	RouteDEXToDEX                      // DEX → different DEX (cross-chain bridge)
	RouteCEXToCEX                      // CEX → different CEX (broker transfer)
	RouteDEXToCEX                      // DEX → CEX (on-chain withdraw → custodial deposit)
	RouteCEXToDEX                      // CEX → DEX (custodial withdraw → on-chain deposit)
	RouteExternalIn                    // Fiat/external → venue (wire, ACH, card)
	RouteExternalOut                   // Venue → fiat/external (wire out)
)

// FundsStatus tracks the lifecycle of a funds movement.
type FundsStatus uint8

const (
	FundsStatusPending    FundsStatus = iota // Created, awaiting processing
	FundsStatusApproved                      // Validated, ready to execute
	FundsStatusProcessing                    // Execution in progress
	FundsStatusConfirming                    // Awaiting on-chain confirmations (DEX)
	FundsStatusSettling                      // Settlement in progress (CEX T+1/T+2)
	FundsStatusCompleted                     // Fully settled
	FundsStatusFailed                        // Failed (insufficient funds, limit exceeded, etc.)
	FundsStatusCancelled                     // Cancelled by user or system
)

// RiskLevel classifies the risk of a funds movement.
type RiskLevel uint8

const (
	RiskLow      RiskLevel = iota // Internal, small amounts, known counterparty
	RiskMedium                    // Cross-venue, moderate amounts
	RiskHigh                      // Large amounts, cross-venue/external
	RiskCritical                  // Requires manual review
)

// FundsMovement is the universal representation of any deposit, withdrawal,
// or transfer across CEX and DEX venues.
type FundsMovement struct {
	ID        string
	Direction FundsDirection
	Route     FundsRoute

	// Source
	SourceVenue VenueKind
	SourceID    string // Venue-specific identifier (CEX account ID, DEX address)
	SourceChain string // For DEX: chain ID. For CEX: empty.

	// Destination
	DestVenue VenueKind
	DestID    string // Venue-specific identifier
	DestChain string // For DEX: chain ID. For CEX: empty.

	// Asset
	Asset     string   // Symbol (e.g. "USDC", "BTC", "XAU")
	Amount    *big.Int // In smallest unit (wei, satoshi, etc.)
	Fee       *big.Int // Total fee charged
	NetAmount *big.Int // Amount after fee

	// Status
	Status      FundsStatus
	Risk        RiskLevel
	CreatedAt   time.Time
	UpdatedAt   time.Time
	CompletedAt *time.Time

	// On-chain (DEX movements)
	SourceTxHash  string // Source chain transaction
	DestTxHash    string // Destination chain transaction
	Confirmations int    // Current confirmations
	RequiredConfs int    // Required confirmations

	// Settlement (CEX movements)
	SettlementType string     // instant, t1, t2, wire
	SettlementDate *time.Time // Expected settlement date
	BrokerRef      string     // Broker reference number

	// Error tracking
	FailureReason string
	RetryCount    int
	MaxRetries    int
}

// RiskProfile describes the risk characteristics of a specific route.
type RiskProfile struct {
	Route             FundsRoute
	CounterpartyRisk  string // "none" (DEX), "custodial" (CEX), "bridge" (cross-chain)
	SmartContractRisk bool   // True for DEX movements
	CustodialRisk     bool   // True for CEX movements
	BridgeRisk        bool   // True for cross-chain
	SettlementTime    string // "instant", "~15min", "T+1", "T+2", "1-5 business days"
	TypicalFees       string // Human-readable fee description
}

// routeRiskProfiles defines the risk characteristics for each route.
var routeRiskProfiles = map[FundsRoute]RiskProfile{
	RouteDEXInternal: {
		Route:             RouteDEXInternal,
		CounterpartyRisk:  "none",
		SmartContractRisk: true,
		SettlementTime:    "instant",
		TypicalFees:       "gas only",
	},
	RouteCEXInternal: {
		Route:            RouteCEXInternal,
		CounterpartyRisk: "custodial",
		CustodialRisk:    true,
		SettlementTime:   "instant",
		TypicalFees:      "none",
	},
	RouteDEXToDEX: {
		Route:             RouteDEXToDEX,
		CounterpartyRisk:  "bridge",
		SmartContractRisk: true,
		BridgeRisk:        true,
		SettlementTime:    "~15min (bridge confirmations)",
		TypicalFees:       "bridge fee (0.1-0.3%) + gas",
	},
	RouteCEXToCEX: {
		Route:            RouteCEXToCEX,
		CounterpartyRisk: "custodial",
		CustodialRisk:    true,
		SettlementTime:   "T+1 to T+2",
		TypicalFees:      "broker transfer fee",
	},
	RouteDEXToCEX: {
		Route:             RouteDEXToCEX,
		CounterpartyRisk:  "custodial",
		SmartContractRisk: true,
		CustodialRisk:     true,
		SettlementTime:    "~30min (on-chain confirm + CEX credit)",
		TypicalFees:       "gas + CEX deposit fee (often free)",
	},
	RouteCEXToDEX: {
		Route:             RouteCEXToDEX,
		CounterpartyRisk:  "custodial",
		CustodialRisk:     true,
		SmartContractRisk: true,
		SettlementTime:    "~15min (CEX processing + on-chain confirm)",
		TypicalFees:       "CEX withdrawal fee + gas",
	},
	RouteExternalIn: {
		Route:            RouteExternalIn,
		CounterpartyRisk: "banking",
		CustodialRisk:    true,
		SettlementTime:   "1-5 business days (wire/ACH)",
		TypicalFees:      "wire fee ($25-50) or free (ACH)",
	},
	RouteExternalOut: {
		Route:            RouteExternalOut,
		CounterpartyRisk: "banking",
		CustodialRisk:    true,
		SettlementTime:   "1-5 business days (wire/ACH)",
		TypicalFees:      "wire fee ($25-50) or free (ACH)",
	},
}

// GetRiskProfile returns the risk profile for a given route.
func GetRiskProfile(route FundsRoute) RiskProfile {
	if rp, ok := routeRiskProfiles[route]; ok {
		return rp
	}
	return RiskProfile{Route: route, CounterpartyRisk: "unknown"}
}

// FundsManager coordinates deposits, withdrawals, and transfers across venues.
type FundsManager struct {
	mu sync.RWMutex

	movements map[string]*FundsMovement
	byAccount map[string][]*FundsMovement // accountID → movements

	// Limits
	dailyWithdrawalLimits map[string]*big.Int // accountID → daily limit
	dailyWithdrawn        map[string]*big.Int // accountID → amount withdrawn today
	lastReset             time.Time

	// Hooks
	onComplete func(m *FundsMovement) // Post-settlement callback
	onFail     func(m *FundsMovement, reason string)

	// Bridge for cross-chain movements
	bridge *EnhancedBridge
}

// NewFundsManager creates a new funds manager.
func NewFundsManager(bridge *EnhancedBridge) *FundsManager {
	return &FundsManager{
		movements:             make(map[string]*FundsMovement),
		byAccount:             make(map[string][]*FundsMovement),
		dailyWithdrawalLimits: make(map[string]*big.Int),
		dailyWithdrawn:        make(map[string]*big.Int),
		lastReset:             time.Now(),
		bridge:                bridge,
	}
}

// SetHooks sets lifecycle callbacks.
func (fm *FundsManager) SetHooks(
	onComplete func(*FundsMovement),
	onFail func(*FundsMovement, string),
) {
	fm.mu.Lock()
	defer fm.mu.Unlock()
	fm.onComplete = onComplete
	fm.onFail = onFail
}

// SetDailyLimit sets the daily withdrawal limit for an account.
func (fm *FundsManager) SetDailyLimit(accountID string, limit *big.Int) {
	fm.mu.Lock()
	defer fm.mu.Unlock()
	fm.dailyWithdrawalLimits[accountID] = new(big.Int).Set(limit)
}

// Initiate creates and validates a new funds movement.
func (fm *FundsManager) Initiate(m *FundsMovement) (*FundsMovement, error) {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	// Reset daily counters if needed
	fm.resetDailyIfNeeded()

	// Validate
	if m.Amount == nil || m.Amount.Sign() <= 0 {
		return nil, fmt.Errorf("amount must be positive")
	}
	if m.Asset == "" {
		return nil, fmt.Errorf("asset is required")
	}
	if m.SourceID == "" {
		return nil, fmt.Errorf("source account is required")
	}

	// Assign ID and timestamps
	m.ID = fmt.Sprintf("fm_%d", time.Now().UnixNano())
	m.CreatedAt = time.Now()
	m.UpdatedAt = time.Now()
	m.Status = FundsStatusPending

	// Calculate fee based on route
	if m.Fee == nil {
		m.Fee = big.NewInt(0)
	}
	m.NetAmount = new(big.Int).Sub(m.Amount, m.Fee)

	// Classify risk
	m.Risk = fm.classifyRisk(m)

	// Check daily withdrawal limit for outbound movements
	if m.Direction == FundsWithdraw {
		if err := fm.checkWithdrawalLimit(m); err != nil {
			m.Status = FundsStatusFailed
			m.FailureReason = err.Error()
			return m, err
		}
	}

	// Permissionless: a validated movement is approved. There is no
	// eligibility/KYC gate — only the funds checks above (positive amount,
	// asset/source present, daily withdrawal limit) can reject a movement.
	m.Status = FundsStatusApproved

	// Store
	fm.movements[m.ID] = m
	fm.byAccount[m.SourceID] = append(fm.byAccount[m.SourceID], m)
	if m.DestID != "" && m.DestID != m.SourceID {
		fm.byAccount[m.DestID] = append(fm.byAccount[m.DestID], m)
	}

	// Track daily withdrawal
	if m.Direction == FundsWithdraw {
		if fm.dailyWithdrawn[m.SourceID] == nil {
			fm.dailyWithdrawn[m.SourceID] = big.NewInt(0)
		}
		fm.dailyWithdrawn[m.SourceID].Add(fm.dailyWithdrawn[m.SourceID], m.Amount)
	}

	return m, nil
}

// Execute processes an approved funds movement.
func (fm *FundsManager) Execute(movementID string) error {
	fm.mu.Lock()
	m, ok := fm.movements[movementID]
	if !ok {
		fm.mu.Unlock()
		return fmt.Errorf("movement %s not found", movementID)
	}
	if m.Status != FundsStatusApproved {
		fm.mu.Unlock()
		return fmt.Errorf("movement %s not in approved state (current: %d)", movementID, m.Status)
	}
	m.Status = FundsStatusProcessing
	m.UpdatedAt = time.Now()
	fm.mu.Unlock()

	var err error

	switch m.Route {
	case RouteDEXInternal:
		err = fm.executeDEXInternal(m)
	case RouteCEXInternal:
		err = fm.executeCEXInternal(m)
	case RouteDEXToDEX:
		err = fm.executeDEXToDEX(m)
	case RouteCEXToCEX:
		err = fm.executeCEXToCEX(m)
	case RouteDEXToCEX:
		err = fm.executeDEXToCEX(m)
	case RouteCEXToDEX:
		err = fm.executeCEXToDEX(m)
	case RouteExternalIn:
		err = fm.executeExternalIn(m)
	case RouteExternalOut:
		err = fm.executeExternalOut(m)
	default:
		err = fmt.Errorf("unsupported route: %d", m.Route)
	}

	fm.mu.Lock()
	defer fm.mu.Unlock()

	if err != nil {
		m.Status = FundsStatusFailed
		m.FailureReason = err.Error()
		m.UpdatedAt = time.Now()
		if fm.onFail != nil {
			fm.onFail(m, err.Error())
		}
		return err
	}

	now := time.Now()
	m.Status = FundsStatusCompleted
	m.CompletedAt = &now
	m.UpdatedAt = now
	if fm.onComplete != nil {
		fm.onComplete(m)
	}

	return nil
}

// GetMovement returns a funds movement by ID.
func (fm *FundsManager) GetMovement(id string) (*FundsMovement, bool) {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	m, ok := fm.movements[id]
	return m, ok
}

// GetAccountMovements returns all movements for an account.
func (fm *FundsManager) GetAccountMovements(accountID string) []*FundsMovement {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	return fm.byAccount[accountID]
}

// --- Risk classification ---

func (fm *FundsManager) classifyRisk(m *FundsMovement) RiskLevel {
	// Large amounts = high risk (threshold: 10,000 USD equivalent)
	threshold := new(big.Int).Mul(big.NewInt(10000), big.NewInt(1e6)) // 10K in 6-decimal units
	if m.Amount.Cmp(threshold) > 0 {
		if m.Route == RouteDEXToCEX || m.Route == RouteCEXToDEX || m.Route == RouteExternalOut {
			return RiskHigh
		}
		return RiskMedium
	}

	// Cross-venue = medium
	if m.Route == RouteDEXToCEX || m.Route == RouteCEXToDEX || m.Route == RouteDEXToDEX {
		return RiskMedium
	}

	// Internal = low
	return RiskLow
}

// --- Withdrawal limits ---

func (fm *FundsManager) checkWithdrawalLimit(m *FundsMovement) error {
	limit := fm.dailyWithdrawalLimits[m.SourceID]
	if limit == nil {
		return nil // No limit set
	}

	withdrawn := fm.dailyWithdrawn[m.SourceID]
	if withdrawn == nil {
		withdrawn = big.NewInt(0)
	}

	total := new(big.Int).Add(withdrawn, m.Amount)
	if total.Cmp(limit) > 0 {
		return fmt.Errorf("daily withdrawal limit exceeded: %s + %s > %s",
			withdrawn.String(), m.Amount.String(), limit.String())
	}
	return nil
}

func (fm *FundsManager) resetDailyIfNeeded() {
	now := time.Now()
	if now.Day() != fm.lastReset.Day() || now.Sub(fm.lastReset) > 24*time.Hour {
		fm.dailyWithdrawn = make(map[string]*big.Int)
		fm.lastReset = now
	}
}

// --- Route executors ---
// Each route has fundamentally different mechanics.

// DEX internal: margin deposit, vault deposit, lending deposit — on-chain, instant.
func (fm *FundsManager) executeDEXInternal(m *FundsMovement) error {
	// On-chain: call smart contract (deposit/withdraw from margin/vault/lending)
	// Risk: smart contract risk only
	// Settlement: instant (same block)
	return nil
}

// CEX internal: account-to-account, margin transfer — custodial, instant.
func (fm *FundsManager) executeCEXInternal(m *FundsMovement) error {
	// Custodial: update internal ledger
	// Risk: custodial risk (exchange holds funds)
	// Settlement: instant
	return nil
}

// DEX→DEX: cross-chain bridge — smart contract + bridge risk, ~15 min.
func (fm *FundsManager) executeDEXToDEX(m *FundsMovement) error {
	if fm.bridge == nil {
		return fmt.Errorf("cross-chain bridge required for DEX-to-DEX transfers")
	}
	// Lock on source chain → bridge validates → mint on dest chain
	// Risk: smart contract + bridge validator risk
	// Settlement: ~15 min (bridge confirmations)
	return nil
}

// CEX→CEX: broker transfer — custodial, T+1/T+2.
func (fm *FundsManager) executeCEXToCEX(m *FundsMovement) error {
	// Broker-to-broker transfer via ACAT, FIX, or proprietary API
	// Risk: custodial + counterparty
	// Settlement: T+1 to T+2
	return nil
}

// DEX→CEX: on-chain withdrawal + CEX deposit — mixed risk, ~30 min.
func (fm *FundsManager) executeDEXToCEX(m *FundsMovement) error {
	// 1. Withdraw from DEX smart contract (on-chain tx)
	// 2. Send to CEX deposit address
	// 3. Wait for CEX to credit (confirmations required)
	// Risk: smart contract (step 1) + custodial (step 3)
	// Settlement: ~30 min (on-chain confirms + CEX processing)
	return nil
}

// CEX→DEX: custodial withdrawal + on-chain deposit — mixed risk, ~15 min.
func (fm *FundsManager) executeCEXToDEX(m *FundsMovement) error {
	// 1. CEX processes withdrawal (2FA, withdrawal limits)
	// 2. CEX sends on-chain transaction
	// 3. DEX contract receives deposit
	// Risk: custodial (step 1) + smart contract (step 3)
	// Settlement: ~15 min
	return nil
}

// External→venue: fiat wire/ACH — banking risk, 1-5 business days.
func (fm *FundsManager) executeExternalIn(m *FundsMovement) error {
	// Fiat deposit via wire transfer, ACH, or card
	// Risk: banking counterparty
	// Settlement: 1-5 business days (wire), 3-5 days (ACH), instant (card, with limits)
	return nil
}

// Venue→external: fiat wire/ACH out — banking risk, 1-5 business days.
func (fm *FundsManager) executeExternalOut(m *FundsMovement) error {
	// Fiat withdrawal via wire transfer or ACH
	// Risk: banking counterparty
	// Settlement: 1-5 business days
	return nil
}
