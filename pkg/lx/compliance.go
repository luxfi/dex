package lx

import "fmt"

// EligibilityChecker is called before every order placement to verify that
// the trader is allowed to trade the given security. ATS operators implement
// this interface with their own KYC/AML/accreditation backend.
type EligibilityChecker interface {
	// CanTrade returns nil if the user is eligible, or an error describing
	// why the trade is rejected.
	CanTrade(userID string, symbol string, side Side, size float64, price float64) error
}

// SettlementHandler is called after each trade execution. ATS operators
// implement this to notify transfer agents, update cap tables, trigger
// on-chain token transfers, or enqueue escrow workflows.
type SettlementHandler interface {
	// OnTrade is called for each executed trade. Returning an error does not
	// reverse the match — settlement failures are handled asynchronously.
	OnTrade(trade *Trade, market *MarketConfig) error
}

// ComplianceErrorCode identifies the category of a compliance rejection.
type ComplianceErrorCode uint8

const (
	ErrNotAccredited      ComplianceErrorCode = iota + 1 // Investor not accredited
	ErrJurisdiction                                      // Investor jurisdiction blocked
	ErrHoldingPeriod                                     // Holding period not elapsed
	ErrMaxHolders                                        // Would exceed max holder count
	ErrKYCRequired                                       // KYC verification missing
	ErrAMLRequired                                       // AML clearance missing
	ErrMinNotional                                       // Below minimum notional value
	ErrMinSize                                           // Below minimum order size
	ErrMaxSize                                           // Exceeds maximum order size
	ErrROFRPending                                       // Right of first refusal pending
	ErrBoardApproval                                     // Board approval required
	ErrTransferRestricted                                // Generic transfer restriction
)

// ComplianceError is returned when a pre-trade check fails.
type ComplianceError struct {
	Code    ComplianceErrorCode
	Message string
	Detail  interface{} // Optional context (threshold value, expiry date, etc.)
}

func (e *ComplianceError) Error() string {
	if e.Detail != nil {
		return fmt.Sprintf("compliance: %s (detail: %v)", e.Message, e.Detail)
	}
	return fmt.Sprintf("compliance: %s", e.Message)
}

// StandardEligibilityChecker provides a reusable implementation that enforces
// TransferRestriction rules using an InvestorProfile lookup function.
// ATS operators can use this directly or wrap it with additional checks.
type StandardEligibilityChecker struct {
	// GetProfile returns the investor profile for a user ID.
	// This is the only function the operator must provide.
	GetProfile func(userID string) (*InvestorProfile, error)

	// GetHolderCount returns the current number of distinct holders for a symbol.
	GetHolderCount func(symbol string) (int, error)

	// Restrictions are the transfer restrictions for the market.
	// Usually copied from MarketConfig.Restrictions.
	Restrictions *TransferRestriction
}

// CanTrade checks all configured restrictions against the investor profile.
func (c *StandardEligibilityChecker) CanTrade(userID string, symbol string, side Side, size float64, price float64) error {
	if c.Restrictions == nil {
		return nil
	}

	profile, err := c.GetProfile(userID)
	if err != nil {
		return &ComplianceError{Code: ErrTransferRestricted, Message: "unable to verify investor profile"}
	}
	if profile == nil {
		return &ComplianceError{Code: ErrKYCRequired, Message: "investor profile not found"}
	}

	// KYC/AML
	if !profile.KYCVerified {
		return &ComplianceError{Code: ErrKYCRequired, Message: "KYC verification required"}
	}
	if !profile.AMLCleared {
		return &ComplianceError{Code: ErrAMLRequired, Message: "AML clearance required"}
	}

	r := c.Restrictions

	// Investor status check (accreditation)
	if profile.Status < r.MinInvestorStatus {
		return &ComplianceError{
			Code:    ErrNotAccredited,
			Message: "investor does not meet minimum status requirement",
			Detail:  r.MinInvestorStatus,
		}
	}

	// Jurisdiction whitelist
	if len(r.JurisdictionWhitelist) > 0 {
		allowed := false
		for _, j := range r.JurisdictionWhitelist {
			if j == profile.Jurisdiction {
				allowed = true
				break
			}
		}
		if !allowed {
			return &ComplianceError{
				Code:    ErrJurisdiction,
				Message: "investor jurisdiction not in whitelist",
				Detail:  profile.Jurisdiction,
			}
		}
	}

	// Jurisdiction blacklist
	for _, j := range r.JurisdictionBlacklist {
		if j == profile.Jurisdiction {
			return &ComplianceError{
				Code:    ErrJurisdiction,
				Message: "investor jurisdiction is restricted",
				Detail:  profile.Jurisdiction,
			}
		}
	}

	// Max holders (only check for buy side — new holders)
	if side == Buy && r.MaxHolders > 0 && c.GetHolderCount != nil {
		// Only check if this would be a NEW holder
		currentHolding := profile.Holdings[symbol]
		if currentHolding == 0 {
			count, err := c.GetHolderCount(symbol)
			if err == nil && count >= r.MaxHolders {
				return &ComplianceError{
					Code:    ErrMaxHolders,
					Message: "maximum holder count reached",
					Detail:  r.MaxHolders,
				}
			}
		}
	}

	// ROFR / board approval (these would typically be checked via an async
	// workflow, but we reject synchronously if the flag is set and no
	// pre-approval exists. The operator's GetProfile should set these
	// flags based on prior approval status.)
	if side == Buy && r.ROFRRequired {
		return &ComplianceError{
			Code:    ErrROFRPending,
			Message: "right of first refusal review required",
		}
	}
	if r.BoardApprovalRequired {
		return &ComplianceError{
			Code:    ErrBoardApproval,
			Message: "board approval required for transfer",
		}
	}

	return nil
}
