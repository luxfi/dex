package lx

import "time"

// AssetClass represents the classification of a traded asset.
type AssetClass uint8

const (
	AssetClassCrypto    AssetClass = iota // Unregulated digital asset
	AssetClassEquity                      // Common/preferred stock
	AssetClassDebt                        // Bonds, notes, convertible notes
	AssetClassFund                        // LP interests, fund shares
	AssetClassSAFE                        // Simple Agreement for Future Equity
	AssetClassCommodity                   // Physical or digital commodity
	AssetClassDerivative                  // Options, futures, swaps
)

// ExemptionType identifies the registration exemption or regulatory framework
// under which a security is offered and traded.
type ExemptionType uint8

const (
	ExemptionNone    ExemptionType = iota // No exemption (registered or crypto)

	// United States (SEC)
	ExemptionRegD506b                     // Reg D Rule 506(b) — up to 35 non-accredited
	ExemptionRegD506c                     // Reg D Rule 506(c) — accredited only, general solicitation OK
	ExemptionRegA1                        // Reg A Tier 1 — up to $20M/year
	ExemptionRegA2                        // Reg A Tier 2 — up to $75M/year
	ExemptionRegCF                        // Regulation Crowdfunding — up to $5M/year
	ExemptionRegS                         // Regulation S — offshore only
	ExemptionRule144                      // Rule 144 resale of restricted securities
	ExemptionSection4a7                   // Section 4(a)(7) — private resales to accredited

	// United Kingdom (FCA)
	ExemptionFCAPrivate                   // FCA private placement (no prospectus required)
	ExemptionFCAProspectus                // FCA prospectus-exempt (qualified investors only)

	// Singapore (MAS)
	ExemptionMASSIP                       // MAS Small Offers (S$5M/12 months, <=50 persons)
	ExemptionMASPrivate                   // MAS Private Placement (<=50 persons in 12 months)
	ExemptionMASInstitutional             // MAS Institutional Investor exemption
)

// InvestorStatus represents the regulatory classification of an investor.
// Values are ordered by privilege level — higher values have more access.
type InvestorStatus uint8

const (
	InvestorRetail              InvestorStatus = iota // Non-accredited retail
	InvestorAccredited                                // US: SEC accredited investor
	InvestorSophisticated                             // UK: FCA certified sophisticated investor
	InvestorProfessional                              // UK: FCA professional client / SG: MAS accredited
	InvestorQualifiedPurchaser                        // US: Qualified purchaser ($5M+ investments)
	InvestorQualifiedClient                           // US: Qualified client ($1.1M+ net worth)
	InvestorInstitutional                             // Institutional investor (bank, fund, etc.)
)

// TransferRestriction defines a constraint on the transferability of a security.
type TransferRestriction struct {
	// HoldingPeriod is the minimum duration a buyer must hold before resale.
	// Zero means no holding period restriction.
	HoldingPeriod time.Duration

	// MaxHolders is the maximum number of beneficial owners allowed.
	// Zero means unlimited.
	MaxHolders int

	// MinInvestorStatus is the minimum investor classification required to trade.
	MinInvestorStatus InvestorStatus

	// JurisdictionWhitelist limits trading to investors in these jurisdictions.
	// Empty means all jurisdictions allowed.
	JurisdictionWhitelist []string

	// JurisdictionBlacklist prohibits trading by investors in these jurisdictions.
	JurisdictionBlacklist []string

	// ROFRRequired indicates the company has a right of first refusal on transfers.
	ROFRRequired bool

	// BoardApprovalRequired indicates transfers require board/GP approval.
	BoardApprovalRequired bool
}

// InvestorProfile holds the compliance-relevant attributes of a trader.
// Provided by the EligibilityChecker implementation, not stored in the DEX.
type InvestorProfile struct {
	UserID       string
	Status       InvestorStatus
	Jurisdiction string // ISO 3166-1 alpha-2 country code
	AccreditedAt time.Time
	KYCVerified  bool
	AMLCleared   bool

	// Holdings tracks current share count per symbol for max-holder checks.
	// The checker implementation is responsible for populating this.
	Holdings map[string]float64
}
