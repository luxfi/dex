// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math/big"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

// custody.go is the funding rail, and it is deliberately boring.
//
// MOVE ONCE, TRADE MANY. Value crosses the chain boundary exactly twice in its
// life — in, and out. Between those, D owns it outright and every order, match and
// trade is a local balance movement. That is what makes D an exchange ledger rather
// than a remote matching oracle bolted onto C.
//
// THE INVARIANT EVERYTHING RESTS ON: a unit is spendable on exactly one chain at a
// time.
//
//	C  --Export-->  CToD  --Import-->  D  --Export-->  DToC  --Import-->  C
//
// Those four are the ONLY transitions. Never C to D directly — that would mean a
// moment where the value is either in both places or neither. Never CToD back to
// CToD — an atomic object is consumed once or not at all.
//
// WHAT THIS SUBSYSTEM MUST NOT DO. It moves ownership between chains. It knows
// nothing about orders, matches, executions, trades, reservations or settlement.
// Braiding those in is how the previous design ended up needing a cross-chain
// certificate per trade; keeping them out is why a million trades now cost zero
// boundary crossings.

// Location is where a unit of value currently lives. It exists to be named in
// errors and assertions — the type system carries the real enforcement, since each
// transition consumes its input and produces its output.
type Location uint8

const (
	OnC Location = iota
	CToD
	OnD
	DToC
)

func (l Location) String() string {
	switch l {
	case OnC:
		return "C"
	case CToD:
		return "C->D"
	case OnD:
		return "D"
	case DToC:
		return "D->C"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(l))
	}
}

var (
	ErrClaimConsumed = errors.New("dexprotocol: this transfer object was already imported")
	ErrClaimWidth    = errors.New("dexprotocol: encoded Claim has the wrong width")
	ErrClaimScope    = errors.New("dexprotocol: transfer object is missing a required field")
	ErrClaimAmount   = errors.New("dexprotocol: transfer object must move a non-zero amount")
	ErrClaimChain    = errors.New("dexprotocol: transfer object is addressed to a different chain")
	ErrNoBalance     = errors.New("dexprotocol: insufficient available balance")
	ErrReserved      = errors.New("dexprotocol: insufficient reserved balance")
	ErrNotOwned      = errors.New("dexprotocol: no balance for that owner and asset")
)

// Claim is a value transfer in flight — the atomic object the source chain wrote and
// the destination chain may consume, exactly once.
//
// It names its BENEFICIARY, which is what makes import safely permissionless: anyone
// may deliver a claim, because delivering it can only credit the account it already
// names. Co-location buys latency, never correctness — a relayer that refuses to
// import cannot strand anyone, since any other participant can.
type Claim struct {
	ClaimID     ids.ID
	Source      ids.ID
	Dest        ids.ID
	Beneficiary common.Address
	Asset       ids.ID
	Amount      [32]byte
}

func (c *Claim) Validate() error {
	if c.ClaimID == ids.Empty {
		return fmt.Errorf("%w: ClaimID", ErrClaimScope)
	}
	if c.Source == ids.Empty || c.Dest == ids.Empty {
		return fmt.Errorf("%w: Source/Dest", ErrClaimScope)
	}
	if c.Source == c.Dest {
		return fmt.Errorf("%w: source and destination are the same chain", ErrClaimChain)
	}
	if c.Beneficiary == (common.Address{}) {
		return fmt.Errorf("%w: Beneficiary", ErrClaimScope)
	}
	if c.Asset == ids.Empty {
		return fmt.Errorf("%w: Asset", ErrClaimScope)
	}
	if c.Amount == ([32]byte{}) {
		return ErrClaimAmount
	}
	return nil
}

func (c Claim) Big() *big.Int { return new(big.Int).SetBytes(c.Amount[:]) }

// ClaimEncodedLen is the canonical width — every field fixed, so concatenation is
// injective and the commitment rests on SHA-256 rather than on a parsing argument.
const ClaimEncodedLen = 32 + 32 + 32 + 20 + 32 + 32 // id, source, dest, beneficiary, asset, amount

const claimDomain = "lux.dex.claim.v1"

var claimDomainTag = sha256.Sum256([]byte(claimDomain))

// Encode returns the canonical encoding. Total, so a malformed claim still has a
// well-defined commitment and a verifier can name what it rejected.
func (c Claim) Encode() []byte {
	b := make([]byte, ClaimEncodedLen)
	n := 0
	n += copy(b[n:], c.ClaimID[:])
	n += copy(b[n:], c.Source[:])
	n += copy(b[n:], c.Dest[:])
	n += copy(b[n:], c.Beneficiary[:])
	n += copy(b[n:], c.Asset[:])
	n += copy(b[n:], c.Amount[:])
	if n != ClaimEncodedLen {
		panic(fmt.Sprintf("dexprotocol: claim encoder wrote %d bytes, layout says %d", n, ClaimEncodedLen))
	}
	return b
}

// DecodeClaim parses the canonical encoding, requiring the EXACT width. A longer
// buffer is refused rather than truncated: accepting a suffix would let two distinct
// wire messages decode to the same claim while carrying different attested bytes.
func DecodeClaim(b []byte) (Claim, error) {
	if len(b) != ClaimEncodedLen {
		return Claim{}, fmt.Errorf("%w: got %d, want %d", ErrClaimWidth, len(b), ClaimEncodedLen)
	}
	var c Claim
	n := 0
	n += copy(c.ClaimID[:], b[n:n+32])
	n += copy(c.Source[:], b[n:n+32])
	n += copy(c.Dest[:], b[n:n+32])
	n += copy(c.Beneficiary[:], b[n:n+20])
	n += copy(c.Asset[:], b[n:n+32])
	copy(c.Amount[:], b[n:n+32])
	return c, nil
}

// Commitment is the claim's identity and the thing the source chain attests to:
//
//	commitment = SHA-256( SHA-256(domain) || canonical encoding )
//
// It covers the CORRIDOR (source and dest) as well as the payment, so an attestation
// for one direction cannot be replayed as the other, and one for a different pair of
// chains cannot be replayed here.
func (c Claim) Commitment() ids.ID {
	h := sha256.New()
	h.Write(claimDomainTag[:])
	h.Write(c.Encode())
	var id ids.ID
	copy(id[:], h.Sum(nil))
	return id
}

// balance is one owner's holding of one asset on D.
//
// Available is spendable: it may back a new order, or leave for C. Reserved is
// committed to open orders and may do NEITHER. A withdrawal drawing on reserved
// balance would let a trader spend the same value twice — once as a resting order and
// once as an exit — so Reserved is simply not reachable by any path that removes
// value from D.
type balance struct {
	available *big.Int
	reserved  *big.Int
}

// Custody is the D-side ledger of who owns what, plus the exactly-once record of
// which transfer objects have been consumed.
type Custody struct {
	chainID  ids.ID
	balances map[common.Address]map[ids.ID]*balance
	consumed map[ids.ID]struct{}
	// Export domains — see export.go. A claim id lives in exactly one of them,
	// which is what makes reclaimable and writable mutually exclusive.
	pending   map[ids.ID]PendingExport
	delivered map[ids.ID]Deliverable
	reclaimed map[ids.ID]Reclaimed
}

func NewCustody(chainID ids.ID) *Custody {
	return &Custody{
		chainID:   chainID,
		balances:  make(map[common.Address]map[ids.ID]*balance),
		consumed:  make(map[ids.ID]struct{}),
		pending:   make(map[ids.ID]PendingExport),
		delivered: make(map[ids.ID]Deliverable),
		reclaimed: make(map[ids.ID]Reclaimed),
	}
}

func (c *Custody) at(owner common.Address, asset ids.ID) *balance {
	byAsset, ok := c.balances[owner]
	if !ok {
		byAsset = make(map[ids.ID]*balance)
		c.balances[owner] = byAsset
	}
	b, ok := byAsset[asset]
	if !ok {
		b = &balance{available: new(big.Int), reserved: new(big.Int)}
		byAsset[asset] = b
	}
	return b
}

// Import consumes a transfer object and credits its beneficiary. THE
// CONSUME AND THE CREDIT ARE ONE OPERATION — that is the whole safety property of
// the rail. If the object could be marked consumed without the credit landing, value
// would vanish; if the credit could land without the consumption, the same object
// would be importable again by a retry, a reorg, a racing relayer or a restart, and
// that is a double-spend of real money.
//
// It is idempotent-by-refusal rather than idempotent-by-silence: a second import
// returns ErrClaimConsumed instead of quietly succeeding, because a caller that
// imported twice has a bug worth surfacing.
func (c *Custody) Import(v VerifiedClaim) error {
	cl := v.Claim()
	if cl.Dest != c.chainID {
		return fmt.Errorf("%w: addressed to %s, this is %s", ErrClaimChain, cl.Dest, c.chainID)
	}
	if _, done := c.consumed[cl.ClaimID]; done {
		return fmt.Errorf("%w: %s", ErrClaimConsumed, cl.ClaimID)
	}
	c.consumed[cl.ClaimID] = struct{}{}
	b := c.at(cl.Beneficiary, cl.Asset)
	b.available.Add(b.available, cl.Big())
	return nil
}

// --- Authenticated claims: an unattested transfer must not compile ----------
//
// THIS WAS A REAL DEFECT, NOT A HYPOTHETICAL. Import used to take a bare Claim.
// Every other value-creating path in this package already had the sentinel —
// VerifiedExecution, VerifiedOrder, AcceptedBlock — and the export side has three
// storage domains. The import side, the ONLY path that creates value on D, took an
// exported struct with exported fields. A literal
//
//	Claim{Beneficiary: attacker, Asset: usdc, Amount: 1e30, ...}
//
// returned nil and credited 1e30. Validate() checks that fields are PRESENT, which
// is not the same question as whether anyone said them.
//
// It survived because permissionless delivery reads as a safety property: "anyone
// may deliver a claim, because delivering it can only credit the account it already
// names." That is true only if the claim is genuine. Permissionless delivery plus an
// unauthenticated object means anyone credits themselves — the argument for the
// feature quietly assumed the check that was missing.

// ClaimVerifier attests that the SOURCE chain really wrote this transfer object. In
// production that attestation is the object's presence in the source chain's
// shared-memory partition — only that chain can write there. It is an interface so
// this package depends on the ABSTRACTION and never on a socket, a database or a
// live peer chain.
type ClaimVerifier interface {
	VerifyClaim(commitment ids.ID, attestation []byte) error
}

// ClaimContext is everything claim verification may consult. A VALUE, like every
// other context here: if a check cannot be made from this struct plus the witness
// bytes, it does not belong in the consensus path.
//
// Dest is the verifier's OWN chain id, never read from the witness. A witness that
// named its own destination would be asserting the one thing that stops a transfer
// addressed elsewhere from being credited here.
type ClaimContext struct {
	Dest     ids.ID
	Verifier ClaimVerifier
}

var (
	ErrNoClaimVerifier = errors.New("dexprotocol: claim verification requires a verifier")
	ErrBadClaim        = errors.New("dexprotocol: transfer object attestation did not verify")
)

// VerifiedClaim is a transfer object whose attestation has been checked. It cannot
// be constructed outside this package: the interface method is unexported, so no
// other package can implement it, and the concrete type is unexported, so no other
// package can build one.
type VerifiedClaim interface {
	Claim() Claim
	Commitment() ids.ID
	// verifiedClaim is unexported and therefore unimplementable elsewhere. It is
	// the enforcement; do not export it.
	verifiedClaim()
}

type verifiedClaim struct {
	claim      Claim
	commitment ids.ID
}

func (v verifiedClaim) Claim() Claim      { return v.claim }
func (v verifiedClaim) Commitment() ids.ID { return v.commitment }
func (verifiedClaim) verifiedClaim()      {}

// VerifyClaim is the ONLY producer of a VerifiedClaim. It checks that the object is
// well formed, that it is addressed to THIS chain as the verifier knows it, and that
// its commitment carries a valid attestation from the source chain.
func VerifyClaim(witness []byte, ctx ClaimContext) (VerifiedClaim, error) {
	if ctx.Verifier == nil {
		return nil, ErrNoClaimVerifier
	}
	if ctx.Dest == ids.Empty {
		return nil, fmt.Errorf("%w: context names no destination chain", ErrClaimScope)
	}
	cl, err := DecodeClaim(claimBody(witness))
	if err != nil {
		return nil, err
	}
	if err := cl.Validate(); err != nil {
		return nil, err
	}
	if cl.Dest != ctx.Dest {
		return nil, fmt.Errorf("%w: addressed to %s, verifying on %s", ErrClaimChain, cl.Dest, ctx.Dest)
	}
	commitment := cl.Commitment()
	if err := ctx.Verifier.VerifyClaim(commitment, claimAttestation(witness)); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrBadClaim, err)
	}
	return verifiedClaim{claim: cl, commitment: commitment}, nil
}

func claimBody(witness []byte) []byte {
	if len(witness) < ClaimEncodedLen {
		return witness
	}
	return witness[:ClaimEncodedLen]
}

func claimAttestation(witness []byte) []byte {
	if len(witness) <= ClaimEncodedLen {
		return nil
	}
	return witness[ClaimEncodedLen:]
}

// Reserve moves value from available to reserved when an order is placed. Nothing
// leaves the account — this is a change of what the value may be used for, and
// Owned is unchanged by construction.
func (c *Custody) Reserve(owner common.Address, asset ids.ID, amount *big.Int) error {
	if amount.Sign() <= 0 {
		return ErrClaimAmount
	}
	b := c.at(owner, asset)
	if b.available.Cmp(amount) < 0 {
		return fmt.Errorf("%w: have %s available, reserving %s", ErrNoBalance, b.available, amount)
	}
	b.available.Sub(b.available, amount)
	b.reserved.Add(b.reserved, amount)
	return nil
}

// Unreserve returns committed value to available — an order cancelled, or the
// unfilled remainder of one that partially traded.
func (c *Custody) Unreserve(owner common.Address, asset ids.ID, amount *big.Int) error {
	if amount.Sign() <= 0 {
		return ErrClaimAmount
	}
	b := c.at(owner, asset)
	if b.reserved.Cmp(amount) < 0 {
		return fmt.Errorf("%w: have %s reserved, releasing %s", ErrReserved, b.reserved, amount)
	}
	b.reserved.Sub(b.reserved, amount)
	b.available.Add(b.available, amount)
	return nil
}

// Trade is the only operation that moves value BETWEEN owners, and it is entirely
// local to D — no boundary crossing, no certificate, no C block. The seller's
// reserved input is spent; the buyer receives available output. This is what "trade
// many" means in practice.
//
// Both legs are applied together or not at all: a partial application would create
// value on one side and destroy it on the other.
func (c *Custody) Trade(seller common.Address, sold ids.ID, soldAmt *big.Int,
	buyer common.Address, bought ids.ID, boughtAmt *big.Int) error {
	if soldAmt.Sign() <= 0 || boughtAmt.Sign() <= 0 {
		return ErrClaimAmount
	}
	sb := c.at(seller, sold)
	bb := c.at(buyer, bought)
	if sb.reserved.Cmp(soldAmt) < 0 {
		return fmt.Errorf("%w: seller has %s reserved, trading %s", ErrReserved, sb.reserved, soldAmt)
	}
	if bb.reserved.Cmp(boughtAmt) < 0 {
		return fmt.Errorf("%w: buyer has %s reserved, trading %s", ErrReserved, bb.reserved, boughtAmt)
	}
	sb.reserved.Sub(sb.reserved, soldAmt)
	bb.reserved.Sub(bb.reserved, boughtAmt)
	c.at(buyer, sold).available.Add(c.at(buyer, sold).available, soldAmt)
	c.at(seller, bought).available.Add(c.at(seller, bought).available, boughtAmt)
	return nil
}

// Balance reports one account's holding.
func (c *Custody) Balance(owner common.Address, asset ids.ID) (available, reserved *big.Int, ok bool) {
	byAsset, ok := c.balances[owner]
	if !ok {
		return nil, nil, false
	}
	b, ok := byAsset[asset]
	if !ok {
		return nil, nil, false
	}
	return new(big.Int).Set(b.available), new(big.Int).Set(b.reserved), true
}

// Owned is the total D holds of one asset: available plus reserved, across every
// account. It must equal imports minus exports for that asset, forever — trading
// moves value between owners and never changes the total.
func (c *Custody) Owned(asset ids.ID) *big.Int {
	total := new(big.Int)
	for _, byAsset := range c.balances {
		if b, ok := byAsset[asset]; ok {
			total.Add(total, b.available)
			total.Add(total, b.reserved)
		}
	}
	return total
}

// Conserved asserts the rail's invariant for one asset:
//
//	imported - delivered  ==  Owned + Earmarked
//
// `delivered` is what was COMMITTED for delivery, not what was merely earmarked: a
// pending export has left the owner's available balance but is still D's and still
// reclaimable, so it counts on D's side of the equation. Leaving it out would make
// every open export look like a shortfall.
//
// The caller supplies the boundary totals it observed, so the check is against the
// RAIL rather than against an internal number that would agree with itself by
// construction.
func (c *Custody) Conserved(asset ids.ID, imported, delivered *big.Int) error {
	want := new(big.Int).Sub(imported, delivered)
	got := new(big.Int).Add(c.Owned(asset), c.Earmarked(asset))
	if got.Cmp(want) != 0 {
		return fmt.Errorf("dexprotocol: D holds %s of %s (owned %s + earmarked %s), rail says %s in - %s delivered = %s",
			got, asset, c.Owned(asset), c.Earmarked(asset), imported, delivered, want)
	}
	return nil
}

// NoNegative is a paranoia check: no balance may ever go below zero. Every path
// above guards its own subtraction, so a failure here means a path was added that
// did not — which is exactly the kind of regression worth catching loudly.
func (c *Custody) NoNegative() error {
	for owner, byAsset := range c.balances {
		for asset, b := range byAsset {
			if b.available.Sign() < 0 || b.reserved.Sign() < 0 {
				return fmt.Errorf("dexprotocol: %s holds %s available %s reserved of %s",
					owner, b.available, b.reserved, asset)
			}
		}
	}
	return nil
}
