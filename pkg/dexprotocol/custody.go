// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
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
func (c *Custody) Import(cl Claim) error {
	if err := cl.Validate(); err != nil {
		return err
	}
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
