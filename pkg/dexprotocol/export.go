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

// export.go is the D->C leg, and it exists because that leg has the sharpest
// monetary edge in the system.
//
// THE HAZARD. D debits the owner and writes a transfer object for C to consume. If
// the write is skipped or lost, the value is gone from D and never arrives on C.
// Refunding it on a timer looks like the obvious repair and is a double-spend
// waiting to happen: the moment a refund and a late delivery can both occur, the
// same value exists twice.
//
// THE PROPERTY THAT MUST HOLD:
//
//	After a reclaim, no copy of the original transfer object — however delayed,
//	retried or replayed — can ever become importable on C.
//
// It is made structural rather than argued, by controlling who can obtain the bytes:
//
//	Available --Export--> Pending --Deliver--> Delivered --(retry the write)--> C
//	                         |
//	                         +----Reclaim----> Available
//
// A PendingExport does not expose a Claim. There is no accessor, so the bytes to
// write simply cannot be obtained from it. Only Deliver produces a Deliverable, and
// Deliver MOVES the record out of the pending domain — so a record is either
// reclaimable or writable, never both, and the choice is one D state transition that
// exactly one of the two contenders wins.
//
// The write is therefore attempted only ever from a COMMITTED Delivered state. That
// is the same commit-then-apply discipline the import leg already uses, and it is
// what makes the property above true by construction rather than by scheduling:
// reclaim requires Pending, a write requires Delivered, and nothing is both.
//
// A TIMEOUT IS NOT THE RECOVERY MECHANISM. Delivery is retryable from durable state
// for as long as it takes, because a Delivered record keeps yielding its Deliverable.
// A deadline may exist as a last resort, but the ordinary repair for a lost write is
// to write again — not to wait a week and refund.

// Export storage domains. Transitions MOVE the record; nothing is updated in place.
const (
	domainPending   = "exports/pending/"
	domainDelivered = "exports/delivered/"
	domainReclaimed = "exports/reclaimed/"
)

var (
	ErrNotPending      = errors.New("dexprotocol: no pending export under that claim id")
	ErrNotDelivered    = errors.New("dexprotocol: no delivered export under that claim id")
	ErrExportExists    = errors.New("dexprotocol: an export already exists under that claim id")
	ErrAlreadyResolved = errors.New("dexprotocol: this export was already delivered or reclaimed")
	ErrReclaimDeliverd = errors.New("dexprotocol: a delivered export cannot be reclaimed — retry the write instead")
)

// PendingExport is value that has left the owner's available balance and not yet
// been committed for delivery. It is reclaimable and DELIBERATELY NOT WRITABLE:
// there is no way to get a Claim out of it, so no code path can write the object
// while a reclaim is still possible.
type PendingExport struct {
	claim Claim
}

func (p PendingExport) ClaimID() ids.ID          { return p.claim.ClaimID }
func (p PendingExport) Owner() common.Address    { return p.claim.Beneficiary }
func (p PendingExport) Asset() ids.ID            { return p.claim.Asset }
func (p PendingExport) Amount() *big.Int         { return p.claim.Big() }
func (p PendingExport) Key() string              { return domainPending + p.claim.ClaimID.String() }

// Deliverable is a committed export whose object may be written to shared memory.
// Obtaining one is the ONLY way to reach the bytes, and obtaining one has already
// moved the record out of the reclaimable domain.
//
// It is safe to ask for repeatedly: a lost write is repaired by writing again, and
// the object is identical every time, so C's exactly-once import handles a duplicate
// arrival as the replay it is.
type Deliverable struct {
	claim Claim
}

func (d Deliverable) Claim() Claim  { return d.claim }
func (d Deliverable) ClaimID() ids.ID { return d.claim.ClaimID }
func (d Deliverable) Key() string   { return domainDelivered + d.claim.ClaimID.String() }

// Reclaimed is terminal. There is no function from Reclaimed back to Pending or
// Delivered, so a reclaimed export can never produce a writable object.
type Reclaimed struct {
	claim Claim
}

func (r Reclaimed) ClaimID() ids.ID { return r.claim.ClaimID }
func (r Reclaimed) Key() string     { return domainReclaimed + r.claim.ClaimID.String() }

// Export debits available balance and opens a PENDING export. It writes nothing and
// produces nothing writable — the object cannot be delivered until Deliver commits
// the record, which is what leaves a clean window in which reclaiming is safe.
//
// Reserved balance is untouchable here, as everywhere: value backing an open order
// cannot also leave the chain.
func (c *Custody) Export(owner common.Address, asset ids.ID, amount *big.Int, dest, claimID ids.ID) (PendingExport, error) {
	if amount.Sign() <= 0 {
		return PendingExport{}, ErrClaimAmount
	}
	if c.exportResolved(claimID) {
		return PendingExport{}, fmt.Errorf("%w: %s", ErrExportExists, claimID)
	}
	if _, open := c.pending[claimID]; open {
		return PendingExport{}, fmt.Errorf("%w: %s", ErrExportExists, claimID)
	}
	cl := Claim{ClaimID: claimID, Source: c.chainID, Dest: dest, Beneficiary: owner, Asset: asset}
	var buf [32]byte
	amount.FillBytes(buf[:])
	cl.Amount = buf
	if err := cl.Validate(); err != nil {
		return PendingExport{}, err
	}
	b := c.at(owner, asset)
	if b.available.Cmp(amount) < 0 {
		return PendingExport{}, fmt.Errorf("%w: have %s available, exporting %s", ErrNoBalance, b.available, amount)
	}
	b.available.Sub(b.available, amount)
	p := PendingExport{claim: cl}
	c.pending[claimID] = p
	return p, nil
}

// Deliver commits a pending export and yields the writable object. This is the
// transition that makes the export irreversible on D: afterwards the value is
// C's problem to receive, and D's only remaining job is to keep trying to hand it
// over.
func (c *Custody) Deliver(claimID ids.ID) (Deliverable, error) {
	p, ok := c.pending[claimID]
	if !ok {
		if c.exportResolved(claimID) {
			return Deliverable{}, fmt.Errorf("%w: %s", ErrAlreadyResolved, claimID)
		}
		return Deliverable{}, fmt.Errorf("%w: %s", ErrNotPending, claimID)
	}
	d := Deliverable{claim: p.claim}
	delete(c.pending, claimID)
	c.delivered[claimID] = d
	return d, nil
}

// Redeliver hands back the object for a committed export whose write did not land.
// This is the ORDINARY repair for a lost write. It is idempotent by nature — the
// object is byte-identical — and C's import is exactly-once, so a duplicate arrival
// is refused there rather than credited twice.
func (c *Custody) Redeliver(claimID ids.ID) (Deliverable, error) {
	d, ok := c.delivered[claimID]
	if !ok {
		return Deliverable{}, fmt.Errorf("%w: %s", ErrNotDelivered, claimID)
	}
	return d, nil
}

// Reclaim returns a PENDING export to available balance. It refuses a delivered
// export outright: once the object is obtainable, refunding it would let a late
// write and a refund both take effect, which is the double-spend this whole file
// exists to prevent. The repair for a stuck delivery is Redeliver, not Reclaim.
func (c *Custody) Reclaim(claimID ids.ID) (Reclaimed, error) {
	if _, ok := c.delivered[claimID]; ok {
		return Reclaimed{}, fmt.Errorf("%w: %s", ErrReclaimDeliverd, claimID)
	}
	p, ok := c.pending[claimID]
	if !ok {
		if _, done := c.reclaimed[claimID]; done {
			return Reclaimed{}, fmt.Errorf("%w: %s", ErrAlreadyResolved, claimID)
		}
		return Reclaimed{}, fmt.Errorf("%w: %s", ErrNotPending, claimID)
	}
	b := c.at(p.claim.Beneficiary, p.claim.Asset)
	b.available.Add(b.available, p.claim.Big())
	delete(c.pending, claimID)
	r := Reclaimed{claim: p.claim}
	c.reclaimed[claimID] = r
	return r, nil
}

func (c *Custody) exportResolved(claimID ids.ID) bool {
	if _, ok := c.delivered[claimID]; ok {
		return true
	}
	_, ok := c.reclaimed[claimID]
	return ok
}

// InFlight is the total committed for delivery but not yet known to have arrived —
// the D->C leg of the cross-chain conservation sum. Pending exports are NOT in
// flight: they are still D's, merely earmarked, and a reclaim returns them.
func (c *Custody) InFlight(asset ids.ID) *big.Int {
	total := new(big.Int)
	for _, d := range c.delivered {
		if d.claim.Asset == asset {
			total.Add(total, d.claim.Big())
		}
	}
	return total
}

// Earmarked is the total sitting in pending exports — debited from available,
// not yet committed, still reclaimable.
func (c *Custody) Earmarked(asset ids.ID) *big.Int {
	total := new(big.Int)
	for _, p := range c.pending {
		if p.claim.Asset == asset {
			total.Add(total, p.claim.Big())
		}
	}
	return total
}

// ExportsExclusive asserts that a claim id lives in exactly one export domain. An id
// that were both pending and delivered would be simultaneously reclaimable and
// writable, which is precisely the state that permits the double-spend.
func (c *Custody) ExportsExclusive() error {
	for id := range c.pending {
		if c.exportResolved(id) {
			return fmt.Errorf("dexprotocol: export %s is pending AND resolved", id)
		}
	}
	for id := range c.delivered {
		if _, ok := c.reclaimed[id]; ok {
			return fmt.Errorf("dexprotocol: export %s is delivered AND reclaimed", id)
		}
	}
	return nil
}
