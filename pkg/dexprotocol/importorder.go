// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"fmt"
	"math/big"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

// importorder.go is the arriving-trader path: bring value across from C and place an
// order with it in one D transition.
//
// THE UNUSUAL SEMANTICS, AND WHY THEY ARE RIGHT. The import is unconditional; the
// order is best effort. Order failure is DATA, not transaction failure.
//
// The temptation is the opposite — make the pair atomic in the strong sense, so a
// trader never ends up funded without the order they wanted. That strands people. A
// deadline can expire in transit: C accepts the export at t, D imports at t+d, and an
// order whose deadline fell in between is dead on arrival through nobody's fault. If
// the bundle failed as a unit, the value would sit in the in-flight state with its
// only consuming transaction permanently invalid, and the trader would have to be
// rescued by hand.
//
// So the money always lands. A rejected order leaves the balance available on D — a
// state the trader can act on alone, by placing another order or exporting home.
//
// THE ONE FAILURE THAT IS REAL is an invalid or already-consumed claim. That is not
// an order problem; it means the transfer object itself is not admissible, and there
// is nothing to land.

// Outcome is what became of the order half of the bundle.
type Outcome uint8

const (
	Placed Outcome = iota
	Rejected
)

func (o Outcome) String() string {
	switch o {
	case Placed:
		return "placed"
	case Rejected:
		return "rejected"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(o))
	}
}

// Result reports both halves. Imported is always true when the error is nil — the
// field exists so the record is explicit rather than implied by the absence of an
// error, since "the money landed" is the fact a trader most needs stated.
type Result struct {
	Imported bool
	Outcome  Outcome
	OrderID  ids.ID
	// Reason is set when Outcome is Rejected. It is a value to report, not an
	// error to propagate: propagating it is exactly the mistake that would roll
	// the import back.
	Reason error
}

// ImportAndOrder consumes a transfer object, credits its beneficiary, and then tries
// to place the accompanying order against the newly available balance.
//
// It returns an error ONLY when the claim itself is inadmissible. Once the credit
// lands, every subsequent disappointment — expired deadline, bad signature, consumed
// nonce, insufficient balance for the lots requested — comes back as a Rejected
// Result with a Reason, and the value stays on D.
//
// lots is the order quantity in the book's units, converted by the caller from
// Order.Input using the market's lot size. That conversion needs market metadata,
// which deliberately does not live in this package.
func ImportAndOrder(
	c *Custody, b *Book,
	vc VerifiedClaim, witness []byte, ctx OrderContext, lots uint64,
) (Result, error) {
	// The only real failure. Nothing has moved yet. The claim arrives ALREADY
	// VERIFIED — an unattested transfer cannot reach this function, because
	// VerifiedClaim is unimplementable outside this package.
	cl := vc.Claim()
	if err := c.Import(vc); err != nil {
		return Result{}, err
	}

	// From here the money has landed and stays landed, whatever else happens.
	res := Result{Imported: true, Outcome: Rejected}

	v, err := VerifyOrder(witness, ctx)
	if err != nil {
		res.Reason = err
		return res, nil
	}
	res.OrderID = v.OrderID()

	order := v.Order()
	if order.Swapper != cl.Beneficiary {
		res.Reason = fmt.Errorf("%w: claim credits %s, order is signed by %s",
			ErrOrderScope, cl.Beneficiary, order.Swapper)
		return res, nil
	}
	if order.Input.Asset != cl.Asset {
		res.Reason = fmt.Errorf("%w: claim carries %s, order spends %s",
			ErrExecAsset, cl.Asset, order.Input.Asset)
		return res, nil
	}
	if lots == 0 {
		res.Reason = ErrZeroOrder
		return res, nil
	}

	// Reserve the value, then open the lots. If either refuses, nothing is left
	// half-done: the reserve is undone before returning, so a rejected order leaves
	// the balance exactly as the import left it.
	amount := order.Input.Big()
	if err := c.Reserve(order.Swapper, order.Input.Asset, amount); err != nil {
		res.Reason = err
		return res, nil
	}
	if err := b.Place(res.OrderID, lots); err != nil {
		if undo := c.Unreserve(order.Swapper, order.Input.Asset, amount); undo != nil {
			// Unreachable in practice — we reserved this exact amount one line
			// ago — but a silent failure here would leak value out of available
			// balance, so it is surfaced as the real error it would be.
			return res, fmt.Errorf("dexprotocol: could not undo reservation after a rejected order: %w", undo)
		}
		res.Reason = err
		return res, nil
	}

	// The nonce is consumed only now, when the order is actually live. Burning it
	// on a rejected order would cost the trader a position they never used.
	if ctx.Nonces != nil {
		if err := ctx.Nonces.Consume(order.Swapper, order.Nonce); err != nil {
			return res, fmt.Errorf("dexprotocol: order placed but its nonce was already consumed: %w", err)
		}
	}

	res.Outcome = Placed
	res.Reason = nil
	return res, nil
}

// Cancel retires an order's open lots and returns the corresponding value to
// available balance. It is the D-local counterpart to placing: no boundary crossing,
// no certificate, no C block.
//
// The value returned is derived from the LOTS actually retired, never from an amount
// the caller supplies — a caller-supplied amount is how a cancel comes to refund more
// than the order ever committed.
func Cancel(c *Custody, b *Book, orderID ids.ID, owner common.Address, asset ids.ID, lotSize *big.Int) (uint64, error) {
	retired, err := b.Cancel(orderID)
	if err != nil {
		return 0, err
	}
	if retired == 0 {
		return 0, nil
	}
	value := new(big.Int).Mul(new(big.Int).SetUint64(retired), lotSize)
	if err := c.Unreserve(owner, asset, value); err != nil {
		return 0, fmt.Errorf("dexprotocol: cancelled %d lots but could not release their value: %w", retired, err)
	}
	return retired, nil
}
