// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"fmt"
	"math/big"
)

// bounds.go answers one question: does this certified execution stay inside what the
// order's signer agreed to?
//
// IT IS THE PARTIAL-EXECUTION RULE THAT NEEDS CARE. An order for 100 USDC with a
// minimum of 10 LUX has an obvious meaning when it executes in full and no obvious
// meaning at all when 23 USDC executes. Left informal, two partial executions can
// each look plausible while their sum violates what the trader thought they signed —
// take the good half of the order at the limit and the rest at anything.
//
// So the rule is stated as a RATIO, and the ratio is checked by cross-multiplying:
//
//	executionOutput * orderInput  >=  executionInput * orderMinOutput
//
// which says "this slice is at least as good, per unit of input, as the whole order
// demanded". No division, so no rounding convention to argue about and no place for
// a remainder to be quietly discarded in the trader's disfavour. Both sides are
// products of two 256-bit values, so they are computed in big.Int rather than
// wrapped into 256 bits — a wrapped comparison would silently invert.
//
// The rule composes: if every slice satisfies it, so does their sum, because summing
// the inequalities is exactly summing the numerators over a common denominator. That
// is the property an informal check would not have given us.

var (
	ErrExecOrder    = errors.New("dexprotocol: execution does not name this order")
	ErrExecMarket   = errors.New("dexprotocol: execution is for a different market")
	ErrExecAsset    = errors.New("dexprotocol: execution moves a different asset than the order authorized")
	ErrExecBound    = errors.New("dexprotocol: execution is worse than the order's limit")
	ErrExecTooLarge = errors.New("dexprotocol: execution input exceeds what the order authorized")
	ErrExecAmount   = errors.New("dexprotocol: execution must move a non-zero amount in and out")
)

// Authorized reports whether the execution is one the order permits. Both arguments
// are unforgeable — a VerifiedOrder comes only from VerifyOrder and a
// VerifiedExecution only from VerifyExecution — so this cannot be asked about an
// order nobody signed or an execution nobody certified.
//
// It deliberately does NOT consult the ledger. Whether the order has enough left is
// a running total across executions and belongs to the ledger; whether THIS
// execution is individually permissible is a pure function of the two objects. Two
// separate questions, two separate places, neither answering for the other.
func Authorized(o VerifiedOrder, v VerifiedExecution) error {
	ord := o.Order()
	e := v.Execution()

	if e.OrderID != o.OrderID() {
		return fmt.Errorf("%w: execution names %s, order is %s", ErrExecOrder, e.OrderID, o.OrderID())
	}
	if e.MarketID != ord.Market {
		return fmt.Errorf("%w: execution %s, order %s", ErrExecMarket, e.MarketID, ord.Market)
	}
	if e.Input.Asset != ord.Input.Asset {
		return fmt.Errorf("%w: input %s, order authorized %s", ErrExecAsset, e.Input.Asset, ord.Input.Asset)
	}
	if e.Output.Asset != ord.MinOutput.Asset {
		return fmt.Errorf("%w: output %s, order wanted %s", ErrExecAsset, e.Output.Asset, ord.MinOutput.Asset)
	}

	in := e.Input.Big()
	// No single execution may consume more than the whole order, whatever the
	// running total says. This is the cheap bound; the ledger holds the real one.
	if in.Cmp(ord.Input.Big()) > 0 {
		return fmt.Errorf("%w: %s of %s", ErrExecTooLarge, in, ord.Input.Big())
	}

	// executionOutput * orderInput >= executionInput * orderMinOutput
	lhs := new(big.Int).Mul(e.Output.Big(), ord.Input.Big())
	rhs := new(big.Int).Mul(in, ord.MinOutput.Big())
	if lhs.Cmp(rhs) < 0 {
		return fmt.Errorf("%w: %s*%s < %s*%s",
			ErrExecBound, e.Output.Big(), ord.Input.Big(), in, ord.MinOutput.Big())
	}
	return nil
}
