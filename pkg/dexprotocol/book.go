// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"fmt"

	"github.com/luxfi/ids"
)

// book.go is the third conservation proof, and the only one denominated in LOTS.
//
// The three are deliberately separate, and none of them can stand in for another:
//
//	cross-chain   C + C->D + D + D->C            constant          token base units
//	D-local       owned == available + reserved                    token base units
//	order-local   original == open + reserved + traded + canceled  LOTS
//
// Mixing the units is how a lot count gets added to a wei balance, so the two
// domains are connected by the market's lot size and never merged into one type.
//
// WHY FOUR TERMS AND NOT THREE. `open` and `reserved` are different things and
// collapsing them loses the property that matters: open quantity is restable and
// cancellable, reserved quantity is committed to a match already in flight. A cancel
// that could reach reserved quantity would let the same lots be cancelled and traded,
// which is the order-book form of a double-spend.
//
// So Cancel takes the OPEN quantity only. The remainder resolves itself when the
// match it is committed to either trades or comes back.

var (
	ErrNoOrder      = errors.New("dexprotocol: no order under that id")
	ErrOrderExists  = errors.New("dexprotocol: an order already exists under that id")
	ErrNotOpen      = errors.New("dexprotocol: insufficient open quantity")
	ErrNotCommitted = errors.New("dexprotocol: insufficient reserved quantity")
	ErrBookBroken   = errors.New("dexprotocol: an order's quantities do not sum to what was placed")
	ErrZeroOrder    = errors.New("dexprotocol: an order must have a non-zero quantity")
)

// orderState is one order's quantity account. Every transition moves lots between
// terms and never changes their sum, so conservation is a property of the shape
// rather than of remembering to update a total.
type orderState struct {
	original uint64
	open     uint64
	reserved uint64
	traded   uint64
	canceled uint64
}

func (o *orderState) sum() uint64 { return o.open + o.reserved + o.traded + o.canceled }

// Book is the D-side order accounting. It holds no value — Custody does that — and
// no prices. It answers exactly one question: where did an order's lots go.
type Book struct {
	orders map[ids.ID]*orderState
}

func NewBook() *Book { return &Book{orders: make(map[ids.ID]*orderState)} }

// Place opens an order for a quantity. All of it starts open.
func (b *Book) Place(orderID ids.ID, quantity uint64) error {
	if quantity == 0 {
		return ErrZeroOrder
	}
	if _, exists := b.orders[orderID]; exists {
		return fmt.Errorf("%w: %s", ErrOrderExists, orderID)
	}
	b.orders[orderID] = &orderState{original: quantity, open: quantity}
	return nil
}

// Commit moves open lots to reserved when the matcher takes them for a match.
func (b *Book) Commit(orderID ids.ID, quantity uint64) error {
	o, ok := b.orders[orderID]
	if !ok {
		return fmt.Errorf("%w: %s", ErrNoOrder, orderID)
	}
	if quantity == 0 || o.open < quantity {
		return fmt.Errorf("%w: %d open, committing %d", ErrNotOpen, o.open, quantity)
	}
	o.open -= quantity
	o.reserved += quantity
	return nil
}

// Trade settles reserved lots. This is the only terminal transition that represents
// value actually changing hands, and it is irreversible.
func (b *Book) Trade(orderID ids.ID, quantity uint64) error {
	o, ok := b.orders[orderID]
	if !ok {
		return fmt.Errorf("%w: %s", ErrNoOrder, orderID)
	}
	if quantity == 0 || o.reserved < quantity {
		return fmt.Errorf("%w: %d reserved, trading %d", ErrNotCommitted, o.reserved, quantity)
	}
	o.reserved -= quantity
	o.traded += quantity
	return nil
}

// Return sends reserved lots back to open — a match that did not happen. The order
// is still live and the lots are restable, which is the whole point of an order
// outliving any single match attempt.
func (b *Book) Return(orderID ids.ID, quantity uint64) error {
	o, ok := b.orders[orderID]
	if !ok {
		return fmt.Errorf("%w: %s", ErrNoOrder, orderID)
	}
	if quantity == 0 || o.reserved < quantity {
		return fmt.Errorf("%w: %d reserved, returning %d", ErrNotCommitted, o.reserved, quantity)
	}
	o.reserved -= quantity
	o.open += quantity
	return nil
}

// Cancel retires the OPEN quantity and returns how much was retired. Lots committed
// to an in-flight match are deliberately untouched: cancelling them would let the
// same lots be both cancelled and traded. They resolve when that match does, and the
// trader may cancel again afterwards to retire whatever came back.
//
// Cancelling an order with nothing open is not an error — it is a no-op returning 0,
// because "cancel everything you can" is the honest meaning of the request and
// failing it would make the caller inspect state first to avoid an error it cannot
// prevent.
func (b *Book) Cancel(orderID ids.ID) (uint64, error) {
	o, ok := b.orders[orderID]
	if !ok {
		return 0, fmt.Errorf("%w: %s", ErrNoOrder, orderID)
	}
	retired := o.open
	o.canceled += retired
	o.open = 0
	return retired, nil
}

// Quantities reports one order's four terms plus what it was placed for.
func (b *Book) Quantities(orderID ids.ID) (original, open, reserved, traded, canceled uint64, ok bool) {
	o, exists := b.orders[orderID]
	if !exists {
		return 0, 0, 0, 0, 0, false
	}
	return o.original, o.open, o.reserved, o.traded, o.canceled, true
}

// Live reports whether the order still has lots that could yet trade.
func (b *Book) Live(orderID ids.ID) bool {
	o, ok := b.orders[orderID]
	return ok && (o.open > 0 || o.reserved > 0)
}

// Conserved asserts the order-local invariant for every order:
//
//	original == open + reserved + traded + canceled
//
// Assert it after every transition. Each one moves lots between two terms, so a
// failure means a term was written rather than moved — which is the mutation this
// shape exists to make visible.
func (b *Book) Conserved() error {
	for id, o := range b.orders {
		if got := o.sum(); got != o.original {
			return fmt.Errorf("%w: order %s placed %d, holds open %d + reserved %d + traded %d + canceled %d = %d",
				ErrBookBroken, id, o.original, o.open, o.reserved, o.traded, o.canceled, got)
		}
	}
	return nil
}
