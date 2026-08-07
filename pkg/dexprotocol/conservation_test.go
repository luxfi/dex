// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/luxfi/ids"
)

// conservation_test.go covers the quantity invariant. Exclusive() proves an ExecID
// cannot sit in two domains; it says nothing about HOW MUCH of an order has been
// committed, and every defect the last review turned up was a conservation bug
// rather than a determinism bug.

// oneOrder is the OrderID every sampleExecution carries.
var oneOrder = ids.ID{0x01}

// execFor builds an execution against the shared order with a chosen id and size.
func execFor(id byte, qty uint64) Execution {
	e := sampleExecution()
	e.ExecID = ids.ID{0xE0, id}
	e.Quantity = qty
	return e
}

// assertSound checks BOTH ledger invariants. Real callers must do exactly this after
// every transition — that is the whole point of exporting them.
func assertSound(t *testing.T, l *Ledger, when string) {
	t.Helper()
	if err := l.Exclusive(); err != nil {
		t.Fatalf("%s: exclusivity broken: %v", when, err)
	}
	if err := l.Conserved(); err != nil {
		t.Fatalf("%s: conservation broken: %v", when, err)
	}
}

// THE BUG THIS EXISTS TO CATCH. Each execution is individually valid — correctly
// certified, correctly scoped to the accepted parent — and only the running total
// reveals that D has committed more than the trader ever signed for.
func TestCannotReserveBeyondWhatTheOrderAuthorized(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)

	if _, err := l.Reserve(p, verified(t, execFor(1, 600)), 1000); err != nil {
		t.Fatal(err)
	}
	assertSound(t, l, "after first")

	if _, err := l.Reserve(p, verified(t, execFor(2, 300)), 1000); err != nil {
		t.Fatal(err)
	}
	assertSound(t, l, "after second")

	// 600 + 300 + 200 = 1100 > 1000.
	_, err := l.Reserve(p, verified(t, execFor(3, 200)), 1000)
	if !errors.Is(err, ErrOverReserved) {
		t.Fatalf("got %v, want ErrOverReserved", err)
	}
	assertSound(t, l, "after refusal")

	// The exact remainder still fits — the guard is a bound, not a margin.
	if _, err := l.Reserve(p, verified(t, execFor(4, 100)), 1000); err != nil {
		t.Fatalf("exact remainder refused: %v", err)
	}
	assertSound(t, l, "after exact remainder")

	_, reserved, traded, ok := l.OrderQuantities(oneOrder)
	if !ok || reserved != 1000 || traded != 0 {
		t.Fatalf("reserved %d traded %d ok %v, want 1000/0/true", reserved, traded, ok)
	}
}

// A released reservation must return its quantity to the order. This is the property
// the portable Order was designed around: an order whose execution lost the race is
// still live and can be matched again. A release that failed to give the quantity
// back would silently shrink every order that ever lost one.
func TestReleaseReturnsQuantityToTheOrder(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)
	e := execFor(1, 1000) // the order's entire quantity

	if _, err := l.Reserve(p, verified(t, e), 1000); err != nil {
		t.Fatal(err)
	}
	// A second reservation cannot fit while the first is held.
	if _, err := l.Reserve(p, verified(t, execFor(2, 1)), 1000); !errors.Is(err, ErrOverReserved) {
		t.Fatalf("got %v, want ErrOverReserved while fully reserved", err)
	}

	// The accepted child does NOT consume it, so it is released.
	q, proof := acceptedConsuming(t, ids.ID{0xC1}, testParent)
	if _, err := l.Release(e.ExecID, q, proof); err != nil {
		t.Fatal(err)
	}
	assertSound(t, l, "after release")

	if _, reserved, _, _ := l.OrderQuantities(oneOrder); reserved != 0 {
		t.Fatalf("released but %d still held", reserved)
	}
	// And now the order is matchable again, in full.
	if _, err := l.Reserve(p, verified(t, execFor(2, 1000)), 1000); err != nil {
		t.Fatalf("released order could not be matched again: %v", err)
	}
	assertSound(t, l, "after re-reserve")
}

// Settling does NOT free quantity — it spends it. The distinction is the whole
// difference between a partial fill and a refund, and getting it backwards is how a
// trader's order gets spent twice.
func TestSettleSpendsQuantityRatherThanFreeingIt(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)
	e := execFor(1, 700)

	if _, err := l.Reserve(p, verified(t, e), 1000); err != nil {
		t.Fatal(err)
	}
	q, proof := acceptedConsuming(t, ids.ID{0xC1}, testParent, e.ExecID)
	if _, err := l.Settle(e.ExecID, q, proof); err != nil {
		t.Fatal(err)
	}
	assertSound(t, l, "after settle")

	_, reserved, traded, _ := l.OrderQuantities(oneOrder)
	if reserved != 0 || traded != 700 {
		t.Fatalf("reserved %d traded %d, want 0/700", reserved, traded)
	}
	// Only the unspent remainder is still available.
	if _, err := l.Reserve(p, verified(t, execFor(2, 301)), 1000); !errors.Is(err, ErrOverReserved) {
		t.Fatalf("got %v, want ErrOverReserved — settled quantity must not come back", err)
	}
	if _, err := l.Reserve(p, verified(t, execFor(3, 300)), 1000); err != nil {
		t.Fatalf("remainder refused: %v", err)
	}
	assertSound(t, l, "after remainder")
}

// The authorized quantity is a property of the signed order, so it cannot change
// between reservations. Otherwise a caller converting Input to lots differently the
// second time could quietly enlarge the order.
func TestAuthorizedQuantityCannotChange(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)
	if _, err := l.Reserve(p, verified(t, execFor(1, 100)), 1000); err != nil {
		t.Fatal(err)
	}
	_, err := l.Reserve(p, verified(t, execFor(2, 100)), 5000)
	if !errors.Is(err, ErrAuthorizedChanged) {
		t.Fatalf("got %v, want ErrAuthorizedChanged", err)
	}
	assertSound(t, l, "after refusal")
}

func TestOrderAuthorizingNothingIsRefused(t *testing.T) {
	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, execFor(1, 1)), 0); !errors.Is(err, ErrOverReserved) {
		t.Fatalf("got %v, want ErrOverReserved", err)
	}
}

// Conservation must hold after EVERY transition, not merely at the end of a happy
// path. Drive a long random sequence of reserve/settle/release and assert both
// invariants after each one — this is what catches an accounting update that drifts
// from the objects it claims to summarise.
func TestConservationHoldsAcrossRandomLifecycles(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)
	rng := rand.New(rand.NewSource(11))

	const authorized = 100_000
	live := map[ids.ID]Execution{}
	var next byte

	for step := 0; step < 400; step++ {
		switch {
		case len(live) == 0 || rng.Intn(2) == 0:
			next++
			e := execFor(next, uint64(rng.Intn(500)+1))
			if _, err := l.Reserve(p, verified(t, e), authorized); err != nil {
				// Refusal is a legitimate outcome once the order fills up; what
				// must never happen is a refusal that leaves the ledger unsound.
				if !errors.Is(err, ErrOverReserved) {
					t.Fatalf("step %d: %v", step, err)
				}
				assertSound(t, l, "after refused reserve")
				continue
			}
			live[e.ExecID] = e
		default:
			// Pick one live execution and finish it, settling or releasing.
			var pick Execution
			for _, e := range live {
				pick = e
				break
			}
			delete(live, pick.ExecID)
			if rng.Intn(2) == 0 {
				q, proof := acceptedConsuming(t, ids.ID{0xC1}, testParent, pick.ExecID)
				if _, err := l.Settle(pick.ExecID, q, proof); err != nil {
					t.Fatalf("step %d settle: %v", step, err)
				}
			} else {
				q, proof := acceptedConsuming(t, ids.ID{0xC1}, testParent)
				if _, err := l.Release(pick.ExecID, q, proof); err != nil {
					t.Fatalf("step %d release: %v", step, err)
				}
			}
		}
		assertSound(t, l, "mid-lifecycle")
	}

	// Final accounting must agree with the domains, independently recomputed.
	authorizedGot, reserved, traded, ok := l.OrderQuantities(oneOrder)
	if !ok {
		t.Fatal("order account vanished")
	}
	if authorizedGot != authorized {
		t.Fatalf("authorized drifted to %d", authorizedGot)
	}
	if reserved+traded > authorized {
		t.Fatalf("committed %d beyond authorized %d", reserved+traded, authorized)
	}
	nReserved, nTraded, nReleased := l.Counts()
	if nReserved != len(live) {
		t.Fatalf("ledger holds %d reserved, tracker says %d", nReserved, len(live))
	}
	if nTraded+nReleased == 0 {
		t.Fatal("the run never reached a terminal state; the test proved nothing")
	}
	t.Logf("reserved=%d traded=%d released=%d, committed %d/%d",
		nReserved, nTraded, nReleased, reserved+traded, authorized)
}

// Conservation is scoped per order: filling one order must not consume another's
// quantity.
func TestOrdersDoNotShareQuantity(t *testing.T) {
	l := NewLedger()
	p := acceptedParent(t)

	a := execFor(1, 900)
	b := execFor(2, 900)
	b.OrderID = ids.ID{0x02}
	// b is scoped to a different order, so it needs its own accepted-parent scope
	// but the same C parent — that part is unchanged.
	if _, err := l.Reserve(p, verified(t, a), 1000); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Reserve(p, verified(t, b), 1000); err != nil {
		t.Fatalf("second order refused because the first was nearly full: %v", err)
	}
	assertSound(t, l, "two orders")

	if _, reserved, _, _ := l.OrderQuantities(ids.ID{0x02}); reserved != 900 {
		t.Fatalf("order 2 holds %d, want 900", reserved)
	}
	if _, reserved, _, _ := l.OrderQuantities(oneOrder); reserved != 900 {
		t.Fatalf("order 1 holds %d, want 900", reserved)
	}
}
