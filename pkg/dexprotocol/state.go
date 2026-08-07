// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"fmt"

	"github.com/luxfi/ids"
)

// state.go is the D-side lifecycle, built so the bug class is UNWRITABLE rather
// than merely tested.
//
//	          Execution
//	          /       \
//	     Trade         Released
//
// Both terminal, both driven by proof of the accepted C successor — never a timer.
//
// WHY NOT A STATUS FIELD. A status enum makes every illegal transition WRITABLE and
// pushes correctness onto whoever remembers to check it. This seam has already paid
// for that: in the shipping code the intent record is written back only inside the
// same-asset refund branch, so the ordinary outcome of a swap — proceeds in the
// opposite asset — leaves the record Open carrying its whole principal, and the
// deadline reclaim later refunds money already spent, out of other traders' pooled
// reserve. Reproduced on a clean tree: 200 of OUTPUT credited AND 300 of INPUT
// principal reclaimed against one intent.
//
// That defect is a field somebody forgot to set on one code path. It cannot exist
// here, because there is no field to forget: a settled execution is a DIFFERENT
// TYPE under a DIFFERENT KEY PREFIX, and settling MOVES the object out of the
// reserved domain. Afterwards there is no reserved execution to operate on —
// Settle(Trade) and Release(Trade) are not rejected at runtime, they do not compile.
//
// The remedy for the shipping defect is therefore to port the STORAGE LAYOUT, not
// to port the enum forward and add the missing write.
//
// 🚫 Never add UpdateExecStatus(id, status) or anything shaped like it. The
// ExecType/OrdStatus pair carries FIX semantics ON THE WIRE and must never become
// the internal source of truth.

// Storage domains. Transitions MOVE the object between them; nothing is ever
// updated in place. Exclusivity across the three is the ledger's core invariant.
const (
	domainReserved = "executions/reserved/"
	domainTraded   = "executions/traded/"
	domainReleased = "executions/released/"
)

// ReservedExecution is a certified execution whose price and quantity D is holding,
// awaiting C's decision. Its fields are unexported and it has no exported
// constructor: the only way to obtain one is Reserve, which requires a
// VerifiedExecution. So an unauthenticated execution cannot enter the lifecycle.
//
// It is a distinct type from Execution on purpose. Execution is the wire struct and
// anyone can construct one; if Settle took an Execution, settling something nobody
// verified would compile.
type ReservedExecution struct {
	exec Execution
}

func (r ReservedExecution) Execution() Execution { return r.exec }
func (r ReservedExecution) ExecID() ids.ID       { return r.exec.ExecID }
func (r ReservedExecution) Key() string          { return domainReserved + r.exec.ExecID.String() }

// Trade is terminal: C accepted a successor that used the execution. This is the
// ONLY object market data may publish — no reserved execution appears in the trades
// feed, volume, candles or history.
//
// There is no function from Trade back to ReservedExecution, so nothing downstream
// can re-open it.
type Trade struct {
	exec   Execution
	cBlock ids.ID
}

func (t Trade) Execution() Execution { return t.exec }
func (t Trade) ExecID() ids.ID       { return t.exec.ExecID }
func (t Trade) CBlock() ids.ID       { return t.cBlock }
func (t Trade) Key() string          { return domainTraded + t.exec.ExecID.String() }

// Released is terminal, symmetrically: C accepted a successor that did not use the
// execution, so the reservation returns to the book.
type Released struct {
	exec   Execution
	cBlock ids.ID
}

func (r Released) Execution() Execution { return r.exec }
func (r Released) ExecID() ids.ID       { return r.exec.ExecID }
func (r Released) CBlock() ids.ID       { return r.cBlock }
func (r Released) Key() string          { return domainReleased + r.exec.ExecID.String() }

// Reserve admits a verified execution into the reserved domain, AGAINST AN ACCEPTED
// C PARENT. This is rule 1 of the proof system:
//
//	Accepted(P)  permits Reserve(E) with E.CParent == P
//
// Both arguments are unforgeable: VerifiedExecution cannot be constructed outside
// this package, and AcceptedBlock only comes from VerifyAcceptedBlock. So a
// reservation can be taken neither against an execution nobody authenticated nor
// against a C parent nobody proved accepted — which is what makes the
// abandoned-parent case not merely handled but nonexistent.
func Reserve(parent AcceptedBlock, v VerifiedExecution) (ReservedExecution, error) {
	if !parent.verified {
		return ReservedExecution{}, ErrAcceptedUnverified
	}
	e := v.Execution()
	if e.CParent != parent.ID() {
		return ReservedExecution{}, fmt.Errorf("%w: execution scoped to %s, reserving against accepted %s",
			ErrWrongParent, e.CParent, parent.ID())
	}
	return ReservedExecution{exec: e}, nil
}

var (
	ErrWrongSuccessor = errors.New("dexprotocol: the accepted block is not a successor of the execution's C parent")
	ErrNotReserved    = errors.New("dexprotocol: no reserved execution under that id")
	ErrAlreadyOpen    = errors.New("dexprotocol: an execution is already reserved under that id")
	ErrTerminalExists = errors.New("dexprotocol: a terminal execution already exists under that id")
	ErrWrongProof     = errors.New("dexprotocol: the proof does not match the transition being attempted")
	ErrWrongParent    = errors.New("dexprotocol: the execution is not scoped to this accepted C parent")

	ErrOverReserved      = errors.New("dexprotocol: this would commit more than the order authorized")
	ErrAuthorizedChanged = errors.New("dexprotocol: the order's authorized quantity changed between reservations")
	ErrNotConserved      = errors.New("dexprotocol: the order account disagrees with the executions it summarises")
)

// THE ENTIRE PROOF SYSTEM, THREE RULES:
//
//	Accepted(P)                                permits Reserve(E) with E.CParent == P
//	Accepted(Q), Q.Parent==P, E ∈ Q.ExecRoot   permits Settle(E) → Trade
//	Accepted(Q), Q.Parent==P, E ∉ Q.ExecRoot   permits Release(E)
//
// No clocks, no timeouts, no abandoned-parent case, and no race between a reclaim
// and a late settlement. Because P is accepted, exactly one child Q of P is
// eventually accepted, and Q's ExecRoot answers the question permanently — in one
// direction or the other, with equal force.
//
// EVERY VALID RESERVATION IS RESOLVABLE, even when it is useless. If C accepts P, D
// sees Accepted(P), C quickly accepts Q, and only THEN D reserves E against P, that
// reservation can never be settled — but anyone can deliver Accepted(Q) with a
// non-inclusion proof and release it. D may optimise by refusing a reservation when
// it already knows Q, but correctness must not depend on winning that race.

// Settle turns a reserved execution into a Trade, on proof that the accepted child
// of its parent CONSUMED it.
func Settle(r ReservedExecution, q AcceptedBlock, proof ExecProof) (Trade, error) {
	if err := checkSuccessor(r, q); err != nil {
		return Trade{}, err
	}
	if !proof.Included {
		return Trade{}, fmt.Errorf("%w: a settle needs an INCLUSION proof", ErrWrongProof)
	}
	if err := VerifyExecProof(q.ExecRoot(), r.exec.ExecID, proof); err != nil {
		return Trade{}, err
	}
	return Trade{exec: r.exec, cBlock: q.ID()}, nil
}

// Release returns the reservation to the book, on proof that the accepted child of
// its parent did NOT consume it.
//
// Non-inclusion is not a fallback here — it is the other half of the protocol, and
// it carries a real proof against the same committed root, verified by the same
// code path as inclusion (see smt.go).
func Release(r ReservedExecution, q AcceptedBlock, proof ExecProof) (Released, error) {
	if err := checkSuccessor(r, q); err != nil {
		return Released{}, err
	}
	if proof.Included {
		return Released{}, fmt.Errorf("%w: a release needs a NON-INCLUSION proof", ErrWrongProof)
	}
	if err := VerifyExecProof(q.ExecRoot(), r.exec.ExecID, proof); err != nil {
		return Released{}, err
	}
	return Released{exec: r.exec, cBlock: q.ID()}, nil
}

// checkSuccessor enforces that q is the accepted CHILD of the execution's accepted
// parent. Both halves matter: q must be verified (so a zero AcceptedBlock cannot be
// passed), and its parent must be exactly the opportunity the execution was scoped
// to.
func checkSuccessor(r ReservedExecution, q AcceptedBlock) error {
	if !q.verified {
		return ErrAcceptedUnverified
	}
	if q.ParentID() != r.exec.CParent {
		return fmt.Errorf("%w: accepted block's parent %s, execution scoped to %s",
			ErrWrongSuccessor, q.ParentID(), r.exec.CParent)
	}
	return nil
}

// ReservationKey is the D custody key: CChain + CParent + ExecID. It is scoped to
// the parent OPPORTUNITY, never to a candidate block, which is what makes many
// competing candidates harmless — they all name the same reservation and none of
// them moves it.
func ReservationKey(cChainID, cParent, execID ids.ID) string {
	return fmt.Sprintf("reserve/%s/%s/%s", cChainID, cParent, execID)
}

// Ledger holds the three PHYSICALLY DISTINCT domains and enforces exclusivity: an
// ExecID lives in exactly one of them, forever. It is the in-memory reference
// model — a persistent implementation must preserve the same move semantics and the
// same exclusivity check, not add a status column.
type Ledger struct {
	reserved map[ids.ID]ReservedExecution
	traded   map[ids.ID]Trade
	released map[ids.ID]Released
	orders   map[ids.ID]*orderAccount
}

func NewLedger() *Ledger {
	return &Ledger{
		reserved: make(map[ids.ID]ReservedExecution),
		traded:   make(map[ids.ID]Trade),
		released: make(map[ids.ID]Released),
		orders:   make(map[ids.ID]*orderAccount),
	}
}

// orderAccount is the per-order quantity account. Exclusivity across the three
// storage domains says an ExecID cannot be in two places; it says NOTHING about how
// much of an order has been committed. Without this, D can certify ten executions
// against one order for its full quantity, every one of them individually valid,
// and settle whichever the C successors happen to include.
//
// EVERY DEFECT THE LAST REVIEW FOUND WAS A CONSERVATION BUG, not a determinism bug.
// So the quantities are tracked here and asserted after every transition, rather
// than left as a property that happens to hold.
//
// UNITS. These are Execution.Quantity's units — the market's LOTS — never the
// order's Input, which is in token base units. The conversion is the market's lot
// size, which is market metadata and deliberately does not live in this package: a
// caller that knows the market converts once and passes `authorized` in. One
// conversion, at one boundary, in the direction the book already speaks.
type orderAccount struct {
	authorized uint64
	reserved   uint64
	traded     uint64
}

// available is what is left to commit against this order.
func (a *orderAccount) available() uint64 { return a.authorized - a.reserved - a.traded }

// terminalExists reports whether the id has already reached a terminal domain.
// Checked on entry and on every transition, so an id can never be resurrected.
func (l *Ledger) terminalExists(id ids.ID) bool {
	if _, ok := l.traded[id]; ok {
		return true
	}
	_, ok := l.released[id]
	return ok
}

// Reserve admits a verified execution. Refuses an id that is already reserved or
// already terminal — the latter is what stops a replayed certificate from
// re-reserving liquidity that was already settled or released.
// authorized is the order's total quantity in the book's units, converted by the
// caller from the signed Order.Input. It is checked for AGREEMENT across every
// reservation against the same order, so a caller cannot enlarge an order by
// converting it differently the second time.
func (l *Ledger) Reserve(parent AcceptedBlock, v VerifiedExecution, authorized uint64) (ReservedExecution, error) {
	e := v.Execution()
	id := e.ExecID
	if l.terminalExists(id) {
		return ReservedExecution{}, fmt.Errorf("%w: %s", ErrTerminalExists, id)
	}
	if _, ok := l.reserved[id]; ok {
		return ReservedExecution{}, fmt.Errorf("%w: %s", ErrAlreadyOpen, id)
	}
	if authorized == 0 {
		return ReservedExecution{}, fmt.Errorf("%w: order %s authorizes nothing", ErrOverReserved, e.OrderID)
	}
	acct, ok := l.orders[e.OrderID]
	if !ok {
		acct = &orderAccount{authorized: authorized}
	} else if acct.authorized != authorized {
		return ReservedExecution{}, fmt.Errorf("%w: order %s was %d, now %d",
			ErrAuthorizedChanged, e.OrderID, acct.authorized, authorized)
	}
	// THE CHECK THAT MAKES OVER-RESERVATION IMPOSSIBLE. Each execution against an
	// order is individually valid; only the running total says whether D has
	// certified more than the trader signed for.
	if e.Quantity > acct.available() {
		return ReservedExecution{}, fmt.Errorf("%w: order %s has %d left, execution wants %d",
			ErrOverReserved, e.OrderID, acct.available(), e.Quantity)
	}
	r, err := Reserve(parent, v)
	if err != nil {
		return ReservedExecution{}, err
	}
	acct.reserved += e.Quantity
	l.orders[e.OrderID] = acct
	l.reserved[id] = r
	return r, nil
}

// Settle MOVES the execution from the reserved domain to the traded domain. The
// delete is what makes the illegal follow-on unrepresentable rather than merely
// refused: afterwards there is no reserved execution to hand to anything.
func (l *Ledger) Settle(id ids.ID, q AcceptedBlock, proof ExecProof) (Trade, error) {
	r, ok := l.reserved[id]
	if !ok {
		return Trade{}, fmt.Errorf("%w: %s", ErrNotReserved, id)
	}
	t, err := Settle(r, q, proof)
	if err != nil {
		return Trade{}, err
	}
	// Reserved becomes traded: the order's committed total is unchanged, which is
	// exactly right — settling spends what was already held, it does not spend more.
	if acct, ok := l.orders[r.exec.OrderID]; ok {
		acct.reserved -= r.exec.Quantity
		acct.traded += r.exec.Quantity
	}
	delete(l.reserved, id)
	l.traded[id] = t
	return t, nil
}

// Release MOVES the execution from the reserved domain to the released domain.
func (l *Ledger) Release(id ids.ID, q AcceptedBlock, proof ExecProof) (Released, error) {
	r, ok := l.reserved[id]
	if !ok {
		return Released{}, fmt.Errorf("%w: %s", ErrNotReserved, id)
	}
	rel, err := Release(r, q, proof)
	if err != nil {
		return Released{}, err
	}
	// The quantity returns to the order's available balance. This is what lets the
	// same unexecuted order be matched AGAIN — the property the portable Order was
	// designed for. A release that failed to give the quantity back would silently
	// shrink every order that ever lost a race.
	if acct, ok := l.orders[r.exec.OrderID]; ok {
		acct.reserved -= r.exec.Quantity
	}
	delete(l.reserved, id)
	l.released[id] = rel
	return rel, nil
}

// Exclusive reports whether the ledger's core invariant holds: every ExecID appears
// in at most one domain. Exported so callers can assert it after every transition
// rather than trusting the implementation.
func (l *Ledger) Exclusive() error {
	for id := range l.reserved {
		if l.terminalExists(id) {
			return fmt.Errorf("dexprotocol: execution %s is reserved AND terminal", id)
		}
	}
	for id := range l.traded {
		if _, ok := l.released[id]; ok {
			return fmt.Errorf("dexprotocol: execution %s is traded AND released", id)
		}
	}
	return nil
}

// Counts reports each domain's size, for conservation assertions and metrics.
func (l *Ledger) Counts() (reserved, traded, released int) {
	return len(l.reserved), len(l.traded), len(l.released)
}

// Conserved is the quantity invariant, the counterpart to Exclusive. Exclusive says
// an ExecID cannot be in two domains; Conserved says quantity is neither created nor
// destroyed:
//
//	per order:  reserved + traded ≤ authorized
//	per order:  reserved == Σ quantity of its executions still in the reserved domain
//	per order:  traded   == Σ quantity of its executions in the traded domain
//
// The second and third clauses are what make this an ASSERTION rather than a
// restatement: they recompute the totals from the domains themselves, so an
// accounting update that drifted from the objects it claims to summarise is caught
// instead of trusted. Assert it after EVERY transition, not only after swaps that
// went the happy way.
func (l *Ledger) Conserved() error {
	reserved := make(map[ids.ID]uint64, len(l.orders))
	traded := make(map[ids.ID]uint64, len(l.orders))
	for _, r := range l.reserved {
		reserved[r.exec.OrderID] += r.exec.Quantity
	}
	for _, t := range l.traded {
		traded[t.exec.OrderID] += t.exec.Quantity
	}
	for id, acct := range l.orders {
		if acct.reserved+acct.traded > acct.authorized {
			return fmt.Errorf("%w: order %s authorized %d, reserved %d + traded %d",
				ErrOverReserved, id, acct.authorized, acct.reserved, acct.traded)
		}
		if acct.reserved != reserved[id] {
			return fmt.Errorf("%w: order %s account says reserved %d, domain holds %d",
				ErrNotConserved, id, acct.reserved, reserved[id])
		}
		if acct.traded != traded[id] {
			return fmt.Errorf("%w: order %s account says traded %d, domain holds %d",
				ErrNotConserved, id, acct.traded, traded[id])
		}
	}
	// An execution whose order has no account is quantity from nowhere.
	for id := range reserved {
		if _, ok := l.orders[id]; !ok {
			return fmt.Errorf("%w: reserved execution for unaccounted order %s", ErrNotConserved, id)
		}
	}
	for id := range traded {
		if _, ok := l.orders[id]; !ok {
			return fmt.Errorf("%w: traded execution for unaccounted order %s", ErrNotConserved, id)
		}
	}
	return nil
}

// OrderQuantities reports one order's account: what the trader authorized, what is
// held against it right now, and what has been permanently spent. Released quantity
// does not appear because it is not a category — it went back to available, which is
// authorized - reserved - traded.
func (l *Ledger) OrderQuantities(orderID ids.ID) (authorized, reserved, traded uint64, ok bool) {
	acct, ok := l.orders[orderID]
	if !ok {
		return 0, 0, 0, false
	}
	return acct.authorized, acct.reserved, acct.traded, true
}
