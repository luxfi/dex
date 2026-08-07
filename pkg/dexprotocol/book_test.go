// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"math/big"
	"math/rand"
	"testing"

	"github.com/luxfi/ids"
)

func bookSound(t *testing.T, b *Book, when string) {
	t.Helper()
	if err := b.Conserved(); err != nil {
		t.Fatalf("%s: %v", when, err)
	}
}

// The ordinary CLOB life of a partially executed order, entirely on D — no C parent,
// no certificate, no settlement round trip. This is what importing first buys.
func TestPartialExecutionIsPurelyLocal(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x01}
	if err := b.Place(id, 100); err != nil {
		t.Fatal(err)
	}
	bookSound(t, b, "placed")

	// Three partial trades: 30, 20, 15.
	for _, q := range []uint64{30, 20, 15} {
		if err := b.Commit(id, q); err != nil {
			t.Fatal(err)
		}
		bookSound(t, b, "committed")
		if err := b.Trade(id, q); err != nil {
			t.Fatal(err)
		}
		bookSound(t, b, "traded")
	}

	_, open, reserved, traded, canceled, _ := b.Quantities(id)
	if open != 35 || reserved != 0 || traded != 65 || canceled != 0 {
		t.Fatalf("open %d reserved %d traded %d canceled %d, want 35/0/65/0", open, reserved, traded, canceled)
	}

	// Cancel the rest.
	retired, err := b.Cancel(id)
	if err != nil {
		t.Fatal(err)
	}
	if retired != 35 {
		t.Fatalf("cancel retired %d, want 35", retired)
	}
	bookSound(t, b, "cancelled")

	original, open, reserved, traded, canceled, _ := b.Quantities(id)
	if original != 100 || open != 0 || reserved != 0 || traded != 65 || canceled != 35 {
		t.Fatalf("final %d = %d + %d + %d + %d", original, open, reserved, traded, canceled)
	}
	if b.Live(id) {
		t.Fatal("a fully resolved order is still live")
	}
}

// THE ORDER-BOOK FORM OF A DOUBLE-SPEND. Cancel must not reach lots committed to a
// match already in flight, or the same lots are both cancelled and traded.
func TestCancelCannotReachCommittedLots(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x02}
	if err := b.Place(id, 100); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(id, 60); err != nil {
		t.Fatal(err)
	}

	retired, err := b.Cancel(id)
	if err != nil {
		t.Fatal(err)
	}
	if retired != 40 {
		t.Fatalf("cancel retired %d, want only the 40 that were open", retired)
	}
	// The committed 60 survives the cancel and can still trade.
	if err := b.Trade(id, 60); err != nil {
		t.Fatalf("committed lots did not survive the cancel: %v", err)
	}
	bookSound(t, b, "traded after cancel")

	original, open, reserved, traded, canceled, _ := b.Quantities(id)
	if original != 100 || open != 0 || reserved != 0 || traded != 60 || canceled != 40 {
		t.Fatalf("%d = %d + %d + %d + %d, want 100 = 0 + 0 + 60 + 40",
			original, open, reserved, traded, canceled)
	}
}

// A match that does not happen returns its lots to open — the order outlives the
// match attempt, which is what makes a resting order useful.
func TestReturnedLotsRestAgain(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x03}
	if err := b.Place(id, 50); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(id, 50); err != nil {
		t.Fatal(err)
	}
	// Nothing is cancellable while it is all committed.
	if retired, err := b.Cancel(id); err != nil || retired != 0 {
		t.Fatalf("cancel retired %d err %v, want 0/nil", retired, err)
	}
	if err := b.Return(id, 50); err != nil {
		t.Fatal(err)
	}
	bookSound(t, b, "returned")
	if !b.Live(id) {
		t.Fatal("an order whose match failed should still be live")
	}
	// And now it is cancellable, including the lots that came back.
	retired, err := b.Cancel(id)
	if err != nil {
		t.Fatal(err)
	}
	if retired != 50 {
		t.Fatalf("retired %d, want 50", retired)
	}
}

// Cancelling twice, or cancelling an order with nothing open, is a no-op rather than
// an error: "retire whatever you can" is the honest reading, and failing it would
// force the caller to inspect state to avoid an error it cannot prevent.
func TestCancelIsIdempotent(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x04}
	if err := b.Place(id, 10); err != nil {
		t.Fatal(err)
	}
	if retired, err := b.Cancel(id); err != nil || retired != 10 {
		t.Fatalf("first cancel: %d, %v", retired, err)
	}
	for i := 0; i < 3; i++ {
		retired, err := b.Cancel(id)
		if err != nil {
			t.Fatalf("repeat cancel %d errored: %v", i, err)
		}
		if retired != 0 {
			t.Fatalf("repeat cancel %d retired %d, want 0", i, retired)
		}
	}
	_, _, _, _, canceled, _ := b.Quantities(id)
	if canceled != 10 {
		t.Fatalf("canceled total is %d after four cancels, want 10", canceled)
	}
	bookSound(t, b, "after repeated cancels")
}

func TestBookRefusesOverCommitAndOverTrade(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x05}
	if err := b.Place(id, 10); err != nil {
		t.Fatal(err)
	}
	if err := b.Commit(id, 11); !errors.Is(err, ErrNotOpen) {
		t.Fatalf("over-commit: got %v, want ErrNotOpen", err)
	}
	if err := b.Commit(id, 10); err != nil {
		t.Fatal(err)
	}
	if err := b.Trade(id, 11); !errors.Is(err, ErrNotCommitted) {
		t.Fatalf("over-trade: got %v, want ErrNotCommitted", err)
	}
	if err := b.Return(id, 11); !errors.Is(err, ErrNotCommitted) {
		t.Fatalf("over-return: got %v, want ErrNotCommitted", err)
	}
	bookSound(t, b, "after refusals")
}

func TestBookRejectsDuplicateAndEmptyOrders(t *testing.T) {
	b := NewBook()
	id := ids.ID{0x06}
	if err := b.Place(id, 0); !errors.Is(err, ErrZeroOrder) {
		t.Fatalf("got %v, want ErrZeroOrder", err)
	}
	if err := b.Place(id, 5); err != nil {
		t.Fatal(err)
	}
	if err := b.Place(id, 5); !errors.Is(err, ErrOrderExists) {
		t.Fatalf("got %v, want ErrOrderExists", err)
	}
	for _, err := range []error{
		b.Commit(ids.ID{0xFF}, 1), b.Trade(ids.ID{0xFF}, 1), b.Return(ids.ID{0xFF}, 1),
	} {
		if !errors.Is(err, ErrNoOrder) {
			t.Fatalf("got %v, want ErrNoOrder", err)
		}
	}
}

// Random sequencing across many orders, asserting the four-term identity after every
// transition. Each transition moves lots between two terms, so a failure means a term
// was written rather than moved.
func TestBookConservesUnderRandomSequencing(t *testing.T) {
	b := NewBook()
	rng := rand.New(rand.NewSource(23))
	var ids32 []ids.ID
	for i := 0; i < 12; i++ {
		id := ids.ID{0xB0, byte(i)}
		if err := b.Place(id, uint64(rng.Intn(500)+50)); err != nil {
			t.Fatal(err)
		}
		ids32 = append(ids32, id)
	}
	bookSound(t, b, "placed")

	var trades, cancels, returns, placed int
	next := len(ids32)
	for step := 0; step < 3000; step++ {
		// Replenish exhausted orders, or the run degenerates into no-ops and the
		// step count flatters coverage it does not have.
		for i, id := range ids32 {
			if !b.Live(id) {
				fresh := ids.ID{0xB0, byte(next), byte(next >> 8)}
				next++
				if err := b.Place(fresh, uint64(rng.Intn(500)+50)); err != nil {
					t.Fatalf("step %d replenish: %v", step, err)
				}
				ids32[i] = fresh
				placed++
			}
		}
		id := ids32[rng.Intn(len(ids32))]
		_, open, reserved, _, _, _ := b.Quantities(id)
		switch rng.Intn(4) {
		case 0:
			if open > 0 {
				q := uint64(rng.Intn(int(open))) + 1
				if err := b.Commit(id, q); err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
			}
		case 1:
			if reserved > 0 {
				q := uint64(rng.Intn(int(reserved))) + 1
				if err := b.Trade(id, q); err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
				trades++
			}
		case 2:
			if reserved > 0 {
				q := uint64(rng.Intn(int(reserved))) + 1
				if err := b.Return(id, q); err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
				returns++
			}
		case 3:
			retired, err := b.Cancel(id)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			if retired > 0 {
				cancels++
			}
		}
		bookSound(t, b, "mid-sequence")
	}
	if trades == 0 || cancels == 0 || returns == 0 {
		t.Fatalf("the run never exercised all three: trades %d cancels %d returns %d",
			trades, cancels, returns)
	}
	t.Logf("%d trades, %d cancels, %d returns, %d orders replenished across 3000 steps",
		trades, cancels, returns, placed)
	if trades+cancels+returns < 1000 {
		t.Fatalf("only %d real transitions in 3000 steps — the run is mostly no-ops",
			trades+cancels+returns)
	}
}

// The book counts LOTS and the custody ledger counts token base units. They are
// connected by the market's lot size and must never be added together — this test
// exists to state that in code, so a future change that merges the types has to
// delete an explicit assertion rather than quietly succeed.
func TestBookAndCustodyStayInDifferentUnits(t *testing.T) {
	b := NewBook()
	c := NewCustody(dChain)
	id := ids.ID{0x07}

	const lotSize = 1_000 // token base units per lot, a market property
	const lots = 7

	if err := c.Import(claim(1, alice, testUSDC, lots*lotSize)); err != nil {
		t.Fatal(err)
	}
	if err := b.Place(id, lots); err != nil {
		t.Fatal(err)
	}
	// Placing an order reserves VALUE in custody and OPENS lots on the book. The
	// conversion happens exactly here, once, at the boundary.
	if err := c.Reserve(alice, testUSDC, n(lots*lotSize)); err != nil {
		t.Fatal(err)
	}

	_, open, _, _, _, _ := b.Quantities(id)
	_, reservedValue, _ := c.Balance(alice, testUSDC)
	if open != lots {
		t.Fatalf("book holds %d lots, want %d", open, lots)
	}
	if reservedValue.Cmp(n(lots*lotSize)) != 0 {
		t.Fatalf("custody holds %s, want %d", reservedValue, lots*lotSize)
	}
	// The numbers differ by exactly the lot size, which is the point: they are not
	// the same quantity and must not share a type.
	if reservedValue.Int64() == int64(open) {
		t.Fatal("lots and token base units coincided — the test is not discriminating")
	}
	bookSound(t, b, "placed")
	if err := c.Conserved(testUSDC, n(lots*lotSize), new(big.Int)); err != nil {
		t.Fatal(err)
	}
}
