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

// export_test.go attacks the D->C leg. The property under attack:
//
//	After a reclaim, no copy of the original transfer object — however delayed,
//	retried or replayed — can ever become importable on C.
//
// The classic way to break it is reclaim-plus-late-delivery: refund on D, then have
// the original write land on C anyway. Every test below tries to reach that state.

func fund(t *testing.T, c *Custody, amt int64) {
	t.Helper()
	if err := c.Import(claim(1, alice, testUSDC, amt)); err != nil {
		t.Fatal(err)
	}
}

// THE ADVERSARIAL CORE. A reclaimed export must never yield a writable object, by
// any route: not Deliver, not Redeliver, not a second Export reusing the id.
func TestReclaimedExportCanNeverBecomeWritable(t *testing.T) {
	c := NewCustody(dChain)
	fund(t, c, 100)
	id := ids.ID{0xE1}

	p, err := c.Export(alice, testUSDC, n(100), cChain, id)
	if err != nil {
		t.Fatal(err)
	}
	if p.ClaimID() != id {
		t.Fatal("pending export lost its id")
	}
	if _, err := c.Reclaim(id); err != nil {
		t.Fatal(err)
	}

	// Attack 1: deliver the reclaimed export.
	if _, err := c.Deliver(id); !errors.Is(err, ErrAlreadyResolved) {
		t.Fatalf("Deliver after Reclaim: got %v, want ErrAlreadyResolved", err)
	}
	// Attack 2: ask for a redelivery of it.
	if _, err := c.Redeliver(id); !errors.Is(err, ErrNotDelivered) {
		t.Fatalf("Redeliver after Reclaim: got %v, want ErrNotDelivered", err)
	}
	// Attack 3: re-export under the same id to manufacture a fresh object.
	if _, err := c.Export(alice, testUSDC, n(100), cChain, id); !errors.Is(err, ErrExportExists) {
		t.Fatalf("re-Export under a reclaimed id: got %v, want ErrExportExists", err)
	}
	// Attack 4: reclaim twice, to double the refund.
	if _, err := c.Reclaim(id); !errors.Is(err, ErrAlreadyResolved) {
		t.Fatalf("second Reclaim: got %v, want ErrAlreadyResolved", err)
	}

	if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("alice has %s after one reclaim, want exactly 100", got)
	}
	if err := c.ExportsExclusive(); err != nil {
		t.Fatal(err)
	}
}

// The mirror attack: a delivered export must never be refundable. Once the object is
// obtainable, a refund would let a late write and the refund both take effect.
func TestDeliveredExportCanNeverBeReclaimed(t *testing.T) {
	c := NewCustody(dChain)
	fund(t, c, 100)
	id := ids.ID{0xE2}

	if _, err := c.Export(alice, testUSDC, n(100), cChain, id); err != nil {
		t.Fatal(err)
	}
	d, err := c.Deliver(id)
	if err != nil {
		t.Fatal(err)
	}
	if d.Claim().Big().Cmp(n(100)) != 0 {
		t.Fatal("deliverable lost its amount")
	}

	if _, err := c.Reclaim(id); !errors.Is(err, ErrReclaimDeliverd) {
		t.Fatalf("Reclaim after Deliver: got %v, want ErrReclaimDeliverd", err)
	}
	if got := avail(t, c, alice, testUSDC); got.Sign() != 0 {
		t.Fatalf("a refused reclaim credited %s back", got)
	}
	if err := c.ExportsExclusive(); err != nil {
		t.Fatal(err)
	}
}

// A lost write is repaired by writing again, for as long as it takes. Redelivery
// must hand back a BYTE-IDENTICAL object, because C's exactly-once import is what
// makes a duplicate arrival harmless — a different object would be a second claim.
func TestRedeliveryIsTheOrdinaryRepairAndIsIdentical(t *testing.T) {
	c := NewCustody(dChain)
	fund(t, c, 100)
	id := ids.ID{0xE3}

	if _, err := c.Export(alice, testUSDC, n(100), cChain, id); err != nil {
		t.Fatal(err)
	}
	first, err := c.Deliver(id)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		again, err := c.Redeliver(id)
		if err != nil {
			t.Fatalf("retry %d refused: %v", i, err)
		}
		if again.Claim() != first.Claim() {
			t.Fatalf("retry %d produced a different object", i)
		}
	}

	// And the receiving side treats the duplicates as the replay they are.
	dest := NewCustody(cChain)
	if err := dest.Import(first.Claim()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := dest.Import(first.Claim()); !errors.Is(err, ErrClaimConsumed) {
			t.Fatalf("duplicate arrival %d: got %v, want ErrClaimConsumed", i, err)
		}
	}
	if got := avail(t, dest, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("five writes and four arrivals credited %s, want 100", got)
	}
}

// THE FULL DOUBLE-SPEND ATTEMPT, end to end across two chains. Reclaim on D, then
// try to make the object land on C anyway. The value must exist in exactly one place
// at the end.
func TestReclaimAndLateDeliveryCannotBothTakeEffect(t *testing.T) {
	d := NewCustody(dChain)
	cSide := NewCustody(cChain)
	fund(t, d, 100)
	id := ids.ID{0xE4}

	if _, err := d.Export(alice, testUSDC, n(100), cChain, id); err != nil {
		t.Fatal(err)
	}
	// The relayer is slow. D gives up and reclaims.
	if _, err := d.Reclaim(id); err != nil {
		t.Fatal(err)
	}

	// The relayer now tries to obtain the bytes it never got. There is no route:
	// a PendingExport never exposed a Claim, and the record is no longer pending.
	if _, err := d.Deliver(id); err == nil {
		t.Fatal("a reclaimed export produced a writable object — this is the double-spend")
	}
	if _, err := d.Redeliver(id); err == nil {
		t.Fatal("a reclaimed export produced a writable object via Redeliver")
	}

	// Value exists once, on D.
	if got := avail(t, d, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("D holds %s, want 100", got)
	}
	if _, _, ok := cSide.Balance(alice, testUSDC); ok {
		t.Fatal("C credited something for an export that was reclaimed")
	}
	total := new(big.Int).Add(d.Owned(testUSDC), cSide.Owned(testUSDC))
	if total.Cmp(n(100)) != 0 {
		t.Fatalf("the value exists %s times over", total)
	}
}

// Reserved balance cannot be exported even into a pending record — the earmark must
// not be a way around the reservation.
func TestExportCannotEarmarkReservedValue(t *testing.T) {
	c := NewCustody(dChain)
	fund(t, c, 100)
	if err := c.Reserve(alice, testUSDC, n(90)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Export(alice, testUSDC, n(11), cChain, ids.ID{0xE5}); !errors.Is(err, ErrNoBalance) {
		t.Fatalf("got %v, want ErrNoBalance", err)
	}
	if got := held(t, c, alice, testUSDC); got.Cmp(n(90)) != 0 {
		t.Fatalf("reserved is %s after a refused export, want 90", got)
	}
}

// Pending value is still D's — it must count on D's side of the rail equation, or
// every open export would read as a shortfall.
func TestPendingValueStillCountsAsDs(t *testing.T) {
	c := NewCustody(dChain)
	fund(t, c, 100)
	imported, delivered := n(100), new(big.Int)

	if err := c.Conserved(testUSDC, imported, delivered); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Export(alice, testUSDC, n(40), cChain, ids.ID{0xE6}); err != nil {
		t.Fatal(err)
	}
	// Debited from available, still D's.
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(60)) != 0 {
		t.Fatalf("available %s, want 60", got)
	}
	if got := c.Earmarked(testUSDC); got.Cmp(n(40)) != 0 {
		t.Fatalf("earmarked %s, want 40", got)
	}
	if err := c.Conserved(testUSDC, imported, delivered); err != nil {
		t.Fatalf("a pending export read as a shortfall: %v", err)
	}
	// Committing it moves the value out of D's column and into the rail's.
	if _, err := c.Deliver(ids.ID{0xE6}); err != nil {
		t.Fatal(err)
	}
	delivered.Add(delivered, n(40))
	if got := c.InFlight(testUSDC); got.Cmp(n(40)) != 0 {
		t.Fatalf("in flight %s, want 40", got)
	}
	if err := c.Conserved(testUSDC, imported, delivered); err != nil {
		t.Fatal(err)
	}
	// Reclaiming it after delivery is refused, so the columns cannot both hold it.
	if _, err := c.Reclaim(ids.ID{0xE6}); !errors.Is(err, ErrReclaimDeliverd) {
		t.Fatalf("got %v, want ErrReclaimDeliverd", err)
	}
}

// Random adversarial sequencing. Every export is driven through an arbitrary mix of
// deliver, redeliver, reclaim and re-export attempts, asserting after each step that
// no id is ever both reclaimable and writable, and that the rail equation holds.
func TestExportLifecycleUnderRandomAttack(t *testing.T) {
	c := NewCustody(dChain)
	rng := rand.New(rand.NewSource(17))
	imported, delivered := new(big.Int), new(big.Int)

	// Seed with plenty of balance.
	for i := 0; i < 10; i++ {
		cl := claim(byte(i), alice, testUSDC, 1000)
		if err := c.Import(cl); err != nil {
			t.Fatal(err)
		}
		imported.Add(imported, n(1000))
	}

	seq := 0
	writable := map[ids.ID]bool{}
	for step := 0; step < 500; step++ {
		seq++
		id := ids.ID{0xE0, byte(seq), byte(seq >> 8)}
		switch rng.Intn(4) {
		case 0: // open one
			if _, err := c.Export(alice, testUSDC, n(int64(rng.Intn(20)+1)), cChain, id); err != nil &&
				!errors.Is(err, ErrNoBalance) {
				t.Fatalf("step %d export: %v", step, err)
			}
		case 1: // commit an arbitrary open one
			for open := range c.pending {
				d, err := c.Deliver(open)
				if err != nil {
					t.Fatalf("step %d deliver: %v", step, err)
				}
				delivered.Add(delivered, d.Claim().Big())
				writable[open] = true
				break
			}
		case 2: // reclaim an arbitrary open one
			for open := range c.pending {
				if _, err := c.Reclaim(open); err != nil {
					t.Fatalf("step %d reclaim: %v", step, err)
				}
				if writable[open] {
					t.Fatalf("step %d: reclaimed an id that was already writable", step)
				}
				break
			}
		case 3: // try to reclaim something committed — must always fail
			for done := range c.delivered {
				if _, err := c.Reclaim(done); !errors.Is(err, ErrReclaimDeliverd) {
					t.Fatalf("step %d: committed export was reclaimable: %v", step, err)
				}
				break
			}
		}
		if err := c.ExportsExclusive(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if err := c.Conserved(testUSDC, imported, delivered); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if err := c.NoNegative(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	t.Logf("imported %s, delivered %s, owned %s, earmarked %s",
		imported, delivered, c.Owned(testUSDC), c.Earmarked(testUSDC))
}
