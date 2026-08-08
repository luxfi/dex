// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"math/big"
	"testing"

	"github.com/luxfi/ids"
)

// importorder_test.go pins the semantics that look wrong until you see the failure
// they prevent: the import is unconditional, the order is best effort, and order
// failure is DATA rather than transaction failure.

const testLot = 10 // token base units per lot, for these fixtures

// arrivingOrder builds an order alice signs, spending exactly what the claim carries.
func arrivingOrder(input uint64) Order {
	o := testOrder()
	o.Swapper = alice
	o.Recipient = alice
	o.Input = AssetAmount{Asset: testUSDC, Amount: amount(input)}
	o.MinOutput = AssetAmount{Asset: testLUX, Amount: amount(1)}
	return o
}

func aliceContext() OrderContext {
	ctx := okContext()
	ctx.Signer = recoverAs{addr: alice}
	return ctx
}

func TestImportAndOrderPlacesBoth(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)

	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), aliceContext(), 10)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Imported || res.Outcome != Placed || res.Reason != nil {
		t.Fatalf("got %+v, want imported/placed/no reason", res)
	}
	// Value is reserved against the order, lots are open on the book.
	if got := held(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("reserved %s, want 100", got)
	}
	if got := avail(t, c, alice, testUSDC); got.Sign() != 0 {
		t.Fatalf("available %s, want 0 — it should all be committed", got)
	}
	_, open, _, _, _, ok := b.Quantities(res.OrderID)
	if !ok || open != 10 {
		t.Fatalf("book open %d ok %v, want 10/true", open, ok)
	}
	if err := b.Conserved(); err != nil {
		t.Fatal(err)
	}
}

// THE NORMATIVE CASE. The deadline expired in transit — C accepted the export before
// it, D imports after it, nobody did anything wrong. The money must still land, or
// the trader is stranded holding an in-flight object whose only consuming
// transaction is permanently invalid.
func TestExpiredOrderStillLandsTheMoney(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)
	o.Deadline = 500
	ctx := aliceContext()
	ctx.BlockTime = 900 // D imports well past the deadline

	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), ctx, 10)
	if err != nil {
		t.Fatalf("an expired order must not fail the import: %v", err)
	}
	if !res.Imported {
		t.Fatal("the money did not land")
	}
	if res.Outcome != Rejected || !errors.Is(res.Reason, ErrOrderExpired) {
		t.Fatalf("got %+v, want rejected/ErrOrderExpired", res)
	}
	// Available, not reserved — the trader can act alone.
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("available %s, want 100", got)
	}
	if got := held(t, c, alice, testUSDC); got.Sign() != 0 {
		t.Fatalf("a rejected order still reserved %s", got)
	}
	// And they can take it home without anyone's help.
	if _, err := c.Export(alice, testUSDC, n(100), cChain, ids.ID{0xE1}); err != nil {
		t.Fatalf("trader cannot recover their own funds: %v", err)
	}
}

// Every other order-side disappointment behaves the same way: data, not failure.
func TestEveryOrderRejectionStillLandsTheMoney(t *testing.T) {
	cases := map[string]func(*Order, *OrderContext){
		"bad signature": func(_ *Order, ctx *OrderContext) {
			ctx.Signer = recoverAs{addr: bob}
		},
		"consumed nonce": func(o *Order, ctx *OrderContext) {
			_ = ctx.Nonces.Consume(alice, o.Nonce)
		},
		"order is someone else's": func(o *Order, _ *OrderContext) {
			o.Swapper = bob
		},
		"spends a different asset": func(o *Order, _ *OrderContext) {
			o.Input.Asset = testLUX
		},
		"wants more than the claim carried": func(o *Order, _ *OrderContext) {
			o.Input.Amount = amount(1000)
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c, b := NewCustody(dChain), NewBook()
			o := arrivingOrder(100)
			ctx := aliceContext()
			mutate(&o, &ctx)
			// "order is someone else's" must still be signed by whoever claims it.
			if o.Swapper == bob {
				ctx.Signer = recoverAs{addr: bob}
			}

			res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), ctx, 10)
			if err != nil {
				t.Fatalf("order rejection became a transaction failure: %v", err)
			}
			if !res.Imported {
				t.Fatal("the money did not land")
			}
			if res.Outcome != Rejected || res.Reason == nil {
				t.Fatalf("got %+v, want rejected with a reason", res)
			}
			if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
				t.Fatalf("available %s, want 100", got)
			}
			if got := held(t, c, alice, testUSDC); got.Sign() != 0 {
				t.Fatalf("a rejected order reserved %s", got)
			}
			if err := b.Conserved(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// The one failure that IS real. An inadmissible claim is not an order problem —
// there is nothing to land, so it must be an error rather than a Result.
func TestInadmissibleClaimIsARealFailure(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)
	cl := attested(t, claim(1, alice, testUSDC, 100))

	if _, err := ImportAndOrder(c, b, cl, orderWitness(o), aliceContext(), 10); err != nil {
		t.Fatal(err)
	}
	// Replaying the same claim must fail outright, not credit again.
	res, err := ImportAndOrder(c, b, cl, orderWitness(o), aliceContext(), 10)
	if !errors.Is(err, ErrClaimConsumed) {
		t.Fatalf("got %v, want ErrClaimConsumed", err)
	}
	if res.Imported {
		t.Fatal("a refused claim reported itself imported")
	}
	// Balance untouched by the refused replay: 100 in, all of it reserved once.
	total := new(big.Int).Add(avail(t, c, alice, testUSDC), held(t, c, alice, testUSDC))
	if total.Cmp(n(100)) != 0 {
		t.Fatalf("the replay moved the balance to %s", total)
	}
}

// A rejected order must NOT burn the trader's nonce — they never got to use it, and
// an unordered nonce is a position they chose.
func TestRejectedOrderDoesNotBurnTheNonce(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)
	o.Deadline = 500
	ctx := aliceContext()
	ctx.BlockTime = 900

	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), ctx, 10)
	if err != nil || res.Outcome != Rejected {
		t.Fatalf("setup: %+v %v", res, err)
	}
	if ctx.Nonces.Used(alice, o.Nonce) {
		t.Fatal("a rejected order consumed the nonce")
	}
	// The trader re-signs with a valid deadline and the SAME nonce, and it works.
	good := arrivingOrder(100)
	good.Nonce = o.Nonce
	ctx.BlockTime = 100
	if _, err := VerifyOrder(orderWitness(good), ctx); err != nil {
		t.Fatalf("the nonce was not reusable after a rejected order: %v", err)
	}
}

// A placed order DOES consume the nonce, at the moment it becomes live.
func TestPlacedOrderConsumesTheNonce(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)
	ctx := aliceContext()

	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), ctx, 10)
	if err != nil || res.Outcome != Placed {
		t.Fatalf("setup: %+v %v", res, err)
	}
	if !ctx.Nonces.Used(alice, o.Nonce) {
		t.Fatal("a placed order left its nonce unconsumed — it is replayable")
	}
}

// The whole arriving-trader round trip, entirely D-local after the one crossing:
// import + place, partially trade, cancel the rest, take the remainder home.
func TestArrivingTraderRoundTrip(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)

	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), aliceContext(), 10)
	if err != nil || res.Outcome != Placed {
		t.Fatalf("setup: %+v %v", res, err)
	}

	// Four of ten lots trade. No boundary crossing, no certificate, no C block.
	if err := b.Commit(res.OrderID, 4); err != nil {
		t.Fatal(err)
	}
	if err := b.Trade(res.OrderID, 4); err != nil {
		t.Fatal(err)
	}
	// Value moves between owners for the traded part.
	if err := c.Import(attested(t, claim(2, bob, testLUX, 4))); err != nil {
		t.Fatal(err)
	}
	if err := c.Reserve(bob, testLUX, n(4)); err != nil {
		t.Fatal(err)
	}
	if err := c.Trade(alice, testUSDC, n(4*testLot), bob, testLUX, n(4)); err != nil {
		t.Fatal(err)
	}
	if err := b.Conserved(); err != nil {
		t.Fatal(err)
	}

	// Cancel the remaining six lots; their value returns to available.
	retired, err := Cancel(c, b, res.OrderID, alice, testUSDC, big.NewInt(testLot))
	if err != nil {
		t.Fatal(err)
	}
	if retired != 6 {
		t.Fatalf("cancel retired %d lots, want 6", retired)
	}
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(60)) != 0 {
		t.Fatalf("available %s after cancel, want 60", got)
	}
	if got := held(t, c, alice, testUSDC); got.Sign() != 0 {
		t.Fatalf("still holding %s reserved after a full cancel", got)
	}

	// Take it home.
	p, err := c.Export(alice, testUSDC, n(60), cChain, ids.ID{0xE9})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Deliver(p.ClaimID()); err != nil {
		t.Fatal(err)
	}
	if err := b.Conserved(); err != nil {
		t.Fatal(err)
	}
	if err := c.NoNegative(); err != nil {
		t.Fatal(err)
	}
	original, open, reserved, traded, canceled, _ := b.Quantities(res.OrderID)
	if original != 10 || open != 0 || reserved != 0 || traded != 4 || canceled != 6 {
		t.Fatalf("%d = %d + %d + %d + %d, want 10 = 0 + 0 + 4 + 6",
			original, open, reserved, traded, canceled)
	}
}

// Cancel derives the refund from the LOTS it actually retired, never from a
// caller-supplied amount — that is how a cancel comes to refund more than the order
// committed.
func TestCancelRefundsOnlyWhatItRetired(t *testing.T) {
	c, b := NewCustody(dChain), NewBook()
	o := arrivingOrder(100)
	res, err := ImportAndOrder(c, b, attested(t, claim(1, alice, testUSDC, 100)), orderWitness(o), aliceContext(), 10)
	if err != nil || res.Outcome != Placed {
		t.Fatalf("setup: %+v %v", res, err)
	}
	// Three lots are committed to a match, so only seven are cancellable.
	if err := b.Commit(res.OrderID, 3); err != nil {
		t.Fatal(err)
	}
	retired, err := Cancel(c, b, res.OrderID, alice, testUSDC, big.NewInt(testLot))
	if err != nil {
		t.Fatal(err)
	}
	if retired != 7 {
		t.Fatalf("retired %d, want 7", retired)
	}
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(70)) != 0 {
		t.Fatalf("refunded to %s available, want 70 — the committed 3 lots must stay held", got)
	}
	if got := held(t, c, alice, testUSDC); got.Cmp(n(30)) != 0 {
		t.Fatalf("holding %s, want 30 for the committed lots", got)
	}
}
