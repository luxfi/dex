// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"math/big"
	"math/rand"
	"testing"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

var (
	cChain = ids.ID{0x0C}
	dChain = ids.ID{0x0D}
	alice  = common.Address{0xA1}
	bob    = common.Address{0xB0}
)

func n(v int64) *big.Int { return big.NewInt(v) }

func claim(id byte, to common.Address, asset ids.ID, amt int64) Claim {
	c := Claim{ClaimID: ids.ID{0xC0, id}, Source: cChain, Dest: dChain, Beneficiary: to, Asset: asset}
	n(amt).FillBytes(c.Amount[:])
	return c
}

func avail(t *testing.T, c *Custody, who common.Address, asset ids.ID) *big.Int {
	t.Helper()
	a, _, ok := c.Balance(who, asset)
	if !ok {
		return new(big.Int)
	}
	return a
}

func held(t *testing.T, c *Custody, who common.Address, asset ids.ID) *big.Int {
	t.Helper()
	_, r, ok := c.Balance(who, asset)
	if !ok {
		return new(big.Int)
	}
	return r
}

// THE SAFETY PROPERTY OF THE WHOLE RAIL. A transfer object may be consumed exactly
// once. A second import — from a retry, a reorg, two relayers racing, or a restart
// mid-transaction — is a double-spend of real value, so it must be refused rather
// than quietly succeed.
func TestImportIsExactlyOnce(t *testing.T) {
	c := NewCustody(dChain)
	cl := claim(1, alice, testUSDC, 100)

	if err := c.Import(cl); err != nil {
		t.Fatal(err)
	}
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("credited %s, want 100", got)
	}
	for i := 0; i < 3; i++ {
		if err := c.Import(cl); !errors.Is(err, ErrClaimConsumed) {
			t.Fatalf("replay %d: got %v, want ErrClaimConsumed", i, err)
		}
	}
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("a refused replay still moved the balance to %s", got)
	}
}

// A claim addressed to another chain must not credit here, or the rail would deliver
// the same value to two destinations.
func TestImportRefusesForeignClaim(t *testing.T) {
	c := NewCustody(dChain)
	cl := claim(1, alice, testUSDC, 100)
	cl.Dest = ids.ID{0xEE}
	if err := c.Import(cl); !errors.Is(err, ErrClaimChain) {
		t.Fatalf("got %v, want ErrClaimChain", err)
	}
	if got := avail(t, c, alice, testUSDC); got.Sign() != 0 {
		t.Fatalf("foreign claim credited %s", got)
	}
}

// THE OTHER HALF OF "SPENDABLE ON EXACTLY ONE CHAIN". Reserved balance backs an open
// order. If a withdrawal could draw on it, the trader would spend the same value
// twice — once as a resting order and once as an exit.
func TestReservedValueCannotLeaveD(t *testing.T) {
	c := NewCustody(dChain)
	if err := c.Import(claim(1, alice, testUSDC, 100)); err != nil {
		t.Fatal(err)
	}
	if err := c.Reserve(alice, testUSDC, n(80)); err != nil {
		t.Fatal(err)
	}
	// 20 available, 80 reserved. Exporting 21 must fail.
	if _, err := c.Export(alice, testUSDC, n(21), cChain, ids.ID{0x01}); !errors.Is(err, ErrNoBalance) {
		t.Fatalf("got %v, want ErrNoBalance — reserved value must not leave", err)
	}
	// Exactly the available amount is fine.
	cl, err := c.Export(alice, testUSDC, n(20), cChain, ids.ID{0x01})
	if err != nil {
		t.Fatalf("exporting available balance refused: %v", err)
	}
	if cl.Big().Cmp(n(20)) != 0 || cl.Beneficiary != alice || cl.Source != dChain || cl.Dest != cChain {
		t.Fatalf("malformed claim: %+v", cl)
	}
	if got := held(t, c, alice, testUSDC); got.Cmp(n(80)) != 0 {
		t.Fatalf("export disturbed reserved balance: %s", got)
	}
}

// A trade moves value BETWEEN owners and never changes what D owns in total. That is
// the whole point of importing first: a million trades cost zero boundary crossings.
func TestTradeMovesValueWithoutCrossingTheBoundary(t *testing.T) {
	c := NewCustody(dChain)
	if err := c.Import(claim(1, alice, testUSDC, 100)); err != nil {
		t.Fatal(err)
	}
	if err := c.Import(claim(2, bob, testLUX, 5)); err != nil {
		t.Fatal(err)
	}
	if err := c.Reserve(alice, testUSDC, n(100)); err != nil {
		t.Fatal(err)
	}
	if err := c.Reserve(bob, testLUX, n(5)); err != nil {
		t.Fatal(err)
	}

	usdcBefore, luxBefore := c.Owned(testUSDC), c.Owned(testLUX)

	if err := c.Trade(alice, testUSDC, n(100), bob, testLUX, n(5)); err != nil {
		t.Fatal(err)
	}
	if err := c.NoNegative(); err != nil {
		t.Fatal(err)
	}

	if got := c.Owned(testUSDC); got.Cmp(usdcBefore) != 0 {
		t.Fatalf("USDC owned changed %s -> %s across a trade", usdcBefore, got)
	}
	if got := c.Owned(testLUX); got.Cmp(luxBefore) != 0 {
		t.Fatalf("LUX owned changed %s -> %s across a trade", luxBefore, got)
	}
	if got := avail(t, c, bob, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("bob holds %s USDC, want 100", got)
	}
	if got := avail(t, c, alice, testLUX); got.Cmp(n(5)) != 0 {
		t.Fatalf("alice holds %s LUX, want 5", got)
	}
	if held(t, c, alice, testUSDC).Sign() != 0 || held(t, c, bob, testLUX).Sign() != 0 {
		t.Fatal("reserved balance survived the trade")
	}
}

// The full round trip: money in, traded repeatedly, money out. Conservation is
// checked against the RAIL totals at each stage, not against an internal number that
// would agree with itself by construction.
func TestRoundTripConserves(t *testing.T) {
	c := NewCustody(dChain)
	imported, exported := new(big.Int), new(big.Int)

	credit := func(id byte, who common.Address, amt int64) {
		t.Helper()
		if err := c.Import(claim(id, who, testUSDC, amt)); err != nil {
			t.Fatal(err)
		}
		imported.Add(imported, n(amt))
		if err := c.Conserved(testUSDC, imported, exported); err != nil {
			t.Fatal(err)
		}
	}
	credit(1, alice, 1000)
	credit(2, bob, 400)

	// Trade back and forth; conservation must hold at every step.
	for i := 0; i < 20; i++ {
		from, to := alice, bob
		if i%2 == 1 {
			from, to = bob, alice
		}
		if err := c.Reserve(from, testUSDC, n(10)); err != nil {
			t.Fatal(err)
		}
		if err := c.Unreserve(from, testUSDC, n(4)); err != nil {
			t.Fatal(err)
		}
		// Move the still-reserved 6 to the counterparty as a one-sided settlement:
		// reserve on both sides, then swap equal notional.
		if err := c.Reserve(to, testUSDC, n(6)); err != nil {
			t.Fatal(err)
		}
		if err := c.Trade(from, testUSDC, n(6), to, testUSDC, n(6)); err != nil {
			t.Fatal(err)
		}
		if err := c.Conserved(testUSDC, imported, exported); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if err := c.NoNegative(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}

	// Take some out.
	if _, err := c.Export(alice, testUSDC, n(250), cChain, ids.ID{0xE1}); err != nil {
		t.Fatal(err)
	}
	exported.Add(exported, n(250))
	if err := c.Conserved(testUSDC, imported, exported); err != nil {
		t.Fatal(err)
	}
	if want := n(1150); c.Owned(testUSDC).Cmp(want) != 0 {
		t.Fatalf("D owns %s, want %s", c.Owned(testUSDC), want)
	}
}

// Random operations, asserting conservation and non-negativity after every one. The
// point is not the happy path; it is that no SEQUENCE reaches an unsound state.
func TestCustodyConservesUnderRandomOperations(t *testing.T) {
	c := NewCustody(dChain)
	rng := rand.New(rand.NewSource(3))
	imported, exported := new(big.Int), new(big.Int)
	who := []common.Address{alice, bob, {0xC3}}
	claimSeq := 0
	seqID := func(tag byte) ids.ID {
		claimSeq++
		return ids.ID{tag, byte(claimSeq), byte(claimSeq >> 8)}
	}

	for step := 0; step < 600; step++ {
		actor := who[rng.Intn(len(who))]
		amt := n(int64(rng.Intn(50) + 1))
		switch rng.Intn(4) {
		case 0:
			cl := claim(0, actor, testUSDC, amt.Int64())
			cl.ClaimID = seqID(0xC0)
			if err := c.Import(cl); err != nil {
				t.Fatalf("step %d import: %v", step, err)
			}
			imported.Add(imported, amt)
		case 1:
			if err := c.Reserve(actor, testUSDC, amt); err != nil && !errors.Is(err, ErrNoBalance) {
				t.Fatalf("step %d reserve: %v", step, err)
			}
		case 2:
			if err := c.Unreserve(actor, testUSDC, amt); err != nil && !errors.Is(err, ErrReserved) {
				t.Fatalf("step %d unreserve: %v", step, err)
			}
		case 3:
			_, err := c.Export(actor, testUSDC, amt, cChain, seqID(0xE0))
			if err == nil {
				exported.Add(exported, amt)
			} else if !errors.Is(err, ErrNoBalance) {
				t.Fatalf("step %d export: %v", step, err)
			}
		}
		if err := c.Conserved(testUSDC, imported, exported); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if err := c.NoNegative(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	t.Logf("imported %s, exported %s, D owns %s", imported, exported, c.Owned(testUSDC))
}

// THE FINDING WORTH THE MOST. An import bundled with an order must credit
// UNCONDITIONALLY and place the order conditionally — never the reverse.
//
// A deadline can expire in transit: C accepts the export at t, D imports at t+d, and
// an order whose deadline fell in between is dead on arrival through nobody's fault.
// If the bundle were atomic in the strong sense — order fails, whole thing fails —
// the funds would be stuck in the in-flight state with their only consuming
// transaction permanently invalid. The trader would have to be rescued.
//
// So the rule is: the money always lands. A failed order leaves the balance sitting
// available on D, which is a state the trader can act on.
func TestFailedOrderStillLandsTheMoney(t *testing.T) {
	c := NewCustody(dChain)
	cl := claim(1, alice, testUSDC, 100)

	// The bundled order has already expired by the time D imports it.
	o := testOrder()
	o.Swapper = alice
	o.Deadline = 500
	ctx := okContext()
	ctx.Signer = recoverAs{addr: alice}
	ctx.BlockTime = 900 // D is past the deadline

	// Import first, unconditionally.
	if err := c.Import(cl); err != nil {
		t.Fatalf("import must not depend on the order: %v", err)
	}
	// Then attempt the order; it legitimately fails.
	if _, err := VerifyOrder(orderWitness(o), ctx); !errors.Is(err, ErrOrderExpired) {
		t.Fatalf("expected the order to be expired, got %v", err)
	}
	// The money is on D and spendable. Nothing is stranded, nobody needs rescuing.
	if got := avail(t, c, alice, testUSDC); got.Cmp(n(100)) != 0 {
		t.Fatalf("funds stranded: alice has %s available, want 100", got)
	}
	if _, err := c.Export(alice, testUSDC, n(100), cChain, ids.ID{0x01}); err != nil {
		t.Fatalf("trader cannot recover their own funds: %v", err)
	}
}

func TestClaimValidation(t *testing.T) {
	base := claim(1, alice, testUSDC, 100)
	cases := map[string]func(*Claim){
		"no id":          func(c *Claim) { c.ClaimID = ids.Empty },
		"no source":      func(c *Claim) { c.Source = ids.Empty },
		"no dest":        func(c *Claim) { c.Dest = ids.Empty },
		"same chain":     func(c *Claim) { c.Dest = c.Source },
		"no beneficiary": func(c *Claim) { c.Beneficiary = common.Address{} },
		"no asset":       func(c *Claim) { c.Asset = ids.Empty },
		"zero amount":    func(c *Claim) { c.Amount = [32]byte{} },
	}
	for name, mutate := range cases {
		cl := base
		mutate(&cl)
		if err := cl.Validate(); err == nil {
			t.Fatalf("%s: Validate accepted it", name)
		}
	}
}

func TestLocationNames(t *testing.T) {
	for l, want := range map[Location]string{OnC: "C", CToD: "C->D", OnD: "D", DToC: "D->C"} {
		if got := l.String(); got != want {
			t.Fatalf("got %q, want %q", got, want)
		}
	}
}
