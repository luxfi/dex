// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// byzantine_matcher_test.go proves the D-Chain CLOB matcher — the SINGLE
// source-of-truth that sequences orders price-time-priority and produces fills —
// upholds its safety invariants against an adversary, AND that a byzantine
// PROPOSER can never make honest validators settle a fabricated matching.
//
// THREAT MODEL. The matcher is authoritative: every validator re-executes it
// deterministically at Verify (block.go: `result.root != b.execRoot` => reject),
// so the block bytes carry the ORDER stream + the proposer's claimed execution
// root, never the fills. A byzantine actor therefore has exactly two levers:
//
//	(1) craft an adversarial ORDER stream (self-cross, worse-price, oversized,
//	    cancelled/withdrawn liquidity) and hope the matcher emits an invalid fill;
//	(2) build an honest order stream but CLAIM a fabricated execution root (assert
//	    self-cross proceeds / over-fill / phantom fills the honest matcher never
//	    produced).
//
// This file closes both. (1) is covered by the four matcher-invariant tests:
// the deterministic matcher every validator runs never self-trades, never fills a
// worse price while a better one rests, never over-fills beyond resting liquidity
// or order quantity, and never fills a cancelled/withdrawn order. (2) is the
// forged-root keystone: ANY block whose claimed execRoot diverges from the honest
// re-derivation is rejected by every validator, and the last-accepted state is
// unchanged — so whatever invalid matching a byzantine proposer wishes to settle,
// the honest quorum refuses it.
//
// All tests are deterministic (fixed accounts, integer prices/sizes, block-derived
// ids/timestamps — no wall-clock, no PRNG) and assert GLOBAL VALUE CONSERVATION
// (Σ available+locked per asset == Σ deposited) after every adversarial step.

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// totalHeld returns a user's controlled total for an asset: available + locked.
// Conservation is a statement about the controlled total (a place moves
// available->locked without changing it), so the byzantine-invariant assertions
// key on this, not on available alone.
func totalHeld(t *testing.T, vm *VM, user string, asset [32]byte) uint64 {
	t.Helper()
	avail, locked, err := vm.Balance(wireUser(t, user), asset)
	if err != nil {
		t.Fatalf("Balance(%s): %v", user, err)
	}
	return avail + locked
}

// assertConserved2 asserts the two-asset book conserves value across a set of
// accounts: for each asset the sum of every account's controlled total equals the
// amount deposited. This is the matcher-level analogue of the ledger property
// I = A + L (no withdraws here, so E = 0).
func assertConserved2(t *testing.T, vm *VM, where string, accts []string, baseAsset, quoteAsset [32]byte, wantBase, wantQuote uint64) {
	t.Helper()
	var gotBase, gotQuote uint64
	for _, a := range accts {
		gotBase += totalHeld(t, vm, a, baseAsset)
		gotQuote += totalHeld(t, vm, a, quoteAsset)
	}
	if gotBase != wantBase {
		t.Fatalf("%s: base conservation VIOLATED: Σheld=%d want deposited=%d (matcher minted/burned base)", where, gotBase, wantBase)
	}
	if gotQuote != wantQuote {
		t.Fatalf("%s: quote conservation VIOLATED: Σheld=%d want deposited=%d (matcher minted/burned quote)", where, gotQuote, wantQuote)
	}
}

// TestByzantineMatcher_SelfCrossNeverTrades is threat #1(b): a single account
// rests a maker and then submits a taker that crosses ITS OWN resting order. A
// correct matcher must NOT trade an account against itself — a self-trade is a
// wash that could desync the two settlement lanes (the seed-4242 regression) and
// is never a real economic event. The invariant: the crossing submit produces
// ZERO fills against the self-maker, and the account's controlled totals are
// exactly conserved (it neither gains nor loses from the wash).
func TestByzantineMatcher_SelfCrossNeverTrades(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const alice = "alice"
	pool := [32]byte{0x5e, 0x1f, 0xc0} // "self-cross"

	// Fund alice on BOTH sides so she can rest a sell (locks base) and submit a
	// crossing buy (locks quote).
	addBlock(t, vm,
		depositTx(t, alice, assetLUX, 100),
		depositTx(t, alice, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	assertConserved2(t, vm, "after fund", []string{alice}, assetLUX, assetLUSD, 100, 1000)

	// alice rests a SELL 10 LUX @ 5 (locks 10 LUX).
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, alice))
	assertBalance(t, vm, alice, assetLUX, 90, 10)

	// alice submits a BUY 10 LUX @ limit 5 — this WOULD cross her own resting sell.
	submit := submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, alice)
	blk := addBlock(t, vm, submit)

	oc := outcomeForTx(blk, submit)
	if n := len(oc.fills); n != 0 {
		t.Fatalf("SELF-TRADE EXECUTED: crossing submit produced %d fills against the account's own resting order "+
			"(a correct matcher must skip the self-maker leg; a self-trade can desync the settlement lanes)", n)
	}

	// CONSERVATION: alice can neither mint nor burn by trading with herself. Her
	// controlled totals are exactly what she deposited, no matter how the resting
	// orders are arranged.
	assertConserved2(t, vm, "after self-cross", []string{alice}, assetLUX, assetLUSD, 100, 1000)
}

// TestByzantineMatcher_SelfCrossCancelsRestingMakerConsistently is the regression
// pin for the self-cross desync fix. Previously the marketable matcher removed a
// self-crossed maker from the IN-MEMORY book only, leaving its durable
// order:/reserve/orderuser: rows — so a later cross tripped "resting order missing
// full settlement identity (orderuser)" / "insufficient locked balance" (seed 42,
// ~step 153). The fix routes the self-cross removal through the SAME cancel-persist
// path a TxCancel uses. This test drives the exact wedge shape — partial fill, then
// self-cross of the remainder, then an external cross — and asserts:
//   - the self-cross CANCELS the resting maker: its reserve returns locked->available
//     (conservation-exact), the order leaves the book, and NO fill is produced;
//   - the later external cross runs cleanly (no refusal), finding no phantom
//     liquidity from the cancelled maker;
//   - value is conserved at every step.
func TestByzantineMatcher_SelfCrossCancelsRestingMakerConsistently(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		alice = "alice"
		bob   = "bob"
	)
	pool := [32]byte{0x5e, 0x1f, 0xed} // "self-fixed"
	accts := []string{alice, bob}

	addBlock(t, vm,
		depositTx(t, alice, assetLUX, 100),
		depositTx(t, bob, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// alice rests SELL 20 LUX @ 5 (locks 20 base).
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 20.0, alice))
	assertBalance(t, vm, alice, assetLUX, 80, 20)

	// bob buys 12 @ 5 — PARTIAL fill: alice's order remaining 8, locked 8.
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 12.0, bob))
	assertBalance(t, vm, alice, assetLUX, 80, 8)  // 12 base sold to bob; 8 still locked
	assertBalance(t, vm, alice, assetLUSD, 60, 0) // received 12*5 = 60 quote

	// alice self-crosses her own remaining ask: BUY 8 @ 6. The fix CANCELS her
	// resting sell — reserve released (8 locked -> available), no fill.
	scSubmit := submitPoolTx(t, pool, zapwire.SideBuy, false, 6.0, 8.0, alice)
	scBlk := addBlock(t, vm, scSubmit)
	if n := len(outcomeForTx(scBlk, scSubmit).fills); n != 0 {
		t.Fatalf("SELF-TRADE EXECUTED: %d fills on a self-cross", n)
	}
	// KEY FIX ASSERTION: the self-maker's 8 locked base returned to available — the
	// order was cancelled consistently in durable state, not stranded.
	assertBalance(t, vm, alice, assetLUX, 88, 0)
	assertConserved2(t, vm, "after self-cross cancel", accts, assetLUX, assetLUSD, 100, 1000)

	// bob crosses again: BUY 8 @ 6. The cancelled maker is GONE from the book (not a
	// phantom that wedges the settle) — this cross runs cleanly with zero fill.
	bobSubmit := submitPoolTx(t, pool, zapwire.SideBuy, false, 6.0, 8.0, bob)
	bobBlk := addBlock(t, vm, bobSubmit)
	if n := len(outcomeForTx(bobBlk, bobSubmit).fills); n != 0 {
		t.Fatalf("PHANTOM FILL: %d fills against a self-cancelled maker (the old wedge/desync bug)", n)
	}
	assertConserved2(t, vm, "after external cross of cancelled book", accts, assetLUX, assetLUSD, 100, 1000)

	// alice's freed collateral is fully withdrawable (the cancel truly released it).
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUX, 88))
	if got := withdrawRealizedOf(wo, TxWithdraw); got != 88 {
		t.Fatalf("self-cancelled maker's freed collateral not fully withdrawable: got %d of 88", got)
	}
}

// TestByzantineMatcher_PriceTimePriority_BestPriceFirst is threat #1(c): with two
// resting asks at DIFFERENT prices, a taker must fill the BEST (lowest) ask first
// and MUST NOT touch the worse ask while better liquidity rests. A byzantine
// matcher that filled the worse price would overcharge the taker and mispay a
// maker — a price-priority violation. The invariant: the fill lands at the best
// price and the worse maker is untouched.
func TestByzantineMatcher_PriceTimePriority_BestPriceFirst(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		makerBest  = "maker-best"  // asks @ 5 (the better price for a buyer)
		makerWorse = "maker-worse" // asks @ 10 (worse for a buyer)
		taker      = "taker"
	)
	pool := [32]byte{0x9a, 0x1c, 0xe0} // "price"
	accts := []string{makerBest, makerWorse, taker}

	// Fund: each maker holds base to sell; the taker holds quote to buy.
	addBlock(t, vm,
		depositTx(t, makerBest, assetLUX, 100),
		depositTx(t, makerWorse, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// Rest two asks: 10 @ 5 (best) and 10 @ 10 (worse).
	addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, makerBest),
		placePoolTx(t, pool, zapwire.SideSell, 10.0, 10.0, makerWorse),
	)
	assertBalance(t, vm, makerBest, assetLUX, 90, 10)
	assertBalance(t, vm, makerWorse, assetLUX, 90, 10)

	// Taker BUYS only 5 LUX at a limit (10) that makes BOTH asks eligible. Price
	// priority must fill the 5-priced ask, never the 10-priced one.
	submit := submitPoolTx(t, pool, zapwire.SideBuy, false, 10.0, 5.0, taker)
	blk := addBlock(t, vm, submit)
	oc := outcomeForTx(blk, submit)

	if len(oc.fills) == 0 {
		t.Fatal("expected the taker to fill against the best resting ask, got zero fills")
	}
	for i, f := range oc.fills {
		if f.Price != 5*uint64(zapwire.PriceScale) { // price 5.0 in ×1e8 fixed-point
			t.Fatalf("PRICE-PRIORITY VIOLATED: fill[%d] executed @ %d while a better resting ask @ 5e8 existed "+
				"(matcher filled a worse price first)", i, f.Price)
		}
	}

	// The WORSE maker must be entirely untouched: 10 LUX still locked, zero quote
	// received. Filling it while the better ask had liquidity is the violation.
	assertBalance(t, vm, makerWorse, assetLUX, 90, 10)
	if q := totalHeld(t, vm, makerWorse, assetLUSD); q != 0 {
		t.Fatalf("PRICE-PRIORITY VIOLATED: worse-priced maker received %d quote — it was filled while a better ask rested", q)
	}
	// The best maker sold exactly 5 (5 of its 10 remain locked) and received 25 quote.
	assertBalance(t, vm, makerBest, assetLUX, 90, 5)
	assertBalance(t, vm, makerBest, assetLUSD, 25, 0)

	assertConserved2(t, vm, "after best-price fill", accts, assetLUX, assetLUSD, 200, 1000)
}

// TestByzantineMatcher_OverFillClampedToRestingLiquidity is threat #1(d): a taker
// larger than the resting liquidity must fill ONLY up to what rests — never more
// than the resting size and never more than the order quantity — with the
// remainder refunded. A byzantine matcher that "over-filled" would fabricate base
// out of nothing. The invariant: Σ fill size <= min(orderQty, restingLiquidity),
// and value is conserved.
func TestByzantineMatcher_OverFillClampedToRestingLiquidity(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		maker = "maker"
		taker = "taker"
	)
	pool := [32]byte{0x0f, 0x11, 0x00} // "overfill"
	accts := []string{maker, taker}

	const restingLiquidity = 4.0 // maker rests only 4 LUX
	const orderQty = 10.0        // taker wants 10 LUX

	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, restingLiquidity, maker))
	assertBalance(t, vm, maker, assetLUX, 96, 4)

	// Taker BUYS 10 @ limit 5 — 2.5x the resting liquidity.
	submit := submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, orderQty, taker)
	blk := addBlock(t, vm, submit)
	oc := outcomeForTx(blk, submit)

	var filled float64
	for _, f := range oc.fills {
		filled += float64(f.Size) // wire fill size is exact base units
	}
	if filled > restingLiquidity {
		t.Fatalf("OVER-FILL: matcher filled %g > resting liquidity %g (fabricated base from nothing)", filled, restingLiquidity)
	}
	if filled > orderQty {
		t.Fatalf("OVER-FILL: matcher filled %g > order quantity %g", filled, orderQty)
	}
	if filled != restingLiquidity {
		t.Fatalf("expected the taker to consume all %g resting, filled %g", restingLiquidity, filled)
	}

	// Taker got exactly 4 LUX, spent 20 quote, refunded 30 (locked 50 - spent 20).
	assertBalance(t, vm, taker, assetLUX, 4, 0)
	assertBalance(t, vm, taker, assetLUSD, 980, 0)
	// Maker fully consumed: 0 locked, received 20 quote.
	assertBalance(t, vm, maker, assetLUX, 96, 0)
	assertBalance(t, vm, maker, assetLUSD, 20, 0)

	// Only the maker deposited base (100 LUX); only the taker deposited quote (1000).
	assertConserved2(t, vm, "after over-fill clamp", accts, assetLUX, assetLUSD, 100, 1000)
}

// TestByzantineMatcher_CancelledOrderNotFilled is threat #1(e): a resting order
// that was CANCELLED (its collateral returned to the owner) must never be filled
// by a later crossing submit — the liquidity no longer exists. A byzantine
// matcher that filled the gone order would move value the owner already reclaimed.
// The invariant: the post-cancel submit produces ZERO fills, the taker is fully
// refunded, and the cancelled maker's freed collateral is fully withdrawable.
func TestByzantineMatcher_CancelledOrderNotFilled(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		maker = "maker"
		taker = "taker"
	)
	pool := [32]byte{0xca, 0x9c, 0xe1} // "cancel"
	accts := []string{maker, taker}

	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// Maker rests a SELL 10 @ 5 (locks 10 LUX) — record its deterministic id.
	placeBlk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker))
	orderID := blockDeterministicID(placeBlk.height, 0)
	assertBalance(t, vm, maker, assetLUX, 90, 10)

	// Maker CANCELS it — 10 LUX unlocks back to available.
	addBlock(t, vm, cancelPoolTx(t, pool, orderID, maker))
	assertBalance(t, vm, maker, assetLUX, 100, 0)

	// Taker BUYS 10 @ limit 5 — the book is empty (order cancelled).
	submit := submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker)
	blk := addBlock(t, vm, submit)
	oc := outcomeForTx(blk, submit)

	if n := len(oc.fills); n != 0 {
		t.Fatalf("PHANTOM FILL: submit matched a CANCELLED order (%d fills) — the liquidity was already reclaimed", n)
	}
	// Taker received nothing; its locked quote is fully refunded (limit buy with no
	// fill either rests or refunds — either way its controlled total is intact).
	if b := totalHeld(t, vm, taker, assetLUX); b != 0 {
		t.Fatalf("PHANTOM FILL: taker received %d base against a cancelled order", b)
	}
	assertConserved2(t, vm, "after cancelled-order submit", accts, assetLUX, assetLUSD, 100, 1000)

	// The cancelled maker's freed collateral is fully withdrawable — proving the
	// cancel truly released it (it was not silently consumed by a phantom fill).
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, maker, assetLUX, 100))
	if got := withdrawRealizedOf(wo, TxWithdraw); got != 100 {
		t.Fatalf("cancelled maker could only withdraw %d of 100 freed LUX (collateral leaked into a phantom fill)", got)
	}
}

// TestByzantineProposer_ForgedExecRootRejected is the KEYSTONE for threat #1(a)
// and the generic "validators reject a byzantine matcher's fabricated fills". A
// byzantine proposer runs the honest order stream but CLAIMS a fabricated
// execution root — the on-chain commitment to the matching result (fills + ledger
// rows). Whatever invalid settlement it wants (self-cross proceeds, an over-fill,
// a phantom fill, a conservation-breaking mint), it must encode as a divergent
// execRoot. Every honest validator re-executes the matcher and rejects the block
// because its derived root differs from the claim — and the last-accepted state is
// UNCHANGED. This subsumes (a)-(e): no fabricated matching can be finalized.
func TestByzantineProposer_ForgedExecRootRejected(t *testing.T) {
	ctx := context.Background()

	// Corruption strategies a byzantine proposer might use to smuggle a fabricated
	// matching into the block header. Each must be rejected at Verify.
	corruptions := []struct {
		name    string
		corrupt func(raw []byte)
	}{
		{
			name: "flip-execroot-high-byte",
			// Claim a different execution root => claim a different matching outcome.
			corrupt: func(raw []byte) { raw[48] ^= 0xff },
		},
		{
			name: "flip-execroot-low-byte",
			corrupt: func(raw []byte) { raw[79] ^= 0x01 },
		},
		{
			name: "zero-execroot",
			// Claim the empty/zero root (a proposer asserting "nothing settled").
			corrupt: func(raw []byte) {
				for i := 48; i < 80; i++ {
					raw[i] = 0
				}
			},
		},
	}

	for _, c := range corruptions {
		t.Run(c.name, func(t *testing.T) {
			vm, _ := newTestVM(t, memdb.New())
			defer vm.Shutdown(ctx)

			const (
				maker = "maker"
				taker = "taker"
			)
			pool := [32]byte{0xf0, 0x69, 0xed} // "forged"

			// Fund + rest liquidity + accept an honest crossing block first, so the
			// chain has real state a forgery would try to corrupt.
			addBlock(t, vm,
				depositTx(t, maker, assetLUX, 100),
				depositTx(t, taker, assetLUSD, 1000),
				openMarketTx(t, pool, assetLUX, assetLUSD),
			)
			addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker))

			headBefore := vm.lastAcceptedID
			heightBefore := vm.lastAcceptedHeight

			// The proposer builds an honest crossing block (real matcher, real root).
			vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
			built, err := vm.BuildBlock(ctx)
			if err != nil {
				t.Fatalf("BuildBlock: %v", err)
			}

			// Byzantine: tamper the block header to claim a fabricated matching.
			raw := append([]byte(nil), built.Bytes()...)
			c.corrupt(raw)

			forged, err := vm.ParseBlock(ctx, raw)
			if err != nil {
				// A parse-level rejection is an equally valid refusal of the forgery.
				t.Logf("forged block rejected at ParseBlock: %v", err)
			} else if verr := forged.Verify(ctx); verr == nil {
				t.Fatalf("FORGED MATCHING ACCEPTED: a block claiming a fabricated execution root passed Verify — "+
					"a byzantine proposer could finalize an invalid matching (%s)", c.name)
			} else {
				t.Logf("forged block rejected at Verify: %v", verr)
			}

			// Honest state is UNCHANGED: the rejected forgery advanced nothing.
			if vm.lastAcceptedID != headBefore || vm.lastAcceptedHeight != heightBefore {
				t.Fatalf("last-accepted advanced on a rejected forgery: head %s->%s height %d->%d",
					headBefore, vm.lastAcceptedID, heightBefore, vm.lastAcceptedHeight)
			}

			// And the HONEST block (unmodified) still verifies + accepts cleanly — the
			// forgery did not poison the mempool or the matcher.
			reparsed, err := vm.ParseBlock(ctx, built.Bytes())
			if err != nil {
				t.Fatalf("ParseBlock honest: %v", err)
			}
			if err := reparsed.Verify(ctx); err != nil {
				t.Fatalf("honest block must still verify after a rejected forgery: %v", err)
			}
			if err := reparsed.Accept(ctx); err != nil {
				t.Fatalf("Accept honest: %v", err)
			}
			if vm.lastAcceptedHeight != heightBefore+1 {
				t.Fatalf("honest block did not advance the chain: height %d, want %d", vm.lastAcceptedHeight, heightBefore+1)
			}
		})
	}
}
