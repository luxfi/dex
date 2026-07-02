// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// redteam_selfcross_interleave_test.go — RED TEAM. Attacks the recently-fixed
// self-trade-prevented maker removal (commit 16c2b00) for a DOUBLE-RELEASE of the
// per-order reserve: the self-cross path releases a maker's reserve (locked ->
// available) via releaseOrderReserve, and a TxCancel of the SAME order id releases
// it via the same primitive. If both could fire for one order, the owner would
// gain its reserve TWICE — a mint. This interleaves self-cross + external cross +
// explicit cancel + withdraw, within one block and across blocks, in both orders,
// and asserts conservation (no mint) after every step.
package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// TestRedteam_SelfCrossThenCancel_NoDoubleRelease drives the exact double-release
// shape: rest a SELL (locks base), self-cross it away (release #1), then explicitly
// cancel the now-dead id (attempted release #2). A double release would push the
// owner's base ABOVE its deposit.
func TestRedteam_SelfCrossThenCancel_NoDoubleRelease(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const alice = "alice"
	pool := [32]byte{0x5e, 0x1f, 0x99}
	accts := []string{alice}

	addBlock(t, vm,
		depositTx(t, alice, assetLUX, 100),
		depositTx(t, alice, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	assertConserved2(t, vm, "fund", accts, assetLUX, assetLUSD, 100, 1000)

	// Alice rests SELL 20 LUX @ 5 (locks 20 base).
	sBlk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 20.0, alice))
	sID := blockDeterministicID(sBlk.height, 0)
	assertBalance(t, vm, alice, assetLUX, 80, 20)

	// Alice submits BUY 20 @ 6 — self-crosses her own SELL. The self-maker is
	// cancelled: 20 base released locked->available (RELEASE #1). The BUY locks
	// ceil(20*6)=120 quote, finds only its own sell => no fill => full refund.
	scBlk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 6.0, 20.0, alice))
	if n := len(outcomeForTx(scBlk, scBlk.txs[0]).fills); n != 0 {
		t.Fatalf("self-cross produced %d fills (must be 0)", n)
	}
	assertBalance(t, vm, alice, assetLUX, 100, 0)    // base fully released
	assertBalance(t, vm, alice, assetLUSD, 1000, 0)  // quote lock fully refunded
	assertConserved2(t, vm, "after self-cross", accts, assetLUX, assetLUSD, 100, 1000)

	// ATTACK: explicitly cancel the now-dead order id (attempted RELEASE #2). If the
	// reserve were released again, alice base would jump to 120 (mint).
	addBlock(t, vm, cancelPoolTx(t, pool, sID, alice))
	assertBalance(t, vm, alice, assetLUX, 100, 0)
	assertConserved2(t, vm, "after double-cancel attempt", accts, assetLUX, assetLUSD, 100, 1000)

	// Withdraw everything: realized must be exactly the deposit, never more.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUX, 1_000_000))
	if got := withdrawRealizedOf(wo, TxWithdraw); got != 100 {
		t.Fatalf("MINT via double-release: withdrew %d base of a 100 deposit", got)
	}
}

// TestRedteam_SelfCrossAndCancelSameBlock exercises the double-release across BOTH
// intra-block orderings: (a) cancel BEFORE the self-crossing submit, (b) submit
// BEFORE the cancel — in a single block each. Conservation must hold regardless of
// the order the two reserve-touching txs execute in.
func TestRedteam_SelfCrossAndCancelSameBlock(t *testing.T) {
	run := func(t *testing.T, cancelFirst bool) {
		vm, _ := newTestVM(t, memdb.New())
		defer vm.Shutdown(context.Background())
		const alice = "alice"
		pool := [32]byte{0x5e, 0x1f, 0xab}
		accts := []string{alice}

		addBlock(t, vm,
			depositTx(t, alice, assetLUX, 100),
			depositTx(t, alice, assetLUSD, 1000),
			openMarketTx(t, pool, assetLUX, assetLUSD),
		)
		sBlk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 20.0, alice))
		sID := blockDeterministicID(sBlk.height, 0)
		assertBalance(t, vm, alice, assetLUX, 80, 20)

		submit := submitPoolTx(t, pool, zapwire.SideBuy, false, 6.0, 20.0, alice)
		cancel := cancelPoolTx(t, pool, sID, alice)
		if cancelFirst {
			addBlock(t, vm, cancel, submit)
		} else {
			addBlock(t, vm, submit, cancel)
		}
		ba, bl, _ := vm.Balance(wireUser(t, alice), assetLUX)
		qa, ql, _ := vm.Balance(wireUser(t, alice), assetLUSD)
		t.Logf("cancelFirst=%v post-interleave: base(avail=%d locked=%d) quote(avail=%d locked=%d)", cancelFirst, ba, bl, qa, ql)
		// Conservation (no mint) must hold either way.
		assertConserved2(t, vm, "same-block interleave", accts, assetLUX, assetLUSD, 100, 1000)
		_, woB := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUX, 1_000_000))
		_, woQ := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUSD, 1_000_000))
		gotB := withdrawRealizedOf(woB, TxWithdraw)
		gotQ := withdrawRealizedOf(woQ, TxWithdraw)
		t.Logf("cancelFirst=%v drained base=%d/100 quote=%d/1000", cancelFirst, gotB, gotQ)
		if gotB > 100 || gotQ > 1000 {
			t.Fatalf("MINT (cancelFirst=%v): drained base=%d quote=%d", cancelFirst, gotB, gotQ)
		}
		if gotB < 100 {
			t.Logf("STUCK FUNDS (cancelFirst=%v): only %d/100 base drainable — %d base frozen", cancelFirst, gotB, 100-gotB)
		}
	}
	t.Run("cancel_before_submit", func(t *testing.T) { run(t, true) })
	t.Run("submit_before_cancel", func(t *testing.T) { run(t, false) })
}

// TestRedteam_SelfCrossPlusExternalCross_Conservation is the full task shape:
// interleave a self-cross with a REAL external cross in one submit, plus a resting
// buy the same account cancels, then withdraw. If the self-cancel reserve or the
// external maker reserve is mis-accounted, conservation breaks or the withdraw
// mints.
func TestRedteam_SelfCrossPlusExternalCross_Conservation(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		alice = "alice"
		bob   = "bob"
	)
	pool := [32]byte{0x5e, 0x1f, 0xcc}
	accts := []string{alice, bob}

	// alice: base+quote. bob: base to sell.
	addBlock(t, vm,
		depositTx(t, alice, assetLUX, 100),
		depositTx(t, alice, assetLUSD, 1000),
		depositTx(t, bob, assetLUX, 100),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	assertConserved2(t, vm, "fund", accts, assetLUX, assetLUSD, 200, 1000)

	// bob rests SELL 10 @ 5 (external liquidity, locks 10 base).
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, bob))
	// alice rests SELL 20 @ 5 (her OWN liquidity she will self-cross, locks 20 base).
	aSell := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 20.0, alice))
	aSellID := blockDeterministicID(aSell.height, 0)
	// alice rests BUY 5 @ 4 (resting bid she'll later cancel, locks ceil(20)=20 quote).
	aBid := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, 4.0, 5.0, alice))
	aBidID := blockDeterministicID(aBid.height, 0)
	assertConserved2(t, vm, "resting", accts, assetLUX, assetLUSD, 200, 1000)

	// alice submits BUY 25 @ 6: crosses bob's 10 @ 5 (external fill) AND self-crosses
	// her own 20 @ 5 (self-cancel, no fill). Locks ceil(25*6)=150 quote up front.
	scBlk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 6.0, 25.0, alice))
	fills := outcomeForTx(scBlk, scBlk.txs[0]).fills
	t.Logf("submit fills=%d (expect 1 external fill against bob, 0 self)", len(fills))
	assertConserved2(t, vm, "after self+external cross", accts, assetLUX, assetLUSD, 200, 1000)

	// alice cancels her resting bid (release its 20 quote) and the dead self-crossed
	// sell id (must be a no-op — already released by the self-cross).
	addBlock(t, vm, cancelPoolTx(t, pool, aBidID, alice))
	addBlock(t, vm, cancelPoolTx(t, pool, aSellID, alice))
	assertConserved2(t, vm, "after cancels", accts, assetLUX, assetLUSD, 200, 1000)

	// Drain both accounts fully; total realized across both assets must never exceed
	// what was deposited (no mint anywhere in the interleave).
	_, woA1 := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUX, 1_000_000))
	_, woA2 := addBlockOutcomes(t, vm, withdrawTx(t, alice, assetLUSD, 1_000_000))
	_, woB1 := addBlockOutcomes(t, vm, withdrawTx(t, bob, assetLUX, 1_000_000))
	_, woB2 := addBlockOutcomes(t, vm, withdrawTx(t, bob, assetLUSD, 1_000_000))
	totBase := withdrawRealizedOf(woA1, TxWithdraw) + withdrawRealizedOf(woB1, TxWithdraw)
	totQuote := withdrawRealizedOf(woA2, TxWithdraw) + withdrawRealizedOf(woB2, TxWithdraw)
	// Any residual still LOCKED (e.g. the ceil/floor freeze) legitimately reduces the
	// drainable total; it can never EXCEED the deposit. > is a mint.
	if totBase > 200 {
		t.Fatalf("MINT: drained %d base of 200 deposited", totBase)
	}
	if totQuote > 1000 {
		t.Fatalf("MINT: drained %d quote of 1000 deposited", totQuote)
	}
	t.Logf("drained base=%d/200 quote=%d/1000 (<= deposit => no mint)", totBase, totQuote)
}
