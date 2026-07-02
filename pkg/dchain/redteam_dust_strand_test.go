// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// redteam_dust_strand_test.go — RED TEAM (adversarial). Attacks the ceil-lock /
// floor-charge asymmetry on a RESTING BUY order.
//
// A resting BUY locks ceil(size*price) quote (settle.go quoteUnitsCeil), but each
// fill charges floor(baseUnits*price) quote (settlement_units.go quoteUnitsFor).
// When the order is CANCELLED while still resting, releaseOrderReserve returns the
// full remaining reserve (ceil-side) — so the over-lock is refunded. But when the
// order FULLY FILLS, decrementMakerReserves only reduces the reserve by the floored
// consumed amount and NEVER releases the ceil-vs-floor residual: the order leaves
// the book (its order: row is deleted) with a non-zero orderasset:/orderuser: pair
// still recording locked quote that is now UNREACHABLE (a filled order cannot be
// cancelled: applyCancel sees GetOrder==nil => Canceled=0 => no release).
//
// Result: quote the maker deposited is PERMANENTLY FROZEN in locked:. It is NOT a
// mint (I = A + L still holds — the residual sits in L), which is exactly why the
// whole-keyspace conservation property (conservation_property_test.go) does NOT
// catch it. It IS a permanent loss/griefing vector, adversarially AMPLIFIED by
// fragmenting the maker's fill (each 1-unit fill strands ~frac(price) more).
package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// TestRedteam_FullFillStrandsOverlock_Minimal is the minimal repro: a single full
// fill of a resting BUY at a fractional price strands ceil(1*1.5)-floor(1*1.5)=1
// quote unit in the maker's locked, unrecoverable by cancel or withdraw.
func TestRedteam_FullFillStrandsOverlock_Minimal(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		maker = "maker"
		taker = "taker"
	)
	pool := [32]byte{0xd0, 0x57} // "dust"
	accts := []string{maker, taker}

	// Fund: maker holds quote (LUSD) to bid with, taker holds base (LUX) to sell.
	addBlock(t, vm,
		depositTx(t, maker, assetLUSD, 10),
		depositTx(t, taker, assetLUX, 5),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	assertConserved2(t, vm, "after fund", accts, assetLUX, assetLUSD, 5, 10)

	// Maker rests BUY 1 LUX @ 1.5 LUSD. Lock = ceil(1 * 1.5) = 2 quote.
	placeBlk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, 1.5, 1.0, maker))
	makerOrderID := blockDeterministicID(placeBlk.height, 0)
	assertBalance(t, vm, maker, assetLUSD, 8, 2) // avail 8, locked 2

	// Taker SELL 1 LUX @ 1.0 crosses the 1.5 bid — fills 1 unit at MAKER price 1.5.
	// quoteUnits = floor(1 * 1.5) = 1 : the maker is charged 1, not the 2 it locked.
	blk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideSell, false, 1.0, 1.0, taker))
	if n := len(outcomeForTx(blk, blk.txs[0]).fills); n != 1 {
		t.Fatalf("expected exactly 1 fill, got %d", n)
	}

	// ATTACK ASSERTION: the maker's order fully filled and left the book, yet 1
	// quote unit remains LOCKED — the ceil(2) reservation minus the floor(1) charge.
	// A correct full-fill would release this residual (as a cancel would); it does
	// not, so it is frozen.
	availQ, lockedQ, err := vm.Balance(wireUser(t, maker), assetLUSD)
	if err != nil {
		t.Fatalf("Balance(maker,LUSD): %v", err)
	}
	t.Logf("after full fill: maker LUSD avail=%d locked=%d (locked SHOULD be 0)", availQ, lockedQ)

	// NOT A MINT: at fill time (E=0), conservation I=A+L holds exactly. This is the
	// blind spot — the whole-keyspace property suite passes because the residual sits
	// in L, yet it is real user funds that can never be moved to A.
	assertConserved2(t, vm, "at fill time (no mint)", accts, assetLUX, assetLUSD, 5, 10)

	if lockedQ == 0 {
		t.Skip("no residual stranded — behavior may have been fixed; nothing to prove")
	}
	if lockedQ != 1 {
		t.Fatalf("unexpected stranded amount: got %d, analysis predicts 1", lockedQ)
	}

	// UNRECOVERABLE #1 — cancel the now-dead order id: it is gone from the book, so
	// applyCancel returns Canceled=0 and releaseOrderReserve is a no-op. The 1 stays.
	addBlock(t, vm, cancelPoolTx(t, pool, makerOrderID, maker))
	_, lockedAfterCancel, _ := vm.Balance(wireUser(t, maker), assetLUSD)
	if lockedAfterCancel != 1 {
		t.Fatalf("cancel unexpectedly changed stranded locked: %d (want still 1)", lockedAfterCancel)
	}

	// UNRECOVERABLE #2 — withdraw ALL quote: withdraw only touches AVAILABLE, so the
	// realized amount is the 8 available; the 1 locked is unreachable forever.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, maker, assetLUSD, 100))
	realized := withdrawRealizedOf(wo, TxWithdraw)
	_, lockedFinal, _ := vm.Balance(wireUser(t, maker), assetLUSD)
	t.Logf("withdraw-all realized=%d, maker LUSD locked FROZEN=%d", realized, lockedFinal)
	if lockedFinal != 1 {
		t.Fatalf("expected 1 quote permanently frozen after withdraw-all, got locked=%d", lockedFinal)
	}

	t.Errorf("RED FINDING: %d quote unit(s) permanently FROZEN in maker.locked after full fill "+
		"(unrecoverable by cancel or withdraw); conservation I=A+L masks it", lockedFinal)
}

// TestRedteam_FragmentationAmplifiesStranding shows an ADVERSARY (the taker) can
// scale the frozen amount by fragmenting the maker's fill: N single-unit sells
// against a BUY N @ 1.5 strand ceil(1.5N)-N = ceil(0.5N) quote in the maker's
// locked. Here N=10 => 15 locked, 10 charged, 5 frozen (33% of the reservation).
func TestRedteam_FragmentationAmplifiesStranding(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const (
		maker = "maker"
		taker = "taker"
	)
	pool := [32]byte{0xd0, 0x58} // "dust2"
	accts := []string{maker, taker}

	addBlock(t, vm,
		depositTx(t, maker, assetLUSD, 20),
		depositTx(t, taker, assetLUX, 20),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// Maker rests BUY 10 LUX @ 1.5 : lock = ceil(10*1.5) = 15 quote, avail 5.
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, 1.5, 10.0, maker))
	assertBalance(t, vm, maker, assetLUSD, 5, 15)

	// Adversarial taker fragments the fill into 10 single-unit sells. Each charges
	// floor(1*1.5)=1 quote; the maker's 15 reservation is only ever debited by 10.
	const N = 10
	for i := 0; i < N; i++ {
		addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideSell, false, 1.0, 1.0, taker))
	}

	availQ, lockedQ, _ := vm.Balance(wireUser(t, maker), assetLUSD)
	t.Logf("after %d fragmented fills: maker LUSD avail=%d locked=%d (locked SHOULD be 0)", N, availQ, lockedQ)

	// NOT A MINT: conservation I=A+L holds at fill time (before any withdraw).
	assertConserved2(t, vm, "at fill time (no mint)", accts, assetLUX, assetLUSD, 20, 20)

	// Maker bought all 10 LUX?
	availBase, _, _ := vm.Balance(wireUser(t, maker), assetLUX)
	if availBase != 10 {
		t.Fatalf("maker did not fully fill: got %d base of 10", availBase)
	}

	// The frozen residual: ceil(15) - 10 charged = 5.
	if lockedQ == 0 {
		t.Skip("no residual stranded — behavior may have been fixed")
	}
	if lockedQ != 5 {
		t.Fatalf("unexpected frozen amount: got %d, analysis predicts 5 (ceil(1.5*10)-10)", lockedQ)
	}

	// Withdraw everything: only the 5 available comes out; 5 stays frozen forever.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, maker, assetLUSD, 1000))
	realized := withdrawRealizedOf(wo, TxWithdraw)
	_, lockedFinal, _ := vm.Balance(wireUser(t, maker), assetLUSD)

	t.Errorf("RED FINDING (amplified): taker fragmentation FROZE %d/%d quote (%.0f%% of the maker's "+
		"reservation) after full fill; withdraw realized only %d; I=A+L masks the loss",
		lockedFinal, 15, float64(lockedFinal)/15*100, realized)
}
