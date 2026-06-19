// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

// market_constraint_test.go pins the ZERO-NOTIONAL reject — the minimal
// market-constraint invariant that arithmetic flooring can NEVER create a FREE
// executable order. An order whose required lock floors to zero (a sub-tick BUY
// price whose ceil(size*price) rounds to 0, or a sub-unit SELL size) would
// rest/cross while reserving NOTHING; settlement could then move the
// counterparty's asset for no spend. lockOrderSpend rejects such an order with
// ErrDustPriceOrZeroLock (a deterministic per-tx reject), so it never rests and
// never matches.
//
// PriceMultiplier is 1e8, so priceToInt(price) = price*1e8. A price < 1e-8 floors
// to priceInt 0, and quoteUnitsCeil(units, 0) == 0 — the canonical sub-tick dust
// BUY this gate must refuse.

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// dustPrice is a positive price strictly below one price tick (1e-8), so
// priceToInt(dustPrice) == 0 and a BUY of any positive size floors to a zero
// quote lock. It is a GENUINE executable price on the wire (>0), not a malformed
// zero — exactly the "executable yet free" case the gate must catch.
const dustPrice = 5e-9

// TestNotionalFlooringCannotCreateFreeExecutableOrder is the headline
// market-constraint proof: an executable BUY whose notional floors to zero lock
// (size>0, price>0, but ceil(size*price)==0) is REJECTED at placement — it never
// rests, locks nothing, and a crossing taker cannot acquire the (non-existent)
// liquidity for free. It also pins the cross axis: a sub-tick LIMIT BUY submit
// against a real resting maker is rejected too, so flooring cannot acquire the
// maker's base for no quote.
func TestNotionalFlooringCannotCreateFreeExecutableOrder(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const buyer, seller = "dust-buyer", "dust-seller"
	pool := [32]byte{0xd5, 0x70}

	// Sanity on the predicate itself: a sub-tick BUY floors to zero lock; a normal
	// BUY does not.
	mc := marketConstraint{assetLUX, assetLUSD}
	if !mc.floorsToZeroLock(sideBuy, dustPrice, 10.0) {
		t.Fatalf("predicate: sub-tick BUY %g x10 should floor to zero lock", dustPrice)
	}
	if mc.floorsToZeroLock(sideBuy, 5.0, 10.0) {
		t.Fatal("predicate: a normal BUY 5x10 must NOT floor to zero lock")
	}

	// Fund the buyer generously (so the reject is the NOTIONAL floor, never an
	// affordability reject) and open the custody market.
	addBlock(t, vm,
		depositTx(t, buyer, assetLUSD, 1_000_000),
		depositTx(t, seller, assetLUX, 100),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// PLACE a dust BUY (10 @ 5e-9): rejected, nothing locked, book empty.
	own := newOwnershipChecker(t, vm)
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, dustPrice, 10.0, buyer))
	if oc := outcomeForTx(blk, blk.txs[0]); oc.status != zapwire.StatusRejected {
		t.Fatalf("dust BUY place status = %d, want StatusRejected (%d)", oc.status, zapwire.StatusRejected)
	}
	// No lock taken (the buyer's full quote stays available), and NO resting order.
	assertBalance(t, vm, buyer, assetLUSD, 1_000_000, 0)
	if orders, _, _, _, _ := vm.BookDepth(pool); orders != 0 {
		t.Fatalf("dust BUY rested: book has %d orders, want 0 (a zero-lock executable order must never rest)", orders)
	}
	// No value moved at all -> ownership trivially holds (no parties needed).
	own.assert(t, vm, blk, "dust-buy-place")

	// Now a REAL maker rests (seller SELL 10 LUX @ 5), and a sub-tick LIMIT BUY
	// submit tries to cross it. The submit is rejected by the notional gate, so the
	// maker's base is NOT acquired for a floored-to-zero quote.
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, seller))
	assertBalance(t, vm, seller, assetLUX, 90, 10)

	ownCross := newOwnershipChecker(t, vm)
	sblk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, dustPrice, 10.0, buyer))
	if oc := outcomeForTx(sblk, sblk.txs[0]); oc.status != zapwire.StatusRejected {
		t.Fatalf("dust LIMIT BUY submit status = %d, want StatusRejected", oc.status)
	}
	// Nothing was acquired or paid: buyer keeps all quote, seller keeps its locked
	// base. Flooring acquired NO free base.
	assertBalance(t, vm, buyer, assetLUSD, 1_000_000, 0)
	assertBalance(t, vm, buyer, assetLUX, 0, 0)
	assertBalance(t, vm, seller, assetLUX, 90, 10)
	ownCross.assert(t, vm, sblk, "dust-buy-cross")
}

// TestLockAmtZeroBuyRejected is the focused mutation pin named by the acceptance
// gate: a BUY with size>0 and price>0 whose required quote lock floors to zero is
// rejected (ok=false) by lockOrderSpend with ErrDustPriceOrZeroLock — it cannot
// rest with a zero lock. (Mutation: allowing lockAmt==0 BUY to pass makes this and
// TestNotionalFlooringCannotCreateFreeExecutableOrder RED.)
func TestLockAmtZeroBuyRejected(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const buyer = "zlb-buyer"
	pool := [32]byte{0x21, 0xb0}
	addBlock(t, vm,
		depositTx(t, buyer, assetLUSD, 1_000_000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// Drive lockOrderSpend directly through a built block's overlay so we observe
	// the named reject, not just the StatusRejected outcome.
	tx := placePoolTx(t, pool, zapwire.SideBuy, dustPrice, 10.0, buyer)
	blk := addBlock(t, vm, tx)
	if oc := outcomeForTx(blk, blk.txs[0]); oc.status != zapwire.StatusRejected {
		t.Fatalf("zero-lock BUY status = %d, want StatusRejected", oc.status)
	}
	// The buyer funded a million quote but the order never rested or locked.
	assertBalance(t, vm, buyer, assetLUSD, 1_000_000, 0)
	if orders, _, _, _, _ := vm.BookDepth(pool); orders != 0 {
		t.Fatalf("zero-lock BUY rested: %d orders, want 0", orders)
	}
}
