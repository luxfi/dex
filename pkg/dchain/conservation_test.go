// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// conservation_test.go is the value-conservation proof for the D-Chain CUSTODY
// ledger — the "money lives in the order book" model. It drives the REAL VM
// (BuildBlock -> Verify -> Accept) through the full lifecycle of a non-native
// asset: DEPOSIT into the D-Chain, OPEN a market, a maker PLACES resting
// liquidity (locking its base), a taker SUBMITS a crossing order (locking its
// quote), the cross SETTLES ENTIRELY inside the D-Chain (balances move between
// maker and taker in the ledger), and both parties WITHDRAW. The acceptance
// criterion is GLOBAL WEI-EXACT CONSERVATION: the sum of every asset deposited
// equals the sum of every asset withdrawn (plus whatever remains as
// available/locked) — nothing is minted, nothing is burned, and crucially the
// MAKER receives exactly what the TAKER paid (the leg the prior proxy-escrow
// model dropped).

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// asset handles for the test market (8-byte ledger handles).
const (
	assetLUX  uint64 = 0x4c5558_00000001 // "LUX" base
	assetLUSD uint64 = 0x4c555344_000001 // "LUSD" quote
)

func depositTx(t *testing.T, user string, asset, amount uint64) *Tx {
	t.Helper()
	tx, err := NewTx(TxDeposit, zapwire.EncodeDeposit(user, asset, amount))
	if err != nil {
		t.Fatalf("NewTx deposit: %v", err)
	}
	return tx
}

func withdrawTx(t *testing.T, user string, asset, amount uint64) *Tx {
	t.Helper()
	tx, err := NewTx(TxWithdraw, zapwire.EncodeWithdraw(user, asset, amount))
	if err != nil {
		t.Fatalf("NewTx withdraw: %v", err)
	}
	return tx
}

func openMarketTx(t *testing.T, pool [32]byte, base, quote uint64) *Tx {
	t.Helper()
	tx, err := NewTx(TxOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
	if err != nil {
		t.Fatalf("NewTx open-market: %v", err)
	}
	return tx
}

// addBlock pushes the given txs into the mempool and runs one
// BuildBlock->Verify->Accept, returning the accepted block. It exercises the
// validator path (re-parse + independent Verify) so the ledger transition is
// proven to re-derive identically.
func addBlock(t *testing.T, vm *VM, txs ...*Tx) *Block {
	t.Helper()
	for _, tx := range txs {
		vm.mempool.Add(tx)
	}
	return buildVerifyAccept(t, vm)
}

// addBlockOutcomes is like addBlock but snapshots the per-tx consensus outcomes
// (Verify sets them on the block; Accept clears them after resolving waiters), so
// a caller can inspect a withdraw's REALIZED amount. It runs the same
// Build->reparse->Verify->Accept path.
func addBlockOutcomes(t *testing.T, vm *VM, txs ...*Tx) []txOutcome {
	t.Helper()
	ctx := context.Background()
	for _, tx := range txs {
		vm.mempool.Add(tx)
	}
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	reparsed, err := vm.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock: %v", err)
	}
	if err := reparsed.Verify(ctx); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	// Snapshot outcomes after Verify (set) and before Accept (cleared).
	blk := reparsed.(*Block)
	snap := make([]txOutcome, len(blk.outcomes))
	copy(snap, blk.outcomes)
	if err := reparsed.Accept(ctx); err != nil {
		t.Fatalf("Accept: %v", err)
	}
	return snap
}

// TestCustodyConservationFullCycle is the headline proof: deposit -> open ->
// place -> cross -> settle -> withdraw for a NON-NATIVE asset pair, asserting
// wei-exact conservation and that the maker received the taker's payment.
//
// Scenario (integer units; price grid = whole LUSD per LUX):
//   - maker deposits 100 LUX (base), taker deposits 1000 LUSD (quote).
//   - market LUX/LUSD opened (base=LUX, quote=LUSD).
//   - maker PLACES a resting SELL of 10 LUX @ 5 LUSD  (locks 10 LUX).
//   - taker SUBMITS a marketable BUY of 10 LUX, limit 5 (locks 50 LUSD).
//   - cross fills 10 LUX @ 5 => 50 LUSD moves taker->maker, 10 LUX moves
//     maker->taker, ALL INSIDE THE LEDGER.
//   - both withdraw everything.
//
// Expected end balances BEFORE withdraw:
//   maker: 90 LUX available, 0 LUX locked, 50 LUSD available.
//   taker: 10 LUX available, 1000-50=950 LUSD available, 0 LUSD locked.
// Conservation: total LUX in (100) == maker(90)+taker(10); total LUSD in (1000)
// == taker(950)+maker(50). After withdraw, exported == deposited, exactly.
func TestCustodyConservationFullCycle(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const (
		maker = "maker-a"
		taker = "taker-x"
	)
	pool := [32]byte{0xde, 0xad, 0xbe, 0xef}

	// --- Block 1: deposits + open market. ---
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// After deposit: available credited, nothing locked.
	assertBalance(t, vm, maker, assetLUX, 100, 0)
	assertBalance(t, vm, taker, assetLUSD, 1000, 0)

	// --- Block 2: maker rests a SELL of 10 LUX @ 5 (locks 10 LUX). ---
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker))
	// Maker's 10 LUX moved available->locked; 90 LUX stays available.
	assertBalance(t, vm, maker, assetLUX, 90, 10)

	// --- Block 3: taker BUYS 10 LUX, limit 5 (locks 50 LUSD), crossing. ---
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))

	// THE MAKER LEG: maker received 50 LUSD (the taker's payment) into available,
	// and its 10 locked LUX left to the taker. This is exactly what the prior
	// proxy-escrow taker-only model dropped.
	assertBalance(t, vm, maker, assetLUX, 90, 0)   // 10 locked LUX spent to taker
	assertBalance(t, vm, maker, assetLUSD, 50, 0)  // received the taker's 50 LUSD
	// THE TAKER LEG: spent 50 LUSD (refunded 0 since limit*size == lock exactly),
	// received 10 LUX.
	assertBalance(t, vm, taker, assetLUX, 10, 0)   // received 10 LUX
	assertBalance(t, vm, taker, assetLUSD, 950, 0) // 1000 - 50 spent, lock fully spent

	// --- CONSERVATION across the trade (before withdraw) ---
	// Every unit deposited is still present as available somewhere; nothing minted.
	totalLUX := bal(t, vm, maker, assetLUX) + bal(t, vm, taker, assetLUX)
	totalLUSD := bal(t, vm, maker, assetLUSD) + bal(t, vm, taker, assetLUSD)
	if totalLUX != 100 {
		t.Fatalf("LUX conservation violated: total available %d, deposited 100", totalLUX)
	}
	if totalLUSD != 1000 {
		t.Fatalf("LUSD conservation violated: total available %d, deposited 1000", totalLUSD)
	}

	// --- Block 4: both parties WITHDRAW everything; the realized amounts (the
	// proxy export legs) must equal exactly the available balances. ---
	wo := addBlockOutcomes(t, vm,
		withdrawTx(t, maker, assetLUX, 90),
		withdrawTx(t, maker, assetLUSD, 50),
		withdrawTx(t, taker, assetLUX, 10),
		withdrawTx(t, taker, assetLUSD, 950),
	)
	// Withdraw outcomes carry the realized amount in orderID.
	var totalWithdrawn uint64
	for _, oc := range wo {
		if oc.typ == TxWithdraw {
			totalWithdrawn += oc.orderID
		}
	}
	const totalDeposited = 100 + 1000
	if totalWithdrawn != totalDeposited {
		t.Fatalf("CONSERVATION VIOLATED: withdrawn %d != deposited %d", totalWithdrawn, totalDeposited)
	}

	// Ledger is now empty (all withdrawn).
	assertBalance(t, vm, maker, assetLUX, 0, 0)
	assertBalance(t, vm, maker, assetLUSD, 0, 0)
	assertBalance(t, vm, taker, assetLUX, 0, 0)
	assertBalance(t, vm, taker, assetLUSD, 0, 0)
}

// TestCustodySameBlockPlaceThenCross proves the custody ledger is correct when a
// maker PLACE and a crossing taker SUBMIT land in the SAME block (a real
// consensus scenario): the place locks first, then the submit locks + crosses the
// just-rested maker, and balances settle to the unit. The in-block ordering is
// the tx order; the lock/match/settle for each tx runs against the same overlay,
// so the submit sees the maker rested earlier in the block.
func TestCustodySameBlockPlaceThenCross(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const (
		maker = "sb-maker"
		taker = "sb-taker"
	)
	pool := [32]byte{0x0a, 0x0b}

	// Fund + open in block 1.
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// Block 2: place (maker SELL 10@5) AND cross (taker BUY 10@5) in ONE block.
	addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker),
		submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker),
	)

	// Same outcome as the two-block case: maker 90 LUX + 50 LUSD, taker 10 LUX + 950 LUSD.
	assertBalance(t, vm, maker, assetLUX, 90, 0)
	assertBalance(t, vm, maker, assetLUSD, 50, 0)
	assertBalance(t, vm, taker, assetLUX, 10, 0)
	assertBalance(t, vm, taker, assetLUSD, 950, 0)

	if got := bal(t, vm, maker, assetLUX) + bal(t, vm, taker, assetLUX); got != 100 {
		t.Fatalf("same-block LUX conservation: %d want 100", got)
	}
	if got := bal(t, vm, maker, assetLUSD) + bal(t, vm, taker, assetLUSD); got != 1000 {
		t.Fatalf("same-block LUSD conservation: %d want 1000", got)
	}
}

// TestCustodyRejectsUnfundedOrder proves the matcher debits only AVAILABLE
// balance: an order the account cannot fund is REJECTED (no lock, no rest, no
// match) and leaves the ledger untouched.
func TestCustodyRejectsUnfundedOrder(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const maker = "broke-maker"
	pool := [32]byte{0x01, 0x02}

	addBlock(t, vm, openMarketTx(t, pool, assetLUX, assetLUSD))
	// No deposit. A sell of 10 LUX needs 10 LUX available; the account has 0.
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker))

	// Nothing locked, nothing available, and NO resting order (the book is empty).
	assertBalance(t, vm, maker, assetLUX, 0, 0)
	orders, _, _, _, _ := vm.BookDepth(pool)
	if orders != 0 {
		t.Fatalf("unfunded order rested: book has %d orders, want 0", orders)
	}
}

// TestCustodyCancelRefundsLock proves a cancelled order's locked reserve returns
// to available, fully — no value stranded in locked.
func TestCustodyCancelRefundsLock(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const maker = "maker-c"
	pool := [32]byte{0x03, 0x04}

	addBlock(t, vm, depositTx(t, maker, assetLUX, 100), openMarketTx(t, pool, assetLUX, assetLUSD))
	// Rest a sell of 30 LUX @ 5 -> locks 30 LUX.
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 30.0, maker))
	assertBalance(t, vm, maker, assetLUX, 70, 30)

	// Recover the resting order id (the block placed it at height/txIndex 0).
	orderID := blockDeterministicID(blk.height, 0)
	addBlock(t, vm, cancelPoolTx(t, pool, orderID))

	// The 30 locked LUX returned to available; nothing stranded.
	assertBalance(t, vm, maker, assetLUX, 100, 0)
}

// TestCustodyPartialFillConservesAndRefundsRemainder crosses a taker against a
// SMALLER maker: the taker fills part, the maker is fully consumed, and the
// taker's unfilled lock remainder is refunded. Conservation holds and the taker
// keeps no quote it did not spend.
func TestCustodyPartialFillConservesAndRefundsRemainder(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const (
		maker = "maker-p"
		taker = "taker-p"
	)
	pool := [32]byte{0x05, 0x06}

	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	// Maker rests only 4 LUX @ 5 (locks 20 LUX).
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 4.0, maker))
	// Taker wants 10 LUX, limit 5 (locks 50 LUSD) but only 4 are available.
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))

	// Fill = 4 LUX @ 5 = 20 LUSD. Taker spent 20 LUSD, refunded 30 of the 50 lock.
	assertBalance(t, vm, maker, assetLUX, 96, 0)    // 4 locked LUX spent to taker, 96 still available
	assertBalance(t, vm, maker, assetLUSD, 20, 0)   // received 20 LUSD
	assertBalance(t, vm, taker, assetLUX, 4, 0)     // received 4 LUX
	assertBalance(t, vm, taker, assetLUSD, 980, 0)  // 1000 - 20 spent (30 refunded back to available)

	totalLUX := bal(t, vm, maker, assetLUX) + bal(t, vm, taker, assetLUX)
	totalLUSD := bal(t, vm, maker, assetLUSD) + bal(t, vm, taker, assetLUSD)
	if totalLUX != 100 || totalLUSD != 1000 {
		t.Fatalf("partial-fill conservation violated: LUX %d (want 100), LUSD %d (want 1000)", totalLUX, totalLUSD)
	}
}

// TestCustodyWithdrawClampsToAvailable proves a withdraw can only export REALIZED
// available balance: an over-withdraw exports only what is there (no mint).
func TestCustodyWithdrawClampsToAvailable(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const user = "holder"
	addBlock(t, vm, depositTx(t, user, assetLUX, 40))

	// Ask to withdraw 100; only 40 is available.
	wo := addBlockOutcomes(t, vm, withdrawTx(t, user, assetLUX, 100))
	var realized uint64
	for _, oc := range wo {
		if oc.typ == TxWithdraw {
			realized = oc.orderID
		}
	}
	if realized != 40 {
		t.Fatalf("over-withdraw realized %d, want clamped 40", realized)
	}
	assertBalance(t, vm, user, assetLUX, 0, 0)
}

// ---- helpers ----

func cancelPoolTx(t *testing.T, pool [32]byte, orderID uint64) *Tx {
	t.Helper()
	tx, err := NewTx(TxCancel, zapwire.EncodeCancel(pool, orderID))
	if err != nil {
		t.Fatalf("NewTx cancel: %v", err)
	}
	return tx
}

func bal(t *testing.T, vm *VM, user string, asset uint64) uint64 {
	t.Helper()
	avail, _, err := vm.Balance(user, asset)
	if err != nil {
		t.Fatalf("Balance(%s,%d): %v", user, asset, err)
	}
	return avail
}

func assertBalance(t *testing.T, vm *VM, user string, asset, wantAvail, wantLocked uint64) {
	t.Helper()
	avail, locked, err := vm.Balance(user, asset)
	if err != nil {
		t.Fatalf("Balance(%s,%d): %v", user, asset, err)
	}
	if avail != wantAvail || locked != wantLocked {
		t.Fatalf("balance %s asset=%#x: available=%d locked=%d, want available=%d locked=%d",
			user, asset, avail, locked, wantAvail, wantLocked)
	}
}
