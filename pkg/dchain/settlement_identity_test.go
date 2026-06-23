// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/binary"
	"errors"
	"sort"
	"testing"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// settlement_identity_test.go is the PERMANENT REGRESSION suite for the custody
// settlement-identity collision class: the matcher's 8-byte DEXOrder.UserID is a
// compact STP handle that MAY collide, and it MUST NEVER bear value. Settlement
// identity is resolved EXCLUSIVELY through the orderuser: index to the full 16-byte
// account. These tests pin that settlement FAILS CLOSED (ErrMissingSettlementUser,
// block rejected, no value moves) whenever the full identity is unavailable —
// rather than falling back to the compact handle, which would make a handle
// collision a theft primitive.
//
// MUTATION CONTRACT: reintroducing ANY fallback from the orderuser: index to the
// 8-byte handle (or the trade's 8-byte-derived carried user) makes the fail-closed
// and exploit tests below RED. That is the whole point.

// twoUsersSharingUserID8 returns a victim/attacker pair of full 16-byte wire users
// that FOLD TO THE SAME 8-byte matcher handle (identical leading 8 bytes) but are
// DISTINCT full accounts (differ in bytes 8..15) — exactly the collision the
// matcher handle permits and settlement must never honor.
func twoUsersSharingUserID8() (victim, attacker string) {
	victim = string([]byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08})
	attacker = string([]byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8})
	return victim, attacker
}

// buildVerify builds a block from the mempool's current txs and runs the VALIDATOR
// path (re-parse + Verify) WITHOUT Accept, returning the verify error (nil on
// success) and the verified block. Used to assert a block FAILS verification
// (fail-closed) without committing.
func buildVerify(t *testing.T, vm *VM) (*Block, error) {
	t.Helper()
	ctx := context.Background()
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		return nil, err
	}
	reparsed, err := vm.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock: %v", err)
	}
	blk := reparsed.(*Block)
	return blk, blk.Verify(ctx)
}

// deleteOrderUserRow removes the orderuser: row for a resting order directly from
// the VM's durable store — modeling the corruption/attack the fail-closed rule
// defends against (a maker fill/cancel whose full settlement identity has been
// lost or hidden). Operates under vm.mu, like the production state writes.
func deleteOrderUserRow(t *testing.T, vm *VM, poolID [32]byte, orderID uint64) {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	if err := vm.db.Delete(orderUserKey(poolID, orderID)); err != nil {
		t.Fatalf("delete orderuser row: %v", err)
	}
}

// hasOrderUserRow reports whether the orderuser: row exists for a resting order.
func hasOrderUserRow(t *testing.T, vm *VM, poolID [32]byte, orderID uint64) bool {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	_, ok, err := getOrderUser(vm.db, poolID, orderID)
	if err != nil {
		t.Fatalf("getOrderUser: %v", err)
	}
	return ok
}

// fundOpenAndRestMaker is the common setup: deposit maker base + taker quote, open
// the LUX/LUSD custody market, and rest a maker SELL of `size` LUX @ `price`.
// Returns the resting maker order id.
func fundOpenAndRestMaker(t *testing.T, vm *VM, pool [32]byte, maker, taker string, makerBase, takerQuote uint64, price, size float64) uint64 {
	t.Helper()
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, makerBase),
		depositTx(t, taker, assetLUSD, takerQuote),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, price, size, maker))
	return blockDeterministicID(blk.height, 0)
}

// ---------------------------------------------------------------------------
// 1. Settlement fails closed when the orderuser mapping is missing.
// ---------------------------------------------------------------------------

// TestMissingOrderUserFailsClosed proves a crossing submit against a resting maker
// whose orderuser: row is gone ABORTS with ErrMissingSettlementUser at Verify —
// no partial settlement, no balance mutation, head unchanged.
func TestMissingOrderUserFailsClosed(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const maker, taker = "maker-fc", "taker-fc"
	pool := [32]byte{0xfc, 0x01}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)
	assertBalance(t, vm, maker, assetLUX, 90, 10)

	// Build the crossing block (probe sees the row), THEN delete the row, THEN Verify
	// (the validator re-executes against state with the row gone).
	vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	deleteOrderUserRow(t, vm, pool, makerOrderID)

	reparsed, err := vm.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock: %v", err)
	}
	if err := reparsed.Verify(ctx); !errors.Is(err, ErrMissingSettlementUser) {
		t.Fatalf("Verify err = %v, want ErrMissingSettlementUser (settlement must fail closed, not fall back to the 8-byte handle)", err)
	}

	// No value moved: maker still has its locked 10 LUX, taker still has all quote.
	assertBalance(t, vm, maker, assetLUX, 90, 10)
	assertBalance(t, vm, maker, assetLUSD, 0, 0)
	assertBalance(t, vm, taker, assetLUSD, 1000, 0)
	assertBalance(t, vm, taker, assetLUX, 0, 0)
}

// ---------------------------------------------------------------------------
// 2. No maker fallback to the 8-byte handle.
// ---------------------------------------------------------------------------

// TestSettlementNeverFallsBackToMatcherUserID proves the maker's settlement
// identity comes ONLY from the orderuser: index: with the row deleted, the maker
// fill cannot resolve a 16-byte account and the block aborts — even though the
// matcher's 8-byte handle is still present on the fill/row. The settlement
// identity is the wallet (AccountID); the matcher's UserID8 is only a matching
// hint and NEVER a fallback for value. (If a fallback existed, the cross would
// settle, mis-routing the maker leg to a handle-derived account.)
func TestSettlementNeverFallsBackToMatcherUserID(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const maker, taker = "mk-nofb", "tk-nofb"
	pool := [32]byte{0xfb, 0x01}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)

	vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	deleteOrderUserRow(t, vm, pool, makerOrderID)
	reparsed, _ := vm.ParseBlock(ctx, built.Bytes())
	if err := reparsed.Verify(ctx); !errors.Is(err, ErrMissingSettlementUser) {
		t.Fatalf("maker-fill Verify err = %v, want ErrMissingSettlementUser (no fallback to 8-byte maker handle)", err)
	}
}

// ---------------------------------------------------------------------------
// 3. No taker fallback to the 8-byte handle.
// ---------------------------------------------------------------------------

// TestNoTakerFallbackToUserID8 proves the TAKER's settlement identity is the FULL
// 16-byte submit user, never an 8-byte fold: two takers that share their leading 8
// bytes (same matcher handle) but differ in bytes 8..15 settle to DISTINCT accounts.
// If the taker side folded to 8 bytes, both takers' proceeds would land on ONE
// colliding account.
func TestNoTakerFallbackToUserID8(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	takerA, takerB := twoUsersSharingUserID8()
	const maker = "mk-taker8"
	pool := [32]byte{0x78, 0x01}

	// Maker rests enough base for two crosses (20 LUX), each taker funded with 50 LUSD.
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 100),
		depositTx(t, takerA, assetLUSD, 50),
		depositTx(t, takerB, assetLUSD, 50),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 20.0, maker))

	// Taker A buys 10 LUX @ 5 (50 LUSD). Then taker B buys 10 LUX @ 5 (50 LUSD).
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, takerA))
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, takerB))

	// Each taker received its OWN 10 LUX — distinct accounts, no collision.
	assertBalance(t, vm, takerA, assetLUX, 10, 0)
	assertBalance(t, vm, takerB, assetLUX, 10, 0)
	assertBalance(t, vm, takerA, assetLUSD, 0, 0)
	assertBalance(t, vm, takerB, assetLUSD, 0, 0)
	// The maker received both takers' 100 LUSD.
	assertBalance(t, vm, maker, assetLUSD, 100, 0)
}

// ---------------------------------------------------------------------------
// 4. A resting order keeps its FULL 16-byte settlement identity across a restart.
// ---------------------------------------------------------------------------

// TestRestingOrderAfterRestartKeepsFullUser16 proves a maker that rests with a full
// 16-byte user, survives a VM restart (book + orderuser: rebuilt from durable
// state), and is then crossed, credits the FULL 16-byte account — not the 8-byte
// handle the rebuilt in-RAM order carries. The maker uses a user whose bytes 8..15
// are non-zero, so an 8-byte-folded credit would land on a DIFFERENT account.
func TestRestingOrderAfterRestartKeepsFullUser16(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)

	// Maker user with significant bytes 8..15 (would be lost by an 8-byte fold).
	maker := string([]byte{0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98})
	const taker = "tk-restart"
	pool := [32]byte{0x9e, 0x01}

	fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)
	vm.Shutdown(ctx)

	// Restart over the SAME db: book + ledger rebuilt from durable rows.
	vm2, _ := newTestVM(t, db)
	defer vm2.Shutdown(ctx)

	// Cross the rebuilt maker.
	addBlock(t, vm2, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))

	// The maker's FULL 16-byte account received the 50 LUSD and gave up 10 LUX.
	assertBalance(t, vm2, maker, assetLUX, 90, 0)
	assertBalance(t, vm2, maker, assetLUSD, 50, 0)
	// The 8-byte-folded would-be account (bytes 8..15 zeroed) got NOTHING.
	folded := string([]byte{0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80})
	assertBalance(t, vm2, folded, assetLUSD, 0, 0)
	assertBalance(t, vm2, folded, assetLUX, 0, 0)
}

// ---------------------------------------------------------------------------
// 5. Crash after book-insert but before orderuser-persist cannot settle.
// ---------------------------------------------------------------------------

// TestCrashAfterBookInsertBeforeOrderUserCannotSettle models the worst ordering
// hazard: a resting maker order exists in the book but its orderuser: row is
// absent (as if a crash interleaved between the two writes — which the production
// ordering forbids, but we force it here). A crossing taker MUST abort fail-closed
// rather than settle the maker leg to the 8-byte handle.
func TestCrashAfterBookInsertBeforeOrderUserCannotSettle(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const maker, taker = "mk-crash", "tk-crash"
	pool := [32]byte{0xc5, 0x01}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)

	// Force the hazard: drop the orderuser row, leaving the book row intact.
	deleteOrderUserRow(t, vm, pool, makerOrderID)

	// A crossing taker now aborts at BuildBlock's probe OR Verify — either way the
	// block never settles. Drive both: BuildBlock executes the probe first.
	vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	_, verr := buildVerify(t, vm)
	if !errors.Is(verr, ErrMissingSettlementUser) {
		t.Fatalf("crossing a book row with no orderuser err = %v, want ErrMissingSettlementUser", verr)
	}
	// Nothing settled.
	assertBalance(t, vm, maker, assetLUX, 90, 10)
	assertBalance(t, vm, taker, assetLUSD, 1000, 0)
}

// ---------------------------------------------------------------------------
// 6. orderuser is persisted BEFORE the order is matchable.
// ---------------------------------------------------------------------------

// TestOrderUserPersistedBeforeBookInsert proves the placement ordering: once a
// place is accepted, the resting order has BOTH a per-order lock reserve AND an
// orderuser: identity row — neither is deferred past the book insert. (The lock +
// identity are written inside lockOrderSpend, which runs BEFORE applyTx inserts the
// order into the book.) A funded resting order therefore NEVER exists without its
// settlement identity.
func TestOrderUserPersistedBeforeBookInsert(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const maker, taker = "mk-order", "tk-order"
	pool := [32]byte{0x0d, 0x01}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)

	// The resting order carries its full 16-byte settlement identity.
	if !hasOrderUserRow(t, vm, pool, makerOrderID) {
		t.Fatal("resting order has NO orderuser row after place (identity not persisted before book insert)")
	}
	vm.mu.Lock()
	gotUser, ok, err := getOrderUser(vm.db, pool, makerOrderID)
	vm.mu.Unlock()
	if err != nil || !ok {
		t.Fatalf("getOrderUser ok=%v err=%v", ok, err)
	}
	if gotUser != userKey16(maker) {
		t.Fatalf("orderuser row = %x, want full 16-byte maker identity %x", gotUser, userKey16(maker))
	}
	// And a per-order lock reserve exists in lockstep.
	vm.mu.Lock()
	_, lockedAmt, lockOK, lerr := getOrderLock(vm.db, pool, makerOrderID)
	vm.mu.Unlock()
	if lerr != nil || !lockOK || lockedAmt == 0 {
		t.Fatalf("resting order missing per-order lock reserve: ok=%v amt=%d err=%v", lockOK, lockedAmt, lerr)
	}
}

// ---------------------------------------------------------------------------
// 7. THE EXPLOIT: a colliding userID8 cannot steal a victim maker fill.
// ---------------------------------------------------------------------------

// TestCollidingUserID8CannotReceiveMakerProceeds is the headline theft PoC. A
// victim deposits and rests a custody maker. An attacker shares the victim's 8-byte
// matcher handle (differs in bytes 8..15). The attacker's orderuser: mapping is then
// hidden (modeling the attacker forcing settlement onto the bare 8-byte handle), and
// the attacker tries to cross/claim the victim's resting fill. The cross MUST abort
// ErrMissingSettlementUser: NO attacker credit, NO victim debit, conservation holds.
// (The ownership axis of this exact class — that the colliding attacker is never an
// explicit party to the fill — is pinned in
// TestOwnershipInvariant_CollidingAttackerIsNotAParty.)
//
// MUTATION: if settlement fell back to the 8-byte handle, the colliding attacker
// would resolve to the victim's handle and the value would mis-route — this test
// goes RED, exactly as required.
func TestCollidingUserID8CannotReceiveMakerProceeds(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	victim, attacker := twoUsersSharingUserID8()
	// Sanity: the two users DO collide on the 8-byte matcher handle but are distinct
	// full accounts (the precondition for the exploit class).
	if userKey16(victim) == userKey16(attacker) {
		t.Fatal("test setup invalid: victim/attacker are the same full account")
	}
	pool := [32]byte{0xae, 0x01}

	// Victim deposits 100 LUX and rests a SELL of 10 LUX @ 5 (locks 10 LUX). Taker
	// (the attacker, sharing the handle) is funded with quote to attempt the cross.
	addBlock(t, vm,
		depositTx(t, victim, assetLUX, 100),
		depositTx(t, attacker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, victim))
	victimOrderID := blockDeterministicID(blk.height, 0)
	assertBalance(t, vm, victim, assetLUX, 90, 10)

	// Build the attacker's crossing buy (probe succeeds), then HIDE the victim's
	// orderuser mapping (the attack: force settlement onto the bare 8-byte handle),
	// then Verify as a validator would.
	vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, attacker))
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	deleteOrderUserRow(t, vm, pool, victimOrderID)
	reparsed, _ := vm.ParseBlock(ctx, built.Bytes())
	if err := reparsed.Verify(ctx); !errors.Is(err, ErrMissingSettlementUser) {
		t.Fatalf("colliding-handle cross Verify err = %v, want ErrMissingSettlementUser (THEFT: attacker would settle the victim's fill)", err)
	}

	// CONSERVATION + no theft: victim keeps its locked LUX, attacker keeps its quote.
	assertBalance(t, vm, victim, assetLUX, 90, 10)
	assertBalance(t, vm, victim, assetLUSD, 0, 0)
	assertBalance(t, vm, attacker, assetLUSD, 1000, 0)
	assertBalance(t, vm, attacker, assetLUX, 0, 0)
	// Global LUX conserved (100 deposited == 90 available + 10 locked, all victim's;
	// attacker holds zero LUX in either bucket).
	vAvail, vLocked := mustBalance(t, vm, victim, assetLUX)
	aAvail, aLocked := mustBalance(t, vm, attacker, assetLUX)
	if got := vAvail + vLocked + aAvail + aLocked; got != 100 {
		t.Fatalf("LUX conservation violated after aborted theft: %d want 100", got)
	}
}

// mustBalance reads (available, locked) for (user, asset), failing the test on a
// ledger read error.
func mustBalance(t *testing.T, vm *VM, user string, asset [32]byte) (uint64, uint64) {
	t.Helper()
	avail, locked, err := vm.Balance(user, asset)
	if err != nil {
		t.Fatalf("Balance(%s): %v", user, err)
	}
	return avail, locked
}

// TestCollidingUserID8CannotWithdrawVictimFunds is the withdraw axis of the
// exploit: the victim's funds live in the ledger keyed by the FULL 16-byte account
// (the wallet / AccountID). The attacker, sharing only the 8-byte matcher handle,
// withdraws — and gets NOTHING from the victim's balance, because the ledger never
// keys by the handle. This proves the balance ledger (not just settlement) is
// collision-safe.
func TestCollidingUserID8CannotWithdrawVictimFunds(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	victim, attacker := twoUsersSharingUserID8()
	pool := [32]byte{0xad, 0x01}

	// Victim deposits 100 LUX and locks 10 in a resting sell; 90 stays available.
	addBlock(t, vm,
		depositTx(t, victim, assetLUX, 100),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, victim))
	assertBalance(t, vm, victim, assetLUX, 90, 10)
	// Attacker has deposited nothing.
	assertBalance(t, vm, attacker, assetLUX, 0, 0)

	// Attacker tries to withdraw 90 LUX (the victim's available). The ledger keys by
	// the FULL 16-byte attacker account, which has zero balance -> realized 0.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, attacker, assetLUX, 90))
	var attackerRealized uint64
	for _, oc := range wo {
		if oc.typ == TxWithdraw {
			attackerRealized = oc.orderID
		}
	}
	if attackerRealized != 0 {
		t.Fatalf("attacker withdrew %d LUX from a colliding handle, want 0 (THEFT via the matcher handle)", attackerRealized)
	}
	// Victim's balance is untouched.
	assertBalance(t, vm, victim, assetLUX, 90, 10)
	assertBalance(t, vm, attacker, assetLUX, 0, 0)
}

// ---------------------------------------------------------------------------
// 8. Every resting book order has an orderuser mapping after a restart.
// ---------------------------------------------------------------------------

// TestEveryBookOrderHasOrderUserMappingAfterRestart proves the settlement-identity
// invariant holds across a restart: after rebooting over a healthy custody chain,
// EVERY resting order on EVERY custody market has an orderuser: row, and Initialize
// succeeds — with NO boot scan (the rows are present because the commit path wrote
// them atomically, not because Initialize re-checks them). That committed state can
// never lack a row is pinned by construction in
// TestCommittedStateNeverHasOrderWithoutIdentity; that boot no longer bricks on a
// state that the commit path cannot produce is pinned in TestRestartDoesNotBrickBoot.
func TestEveryBookOrderHasOrderUserMappingAfterRestart(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)

	const m1, m2, taker = "rest-m1", "rest-m2", "rest-tk"
	pool := [32]byte{0xeb, 0x01}

	addBlock(t, vm,
		depositTx(t, m1, assetLUX, 100),
		depositTx(t, m2, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	// Two resting makers on the custody market.
	addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, m1),
		placePoolTx(t, pool, zapwire.SideSell, 6.0, 12.0, m2),
	)
	vm.Shutdown(ctx)

	// Restart: books are folded from the committed order:* rows (no boot scan).
	vm2, _ := newTestVM(t, db)
	defer vm2.Shutdown(ctx)

	// Every resting order on the market has an orderuser row.
	vm2.mu.Lock()
	ob := vm2.books[pool]
	vm2.mu.Unlock()
	if ob == nil {
		t.Fatal("market book did not rebuild on restart")
	}
	if len(ob.Orders) != 2 {
		t.Fatalf("rebuilt book has %d resting orders, want 2", len(ob.Orders))
	}
	for _, o := range ob.Orders {
		if !hasOrderUserRow(t, vm2, pool, o.ID) {
			t.Fatalf("resting order %d has NO orderuser row after restart", o.ID)
		}
	}
}

// TestMultiPlaceInBlockKeysOrderUserByMatcherIndex proves the per-order reserve and
// identity rows key under the EXACT order id the matcher assigns — the index is
// threaded from execute's loop (not recovered by scanning the block for the tx
// pointer). Two maker PLACEs in ONE block land at txIndex 0 and 1; each rests under
// blockDeterministicID(height, its-own-index) with a matching orderuser: row, and
// each is independently crossable to its OWN account. (A pointer-scan id recovery
// could mis-resolve the second place's index to the first, keying its identity row
// under the wrong order id and tripping ErrMissingSettlementUser at fill time.)
func TestMultiPlaceInBlockKeysOrderUserByMatcherIndex(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const m1, m2, taker = "mp-m1", "mp-m2", "mp-tk"
	pool := [32]byte{0x37, 0x01}

	addBlock(t, vm,
		depositTx(t, m1, assetLUX, 100),
		depositTx(t, m2, assetLUX, 100),
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	// Two places in ONE block: m1 at txIndex 0, m2 at txIndex 1.
	blk := addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, m1),
		placePoolTx(t, pool, zapwire.SideSell, 7.0, 8.0, m2),
	)
	id0 := blockDeterministicID(blk.height, 0)
	id1 := blockDeterministicID(blk.height, 1)

	// Each resting order's orderuser row is keyed under its OWN matcher index and
	// names its OWN full account.
	vm.mu.Lock()
	u0, ok0, _ := getOrderUser(vm.db, pool, id0)
	u1, ok1, _ := getOrderUser(vm.db, pool, id1)
	vm.mu.Unlock()
	if !ok0 || u0 != userKey16(m1) {
		t.Fatalf("order %d (txIndex 0) orderuser = %x ok=%v, want m1 %x", id0, u0, ok0, userKey16(m1))
	}
	if !ok1 || u1 != userKey16(m2) {
		t.Fatalf("order %d (txIndex 1) orderuser = %x ok=%v, want m2 %x", id1, u1, ok1, userKey16(m2))
	}

	// Cross m1's order (sell 10@5): taker buys 10@5 -> m1 receives 50 LUSD.
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	assertBalance(t, vm, m1, assetLUSD, 50, 0)
	assertBalance(t, vm, m1, assetLUX, 90, 0)
	// m2 untouched (its order still rests, 8 LUX locked).
	assertBalance(t, vm, m2, assetLUX, 92, 8)
	assertBalance(t, vm, m2, assetLUSD, 0, 0)
}

// TestCommittedStateNeverHasOrderWithoutIdentity is the DECOMPLECT proof: it
// demonstrates the settlement-identity invariant is enforced BY CONSTRUCTION at
// order creation, so a resting custody order WITHOUT its orderuser: row is
// unrepresentable in committed state. It drives the ONLY path that writes committed
// state — funded places that REST, an unfunded place that is REJECTED (never rests),
// and a cross that removes a maker — then scans the committed key space and asserts
// the set of order:<pool> rows is EXACTLY the set of orderuser:<pool> rows. There is
// no order row without its identity row and no identity row without its order row,
// because lockOrderSpend (orderuser:) and putOrderRow (order:) write to the SAME
// versiondb overlay and the block commits in ONE atomic overlay.Commit() — they land
// together or not at all. This is why the boot scan (formerly assertOrderUserCoverage)
// is provably dead and was deleted: the bad state it scanned for cannot exist.
func TestCommittedStateNeverHasOrderWithoutIdentity(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)
	defer vm.Shutdown(ctx)

	const m1, m2, poor, taker = "ci-m1", "ci-m2", "ci-poor", "ci-tk"
	pool := [32]byte{0xc1, 0x07}

	addBlock(t, vm,
		depositTx(t, m1, assetLUX, 100),
		depositTx(t, m2, assetLUX, 100),
		depositTx(t, poor, assetLUX, 0), // funded with NOTHING: its place cannot lock, so it cannot rest
		depositTx(t, taker, assetLUSD, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	// Two funded maker places (rest) + one unfunded place (rejected, never rests) in
	// ONE block. The unfunded order is the hostile case: if a rejected place could
	// leave an order: row, it would be one with no orderuser: row — exactly the state
	// the deleted boot-brick guarded against. It cannot: lockOrderSpend returns !ok
	// and execute skips applyTx, so no order row is written.
	addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, m1),
		placePoolTx(t, pool, zapwire.SideSell, 6.0, 12.0, m2),
		placePoolTx(t, pool, zapwire.SideSell, 7.0, 5.0, poor), // unfunded -> rejected
	)
	// Cross m1 fully (taker buys 10@5): m1's order leaves the book — its order: row
	// AND its orderuser: row must be removed in lockstep (no orphan either way).
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))

	// Scan committed state: collect every order:<pool> id and every orderuser:<pool> id.
	orderIDs := scanIDs(t, vm, orderPrefixFor(pool), len(prefixOrder))
	userIDs := scanIDs(t, vm, append([]byte(prefixOrderUser), pool[:]...), len(prefixOrderUser))

	// The two sets are IDENTICAL: every resting order has an identity row, and no
	// identity row is orphaned. (m2 rests -> present in both; m1 fully filled -> in
	// neither; poor rejected -> in neither.)
	if !idSetsEqual(orderIDs, userIDs) {
		t.Fatalf("committed order: set %v != orderuser: set %v — illegal state (order without identity, or orphan identity) is representable",
			sortedIDs(orderIDs), sortedIDs(userIDs))
	}
	// Exactly one resting order survives (m2); sanity that the path actually exercised
	// rest + reject + fill, not a degenerate empty book.
	if len(orderIDs) != 1 {
		t.Fatalf("expected exactly 1 resting order after rest+reject+fill, got %d (%v)", len(orderIDs), sortedIDs(orderIDs))
	}
}

// TestRestartDoesNotBrickBoot proves the boot-brick is GONE. It forces the EXACT
// corrupt state the deleted assertOrderUserCoverage scanned for — a resting custody
// order whose orderuser: row is removed by a raw db.Delete (a state the commit path
// can never produce; see TestCommittedStateNeverHasOrderWithoutIdentity) — and shows
// that a restart now SUCCEEDS instead of bricking. The settlement-identity guarantee
// has not weakened: the single remaining guard is consensus-time, so a fill or cancel
// that actually TOUCHES the corrupt order still fails closed with
// ErrMissingSettlementUser. Boot is no longer a place where that invariant is checked.
func TestRestartDoesNotBrickBoot(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)

	const maker, taker = "boot-mk", "boot-tk"
	pool := [32]byte{0xbf, 0x01}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)
	vm.Shutdown(ctx)

	// Corrupt the durable state by hand: drop the resting order's orderuser: row while
	// its order: row survives. This is unreachable through the commit path — we reach
	// in past it purely to prove boot no longer scans for (and bricks on) it.
	if err := db.Delete(orderUserKey(pool, makerOrderID)); err != nil {
		t.Fatalf("delete orderuser row: %v", err)
	}

	// A restart now SUCCEEDS (no boot-brick): the VM folds the book and comes up.
	toEngine := make(chan block.Message, 16)
	vm2 := &VM{}
	if err := vm2.Initialize(ctx, block.Init{DB: db, Log: log.NewNoOpLogger(), ToEngine: toEngine}); err != nil {
		t.Fatalf("Initialize over (hand-)corrupted custody state err = %v, want nil (boot must NOT brick — the invariant lives at creation + consensus-time, not boot)", err)
	}
	defer vm2.Shutdown(ctx)

	// The book rebuilt with the maker order resting.
	vm2.mu.Lock()
	ob := vm2.books[pool]
	vm2.mu.Unlock()
	if ob == nil || ob.GetOrder(makerOrderID) == nil {
		t.Fatal("maker order did not rebuild after restart")
	}

	// Settlement identity is STILL enforced where it matters: a taker that crosses the
	// corrupt maker order fails closed at consensus-time (Verify), not at boot.
	vm2.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	if _, verr := buildVerify(t, vm2); !errors.Is(verr, ErrMissingSettlementUser) {
		t.Fatalf("crossing the corrupt maker err = %v, want ErrMissingSettlementUser (consensus-time guard must still fail closed)", verr)
	}
}

// scanIDs returns the orderID suffix of every key under prefix in the VM's committed
// db. keyHeadLen is the length of the constant key head BEFORE the 8-byte big-endian
// orderID (i.e. len(prefix)+32 for an order:/orderuser: key, since the caller passes
// the prefix INCLUDING the 32-byte poolID).
func scanIDs(t *testing.T, vm *VM, prefix []byte, keyHeadLen int) map[uint64]struct{} {
	t.Helper()
	_ = keyHeadLen // the prefix already includes the poolID; the id is the trailing 8 bytes
	ids := map[uint64]struct{}{}
	vm.mu.Lock()
	defer vm.mu.Unlock()
	it := vm.db.NewIteratorWithPrefix(prefix)
	defer it.Release()
	for it.Next() {
		k := it.Key()
		if len(k) < 8 {
			continue
		}
		ids[binary.BigEndian.Uint64(k[len(k)-8:])] = struct{}{}
	}
	if err := it.Error(); err != nil {
		t.Fatalf("scan %q: %v", prefix, err)
	}
	return ids
}

// idSetsEqual reports whether two orderID sets are identical.
func idSetsEqual(a, b map[uint64]struct{}) bool {
	if len(a) != len(b) {
		return false
	}
	for id := range a {
		if _, ok := b[id]; !ok {
			return false
		}
	}
	return true
}

// sortedIDs renders an id set as a sorted slice for stable failure messages.
func sortedIDs(s map[uint64]struct{}) []uint64 {
	out := make([]uint64, 0, len(s))
	for id := range s {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// ---------------------------------------------------------------------------
// 9. A sub-dust BUY never rests, so it never creates a zero-lock resting order
//    with no settlement identity.
// ---------------------------------------------------------------------------

// TestSubDustBuyCannotRestWithoutOrderUser ties the zero-notional reject to the
// settlement-identity invariant: a BUY whose notional floors to zero lock (a
// sub-tick price) is REJECTED at placement, so it NEVER rests — and therefore can
// never become a resting book order that lacks a non-zero lock AND an orderuser:
// row (the degraded path settlement must never hit). This is the structural reason
// "every resting custody order carries a lock + identity" stays total: an order
// that cannot lock cannot rest, so it cannot be a resting order missing its
// identity. (Mutation: allowing a zero-lock BUY to rest would create exactly such
// an identity-less resting order; this test goes RED.)
func TestSubDustBuyCannotRestWithoutOrderUser(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const buyer = "subdust-buyer"
	pool := [32]byte{0x5d, 0x05}
	addBlock(t, vm,
		depositTx(t, buyer, assetLUSD, 1_000_000), // funded -> the reject is the notional floor, not affordability
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	// A sub-tick BUY (price 5e-9 < 1 tick, size 10) floors ceil(size*price) to 0.
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, dustPrice, 10.0, buyer))
	orderID := blockDeterministicID(blk.height, 0)

	// It did NOT rest: no book order, no per-order lock, and crucially NO orderuser
	// identity row (the order never reached the book, so it has no settlement
	// identity to be missing).
	if orders, _, _, _, _ := vm.BookDepth(pool); orders != 0 {
		t.Fatalf("sub-dust BUY rested: book has %d orders, want 0", orders)
	}
	if hasOrderUserRow(t, vm, pool, orderID) {
		t.Fatal("sub-dust BUY created an orderuser row for an order that never rested")
	}
	vm.mu.Lock()
	_, lockedAmt, lockOK, lerr := getOrderLock(vm.db, pool, orderID)
	vm.mu.Unlock()
	if lerr != nil {
		t.Fatalf("getOrderLock: %v", lerr)
	}
	if lockOK || lockedAmt != 0 {
		t.Fatalf("sub-dust BUY held a lock: ok=%v amt=%d, want none", lockOK, lockedAmt)
	}
	// The buyer's quote is fully available — nothing reserved.
	assertBalance(t, vm, buyer, assetLUSD, 1_000_000, 0)
}

// ---------------------------------------------------------------------------
// 10. The settlement identity is exactly the full wire-user width.
// ---------------------------------------------------------------------------

// TestSettlementIdentityWidthEnforced pins that the settlement identity (the
// wallet / AccountID the ledger keys value by) is the FULL 16-byte wire user
// (zapwire.UserSize), NOT the matcher's 8-byte STP handle. It asserts: (a) the
// userKey type is 16 bytes; (b) a resting order's persisted orderuser: row is
// exactly 16 bytes and equals the full wire user; (c) two accounts sharing the
// leading 8 bytes resolve to DISTINCT 16-byte identities (so a handle collision is
// not an identity collision). Width is the load-bearing property: a fold to 8
// bytes is what would make the matcher handle value-bearing.
func TestSettlementIdentityWidthEnforced(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	// (a) The settlement identity type is the full wire width.
	if got := len(userKey{}); got != zapwire.UserSize {
		t.Fatalf("settlement identity width = %d, want zapwire.UserSize (%d)", got, zapwire.UserSize)
	}
	if zapwire.UserSize != 16 {
		t.Fatalf("zapwire.UserSize = %d, want 16 (the settlement identity must be 16 bytes, not the 8-byte handle)", zapwire.UserSize)
	}

	// (b) A resting order persists its identity at full 16-byte width.
	victim, attacker := twoUsersSharingUserID8()
	pool := [32]byte{0x1d, 0x77}
	makerOrderID := fundOpenAndRestMaker(t, vm, pool, victim, "tk-width", 100, 1000, 5.0, 10.0)
	vm.mu.Lock()
	raw, rerr := vm.db.Get(orderUserKey(pool, makerOrderID))
	gotUser, ok, gerr := getOrderUser(vm.db, pool, makerOrderID)
	vm.mu.Unlock()
	if rerr != nil || gerr != nil || !ok {
		t.Fatalf("orderuser read: raw_err=%v get_err=%v ok=%v", rerr, gerr, ok)
	}
	if len(raw) != zapwire.UserSize {
		t.Fatalf("persisted orderuser row width = %d, want %d (settlement identity must be stored full-width)", len(raw), zapwire.UserSize)
	}
	if gotUser != userKey16(victim) {
		t.Fatalf("orderuser = %x, want full 16-byte victim %x", gotUser, userKey16(victim))
	}

	// (c) Two accounts sharing the leading 8 bytes are DISTINCT 16-byte identities.
	if userKey16(victim) == userKey16(attacker) {
		t.Fatal("colliding-handle accounts resolved to the SAME 16-byte identity (width fold lost bytes 8..15)")
	}
	// And their leading 8 bytes DO match (the precondition: the collision is at the
	// handle, not the identity).
	v16, a16 := userKey16(victim), userKey16(attacker)
	if string(v16[:8]) != string(a16[:8]) {
		t.Fatal("test setup invalid: the two accounts do not share the 8-byte matcher handle")
	}
	if string(v16[8:]) == string(a16[8:]) {
		t.Fatal("test setup invalid: the two accounts do not differ in bytes 8..15")
	}
}

// ---------------------------------------------------------------------------
// 11. MUTATION: rebuilding a maker from the 8-byte handle after restart breaks
//     settlement (the identity must come from the full-width orderuser: row).
// ---------------------------------------------------------------------------

// TestRestartRebuildsMakerFromFullIdentityNotUserID8 is the explicit mutation pin
// for the restart path: a maker that rests with a full 16-byte wallet, survives a
// VM restart (its in-RAM order is rebuilt from the GPU row carrying only 8 user
// bytes), and is then crossed, MUST credit the FULL 16-byte account — resolved
// from the orderuser: row, NOT the rebuilt order's 8-byte handle. The maker's
// bytes 8..15 are non-zero, so an 8-byte-fold rebuild would credit a DIFFERENT
// account and this test goes RED. (This is the named restart-axis companion to
// TestRestingOrderAfterRestartKeepsFullUser16.)
func TestRestartRebuildsMakerFromFullIdentityNotUserID8(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	vm, _ := newTestVM(t, db)

	// A maker wallet with significant bytes 8..15 (an 8-byte fold would lose them).
	maker := string([]byte{0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8})
	const taker = "tk-rebuild"
	pool := [32]byte{0x2b, 0x11}

	fundOpenAndRestMaker(t, vm, pool, maker, taker, 100, 1000, 5.0, 10.0)
	vm.Shutdown(ctx)

	// Restart over the SAME db: the book is rebuilt from the durable order:* rows
	// (whose UserID is only 8 bytes); the settlement identity must still come from
	// the full-width orderuser: row.
	vm2, _ := newTestVM(t, db)
	defer vm2.Shutdown(ctx)

	// Cross the rebuilt maker; ownership must hold (only the full-id maker and the
	// taker change value), and the maker's FULL account is credited.
	own := newOwnershipChecker(t, vm2)
	blk := addBlock(t, vm2, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, taker))
	own.assert(t, vm2, blk, "restart-rebuild-cross")

	assertBalance(t, vm2, maker, assetLUX, 90, 0)
	assertBalance(t, vm2, maker, assetLUSD, 50, 0)
	// The 8-byte-folded would-be account (bytes 8..15 zeroed) got NOTHING.
	folded := string([]byte{0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8})
	assertBalance(t, vm2, folded, assetLUSD, 0, 0)
	assertBalance(t, vm2, folded, assetLUX, 0, 0)
}
