// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

// ownership_completeness_test.go ADVERSARIALLY tests the ownership harness itself
// (ownership_test.go) — the new safety net. A blind spot in the harness is false
// confidence: a conserving theft the harness fails to flag would slip the gate.
//
// Strategy for each violation class:
//   1. Drive a REAL block through the production path (addBlock / restart), so the
//      before-snapshot, the derived party set, and the real after-snapshot are all
//      exactly what the gate would see in the property/full-cycle suites.
//   2. Assert the gate is CLEAN on the real (honest) after-state — no false flag.
//   3. PLANT a value-conserving violation into a CLONE of the after-state (so the
//      live ledger is untouched) and assert findOwnershipViolation FIRES, naming
//      the expected non-party coordinate.
//
// All plants are value-conserving (Σ per asset unchanged) — exactly the class
// conservation cannot see, so the only thing standing between it and theft is the
// ownership gate firing. If ANY plant slips past, that is a HARNESS GAP.
//
// These tests reuse the SAME functions the gate uses (findOwnershipViolation,
// partiesOfBlock via ownershipChecker.parties) so they exercise the real net, not
// a re-implementation.

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

// thirdParty is an account that is NEVER a party to any event in the planted
// blocks — the canonical "value arrived at an account no event named" recipient.
// Its bytes 8..15 are non-zero so it is also distinct from any 8-byte fold.
var thirdParty = string([]byte{0x9e, 0x9e, 0x9e, 0x9e, 0x9e, 0x9e, 0x9e, 0x9e, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33})

// creditControlled adds amt to (user,asset) available in a snapshot WITHOUT a
// matching debit — models value APPEARING at an account (a one-sided credit). Used
// when the test asserts conservation is intentionally NOT the thing under test (a
// pure ownership question about a single account).
func creditControlled(snap map[acctAsset]balPair, user userKey, asset [32]byte, amt uint64) {
	k := acctAsset{user, asset}
	p := snap[k]
	p.available += amt
	snap[k] = p
}

// assertFires asserts findOwnershipViolation FIRES for the given before/after over
// the party set, and (when wantUser/wantAsset are non-nil) that it names the
// expected coordinate. The harness is the net; this proves the net catches the fish.
func assertFires(t *testing.T, before, after map[acctAsset]balPair, parties partySet, wantUser *userKey, wantAsset *[32]byte, what string) {
	t.Helper()
	k, delta, bad := findOwnershipViolation(before, after, parties)
	if !bad {
		t.Fatalf("%s: ownership gate FAILED to flag the violation (HARNESS GAP — a conserving theft would slip the gate)", what)
	}
	if wantUser != nil && k.user != *wantUser {
		t.Fatalf("%s: gate flagged user %x, want %x", what, k.user, *wantUser)
	}
	if wantAsset != nil && k.asset != *wantAsset {
		t.Fatalf("%s: gate flagged asset %x, want %x (delta %d)", what, k.asset, *wantAsset, delta)
	}
	t.Logf("%s: gate fired on %x/%x delta %d", what, k.user[:8], k.asset[:8], delta)
}

// assertClean asserts findOwnershipViolation does NOT fire — the honest after-state
// must pass (no false positive against real settlement).
func assertClean(t *testing.T, before, after map[acctAsset]balPair, parties partySet, what string) {
	t.Helper()
	if k, delta, bad := findOwnershipViolation(before, after, parties); bad {
		t.Fatalf("%s: ownership gate FALSE-flagged honest settlement: %x/%x delta %d (over-strict gate)", what, k.user, k.asset, delta)
	}
}

// uk resolves a human name to the real key-derived 16-byte account the gate and
// ledger key by (so plants debit/credit the SAME account real settlement used).
func uk(t *testing.T, s string) userKey { return acctFor(t, s).account }

// restMakerCross is the common single-maker / single-taker cross fixture: victim
// (maker) rests a SELL of size LUX @ price, a legit taker crosses it fully. Returns
// the checker snapshotted BEFORE the cross, the cross block, and the maker/taker.
// All callers then plant a conserving theft and assert the gate fires.
func restMakerCross(t *testing.T, vm *VM, pool [32]byte, maker, taker string, price, size float64) (*ownershipChecker, *Block) {
	t.Helper()
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 1000),
		depositTx(t, taker, assetLUSD, 100000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, price, size, maker))
	oc := newOwnershipChecker(t, vm)
	blk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, price, size, taker))
	return oc, blk
}

// ---------------------------------------------------------------------------
// (B.1) maker proceeds routed to a WRONG / EXTRA account.
// ---------------------------------------------------------------------------

func TestOwnershipHarness_MakerProceedsToWrongAccount(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const maker, taker = "b1-maker", "b1-taker"
	pool := [32]byte{0xb1}
	oc, blk := restMakerCross(t, vm, pool, maker, taker, 5.0, 10.0)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)
	assertClean(t, before, after, parties, "honest cross")

	// Plant: maker's 50-LUSD proceeds re-routed to thirdParty (a non-party). Conserving.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, maker), assetLUSD, uk(t, thirdParty), assetLUSD, 50)
	wantU, wantA := uk(t, thirdParty), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "maker proceeds -> wrong account")
}

// EXTRA account: maker keeps its proceeds AND thirdParty also receives a credit (a
// one-sided extra credit — not conserving, but the ownership gate must still flag
// the unexplained recipient regardless of conservation).
func TestOwnershipHarness_MakerProceedsToExtraAccount(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const maker, taker = "b1e-maker", "b1e-taker"
	pool := [32]byte{0xb1, 0xee}
	oc, blk := restMakerCross(t, vm, pool, maker, taker, 5.0, 10.0)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)

	vuln := cloneBalances(after)
	creditControlled(vuln, uk(t, thirdParty), assetLUSD, 50) // extra, unexplained
	wantU, wantA := uk(t, thirdParty), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "maker proceeds -> extra account")
}

// ---------------------------------------------------------------------------
// (B.2) TAKER proceeds mis-routed.
// ---------------------------------------------------------------------------

func TestOwnershipHarness_TakerProceedsMisrouted(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const maker, taker = "b2-maker", "b2-taker"
	pool := [32]byte{0xb2}
	oc, blk := restMakerCross(t, vm, pool, maker, taker, 5.0, 10.0)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)
	assertClean(t, before, after, parties, "honest cross")

	// Plant: taker's 10-LUX proceeds re-routed to thirdParty. Conserving.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, taker), assetLUX, uk(t, thirdParty), assetLUX, 10)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "taker proceeds -> wrong account")
}

// ---------------------------------------------------------------------------
// (B.3) CANCEL / refund credited to a non-owner.
// ---------------------------------------------------------------------------

func TestOwnershipHarness_CancelRefundToNonOwner(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const owner = "b3-owner"
	pool := [32]byte{0xb3}
	addBlock(t, vm,
		depositTx(t, owner, assetLUX, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	pblk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, owner))
	orderID := blockDeterministicID(pblk.height, 0)

	oc := newOwnershipChecker(t, vm)
	cblk := addBlock(t, vm, cancelPoolTx(t, pool, orderID, owner))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, cblk)
	assertClean(t, before, after, parties, "honest cancel (refund to owner)")

	// The owner is a party (cancel resolves owner via orderuser:). Plant the refund
	// landing on thirdParty instead: thirdParty's controlled LUX rises, owner's falls.
	// Conserving. thirdParty is not the order's owner, so the gate must fire.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, owner), assetLUX, uk(t, thirdParty), assetLUX, 10)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "cancel refund -> non-owner")
}

// ---------------------------------------------------------------------------
// (B.4) WITHDRAW crediting a non-withdrawer.
// ---------------------------------------------------------------------------

func TestOwnershipHarness_WithdrawCreditsNonWithdrawer(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const w = "b4-withdrawer"
	addBlock(t, vm, depositTx(t, w, assetLUX, 1000))

	oc := newOwnershipChecker(t, vm)
	wblk := addBlock(t, vm, withdrawTx(t, w, assetLUX, 400))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, wblk)
	assertClean(t, before, after, parties, "honest withdraw")

	// A withdraw is a one-sided DEBIT from the withdrawer (value leaves the ledger).
	// The withdrawer IS a party. Plant a credit to thirdParty (modeling a withdraw
	// that, instead of removing value, mis-credits a non-withdrawer). Gate must fire.
	vuln := cloneBalances(after)
	creditControlled(vuln, uk(t, thirdParty), assetLUX, 400)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "withdraw credits non-withdrawer")
}

// ---------------------------------------------------------------------------
// (B.5) A balance delta for an account with NO accepted event at all.
// ---------------------------------------------------------------------------

func TestOwnershipHarness_DeltaForAccountWithNoEvent(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	// A block that touches only maker+taker; thirdParty is in NO event.
	const maker, taker = "b5-maker", "b5-taker"
	pool := [32]byte{0xb5}
	oc, blk := restMakerCross(t, vm, pool, maker, taker, 5.0, 10.0)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)

	// Sanity: thirdParty is genuinely not a party in any asset.
	if parties.has(uk(t, thirdParty), assetLUX) || parties.has(uk(t, thirdParty), assetLUSD) {
		t.Fatal("test precondition broken: thirdParty wrongly in the party set")
	}

	// Plant: a bare delta on thirdParty in BOTH assets (no event references it).
	vuln := cloneBalances(after)
	creditControlled(vuln, uk(t, thirdParty), assetLUX, 1)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "delta for account with no event")
}

// ---------------------------------------------------------------------------
// (B.6) A multi-maker fill where ONE maker is mis-attributed.
// ---------------------------------------------------------------------------

// TestOwnershipHarness_MultiMakerOneMisattributed rests TWO makers at the same
// price, then a single taker BUY consumes BOTH in one cross (two fills -> two trade
// rows). The honest gate sees both makers + the taker as parties. Plant: maker2's
// proceeds re-routed to thirdParty. The gate must fire even though maker1's leg is
// honest and the taker's legs are honest — a single mis-attributed maker in a
// multi-fill cross is exactly the subtle case.
func TestOwnershipHarness_MultiMakerOneMisattributed(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const maker1, maker2, taker = "b6-m1", "b6-m2", "b6-tk"
	pool := [32]byte{0xb6}
	addBlock(t, vm,
		depositTx(t, maker1, assetLUX, 1000),
		depositTx(t, maker2, assetLUX, 1000),
		depositTx(t, taker, assetLUSD, 100000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	// Two resting SELLs at the same price 5, 10 LUX each.
	addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker1),
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, maker2),
	)

	oc := newOwnershipChecker(t, vm)
	// Taker buys 20 LUX @ 5 -> crosses BOTH makers (two fills).
	blk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 20.0, taker))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)

	// Both makers must be parties (resolved through the block's two trade rows).
	if !parties.has(uk(t, maker1), assetLUSD) || !parties.has(uk(t, maker2), assetLUSD) {
		t.Fatalf("multi-maker party set missing a maker: m1=%v m2=%v",
			parties.has(uk(t, maker1), assetLUSD), parties.has(uk(t, maker2), assetLUSD))
	}
	assertClean(t, before, after, parties, "honest multi-maker cross")

	// Plant: maker2's 50-LUSD leg re-routed to thirdParty. Conserving.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, maker2), assetLUSD, uk(t, thirdParty), assetLUSD, 50)
	wantU, wantA := uk(t, thirdParty), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "multi-maker: one maker mis-attributed")
}

// ---------------------------------------------------------------------------
// (B.7) A partial-fill maker spanning MULTIPLE blocks (resting across a block) —
//       is it still resolved to the full party?
// ---------------------------------------------------------------------------

// TestOwnershipHarness_PartialFillMakerAcrossBlocks rests a LARGE maker (50 LUX) in
// block N, then a taker partially fills 10 LUX in block N+2 (maker still rests with
// 40), then ANOTHER taker fills another 10 in block N+3. For the SECOND cross the
// maker's orderuser: row was written TWO blocks earlier — the harness must resolve
// the still-resting maker to its FULL account from the pre-block orderuser: snapshot
// (not from this block's placements). We assert the gate fires when that second
// cross's maker leg is mis-routed.
func TestOwnershipHarness_PartialFillMakerAcrossBlocks(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	maker := string([]byte{0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8})
	const t1, t2 = "b7-t1", "b7-t2"
	pool := [32]byte{0xb7}
	addBlock(t, vm,
		depositTx(t, maker, assetLUX, 1000),
		depositTx(t, t1, assetLUSD, 100000),
		depositTx(t, t2, assetLUSD, 100000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	pblk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 50.0, maker))
	makerOrderID := blockDeterministicID(pblk.height, 0)

	// First partial fill (10 of 50). Maker still rests with 40.
	addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, t1))
	// The maker still has an orderuser: row (resting), keyed under makerOrderID.
	if !hasOrderUserRow(t, vm, pool, makerOrderID) {
		t.Fatal("partially-filled maker lost its orderuser row (should still rest)")
	}

	// SECOND cross, in a later block — the maker rested 2+ blocks ago.
	oc := newOwnershipChecker(t, vm)
	blk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, t2))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)

	// The maker (resting across blocks) MUST be resolved to its full account as a
	// party of this cross — from the pre-block orderuser: snapshot.
	if !parties.has(uk(t, maker), assetLUSD) {
		t.Fatal("HARNESS GAP: cross-block resting maker not resolved to a party (orderuser: pre-snapshot miss)")
	}
	// And the 8-byte fold of the maker is NOT separately admitted.
	folded := string([]byte{0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC})
	if parties.has(uk(t, folded), assetLUSD) {
		t.Fatal("party set wrongly admitted the maker's 8-byte fold as a distinct party")
	}
	assertClean(t, before, after, parties, "honest cross-block partial fill")

	// Plant: this cross's maker leg (50 LUSD) re-routed to the 8-byte fold (the exact
	// theft a fold-settlement would produce). Conserving. The fold is NOT a party.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, maker), assetLUSD, uk(t, folded), assetLUSD, 50)
	wantU, wantA := uk(t, folded), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "cross-block maker leg -> 8-byte fold")
}

// ---------------------------------------------------------------------------
// (B.8) FEE attribution — does a fee skim leave an unexplained delta?
// ---------------------------------------------------------------------------

// TestOwnershipHarness_NoFeeSkimAndFeeRecipientWouldFire documents the fee axis.
// This ledger takes NO fee (settleFills moves maker<->taker with no skim; the
// conservation property asserts F==0). So in the honest after-state there is no fee
// recipient delta. We assert (a) the honest cross has no extra delta, and (b) IF a
// fee were skimmed to a recipient that is not a party, the gate WOULD fire — i.e.
// the harness does not have a fee blind spot (a fee recipient is not silently
// admitted as a party). If a fee leg is ever added, its recipient must be made an
// explicit party in partiesOfBlock or this gate fires.
func TestOwnershipHarness_NoFeeSkimAndFeeRecipientWouldFire(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const maker, taker = "b8-maker", "b8-taker"
	pool := [32]byte{0xb8}
	oc, blk := restMakerCross(t, vm, pool, maker, taker, 5.0, 10.0)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)
	assertClean(t, before, after, parties, "honest cross (no fee)")

	feeRecipient := string([]byte{0xfe, 0xe0, 0xfe, 0xe0, 0xfe, 0xe0, 0xfe, 0xe0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08})
	if parties.has(uk(t, feeRecipient), assetLUSD) {
		t.Fatal("a fee recipient is wrongly already a party (harness over-permits)")
	}
	// Plant a fee skim: 1 LUSD diverted from the maker's proceeds to a fee recipient.
	// Conserving (maker 49, fee 1). The fee recipient is NOT a party -> gate fires.
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, maker), assetLUSD, uk(t, feeRecipient), assetLUSD, 1)
	wantU, wantA := uk(t, feeRecipient), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "fee skim to non-party recipient")
}

// ---------------------------------------------------------------------------
// (B.9) SELF-TRADE (maker == taker) — does partiesOfBlock handle it?
// ---------------------------------------------------------------------------

// TestOwnershipHarness_SelfTradeHandled rests a maker and then crosses it with a
// submit from the SAME account. The matcher's self-trade prevention may reject the
// cross outright; either way the harness must (a) not crash, (b) not false-flag the
// honest outcome, and (c) still fire if value is diverted to a third party. We drive
// the scenario and assert the gate behaves on whatever outcome the matcher produces,
// then plant a third-party diversion sized to the maker's locked reserve and confirm
// the gate fires.
func TestOwnershipHarness_SelfTradeHandled(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	self := string([]byte{0x5e, 0x1f, 0x5e, 0x1f, 0x5e, 0x1f, 0x5e, 0x1f, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8})
	pool := [32]byte{0xb9}
	// Self funds BOTH legs (base to rest a sell, quote to attempt the buy cross).
	addBlock(t, vm,
		depositTx(t, self, assetLUX, 1000),
		depositTx(t, self, assetLUSD, 100000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, self))

	oc := newOwnershipChecker(t, vm)
	// Submit a BUY from the same account — self-trade. partiesOfBlock must handle the
	// taker==maker identity (the taker is added; the maker resolves to the same uid).
	blk := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideBuy, false, 5.0, 10.0, self))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk) // must not panic on self-trade

	// `self` is a party in both assets regardless of whether the cross filled or was
	// self-trade-rejected (it is the taker; and, if filled, also the maker).
	if !parties.has(uk(t, self), assetLUX) || !parties.has(uk(t, self), assetLUSD) {
		t.Fatalf("self-trade: self not a party in both assets (lux=%v lusd=%v)",
			parties.has(uk(t, self), assetLUX), parties.has(uk(t, self), assetLUSD))
	}
	assertClean(t, before, after, parties, "honest self-trade outcome")

	// Plant: divert 1 LUX of self's controlled value to thirdParty. Conserving.
	// (self controls its LUX whether it filled-to-itself or the cross was rejected
	// and the lock refunded — either way the total is unchanged and self is a party.)
	vuln := cloneBalances(after)
	moveControlled(vuln, uk(t, self), assetLUX, uk(t, thirdParty), assetLUX, 1)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "self-trade: value diverted to third party")
}

// ---------------------------------------------------------------------------
// OVER-PERMIT direction: partiesOfBlock must NOT mark an account a party when it
// isn't. A delta on a placed-order OWNER for an asset the place does NOT touch must
// still be flagged.
// ---------------------------------------------------------------------------

// TestOwnershipHarness_DoesNotOverPermitPlaceOwner places a SELL (locks base=LUX)
// from an owner. The place makes the owner a party for LUX only (the spend asset).
// A planted delta on that same owner in QUOTE (LUSD) — an asset the place never
// touches — must STILL be flagged: a place-owner is a party only for the asset it
// locks, not a blanket pass for every asset.
func TestOwnershipHarness_DoesNotOverPermitPlaceOwner(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const owner = "op-owner"
	pool := [32]byte{0xb0, 0x09}
	addBlock(t, vm,
		depositTx(t, owner, assetLUX, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)

	oc := newOwnershipChecker(t, vm)
	blk := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, owner))

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)
	assertClean(t, before, after, parties, "honest place")

	// The owner is a party for LUX (the locked spend asset) ...
	if !parties.has(uk(t, owner), assetLUX) {
		t.Fatal("place owner not a party for the locked base asset")
	}
	// ... but NOT for LUSD (a SELL never touches quote).
	if parties.has(uk(t, owner), assetLUSD) {
		t.Fatal("HARNESS OVER-PERMIT: place SELL owner wrongly a party for the quote asset it never locks")
	}
	// Plant a delta on the owner in LUSD -> must fire (owner is not a quote party here).
	vuln := cloneBalances(after)
	creditControlled(vuln, uk(t, owner), assetLUSD, 7)
	wantU, wantA := uk(t, owner), assetLUSD
	assertFires(t, before, vuln, parties, &wantU, &wantA, "place SELL owner delta in untouched quote asset")
}

// TestOwnershipHarness_DoesNotOverPermitUninvolvedAccountOnBusyBlock builds a block
// with several places by DIFFERENT owners; an account that placed nothing and is in
// no event gets a planted delta and must be flagged — the harness must not admit
// "any account that appears anywhere" as a party.
func TestOwnershipHarness_DoesNotOverPermitUninvolvedAccountOnBusyBlock(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)
	const a, b, c = "busy-a", "busy-b", "busy-c"
	pool := [32]byte{0xb0, 0x5b}
	addBlock(t, vm,
		depositTx(t, a, assetLUX, 1000),
		depositTx(t, b, assetLUX, 1000),
		depositTx(t, c, assetLUX, 1000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	oc := newOwnershipChecker(t, vm)
	// Three places by a, b, c.
	blk := addBlock(t, vm,
		placePoolTx(t, pool, zapwire.SideSell, 5.0, 10.0, a),
		placePoolTx(t, pool, zapwire.SideSell, 6.0, 10.0, b),
		placePoolTx(t, pool, zapwire.SideSell, 7.0, 10.0, c),
	)

	before := oc.before
	after := ownershipSnapshot(t, vm)
	parties := oc.parties(t, vm, blk)
	assertClean(t, before, after, parties, "honest busy block")

	// thirdParty placed nothing. A delta on it must fire.
	vuln := cloneBalances(after)
	creditControlled(vuln, uk(t, thirdParty), assetLUX, 3)
	wantU, wantA := uk(t, thirdParty), assetLUX
	assertFires(t, before, vuln, parties, &wantU, &wantA, "uninvolved account delta on busy block")
}
