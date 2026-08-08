// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// custody_timeout_test.go is the REGRESSION PROOF for the orphan-custody-tx bug
// (RED #4): handler.submitTx did mempool.Add(tx) then blocked on the waiter; on
// ctx.Done() it cancelled ONLY the waiter, never removing the tx. The mempool has
// no dedup and BuildBlock drains ALL pending, so a client timeout left the tx
// QUEUED -> it committed in a LATER block = orphan. A DEPOSIT orphan is the worst:
// the EVM reverts (the vault msg.value is rolled back by the StateDB snapshot when
// the precompile's ZAP call timed out) but the D-Chain STILL credits = MINT.
//
// THE FIX: submitTx now calls mempool.cancel(txID) on ctx.Done(). cancel removes a
// still-pending tx outright; if the tx already drained into an in-flight block it
// is TOMBSTONED so a later Drain/Requeue excludes it. Either way a cancelled
// custody tx can NEVER commit in a later block.
//
// These tests drive the d-chain VM directly (no live socket) so the timeout window
// is deterministic: a custody tx submitted with a ctx that expires WHILE it sits in
// the mempool (no sealer draining) must (a) make submitTx return ctx.Err(), (b)
// leave the mempool empty (removed), and (c) NOT credit the ledger on a subsequent
// build — no orphan, no mint. (The existing handler/relay tests only cover
// dial-fail, never this timeout-after-Add path.)

// newTimeoutVM builds an initialized VM with NO auto-sealer, so a submitted tx
// sits in the mempool until the test explicitly builds — giving a deterministic
// window to time out submitTx while the tx is still pending.
func newTimeoutVM(t *testing.T) *VM {
	t.Helper()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		Genesis:  []byte(testDocument),
		DB:       memdb.New(),
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	t.Cleanup(func() { _ = vm.Shutdown(context.Background()) })
	return vm
}

// TestCustodyDepositTimeoutDoesNotCreditOrOrphan is the #4 proof for a DEPOSIT:
// submitTx times out while the deposit sits in the mempool. The fix must remove it
// so it never commits — no D-Chain credit (no mint to match the EVM's rolled-back
// vault lock).
func TestCustodyDepositTimeoutDoesNotCreditOrOrphan(t *testing.T) {
	vm := newTimeoutVM(t)
	const (
		user   = "alice"
		amount = uint64(1000)
	)
	asset := a32(0x4c5558_00000001)
	depRef := contentRef(byte(TxDeposit), user, asset, amount)
	depBody := encodeDepositBody(wireUser(t, user), asset, amount, depRef)
	// A deposit is authorized by the trusted bridge AUTHORITY (F9), not the user.
	depPayload := signedPayload(t, depositAuthorityName, TxDeposit, depBody)

	// submitTx with a ctx that expires almost immediately. No sealer is running, so
	// the tx is Add'ed to the mempool and the ctx fires before anything drains it.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	_, err := vm.submitTx(ctx, TxDeposit, depPayload)
	if err == nil {
		t.Fatal("submitTx returned nil error on a timed-out deposit; want ctx deadline error")
	}
	if err != context.DeadlineExceeded && ctx.Err() == nil {
		t.Fatalf("submitTx err = %v, want a ctx timeout", err)
	}

	// THE FIX: the timed-out deposit was REMOVED from the mempool, not left queued.
	if n := vm.mempool.Len(); n != 0 {
		t.Fatalf("mempool holds %d txs after a timed-out deposit — the orphan is still QUEUED "+
			"and WILL commit in a later block (mint). cancel() must have removed it.", n)
	}

	// And a subsequent build finds nothing -> no block -> no credit. (BuildBlock
	// returns ErrEmptyMempool on an empty pool; that is the correct "nothing to
	// commit" outcome — the deposit never reaches the ledger.)
	if _, berr := vm.BuildBlock(context.Background()); berr != ErrEmptyMempool {
		t.Fatalf("BuildBlock after timeout = %v, want ErrEmptyMempool (the orphan deposit must not build a block)", berr)
	}

	// The ledger never credited the timed-out deposit: available is 0 (no mint).
	avail, locked, err := vm.Balance(wireUser(t, user), asset)
	if err != nil {
		t.Fatalf("Balance: %v", err)
	}
	if avail != 0 || locked != 0 {
		t.Fatalf("ledger credited a TIMED-OUT deposit: available=%d locked=%d, want 0/0 — "+
			"this is the orphan->mint the EVM's rolled-back vault lock cannot back", avail, locked)
	}
}

// TestCustodyWithdrawTimeoutDoesNotCommit is the #4 proof for a WITHDRAW: a
// timed-out withdraw must not later commit (it would export against a vault the EVM
// did not release). It also confirms the deposit it follows is unaffected.
func TestCustodyWithdrawTimeoutDoesNotCommit(t *testing.T) {
	vm := newTimeoutVM(t)
	const (
		user = "bob"
		dep  = uint64(500)
	)
	asset := a32(0x4c5558_00000001)

	// First, a real deposit committed via the explicit build path (so there IS a
	// balance a stray withdraw could illegitimately export).
	addBlock(t, vm, depositTx(t, user, asset, dep))
	if avail, _, _ := vm.Balance(wireUser(t, user), asset); avail != dep {
		t.Fatalf("setup deposit available = %d, want %d", avail, dep)
	}

	// Now a withdraw whose submitTx times out while queued.
	wRef := contentRef(byte(TxWithdraw), user, asset, dep)
	wBody := encodeWithdrawBody(wireUser(t, user), asset, dep, wRef)
	wPayload := signedPayload(t, user, TxWithdraw, wBody)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	if _, err := vm.submitTx(ctx, TxWithdraw, wPayload); err == nil {
		t.Fatal("submitTx returned nil on a timed-out withdraw; want ctx timeout")
	}

	// The withdraw was removed; the mempool is empty; no block commits it.
	if n := vm.mempool.Len(); n != 0 {
		t.Fatalf("mempool holds %d after a timed-out withdraw — orphan would export against an unreleased vault", n)
	}
	if _, berr := vm.BuildBlock(context.Background()); berr != ErrEmptyMempool {
		t.Fatalf("BuildBlock after withdraw timeout = %v, want ErrEmptyMempool", berr)
	}

	// The balance is UNCHANGED: the timed-out withdraw debited nothing (no export
	// leg the EVM never authorized).
	if avail, _, _ := vm.Balance(wireUser(t, user), asset); avail != dep {
		t.Fatalf("balance after a timed-out withdraw = %d, want %d (the orphan withdraw must NOT debit)", avail, dep)
	}
}

// TestMempoolCancelRemovesPending is the unit proof that cancel removes a pending
// tx so Drain (BuildBlock's mechanism) never sees it.
func TestMempoolCancelRemovesPending(t *testing.T) {
	m := newMempool(nil)
	keep := mustTx(t, TxEnsureMarket, encodeEnsureBody([32]byte{0x01}))
	drop := depositTxRef(t, "u", a32(7), 100, [32]byte{0xab})

	m.Add(keep)
	m.Add(drop)
	if m.Len() != 2 {
		t.Fatalf("len = %d, want 2", m.Len())
	}

	m.cancel(drop.ID(), 1) // pending: removed outright (height arg unused on this path)
	if m.Len() != 1 {
		t.Fatalf("len after cancel = %d, want 1 (the cancelled tx must be gone)", m.Len())
	}
	got := m.Drain(0)
	if len(got) != 1 || got[0].ID() != keep.ID() {
		t.Fatalf("Drain returned %d txs; want exactly the non-cancelled one", len(got))
	}
}

// TestMempoolTombstoneSurvivesRequeue is the unit proof for the in-flight-block
// race: a tx drained into a block, then cancelled (tombstoned), must NOT come back
// when that block is rejected and Requeued — it can never commit in a later block.
func TestMempoolTombstoneSurvivesRequeue(t *testing.T) {
	m := newMempool(nil)
	a := depositTxRef(t, "a", a32(1), 10, [32]byte{0x01})
	b := depositTxRef(t, "b", a32(2), 20, [32]byte{0x02})

	m.Add(a)
	m.Add(b)
	drained := m.Drain(0) // both leave the pending queue (into a "building block")
	if len(drained) != 2 {
		t.Fatalf("drained %d, want 2", len(drained))
	}

	// The submitter of `b` times out AFTER it was drained -> tombstone, stamped with
	// the in-flight block height (1).
	m.cancel(b.ID(), 1)

	// The block is rejected; its txs are Requeued. `b` must be dropped (tombstoned),
	// only `a` returns.
	m.Requeue(drained)
	back := m.Drain(0)
	if len(back) != 1 || back[0].ID() != a.ID() {
		t.Fatalf("after tombstone+requeue, Drain returned %d txs; want only the non-cancelled `a` "+
			"(the cancelled `b` must never re-enter a block)", len(back))
	}

	// The tombstone was consumed (it does not linger to drop a future legitimate tx
	// that happens to share the id — ids are content+ref unique, but hygiene matters).
	m.mu.Lock()
	nLeft := len(m.tombstones)
	m.mu.Unlock()
	if nLeft != 0 {
		t.Fatalf("tombstones not cleared after consumption: %d left", nLeft)
	}
}

// ---- local frame/body builders (avoid importing zapwire ref-codecs into the
// test's hot path; these mirror the bodies NewTx expects) ----

func mustTx(t *testing.T, typ TxType, body []byte) *Tx {
	t.Helper()
	tx, err := NewTx(typ, body)
	if err != nil {
		t.Fatalf("NewTx(%v): %v", typ, err)
	}
	return tx
}

// the following thin wrappers keep the test readable and route through the same
// zapwire encoders production uses (so the bodies are byte-identical to the wire).
func encodeDepositBody(user string, asset [32]byte, amount uint64, ref [32]byte) []byte {
	return zapwire.EncodeDeposit(user, asset, amount, ref)
}
func encodeWithdrawBody(user string, asset [32]byte, amount uint64, ref [32]byte) []byte {
	return zapwire.EncodeWithdraw(user, asset, amount, ref)
}
func encodeEnsureBody(pool [32]byte) []byte { return zapwire.EncodeEnsureMarket(pool) }
