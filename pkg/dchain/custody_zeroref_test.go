// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"errors"
	"testing"

	"github.com/luxfi/database/memdb"
)

// custody_zeroref_test.go pins R3 on the consensus side: a custody tx whose
// idempotency ref is all-zero is REJECTED at execute (every validator decodes the
// same frame and refuses identically, so the rejection is consensus-neutral) before
// any credit/debit. The ref is the originating-tx identity folded into tx.ID() and
// the seen: dedup key; a committed EVM tx always has a unique non-zero hash, so a
// zero ref is an unidentified (test-mock) or proxy fillRef=0 frame that must not
// mint an unbacked credit or release value against an untracked identity.

// zeroRef is the all-zero idempotency ref the handlers must refuse.
var zeroRef = [32]byte{}

// TestCustody_ZeroRefDepositRejected: a deposit carrying a zero ref fails at
// execute (surfaced by BuildBlock) and credits nothing.
func TestCustody_ZeroRefDepositRejected(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const user = "zero-dep"
	// A correctly-SIGNED deposit (so it passes the auth gate) but carrying a zero
	// ref: the zero-ref refusal must fire at execute regardless.
	tx := depositTxRef(t, user, assetLUX, 100, zeroRef)
	vm.mempool.Add(tx)

	// BuildBlock runs the execute probe; a zero-ref deposit must make it error.
	if _, err := vm.BuildBlock(ctx); err == nil {
		t.Fatal("BuildBlock with a zero-ref deposit must error, got nil")
	} else if !errors.Is(err, errZeroCustodyRef) {
		t.Fatalf("BuildBlock err = %v, want errZeroCustodyRef", err)
	}

	// Nothing was credited (the tx never committed).
	if avail, _, _ := vm.Balance(wireUser(t, user), assetLUX); avail != 0 {
		t.Fatalf("balance after refused zero-ref deposit = %d, want 0 (no mint)", avail)
	}
}

// TestCustody_ZeroRefWithdrawRejected: a withdraw carrying a zero ref fails at
// execute and debits nothing — even when the user holds a real (bound-deposited)
// balance the withdraw could otherwise release.
func TestCustody_ZeroRefWithdrawRejected(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const user = "zero-wd"
	// Seed a real balance via a NON-zero-ref deposit committed in a block.
	depRef := contentRef(byte(TxDeposit), user, assetLUX, 500)
	addBlock(t, vm, depositTxRef(t, user, assetLUX, 500, depRef))
	if avail, _, _ := vm.Balance(wireUser(t, user), assetLUX); avail != 500 {
		t.Fatalf("seed balance = %d, want 500", avail)
	}

	// A correctly-SIGNED but zero-ref withdraw must be refused at execute.
	wtx := withdrawTxRef(t, user, assetLUX, 500, zeroRef)
	vm.mempool.Add(wtx)
	if _, err := vm.BuildBlock(ctx); err == nil {
		t.Fatal("BuildBlock with a zero-ref withdraw must error, got nil")
	} else if !errors.Is(err, errZeroCustodyRef) {
		t.Fatalf("BuildBlock err = %v, want errZeroCustodyRef", err)
	}

	// The balance is UNCHANGED: the refused withdraw debited nothing.
	if avail, _, _ := vm.Balance(wireUser(t, user), assetLUX); avail != 500 {
		t.Fatalf("balance after refused zero-ref withdraw = %d, want 500 (no debit)", avail)
	}
}

// TestCustody_NonZeroRefStillWorks: the normal path — a non-zero ref deposit then
// withdraw both commit and move the ledger exactly, proving R3 only refuses the
// zero-ref door and does not regress real custody.
func TestCustody_NonZeroRefStillWorks(t *testing.T) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const user = "ok-ref"
	depRef := contentRef(byte(TxDeposit), user, assetLUX, 250)
	addBlock(t, vm, depositTxRef(t, user, assetLUX, 250, depRef))
	if avail, _, _ := vm.Balance(wireUser(t, user), assetLUX); avail != 250 {
		t.Fatalf("balance after non-zero-ref deposit = %d, want 250", avail)
	}

	wRef := contentRef(byte(TxWithdraw), user, assetLUX, 250)
	_, out := addBlockOutcomes(t, vm, withdrawTxRef(t, user, assetLUX, 250, wRef))
	var realized uint64
	for _, o := range out {
		if o.typ == TxWithdraw {
			realized = o.orderID
		}
	}
	if realized != 250 {
		t.Fatalf("non-zero-ref withdraw realized = %d, want 250", realized)
	}
	if avail, _, _ := vm.Balance(wireUser(t, user), assetLUX); avail != 0 {
		t.Fatalf("balance after non-zero-ref withdraw = %d, want 0 (debited)", avail)
	}
}
