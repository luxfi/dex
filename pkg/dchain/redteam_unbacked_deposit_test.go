// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// redteam_unbacked_deposit_test.go — RED TEAM (infinite money), now a REGRESSION
// GUARD for the F9 fix. A TxDeposit MINTS ledger balance, so it must be authorized
// by the trusted deposit AUTHORITY (the bridge/proxy that custodies the backing
// C-side value), NEVER by the crediting account self-signing. This test proves:
//
//   1. a SELF-SIGNED deposit (the attacker signs their own credit) is REJECTED at
//      the public RPC boundary AND at consensus — it mints 0, withdraws 0;
//   2. a legitimate AUTHORITY-signed deposit credits the beneficiary correctly and
//      is fully withdrawable (the real bridge funding flow still works).
//
// Backing invariant: Σ(ledger credits) ≤ Σ(authority-authorized deposits); an
// authenticated principal can no longer mint unbacked balance.
package dchain

import (
	"context"
	"errors"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

func TestRedteam_UnbackedSelfSignedDepositMints(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const attacker = "attacker"
	const minted uint64 = 500_000_000_000

	att := acctFor(t, attacker)
	if a, l, _ := vm.Balance(att.user, assetLUX); a != 0 || l != 0 {
		t.Fatalf("precondition: attacker not empty (avail=%d locked=%d)", a, l)
	}

	// The attacker SELF-SIGNS a deposit crediting their own account — the F9 mint.
	ref := contentRef(byte(TxDeposit), attacker, assetLUX, minted)
	selfBody := zapwire.EncodeDeposit(att.user, assetLUX, minted, ref)
	selfTx := att.signed(t, TxDeposit, selfBody)

	// (1) PUBLIC-BOUNDARY: the RPC pre-screen REJECTS a deposit not signed by the
	// configured bridge authority (submitTx). The unbacked mint never enters the mempool.
	frame := append(append([]byte{}, selfTx.Body...), selfTx.Auth.encode()...)
	if _, verr := vm.submitTx(context.Background(), TxDeposit, frame); !errors.Is(verr, ErrTxBadSignature) {
		t.Fatalf("public dex_deposit boundary admitted a self-signed deposit: err=%v, want ErrTxBadSignature", verr)
	}

	// (2) CONSENSUS: even if a malicious proposer smuggles the tx into a block, the
	// AUTHORITATIVE gate (authorizeTx) rejects it — mints 0.
	addBlock(t, vm, selfTx)
	if avail, _, _ := vm.Balance(att.user, assetLUX); avail != 0 {
		t.Fatalf("SELF-SIGNED DEPOSIT MINTED %d — the F9 infinite-money glitch is BACK", avail)
	}

	// (3) DRAIN attempt: nothing was minted, so a withdraw realizes 0.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, attacker, assetLUX, minted))
	if realized := withdrawRealizedOf(wo, TxWithdraw); realized != 0 {
		t.Fatalf("attacker withdrew %d from an unbacked deposit — mint not fully closed", realized)
	}
	t.Logf("GUARD OK: self-signed deposit rejected at boundary + consensus; minted 0, withdrew 0")
}

// TestBackedAuthoritySignedDepositCredits proves the LEGITIMATE flow: a deposit
// signed by the configured bridge authority credits the beneficiary exactly and is
// fully withdrawable. This is the real funding path the backing invariant permits.
func TestBackedAuthoritySignedDepositCredits(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const user = "honest-depositor"
	const amount uint64 = 777_000

	ben := acctFor(t, user)
	// fundAcct signs with the authority; the body names `ben` as the beneficiary.
	ref := contentRef(byte(TxDeposit), user, assetLUX, amount)
	addBlock(t, vm, fundAcct(t, ben, assetLUX, amount, ref))

	if avail, locked, _ := vm.Balance(ben.user, assetLUX); avail != amount || locked != 0 {
		t.Fatalf("authority deposit credited avail=%d locked=%d, want %d / 0", avail, locked, amount)
	}

	// The credited balance is real and fully withdrawable by its owner.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, user, assetLUX, amount))
	if realized := withdrawRealizedOf(wo, TxWithdraw); realized != amount {
		t.Fatalf("withdraw of a backed deposit realized %d, want %d", realized, amount)
	}
	if avail, _, _ := vm.Balance(ben.user, assetLUX); avail != 0 {
		t.Fatalf("after full withdraw avail=%d, want 0", avail)
	}
	t.Logf("GUARD OK: authority-signed deposit credited %d and withdrew %d (legitimate backed flow works)", amount, amount)
}
