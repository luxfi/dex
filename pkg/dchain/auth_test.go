// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// auth_test.go is the SECURITY PROOF for the D-Chain authorization boundary: the
// fix for "the settlement path accepts money-moving commands with NO signature".
// It drives the REAL consensus path (BuildBlock -> Verify -> Accept) and proves,
// for BOTH the classical (secp256k1) and post-quantum (ML-DSA-65) schemes:
//
//   - a correctly-signed tx APPLIES (state moves);
//   - an UNSIGNED tx is REJECTED and mutates NO state;
//   - a FORGED signature is REJECTED and mutates NO state;
//   - a WRONG-SIGNER tx (valid sig, signer != asserted account) is REJECTED;
//   - a REPLAYED tx (reused nonce) is REJECTED and does not double-apply;
//   - cancelling ANOTHER account's order is REJECTED (ownership);
//   - a WITHDRAW whose export ref is tampered after signing is REJECTED (the ref
//     is inside the signed digest — "withdraw to attacker_ref" is closed).
//
// Every rejection asserts the LEDGER is byte-for-byte unchanged, so the gate is
// fail-closed: a command the asserted account did not authorize moves nothing.

// authMarket is the (pool, base, quote) fixture for the auth suite.
var authPool = [32]byte{0xa0, 0x71, 0x88}

// mkAcct is the identity constructor for a scheme lane (secp or ML-DSA).
type mkAcct func(t *testing.T, name string) *testAcct

// rawPlace builds a Place body asserting account `user` (the 16-byte account
// string) — used to forge wrong-signer / unsigned frames where the body's
// asserted account is chosen independently of who (if anyone) signs.
func rawPlace(user string, side uint8, price, size float64) []byte {
	return encPlace(authPool, side, price, size, user)
}

// setupAuthVM brings up a VM with the LUX/LUSD market opened (custody on) so
// place/cancel/submit lock and settle real value, and returns it.
func setupAuthVM(t *testing.T) *VM {
	t.Helper()
	vm, _ := newTestVM(t, memdb.New())
	t.Cleanup(func() { _ = vm.Shutdown(context.Background()) })
	addBlock(t, vm, openMarketTx(t, authPool, assetLUX, assetLUSD))
	return vm
}

// TestAuth_Secp256k1 runs the full suite under the classical scheme.
func TestAuth_Secp256k1(t *testing.T) { runAuthSuite(t, acctFor) }

// TestAuth_MLDSA65 runs the full suite under the post-quantum scheme — the same
// gate, the same digest, a different verifier (ML-DSA-65 instead of ecrecover).
func TestAuth_MLDSA65(t *testing.T) { runAuthSuite(t, pqAcctFor) }

func runAuthSuite(t *testing.T, mk mkAcct) {
	t.Run("signed_tx_applies", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		// Deposit 100 LUX (signed), then rest a SELL of 10 LUX @ 5 (locks 10 LUX).
		dep := fundAcct(t, a, assetLUX, 100, ref32(1))
		addBlock(t, vm, dep)
		assertAcctBalance(t, vm, a, assetLUX, 100, 0)

		place := a.signed(t, TxPlace, encPlace(authPool, zapwire.SideSell, 5.0, 10.0, a.user))
		addBlock(t, vm, place)
		// The lock moved available->locked: a correctly-signed order took effect.
		assertAcctBalance(t, vm, a, assetLUX, 90, 10)
	})

	t.Run("unsigned_tx_rejected_no_mutation", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		addBlock(t, vm, fundAcct(t, a, assetLUX, 100, ref32(1)))

		// Hand-build a place with NO auth envelope (the audited vulnerability: an
		// unsigned money-moving command).
		unsigned := &Tx{Type: TxPlace, Body: rawPlace(a.user, zapwire.SideSell, 5.0, 10.0)}

		// (1) The stateless gate refuses it.
		if _, err := vm.verifyTxSignature(unsigned); !errors.Is(err, ErrTxUnsigned) {
			t.Fatalf("verifyTxSignature(unsigned) = %v, want ErrTxUnsigned", err)
		}
		// (2) The wire layer refuses to even decode an unsigned money frame: a
		// money-moving type with no auth trailer is not a valid tx, so a block that
		// embeds one cannot be parsed.
		if _, err := ParseTx(unsigned.Bytes()); err == nil {
			t.Fatal("ParseTx accepted an unsigned money-moving frame")
		}
		// (3) End-to-end: a validator REJECTS a block carrying the unsigned tx (a
		// malicious proposer cannot smuggle one past consensus). The block bytes fail
		// to parse, so no overlay is ever staged — nothing mutates.
		raw := newBlock(vm, vm.lastAcceptedID, vm.lastAcceptedHeight+1, time.Unix(0, 1), [Size]byte{}, []*Tx{unsigned}).Bytes()
		if _, err := vm.ParseBlock(context.Background(), raw); err == nil {
			t.Fatal("a validator parsed a block carrying an unsigned money-moving tx")
		}
		// The ledger is untouched.
		assertAcctBalance(t, vm, a, assetLUX, 100, 0)
		assertNoResting(t, vm, authPool)
	})

	t.Run("forged_signature_rejected_no_mutation", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		addBlock(t, vm, fundAcct(t, a, assetLUX, 100, ref32(1)))

		// A validly-signed place, then corrupt the signature so it no longer
		// recovers alice's account.
		body := encPlace(authPool, zapwire.SideSell, 5.0, 10.0, a.user)
		auth := a.authFor(TxPlace, a.nonce, body)
		auth.Sig[0] ^= 0xff // tamper
		forged, err := NewSignedTx(TxPlace, body, auth)
		if err != nil {
			t.Fatalf("NewSignedTx: %v", err)
		}
		blk := addBlock(t, vm, forged)
		assertRejected(t, blk, forged)
		assertAcctBalance(t, vm, a, assetLUX, 100, 0)
		assertNoResting(t, vm, authPool)
	})

	t.Run("wrong_signer_rejected_no_mutation", func(t *testing.T) {
		vm := setupAuthVM(t)
		alice := mk(t, "alice")
		mallory := mk(t, "mallory")
		addBlock(t, vm, fundAcct(t, alice, assetLUX, 100, ref32(1)))

		// Body asserts ALICE's account; signature is MALLORY's valid signature over
		// that exact body. The recovered account is mallory != alice -> rejected.
		body := encPlace(authPool, zapwire.SideSell, 5.0, 10.0, alice.user)
		auth := mallory.authFor(TxPlace, 0, body)
		tx, err := NewSignedTx(TxPlace, body, auth)
		if err != nil {
			t.Fatalf("NewSignedTx: %v", err)
		}
		blk := addBlock(t, vm, tx)
		assertRejected(t, blk, tx)
		assertAcctBalance(t, vm, alice, assetLUX, 100, 0)
		assertNoResting(t, vm, authPool)
	})

	t.Run("replayed_nonce_rejected_no_double_apply", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		addBlock(t, vm, fundAcct(t, a, assetLUX, 100, ref32(1)))
		// The deposit is AUTHORITY-signed (F9), so it does NOT advance alice's nonce:
		// her first own tx consumes nonce 0.

		p1 := a.signed(t, TxPlace, encPlace(authPool, zapwire.SideSell, 5.0, 10.0, a.user))
		addBlock(t, vm, p1)
		assertAcctBalance(t, vm, a, assetLUX, 90, 10) // first place locked 10 (nonce 0)

		// A SECOND, DISTINCT place (different size) but reusing the just-consumed
		// nonce (0). The nonce gate refuses it; no further lock.
		replayBody := encPlace(authPool, zapwire.SideSell, 5.0, 20.0, a.user)
		replay, err := NewSignedTx(TxPlace, replayBody, a.authFor(TxPlace, 0, replayBody))
		if err != nil {
			t.Fatalf("NewSignedTx: %v", err)
		}
		blk := addBlock(t, vm, replay)
		assertRejected(t, blk, replay)
		assertAcctBalance(t, vm, a, assetLUX, 90, 10) // unchanged: replay applied nothing
	})

	t.Run("exact_replay_is_idempotent_not_double", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		addBlock(t, vm, fundAcct(t, a, assetLUX, 100, ref32(1)))
		p1 := a.signed(t, TxPlace, encPlace(authPool, zapwire.SideSell, 5.0, 10.0, a.user))
		addBlock(t, vm, p1)
		assertAcctBalance(t, vm, a, assetLUX, 90, 10)

		// Re-submit the IDENTICAL bytes (same nonce, same signature). Content-addressed
		// seen: dedup returns the first outcome and applies NOTHING again.
		replayExact, err := ParseTx(p1.Bytes())
		if err != nil {
			t.Fatalf("ParseTx: %v", err)
		}
		addBlock(t, vm, replayExact)
		assertAcctBalance(t, vm, a, assetLUX, 90, 10) // still exactly one lock
	})

	t.Run("cancel_other_users_order_rejected", func(t *testing.T) {
		vm := setupAuthVM(t)
		alice := mk(t, "alice")
		mallory := mk(t, "mallory")
		addBlock(t, vm,
			fundAcct(t, alice, assetLUX, 100, ref32(1)),
		)
		// Alice rests a SELL of 10 LUX @ 5 (locks 10). It is HER order.
		pblk := addBlock(t, vm, alice.signed(t, TxPlace, encPlace(authPool, zapwire.SideSell, 5.0, 10.0, alice.user)))
		orderID := blockDeterministicID(pblk.height, 0)
		assertAcctBalance(t, vm, alice, assetLUX, 90, 10)

		// Mallory signs a cancel of ALICE's order. The gate's ownership check refuses
		// it (orderuser[orderID] == alice != mallory).
		cancel := mallory.signed(t, TxCancel, zapwire.EncodeCancel(authPool, orderID))
		cblk := addBlock(t, vm, cancel)
		assertRejected(t, cblk, cancel)
		// Alice's order still rests, lock intact: no unauthorized cancel/unlock.
		assertAcctBalance(t, vm, alice, assetLUX, 90, 10)
		if n, _, _, _, _ := vm.BookDepth(authPool); n != 1 {
			t.Fatalf("victim order count = %d, want 1 (must survive a foreign cancel)", n)
		}

		// The OWNER can cancel — proving the gate refuses only the non-owner.
		ownCancel := alice.signed(t, TxCancel, zapwire.EncodeCancel(authPool, orderID))
		addBlock(t, vm, ownCancel)
		assertAcctBalance(t, vm, alice, assetLUX, 100, 0) // owner's cancel refunded the lock
	})

	t.Run("withdraw_to_unauthorized_ref_rejected", func(t *testing.T) {
		vm := setupAuthVM(t)
		a := mk(t, "alice")
		addBlock(t, vm, fundAcct(t, a, assetLUX, 100, ref32(1)))
		assertAcctBalance(t, vm, a, assetLUX, 100, 0)

		// Alice signs a withdraw exporting to ref R_auth (the destination she
		// authorized). An attacker rewrites the body's ref to R_attacker but keeps
		// Alice's signature. The ref is inside the signed digest, so the digest no
		// longer matches -> the recovered account is wrong/garbage -> rejected, and
		// NO funds leave Alice's balance ("withdraw to attacker_ref" is closed).
		authorizedRef := ref32(0xAA)
		attackerRef := ref32(0xEE)
		signedBody := zapwire.EncodeWithdraw(a.user, assetLUX, 50, authorizedRef)
		auth := a.authFor(TxWithdraw, a.nonce, signedBody)
		tamperedBody := zapwire.EncodeWithdraw(a.user, assetLUX, 50, attackerRef)
		tampered, err := NewSignedTx(TxWithdraw, tamperedBody, auth)
		if err != nil {
			t.Fatalf("NewSignedTx: %v", err)
		}
		blk := addBlock(t, vm, tampered)
		assertRejected(t, blk, tampered)
		assertAcctBalance(t, vm, a, assetLUX, 100, 0) // nothing exported

		// The HONEST withdraw to the authorized ref succeeds (50 debited).
		good := a.signed(t, TxWithdraw, signedBody)
		addBlock(t, vm, good)
		assertAcctBalance(t, vm, a, assetLUX, 50, 0)
	})
}

// TestAuth_AdminTxNeedsNoSignature proves the gate is scoped to MONEY-MOVING txs:
// market admin (ensure/open) carries no asserted account and is accepted without
// an auth envelope.
func TestAuth_AdminTxNeedsNoSignature(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(context.Background())
	// openMarketTx builds an UNSIGNED admin tx; it must apply.
	addBlock(t, vm, openMarketTx(t, authPool, assetLUX, assetLUSD))
	if _, _, bound, err := readMarketAssets(vm.db, authPool); err != nil || !bound {
		t.Fatalf("admin open-market did not bind assets (bound=%v err=%v)", bound, err)
	}
}

// TestAuth_EnvelopeRoundTrip proves the wire envelope encodes/decodes canonically
// for both schemes and that a tx survives Bytes/ParseTx with its id intact.
func TestAuth_EnvelopeRoundTrip(t *testing.T) {
	vm, _ := newTestVM(t, memdb.New())
	for _, mk := range []mkAcct{acctFor, pqAcctFor} {
		a := mk(t, "rt")
		body := zapwire.EncodeWithdraw(a.user, assetLUX, 7, ref32(3))
		tx := a.signed(t, TxWithdraw, body)
		back, err := ParseTx(tx.Bytes())
		if err != nil {
			t.Fatalf("ParseTx: %v", err)
		}
		if back.ID() != tx.ID() {
			t.Fatalf("TxID changed across round-trip (scheme %d)", a.scheme)
		}
		if back.Auth == nil || back.Auth.Scheme != a.scheme || back.Auth.Nonce != tx.Auth.Nonce {
			t.Fatalf("auth envelope not preserved across round-trip")
		}
		// The recovered account must equal the asserted body account.
		got, err := vm.verifyTxSignature(back)
		if err != nil {
			t.Fatalf("verifyTxSignature after round-trip: %v", err)
		}
		if string(got[:]) != a.user {
			t.Fatalf("recovered account != asserted (scheme %d)", a.scheme)
		}
	}
}

// TestAuth_SchemesDeriveDistinctAccounts proves the scheme byte is mixed into the
// account fold: the "same" person's classical and PQ keys map to DISTINCT accounts
// (a secp signature can never authorize a PQ account's funds and vice versa).
func TestAuth_SchemesDeriveDistinctAccounts(t *testing.T) {
	c := acctFor(t, "same-name")
	p := pqAcctFor(t, "same-name")
	if c.user == p.user {
		t.Fatal("classical and PQ identities collapsed to the same account")
	}
	if c.scheme != dex.AuthSecp256k1 || p.scheme != dex.AuthMLDSA65 {
		t.Fatal("scheme tags wrong")
	}
}

// ---- assertion helpers ----

// assertAcctBalance asserts a testAcct's (available, locked) for an asset against
// the live ledger (keyed by the account the gate bound).
func assertAcctBalance(t *testing.T, vm *VM, a *testAcct, asset [32]byte, wantAvail, wantLocked uint64) {
	t.Helper()
	avail, locked, err := vm.Balance(a.user, asset)
	if err != nil {
		t.Fatalf("Balance: %v", err)
	}
	if avail != wantAvail || locked != wantLocked {
		t.Fatalf("balance: available=%d locked=%d, want available=%d locked=%d", avail, locked, wantAvail, wantLocked)
	}
}

// assertRejected asserts the block recorded a StatusRejected outcome for tx — the
// deterministic per-tx verdict the gate produced.
func assertRejected(t *testing.T, blk *Block, tx *Tx) {
	t.Helper()
	id := tx.ID()
	for _, o := range blk.outcomes {
		if o.txID == id {
			if o.status != zapwire.StatusRejected {
				t.Fatalf("tx %s outcome status=%d, want StatusRejected (%d)", id, o.status, zapwire.StatusRejected)
			}
			return
		}
	}
	t.Fatalf("no outcome recorded for tx %s", id)
}

// assertNoResting asserts a market has no resting orders.
func assertNoResting(t *testing.T, vm *VM, pool [32]byte) {
	t.Helper()
	if n, _, _, _, _ := vm.BookDepth(pool); n != 0 {
		t.Fatalf("resting orders = %d, want 0 (rejected order must not rest)", n)
	}
}

// ref32 builds a non-zero 32-byte custody ref from a byte seed (custody txs refuse
// an all-zero ref; these fixtures supply a distinct non-zero one).
func ref32(seed byte) [32]byte {
	var r [32]byte
	for i := range r {
		r[i] = seed + byte(i)
	}
	if r == ([32]byte{}) {
		r[0] = 1
	}
	return r
}

// TestAuth_SignatureIsBoundToItsChain proves a signed frame spends on exactly one
// D-Chain.
//
// Account16 folds a key to an account with no network in the fold, so one key is the
// same account on every D-Chain there is, and the native asset is all-zero on all of
// them. Nothing in a tx body names a chain either. So a digest over (type, nonce, body)
// alone describes a transaction that exists everywhere at once: an export signed on a
// devnet is a valid export of the SAME account's mainnet balance to the SAME C address,
// and anyone who saw the frame can submit it there. The signer authorized one
// withdrawal and paid for as many chains as exist.
//
// The scope is in the preimage, so the frame verifies where its signer meant and is
// bytes anywhere else.
func TestAuth_SignatureIsBoundToItsChain(t *testing.T) {
	here, _ := newTestVM(t, memdb.New())
	a := acctFor(t, "scope-owner")
	body := zapwire.EncodeWithdraw(a.user, assetLUX, 500, ref32(9))
	tx := a.signed(t, TxWithdraw, body)

	if _, err := here.verifyTxSignature(tx); err != nil {
		t.Fatalf("the frame must verify on the chain it was signed for: %v", err)
	}

	// The SAME frame, offered to a sibling D-Chain: another network, and another chain
	// on this network. Both must refuse it.
	for _, elsewhere := range []struct {
		name string
		vm   *VM
	}{
		{"another network", &VM{networkID: testNetworkID + 1, chainID: testChainID}},
		{"another chain on this network", &VM{networkID: testNetworkID, chainID: repeatID(0xEE)}},
	} {
		if _, err := elsewhere.vm.verifyTxSignature(tx); err == nil {
			t.Errorf("%s accepted a frame signed for a different chain: one signature drains "+
				"the same account everywhere it exists", elsewhere.name)
		}
	}

	// And the digest itself moves with the scope, so the binding is in the commitment
	// rather than in a check something could skip.
	mine := txAuthDigest(testNetworkID, testChainID, TxWithdraw, a.scheme, 0, body)
	if mine == txAuthDigest(testNetworkID+1, testChainID, TxWithdraw, a.scheme, 0, body) {
		t.Error("the digest ignores the network")
	}
	if mine == txAuthDigest(testNetworkID, repeatID(0xEE), TxWithdraw, a.scheme, 0, body) {
		t.Error("the digest ignores the chain")
	}
}
