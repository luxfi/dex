// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/crypto/hash"
	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
)

// auth.go is the D-Chain AUTHORIZATION boundary: the single place that turns a
// money-moving tx into "the asserted account cryptographically authorized this
// exact operation". It is decomplected into three orthogonal concerns:
//
//   - WIRE      : the TxAuth envelope (scheme, nonce, signature, pubkey) carried
//                 on the dchain tx frame (tx.go), encoded canonically.
//   - DIGEST    : txAuthDigest binds the tx TYPE, the per-account NONCE, and the
//                 FROZEN zapwire body (which already carries user, asset, amount,
//                 the withdraw export ref, the cancel order id, price/size) into
//                 one keccak256 commitment — so a signature authorizes exactly
//                 one operation, on one account, at one nonce, and a withdraw can
//                 only ever export to the ref the user signed over.
//   - VERIFY    : verifyTxSignature recovers the signer via the shared primitive
//                 (dex.RecoverAuthAddress — the same ecrecover / ML-DSA-65 verify
//                 the order path uses), folds it to the 16-byte settlement
//                 account, and REQUIRES it equals the account the tx asserts.
//
// The STATEFUL policy (monotonic per-account nonce; cancel ownership) lives in
// block.execute against the consensus overlay, not here — verifyTxSignature is a
// pure function of the tx bytes so it can also pre-screen at the RPC boundary
// (handler.go) without touching state.

// Auth-layer errors. All are FAIL-CLOSED deterministic per-tx rejects on the
// consensus path: every validator decodes the same frame and reaches the same
// verdict, so a forged/replayed/unauthorized tx is a consensus-neutral no-op
// that mutates NO ledger state — never a block abort.
var (
	// ErrTxUnsigned: a money-moving tx arrived with no auth envelope. The
	// settlement path REFUSES to move value for an unauthenticated command.
	ErrTxUnsigned = errors.New("dchain: state-mutating tx is not signed")
	// ErrTxBadSignature: the signature did not recover/verify, or the recovered
	// signer's account does not equal the account the tx asserts (the wire user).
	ErrTxBadSignature = errors.New("dchain: tx signature invalid or signer is not the asserted account")
	// ErrTxBadNonce: the tx nonce is not the account's expected next nonce (a
	// replay, a reorder, or a gap) — monotonic per-account replay protection.
	ErrTxBadNonce = errors.New("dchain: tx nonce is not the account's next nonce (replay/reorder)")
	// ErrTxNotOwner: a cancel whose signer does not own the resting order it
	// targets — only the order's placer may cancel it.
	ErrTxNotOwner = errors.New("dchain: cancel signer does not own the target order")
	// ErrTxMalformedAuth: the auth envelope bytes are structurally invalid.
	ErrTxMalformedAuth = errors.New("dchain: malformed tx auth envelope")
)

// Domain-separation tags. Distinct, versioned constants keep a D-Chain tx
// signature from ever being valid as any other lux-stack signature, and bind the
// account-id derivation to its own namespace.
const (
	// txAuthDomain tags the tx-authorization digest.
	txAuthDomain = "lux.dchain.tx.auth.v1"
	// accountDomain tags the address→16-byte-account fold so a secp256k1 and an
	// ML-DSA identity for the "same" 20-byte address map to DISTINCT accounts
	// (the scheme byte is mixed in), and so the account id is not interchangeable
	// with any other 16-byte handle in the stack.
	accountDomain = "lux.dchain.account.v1"
)

// TxAuth is the authorization envelope carried on a money-moving dchain tx. It is
// orthogonal to the FROZEN zapwire body: the body says WHAT to do, the envelope
// proves WHO authorized it and at WHICH nonce.
type TxAuth struct {
	Scheme dex.AuthScheme // AuthSecp256k1 or AuthMLDSA65
	Nonce  uint64         // the signer-account's monotonic sequence number
	Sig    []byte         // 65-byte r‖s‖v (secp) or 3293-byte ML-DSA-65 signature
	PubKey []byte         // empty for secp (recovered); 1952-byte key for ML-DSA
}

// authHeaderSize is the fixed prefix of an encoded envelope: scheme[1] +
// nonce[8] + sigLen[2] + pubLen[2].
const authHeaderSize = 1 + 8 + 2 + 2

// encode serializes the envelope canonically:
//
//	scheme[1] ‖ nonce[8] ‖ sigLen[2] ‖ pubLen[2] ‖ sig ‖ pubKey
//
// Fixed field order and big-endian lengths make the encoding deterministic, so
// the tx bytes (hence the TxID) are stable across hosts.
func (a *TxAuth) encode() []byte {
	out := make([]byte, authHeaderSize+len(a.Sig)+len(a.PubKey))
	out[0] = byte(a.Scheme)
	binary.BigEndian.PutUint64(out[1:9], a.Nonce)
	binary.BigEndian.PutUint16(out[9:11], uint16(len(a.Sig)))
	binary.BigEndian.PutUint16(out[11:13], uint16(len(a.PubKey)))
	off := authHeaderSize
	off += copy(out[off:], a.Sig)
	copy(out[off:], a.PubKey)
	return out
}

// decodeTxAuth parses an encoded envelope, bounds-checking every length so a
// truncated/garbage trailer is refused rather than panicking.
func decodeTxAuth(b []byte) (*TxAuth, error) {
	if len(b) < authHeaderSize {
		return nil, fmt.Errorf("%w: header short (%d)", ErrTxMalformedAuth, len(b))
	}
	a := &TxAuth{Scheme: dex.AuthScheme(b[0]), Nonce: binary.BigEndian.Uint64(b[1:9])}
	sigLen := int(binary.BigEndian.Uint16(b[9:11]))
	pubLen := int(binary.BigEndian.Uint16(b[11:13]))
	off := authHeaderSize
	if off+sigLen+pubLen != len(b) {
		return nil, fmt.Errorf("%w: length mismatch (have %d want %d)", ErrTxMalformedAuth, len(b), off+sigLen+pubLen)
	}
	a.Sig = append([]byte(nil), b[off:off+sigLen]...)
	a.PubKey = append([]byte(nil), b[off+sigLen:off+sigLen+pubLen]...)
	return a, nil
}

// txAuthDigest is the 32-byte commitment the signature covers:
//
//	keccak256( txAuthDomain ‖ scheme[1] ‖ type[1] ‖ nonce[8] ‖ body )
//
// Binding the TYPE stops a signed place being replayed as a submit; binding the
// NONCE makes each signature single-use under the monotonic gate; binding the
// whole frozen BODY binds every value-bearing field — crucially the withdraw
// export ref[32] and the cancel order id — so a signed withdraw can only export
// to the address the user authorized, and a signed cancel can only target the
// order id the user named.
func txAuthDigest(typ TxType, scheme dex.AuthScheme, nonce uint64, body []byte) [32]byte {
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], nonce)
	return hash.ComputeKeccak256Array(
		[]byte(txAuthDomain),
		[]byte{byte(scheme)},
		[]byte{byte(typ)},
		n[:],
		body,
	)
}

// Account16 folds a verified 20-byte signer address to the D-Chain's 16-byte
// settlement identity, domain-separated and scheme-tagged:
//
//	keccak256( accountDomain ‖ scheme[1] ‖ addr[20] )[:16]
//
// This is the ONE derivation from a cryptographic identity to the wire/ledger
// account the rest of the stack keys balances by (state.go userKey16). 16 bytes
// is the birthday-safe (2^64) account axis the settlement model already commits
// to; the scheme byte makes a classical and a PQ key never silently share an
// account. The result, rendered as a string, IS the zapwire user[16] field a
// caller must put on the wire.
func Account16(scheme dex.AuthScheme, addr common.Address) [zapwire.UserSize]byte {
	d := hash.ComputeKeccak256Array([]byte(accountDomain), []byte{byte(scheme)}, addr[:])
	var a [zapwire.UserSize]byte
	copy(a[:], d[:zapwire.UserSize])
	return a
}

// txWireUser returns the account a money-moving tx ASSERTS — the user[16] field
// the FROZEN zapwire body carries. Cancel has no user field (its authorization
// is ownership of the target order, proven against the orderuser: row), so it
// returns ok=false.
func txWireUser(typ TxType, body []byte) (userKey, bool) {
	var u userKey
	switch typ {
	case TxPlace:
		copy(u[:], body[zapwire.PoolIDSize+17:zapwire.PoolIDSize+17+zapwire.UserSize])
		return u, true
	case TxSubmit:
		copy(u[:], body[zapwire.PoolIDSize+18:zapwire.PoolIDSize+18+zapwire.UserSize])
		return u, true
	case TxWithdraw:
		// A withdraw debits the caller's OWN balance, so the signer MUST be the
		// account named in the body (signer == beneficiary).
		copy(u[:], body[0:zapwire.UserSize])
		return u, true
	case TxDeposit:
		// A deposit MINTS balance for the body's beneficiary, so it is NOT authorized
		// by the beneficiary — it is authorized by the trusted deposit AUTHORITY (the
		// bridge/proxy that holds the backing C-side value). The signature still binds
		// the beneficiary via the digest (the whole body is signed), but the signer is
		// the authority, checked statefully in authorizeTx against vm.depositAuthority.
		// Returning ok=false here means verifyTxSignature does NOT require
		// signer==beneficiary; it just recovers the signer for the authority gate.
		return u, false
	default:
		return u, false
	}
}

// verifyTxSignature is the STATELESS half of the gate: it verifies the envelope
// signature over the tx digest, folds the recovered signer to the 16-byte
// account, and for the four user-asserting tx types REQUIRES that account equals
// the account the body asserts. It returns the authenticated account so the
// stateful gate (nonce, cancel ownership) can key on it.
//
// Pure function of the tx bytes: no state read, so handler.go can run it to
// reject a forged frame before it ever enters the mempool, and block.execute
// runs the identical check authoritatively at consensus time.
func verifyTxSignature(tx *Tx) (userKey, error) {
	var zero userKey
	if !tx.Type.requiresAuth() {
		return zero, nil // market admin (ensure/open) moves no user funds
	}
	if tx.Auth == nil {
		return zero, fmt.Errorf("%w: %s", ErrTxUnsigned, tx.Type)
	}
	digest := txAuthDigest(tx.Type, tx.Auth.Scheme, tx.Auth.Nonce, tx.Body)
	addr, err := dex.RecoverAuthAddress(tx.Auth.Scheme, digest, tx.Auth.Sig, tx.Auth.PubKey)
	if err != nil {
		return zero, fmt.Errorf("%w: %v", ErrTxBadSignature, err)
	}
	account := Account16(tx.Auth.Scheme, addr)
	if asserted, ok := txWireUser(tx.Type, tx.Body); ok && asserted != account {
		return zero, fmt.Errorf("%w: signer-account %x != asserted %x", ErrTxBadSignature, account[:8], asserted[:8])
	}
	return account, nil
}

// authReader is the read+write surface the stateful gate needs against the
// consensus overlay (nonce read/advance; orderuser ownership read).
type authReader interface {
	database.KeyValueReader
	database.KeyValueWriter
}

// authorizeTx is the AUTHORITATIVE gate run by block.execute for every
// money-moving tx, against the block's versiondb overlay, BEFORE any ledger
// mutation. It composes three fail-closed checks; ALL must pass to proceed:
//
//  1. SIGNATURE (stateless): verifyTxSignature — the envelope signs this exact
//     (type, nonce, body), and the recovered account equals the asserted user.
//  2. OWNERSHIP (cancel only): the signer's account owns the resting order being
//     cancelled (its orderuser: row equals the account). A cancel of an unknown
//     order is allowed through (applyCancel no-ops it; nothing moves).
//  3. NONCE (stateful): the nonce equals the account's expected-next; on success
//     the expected-next is advanced by one in the overlay, so a replay/reorder
//     of this authorization can never apply again.
//
// Return contract: (rejected, err). err != nil is an overlay fault that ABORTS
// the block (every validator faults identically — consensus-neutral). rejected
// == true is a DETERMINISTIC per-tx reject: the caller records a StatusRejected
// outcome and moves on, mutating NO ledger state and (because steps fail before
// step 3 writes) leaving the nonce unconsumed. An admin tx (no auth) returns
// (false, nil) untouched.
func (b *Block) authorizeTx(db authReader, tx *Tx) (rejected bool, err error) {
	if !tx.Type.requiresAuth() {
		return false, nil
	}
	account, verr := verifyTxSignature(tx)
	if verr != nil {
		return true, nil // forged / unsigned / wrong-signer: fail closed, no mutation
	}

	// DEPOSIT BACKING GATE (the F9 fix). A TxDeposit MINTS ledger balance, so it is
	// authorized ONLY by the configured deposit authority — the trusted bridge/proxy
	// that custodies the backing C-side value — NEVER by the crediting account
	// itself. A self-signed deposit (signer == beneficiary) is rejected here, so an
	// authenticated principal can no longer mint. Fail-closed when no authority is
	// configured: with no bridge, value enters the ledger ONLY via the trustless
	// atomic import (TxImport), which is backed by a consumed C->D object. This gate
	// is deterministic (vm.depositAuthority is a consensus config param), so every
	// validator reaches the same verdict — a consensus-neutral per-tx reject, no mint.
	if tx.Type == TxDeposit {
		if b.vm.depositAuthority == (userKey{}) || account != b.vm.depositAuthority {
			return true, nil // unbacked / unauthorized deposit: no mint, nonce not consumed
		}
	}

	// Monotonic per-account nonce. Checked (and, on a match, CONSUMED) before the
	// cancel-ownership decision so that an AUTHENTICATED order advances the
	// account's sequence even if its specific action is refused below — the same
	// nonce semantics as a reverted-but-included Ethereum tx, and what keeps a
	// signer's nonce stream total-ordered regardless of per-action outcome. A
	// forged/unsigned/wrong-signer tx never reaches here, so an attacker can never
	// burn a victim's nonce.
	expected, nerr := getNonce(db, account)
	if nerr != nil {
		return false, nerr
	}
	if tx.Auth.Nonce != expected {
		return true, nil // replay (< expected), reorder/gap (> expected): no mutation
	}

	// Cancel ownership: only the order's placer may cancel it. The owner of record
	// is the orderuser: row written when the order rested (its FULL 16-byte
	// settlement identity). A cancel targeting a non-existent order has no owner
	// row — let it through as a harmless no-op (applyCancel moves nothing).
	if tx.Type == TxCancel {
		poolID, orderID, derr := zapwire.DecodeCancel(tx.Body)
		if derr != nil {
			return true, nil // malformed cancel body: reject (nonce not yet consumed)
		}
		owner, ok, oerr := getOrderUser(db, poolID, orderID)
		if oerr != nil {
			return false, oerr
		}
		if ok && owner != account {
			// Authenticated but not the owner: consume the nonce (the order was
			// authentic) and refuse the action — no unlock, no state move.
			if err := setNonce(db, account, expected+1); err != nil {
				return false, err
			}
			return true, nil
		}
	}

	if err := setNonce(db, account, expected+1); err != nil {
		return false, err
	}
	return false, nil
}
