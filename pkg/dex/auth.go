// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"fmt"

	"github.com/luxfi/crypto"
	"github.com/luxfi/crypto/mldsa"
	"github.com/luxfi/geth/common"
)

// auth.go decomplects the SIGNATURE-VERIFICATION PRIMITIVE from the order
// encoding it was first written for. SignedOrder.SigningHash (signed_order.go)
// fixes WHAT bytes an order signs; the recover-and-bind step below is HOW any
// 32-byte digest is authenticated — the exact ecrecover (secp256k1) and
// ML-DSA-65 verify the order path already uses (signed_order_cpu.go /
// signed_order_pq.go), with the order-specific field packing removed.
//
// One verification surface, two callers: the GPU-batched order ingest
// (BatchVerifyOrders / VerifyOrderForMode) keeps its order-typed wrapper, and
// the D-Chain custody settlement path (pkg/dchain) authenticates its tx digests
// through RecoverAuthAddress here — neither reinvents ecrecover or the ML-DSA
// pubkey→address binding.

// AuthScheme is the signature scheme authenticating a digest.
type AuthScheme uint8

const (
	// AuthSecp256k1 is the classical Ethereum scheme: a 65-byte r‖s‖v signature
	// over a keccak256 digest, signer recovered via ecrecover. v ∈ {0,1}.
	AuthSecp256k1 AuthScheme = 0
	// AuthMLDSA65 is FIPS 204 ML-DSA-65 (NIST Level 3, post-quantum): a
	// 3293-byte signature verified against an inline 1952-byte public key,
	// signer = AddressFromMLDSAPubKey(pubKey).
	AuthMLDSA65 AuthScheme = 1
)

// ErrAuthInvalidSignature is the single fail-closed verdict for any signature
// that does not verify (bad recovery, wrong key, verify=false). Callers compare
// the returned signer to an asserted identity; an error here means "no valid
// signer", never "trust the recovered value anyway".
var ErrAuthInvalidSignature = errors.New("lx/auth: signature verification failed")

// RecoverAuthAddress verifies sig over the 32-byte digest under scheme and
// returns the authenticated 20-byte signer address. It FAILS CLOSED: a malformed
// length, a non-canonical recovery id, an unparseable pubkey, or a verify=false
// all return an error and a zero address — there is no path that returns a
// non-error address without a cryptographically valid signature.
//
//   - AuthSecp256k1: ecrecover(digest, sig[65]) → uncompressed pubkey →
//     address = keccak256(pub[1:])[12:]. pubKey is ignored (recovered from sig).
//   - AuthMLDSA65: VerifySignature(digest, sig) under pubKey; on success the
//     address is AddressFromMLDSAPubKey(pubKey) (the same sha256-based
//     derivation the strict-PQ order path uses), so the returned identity is
//     bound to the verifying key.
func RecoverAuthAddress(scheme AuthScheme, digest [32]byte, sig, pubKey []byte) (common.Address, error) {
	switch scheme {
	case AuthSecp256k1:
		if len(sig) != 65 {
			return common.Address{}, fmt.Errorf("lx/auth: secp256k1 signature len=%d, want 65", len(sig))
		}
		// Reject a non-canonical recovery id up front (ecrecover only accepts
		// {0,1}); a v outside that range is a malformed signature, not a verify
		// failure to mask.
		if sig[64] > 1 {
			return common.Address{}, fmt.Errorf("lx/auth: secp256k1 recovery id v=%d, want 0 or 1", sig[64])
		}
		pub, err := crypto.Ecrecover(digest[:], sig)
		if err != nil || len(pub) != 65 {
			return common.Address{}, ErrAuthInvalidSignature
		}
		// pub is the uncompressed pubkey (0x04 ‖ X ‖ Y); the address is the low
		// 20 bytes of keccak256(X ‖ Y) — identical to verifyOrdersCPU.
		return common.BytesToAddress(crypto.Keccak256(pub[1:])[12:]), nil

	case AuthMLDSA65:
		if len(pubKey) != mldsa.MLDSA65PublicKeySize {
			return common.Address{}, fmt.Errorf("lx/auth: ML-DSA-65 pubkey len=%d, want %d", len(pubKey), mldsa.MLDSA65PublicKeySize)
		}
		if len(sig) != mldsa.MLDSA65SignatureSize {
			return common.Address{}, fmt.Errorf("lx/auth: ML-DSA-65 signature len=%d, want %d", len(sig), mldsa.MLDSA65SignatureSize)
		}
		pub, err := mldsa.PublicKeyFromBytes(pubKey, mldsa.MLDSA65)
		if err != nil {
			return common.Address{}, fmt.Errorf("lx/auth: parse ML-DSA-65 pubkey: %w", err)
		}
		if !pub.VerifySignature(digest[:], sig) {
			return common.Address{}, ErrAuthInvalidSignature
		}
		// The address IS a function of the verifying key, so it cannot be forged
		// independently of the signature that just verified under that key.
		return AddressFromMLDSAPubKey(pubKey), nil

	default:
		return common.Address{}, fmt.Errorf("lx/auth: unknown auth scheme %d", scheme)
	}
}
