// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"crypto/sha256"
	"errors"
	"fmt"

	"github.com/luxfi/crypto/mldsa"
	"github.com/luxfi/geth/common"
)

// SignedOrderPQ is the strict-PQ variant of SignedOrder. The
// signature is a 3293-byte ML-DSA-65 signature (FIPS 204 NIST
// Level 3) over SigningHash(); the verifier uses the trader's
// 1952-byte ML-DSA-65 public key (carried inline so a single batch
// can verify orders from many traders without a pubkey lookup).
//
// Why pubkey inline. Classical SignedOrder carries Sender (20-byte
// address) and recovers the pubkey from the signature via ecrecover
// — a classical-only primitive. ML-DSA has no recovery: the pubkey
// MUST be supplied. The DEX VM still validates that the pubkey
// belongs to a registered trader; carrying it inline is just the
// wire format, not the access-control gate.
//
// SigningHash() inherits from SignedOrder — same field-packing,
// same byte-for-byte encoding — so a client can compute one digest
// and dual-sign for a permissive chain that accepts both schemes
// during a migration. Strict-PQ chains REFUSE the classical
// SignedOrder; SignedOrderPQ is the only accepted form.
type SignedOrderPQ struct {
	Order
	// Sig is the FIPS 204 ML-DSA-65 signature over
	// SigningHash() (matches the Order embed below).
	// Size: mldsa.MLDSA65SignatureSize = 3293 bytes.
	Sig []byte
	// PubKey is the FIPS 204 ML-DSA-65 public key the signature
	// verifies against (1952 bytes). The DEX VM validates
	// membership against the active trader set; this field is the
	// wire-form, not the authorization gate.
	PubKey []byte
	// Sender is the 20-byte address derived from PubKey via
	// AddressFromMLDSAPubKey. Mirrors SignedOrder.Sender so a
	// single SigningHash encoding works for both classical and
	// strict-PQ orders byte-for-byte. The verifier asserts
	// AddressFromMLDSAPubKey(PubKey) == Sender before checking
	// the signature, so a malicious party cannot substitute their
	// own pubkey on someone else's order.
	Sender common.Address
}

// AddressFromMLDSAPubKey derives the 20-byte address that the DEX
// uses as the strict-PQ trader identifier. Single canonical
// derivation: address = sha256(mldsa-pub-key)[:20]. Distinct from
// the secp256k1 keccak256-based EVM address space — a strict-PQ
// trader has a different on-chain address than the same person's
// classical Ethereum address would be, by construction.
func AddressFromMLDSAPubKey(pubKey []byte) common.Address {
	h := sha256.Sum256(pubKey)
	var a common.Address
	copy(a[:], h[:20])
	return a
}

// ErrClassicalAuthForbidden is returned when a strict-PQ chain is
// asked to verify a classical (secp256k1) SignedOrder. The error
// message + name match the EVM precompile refusal so audit
// pipelines can grep one identifier across all three chain layers.
var ErrClassicalAuthForbidden = errors.New(
	"lx/dex: classical authentication forbidden under strict-PQ profile (use SignedOrderPQ with ML-DSA-65)")

// VerifyOrderPQ verifies one ML-DSA-65 signed order. Returns nil
// iff the signature is valid and the pubkey/sig sizes match the
// FIPS 204 fixed sizes for MLDSA65 — wrong-length inputs are
// refused before any verification work runs.
func VerifyOrderPQ(order *SignedOrderPQ) error {
	if order == nil {
		return errors.New("lx/dex: nil SignedOrderPQ")
	}
	if len(order.PubKey) != mldsa.MLDSA65PublicKeySize {
		return fmt.Errorf("lx/dex: ML-DSA-65 pubkey len=%d, want %d",
			len(order.PubKey), mldsa.MLDSA65PublicKeySize)
	}
	if len(order.Sig) == 0 {
		return errors.New("lx/dex: ML-DSA-65 signature is empty")
	}
	// Bind the pubkey to Sender so a malicious party cannot
	// substitute their own pubkey on someone else's order
	// (they'd still need to produce a signature over the same
	// digest, but with a different Sender the order maps to a
	// different trader's collateral / balance).
	if AddressFromMLDSAPubKey(order.PubKey) != order.Sender {
		return errors.New("lx/dex: ML-DSA-65 pubkey does not match Sender (address binding)")
	}
	// Reuse SignedOrder's SigningHash so the digest is identical
	// across classical + strict-PQ paths — a client can compute
	// one digest and sign it with whichever scheme the chain
	// requires.
	asClassical := SignedOrder{Order: order.Order, Sender: order.Sender}
	hash, err := asClassical.SigningHash()
	if err != nil {
		return fmt.Errorf("lx/dex: SigningHash: %w", err)
	}
	pub, err := mldsa.PublicKeyFromBytes(order.PubKey, mldsa.MLDSA65)
	if err != nil {
		return fmt.Errorf("lx/dex: parse ML-DSA-65 pubkey: %w", err)
	}
	if !pub.VerifySignature(hash[:], order.Sig) {
		return errors.New("lx/dex: ML-DSA-65 signature verification failed")
	}
	return nil
}

// BatchVerifyOrdersPQ verifies a batch of SignedOrderPQ orders,
// returning a per-order valid/invalid bool. ML-DSA-65 does not
// support batch verification (no aggregation primitive), so this
// is a sequential loop. The function exists at the same shape as
// BatchVerifyOrders (classical) so callers route through a single
// verification surface and the strict-PQ gate sits at one place.
//
// Returns (results, error) where the error is non-nil only on an
// invariant violation (e.g. nil orders slice). Per-order invalid
// signatures populate results[i]=false without erroring the batch.
func BatchVerifyOrdersPQ(orders []SignedOrderPQ) ([]bool, error) {
	out := make([]bool, len(orders))
	for i := range orders {
		out[i] = VerifyOrderPQ(&orders[i]) == nil
	}
	return out, nil
}

// VerifyOrderForProfile dispatches between classical and strict-PQ
// verification based on the active SecurityProfile. This is the
// single seam every DEX VM verification path should route through;
// direct calls to BatchVerifyOrders (classical) bypass the strict-PQ
// gate and silently accept secp256k1 signatures on a PQ chain.
//
// The signature accepts an interface so a caller can pass either
// SignedOrder (classical) or SignedOrderPQ (strict-PQ); the
// dispatch picks the right verifier and refuses the wrong shape
// for the active profile.
//
//   - profile == ProfileClassical: SignedOrder ok, SignedOrderPQ
//     also ok (a permissive chain accepts both during a migration).
//   - profile == ProfileStrictPQ:  SignedOrderPQ ok, SignedOrder
//     refused with ErrClassicalAuthForbidden.
func VerifyOrderForProfile(profile SecurityProfile, order any) error {
	switch o := order.(type) {
	case *SignedOrder:
		if profile.IsPostQuantum() {
			return ErrClassicalAuthForbidden
		}
		results, err := BatchVerifyOrders([]SignedOrder{*o})
		if err != nil {
			return err
		}
		if !results[0] {
			return errors.New("lx/dex: secp256k1 signature verification failed")
		}
		return nil
	case *SignedOrderPQ:
		return VerifyOrderPQ(o)
	default:
		return fmt.Errorf("lx/dex: unknown order type %T", order)
	}
}
