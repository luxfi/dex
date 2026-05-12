// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"crypto/rand"
	"errors"
	"testing"

	"github.com/luxfi/crypto/mldsa"
)

// signingMaterial produces a fresh ML-DSA-65 keypair, the derived
// trader address, and an Order ready for SignedOrderPQ tests.
func signingMaterial(t *testing.T) (*mldsa.PrivateKey, []byte, Order) {
	t.Helper()
	priv, err := mldsa.GenerateKey(rand.Reader, mldsa.MLDSA65)
	if err != nil {
		t.Fatalf("mldsa keygen: %v", err)
	}
	pubBytes := priv.PublicKey.Bytes()
	order := Order{
		ID:       42,
		Symbol:   "BTC-USD",
		Side:     Buy,
		Type:     Limit,
		Price:    50000,
		Size:     1,
		ClientID: "test-1",
	}
	return priv, pubBytes, order
}

func TestVerifyOrderPQ_HappyPath(t *testing.T) {
	priv, pubBytes, order := signingMaterial(t)
	sender := AddressFromMLDSAPubKey(pubBytes)

	asClassical := SignedOrder{Order: order, Sender: sender}
	hash, err := asClassical.SigningHash()
	if err != nil {
		t.Fatalf("SigningHash: %v", err)
	}
	sig, err := priv.Sign(rand.Reader, hash[:], nil)
	if err != nil {
		t.Fatalf("mldsa sign: %v", err)
	}
	pqOrder := SignedOrderPQ{
		Order:  order,
		Sig:    sig,
		PubKey: pubBytes,
		Sender: sender,
	}
	if err := VerifyOrderPQ(&pqOrder); err != nil {
		t.Fatalf("VerifyOrderPQ on valid order: %v", err)
	}
}

// TestVerifyOrderPQ_AddressMismatch covers the case where an
// attacker substitutes their own pubkey on someone else's order.
// The Sender binding refuses before the signature is even checked.
func TestVerifyOrderPQ_AddressMismatch(t *testing.T) {
	priv, _, order := signingMaterial(t)
	// Different pubkey, different derived address.
	_, attackerPub, _ := signingMaterial(t)
	hash, _ := (&SignedOrder{Order: order, Sender: AddressFromMLDSAPubKey(attackerPub)}).SigningHash()
	sig, _ := priv.Sign(rand.Reader, hash[:], nil) // signed over wrong sender

	pqOrder := SignedOrderPQ{
		Order:  order,
		Sig:    sig,
		PubKey: attackerPub,                                 // attacker substitutes their pubkey
		Sender: AddressFromMLDSAPubKey(priv.PublicKey.Bytes()), // but keeps original sender
	}
	if err := VerifyOrderPQ(&pqOrder); err == nil {
		t.Fatal("VerifyOrderPQ accepted address-mismatched order")
	}
}

// TestVerifyOrderPQ_WrongPubkeySize refuses inputs whose pubkey
// length doesn't match FIPS 204 MLDSA-65 (1952 bytes). The check
// runs BEFORE any verification work so an attacker can't trigger
// expensive crypto with malformed payloads.
func TestVerifyOrderPQ_WrongPubkeySize(t *testing.T) {
	_, _, order := signingMaterial(t)
	pqOrder := SignedOrderPQ{
		Order:  order,
		Sig:    []byte{0x00},
		PubKey: []byte{0xde, 0xad}, // 2 bytes != 1952
		Sender: AddressFromMLDSAPubKey([]byte{0xde, 0xad}),
	}
	if err := VerifyOrderPQ(&pqOrder); err == nil {
		t.Fatal("VerifyOrderPQ accepted wrong-size pubkey")
	}
}

func TestVerifyOrderForProfile_StrictPQRefusesClassical(t *testing.T) {
	order := SignedOrder{Order: Order{ID: 1}, Sig: [65]byte{}}
	err := VerifyOrderForProfile(ProfileStrictPQ, &order)
	if !errors.Is(err, ErrClassicalAuthForbidden) {
		t.Fatalf("VerifyOrderForProfile(StrictPQ, *SignedOrder) = %v, want ErrClassicalAuthForbidden", err)
	}
}

func TestVerifyOrderForProfile_StrictPQAcceptsPQ(t *testing.T) {
	priv, pubBytes, order := signingMaterial(t)
	sender := AddressFromMLDSAPubKey(pubBytes)
	hash, _ := (&SignedOrder{Order: order, Sender: sender}).SigningHash()
	sig, _ := priv.Sign(rand.Reader, hash[:], nil)
	pqOrder := SignedOrderPQ{Order: order, Sig: sig, PubKey: pubBytes, Sender: sender}
	if err := VerifyOrderForProfile(ProfileStrictPQ, &pqOrder); err != nil {
		t.Fatalf("VerifyOrderForProfile(StrictPQ, *SignedOrderPQ): %v", err)
	}
}

func TestSecurityProfile_String(t *testing.T) {
	if ProfileClassical.String() != "classical" {
		t.Errorf("ProfileClassical.String() = %q", ProfileClassical.String())
	}
	if ProfileStrictPQ.String() != "strict-pq" {
		t.Errorf("ProfileStrictPQ.String() = %q", ProfileStrictPQ.String())
	}
	if !ProfileStrictPQ.IsPostQuantum() {
		t.Error("ProfileStrictPQ.IsPostQuantum() = false")
	}
	if ProfileClassical.IsPostQuantum() {
		t.Error("ProfileClassical.IsPostQuantum() = true")
	}
	if ProfileFromPQFlag(true) != ProfileStrictPQ {
		t.Error("ProfileFromPQFlag(true) != StrictPQ")
	}
	if ProfileFromPQFlag(false) != ProfileClassical {
		t.Error("ProfileFromPQFlag(false) != Classical")
	}
}
