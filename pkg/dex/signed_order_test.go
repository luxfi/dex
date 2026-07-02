// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"crypto/ecdsa"
	"fmt"
	"testing"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// makeSignedOrder is a test helper that builds a SignedOrder with a fresh
// secp256k1 keypair, fills the public Order fields with deterministic-ish
// values from id, signs SigningHash() with the freshly generated key, and
// stamps Sender with the corresponding address.
func makeSignedOrder(t testing.TB, id uint64) (SignedOrder, *ecdsa.PrivateKey) {
	t.Helper()
	key, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	// crypto.PubkeyToAddress returns crypto/common.Address (a sibling type
	// to geth/common.Address). The dex package universally uses
	// geth/common.Address, so we build it directly from the pubkey bytes:
	// addr = last 20 bytes of keccak256(uncompressed_pubkey[1:]).
	pub := crypto.FromECDSAPub(&key.PublicKey)
	var addr common.Address
	copy(addr[:], crypto.Keccak256(pub[1:])[12:])

	so := SignedOrder{
		Order: Order{
			ID:       id,
			Symbol:   "BTC-USD",
			Type:     Limit,
			Side:     Buy,
			Price:    100 + float64(id%100)/10,
			Size:     1 + float64(id%5),
			ClientID: fmt.Sprintf("c-%d", id),
		},
		Sender: addr,
	}
	hash, err := so.SigningHash()
	if err != nil {
		t.Fatalf("signing hash: %v", err)
	}
	sig, err := crypto.Sign(hash[:], key)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	if len(sig) != 65 {
		t.Fatalf("sig len = %d", len(sig))
	}
	copy(so.Sig[:], sig)
	return so, key
}

// TestBatchVerifyOrders_AllValid: 64 random orders, each signed by its own
// fresh key, all 64 must verify true.
func TestBatchVerifyOrders_AllValid(t *testing.T) {
	const N = 64
	orders := make([]SignedOrder, N)
	for i := 0; i < N; i++ {
		orders[i], _ = makeSignedOrder(t, uint64(i+1))
	}

	got, err := BatchVerifyOrders(orders)
	if err != nil {
		t.Fatalf("BatchVerifyOrders: %v", err)
	}
	if len(got) != N {
		t.Fatalf("len(got) = %d, want %d", len(got), N)
	}
	for i, ok := range got {
		if !ok {
			t.Errorf("orders[%d] expected true, got false", i)
		}
	}
}

// TestBatchVerifyOrders_OneBad: one order in the middle has its r-component
// flipped. That index must come back false; the rest still true.
func TestBatchVerifyOrders_OneBad(t *testing.T) {
	const N = 64
	const bad = 17
	orders := make([]SignedOrder, N)
	for i := 0; i < N; i++ {
		orders[i], _ = makeSignedOrder(t, uint64(i+1))
	}
	// Flip a high bit in r — guaranteed to break signature recovery.
	orders[bad].Sig[0] ^= 0x80

	got, err := BatchVerifyOrders(orders)
	if err != nil {
		t.Fatalf("BatchVerifyOrders: %v", err)
	}
	for i, ok := range got {
		want := i != bad
		if ok != want {
			t.Errorf("orders[%d] = %v, want %v", i, ok, want)
		}
	}
}

// TestBatchVerifyOrders_Empty: zero-length input must round-trip without
// touching the C kernel.
func TestBatchVerifyOrders_Empty(t *testing.T) {
	got, err := BatchVerifyOrders(nil)
	if err != nil {
		t.Fatalf("BatchVerifyOrders(nil): %v", err)
	}
	if got != nil {
		t.Fatalf("got = %v, want nil", got)
	}
}
