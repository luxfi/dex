// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"encoding/hex"
	"testing"

	"github.com/luxfi/ids"
)

// assetid_test.go pins the CANONICAL asset identity. The decisive property is CROSS-HOME
// PARITY: dex.DeriveAssetID must produce the byte-identical id that
// chains/dexvm/registry.DeriveAssetID produces for the same inputs, because the value
// path (precompile/dex, over dex) and the admission authority (the registry) must
// name the SAME asset by the SAME 32-byte id — a registered AssetID and a swap-derived
// AssetID are compared directly. The expected vectors below were emitted by the chains
// registry's own DeriveAssetID; if either implementation's preimage discipline drifts,
// this test fails (a real consensus-identity divergence, caught here, never in prod).

// chainAllOnes is the fixed 32-byte source chain id (every byte 0x11) the parity
// vectors were generated against.
func chainAllOnes() ids.ID {
	var c ids.ID
	for i := range c {
		c[i] = 0x11
	}
	return c
}

// AssetIDGoldenVectors is the CANONICAL cross-home golden KAT (MED-1): the fixed
// (networkID, chainID, kind, ref) -> 32-byte AssetID mapping that BOTH dex and
// chains/dexvm/registry MUST reproduce byte-for-byte. The byte-identity these vectors
// pin is the foundation of the resolve<->register equivalence the value path depends on:
// a swap-derived AssetID (dex, via the precompile resolver) and a registered AssetID
// (the chains registry) name the SAME asset by the SAME id, by construction. The mirror
// test in chains/dexvm/registry asserts these EXACT bytes, so a drift in either home's
// preimage discipline fails CI in both places, never silently in prod.
//
// All vectors use networkID=2 and the all-0x11 source chain id (chainAllOnes). Generated
// by the chains registry's own DeriveAssetID; do NOT edit a vector to make a test pass —
// a changed id is a real consensus-identity fork and must be investigated, not papered over.
var AssetIDGoldenVectors = []struct {
	Name string
	Kind AssetKind
	Ref  []byte
	Want string // hex of the 32-byte AssetID
}{
	{"ERC20/addr..01", AssetKindERC20, erc20RefEnding(0x01), "dc392784b1b0764f885a2b24786850dae0a221fe7eaa218065ac1497473fa868"},
	{"EVM_NATIVE/marker", AssetKindEVMNative, EVMNativeMarker, "5941ecf871f909bac11b9b3d34fff1d05c7a0182f3a1c5b905ee6059dbb6dc72"},
	{"UTXO/asset..07", AssetKindUTXO, utxoRefEnding(0x07), "5cd895b8a577437bdf39e921902cbf06c29a11ae4f1776b369584c46fc0d647d"},
}

// erc20RefEnding builds a 20-byte ERC-20 address whose last byte is b (rest zero).
func erc20RefEnding(b byte) []byte { r := make([]byte, 20); r[19] = b; return r }

// utxoRefEnding builds a 32-byte UTXO assetID whose last byte is b (rest zero).
func utxoRefEnding(b byte) []byte { r := make([]byte, 32); r[31] = b; return r }

func TestAssetID_CrossHomeParity_MatchesChainsRegistry(t *testing.T) {
	chain := chainAllOnes()
	for _, v := range AssetIDGoldenVectors {
		got, err := DeriveAssetID(2, chain, v.Kind, v.Ref)
		if err != nil {
			t.Fatalf("%s: derive: %v", v.Name, err)
		}
		if h := hex.EncodeToString(got[:]); h != v.Want {
			t.Fatalf("%s: AssetID diverged from chains/dexvm/registry golden KAT:\n got  %s\n want %s", v.Name, h, v.Want)
		}
	}
}

func TestAssetID_DomainSeparation_KindAndRef(t *testing.T) {
	chain := ids.GenerateTestID()
	ref := make([]byte, 20)
	ref[19] = 0x42

	// Same (network, chain, ref) under ERC20 vs (a hypothetical) — the kind byte must
	// domain-separate, and CanonicalRefFor forbids reusing the 20-byte ref as the native
	// marker (it is non-zero), so the kinds occupy disjoint spaces.
	erc20ID, err := DeriveAssetID(2, chain, AssetKindERC20, ref)
	if err != nil {
		t.Fatalf("erc20: %v", err)
	}
	nativeID, err := DeriveAssetID(2, chain, AssetKindEVMNative, EVMNativeMarker)
	if err != nil {
		t.Fatalf("native: %v", err)
	}
	if erc20ID == nativeID {
		t.Fatal("ERC20 and EVM_NATIVE asset ids must never collide (domain separation)")
	}

	// A zero-address ERC-20 ref is refused (the zero address is the native marker, never
	// a token) — the shape rule lives in exactly one place (CanonicalRefFor).
	if _, err := DeriveAssetID(2, chain, AssetKindERC20, make([]byte, 20)); err == nil {
		t.Fatal("an ERC20 ref equal to the zero address must be refused")
	}
	// An empty source chain is refused.
	if _, err := DeriveAssetID(2, ids.Empty, AssetKindERC20, ref); err == nil {
		t.Fatal("an empty source chain id must be refused")
	}
}
