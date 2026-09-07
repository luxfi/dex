// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"testing"

	"github.com/luxfi/constants"
	"github.com/luxfi/ids"
)

// canonicalVMIDCB58 is the D-Chain's vmID in the encoding the node's plugin
// registry uses. The dexd binary MUST be installed under exactly this
// filename: the registry resolves a CreateChainTx's vmID to an implementation
// by looking for a file with this name in the plugin directory.
//
// Written out literally rather than computed, so a change to the vmID — for
// any reason, including an "obviously equivalent" refactor — fails here
// instead of silently producing a chain no node can start.
const canonicalVMIDCB58 = "mDVT5EWMumBp3LCqvKwuyZQeY1VXr1jvjGNAt8nL4UFiXvqXr"

// TestVMID_IsCanonicalAndStable checks the two declarations of the D-Chain's
// vmID are one value. This repo declares VMID beside the Factory; luxfi/
// constants declares DexVMID, and genesis writes THAT into the CreateChainTx.
// Nothing makes them agree. If this repo's literal moves, a node looks for a
// plugin under one name while genesis named the other, and the chain simply
// never appears — with no error, because a plugin that is not there and a
// plugin that was never asked for look identical to a directory scan.
func TestVMID_IsCanonicalAndStable(t *testing.T) {
	if VMID != constants.DexVMID {
		t.Fatalf("VMID = %s, want constants.DexVMID = %s", VMID, constants.DexVMID)
	}
	if got := VMID.String(); got != canonicalVMIDCB58 {
		t.Fatalf("VMID CB58 = %q, want %q\n"+
			"A vmID cannot change after a chain is created with it. If this is an\n"+
			"intentional pre-launch change, every declaration must move together:\n"+
			"  luxfi/constants  vm_ids.go DexVMID\n"+
			"  luxfi/dex        pkg/dchain/factory.go VMID, this test",
			got, canonicalVMIDCB58)
	}
	if VMID == ids.Empty {
		t.Fatal("VMID is the empty id")
	}
}

// TestVMID_IsANameNotADigest is the property that decides how a plugin is
// pinned, so it is worth asserting rather than remembering.
//
// The id is an ASCII name padded into 32 bytes — it is NOT a hash of the
// binary, and it must not become one. Genesis references the vmID in a
// CreateChainTx that is already signed and already in the chain, so an id
// derived from the binary would change on every rebuild and the chain it
// created would stop resolving. Identity has to be stable across versions.
//
// Which leaves integrity as a SEPARATE value: nothing here says which build of
// dexd is the right one, and the registry runs whatever sits at the filename.
// That pin belongs beside the artifact, not inside the identity.
func TestVMID_IsANameNotADigest(t *testing.T) {
	want := ids.ID{'d', 'e', 'x', 'v', 'm'}
	if VMID != want {
		t.Fatalf("VMID bytes = %v, want the ASCII name %q padded to 32 bytes", VMID[:8], "dexvm")
	}
}

// TestVMID_IsNotAnotherChains: the shims across the estate are near-copies of
// each other, and a copy that kept the wrong id would still compile, still
// pass a byte test if that were copied too, and produce two plugins racing for
// one filename.
func TestVMID_IsNotAnotherChains(t *testing.T) {
	for name, other := range map[string]ids.ID{
		"BridgeVMID":   constants.BridgeVMID,
		"MPCVMID":      constants.MPCVMID,
		"OracleVMID":   constants.OracleVMID,
		"RelayVMID":    constants.RelayVMID,
		"IdentityVMID": constants.IdentityVMID,
		"EVMID":        constants.EVMID,
	} {
		if VMID == other {
			t.Fatalf("dchain.VMID equals constants.%s (%s)", name, other)
		}
	}
}

// TestVMID_FilenameTypoIsRefused is the one property CB58 buys as a filename:
// base58 plus a four-byte checksum, so a mistyped plugin name fails to decode
// instead of resolving to a different id or to nothing at all.
func TestVMID_FilenameTypoIsRefused(t *testing.T) {
	typo := []byte(canonicalVMIDCB58)
	if typo[len(typo)-1] == 'q' {
		typo[len(typo)-1] = 'r'
	} else {
		typo[len(typo)-1] = 'q'
	}
	if _, err := ids.FromString(string(typo)); err == nil {
		t.Fatal("a one-character change in the plugin filename decoded cleanly; " +
			"the checksum is what makes a typo an error rather than a missing chain")
	}
}
