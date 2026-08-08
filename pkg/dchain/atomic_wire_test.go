// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/hex"
	"testing"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

// atomic_wire_test.go is the D-side half of the cross-repo funding-wire contract.
// precompile/dex/native_seam_parity_test.go pins the IDENTICAL strings. The two repos
// cannot import each other, so these golden vectors ARE the contract: a one-sided
// change to the layout or the id derivation fails HERE, at test time, rather than in
// consensus — where it would either revert every crossing or, worse, be a width both
// sides accept and read differently.
//
// HOW THE GOLDEN WAS PRODUCED (reproducible). encodeClaim on:
//
//	beneficiary = 0x112233445566778899aabbccddeeff0102030405   (20 bytes)
//	asset       = 0xa0a1a2a3...bebf                            (32 bytes, ascending from 0xA0)
//	amount      = 0x0102030405060708
const claimGoldenHex = "112233445566778899aabbccddeeff0102030405" + // beneficiary (20)
	"a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf" + // asset (32)
	"0102030405060708" // amount (8, big-endian)

// claimIDGoldenHex pins DeriveClaimID over source=0x01…, dest=0x02…, sourceTx=0x03…,
// index=7. A drift here does not corrupt value, it LOSES it: one side writes the claim
// under one key and the other looks for it under another.
const claimIDGoldenHex = "0a5e96264cb3e4649c30aba257749a203119d0363677b1be3ead32ebf989f430"

func goldenBeneficiary() common.Address {
	return common.HexToAddress("0x112233445566778899aabbccddeeff0102030405")
}

func goldenAsset() [32]byte {
	var a [32]byte
	for i := range a {
		a[i] = byte(0xA0 + i)
	}
	return a
}

func repeatID(b byte) ids.ID {
	var id ids.ID
	for i := range id {
		id[i] = b
	}
	return id
}

func TestClaimWire_GoldenMatchesPrecompile(t *testing.T) {
	want, err := hex.DecodeString(claimGoldenHex)
	if err != nil {
		t.Fatalf("bad golden hex: %v", err)
	}
	if len(want) != claimSize {
		t.Fatalf("golden width %d != claimSize %d — the wire changed on ONE side only "+
			"(consensus break). Update BOTH repos in lockstep.", len(want), claimSize)
	}
	got := encodeClaim(goldenBeneficiary(), goldenAsset(), 0x0102030405060708)
	if hex.EncodeToString(got) != claimGoldenHex {
		t.Fatalf("funding wire DIVERGED from the precompile golden:\n got=%s\nwant=%s",
			hex.EncodeToString(got), claimGoldenHex)
	}
}

func TestClaimID_GoldenMatchesPrecompile(t *testing.T) {
	got := DeriveClaimID(repeatID(0x01), repeatID(0x02), repeatID(0x03), 7)
	if hex.EncodeToString(got[:]) != claimIDGoldenHex {
		t.Fatalf("claim id DIVERGED from the precompile golden:\n got=%s\nwant=%s",
			hex.EncodeToString(got[:]), claimIDGoldenHex)
	}
}

func TestClaimWire_RoundTrip(t *testing.T) {
	beneficiary, asset := goldenBeneficiary(), goldenAsset()
	const amount uint64 = 9_000_000
	gotB, gotA, gotAmt, ok := decodeClaim(encodeClaim(beneficiary, asset, amount))
	if !ok || gotB != beneficiary || gotA != asset || gotAmt != amount {
		t.Fatalf("round-trip mismatch: ok=%v beneficiary=%x asset=%x amount=%d", ok, gotB, gotA, gotAmt)
	}
}

// TestClaimWire_WrongWidthRejected pins that a non-canonical width never decodes into
// a credit — including the widths of the OLD braided object, 69 (value + lane byte +
// spent witness) and 118 (that, plus a market/side/limit/size tail).
func TestClaimWire_WrongWidthRejected(t *testing.T) {
	for _, n := range []int{0, 59, 61, 69, 80, 118} {
		if _, _, _, ok := decodeClaim(make([]byte, n)); ok {
			t.Fatalf("a %d-byte object must NOT decode (only the canonical %d is valid)", n, claimSize)
		}
	}
}

// TestBodyWidths pins the two transaction body widths against the wire they carry, so
// a claim-size change cannot silently leave a body sized for the old object.
func TestBodyWidths(t *testing.T) {
	if importBodySize != 32+claimSize {
		t.Fatalf("import body %d != claimID(32) + claim(%d)", importBodySize, claimSize)
	}
	if exportBodySize != 16+32+8+20 {
		t.Fatalf("export body %d != owner(16)+asset(32)+amount(8)+beneficiary(20)", exportBodySize)
	}
	claimID := repeatID(0x0A)
	object := encodeClaim(goldenBeneficiary(), goldenAsset(), 42)
	gotID, gotObj, err := decodeImportBody(EncodeImportBody(claimID, object))
	if err != nil || gotID != claimID || string(gotObj) != string(object) {
		t.Fatalf("import body round trip: id=%s err=%v", gotID, err)
	}

	var owner userKey
	copy(owner[:], "owner-16-bytes--")
	gotOwner, gotAsset, gotAmt, gotBen, err := decodeExportBody(
		EncodeExportBody(owner, goldenAsset(), 4242, goldenBeneficiary()))
	if err != nil || gotOwner != owner || gotAsset != goldenAsset() ||
		gotAmt != 4242 || gotBen != goldenBeneficiary() {
		t.Fatalf("export body round trip: err=%v owner=%x amount=%d", err, gotOwner, gotAmt)
	}
}

// TestExportIsSigned pins that an export — which removes value from a real account and
// names where it lands — requires a signature, and that the signature is bound to the
// OWNER the body asserts. Without both, anyone could drain any account to an address
// of their choosing. An import needs neither: a claim can only credit the account it
// already names, which is what makes delivery permissionless.
func TestExportIsSigned(t *testing.T) {
	if !TxExport.requiresAuth() {
		t.Fatal("TxExport must require a signature: it debits a real account and names " +
			"the address the value lands on")
	}
	if TxImport.requiresAuth() {
		t.Fatal("TxImport must NOT require a signature: a claim names its own beneficiary, " +
			"so delivering it can only pay that account — requiring one would let a " +
			"refusing relayer strand a depositor")
	}
	var owner userKey
	copy(owner[:], "owner-16-bytes--")
	body := EncodeExportBody(owner, goldenAsset(), 100, goldenBeneficiary())
	asserted, ok := txWireUser(TxExport, body)
	if !ok || asserted != owner {
		t.Fatalf("an export must assert its owner on the wire so the signer can be bound "+
			"to it: ok=%v asserted=%x", ok, asserted)
	}
}
