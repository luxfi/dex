// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package main

import (
	"encoding/hex"
	"testing"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// devnetCChain is the canonical devnet C-Chain blockchain id the seeder derives asset
// ids under. It is the same id the live devnet validators serve.
const devnetCChain = "Bm8X7THQtS2txLrQWTDXN8a4JDuhP4KGybUE754LLSiVFLQ7v"

// Golden AssetID vectors — pinned bytes the dex identity primitive MUST produce for
// fixed (networkID, C-chainID, kind, ref) tuples. These are NOT recomputed from the
// same call under test; they are CONSTANTS frozen here, so a drift in DeriveAssetID's
// fold (domain tag, length-prefix discipline, field order) breaks this test. The C-Chain
// id below decodes to 186f101d…91e57b71 (32 bytes).
const (
	// goldenNativeID3 = DeriveAssetID(3, devnetCChain, EVM_NATIVE, EVMNativeMarker).
	goldenNativeID3 = "b3e7122cf728552ca6e7336e95ade83c2f48711b7ac963cebe549afeaf6a73f4"
	// goldenERC20ID3 = DeriveAssetID(3, devnetCChain, ERC20, 0x00..01) (20-byte address,
	// last byte 0x01).
	goldenERC20ID3 = "c7e26133e8a0c2201a65341b24b98446d6a5bd4dfa369a34682bfa6da2b4455b"
)

func mustCChain(t *testing.T) ids.ID {
	t.Helper()
	cid, err := ids.FromString(devnetCChain)
	if err != nil {
		t.Fatalf("parse devnet C-Chain id: %v", err)
	}
	return cid
}

func erc20Addr(lastByte byte) []byte {
	a := make([]byte, 20)
	a[19] = lastByte
	return a
}

// TestGoldenAssetIDs is the GOLDEN VECTOR proof: DeriveAssetID produces the exact,
// frozen 32-byte ids for known tuples. This pins the consensus-native identity the
// seeder uses to the chain's own derivation — if either drifts, a seeded market would
// bind a different id than the chain admits, and this test fails first.
func TestGoldenAssetIDs(t *testing.T) {
	cid := mustCChain(t)

	natID, err := dex.DeriveAssetID(3, cid, dex.AssetKindEVMNative, dex.EVMNativeMarker)
	if err != nil {
		t.Fatalf("derive native id: %v", err)
	}
	if got := hex.EncodeToString(natID[:]); got != goldenNativeID3 {
		t.Fatalf("native(3) AssetID drift:\n got  %s\n want %s", got, goldenNativeID3)
	}

	ercID, err := dex.DeriveAssetID(3, cid, dex.AssetKindERC20, erc20Addr(0x01))
	if err != nil {
		t.Fatalf("derive erc20 id: %v", err)
	}
	if got := hex.EncodeToString(ercID[:]); got != goldenERC20ID3 {
		t.Fatalf("erc20(3,..01) AssetID drift:\n got  %s\n want %s", got, goldenERC20ID3)
	}

	// The two ids MUST differ (kind domain-separation): native and an ERC-20 are never
	// the same asset even by accident.
	if natID == ercID {
		t.Fatal("native and ERC-20 ids collided — kind domain separation broken")
	}
}

// TestResolveID_MatchesGolden proves the seeder's assetSpec.resolveID delegates to the
// SAME primitive: a native spec and an ERC-20 spec resolve to the golden ids.
func TestResolveID_MatchesGolden(t *testing.T) {
	cid := mustCChain(t)

	natID, err := nativeSpec().resolveID(3, cid)
	if err != nil {
		t.Fatalf("native resolveID: %v", err)
	}
	if got := hex.EncodeToString(natID[:]); got != goldenNativeID3 {
		t.Fatalf("nativeSpec.resolveID drift:\n got  %s\n want %s", got, goldenNativeID3)
	}

	ercID, err := erc20Spec(erc20Addr(0x01)).resolveID(3, cid)
	if err != nil {
		t.Fatalf("erc20 resolveID: %v", err)
	}
	if got := hex.EncodeToString(ercID[:]); got != goldenERC20ID3 {
		t.Fatalf("erc20Spec.resolveID drift:\n got  %s\n want %s", got, goldenERC20ID3)
	}
}

// TestResolveID_RejectsSynthetic proves the seeder REFUSES synthetic / malformed asset
// refs at derivation time — the property that makes dexseed safe against a shared
// cluster. None of these can ever become a poolID or a bound asset.
func TestResolveID_RejectsSynthetic(t *testing.T) {
	cid := mustCChain(t)

	cases := []struct {
		name string
		spec assetSpec
	}{
		{
			// THE BRICKING FORM: an ascii-of-symbol "address" — "LUX" left in a byte
			// slice that is NOT a 20-byte token address.
			name: "ascii-ticker-LUX (not 20 bytes)",
			spec: assetSpec{Kind: dex.AssetKindERC20, Ref: []byte("LUX")},
		},
		{
			// An ascii ticker right-shaped to 20 bytes is still NOT a real token, but the
			// derivation can only refuse it on shape; the CHAIN's reality gate (no code)
			// refuses the rest. We assert the seeder refuses the clearly-malformed shapes
			// here and rely on the chain's EXTCODESIZE gate for shaped-but-fake. The
			// all-zero ERC-20 address (== the native marker) MUST be refused locally.
			name: "ERC20 zero address (== native marker, not a token)",
			spec: assetSpec{Kind: dex.AssetKindERC20, Ref: make([]byte, 20)},
		},
		{
			name: "ERC20 wrong length (33 bytes)",
			spec: assetSpec{Kind: dex.AssetKindERC20, Ref: make([]byte, 33)},
		},
		{
			name: "invalid kind (0)",
			spec: assetSpec{Kind: dex.AssetKindInvalid, Ref: erc20Addr(0x01)},
		},
		{
			name: "EVM_NATIVE with a non-marker ref",
			spec: assetSpec{Kind: dex.AssetKindEVMNative, Ref: erc20Addr(0x01)},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if id, err := tc.spec.resolveID(3, cid); err == nil {
				t.Fatalf("resolveID MUST reject %s, but it derived id %x", tc.name, id)
			}
		})
	}
}

// TestPoolIDDeterministicAndNonAscii proves the poolID derivation is (a) deterministic
// and (b) structurally NOT an ascii-of-symbol id (the removed synthetic-seeder form).
func TestPoolIDDeterministicAndNonAscii(t *testing.T) {
	cid := mustCChain(t)
	baseID, err := nativeSpec().resolveID(3, cid)
	if err != nil {
		t.Fatalf("native resolveID: %v", err)
	}
	quoteID, err := erc20Spec(erc20Addr(0x01)).resolveID(3, cid)
	if err != nil {
		t.Fatalf("erc20 resolveID: %v", err)
	}

	p1 := poolIDFor(baseID, quoteID)
	p2 := poolIDFor(baseID, quoteID)
	if p1 != p2 {
		t.Fatalf("poolIDFor is not deterministic: %x != %x", p1, p2)
	}

	// Order matters: (base,quote) and (quote,base) are distinct markets, so distinct ids.
	if pSwapped := poolIDFor(quoteID, baseID); pSwapped == p1 {
		t.Fatal("poolIDFor(base,quote) must differ from poolIDFor(quote,base)")
	}

	// NOT ascii-of-symbol: the removed synthetic-seeder form copied a ticker into the high
	// bytes and set the last byte to 0xD0. Our derivation must match neither the
	// "LUX/<ERC20>" ascii prefix nor the 0xD0 suffix sentinel.
	ascii := asciiPoolID("LUX/ERC20")
	if p1 == ascii {
		t.Fatal("poolID collided with an ascii-of-symbol poolID")
	}
	if p1[31] == 0xD0 {
		t.Fatal("poolID ends in the 0xD0 synthetic sentinel — looks ascii-shaped")
	}
	// No leading run of printable-ascii ticker bytes (the ascii form starts with the
	// symbol). A real hash is uniformly random; assert the first 8 bytes are not all in
	// the printable-ascii band an ascii ticker would occupy.
	allPrintable := true
	for _, b := range p1[:8] {
		if b < 0x20 || b > 0x7e {
			allPrintable = false
			break
		}
	}
	if allPrintable {
		t.Fatalf("poolID first 8 bytes are all printable ascii (%q) — looks like a synthetic ticker id", p1[:8])
	}
}

// asciiPoolID reproduces the removed synthetic-seeder poolID form — the SYNTHETIC shape
// dexseed must never produce (kept here only as the negative oracle for the test above).
func asciiPoolID(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}

// TestRealMarketFrames proves the frozen frames are built correctly and bind the
// canonical real ids: the OpenMarket frame is exactly 96 bytes, its base/quote slots
// equal the derived ids, the Deposit frame is 88 bytes binding the base id, and the
// Place frame is a 65-byte resting SELL by the maker at the given price/size.
func TestRealMarketFrames(t *testing.T) {
	cid := mustCChain(t)
	const (
		net      uint32 = 3
		maker           = "maker"
		askPrice        = 1.25
		askSize         = 7.0
		depAmt   uint64 = 9
	)
	var depRef [32]byte
	depRef[0] = 0xAB

	open, dep, place, plan, err := realMarketFrames(net, cid, nativeSpec(), erc20Spec(erc20Addr(0x01)), maker, askPrice, askSize, depAmt, depRef)
	if err != nil {
		t.Fatalf("realMarketFrames: %v", err)
	}

	// The plan's ids must be the golden ids.
	if got := hex.EncodeToString(plan.BaseID[:]); got != goldenNativeID3 {
		t.Fatalf("plan baseID = %s, want golden native %s", got, goldenNativeID3)
	}
	if got := hex.EncodeToString(plan.QuoteID[:]); got != goldenERC20ID3 {
		t.Fatalf("plan quoteID = %s, want golden erc20 %s", got, goldenERC20ID3)
	}

	// (a) OpenMarket is 96 bytes and its slots equal the derived ids.
	if len(open) != zapwire.OpenMarketReqSize {
		t.Fatalf("OpenMarket frame len = %d, want %d", len(open), zapwire.OpenMarketReqSize)
	}
	gotPool, gotBase, gotQuote, err := zapwire.DecodeOpenMarket(open)
	if err != nil {
		t.Fatalf("DecodeOpenMarket: %v", err)
	}
	if gotPool != plan.PoolID {
		t.Fatalf("OpenMarket poolID slot = %x, want %x", gotPool, plan.PoolID)
	}
	if hex.EncodeToString(gotBase[:]) != goldenNativeID3 {
		t.Fatalf("OpenMarket base slot = %x, want golden native", gotBase)
	}
	if hex.EncodeToString(gotQuote[:]) != goldenERC20ID3 {
		t.Fatalf("OpenMarket quote slot = %x, want golden erc20", gotQuote)
	}

	// (b) Deposit is 88 bytes, binds the maker + base id + amount + ref.
	if len(dep) != zapwire.DepositReqSize {
		t.Fatalf("Deposit frame len = %d, want %d", len(dep), zapwire.DepositReqSize)
	}
	gotUser, gotAsset, gotAmt, gotRef, err := zapwire.DecodeDeposit(dep)
	if err != nil {
		t.Fatalf("DecodeDeposit: %v", err)
	}
	if gotUser != maker {
		t.Fatalf("Deposit user = %q, want %q", gotUser, maker)
	}
	if hex.EncodeToString(gotAsset[:]) != goldenNativeID3 {
		t.Fatalf("Deposit asset = %x, want golden native (base)", gotAsset)
	}
	if gotAmt != depAmt {
		t.Fatalf("Deposit amount = %d, want %d", gotAmt, depAmt)
	}
	if gotRef != depRef {
		t.Fatalf("Deposit ref = %x, want %x", gotRef, depRef)
	}

	// (c) Place is 65 bytes, a resting SELL by the maker at askPrice x askSize on poolID.
	if len(place) != zapwire.PlaceReqSize {
		t.Fatalf("Place frame len = %d, want %d", len(place), zapwire.PlaceReqSize)
	}
	gotPP, gotSide, gotPrice, gotSize, gotPlaceUser, err := zapwire.DecodePlace(place)
	if err != nil {
		t.Fatalf("DecodePlace: %v", err)
	}
	if gotPP != plan.PoolID {
		t.Fatalf("Place poolID = %x, want %x", gotPP, plan.PoolID)
	}
	if gotSide != zapwire.SideSell {
		t.Fatalf("Place side = %d, want SELL(%d)", gotSide, zapwire.SideSell)
	}
	if gotPrice != askPrice || gotSize != askSize {
		t.Fatalf("Place price/size = %v/%v, want %v/%v", gotPrice, gotSize, askPrice, askSize)
	}
	if gotPlaceUser != maker {
		t.Fatalf("Place user = %q, want %q", gotPlaceUser, maker)
	}
}

// TestRealMarketFrames_RejectsSyntheticQuote proves the whole frame-build refuses a
// synthetic quote ref — no frames are returned for a market the chain would refuse.
func TestRealMarketFrames_RejectsSyntheticQuote(t *testing.T) {
	cid := mustCChain(t)
	syntheticQuote := assetSpec{Kind: dex.AssetKindERC20, Ref: []byte("LUSD")} // ascii ticker, not 20 bytes
	open, dep, place, _, err := realMarketFrames(3, cid, nativeSpec(), syntheticQuote, "maker", 1.0, 1.0, 3, [32]byte{})
	if err == nil {
		t.Fatal("realMarketFrames MUST reject a synthetic (ascii-ticker) quote ref")
	}
	if open != nil || dep != nil || place != nil {
		t.Fatal("no frames may be returned when asset admission fails")
	}
}

// TestRealMarketFrames_RejectsSameBaseQuote proves a market whose base and quote
// resolve to the SAME id is refused (it is not a market).
func TestRealMarketFrames_RejectsSameBaseQuote(t *testing.T) {
	cid := mustCChain(t)
	same := erc20Spec(erc20Addr(0x07))
	if _, _, _, _, err := realMarketFrames(3, cid, same, same, "maker", 1.0, 1.0, 3, [32]byte{}); err == nil {
		t.Fatal("realMarketFrames MUST reject base==quote")
	}
}

// TestDepositRefDeterministic proves the deposit idempotency ref is deterministic in
// (poolID, asset, amount) — a byte-identical re-seed dedups to exactly-once on the
// content-addressed D-Chain.
func TestDepositRefDeterministic(t *testing.T) {
	cid := mustCChain(t)
	baseID, err := nativeSpec().resolveID(3, cid)
	if err != nil {
		t.Fatalf("resolveID: %v", err)
	}
	pid := poolIDFor(baseID, baseID) // any pid for the test
	r1 := depositRefFor(pid, baseID, 100)
	r2 := depositRefFor(pid, baseID, 100)
	if r1 != r2 {
		t.Fatalf("depositRefFor not deterministic: %x != %x", r1, r2)
	}
	if r3 := depositRefFor(pid, baseID, 101); r3 == r1 {
		t.Fatal("depositRefFor must change with amount")
	}
}

// TestParseERC20Addr exercises the boundary parser: it accepts a 0x-prefixed and a bare
// 20-byte address, and rejects wrong-length / non-hex.
func TestParseERC20Addr(t *testing.T) {
	ok := []string{
		"0x0000000000000000000000000000000000000001",
		"0000000000000000000000000000000000000001",
	}
	for _, s := range ok {
		b, err := parseERC20Addr(s)
		if err != nil {
			t.Fatalf("parseERC20Addr(%q) unexpected error: %v", s, err)
		}
		if len(b) != 20 {
			t.Fatalf("parseERC20Addr(%q) len = %d, want 20", s, len(b))
		}
	}
	bad := []string{"", "0x", "0x01", "0xZZ00000000000000000000000000000000000001",
		"0x000000000000000000000000000000000000000001"} // 21 bytes
	for _, s := range bad {
		if _, err := parseERC20Addr(s); err == nil {
			t.Fatalf("parseERC20Addr(%q) must error", s)
		}
	}
}
