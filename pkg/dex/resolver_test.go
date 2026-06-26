// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"testing"

	"github.com/luxfi/ids"
)

// resolver_test.go is the PERMISSIONLESS-PUBLIC admission proof: the value path's
// market-open RESOLVES each side's canonical identity (permissionless — any well-formed
// real reference on the bound network resolves) and PROVES each side's live on-chain
// reality (the AUTHORITATIVE EXTCODESIZE check). A fabricated/synthetic asset is refused by
// the reality check (no code), NOT by an allowlist; a REAL ERC-20 that was never
// pre-registered trades; an explicitly DISABLED asset is refused. This pins the owner's
// model: product = permissionless + public; safety = the canonical-identity proof + the
// authoritative on-chain reality check, not membership.

// permissionlessResolver is a resolver bound to a fixed network identity that resolves the
// CANONICAL IDENTITY of ANY well-formed reference on its bound chain — exactly as the real
// chains/dexvm AssetResolver resolves an EVM asset permissionlessly. It does NOT gate on
// pre-registration: the authoritative reality check is the OnChainAssetVerifier, run next.
// It refuses ONLY (a) a malformed reference (DeriveAssetID rejects it) and (b) an
// EXPLICITLY disabled reference (a per-asset deny / emergency halt — modeled by `disabled`).
// A deny is not membership: it is an out-of-band kill switch over an otherwise-real asset.
type permissionlessResolver struct {
	networkID uint32
	chainID   ids.ID
	// decimals is an OPTIONAL metadata cache (a manifest a node may ship). A ref absent
	// here still resolves (permissionless) with the default decimals — the cache only
	// supplies the on-chain precision when known; it never gates admission.
	decimals map[ids.ID]uint8
	// disabled is the explicit deny set (a per-asset kill / emergency halt). A ref present
	// here is refused even though it is well-formed and real — the only non-membership
	// reason the resolver itself refuses an EVM asset.
	disabled map[ids.ID]bool
}

func newPermissionlessResolver(networkID uint32, chainID ids.ID) *permissionlessResolver {
	return &permissionlessResolver{
		networkID: networkID,
		chainID:   chainID,
		decimals:  map[ids.ID]uint8{},
		disabled:  map[ids.ID]bool{},
	}
}

// cacheDecimals records OPTIONAL on-chain decimals metadata for a reference (a manifest
// cache entry). It does NOT register/admit the asset — resolution is permissionless either
// way; this only supplies the precision when the node happens to know it.
func (r *permissionlessResolver) cacheDecimals(t *testing.T, kind AssetKind, ref []byte, decimals uint8) ids.ID {
	t.Helper()
	id, err := DeriveAssetID(r.networkID, r.chainID, kind, ref)
	if err != nil {
		t.Fatalf("cacheDecimals: derive id: %v", err)
	}
	r.decimals[id] = decimals
	return id
}

// disable marks a well-formed real asset as explicitly DENIED (kill switch / emergency
// halt). The resolver refuses it even though it is real on-chain.
func (r *permissionlessResolver) disable(t *testing.T, kind AssetKind, ref []byte) ids.ID {
	t.Helper()
	id, err := DeriveAssetID(r.networkID, r.chainID, kind, ref)
	if err != nil {
		t.Fatalf("disable: derive id: %v", err)
	}
	r.disabled[id] = true
	return id
}

// defaultDecimals is the precision a permissionless resolve returns when the node ships no
// cached metadata for the asset. Decimals is metadata on the value path (the conservation
// keying is by value key, not decimals), so any sane default is sound for admission.
const defaultDecimals uint8 = 18

func (r *permissionlessResolver) ResolveAsset(kind AssetKind, ref []byte) (ids.ID, uint8, error) {
	// Permissionless: derive the canonical id under the bound network identity. A malformed
	// reference (bad shape, wrong length, zero address) is refused HERE by DeriveAssetID —
	// that is the "synthetic/malformed reference" refusal, not a membership check.
	id, err := DeriveAssetID(r.networkID, r.chainID, kind, ref)
	if err != nil {
		return ids.Empty, 0, err
	}
	// Explicit deny (kill switch / emergency halt) is the ONLY non-membership reason the
	// resolver itself refuses an otherwise-real asset.
	if r.disabled[id] {
		return ids.Empty, 0, errors.New("asset explicitly disabled (kill switch)")
	}
	// Any well-formed, non-denied reference resolves — no pre-registration required. Decimals
	// come from the optional cache, else the default.
	dec, ok := r.decimals[id]
	if !ok {
		dec = defaultDecimals
	}
	return id, dec, nil
}

// erc20Ref builds a deterministic non-zero 20-byte token address ref from a tag.
func erc20Ref(tag byte) []byte {
	a := make([]byte, 20)
	a[19] = tag
	return a
}

// fakeOnChainVerifier is a test OnChainAssetVerifier that models live-state code presence —
// the AUTHORITATIVE reality gate. An ERC-20 ref is "real on chain" iff it was seeded with
// code; native is always real; UTXO is reported real (not an EVM-state object). It mirrors
// the precompile's code-size-backed verifier without an EVM — a token whose ref was NOT
// seeded has no code, so admission fails the reality gate exactly as a self-destructed
// contract (or a fabricated/synthetic ASCII-ticker address) would.
type fakeOnChainVerifier struct {
	hasCode map[string]bool // hex(ref) -> code present
}

func newFakeOnChainVerifier() *fakeOnChainVerifier {
	return &fakeOnChainVerifier{hasCode: map[string]bool{}}
}

// seedCode marks an ERC-20 ref as having live contract code (a real deployed token).
func (v *fakeOnChainVerifier) seedCode(ref []byte) { v.hasCode[string(ref)] = true }

func (v *fakeOnChainVerifier) VerifyOnChainAsset(kind AssetKind, ref []byte) error {
	switch kind {
	case AssetKindEVMNative:
		return nil // the chain's own coin: always real
	case AssetKindUTXO:
		return nil // not an EVM-state object; reality is the resolver's X-chain attestation
	case AssetKindERC20:
		if v.hasCode[string(ref)] {
			return nil
		}
		return errors.New("test verifier: no contract code at token address")
	default:
		return ErrInvalidKind
	}
}

// realVerifierFor builds a verifier that treats every given ERC-20 ref as backed by live
// code (a deployed token). Native is always real. Refs NOT passed have no code.
func realVerifierFor(refs ...[]byte) *fakeOnChainVerifier {
	v := newFakeOnChainVerifier()
	for _, r := range refs {
		v.seedCode(r)
	}
	return v
}

// TestPermissionless_OpenMarketChecked_AdmitsUnregisteredRealAsset is the PERMISSIONLESS
// positive proof and the test that FAILS without the resolver change: a REAL (has live
// on-chain code) ERC-20 that was NEVER pre-registered (no manifest/cache entry) MUST open a
// market over a real quote — permissionlessly. Under the old allowlist model this base
// would be refused ErrAssetNotAdmitted for not being registered; under the permissionless
// model it resolves (well-formed real ref) and verifies (has code), so the market opens.
func TestPermissionless_OpenMarketChecked_AdmitsUnregisteredRealAsset(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	// The QUOTE is a real ERC-20 whose precision the node happens to cache; the BASE is a
	// real ERC-20 the node has NEVER seen (no cache, no registration) — but it has live code.
	quoteRef := erc20Ref(0x01)
	r.cacheDecimals(t, AssetKindERC20, quoteRef, 6)
	unregisteredRealBaseRef := erc20Ref(0xAB) // never registered, never cached

	db := newStore()
	pid := market(0x10)

	// Both refs are backed by live on-chain code -> the AUTHORITATIVE reality gate passes;
	// the permissionless resolver admits the unregistered base by IDENTITY, not membership.
	v := realVerifierFor(unregisteredRealBaseRef, quoteRef)
	if err := OpenMarketChecked(db, r, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: unregisteredRealBaseRef},
		AssetSide{Kind: AssetKindERC20, Ref: quoteRef}); err != nil {
		t.Fatalf("permissionless OpenMarketChecked must ADMIT a REAL, unregistered ERC-20 over a real quote: %v", err)
	}
	// The market must be bound to the established left-padded VALUE KEYS (unchanged keying).
	gotBase, gotQuote, ok, err := ReadMarketAssets(db, pid)
	if err != nil {
		t.Fatalf("ReadMarketAssets: %v", err)
	}
	if !ok {
		t.Fatal("a permissionlessly admitted market must have its assets bound")
	}
	baseSide := AssetSide{Kind: AssetKindERC20, Ref: unregisteredRealBaseRef}
	quoteSide := AssetSide{Kind: AssetKindERC20, Ref: quoteRef}
	if gotBase != baseSide.valueKey() || gotQuote != quoteSide.valueKey() {
		t.Fatalf("market bound to base=%x quote=%x, want value keys base=%x quote=%x",
			gotBase, gotQuote, baseSide.valueKey(), quoteSide.valueKey())
	}
}

// TestPermissionless_OpenMarketChecked_RefusesSyntheticBase proves the SAFETY property
// (a synthetic/fabricated asset can never trade) is preserved under the permissionless
// model — but enforced by the AUTHORITATIVE reality gate, not by a membership list. The
// base is a fabricated ERC-20 (a left-padded ASCII-ticker-shaped address) with NO live
// on-chain code. It resolves to a well-formed id (permissionless), but the EXTCODESIZE
// verifier refuses it (ErrAssetNotOnChain), so no market is created (fail-closed).
func TestPermissionless_OpenMarketChecked_RefusesSyntheticBase(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	quoteRef := erc20Ref(0x01)
	r.cacheDecimals(t, AssetKindERC20, quoteRef, 6)
	syntheticBaseRef := erc20Ref(0xEE) // fabricated address, no deployed contract

	db := newStore()
	pid := market(0x11)

	// THE ATTACK: open a market with a fabricated base over a real quote. The base resolves
	// (its left-pad is a well-formed 32-byte id) but has NO live code, so the AUTHORITATIVE
	// reality gate must REFUSE it. Only the quote is seeded with code.
	v := realVerifierFor(quoteRef)
	err := OpenMarketChecked(db, r, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: syntheticBaseRef},
		AssetSide{Kind: AssetKindERC20, Ref: quoteRef})
	if err == nil {
		t.Fatal("OpenMarketChecked must REFUSE a market whose base is a fabricated/synthetic (no-code) asset")
	}
	if !errors.Is(err, ErrAssetNotOnChain) {
		t.Fatalf("a synthetic asset must be refused by the authoritative reality gate (ErrAssetNotOnChain), got: %v", err)
	}
	// And NO market binding may have been written (fail-closed: nothing persisted on a
	// refused open).
	if _, _, ok, rerr := ReadMarketAssets(db, pid); rerr != nil {
		t.Fatalf("ReadMarketAssets: %v", rerr)
	} else if ok {
		t.Fatal("a refused OpenMarketChecked must NOT have bound any market assets")
	}
}

// TestPermissionless_OpenMarketChecked_RefusesDisabledQuote proves the explicit-deny
// (kill switch / emergency halt) path: a well-formed, real asset that is EXPLICITLY
// DISABLED is refused at resolve (ErrAssetNotResolved). This is NOT membership — it is an
// out-of-band deny over an otherwise-tradeable real asset.
func TestPermissionless_OpenMarketChecked_RefusesDisabledQuote(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	baseRef := erc20Ref(0x02)
	disabledQuoteRef := erc20Ref(0x03)
	r.disable(t, AssetKindERC20, disabledQuoteRef) // explicit kill switch

	db := newStore()
	pid := market(0x12)
	// Both have live code, so the reality gate would pass — the explicit DENY at resolve is
	// what must fire.
	v := realVerifierFor(baseRef, disabledQuoteRef)
	err := OpenMarketChecked(db, r, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: baseRef},
		AssetSide{Kind: AssetKindERC20, Ref: disabledQuoteRef})
	if err == nil {
		t.Fatal("OpenMarketChecked must REFUSE a market whose quote is an explicitly DISABLED (killed) asset")
	}
	if !errors.Is(err, ErrAssetNotResolved) {
		t.Fatalf("an explicitly disabled asset must be refused at resolve (ErrAssetNotResolved), got: %v", err)
	}
}

// TestPermissionless_OpenMarketChecked_RefusesMalformedRef proves a MALFORMED reference is
// refused at resolve — the resolver derives no id for a wrong-length/zero reference, so a
// malformed side can never bind a market.
func TestPermissionless_OpenMarketChecked_RefusesMalformedRef(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	baseRef := erc20Ref(0x04)
	malformedQuoteRef := []byte{0x01, 0x02, 0x03} // not a 20-byte address

	db := newStore()
	pid := market(0x13)
	v := realVerifierFor(baseRef) // base real; quote is malformed regardless of code
	err := OpenMarketChecked(db, r, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: baseRef},
		AssetSide{Kind: AssetKindERC20, Ref: malformedQuoteRef})
	if err == nil {
		t.Fatal("OpenMarketChecked must REFUSE a market whose quote reference is malformed")
	}
	if !errors.Is(err, ErrAssetNotResolved) {
		t.Fatalf("a malformed reference must be refused at resolve (ErrAssetNotResolved), got: %v", err)
	}
}

// TestPermissionless_OpenMarketChecked_AdmitsNativePlusRealERC20 proves the common public
// pair (native LUX + a real ERC-20 quote) opens permissionlessly and keys to the established
// value keys, with the native base keyed to the all-zero id.
func TestPermissionless_OpenMarketChecked_AdmitsNativePlusRealERC20(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	quoteRef := erc20Ref(0x01)
	r.cacheDecimals(t, AssetKindERC20, quoteRef, 6)

	db := newStore()
	pid := market(0x14)
	baseSide := AssetSide{Kind: AssetKindEVMNative, Ref: EVMNativeMarker}
	quoteSide := AssetSide{Kind: AssetKindERC20, Ref: quoteRef}
	v := realVerifierFor(quoteRef) // native always real; quote has live code
	if err := OpenMarketChecked(db, r, v, pid, baseSide, quoteSide); err != nil {
		t.Fatalf("OpenMarketChecked must ADMIT a market over native + a real ERC-20: %v", err)
	}
	gotBase, gotQuote, ok, err := ReadMarketAssets(db, pid)
	if err != nil {
		t.Fatalf("ReadMarketAssets: %v", err)
	}
	if !ok {
		t.Fatal("an admitted market must have its assets bound")
	}
	if gotBase != baseSide.valueKey() {
		t.Fatalf("market base bound to %x, want value key %x", gotBase, baseSide.valueKey())
	}
	if gotQuote != quoteSide.valueKey() {
		t.Fatalf("market quote bound to %x, want value key %x", gotQuote, quoteSide.valueKey())
	}
	if (gotBase != AssetID{}) {
		t.Fatalf("native base must key to the all-zero value id, got %x", gotBase)
	}
}

func TestPermissionless_OpenMarketChecked_NilResolverFailsClosed(t *testing.T) {
	db := newStore()
	pid := market(0x15)
	// No resolver injected: the value path must FAIL CLOSED rather than admit a left-pad.
	v := realVerifierFor(erc20Ref(0x01), erc20Ref(0x02))
	err := OpenMarketChecked(db, nil, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: erc20Ref(0x01)},
		AssetSide{Kind: AssetKindERC20, Ref: erc20Ref(0x02)})
	if !errors.Is(err, ErrNoAssetResolver) {
		t.Fatalf("a nil resolver must fail closed with ErrNoAssetResolver, got: %v", err)
	}
}

// TestPermissionless_OpenMarketChecked_RefusesAssetWithNoOnChainCode proves the
// AUTHORITATIVE reality gate stands alone: an asset that RESOLVES (well-formed, real ref,
// not disabled) but has NO contract code in the executing state (a self-destructed /
// never-deployed token) is REFUSED with ErrAssetNotOnChain, and nothing is bound.
func TestPermissionless_OpenMarketChecked_RefusesAssetWithNoOnChainCode(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)

	baseRef := erc20Ref(0x21)
	quoteRef := erc20Ref(0x22)

	db := newStore()
	pid := market(0x16)

	// The QUOTE has NO live on-chain code (only the base was seeded). Both resolve; the
	// live-reality verifier must REFUSE the code-less quote.
	v := realVerifierFor(baseRef) // quoteRef deliberately absent => no code
	err := OpenMarketChecked(db, r, v, pid,
		AssetSide{Kind: AssetKindERC20, Ref: baseRef},
		AssetSide{Kind: AssetKindERC20, Ref: quoteRef})
	if !errors.Is(err, ErrAssetNotOnChain) {
		t.Fatalf("a resolvable asset with no live on-chain code must be refused (ErrAssetNotOnChain), got: %v", err)
	}
	if _, _, ok, rerr := ReadMarketAssets(db, pid); rerr != nil {
		t.Fatalf("ReadMarketAssets: %v", rerr)
	} else if ok {
		t.Fatal("a market refused at the live-reality gate must NOT have bound any assets")
	}
}

// TestPermissionless_OpenMarketChecked_NilVerifierFailsClosed proves that with a resolver
// but NO on-chain verifier, the value path FAILS CLOSED (ErrNoOnChainVerifier) rather than
// admit an asset it cannot prove is backed by live code.
func TestPermissionless_OpenMarketChecked_NilVerifierFailsClosed(t *testing.T) {
	const net uint32 = 2
	chain := ids.GenerateTestID()
	r := newPermissionlessResolver(net, chain)
	baseRef := erc20Ref(0x31)
	quoteRef := erc20Ref(0x32)

	db := newStore()
	pid := market(0x17)
	err := OpenMarketChecked(db, r, nil, pid,
		AssetSide{Kind: AssetKindERC20, Ref: baseRef},
		AssetSide{Kind: AssetKindERC20, Ref: quoteRef})
	if !errors.Is(err, ErrNoOnChainVerifier) {
		t.Fatalf("a nil on-chain verifier must fail closed with ErrNoOnChainVerifier, got: %v", err)
	}
}
