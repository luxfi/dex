// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"errors"

	"github.com/luxfi/ids"
)

// resolver.go is the ADMISSION port of the value path, in the owner's PERMISSIONLESS
// PUBLIC model: the abstraction the 0x9999 cEVM router depends on to ADMIT a market (or
// swap) over an asset by PROVING its canonical identity — NOT by checking it against an
// allowlist. dexcore (the lowest shared leaf) defines the port; the chains/dexvm
// AssetResolver satisfies it; the EVM plugin boot injects a value into the precompile env.
// This is dependency inversion: the value path names the property it needs ("what is the
// canonical id of this real asset, and is it a well-formed real reference on MY network?")
// and the resolver answers it, WITHOUT dexcore or the precompile importing the resolver
// package (no upward dependency, no cycle).
//
// PERMISSIONLESS + PUBLIC (the protocol rule). An asset trades IFF its canonical identity
// VERIFIES ON-CHAIN — there is NO manifest-membership requirement. The two orthogonal
// admission steps are:
//
//   - RESOLVE (this AssetResolver port): prove the canonical identity. Derive the 32-byte
//     AssetID from the asset's real coordinates (kind + canonical ref) rooted at the node's
//     bound (networkID, sourceChainID). This refuses ONLY a MALFORMED reference, a
//     WRONG-NETWORK/WRONG-CHAIN binding, an explicitly-DISABLED asset (a per-asset deny or
//     emergency halt — a deny is not membership), or a synthetic-fake reference. It does
//     NOT require the asset to be pre-registered: any well-formed EVM-native or ERC-20
//     reference on the bound chain resolves. A UTXO's reality is its X-chain assetID proof,
//     which the resolver still attests (an EVM verifier cannot read the X-chain).
//   - VERIFY ON-CHAIN (the OnChainAssetVerifier port): prove the asset is REAL in the live
//     state this block mutates — an ERC-20 must have EXTCODESIZE>0 at its address. THIS is
//     the AUTHORITATIVE reality check. A fabricated/synthetic asset (a left-padded ASCII
//     ticker, a never-deployed or self-destructed address) has no code, so it is refused
//     HERE — not by a membership list.
//
// The left-padded address (assetID) is sufficient to NAME an asset; it is NOT sufficient
// to TRADE one. Trading requires the asset to resolve (well-formed, right network) AND
// verify on-chain (real code). The manifest, if a node ships one, is a CACHE of metadata
// (decimals/symbol) and the X-chain reality source for UTXO — never the membership gate.
//
// WHY identity-bound: the resolver is bound to (networkID, sourceChainID) at construction,
// exactly like the startup RuntimeVerifier. So it answers ONLY for the chain the node
// really runs; an asset cannot resolve under a cross-network identity, because the derived
// AssetID would not match the chain the node serves.

// AssetResolver proves, for the node's bound network identity, the CANONICAL identity of a
// real on-chain asset (identified by its kind + canonical reference) and returns its
// canonical AssetID + on-chain decimals. It is PERMISSIONLESS: it admits any well-formed
// real reference on the bound network and refuses only a malformed/wrong-network/
// explicitly-disabled/synthetic reference. It is the single resolver every market-open and
// swap consults so a synthetic asset is structurally untradeable — that refusal is enforced
// jointly with the OnChainAssetVerifier (the authoritative EXTCODESIZE reality check), not
// by a pre-registration allowlist.
//
// The resolver is bound to (networkID, sourceChainID) at construction, so callers pass only
// the per-asset (kind, ref). An EVM_NATIVE/ERC20 asset binds to the bound C-Chain id; the
// resolver derives the canonical AssetID via DeriveAssetID with its OWN bound ids. A
// malformed reference, a wrong-network binding, or an explicitly disabled/halted asset is
// REFUSED (non-nil error) — the value path then reverts.
type AssetResolver interface {
	// ResolveAsset returns the canonical AssetID and on-chain decimals for a real asset of
	// the given kind with the given canonical reference, rooted at the resolver's bound
	// network identity. It is PERMISSIONLESS for the EVM kinds: any well-formed
	// EVM_NATIVE/ERC20 reference resolves (the authoritative reality check is the
	// OnChainAssetVerifier, run next). It returns a non-nil error ONLY when the reference
	// is malformed, bound to the wrong network/chain, explicitly disabled (a per-asset deny
	// or emergency halt), or a synthetic-fake reference — so the caller reverts the
	// swap/market-open. The returned AssetID equals
	// DeriveAssetID(boundNetworkID, boundSourceChainID, kind, ref).
	ResolveAsset(kind AssetKind, ref []byte) (assetID ids.ID, decimals uint8, err error)
}

// OnChainAssetVerifier proves an asset is REAL in the LIVE chain state the swap is
// executing against — this is the AUTHORITATIVE reality check, run alongside the
// permissionless resolve. The resolver answers "what is this asset's canonical id, and is
// it a well-formed reference on my network?"; the verifier answers "does the token CONTRACT
// actually exist (have bytecode) in the state this block mutates, RIGHT NOW?". Both run on
// every value-path market open, BEFORE any C-balance debit, so a swap is admitted only over
// an asset whose canonical identity resolves AND whose reality is proven by live on-chain
// code.
//
// WHY THIS IS THE REALITY AUTHORITY (not a membership list): a fabricated/synthetic asset
// (a left-padded ASCII ticker, a never-deployed address) has no code at its address, so it
// is refused HERE. A real ERC-20 deployed on the bound C-Chain has code, so it trades —
// permissionlessly, with no pre-registration. The precompile injects a verifier backed by
// the live EVM StateDB's code-size; a non-EVM caller (no verifier) FAILS CLOSED
// (ErrNoOnChainVerifier).
type OnChainAssetVerifier interface {
	// VerifyOnChainAsset returns nil iff the asset of the given kind with the given
	// canonical reference is backed by live on-chain reality in the state being executed:
	// an ERC20 must have non-empty contract code at its address; an EVM_NATIVE marker is
	// always real (the chain's own coin). A UTXO asset is not an EVM-state object, so an
	// EVM verifier reports it real (its reality is the resolver's X-chain attestation). A
	// non-nil error reverts the swap/market-open BEFORE any debit.
	VerifyOnChainAsset(kind AssetKind, ref []byte) error
}

var (
	// ErrAssetNotResolved is returned by the value path when a swap/market references an
	// asset the resolver refuses — a MALFORMED reference, a WRONG-NETWORK/WRONG-CHAIN
	// binding, an explicitly DISABLED asset (a per-asset deny or emergency halt), or a
	// synthetic-fake reference. It is NOT "not in an allowlist": a well-formed real
	// reference on the bound network always resolves. The whole swap/market-open reverts.
	ErrAssetNotResolved = errors.New("dexcore: asset canonical identity does not resolve on this network (malformed, wrong network/chain, disabled, or synthetic)")

	// ErrNoAssetResolver is returned when a market-open/swap requires real-asset admission
	// but no resolver was injected. FAIL-CLOSED: with the value path live and no resolver
	// wired, it refuses rather than admitting a left-padded address as if it were a real,
	// network-bound asset. (A resolver is injected at boot on any node that runs the value
	// path; a paper/non-value node never reaches the value path.)
	ErrNoAssetResolver = errors.New("dexcore: no asset resolver configured (refusing real-asset market/swap; fail-closed)")

	// ErrAssetNotOnChain is the AUTHORITATIVE reality refusal: an asset RESOLVES (its
	// canonical identity is well-formed on this network) but is NOT backed by live on-chain
	// reality in the state being executed (an ERC-20 whose address has no contract code
	// right now). This is the live-state form of "synthetic asset": a fabricated reference
	// resolves to a well-formed id but carries no code, so it is refused here. The
	// swap/market-open reverts BEFORE any debit.
	ErrAssetNotOnChain = errors.New("dexcore: asset has no live on-chain backing (token address has no contract code); refusing market/swap")

	// ErrNoOnChainVerifier is returned when real-asset admission requires the live on-
	// chain reality check but no verifier was injected. FAIL-CLOSED: the value path will
	// not admit an asset it cannot prove is backed by live code, exactly as it will not
	// admit one it cannot resolve. (The precompile injects a code-size-backed verifier on
	// every value-path market open; a caller that cannot supply one must not trade value.)
	ErrNoOnChainVerifier = errors.New("dexcore: no on-chain asset verifier configured (refusing real-asset market/swap; fail-closed)")
)

// RequireRealOnChain proves an asset is backed by live on-chain code via v and maps any
// refusal to the canonical ErrAssetNotOnChain (preserving the underlying cause). A nil
// verifier fails closed with ErrNoOnChainVerifier. It is the ONE helper the value path
// calls for each side of a market so the AUTHORITATIVE reality refusal shape is identical
// everywhere — the live-state half of permissionless admission.
func RequireRealOnChain(v OnChainAssetVerifier, kind AssetKind, ref []byte) error {
	if v == nil {
		return ErrNoOnChainVerifier
	}
	if err := v.VerifyOnChainAsset(kind, ref); err != nil {
		return errors.Join(ErrAssetNotOnChain, err)
	}
	return nil
}

// RequireResolvedAsset resolves an asset's canonical identity through r and maps a refusal
// to the canonical ErrAssetNotResolved (preserving the underlying cause). A nil resolver
// fails closed with ErrNoAssetResolver. It is the ONE helper the value path calls for each
// side of a market so the identity-resolve refusal shape is identical everywhere. It is
// PERMISSIONLESS: a well-formed real reference on the bound network resolves; the
// authoritative reality check is RequireRealOnChain, called next.
func RequireResolvedAsset(r AssetResolver, kind AssetKind, ref []byte) (ids.ID, uint8, error) {
	if r == nil {
		return ids.Empty, 0, ErrNoAssetResolver
	}
	id, dec, err := r.ResolveAsset(kind, ref)
	if err != nil {
		return ids.Empty, 0, errors.Join(ErrAssetNotResolved, err)
	}
	return id, dec, nil
}
