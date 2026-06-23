// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package main

import (
	"crypto/sha256"
	"fmt"

	"github.com/luxfi/dex/pkg/dexcore"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// seed.go is the PURE, testable core of the canonical real-market seeder. It holds
// no I/O: every function here turns inputs into bytes (frozen zapwire frames) or
// derives the canonical 32-byte identities the native D-Chain keys its ledger by.
// main.go is the thin transport that POSTs these frames over HTTP.
//
// THE IDENTITY RULE this file enforces (and the reason this seeder is SAFE against a
// shared cluster, with no guard needed): every asset handle is the dexcore CANONICAL
// AssetID — DeriveAssetID(networkID, C-chainID, kind, canonicalRef) — the SAME
// consensus-native identity primitive the on-chain resolver computes. It is NEVER an
// ascii-of-symbol id (the synthetic form, e.g. 0x4c5558 = "LUX", that bricked the
// chain by binding a resting order to a market with no real asset). A synthetic /
// malformed ref is rejected HERE, at frame-build time, before a single byte is sent,
// by CanonicalRefFor inside DeriveAssetID.

// poolIDDomain is the domain-separation tag for the canonical poolID derivation. It
// is folded as the FIRST field so a poolID preimage can never collide with any other
// SHA-256 use, and so the result is structurally unrelated to an ascii ticker.
var poolIDDomain = []byte("lux:dex:dexseed:poolid:v1")

// assetSpec names a real on-chain asset by WHERE IT LIVES: its kind (EVM_NATIVE /
// ERC20 / UTXO) and its canonical on-chain reference (the native marker for
// EVM_NATIVE, the 20-byte token address for ERC20, the 32-byte source assetID for
// UTXO). It is the seeder's input unit; resolveID turns it into the canonical
// AssetID the chain keys by. It deliberately carries NO ticker/symbol — identity is
// the derived id, never a string.
type assetSpec struct {
	Kind dexcore.AssetKind
	Ref  []byte
}

// nativeSpec is the canonical EVM_NATIVE (LUX) asset spec: the EVM zero-address
// native marker. Its derived id is the chain's own coin under the bound network.
func nativeSpec() assetSpec {
	return assetSpec{Kind: dexcore.AssetKindEVMNative, Ref: dexcore.EVMNativeMarker}
}

// erc20Spec builds an ERC20 asset spec from a 20-byte token contract address. The
// address is validated (length, non-zero) when the spec is resolved, not here.
func erc20Spec(addr []byte) assetSpec {
	ref := make([]byte, len(addr))
	copy(ref, addr)
	return assetSpec{Kind: dexcore.AssetKindERC20, Ref: ref}
}

// resolveID derives the canonical, consensus-native 32-byte AssetID for this spec
// under (networkID, cChainID). It is the SINGLE admission gate in the seeder: a
// synthetic/malformed ref (wrong length, zero ERC-20 address, non-marker native ref,
// invalid kind) is REFUSED here by dexcore.DeriveAssetID — so no frame is ever built,
// let alone sent, for an asset that is not a well-formed real on-chain identity. This
// is byte-identical to what the chain's resolver derives, so a market this seeder
// opens binds the SAME id the chain admits.
func (a assetSpec) resolveID(networkID uint32, cChainID ids.ID) (ids.ID, error) {
	id, err := dexcore.DeriveAssetID(networkID, cChainID, a.Kind, a.Ref)
	if err != nil {
		return ids.Empty, fmt.Errorf("resolve %s asset id: %w", a.Kind, err)
	}
	return id, nil
}

// poolIDFor derives the canonical, deterministic, NON-ascii poolID for a (base,
// quote) market from the two REAL derived asset ids:
//
//	poolID = SHA-256( len|domain ‖ len|baseID ‖ len|quoteID )
//
// It is reproducible (the same base/quote always yield the same poolID, so a re-seed
// is idempotent at the market level) and structurally cannot equal an ascii ticker
// (the domain tag dominates the preimage). It is keyed by the DERIVED ids, not the
// refs, so two markets over the same economic pair on the same network share one
// poolID by construction.
func poolIDFor(baseID, quoteID ids.ID) [32]byte {
	h := sha256.New()
	writeLenPrefixed(h, poolIDDomain)
	writeLenPrefixed(h, baseID[:])
	writeLenPrefixed(h, quoteID[:])
	var pid [32]byte
	copy(pid[:], h.Sum(nil))
	return pid
}

// writeLenPrefixed writes an 8-byte big-endian length followed by b, so the fold is
// injective (no two distinct field tuples share a preimage) — the same discipline as
// dexcore's assetFolder.
func writeLenPrefixed(h interface{ Write([]byte) (int, error) }, b []byte) {
	var lp [8]byte
	lp[0] = byte(len(b) >> 56)
	lp[1] = byte(len(b) >> 48)
	lp[2] = byte(len(b) >> 40)
	lp[3] = byte(len(b) >> 32)
	lp[4] = byte(len(b) >> 24)
	lp[5] = byte(len(b) >> 16)
	lp[6] = byte(len(b) >> 8)
	lp[7] = byte(len(b))
	_, _ = h.Write(lp[:])
	_, _ = h.Write(b)
}

// marketPlan is the fully-derived, validated description of one real market the
// seeder will create: the canonical poolID and both REAL asset ids. It is produced by
// realMarketFrames (which also returns the frozen frames) and is what main.go reads
// back the book/markets against.
type marketPlan struct {
	PoolID  [32]byte
	BaseID  ids.ID
	QuoteID ids.ID
}

// realMarketFrames is the pure heart of the seeder. Given the network identity, the
// C-Chain id, the base/quote REAL asset specs, the maker's 16-byte CLOB user handle,
// and the resting ask price/size, it:
//
//  1. derives both canonical asset ids (REFUSING any synthetic/malformed ref);
//  2. derives the canonical NON-ascii poolID from those ids;
//  3. builds the three FROZEN zapwire frames that seed the market end-to-end:
//     - OpenMarket(poolID, baseID, quoteID)         — bind the real (base,quote);
//     - Deposit(maker, baseID, depositAmount, ref)  — fund the maker's base balance;
//     - Place(poolID, SELL, askPrice, askSize, maker) — rest a maker limit ask.
//
// It returns the raw frames, the validated plan, and any derivation error. It does NO
// I/O — main.go posts the frames. depositAmount must cover the ask lock (>= askSize in
// base whole units); ref is the deposit idempotency reference (originating-op id).
func realMarketFrames(
	networkID uint32,
	cChainID ids.ID,
	base, quote assetSpec,
	maker string,
	askPrice, askSize float64,
	depositAmount uint64,
	depositRef [32]byte,
) (openMarket, deposit, place []byte, plan marketPlan, err error) {
	baseID, err := base.resolveID(networkID, cChainID)
	if err != nil {
		return nil, nil, nil, marketPlan{}, fmt.Errorf("base: %w", err)
	}
	quoteID, err := quote.resolveID(networkID, cChainID)
	if err != nil {
		return nil, nil, nil, marketPlan{}, fmt.Errorf("quote: %w", err)
	}
	if baseID == quoteID {
		return nil, nil, nil, marketPlan{}, fmt.Errorf("base and quote resolve to the same asset id %s — not a market", baseID)
	}

	pid := poolIDFor(baseID, quoteID)

	openMarket = zapwire.EncodeOpenMarket(pid, baseID, quoteID)
	deposit = zapwire.EncodeDeposit(maker, baseID, depositAmount, depositRef)
	place = zapwire.EncodePlace(pid, zapwire.SideSell, askPrice, askSize, maker)

	return openMarket, deposit, place, marketPlan{PoolID: pid, BaseID: baseID, QuoteID: quoteID}, nil
}

// depositRefFor derives a deterministic idempotency reference for the maker's seed
// deposit, domain-separated and bound to (poolID, asset, amount). The native D-Chain
// content-addresses its dedup index on the full frame, so a byte-identical re-seed
// (same pool, same asset, same amount) dedups to exactly-once — a re-run does not
// double-credit. It folds the canonical asset id (32 bytes), not a symbol.
func depositRefFor(poolID [32]byte, assetID ids.ID, amount uint64) [32]byte {
	h := sha256.New()
	writeLenPrefixed(h, []byte("lux:dex:dexseed:depositref:v1"))
	writeLenPrefixed(h, poolID[:])
	writeLenPrefixed(h, assetID[:])
	var amt [8]byte
	amt[0] = byte(amount >> 56)
	amt[1] = byte(amount >> 48)
	amt[2] = byte(amount >> 40)
	amt[3] = byte(amount >> 32)
	amt[4] = byte(amount >> 24)
	amt[5] = byte(amount >> 16)
	amt[6] = byte(amount >> 8)
	amt[7] = byte(amount)
	writeLenPrefixed(h, amt[:])
	var ref [32]byte
	copy(ref[:], h.Sum(nil))
	return ref
}
