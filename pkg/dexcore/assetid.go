// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/ids"
)

// assetid.go is the CANONICAL real-asset identity primitive for the whole DEX value
// path. It is the SINGLE place that derives the 32-byte AssetID of a real on-chain
// asset from WHERE IT REALLY LIVES — (networkID, sourceChainID, assetKind,
// canonicalRef). It lives in dexcore (the lowest shared leaf) so BOTH the 0x9999
// cEVM value path (precompile/dex) AND the chains/dexvm registry compute the SAME
// id from the same fields: a registered AssetID and a swap-derived AssetID are
// directly comparable, by construction, with no risk of two derivations drifting.
//
// The identity is a length-prefixed, domain-separated SHA-256 fold:
//
//	domAssetV1 | networkID | sourceChainID | assetKind | canonicalRef
//
// matching the per-kind formulas exactly:
//
//	ERC20:      H(networkID, C-chainID, ERC20,      token-address)
//	EVM_NATIVE: H(networkID, C-chainID, EVM_NATIVE, native-marker)
//	UTXO:       H(networkID, X-chain-id, UTXO,      assetID)
//
// Every field's length precedes its bytes, so the fold is INJECTIVE (no two distinct
// field tuples share a preimage), and the kind byte domain-separates the three
// classes (an ERC-20 whose 20-byte address numerically equals the low bytes of a
// UTXO assetID can never collide). The result is an ids.ID — the SAME identity space
// the on-chain atomic objects already use — never a string ticker.
//
// This is the consensus-native asset identity. The dexcore ledger's AssetID alias is
// [32]byte and an ids.ID is [32]byte, so DeriveAssetID's output is directly usable as
// a ledger AssetID via AssetIDFromID.

// AssetKind is the CLOSED set of real asset classes the DEX admits. There are exactly
// three. There is NO synthetic / D-native / declared class — adding one is a breaking
// change to the wire identity and is intentionally hard. The values are wire-pinned
// (they enter the AssetID preimage), identical to chains/dexvm/registry.AssetKind.
type AssetKind uint8

const (
	// AssetKindInvalid is the zero value and is never admissible. Keeping it at 0 means
	// a zero-initialised value fails closed.
	AssetKindInvalid AssetKind = 0
	// AssetKindEVMNative is the C-Chain native coin (e.g. LUX). Its on-chain reference
	// is the fixed native marker (the EVM zero address).
	AssetKindEVMNative AssetKind = 1
	// AssetKindERC20 is an ERC-20 token on the C-Chain. Its reference is the 20-byte
	// token contract address.
	AssetKindERC20 AssetKind = 2
	// AssetKindUTXO is a UTXO-model asset native to the X-Chain. Its reference is the
	// 32-byte source-chain assetID.
	AssetKindUTXO AssetKind = 3
)

// Valid reports whether k is one of the three admissible kinds.
func (k AssetKind) Valid() bool {
	return k == AssetKindEVMNative || k == AssetKindERC20 || k == AssetKindUTXO
}

// String renders the kind as its canonical wire token (these exact strings are part
// of the manifest/policy contract, not cosmetic).
func (k AssetKind) String() string {
	switch k {
	case AssetKindEVMNative:
		return "EVM_NATIVE"
	case AssetKindERC20:
		return "ERC20"
	case AssetKindUTXO:
		return "UTXO"
	default:
		return "INVALID"
	}
}

// EVMNativeMarker is the fixed on-chain reference for the C-Chain native coin: 20 zero
// bytes (the EVM zero address, the EVM's own sentinel for "the native coin"). Folding a
// fixed, kind-tagged marker means every party derives the same EVM_NATIVE AssetID for a
// given (networkID, C-chainID) and nobody can invent a second native asset.
var EVMNativeMarker = make([]byte, 20)

var (
	// ErrInvalidKind is returned when an asset's kind is not one of the three.
	ErrInvalidKind = errors.New("dexcore: asset kind is not EVM_NATIVE, ERC20 or UTXO")
	// ErrBadRef is returned when a canonical reference does not match its kind's shape.
	ErrBadRef = errors.New("dexcore: canonical reference does not match asset kind")
	// ErrEmptyChainID is returned when the source chain of an asset is the empty id.
	ErrEmptyChainID = errors.New("dexcore: asset source chain id is empty")
)

// domAssetV1 is the domain-separation tag folded as the FIRST field of every AssetID
// preimage. It is BYTE-IDENTICAL to chains/dexvm/registry's domAssetV1 so the two
// homes derive the same id. Versioned so a future migration can re-domain cleanly.
var domAssetV1 = []byte("lux:dex:asset:v1")

// CanonicalRefFor validates and returns the canonical on-chain reference bytes for a
// (kind, ref) pair. This is the SINGLE place that decides what a well-formed reference
// looks like per kind:
//
//   - EVM_NATIVE: ref must be the native marker (20 zero bytes). Normalised so no
//     caller can smuggle a non-native ref into the native kind.
//   - ERC20:      ref must be a 20-byte token address, not the zero address (the zero
//     address is the native marker, never a token).
//   - UTXO:       ref must be a 32-byte source-chain assetID.
func CanonicalRefFor(kind AssetKind, ref []byte) ([]byte, error) {
	switch kind {
	case AssetKindEVMNative:
		if len(ref) != 20 {
			return nil, fmt.Errorf("%w: EVM_NATIVE marker must be 20 bytes, got %d", ErrBadRef, len(ref))
		}
		if !allZeroBytes(ref) {
			return nil, fmt.Errorf("%w: EVM_NATIVE reference must be the native marker (address zero)", ErrBadRef)
		}
		return append([]byte(nil), EVMNativeMarker...), nil
	case AssetKindERC20:
		if len(ref) != 20 {
			return nil, fmt.Errorf("%w: ERC20 token address must be 20 bytes, got %d", ErrBadRef, len(ref))
		}
		if allZeroBytes(ref) {
			return nil, fmt.Errorf("%w: ERC20 token address must not be the zero address", ErrBadRef)
		}
		return append([]byte(nil), ref...), nil
	case AssetKindUTXO:
		if len(ref) != 32 {
			return nil, fmt.Errorf("%w: UTXO assetID must be 32 bytes, got %d", ErrBadRef, len(ref))
		}
		if allZeroBytes(ref) {
			return nil, fmt.Errorf("%w: UTXO assetID must not be empty", ErrBadRef)
		}
		return append([]byte(nil), ref...), nil
	default:
		return nil, ErrInvalidKind
	}
}

// DeriveAssetID computes the canonical, consensus-native 32-byte identity of a real
// on-chain asset. sourceChainID is the C-Chain id for EVM_NATIVE/ERC20 and the UTXO
// source chain id for UTXO. The fold is length-prefixed and domain-separated (see the
// file header). The output is byte-identical to chains/dexvm/registry.DeriveAssetID
// for the same inputs — they share this primitive's algorithm exactly.
func DeriveAssetID(networkID uint32, sourceChainID ids.ID, kind AssetKind, ref []byte) (ids.ID, error) {
	if sourceChainID == ids.Empty {
		return ids.Empty, ErrEmptyChainID
	}
	cref, err := CanonicalRefFor(kind, ref)
	if err != nil {
		return ids.Empty, err
	}
	var f assetFolder
	f.tag(domAssetV1)
	f.u32(networkID)
	f.bytes(sourceChainID[:])
	f.u8(uint8(kind))
	f.bytes(cref)
	return f.sum(), nil
}

// AssetIDFromID adapts a consensus-native ids.ID to the dexcore ledger AssetID alias
// ([32]byte). They are the same underlying array; this is the one named conversion so
// the value path keys the ledger by the canonical derived id, never an ad-hoc handle.
func AssetIDFromID(id ids.ID) AssetID { return AssetID(id) }

// assetFolder accumulates a length-prefixed, domain-separated SHA-256 preimage — the
// SAME primitive and discipline as chains/dexvm/registry's folder and the consensus
// state hash. The length prefix on every field is what makes the fold injective.
type assetFolder struct{ buf []byte }

func (f *assetFolder) raw(b []byte) {
	var lp [8]byte
	binary.BigEndian.PutUint64(lp[:], uint64(len(b)))
	f.buf = append(f.buf, lp[:]...)
	f.buf = append(f.buf, b...)
}
func (f *assetFolder) tag(b []byte)   { f.raw(b) }
func (f *assetFolder) bytes(b []byte) { f.raw(b) }
func (f *assetFolder) u8(v uint8)     { f.raw([]byte{v}) }
func (f *assetFolder) u32(v uint32) {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], v)
	f.raw(b[:])
}
func (f *assetFolder) sum() ids.ID { return ids.ID(sha256.Sum256(f.buf)) }

func allZeroBytes(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}
