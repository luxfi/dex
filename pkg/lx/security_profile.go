// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

// SecurityProfile is the security profile a Partner DEX chain pins.
// A chain is strict-PQ or it isn't — there is no relaxed middle.
//
// Classical chains accept SignedOrder (secp256k1 r||s||v Ethereum
// signatures, verified via ecrecover). Strict-PQ chains REFUSE
// SignedOrder at the verification boundary and only accept
// SignedOrderPQ (FIPS 204 ML-DSA-65 signatures). Refusal returns
// ErrClassicalAuthForbidden, matching the EVM precompile gate
// shape so observable behaviour stays consistent across the EVM,
// DEX, and FHE chain layers.
type SecurityProfile int

const (
	// ProfileClassical accepts secp256k1 SignedOrder via ecrecover.
	// Acceptable for legacy DEX deployments and the Lux-Permissive
	// profile; a strict-PQ Liquid chain MUST NOT boot under this.
	ProfileClassical SecurityProfile = iota

	// ProfileStrictPQ refuses every classical SignedOrder at the
	// verification boundary. Only SignedOrderPQ (ML-DSA-65) is
	// accepted. Canonical Partner DEX profile.
	ProfileStrictPQ
)

// String returns the canonical wire name. Audit pipelines match on
// these strings; renaming here breaks every downstream consumer.
func (p SecurityProfile) String() string {
	switch p {
	case ProfileClassical:
		return "classical"
	case ProfileStrictPQ:
		return "strict-pq"
	default:
		return "unknown"
	}
}

// IsPostQuantum reports whether this profile refuses classical
// signature schemes.
func (p SecurityProfile) IsPostQuantum() bool {
	return p == ProfileStrictPQ
}

// ProfileFromPQFlag maps a chain-config "pq" boolean (as written
// by liquidity/operator into /data/configs/chains/<id>/config.json)
// to a DEX SecurityProfile. true → ProfileStrictPQ;
// false → ProfileClassical.
func ProfileFromPQFlag(pq bool) SecurityProfile {
	if pq {
		return ProfileStrictPQ
	}
	return ProfileClassical
}
