// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package rfq

import "time"

// timeout.go is the timeout-ordering policy from the HTLC atomic-swap spec
// ("The Timeout-Ordering Rule"). It is a PURE table: no state, no clock reads.
//
// The maker's leg (HTLC_A) expires first; the taker's leg (HTLC_B) expires last,
// and the safety buffer between them must satisfy
//
//	T_B > T_A + Δ,   Δ ≥ R_late + C_late
//
// where R_late + C_late is the reorg-safety depth plus worst-case claim
// confirmation time of whichever chain can reorg. Δ is the FLOOR; AssignTimeouts
// recommends timeouts strictly above it, and a session rejects any submitted leg
// pair that does not clear it.

// ChainKind identifies a settlement chain by its reorg/finality character — the
// only chain property the timeout policy needs. It is a value: the coordinator
// dials no chain and verifies no proof, so a chain is just its safety class.
type ChainKind uint8

const (
	ChainLux     ChainKind = iota // sub-second deterministic finality (R ≈ 0)
	ChainEVM                      // probabilistic EVM L1/L2 (a few minutes)
	ChainBitcoin                  // 6-confirmation reorg safety (~2h with claim confirm)
)

func (c ChainKind) String() string {
	switch c {
	case ChainLux:
		return "lux"
	case ChainEVM:
		return "evm"
	case ChainBitcoin:
		return "bitcoin"
	default:
		return "unknown"
	}
}

// reorgBuffer is R + C per chain: the time a leg's timeout must lead any
// dependent claim by, to survive that chain's worst-case reorg plus claim
// confirmation. From the spec's "Concrete parameters": Bitcoin ≥ 2h (6-block
// reorg ≈ 60min plus claim confirmation), EVM a few minutes, Lux ≈ 0.
var reorgBuffer = map[ChainKind]time.Duration{
	ChainLux:     0,
	ChainEVM:     5 * time.Minute,
	ChainBitcoin: 2 * time.Hour,
}

const (
	// MakerLegLifetime is T_A − t0: the short window the maker's leg stays locked
	// before auto-refund (spec example: 40 min), bounding maker capital lockup.
	MakerLegLifetime = 40 * time.Minute
	// ClaimMargin is the cushion AssignTimeouts adds above the Δ floor so the
	// recommended (T_A, T_B) strictly satisfy T_B > T_A + Δ.
	ClaimMargin = 5 * time.Minute
)

// Delta is the minimum safety buffer Δ between the two legs' timeouts for a swap
// over (chainA → chainB). Δ must cover whichever involved chain can reorg, so it
// is the larger of the two chains' reorg buffers — Lux↔BTC yields 2h, EVM↔EVM a
// few minutes, exactly as the spec's concrete parameters.
func Delta(chainA, chainB ChainKind) time.Duration {
	a, b := reorgBuffer[chainA], reorgBuffer[chainB]
	if a > b {
		return a
	}
	return b
}

// OrderingHolds reports the atomicity invariant T_B > T_A + Δ for a maker-leg
// timeout tA and taker-leg timeout tB (both absolute unix seconds) under buffer
// delta. This is the single predicate the session enforces on a submitted leg
// pair.
func OrderingHolds(tA, tB uint64, delta time.Duration) bool {
	return tB > tA+uint64(delta/time.Second)
}

// AssignTimeouts recommends the absolute timeouts (tA, tB) for a swap started at
// t0 over (chainA → chainB): the maker's leg expires at t0 + MakerLegLifetime,
// the taker's leg at T_A + Δ + ClaimMargin. The result strictly satisfies
// OrderingHolds(tA, tB, Delta(chainA, chainB)). Parties may choose their own
// timeouts; this is the venue's advisory, and the session validates whatever is
// submitted against the same invariant.
func AssignTimeouts(t0 time.Time, chainA, chainB ChainKind) (tA, tB uint64) {
	at := t0.Add(MakerLegLifetime)
	bt := at.Add(Delta(chainA, chainB) + ClaimMargin)
	return uint64(at.Unix()), uint64(bt.Unix())
}
