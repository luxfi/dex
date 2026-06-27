// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

// ReservePair is the (reserve_x, reserve_y) tuple for a single pool, the input
// to the constant-product (xy=k) AMM math used by the on-chain DEX router's
// AMM source (pkg/dex/source_amm.go).
type ReservePair struct {
	ReserveX uint64
	ReserveY uint64
}

// ConstantProductOut is the canonical CPU oracle for the xy=k AMM curve:
//
//	out = (amount * reserve_y) / (reserve_x + amount)
//
// Computed in 128-bit so amount*reserve_y can't silently overflow. Returns
// 0 for the degenerate (rx + amount == 0) case. It is the deterministic,
// consensus-safe pricing used by the DEX router's AMM source
// (pkg/dex/source_amm.go::AMMSource.curveOut).
func ConstantProductOut(rx, ry, amount uint64) uint64 {
	denom := rx + amount
	if denom == 0 {
		return 0
	}
	return mulDiv64(amount, ry, denom)
}

// mulDiv64 = (a * b) / d using 128-bit intermediates so a*b can exceed 2^64
// without truncating. The result is undefined (and the caller's bug) if the
// quotient itself doesn't fit in 64 bits.
func mulDiv64(a, b, d uint64) uint64 {
	// Split into 32-bit halves; each partial product fits in 64 bits.
	aLo, aHi := a&0xFFFFFFFF, a>>32
	bLo, bHi := b&0xFFFFFFFF, b>>32

	ll := aLo * bLo
	lh := aLo * bHi
	hl := aHi * bLo
	hh := aHi * bHi

	mid := (ll >> 32) + (lh & 0xFFFFFFFF) + (hl & 0xFFFFFFFF)
	lo := (ll & 0xFFFFFFFF) | (mid << 32)
	hi := hh + (lh >> 32) + (hl >> 32) + (mid >> 32)

	return divU128byU64(hi, lo, d)
}

// divU128byU64 — straight shift-subtract long division, 128/64 -> 64.
func divU128byU64(hi, lo, d uint64) uint64 {
	if hi == 0 {
		return lo / d
	}
	var quotient, remainder uint64
	for i := 127; i >= 0; i-- {
		remainder <<= 1
		var bit uint64
		if i >= 64 {
			bit = (hi >> uint(i-64)) & 1
		} else {
			bit = (lo >> uint(i)) & 1
		}
		remainder |= bit
		if remainder >= d {
			remainder -= d
			if i < 64 {
				quotient |= uint64(1) << uint(i)
			}
		}
	}
	return quotient
}

// BatchEvalConstantProductCPU is the per-pool CPU batch evaluator over
// ConstantProductOut, exposed for tests and batch callers.
func BatchEvalConstantProductCPU(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	if len(reserves) != len(amounts) {
		return nil, ErrInvalidOrder
	}
	out := make([]uint64, len(reserves))
	for i := range reserves {
		out[i] = ConstantProductOut(reserves[i].ReserveX, reserves[i].ReserveY, amounts[i])
	}
	return out, nil
}
