// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo || !darwin

package lx

// BatchEvalConstantProduct on non-Apple / non-cgo builds runs the CPU
// oracle. The Metal-backed implementation is in amm_gpu.go.
func BatchEvalConstantProduct(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	return BatchEvalConstantProductCPU(reserves, amounts)
}
