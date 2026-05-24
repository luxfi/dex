// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

package lx

import "fmt"

// BatchEvalConstantProduct under !cgo is the CPU oracle. The cgo build
// provides this symbol via amm_gpu.go (which dispatches to the
// per-platform Metal / CUDA / stub implementation).
func BatchEvalConstantProduct(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	if len(reserves) != len(amounts) {
		return nil, fmt.Errorf("lx: reserves/amounts length mismatch: %d vs %d",
			len(reserves), len(amounts))
	}
	return BatchEvalConstantProductCPU(reserves, amounts)
}
