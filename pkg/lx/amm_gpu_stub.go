// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && !darwin && !linux

package lx

// gpuBatchEval has no GPU backend on this OS. The dispatcher in
// amm_gpu.go turns errAMMGPUUnsupported into a transparent CPU fallback.
func gpuBatchEval(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	return nil, errAMMGPUUnsupported
}
