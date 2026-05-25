// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

package lx

import (
	"fmt"

	"github.com/luxfi/crypto/backend"
)

// BatchEvalConstantProduct dispatches an N-tuple xy=k batch onto the host
// GPU and returns the per-tuple out_amount. Output is byte-identical to
// the CPU oracle BatchEvalConstantProductCPU.
//
// Backend selection at compile time (CGo on):
//   darwin  → amm_gpu_metal.go → Metal
//   linux   → amm_gpu_cuda.go  → CUDA
//   other   → amm_gpu_stub.go  → returns errAMMGPUUnsupported (caller falls back to CPU)
//
// Fallback policy (recorded via backend.RecordFallback):
//   GPU_DISABLE=1            → reason "disabled"
//   gpuBatchEval returns sentinel → reason "unsupported"
//   other gpuBatchEval error     → surfaced to caller (no fallback)
//
// CGo off → amm_nogpu.go provides a CPU-only implementation of this
// entry point. One and only one BatchEvalConstantProduct symbol per build.
func BatchEvalConstantProduct(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	if len(reserves) != len(amounts) {
		return nil, fmt.Errorf("lx: reserves/amounts length mismatch: %d vs %d",
			len(reserves), len(amounts))
	}
	n := len(reserves)
	if n == 0 {
		return nil, nil
	}

	if backend.GPUDisabled() {
		backend.RecordFallback(backend.FallbackDisabled, "amm")
		return BatchEvalConstantProductCPU(reserves, amounts)
	}

	out, err := gpuBatchEval(reserves, amounts)
	if err == errAMMGPUUnsupported {
		backend.RecordFallback(backend.FallbackUnsupported, "amm")
		return BatchEvalConstantProductCPU(reserves, amounts)
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}

// errAMMGPUUnsupported is returned by the per-platform gpuBatchEval
// when no GPU dispatch is available on this host. The dispatcher above
// turns that into a transparent CPU fallback.
var errAMMGPUUnsupported = fmt.Errorf("lx: amm GPU dispatch not available on this host")
