// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && linux

package lx

// CUDA backend for batch xy=k AMM evaluation. Linked via
//   lux-dex-amm-cuda pkg-config bundle
//     -> libamm_xyk_cuda.a (luxcpp/dex/gpu/cuda/amm_xyk_driver.cu
//                          + luxcpp/cuda/kernels/dex/amm_xyk.cu)

/*
#cgo pkg-config: lux-dex-amm-cuda

#include <stdint.h>
#include "amm_xyk_driver.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// gpuBatchEval is the CUDA implementation called by amm_gpu.go::BatchEvalConstantProduct.
// When no NVIDIA device is present at runtime, the driver returns -2 and we
// surface errAMMGPUUnsupported so the dispatcher falls back to CPU.
func gpuBatchEval(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	n := len(reserves)
	out := make([]uint64, n)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&reserves[0])
	pinner.Pin(&amounts[0])
	pinner.Pin(&out[0])

	rc := C.amm_xyk_batch_cuda_host(
		(*C.AMMReservePair)(unsafe.Pointer(&reserves[0])),
		(*C.uint64_t)(unsafe.Pointer(&amounts[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(n),
	)
	runtime.KeepAlive(reserves)
	runtime.KeepAlive(amounts)
	runtime.KeepAlive(out)

	switch rc {
	case 0:
		return out, nil
	case -2:
		// No CUDA device / driver — fall back to CPU.
		return nil, errAMMGPUUnsupported
	default:
		return nil, fmt.Errorf("lx: amm_xyk_batch_cuda_host rc=%d", int(rc))
	}
}
