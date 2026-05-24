// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && linux

package lx

// CUDA backend for the DEX CLOB match_order kernel. Linked via
//   lux-dex-clob-cuda pkg-config bundle
//     -> libdex_clob_cuda.a (luxcpp/dex/gpu/cuda/dex_clob_host.cu
//                          + luxcpp/cuda/kernels/gpu/dex_swap.cu)

/*
#cgo pkg-config: lux-dex-clob-cuda

#include <stdint.h>
#include "dex_clob_host.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// gpuMatchOrder is the CUDA implementation called by orderbook_gpu.go.
// Returns errCLOBGPUUnsupported when no NVIDIA device is present so the
// dispatcher falls back to MatchOrderCPU.
func gpuMatchOrder(
	incoming *DEXOrder,
	book []DEXOrder,
	bookIndices []uint32,
	tradeIDBase uint64,
	timestamp uint64,
) ([]DEXTrade, uint64, error) {
	if len(bookIndices) == 0 {
		return nil, incoming.Quantity, nil
	}

	tradesOut := make([]DEXTrade, len(bookIndices))
	var (
		tradesWritten uint32
		remaining     uint64
	)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(incoming)
	pinner.Pin(&book[0])
	pinner.Pin(&bookIndices[0])
	pinner.Pin(&tradesOut[0])
	pinner.Pin(&tradesWritten)
	pinner.Pin(&remaining)

	rc := C.dex_clob_match_order_host(
		(*C.DEXOrder)(unsafe.Pointer(incoming)),
		(*C.DEXOrder)(unsafe.Pointer(&book[0])),
		(*C.uint32_t)(unsafe.Pointer(&bookIndices[0])),
		C.uint32_t(len(bookIndices)),
		(*C.DEXTrade)(unsafe.Pointer(&tradesOut[0])),
		C.uint64_t(tradeIDBase),
		C.uint64_t(timestamp),
		(*C.uint32_t)(unsafe.Pointer(&tradesWritten)),
		(*C.uint64_t)(unsafe.Pointer(&remaining)),
	)
	runtime.KeepAlive(incoming)
	runtime.KeepAlive(book)
	runtime.KeepAlive(bookIndices)
	runtime.KeepAlive(tradesOut)

	switch rc {
	case 0:
		// Padding normalization: the host driver already zeros the
		// padding bytes for emitted trades and touched book rows, but
		// belt-and-braces here in case a future kernel revision
		// changes the contract.
		trades := tradesOut[:int(tradesWritten)]
		for i := range trades {
			trades[i].ClearPadding()
		}
		for i := range book {
			book[i].ClearPadding()
		}
		return trades, remaining, nil
	case -2:
		return nil, 0, errCLOBGPUUnsupported
	default:
		return nil, 0, fmt.Errorf("lx: dex_clob_match_order_host rc=%d", int(rc))
	}
}
