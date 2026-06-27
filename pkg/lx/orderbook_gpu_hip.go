// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && linux && hip && !vulkan

package lx

// HIP/ROCm backend for the DEX OrderBook match_order kernel (AMD GPUs).
// Opt-in via -tags hip (mutually exclusive with the default CUDA backend and
// the -tags vulkan backend). Linked via
//   lux-dex-orderbook-hip pkg-config bundle
//     -> libdex_orderbook_hip.a (luxcpp/dex/gpu/hip/dex_orderbook_hip.cpp)
//
// The kernel is the SINGLE-THREAD deterministic matcher whose output is byte-
// identical to MatchOrderCPU by construction (see dex_orderbook_hip.cpp). That
// contract is enforced on every run by orderbook_gpu_test.go and the
// determinism corpus in match_verify_test.go.

/*
#cgo pkg-config: lux-dex-orderbook-hip

#include <stdint.h>
#include "dex_orderbook_hip.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// dexBackendLabel reports the linked DEX GPU backend for corpus logging.
func dexBackendLabel() string { return "hip" }

// gpuMatchOrder is the HIP implementation called by orderbook_gpu.go.
// Returns errOrderBookGPUUnsupported when no AMD/ROCm device is present so the
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

	rc := C.dex_orderbook_match_order_hip(
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
		trades := tradesOut[:int(tradesWritten)]
		for i := range trades {
			trades[i].ClearPadding()
		}
		for i := range book {
			book[i].ClearPadding()
		}
		return trades, remaining, nil
	case -2:
		return nil, 0, errOrderBookGPUUnsupported
	default:
		return nil, 0, fmt.Errorf("lx: dex_orderbook_match_order_hip rc=%d", int(rc))
	}
}
