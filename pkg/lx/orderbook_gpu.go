// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

package lx

// orderbook_gpu.go is the ONE cgo binding for the DEX OrderBook match_order
// kernel. There is deliberately NO per-OS / per-vendor build-tag file: the
// backend is a RUNTIME value, not a build-time place. This file links a single
// unified pkg-config bundle, lux-gpu, whose liblux_gpu selects the best
// available device at runtime (accel's SelectBestBackend: CUDA > HIP > Metal >
// CPU) and exposes it behind one stable C ABI (dex_gpu.h):
//
//	int  lux_gpu_dex_match_order(...)    // the matcher; rc 0 ok, -2 unsupported
//	const char* lux_gpu_backend_name()   // the runtime-selected backend name
//
// The kernel is the SINGLE-THREAD deterministic matcher whose output is byte-
// identical to MatchOrderCPU by construction. That contract is enforced on
// every run by orderbook_gpu_test.go and the determinism corpus in
// match_verify_test.go, and byte-verified live against the CPU authority by the
// GPU shadow gate in match_verify.go.
//
// CGo off → orderbook_nogpu.go provides a CPU-only MatchOrderGPU and a
// dexBackendLabel of "cpu (cgo disabled)". One and only one of each symbol per
// build; the seam is cgo vs !cgo, nothing finer.

/*
#cgo pkg-config: lux-gpu

#include <stdint.h>
#include "dex_gpu.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/luxfi/crypto/backend"
)

// MatchOrderGPU matches one incoming taker against a subset of book orders
// identified by bookIndices, dispatching to the runtime-selected GPU backend
// when available and falling back to MatchOrderCPU otherwise. Output is
// byte-equal across paths after padding normalization — that contract is
// enforced by orderbook_gpu_test.go on every run.
//
// This is the flat-buffer matching primitive used for parity testing, batch
// backtests, and future GPU-resident architectures. The live
// OrderBook.MatchOrders() engine retains its richer order-type and linked-list
// semantics and is NOT touched by this dispatch.
//
// Fallback policy (recorded via backend.RecordFallback):
//
//	GPU_DISABLE=1                     → reason "disabled"
//	lux_gpu_dex_match_order rc == -2  → reason "unsupported"
//	other rc                          → surfaced to caller (no fallback)
//
// Returns: trades emitted, remaining quantity on the incoming order.
// `book` may be mutated in place to reflect updated Remaining / Status.
func MatchOrderGPU(
	incoming *DEXOrder,
	book []DEXOrder,
	bookIndices []uint32,
	tradeIDBase uint64,
	timestamp uint64,
) ([]DEXTrade, uint64, error) {
	if incoming == nil {
		return nil, 0, fmt.Errorf("lx: MatchOrderGPU: nil incoming")
	}

	if backend.GPUDisabled() {
		backend.RecordFallback(backend.FallbackDisabled, "orderBook")
		t, r := MatchOrderCPU(incoming, book, bookIndices, tradeIDBase, timestamp)
		return t, r, nil
	}

	trades, remaining, err := gpuMatchOrder(incoming, book, bookIndices, tradeIDBase, timestamp)
	if err == errOrderBookGPUUnsupported {
		backend.RecordFallback(backend.FallbackUnsupported, "orderBook")
		t, r := MatchOrderCPU(incoming, book, bookIndices, tradeIDBase, timestamp)
		return t, r, nil
	}
	if err != nil {
		return nil, 0, err
	}
	return trades, remaining, nil
}

// errOrderBookGPUUnsupported is returned by gpuMatchOrder when liblux_gpu
// reports no usable device dispatch (rc -2). The dispatcher turns that into a
// transparent CPU fallback rather than surfacing it.
var errOrderBookGPUUnsupported = fmt.Errorf("lx: orderBook GPU dispatch not available on this host")

// dexBackendLabel reports the runtime-selected DEX GPU backend for corpus
// logging ("cuda" | "hip" | "metal" | "cpu"), as chosen by liblux_gpu.
func dexBackendLabel() string { return C.GoString(C.lux_gpu_backend_name()) }

// gpuMatchOrder calls the unified liblux_gpu matcher. Returns
// errOrderBookGPUUnsupported when the library reports rc -2 (no usable device)
// so the dispatcher falls back to MatchOrderCPU. Semantics are identical to the
// CPU oracle: byte-equal trades + remaining, with `book` mutated in place.
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

	rc := C.lux_gpu_dex_match_order(
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
		// Padding normalization to match the CPU oracle's ClearPadding output,
		// so raw bytewise comparisons (EncodeTrade/EncodeRow) are deterministic.
		// The unified driver already zeros the padding for emitted trades and
		// touched book rows; this is belt-and-braces against a future kernel
		// revision that changes the contract.
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
		return nil, 0, fmt.Errorf("lx: lux_gpu_dex_match_order rc=%d", int(rc))
	}
}
