// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

package lx

import (
	"fmt"

	"github.com/luxfi/crypto/backend"
)

// MatchOrderGPU matches one incoming taker against a subset of book
// orders identified by bookIndices, dispatching to the host GPU when
// available and falling back to MatchOrderCPU otherwise. Output is
// byte-equal across paths after padding normalization — that contract is
// enforced by orderbook_gpu_test.go on every run.
//
// This is the flat-buffer matching primitive used for parity testing,
// batch backtests, and future GPU-resident architectures. The live
// OrderBook.MatchOrders() engine retains its richer order-type and
// linked-list semantics and is NOT touched by this dispatch.
//
// Backend selection at compile time (CGo on):
//
//	linux  → orderbook_gpu_cuda.go → CUDA
//	other  → orderbook_gpu_stub.go → returns errOrderBookGPUUnsupported (caller falls back to CPU)
//
// Fallback policy (recorded via backend.RecordFallback):
//
//	GPU_DISABLE=1                  → reason "disabled"
//	gpuMatchOrder returns sentinel → reason "unsupported"
//	other gpuMatchOrder error      → surfaced to caller (no fallback)
//
// CGo off → orderbook_nogpu.go provides a CPU-only implementation of this
// entry point. One and only one MatchOrderGPU symbol per build.
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

// errOrderBookGPUUnsupported is returned by the per-platform gpuMatchOrder
// when no GPU dispatch is available on this host. The dispatcher above
// turns that into a transparent CPU fallback rather than surfacing it.
var errOrderBookGPUUnsupported = fmt.Errorf("lx: orderBook GPU dispatch not available on this host")
