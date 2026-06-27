// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

package lx

import "fmt"

// dexBackendLabel reports the linked DEX GPU backend for corpus logging.
func dexBackendLabel() string { return "cpu (cgo disabled)" }

// MatchOrderGPU under !cgo is the CPU oracle. The cgo build provides this
// symbol via orderbook_gpu.go (which dispatches to the per-platform CUDA /
// stub implementation).
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
	t, r := MatchOrderCPU(incoming, book, bookIndices, tradeIDBase, timestamp)
	return t, r, nil
}
