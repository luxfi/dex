// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo || !linux

package lx

// gpuMatchOrder has no GPU backend on non-Linux or no-cgo builds. The
// dispatcher in orderbook_gpu.go turns errOrderBookGPUUnsupported into a
// transparent fallback to MatchOrderCPU.
func gpuMatchOrder(
	incoming *DEXOrder,
	book []DEXOrder,
	bookIndices []uint32,
	tradeIDBase uint64,
	timestamp uint64,
) ([]DEXTrade, uint64, error) {
	return nil, 0, errOrderBookGPUUnsupported
}
