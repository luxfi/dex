// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

package lx

// BatchVerifyOrders verifies a batch of SignedOrders, returning a per-order
// validity bool. On non-cgo builds we run the per-order CPU oracle; the
// luxcpp-backed batch path is in signed_order_gpu.go.
func BatchVerifyOrders(orders []SignedOrder) ([]bool, error) {
	return verifyOrdersCPU(orders)
}
