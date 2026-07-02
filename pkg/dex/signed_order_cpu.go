// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// verifyOrdersCPU is the per-order ecrecover oracle used by the test as the
// reference and as the fallback for non-cgo / non-darwin builds. For each
// order it recomputes the signing hash, recovers the pubkey from the sig,
// derives the 20-byte address and compares with order.Sender.
func verifyOrdersCPU(orders []SignedOrder) ([]bool, error) {
	// Return nil (not an empty slice) for empty input so the CPU oracle matches
	// the luxcpp-backed batch path's contract (signed_order_gpu.go) byte-for-
	// byte at the boundary — TestBatchVerifyOrders_Empty pins this.
	if len(orders) == 0 {
		return nil, nil
	}
	out := make([]bool, len(orders))
	for i := range orders {
		hash, err := orders[i].SigningHash()
		if err != nil {
			out[i] = false
			continue
		}
		pub, err := crypto.Ecrecover(hash[:], orders[i].Sig[:])
		if err != nil || len(pub) != 65 {
			out[i] = false
			continue
		}
		// pub is the uncompressed pubkey (0x04 || X || Y). Address is
		// last 20 bytes of keccak256(X || Y).
		addr := common.BytesToAddress(crypto.Keccak256(pub[1:])[12:])
		out[i] = addr == orders[i].Sender
	}
	return out, nil
}
