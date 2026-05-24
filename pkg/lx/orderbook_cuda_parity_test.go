// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/rand"
	"testing"
)

// TestMatchOrder_GPUMatchesCPU pins the field-equality contract between
// MatchOrderGPU and MatchOrderCPU. On hosts without CUDA the GPU path
// transparently falls back to CPU, so this test still passes (it proves
// CPU == CPU and exercises the dispatcher path). On Linux+NVIDIA hosts
// it proves the CUDA dispatch produces the same trade list and remaining
// quantity field-for-field.
//
// We compare via DEXTrade.EqualFields / DEXOrder.EqualFields, NOT `==`,
// so padding bytes are not allowed to influence the result. Padding is
// also explicitly cleared on both paths (host driver post-copy + Go
// MatchOrderCPU) so the production objects stay deterministic too.
func TestMatchOrder_GPUMatchesCPU(t *testing.T) {
	for seed := int64(1); seed <= 8; seed++ {
		seed := seed
		t.Run("", func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))

			incoming := DEXOrder{
				OrderID:   uint64(seed*1000 + 1),
				UserID:    1,
				Price:     DEXPrice{Integer: uint64(100 + rng.Intn(20)), Fraction: 0},
				Quantity:  uint64(50 + rng.Intn(450)),
				Timestamp: uint64(seed),
				Side:      DEXSideBid,
				Type:      DEXOrderLimit,
				Status:    DEXStatusOpen,
			}
			incoming.Remaining = incoming.Quantity

			bookSize := 16 + rng.Intn(64)
			book := make([]DEXOrder, bookSize)
			indices := make([]uint32, bookSize)
			for i := range book {
				book[i] = DEXOrder{
					OrderID:   uint64(seed*10000) + uint64(i),
					UserID:    2 + uint64(i)%4,
					Price:     DEXPrice{Integer: uint64(90 + rng.Intn(30)), Fraction: 0},
					Quantity:  uint64(10 + rng.Intn(100)),
					Timestamp: uint64(seed*100) + uint64(i),
					Side:      DEXSideAsk,
					Type:      DEXOrderLimit,
					Status:    DEXStatusOpen,
				}
				book[i].Remaining = book[i].Quantity
				indices[i] = uint32(i)
			}

			// Deep-copy book so CPU and GPU runs start from identical state.
			bookCPU := append([]DEXOrder(nil), book...)
			bookGPU := append([]DEXOrder(nil), book...)

			incomingCPU := incoming
			tradesCPU, remCPU := MatchOrderCPU(&incomingCPU, bookCPU, indices, 1_000_000, 9_999)

			incomingGPU := incoming
			tradesGPU, remGPU, err := MatchOrderGPU(&incomingGPU, bookGPU, indices, 1_000_000, 9_999)
			if err != nil {
				t.Fatalf("MatchOrderGPU error: %v", err)
			}

			if remCPU != remGPU {
				t.Fatalf("remaining mismatch: cpu=%d gpu=%d", remCPU, remGPU)
			}
			if len(tradesCPU) != len(tradesGPU) {
				t.Fatalf("trade count mismatch: cpu=%d gpu=%d", len(tradesCPU), len(tradesGPU))
			}
			for i := range tradesCPU {
				if !tradesCPU[i].EqualFields(tradesGPU[i]) {
					t.Fatalf("trade[%d] field mismatch:\n  cpu=%+v\n  gpu=%+v",
						i, tradesCPU[i], tradesGPU[i])
				}
			}
			for i := range bookCPU {
				if !bookCPU[i].EqualFields(bookGPU[i]) {
					t.Fatalf("book[%d] field mismatch:\n  cpu=%+v\n  gpu=%+v",
						i, bookCPU[i], bookGPU[i])
				}
			}
		})
	}
}
