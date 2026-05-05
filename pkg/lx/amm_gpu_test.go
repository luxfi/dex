// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/rand"
	"testing"
)

// TestBatchEvalConstantProduct_GPUMatchesCPU dispatches 1000 random
// (reserve_x, reserve_y, amount) tuples and asserts the GPU output is
// byte-equal to the in-process CPU oracle. We use a fixed seed so a failure
// reproduces with the index printed.
func TestBatchEvalConstantProduct_GPUMatchesCPU(t *testing.T) {
	const N = 1000
	r := rand.New(rand.NewSource(1))

	reserves := make([]ReservePair, N)
	amounts := make([]uint64, N)
	for i := 0; i < N; i++ {
		// Keep reserves in [1, 2^48) and amount in [0, 2^32) so
		// amount*reserve_y fits in 80 bits — well inside the kernel's
		// 128-bit numerator path but big enough to exercise the
		// non-trivial divide branch.
		reserves[i] = ReservePair{
			ReserveX: r.Uint64()%(1<<48) + 1,
			ReserveY: r.Uint64()%(1<<48) + 1,
		}
		amounts[i] = r.Uint64() % (1 << 32)
	}

	cpu, err := BatchEvalConstantProductCPU(reserves, amounts)
	if err != nil {
		t.Fatalf("CPU oracle: %v", err)
	}
	gpu, err := BatchEvalConstantProduct(reserves, amounts)
	if err != nil {
		t.Fatalf("GPU batch: %v", err)
	}
	if len(cpu) != N || len(gpu) != N {
		t.Fatalf("unexpected output lengths cpu=%d gpu=%d", len(cpu), len(gpu))
	}
	for i := 0; i < N; i++ {
		if cpu[i] != gpu[i] {
			t.Fatalf("mismatch at %d: rx=%d ry=%d a=%d cpu=%d gpu=%d",
				i, reserves[i].ReserveX, reserves[i].ReserveY,
				amounts[i], cpu[i], gpu[i])
		}
	}
}

// TestBatchEvalConstantProduct_Edge covers the known degenerate inputs:
// zero reserves, zero amount, very small pools.
func TestBatchEvalConstantProduct_Edge(t *testing.T) {
	reserves := []ReservePair{
		{ReserveX: 0, ReserveY: 0},
		{ReserveX: 1, ReserveY: 1},
		{ReserveX: 1_000_000, ReserveY: 1_000_000},
		{ReserveX: 0, ReserveY: 100},
	}
	amounts := []uint64{0, 1, 1_000_000, 0}
	want := []uint64{
		0,
		1 / 2,
		(1_000_000 * 1_000_000) / (1_000_000 + 1_000_000),
		0,
	}
	got, err := BatchEvalConstantProduct(reserves, amounts)
	if err != nil {
		t.Fatalf("BatchEvalConstantProduct: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestBatchEvalConstantProduct_LengthMismatch verifies the input contract.
func TestBatchEvalConstantProduct_LengthMismatch(t *testing.T) {
	if _, err := BatchEvalConstantProduct(
		[]ReservePair{{1, 1}, {2, 2}},
		[]uint64{1},
	); err == nil {
		t.Fatal("expected error for length mismatch, got nil")
	}
}
