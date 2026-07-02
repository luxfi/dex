// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

// three_mode_parity_test.go — proves the DEX produces IDENTICAL match outcomes
// across the three execution engines for the same order sequence:
//
//	1. pure-Go    : dex.OrderBook (the canonical Go/CPU reference matcher that
//	                lives in luxfi/dex — the d-chain's matcher)
//	2. CPU/C++    : orderBookMatchStepCPU — the Go port of the canonical C++ oracle
//	                (dex_cpu_oracle.hpp), uint256, byte-equal to the C++ backend
//	3. GPU        : OrderBookMatch via the loaded plugin (Metal locally), uint256,
//	                device-resident BookArena
//
// The three engines have DIFFERENT data models (dex.OrderBook is float/uint64
// and price-level aggregated; the GPU/oracle are uint256 and per-level arenas),
// so parity is asserted at the shared, model-independent abstraction every OrderBook
// must agree on: for each incoming order, the TOTAL FILLED QUANTITY and the
// volume-weighted average execution price (VWAP).
//
// Modes 2 and 3 are already proven byte-identical to each other by
// TestOrderBookGPUvsCPUByteParity. This test adds the third leg — the dex.OrderBook
// pure-Go matcher — and ties all three together.
//
// Run (Metal, this Mac):
//
//	export SDKROOT=$(xcrun --show-sdk-path)
//	export LUX_GPU_PLUGIN_DIR=/Users/z/work/luxcpp/gpu/build/metal_backend
//	GOWORK=off go test ./pkg/lxgpu/ -run TestThreeModeMatchParity -v

package lxgpu

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
)

// u256ToU64 collapses a uint256 to uint64, asserting the high 192 bits are
// zero. Every value in the parity scenarios fits in uint64 (that is the pure-Go
// book's native width), so this is lossless for the test inputs and lets us
// compare GPU/oracle output against the uint64 engine directly.
func u256ToU64(t *testing.T, v u256) uint64 {
	t.Helper()
	for i := 0; i < 24; i++ {
		if v[i] != 0 {
			t.Fatalf("u256 value exceeds uint64: %x", v)
		}
	}
	var r uint64
	for i := 24; i < 32; i++ {
		r = (r << 8) | uint64(v[i])
	}
	return r
}

// goOrderbookStep feeds one order into the canonical dex.OrderBook and returns
// the (filledQty, vwap) outcome for that order, computed from the trades it
// produced — the same (filled, vwap) abstraction the GPU/oracle report.
//
// userID is derived from the side so a taker never self-trades the resting
// makers (dex.OrderBook skips self-trades; the GPU/oracle have no owner
// concept). Two stable users — one per side — keep the models comparable. The
// order ID and timestamp are set explicitly (never auto-assigned) so the run is
// deterministic and independent of wall-clock time or process-local counters.
func goOrderbookStep(ob *dex.OrderBook, s orderBookStep, seq int64) (filled, vwap uint64) {
	side := dex.Side(s.side) // 0 = bid/buy, 1 = ask/sell
	user := "bid"            // bid maker/taker
	if side == dex.Sell {
		user = "ask" // ask user — distinct so crosses never self-trade
	}

	order := &dex.Order{
		ID:          uint64(seq), // explicit, distinct id per step
		Symbol:      "PARITY",
		Side:        side,
		Type:        dex.Limit,
		Price:       float64(s.price),
		Size:        float64(s.qty),
		User:        user,
		UserID:      user,
		TimeInForce: "GTC",
		Timestamp:   time.Unix(0, seq), // deterministic price-time priority
	}

	// Capture the trade-log tail BEFORE matching so we slice exactly the trades
	// this order produced. AddOrder matches on insert (continuous OrderBook) and
	// appends each fill to ob.Trades.
	start := len(ob.Trades)
	if ob.AddOrder(order) == 0 && order.Status == dex.Rejected {
		return 0, 0
	}

	var notional uint64
	for _, tr := range ob.Trades[start:] {
		q := uint64(tr.Size)
		filled += q
		notional += q * uint64(tr.Price)
	}
	if filled > 0 {
		vwap = notional / filled // floor division — matches the oracle's u256VWAP
	}
	return filled, vwap
}

// TestThreeModeMatchParity drives the fixed parity sequence through all three
// engines and asserts the (filled, vwap) outcome of every order is identical
// across pure-Go, the CPU oracle, and the GPU.
func TestThreeModeMatchParity(t *testing.T) {
	backend := AutoBackend()
	gpuActive := backend != GPUBackendNone
	if !gpuActive {
		t.Logf("no GPU plugin loaded (probe order %v) — running pure-Go vs CPU "+
			"oracle only; set LUX_GPU_PLUGIN_DIR for the GPU leg", probeOrder())
	} else {
		t.Logf("GPU backend=%s plugin=%s", backend, GPUPluginPath())
	}

	seq := paritySequence()

	// --- pure-Go engine (the canonical dex.OrderBook matcher) ---
	// EnableImmediateMatching makes AddOrder cross the incoming order against the
	// book and rest the residual — the same "match-on-insert, residual rests"
	// semantic the CPU oracle (orderBookMatchStepCPU) and the GPU kernel implement.
	ob := dex.NewOrderBook("PARITY")
	ob.EnableImmediateMatching = true
	goOut := make([][2]uint64, len(seq))
	for i, s := range seq {
		f, v := goOrderbookStep(ob, s, int64(i+1))
		goOut[i] = [2]uint64{f, v}
	}

	// --- CPU oracle engine ---
	var cpu bookArena
	cpuOut := make([][2]uint64, len(seq))
	for i, s := range seq {
		out := orderBookMatchStepCPU(&cpu, s.side, u256FromU64(s.price), u256FromU64(s.qty))
		cpuOut[i] = [2]uint64{u256ToU64(t, out.filled), u256ToU64(t, out.avgPrice)}
	}

	// --- GPU engine (optional) ---
	var gpuOut [][2]uint64
	if gpuActive {
		arena, err := ArenaCreate()
		if err != nil {
			t.Fatalf("ArenaCreate: %v", err)
		}
		t.Cleanup(func() {
			if derr := ArenaDestroy(arena); derr != nil {
				t.Errorf("ArenaDestroy: %v", derr)
			}
		})
		var user [20]byte
		var bookID [32]byte
		bookID[31] = 7
		gpuOut = make([][2]uint64, len(seq))
		for i, s := range seq {
			cd := encodeOrderBookCalldata(s.side, u256FromU64(s.price), u256FromU64(s.qty), user, bookID)
			out, _, merr := OrderBookMatch(arena, cd)
			if merr != nil {
				t.Fatalf("OrderBookMatch[%s]: %v", s.name, merr)
			}
			var filled, avg u256
			copy(filled[:], out[0:32])
			copy(avg[:], out[32:64])
			gpuOut[i] = [2]uint64{u256ToU64(t, filled), u256ToU64(t, avg)}
		}
	}

	// --- assert identical (filled, vwap) across all active engines ---
	for i, s := range seq {
		if goOut[i] != cpuOut[i] {
			t.Errorf("pure-Go vs CPU mismatch at %q: go=(filled=%d,vwap=%d) cpu=(filled=%d,vwap=%d)",
				s.name, goOut[i][0], goOut[i][1], cpuOut[i][0], cpuOut[i][1])
		}
		if gpuActive {
			if gpuOut[i] != cpuOut[i] {
				t.Errorf("GPU vs CPU mismatch at %q: gpu=(filled=%d,vwap=%d) cpu=(filled=%d,vwap=%d)",
					s.name, gpuOut[i][0], gpuOut[i][1], cpuOut[i][0], cpuOut[i][1])
			}
			if gpuOut[i] != goOut[i] {
				t.Errorf("GPU vs pure-Go mismatch at %q: gpu=(filled=%d,vwap=%d) go=(filled=%d,vwap=%d)",
					s.name, gpuOut[i][0], gpuOut[i][1], goOut[i][0], goOut[i][1])
			}
		}
		t.Logf("%-22s filled/vwap  go=%d/%d cpu=%d/%d%s", s.name,
			goOut[i][0], goOut[i][1], cpuOut[i][0], cpuOut[i][1],
			gpuLogSuffix(gpuActive, gpuOut, i))
	}
}

func gpuLogSuffix(active bool, gpuOut [][2]uint64, i int) string {
	if !active {
		return "  gpu=SKIP"
	}
	return sprintGPU(gpuOut[i])
}

func sprintGPU(v [2]uint64) string {
	return "  gpu=" + itoa(v[0]) + "/" + itoa(v[1])
}

func itoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
