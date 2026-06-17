// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

// dexvm_clob_parity_test.go — GPU-vs-CPU byte-parity KAT for the
// device-resident CLOB matcher.
//
// This is the load-bearing proof for the "full native GPU DEX, crypto in
// GPU memory" claim: the GPU CLOB path (CLOBMatch in dexvm_gpu.go, which
// dlsym's lux_<backend>_dex_clob_match and runs the matcher against a
// device-resident BookArena) MUST produce a 68-byte output that is
// byte-identical to the canonical CPU oracle for the same order sequence.
//
// The CPU oracle here is a self-contained Go port of the canonical C++
// oracle at
//   ~/work/lux-private/gpu-kernels/tools/kat/dex_cpu_oracle.hpp
//     ::clob_match_step_cpu
// which is itself the single source of truth that every GPU backend
// (cuda/hip/metal/vulkan/webgpu) and the C++ dex_match_cpu.cpp are held
// to. The Metal kernel
//   ~/work/lux-private/gpu-kernels/ops/dex/metal/clob_match.metal
//     ::clob_match_step
// implements the identical algorithm (same crossing rules, FIFO front-pop,
// price-sorted insertion: bids high→low, asks low→high; VWAP = 512/256
// shift-subtract long division). This test proves the Go bridge → plugin
// → kernel path reproduces that oracle byte-for-byte on real silicon.
//
// To run on a host with a built plugin (e.g. this Mac's Metal GPU):
//
//	export LUX_GPU_PLUGIN_DIR=/Users/z/work/luxcpp/gpu/build/metal_backend
//	export SDKROOT=$(xcrun --show-sdk-path)
//	GOWORK=off go test ./dexvm/ -run TestCLOBGPUvsCPUByteParity -v
//
// When no plugin is on disk (AutoBackend()==None), the test SKIPs with the
// reason — the parity assertion needs a real GPU plugin; the CPU oracle is
// still self-checked so a CPU-only host proves the oracle is internally
// consistent.

package lxgpu

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// =============================================================================
// Go CPU oracle — byte-equal port of dex_cpu_oracle.hpp::clob_match_step_cpu.
// uint256 is a 32-byte big-endian array, exactly the kernel-side layout.
// =============================================================================

type u256 = [32]byte

func u256FromU64(v uint64) u256 {
	var r u256
	for i := 0; i < 8; i++ {
		r[31-i] = byte(v >> (uint(i) * 8))
	}
	return r
}

func u256Cmp(a, b *u256) int {
	for i := 0; i < 32; i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	return 0
}

func u256IsZero(a *u256) bool {
	for _, b := range a {
		if b != 0 {
			return false
		}
	}
	return true
}

func u256Sub(a, b *u256) u256 {
	var r u256
	borrow := 0
	for i := 31; i >= 0; i-- {
		diff := int(a[i]) - int(b[i]) - borrow
		if diff < 0 {
			diff += 256
			borrow = 1
		} else {
			borrow = 0
		}
		r[i] = byte(diff)
	}
	return r
}

func u256AddInplace(dst, a *u256) {
	carry := 0
	for i := 31; i >= 0; i-- {
		sum := int(dst[i]) + int(a[i]) + carry
		dst[i] = byte(sum & 0xff)
		carry = sum >> 8
	}
}

// (outHi:outLo) = a * b (256x256 -> 512). Schoolbook 4x4 64-bit limbs over
// big-endian bytes — identical limb extraction to the C++/Metal body.
func u256Mul(a, b *u256) (outHi, outLo u256) {
	var al, bl [4]uint64
	for i := 0; i < 4; i++ {
		var va, vb uint64
		for j := 0; j < 8; j++ {
			va |= uint64(a[31-i*8-j]) << (uint(j) * 8)
			vb |= uint64(b[31-i*8-j]) << (uint(j) * 8)
		}
		al[i] = va
		bl[i] = vb
	}
	var r [8]uint64
	for i := 0; i < 4; i++ {
		var carry uint64
		for j := 0; j < 4; j++ {
			hi, lo := mulU64(al[i], bl[j])
			// cur = (hi:lo) + carry + r[i+j]
			t := lo + carry
			cf := uint64(0)
			if t < lo {
				cf = 1
			}
			sumLo := t + r[i+j]
			if sumLo < t {
				cf++
			}
			r[i+j] = sumLo
			carry = hi + cf
		}
		r[i+4] += carry
	}
	for i := 0; i < 4; i++ {
		vl := r[i]
		vh := r[i+4]
		for j := 0; j < 8; j++ {
			outLo[31-i*8-j] = byte(vl >> (uint(j) * 8))
			outHi[31-i*8-j] = byte(vh >> (uint(j) * 8))
		}
	}
	return outHi, outLo
}

// mulU64 returns the 128-bit product of a*b as (hi, lo) via 32-bit splits —
// matches dex_cpu_oracle.hpp::mul_u64_u64 limb-for-limb.
func mulU64(a, b uint64) (hi, lo uint64) {
	aLo, aHi := a&0xFFFFFFFF, a>>32
	bLo, bHi := b&0xFFFFFFFF, b>>32
	ll := aLo * bLo
	lh := aLo * bHi
	hl := aHi * bLo
	hh := aHi * bHi
	mid := (ll >> 32) + (lh & 0xFFFFFFFF) + (hl & 0xFFFFFFFF)
	lo = (ll & 0xFFFFFFFF) | (mid << 32)
	hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32)
	return hi, lo
}

// (outHi:outLo) += (aHi:aLo). 512-bit accumulator add.
func u512AddInplace(outHi, outLo, aHi, aLo *u256) {
	carry := 0
	for i := 31; i >= 0; i-- {
		sum := int(outLo[i]) + int(aLo[i]) + carry
		outLo[i] = byte(sum & 0xff)
		carry = sum >> 8
	}
	for i := 31; i >= 0; i-- {
		sum := int(outHi[i]) + int(aHi[i]) + carry
		outHi[i] = byte(sum & 0xff)
		carry = sum >> 8
	}
}

// u256VWAP = floor(notional / total) via 512/256 bit-by-bit long division.
func u256VWAP(notionalHi, notionalLo, total *u256) u256 {
	var quot u256
	if u256IsZero(total) {
		return quot
	}
	var rem u256
	for bit := 511; bit >= 0; bit-- {
		src := notionalLo
		if bit >= 256 {
			src = notionalHi
		}
		b := bit % 256
		byteIdx := 31 - (b / 8)
		shift := b % 8
		incBit := int((src[byteIdx] >> uint(shift)) & 1)
		// rem = (rem << 1) | incBit
		carry := incBit
		for i := 31; i >= 0; i-- {
			v := (int(rem[i]) << 1) | carry
			rem[i] = byte(v & 0xff)
			carry = (v >> 8) & 1
		}
		ge := u256Cmp(&rem, total) >= 0
		qbit := 0
		if ge {
			qbit = 1
			rem = u256Sub(&rem, total)
		}
		carry = qbit
		for i := 31; i >= 0; i-- {
			v := (int(quot[i]) << 1) | carry
			quot[i] = byte(v & 0xff)
			carry = (v >> 8) & 1
		}
	}
	return quot
}

// bookArena is the host-side mirror of the device-resident BookArena.
// Fixed-cap 1024 levels per side, price-sorted (bids high→low, asks low→high).
type bookArena struct {
	nBids, nAsks int
	bidPrice     [LuxCLOBMaxLevels]u256
	bidQty       [LuxCLOBMaxLevels]u256
	askPrice     [LuxCLOBMaxLevels]u256
	askQty       [LuxCLOBMaxLevels]u256
}

type clobOutput struct {
	filled   u256
	avgPrice u256
	numFills uint32
}

// clobMatchStepCPU is the byte-equal Go port of clob_match_step_cpu.
func clobMatchStepCPU(a *bookArena, side uint8, price, qty u256) clobOutput {
	remaining := qty
	var filled, notionalHi, notionalLo u256
	numFills := uint32(0)

	matchLoop := func(oppPrice, oppQty *[LuxCLOBMaxLevels]u256, nOpp *int, incomingIsBid bool) {
		for *nOpp > 0 && !u256IsZero(&remaining) {
			topPrice := oppPrice[0]
			topQty := oppQty[0]
			var crosses bool
			if incomingIsBid {
				crosses = u256Cmp(&topPrice, &price) <= 0 // ask price <= bid limit
			} else {
				crosses = u256Cmp(&price, &topPrice) <= 0 // ask limit <= bid price
			}
			if !crosses {
				break
			}
			fillQty := topQty
			if u256Cmp(&remaining, &topQty) < 0 {
				fillQty = remaining
			}
			addHi, addLo := u256Mul(&fillQty, &topPrice)
			u512AddInplace(&notionalHi, &notionalLo, &addHi, &addLo)
			u256AddInplace(&filled, &fillQty)
			remaining = u256Sub(&remaining, &fillQty)

			if u256Cmp(&fillQty, &topQty) < 0 {
				oppQty[0] = u256Sub(&topQty, &fillQty)
			} else {
				for i := 1; i < *nOpp; i++ {
					oppPrice[i-1] = oppPrice[i]
					oppQty[i-1] = oppQty[i]
				}
				*nOpp--
			}
			numFills++
		}
	}

	if side == 0 {
		matchLoop(&a.askPrice, &a.askQty, &a.nAsks, true)
		if !u256IsZero(&remaining) && a.nBids < LuxCLOBMaxLevels {
			pos := a.nBids
			for i := 0; i < a.nBids; i++ {
				if u256Cmp(&a.bidPrice[i], &price) < 0 {
					pos = i
					break
				}
			}
			for i := a.nBids; i > pos; i-- {
				a.bidPrice[i] = a.bidPrice[i-1]
				a.bidQty[i] = a.bidQty[i-1]
			}
			a.bidPrice[pos] = price
			a.bidQty[pos] = remaining
			a.nBids++
		}
	} else {
		matchLoop(&a.bidPrice, &a.bidQty, &a.nBids, false)
		if !u256IsZero(&remaining) && a.nAsks < LuxCLOBMaxLevels {
			pos := a.nAsks
			for i := 0; i < a.nAsks; i++ {
				if u256Cmp(&price, &a.askPrice[i]) < 0 {
					pos = i
					break
				}
			}
			for i := a.nAsks; i > pos; i-- {
				a.askPrice[i] = a.askPrice[i-1]
				a.askQty[i] = a.askQty[i-1]
			}
			a.askPrice[pos] = price
			a.askQty[pos] = remaining
			a.nAsks++
		}
	}

	return clobOutput{
		filled:   filled,
		avgPrice: u256VWAP(&notionalHi, &notionalLo, &filled),
		numFills: numFills,
	}
}

// encodeCLOBOutput packs a clobOutput into the 68-byte wire format the GPU
// kernel emits: filled(32 BE) | avg_price(32 BE) | num_fills(4 BE).
func encodeCLOBOutput(o clobOutput) [LuxCLOBOutLen]byte {
	var out [LuxCLOBOutLen]byte
	copy(out[0:32], o.filled[:])
	copy(out[32:64], o.avgPrice[:])
	binary.BigEndian.PutUint32(out[64:68], o.numFills)
	return out
}

// encodeCLOBCalldata builds the 117-byte EVM 0x100 calldata:
// side(1) | price(32 BE) | qty(32 BE) | user(20) | book_id(32).
func encodeCLOBCalldata(side uint8, price, qty u256, user [20]byte, bookID [32]byte) []byte {
	cd := make([]byte, LuxCLOBCalldataLen)
	cd[0] = side
	copy(cd[1:33], price[:])
	copy(cd[33:65], qty[:])
	copy(cd[65:85], user[:])
	copy(cd[LuxCLOBBookIDOffset:LuxCLOBBookIDOffset+LuxCLOBBookIDLen], bookID[:])
	return cd
}

// clobStep is one order in a parity sequence.
type clobStep struct {
	name  string
	side  uint8 // 0 = bid, 1 = ask
	price uint64
	qty   uint64
}

// =============================================================================
// Self-check of the Go oracle — runs on every host (no GPU needed).
// =============================================================================

// TestCLOBOracleSelfConsistent verifies the Go oracle against hand-computed
// values, so a CPU-only host still proves the oracle is internally correct
// before it is used as the parity reference.
func TestCLOBOracleSelfConsistent(t *testing.T) {
	var a bookArena

	// 1) Rest an ask: 100 @ price 10. No cross (empty book).
	out := clobMatchStepCPU(&a, 1, u256FromU64(10), u256FromU64(100))
	if out.numFills != 0 || !u256IsZero(&out.filled) {
		t.Fatalf("rest ask: expected no fill, got numFills=%d filled=%x", out.numFills, out.filled)
	}
	if a.nAsks != 1 || u256Cmp(&a.askPrice[0], &[32]byte{31: 10}) != 0 {
		t.Fatalf("rest ask: arena not updated: nAsks=%d", a.nAsks)
	}

	// 2) Incoming bid 60 @ limit 12 crosses the ask. Fill 60 @ 10.
	out = clobMatchStepCPU(&a, 0, u256FromU64(12), u256FromU64(60))
	if out.numFills != 1 {
		t.Fatalf("cross bid: numFills=%d, want 1", out.numFills)
	}
	if want := u256FromU64(60); u256Cmp(&out.filled, &want) != 0 {
		t.Fatalf("cross bid: filled=%x, want 60", out.filled)
	}
	// VWAP = 60*10 / 60 = 10.
	if want := u256FromU64(10); u256Cmp(&out.avgPrice, &want) != 0 {
		t.Fatalf("cross bid: avg=%x, want 10", out.avgPrice)
	}
	// Ask partially filled: 40 remains resting; nothing rests on bid side.
	if a.nAsks != 1 || a.nBids != 0 {
		t.Fatalf("cross bid: arena nAsks=%d nBids=%d, want 1/0", a.nAsks, a.nBids)
	}
	if want := u256FromU64(40); u256Cmp(&a.askQty[0], &want) != 0 {
		t.Fatalf("cross bid: residual ask qty=%x, want 40", a.askQty[0])
	}

	// 3) Big bid 200 @ limit 15: consumes the 40 remaining ask, residual 160
	//    rests as a bid. VWAP = 40*10/40 = 10.
	out = clobMatchStepCPU(&a, 0, u256FromU64(15), u256FromU64(200))
	if out.numFills != 1 {
		t.Fatalf("drain ask: numFills=%d, want 1", out.numFills)
	}
	if want := u256FromU64(40); u256Cmp(&out.filled, &want) != 0 {
		t.Fatalf("drain ask: filled=%x, want 40", out.filled)
	}
	if a.nAsks != 0 || a.nBids != 1 {
		t.Fatalf("drain ask: arena nAsks=%d nBids=%d, want 0/1", a.nAsks, a.nBids)
	}
	if want := u256FromU64(160); u256Cmp(&a.bidQty[0], &want) != 0 {
		t.Fatalf("drain ask: resting bid qty=%x, want 160", a.bidQty[0])
	}
}

// =============================================================================
// GPU-vs-CPU byte-parity KAT — needs a real plugin (LUX_GPU_PLUGIN_DIR).
// =============================================================================

// paritySequence is a fixed multi-order book that exercises: resting both
// sides, partial fills, full-level drain (front-pop), multi-level sweep, and
// non-trivial VWAP (averaging across two price levels).
func paritySequence() []clobStep {
	return []clobStep{
		{"rest ask 100@10", 1, 10, 100},
		{"rest ask 50@11", 1, 11, 50},
		{"rest ask 30@12", 1, 12, 30},
		{"rest bid 40@9", 0, 9, 40},
		// Crosses both 10 and 11 levels + part of 12: 100+50+30=180 available
		// up to limit 12; taker qty 170 → fills 100@10, 50@11, 20@12.
		// VWAP = (100*10 + 50*11 + 20*12) / 170 = (1000+550+240)/170 = 1790/170.
		{"sweep bid 170@12", 0, 12, 170},
		// Now asks: 10@12 left. Incoming ask 25@8 crosses resting bids
		// (the 40@9 + whatever rested). Tests the ask-side match loop.
		{"cross ask 25@8", 1, 8, 25},
		// Rest a fresh bid then drain it exactly.
		{"rest bid 15@7", 0, 7, 15},
		{"drain ask 15@7", 1, 7, 15},
		// No-cross resting order (limit far from book).
		{"rest ask 5@99", 1, 99, 5},
	}
}

// runParity drives one order sequence through the GPU plugin against a fresh
// device-resident arena, and asserts every 68-byte output is byte-identical
// to the Go CPU oracle stepping a parallel host-side arena. Returns the
// concatenated GPU outputs so the caller can prove determinism across arenas.
func runParity(t *testing.T, seq []clobStep) [][LuxCLOBOutLen]byte {
	t.Helper()

	arena, err := ArenaCreate()
	if err != nil {
		t.Fatalf("ArenaCreate: %v", err)
	}
	// Lifecycle: arena MUST be destroyed exactly once; verify no error.
	t.Cleanup(func() {
		if derr := ArenaDestroy(arena); derr != nil {
			t.Errorf("ArenaDestroy: %v", derr)
		}
	})

	var cpu bookArena
	var user [20]byte
	var bookID [32]byte
	bookID[31] = 1 // arbitrary stable book id

	outs := make([][LuxCLOBOutLen]byte, 0, len(seq))
	for _, s := range seq {
		price := u256FromU64(s.price)
		qty := u256FromU64(s.qty)

		calldata := encodeCLOBCalldata(s.side, price, qty, user, bookID)
		gpuOut, gpuNF, merr := CLOBMatch(arena, calldata)
		if merr != nil {
			t.Fatalf("CLOBMatch[%s]: %v", s.name, merr)
		}

		// CPU oracle steps its parallel arena.
		want := clobMatchStepCPU(&cpu, s.side, price, qty)
		wantBytes := encodeCLOBOutput(want)

		if !bytes.Equal(gpuOut[:], wantBytes[:]) {
			t.Errorf("BYTE PARITY MISMATCH at step %q:\n  GPU : %x\n  CPU : %x",
				s.name, gpuOut[:], wantBytes[:])
		}
		// num_fills mirror must also agree with both the 68-byte tail and CPU.
		if gpuNF != want.numFills {
			t.Errorf("num_fills mismatch at %q: GPU=%d CPU=%d", s.name, gpuNF, want.numFills)
		}
		if tailNF := binary.BigEndian.Uint32(gpuOut[64:68]); tailNF != gpuNF {
			t.Errorf("GPU num_fills tail %d != mirror %d at %q", tailNF, gpuNF, s.name)
		}
		outs = append(outs, gpuOut)
	}
	return outs
}

// TestCLOBGPUvsCPUByteParity is the headline proof: the device-resident GPU
// matcher returns byte-identical output to the canonical CPU oracle for a
// fixed multi-order book. Also proves arena lifecycle + determinism by
// replaying the same sequence on a second fresh arena and asserting identical
// GPU output across both runs.
func TestCLOBGPUvsCPUByteParity(t *testing.T) {
	backend := AutoBackend()
	if backend == GPUBackendNone {
		t.Skipf("no GPU plugin loaded — set LUX_GPU_PLUGIN_DIR to a dir containing "+
			"libluxgpu_backend_*.{dylib,so} (e.g. .../metal_backend). probe order: %v. "+
			"CPU oracle is still self-checked by TestCLOBOracleSelfConsistent.",
			probeOrder())
	}
	t.Logf("GPU backend=%s plugin=%s", backend, GPUPluginPath())

	seq := paritySequence()

	// Run 1: parity vs CPU oracle on a fresh arena.
	run1 := runParity(t, seq)

	// Run 2: same sequence, second fresh arena. Output MUST be identical to
	// run 1 byte-for-byte (deterministic, no cross-arena state bleed).
	run2 := runParity(t, seq)

	if len(run1) != len(run2) {
		t.Fatalf("determinism: run lengths differ %d vs %d", len(run1), len(run2))
	}
	for i := range run1 {
		if !bytes.Equal(run1[i][:], run2[i][:]) {
			t.Errorf("DETERMINISM MISMATCH at step %d (%s):\n  run1: %x\n  run2: %x",
				i, seq[i].name, run1[i][:], run2[i][:])
		}
	}
}

// TestCLOBArenaLifecycle exercises create → (no match) → destroy and the
// double-destroy no-op contract, independent of any matching. Skips without
// a plugin.
func TestCLOBArenaLifecycle(t *testing.T) {
	if AutoBackend() == GPUBackendNone {
		t.Skipf("no GPU plugin loaded — set LUX_GPU_PLUGIN_DIR (probe order: %v)", probeOrder())
	}
	arena, err := ArenaCreate()
	if err != nil {
		t.Fatalf("ArenaCreate: %v", err)
	}
	if arena == nil {
		t.Fatal("ArenaCreate returned nil arena with no error")
	}
	if err := ArenaDestroy(arena); err != nil {
		t.Fatalf("ArenaDestroy: %v", err)
	}
	// After destroy the handle is cleared; a second destroy is a safe no-op
	// (a.ptr == nil short-circuits before any plugin call).
	if err := ArenaDestroy(arena); err != nil {
		t.Fatalf("ArenaDestroy (second, should be no-op): %v", err)
	}
	// Matching against a destroyed arena must error, not crash.
	calldata := encodeCLOBCalldata(0, u256FromU64(1), u256FromU64(1), [20]byte{}, [32]byte{})
	if _, _, merr := CLOBMatch(arena, calldata); merr == nil {
		t.Fatal("CLOBMatch on destroyed arena: expected error, got nil")
	}
}
