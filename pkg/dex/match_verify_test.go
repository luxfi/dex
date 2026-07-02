// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// match_verify_test.go is the real-hardware determinism corpus + the fail-closed
// negative test for the GPU shadow gate.
//
// The corpus builds a battery of order books (empty, partial, full cross, multi-
// level sweep, price barriers, max depth, randomized) and asserts the GPU match
// result is BYTE-IDENTICAL to the CPU oracle (MatchOrderCPU) across the full fill
// set + resulting book + remaining quantity. On a host with a real GPU kernel
// linked (Metal on Apple, CUDA on NVIDIA) these assertions run the kernel on the
// device; on a host without one, MatchOrderGPU transparently falls back to the
// CPU oracle and the corpus still proves the dispatch path. The test prints which
// backend actually ran so "real GPU vs simulated" is never ambiguous in the log.

// ---- flat corpus -------------------------------------------------------------

type dexScenario struct {
	name     string
	incoming DEXOrder
	book     []DEXOrder
	indices  []uint32
}

func bid(id, qty uint64, price uint64) DEXOrder {
	o := DEXOrder{OrderID: id, UserID: id, Price: DEXPrice{Integer: price}, Quantity: qty,
		Side: DEXSideBid, Type: DEXOrderLimit, Status: DEXStatusOpen}
	o.Remaining = qty
	return o
}

func ask(id, qty uint64, price uint64) DEXOrder {
	o := DEXOrder{OrderID: id, UserID: id, Price: DEXPrice{Integer: price}, Quantity: qty,
		Side: DEXSideAsk, Type: DEXOrderLimit, Status: DEXStatusOpen}
	o.Remaining = qty
	return o
}

func idxOf(n int) []uint32 {
	idx := make([]uint32, n)
	for i := range idx {
		idx[i] = uint32(i)
	}
	return idx
}

// fixedScenarios are the hand-curated edge cases the prompt calls out: empty
// book, single full/partial fill, multi-level sweep, price barrier, no cross,
// exact-cross boundary, and a max-depth book that crosses past a single GPU
// block (the case the legacy parallel kernel got nondeterministically wrong).
func fixedScenarios() []dexScenario {
	taker := func(id, qty, price uint64) DEXOrder {
		o := bid(id, qty, price)
		return o
	}
	var s []dexScenario

	s = append(s, dexScenario{"empty_book", taker(1, 100, 100), nil, nil})

	s = append(s, dexScenario{"single_full_fill",
		taker(1, 50, 100), []DEXOrder{ask(10, 50, 100)}, idxOf(1)})

	s = append(s, dexScenario{"maker_partial",
		taker(1, 40, 100), []DEXOrder{ask(10, 100, 100)}, idxOf(1)})

	s = append(s, dexScenario{"taker_partial_remainder",
		taker(1, 50, 100), []DEXOrder{ask(10, 30, 100)}, idxOf(1)})

	s = append(s, dexScenario{"multi_level_sweep",
		taker(1, 45, 102),
		[]DEXOrder{ask(10, 10, 100), ask(11, 20, 101), ask(12, 30, 102)}, idxOf(3)})

	s = append(s, dexScenario{"price_barrier_stops_sweep",
		taker(1, 50, 101),
		[]DEXOrder{ask(10, 10, 100), ask(11, 20, 105)}, idxOf(2)})

	s = append(s, dexScenario{"no_cross",
		taker(1, 50, 100), []DEXOrder{ask(10, 10, 110)}, idxOf(1)})

	// Exact-cross boundary in Q64.64: a bid at exactly the ask price crosses
	// (>=); a hair below does not.
	atE := DEXOrder{OrderID: 10, UserID: 10, Price: DEXPrice{Integer: 100, Fraction: 0},
		Quantity: 10, Remaining: 10, Side: DEXSideAsk, Type: DEXOrderLimit, Status: DEXStatusOpen}
	exactBid := DEXOrder{OrderID: 1, UserID: 1, Price: DEXPrice{Integer: 100, Fraction: 0},
		Quantity: 10, Remaining: 10, Side: DEXSideBid, Type: DEXOrderLimit, Status: DEXStatusOpen}
	belowBid := exactBid
	belowBid.Price = DEXPrice{Integer: 99, Fraction: ^uint64(0)} // 99.9999… < 100
	s = append(s, dexScenario{"exact_cross", exactBid, []DEXOrder{atE}, idxOf(1)})
	s = append(s, dexScenario{"one_ulp_below_no_cross", belowBid, []DEXOrder{atE}, idxOf(1)})

	// Max depth: 300 asks at strictly increasing prices, a taker that sweeps the
	// first ~200. Beyond 256 entries this is the regime where the legacy parallel
	// CUDA kernel (block(256), per-block shared remaining, atomicAdd trade slots)
	// produced nondeterministic / racing output. The deterministic single-thread
	// kernel + the CPU oracle agree byte-for-byte.
	const depth = 300
	deep := make([]DEXOrder, depth)
	for i := 0; i < depth; i++ {
		deep[i] = ask(uint64(1000+i), 5, uint64(100+i))
	}
	s = append(s, dexScenario{"max_depth_sweep",
		taker(1, 5*200+3, uint64(100+250)), deep, idxOf(depth)})

	// A sell taker against bids (mirror side).
	sellTaker := DEXOrder{OrderID: 1, UserID: 1, Price: DEXPrice{Integer: 98}, Quantity: 25,
		Remaining: 25, Side: DEXSideAsk, Type: DEXOrderLimit, Status: DEXStatusOpen}
	s = append(s, dexScenario{"sell_taker_sweeps_bids",
		sellTaker,
		[]DEXOrder{bid(10, 10, 100), bid(11, 20, 99), bid(12, 30, 97)}, idxOf(3)})

	// Market taker (price ignored; sweeps unconditionally). The flat ABI models a
	// market order as a max-price taker.
	mkt := DEXOrder{OrderID: 1, UserID: 1, Price: DEXPrice{Integer: ^uint64(0), Fraction: ^uint64(0)},
		Quantity: 1000, Remaining: 1000, Side: DEXSideBid, Type: DEXOrderMarket, Status: DEXStatusOpen}
	s = append(s, dexScenario{"market_sweep",
		mkt, []DEXOrder{ask(10, 10, 100), ask(11, 20, 200), ask(12, 30, 300)}, idxOf(3)})

	return s
}

// randomScenario builds a randomized cross for the given seed: a bid taker
// against a book of random-priced asks (some crossable, some not), random sizes.
func randomScenario(seed int64) dexScenario {
	rng := rand.New(rand.NewSource(seed))
	n := 8 + rng.Intn(120)
	book := make([]DEXOrder, n)
	for i := 0; i < n; i++ {
		book[i] = ask(uint64(seed*10000)+uint64(i), uint64(1+rng.Intn(100)), uint64(90+rng.Intn(40)))
	}
	taker := bid(uint64(seed*1000+1), uint64(1+rng.Intn(2000)), uint64(100+rng.Intn(30)))
	return dexScenario{fmt.Sprintf("rand_seed_%d", seed), taker, book, idxOf(n)}
}

// runFlatParity is the core assertion: GPU == CPU, byte-identical, over fills +
// remaining + resulting book. Returns the trade count + the backend label so the
// caller can log whether a real device ran.
func runFlatParity(t *testing.T, sc dexScenario) (int, string) {
	t.Helper()
	bookCPU := append([]DEXOrder(nil), sc.book...)
	bookGPU := append([]DEXOrder(nil), sc.book...)
	incCPU := sc.incoming
	incGPU := sc.incoming

	cpuTr, cpuRem := MatchOrderCPU(&incCPU, bookCPU, sc.indices, 1_000_000, 7_777)
	gpuTr, gpuRem, err := MatchOrderGPU(&incGPU, bookGPU, sc.indices, 1_000_000, 7_777)
	if err != nil {
		t.Fatalf("%s: MatchOrderGPU error: %v", sc.name, err)
	}
	if reason, ok := flatResultsEqual(cpuTr, cpuRem, bookCPU, gpuTr, gpuRem, bookGPU); !ok {
		t.Fatalf("%s: GPU != CPU byte-identity FAILED: %s\n cpu=%+v\n gpu=%+v",
			sc.name, reason, cpuTr, gpuTr)
	}
	return len(cpuTr), dexBackendLabel()
}

func TestDEXShadow_GPUMatchesCPU_Corpus(t *testing.T) {
	scenarios := fixedScenarios()
	for seed := int64(1); seed <= 64; seed++ {
		scenarios = append(scenarios, randomScenario(seed))
	}

	backend := ""
	totalFills := 0
	for _, sc := range scenarios {
		n, b := runFlatParity(t, sc)
		totalFills += n
		backend = b
	}
	t.Logf("DEX determinism corpus: %d scenarios, %d total fills, GPU==CPU byte-identical, backend=%s",
		len(scenarios), totalFills, backend)
}

// ---- end-to-end gate through the consensus path ------------------------------

func tsAt(i int) time.Time {
	return time.Unix(0, 1_700_000_000_000_000_000).UTC().Add(time.Duration(i) * time.Millisecond)
}

// restMaker rests a limit order on a fresh consensus book.
func restMaker(t *testing.T, ob *OrderBook, id uint64, side Side, price, size float64, user string, i int) {
	t.Helper()
	o := &Order{ID: id, Type: Limit, Side: side, Price: price, Size: size,
		User: user, UserID: user, Symbol: ob.Symbol, Timestamp: tsAt(i)}
	if ob.ConsensusAddOrder(o) == 0 {
		t.Fatalf("ConsensusAddOrder rejected maker id=%d", id)
	}
}

// TestDEXGate_EndToEnd_Verified drives a realistic multi-level cross through the
// consensus entrypoint with the GPU shadow ON and asserts the gate recorded a
// verified match with zero divergences and zero errors.
func TestDEXGate_EndToEnd_Verified(t *testing.T) {
	prev := SetMatchEngineForTest(EngineGPUVerified)
	defer SetMatchEngineForTest(prev)
	resetMatchShadowStats()

	ob := NewOrderBook("BTC-USD")
	restMaker(t, ob, 1, Sell, 101.0, 5.0, "maker-a", 0)
	restMaker(t, ob, 2, Sell, 101.5, 3.0, "maker-b", 1)
	restMaker(t, ob, 3, Sell, 102.0, 4.0, "maker-c", 2)

	taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 102.0, Size: 7.0,
		User: "taker-x", UserID: "taker-x", Symbol: "BTC-USD", Timestamp: tsAt(5)}
	fills, err := ob.SubmitMarketableVerified(taker, EngineGPUVerified)
	if err != nil {
		t.Fatalf("SubmitMarketableVerified: %v", err)
	}
	if len(fills) == 0 {
		t.Fatal("expected fills")
	}
	st := MatchShadowSnapshot()
	if st.Verified == 0 {
		t.Errorf("expected a verified shadow match, got %+v", st)
	}
	if st.Divergences != 0 || st.Errors != 0 {
		t.Errorf("GPU shadow diverged on a clean cross: %+v", st)
	}
	t.Logf("end-to-end verified cross: %d fills, stats=%+v, backend=%s", len(fills), st, dexBackendLabel())
}

// TestDEXGate_NoFork_CPUvsGPU is the anti-fork property: the COMMITTED fills are
// byte-identical whether a validator runs EngineCPU or EngineGPUVerified. A mixed
// validator set therefore cannot fork on a marketable cross.
func TestDEXGate_NoFork_CPUvsGPU(t *testing.T) {
	build := func() *OrderBook {
		ob := NewOrderBook("BTC-USD")
		restMaker(t, ob, 1, Sell, 101.0, 5.0, "maker-a", 0)
		restMaker(t, ob, 2, Sell, 101.5, 3.0, "maker-b", 1)
		restMaker(t, ob, 3, Sell, 102.0, 4.0, "maker-c", 2)
		return ob
	}
	newTaker := func() *Order {
		return &Order{ID: 9, Type: Limit, Side: Buy, Price: 102.0, Size: 7.0,
			User: "taker-x", UserID: "taker-x", Symbol: "BTC-USD", Timestamp: tsAt(5)}
	}

	cpuFills, err := build().SubmitMarketableVerified(newTaker(), EngineCPU)
	if err != nil {
		t.Fatal(err)
	}
	gpuFills, err := build().SubmitMarketableVerified(newTaker(), EngineGPUVerified)
	if err != nil {
		t.Fatal(err)
	}
	if len(cpuFills) != len(gpuFills) {
		t.Fatalf("committed fill count differs: cpu=%d gpu=%d", len(cpuFills), len(gpuFills))
	}
	for i := range cpuFills {
		a := EncodeTrade(TradeToRow(cpuFills[i], Buy))
		b := EncodeTrade(TradeToRow(gpuFills[i], Buy))
		if !bytes.Equal(a, b) {
			t.Fatalf("committed fill[%d] differs between EngineCPU and EngineGPUVerified", i)
		}
	}
}

// TestDEXGate_GPUMatchesSubmitMarketable proves the headline equivalence on a
// clean (no self-trade) cross: the flat GPU result, projected to canonical fill
// rows, is byte-identical to what the rich SubmitMarketable authority committed.
func TestDEXGate_GPUMatchesSubmitMarketable(t *testing.T) {
	ob := NewOrderBook("BTC-USD")
	restMaker(t, ob, 1, Sell, 100.0, 10.0, "m1", 0)
	restMaker(t, ob, 2, Sell, 101.0, 20.0, "m2", 1)
	restMaker(t, ob, 3, Sell, 102.0, 30.0, "m3", 2)

	// Capture the inputs the way the gate does, BEFORE the cross.
	taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 102.0, Size: 45.0,
		User: "tk", UserID: "tk", Symbol: "BTC-USD", Timestamp: tsAt(5)}

	// Snapshot via the same path the gate uses.
	ob.mu.Lock()
	taker.Status = Open
	taker.RemainingSize = taker.Size
	shadow := ob.captureDEXShadowLocked(taker)
	ob.mu.Unlock()

	// Authority.
	richFills, err := ob.SubmitMarketable(taker)
	if err != nil {
		t.Fatal(err)
	}

	// Flat GPU on the captured inputs.
	bookGPU := append([]DEXOrder(nil), shadow.book...)
	inc := shadow.incoming
	gpuTr, _, err := MatchOrderGPU(&inc, bookGPU, shadow.indices, shadow.tradeBase, shadow.timestamp)
	if err != nil {
		t.Fatal(err)
	}

	if reason, ok := flatReproducesRich(gpuTr, 0, richFills, Buy); !ok {
		t.Fatalf("GPU flat result != rich SubmitMarketable: %s\n rich=%+v\n gpu=%+v",
			reason, richFills, gpuTr)
	}
	t.Logf("GPU flat == rich SubmitMarketable byte-identical over %d fills (backend=%s)",
		len(richFills), dexBackendLabel())
}

// ---- the fail-closed negative test -------------------------------------------

// corruptGPUMatch is a deliberately-divergent "GPU": it runs the real CPU oracle
// then mutates the first trade so the result differs from the authority by one
// field. It models a buggy kernel / driver regression.
func corruptGPUMatch(incoming *DEXOrder, book []DEXOrder, idx []uint32, base, ts uint64) ([]DEXTrade, uint64, error) {
	tr, rem := MatchOrderCPU(incoming, book, idx, base, ts)
	if len(tr) > 0 {
		tr[0].Quantity++ // a single-unit lie: enough to fork a chain
	}
	return tr, rem, nil
}

// errGPUMatch models a GPU dispatch failure.
func errGPUMatch(incoming *DEXOrder, book []DEXOrder, idx []uint32, base, ts uint64) ([]DEXTrade, uint64, error) {
	return nil, 0, fmt.Errorf("simulated GPU driver failure")
}

// TestDEXGate_FailClosed_DivergentGPU is the load-bearing safety test: when the
// GPU produces a result that differs from the CPU authority, the gate (1) commits
// the CPU authority UNCHANGED, (2) records a divergence, (3) does NOT record a
// verified match. A divergent GPU result is structurally impossible to commit.
func TestDEXGate_FailClosed_DivergentGPU(t *testing.T) {
	prevEng := SetMatchEngineForTest(EngineGPUVerified)
	defer SetMatchEngineForTest(prevEng)

	// Capture the authority result with the GPU OFF (the ground truth).
	want := func() []Trade {
		ob := NewOrderBook("BTC-USD")
		restMaker(t, ob, 1, Sell, 100.0, 10.0, "m1", 0)
		restMaker(t, ob, 2, Sell, 101.0, 20.0, "m2", 1)
		taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 101.0, Size: 25.0,
			User: "tk", UserID: "tk", Symbol: "BTC-USD", Timestamp: tsAt(5)}
		f, _ := ob.SubmitMarketable(taker)
		return f
	}()

	// Now run the SAME cross with a corrupt GPU installed.
	saved := dexGPUMatch
	dexGPUMatch = corruptGPUMatch
	defer func() { dexGPUMatch = saved }()
	resetMatchShadowStats()

	ob := NewOrderBook("BTC-USD")
	restMaker(t, ob, 1, Sell, 100.0, 10.0, "m1", 0)
	restMaker(t, ob, 2, Sell, 101.0, 20.0, "m2", 1)
	taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 101.0, Size: 25.0,
		User: "tk", UserID: "tk", Symbol: "BTC-USD", Timestamp: tsAt(5)}

	got, err := ob.SubmitMarketableVerified(taker, EngineGPUVerified)
	if err != nil {
		t.Fatal(err)
	}

	// (1) The committed fills are the CPU authority's, NOT the corrupt GPU's.
	if len(got) != len(want) {
		t.Fatalf("committed fill count changed under corrupt GPU: got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if !bytes.Equal(EncodeTrade(TradeToRow(got[i], Buy)), EncodeTrade(TradeToRow(want[i], Buy))) {
			t.Fatalf("corrupt GPU result LEAKED into committed fill[%d]", i)
		}
	}
	// (2) a divergence was recorded, (3) no verified match was recorded.
	st := MatchShadowSnapshot()
	if st.Divergences == 0 {
		t.Errorf("corrupt GPU did not register a divergence: %+v", st)
	}
	if st.Verified != 0 {
		t.Errorf("corrupt GPU was wrongly counted as verified: %+v", st)
	}
	t.Logf("fail-closed proven: corrupt GPU discarded, CPU authority committed, stats=%+v", st)
}

// TestDEXGate_FailClosed_GPUError proves a GPU dispatch error also fails closed:
// the CPU authority is committed and the error is recorded, never surfaced.
func TestDEXGate_FailClosed_GPUError(t *testing.T) {
	saved := dexGPUMatch
	dexGPUMatch = errGPUMatch
	defer func() { dexGPUMatch = saved }()
	resetMatchShadowStats()

	ob := NewOrderBook("BTC-USD")
	restMaker(t, ob, 1, Sell, 100.0, 10.0, "m1", 0)
	taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 100.0, Size: 10.0,
		User: "tk", UserID: "tk", Symbol: "BTC-USD", Timestamp: tsAt(5)}

	got, err := ob.SubmitMarketableVerified(taker, EngineGPUVerified)
	if err != nil {
		t.Fatalf("a GPU dispatch error must not surface to the caller: %v", err)
	}
	if len(got) != 1 || got[0].Size != 10.0 {
		t.Fatalf("CPU authority result wrong under GPU error: %+v", got)
	}
	st := MatchShadowSnapshot()
	if st.Errors == 0 {
		t.Errorf("GPU error not recorded: %+v", st)
	}
}

// TestDEXGate_SelfCrossStaysSafe proves the consensus-safety gate (GPU flat ==
// CPU flat) holds even on a self-trade-prevention cross — the flat primitive
// does not model STP, so it is recorded as a model divergence, but the GPU and
// CPU FLAT paths still agree, and the rich authority (which DOES apply STP) is
// what gets committed.
func TestDEXGate_SelfCrossStaysSafe(t *testing.T) {
	prevEng := SetMatchEngineForTest(EngineGPUVerified)
	defer SetMatchEngineForTest(prevEng)
	resetMatchShadowStats()

	ob := NewOrderBook("BTC-USD")
	// Two asks from the SAME user as the taker -> rich matcher skips-and-cancels.
	restMaker(t, ob, 1, Sell, 100.0, 10.0, "same", 0)
	restMaker(t, ob, 2, Sell, 101.0, 20.0, "other", 1)
	taker := &Order{ID: 9, Type: Limit, Side: Buy, Price: 101.0, Size: 25.0,
		User: "same", UserID: "same", Symbol: "BTC-USD", Timestamp: tsAt(5)}

	if _, err := ob.SubmitMarketableVerified(taker, EngineGPUVerified); err != nil {
		t.Fatal(err)
	}
	st := MatchShadowSnapshot()
	// The byte-for-byte consensus-safety gate (GPU flat == CPU flat) must NOT
	// have diverged; STP shows up only as a (non-consensus) model divergence.
	if st.Divergences != 0 {
		t.Errorf("GPU-vs-CPU flat gate diverged on a self-cross (must not): %+v", st)
	}
	t.Logf("self-cross: GPU==CPU flat gate held; model divergence=%d (expected, STP not modeled by flat primitive)",
		st.ModelDivergences)
}
