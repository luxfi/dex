// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"os"
	"sync"
	"sync/atomic"
)

// VerifyEngine selects the matcher acceleration mode for the consensus match
// path. It is a NODE-LOCAL choice, never a chain parameter, because the gate is
// designed so the COMMITTED result is identical for every value of this enum:
// the GPU never produces a value that reaches state. That property — proven by
// the shadow gate below — is exactly why a validator set mixing GPU-verified and
// CPU-only nodes cannot fork.
//
//	EngineCPU         the matcher runs the pure-Go authority only. Default.
//	EngineGPUVerified the matcher ALSO dispatches the GPU kernel on the exact
//	                  pre-cross inputs, byte-compares the GPU fills + book delta
//	                  against the CPU authority, records the outcome, and returns
//	                  the CPU authority's result REGARDLESS. A GPU result that
//	                  differs from the CPU by a single byte is recorded as a
//	                  divergence and discarded (fail-closed to CPU); it can never
//	                  be committed.
type VerifyEngine int32

const (
	EngineCPU VerifyEngine = iota
	EngineGPUVerified
)

func (e VerifyEngine) String() string {
	switch e {
	case EngineGPUVerified:
		return "gpu-verified"
	default:
		return "cpu"
	}
}

// envMatchEngine is the environment variable a node operator sets to opt a
// validator into running the GPU matcher as a checked shadow. Absent / "cpu"
// (the default) means the GPU is never dispatched in the match path. Only
// "gpu-verified" turns the shadow on. There is deliberately NO "gpu-trusted"
// mode: a mode that commits GPU bytes without the CPU cross-check is out of
// scope until the kernels have accumulated enough live divergence-free evidence,
// and would be a separate, explicitly-named enum value.
const envMatchEngine = "LUX_DEX_MATCH_ENGINE"

var (
	matchEngineOnce sync.Once
	matchEngine     VerifyEngine
)

// MatchEngine returns the process-wide match engine, parsed once from the
// environment. Default EngineCPU. This is read by the consensus match path
// (applySubmit -> SubmitMarketableVerified) on every cross; the value is a local
// accelerator toggle, not consensus state.
func MatchEngine() VerifyEngine {
	matchEngineOnce.Do(func() {
		switch os.Getenv(envMatchEngine) {
		case "gpu-verified", "gpu_verified", "gpu":
			matchEngine = EngineGPUVerified
		default:
			matchEngine = EngineCPU
		}
	})
	return matchEngine
}

// SetMatchEngineForTest overrides the parsed engine. Test-only: it bypasses the
// sync.Once so a test can drive both branches deterministically. Returns the
// previous value so the test can restore it.
func SetMatchEngineForTest(e VerifyEngine) VerifyEngine {
	matchEngineOnce.Do(func() {}) // consume the Once so MatchEngine() won't re-parse
	prev := VerifyEngine(atomic.LoadInt32((*int32)(&matchEngine)))
	atomic.StoreInt32((*int32)(&matchEngine), int32(e))
	return prev
}

// --- Shadow-gate metrics (the "loud metric") ----------------------------------

// MatchShadowStats is the running tally of GPU shadow-gate outcomes. These are
// the consensus-safety telemetry an operator scrapes: Divergences MUST stay
// zero. A non-zero Divergences/Errors count means a GPU kernel or driver
// produced a result that differed from the CPU authority and was (correctly)
// discarded — a signal to disable the GPU engine on that node and investigate.
type MatchShadowStats struct {
	// Verified counts crosses where the GPU result was byte-identical to the
	// CPU authority (the happy path; the GPU genuinely ran and agreed).
	Verified uint64
	// Divergences counts crosses where the GPU fills/book-delta differed from
	// the CPU authority. The CPU result was committed; the GPU result discarded.
	Divergences uint64
	// Errors counts crosses where the GPU dispatch itself failed (driver error,
	// allocation failure). The CPU result was committed.
	Errors uint64
	// ModelDivergences counts crosses where the flat GPU/CPU primitive could not
	// reproduce the rich authority's fills (e.g. self-trade-prevention paths the
	// flat primitive does not model). This is a fidelity signal only: the CPU
	// authority is always committed, and the GPU-vs-CPU-flat gate is unaffected.
	ModelDivergences uint64
}

var (
	statVerified  atomic.Uint64
	statDivergent atomic.Uint64
	statGPUError  atomic.Uint64
	statModelDiff atomic.Uint64
)

// MatchShadowSnapshot returns a snapshot of the shadow-gate counters.
func MatchShadowSnapshot() MatchShadowStats {
	return MatchShadowStats{
		Verified:         statVerified.Load(),
		Divergences:      statDivergent.Load(),
		Errors:           statGPUError.Load(),
		ModelDivergences: statModelDiff.Load(),
	}
}

// resetMatchShadowStats zeroes the counters. Test-only.
func resetMatchShadowStats() {
	statVerified.Store(0)
	statDivergent.Store(0)
	statGPUError.Store(0)
	statModelDiff.Store(0)
}

// --- Divergence sink (the "loud log") -----------------------------------------

// MatchDivergence is the payload describing a shadow-gate event where the GPU
// did NOT agree with the CPU authority. It is handed to the divergence sink so a
// node can surface it at ERROR (with full context for forensics) while the chain
// keeps running on the CPU authority.
type MatchDivergence struct {
	Kind   string // "dex" (order-book match shadow-gate divergence)
	Reason string // "gpu_ne_cpu" | "gpu_error" | "model"
	Detail string // human-readable specifics (first differing field / index)
}

var (
	sinkMu       sync.RWMutex
	divergenceFn = defaultDivergenceSink
)

// defaultDivergenceSink writes a single loud line to stderr. It exists so a
// divergence is never silently swallowed even before the node wires its logger.
func defaultDivergenceSink(d MatchDivergence) {
	// One line, easy to grep/alert on. os.Stderr.Write is allocation-light and
	// safe to call from the match path (divergence is, by design, never the
	// happy path — the kernels are KAT-byte-identical to the CPU oracle).
	os.Stderr.WriteString("LUX-DEX GPU SHADOW DIVERGENCE kind=" + d.Kind +
		" reason=" + d.Reason + " detail=" + d.Detail + "\n")
}

// SetMatchDivergenceSink installs a sink for shadow-gate divergence events (the
// dchain VM routes it to the node logger at ERROR). Passing nil restores the
// default stderr sink. Returns no value; idempotent and safe to call at startup.
func SetMatchDivergenceSink(fn func(MatchDivergence)) {
	sinkMu.Lock()
	defer sinkMu.Unlock()
	if fn == nil {
		divergenceFn = defaultDivergenceSink
		return
	}
	divergenceFn = fn
}

// emitDivergence records the metric and fires the sink. Called only off the
// happy path.
func emitDivergence(d MatchDivergence) {
	switch d.Reason {
	case "gpu_error":
		statGPUError.Add(1)
	case "model":
		statModelDiff.Add(1)
	default:
		statDivergent.Add(1)
	}
	sinkMu.RLock()
	fn := divergenceFn
	sinkMu.RUnlock()
	if fn != nil {
		fn(d)
	}
}
