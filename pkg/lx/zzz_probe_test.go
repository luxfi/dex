// TEMPORARY probe — NOT for commit. Times single-shot Metal dispatch
// (under the ~600-call failure ceiling) to get honest per-call latency
// and per-eval throughput for the regime where the driver actually works.
package lx

import (
	"math/rand"
	"testing"
	"time"
)

func TestProbeAMMMetalSingleShot(t *testing.T) {
	for _, n := range []int{1024, 16384, 262144, 1048576, 4194304} {
		rng := rand.New(rand.NewSource(1))
		reserves := make([]ReservePair, n)
		amounts := make([]uint64, n)
		for i := 0; i < n; i++ {
			reserves[i] = ReservePair{ReserveX: rng.Uint64()%(1<<48) + 1, ReserveY: rng.Uint64()%(1<<48) + 1}
			amounts[i] = rng.Uint64() % (1 << 32)
		}
		// 50 dispatches (well under the ~600 ceiling), report median-ish via min.
		const iters = 50
		best := time.Hour
		var sum time.Duration
		ok := true
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			if _, err := BatchEvalConstantProduct(reserves, amounts); err != nil {
				t.Logf("N=%d FAILED at iter %d: %v", n, i, err)
				ok = false
				break
			}
			d := time.Since(t0)
			sum += d
			if d < best {
				best = d
			}
		}
		if ok {
			avg := sum / iters
			t.Logf("GPU N=%-8d best=%-10v avg=%-10v evals/sec(best)=%.3g evals/sec(avg)=%.3g",
				n, best, avg,
				float64(n)/best.Seconds(), float64(n)/avg.Seconds())
		}
	}
}

func TestProbeAMMCPUSingleShot(t *testing.T) {
	for _, n := range []int{1024, 16384, 262144, 1048576, 4194304} {
		rng := rand.New(rand.NewSource(1))
		reserves := make([]ReservePair, n)
		amounts := make([]uint64, n)
		for i := 0; i < n; i++ {
			reserves[i] = ReservePair{ReserveX: rng.Uint64()%(1<<48) + 1, ReserveY: rng.Uint64()%(1<<48) + 1}
			amounts[i] = rng.Uint64() % (1 << 32)
		}
		const iters = 50
		best := time.Hour
		var sum time.Duration
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			if _, err := BatchEvalConstantProductCPU(reserves, amounts); err != nil {
				t.Fatalf("cpu N=%d: %v", n, err)
			}
			d := time.Since(t0)
			sum += d
			if d < best {
				best = d
			}
		}
		avg := sum / iters
		t.Logf("CPU N=%-8d best=%-10v avg=%-10v evals/sec(best)=%.3g evals/sec(avg)=%.3g",
			n, best, avg,
			float64(n)/best.Seconds(), float64(n)/avg.Seconds())
	}
}
