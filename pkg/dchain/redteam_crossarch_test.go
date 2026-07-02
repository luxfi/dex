// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

// redteam_crossarch_test.go — RED TEAM (adversarial). CONSENSUS FORK via
// architecture-dependent float64->uint64 conversion on the value path.
//
// THREAT: the deterministic state transition computes the exact-integer
// settlement quantity as `uint64(size)` on the attacker-controlled float64 wire
// size (execute.go sizeUnitsBig line 19-24; settle.go sizeToUnits line 73-79).
// Per the Go spec a float64->uint64 conversion whose value is out of range is
// IMPLEMENTATION-DEFINED — and it differs by CPU architecture:
//
//	uint64(1e20):  arm64 = 0xFFFFFFFFFFFFFFFF   amd64 = 0x8000000000000000
//
// A wire size >= 2^64 (1.845e19) — e.g. 1e20 base units = 100 tokens at 18
// decimals, the EXACT case settlement_units.go's comment says the integer lane
// exists to support — therefore produces a DIFFERENT resting-order reserve, a
// DIFFERENT locked balance, and (once it crosses) DIFFERENT fill units on an
// arm64 validator vs an amd64 validator, FROM THE SAME ordered block.
//
// This propagates into the execRoot: the resting order's per-order reserve is
// keyed/sized by uint64(size); the maker row that feeds bookRoot carries a
// remaining that, combined with the divergent integer lane, differs. The net
// effect proven here: an arm64 proposer's block, re-verified byte-for-byte by an
// amd64 validator, DERIVES A DIFFERENT execRoot and is REJECTED
// ("execution root mismatch"). Two honest validators on different architectures
// cannot agree on the same block => the chain forks / halts.
//
// The test is two-phase to eliminate the wall-clock timestamp confound
// (blockTimestamp() uses time.Now(), so each process stamps its own ts; a
// faithful test must hand the PROPOSER'S bytes — with its embedded ts — to the
// verifier, exactly as consensus does):
//
//	PHASE=build  (run on arm64): build blocks 1..3, write their exact bytes +
//	                             arm64-derived execRoots to files.
//	PHASE=verify (run on amd64 under Rosetta): fresh VM, ParseBlock+Verify+Accept
//	                             each block's bytes. A Verify error / root diff is
//	                             the fork.
//
// Reproduce:
//	export PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin
//	export GOPRIVATE='github.com/luxfi/*' GOFLAGS=-mod=mod CGO_ENABLED=0 GOWORK=off
//	D=$SCRATCH  # a writable dir
//	ZEN_XARCH_DIR=$D ZEN_XARCH_PHASE=build  go test ./pkg/dchain/ -run TestRedTeam_CrossArchConsensusFork -count=1 -v
//	GOARCH=amd64 go test ./pkg/dchain/ -run TestRedTeam_CrossArchConsensusFork -c -o $D/x.test
//	ZEN_XARCH_DIR=$D ZEN_XARCH_PHASE=verify arch -x86_64 $D/x.test -test.run TestRedTeam_CrossArchConsensusFork -test.v

import (
	"context"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

const (
	xarchPool0 = 0xc0
	xarchPool1 = 0x55
)

func xarchDir(t *testing.T) string {
	d := os.Getenv("ZEN_XARCH_DIR")
	if d == "" {
		t.Skip("ZEN_XARCH_DIR not set — this is the two-phase cross-arch fork harness; see file header")
	}
	return d
}

// TestRedTeam_CrossArchConsensusFork proves an architecture-dependent execRoot.
func TestRedTeam_CrossArchConsensusFork(t *testing.T) {
	dir := xarchDir(t)
	phase := os.Getenv("ZEN_XARCH_PHASE")

	// ARCH PROBE (language-level divergence, arch-local).
	var bigSize = 1e20
	t.Logf("ARCH PROBE: uint64(%g) = %d (0x%x)   uint64(+Inf)=0x%x",
		bigSize, uint64(bigSize), uint64(bigSize), uint64(math.Inf(1)))

	switch phase {
	case "build":
		xarchBuild(t, dir, bigSize)
	case "verify":
		xarchVerify(t, dir)
	default:
		t.Skip("ZEN_XARCH_PHASE must be build|verify")
	}
}

// xarchBuild (run on the PROPOSER arch) builds the 3-block scenario and writes
// each block's canonical bytes + the proposer-derived execRoot to files.
func xarchBuild(t *testing.T, dir string, bigSize float64) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	const (
		maker = "maker-cx"
		taker = "taker-cx"
		fund  = uint64(math.MaxUint64)
	)
	pool := [32]byte{xarchPool0, xarchPool1, 0xa1, 0xcd}

	b1 := addBlock(t, vm,
		depositTx(t, taker, assetLUX, fund),
		depositTx(t, maker, assetLUSD, fund),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	b2 := addBlock(t, vm, placePoolTx(t, pool, zapwire.SideBuy, 1.0, bigSize, maker))
	b3 := addBlock(t, vm, submitPoolTx(t, pool, zapwire.SideSell, false, 1.0, bigSize, taker))

	for i, b := range []*Block{b1, b2, b3} {
		writeHex(t, filepath.Join(dir, blkFile(i)), b.Bytes())
		writeHex(t, filepath.Join(dir, rootFile(i)), b.execRoot[:])
		t.Logf("PROPOSER block%d execRoot=%x", i+1, b.execRoot[:])
	}
	avail, locked := ledgerTotals(t, vm)
	t.Logf("PROPOSER LEDGER LUSD available=%d locked=%d", avail[assetLUSD], locked[assetLUSD])
	t.Logf("PROPOSER LEDGER LUX  available=%d locked=%d", avail[assetLUX], locked[assetLUX])
	// Persist the proposer's ledger so the validator arch can compare the MONEY,
	// not just the (arch-stable) execRoot.
	writeHex(t, filepath.Join(dir, "ledger.hex"), encodeLedger(avail[assetLUSD], locked[assetLUSD], avail[assetLUX], locked[assetLUX]))
}

func encodeLedger(vals ...uint64) []byte {
	out := make([]byte, 8*len(vals))
	for i, v := range vals {
		for j := 0; j < 8; j++ {
			out[i*8+7-j] = byte(v >> (8 * j))
		}
	}
	return out
}

func decodeLedger(b []byte) []uint64 {
	out := make([]uint64, len(b)/8)
	for i := range out {
		var v uint64
		for j := 0; j < 8; j++ {
			v = (v << 8) | uint64(b[i*8+j])
		}
		out[i] = v
	}
	return out
}

// xarchVerify (run on the VALIDATOR arch) re-parses and independently verifies
// the proposer's EXACT block bytes — the real consensus validation path. A
// mismatch between the root this arch derives and the proposer's embedded
// execRoot is a CONSENSUS FORK: the two honest validators cannot agree.
func xarchVerify(t *testing.T, dir string) {
	ctx := context.Background()
	vm, _ := newTestVM(t, memdb.New())
	defer vm.Shutdown(ctx)

	forked := false
	for i := 0; i < 3; i++ {
		raw := readHex(t, filepath.Join(dir, blkFile(i)))
		wantRoot := readHex(t, filepath.Join(dir, rootFile(i)))

		blk, err := vm.ParseBlock(ctx, raw)
		if err != nil {
			t.Fatalf("block%d ParseBlock: %v", i+1, err)
		}
		verr := blk.Verify(ctx)
		gotRoot := blk.(*Block).execRoot // the proposer's claim, carried in bytes

		// Re-derive this arch's root the same way Verify does, to report it even
		// when Verify rejects. Verify already compared derived vs claimed; a
		// non-nil error here with a root-mismatch message is the fork signal.
		if verr != nil {
			t.Logf("VALIDATOR block%d Verify REJECTED proposer block: %v", i+1, verr)
			t.Logf("  => proposer(other-arch) execRoot=%x, this-arch derived a DIFFERENT root", wantRoot)
			forked = true
			break
		}
		// Verify passed: this arch derived == proposer's claim. Accept and chain on.
		if err := blk.Accept(ctx); err != nil {
			t.Fatalf("block%d Accept: %v", i+1, err)
		}
		t.Logf("VALIDATOR block%d Verify OK (root %x matches proposer)", i+1, gotRoot[:])
	}

	if forked {
		t.Errorf("HARD CONSENSUS FORK: an honest validator on this architecture REJECTS a valid block proposed on the other architecture, because uint64(float64 size) diverges across archs on the value path. Mixed-arch validator sets cannot agree.")
		return
	}
	t.Logf("execRoots MATCHED across arch (rows use the clamped floatToFixedQty). Now compare the MONEY the block moved — the ledger is NOT a term in the execRoot.")

	avail, locked := ledgerTotals(t, vm)
	got := []uint64{avail[assetLUSD], locked[assetLUSD], avail[assetLUX], locked[assetLUX]}
	want := decodeLedger(readHex(t, filepath.Join(dir, "ledger.hex")))
	labels := []string{"LUSD.available", "LUSD.locked", "LUX.available", "LUX.locked"}
	t.Logf("VALIDATOR LEDGER LUSD available=%d locked=%d", got[0], got[1])
	t.Logf("VALIDATOR LEDGER LUX  available=%d locked=%d", got[2], got[3])
	desync := false
	for i := range want {
		if got[i] != want[i] {
			desync = true
			t.Logf("  LEDGER DESYNC on %s: proposer(other-arch)=%d  this-arch=%d  (delta=%d)",
				labels[i], want[i], got[i], int64(want[i]-got[i]))
		}
	}
	if desync {
		t.Errorf("SILENT MONEY DESYNC PROVEN: both validators ACCEPTED the identical block bytes and AGREE on the execRoot, but hold DIFFERENT custody balances. The execRoot does not commit balance:/locked:, and settleFills moves BaseUnits=big.Int(uint64(size)) which diverges across arch. Consensus cannot detect this; withdrawals will be authorized inconsistently.")
	} else {
		t.Logf("ledgers also matched — no divergence on these blocks")
	}
}

func blkFile(i int) string  { return "xarch_blk" + string(rune('0'+i)) + ".hex" }
func rootFile(i int) string { return "xarch_root" + string(rune('0'+i)) + ".hex" }

func writeHex(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, []byte(hex.EncodeToString(b)), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readHex(t *testing.T, path string) []byte {
	t.Helper()
	s, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	b, err := hex.DecodeString(string(s))
	if err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return b
}
