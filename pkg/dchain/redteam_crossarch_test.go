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
	agA, agL := ledgerTotals(t, vm)
	t.Logf("PROPOSER AGGREGATE LUSD avail=%d locked=%d (conserved sum)", agA[assetLUSD], agL[assetLUSD])
	pa := perAccount(t, vm, maker, taker)
	logAccounts(t, "PROPOSER", pa)
	// Persist the proposer's PER-ACCOUNT ledger so the validator arch can compare
	// the actual withdrawable money, not just the (arch-stable) aggregate/root.
	writeHex(t, filepath.Join(dir, "ledger.hex"), encodeLedger(pa...))
}

// perAccount returns [makerLUX.avail, makerLUX.locked, makerLUSD.avail,
// makerLUSD.locked, takerLUX.avail, takerLUX.locked, takerLUSD.avail,
// takerLUSD.locked] — the withdrawable/escrowed money each account actually holds.
func perAccount(t *testing.T, vm *VM, maker, taker string) []uint64 {
	t.Helper()
	g := func(u string, a [32]byte, locked bool) uint64 {
		var v uint64
		var err error
		if locked {
			v, err = getLocked(vm.db, userKey16(u), a)
		} else {
			v, err = getAvailable(vm.db, userKey16(u), a)
		}
		if err != nil {
			t.Fatalf("read balance: %v", err)
		}
		return v
	}
	return []uint64{
		g(maker, assetLUX, false), g(maker, assetLUX, true),
		g(maker, assetLUSD, false), g(maker, assetLUSD, true),
		g(taker, assetLUX, false), g(taker, assetLUX, true),
		g(taker, assetLUSD, false), g(taker, assetLUSD, true),
	}
}

func logAccounts(t *testing.T, who string, v []uint64) {
	t.Helper()
	t.Logf("%s MAKER  LUX(avail=%d locked=%d) LUSD(avail=%d locked=%d)", who, v[0], v[1], v[2], v[3])
	t.Logf("%s TAKER  LUX(avail=%d locked=%d) LUSD(avail=%d locked=%d)", who, v[4], v[5], v[6], v[7])
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
	t.Logf("execRoots MATCHED across arch (rows use the clamped floatToFixedQty). Now compare the PER-ACCOUNT withdrawable MONEY — the ledger is NOT a term in the execRoot.")

	agA, agL := ledgerTotals(t, vm)
	t.Logf("VALIDATOR AGGREGATE LUSD avail=%d locked=%d (conserved sum — matches proposer, MASKS the split)", agA[assetLUSD], agL[assetLUSD])
	got := perAccount(t, vm, "maker-cx", "taker-cx")
	logAccounts(t, "VALIDATOR", got)
	want := decodeLedger(readHex(t, filepath.Join(dir, "ledger.hex")))
	labels := []string{"maker.LUX.avail", "maker.LUX.locked", "maker.LUSD.avail", "maker.LUSD.locked",
		"taker.LUX.avail", "taker.LUX.locked", "taker.LUSD.avail", "taker.LUSD.locked"}
	desync := false
	for i := range want {
		if got[i] != want[i] {
			desync = true
			t.Logf("  PER-ACCOUNT DESYNC on %s: proposer(other-arch)=%d  this-arch=%d  (delta=%d)",
				labels[i], want[i], got[i], int64(want[i]-got[i]))
		}
	}
	if desync {
		t.Errorf("SILENT MONEY DESYNC PROVEN: both validators ACCEPTED the identical block bytes and AGREE on the execRoot, yet a given account holds DIFFERENT withdrawable balances on the two architectures. The execRoot does not commit balance:/locked:, and settleFills moves BaseUnits=big.Int(uint64(size)), which diverges across arch. Conservation (aggregate sum) still holds, so no mint — but the PER-ACCOUNT split forks: a withdrawal is authorized for MaxUint64 on one arch and ~half that on the other. Consensus cannot detect this.")
	} else {
		t.Logf("per-account ledgers also matched — no divergence on these blocks")
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
