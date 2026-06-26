// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"bytes"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/lx"
)

// replay_test.go proves the matcher-at-Verify determinism: the SAME swap over the
// SAME prior state on TWO independently-constructed stores produces byte-identical
// ledger state. This is the property that lets the cEVM block commit the route by a
// single state root — every validator re-runs the swap and derives the same result,
// no fill vector carried in the block bytes.

// buildSeededEconomy constructs an identical starting economy on the given store:
// a market, a maker resting a bid, and a funded taker. Returns the pool id.
func buildSeededEconomy(t *testing.T, db Store, tag byte) [32]byte {
	t.Helper()
	pid := seedMarket(t, db, tag, tokLETH, tokLUSD)
	maker := account(0xAA)
	taker := account(0xBB)
	seedDeposit(t, db, maker, tokLUSD, 1_000_000)
	restOrder(t, db, pid, maker, lx.Buy, 50, 1000, 1)
	seedDeposit(t, db, taker, tokLETH, 500)
	return pid
}

// snapshotAllRows dumps every (key,value) under the dex ledger/book prefixes
// into a sorted, canonical byte image — the cross-store equality witness. Two stores
// with the same image hold byte-identical dex state.
func snapshotAllRows(t *testing.T, db *memdb.Database) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, prefix := range []string{
		prefixBalance, prefixLocked, prefixOrder, prefixBook, prefixMarket,
		prefixAsset, prefixOrderLock, prefixOrderUser,
	} {
		it := db.NewIteratorWithPrefix([]byte(prefix))
		for it.Next() {
			buf.Write(it.Key())
			buf.WriteByte(0)
			buf.Write(it.Value())
			buf.WriteByte(0)
		}
		it.Release()
	}
	return buf.Bytes()
}

// TestReplay_SameInputSameStateAllValidators runs the identical swap on two
// independently-built stores and asserts byte-identical resulting state.
func TestReplay_SameInputSameStateAllValidators(t *testing.T) {
	const N = 3
	images := make([][]byte, N)
	for v := 0; v < N; v++ {
		db := memdb.New()
		pid := buildSeededEconomy(t, db, 30)
		req := SwapRequest{
			PoolID: pid, TakerUser: account(0xBB), Side: lx.Sell,
			Base: tokLETH, Quote: tokLUSD, AmountIn: 137, OrderID: 5_000, TimestampN: 9_000,
			Class: ClassPublicDEX,
		}
		res, err := ExecuteSwap(db, orderBookRouter(), req)
		if err != nil {
			t.Fatalf("validator %d ExecuteSwap: %v", v, err)
		}
		if res.AmountOut != 137*50 {
			t.Fatalf("validator %d AmountOut = %d, want %d", v, res.AmountOut, 137*50)
		}
		images[v] = snapshotAllRows(t, db)
	}
	for v := 1; v < N; v++ {
		if !bytes.Equal(images[0], images[v]) {
			t.Fatalf("validator %d state differs from validator 0 (non-deterministic replay)", v)
		}
	}
}

// TestReplay_ExecutionRootDeterministic asserts the execution root over the same
// resulting rows/fills is identical across independent computations (the cEVM block
// state-root analog), and that a different fill set yields a different root (so a
// fabricated fill cannot reproduce the honest root).
func TestReplay_ExecutionRootDeterministic(t *testing.T) {
	db1 := memdb.New()
	db2 := memdb.New()
	pid := buildSeededEconomy(t, db1, 31)
	_ = buildSeededEconomy(t, db2, 31)

	req := SwapRequest{
		PoolID: pid, TakerUser: account(0xBB), Side: lx.Sell,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 200, OrderID: 5_000, TimestampN: 9_000,
		Class: ClassPublicDEX,
	}
	r1, err := ExecuteSwap(db1, orderBookRouter(), req)
	if err != nil {
		t.Fatalf("db1: %v", err)
	}
	r2, err := ExecuteSwap(db2, orderBookRouter(), req)
	if err != nil {
		t.Fatalf("db2: %v", err)
	}

	root1 := SwapExecutionRoot(db1, pid, r1)
	root2 := SwapExecutionRoot(db2, pid, r2)
	if root1 != root2 {
		t.Fatalf("execution roots differ across independent stores: %x vs %x", root1[:8], root2[:8])
	}

	// A fabricated fill (one extra unit credited) must change the root — the basis
	// for rejecting a Byzantine proposer who claims a fill the matcher would not
	// produce.
	fakeRes := *r1
	fakeRes.AmountOut = r1.AmountOut + 1
	fakeRoot := SwapExecutionRoot(db1, pid, &fakeRes)
	if fakeRoot == root1 {
		t.Fatalf("a fabricated fill produced the SAME root — fake fills would be undetectable")
	}
}
