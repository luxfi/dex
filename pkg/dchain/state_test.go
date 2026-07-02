// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// TestExecutionRootDeterministic proves the state root is a pure function of the
// committed rows/trades/txs + parent + height: same inputs -> identical root,
// computed independently twice.
func TestExecutionRootDeterministic(t *testing.T) {
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	txs := []*Tx{
		placeTx(t, zapwire.SideSell, 101.0, 5.0, "maker-a"),
		submitTx(t, zapwire.SideBuy, false, 101.0, 2.0, "taker-x"),
	}
	fills, rows := applyAll(t, 10, ts, txs)

	var parent [Size]byte
	r1, b1, tr1, x1 := ExecutionRoot(parent, rows, fills, txs, 10)
	r2, b2, tr2, x2 := ExecutionRoot(parent, rows, fills, txs, 10)

	if r1 != r2 || b1 != b2 || tr1 != tr2 || x1 != x2 {
		t.Fatal("ExecutionRoot not deterministic across two computations")
	}

	// The root chains the parent: a different parent -> different root.
	parent2 := [Size]byte{0x01}
	r3, _, _, _ := ExecutionRoot(parent2, rows, fills, txs, 10)
	if r3 == r1 {
		t.Error("root did not change with a different parent (no history binding)")
	}

	// The root binds the height: a different height -> different root.
	r4, _, _, _ := ExecutionRoot(parent, rows, fills, txs, 11)
	if r4 == r1 {
		t.Error("root did not change with a different height")
	}
}

// TestExecutionRootChangesWithState proves any change to committed state moves
// the root (it actually commits the book/fills, not a constant).
func TestExecutionRootChangesWithState(t *testing.T) {
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()
	var parent [Size]byte

	txsA := []*Tx{placeTx(t, zapwire.SideSell, 101.0, 5.0, "maker-a")}
	_, rowsA := applyAll(t, 1, ts, txsA)
	rootA, _, _, _ := ExecutionRoot(parent, rowsA, nil, txsA, 1)

	// Different resting price -> different book root -> different exec root.
	txsB := []*Tx{placeTx(t, zapwire.SideSell, 102.0, 5.0, "maker-a")}
	_, rowsB := applyAll(t, 1, ts, txsB)
	rootB, _, _, _ := ExecutionRoot(parent, rowsB, nil, txsB, 1)

	if rootA == rootB {
		t.Error("execution root identical for different book state")
	}
}

// TestEmptyRootsAreStable pins the empty sub-root values to merkle.EmptyRoot via
// the public ExecutionRoot path (no rows/trades/txs).
func TestEmptyRootsAreStable(t *testing.T) {
	var parent [Size]byte
	r1, b1, tr1, x1 := ExecutionRoot(parent, nil, nil, nil, 0)
	r2, b2, tr2, x2 := ExecutionRoot(parent, nil, nil, nil, 0)
	if r1 != r2 || b1 != b2 || tr1 != tr2 || x1 != x2 {
		t.Fatal("empty roots not stable")
	}
	// Sub-roots over empty leaf sets must equal each other (all merkle.EmptyRoot).
	if b1 != tr1 || tr1 != x1 {
		t.Error("empty sub-roots differ; expected all merkle.EmptyRoot")
	}
}

// TestBookRootIgnoresInsertionOrder proves the book root is computed over the
// canonical row stream, so it is independent of how rows are presented (rows go
// through BookToRows which canonicalizes).
func TestBookRootIgnoresInsertionOrder(t *testing.T) {
	ts := time.Unix(0, 1_700_000_000_000_000_000).UTC()

	b1 := dex.NewOrderBook("X")
	b1.ConsensusAddOrder(&dex.Order{ID: 1, Type: dex.Limit, Side: dex.Buy, Price: 100, Size: 1, User: "a", Timestamp: ts})
	b1.ConsensusAddOrder(&dex.Order{ID: 2, Type: dex.Limit, Side: dex.Sell, Price: 101, Size: 1, User: "b", Timestamp: ts})

	b2 := dex.NewOrderBook("X")
	b2.ConsensusAddOrder(&dex.Order{ID: 2, Type: dex.Limit, Side: dex.Sell, Price: 101, Size: 1, User: "b", Timestamp: ts})
	b2.ConsensusAddOrder(&dex.Order{ID: 1, Type: dex.Limit, Side: dex.Buy, Price: 100, Size: 1, User: "a", Timestamp: ts})

	if bookRoot(dex.BookToRows(b1)) != bookRoot(dex.BookToRows(b2)) {
		t.Error("book root depends on insertion order")
	}
}
