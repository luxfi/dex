// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"encoding/hex"
	"testing"
	"time"

	"golang.org/x/crypto/sha3"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// The block root is pinned here the same way the genesis root is pinned in
// genesis_test.go, and for the same reason. Genesis was the fork that happened;
// every root above height 0 is the identical fork waiting to happen, because two
// nodes that compose the root differently disagree on the first block they build
// and neither can tell why.
//
// Determinism tests do not cover this. Comparing ComposeRoot against ComposeRoot
// moves both sides together, so a change to the composition keeps every such test
// green — which is exactly how the ledger operand landed in v1.14.0 unnoticed. A
// vector has to say what the value IS.
//
// blockRootVector's operands are all distinct and none is zero, so a reordering is
// caught as surely as an added or dropped term.
var blockRootVector = struct {
	parent, book, trade, txr, ledger [Size]byte
	height                           uint64
	root                             string
}{
	parent: [Size]byte{0x01, 0x11},
	book:   [Size]byte{0x02, 0x22},
	trade:  [Size]byte{0x03, 0x33},
	txr:    [Size]byte{0x04, 0x44},
	ledger: [Size]byte{0x05, 0x55},
	height: 0x0102030405060708,
	root:   "5120604309ac9adc9c63e4f5ac46ca81d543c787e2e41458c5d0f0a2e77ff79e",
}

// specBlockRoot recomputes a block root straight from the written construction:
//
//	BlockRoot = keccak256( "dex/block-root/v1" ‖ parent ‖ book ‖ trade ‖ tx ‖ ledger ‖ height_u64_le )
//
// Independent of the VM: a different keccak implementation and the concatenation
// spelled out, so the vector cannot be brought back into agreement by editing the
// code it checks.
func specBlockRoot(parent, book, trade, txr, ledger [Size]byte, height uint64) [Size]byte {
	var h [8]byte
	binary.LittleEndian.PutUint64(h[:], height)

	k := sha3.NewLegacyKeccak256()
	k.Write([]byte("dex/block-root/v1"))
	k.Write(parent[:])
	k.Write(book[:])
	k.Write(trade[:])
	k.Write(txr[:])
	k.Write(ledger[:])
	k.Write(h[:])

	var root [Size]byte
	copy(root[:], k.Sum(nil))
	return root
}

// TestBlockRootGolden pins the composition every root above height 0 is built
// from. A build that changes it forks the chain at the first block it produces.
func TestBlockRootGolden(t *testing.T) {
	v := blockRootVector
	got := ComposeRoot(v.parent, v.book, v.trade, v.txr, v.ledger, v.height)

	if want := specBlockRoot(v.parent, v.book, v.trade, v.txr, v.ledger, v.height); got != want {
		t.Fatalf("block root = %x, specification says %x", got, want)
	}
	if hex.EncodeToString(got[:]) != v.root {
		t.Fatalf("block root = %x, pinned %s\n"+
			"A node composing roots this way cannot agree with one that does not, from height 1 on.",
			got, v.root)
	}

	// Height is in the preimage and it is the operand a refactor drops or moves
	// without noticing: it is the only one that is not already a root, and every
	// other term is what the block committed. Both values are pinned so a change
	// cannot be absorbed by updating one of them.
	next := ComposeRoot(v.parent, v.book, v.trade, v.txr, v.ledger, v.height+1)
	if next == got {
		t.Fatal("the height operand does not reach the root")
	}
	if want := specBlockRoot(v.parent, v.book, v.trade, v.txr, v.ledger, v.height+1); next != want {
		t.Fatalf("block root at height+1 = %x, specification says %x", next, want)
	}

	// Every other operand must reach the root too, in its own position. Swapping
	// any pair of them is a different chain and must be a different root.
	for _, swap := range []struct {
		name string
		root [Size]byte
	}{
		{"parent<->book", ComposeRoot(v.book, v.parent, v.trade, v.txr, v.ledger, v.height)},
		{"trade<->tx", ComposeRoot(v.parent, v.book, v.txr, v.trade, v.ledger, v.height)},
		{"tx<->ledger", ComposeRoot(v.parent, v.book, v.trade, v.ledger, v.txr, v.height)},
	} {
		if swap.root == got {
			t.Errorf("%s: swapping two operands left the root unchanged", swap.name)
		}
	}
}

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
	r1, b1, tr1, x1 := ExecutionRoot(parent, rows, fills, txs, [Size]byte{}, 10)
	r2, b2, tr2, x2 := ExecutionRoot(parent, rows, fills, txs, [Size]byte{}, 10)

	if r1 != r2 || b1 != b2 || tr1 != tr2 || x1 != x2 {
		t.Fatal("ExecutionRoot not deterministic across two computations")
	}

	// The root chains the parent: a different parent -> different root.
	parent2 := [Size]byte{0x01}
	r3, _, _, _ := ExecutionRoot(parent2, rows, fills, txs, [Size]byte{}, 10)
	if r3 == r1 {
		t.Error("root did not change with a different parent (no history binding)")
	}

	// The root binds the height: a different height -> different root.
	r4, _, _, _ := ExecutionRoot(parent, rows, fills, txs, [Size]byte{}, 11)
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
	rootA, _, _, _ := ExecutionRoot(parent, rowsA, nil, txsA, [Size]byte{}, 1)

	// Different resting price -> different book root -> different exec root.
	txsB := []*Tx{placeTx(t, zapwire.SideSell, 102.0, 5.0, "maker-a")}
	_, rowsB := applyAll(t, 1, ts, txsB)
	rootB, _, _, _ := ExecutionRoot(parent, rowsB, nil, txsB, [Size]byte{}, 1)

	if rootA == rootB {
		t.Error("execution root identical for different book state")
	}
}

// TestEmptyRootsAreStable pins the empty sub-root values to merkle.EmptyRoot via
// the public ExecutionRoot path (no rows/trades/txs).
func TestEmptyRootsAreStable(t *testing.T) {
	var parent [Size]byte
	r1, b1, tr1, x1 := ExecutionRoot(parent, nil, nil, nil, [Size]byte{}, 0)
	r2, b2, tr2, x2 := ExecutionRoot(parent, nil, nil, nil, [Size]byte{}, 0)
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
