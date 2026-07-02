// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// redteam_restart_refutation_test.go — RED TEAM verification of the "restart forks
// execRoot" hypothesis at the REAL VM consensus boundary (not the persist-layer
// helper functions in isolation).
//
// The hypothesis (from the persist.go floatToFixedQty non-idempotence): a resting
// order with a fractional wire size whose round(size*1e8) drifts by 1 on the
// fixedQtyToFloat reload makes a RESTARTED validator derive a different bookRoot /
// execRoot than a non-restarted one, forking the set.
//
// This test REFUTES that at the integration level. Block.execute rebuilds every
// touched market's book FROM THE COMMITTED DB ROWS via rebuildBookFromDB on EVERY
// block (block.go: "the SAME function used at startup ... and during execution"),
// on every node, restart or not. So the in-RAM wire float only reaches bookRoot in
// the order's PLACE block (which all nodes process from identical tx bytes); every
// subsequent block re-derives the same reloaded float on all nodes. A restarted
// validator therefore VERIFIES a continuing proposer's post-place block cleanly —
// no root divergence, no fork.
package dchain

import (
	"context"
	"math"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

func TestRedteam_RestartDoesNotForkExecRoot(t *testing.T) {
	db := memdb.New()
	vm1, _ := newTestVM(t, db) // the CONTINUING (never-restarted) proposer
	defer vm1.Shutdown(context.Background())

	const seller = "seller"
	pool := [32]byte{0xf0, 0x2c} // "fork?"

	// The exact wire size whose 1e8-fixed image is non-idempotent under reload
	// (Q=4159461352541877 -> reload -> Q'=4159461352541878). If ANYTHING forks on
	// restart, this size triggers it.
	wireSize := math.Float64frombits(0x4183d577ac340ec1) // ~41,594,613.525419

	// Fund + open market + rest a SELL of the fractional size (locks base=uint64(size)).
	addBlock(t, vm1,
		depositTx(t, seller, assetLUX, 50_000_000),
		openMarketTx(t, pool, assetLUX, assetLUSD),
	)
	addBlock(t, vm1, placePoolTx(t, pool, zapwire.SideSell, 2.5, wireSize, seller))
	rootAtPlace := vm1.lastRoot // committed accepted root through the place block (height Hp)

	// RESTART: a fresh VM re-Initializes over the SAME committed db. It rebuilds
	// every book from the durable rows (RowToOrder -> reloaded float S').
	vm2, _ := newTestVM(t, db)
	defer vm2.Shutdown(context.Background())

	// A restart must preserve the accepted execution root exactly.
	if vm2.lastRoot != rootAtPlace {
		t.Fatalf("RESTART CHANGED ACCEPTED ROOT: continuing=%x restarted=%x", rootAtPlace[:8], vm2.lastRoot[:8])
	}

	// The CONTINUING proposer (vm1) builds height Hp+1: a benign second place in the
	// same market (touches the market => bookRoot includes the resting fractional
	// SELL, re-derived from the committed row). Build WITHOUT accepting so the shared
	// db stays at Hp for the restarted verifier.
	ctx := context.Background()
	vm1.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 1.0, 1.0, seller))
	built, err := vm1.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("continuing proposer BuildBlock: %v", err)
	}
	proposerRoot := built.(*Block).execRoot

	// The RESTARTED validator (vm2) independently re-derives and verifies the SAME
	// block bytes. Verify recomputes the execRoot from vm2's restarted view and
	// checks it against the proposer's claim (block.go:188 result.root != execRoot).
	reparsed, err := vm2.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("restarted validator ParseBlock: %v", err)
	}
	if err := reparsed.Verify(ctx); err != nil {
		t.Fatalf("CONSENSUS FORK CONFIRMED: restarted validator REJECTED the continuing proposer's "+
			"post-place block (root divergence): %v", err)
	}
	if got := reparsed.(*Block).execRoot; got != proposerRoot {
		t.Fatalf("root divergence continuing=%x restarted=%x", proposerRoot[:8], got[:8])
	}
	t.Logf("REFUTED: restarted validator re-derived the identical execRoot %x for the post-place "+
		"block and Verify passed — no fork. execute() rebuilds the book from committed DB rows every "+
		"block, so the wire-float only reaches the root in the place block (processed identically by all).",
		proposerRoot[:8])
}
