// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// vm_test.go drives the real dchain.VM through its block.ChainVM lifecycle —
// the methods the consensus engine calls, in the exact order it calls them:
// Initialize -> WaitForEvent -> BuildBlock -> Verify -> Accept, then a restart
// (a fresh VM over the SAME durable DB) to prove the chainstate survives and the
// books rebuild from the order:* rows. This is the step-3 acceptance proof.

// newTestVM builds a VM over db with a buffered toEngine channel and Initializes
// it. The buffered channel mirrors the engine's receive side so the mempool's
// non-blocking PendingTxs signal never blocks the test.
func newTestVM(t *testing.T, db database.Database) (*VM, chan block.Message) {
	t.Helper()
	toEngine := make(chan block.Message, 16)
	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   authConfig(t), // configure the test deposit authority (F9)
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	return vm, toEngine
}

// submitFrame builds the [type:1][body] tx bytes for a dex op, the exact bytes
// the handler/proxy would Add to the mempool.
func ensureMarketTx(t *testing.T, pool [32]byte) *Tx {
	t.Helper()
	tx, err := NewTx(TxEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	if err != nil {
		t.Fatalf("NewTx ensure: %v", err)
	}
	return tx
}

// wirePriceUnits / wireSizeUnits convert a test's human price/size into the EXACT
// wire integers (price × 1e8 fixed-point, whole base units) — the SAME derivation
// the old float path applied downstream (int64(price*1e8), uint64(size)), so every
// balance the conservation/byzantine tests assert is unchanged. The conversion
// happens ONCE here (the client/proposer), never per-validator on the value path.
func wirePriceUnits(price float64) uint64 {
	if !(price > 0) {
		return 0
	}
	return uint64(price * float64(zapwire.PriceScale))
}
func wireSizeUnits(size float64) uint64 {
	if !(size > 0) {
		return 0
	}
	return uint64(size)
}

// encPlace / encSubmit are float-taking convenience wrappers over the exact-integer
// wire codec: they scale a human price to the ×1e8 fixed-point wire integer and a
// human size to whole base units (the SAME derivation the consensus path applies),
// so existing tests keep their human-readable literals while the wire stays integer.
func encPlace(pool [32]byte, side uint8, price, size float64, user string) []byte {
	return zapwire.EncodePlace(pool, side, wirePriceUnits(price), wireSizeUnits(size), user)
}
func encSubmit(pool [32]byte, side uint8, isMarket bool, price, size float64, user string) []byte {
	return zapwire.EncodeSubmit(pool, side, isMarket, wirePriceUnits(price), wireSizeUnits(size), user)
}

func placePoolTx(t *testing.T, pool [32]byte, side uint8, price, size float64, user string) *Tx {
	t.Helper()
	a := acctFor(t, user)
	return a.signed(t, TxPlace, zapwire.EncodePlace(pool, side, wirePriceUnits(price), wireSizeUnits(size), a.user))
}

func submitPoolTx(t *testing.T, pool [32]byte, side uint8, isMarket bool, price, size float64, user string) *Tx {
	t.Helper()
	a := acctFor(t, user)
	return a.signed(t, TxSubmit, zapwire.EncodeSubmit(pool, side, isMarket, wirePriceUnits(price), wireSizeUnits(size), a.user))
}

// buildVerifyAccept runs the proposer+validator path for the txs currently in the
// mempool: BuildBlock (proposer) -> ParseBlock+Verify (validator re-derives) ->
// Accept (commit). It asserts the parsed/verified block is byte-identical to the
// built one (a peer reconstructs the same block) and returns the accepted block.
func buildVerifyAccept(t *testing.T, vm *VM) *Block {
	t.Helper()
	ctx := context.Background()

	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}

	// A validator re-parses the proposer's bytes and verifies independently.
	reparsed, err := vm.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock: %v", err)
	}
	if string(reparsed.Bytes()) != string(built.Bytes()) {
		t.Fatal("re-parsed block bytes differ from built block")
	}
	if err := reparsed.Verify(ctx); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	// Snapshot the per-tx outcomes set at Verify so the accepted block retains them
	// for test inspection (Accept clears blk.outcomes after resolving waiters). The
	// ownership harness reads a submit block's fills from here.
	blk := reparsed.(*Block)
	snap := make([]txOutcome, len(blk.outcomes))
	copy(snap, blk.outcomes)
	if err := reparsed.Accept(ctx); err != nil {
		t.Fatalf("Accept: %v", err)
	}
	blk.outcomes = snap
	return blk
}

// TestVMLifecycle is the step-3 end-to-end proof: Initialize a fresh VM, place
// resting liquidity through a built+verified+accepted block, cross it with a
// submit in a second block, then restart (new VM, same DB) and confirm the book
// and the accepted head survive — rebuilt purely from the durable rows.
func TestVMLifecycle(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	pool := [32]byte{0x01, 0x02, 0x03}

	vm, toEngine := newTestVM(t, db)

	// Genesis: fresh chain is at height 0 with no markets.
	if vm.lastAcceptedHeight != 0 {
		t.Fatalf("fresh VM height = %d, want 0", vm.lastAcceptedHeight)
	}
	genesisID := vm.lastAcceptedID

	// ---- Block 1: ensure market + rest three orders ----
	vm.mempool.Add(ensureMarketTx(t, pool))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 101.0, 5.0, "maker-a"))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 102.0, 3.0, "maker-b"))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 99.0, 4.0, "maker-c"))

	// The mempool signalled the engine; WaitForEvent must return PendingTxs.
	msg, err := vm.WaitForEvent(ctx)
	if err != nil {
		t.Fatalf("WaitForEvent: %v", err)
	}
	if msg.Type != block.PendingTxs {
		t.Fatalf("WaitForEvent type = %v, want PendingTxs", msg.Type)
	}
	drainSignals(toEngine)

	blk1 := buildVerifyAccept(t, vm)
	if blk1.Height() != 1 {
		t.Fatalf("block1 height = %d, want 1", blk1.Height())
	}
	if blk1.Parent() != genesisID {
		t.Fatal("block1 parent != genesis")
	}
	if vm.lastAcceptedID != blk1.ID() {
		t.Fatal("VM head not advanced to block1")
	}

	// Book now has 3 resting orders (2 asks, 1 bid).
	ob := vm.books[pool]
	if ob == nil {
		t.Fatal("market book not present after accept")
	}
	if got := len(ob.Orders); got != 3 {
		t.Fatalf("resting orders = %d, want 3", got)
	}

	// ---- Block 2: a buy submit crosses the 101 ask fully + part of 102 ----
	vm.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 102.0, 6.0, "taker-x"))
	drainSignals(toEngine)
	blk2 := buildVerifyAccept(t, vm)
	if blk2.Height() != 2 {
		t.Fatalf("block2 height = %d, want 2", blk2.Height())
	}

	// After the cross: the 101 ask (5.0) is gone, the 102 ask has 2.0 left, the
	// 99 bid is untouched. So 2 resting orders remain.
	ob = vm.books[pool]
	if got := len(ob.Orders); got != 2 {
		t.Fatalf("resting orders after cross = %d, want 2", got)
	}

	// Capture the authoritative head + a canonical book image before restart.
	headBefore := vm.lastAcceptedID
	heightBefore := vm.lastAcceptedHeight
	rootBefore := vm.lastRoot
	rowsBefore := dex.BookToRows(ob)

	if err := vm.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// ---- RESTART: new VM over the SAME db (the rpc.Serve harness reopens the
	// persistent store; here memdb persists in-process) ----
	vm2, _ := newTestVM(t, db)

	if vm2.lastAcceptedID != headBefore {
		t.Fatalf("restart head = %s, want %s", vm2.lastAcceptedID, headBefore)
	}
	if vm2.lastAcceptedHeight != heightBefore {
		t.Fatalf("restart height = %d, want %d", vm2.lastAcceptedHeight, heightBefore)
	}
	if vm2.lastRoot != rootBefore {
		t.Fatalf("restart root = %x, want %x", vm2.lastRoot[:8], rootBefore[:8])
	}

	// The book rebuilt from the durable order:* rows must be byte-identical.
	ob2 := vm2.books[pool]
	if ob2 == nil {
		t.Fatal("market book not rebuilt after restart")
	}
	rowsAfter := dex.BookToRows(ob2)
	if len(rowsAfter) != len(rowsBefore) {
		t.Fatalf("rebuilt book has %d rows, want %d", len(rowsAfter), len(rowsBefore))
	}
	for i := range rowsBefore {
		if string(dex.EncodeRow(rowsBefore[i])) != string(dex.EncodeRow(rowsAfter[i])) {
			t.Errorf("rebuilt row %d differs after restart", i)
		}
	}

	// ---- The restarted VM is fully live: it can build/verify/accept again, and
	// the new block's root chains the surviving parent root. ----
	vm2.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 100.0, 1.0, "maker-d"))
	blk3 := buildVerifyAccept(t, vm2)
	if blk3.Height() != heightBefore+1 {
		t.Fatalf("post-restart block height = %d, want %d", blk3.Height(), heightBefore+1)
	}
	if blk3.Parent() != headBefore {
		t.Fatal("post-restart block does not chain the surviving head")
	}
	if got := len(vm2.books[pool].Orders); got != 3 {
		t.Fatalf("resting orders after post-restart place = %d, want 3", got)
	}
}

// TestVMRejectAbortsState proves a rejected block leaves no durable trace and
// returns its txs to the mempool: Verify stages an overlay, Reject aborts it, the
// committed state and in-RAM book are unchanged, and the dropped txs can be
// rebuilt into a later block.
func TestVMRejectAbortsState(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	pool := [32]byte{0x09}
	vm, toEngine := newTestVM(t, db)

	// Establish a market with one resting ask via an accepted block.
	vm.mempool.Add(ensureMarketTx(t, pool))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 50.0, 2.0, "maker"))
	drainSignals(toEngine)
	buildVerifyAccept(t, vm)

	headBefore := vm.lastAcceptedID
	heightBefore := vm.lastAcceptedHeight
	restingBefore := len(vm.books[pool].Orders)

	// Build a block that rests another order, Verify it (stages overlay), then
	// Reject it.
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 49.0, 1.0, "maker2"))
	drainSignals(toEngine)
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	if err := built.Verify(ctx); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if err := built.Reject(ctx); err != nil {
		t.Fatalf("Reject: %v", err)
	}

	// State must be exactly as before the rejected block.
	if vm.lastAcceptedID != headBefore || vm.lastAcceptedHeight != heightBefore {
		t.Fatal("rejected block advanced the head")
	}
	if got := len(vm.books[pool].Orders); got != restingBefore {
		t.Fatalf("rejected block mutated the in-RAM book: %d != %d", got, restingBefore)
	}
	// Nothing durable was written for height+1.
	if _, err := db.Get(orderKey(pool, blockDeterministicID(heightBefore+1, 1))); err != database.ErrNotFound {
		t.Fatalf("rejected order row leaked to disk: err=%v", err)
	}

	// The rejected txs are back in the mempool and can be rebuilt+accepted.
	drainSignals(toEngine)
	if vm.mempool.Len() == 0 {
		t.Fatal("rejected block did not requeue its txs")
	}
	blk := buildVerifyAccept(t, vm)
	if blk.Height() != heightBefore+1 {
		t.Fatalf("requeued block height = %d, want %d", blk.Height(), heightBefore+1)
	}
	if got := len(vm.books[pool].Orders); got != restingBefore+1 {
		t.Fatalf("requeued+accepted order not resting: %d", got)
	}
}

// TestVMVerifyRejectsLyingRoot proves the matcher-at-Verify guarantee: a block
// whose claimed execution root does not match the locally derived one is rejected
// by Verify. This is what lets every validator reject a lying proposer.
func TestVMVerifyRejectsLyingRoot(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	pool := [32]byte{0x07}
	vm, toEngine := newTestVM(t, db)

	vm.mempool.Add(ensureMarketTx(t, pool))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 10.0, 1.0, "m"))
	drainSignals(toEngine)
	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	good := built.(*Block)

	// Forge a block with the same txs but a tampered execution root.
	bad := newBlock(vm, good.parentID, good.height, good.timestamp, [Size]byte{0xde, 0xad}, good.txs)
	if err := bad.Verify(ctx); err == nil {
		t.Fatal("Verify accepted a block with a lying execution root")
	}

	// The honest block still verifies + accepts.
	if err := good.Verify(ctx); err != nil {
		t.Fatalf("honest Verify: %v", err)
	}
	if err := good.Accept(ctx); err != nil {
		t.Fatalf("honest Accept: %v", err)
	}
}

// TestVMDeterministicAcrossInstances proves the cross-validator determinism the
// whole design rests on: a block PROPOSED by one VM (carrying the proposer's
// timestamp in its bytes) is re-executed by a SECOND, independent VM (separate
// DB) to the byte-identical execution root. This models real consensus — the
// proposer mints the timestamp once, every validator replays the carried value
// (never re-samples its own clock), so the derived root is identical everywhere.
//
// (Two INDEPENDENTLY-proposed blocks legitimately differ by the proposer's
// timestamp; that non-determinism lives only in BuildBlock and is committed to
// the block bytes, which is why Verify replays the carried timestamp.)
func TestVMDeterministicAcrossInstances(t *testing.T) {
	ctx := context.Background()
	pool := [32]byte{0x42}

	// fresh brings up a new VM over its own DB (a distinct validator).
	fresh := func() (*VM, chan block.Message) {
		db := memdb.New()
		return newTestVM(t, db)
	}

	// applyProposed makes vm adopt a proposer's exact block bytes (ParseBlock ->
	// Verify -> Accept). Every validator runs this on the SAME bytes, so they all
	// replay the proposer's carried timestamp and converge to the identical head
	// and parent root — the precondition for block N+1's root to match too.
	applyProposed := func(vm *VM, raw []byte) *Block {
		t.Helper()
		blk, err := vm.ParseBlock(ctx, raw)
		if err != nil {
			t.Fatalf("ParseBlock: %v", err)
		}
		if err := blk.Verify(ctx); err != nil {
			t.Fatalf("Verify: %v", err)
		}
		if err := blk.Accept(ctx); err != nil {
			t.Fatalf("Accept: %v", err)
		}
		return blk.(*Block)
	}

	// vm1 is the proposer; vm2 is an independent validator on a separate DB.
	vm1, eng1 := fresh()
	vm2, _ := fresh()

	// ---- Block 1: vm1 proposes; BOTH adopt the identical bytes. ----
	vm1.mempool.Add(ensureMarketTx(t, pool))
	vm1.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 101.0, 5.0, "maker-a"))
	vm1.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 99.0, 2.0, "maker-c"))
	drainSignals(eng1)
	prop1, err := vm1.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock 1: %v", err)
	}
	b1a := applyProposed(vm1, prop1.Bytes())
	b1b := applyProposed(vm2, prop1.Bytes())
	if b1a.execRoot != b1b.execRoot {
		t.Fatalf("block 1 root diverged: %x vs %x", b1a.execRoot[:8], b1b.execRoot[:8])
	}

	// ---- Block 2: vm1 proposes a cross; vm2 re-derives the same root. ----
	vm1.mempool.Add(submitPoolTx(t, pool, zapwire.SideBuy, false, 101.0, 3.0, "taker"))
	drainSignals(eng1)
	prop2, err := vm1.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock 2: %v", err)
	}
	// vm2 re-parses + re-executes the proposer's exact bytes. Verify enforces the
	// derived root equals the claimed root; we assert vm2's derived root matches.
	reparsed, err := vm2.ParseBlock(ctx, prop2.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock 2 on validator: %v", err)
	}
	if err := reparsed.Verify(ctx); err != nil {
		t.Fatalf("validator Verify 2 (root divergence surfaces here): %v", err)
	}
	if reparsed.(*Block).execRoot != prop2.(*Block).execRoot {
		t.Fatalf("execution root diverged across instances: %x vs %x",
			prop2.(*Block).execRoot[:8], reparsed.(*Block).execRoot[:8])
	}
}

// drainSignals empties the buffered toEngine channel so a later WaitForEvent /
// Add does not see a stale signal. Tests drive BuildBlock directly, so the signal
// is only asserted once (TestVMLifecycle); elsewhere we just clear it.
func drainSignals(ch chan block.Message) {
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}

// TestVMRestartGetBlockHead reproduces the production restart failure: after a
// chain advances past genesis and the VM restarts (fresh VM, same DB), the
// consensus engine calls GetBlock(lastAccepted) immediately after Initialize.
// The block store is the in-RAM acceptedBlocks map (empty after restart), so
// before head blocks were persisted this returned ErrNotFound and luxd reported
// "zap initialize: remote error: get last accepted block: not found", taking the
// D-Chain down on the first validator to restart. loadHead now reconstructs the
// head block from durable bytes, so GetBlock(head) succeeds at every height.
func TestVMRestartGetBlockHead(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	pool := [32]byte{0x07}

	vm, toEngine := newTestVM(t, db)

	// The engine queries the head block right after Initialize even on a FRESH
	// (genesis) chain — that must already work.
	genesisID := vm.lastAcceptedID
	if blk, err := vm.GetBlock(ctx, genesisID); err != nil {
		t.Fatalf("GetBlock(genesis) on fresh VM: %v", err)
	} else if blk.ID() != genesisID {
		t.Fatalf("GetBlock(genesis) id = %s, want %s", blk.ID(), genesisID)
	}

	// Advance two blocks so the head is well past genesis.
	vm.mempool.Add(ensureMarketTx(t, pool))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 101.0, 5.0, "maker-a"))
	drainSignals(toEngine)
	buildVerifyAccept(t, vm)

	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 99.0, 3.0, "maker-c"))
	drainSignals(toEngine)
	blk2 := buildVerifyAccept(t, vm)

	headID := vm.lastAcceptedID
	headHeight := vm.lastAcceptedHeight
	headBytes := append([]byte(nil), blk2.Bytes()...)
	if headHeight != 2 {
		t.Fatalf("head height = %d, want 2", headHeight)
	}

	if err := vm.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// RESTART: fresh VM, same DB — mirrors the rpc.Serve harness reopening the
	// persistent store. This is exactly what failed in production.
	vm2, _ := newTestVM(t, db)
	if vm2.lastAcceptedID != headID {
		t.Fatalf("restart head = %s, want %s", vm2.lastAcceptedID, headID)
	}

	// THE REGRESSION ASSERTION: the engine's first call after Initialize.
	got, err := vm2.GetBlock(ctx, headID)
	if err != nil {
		t.Fatalf("GetBlock(head) after restart: %v (the production failure)", err)
	}
	if got.ID() != headID {
		t.Fatalf("GetBlock(head) id = %s, want %s", got.ID(), headID)
	}
	if got.Height() != headHeight {
		t.Fatalf("GetBlock(head) height = %d, want %d", got.Height(), headHeight)
	}
	if string(got.Bytes()) != string(headBytes) {
		t.Fatalf("GetBlock(head) bytes differ from the accepted block")
	}
	if got.Status() != statusAccepted {
		t.Fatalf("GetBlock(head) status = %d, want accepted(%d)", got.Status(), statusAccepted)
	}
}

// TestVMAcceptViaGetBlock reproduces the production consensus stall: the luxd
// plugin transport (vm/rpc) resolves Block.Accept by ID via VM.GetBlock — unlike
// Verify, which carries the full block bytes — and then calls Accept on whatever
// GetBlock returns. Before the processing-block index, GetBlock(builtID) returned
// ErrNotFound (only accepted blocks were indexed), so the engine's self-finalize
// Accept was a silent no-op, the dex_* submitTx waiter hung forever, and no
// further blocks were built. This drives the EXACT plugin-shaped path the live
// node uses — BuildBlock -> GetBlock(id) -> Accept — which every existing test
// skipped by accepting the *Block returned from BuildBlock directly.
func TestVMAcceptViaGetBlock(t *testing.T) {
	ctx := context.Background()
	db := memdb.New()
	pool := [32]byte{0x09}

	vm, toEngine := newTestVM(t, db)

	// A dex tx in the mempool, then the proposer's lifecycle exactly as the
	// engine drives it over the plugin transport.
	vm.mempool.Add(ensureMarketTx(t, pool))
	vm.mempool.Add(placePoolTx(t, pool, zapwire.SideSell, 101.0, 5.0, "maker-a"))
	drainSignals(toEngine)

	built, err := vm.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	builtID := built.ID()

	// THE PLUGIN-TRANSPORT CONTRACT: Verify the built block, then resolve it by ID
	// (not by holding the BuildBlock return) and Accept that instance.
	if err := built.Verify(ctx); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	got, err := vm.GetBlock(ctx, builtID)
	if err != nil {
		t.Fatalf("GetBlock(builtID) before accept: %v (the production stall)", err)
	}
	if got.ID() != builtID {
		t.Fatalf("GetBlock returned %s, want %s", got.ID(), builtID)
	}
	if err := got.Accept(ctx); err != nil {
		t.Fatalf("Accept via GetBlock: %v", err)
	}

	// The head advanced and the accepted block is now resolvable by id; the
	// processing entry was pruned.
	if vm.lastAcceptedID != builtID {
		t.Fatalf("head = %s, want %s", vm.lastAcceptedID, builtID)
	}
	if _, ok := vm.processingBlocks[builtID]; ok {
		t.Fatalf("processing entry not pruned after Accept")
	}
	if _, err := vm.GetBlock(ctx, builtID); err != nil {
		t.Fatalf("GetBlock(head) after accept: %v", err)
	}
	if len(vm.books[pool].Orders) != 1 {
		t.Fatalf("resting orders = %d, want 1", len(vm.books[pool].Orders))
	}
}

// TestVMParseVerifyAcceptByID covers the follower path: ParseBlock(bytes) ->
// Verify -> Accept resolved by ID through GetBlock. ParseBlock must return a
// STABLE instance so the overlay Verify stashes is the one Accept commits.
func TestVMParseVerifyAcceptByID(t *testing.T) {
	ctx := context.Background()
	pool := [32]byte{0x0A}

	// Proposer builds the block bytes.
	dbP := memdb.New()
	vmP, engP := newTestVM(t, dbP)
	vmP.mempool.Add(ensureMarketTx(t, pool))
	vmP.mempool.Add(placePoolTx(t, pool, zapwire.SideBuy, 99.0, 4.0, "maker-c"))
	drainSignals(engP)
	prop, err := vmP.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("proposer BuildBlock: %v", err)
	}
	raw := prop.Bytes()
	propID := prop.ID()

	// Follower adopts the bytes the way the engine does: ParseBlock, Verify, then
	// resolve by ID and Accept.
	dbF := memdb.New()
	vmF, _ := newTestVM(t, dbF)
	parsed, err := vmF.ParseBlock(ctx, raw)
	if err != nil {
		t.Fatalf("follower ParseBlock: %v", err)
	}
	if parsed.ID() != propID {
		t.Fatalf("parsed id %s, want %s", parsed.ID(), propID)
	}
	if err := parsed.Verify(ctx); err != nil {
		t.Fatalf("follower Verify: %v", err)
	}
	got, err := vmF.GetBlock(ctx, propID)
	if err != nil {
		t.Fatalf("follower GetBlock(id): %v", err)
	}
	if err := got.Accept(ctx); err != nil {
		t.Fatalf("follower Accept via GetBlock: %v", err)
	}
	if vmF.lastAcceptedID != propID {
		t.Fatalf("follower head = %s, want %s", vmF.lastAcceptedID, propID)
	}
}

// ensure the VM satisfies the consensus ChainVM interface at compile time.
var _ block.ChainVM = (*VM)(nil)

// silence unused import in case time is only used transitively.
var _ = time.Now
