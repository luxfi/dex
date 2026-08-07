// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/database/prefixdb"
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/runtime"
	luxvm "github.com/luxfi/vm"
	"github.com/luxfi/vm/chains/atomic"
)

// seam_reproducible_test.go proves the property the whole design turns on: a block
// carrying a cross-chain import can be RE-EXECUTED, by a node that has no access to the
// shared-memory object it consumed, and reach the identical state root.
//
// A settle that works live and wedges a syncing node is not a fix. That was the exact
// shape of the defect on both sides of the seam: execution read the object out of shared
// memory while the consuming Remove landed at Accept, so the first node to replay the
// block found the object gone, computed a different result, and died on the root.

// TestSeam_ImportReplaysWithoutSharedMemory is the bootstrap proof. It executes a block
// containing an import, then re-executes the SAME block bytes against the SAME parent
// state with the shared-memory object REMOVED — which is precisely the world a
// bootstrapping node lives in, because the object was consumed by this block's own
// accept. The two roots must be equal.
//
// Before the fix this test could not pass by construction: executeImport called sm.Get,
// found nothing, rejected the import, wrote no credit and no escrow, and derived a
// different ledgerRoot.
func TestSeam_ImportReplaysWithoutSharedMemory(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x5E, 0xA1}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	maker := acctFor(t, "replay-root-maker")
	takerAddr := h.addr(t, "replay-root-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "replay-root-maker", base, 100))
	h.buildAccept(t)
	h.vm.mempool.Add(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))
	h.buildAccept(t)

	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})

	// Build the seam block WITHOUT accepting it, so the parent state is still the state
	// a replaying node would start from.
	blkI, err := h.vm.BuildBlock(context.Background())
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	blk := blkI.(*Block)
	if _, ok := findTx(blk, TxImport); !ok {
		t.Fatal("no import in the block under test")
	}

	// First execution: with the object present, exactly as the proposer saw it.
	overlay1 := versiondb.New(h.vm.db)
	res1, err := blk.execute(context.Background(), overlay1)
	overlay1.Abort()
	if err != nil {
		t.Fatalf("execute #1: %v", err)
	}
	if res1.root != blk.execRoot {
		t.Fatalf("execute #1 root %x != claimed %x", res1.root[:8], blk.execRoot[:8])
	}

	// Now DESTROY the input the old code depended on: consume the object, exactly as
	// this block's own Accept would. A replaying node sees this world, never the one
	// above.
	if err := h.dSM.Apply(map[ids.ID]*atomic.Requests{
		h.cChainID: {RemoveRequests: [][]byte{intentID[:]}},
	}); err != nil {
		t.Fatalf("consume object: %v", err)
	}
	if vals, _ := h.dSM.Get(h.cChainID, [][]byte{intentID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("the object must be gone for this test to mean anything")
	}

	// Second execution of the SAME block bytes against the SAME parent state.
	overlay2 := versiondb.New(h.vm.db)
	res2, err := blk.execute(context.Background(), overlay2)
	overlay2.Abort()
	if err != nil {
		t.Fatalf("execute #2 (object consumed): %v", err)
	}
	if res2.root != res1.root {
		t.Fatalf("UNSYNCABLE: re-executing the block with the object consumed derived root %x, "+
			"the network derived %x. A bootstrapping node would die on `execution root mismatch` "+
			"at the first swap the seam ever settles.", res2.root[:8], res1.root[:8])
	}
	if res2.root != blk.execRoot {
		t.Fatalf("replay root %x != the block's claimed root %x", res2.root[:8], blk.execRoot[:8])
	}
}

// TestSeam_AcceptChecksTheRootToo pins the other half of syncability. Accept re-executes
// when it was not preceded by Verify — which is the bootstrap path — and it used to
// commit whatever it derived while persisting the block's CLAIMED root. A node that
// derived something different wrote a ledger disagreeing with its own meta:root and only
// found out one block later, with the divergence already committed.
//
// The test hands Accept a block whose claimed root is wrong and requires it to refuse.
func TestSeam_AcceptChecksTheRootToo(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x5E, 0xA2}
	h.vm.mempool.Add(openMarketTx(t, pool, assetID32(0xB0), assetID32(0x90)))
	blkI, err := h.vm.BuildBlock(context.Background())
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	good := blkI.(*Block)

	// A block with the same txs but a LIE for a root, and no overlay stashed — the
	// bootstrap shape (Accept without Verify).
	var lie [Size]byte
	lie[0] = 0xFF
	bad := newBlock(h.vm, good.parentID, good.height, good.timestamp, lie, good.txs)
	if err := bad.Accept(context.Background()); err == nil {
		t.Fatal("SILENT FORK: Accept committed a block whose execution it never re-derived. " +
			"A bootstrapping node would sync onto a ledger that disagrees with its own " +
			"persisted root and only discover it a block later, already committed.")
	}
}

// TestSeam_ForgedObjectKillsTheBlockNotTheRoot proves where the ship rule moved to.
// Execution binds the bytes it was handed — it must, or it is not replayable — so the
// proof that those bytes are the real recorded object is a BLOCK-level rule. A forged
// object therefore costs the producer a block instead of forking a receipt root, and the
// producer verifies its own block before proposing it, so a forgery never leaves the
// node.
func TestSeam_ForgedObjectKillsTheBlockNotTheRoot(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())
	h.vm.normalOp = true // above the bootstrap frontier, where the rule applies

	pool := [32]byte{0x5E, 0xA3}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "forge-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	const locked = 200
	op := seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100}
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked, op)

	// An honest block passes.
	honest := h.importTx(t, intentID)
	h.vm.autoDriveSeam = false
	h.vm.mempool.Add(honest)
	h.buildAccept(t)

	// A forged block: the id names a real object, but the bytes claim TEN TIMES the
	// value. Nothing in execution can catch this — it binds what it is handed.
	forgedID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 1)
	h.writeCToDIntentOp(t, forgedID, takerAddr, quote, locked, op)
	forgedObj := encodeSeamIntentObject(takerAddr, quote, locked*10, op)
	forged, err := NewTx(TxImport, EncodeSeamImportBody(forgedID, forgedObj))
	if err != nil {
		t.Fatalf("NewTx: %v", err)
	}
	blk := newBlock(h.vm, h.vm.lastAcceptedID, h.vm.lastAcceptedHeight+1, h.vm.blockTimestamp(), [Size]byte{}, []*Tx{forged})
	if err := blk.Verify(context.Background()); err == nil {
		t.Fatal("a block declaring an object shared memory does not hold was accepted: the ship " +
			"rule is gone and D can be funded without C ever locking anything")
	}

	// And an import naming an object that is not there at all is equally refused.
	absentID := ids.ID{0xAB, 0x5E, 0x17}
	absent, _ := NewTx(TxImport, EncodeSeamImportBody(absentID, encodeSeamIntentObject(takerAddr, quote, locked, op)))
	blk2 := newBlock(h.vm, h.vm.lastAcceptedID, h.vm.lastAcceptedHeight+1, h.vm.blockTimestamp(), [Size]byte{}, []*Tx{absent})
	if err := blk2.Verify(context.Background()); err == nil {
		t.Fatal("a block importing an object that is not in shared memory was accepted (unbacked mint)")
	}
}

// TestSeam_BootstrapGateIsRequired pins that the block-level check is SKIPPED below the
// bootstrap frontier — and why that is required rather than an optimization. C and D
// bootstrap independently, so while replaying history the C->D object may legitimately be
// absent: not yet exported, or already consumed by this very block's own accepted Remove.
// Those blocks carry the network's acceptance as their authority.
//
// Without the gate, the fix for unsyncable execution would simply have MOVED the wedge
// from execute() into Verify().
func TestSeam_BootstrapGateIsRequired(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	takerAddr := h.addr(t, "gate-taker")
	quote := assetID32(0x90)
	op := seamOp{Market: [32]byte{0x5E, 0xA4}, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100}
	absentID := ids.ID{0x60, 0x0F}
	tx, _ := NewTx(TxImport, EncodeSeamImportBody(absentID, encodeSeamIntentObject(takerAddr, quote, 200, op)))
	blk := newBlock(h.vm, h.vm.lastAcceptedID, h.vm.lastAcceptedHeight+1, h.vm.blockTimestamp(), [Size]byte{}, []*Tx{tx})

	h.vm.normalOp = false
	if err := blk.verifySeamImports(); err != nil {
		t.Fatalf("below the frontier the check must not run (a replaying node cannot see the "+
			"peer chain's object): %v", err)
	}
	h.vm.normalOp = true
	if err := blk.verifySeamImports(); err == nil {
		t.Fatal("above the frontier the check must run")
	}

	// And SetState is what moves the gate — the engine's own signal, not a guess.
	if err := h.vm.SetState(context.Background(), uint32(luxvm.Bootstrapping)); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	if h.vm.normalOp {
		t.Fatal("Bootstrapping must not be normal operation")
	}
	if err := h.vm.SetState(context.Background(), uint32(luxvm.Ready)); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	if !h.vm.normalOp {
		t.Fatal("Ready must be normal operation")
	}
}

// TestSeam_FlagDay_OldAndNewCannotDisagree is the wire-change safety proof. The intent
// object and the TxImport body both changed width, which is a hard fork — so the
// question that matters is not "can a mixed fleet keep working" (it cannot, that is what
// a flag day means) but "can a mixed fleet DISAGREE about a block". It cannot, and the
// reason is that the two generations are mutually UNPARSEABLE rather than mutually
// executable:
//
//   - An old node's bodySize(TxImport) is 32. Handed a new 150-byte frame, parseTxFrame
//     refuses it outright ("carries unexpected trailing bytes"), decodeTxList fails, and
//     ParseBlock fails. The old node never computes a state root for the block, so it
//     cannot compute a different one. It stalls — loudly, at a block boundary.
//   - A new node handed an old 33-byte frame fails the same way, from the other side
//     (ErrShortTxBody).
//
// This test asserts the mechanism directly: a frame of the wrong width for its type is
// refused at parse, in both directions, and a block containing one cannot be decoded.
func TestSeam_FlagDay_OldAndNewCannotDisagree(t *testing.T) {
	intentID := ids.ID{0xF1, 0xA6}
	owner := common.HexToAddress("0x00112233445566778899aabbccddeeff00112233")
	op := seamOp{Market: [32]byte{0x01}, Side: seamSideSell, LimitPrice: 2 * dex.PriceMultiplier, Size: 10}
	object := encodeSeamIntentObject(owner, assetID32(0x90), 100, op)

	// The new frame, as this build writes it.
	newTx, err := NewTx(TxImport, EncodeSeamImportBody(intentID, object))
	if err != nil {
		t.Fatalf("NewTx: %v", err)
	}
	if got := len(newTx.Bytes()); got != 1+seamImportBodySize {
		t.Fatalf("new TxImport frame = %d bytes, want %d", got, 1+seamImportBodySize)
	}

	// What an OLD node would put on the wire: the id alone. A new node must REFUSE it,
	// not execute it against an invented operation.
	oldFrame := append([]byte{byte(TxImport)}, intentID[:]...)
	if _, err := ParseTx(oldFrame); err == nil {
		t.Fatal("a new node parsed an OLD 32-byte TxImport body. It would then have to import " +
			"value with no operation attached — an order the taker never authorized.")
	}

	// The reverse direction, asserted structurally: an old node's parser is
	// parseTxFrame with bodySize(TxImport) == 32, and its rule for an unauthenticated
	// type is EXACT-WIDTH. Feed that rule the new frame's tail.
	if len(newTx.Body) == 32 {
		t.Fatal("the import body did not actually change width; there is no flag day to reason about")
	}

	// A block containing a frame of the wrong width does not decode at all — the
	// property that makes "stall" rather than "fork" the outcome.
	if _, err := decodeTxList(encodeRawTxList([][]byte{oldFrame})); err == nil {
		t.Fatal("a block carrying an old-format import decoded on a new node")
	}

	// And an object of the OLD width is refused by the decoder even if it reaches it:
	// no operation, no import.
	valueOnly := encodeSeamObject(railSwap, owner, assetID32(0x90), 100, 0)
	if _, _, _, _, ok := decodeSeamIntentObject(valueOnly); ok {
		t.Fatal("a 69-byte value object decoded as an intent")
	}
}

// TestSeamIntentWire_GoldenMatchesPrecompile pins THIS repo's intent encoder against the
// SAME cross-repo golden precompile/dex pins its encodeIntentObject to. The two repos
// cannot import each other, so this vector is the only thing keeping the operation wire
// in lockstep — and an operation that decodes differently on the two sides is a taker's
// order executed at terms they never authorized.
func TestSeamIntentWire_GoldenMatchesPrecompile(t *testing.T) {
	const golden = "00" + // rail (swap)
		"112233445566778899aabbccddeeff0102030405" + // owner (20)
		"a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf" + // asset (32)
		"0102030405060708" + // amount (8)
		"0000000000000000" + // spent (8) — always 0 on a C->D leg
		"b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf" + // market (32)
		"01" + // side (sell)
		"000000000bebc200" + // limitPrice (8) = 2.0 * 1e8
		"00000000000003e8" //   size (8) = 1000

	owner := common.HexToAddress("0x112233445566778899aabbccddeeff0102030405")
	var asset, market [32]byte
	for i := range asset {
		asset[i] = byte(0xA0 + i)
		market[i] = byte(0xB0 + i)
	}
	op := seamOp{Market: market, Side: seamSideSell, LimitPrice: 2 * dex.PriceMultiplier, Size: 1000}
	got := encodeSeamIntentObject(owner, asset, 0x0102030405060708, op)
	if len(got) != seamIntentObjectSize {
		t.Fatalf("intent object width %d, want %d", len(got), seamIntentObjectSize)
	}
	if hexOf(got) != golden {
		t.Fatalf("C->D intent wire DIVERGED from the precompile golden:\n got=%s\nwant=%s\n"+
			"the operation would decode differently on the two sides — a taker's order run at "+
			"terms they never authorized. Re-align dex/pkg/dchain and precompile/dex in lockstep.",
			hexOf(got), golden)
	}
	// Round trip, and the value head is byte-identical to the canonical 69-byte object.
	gotOwner, gotAsset, gotAmount, gotOp, ok := decodeSeamIntentObject(got)
	if !ok || gotOwner != owner || gotAsset != asset || gotAmount != 0x0102030405060708 || gotOp != op {
		t.Fatalf("round trip lost a field: ok=%v op=%+v", ok, gotOp)
	}
	head := encodeSeamObject(railSwap, owner, asset, 0x0102030405060708, 0)
	if hexOf(got[:seamObjectSize]) != hexOf(head) {
		t.Fatal("the intent's value head diverged from the canonical 69-byte object: every " +
			"settlement-side decoder reads exactly those bytes")
	}
}

// TestSeamPendingTrait_MatchesPrecompileGolden pins the DISCOVERY trait against the
// precompile's own constant. Drift here is silent in the worst way: D would query a
// trait C never writes, enumerate nothing, and every swap would sit unmatched forever
// with no error logged on either side — indistinguishable from "nobody is trading".
func TestSeamPendingTrait_MatchesPrecompileGolden(t *testing.T) {
	// sha256("lux.dex.native.intent.pending.v2") — the .v2 domain is the flag-day
	// switch that makes the op-less and op-carrying generations mutually invisible.
	const golden = "5f10f9ad8c53b78043df24cb69f8a5acb745351107470cb7e19f44748fe8bd5b"
	if len(SeamPendingTrait) != 32 {
		t.Fatalf("trait width %d, want 32", len(SeamPendingTrait))
	}
	if hexOf(SeamPendingTrait) != golden {
		t.Fatalf("SeamPendingTrait = %s\nwant                %s\n"+
			"the discovery trait drifted from the precompile's constant: D would enumerate "+
			"nothing and every swap would sit unmatched with no error anywhere",
			hexOf(SeamPendingTrait), golden)
	}
}

// --- small helpers -------------------------------------------------------------

func hexOf(b []byte) string {
	const hexdigits = "0123456789abcdef"
	out := make([]byte, 0, 2*len(b))
	for _, c := range b {
		out = append(out, hexdigits[c>>4], hexdigits[c&0x0f])
	}
	return string(out)
}

// encodeRawTxList frames arbitrary tx bytes the way a block body does, so a
// wrong-width frame can be fed to the real block decoder.
func encodeRawTxList(raws [][]byte) []byte {
	size := 4
	for _, r := range raws {
		size += 4 + len(r)
	}
	out := make([]byte, size)
	out[3] = byte(len(raws))
	off := 4
	for _, r := range raws {
		out[off+3] = byte(len(r))
		off += 4
		off += copy(out[off:], r)
	}
	return out
}

// TestSeam_RejectDoesNotRequeueDriveLegs pins that a rejected block's seam legs do NOT
// go back into the mempool. They came from the drive, which regenerates exactly the ones
// still warranted from committed state, so requeuing them would make the next block
// carry each leg twice — once from the mempool, once from the drive. Both copies are
// refused by the replay guards, so this is duplication rather than a double spend, but a
// block that carries the same import twice is a block nobody can read.
func TestSeam_RejectDoesNotRequeueDriveLegs(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x5E, 0xA5}
	h.vm.mempool.Add(openMarketTx(t, pool, assetID32(0xB0), assetID32(0x90)))
	h.buildAccept(t)

	takerAddr := h.addr(t, "requeue-taker")
	quote := assetID32(0x90)
	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})

	// A client tx rides along so the test can tell "requeued nothing" from "requeued
	// only the client's".
	h.vm.mempool.Add(depositTx(t, "requeue-taker", assetID32(0xB0), 1))

	blkI, err := h.vm.BuildBlock(context.Background())
	if err != nil {
		t.Fatalf("BuildBlock: %v", err)
	}
	blk := blkI.(*Block)
	if _, ok := findTx(blk, TxImport); !ok {
		t.Fatal("no seam leg in the block under test")
	}
	if h.vm.mempool.Len() != 0 {
		t.Fatalf("mempool should be drained, have %d", h.vm.mempool.Len())
	}

	if err := blk.Reject(context.Background()); err != nil {
		t.Fatalf("Reject: %v", err)
	}
	for _, tx := range h.vm.mempool.Drain(0) {
		if tx.Type.isSeamDriven() {
			t.Fatalf("a rejected block requeued a proposer-generated %s: the next block would "+
				"carry it twice, once from the mempool and once from the drive", tx.Type)
		}
		if tx.Type != TxDeposit {
			t.Fatalf("unexpected requeued tx type %s", tx.Type)
		}
	}
}

// TestSeam_TwoNodesAgreeOnASeamBlock is the closest deterministic stand-in for the
// devnet gate: a SECOND, independent node — its own state store, its own shared-memory
// handle, never having seen the proposer's mempool — verifies the proposer's seam block
// and derives the identical state root.
//
// It is the cross-node half of syncability. TestSeam_ImportReplaysWithoutSharedMemory
// proves one node can re-execute its own block after the object is gone; this proves a
// different node reaches the same answer in the first place. Together they are what
// "the settle does not wedge a syncing node" means.
func TestSeam_TwoNodesAgreeOnASeamBlock(t *testing.T) {
	ctx := context.Background()
	logger := log.NewNoOpLogger()
	baseDB := memdb.New()
	dChainID := ids.GenerateTestID()
	cChainID := ids.GenerateTestID()

	// SHARED MEMORY IS PER-NODE, not per-network: it is that node's local
	// materialization of both chains' accepted history, and each validator applies its
	// own copy of a block's cross-chain operations. Modelling it as one store shared
	// between the two nodes is wrong in a way that hides the real question and invents
	// a fake one — the second node's Apply then collides with the first's ("duplicate
	// put", a fatal Accept) for a Put that in production it would be making into its
	// own store. Two memories, two C-side flushes of the same object.
	type node struct {
		vm  *VM
		cSM atomic.SharedMemory
	}
	newNode := func(memPrefix, statePrefix byte) *node {
		m := atomic.NewMemory(prefixdb.New([]byte{memPrefix}, baseDB))
		vm := &VM{}
		if err := vm.Initialize(ctx, block.Init{
			Runtime: &runtime.Runtime{
				ChainID:      dChainID,
				CChainID:     cChainID,
				NetworkID:    96369,
				Log:          logger,
				SharedMemory: m.NewSharedMemory(dChainID),
			},
			DB:       prefixdb.New([]byte{statePrefix}, baseDB),
			Log:      logger,
			ToEngine: make(chan block.Message, 16),
			Config:   authConfig(t),
		}); err != nil {
			t.Fatalf("Initialize: %v", err)
		}
		if !vm.autoDriveSeam {
			t.Fatal("the seam must be wired for this test to mean anything")
		}
		return &node{vm: vm, cSM: m.NewSharedMemory(cChainID)}
	}
	pNode := newNode(0x10, 0x11)
	fNode := newNode(0x20, 0x21)
	proposer, follower := pNode.vm, fNode.vm
	defer proposer.Shutdown(ctx)
	defer follower.Shutdown(ctx)

	pool := [32]byte{0x7A, 0x70}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	maker := acctFor(t, "twonode-maker")
	takerAddr := common.HexToAddress("0x00112233445566778899aabbccddeeff00112233")

	// Both nodes accept the SAME setup blocks, so they start from identical state.
	// (The follower verifies each one, which is what keeps them in lockstep.)
	step := func(txs ...*Tx) *Block {
		t.Helper()
		for _, tx := range txs {
			proposer.mempool.Add(tx)
		}
		blkI, err := proposer.BuildBlock(ctx)
		if err != nil {
			t.Fatalf("BuildBlock: %v", err)
		}
		blk := blkI.(*Block)
		// The follower parses the proposer's BYTES — not its objects — and verifies.
		fblkI, err := follower.ParseBlock(ctx, blk.Bytes())
		if err != nil {
			t.Fatalf("follower ParseBlock: %v", err)
		}
		fblk := fblkI.(*Block)
		if err := fblk.Verify(ctx); err != nil {
			t.Fatalf("follower Verify (height %d): %v", blk.height, err)
		}
		if err := blk.Verify(ctx); err != nil {
			t.Fatalf("proposer Verify: %v", err)
		}
		if err := blk.Accept(ctx); err != nil {
			t.Fatalf("proposer Accept: %v", err)
		}
		if err := fblk.Accept(ctx); err != nil {
			t.Fatalf("follower Accept: %v", err)
		}
		return blk
	}

	step(openMarketTx(t, pool, base, quote), depositTx(t, "twonode-maker", base, 100))
	step(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))

	// One C->D intent, flushed once into the shared partition both nodes read.
	const locked = 200
	intentID := DeriveIntentID(96369, cChainID, dChainID, takerAddr, quote, locked, pool, 0)
	obj := encodeSeamIntentObject(takerAddr, quote, locked, seamOp{
		Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100,
	})
	// C's accept flushes the object into EVERY node's own shared memory, because every
	// node runs C too. Both nodes therefore see the identical object.
	for _, n := range []*node{pNode, fNode} {
		if err := n.cSM.Apply(map[ids.ID]*atomic.Requests{
			dChainID: {PutRequests: []*atomic.Element{{
				Key: intentID[:], Value: obj, Traits: [][]byte{takerAddr[:], SeamPendingTrait},
			}}},
		}); err != nil {
			t.Fatalf("flush intent: %v", err)
		}
	}

	// The seam block. The follower's Verify enforces root equality internally, so
	// reaching here at all is the agreement — but assert the shape too.
	seamBlk := step()
	if _, ok := findTx(seamBlk, TxImport); !ok {
		t.Fatal("the proposer's block carries no import")
	}
	if _, ok := findTx(seamBlk, TxIntentSubmit); !ok {
		t.Fatal("the proposer's block carries no taker order")
	}
	if proposer.lastRoot != follower.lastRoot {
		t.Fatalf("nodes diverged on the seam block: proposer %x, follower %x",
			proposer.lastRoot[:8], follower.lastRoot[:8])
	}

	// And the trade actually happened, on BOTH nodes' ledgers.
	acct := seamAccount(intentID)
	for name, vm := range map[string]*VM{"proposer": proposer, "follower": follower} {
		got, err := getAvailable(vm.db, acct, base)
		if err != nil || got != 100 {
			t.Fatalf("%s: seam account base = %d (err=%v), want 100 — the cross did not "+
				"replicate", name, got, err)
		}
		mk, _ := getAvailable(vm.db, maker.account, quote)
		if mk != 200 {
			t.Fatalf("%s: maker quote = %d, want 200", name, mk)
		}
	}

	// The export block: the follower must agree on the D->C object too.
	expBlk := step()
	exp, ok := findTx(expBlk, TxExport)
	if !ok {
		t.Fatal("no export driven for the settled intent")
	}
	if proposer.lastRoot != follower.lastRoot {
		t.Fatalf("nodes diverged on the export block: proposer %x, follower %x",
			proposer.lastRoot[:8], follower.lastRoot[:8])
	}
	// BOTH nodes produced the SAME D->C object into their own shared memory — byte for
	// byte, under the same key. That is what lets C consume it exactly once no matter
	// which validator it reads from.
	outputID := deriveSeamOutputID(exp.ID(), 0)
	var first []byte
	for name, n := range map[string]*node{"proposer": pNode, "follower": fNode} {
		vals, gerr := n.cSM.Get(dChainID, [][]byte{outputID[:]})
		if gerr != nil || len(vals) != 1 || len(vals[0]) != seamObjectSize {
			t.Fatalf("%s: D->C settlement object: err=%v vals=%d", name, gerr, len(vals))
		}
		if first == nil {
			first = vals[0]
		} else if string(first) != string(vals[0]) {
			t.Fatalf("the two nodes exported DIFFERENT bytes under the same key: %x vs %x",
				first, vals[0])
		}
	}
}
