// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/database/prefixdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/runtime"
	"github.com/luxfi/vm/chains/atomic"
)

// drive_seam_e2e_test.go proves the PROPOSER-SIDE DRIVE (drive.go): the dexvm's OWN
// BuildBlock — NOT a manual mempool.Add — autonomously imports C->D intents and exports
// D->C settlements during normal block production. atomic_seam_e2e_test.go drives the seam
// by hand (it plays the keeper); these tests inject NO TxImport/TxExport at all. The only
// inputs are (a) a Phase-A C->D intent object placed in shared memory (modeling 0x9999
// SubmitSwapIntent's flush) and (b) the taker's own signed crossing order. The VM does the
// rest, and the D->C object 0x9999 ImportSettlement consumes is produced by the VM itself.

// findTx returns the first tx of a given type in an accepted block (the drive-generated
// import/export the test asserts the VM produced on its own).
func findTx(blk *Block, typ TxType) (*Tx, bool) {
	for _, tx := range blk.txs {
		if tx.Type == typ {
			return tx, true
		}
	}
	return nil, false
}

// TestDrive_AutonomousImportCrossExport is the headline proof of the closed gap: with NO
// manual TxImport/TxExport, the VM's OWN BuildBlock (a) imports a Phase-A C->D intent and
// funds the taker, the taker's signed order (b) crosses a resting maker, and the VM's OWN
// BuildBlock (c) exports the realized proceeds as a D->C object the modeled 0x9999 Phase B
// consumes exactly once — conservation holds and the intent escrow is closed.
func TestDrive_AutonomousImportCrossExport(t *testing.T) {
	h := newSeamHarness(t)
	if !h.vm.autoDriveSeam {
		t.Fatal("seam is wired, so the proposer drive must be enabled by default")
	}

	pool := [32]byte{0xDE, 0xAD, 0xBE, 0xEF, 0x42}
	base := assetID32(0xB0)  // the asset the taker BUYS / the maker SELLS / the export proceeds
	quote := assetID32(0x90) // the asset the taker LOCKS on C (tokenIn) / the import funds

	maker := acctFor(t, "drive-maker")
	taker := acctFor(t, "drive-taker")
	takerAddr := h.addr(t, "drive-taker")

	// ---- Setup: open the custody market + rest a maker SELL 100 base @ 2 ----
	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "drive-maker", base, 100))
	h.buildAccept(t)
	h.vm.mempool.Add(maker.signed(t, TxPlace, zapwire.EncodePlace(pool, sideSell, 2.0, 100, maker.user)))
	h.buildAccept(t)

	// ---- PHASE A (C side, modeled): lock 200 quote -> a tagged C->D intent object ----
	const lockedQuote = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, lockedQuote, pool, 0)
	h.writeCToDIntent(t, intentID, takerAddr, quote, lockedQuote)

	// ---- (b) AUTONOMOUS IMPORT: BuildBlock with an EMPTY mempool drives the import ----
	if h.vm.mempool.Len() != 0 {
		t.Fatalf("mempool must be empty so the import is purely drive-generated, have %d", h.vm.mempool.Len())
	}
	if got := h.avail(t, taker.account, quote); got != 0 {
		t.Fatalf("pre-import taker quote = %d, want 0", got)
	}
	importBlk := h.buildAccept(t)
	importTx, ok := findTx(importBlk, TxImport)
	if !ok {
		t.Fatal("the VM's OWN BuildBlock must produce a TxImport for the pending intent (none found)")
	}
	if body := EncodeSeamImportBody(intentID); string(importTx.Body) != string(body) {
		t.Fatal("drive-generated TxImport names the wrong intent id")
	}
	if got := h.avail(t, taker.account, quote); got != lockedQuote {
		t.Fatalf("autonomous import: taker quote = %d, want %d", got, lockedQuote)
	}
	// The C->D object was consumed (removed) at Accept by the drive's import.
	if vals, _ := h.dSM.Get(h.cChainID, [][]byte{intentID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("C->D intent object must be consumed after autonomous import")
	}
	// The escrow now records the intent (so it is import-replay-guarded and export-bindable).
	rec, exists, err := getSeamIntent(h.vm.db, intentID)
	if err != nil || !exists || rec.Owner != takerAddr || rec.Remaining != lockedQuote {
		t.Fatalf("seam escrow after autonomous import: exists=%v err=%v rec=%+v", exists, err, rec)
	}

	// ---- The taker submits a SIGNED crossing order (the normal mempool flow) ----
	// The drive funds and settles; the taker still authors their own trade.
	h.vm.mempool.Add(taker.signed(t, TxSubmit, zapwire.EncodeSubmit(pool, sideBuy, false, 2.0, 100, taker.user)))
	crossBlk := h.buildAccept(t)
	// The cross is realized IN this block; at this block's BUILD the committed state had no
	// output yet, so the drive correctly emitted NO export here (no-export-before-trade).
	if _, drove := findTx(crossBlk, TxExport); drove {
		t.Fatal("export must not be driven in the same block the cross is realized (no committed output at build)")
	}
	if got := h.avail(t, taker.account, base); got != 100 {
		t.Fatalf("post-cross taker base = %d, want 100 (proceeds)", got)
	}
	if got := h.avail(t, taker.account, quote); got != 0 {
		t.Fatalf("post-cross taker quote = %d, want 0 (all spent)", got)
	}

	// ---- (c) AUTONOMOUS EXPORT: BuildBlock with an EMPTY mempool drives the export ----
	if h.vm.mempool.Len() != 0 {
		t.Fatalf("mempool must be empty so the export is purely drive-generated, have %d", h.vm.mempool.Len())
	}
	exportBlk := h.buildAccept(t)
	exportTx, ok := findTx(exportBlk, TxExport)
	if !ok {
		t.Fatal("the VM's OWN BuildBlock must produce a TxExport for the realized proceeds (none found)")
	}
	// Decode the drive's export and assert it settles the proceeds (base, 100) with the
	// matched-input spent witness (200), exactly what a correct keeper would build.
	gotIntent, gotAsset, gotAmount, gotSpent, derr := decodeSeamExportBody(exportTx.Body)
	if derr != nil || gotIntent != intentID || gotAsset != base || gotAmount != 100 || gotSpent != 200 {
		t.Fatalf("drive export body = {intent %x asset %x amount %d spent %d}, want {%x %x 100 200}",
			gotIntent[:6], gotAsset[:4], gotAmount, gotSpent, intentID[:6], base[:4])
	}
	if got := h.avail(t, taker.account, base); got != 0 {
		t.Fatalf("post-export taker base = %d, want 0 (exported)", got)
	}
	// The intent escrow is CLOSED (one-shot) once the account is drained.
	if rec, _, _ := getSeamIntent(h.vm.db, intentID); rec.Status != seamIntentReclaimed {
		t.Fatalf("escrow status after autonomous export = %d, want reclaimed (%d)", rec.Status, seamIntentReclaimed)
	}

	// ---- PHASE B (C side, modeled 0x9999 ImportSettlement): consume the D->C object ----
	outputID := deriveSeamOutputID(exportTx.ID(), 0)
	vals, gerr := h.cSM.Get(h.dChainID, [][]byte{outputID[:]})
	if gerr != nil || len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatalf("the VM-produced D->C settlement object is not where 0x9999 reads it (err=%v)", gerr)
	}
	recRail, recOwner, recAsset, recAmount, recSpent, decOK := decodeSeamObject(vals[0])
	if !decOK || recRail != railSwap || recOwner != takerAddr || recAsset != base || recAmount != 100 || recSpent != 200 {
		t.Fatalf("D->C object binds = {rail %d owner %s asset %x amount %d spent %d}, want {0 %s %x 100 200}",
			recRail, recOwner.Hex(), recAsset[:4], recAmount, recSpent, takerAddr.Hex(), base[:4])
	}
	// Consume once; replay must find it gone (0x9999 reverts ErrNativeNoSettlement).
	if err := h.cSM.Apply(map[ids.ID]*atomic.Requests{h.dChainID: {RemoveRequests: [][]byte{outputID[:]}}}); err != nil {
		t.Fatalf("0x9999 Phase-B consume: %v", err)
	}
	if vals2, _ := h.cSM.Get(h.dChainID, [][]byte{outputID[:]}); len(vals2) == 1 && len(vals2[0]) != 0 {
		t.Fatal("D->C object must be consumable exactly once")
	}

	// CONSERVATION across the seam: 200 quote entered D (C->D import), 100 base left D
	// (D->C export); inside D the maker provided 100 base and now holds 200 quote.
	if got := h.avail(t, maker.account, quote); got != 200 {
		t.Fatalf("maker quote (taker's conserved input) = %d, want 200", got)
	}
	if got := h.avail(t, maker.account, base); got != 0 {
		t.Fatalf("maker base after selling all = %d, want 0", got)
	}
	t.Logf("PASS: VM's OWN BuildBlock imported intent %x and exported D->C object %x (no manual injection)",
		intentID[:6], outputID[:6])
}

// TestDrive_NoExportBeforeImportOrTrade proves the two gates that keep the export drive from
// ever running ahead of the swap: (1) an intent that was never imported has no escrow, so
// driveSeamExports never enumerates it; (2) an imported-but-not-yet-traded intent holds only
// its input (no realized output), so the drive emits NO export and BuildBlock with an empty
// mempool reports nothing to do — the imported principal is never refunded out from under a
// pending swap.
func TestDrive_NoExportBeforeImportOrTrade(t *testing.T) {
	h := newSeamHarness(t)
	pool := [32]byte{0x07, 0x07, 0x07}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "noexp-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	// (1) Intent staged but NOT imported (no escrow). The export drive must produce nothing.
	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntent(t, intentID, takerAddr, quote, locked)
	if exports, err := h.vm.driveSeamExports(); err != nil || len(exports) != 0 {
		t.Fatalf("no-export-before-import: driveSeamExports = %d txs err=%v, want 0", len(exports), err)
	}

	// Autonomously import it (drive emits the TxImport), funding the taker — but the taker
	// does NOT trade.
	h.buildAccept(t)
	if got := h.avail(t, idOf(t, "noexp-taker"), quote); got != locked {
		t.Fatalf("after autonomous import, taker quote = %d, want %d", got, locked)
	}

	// (2) Imported, not traded: only the input is held, no realized output. No export, and an
	// empty-mempool BuildBlock has nothing to do (the input is NOT refunded prematurely).
	if exports, err := h.vm.driveSeamExports(); err != nil || len(exports) != 0 {
		t.Fatalf("no-export-before-trade: driveSeamExports = %d txs err=%v, want 0", len(exports), err)
	}
	if _, err := h.vm.BuildBlock(context.Background()); err != ErrEmptyMempool {
		t.Fatalf("BuildBlock with an imported-but-untraded intent = %v, want ErrEmptyMempool (no premature refund)", err)
	}
}

// TestDrive_ReplayImportedOnce proves the import drive funds a given intent EXACTLY ONCE.
// After the autonomous import, the C->D object is consumed; even if the source re-flushes
// the SAME object (a reorg/retry), the committed escrow filter (and executeImport's own
// replay-reject) make a second enumeration a no-op — no double credit.
func TestDrive_ReplayImportedOnce(t *testing.T) {
	h := newSeamHarness(t)
	pool := [32]byte{0x11, 0x22, 0x33}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "replay-drive-taker")
	takerAcct := idOf(t, "replay-drive-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntent(t, intentID, takerAddr, quote, locked)

	// First autonomous import.
	h.buildAccept(t)
	if got := h.avail(t, takerAcct, quote); got != locked {
		t.Fatalf("after first autonomous import, quote = %d, want %d", got, locked)
	}

	// Re-flush the SAME C->D object (object was consumed; simulate a source retry/reorg that
	// re-stages it) and drive again. The escrow already exists, so the import drive filters
	// it out: no re-import, no double credit.
	h.writeCToDIntent(t, intentID, takerAddr, quote, locked)
	imports, err := h.vm.driveSeamImports()
	if err != nil {
		t.Fatalf("driveSeamImports: %v", err)
	}
	if len(imports) != 0 {
		t.Fatalf("re-flushed already-imported intent must NOT be re-imported, got %d TxImport", len(imports))
	}
	// And an empty-mempool build has nothing to do (no re-import, no export — taker untraded).
	if _, err := h.vm.BuildBlock(context.Background()); err != ErrEmptyMempool {
		t.Fatalf("BuildBlock after replay = %v, want ErrEmptyMempool", err)
	}
	if got := h.avail(t, takerAcct, quote); got != locked {
		t.Fatalf("after replay, quote = %d, want %d (no double credit)", got, locked)
	}
}

// TestDrive_DeterministicOrdering proves two independent proposers build the SAME import
// sequence from the SAME committed shared-memory partition — the leaderless-build /
// matcher-at-Verify determinism requirement. Two VMs share one atomic memory (so they see
// the identical pending intents) but have independent state stores; their BuildBlock import
// tx lists must be byte-identical AND in ascending intentID order, regardless of the order
// the intents were flushed.
func TestDrive_DeterministicOrdering(t *testing.T) {
	ctx := context.Background()
	logger := log.NewNoOpLogger()

	baseDB := memdb.New()
	memoryDB := prefixdb.New([]byte{0}, baseDB)
	m := atomic.NewMemory(memoryDB)
	dChainID := ids.GenerateTestID()
	cChainID := ids.GenerateTestID()
	cSM := m.NewSharedMemory(cChainID)

	newVM := func(statePrefix byte) *VM {
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
		}); err != nil {
			t.Fatalf("Initialize vm: %v", err)
		}
		if !vm.autoDriveSeam {
			t.Fatal("seam wired: drive must be enabled")
		}
		return vm
	}
	vm1 := newVM(1)
	vm2 := newVM(2)

	// Flush several tagged intents. The nonce varies the intent id so the keys land in NO
	// particular order in shared memory; the drive must impose the total intentID sort.
	owner := common.HexToAddress("0x00112233445566778899aabbccddeeff00112233")
	quote := assetID32(0x90)
	pool := [32]byte{0xAB, 0xCD}
	const n = 6
	want := make([][]byte, 0, n)
	for i := 0; i < n; i++ {
		amount := uint64(100 + i)
		intentID := DeriveIntentID(96369, cChainID, dChainID, owner, quote, amount, pool, uint64(i*7919))
		obj := encodeSeamObject(railSwap, owner, quote, amount, 0)
		if err := cSM.Apply(map[ids.ID]*atomic.Requests{
			dChainID: {PutRequests: []*atomic.Element{{
				Key:    intentID[:],
				Value:  obj,
				Traits: [][]byte{owner[:], SeamPendingTrait},
			}}},
		}); err != nil {
			t.Fatalf("flush intent %d: %v", i, err)
		}
		want = append(want, EncodeSeamImportBody(intentID))
	}

	b1, err := vm1.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("vm1 BuildBlock: %v", err)
	}
	b2, err := vm2.BuildBlock(ctx)
	if err != nil {
		t.Fatalf("vm2 BuildBlock: %v", err)
	}
	txs1 := b1.(*Block).txs
	txs2 := b2.(*Block).txs

	if len(txs1) != n || len(txs2) != n {
		t.Fatalf("import tx counts = (%d, %d), want (%d, %d)", len(txs1), len(txs2), n, n)
	}
	// Two proposers => byte-identical import sequence.
	for i := range txs1 {
		if string(txs1[i].Bytes()) != string(txs2[i].Bytes()) {
			t.Fatalf("proposers diverged at tx %d: %x vs %x", i, txs1[i].Bytes(), txs2[i].Bytes())
		}
	}
	// And that sequence is ascending intentID order (the deterministic total order).
	for i := 0; i < n; i++ {
		if txs1[i].Type != TxImport {
			t.Fatalf("tx %d type = %s, want import", i, txs1[i].Type)
		}
		if i > 0 && string(txs1[i-1].Body) >= string(txs1[i].Body) {
			t.Fatalf("imports not in ascending intentID order at %d: %x then %x", i, txs1[i-1].Body, txs1[i].Body)
		}
	}
}
