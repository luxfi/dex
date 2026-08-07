// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/database/prefixdb"
	"github.com/luxfi/dex/pkg/dex"
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

// TestDrive_IntentBecomesATrade is the headline proof that the seam is reachable at
// all. With NO manual transaction of any kind, one Phase-A C->D intent object in shared
// memory becomes: a funded D account, a REAL ORDER placed through the seam, a cross
// against a resting maker, and a D->C settlement object carrying the proceeds — every
// leg produced by the VM's own BuildBlock.
//
// The gap this closes: the C->D object used to carry VALUE ONLY, so executeImport
// credited the taker and stopped. Nothing placed an order, nothing crossed, no proceeds
// existed, the export drive (which fired only on realized output) never fired, Phase B
// had no object, and the settle reverted. Carrying the OPERATION in the object — and
// having the drive construct the taker's order from it — is the whole difference
// between this test and a chain that has never emitted a single 0x9999 event.
func TestDrive_IntentBecomesATrade(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x0D, 0x0D, 0x0D}
	base := assetID32(0xBB)
	quote := assetID32(0xCC)
	maker := acctFor(t, "drive-maker")
	takerAddr := h.addr(t, "drive-taker")

	// ---- Setup: open the custody market + rest a maker SELL 100 base @ 2 ----
	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "drive-maker", base, 100))
	h.buildAccept(t)
	h.vm.mempool.Add(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))
	h.buildAccept(t)

	// ---- PHASE A (C side, modeled): lock 200 quote and write the intent object. The
	// operation is the taker's: BUY 100 base on this market, no worse than 2.0. ----
	const lockedQuote = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, lockedQuote, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, lockedQuote,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})

	acct := seamAccount(intentID)
	if got := h.avail(t, acct, quote); got != 0 {
		t.Fatalf("pre-import seam account quote = %d, want 0", got)
	}
	if h.vm.mempool.Len() != 0 {
		t.Fatalf("mempool must be empty so every leg is drive-generated, have %d", h.vm.mempool.Len())
	}

	// ---- ONE BLOCK: the drive imports the object AND submits the taker's order ----
	blk := h.buildAccept(t)
	imp, ok := findTx(blk, TxImport)
	if !ok {
		t.Fatal("BuildBlock must produce a TxImport for the pending intent (none found)")
	}
	gotID, gotObj, derr := decodeSeamImportBody(imp.Body)
	if derr != nil || gotID != intentID {
		t.Fatalf("drive-generated TxImport names the wrong intent: id=%s err=%v", gotID, derr)
	}
	// THE OBJECT RIDES IN THE TRANSACTION. This is what makes execution replayable:
	// a node re-executing this block later reads these bytes, not shared memory.
	if len(gotObj) != seamIntentObjectSize {
		t.Fatalf("TxImport carries a %d-byte object, want %d", len(gotObj), seamIntentObjectSize)
	}
	if _, ok := findTx(blk, TxIntentSubmit); !ok {
		t.Fatal("THE GAP: BuildBlock imported the value but never constructed the taker's order. " +
			"Nothing crosses, so no proceeds exist, so no D->C object is ever produced and the " +
			"0x9999 settle reverts — the seam is unreachable by construction.")
	}

	// The cross happened in that same block: 100 base in, 200 quote out.
	if got := h.avail(t, acct, base); got != 100 {
		t.Fatalf("post-cross seam account base = %d, want 100 (the swap's proceeds)", got)
	}
	if got := h.avail(t, acct, quote); got != 0 {
		t.Fatalf("post-cross seam account quote = %d, want 0 (all 200 spent at limit 2.0)", got)
	}
	// The maker was paid exactly what the taker spent.
	if got := h.avail(t, maker.account, quote); got != 200 {
		t.Fatalf("maker quote = %d, want 200 — the taker's spend must equal the maker's credit", got)
	}
	// The escrow records that the order has RUN. That fact, not the shape of the
	// balances, is what the export drive gates on.
	rec, exists, err := getSeamIntent(h.vm.db, intentID)
	if err != nil || !exists || rec.Status != seamIntentSubmitted {
		t.Fatalf("escrow after submit: exists=%v status=%d err=%v", exists, rec.Status, err)
	}
	if rec.Owner != takerAddr || rec.Remaining != lockedQuote || rec.Op.Size != 100 {
		t.Fatalf("escrow lost the intent's identity or operation: %+v", rec)
	}
	// The C->D object was consumed at Accept.
	if vals, _ := h.dSM.Get(h.cChainID, [][]byte{intentID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("C->D intent object must be consumed after import")
	}

	// ---- AUTONOMOUS EXPORT: the next BuildBlock settles the proceeds back to C ----
	if h.vm.mempool.Len() != 0 {
		t.Fatalf("mempool must be empty so the export is purely drive-generated, have %d", h.vm.mempool.Len())
	}
	exportBlk := h.buildAccept(t)
	exp, ok := findTx(exportBlk, TxExport)
	if !ok {
		t.Fatal("BuildBlock must produce a TxExport for the settled intent (none found)")
	}
	eID, eAsset, eAmount, eSpent, eerr := decodeSeamExportBody(exp.Body)
	if eerr != nil || eID != intentID || eAsset != base || eAmount != 100 {
		t.Fatalf("drive export: id=%s asset=%x amount=%d err=%v", eID, eAsset[:4], eAmount, eerr)
	}
	// The spent witness the taker-authenticated MEV floor reads on C. It is EXACT
	// because the seam account is this intent's alone: nothing else could have moved
	// either number.
	if eSpent != lockedQuote {
		t.Fatalf("spent witness = %d, want %d — C's price floor reads out/spent and would "+
			"compute the wrong realized price", eSpent, lockedQuote)
	}

	// ---- The D->C object 0x9999 ImportSettlement consumes now EXISTS on C ----
	outputID := deriveSeamOutputID(exp.ID(), 0)
	vals, gerr := h.cSM.Get(h.dChainID, [][]byte{outputID[:]})
	if gerr != nil || len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatalf("no D->C settlement object in C's partition: err=%v", gerr)
	}
	rail, owner, asset, amount, spent, ok := decodeSeamObject(vals[0])
	if !ok || rail != railSwap || owner != takerAddr || asset != base || amount != 100 || spent != lockedQuote {
		t.Fatalf("D->C object: rail=%d owner=%s asset=%x amount=%d spent=%d", rail, owner.Hex(), asset[:4], amount, spent)
	}
	// A SETTLEMENT object is a VALUE object: 69 bytes, no operation. Only the C->D leg
	// instructs.
	if len(vals[0]) != seamObjectSize {
		t.Fatalf("D->C settlement object is %d bytes, want the canonical %d — C's "+
			"ImportSettlement decodes exactly that width", len(vals[0]), seamObjectSize)
	}

	// ---- CONSERVATION: the account is empty and the escrow is closed. ----
	if got := h.avail(t, acct, base); got != 0 {
		t.Fatalf("post-export seam account base = %d, want 0 (all exported)", got)
	}
	rec, _, _ = getSeamIntent(h.vm.db, intentID)
	if rec.Status != seamIntentReclaimed {
		t.Fatalf("escrow status = %d, want reclaimed — a drained intent must not be re-exported", rec.Status)
	}
	// Nothing more to drive.
	if h.vm.hasSeamWork() {
		t.Fatal("hasSeamWork still reports work after the intent fully settled")
	}
}

// TestDrive_ZeroFillStillSettles is the mint this design closes. An order that fills
// NOTHING leaves only its input asset in the account — which the old export gate ("does
// the account hold some other asset?") could never satisfy. The principal sat on D
// forever while C's deadline reclaim refunded the taker the same principal: paid twice.
//
// Gating on the RECORDED fact that the order ran makes the zero-fill case settle like
// any other: the whole principal goes back to C as a refund leg, and C's reclaim then
// finds nothing left to refund.
func TestDrive_ZeroFillStillSettles(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x0E, 0x0E, 0x0E}
	base := assetID32(0xBB)
	quote := assetID32(0xCC)
	takerAddr := h.addr(t, "zerofill-taker")

	// Market exists and has assets bound, but the book is EMPTY: nothing to cross.
	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})

	blk := h.buildAccept(t)
	if _, ok := findTx(blk, TxIntentSubmit); !ok {
		t.Fatal("the drive must still place the order against an empty book")
	}
	acct := seamAccount(intentID)
	if got := h.avail(t, acct, quote); got != locked {
		t.Fatalf("nothing crossed, so the whole principal must be available: got %d want %d", got, locked)
	}
	rec, _, _ := getSeamIntent(h.vm.db, intentID)
	if rec.Status != seamIntentSubmitted {
		t.Fatalf("a zero-fill order still RAN: status = %d, want submitted", rec.Status)
	}

	// The export drive settles the untouched principal back to C.
	exportBlk := h.buildAccept(t)
	exp, ok := findTx(exportBlk, TxExport)
	if !ok {
		t.Fatal("MINT: a zero-fill intent produced no export. Its principal is stranded on D " +
			"while C's deadline reclaim refunds the taker the same principal — paid twice.")
	}
	eID, eAsset, eAmount, eSpent, _ := decodeSeamExportBody(exp.Body)
	if eID != intentID || eAsset != quote || eAmount != locked || eSpent != 0 {
		t.Fatalf("refund leg: id=%s asset=%x amount=%d spent=%d", eID, eAsset[:4], eAmount, eSpent)
	}
	if got := h.avail(t, acct, quote); got != 0 {
		t.Fatalf("post-refund seam account quote = %d, want 0", got)
	}
	rec, _, _ = getSeamIntent(h.vm.db, intentID)
	if rec.Status != seamIntentReclaimed {
		t.Fatalf("escrow status = %d, want reclaimed", rec.Status)
	}
}

// TestDrive_SeamCannotSweepTheTakersOwnBalance is the conservation hole that opens the
// moment the seam works, if the import credits the taker's ORDINARY account.
//
// The export drive settles an account's balances back to C. Sharing the account with
// the taker's native D activity makes it impossible to tell this intent's proceeds from
// balance the taker deposited themselves — so ALL of it would be exported, and C would
// credit the full amount out of the shared seam reserve while only the swap's share was
// ever backed by this seam. That is a raid on other takers' pooled input.
//
// One account per intent removes the question instead of bounding it.
func TestDrive_SeamCannotSweepTheTakersOwnBalance(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())

	pool := [32]byte{0x0F, 0x0F, 0x0F}
	base := assetID32(0xBB)
	quote := assetID32(0xCC)
	taker := acctFor(t, "sweep-taker")
	takerAddr := h.addr(t, "sweep-taker")
	maker := acctFor(t, "sweep-maker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "sweep-maker", base, 100))
	// The taker's OWN native D balance, nothing to do with any cross-chain swap.
	h.vm.mempool.Add(depositTx(t, "sweep-taker", base, 5_000))
	h.buildAccept(t)
	h.vm.mempool.Add(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))
	h.buildAccept(t)

	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})

	h.buildAccept(t)              // import + submit + cross
	exportBlk := h.buildAccept(t) // export

	// Every export leg must be bounded by what the intent itself brought in.
	var exported uint64
	for _, tx := range exportBlk.txs {
		if tx.Type != TxExport {
			continue
		}
		_, asset, amount, _, _ := decodeSeamExportBody(tx.Body)
		if asset == base {
			exported += amount
		}
	}
	if exported != 100 {
		t.Fatalf("the seam exported %d base, but this intent only ever bought 100. The taker's own "+
			"5000 native balance was swept to C, and C credits it out of the shared seam reserve — "+
			"draining other takers' pooled input.", exported)
	}
	// The taker's native balance is untouched.
	if got := h.avail(t, taker.account, base); got != 5_000 {
		t.Fatalf("taker native base = %d, want 5000 untouched", got)
	}
}

// TestDrive_NoExportBeforeTheOrderRuns proves the two gates that keep the export drive
// from ever running ahead of the swap: (1) an intent that was never imported has no
// escrow, so driveSeamExports never enumerates it; (2) an escrow whose order has not
// RUN — status `open`, reachable when an import is injected without its submit — is not
// exportable, so the imported principal is never refunded out from under a pending swap.
//
// The gate is the recorded status, not the shape of the account's balances. That is
// what lets a zero-fill order settle (TestDrive_ZeroFillStillSettles) while a
// not-yet-run order still cannot.
func TestDrive_NoExportBeforeTheOrderRuns(t *testing.T) {
	h := newSeamHarness(t)
	defer h.vm.Shutdown(context.Background())
	pool := [32]byte{0x07, 0x07, 0x07}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "noexp-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	// (1) Intent staged but NOT imported (no escrow). The export drive produces nothing.
	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})
	if exports, err := h.vm.driveSeamExports(); err != nil || len(exports) != 0 {
		t.Fatalf("no-export-before-import: driveSeamExports = %d txs err=%v, want 0", len(exports), err)
	}

	// (2) Import the object BY HAND, with no submit leg. The escrow is `open`: the value
	// is on D but the taker's order has not run, so nothing may be settled back yet.
	h.vm.autoDriveSeam = false // suppress the drive so only the injected import lands
	h.vm.mempool.Add(h.importTx(t, intentID))
	h.buildAccept(t)
	h.vm.autoDriveSeam = true

	acct := seamAccount(intentID)
	if got := h.avail(t, acct, quote); got != locked {
		t.Fatalf("after import, seam account quote = %d, want %d", got, locked)
	}
	rec, exists, _ := getSeamIntent(h.vm.db, intentID)
	if !exists || rec.Status != seamIntentOpen {
		t.Fatalf("escrow status = %d (exists=%v), want open", rec.Status, exists)
	}
	if exports, err := h.vm.driveSeamExports(); err != nil || len(exports) != 0 {
		t.Fatalf("no-export-before-the-order-runs: driveSeamExports = %d txs err=%v, want 0", len(exports), err)
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

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	op := seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100}
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked, op)
	acct := seamAccount(intentID)

	// First autonomous import (the book is empty, so the submit leg fills nothing and the
	// whole principal stays available — which is what makes a double credit visible).
	h.buildAccept(t)
	if got := h.avail(t, acct, quote); got != locked {
		t.Fatalf("after first autonomous import, quote = %d, want %d", got, locked)
	}

	// Re-flush the SAME C->D object (it was consumed; simulate a source retry/reorg that
	// re-stages it) and drive again. The escrow already exists, so the import drive filters
	// it out: no re-import, no double credit.
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked, op)
	imports, err := h.vm.driveSeamImports()
	if err != nil {
		t.Fatalf("driveSeamImports: %v", err)
	}
	if len(imports) != 0 {
		t.Fatalf("re-flushed already-imported intent must NOT be re-imported, got %d txs", len(imports))
	}
	if got := h.avail(t, acct, quote); got != locked {
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
			Config:   authConfig(t),
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
		obj := encodeSeamIntentObject(owner, quote, amount, seamOp{
			Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: amount,
		})
		if err := cSM.Apply(map[ids.ID]*atomic.Requests{
			dChainID: {PutRequests: []*atomic.Element{{
				Key:    intentID[:],
				Value:  obj,
				Traits: [][]byte{owner[:], SeamPendingTrait},
			}}},
		}); err != nil {
			t.Fatalf("flush intent %d: %v", i, err)
		}
		want = append(want, EncodeSeamImportBody(intentID, obj))
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

	// Each intent contributes a PAIR: the import that funds it and the submit that runs
	// its order.
	if len(txs1) != 2*n || len(txs2) != 2*n {
		t.Fatalf("seam tx counts = (%d, %d), want (%d, %d)", len(txs1), len(txs2), 2*n, 2*n)
	}
	// Two proposers => byte-identical import sequence.
	for i := range txs1 {
		if string(txs1[i].Bytes()) != string(txs2[i].Bytes()) {
			t.Fatalf("proposers diverged at tx %d: %x vs %x", i, txs1[i].Bytes(), txs2[i].Bytes())
		}
	}
	// And that sequence is (import, submit) pairs in ascending intentID order — the
	// deterministic total order, with each intent's order placed immediately after the
	// value that funds it so there is never a block boundary between them.
	var prev string
	for i := 0; i < n; i++ {
		imp, sub := txs1[2*i], txs1[2*i+1]
		if imp.Type != TxImport || sub.Type != TxIntentSubmit {
			t.Fatalf("pair %d types = (%s, %s), want (seam_import, seam_submit)", i, imp.Type, sub.Type)
		}
		gotID, gotObj, err := decodeSeamImportBody(imp.Body)
		if err != nil {
			t.Fatalf("pair %d import body: %v", i, err)
		}
		if string(sub.Body) != string(gotID[:]) {
			t.Fatalf("pair %d: the submit names a different intent than the import beside it", i)
		}
		if string(imp.Body) != string(EncodeSeamImportBody(gotID, gotObj)) {
			t.Fatalf("pair %d import body is not canonical", i)
		}
		if i > 0 && prev >= string(gotID[:]) {
			t.Fatalf("intents not in ascending id order at %d", i)
		}
		prev = string(gotID[:])
	}
	_ = want
}
