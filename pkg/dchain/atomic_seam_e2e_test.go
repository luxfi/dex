// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"testing"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/crypto"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/database/prefixdb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/runtime"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic_seam_e2e_test.go drives the FULL native co-located two-phase settlement
// flow end-to-end against the real dchain.VM and REAL primary-network atomic shared
// memory — the owner's chosen model (the D-Chain matches under its own consensus and
// C-Chain 0x9999 settles by consuming a node-local atomic object D produces):
//
//	PHASE A (C side, modeled): the 0x9999 precompile's SubmitSwapIntent locks the
//	  taker's tokenIn on C and writes a C->D atomic INTENT object into shared memory,
//	  keyed by DeriveIntentID. We reproduce exactly that object + key.
//	NATIVE D: TxImport CONSUMES the C->D object and funds the taker's native D
//	  account; a signed taker order CROSSES a resting maker under D consensus; TxExport
//	  debits the realized proceeds and writes a D->C atomic SETTLEMENT object.
//	PHASE B (C side, modeled): the precompile's ImportSettlement reads the D->C object
//	  (Get under the D chain), binds rail/owner/asset/amount, consumes it EXACTLY ONCE
//	  (shared-memory Remove), and credits C. We reproduce that consume + its binds.
//
// The wire is the precompile's, byte-for-byte (encodeSeamObject == native_wire.go), so
// the object D exports here is the object 0x9999 ImportSettlement consumes. Phase B's
// binds and one-time consumption are reproduced over the SAME real shared memory, so
// "0x9999 settles it" is proven by the real cross-chain primitive, not a mock.

// seamHarness is a dchain.VM wired to REAL two-chain shared memory: the VM's own
// handle (dSM, used by the seam's import/export) and the C-Chain's handle (cSM, the
// modeled 0x9999 side that writes Phase-A intents and consumes Phase-B settlements).
type seamHarness struct {
	vm       *VM
	dSM      atomic.SharedMemory // D-Chain shared memory (this VM uses it via runtime)
	cSM      atomic.SharedMemory // C-Chain shared memory (the modeled precompile side)
	cChainID ids.ID
	dChainID ids.ID
	netID    uint32
}

// newSeamHarness mirrors the proxy conservation harness: ONE base db, a prefixed
// atomic-memory namespace and a prefixed VM db over it (so sm.Apply commits the VM
// overlay batch atomically with the shared-memory ops), two chain ids, and a runtime
// carrying the D-Chain's shared-memory handle + the C-Chain id the seam imports from /
// exports to.
func newSeamHarness(t *testing.T) *seamHarness {
	t.Helper()
	logger := log.NewNoOpLogger()

	baseDB := memdb.New()
	memoryDB := prefixdb.New([]byte{0}, baseDB)
	m := atomic.NewMemory(memoryDB)

	dChainID := ids.GenerateTestID()
	cChainID := ids.GenerateTestID()
	dSM := m.NewSharedMemory(dChainID)
	cSM := m.NewSharedMemory(cChainID)

	rt := &runtime.Runtime{
		ChainID:      dChainID,
		CChainID:     cChainID,
		NetworkID:    96369,
		Log:          logger,
		SharedMemory: dSM,
	}

	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		Runtime:  rt,
		DB:       prefixdb.New([]byte{1}, baseDB),
		Log:      logger,
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize seam vm: %v", err)
	}
	return &seamHarness{vm: vm, dSM: dSM, cSM: cSM, cChainID: cChainID, dChainID: dChainID, netID: 96369}
}

// addr returns the 20-byte C address of a test account (the cross-chain owner the
// precompile would carry on a C->D object). crossChainAccount(addr) == acct.account,
// so an import funds the exact account the taker's signed D orders draw from.
func (h *seamHarness) addr(t *testing.T, name string) common.Address {
	t.Helper()
	a := acctFor(t, name)
	return common.BytesToAddress(acctAddrBytes(a))
}

// testOp is the default CLOB operation a harness intent carries — a BUY of [size] base
// at a limit of 2.0 on the PriceInt grid. Real intents carry the taker's own; the
// mechanics tests only need one that is valid.
func testOp(market [32]byte, side uint8, size uint64) seamOp {
	return seamOp{Market: market, Side: side, LimitPrice: 2 * dex.PriceMultiplier, Size: size}
}

// writeCToDIntent reproduces the precompile Phase-A host flush: it writes a C->D swap
// INTENT object — the 69-byte value head (rail=railSwap, owner, asset, amount, spent=0)
// plus the 49-byte operation the taker authorized — into shared memory keyed by
// intentID under the D chain's partition, exactly what cSM.Apply(map[dChainID]{Put})
// does after SubmitSwapIntent stages it.
func (h *seamHarness) writeCToDIntent(t *testing.T, intentID ids.ID, owner common.Address, asset [32]byte, amount uint64) {
	t.Helper()
	h.writeCToDIntentOp(t, intentID, owner, asset, amount, testOp([32]byte{0xAA}, seamSideBuy, amount))
}

func (h *seamHarness) writeCToDIntentOp(t *testing.T, intentID ids.ID, owner common.Address, asset [32]byte, amount uint64, op seamOp) {
	t.Helper()
	obj := encodeSeamIntentObject(owner, asset, amount, op)
	// Tag the object with BOTH the per-owner trait (the precompile's existing index) AND
	// the fixed SeamPendingTrait — modeling the corrected C-side flush (native_staging.go
	// collectRange must add SeamPendingTrait so the D-side proposer drive can enumerate the
	// intent owner-agnostically; see drive.go). The trait is inert for the manual-drive
	// mechanics tests (Get-by-key ignores traits) and is what the autonomous drive walks.
	err := h.cSM.Apply(map[ids.ID]*atomic.Requests{
		h.dChainID: {PutRequests: []*atomic.Element{{
			Key:    intentID[:],
			Value:  obj,
			Traits: [][]byte{owner[:], SeamPendingTrait},
		}}},
	})
	if err != nil {
		t.Fatalf("write C->D intent: %v", err)
	}
}

// buildAccept drains the mempool into one block and runs Build->Verify->Accept.
func (h *seamHarness) buildAccept(t *testing.T) *Block {
	t.Helper()
	return buildVerifyAccept(t, h.vm)
}

// avail reads an account's available balance for an asset from the committed ledger.
func (h *seamHarness) avail(t *testing.T, acct userKey, asset [32]byte) uint64 {
	t.Helper()
	v, err := getAvailable(h.vm.db, acct, asset)
	if err != nil {
		t.Fatalf("getAvailable: %v", err)
	}
	return v
}

// ---------------------------------------------------------------------------------

// TestSeam_TwoPhase_FullFill is the headline proof: a Phase-A intent funds a native D
// account, a signed taker order crosses a resting maker under D consensus, the seam
// exports the realized proceeds as a D->C object, and the modeled 0x9999 Phase B
// consumes it exactly once — replay rejected, conservation holds.
func TestSeam_TwoPhase_FullFill(t *testing.T) {
	h := newSeamHarness(t)

	pool := [32]byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01}
	base := assetID32(0xB0)  // the asset the taker BUYS / the maker SELLS
	quote := assetID32(0x90) // the asset the taker LOCKS on C (tokenIn) / pays with

	maker := acctFor(t, "seam-maker")
	takerAddr := h.addr(t, "seam-taker")

	// ---- Setup: open the custody market + seed a resting maker SELL ----
	// Maker deposits 100 base (native dchain deposit) and rests a SELL 100 base @ 2.
	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "seam-maker", base, 100))
	h.buildAccept(t)

	h.vm.mempool.Add(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))
	h.buildAccept(t)
	if got := h.avail(t, maker.account, base); got != 0 {
		t.Fatalf("maker base available after resting SELL = %d, want 0 (locked)", got)
	}

	// ---- PHASE A (C side): lock 200 quote -> C->D intent object ----
	// To buy 100 base @ 2 the taker locks 200 quote on C; the precompile writes the
	// C->D object owner=taker, asset=quote, amount=200, keyed by DeriveIntentID.
	const lockedQuote = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, lockedQuote, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, lockedQuote,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})
	// The intent runs in its OWN ledger account, isolated from anything the taker holds
	// natively on D, so everything left in it at the end is this intent's answer.
	seamAcct := seamAccount(intentID)

	// ---- NATIVE D: import the intent (fund the intent's account) ----
	h.vm.autoDriveSeam = false // hand-drive each leg so the test can inspect them one at a time
	h.vm.mempool.Add(h.importTx(t, intentID))
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, quote); got != lockedQuote {
		t.Fatalf("seam account quote available after import = %d, want %d", got, lockedQuote)
	}
	// The escrow recorded the full 20-byte owner + the locked principal.
	rec, ok, err := getSeamIntent(h.vm.db, intentID)
	if err != nil || !ok {
		t.Fatalf("seam intent escrow missing after import (ok=%v err=%v)", ok, err)
	}
	if rec.Owner != takerAddr || rec.AssetIn != quote || rec.Remaining != lockedQuote {
		t.Fatalf("escrow = {owner %s assetIn %x remaining %d}, want {%s %x %d}",
			rec.Owner.Hex(), rec.AssetIn[:4], rec.Remaining, takerAddr.Hex(), quote[:4], lockedQuote)
	}
	// The C->D object was consumed (removed) at Accept: a second read is empty.
	if vals, _ := h.dSM.Get(h.cChainID, [][]byte{intentID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("C->D intent object must be consumed (removed) after import accept")
	}

	// ---- NATIVE D: the SEAM crosses the maker under D consensus ----
	// The order is the taker's own recorded operation — a limit BUY 100 base @ 2 — run
	// under the intent's authority, not a signature. It locks 200 quote and fills.
	h.vm.mempool.Add(submitTxFor(t, intentID))
	h.buildAccept(t)
	// Post-fill ledger: 200 quote spent, 100 base received; maker received 200 quote.
	if got := h.avail(t, seamAcct, quote); got != 0 {
		t.Fatalf("seam account quote after fill = %d, want 0 (all spent)", got)
	}
	if got := h.avail(t, seamAcct, base); got != 100 {
		t.Fatalf("seam account base after fill = %d, want 100 (proceeds)", got)
	}
	if got := h.avail(t, maker.account, quote); got != 200 {
		t.Fatalf("maker quote available after fill = %d, want 200", got)
	}

	// ---- NATIVE D: export the realized proceeds as a D->C object ----
	// The taker received 100 base by spending 200 quote -> spent witness = 200.
	const proceeds = 100
	const spent = 200
	exportTx := exportTx(t, intentID, base, proceeds, spent)
	outputID := deriveSeamOutputID(exportTx.ID(), 0) // the key Phase B will claim
	h.vm.mempool.Add(exportTx)
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, base); got != 0 {
		t.Fatalf("seam account base after export = %d, want 0 (exported)", got)
	}

	// ---- PHASE B (C side, modeled 0x9999 ImportSettlement) ----
	// Read the D->C object under the D chain (exactly sm.Get(dChainID, claim.OutputID)).
	vals, gerr := h.cSM.Get(h.dChainID, [][]byte{outputID[:]})
	if gerr != nil || len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatalf("D->C settlement object not found for 0x9999 (err=%v)", gerr)
	}
	recRail, recOwner, recAsset, recAmount, recSpent, decOK := decodeSeamObject(vals[0])
	if !decOK {
		t.Fatal("D->C object malformed (0x9999 decodeAtomicObject would reject)")
	}
	// The exact binds ImportSettlement enforces (rail/owner/asset/amount + the MEV witness).
	if recRail != railSwap {
		t.Fatalf("D->C object rail = %d, want railSwap (0)", recRail)
	}
	if recOwner != takerAddr {
		t.Fatalf("D->C object owner = %s, want taker %s (recOwner==Recipient bind)", recOwner.Hex(), takerAddr.Hex())
	}
	if recAsset != base {
		t.Fatalf("D->C object asset mismatch (recAsset==claim.Asset bind)")
	}
	if recAmount != proceeds {
		t.Fatalf("D->C object amount = %d, want %d (recAmount==claim.Amount bind)", recAmount, proceeds)
	}
	if recSpent != spent {
		t.Fatalf("D->C object spent witness = %d, want %d (MEV-floor binding)", recSpent, spent)
	}

	// CONSUME ONCE: the precompile applies a shared-memory Remove under the D chain.
	if err := h.cSM.Apply(map[ids.ID]*atomic.Requests{h.dChainID: {RemoveRequests: [][]byte{outputID[:]}}}); err != nil {
		t.Fatalf("0x9999 Phase-B consume (remove): %v", err)
	}
	// REPLAY REJECTED: a second read finds nothing -> ImportSettlement reverts
	// ErrNativeNoSettlement; the object can never be settled twice.
	if vals2, _ := h.cSM.Get(h.dChainID, [][]byte{outputID[:]}); len(vals2) == 1 && len(vals2[0]) != 0 {
		t.Fatal("D->C object must be consumed exactly once (replay must find it gone)")
	}

	// CONSERVATION across the seam: 200 quote entered D (C->D), 100 base left D (D->C);
	// inside D the maker provided 100 base and now holds 200 quote. Nothing minted.
	if got := h.avail(t, maker.account, quote); got != 200 {
		t.Fatalf("maker quote (the taker's spent input, conserved on D) = %d, want 200", got)
	}
	if got := h.avail(t, maker.account, base); got != 0 {
		t.Fatalf("maker base after selling all = %d, want 0", got)
	}
	// The recorded settlement (seamout:) bound the object to its intent for the keeper.
	t.Logf("PASS: native D fill -> D->C object %x (rail=%d owner=%s asset=%x amount=%d spent=%d) consumed once by 0x9999",
		outputID[:6], recRail, recOwner.Hex(), recAsset[:4], recAmount, recSpent)
}

// TestSeam_Import_ReplayNoDoubleCredit proves a re-submitted TxImport for an
// already-consumed intent never double-credits: the escrow row (and the seen: index)
// make a second import a deterministic no-op, and the C->D object is already gone.
func TestSeam_Import_ReplayNoDoubleCredit(t *testing.T) {
	h := newSeamHarness(t)
	pool := [32]byte{0x11, 0x22, 0x33}
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "replay-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, assetID32(0xB0), quote))
	h.buildAccept(t)

	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, 200, pool, 0)
	h.writeCToDIntent(t, intentID, takerAddr, quote, 200)
	seamAcct := seamAccount(intentID)

	h.vm.autoDriveSeam = false // hand-drive: only the injected import lands
	imp := h.importTx(t, intentID)
	h.vm.mempool.Add(imp)
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, quote); got != 200 {
		t.Fatalf("after first import quote = %d, want 200", got)
	}

	// Re-submit the SAME import. seen: returns the prior outcome (idempotent); the
	// escrow already exists and the C->D object is consumed — no second credit.
	h.vm.mempool.Add(imp)
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, quote); got != 200 {
		t.Fatalf("after replay import quote = %d, want 200 (no double credit)", got)
	}
}

// TestSeam_OverExportRefund_Rejected proves the D-side per-taker cap: a same-asset
// (refund) export that exceeds the intent's remaining locked principal is rejected,
// so an over-refund can never draw on other takers' pooled input — the export analog
// of the precompile's ErrSettleExceedsIntent.
func TestSeam_OverExportRefund_Rejected(t *testing.T) {
	h := newSeamHarness(t)
	pool := [32]byte{0x44, 0x55, 0x66}
	base := assetID32(0xB0)
	quote := assetID32(0x90)
	takerAddr := h.addr(t, "cap-taker")

	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.buildAccept(t)

	// Import 200 quote and run the order against an EMPTY book: nothing crosses, so all
	// 200 stays available and the intent's remaining principal is 200.
	const locked = 200
	intentID := DeriveIntentID(h.netID, h.cChainID, h.dChainID, takerAddr, quote, locked, pool, 0)
	h.writeCToDIntentOp(t, intentID, takerAddr, quote, locked,
		seamOp{Market: pool, Side: seamSideBuy, LimitPrice: 2 * dex.PriceMultiplier, Size: 100})
	seamAcct := seamAccount(intentID)
	h.vm.autoDriveSeam = false
	h.vm.mempool.Add(h.importTx(t, intentID))
	h.vm.mempool.Add(submitTxFor(t, intentID))
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, quote); got != locked {
		t.Fatalf("imported quote = %d, want %d", got, locked)
	}

	// Over-export: a REFUND (asset == intent.AssetIn == quote) of 201 > remaining 200.
	// Rejected by the per-taker cap BEFORE any debit — the ledger is untouched.
	over := exportTx(t, intentID, quote, locked+1, 0)
	h.vm.mempool.Add(over)
	blk := h.buildAccept(t)
	if st := outcomeStatus(blk, over.ID()); st != zapwire.StatusRejected {
		t.Fatalf("over-export status = %d, want StatusRejected (%d)", st, zapwire.StatusRejected)
	}
	if got := h.avail(t, seamAcct, quote); got != locked {
		t.Fatalf("quote after rejected over-export = %d, want %d (untouched)", got, locked)
	}
	// No D->C object was produced for the rejected export.
	outID := deriveSeamOutputID(over.ID(), 0)
	if vals, _ := h.cSM.Get(h.dChainID, [][]byte{outID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("a rejected over-export must produce NO D->C object")
	}

	// A refund AT the cap (== remaining 200) succeeds and exports a valid object.
	ok := exportTx(t, intentID, quote, locked, 0)
	okID := deriveSeamOutputID(ok.ID(), 0)
	h.vm.mempool.Add(ok)
	h.buildAccept(t)
	if got := h.avail(t, seamAcct, quote); got != 0 {
		t.Fatalf("quote after at-cap refund = %d, want 0", got)
	}
	vals, _ := h.cSM.Get(h.dChainID, [][]byte{okID[:]})
	if len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatal("at-cap refund must produce a D->C object")
	}
	_, recOwner, recAsset, recAmount, _, decOK := decodeSeamObject(vals[0])
	if !decOK || recOwner != takerAddr || recAsset != quote || recAmount != locked {
		t.Fatalf("at-cap refund object = {owner %s asset %x amount %d}, want {%s %x %d}",
			recOwner.Hex(), recAsset[:4], recAmount, takerAddr.Hex(), quote[:4], locked)
	}
}

// TestSeamWire_GoldenMatchesPrecompile pins encodeSeamObject against the SAME
// cross-repo golden the precompile (precompile/dex native_seam_parity_test.go) and the
// proxy (chains/dexvm) pin THEIR encoders to: rail(1)|owner(20)|asset(32)|amount(8)|
// spent(8) = 69 bytes, big-endian. The precompile's parity test asserts its
// encodeAtomicObjectSpent == this golden; this test asserts the native dexvm
// encodeSeamObject == the SAME golden. By transitivity the three encoders are
// byte-identical — the ONE-wire reconciliation, proven without a module cycle (dex
// cannot import precompile). Any one-sided change to the width/offsets breaks exactly
// one of the three pins.
func TestSeamWire_GoldenMatchesPrecompile(t *testing.T) {
	// The identical inputs the precompile parity golden uses.
	owner := common.HexToAddress("0x112233445566778899aabbccddeeff0102030405")
	var asset [32]byte
	for i := range asset {
		asset[i] = byte(0xA0 + i)
	}
	const amount, spent = uint64(0x0102030405060708), uint64(0x1112131415161718)

	const goldenHex = "00" + // rail = railSwap
		"112233445566778899aabbccddeeff0102030405" + // owner (20)
		"a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf" + // asset (32)
		"0102030405060708" + // amount (8)
		"1112131415161718" // spent (8)

	got := encodeSeamObject(railSwap, owner, asset, amount, spent)
	if len(got) != seamObjectSize || seamObjectSize != 69 {
		t.Fatalf("seam object width %d (const %d) != canonical 69 — the wire changed on ONE side", len(got), seamObjectSize)
	}
	if gotHex := hexEncode(got); gotHex != goldenHex {
		t.Fatalf("native seam wire DIVERGED from the precompile/proxy golden:\n got=%s\nwant=%s", gotHex, goldenHex)
	}
	// Round-trip the trailing spent witness (the field the precompile MEV floor reads).
	rail, gotOwner, gotAsset, gotAmount, gotSpent, ok := decodeSeamObject(got)
	if !ok || rail != railSwap || gotOwner != owner || gotAsset != asset || gotAmount != amount || gotSpent != spent {
		t.Fatalf("round-trip mismatch: ok=%v rail=%d amount=%d spent=%d", ok, rail, gotAmount, gotSpent)
	}
}

func hexEncode(b []byte) string {
	const hexdigits = "0123456789abcdef"
	out := make([]byte, len(b)*2)
	for i, c := range b {
		out[i*2] = hexdigits[c>>4]
		out[i*2+1] = hexdigits[c&0x0f]
	}
	return string(out)
}

// --- small builders -------------------------------------------------------------

func assetID32(b byte) [32]byte {
	var a [32]byte
	a[0] = b
	a[31] = 0x01 // non-native (native is all-zero)
	return a
}

// importTx builds the TxImport for an intent the way the production drive does: it
// READS the object out of shared memory and carries its bytes in the transaction, so
// execution never has to look at shared memory again.
func (h *seamHarness) importTx(t *testing.T, intentID ids.ID) *Tx {
	t.Helper()
	object, ok, err := readSeamObject(h.vm.seamSharedMemory(), h.cChainID, intentID)
	if err != nil || !ok {
		t.Fatalf("read C->D object for %s: ok=%v err=%v", intentID, ok, err)
	}
	tx, err := NewTx(TxImport, EncodeSeamImportBody(intentID, object))
	if err != nil {
		t.Fatalf("NewTx import: %v", err)
	}
	return tx
}

// submitTxFor builds the seam's taker-order tx for an imported intent.
func submitTxFor(t *testing.T, intentID ids.ID) *Tx {
	t.Helper()
	tx, err := NewTx(TxIntentSubmit, EncodeSeamSubmitBody(intentID))
	if err != nil {
		t.Fatalf("NewTx seam submit: %v", err)
	}
	return tx
}

func exportTx(t *testing.T, intentID ids.ID, asset [32]byte, amount, spent uint64) *Tx {
	t.Helper()
	tx, err := NewTx(TxExport, EncodeSeamExportBody(intentID, asset, amount, spent))
	if err != nil {
		t.Fatalf("NewTx export: %v", err)
	}
	return tx
}

// outcomeStatus returns the recorded zapwire status for a tx id in an accepted block.
func outcomeStatus(blk *Block, txID ids.ID) uint8 {
	for _, o := range blk.outcomes {
		if o.txID == txID {
			return o.status
		}
	}
	return 0xFF
}

// acctAddrBytes returns the 20-byte address bytes of a test account's secp key — the
// SAME derivation acctFor uses, so crossChainAccount(addr)==acct.account and an import
// funds the exact account the taker's signed D orders draw from.
func acctAddrBytes(a *testAcct) []byte {
	return crypto.PubkeyToAddress(a.priv.PublicKey).Bytes()
}
