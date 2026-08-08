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
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/runtime"
	luxvm "github.com/luxfi/vm"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic_rail_e2e_test.go drives the funding rail end to end against the REAL VM and
// REAL primary-network shared memory:
//
//	C (modeled): the 0x9999 precompile debits a depositor and writes a claim.
//	D (real):    TxImport consumes it and credits the beneficiary's OWN account.
//	D (real):    the beneficiary TRADES that money with an ordinary signed order.
//	D (real):    TxExport debits them and writes a claim back to C.
//
// The wire is the precompile's, byte-for-byte (atomic_wire_test.go pins it), so the
// claim written here is the claim 0x9999 consumes.
//
// THE POINT OF THE WHOLE EXERCISE is the third step: the crossing creates no order,
// so the trade is an ordinary D transaction against money that is already the
// beneficiary's. Move once, trade many.

// railHarness is a VM wired to REAL two-chain shared memory: its own handle (used by
// import/export) and the C-Chain's (the modeled precompile side).
type railHarness struct {
	vm       *VM
	dSM      atomic.SharedMemory
	cSM      atomic.SharedMemory
	cChainID ids.ID
	dChainID ids.ID
}

func newRailHarness(t *testing.T) *railHarness {
	t.Helper()
	logger := log.NewNoOpLogger()

	baseDB := memdb.New()
	m := atomic.NewMemory(prefixdb.New([]byte{0}, baseDB))

	dChainID := ids.GenerateTestID()
	cChainID := ids.GenerateTestID()
	dSM := m.NewSharedMemory(dChainID)
	cSM := m.NewSharedMemory(cChainID)

	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		Genesis: []byte(testDocument),
		Runtime: &runtime.Runtime{
			ChainID:      dChainID,
			CChainID:     cChainID,
			NetworkID:    96369,
			Log:          logger,
			SharedMemory: dSM,
		},
		DB:       prefixdb.New([]byte{1}, baseDB),
		Log:      logger,
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize rail vm: %v", err)
	}
	return &railHarness{vm: vm, dSM: dSM, cSM: cSM, cChainID: cChainID, dChainID: dChainID}
}

// newUnwiredVM builds a VM in the state EVERY LIVE NETWORK IS IN TODAY: the plugin
// server hands it a CChainID unconditionally, and SharedMemory stays nil because the
// node wired no atomic server. It is the exact configuration under which the two
// halves of the rail must still agree that there is no rail.
func newUnwiredVM(t *testing.T) *VM {
	t.Helper()
	logger := log.NewNoOpLogger()
	baseDB := memdb.New()
	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		Genesis: []byte(testDocument),
		Runtime: &runtime.Runtime{
			ChainID:   ids.GenerateTestID(),
			CChainID:  ids.GenerateTestID(),
			NetworkID: 96369,
			Log:       logger,
			// SharedMemory deliberately absent.
		},
		DB:       prefixdb.New([]byte{1}, baseDB),
		Log:      logger,
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize unwired vm: %v", err)
	}
	if err := vm.SetState(context.Background(), uint32(luxvm.Ready)); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	return vm
}

// TestRail_UnwiredRailCreditsNothing pins BOTH halves of the rail to the SAME wiring
// condition, because a rail with one end missing is not half a rail — it is a mint.
//
// The two halves used to disagree. Export refused without shared memory; import refused
// only without a C chain id, and the plugin server sets a C chain id unconditionally. So
// on a node with no atomic server — every live network, right now — 60 self-authored
// bytes named a beneficiary and an amount and became real, spendable balance. The block
// authenticator that exists to catch exactly that returned nil on the same nil handle,
// reasoning that import rejects without a C chain ID; it rejects without a C chain id,
// which was present. Two guards, both keyed to the wrong fact, and the money was free.
//
// The property is one sentence: WITHOUT SHARED MEMORY THERE IS NO PROOF, AND WITHOUT
// PROOF THERE IS NO CREDIT. Execution refuses the credit, and the block that carried the
// import is rejected rather than accepted unproven — either alone is sufficient, which is
// the point of having both.
func TestRail_UnwiredRailCreditsNothing(t *testing.T) {
	vm := newUnwiredVM(t)
	if vm.sharedMemory() != nil {
		t.Fatal("setup: this test is only meaningful with no shared memory wired")
	}
	if vm.cChainID == ids.Empty {
		t.Fatal("setup: the C chain id must be set — an absent one is the condition the old " +
			"guard keyed on, and the whole finding is that it is present anyway")
	}

	mallory := addrOf(t, "unwired-mallory")
	acct := Account16(dex.AuthSecp256k1, mallory)
	var native [32]byte
	const minted uint64 = 1_000_000_000_000

	// A claim id nobody ever wrote, and an object Mallory authored herself. There is no
	// C-side debit anywhere behind these bytes.
	forgedID := ids.GenerateTestID()
	forgedObject := encodeClaim(mallory, native, minted)

	overlay := versiondb.New(vm.db)
	before, err := getAvailable(overlay, acct, native)
	if err != nil {
		t.Fatalf("getAvailable: %v", err)
	}

	credited, ok, err := vm.executeImport(overlay, newAtomicRequests(), forgedID, forgedObject)
	if err != nil {
		t.Fatalf("executeImport: %v", err)
	}
	after, aerr := getAvailable(overlay, acct, native)
	if aerr != nil {
		t.Fatalf("getAvailable: %v", aerr)
	}

	if ok || credited != 0 {
		t.Errorf("executeImport accepted a claim it cannot prove: ok=%v credited=%d — with no "+
			"shared memory there is nothing behind these bytes but the bytes", ok, credited)
	}
	if after != before {
		t.Errorf("balance moved %d -> %d on an unprovable import: %d units minted from nothing",
			before, after, after-before)
	}

	// The block-level half, independently. It must reject rather than wave the import
	// through, and it must say so about THIS claim.
	blk := &Block{vm: vm, txs: []*Tx{{Type: TxImport, Body: EncodeImportBody(forgedID, forgedObject)}}}
	if verr := blk.verifyImports(); verr == nil {
		t.Error("verifyImports accepted a block carrying an import it had no way to authenticate; " +
			"an unprovable import must cost the block, not become balance")
	}

	// And a block with no imports is still fine on an unwired node: the rail being absent
	// is not an error, only crossing on it is.
	plain := &Block{vm: vm, txs: []*Tx{depositTx(t, "unwired-plain", native, 1)}}
	if verr := plain.verifyImports(); verr != nil {
		t.Errorf("verifyImports rejected a block that carries no import at all: %v", verr)
	}
}

// addrOf returns a test account's 20-byte C address. Account16 of it is the account
// their signed D orders draw from, so an import that credits the address's account
// credits exactly the balance they can trade with.
func addrOf(t *testing.T, name string) common.Address {
	t.Helper()
	a := acctFor(t, name)
	return common.BytesToAddress(crypto.PubkeyToAddress(a.priv.PublicKey).Bytes())
}

// cross models the C side: the precompile debited the depositor and its host flushed
// the staged claim into shared memory under D's partition. Returns the claim id.
func (h *railHarness) cross(t *testing.T, sourceTx ids.ID, index uint32, beneficiary common.Address, asset [32]byte, amount uint64) ids.ID {
	t.Helper()
	claimID := DeriveClaimID(h.cChainID, h.dChainID, sourceTx, index)
	err := h.cSM.Apply(map[ids.ID]*atomic.Requests{
		h.dChainID: {PutRequests: []*atomic.Element{{
			Key:    claimID[:],
			Value:  encodeClaim(beneficiary, asset, amount),
			Traits: [][]byte{beneficiary[:]},
		}}},
	})
	if err != nil {
		t.Fatalf("write C->D claim: %v", err)
	}
	return claimID
}

// deliver builds the permissionless TxImport that hands a claim to D, carrying the
// object's own bytes so execution never reads shared memory.
func (h *railHarness) deliver(t *testing.T, claimID ids.ID) *Tx {
	t.Helper()
	vals, err := h.dSM.Get(h.cChainID, [][]byte{claimID[:]})
	if err != nil || len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatalf("claim %s not readable from shared memory (err=%v)", claimID, err)
	}
	tx, terr := NewTx(TxImport, EncodeImportBody(claimID, vals[0]))
	if terr != nil {
		t.Fatalf("build TxImport: %v", terr)
	}
	return tx
}

func (h *railHarness) avail(t *testing.T, acct userKey, asset [32]byte) uint64 {
	t.Helper()
	v, err := getAvailable(h.vm.db, acct, asset)
	if err != nil {
		t.Fatalf("getAvailable: %v", err)
	}
	return v
}

// restingOrders counts every resting order row in the committed ledger. It is how
// this file proves a NEGATIVE — that a crossing placed nothing.
func (h *railHarness) restingOrders(t *testing.T) int {
	t.Helper()
	it := h.vm.db.NewIteratorWithPrefix([]byte(prefixOrder))
	defer it.Release()
	n := 0
	for it.Next() {
		n++
	}
	if err := it.Error(); err != nil {
		t.Fatalf("iterate orders: %v", err)
	}
	return n
}

// TestRail_DepositCreditsAndCreatesNoOrder is the headline proof. A deposit crosses,
// the beneficiary's own account is credited, and NOTHING ELSE HAPPENS — no order, no
// escrow, no book entry. Then the beneficiary trades that money with an ordinary
// signed order, which is the whole reason the crossing is allowed to be this boring.
func TestRail_DepositCreditsAndCreatesNoOrder(t *testing.T) {
	h := newRailHarness(t)

	pool := [32]byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01}
	base := a32(0xB0)
	quote := a32(0x90)

	maker := acctFor(t, "rail-maker")
	taker := acctFor(t, "rail-taker")
	takerAddr := addrOf(t, "rail-taker")

	// The credited account is the one the beneficiary's SIGNED orders draw from. If
	// these ever diverge, the money lands somewhere the owner cannot trade from and
	// "move once, trade many" is false.
	if Account16(dex.AuthSecp256k1, takerAddr) != taker.account {
		t.Fatal("the account a claim credits is not the account the beneficiary trades from")
	}

	// Setup: open the market and rest a maker SELL of 100 base @ 2.
	h.vm.mempool.Add(openMarketTx(t, pool, base, quote))
	h.vm.mempool.Add(depositTx(t, "rail-maker", base, 100))
	buildVerifyAccept(t, h.vm)
	h.vm.mempool.Add(maker.signed(t, TxPlace, encPlace(pool, sideSell, 2.0, 100, maker.user)))
	buildVerifyAccept(t, h.vm)

	ordersBefore := h.restingOrders(t)

	// ---- THE CROSSING ----
	const deposited = 200
	claimID := h.cross(t, repeatID(0xC1), 0, takerAddr, quote, deposited)
	h.vm.mempool.Add(h.deliver(t, claimID))
	buildVerifyAccept(t, h.vm)

	// It CREDITED: the money is in the beneficiary's own account, available.
	if got := h.avail(t, taker.account, quote); got != deposited {
		t.Fatalf("after the crossing the beneficiary holds %d quote available, want %d", got, deposited)
	}

	// It CREATED NO ORDER. The crossing carried a beneficiary, an asset and an amount;
	// there was nothing in it that could name a market, a side, a price or a size, so
	// there is nothing it could have placed.
	if got := h.restingOrders(t); got != ordersBefore {
		t.Fatalf("the crossing placed %d order(s); a funding claim must place none", got-ordersBefore)
	}

	// It CONSUMED the claim exactly once: the object is gone from shared memory and
	// the durable consumed-set row is written.
	if vals, _ := h.dSM.Get(h.cChainID, [][]byte{claimID[:]}); len(vals) == 1 && len(vals[0]) != 0 {
		t.Fatal("the claim must be removed from shared memory once imported")
	}
	if done, err := claimConsumed(h.vm.db, claimID); err != nil || !done {
		t.Fatalf("the claim must be in the consumed set (done=%v err=%v)", done, err)
	}

	// ---- TRADE MANY: an ORDINARY signed order, against money already theirs ----
	h.vm.mempool.Add(taker.signed(t, TxSubmit, encSubmit(pool, sideBuy, false, 2.0, 100, taker.user)))
	buildVerifyAccept(t, h.vm)

	if got := h.avail(t, taker.account, base); got != 100 {
		t.Fatalf("the taker holds %d base after the trade, want 100 — the trade must be a "+
			"purely D-local balance movement, with no second crossing", got)
	}
	if got := h.avail(t, taker.account, quote); got != 0 {
		t.Fatalf("the taker holds %d quote after paying 200, want 0", got)
	}
	if got := h.avail(t, maker.account, quote); got != 200 {
		t.Fatalf("the maker received %d quote, want 200", got)
	}

	// ---- THE WAY OUT: a signed export debits and writes a claim back to C ----
	body := EncodeExportBody(taker.account, base, 100, takerAddr)
	h.vm.mempool.Add(taker.signed(t, TxExport, body))
	blk := buildVerifyAccept(t, h.vm)

	if got := h.avail(t, taker.account, base); got != 0 {
		t.Fatalf("the taker holds %d base after exporting all of it, want 0", got)
	}
	outID := DeriveClaimID(h.dChainID, h.cChainID, blk.txs[0].ID(), 0)
	vals, gerr := h.cSM.Get(h.dChainID, [][]byte{outID[:]})
	if gerr != nil || len(vals) != 1 || len(vals[0]) == 0 {
		t.Fatalf("the export must leave a claim C can consume (err=%v)", gerr)
	}
	ben, asset, amount, ok := decodeClaim(vals[0])
	if !ok || ben != takerAddr || asset != base || amount != 100 {
		t.Fatalf("D->C claim = {ok %v beneficiary %s asset %x amount %d}, want {true %s %x 100}",
			ok, ben.Hex(), asset[:4], amount, takerAddr.Hex(), base[:4])
	}
	if len(vals[0]) != claimSize {
		t.Fatalf("the D->C object is %d bytes; the funding object is %d and carries value only",
			len(vals[0]), claimSize)
	}
}

// TestRail_ClaimImportsOnce pins the exactly-once property the whole rail rests on: a
// second delivery of the same claim credits nothing, even though the transaction is
// perfectly well-formed and carries the real recorded bytes.
func TestRail_ClaimImportsOnce(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)
	who := addrOf(t, "rail-once")
	acct := acctFor(t, "rail-once").account

	claimID := h.cross(t, repeatID(0xC2), 0, who, asset, 500)
	deliverTx := h.deliver(t, claimID)

	h.vm.mempool.Add(deliverTx)
	buildVerifyAccept(t, h.vm)
	if got := h.avail(t, acct, asset); got != 500 {
		t.Fatalf("first delivery credited %d, want 500", got)
	}

	// Re-deliver the IDENTICAL claim. A different transaction id, so the mempool and
	// the seen-set both accept it; only the consumed set stops the second credit.
	replay, err := NewTx(TxImport, EncodeImportBody(claimID, encodeClaim(who, asset, 500)))
	if err != nil {
		t.Fatalf("build replay: %v", err)
	}
	h.vm.mempool.Add(replay)
	h.vm.mempool.Add(depositTx(t, "rail-once-filler", a32(0xFF), 1)) // keep the block non-empty
	buildVerifyAccept(t, h.vm)

	if got := h.avail(t, acct, asset); got != 500 {
		t.Fatalf("a replayed claim credited again: balance %d, want 500 — this is a double spend", got)
	}
}

// TestRail_UnsignedExportCannotDrain pins the one authorization the un-braid had to
// add. An export used to draw on a per-order account whose only possible payee was
// the order's recorded owner, so the object was its own authority. Draining a real
// account to a caller-named address is not, so an export without the owner's
// signature must move nothing.
func TestRail_UnsignedExportCannotDrain(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)
	victim := acctFor(t, "rail-victim")
	victimAddr := addrOf(t, "rail-victim")
	thiefAddr := addrOf(t, "rail-thief")

	claimID := h.cross(t, repeatID(0xC3), 0, victimAddr, asset, 1000)
	h.vm.mempool.Add(h.deliver(t, claimID))
	buildVerifyAccept(t, h.vm)
	if got := h.avail(t, victim.account, asset); got != 1000 {
		t.Fatalf("setup: victim holds %d, want 1000", got)
	}

	// An UNSIGNED export naming the victim as owner and the thief as beneficiary
	// cannot even be CONSTRUCTED: the constructor refuses a money-moving type with no
	// authorization, so the gate is not something a caller has to remember.
	body := EncodeExportBody(victim.account, asset, 1000, thiefAddr)
	if _, err := NewTx(TxExport, body); err == nil {
		t.Fatal("an unsigned export must not be constructible")
	}

	// Hand-forge the raw frame anyway — [type][body] with no auth trailer — and prove
	// the PARSE gate refuses it too, so a peer cannot skip the constructor.
	raw := append([]byte{byte(TxExport)}, body...)
	if _, err := ParseTx(raw); err == nil {
		t.Fatal("an unsigned export frame must not parse: any peer could then drain any " +
			"account to an address of their choosing")
	}

	// Signed by the THIEF but asserting the VICTIM's account: the signature is real,
	// the account it authorizes is not the one the body names.
	thief := acctFor(t, "rail-thief")
	forged := thief.signed(t, TxExport, body)
	h.vm.mempool.Add(forged)
	h.vm.mempool.Add(depositTx(t, "rail-filler", a32(0xFF), 1)) // keep the block non-empty
	buildVerifyAccept(t, h.vm)

	if got := h.avail(t, victim.account, asset); got != 1000 {
		t.Fatalf("an export signed by someone else drained the victim: balance %d, want 1000", got)
	}
}
