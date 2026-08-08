// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"errors"
	"testing"

	"strings"

	"github.com/luxfi/database"
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/ids"
	luxvm "github.com/luxfi/vm"
	"github.com/luxfi/vm/chains/atomic"
)

// atomic_prove_test.go exercises proveClaim — the rail's block-level authenticator —
// through both of its callers.
//
// Until the rail harness reached normal operation these properties had NO coverage at
// all: verifyImports gates on normalOp, nothing in the package ever set it, so the
// forged-object rejection, the absent-object rejection and the bootstrap gate were
// assertions no test had ever run. The first thing running them revealed was that an
// ordinary duplicate delivery halted the chain permanently.

// forgedImport builds a TxImport that carries claimID with bytes of the caller's
// choosing — the shape of every attack on the ship rule.
func forgedImport(t *testing.T, claimID ids.ID, object []byte) *Tx {
	t.Helper()
	tx, err := NewTx(TxImport, EncodeImportBody(claimID, object))
	if err != nil {
		t.Fatalf("build TxImport: %v", err)
	}
	return tx
}

// TestProve_DuplicateDeliveryDoesNotHaltTheChain is the regression for a PERMANENT
// CHAIN HALT reachable by an ordinary, honest, expected event.
//
// Delivery is permissionless on purpose: "a relayer that refuses to deliver cannot
// strand anyone — any other participant can." Two relayers delivering the same claim,
// or one retrying after a timeout, is therefore the steady state and not an attack. The
// second delivery is well-formed and carries the real recorded bytes; the object is just
// gone from shared memory, because this chain's own accepted Remove took it.
//
// The authenticator read that as "unbacked" and rejected the block. Reject requeues
// every tx in a rejected block, so the next build drained the same delivery and failed
// identically — measured over four rounds, the chain never left the height it halted at.
// The claim is in the committed consumed set, which is exactly the fact that explains
// the absence, so the fix is to consult it first: an import that cannot credit needs no
// proof.
func TestProve_DuplicateDeliveryDoesNotHaltTheChain(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)
	who := addrOf(t, "dup-beneficiary")
	acct := acctFor(t, "dup-beneficiary").account

	claimID := h.cross(t, repeatID(0xD1), 0, who, asset, 500)
	h.vm.mempool.Add(h.deliver(t, claimID))
	buildVerifyAccept(t, h.vm)

	if got := h.avail(t, acct, asset); got != 500 {
		t.Fatalf("setup: first delivery credited %d, want 500", got)
	}
	acceptedBefore := h.vm.lastAcceptedHeight

	// A second relayer delivers the same claim, carrying the same real bytes.
	h.vm.mempool.Add(forgedImport(t, claimID, encodeClaim(who, asset, 500)))
	h.vm.mempool.Add(depositTx(t, "dup-filler", a32(0xFF), 1))
	buildVerifyAccept(t, h.vm)

	if h.vm.lastAcceptedHeight != acceptedBefore+1 {
		t.Fatalf("the chain is stuck at height %d: a duplicate delivery must cost the delivery, "+
			"not block production", h.vm.lastAcceptedHeight)
	}
	if got := h.avail(t, acct, asset); got != 500 {
		t.Fatalf("the duplicate credited again: balance %d, want 500 — this is a double spend", got)
	}
}

// TestProve_ForgedObjectCostsTheBlock is the ship rule itself: the bytes a transaction
// carries are only a CLAIM about a recorded object, and a block whose claim does not
// match the record is not a block. Execution binds the carried bytes because that is
// what makes it replayable; this is where those bytes are made true.
func TestProve_ForgedObjectCostsTheBlock(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)
	honest := addrOf(t, "forge-honest")
	mallory := addrOf(t, "forge-mallory")

	// A real claim for 10, recorded on C and readable here.
	claimID := h.cross(t, repeatID(0xD3), 0, honest, asset, 10)

	// Same claim id, rewritten to pay Mallory a million.
	blk := &Block{vm: h.vm, txs: []*Tx{forgedImport(t, claimID, encodeClaim(mallory, asset, 1_000_000))}}
	err := blk.verifyImports()
	if err == nil {
		t.Fatal("a block carrying an object that differs from the record was accepted; the " +
			"carried bytes are a claim about the record, not the record")
	}
	if !strings.Contains(err.Error(), "forged") {
		t.Errorf("expected the rejection to name the forgery, got: %v", err)
	}

	// The honest delivery of the same claim still passes, so the check discriminates on
	// the bytes and not on the claim id.
	ok := &Block{vm: h.vm, txs: []*Tx{h.deliver(t, claimID)}}
	if verr := ok.verifyImports(); verr != nil {
		t.Fatalf("the real recorded object was rejected: %v", verr)
	}
}

// TestProve_UnrecordedClaimCostsTheBlock covers the other half: bytes that are perfectly
// well-formed and name a claim id nothing ever recorded. This is the shape CRITICAL-1
// turned into free money when there was no shared memory to consult; with a handle
// wired, the absence is dispositive.
func TestProve_UnrecordedClaimCostsTheBlock(t *testing.T) {
	h := newRailHarness(t)
	mallory := addrOf(t, "unrecorded-mallory")

	blk := &Block{vm: h.vm, txs: []*Tx{
		forgedImport(t, ids.GenerateTestID(), encodeClaim(mallory, a32(0x90), 1_000_000)),
	}}
	err := blk.verifyImports()
	if err == nil {
		t.Fatal("a block carrying a claim id nothing ever recorded was accepted")
	}
	if !strings.Contains(err.Error(), "unbacked") {
		t.Errorf("expected the rejection to name the missing record, got: %v", err)
	}
}

// TestProve_BootstrapGateAcceptsBelowTheFrontier covers the one deliberate exemption.
// C and D bootstrap independently, so below the frontier a C->D object may legitimately
// be absent — not yet exported on the replaying node, or already consumed by this very
// block's own accepted Remove. Those blocks carry the network's acceptance as their
// authority. The gate must therefore be OFF during bootstrap and ON at normal operation,
// and both halves of that need proving, because a gate stuck open is a gate that never
// checks anything — which is precisely the state the whole suite was in.
func TestProve_BootstrapGateAcceptsBelowTheFrontier(t *testing.T) {
	h := newRailHarness(t)
	blk := &Block{vm: h.vm, txs: []*Tx{
		forgedImport(t, ids.GenerateTestID(), encodeClaim(addrOf(t, "boot"), a32(0x90), 7)),
	}}

	if err := blk.verifyImports(); err == nil {
		t.Fatal("setup: at normal operation an unrecorded claim must be rejected")
	}

	if err := h.vm.SetState(context.Background(), uint32(luxvm.Bootstrapping)); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	if err := blk.verifyImports(); err != nil {
		t.Fatalf("below the frontier the authenticator must stand down — a replaying node has "+
			"not necessarily applied C's block yet: %v", err)
	}

	if err := h.vm.SetState(context.Background(), uint32(luxvm.Ready)); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	if err := blk.verifyImports(); err == nil {
		t.Fatal("reaching normal operation must re-arm the authenticator")
	}
}

// TestProve_ProposerNeverProposesAnImportItCannotProve covers the liveness half. A
// validator has no choice but to reject a block carrying an unprovable import — the
// proposer's execution root already contains the credit — so proposing one is self-harm.
// The proposer screens the same predicate at build, which turns "unprovable" from a halt
// into a retry.
func TestProve_ProposerNeverProposesAnImportItCannotProve(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)

	// One unprovable delivery and one ordinary tx, offered together.
	h.vm.mempool.Add(forgedImport(t, ids.GenerateTestID(), encodeClaim(addrOf(t, "screen"), asset, 999)))
	h.vm.mempool.Add(depositTx(t, "screen-filler", asset, 1))

	built, err := h.vm.BuildBlock(context.Background())
	if err != nil {
		t.Fatalf("BuildBlock refused to build around an unprovable import: %v", err)
	}
	blk := built.(*Block)
	for _, tx := range blk.txs {
		if tx.Type == TxImport {
			t.Fatal("the proposer put an import in a block it cannot prove; a validator must " +
				"reject that block, and Reject requeues it, so the chain stops")
		}
	}
	if verr := built.Verify(context.Background()); verr != nil {
		t.Fatalf("the proposer's own block failed its own verification: %v", verr)
	}
	if aerr := built.Accept(context.Background()); aerr != nil {
		t.Fatalf("Accept: %v", aerr)
	}

	// The delivery was HELD, not discarded: a claim C has recorded but not yet flushed
	// here is unprovable now and provable in a moment, and that lag is the rail's known
	// liveness property, not an invalidity.
	if h.vm.mempool.Len() == 0 {
		t.Fatal("the unprovable delivery was dropped; a not-yet-flushed claim must survive to " +
			"be screened again")
	}
}

// TestProve_OverExportWritesNoClaim pins the export side's refusal. An export debits
// before it writes, so an over-debit must leave BOTH sides untouched: the balance
// unmoved and — the part that matters — no C-bound claim accumulated. A claim written
// for value D never debited is a mint on C.
func TestProve_OverExportWritesNoClaim(t *testing.T) {
	h := newRailHarness(t)
	asset := a32(0x90)
	owner := acctFor(t, "over-export")
	ownerAddr := addrOf(t, "over-export")

	claimID := h.cross(t, repeatID(0xD5), 0, ownerAddr, asset, 100)
	h.vm.mempool.Add(h.deliver(t, claimID))
	buildVerifyAccept(t, h.vm)
	if got := h.avail(t, owner.account, asset); got != 100 {
		t.Fatalf("setup: owner holds %d, want 100", got)
	}

	overlay := versiondb.New(h.vm.db)
	ar := newAtomicRequests()
	outID, ok, err := h.vm.executeExport(overlay, ar, repeatID(0xD6), owner.account, asset, 101, ownerAddr)
	if err != nil {
		t.Fatalf("an over-export must be a deterministic per-tx reject, not a block abort: %v", err)
	}
	if ok || outID != ids.Empty {
		t.Fatalf("an export of 101 against a balance of 100 succeeded (ok=%v claim=%s)", ok, outID)
	}
	if !ar.empty() {
		t.Fatal("a refused export accumulated a C-bound claim; C would be credited value D " +
			"never debited")
	}
	if got, _ := getAvailable(overlay, owner.account, asset); got != 100 {
		t.Fatalf("a refused export moved the balance to %d, want 100", got)
	}

	// And the error the ledger raises is the one the export path translates into that
	// reject, rather than a fault that would abort the block.
	if derr := debitAvailable(overlay, owner.account, asset, 101); !errors.Is(derr, ErrInsufficientAvailable) {
		t.Fatalf("over-debit raised %v, want ErrInsufficientAvailable", derr)
	}
}

// --- commitAtomic: the no-batch rule and the commit-then-apply order ----------------

// recordingOverlay and recordingMemory observe the ONE ordering decision commitAtomic
// makes. Both record into a shared trace so the sequence is a single readable list.
type railTrace struct {
	steps []string
	sm    atomic.SharedMemory
}

type recordingOverlay struct {
	tr        *railTrace
	commitErr error
}

func (o *recordingOverlay) Commit() error {
	o.tr.steps = append(o.tr.steps, "commit")
	return o.commitErr
}
func (o *recordingOverlay) CommitBatch() (database.Batch, error) {
	o.tr.steps = append(o.tr.steps, "batch")
	return nil, errors.New("a batch cannot cross a process boundary")
}
func (o *recordingOverlay) Abort() { o.tr.steps = append(o.tr.steps, "abort") }

type recordingMemory struct {
	tr       *railTrace
	applyErr error
}

func (m *recordingMemory) Apply(reqs map[ids.ID]*atomic.Requests, batches ...database.Batch) error {
	if len(batches) > 0 {
		m.tr.steps = append(m.tr.steps, "apply+batch")
		return errors.New("dchain test: a batch reached Apply")
	}
	m.tr.steps = append(m.tr.steps, "apply")
	return m.applyErr
}
func (m *recordingMemory) Get(peer ids.ID, keys [][]byte) ([][]byte, error) {
	return m.tr.sm.Get(peer, keys)
}
func (m *recordingMemory) Indexed(peer ids.ID, traits [][]byte, start []byte, startKey []byte, limit int) ([][]byte, []byte, []byte, error) {
	return m.tr.sm.Indexed(peer, traits, start, startKey, limit)
}

// TestProve_CommitLandsBeforeApplyAndCarriesNoBatch pins the two properties the
// accept-time commit rests on, neither of which had a single reference.
//
// NO BATCH: the D-Chain is an out-of-process plugin and its SharedMemory is the ZAP
// client, so a database.Batch cannot reach Apply — the transport refuses one outright.
// That is what makes a single atomic write unavailable and forces the choice below.
//
// COMMIT FIRST: without a shared batch the choice is at-most-once (commit first, a crash
// SKIPS an op) or at-least-once (apply first, a crash REPLAYS one). Replay is not
// survivable — a duplicate Put is a fatal Accept, and a Put replayed after the peer
// consumed the object RE-CREATES value. A skip is survivable on both legs, because
// neither side's replay protection depends on the shared-memory op.
func TestProve_CommitLandsBeforeApplyAndCarriesNoBatch(t *testing.T) {
	h := newRailHarness(t)
	tr := &railTrace{sm: h.dSM}
	h.vm.runtime.SharedMemory = &recordingMemory{tr: tr}

	ar := newAtomicRequests()
	ar.forChain(h.cChainID).RemoveRequests = [][]byte{{1, 2, 3}}

	if err := h.vm.commitAtomic(&recordingOverlay{tr: tr}, ar); err != nil {
		t.Fatalf("commitAtomic: %v", err)
	}
	if got := strings.Join(tr.steps, ","); got != "commit,apply" {
		t.Fatalf("commitAtomic ordering = %q, want \"commit,apply\" — applying first makes a "+
			"crash REPLAY a shared-memory op, and a replayed Put re-creates value", got)
	}

	// An Apply that ERRORS stays fatal: a node that cannot mutate shared memory must stop
	// rather than run on with a divergent view of it.
	tr.steps = nil
	h.vm.runtime.SharedMemory = &recordingMemory{tr: tr, applyErr: errors.New("peer gone")}
	if err := h.vm.commitAtomic(&recordingOverlay{tr: tr}, ar); err == nil {
		t.Fatal("a failed Apply must stop the node, not be swallowed")
	}

	// With no operations there is nothing to apply, so shared memory is never touched.
	tr.steps = nil
	if err := h.vm.commitAtomic(&recordingOverlay{tr: tr}, newAtomicRequests()); err != nil {
		t.Fatalf("commitAtomic with no ops: %v", err)
	}
	if got := strings.Join(tr.steps, ","); got != "commit" {
		t.Fatalf("a block with no crossings did %q, want just \"commit\"", got)
	}
}
