// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"testing"

	"github.com/luxfi/database"
	"github.com/luxfi/ids"
	"github.com/luxfi/runtime"
	"github.com/luxfi/vm/chains/atomic"
)

// seam_commit_nobatch_test.go pins the ONE property whose regression halts this
// chain: commitSeamAtomic must never hand a database.Batch to SharedMemory.Apply.
//
// The D-Chain ships as an out-of-process plugin, so its SharedMemory is the ZAP
// client, and that client REFUSES a batch (a batch cannot cross a process boundary).
// The previous code passed one unconditionally and had no batch-less path, so the
// first block carrying any cross-chain seam op failed at Accept — which is fatal.
// The chain would have stopped on the very first order it imported.
//
// This test is deliberately about the CALL SHAPE rather than the outcome, because
// the outcome it guards against is "the chain is dead" and there is no cheaper
// signal for it.

// recordingSharedMemory records how Apply was called. It implements only what
// commitSeamAtomic touches.
type recordingSharedMemory struct {
	applyCalls  int
	batchesSeen int
	reqs        map[ids.ID]*atomic.Requests
}

func (m *recordingSharedMemory) Get(ids.ID, [][]byte) ([][]byte, error) { return nil, nil }
func (m *recordingSharedMemory) Indexed(ids.ID, [][]byte, []byte, []byte, int) ([][]byte, []byte, []byte, error) {
	return nil, nil, nil, nil
}

func (m *recordingSharedMemory) Apply(reqs map[ids.ID]*atomic.Requests, batches ...database.Batch) error {
	m.applyCalls++
	m.batchesSeen += len(batches)
	m.reqs = reqs
	return nil
}

var _ atomic.SharedMemory = (*recordingSharedMemory)(nil)

// recordingOverlay records which commit path commitSeamAtomic took.
type recordingOverlay struct {
	commits      int
	commitBatch  int
	aborts       int
	commitErr    error
	batchToBuild database.Batch
}

func (o *recordingOverlay) Commit() error { o.commits++; return o.commitErr }
func (o *recordingOverlay) CommitBatch() (database.Batch, error) {
	o.commitBatch++
	return o.batchToBuild, nil
}
func (o *recordingOverlay) Abort() { o.aborts++ }

var _ versionDB = (*recordingOverlay)(nil)

// TestCommitSeamAtomicNeverPassesABatch is the regression gate. If this fails, the
// D-Chain halts on its first cross-chain seam op under the shipping plugin
// transport — do not "fix" it by teaching atomiczap to take a batch; a batch
// genuinely cannot cross the process boundary.
func TestCommitSeamAtomicNeverPassesABatch(t *testing.T) {
	sm := &recordingSharedMemory{}
	vm := &VM{runtime: &runtime.Runtime{SharedMemory: sm}, cChainID: ids.ID{0xCC}}

	ar := newAtomicRequests()
	req := ar.forChain(ids.ID{0xCC})
	req.RemoveRequests = append(req.RemoveRequests, []byte("an-imported-order-id-32-bytes..."))

	overlay := &recordingOverlay{}
	if err := vm.commitSeamAtomic(overlay, ar); err != nil {
		t.Fatalf("commitSeamAtomic returned an error: %v", err)
	}

	if sm.applyCalls != 1 {
		t.Fatalf("Apply must be called exactly once, got %d", sm.applyCalls)
	}
	if sm.batchesSeen != 0 {
		t.Fatalf("Apply was handed %d batch(es); the ZAP transport refuses ANY batch and "+
			"the chain dies at Accept on the first seam op", sm.batchesSeen)
	}
	if overlay.commitBatch != 0 {
		t.Fatalf("the overlay must be committed with Commit(), not CommitBatch(); got %d CommitBatch calls", overlay.commitBatch)
	}
	if overlay.commits != 1 {
		t.Fatalf("the overlay state must be committed exactly once, got %d", overlay.commits)
	}
}

// TestCommitSeamAtomicCommitsStateBeforeApplying pins the ORDER, which is the part
// that decides what a crash between the two writes costs.
//
// State first means an op can be SKIPPED (at-most-once). Apply first would mean an
// op can be REPLAYED (at-least-once), and a replayed Put is not survivable here: the
// shared-memory layer errors on a duplicate put (fatal Accept) and, if the peer has
// already consumed the object, re-creates it. A skip is survivable because neither
// side's replay protection depends on the shared-memory op — D's guard is the
// durable `seamintent:` escrow row, written by the state commit this test pins as
// happening first.
func TestCommitSeamAtomicCommitsStateBeforeApplying(t *testing.T) {
	sm := &recordingSharedMemory{}
	vm := &VM{runtime: &runtime.Runtime{SharedMemory: sm}, cChainID: ids.ID{0xCC}}

	ar := newAtomicRequests()
	req := ar.forChain(ids.ID{0xCC})
	req.RemoveRequests = append(req.RemoveRequests, []byte("an-imported-order-id-32-bytes..."))

	// A failing state commit must abort BEFORE shared memory is touched: if the state
	// did not land, the consumption it records did not land either, so mutating shared
	// memory would consume an object with no committed record of having done so.
	overlay := &recordingOverlay{commitErr: errCommitBoom}
	if err := vm.commitSeamAtomic(overlay, ar); err == nil {
		t.Fatal("a failed state commit must surface as an error")
	}
	if sm.applyCalls != 0 {
		t.Fatalf("shared memory was mutated after the state commit FAILED (%d Apply calls) — "+
			"the consumption would have no committed record", sm.applyCalls)
	}
}

var errCommitBoom = errBoom{}

type errBoom struct{}

func (errBoom) Error() string { return "dchain: injected commit failure" }
