// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import "testing"

// mempool_gc_test.go pins R4: a tombstone (for a custody tx the submitter gave up
// on AFTER it was drained into an in-flight block) is reclaimed once its stamped
// in-flight height has been Accepted — such a tx can no longer be in flight (it
// either committed in an accepted block, where seen: dedup covers any retry, or its
// block was rejected and Requeue already dropped the tombstone). This bounds
// tombstone-map growth under repeated timed-out-but-committed custody ops.

// TestMempoolTombstoneGCReclaimsAcceptedHeight: a tombstone stamped at in-flight
// height H is dropped by gcTombstones(H). Observable: after GC the same tx, re-added
// as pending, SURVIVES Drain (it is no longer filtered).
func TestMempoolTombstoneGCReclaimsAcceptedHeight(t *testing.T) {
	m := newMempool(nil)
	tx := depositTxRef(t, "gc", a32(1), 10, [32]byte{0x01})

	// Tombstone it as if drained into the in-flight block at height 7 (the not-pending
	// path: cancel on an empty queue tombstones rather than removing).
	m.cancel(tx.ID(), 7)
	m.mu.Lock()
	n := len(m.tombstones)
	m.mu.Unlock()
	if n != 1 {
		t.Fatalf("tombstones after cancel = %d, want 1", n)
	}

	// Height 7 is now accepted -> the tombstone is dead weight and reclaimed.
	m.gcTombstones(7)
	m.mu.Lock()
	n = len(m.tombstones)
	m.mu.Unlock()
	if n != 0 {
		t.Fatalf("tombstones after gc(7) = %d, want 0 (reclaimed)", n)
	}

	// Proof of behavior: the reclaimed tx, re-added, is NOT filtered out of Drain.
	m.Add(tx)
	got := m.Drain(0)
	if len(got) != 1 || got[0].ID() != tx.ID() {
		t.Fatalf("after GC the reclaimed tx must survive Drain; got %d txs", len(got))
	}
}

// TestMempoolTombstoneGCRetainsInFlightHeight: a tombstone stamped at in-flight
// height H is RETAINED by gcTombstones(H-1) — the block at height H has NOT been
// accepted, so the tx could still be in flight and the tombstone must persist.
// Observable: the tx, re-added as pending, is still filtered out of Drain.
func TestMempoolTombstoneGCRetainsInFlightHeight(t *testing.T) {
	m := newMempool(nil)
	tx := depositTxRef(t, "keep", a32(2), 20, [32]byte{0x02})

	// Tombstone stamped at in-flight height 7.
	m.cancel(tx.ID(), 7)

	// Only height 6 is accepted -> height 7 is still in flight -> tombstone retained.
	m.gcTombstones(6)
	m.mu.Lock()
	n := len(m.tombstones)
	m.mu.Unlock()
	if n != 1 {
		t.Fatalf("tombstones after gc(6) = %d, want 1 (retained — block 7 not yet accepted)", n)
	}

	// Proof of behavior: the still-tombstoned tx, re-added, is filtered out of Drain
	// (it must never land in a later block while it could still be in flight).
	m.Add(tx)
	got := m.Drain(0)
	if len(got) != 0 {
		t.Fatalf("a retained-tombstone tx must be filtered from Drain; got %d txs", len(got))
	}
}

// TestMempoolGCTombstonesEmptyIsNoop: gcTombstones on an empty map (the hot path,
// no custody op ever timed out) is a safe no-op.
func TestMempoolGCTombstonesEmptyIsNoop(t *testing.T) {
	m := newMempool(nil)
	m.gcTombstones(100) // must not panic
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.tombstones != nil {
		t.Fatalf("gc on empty mempool allocated tombstones, want nil")
	}
}
