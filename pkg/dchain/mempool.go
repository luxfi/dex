// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"sync"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/ids"
)

// mempool.go is the d-chain pending-transaction queue. ZAP write frames
// (clob_place/cancel/submit) are NOT executed synchronously — the handler decodes
// the frame into a Tx, Adds it here, and the mempool signals the consensus engine
// (init.ToEngine <- PendingTxs) that there is work. The engine responds by
// calling VM.BuildBlock, which drains the mempool in arrival (sequence) order.
// This is the single seam between "a client asked for an order" and "consensus
// decides the order's effect": no order mutates the book outside a verified,
// accepted block.

// mempool is a FIFO of pending transactions guarded by a mutex. It is small and
// deliberately simple: ordering is arrival order, which is the sequence the
// matcher consumes. (A fee-priority mempool is a future concern; the venue's CLOB
// is sequence-fair by design.)
//
// CANCEL / TOMBSTONE (synchronous-or-nothing custody, RED #4): a custody op that
// the caller GAVE UP on (its ZAP submission timed out) must NOT silently commit in
// a LATER block — the EVM rolled back its vault leg, so a stranded commit would
// MINT (deposit) or double-release (withdraw). submitTx therefore calls cancel()
// on ctx-timeout: if the tx is still pending it is removed outright; if it has
// already been drained into an in-flight block it is TOMBSTONED so it is excluded
// from any subsequent Drain (covering a rejected-block Requeue). Tombstones are a
// PROPOSER-LOCAL block-CONSTRUCTION concern only — they decide what the proposer
// puts INTO a block; they never affect how a validator VERIFIES a built block, so
// they are consensus-neutral (a validator that never saw the cancelled tx is
// unaffected; the one node that did simply won't re-propose it).
type mempool struct {
	mu  sync.Mutex
	txs []*Tx

	// tombstones maps a cancelled tx id (cancelled AFTER it left the pending queue,
	// i.e. drained into an in-flight block) to the height of the in-flight block it
	// was drained into (lastAcceptedHeight+1 at cancel time — the proxy builds one
	// block at a time on this linear chain). A tombstoned id is filtered out of every
	// future Drain/Requeue so a cancelled custody tx can never land in a LATER block.
	// The stamped height lets gcTombstones reclaim a tombstone once that height has
	// been Accepted (the tx can no longer be in flight: either it committed in an
	// accepted block — seen: dedup covers any retry — or its block was rejected and
	// Requeue already dropped the tombstone). Lazily allocated; empty on the hot path
	// so order-frame draining pays nothing.
	tombstones map[ids.ID]uint64

	// toEngine is the VM->engine notification channel from block.Init. Sending
	// PendingTxs tells the engine to build a block. nil before Initialize.
	toEngine chan<- block.Message
}

// newMempool creates an empty mempool wired to the engine notification channel.
func newMempool(toEngine chan<- block.Message) *mempool {
	return &mempool{toEngine: toEngine}
}

// Add appends a tx and signals the engine that a block can be built. The signal
// is non-blocking: PendingTxs is a level-trigger ("there is work"), so a full
// channel (engine already notified) is fine to drop — the engine will drain the
// mempool fully on its next BuildBlock regardless of how many signals it saw.
func (m *mempool) Add(tx *Tx) {
	m.mu.Lock()
	m.txs = append(m.txs, tx)
	m.mu.Unlock()
	m.signal()
}

// signal sends a non-blocking PendingTxs to the engine.
func (m *mempool) signal() {
	if m.toEngine == nil {
		return
	}
	select {
	case m.toEngine <- block.Message{Type: block.PendingTxs}:
	default:
	}
}

// Len returns the number of pending txs.
func (m *mempool) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.txs)
}

// cancel withdraws a tx the submitter gave up on (RED #4). If the tx is still
// pending it is removed from the queue outright (the common timeout case: a
// queued, not-yet-built tx — exactly the orphan the bug described). If it has
// already been drained into an in-flight block it is TOMBSTONED (stamped with
// inFlightHeight, the height of that block) so a later Drain (e.g. after that
// block is rejected and Requeued) excludes it. Either way the cancelled tx can
// never commit in a LATER block. Idempotent and lock-safe.
func (m *mempool) cancel(txID ids.ID, inFlightHeight uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, tx := range m.txs {
		if tx.ID() == txID {
			m.txs = append(m.txs[:i], m.txs[i+1:]...)
			if len(m.txs) == 0 {
				m.txs = nil
			}
			return
		}
	}
	// Not pending: already drained into a building block at inFlightHeight.
	// Tombstone it (stamped with that height) so it is dropped from any subsequent
	// Drain/Requeue and reclaimed by gcTombstones once the height is accepted.
	if m.tombstones == nil {
		m.tombstones = make(map[ids.ID]uint64)
	}
	m.tombstones[txID] = inFlightHeight
}

// gcTombstones reclaims tombstones whose stamped in-flight height has been
// accepted (height <= acceptedHeight): such a tx can no longer be in flight, so
// its tombstone is dead weight. Called from Block.Accept after the VM's
// lastAcceptedHeight advances. Bounds tombstone-map growth under repeated
// timed-out-but-committed custody ops (RED #4 / R4). Hot path: the map is empty
// when no custody op ever timed out, so this returns immediately. MUST be called
// without holding any caller lock on the mempool (it takes m.mu itself).
func (m *mempool) gcTombstones(acceptedHeight uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.tombstones) == 0 {
		return
	}
	for id, h := range m.tombstones {
		if h <= acceptedHeight {
			delete(m.tombstones, id)
		}
	}
	if len(m.tombstones) == 0 {
		m.tombstones = nil
	}
}

// dropTombstoned filters cancelled (tombstoned) txs out of a selected batch and
// clears the tombstones it consumed. MUST be called under m.mu. On the hot path
// (no tombstones) it returns the input untouched WITHOUT hashing any tx id, so
// order-frame draining pays nothing for the custody-cancel machinery.
func (m *mempool) dropTombstoned(txs []*Tx) []*Tx {
	if len(m.tombstones) == 0 {
		return txs
	}
	out := txs[:0]
	for _, tx := range txs {
		id := tx.ID()
		if _, dead := m.tombstones[id]; dead {
			delete(m.tombstones, id)
			continue
		}
		out = append(out, tx)
	}
	return out
}

// Drain removes and returns up to max pending txs in arrival order (all of them
// if max <= 0), excluding any tombstoned (cancelled) tx. The returned slice is
// owned by the caller.
func (m *mempool) Drain(max int) []*Tx {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := len(m.txs)
	if max > 0 && max < n {
		n = max
	}
	if n == 0 {
		return nil
	}
	out := make([]*Tx, n)
	copy(out, m.txs[:n])
	m.txs = m.txs[n:]
	// Reclaim the backing array when fully drained.
	if len(m.txs) == 0 {
		m.txs = nil
	}
	out = m.dropTombstoned(out)
	if len(out) == 0 {
		return nil
	}
	return out
}

// Requeue puts txs back at the FRONT of the mempool in their original order,
// dropping any that were tombstoned (cancelled) while the block was in flight.
// Used when a built block is rejected: its (still-live) txs return to the pool to
// be reconsidered in the next block, preserving arrival order — but a cancelled
// custody tx does NOT come back (it must never commit in a later block).
func (m *mempool) Requeue(txs []*Tx) {
	if len(txs) == 0 {
		return
	}
	m.mu.Lock()
	txs = m.dropTombstoned(txs)
	if len(txs) == 0 {
		m.mu.Unlock()
		return
	}
	m.txs = append(txs, m.txs...)
	m.mu.Unlock()
	m.signal()
}
