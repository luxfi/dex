// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"sync"

	"github.com/luxfi/consensus/engine/chain/block"
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
type mempool struct {
	mu  sync.Mutex
	txs []*Tx

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

// Drain removes and returns up to max pending txs in arrival order (all of them
// if max <= 0). The returned slice is owned by the caller.
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
	return out
}

// Requeue puts txs back at the FRONT of the mempool in their original order. Used
// when a built block is rejected: its txs return to the pool to be reconsidered
// in the next block, preserving arrival order.
func (m *mempool) Requeue(txs []*Tx) {
	if len(txs) == 0 {
		return
	}
	m.mu.Lock()
	m.txs = append(txs, m.txs...)
	m.mu.Unlock()
	m.signal()
}
