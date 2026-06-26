// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"sync"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// results.go is the seam between consensus and a synchronous client: the per-tx
// OUTCOME a block's execution produced, and the VM's registry that lets a ZAP
// handler block until "its" tx is decided by an accepted block.
//
// The decomplected shape: a dex_* write is NOT executed when it arrives — it is
// queued, ordered by consensus, matched at Verify, committed at Accept. But the
// caller (the proxy's synchronous Relay, the precompile's swap) needs the
// resulting fills/ack back. So the handler registers a waiter keyed by the tx's
// content-addressed id, Adds the tx to the mempool, and parks. When the block
// carrying that tx is Accepted, the VM resolves the waiter with the consensus
// outcome. Verify/Reject never resolve (a rejected block's txs return to the
// mempool to be retried; only an accepted block produces a final outcome).

// txOutcome is the consensus-decided result of one transaction, attributed by tx
// id. For a submit it carries the fills; for a place it carries the resting
// order id; for a cancel the cancelled id; for ensure_market just status. status
// uses the FROZEN zapwire status bytes so the handler encodes a byte-identical
// ack/fills response without re-deriving anything.
type txOutcome struct {
	txID    ids.ID
	typ     TxType
	orderID uint64         // placed / cancelled order id (0 for submit/ensure)
	fills   []zapwire.Fill // submit fills (nil otherwise)
	status  uint8          // zapwire.StatusPlaced / StatusCanceled / StatusRejected
}

// toWireFills converts a submit's matched trades into the FROZEN wire Fill form
// (price, size, takerSide). This is the single conversion from the internal
// lx.Trade to the 17-byte wire fill; the handler then calls zapwire.EncodeFills.
func toWireFills(fills []lx.Trade, takerSide lx.Side) []zapwire.Fill {
	out := make([]zapwire.Fill, len(fills))
	for i, f := range fills {
		out[i] = zapwire.Fill{
			Price:     f.Price,
			Size:      f.Size,
			TakerSide: uint8(takerSide),
		}
	}
	return out
}

// outcomeFromApply builds the per-tx outcome for an applied tx. It is called in
// execution order so the outcome's identity (the tx id) is exactly what the
// handler registered its waiter under.
func outcomeFromApply(tx *Tx, res applyResult) txOutcome {
	o := txOutcome{txID: tx.ID(), typ: tx.Type}
	switch tx.Type {
	case TxPlace:
		if res.Placed != nil {
			o.orderID = res.Placed.ID
			o.status = zapwire.StatusPlaced
		} else {
			o.status = zapwire.StatusRejected
		}
	case TxCancel:
		if res.Canceled != 0 {
			o.orderID = res.Canceled
			o.status = zapwire.StatusCanceled
		} else {
			o.status = zapwire.StatusRejected
		}
	case TxSubmit:
		o.fills = toWireFills(res.Fills, res.TakerSide)
		o.status = zapwire.StatusPlaced
	case TxEnsureMarket:
		o.status = zapwire.StatusPlaced
	}
	return o
}

// outcomeRegistry maps a pending tx id to the channel a parked handler waits on.
// A buffered (cap 1) channel means resolve never blocks even if the waiter has
// already given up (ctx timeout) and stopped receiving.
type outcomeRegistry struct {
	mu      sync.Mutex
	waiters map[ids.ID]chan txOutcome
}

func newOutcomeRegistry() *outcomeRegistry {
	return &outcomeRegistry{waiters: map[ids.ID]chan txOutcome{}}
}

// register returns a channel that will receive the outcome for txID. If a waiter
// already exists for the id (a duplicate in-flight submit), the existing channel
// is returned so both callers observe the same single consensus outcome.
func (r *outcomeRegistry) register(txID ids.ID) chan txOutcome {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ch, ok := r.waiters[txID]; ok {
		return ch
	}
	ch := make(chan txOutcome, 1)
	r.waiters[txID] = ch
	return ch
}

// resolve delivers an outcome to its waiter (if any) and removes the entry. The
// send is non-blocking on a cap-1 channel; a missing waiter (no local handler
// parked — e.g. on a follower validator) is a no-op.
func (r *outcomeRegistry) resolve(o txOutcome) {
	r.mu.Lock()
	ch, ok := r.waiters[o.txID]
	if ok {
		delete(r.waiters, o.txID)
	}
	r.mu.Unlock()
	if ok {
		ch <- o // cap-1, never blocks
	}
}

// cancel drops a waiter without delivering (used when the handler abandons the
// wait, so a later resolve does not leak the entry).
func (r *outcomeRegistry) cancel(txID ids.ID) {
	r.mu.Lock()
	delete(r.waiters, txID)
	r.mu.Unlock()
}
