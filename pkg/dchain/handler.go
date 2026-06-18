// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"fmt"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// handler.go is the d-chain's CLOB ZAP surface: the clob_* methods a client (the
// chains/dexvm proxy relay, the EVM precompile adapter, the maker) calls over
// github.com/luxfi/rpc. It is the gateway INTO consensus, not a matcher.
//
// THE CRITICAL DECOMPLECT: a write (place/cancel/submit) is NOT executed when it
// arrives. The handler decodes the FROZEN zapwire frame into a Tx, Adds it to the
// mempool, and BLOCKS until consensus decides it (an accepted block resolves the
// waiter with the matched fills/ack). So the bytes a caller gets back are the
// CONSENSUS-COMPUTED result — every validator re-derived the same fills at Verify
// — not a synchronous, single-process book mutation. Reads (book/state) are
// served directly from the authoritative in-RAM book (a fold of the committed
// rows), needing no consensus round-trip.
//
// WIRE PARITY: every frame here is the FROZEN form from pkg/zapwire — the same
// method names (clob_ensure_market/place/cancel/submit), the same request sizes,
// and the same response codecs (zapwire.EncodeAck / zapwire.EncodeFills, 17-byte
// fill). The chains/dexvm relay client and the precompile adapter re-define the
// identical constants (they cannot import this cgo-tagged package), so a frame
// built by any of them is byte-identical to what this server consumes/produces.

// RegisterCLOB wires the four clob_* handlers onto an rpc.Server. It is the
// integration seam the venue entrypoint (cmd/dchain) and tests use to expose the
// d-chain matcher over ZAP. Additive: the caller may register other methods on
// the same server.
func (vm *VM) RegisterCLOB(server rpc.Server) error {
	regs := []struct {
		method  string
		handler rpc.RawHandler
	}{
		{zapwire.MethodEnsureMarket, vm.handleEnsureMarket},
		{zapwire.MethodPlace, vm.handlePlace},
		{zapwire.MethodCancel, vm.handleCancel},
		{zapwire.MethodSubmit, vm.handleSubmit},
	}
	for _, r := range regs {
		if err := server.RegisterRaw(r.method, r.handler); err != nil {
			return fmt.Errorf("dchain: register %s: %w", r.method, err)
		}
	}
	return nil
}

// submitTx builds a Tx from a method's frozen payload, Adds it to the mempool,
// and blocks until consensus resolves it (or ctx is done). It returns the
// per-tx outcome the accepted block produced. This is the single write path for
// all three write methods — one place that turns a wire frame into a
// consensus-decided outcome.
func (vm *VM) submitTx(ctx context.Context, t TxType, payload []byte) (txOutcome, error) {
	tx, err := NewTx(t, payload)
	if err != nil {
		return txOutcome{}, err
	}
	txID := tx.ID()

	// Register the waiter BEFORE Add so the outcome cannot be resolved (by a fast
	// Accept) before we are listening. register is idempotent on txID, so a
	// duplicate in-flight frame shares the one waiter.
	ch := vm.outcomes.register(txID)

	vm.mempool.Add(tx)

	select {
	case o := <-ch:
		return o, nil
	case <-ctx.Done():
		// The caller gave up; drop the waiter so a later Accept does not leak it.
		// The tx may still be accepted (its effect is durable) — idempotency makes
		// a retry safe.
		vm.outcomes.cancel(txID)
		return txOutcome{}, ctx.Err()
	}
}

// handleEnsureMarket: payload = poolId[32]. Queues an idempotent market-create tx
// and returns ack(0, placed) once consensus records the market.
func (vm *VM) handleEnsureMarket(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.EnsureMarketReqSize {
		return encodeReject(0, "clob_ensure_market: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxEnsureMarket, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.EncodeAck(0, o.status, 0), nil
}

// handlePlace: payload = zapwire Place (65B). Queues a resting-limit tx and
// returns ack(orderId, placed|rejected) with the CONSENSUS-assigned order id.
func (vm *VM) handlePlace(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.PlaceReqSize {
		return encodeReject(0, "clob_place: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxPlace, payload)
	if err != nil {
		return nil, err
	}
	if o.status == zapwire.StatusRejected {
		return encodeReject(0, "clob_place: order rejected"), nil
	}
	return zapwire.EncodeAck(o.orderID, o.status, 0), nil
}

// handleCancel: payload = zapwire Cancel (40B). Queues a cancel tx and returns
// ack(orderId, canceled|rejected).
func (vm *VM) handleCancel(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.CancelReqSize {
		return encodeReject(0, "clob_cancel: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxCancel, payload)
	if err != nil {
		return nil, err
	}
	if o.status == zapwire.StatusRejected {
		return encodeReject(o.orderID, "clob_cancel: order not canceled"), nil
	}
	return zapwire.EncodeAck(o.orderID, o.status, 0), nil
}

// handleSubmit: payload = zapwire Submit (66B). Queues a marketable-order tx and
// returns the CONSENSUS-computed fills in the frozen response form: count[4] then
// count × (price[8] + size[8] + takerSide[1]). A deduped (replayed) submit
// returns zero fills — it was already executed exactly once.
func (vm *VM) handleSubmit(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.SubmitReqSize {
		return nil, fmt.Errorf("clob_submit: short payload: %d", len(payload))
	}
	o, err := vm.submitTx(ctx, TxSubmit, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.EncodeFills(o.fills), nil
}

// encodeReject builds the legacy reject frame the gateway uses:
// orderId[8] + status[1=rejected] + reasonLen[2] + reason. The proxy's DecodeAck
// reads status from byte 8 and ignores the rest, so a reject is wire-compatible
// with an ack on the status-read path; DecodeFills on a reject errors (too
// short), which the proxy treats as "no fills".
func encodeReject(orderID uint64, reason string) []byte {
	rb := []byte(reason)
	if len(rb) > 256 {
		rb = rb[:256]
	}
	out := make([]byte, 11+len(rb))
	// orderId[0:8] big-endian.
	for i := 0; i < 8; i++ {
		out[7-i] = byte(orderID >> (8 * i))
	}
	out[8] = zapwire.StatusRejected
	out[9] = byte(len(rb) >> 8)
	out[10] = byte(len(rb))
	copy(out[11:], rb)
	return out
}
