// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"fmt"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// handler.go is the d-chain's DEX ZAP surface: the dex_* methods a client (the
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
// method names (dex_ensure_market/place/cancel/submit), the same request sizes,
// and the same response codecs (zapwire.EncodeAck / zapwire.EncodeFills, 17-byte
// fill). The chains/dexvm relay client and the precompile adapter re-define the
// identical constants (they cannot import this cgo-tagged package), so a frame
// built by any of them is byte-identical to what this server consumes/produces.

// dexMethod pairs a frozen ZAP method name with its raw handler. The handler
// signature is github.com/luxfi/rpc.RawHandler (func(ctx, payload) (resp, err)) —
// the SAME func value whether it is reached over the ZAP socket transport
// (RegisterDEX) or over the node's HTTP router (CreateHandlers / ingest.go). One
// table, two transports: the ingestion seam never duplicates the handler set.
type dexMethod struct {
	method  string
	handler rpc.RawHandler
}

// dexMethods is the single authoritative method->handler table for the DEX
// surface. Both transports fold over it: RegisterDEX registers each on a ZAP
// rpc.Server; the HTTP mux (ingest.go) routes POST .../<method> to the same
// handler. A method added here is reachable over both transports automatically.
//
// THE FUNDING RAIL HAS NO METHOD HERE, IN EITHER DIRECTION, AND THAT IS THE WHOLE
// REASON IT MOVES NO VALUE TODAY. submitTx is the only caller of mempool.Add and this
// table is the only caller of submitTx; there is no gossip handler, so a tx that is not
// in this table cannot exist. TxImport and TxExport are therefore unreachable, which
// means every C->D deposit would lock on activation: C debits the depositor into
// custody, stages a claim, and nothing on D can consume it.
//
// The thing that used to build these was a proposer-side drive that scanned shared
// memory and ran ORDERS off the crossing. Removing it is what made the rail carry value
// and nothing else, so it is not coming back. What replaces it delivers a claim and
// stops there.
//
// AN IMPORT METHOD IS NOT A METHOD, IT IS AN AUTHORIZATION MODEL. TxImport carries no
// signature on purpose — a claim can only credit the account it already names, so
// anyone may deliver one and a relayer who refuses cannot strand a depositor. That
// makes the ingress an unauthenticated, free-to-call, value-moving endpoint, and the
// only thing standing between it and a chain halt is that the proposer screens what it
// cannot prove (vm.screenImports) and that an already-consumed claim needs no proof
// (vm.proveClaim). Both of those are new. Neither had ever run in a test before the
// authenticator was switched on, and the first thing switching it on revealed was that
// an ordinary duplicate delivery stopped the chain permanently.
//
// So the order is: those hold first, then who may deliver gets decided, then a method
// goes here. Adding one before that is adding a public halt.
func (vm *VM) dexMethods() []dexMethod {
	return []dexMethod{
		{zapwire.MethodEnsureMarket, vm.handleEnsureMarket},
		{zapwire.MethodPlace, vm.handlePlace},
		{zapwire.MethodCancel, vm.handleCancel},
		{zapwire.MethodSubmit, vm.handleSubmit},
		{zapwire.MethodOpenMarket, vm.handleOpenMarket},
		{zapwire.MethodDeposit, vm.handleDeposit},
		{zapwire.MethodWithdraw, vm.handleWithdraw},
	}
}

// RegisterDEX wires the dex_* handlers onto an rpc.Server (the ZAP socket
// transport). It is the seam the standalone venue entrypoint (cmd/dvenue) and the
// socket-level tests use. The in-luxd plugin does NOT use this path — it exposes
// the SAME handlers over the node's HTTP router via CreateHandlers (ingest.go).
// Additive: the caller may register other methods on the same server.
func (vm *VM) RegisterDEX(server rpc.Server) error {
	for _, m := range vm.dexMethods() {
		if err := server.RegisterRaw(m.method, m.handler); err != nil {
			return fmt.Errorf("dchain: register %s: %w", m.method, err)
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
	// The RPC payload for a money-moving method is [frozen-body][auth-envelope];
	// parseTxFrame splits it on the fixed body size. An admin method is body-only.
	tx, err := parseTxFrame(t, payload)
	if err != nil {
		return txOutcome{}, err
	}
	// STATELESS pre-screen at the boundary: reject a forged/unsigned/wrong-signer
	// frame before it enters the mempool (fail-fast UX + mempool hygiene). This is
	// NOT the security boundary — the AUTHORITATIVE gate (signature + monotonic
	// nonce + cancel ownership) re-runs deterministically for every validator in
	// block.execute, so a tx that bypasses this RPC path is still caught at
	// consensus. The nonce is intentionally NOT checked here (it is stateful and
	// racy against in-flight blocks); only the consensus gate consumes nonces.
	signer, verr := vm.verifyTxSignature(tx)
	if verr != nil {
		return txOutcome{}, verr
	}
	// Boundary pre-screen for the deposit backing gate: reject an unauthorized
	// deposit (not from the configured bridge authority) at the PUBLIC RPC edge so
	// it never enters the mempool. This mirrors the AUTHORITATIVE gate (authorizeTx)
	// that re-runs at consensus, so a deposit that bypasses this RPC path is still
	// rejected — but rejecting here keeps the public route from admitting an unbacked
	// mint attempt at all. Fail-closed when no authority is configured.
	if tx.Type == TxDeposit && (vm.depositAuthority == (userKey{}) || signer != vm.depositAuthority) {
		return txOutcome{}, fmt.Errorf("%w: deposit not authorized by the configured bridge authority", ErrTxBadSignature)
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
		// The caller gave up. Drop the waiter AND withdraw the tx from the mempool
		// (RED #4): a custody op (deposit/withdraw) whose ZAP submission timed out
		// must NOT silently commit in a LATER block — the EVM rolled back its vault
		// leg, so a stranded deposit would MINT and a stranded withdraw would
		// double-release. mempool.cancel removes it if still pending, or tombstones
		// it if already drained into an in-flight block, so it can never land later.
		// (For an order frame this is also correct: a submit the caller abandoned
		// should not rest/cross in a future block under the caller's account.)
		// Idempotency still makes an explicit RETRY of the same op safe — a retry
		// re-Adds a fresh tx (and, for custody, the same ref dedups it exactly once
		// if the original DID commit before the timeout).
		vm.outcomes.cancel(txID)
		// Stamp the tombstone with the height of the single in-flight block the tx
		// could be in (lastAcceptedHeight+1) so gcTombstones can reclaim it once that
		// height is accepted (R4). inFlightHeight reads the watermark under vm.mu.
		vm.mempool.cancel(txID, vm.inFlightHeight())
		return txOutcome{}, ctx.Err()
	}
}

// handleEnsureMarket: payload = poolId[32]. Queues an idempotent market-create tx
// and returns ack(0, placed) once consensus records the market.
func (vm *VM) handleEnsureMarket(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.EnsureMarketReqSize {
		return encodeReject(0, "dex_ensure_market: short payload"), nil
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
		return encodeReject(0, "dex_place: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxPlace, payload)
	if err != nil {
		return nil, err
	}
	if o.status == zapwire.StatusRejected {
		return encodeReject(0, "dex_place: order rejected"), nil
	}
	return zapwire.EncodeAck(o.orderID, o.status, 0), nil
}

// handleCancel: payload = zapwire Cancel (40B). Queues a cancel tx and returns
// ack(orderId, canceled|rejected).
func (vm *VM) handleCancel(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.CancelReqSize {
		return encodeReject(0, "dex_cancel: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxCancel, payload)
	if err != nil {
		return nil, err
	}
	if o.status == zapwire.StatusRejected {
		return encodeReject(o.orderID, "dex_cancel: order not canceled"), nil
	}
	return zapwire.EncodeAck(o.orderID, o.status, 0), nil
}

// handleSubmit: payload = zapwire Submit (66B). Queues a marketable-order tx and
// returns the CONSENSUS-computed fills in the frozen response form: count[4] then
// count × (price[8] + size[8] + takerSide[1]). A deduped (replayed) submit
// returns zero fills — it was already executed exactly once.
func (vm *VM) handleSubmit(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.SubmitReqSize {
		return nil, fmt.Errorf("dex_submit: short payload: %d", len(payload))
	}
	o, err := vm.submitTx(ctx, TxSubmit, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.EncodeFills(o.fills), nil
}

// handleOpenMarket: payload = zapwire OpenMarket (48B). Binds the market's
// (base,quote) asset handles so orders on it can be value-checked, then acks.
func (vm *VM) handleOpenMarket(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.OpenMarketReqSize {
		return encodeReject(0, "dex_open_market: short payload"), nil
	}
	o, err := vm.submitTx(ctx, TxOpenMarket, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.EncodeAck(0, o.status, 0), nil
}

// handleDeposit: payload = zapwire Deposit (32B). Credits the account's available
// balance from value the proxy atomically imported, then returns the realized
// credited amount (status[1] + amount[8]).
func (vm *VM) handleDeposit(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.DepositReqSize {
		return zapwire.EncodeBalanceResp(zapwire.StatusRejected, 0), nil
	}
	o, err := vm.submitTx(ctx, TxDeposit, payload)
	if err != nil {
		return nil, err
	}
	// orderID carries the realized credited amount.
	return zapwire.EncodeBalanceResp(o.status, o.orderID), nil
}

// handleWithdraw: payload = zapwire Withdraw (32B). Debits the account's realized
// available balance (clamped) and returns the REALIZED amount the proxy must
// export (status[1] + amount[8]). A zero realized withdraw is StatusRejected so
// the proxy builds no empty export leg.
func (vm *VM) handleWithdraw(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < zapwire.WithdrawReqSize {
		return zapwire.EncodeBalanceResp(zapwire.StatusRejected, 0), nil
	}
	o, err := vm.submitTx(ctx, TxWithdraw, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.EncodeBalanceResp(o.status, o.orderID), nil
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
