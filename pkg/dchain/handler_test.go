// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
	"github.com/luxfi/rpc"
)

// handler_test.go proves the d-chain DEX ZAP surface end to end over a REAL
// rpc.Listen socket, using the SAME client (rpc.ZAPDial) the chains/dexvm proxy
// relay uses, and decoding responses with a locally-redefined framing that is
// byte-identical to chains/dexvm/relay.go. This is the step-4 proxy-boundary
// byte-parity proof: a frame built/sent/decoded exactly as the proxy would.
//
// Because a write blocks until consensus accepts it, the test runs an auto-sealer
// goroutine that plays the consensus engine: on a PendingTxs signal it drains the
// mempool via BuildBlock -> Verify -> Accept. This is the harness standing in for
// the real engine that cmd/dexd (plugin mode) wires via rpc.Serve.

// ---- relay-side framing, redefined locally to prove byte-parity (must equal
// chains/dexvm/relay.go exactly: methods, FillWireSize=17, count[4]+N*17) ----

const relayFillWireSize = 17

const (
	relayMethodEnsureMarket = "dex_ensure_market"
	relayMethodPlace        = "dex_place"
	relayMethodCancel       = "dex_cancel"
	relayMethodSubmit       = "dex_submit"
)

type relayFill struct {
	Price float64
	Size  float64
	Side  uint8
}

// relayDecodeFills mirrors chains/dexvm/relay.go::DecodeFills byte for byte,
// including the finite-positive and side∈{0,1} validation, so a successful decode
// here proves the d-chain's dex_submit response is exactly what the proxy parses.
func relayDecodeFills(data []byte) ([]relayFill, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("fills response too short: %d", len(data))
	}
	n := int(binary.BigEndian.Uint32(data[0:4]))
	if 4+n*relayFillWireSize > len(data) {
		return nil, fmt.Errorf("fills response truncated: count=%d len=%d", n, len(data))
	}
	fills := make([]relayFill, 0, n)
	off := 4
	for i := 0; i < n; i++ {
		p := math.Float64frombits(binary.BigEndian.Uint64(data[off : off+8]))
		s := math.Float64frombits(binary.BigEndian.Uint64(data[off+8 : off+16]))
		side := data[off+16]
		off += relayFillWireSize
		if !(p > 0) || math.IsInf(p, 0) {
			return nil, fmt.Errorf("fill %d: invalid price %v", i, p)
		}
		if !(s > 0) || math.IsInf(s, 0) {
			return nil, fmt.Errorf("fill %d: invalid size %v", i, s)
		}
		if side > 1 {
			return nil, fmt.Errorf("fill %d: invalid side %d", i, side)
		}
		fills = append(fills, relayFill{Price: p, Size: s, Side: side})
	}
	return fills, nil
}

// relayDecodeAck mirrors zapwire/relay ack-read: orderId[0:8], status[8].
func relayDecodeAck(resp []byte) (orderID uint64, status uint8, err error) {
	if len(resp) < 9 {
		return 0, 0, fmt.Errorf("ack too short: %d", len(resp))
	}
	return binary.BigEndian.Uint64(resp[0:8]), resp[8], nil
}

// autoSealer plays the consensus engine for tests: it waits for PendingTxs and
// seals every pending tx into an accepted block (Build -> Verify -> Accept). It
// stops when ctx is cancelled.
func autoSealer(ctx context.Context, t *testing.T, vm *VM) {
	t.Helper()
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		msg, err := vm.WaitForEvent(ctx)
		if err != nil {
			return // ctx cancelled
		}
		if msg.Type != block.PendingTxs {
			continue
		}
		blk, err := vm.BuildBlock(ctx)
		if err != nil {
			// Empty mempool race: nothing to do, loop.
			continue
		}
		if err := blk.Verify(ctx); err != nil {
			t.Errorf("autoSealer Verify: %v", err)
			return
		}
		if err := blk.Accept(ctx); err != nil {
			t.Errorf("autoSealer Accept: %v", err)
			return
		}
	}
}

// startServer brings up a VM + a real rpc.Listen ZAP server with the dex_*
// handlers, plus the auto-sealer. Returns the dial address and a stop func.
func startServer(t *testing.T) (*VM, string, func()) {
	t.Helper()
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	server, err := rpc.Listen("127.0.0.1:0")
	if err != nil {
		t.Fatalf("rpc.Listen: %v", err)
	}
	if err := vm.RegisterDEX(server); err != nil {
		t.Fatalf("RegisterDEX: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _ = server.Serve(ctx) }()
	go func() { defer wg.Done(); autoSealer(ctx, t, vm) }()

	addr := server.Addr()
	stop := func() {
		cancel()
		_ = server.Close()
		_ = vm.Shutdown(context.Background())
		wg.Wait()
	}
	return vm, addr, stop
}

// TestHandlerByteParityOverSocket is the step-4 proxy-boundary proof: build the
// FROZEN frames, send them over a real ZAP socket via the proxy's exact client,
// and decode every response with the proxy's exact framing. ensure_market ->
// place a maker -> submit a crossing taker -> assert the fills decode (proving
// the d-chain returns CONSENSUS-computed fills in the byte-identical wire form).
func TestHandlerByteParityOverSocket(t *testing.T) {
	_, addr, stop := startServer(t)
	defer stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := rpc.ZAPDial(ctx, addr)
	if err != nil {
		t.Fatalf("ZAPDial: %v", err)
	}
	defer conn.Close()

	var pool [32]byte
	pool[0], pool[1] = 0xbe, 0xef

	// 1) ensure_market — frozen EnsureMarket frame, ack response.
	resp, err := conn.Call(ctx, relayMethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	if err != nil {
		t.Fatalf("ensure_market Call: %v", err)
	}
	if _, status, err := relayDecodeAck(resp); err != nil || status != zapwire.StatusPlaced {
		t.Fatalf("ensure_market ack: status=%d err=%v", status, err)
	}

	// 2) place a resting ask @101 size 5 — frozen Place frame + auth envelope (the
	// gate refuses an unsigned money-moving frame); wire user is the maker's
	// key-derived account, signed at its next nonce.
	resp, err = conn.Call(ctx, relayMethodPlace,
		signedPayload(t, "maker", TxPlace, zapwire.EncodePlace(pool, zapwire.SideSell, 101.0, 5.0, wireUser(t, "maker"))))
	if err != nil {
		t.Fatalf("place Call: %v", err)
	}
	makerID, status, err := relayDecodeAck(resp)
	if err != nil || status != zapwire.StatusPlaced {
		t.Fatalf("place ack: status=%d err=%v", status, err)
	}
	if makerID == 0 {
		t.Fatal("place returned zero order id")
	}

	// 3) submit a buy that crosses — frozen Submit frame, FILLS response decoded
	// with the proxy's exact DecodeFills (the byte-parity assertion).
	resp, err = conn.Call(ctx, relayMethodSubmit,
		signedPayload(t, "taker", TxSubmit, zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, 101.0, 3.0, wireUser(t, "taker"))))
	if err != nil {
		t.Fatalf("submit Call: %v", err)
	}
	fills, err := relayDecodeFills(resp)
	if err != nil {
		t.Fatalf("proxy DecodeFills rejected the d-chain response (BYTE PARITY BROKEN): %v", err)
	}
	if len(fills) != 1 {
		t.Fatalf("expected 1 fill, got %d", len(fills))
	}
	if fills[0].Side != zapwire.SideBuy {
		t.Errorf("fill taker side = %d, want buy(0)", fills[0].Side)
	}
	if fills[0].Size != 3.0 {
		t.Errorf("fill size = %v, want 3.0", fills[0].Size)
	}
	if fills[0].Price != 101.0 {
		t.Errorf("fill price = %v, want 101.0", fills[0].Price)
	}

	// Cross-check: the same response decodes identically with the dex's own
	// zapwire.DecodeFills (the two definitions are byte-identical).
	zfills, err := zapwire.DecodeFills(resp)
	if err != nil {
		t.Fatalf("zapwire.DecodeFills: %v", err)
	}
	if len(zfills) != len(fills) || zfills[0].Price != fills[0].Price ||
		zfills[0].Size != fills[0].Size || zfills[0].TakerSide != fills[0].Side {
		t.Error("zapwire and relay fill decodes disagree (frame not byte-identical)")
	}
}

// TestHandlerIdempotentReplay proves the d-chain STATE idempotency: replaying the
// identical dex_submit frame (a proxy retry on a dropped socket) executes the
// order EXACTLY ONCE — the second call returns zero fills, and the resting book
// is consumed only once.
func TestHandlerIdempotentReplay(t *testing.T) {
	vm, addr, stop := startServer(t)
	defer stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := rpc.ZAPDial(ctx, addr)
	if err != nil {
		t.Fatalf("ZAPDial: %v", err)
	}
	defer conn.Close()

	var pool [32]byte
	pool[0] = 0x11

	mustCall := func(method string, payload []byte) []byte {
		t.Helper()
		resp, err := conn.Call(ctx, method, payload)
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		return resp
	}

	mustCall(relayMethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	mustCall(relayMethodPlace, signedPayload(t, "maker", TxPlace, zapwire.EncodePlace(pool, zapwire.SideSell, 50.0, 10.0, wireUser(t, "maker"))))

	// First submit: a 4-unit buy crosses for 4 units. The signed payload is built
	// ONCE so the replay below sends BYTE-IDENTICAL bytes (same nonce, same
	// signature -> same tx id), exercising content-addressed idempotency rather than
	// a fresh (distinct-nonce) order.
	submit := signedPayload(t, "taker", TxSubmit, zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, 50.0, 4.0, wireUser(t, "taker")))
	fills1, err := zapwire.DecodeFills(mustCall(relayMethodSubmit, submit))
	if err != nil {
		t.Fatalf("submit1 decode: %v", err)
	}
	if len(fills1) != 1 || fills1[0].Size != 4.0 {
		t.Fatalf("submit1 fills = %+v, want one 4.0 fill", fills1)
	}

	// Replay the EXACT same submit frame (same bytes -> same tx id). Idempotency
	// must return the ORIGINAL outcome verbatim (the same single 4.0 fill) WITHOUT
	// re-executing against the book — the contract a caller (precompile/proxy)
	// needs when the host runs one tx multiple times. Exactly-once is proven by
	// the book invariant below (a double-execution would consume 4 MORE units);
	// the return value being the first result, not a spurious empty/reject, is
	// what lets the caller's settlement see a consistent answer on every replay.
	fills2, err := zapwire.DecodeFills(mustCall(relayMethodSubmit, submit))
	if err != nil {
		t.Fatalf("submit2 (replay) decode: %v", err)
	}
	if len(fills2) != 1 || fills2[0].Size != 4.0 {
		t.Fatalf("replayed submit = %+v, want the original one 4.0 fill (idempotent return)", fills2)
	}

	// The maker's remaining size confirms exactly-once: 10 - 4 = 6 left. THIS is
	// the no-double-execution invariant — unchanged by returning the cached fills.
	ob := vm.books[pool]
	if ob == nil {
		t.Fatal("market book missing")
	}
	var remaining float64
	for _, o := range ob.Orders {
		remaining += o.RemainingSize
	}
	if remaining != 6.0 {
		t.Errorf("resting size = %v, want 6.0 (replay must not double-consume)", remaining)
	}
}
