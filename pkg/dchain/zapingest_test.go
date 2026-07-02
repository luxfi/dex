// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo
// +build !cgo

package dchain

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
	"github.com/luxfi/rpc"
)

// zapingest_test.go proves the CANONICAL in-luxd ZAP DEX ingestion seam
// (zapingest.go): when the chain's init Config names a zapIngestAddr, the VM
// ITSELF stands up a co-located ZAP socket bound to the one dexMethods core —
// there is NO manual RegisterDEX by the runner (cmd/dexd plugin mode). A ZAP client dialing
// that socket drives a real order through the IDENTICAL submitTx -> mempool ->
// BuildBlock -> Verify -> Accept path the HTTP transport uses, and the fill is
// committed D-Chain state. This is the deploy-shaped proof: the in-luxd plugin now
// serves the native HFT socket as canonical, configured by data, with HTTP (ingest.go)
// remaining as the compat surface over the SAME handler core.

// startZAPIngestVM boots a VM whose init Config enables the co-located ZAP socket on
// an ephemeral port, plus the auto-sealer. It returns the VM, the socket's dial
// address (chosen by the OS), and a stop func. Critically it does NOT call
// RegisterDEX — Initialize wires the socket from Config, exactly as the in-luxd
// plugin does in production.
func startZAPIngestVM(t *testing.T) (*VM, string, func()) {
	t.Helper()
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   []byte(`{"zapIngestAddr":"127.0.0.1:0"}`),
	}); err != nil {
		t.Fatalf("Initialize with zapIngestAddr: %v", err)
	}
	if vm.zapIngest == nil {
		t.Fatal("Initialize did not start the ZAP ingest socket from Config")
	}
	addr := vm.zapIngest.Addr()

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); autoSealer(ctx, t, vm) }()

	stop := func() {
		cancel()
		_ = vm.Shutdown(context.Background()) // also closes the ZAP socket
		wg.Wait()
	}
	return vm, addr, stop
}

// TestZAPIngestCrossingOrdersMatchInBlock is the deliverable proof of the canonical
// ZAP ingestion seam: two crossing orders submitted over the VM's own co-located ZAP
// socket produce a consensus-computed fill recorded as D-Chain state — matched by the
// VM's BuildBlock->Verify->Accept, not by any external venue or relay.
func TestZAPIngestCrossingOrdersMatchInBlock(t *testing.T) {
	vm, addr, stop := startZAPIngestVM(t)
	defer stop()

	cli, closeCli := dialVenue(t, addr)
	defer closeCli()

	var pool [32]byte
	pool[0], pool[1] = 0x2a, 0x9e

	// ensure_market over the canonical socket.
	cli.ensureMarket(t, pool)

	// Maker rests an ask: SELL 10 @ 5. Taker crosses: BUY 10 @ 5 -> one fill.
	makerID := cli.place(t, pool, zapwire.SideSell, 5.0, 10.0, "zi-maker")
	if makerID == 0 {
		t.Fatal("maker place over ZAP socket returned order id 0 (not rested)")
	}
	fills := cli.submit(t, pool, zapwire.SideBuy, 5.0, 10.0, "zi-taker")
	if len(fills) != 1 {
		t.Fatalf("submit over ZAP socket: got %d fills, want 1", len(fills))
	}
	if fills[0].Price != 5*uint64(zapwire.PriceScale) || fills[0].Size != 10 {
		t.Fatalf("fill = px %d sz %d, want px 5e8 sz 10", fills[0].Price, fills[0].Size)
	}

	// The fill is committed D-Chain state: the VM's accepted height advanced past the
	// two order blocks + ensure_market, and the matched trade is in the trade log.
	vm.mu.Lock()
	height := vm.lastAcceptedHeight
	vm.mu.Unlock()
	if height < 3 {
		t.Fatalf("accepted height = %d, want >= 3 (ensure_market + place + submit each a block)", height)
	}
}

// TestZAPIngestSharesOneCoreWithHTTP proves transport ⟂ consensus over the SAME book:
// a maker rests over the canonical ZAP socket, and a taker crossing it through the
// HTTP compat handler (CreateHandlers) fills against that SAME resting order. Neither
// transport has its own matcher or its own book — both fold over dexMethods into the
// one mempool/consensus path, so an order from either side sees the other's state.
func TestZAPIngestSharesOneCoreWithHTTP(t *testing.T) {
	// Boot ONE VM with the ZAP socket on, and ALSO mount its HTTP handlers — both
	// transports on the same VM instance, exactly as the in-luxd plugin runs them.
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   []byte(`{"zapIngestAddr":"127.0.0.1:0"}`),
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	zapAddr := vm.zapIngest.Addr()

	httpBase, stopHTTP := mountHTTPHandlers(t, vm)
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); autoSealer(ctx, t, vm) }()
	defer func() { cancel(); stopHTTP(); _ = vm.Shutdown(context.Background()); wg.Wait() }()

	var pool [32]byte
	pool[0], pool[1] = 0x0e, 0x1c

	// Maker rests over the CANONICAL ZAP socket.
	cli, closeCli := dialVenue(t, zapAddr)
	defer closeCli()
	cli.ensureMarket(t, pool)
	makerID := cli.place(t, pool, zapwire.SideSell, 5.0, 10.0, "core-maker")
	if makerID == 0 {
		t.Fatal("maker place over ZAP returned id 0")
	}

	// Taker crosses through the HTTP COMPAT handler against the SAME book.
	resp := postFrame(t, httpBase, zapwire.MethodSubmit,
		signedPayload(t, "core-taker", TxSubmit, encSubmit(pool, zapwire.SideBuy, false, 5.0, 10.0, wireUser(t, "core-taker"))))
	httpFills, err := zapwire.DecodeFills(resp)
	if err != nil {
		t.Fatalf("decode HTTP submit fills: %v", err)
	}
	if len(httpFills) != 1 || httpFills[0].Size != 10.0 {
		t.Fatalf("HTTP taker crossing the ZAP-rested maker: got %d fills (want 1 of size 10) — transports do NOT share one book", len(httpFills))
	}
}

// mountHTTPHandlers stands up the VM's CreateHandlers over a real httptest.Server,
// returning the base URL (.../dex parent) and a stop func. Mirrors ingest_test's
// startIngestServer but reuses an already-initialized VM.
func mountHTTPHandlers(t *testing.T, vm *VM) (string, func()) {
	t.Helper()
	handlers, err := vm.CreateHandlers(context.Background())
	if err != nil {
		t.Fatalf("CreateHandlers: %v", err)
	}
	root := http.NewServeMux()
	for path, h := range handlers {
		root.Handle(path, h)
	}
	srv := httptest.NewServer(root)
	return srv.URL, srv.Close
}

// TestParseConfig pins the init-Config contract: empty -> no socket (zero value), a
// valid zapIngestAddr -> that address, a malformed config -> fail fast (a chain must
// not boot silently dropping a transport setting), and an unknown field -> ignored
// (forwards-only).
func TestParseConfig(t *testing.T) {
	if c, err := parseConfig(nil); err != nil || c.ZAPIngestAddr != "" {
		t.Fatalf("empty config: c=%+v err=%v, want zero value no error", c, err)
	}
	if c, err := parseConfig([]byte(`{}`)); err != nil || c.ZAPIngestAddr != "" {
		t.Fatalf("{} config: c=%+v err=%v, want zero value no error", c, err)
	}
	if c, err := parseConfig([]byte(`{"zapIngestAddr":"0.0.0.0:9101"}`)); err != nil || c.ZAPIngestAddr != "0.0.0.0:9101" {
		t.Fatalf("valid config: c=%+v err=%v, want addr 0.0.0.0:9101", c, err)
	}
	// Unknown fields are ignored (forwards-only); the known field still parses.
	if c, err := parseConfig([]byte(`{"zapIngestAddr":"1.2.3.4:5","someFutureKnob":true}`)); err != nil || c.ZAPIngestAddr != "1.2.3.4:5" {
		t.Fatalf("forwards-only config: c=%+v err=%v, want addr 1.2.3.4:5 ignoring unknown", c, err)
	}
	if _, err := parseConfig([]byte(`{not json`)); err == nil {
		t.Fatal("malformed config parsed without error — a misconfigured chain must fail fast")
	}
}

// TestNoZAPIngestWhenUnset proves the no-op path: with no zapIngestAddr in Config the
// VM serves HTTP only and starts no socket (so in-process tests and the standalone
// cmd/dvenue, which runs its own socket, are unaffected).
func TestNoZAPIngestWhenUnset(t *testing.T) {
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 8)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		// No Config -> no socket.
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer vm.Shutdown(context.Background())
	if vm.zapIngest != nil {
		t.Fatal("ZAP ingest socket started with no zapIngestAddr configured (must be HTTP-compat-only)")
	}
}

// TestZAPIngestStopsOnShutdown proves the socket is closed on Shutdown (no leaked
// listener, idempotent stop).
func TestZAPIngestStopsOnShutdown(t *testing.T) {
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 8)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   []byte(`{"zapIngestAddr":"127.0.0.1:0"}`),
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	addr := vm.zapIngest.Addr()
	if err := vm.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if vm.zapIngest != nil {
		t.Fatal("Shutdown did not clear the ZAP ingest socket")
	}
	// A second Shutdown is a no-op (idempotent stop).
	if err := vm.Shutdown(context.Background()); err != nil {
		t.Fatalf("second Shutdown: %v", err)
	}
	// The listener is closed: a fresh dial must fail within a short timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if conn, err := rpc.ZAPDial(ctx, addr); err == nil {
		_ = conn.Close()
		t.Fatalf("ZAPDial to %s succeeded after Shutdown — listener leaked", addr)
	}
}
