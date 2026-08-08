// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo
// +build !cgo

// Built under //go:build !cgo: this venue e2e exercises the PURE-GO consensus
// path (rebuildable book, zapdb persistence, ZAP socket, match->Verify->Accept)
// and needs no cgo/GPU matcher — the GPU accelerator is an opt-in venue
// accelerator (cmd/dexd run, under cgo), and the matching CORRECTNESS proven
// here is identical on the CPU and GPU paths. The !cgo gate also keeps this test
// in the package's CGO_ENABLED=0 test binary (the only one that links on a host
// without luxcpp), so `CGO_ENABLED=0 go test ./pkg/dchain` runs the full venue
// e2e green. (Was `dchain && cgo`; the dchain tag gated a removed kludge and the
// cgo tag wrongly excluded a pure-Go proof — decomplected to the real axis.)

package dchain

// localnet_e2e_test.go is the CAPSTONE venue-level localnet proof for the
// standalone D-Chain DEX VM. Unlike handler_test.go (memdb, single market), it
// stands the venue up exactly as cmd/dexd would at runtime:
//
//   - a REAL on-disk zapdb (the authoritative persisted chainstate, not memdb),
//   - the dex_* ZAP surface served over a REAL rpc.Listen socket (the same
//     transport the chains/dexvm proxy + 0x9010 precompile dial),
//   - an auto-sealer goroutine playing the consensus engine (BuildBlock ->
//     Verify -> Accept) — the harness that cmd/dexd's plugin rpc.Serve gets from the
//     node runtime,
//   - the TWO canonical first-pair markets the maker seeds: LUX/LUSD and
//     LUX/LETH (dex-architecture-canonical: maker seeds REAL D-Chain pools via
//     resting limit orders = V4 "liquidity"),
//   - an external ZAP client (rpc.ZAPDial) that SEEDS resting liquidity
//     (dex_place), then crosses it with marketable takers (dex_submit), and
//     verifies the CONSENSUS-COMPUTED fills + the resulting book,
//   - a Shutdown + restart that rebuilds the authoritative book from persisted
//     rows and proves the resting liquidity SURVIVED (it is a real chain, not an
//     in-memory authority).
//
// Every order crosses through the full match -> Verify -> Accept consensus path:
// the fills a caller receives are what every validator re-derived at Verify, not
// a synchronous single-process book mutation. This is the on-chain e2e the
// capstone requires, at the venue (single-operator d-chain) layer.

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database"
	"github.com/luxfi/database/zapdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
	"github.com/luxfi/rpc"
)

// poolIDForSymbol lives in poolid_helper_test.go (untagged): it is shared with
// the always-built read RPC tests and is pure-Go, so it must compile in both the
// cgo and !cgo test binaries.

// venue is a booted standalone D-Chain venue: an on-disk VM + real ZAP socket +
// auto-sealer, plus the dial address. dir is the zapdb path so a restart can
// reopen the same authoritative state.
type venue struct {
	vm     *VM
	db     database.Database
	addr   string
	dir    string
	cancel context.CancelFunc
	wg     *sync.WaitGroup
	server rpc.Server
}

// bootVenue stands up the venue on an on-disk badger/zapdb at dir. If dir
// already holds a chain (restart), Initialize rebuilds the authoritative book
// from the persisted order rows rather than bootstrapping a fresh genesis.
func bootVenue(t *testing.T, dir string) *venue {
	t.Helper()
	db, err := zapdb.New(dir, nil, "dchain", nil)
	if err != nil {
		t.Fatalf("zapdb.New(%s): %v", dir, err)
	}
	vm := &VM{}
	toEngine := make(chan block.Message, 256)
	if err := vm.Initialize(context.Background(), block.Init{
		Runtime:  testRuntime(),
		Genesis:  []byte(testDocument),
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("VM.Initialize: %v", err)
	}

	server, err := rpc.Listen("127.0.0.1:0")
	if err != nil {
		t.Fatalf("rpc.Listen: %v", err)
	}
	if err := vm.RegisterDEX(server); err != nil {
		t.Fatalf("RegisterDEX: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	wg := &sync.WaitGroup{}
	wg.Add(2)
	go func() { defer wg.Done(); _ = server.Serve(ctx) }()
	go func() { defer wg.Done(); autoSealer(ctx, t, vm) }()

	return &venue{vm: vm, db: db, addr: server.Addr(), dir: dir, cancel: cancel, wg: wg, server: server}
}

// stop tears down the serving goroutines, shuts the VM down, and closes the
// on-disk db (the VM does NOT own the injected db — at runtime the node runtime
// owns it; here the harness owns it), releasing the badger directory lock so a
// restart can reopen the same authoritative chainstate.
func (v *venue) stop() {
	v.cancel()
	_ = v.server.Close()
	_ = v.vm.Shutdown(context.Background())
	v.wg.Wait()
	_ = v.db.Close()
}

// e2eClient is an external ZAP client speaking the FROZEN dex_* frames — the
// exact surface the chains/dexvm proxy relay and the 0x9010 precompile use.
type e2eClient struct {
	conn *rpc.ZAPConn
	ctx  context.Context
}

func dialVenue(t *testing.T, addr string) (*e2eClient, func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	conn, err := rpc.ZAPDial(ctx, addr)
	if err != nil {
		cancel()
		t.Fatalf("ZAPDial(%s): %v", addr, err)
	}
	return &e2eClient{conn: conn, ctx: ctx}, func() { _ = conn.Close(); cancel() }
}

func (c *e2eClient) ensureMarket(t *testing.T, pool [32]byte) {
	t.Helper()
	resp, err := c.conn.Call(c.ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	if err != nil {
		t.Fatalf("dex_ensure_market: %v", err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		t.Fatalf("dex_ensure_market not placed: resp=%x", resp)
	}
}

// place rests a limit order (maker liquidity). Returns the consensus-assigned id.
func (c *e2eClient) place(t *testing.T, pool [32]byte, side uint8, price, size float64, user string) uint64 {
	t.Helper()
	body := encPlace(pool, side, price, size, wireUser(t, user))
	resp, err := c.conn.Call(c.ctx, zapwire.MethodPlace, signedPayload(t, user, TxPlace, body))
	if err != nil {
		t.Fatalf("dex_place %s %v@%v: %v", sideName(side), size, price, err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		t.Fatalf("dex_place rejected %s %v@%v: resp=%x", sideName(side), size, price, resp)
	}
	return binary.BigEndian.Uint64(resp[0:8])
}

// submit crosses a marketable order against the book and returns the
// consensus-computed fills (decoded via the proxy's exact frame).
func (c *e2eClient) submit(t *testing.T, pool [32]byte, side uint8, limit, size float64, user string) []zapwire.Fill {
	t.Helper()
	body := encSubmit(pool, side, false, limit, size, wireUser(t, user))
	resp, err := c.conn.Call(c.ctx, zapwire.MethodSubmit, signedPayload(t, user, TxSubmit, body))
	if err != nil {
		t.Fatalf("dex_submit %s %v@%v: %v", sideName(side), size, limit, err)
	}
	fills, err := zapwire.DecodeFills(resp)
	if err != nil {
		t.Fatalf("dex_submit fills decode (BYTE PARITY): %v (resp=%x)", err, resp)
	}
	return fills
}

func sideName(s uint8) string {
	if s == zapwire.SideBuy {
		return "buy"
	}
	return "sell"
}

// restingByMarket sums the remaining resting size on a market's authoritative
// in-RAM book (a fold of the committed rows).
func (v *venue) restingByMarket(pool [32]byte) (orders int, remaining float64) {
	ob := v.vm.books[pool]
	if ob == nil {
		return 0, 0
	}
	for _, o := range ob.Orders {
		orders++
		remaining += o.RemainingSize
	}
	return orders, remaining
}

// TestLocalnetVenueE2E is the capstone venue bring-up + on-chain e2e:
//  1. boot the venue on an on-disk zapdb (real persisted chainstate),
//  2. seed LUX/LUSD and LUX/LETH with resting maker liquidity (dex_place),
//  3. cross both books with marketable takers (dex_submit),
//  4. assert the consensus-computed fills are correct + books updated,
//  5. restart the venue and prove the resting liquidity SURVIVED (real chain).
func TestLocalnetVenueE2E(t *testing.T) {
	dir := t.TempDir()

	v := bootVenue(t, dir)
	t.Logf("venue d-chain up: ZAP endpoint=%s zapdb=%s", v.addr, dir)

	cli, closeCli := dialVenue(t, v.addr)

	lusd := poolIDForSymbol("LUX/LUSD")
	leth := poolIDForSymbol("LUX/LETH")

	// --- maker seeding: ensure markets, then rest two-sided liquidity. This is
	// what `maker -dex <endpoint>` does for the first pairs, in dex_* form. ---
	cli.ensureMarket(t, lusd)
	cli.ensureMarket(t, leth)
	t.Logf("markets ensured: LUX/LUSD, LUX/LETH")

	// LUX/LUSD book: asks 10@100, 5@101 ; bids 8@99, 12@98.
	cli.place(t, lusd, zapwire.SideSell, 100.0, 10.0, "maker")
	cli.place(t, lusd, zapwire.SideSell, 101.0, 5.0, "maker")
	cli.place(t, lusd, zapwire.SideBuy, 99.0, 8.0, "maker")
	cli.place(t, lusd, zapwire.SideBuy, 98.0, 12.0, "maker")

	// LUX/LETH book: asks 4@0.05, 6@0.051 ; bids 3@0.049.
	cli.place(t, leth, zapwire.SideSell, 0.050, 4.0, "maker")
	cli.place(t, leth, zapwire.SideSell, 0.051, 6.0, "maker")
	cli.place(t, leth, zapwire.SideBuy, 0.049, 3.0, "maker")

	if n, rem := v.restingByMarket(lusd); n != 4 || rem != 35.0 {
		t.Fatalf("LUX/LUSD seeded book: orders=%d remaining=%v, want 4 / 35.0", n, rem)
	}
	if n, rem := v.restingByMarket(leth); n != 3 || rem != 13.0 {
		t.Fatalf("LUX/LETH seeded book: orders=%d remaining=%v, want 3 / 13.0", n, rem)
	}
	t.Logf("seeded LUX/LUSD: 4 resting orders / 35.0 total; LUX/LETH: 3 resting / 13.0 total")

	// --- on-chain cross #1: a 12-unit buy @101 on LUX/LUSD sweeps 10@100 +
	// 2@101 (two fills, multi-level), VWAP between 100 and 101. ---
	f1 := cli.submit(t, lusd, zapwire.SideBuy, 101.0, 12.0, "taker")
	if len(f1) != 2 {
		t.Fatalf("LUX/LUSD cross: got %d fills, want 2 (multi-level sweep)", len(f1))
	}
	var got1 float64
	for i, f := range f1 {
		if f.TakerSide != zapwire.SideBuy {
			t.Errorf("fill %d takerSide=%d want buy", i, f.TakerSide)
		}
		got1 += float64(f.Size) // exact base units
		t.Logf("  LUX/LUSD fill %d: %d @ %d/1e8 (takerSide=%s)", i, f.Size, f.Price, sideName(f.TakerSide))
	}
	if got1 != 12.0 {
		t.Fatalf("LUX/LUSD filled %v, want 12.0", got1)
	}
	if f1[0].Price != 100*uint64(zapwire.PriceScale) || f1[1].Price != 101*uint64(zapwire.PriceScale) {
		t.Fatalf("LUX/LUSD fill prices = %d,%d want 100e8,101e8 (price-time priority)", f1[0].Price, f1[1].Price)
	}
	// After: the 5@101 ask is fully consumed by the 2-unit take? No: 10@100 fully
	// + 2 of 5@101 -> 3@101 remains. Asks remaining = 3; bids untouched = 20.
	if n, rem := v.restingByMarket(lusd); rem != 23.0 {
		t.Fatalf("LUX/LUSD post-cross resting=%v (orders=%d), want 23.0 (3@101 ask + 8@99 + 12@98 bids)", rem, n)
	}
	t.Logf("LUX/LUSD post-cross: 23.0 resting (3@101 ask + 20 bid) — consensus-settled")

	// --- on-chain cross #2: a 5-unit sell @0.049 on LUX/LETH hits the 3@0.049
	// bid (one fill of 3), residual 2 does NOT rest (dex_submit is IOC-style:
	// marketable, unfilled remainder dropped). ---
	f2 := cli.submit(t, leth, zapwire.SideSell, 0.049, 5.0, "taker")
	if len(f2) != 1 {
		t.Fatalf("LUX/LETH cross: got %d fills, want 1", len(f2))
	}
	if f2[0].Size != 3 || f2[0].Price != wirePriceUnits(0.049) || f2[0].TakerSide != zapwire.SideSell {
		t.Fatalf("LUX/LETH fill = %d@%d side=%d, want 3@%d sell", f2[0].Size, f2[0].Price, f2[0].TakerSide, wirePriceUnits(0.049))
	}
	t.Logf("  LUX/LETH fill: %d @ %d/1e8 (takerSide=sell)", f2[0].Size, f2[0].Price)
	// LUX/LETH bids now empty; asks (4@0.05 + 6@0.051 = 10) untouched.
	if n, rem := v.restingByMarket(leth); rem != 10.0 {
		t.Fatalf("LUX/LETH post-cross resting=%v (orders=%d), want 10.0 (asks only)", rem, n)
	}
	t.Logf("LUX/LETH post-cross: 10.0 resting (asks only, bid fully consumed) — consensus-settled")

	lastHeightBefore := v.vm.lastAcceptedHeight
	closeCli()
	v.stop()
	t.Logf("venue stopped at height=%d; restarting from the same zapdb...", lastHeightBefore)

	// --- restart: reopen the SAME on-disk zapdb. The authoritative book is
	// rebuilt from persisted rows; resting liquidity must be byte-identical. ---
	v2 := bootVenue(t, dir)
	defer v2.stop()

	if h := v2.vm.lastAcceptedHeight; h != lastHeightBefore {
		t.Fatalf("restart height=%d, want %d (persisted chainstate must survive)", h, lastHeightBefore)
	}
	if n, rem := v2.restingByMarket(lusd); rem != 23.0 {
		t.Fatalf("LUX/LUSD after restart resting=%v (orders=%d), want 23.0 (must survive restart)", rem, n)
	}
	if n, rem := v2.restingByMarket(leth); rem != 10.0 {
		t.Fatalf("LUX/LETH after restart resting=%v (orders=%d), want 10.0 (must survive restart)", rem, n)
	}
	t.Logf("RESTART OK: height=%d preserved; LUX/LUSD=23.0, LUX/LETH=10.0 resting rebuilt from persisted rows", v2.vm.lastAcceptedHeight)

	// --- prove the venue still matches after restart: cross the surviving
	// LUX/LUSD 3@101 ask with a 3@101 buy. One fill, book empties to bids only. ---
	cli2, closeCli2 := dialVenue(t, v2.addr)
	defer closeCli2()
	f3 := cli2.submit(t, lusd, zapwire.SideBuy, 101.0, 3.0, "taker2")
	if len(f3) != 1 || f3[0].Size != 3 || f3[0].Price != 101*uint64(zapwire.PriceScale) {
		t.Fatalf("post-restart LUX/LUSD cross = %+v, want one 3@101e8 fill", f3)
	}
	if _, rem := v2.restingByMarket(lusd); rem != 20.0 {
		t.Fatalf("post-restart LUX/LUSD resting=%v, want 20.0 (bids only, ask fully consumed)", rem)
	}
	t.Logf("post-restart cross OK: LUX/LUSD ask consumed, 20.0 bid liquidity remains")

	fmt.Printf("\n=== VENUE LOCALNET E2E: PASS ===\n"+
		"  endpoint=%s\n"+
		"  LUX/LUSD + LUX/LETH seeded, crossed on-chain (match->Verify->Accept), restarted, re-crossed\n"+
		"  final: LUX/LUSD 20.0 bid liquidity; height=%d (persisted)\n",
		v2.addr, v2.vm.lastAcceptedHeight)
}
