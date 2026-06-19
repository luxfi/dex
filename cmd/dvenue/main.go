// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dvenue runs the standalone D-Chain DEX VENUE as a long-lived ZAP
// server on a fixed TCP address — the dial-able endpoint the 0x9010 EVM
// precompile (NewZAPEngine(dex-zap-endpoint)) and the chains/dexvm atomic proxy
// relay connect to. It is the production form of the capstone venue proof
// (pkg/dchain/localnet_e2e_test.go), promoted from a test harness to a daemon:
//
//   - opens the authoritative on-disk badger/zapdb (real persisted chainstate),
//   - serves the FROZEN clob_* surface (RegisterCLOB) over rpc.Listen,
//   - runs the consensus sealer loop (WaitForEvent -> BuildBlock -> Verify ->
//     Accept), so every write crosses the full match -> Verify -> Accept path
//     and the bytes a caller gets back are consensus-computed fills,
//   - exposes read-only clob_depth + clob_balance observation methods for
//     settled book and custody state (reads need no consensus round-trip).
//
// This is the ONE venue binary. CGO_ENABLED is the single axis that picks the
// matcher: pkg/lx links the GPU matcher under cgo (amm_gpu_cuda / amm_gpu_metal
// / orderbook_cuda) and the pure-Go CPU matcher without it (amm_nogpu /
// orderbook_cuda_stub). The main is matcher-agnostic — it drives dchain.VM
// either way; engineName (engine_{cgo,nocgo}.go) only labels the active build.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"math"
	"os"
	"os/signal"
	"syscall"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database"
	"github.com/luxfi/database/zapdb"
	"github.com/luxfi/dex/pkg/dchain"
	"github.com/luxfi/dex/pkg/zapwire"
	log "github.com/luxfi/log"
	"github.com/luxfi/rpc"
)

// clobDepthMethod is the read-only book-observation method this daemon adds on
// top of the four frozen clob_* write methods. Request: poolId[32]. Response:
// orders[4] | remaining[f64:8] | bestBid[f64:8] | bestAsk[f64:8] | found[1].
const clobDepthMethod = "clob_depth"

// clobBalanceMethod is the read-only custody-balance observation method. It lets
// the bring-up harness watch funds DEPOSIT into the book, move on a fill, and
// WITHDRAW out — the "money lives in the order book" proof. Request: user[16] +
// asset[8]. Response: available[8] | locked[8] (big-endian uint64 atomic units).
const clobBalanceMethod = "clob_balance"

func main() {
	addr := flag.String("addr", "127.0.0.1:9099", "ZAP listen address for the venue")
	dir := flag.String("db", "/tmp/dchain_venue_db", "on-disk zapdb directory (authoritative chainstate)")
	flag.Parse()

	logger := log.Root()

	db, err := zapdb.New(*dir, nil, "dchain", nil)
	if err != nil {
		logger.Error("zapdb.New", "dir", *dir, "err", err)
		os.Exit(1)
	}

	vm := &dchain.VM{}
	toEngine := make(chan block.Message, 256)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      logger,
		ToEngine: toEngine,
	}); err != nil {
		logger.Error("VM.Initialize", "err", err)
		_ = db.Close()
		os.Exit(1)
	}

	server, err := rpc.Listen(*addr)
	if err != nil {
		logger.Error("rpc.Listen", "addr", *addr, "err", err)
		_ = vm.Shutdown(context.Background())
		_ = db.Close()
		os.Exit(1)
	}
	if err := vm.RegisterCLOB(server); err != nil {
		logger.Error("RegisterCLOB", "err", err)
		os.Exit(1)
	}
	if err := server.RegisterRaw(clobDepthMethod, depthHandler(vm)); err != nil {
		logger.Error("RegisterRaw clob_depth", "err", err)
		os.Exit(1)
	}
	if err := server.RegisterRaw(clobBalanceMethod, balanceHandler(vm)); err != nil {
		logger.Error("RegisterRaw clob_balance", "err", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go sealer(ctx, vm, logger)
	go func() {
		if serr := server.Serve(ctx); serr != nil && ctx.Err() == nil {
			logger.Error("server.Serve", "err", serr)
		}
	}()

	logger.Info("D-Chain venue serving",
		"addr", server.Addr(), "db", *dir, "version", dchain.Version, "engine", engineName)
	// Emit a machine-greppable readiness line for the bring-up harness.
	os.Stdout.WriteString("DVENUE_READY addr=" + server.Addr() + " db=" + *dir + " engine=" + engineName + "\n")

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	<-sigc

	logger.Info("D-Chain venue shutting down")
	cancel()
	_ = server.Close()
	_ = vm.Shutdown(context.Background())
	_ = db.Close()
}

// sealer is the production consensus loop: on each PendingTxs signal it builds a
// block, verifies it (re-derives fills against a versiondb overlay), and accepts
// it (atomically commits to zapdb). This is the engine the node runtime would
// drive for the in-process chain; the standalone venue runs it itself.
func sealer(ctx context.Context, vm *dchain.VM, logger log.Logger) {
	for {
		msg, err := vm.WaitForEvent(ctx)
		if err != nil {
			return // ctx cancelled
		}
		if msg.Type != block.PendingTxs {
			continue
		}
		// Drain every pending tx the signal may have batched.
		for {
			blk, berr := vm.BuildBlock(ctx)
			if berr != nil {
				break // empty mempool
			}
			if verr := blk.Verify(ctx); verr != nil {
				logger.Error("sealer Verify", "err", verr)
				break
			}
			if aerr := blk.Accept(ctx); aerr != nil {
				logger.Error("sealer Accept", "err", aerr)
				break
			}
		}
	}
}

// depthHandler serves clob_depth: a read-only observation of a market's settled
// resting book. It reads the in-RAM book via the VM's BookDepth accessor (a fold
// of committed rows) — no consensus round-trip.
func depthHandler(vm *dchain.VM) rpc.RawHandler {
	return func(_ context.Context, payload []byte) ([]byte, error) {
		var pool [32]byte
		copy(pool[:], payload[:min(len(payload), zapwire.PoolIDSize)])
		orders, remaining, bestBid, bestAsk, ok := vm.BookDepth(pool)
		out := make([]byte, 4+8+8+8+1)
		binary.BigEndian.PutUint32(out[0:4], uint32(orders))
		binary.BigEndian.PutUint64(out[4:12], math.Float64bits(remaining))
		binary.BigEndian.PutUint64(out[12:20], math.Float64bits(bestBid))
		binary.BigEndian.PutUint64(out[20:28], math.Float64bits(bestAsk))
		if ok {
			out[28] = 1
		}
		return out, nil
	}
}

// balanceHandler serves clob_balance: a read-only observation of an account's
// custody balances for an asset (available + locked), read from the durable
// ledger via the VM's Balance accessor. Request: user[16] + asset[32]. Response:
// available[8] | locked[8].
func balanceHandler(vm *dchain.VM) rpc.RawHandler {
	return func(_ context.Context, payload []byte) ([]byte, error) {
		out := make([]byte, 16)
		if len(payload) < zapwire.UserSize+zapwire.AssetIDSize {
			return out, nil
		}
		user := string(trimNullBytes(payload[0:zapwire.UserSize]))
		var asset [32]byte
		copy(asset[:], payload[zapwire.UserSize:zapwire.UserSize+zapwire.AssetIDSize])
		available, locked, err := vm.Balance(user, asset)
		if err != nil {
			return out, err
		}
		binary.BigEndian.PutUint64(out[0:8], available)
		binary.BigEndian.PutUint64(out[8:16], locked)
		return out, nil
	}
}

// trimNullBytes removes trailing NUL padding from a fixed-width user field.
func trimNullBytes(b []byte) []byte {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] != 0 {
			return b[:i+1]
		}
	}
	return b[:0]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ensure the import is used even if database alias drift occurs.
var _ database.Database
