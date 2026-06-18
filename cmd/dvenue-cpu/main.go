// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build dchain
// +build dchain

// Command dvenue-cpu runs the standalone D-Chain DEX VENUE as a long-lived ZAP
// server — the dial-able endpoint the 0x9010 EVM precompile
// (NewZAPEngine(dex-zap-endpoint)) and the chains/dexvm atomic proxy relay
// connect to. It is byte-identical in behavior to cmd/dvenue but builds PURE-GO
// (CGO_ENABLED=0): it drives the pure-Go CPU matcher via dchain.VM, with NO
// cgo/luxcpp/gpu-kernels dependency.
//
// This is the FREE, open base venue: any operator can run a correct CPU D-Chain
// from this single static binary. GPU acceleration is the licensed opt-in
// (cmd/dvenue, //go:build dchain && cgo, links the private CUDA matcher) — same
// surface, same consensus, just the HFT-grade accelerator.
//
//   - opens the authoritative on-disk badger/zapdb (real persisted chainstate),
//   - serves the FROZEN clob_* surface (RegisterCLOB) over rpc.Listen,
//   - runs the consensus sealer loop (WaitForEvent -> BuildBlock -> Verify ->
//     Accept), so every write crosses the full match -> Verify -> Accept path
//     and the bytes a caller gets back are consensus-computed fills,
//   - exposes read-only clob_depth + clob_balance observation methods.
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

const clobDepthMethod = "clob_depth"
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

	logger.Info("D-Chain venue (CPU) serving",
		"addr", server.Addr(), "db", *dir, "version", dchain.Version)
	os.Stdout.WriteString("DVENUE_READY addr=" + server.Addr() + " db=" + *dir + " engine=cpu\n")

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	<-sigc

	logger.Info("D-Chain venue shutting down")
	cancel()
	_ = server.Close()
	_ = vm.Shutdown(context.Background())
	_ = db.Close()
}

func sealer(ctx context.Context, vm *dchain.VM, logger log.Logger) {
	for {
		msg, err := vm.WaitForEvent(ctx)
		if err != nil {
			return
		}
		if msg.Type != block.PendingTxs {
			continue
		}
		for {
			blk, berr := vm.BuildBlock(ctx)
			if berr != nil {
				break
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

func balanceHandler(vm *dchain.VM) rpc.RawHandler {
	return func(_ context.Context, payload []byte) ([]byte, error) {
		out := make([]byte, 16)
		if len(payload) < zapwire.UserSize+zapwire.AssetIDSize {
			return out, nil
		}
		user := string(trimNullBytes(payload[0:zapwire.UserSize]))
		asset := binary.BigEndian.Uint64(payload[zapwire.UserSize : zapwire.UserSize+zapwire.AssetIDSize])
		available, locked, err := vm.Balance(user, asset)
		if err != nil {
			return out, err
		}
		binary.BigEndian.PutUint64(out[0:8], available)
		binary.BigEndian.PutUint64(out[8:16], locked)
		return out, nil
	}
}

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

var _ database.Database
