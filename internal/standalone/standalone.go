// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package standalone runs the D-Chain DEX VENUE as a long-lived ZAP server on a
// fixed TCP address — the dial-able endpoint the 0x9010 EVM precompile
// (NewZAPEngine(dex-zap-endpoint)) and the chains/dexvm atomic proxy relay
// connect to. It:
//
//   - opens the authoritative on-disk badger/zapdb (real persisted chainstate),
//   - serves the FROZEN dex_* surface (RegisterDEX) over rpc.Listen,
//   - runs the consensus sealer loop (WaitForEvent -> BuildBlock -> Verify ->
//     Accept), so every write crosses the full match -> Verify -> Accept path
//     and the bytes a caller gets back are consensus-computed fills,
//   - exposes read-only dex_depth + dex_balance observation methods for the
//     settled book and custody state (reads need no consensus round-trip),
//   - optionally (-http) serves the SAME surface luxd mounts for the in-process
//     plugin under /v1/dex — the JSON dex_get_* reads + the dex_* writes
//     (vm.CreateHandlers, ingest.go) — so the exchange-api/maker reach a
//     standalone single-operator venue byte-identically to the in-luxd plugin.
//
// CGO_ENABLED is the single axis that picks the matcher: pkg/dex links the GPU
// matcher under cgo (amm_gpu_cuda / amm_gpu_metal / orderbook_gpu_cuda) and the
// pure-Go CPU matcher without it (amm_nogpu / orderbook_nogpu). This package
// is matcher-agnostic — it drives dchain.VM either way; engineName
// (engine_{cgo,nocgo}.go) only labels the active build.
package standalone

import (
	"context"
	"encoding/binary"
	"flag"
	"math"
	"net/http"
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

// dexDepthMethod is the read-only book-observation method this daemon adds on
// top of the four frozen dex_* write methods. Request: poolId[32]. Response:
// orders[4] | remaining[f64:8] | bestBid[f64:8] | bestAsk[f64:8] | found[1].
const dexDepthMethod = "dex_depth"

// dexBalanceMethod is the read-only custody-balance observation method. It lets
// the bring-up harness watch funds DEPOSIT into the book, move on a fill, and
// WITHDRAW out — the "money lives in the order book" proof. Request: user[16] +
// asset[8]. Response: available[8] | locked[8] (big-endian uint64 atomic units).
const dexBalanceMethod = "dex_balance"

// httpChainPrefix is the chain route prefix the optional -http surface mounts
// under. It mirrors the EXACT external path luxd gives the in-process plugin
// (/v1/chain/<chainID-or-alias>), using the canonical "D" alias the exchange-api
// and maker dial (DEX_DCHAIN_URL / DEX_HTTP_URL = http://<host>/v1/dex). So a
// standalone single-operator venue is a byte-identical drop-in for the in-luxd
// D-Chain read+write surface — no luxd required to serve the markets pipeline.
// httpChainPrefix is what the venue mounts its chain surface under. The keys
// httpHandlers returns already carry the chain's own namespace ("/dex/<method>"
// — the same segment a node serves under /v1/chain/<D>/), so the prefix supplies
// only the version. Spelling "dex" here too produced /v1/dex/dex/<method>: a
// path no client written against a node would try, and not the one this flag
// promises.
const httpChainPrefix = "/v1"

// Run serves the standalone venue. args is the post-subcommand argv
// (os.Args[2:] from `dexd run`/`dexd standalone`); it carries -addr and -db.
func Run(args []string) {
	fs := flag.NewFlagSet("dexd run", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:9099", "ZAP listen address for the venue")
	dir := fs.String("db", "/tmp/dchain_venue_db", "on-disk zapdb directory (authoritative chainstate)")
	httpAddr := fs.String("http", "", "HTTP listen address for the chain read+write surface mounted under /v1/dex (e.g. 0.0.0.0:9650); empty = ZAP only")
	genesisPath := fs.String("genesis", "", "path to this chain's creation document (the dchain.json a P-Chain CreateChainTx records). Required to found a chain: the VM refuses to invent one, because a genesis the binary derived is one no other node would derive.")
	_ = fs.Parse(args)

	logger := log.Root()

	// The creation document is delivered, never derived. Under luxd it arrives
	// as InitializeRequest.GenesisBytes; standalone has no P-Chain to read it
	// from, so it is named on the command line. Without this the venue could
	// not start at all — Initialize was called with no document and the VM
	// refused, correctly, every time.
	//
	// The bytes are opaque here: what matters is that an operator chose them,
	// so two nodes told to be the same chain are, and one told nothing is not
	// quietly founded on whatever this binary happened to be built from.
	var document []byte
	if *genesisPath != "" {
		b, err := os.ReadFile(*genesisPath)
		if err != nil {
			logger.Error("reading the creation document", "path", *genesisPath, "err", err)
			os.Exit(1)
		}
		document = b
	}

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
		Genesis:  document,
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
	if err := vm.RegisterDEX(server); err != nil {
		logger.Error("RegisterDEX", "err", err)
		os.Exit(1)
	}
	if err := server.RegisterRaw(dexDepthMethod, depthHandler(vm)); err != nil {
		logger.Error("RegisterRaw dex_depth", "err", err)
		os.Exit(1)
	}
	if err := server.RegisterRaw(dexBalanceMethod, balanceHandler(vm)); err != nil {
		logger.Error("RegisterRaw dex_balance", "err", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go sealer(ctx, vm, logger)
	go func() {
		if serr := server.Serve(ctx); serr != nil && ctx.Err() == nil {
			logger.Error("server.Serve", "err", serr)
		}
	}()

	// Optional HTTP surface: the SAME handler table luxd mounts for the in-process
	// plugin (vm.CreateHandlers -> ingest.go httpHandlers: dex_* writes + dex_get_*
	// JSON reads), served under /v1/dex so the exchange-api (reads) and maker
	// (writes via DEX_HTTP_URL) reach this single-operator venue byte-identically
	// to the in-luxd D-Chain. Writes still cross the full sealer consensus path.
	var httpSrv *http.Server
	if *httpAddr != "" {
		handlers, herr := vm.CreateHandlers(ctx)
		if herr != nil {
			logger.Error("CreateHandlers", "err", herr)
			cancel()
			_ = server.Close()
			_ = vm.Shutdown(context.Background())
			_ = db.Close()
			os.Exit(1)
		}
		mux := http.NewServeMux()
		for key, h := range handlers {
			mux.Handle(httpChainPrefix+key, h)
		}
		httpSrv = &http.Server{Addr: *httpAddr, Handler: mux}
		go func() {
			if serr := httpSrv.ListenAndServe(); serr != nil && serr != http.ErrServerClosed {
				logger.Error("httpSrv.ListenAndServe", "err", serr)
			}
		}()
		logger.Info("D-Chain venue HTTP surface serving", "addr", *httpAddr, "prefix", httpChainPrefix, "routes", len(handlers))
	}

	logger.Info("D-Chain venue serving",
		"addr", server.Addr(), "db", *dir, "version", dchain.Version, "engine", engineName)
	// Emit a machine-greppable readiness line for the bring-up harness.
	os.Stdout.WriteString("DVENUE_READY addr=" + server.Addr() + " db=" + *dir + " engine=" + engineName + " http=" + *httpAddr + "\n")

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	<-sigc

	logger.Info("D-Chain venue shutting down")
	cancel()
	if httpSrv != nil {
		_ = httpSrv.Shutdown(context.Background())
	}
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

// depthHandler serves dex_depth: a read-only observation of a market's settled
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

// balanceHandler serves dex_balance: a read-only observation of an account's
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
