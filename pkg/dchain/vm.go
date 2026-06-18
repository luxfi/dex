// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database"
	"github.com/luxfi/database/versiondb"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/version"
)

// vm.go is the standalone D-Chain DEX VM: an implementation of
// block.ChainVM (== github.com/luxfi/vm/chain.ChainVM) served as an OS process by
// github.com/luxfi/vm/rpc.Serve — the same ZAP plugin harness lux/evm boots
// through. rpc.Serve opens the chain's badgerdb/zapdb at ChainDataDir/db and
// hands it to Initialize as init.DB; the VM treats that as its sole authoritative
// store.
//
// The VM owns: the durable state (init.DB), the in-RAM book cache (rebuilt from
// order:* rows on Initialize), the mempool, and the accepted-block index. It is
// deterministic end to end: BuildBlock drains the mempool in sequence order,
// Block.Verify runs the matcher against a versiondb overlay, Block.Accept commits
// the overlay atomically.

// Version is the VM's semantic version string.
const Version = "dchain/v1.0.0"

// genesisParent is the synthetic parent id of the genesis block (all zero).
var genesisParent = ids.Empty

// VM implements block.ChainVM for the d-chain DEX.
type VM struct {
	// mu guards all mutable VM state below. Block lifecycle methods
	// (Verify/Accept/Reject) and BuildBlock take it; the matcher's own ob.mu is
	// nested under it.
	mu sync.Mutex

	log log.Logger
	db  database.Database

	mempool *mempool

	// outcomes lets a ZAP handler block until its tx is decided by an accepted
	// block (Accept resolves the waiter with the consensus fills/ack).
	outcomes *outcomeRegistry

	// In-RAM accelerator: poolId -> rebuildable book. Truth is the order:* rows.
	books map[[32]byte]*lx.OrderBook

	// Accepted-block index + linear-chain head.
	acceptedBlocks    map[ids.ID]*Block
	heightIndex       map[uint64]ids.ID
	lastAcceptedID    ids.ID
	lastAcceptedHeight uint64
	lastRoot          [Size]byte

	// preferred is the block the engine asked us to build on next.
	preferred ids.ID

	// toEngine is retained for the mempool signal path.
	toEngine chan<- block.Message

	initialized bool
}

// Initialize wires the VM to its runtime, durable store, and engine channel,
// then rebuilds in-RAM state from the durable rows. On a fresh chain it commits
// the genesis block; on restart it loads the last-accepted head and folds every
// market's book from its order:* rows (the rebuildable-accelerator path).
func (vm *VM) Initialize(ctx context.Context, init block.Init) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	if init.DB == nil {
		return fmt.Errorf("dchain: Initialize requires a non-nil DB")
	}
	vm.log = init.Log
	if vm.log == nil {
		vm.log = log.NewNoOpLogger()
	}
	vm.db = init.DB
	vm.toEngine = init.ToEngine
	vm.mempool = newMempool(init.ToEngine)
	vm.outcomes = newOutcomeRegistry()
	vm.books = map[[32]byte]*lx.OrderBook{}
	vm.acceptedBlocks = map[ids.ID]*Block{}
	vm.heightIndex = map[uint64]ids.ID{}

	// Load the head, or bootstrap genesis.
	if _, err := readLastAccepted(vm.db); err == database.ErrNotFound {
		if err := vm.bootstrapGenesis(); err != nil {
			return err
		}
	} else if err != nil {
		return fmt.Errorf("dchain: read last accepted: %w", err)
	} else {
		if err := vm.loadHead(); err != nil {
			return err
		}
		if err := vm.rebuildAllBooks(); err != nil {
			return err
		}
	}

	vm.preferred = vm.lastAcceptedID
	vm.initialized = true
	vm.log.Info("dchain VM initialized",
		"height", vm.lastAcceptedHeight,
		"lastAccepted", vm.lastAcceptedID,
		"markets", len(vm.books),
	)
	return nil
}

// bootstrapGenesis writes the genesis block (height 0, empty, zero parent root)
// and sets it as the accepted head. Must be called under vm.mu on a fresh DB.
func (vm *VM) bootstrapGenesis() error {
	var parentRoot [Size]byte // zero
	root, _, _, _ := ExecutionRoot(parentRoot, nil, nil, nil, 0)
	gen := newBlock(vm, genesisParent, 0, time.Unix(0, 0).UTC(), root, nil)

	batch := vm.db.NewBatch()
	if err := writeLastAccepted(batch, gen.id); err != nil {
		return err
	}
	if err := writeHeight(batch, 0); err != nil {
		return err
	}
	if err := writeRoot(batch, root); err != nil {
		return err
	}
	if err := batch.Write(); err != nil {
		return fmt.Errorf("dchain: write genesis: %w", err)
	}

	gen.status = statusAccepted
	vm.lastAcceptedID = gen.id
	vm.lastAcceptedHeight = 0
	vm.lastRoot = root
	vm.acceptedBlocks[gen.id] = gen
	vm.heightIndex[0] = gen.id
	return nil
}

// loadHead reads the accepted head (id, height, root) from durable meta. Must be
// called under vm.mu.
func (vm *VM) loadHead() error {
	id, err := readLastAccepted(vm.db)
	if err != nil {
		return fmt.Errorf("dchain: load head id: %w", err)
	}
	height, err := readHeight(vm.db)
	if err != nil {
		return fmt.Errorf("dchain: load head height: %w", err)
	}
	root, err := readRoot(vm.db)
	if err != nil {
		return fmt.Errorf("dchain: load head root: %w", err)
	}
	vm.lastAcceptedID = id
	vm.lastAcceptedHeight = height
	vm.lastRoot = root
	vm.heightIndex[height] = id
	return nil
}

// rebuildAllBooks folds every market's book from its order:* rows. This is the
// restart path: after a crash the in-RAM books are reconstructed exactly from the
// durable resting set, so trading resumes with the committed state. Must be
// called under vm.mu.
func (vm *VM) rebuildAllBooks() error {
	it := vm.db.NewIteratorWithPrefix([]byte(prefixMarket))
	defer it.Release()
	for it.Next() {
		key := it.Key()
		if len(key) != len(prefixMarket)+32 {
			continue
		}
		var poolID [32]byte
		copy(poolID[:], key[len(prefixMarket):])
		symbol := string(it.Value())
		ob, err := rebuildBookFromDB(vm.db, poolID, symbol)
		if err != nil {
			return fmt.Errorf("dchain: rebuild book %x: %w", poolID[:8], err)
		}
		vm.books[poolID] = ob
	}
	return it.Error()
}

// BuildBlock drains the mempool into a new block on top of the preferred head,
// executes it locally to compute the execution root (the proposer's claim), and
// returns it. The block is NOT accepted here — consensus votes, then calls
// Verify/Accept. Returns an error when the mempool is empty (the engine only
// calls BuildBlock after a PendingTxs signal, but a race can leave it empty).
func (vm *VM) BuildBlock(ctx context.Context) (block.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	txs := vm.mempool.Drain(0)
	if len(txs) == 0 {
		return nil, ErrEmptyMempool
	}

	height := vm.lastAcceptedHeight + 1
	ts := vm.blockTimestamp()

	// Execute against a throwaway overlay to derive the root the proposer claims.
	// (Accept re-executes against a fresh overlay; this overlay is discarded.)
	probe := newBlock(vm, vm.lastAcceptedID, height, ts, [Size]byte{}, txs)
	overlay := versiondb.New(vm.db)
	res, err := probe.execute(ctx, overlay)
	overlay.Abort()
	if err != nil {
		// A tx that cannot execute (e.g. unknown market) must not wedge the
		// chain: drop the batch back to the mempool and surface the error.
		vm.mempool.Requeue(txs)
		return nil, fmt.Errorf("dchain: build execute: %w", err)
	}

	blk := newBlock(vm, vm.lastAcceptedID, height, ts, res.root, txs)
	vm.log.Debug("dchain built block", "height", height, "txs", len(txs), "fills", len(res.fills))
	return blk, nil
}

// blockTimestamp returns a deterministic, monotone block timestamp: the max of
// "now" and (last accepted ts + 1ns). The block's own timestamp is what feeds
// applyTx, so it is recorded in the block bytes and replayed identically by every
// validator (they read it from the block, never re-sample the clock). Must be
// called under vm.mu.
func (vm *VM) blockTimestamp() time.Time {
	now := time.Now().UTC()
	if last, ok := vm.acceptedBlocks[vm.lastAcceptedID]; ok {
		min := last.timestamp.Add(time.Nanosecond)
		if now.Before(min) {
			return min
		}
	}
	return now
}

// ParseBlock decodes block bytes into a Block. Used by the engine to reconstruct
// a peer's proposed block; Verify then re-executes it.
func (vm *VM) ParseBlock(ctx context.Context, b []byte) (block.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	// Return the cached accepted block if we already have it (stable identity).
	blk, err := parseBlock(vm, b)
	if err != nil {
		return nil, err
	}
	if existing, ok := vm.acceptedBlocks[blk.id]; ok {
		return existing, nil
	}
	return blk, nil
}

// GetBlock returns an accepted block by id, or database.ErrNotFound.
func (vm *VM) GetBlock(ctx context.Context, id ids.ID) (block.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	if blk, ok := vm.acceptedBlocks[id]; ok {
		return blk, nil
	}
	return nil, database.ErrNotFound
}

// LastAccepted returns the id of the last accepted block.
func (vm *VM) LastAccepted(ctx context.Context) (ids.ID, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	return vm.lastAcceptedID, nil
}

// SetPreference records the block the engine wants the VM to build on. The
// d-chain is linear; the preference always tracks the last accepted block.
func (vm *VM) SetPreference(ctx context.Context, id ids.ID) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	vm.preferred = id
	return nil
}

// GetBlockIDAtHeight returns the accepted block id at a height (height indexer).
func (vm *VM) GetBlockIDAtHeight(ctx context.Context, height uint64) (ids.ID, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	if id, ok := vm.heightIndex[height]; ok {
		return id, nil
	}
	return ids.Empty, database.ErrNotFound
}

// SetState transitions the VM lifecycle state (bootstrapping/normal-op/etc).
// The d-chain has no state-specific behavior beyond logging — it is ready to
// build/verify as soon as Initialize completes.
func (vm *VM) SetState(ctx context.Context, state uint32) error {
	vm.log.Debug("dchain SetState", "state", state)
	return nil
}

// Version returns the VM version string.
func (vm *VM) Version(ctx context.Context) (string, error) { return Version, nil }

// NewHTTPHandler returns the VM's HTTP handler. The d-chain serves its CLOB API
// over the ZAP gateway (handler.go), not HTTP, so this is nil.
func (vm *VM) NewHTTPHandler(ctx context.Context) (http.Handler, error) { return nil, nil }

// CreateHandlers returns named HTTP handlers. None — the CLOB surface is ZAP.
func (vm *VM) CreateHandlers(ctx context.Context) (map[string]http.Handler, error) {
	return map[string]http.Handler{}, nil
}

// Connected/Disconnected are p2p lifecycle callbacks; the d-chain matcher needs
// no per-peer state, so these are no-ops.
func (vm *VM) Connected(ctx context.Context, nodeID ids.NodeID, app *version.Application) error {
	return nil
}
func (vm *VM) Disconnected(ctx context.Context, nodeID ids.NodeID) error { return nil }

// HealthCheck reports the VM healthy once initialized, with the current height.
func (vm *VM) HealthCheck(ctx context.Context) (block.HealthCheckResult, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	return block.HealthCheckResult{
		Healthy: vm.initialized,
		Details: map[string]string{
			"height":  fmt.Sprintf("%d", vm.lastAcceptedHeight),
			"markets": fmt.Sprintf("%d", len(vm.books)),
			"pending": fmt.Sprintf("%d", vm.mempool.Len()),
		},
	}, nil
}

// WaitForEvent blocks until there is work to build a block. It returns PendingTxs
// when the mempool is non-empty. The engine calls this in a loop; we poll the
// mempool with a short backoff rather than holding a dedicated condition var,
// keeping the VM's locking simple (the mempool also pushes a PendingTxs signal on
// Add, so latency is bounded by the push, not the poll).
func (vm *VM) WaitForEvent(ctx context.Context) (block.Message, error) {
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		if vm.mempool != nil && vm.mempool.Len() > 0 {
			return block.Message{Type: block.PendingTxs}, nil
		}
		select {
		case <-ctx.Done():
			return block.Message{}, ctx.Err()
		case <-ticker.C:
		}
	}
}

// Shutdown releases VM resources. The durable DB is owned and closed by the
// rpc.Serve harness (it opened it), so we do not close it here — closing it twice
// is an error. We only drop in-RAM state.
func (vm *VM) Shutdown(ctx context.Context) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	vm.books = nil
	vm.acceptedBlocks = nil
	vm.heightIndex = nil
	vm.initialized = false
	return nil
}
