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

	// processingBlocks indexes built/verified blocks not yet accepted, so a
	// plugin-transport Accept — which carries only the block ID and re-resolves
	// the block via GetBlock (vm/rpc handleBlockAccept), unlike Verify which
	// carries the full bytes — can find the SAME instance (and its stashed Verify
	// overlay) to commit. Without it GetBlock(builtID) returns ErrNotFound, the
	// engine's self-finalize Accept is a silent no-op, and the submitTx waiter
	// hangs forever. The EVM gets this for free from geth's block cache. Pruned on
	// Accept/Reject. Guarded by vm.mu.
	processingBlocks map[ids.ID]*Block

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
	vm.processingBlocks = map[ids.ID]*Block{}
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

// genesisBlock deterministically reconstructs the genesis block (height 0, empty,
// zero parent root). It is a pure function of the VM binding — every validator and
// every restart computes the identical id — so it is the single source of the
// genesis block for both bootstrapGenesis (fresh DB) and loadHead's legacy-DB
// recovery (a DB written before head blocks were persisted).
func (vm *VM) genesisBlock() (*Block, [Size]byte) {
	var parentRoot [Size]byte // zero
	root, _, _, _ := ExecutionRoot(parentRoot, nil, nil, nil, 0)
	return newBlock(vm, genesisParent, 0, time.Unix(0, 0).UTC(), root, nil), root
}

// bootstrapGenesis writes the genesis block (height 0, empty, zero parent root)
// and sets it as the accepted head. Must be called under vm.mu on a fresh DB.
func (vm *VM) bootstrapGenesis() error {
	gen, root := vm.genesisBlock()

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
	if err := writeHeadBlock(batch, gen.bytes); err != nil {
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

// loadHead reads the accepted head (id, height, root) from durable meta AND
// reconstructs the head Block into the in-RAM acceptedBlocks map. The block store
// is the acceptedBlocks map (empty after a restart), so without this the engine's
// GetBlock(lastAccepted) — invoked immediately after Initialize — returns
// ErrNotFound and VM init fails ("get last accepted block: not found") on any
// chain that advanced past genesis. Must be called under vm.mu.
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

	head, err := vm.loadHeadBlock(id, height)
	if err != nil {
		return err
	}

	vm.lastAcceptedID = id
	vm.lastAcceptedHeight = height
	vm.lastRoot = root
	vm.heightIndex[height] = id
	vm.acceptedBlocks[id] = head
	return nil
}

// loadHeadBlock reconstructs the accepted head block named by (id, height). It
// reads the persisted head-block bytes and parses them, asserting the reconstructed
// id matches the head pointer. For a DB written before head blocks were persisted
// (legacy), the bytes are absent: a height-0 head is recovered from the
// deterministic genesis block (verified to match the stored id); a height>0 head
// with no persisted block is unrecoverable corruption and errors loudly rather
// than silently resetting the chain. Must be called under vm.mu.
func (vm *VM) loadHeadBlock(id ids.ID, height uint64) (*Block, error) {
	raw, err := readHeadBlock(vm.db)
	if err == database.ErrNotFound {
		// Legacy DB (head block never persisted). Only genesis is reconstructible
		// without the bytes.
		if height != 0 {
			return nil, fmt.Errorf("dchain: head block missing at height %d (corrupt or pre-persistence state); cannot recover", height)
		}
		gen, _ := vm.genesisBlock()
		if gen.id != id {
			return nil, fmt.Errorf("dchain: genesis id mismatch on legacy DB: stored %s computed %s", id, gen.id)
		}
		gen.status = statusAccepted
		return gen, nil
	}
	if err != nil {
		return nil, fmt.Errorf("dchain: read head block: %w", err)
	}

	head, err := parseBlock(vm, raw)
	if err != nil {
		return nil, fmt.Errorf("dchain: parse head block: %w", err)
	}
	if head.id != id {
		return nil, fmt.Errorf("dchain: head block id mismatch: pointer %s parsed %s", id, head.id)
	}
	if head.height != height {
		return nil, fmt.Errorf("dchain: head block height mismatch: meta %d block %d", height, head.height)
	}
	head.status = statusAccepted
	return head, nil
}

// rebuildAllBooks folds every market's book from its order:* rows. This is the
// restart path: after a crash the in-RAM books are reconstructed exactly from the
// durable resting set, so trading resumes with the committed state. Must be
// called under vm.mu.
//
// SETTLEMENT-IDENTITY BOOT ASSERTION: for every market whose (base,quote) assets
// are bound (a CUSTODY market that locks/settles value), every resting order MUST
// have an orderuser: mapping to its full 16-byte settlement identity. An order
// reaching this point on a custody market always has one (lockOrderSpend persists
// orderuser: BEFORE the order is matchable, and a custody order with no lock is
// rejected), so a missing row is state corruption: FAIL BOOT rather than continue
// on a degraded 8-byte path where a maker fill/cancel could only fall back to the
// compact handle. (Asset-less / pre-custody markets never lock or settle value, so
// their orders legitimately carry no orderuser: row and are not asserted.)
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
		if err := vm.assertOrderUserCoverage(poolID, ob); err != nil {
			return err
		}
		vm.books[poolID] = ob
	}
	return it.Error()
}

// assertOrderUserCoverage enforces the settlement-identity boot invariant for one
// market: if the market's assets are bound (custody market), every resting order
// in the rebuilt book MUST have an orderuser: row. A missing row on a custody
// market is unrecoverable state corruption (a maker fill/cancel could not resolve
// the full account) — return an error so Initialize FAILS rather than booting a VM
// that would settle on the degraded 8-byte handle. Asset-less markets are skipped:
// they never lock or settle value, so their orders carry no orderuser: row by
// design. Must be called under vm.mu.
func (vm *VM) assertOrderUserCoverage(poolID [32]byte, ob *lx.OrderBook) error {
	_, _, assetsBound, err := readMarketAssets(vm.db, poolID)
	if err != nil {
		return fmt.Errorf("dchain: read market %x assets for orderuser audit: %w", poolID[:8], err)
	}
	if !assetsBound {
		return nil // asset-less / pre-custody market: no settlement, no orderuser invariant
	}
	for _, o := range ob.Orders {
		if o == nil {
			continue
		}
		if _, ok, gerr := getOrderUser(vm.db, poolID, o.ID); gerr != nil {
			return fmt.Errorf("dchain: orderuser audit market %x order %d: %w", poolID[:8], o.ID, gerr)
		} else if !ok {
			return fmt.Errorf("dchain: %w: resting order %d on custody market %x has no orderuser row at boot (refusing to start on a degraded settlement path)",
				ErrMissingSettlementUser, o.ID, poolID[:8])
		}
	}
	return nil
}

// BookDepth reports the authoritative resting-book state for a market: the count
// of resting orders, their total remaining size, and the best bid/ask. It reads
// the in-RAM book (a fold of the committed order:* rows) under the VM lock —
// reads need no consensus round-trip (see handler.go). It is the read accessor
// the venue daemon and out-of-package callers use to observe consensus-settled
// state without reaching into unexported fields. Returns ok=false for an unknown
// market.
func (vm *VM) BookDepth(poolID [32]byte) (orders int, remaining, bestBid, bestAsk float64, ok bool) {
	vm.mu.Lock()
	ob := vm.books[poolID]
	vm.mu.Unlock()
	if ob == nil {
		return 0, 0, 0, 0, false
	}
	for _, o := range ob.Orders {
		orders++
		remaining += o.RemainingSize
	}
	d := ob.GetDepth(1)
	if d != nil {
		if len(d.Bids) > 0 {
			bestBid = d.Bids[0].Price
		}
		if len(d.Asks) > 0 {
			bestAsk = d.Asks[0].Price
		}
	}
	return orders, remaining, bestBid, bestAsk, true
}

// Balance reports an account's custody balances for an asset: the AVAILABLE
// (deposited, un-escrowed, withdrawable) amount and the LOCKED (reserved by live
// orders) amount, both in atomic asset units. It reads the durable ledger under
// the VM lock — the authoritative committed state, not the in-RAM book — so a
// caller (the proxy's withdraw leg, the venue daemon, tests) observes exactly
// what consensus settled. user is the identity string; it is rendered to the FULL
// 16-byte ledger identity internally (no 8-byte fold). asset is the FULL 32-byte
// injective asset id.
func (vm *VM) Balance(user string, asset [32]byte) (available, locked uint64, err error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	uid := userKey16(user)
	if available, err = getAvailable(vm.db, uid, asset); err != nil {
		return 0, 0, err
	}
	if locked, err = getLocked(vm.db, uid, asset); err != nil {
		return 0, 0, err
	}
	return available, locked, nil
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
	// Index the built block so a later Accept (which the plugin transport resolves
	// by ID via GetBlock, not by bytes) finds THIS instance to commit.
	vm.processingBlocks[blk.id] = blk
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
// a peer's proposed block; Verify then re-executes it. It returns a STABLE
// instance per id: the accepted block if already accepted, else the processing
// block already indexed (so the overlay Verify stashes on it is the one a later
// Accept commits), else a freshly parsed block which it records as processing.
func (vm *VM) ParseBlock(ctx context.Context, b []byte) (block.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	blk, err := parseBlock(vm, b)
	if err != nil {
		return nil, err
	}
	// Return the cached accepted block if we already have it (stable identity).
	if existing, ok := vm.acceptedBlocks[blk.id]; ok {
		return existing, nil
	}
	// Reuse the processing instance if present, so Verify's stashed overlay is the
	// same one Accept commits; otherwise record this one as processing.
	if existing, ok := vm.processingBlocks[blk.id]; ok {
		return existing, nil
	}
	vm.processingBlocks[blk.id] = blk
	return blk, nil
}

// GetBlock returns a block by id — accepted or still processing (built/verified
// but not yet accepted) — or database.ErrNotFound. The processing lookup is what
// lets the plugin transport's ID-only Accept resolve the block it must commit.
func (vm *VM) GetBlock(ctx context.Context, id ids.ID) (block.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	if blk, ok := vm.acceptedBlocks[id]; ok {
		return blk, nil
	}
	if blk, ok := vm.processingBlocks[id]; ok {
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

// inFlightHeight returns the height of the single block that could currently be in
// flight on this linear-chain proposer: lastAcceptedHeight+1. submitTx stamps a
// tombstone with it (R4) so gcTombstones can reclaim the tombstone once that height
// is accepted. Read under vm.mu to avoid a race with Accept's height advance.
func (vm *VM) inFlightHeight() uint64 {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	return vm.lastAcceptedHeight + 1
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

// CreateHandlers returns the VM's named HTTP handlers for luxd to mount under
// /ext/bc/<DCHAIN_ID>/ (and the "D" alias). It returns one handler per CLOB method
// keyed by its full sub-path ("/dex/<method>"), so an order POSTed to
// /ext/bc/D/dex/clob_submit reaches the matcher through the node's own router —
// the in-luxd ingestion seam (ingest.go). This is how an order enters the native
// VM: submitTx -> mempool -> consensus -> Verify-match. The plugin transport
// (github.com/luxfi/vm/rpc) serves these handlers from a local http.Server inside
// the plugin process and the node reverse-proxies to it.
func (vm *VM) CreateHandlers(ctx context.Context) (map[string]http.Handler, error) {
	return vm.httpHandlers(), nil
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
