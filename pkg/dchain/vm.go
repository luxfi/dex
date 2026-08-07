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
	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/rpc"
	"github.com/luxfi/runtime"
	"github.com/luxfi/version"
	luxvm "github.com/luxfi/vm"
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

	// runtime is the chain-scoped wiring the consensus engine hands the VM at
	// Initialize (block.Init.Runtime). It is the source of the cross-chain atomic
	// shared-memory handle (GetSharedMemory) and the C-Chain id the native settlement
	// seam (atomic.go) imports from / exports to. nil when no runtime was wired
	// (single-chain unit tests with no seam) — the seam then refuses (no mint).
	runtime *runtime.Runtime
	// cChainID is the C-Chain id resolved from the runtime: the partition a C->D
	// order object is read under (import) and a D->C settlement object is written
	// under (export). ids.Empty when no runtime / no C-Chain — the seam stays closed.
	cChainID ids.ID

	// normalOp is true once the engine has taken this VM to Ready — past the bootstrap
	// frontier, validating rather than replaying. It gates the BLOCK-LEVEL cross-chain
	// authentication only (verifySeamImports); nothing in the state transition reads
	// it, so it can never affect a state root.
	normalOp bool

	// depositAuthority is the ONLY account allowed to authorize a TxDeposit — the
	// trusted bridge/proxy that custodies the backing C-side value. A deposit MINTS
	// ledger balance, so it must be signed by this authority, NEVER by the crediting
	// account itself (a self-signed deposit is an unbacked mint — the F9 glitch).
	// The ZERO value means NO authority is configured, and EVERY TxDeposit is
	// rejected (fail-closed): with no bridge, money enters ONLY via the trustless
	// atomic import (TxImport, atomic.go), which is backed by a consumed C->D object.
	// It is a CONSENSUS parameter (from the chain config, identical on every
	// validator); authorizeTx reads it deterministically. Write-once at Initialize.
	depositAuthority userKey

	mempool *mempool

	// outcomes lets a ZAP handler block until its tx is decided by an accepted
	// block (Accept resolves the waiter with the consensus fills/ack).
	outcomes *outcomeRegistry

	// In-RAM accelerator: poolId -> rebuildable book. Truth is the order:* rows.
	books map[[32]byte]*dex.OrderBook

	// Accepted-block index + linear-chain head.
	acceptedBlocks     map[ids.ID]*Block
	heightIndex        map[uint64]ids.ID
	lastAcceptedID     ids.ID
	lastAcceptedHeight uint64
	lastRoot           [Size]byte

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

	// zapIngest is the canonical co-located ZAP DEX socket (zapingest.go), served
	// in-plugin when init Config sets zapIngestAddr. nil when HTTP-compat-only.
	// zapIngestCancel stops its serve goroutine on Shutdown. Both guarded by vm.mu.
	zapIngest       rpc.Server
	zapIngestCancel context.CancelFunc

	// autoDriveSeam enables the PROPOSER-SIDE DRIVE (drive.go): when set, BuildBlock
	// autonomously enumerates the committed cross-chain atomic state and emits the
	// TxImport / TxExport txs that import C->D orders and export D->C settlements —
	// the keeper-less native settlement seam. It is set true at Initialize exactly when
	// the cross-chain seam is wired (a runtime with shared memory + a resolved C-Chain
	// id); a single-chain runtime (no seam) leaves it false, and the seam unit tests
	// (atomic_seam_e2e_test) clear it to exercise executeImport/executeExport in
	// isolation via manual mempool injection. Guarded by vm.mu.
	autoDriveSeam bool

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

	// Route GPU shadow-gate divergences (a GPU match result that did NOT agree
	// with the CPU authority, byte-for-byte) into the node logger at ERROR. With
	// the default EngineCPU the gate never runs, so this is dormant until a node
	// operator opts in via LUX_DEX_MATCH_ENGINE=gpu-verified. A divergence here is
	// a hard alarm: the CPU authority was committed and the GPU output discarded,
	// and the GPU engine should be disabled on this node pending investigation.
	dex.SetMatchDivergenceSink(func(d dex.MatchDivergence) {
		vm.log.Error("dchain GPU shadow divergence (CPU authority committed; GPU output discarded)",
			"kind", d.Kind, "reason", d.Reason, "detail", d.Detail)
	})
	if engine := dex.MatchEngine(); engine != dex.EngineCPU {
		vm.log.Warn("dchain match engine: GPU shadow ENABLED — CPU remains the committed authority",
			"engine", engine.String())
	}

	vm.db = init.DB
	vm.toEngine = init.ToEngine
	// Capture the cross-chain wiring for the native settlement seam (atomic.go). The
	// runtime carries the shared-memory handle + the C-Chain id; both are absent in a
	// single-chain unit test, where the seam refuses rather than mint.
	vm.runtime = init.Runtime
	if init.Runtime != nil {
		vm.cChainID = init.Runtime.CChainID
	}
	// Drive the native settlement seam autonomously exactly when it is wired: a runtime
	// with cross-chain shared memory AND a resolved C-Chain id. With either absent the
	// seam is closed (executeImport/executeExport refuse — no mint), so the drive has
	// nothing to do and stays off. This is the keeper-less production default; the seam
	// mechanics tests clear it to inject TxImport/TxExport by hand.
	vm.autoDriveSeam = vm.runtime != nil && vm.runtime.GetSharedMemory() != nil && vm.cChainID != ids.Empty
	vm.mempool = newMempool(init.ToEngine)
	vm.outcomes = newOutcomeRegistry()
	vm.books = map[[32]byte]*dex.OrderBook{}
	vm.acceptedBlocks = map[ids.ID]*Block{}
	vm.processingBlocks = map[ids.ID]*Block{}
	vm.heightIndex = map[uint64]ids.ID{}

	// Resolve height 0 before anything else touches the chain. The genesis comes
	// from the chain-creation document the node supplies (init.Genesis) and must
	// match this node's own record of it; a disagreement refuses to start rather
	// than fork silently. See genesis.go.
	stored, err := readGenesis(vm.db)
	if err != nil && err != database.ErrNotFound {
		return fmt.Errorf("dchain: read stored genesis: %w", err)
	}
	_, err = readLastAccepted(vm.db)
	if err != nil && err != database.ErrNotFound {
		return fmt.Errorf("dchain: read last accepted: %w", err)
	}
	born := err == nil // this database already holds a chain

	if born && len(stored) == 0 {
		// Born under a build that kept no genesis record. Recoverable at height 0
		// from the chain's own head block, and nowhere else.
		if stored, err = vm.recoverGenesis(); err != nil {
			return err
		}
	}

	gen, err := vm.genesis(init.Genesis, stored)
	if err != nil {
		return err
	}

	if born {
		if err := vm.loadHead(); err != nil {
			return err
		}
		if vm.lastAcceptedHeight == 0 && vm.lastAcceptedID != gen.id {
			return fmt.Errorf("dchain: refusing to start: the head at height 0 is %s but this chain's genesis is %s",
				vm.lastAcceptedID, describeGenesis(gen))
		}
		if err := vm.rebuildAllBooks(); err != nil {
			return err
		}
	} else if err := vm.bootstrapGenesis(gen); err != nil {
		return err
	}

	vm.preferred = vm.lastAcceptedID

	// Start the canonical co-located ZAP DEX ingestion socket if the chain config
	// names a listen address (zapingest.go). This runs AFTER the durable state is
	// loaded so the socket never accepts an order the VM cannot yet sequence. It is
	// a no-op (HTTP-compat-only) when no address is configured. Transport ⟂
	// consensus: orders from this socket take the identical submitTx -> mempool ->
	// Verify-match path as the HTTP route.
	cfg, err := parseConfig(init.Config)
	if err != nil {
		return err
	}
	// Resolve the deposit authority (the trusted bridge/proxy account) from the
	// chain config. Empty config => zero authority => all TxDeposit fail-closed
	// (deposits then enter ONLY via the backed atomic import). A malformed value is
	// a hard Initialize error — a misconfigured backing authority must not boot.
	auth, aerr := cfg.depositAuthorityKey()
	if aerr != nil {
		return fmt.Errorf("dchain: deposit authority config: %w", aerr)
	}
	vm.depositAuthority = auth
	if err := vm.startZAPIngest(cfg.ZAPIngestAddr); err != nil {
		return err
	}

	vm.initialized = true
	// genesis is logged on every start so which chain a node is on is answerable
	// from its logs alone. Two nodes that cannot agree here agree about nothing.
	vm.log.Info("dchain VM initialized",
		"genesis", gen.id,
		"createdFrom", fmt.Sprintf("%x", genesisOrigin(init.Genesis)),
		"height", vm.lastAcceptedHeight,
		"lastAccepted", vm.lastAcceptedID,
		"markets", len(vm.books),
		"zapIngest", cfg.ZAPIngestAddr != "",
	)
	return nil
}

// bootstrapGenesis commits gen as the chain's height-0 head and, in the same
// batch, records it as the chain's genesis — the node's permanent answer to "which
// chain am I on". Every later start reads that record back and asserts against it,
// so this is the one and only moment the answer is decided. Must be called under
// vm.mu on a database that holds no chain.
func (vm *VM) bootstrapGenesis(gen *Block) error {
	batch := vm.db.NewBatch()
	if err := writeGenesis(batch, gen.bytes); err != nil {
		return err
	}
	if err := writeLastAccepted(batch, gen.id); err != nil {
		return err
	}
	if err := writeHeight(batch, 0); err != nil {
		return err
	}
	if err := writeRoot(batch, gen.execRoot); err != nil {
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
	vm.lastRoot = gen.execRoot
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

// loadHeadBlock reconstructs the accepted head block named by (id, height) from
// its persisted bytes, asserting the reconstructed id and height match the head
// pointer. A head pointer with no block behind it is unrecoverable corruption and
// errors loudly rather than silently resetting the chain — reconstructing height 0
// from the binary instead is exactly the silent fork this package refuses to
// commit. Must be called under vm.mu.
func (vm *VM) loadHeadBlock(id ids.ID, height uint64) (*Block, error) {
	raw, err := readHeadBlock(vm.db)
	if err != nil {
		return nil, fmt.Errorf("dchain: read head block at height %d: %w", height, err)
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
// SETTLEMENT-IDENTITY HOLDS BY CONSTRUCTION — no boot scan. A resting custody order
// without its orderuser: row is UNREPRESENTABLE in committed state: a place's lock,
// its per-order reserve, AND its orderuser: identity row are written in lockOrderSpend
// (block.go) against the SAME versiondb overlay as the order:* row (putOrderRow), and
// the whole block commits in ONE atomic overlay.Commit() (Block.Accept). An unfunded
// place never rests (lockOrderSpend returns !ok, execute skips applyTx), so the order
// row is never written without the matching orderuser: row — they land together or
// not at all. There is therefore no degraded-boot state to scan for, so no boot-brick:
// the invariant is enforced where the state is created, not re-checked where it is
// loaded (illegal state unrepresentable; see TestOrderUserPersistedBeforeBookInsert and
// TestCommittedStateNeverHasOrderWithoutIdentity). The single remaining settlement-
// identity guard is the consensus-time fail-closed in settleOrderEffects: a fill/cancel
// that cannot resolve an orderuser: row aborts the block with ErrMissingSettlementUser.
// On honest committed state it can never fire (the row is always present by the above);
// it stands only as a defense against an in-block hazard, never as a boot gate.
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

	// PROPOSER-SIDE DRIVE (drive.go): before draining the mempool, enumerate the committed
	// cross-chain atomic state and autonomously generate the seam txs — TxImport for every
	// un-imported C->D order, TxExport for every open order whose owner now holds realized
	// proceeds. Both are pure, deterministic functions of committed state (shared memory +
	// the seamintent: escrow + the ledger), so every validator re-derives the identical
	// effects in execute and a divergent proposer is rejected on the tx root. This is the
	// keeper-less native settlement seam — the VM imports C->D orders and exports D->C
	// settlements during normal block production. A no-op when the seam is unwired.
	imports, err := vm.driveSeamImports(maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: drive imports: %w", err)
	}
	orders, err := vm.driveSeamOrders(maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: drive orders: %w", err)
	}

	exports, err := vm.driveSeamExports(maxSeamDrivePerBlock)
	if err != nil {
		return nil, fmt.Errorf("dchain: drive exports: %w", err)
	}

	mtxs := vm.mempool.Drain(0)

	// Assemble [imports || exports || mempool]. Imports come FIRST so a same-block taker
	// order draws from the just-imported funds; exports come before the mempool so they
	// settle committed proceeds before any new mempool order can touch the same account.
	// The seam txs and the mempool txs are disjoint by construction (an order is imported,
	// then traded, then exported across distinct blocks).
	txs := make([]*Tx, 0, len(imports)+len(orders)+len(exports)+len(mtxs))
	txs = append(txs, imports...)
	txs = append(txs, orders...)
	txs = append(txs, exports...)
	txs = append(txs, mtxs...)
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
		// chain: drop the MEMPOOL batch back so it is retried. The seam txs are
		// regenerated from committed state on the next build, so they are not
		// requeued (they are not mempool-owned).
		vm.mempool.Requeue(mtxs)
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

// SetState transitions the VM lifecycle state (bootstrapping/normal-op/etc). The
// d-chain's own execution has no state-specific behavior — it is ready to build/verify
// as soon as Initialize completes — but the BLOCK-LEVEL cross-chain authentication
// does: below the bootstrap frontier a C->D object may legitimately be absent from
// shared memory (C and D bootstrap independently, and this node's own accepted Remove
// may already have consumed it), and those blocks carry the network's acceptance as
// their authority. So the one thing tracked here is whether we have reached normal
// operation — exactly the gate verifySeamImports needs.
func (vm *VM) SetState(ctx context.Context, state uint32) error {
	vm.mu.Lock()
	vm.normalOp = state == uint32(luxvm.Ready)
	normalOp := vm.normalOp
	vm.mu.Unlock()
	vm.log.Debug("dchain SetState", "state", state, "normalOp", normalOp)
	return nil
}

// Version returns the VM version string.
func (vm *VM) Version(ctx context.Context) (string, error) { return Version, nil }

// NewHTTPHandler returns the VM's HTTP handler. The d-chain serves its DEX API
// over the ZAP gateway (handler.go), not HTTP, so this is nil.
func (vm *VM) NewHTTPHandler(ctx context.Context) (http.Handler, error) { return nil, nil }

// CreateHandlers returns the VM's named HTTP handlers for luxd to mount under
// /v1/bc/<DCHAIN_ID>/ (and the "D" alias). It returns one handler per DEX method
// keyed by its full sub-path ("/dex/<method>"), so an order POSTed to
// /v1/dex/dex/dex_submit reaches the matcher through the node's own router —
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
	// The seam drive (drive.go) can have work even with an empty mempool — a pending
	// C->D order to import, or an open order with realized proceeds to export. Those
	// appear at the source chain's block cadence (seconds), so they are polled on a
	// slower, separate tick to keep the hot mempool path cheap.
	seamTicker := time.NewTicker(time.Second)
	defer seamTicker.Stop()
	for {
		if vm.mempool != nil && vm.mempool.Len() > 0 {
			return block.Message{Type: block.PendingTxs}, nil
		}
		select {
		case <-ctx.Done():
			return block.Message{}, ctx.Err()
		case <-ticker.C:
		case <-seamTicker.C:
			vm.mu.Lock()
			work := vm.hasSeamWork()
			vm.mu.Unlock()
			if work {
				return block.Message{Type: block.PendingTxs}, nil
			}
		}
	}
}

// Shutdown releases VM resources. The durable DB is owned and closed by the
// rpc.Serve harness (it opened it), so we do not close it here — closing it twice
// is an error. We only drop in-RAM state.
func (vm *VM) Shutdown(ctx context.Context) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	vm.stopZAPIngest()
	vm.books = nil
	vm.acceptedBlocks = nil
	vm.heightIndex = nil
	vm.initialized = false
	return nil
}
