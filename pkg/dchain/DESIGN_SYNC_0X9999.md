# Synchronous Native 0x9999 Swaps — One Consensus Domain, Two Execution Surfaces

Status: DESIGN ONLY. Changes no code. The BFT quorum-finality fix is a separate
prerequisite (scoped elsewhere); this design assumes quorum finality exists as the
substrate (i.e. a block is Accepted only after alpha-of-k votes, no proposer self-
finalize). Every architectural claim below cites the actual code it rests on.

Target model: ONE consensus domain (one BFT instance), TWO execution
surfaces over ONE state machine:
  - The EVM-adapter surface = the C/cEVM surface: 0x9999 PoolManager.swap, 0x9998 Quoter,
    0x9997 StateView, 0x9996 PositionManager (Uniswap-v4-shaped EVM ABI).
  - The native DEX core = the D/native core: the CLOB matcher + DEX balances/locks +
    LP/order/position state + fills/trades — `dex/pkg/lx` (matcher) and the state
    machine in `dex/pkg/dchain` (VM).
0x9999.swap is SYNCHRONOUS: decode V4 PoolKey/SwapParams → call the native DEX core
deterministically, in-process → apply C balance/allowance changes AND D book/fill/
position changes → emit V4-compatible events → return amountOut/BalanceDelta or
revert. The whole transition finalizes in ONE consensus block, both-or-neither.

---

## 1. Current structure (file-grounded)

### 1.1 C and D are SEPARATE OS PROCESSES, each with its OWN consensus engine

This is the single most important fact and it is the opposite of the one-consensus
target model: today C and D are two consensus domains, not one.

- C-Chain (cEVM) and D-Chain (dexvm) are BOTH `OptionalVMs`, i.e. plugin-only,
  loaded from PluginDir and launched as subprocesses via rpcchainvm/ZAP transport:
  - `node/node/vms.go:117-120` — `OptionalVMs` lists `DexVMID {Name:"dexvm"}` and
    `EVMID {Name:"evm"}`; the file's own doc-comment (`vms.go:22-40`) states
    "OptionalVMs: plugin-only (loaded from PluginDir … via the upstream VMRegistry
    scan)" and "CoreVMs: in-process … {P, X, Q, Z}". C and D are explicitly NOT in
    the in-process set.
  - `node/vms/registry/registry.go:136-145` — a PluginDir VM is wrapped by
    `rpcchainvm.NewFactory(pluginPath, …)`.
  - `node/vms/rpcchainvm/factory.go` — `factory.New` does `subprocess.NewCmd(path)`
    with `VM_TRANSPORT=zap` and dials the plugin over ZAP. Each VM is a separate
    OS process.
- Each chain gets its OWN consensus engine instance:
  - `node/chains/manager.go:1223` — `consensuschain.NewRuntime(NetworkConfig{ChainID, …, VM: blockBuilder, Params:&consensusParams})` is constructed once PER chain inside `buildChain`; the engine is stored on `chainInfo.Engine` (`manager.go:234-248`, `:1304-1311`). There is no shared engine.
- The cross-chain substrate between them is `atomic.SharedMemory`:
  - `node/node/node.go` `initSharedMemory()` builds ONE global `atomic.NewMemory(...)`; `node/chains/manager.go:985-987` hands each chain a per-chain slice `AtomicMemory.NewSharedMemory(chainID)`.

Consequence: today, "C calls D" can only mean one of (a) a network ZAP call to the D
process, or (b) a cross-process atomic shared-memory handoff. Both cross a process
boundary and a separate-consensus boundary. A synchronous in-process function call
from the 0x9999 precompile into the matcher does NOT exist and is not possible in the
current topology without restructuring.

### 1.2 Where 0x9999 lives today, and how it reaches D (the deprecated async path)

0x9999 is a cEVM precompile (`precompile/dex`), dispatched inside the C-Chain EVM
plugin (`evm/precompile/registry/registry.go`, `evm/plugin/evm/vm.go`). It is the
SOLE DEX money path (`precompile/dex/settle_addr.go:13-16`: "0x9999 is the sole DEX
precompile … exactly one money path"). Its production engine is
`NativeDChainClient` (`precompile/dex/native_dchain_client.go`).

The current model is async two-phase atomic, NOT synchronous:
- `SubmitSwapIntent` (`native_dchain_client.go:207-299`) LOCKS the taker's input on
  C into the 0x9999 seam reserve, derives an intent id, and STAGES a C→D atomic
  object — it "returns an intent id ONLY — never a live fill" (`:30-46`).
- `ImportSettlement` (`:590-732`) is the ONLY path that CREDITS C, by consuming a
  D→C atomic object exactly once, bound to the recorded value.
- The synchronous `Engine.Swap` surface is deliberately CLOSED:
  `NativeDChainClient.Swap` returns `ErrDChainUnavailable` with the comment "A
  synchronous in-block fill via a live query forks consensus (the whole reason for
  the async atomic model)" (`native_dchain_client.go:918-923`). The `Engine`
  interface doc says the same: a live query to a separate chain's moving book inside
  C-Chain block execution "forks (each validator observes independently-timed fills
  => divergent StateRoot)" (`precompile/dex/engine.go:20-27`).

The cross-domain commit is deferred to block accept and is revert-safe:
- `precompile/dex/native_staging.go` — the precompile NEVER calls
  `atomic.SharedMemory.Apply` inside `Run`; it STAGES Put/Remove into its own StateDB
  namespace (snapshot/revert-aware), and the HOST flushes the window at block accept.
- The host hook ALREADY EXISTS on the C side: `evm/plugin/evm/vm.go:1855-1918`
  `stageDexAtomic` reads `dex.ReadStagedAtomicSeq(parent/current)`, calls
  `dex.CollectStagedAtomicRange`, and `Block.Accept` does `sm.Apply(reqs, batch)` in
  ONE atomic database write (the platformvm acceptor pattern).

### 1.3 The native D-Chain core is already a deterministic matcher-at-Verify VM

- `dex/pkg/dchain/vm.go` — a standalone `block.ChainVM`; "BuildBlock drains the
  mempool in sequence order, Block.Verify runs the matcher against a versiondb
  overlay, Block.Accept commits the overlay atomically" (`vm.go:30-33`).
- `dex/pkg/dchain/block.go:153-189` — `Block.Verify` builds a `versiondb.New(vm.db)`
  overlay, calls `execute` (which runs the matcher), and CHECKS the proposer's
  claimed `execRoot` against the locally-derived root (`:181-184`). Every validator
  re-derives fills; a lying proposer is rejected. `Accept` commits that overlay
  atomically (`block.go:195-…`, with `writeLastAccepted/Height/Root/HeadBlock` in the
  SAME overlay batch).
- `dex/pkg/dchain/execute.go` — `applyTx` is "a pure function of (tx, book state,
  height, ts, txIndex): no time.Now(), no LastOrderID mint" (`:69-78`). Order ids are
  `blockDeterministicID(height, txIndex)` (`:42-45`). The fills are byte-identical on
  every validator.
- The matcher core `dex/pkg/lx` is PURE GO on its consensus path:
  `ConsensusAddOrder` (`consensus.go:52`) and `SubmitMarketable`
  (`orderbook.go:282`) take no clock, mint nothing. The only cgo files
  (`amm_gpu_cuda.go`, `orderbook_gpu_cuda.go`, `*_metal*.go`) are GPU-accel, separately
  build-tagged — they are NOT on the deterministic match path. So `pkg/lx` and
  `pkg/dchain` are importable in-process by another VM with no cgo and no heavy deps
  (verified: no `import "C"` in `pkg/dchain`).

Net: the native-DEX-core matcher already runs deterministically inside a Lux VM
block. What it does NOT yet do is run inside the SAME block as the cEVM that exposes
the V4 ABI. That is the whole gap.

---

## 2. The key decision — make C+D ONE atomic transition

### Options (as posed), evaluated against the code above

(a) **dexcore as an in-process MODULE the cEVM 0x9999 calls directly; combined C+D
    state committed in ONE block.** This is the one-consensus model: the native DEX
    core is not a separate chain — it is a second execution surface over the same
    single-consensus
    state machine. In Lux terms: the cEVM VM (C) imports `dex/pkg/lx` (+ a thin
    state-binding lifted from `dex/pkg/dchain`) as an in-process library. 0x9999.swap
    calls the matcher directly; the matcher's book/fill/position writes land in the
    cEVM block's own state commit alongside the EVM balance/allowance writes. ONE
    block, ONE state root, ONE consensus engine (C's). No shared memory, no second
    consensus, no async settle. The standalone D-Chain VM is RETIRED as a separate
    chain; its reusable core (`pkg/lx`, and the deterministic execute/state logic in
    `pkg/dchain`) is reused as the in-process module.

(b) **Keep D a separate chain, make C+D blocks atomic in one consensus round.** This
    means co-finalizing two separate VMs' blocks (C-engine and D-engine per
    `manager.go:1223`) in a single round — a cross-chain atomic-commit protocol the
    engine does not have and would have to grow. Two block builders, two verify
    paths, two state roots that must be made jointly-final, plus a deterministic
    inter-VM call channel inside Verify. This is strictly more machinery than (a) for
    the same result, and it re-introduces exactly the "live query into another
    consensus instance inside Verify" hazard the current code calls a fork
    (`engine.go:20-27`). Rejected.

(c) **Tighten the existing atomic import/export into one logical transition.** The
    import/export legs are two SEPARATE C transactions in (generally) two different
    blocks: `SubmitSwapIntent` locks now, `ImportSettlement` credits after D matches
    and exports (`native_dchain_client.go:30-57`). "One logical transition" over a
    two-block, two-process, cross-consensus handoff is a UX veneer, not atomicity:
    between the legs the taker's funds are locked with the fill not yet realized, and
    liveness depends on a keeper + D + the deadline-gated `ReclaimIntent`
    (`:800-897`). It cannot deliver "both-or-neither in ONE block." Rejected for
    normal swaps (it remains valid for genuine cross-chain settlement; see §5).

### DECISION

Decision: **(a) — dexcore as an in-process module of the cEVM. Combined C+D state
commits in ONE cEVM block under ONE consensus engine. The standalone D-Chain
consensus instance is retired for spot/CLOB trading; its core libraries are reused
in-process.**

Rationale:
- It is the one-consensus topology (one consensus, two execution surfaces),
  and it is the only option that yields true single-block both-or-neither without
  inventing a cross-chain co-finalization protocol.
- The pieces already fit: the matcher (`pkg/lx`) and the deterministic state logic
  (`pkg/dchain`) are pure-Go and importable; the cEVM already runs a host hook at
  block accept (`evm/plugin/evm/vm.go:1855`) where combined state is committed; the
  matcher-at-Verify discipline (`dchain/block.go:161`) is exactly the determinism
  contract the EVM block already enforces on every validator.
- It deletes the entire async surface (shared-memory staging, intent/settle, keeper,
  reclaim) from the NORMAL swap path — less code, one money path, one replay
  namespace, which is the standing project law ("exactly one way").

Trade-off (named, and why acceptable):
- We give up D as an independently-schedulable, independently-validatable chain with
  its own block cadence. The D matcher now advances at the cEVM's block rate and is
  validated by the C validator set, not a separate DEX-operator set. Acceptable:
  this is the deliberate one-consensus choice (the native DEX core and the EVM-adapter
  surface share one validator set
  and one block), and the Lux DEX product goal is a single trustless venue, not a
  federation of independently-finalized order books. The `pkg/dchain` standalone VM
  is not wasted — it becomes the library home for the deterministic execute/state
  code the cEVM module imports (see §5), and it remains runnable standalone for
  local matcher testing.

### Where the D book-state lives, and how it commits atomically with C

D book/order/position/balance state moves INTO the cEVM state trie under the 0x9999
account's storage namespace, exactly where the seam records already live
(`settleStateNamespace = "dex.precompile.v1.9999."`, `settle_addr.go:27`). Concretely:
- Balances/locks: the existing seam pots already model this — `seamReserve`,
  `committedPositions`, depositor `settleVault`, `makerLockedVault`
  (`native_state.go:206-285`) with the per-asset vault invariant
  `realHolding(0x9999,a) == settleVault + makerLockedVault + seamReserve +
  committedPositions`. Synchronous swaps keep these pots and the invariant; they just
  debit/credit them in the SAME call instead of across two blocks.
- Resting book + orders + positions: persisted as 0x9999 storage rows, the in-trie
  analog of `dchain` `order:*`/`market:*` rows. The in-RAM `OrderBook`
  (`pkg/lx.OrderBook`) is a rebuildable accelerator folded from those rows on VM init
  — the same rebuild discipline `dchain/vm.go:251-316` (`rebuildAllBooks`) already
  implements, ported to read 0x9999 storage instead of a standalone db.
- Fills/trades: emitted as V4-compatible logs in the same block (see §3), and any
  durable fill index written as 0x9999 rows.

Because every D write is a 0x9999 StateDB write, it is part of the cEVM block's state
root and is committed by the cEVM's normal `Block.Accept` (the trie commit), the SAME
batch the EVM balance writes commit in. Atomicity is then free and total: there is no
second store and no shared-memory leg to keep in sync — the "two domains, two commit
semantics" problem that `native_staging.go:18-33` was built to paper over simply
disappears for the synchronous path.

Determinism: the matcher is already a pure function of (state, ordered txs, height,
ts) (`execute.go:69-78`, `dchain/block.go:158-160`). Running it inside the EVM does
not change that — the EVM block already supplies a deterministic ordered tx list, a
fixed block timestamp, and a fixed height, replayed identically by every validator in
Verify. The one rule the module MUST keep: the matcher reads its block time and order
ids from EVM block context (`state.GetBlockContext()`), never from `time.Now()` or a
process-local counter — identical to what `applyTx` already enforces.

Fate of the just-built native D-Chain VM: the standalone `cmd/dchain` /
`cmd/dvenue` consensus deployment is deprecated for production trading. The library
(`pkg/lx`, plus the deterministic `execute.go`/`state.go`/settlement logic of
`pkg/dchain`) is PROMOTED to the shared core both the (optional, test-only) standalone
VM and the in-process cEVM module call. Nothing in the proven matcher is rewritten;
it changes callers, not logic.

---

## 3. V4SwapEnvelope + deterministic execution path

### 3.1 The envelope (a pure value carried in the cEVM tx; no live external input)

`V4SwapEnvelope` is the in-block representation of one synchronous swap. It is fully
derivable from the EVM calldata + prior committed state, so every validator
reconstructs it identically. No field is a live query result.

```
V4SwapEnvelope {
  // cCall — straight from the 0x9999.swap ABI decode (V4 PoolManager.swap shape):
  caller      common.Address     // EVM tx caller; the taker / spender
  poolKey     PoolKey            // currency0, currency1, fee, tickSpacing, hooks
  swapParams  SwapParams         // zeroForOne, amountSpecified (signed), sqrtPriceLimitX96
  hookData    []byte

  // dAction — derived deterministically from cCall (NOT a separate user input):
  marketID    [32]byte           // = poolKey.ID() (precompile/dex pool_manager: key.ID())
  side        lx.Side            // from zeroForOne
  amountIn    uint64             // |amountSpecified| for exact-input (asset-unit, observed)
  limitOrMin  { priceLimit uint64; limitIsUpper bool; minOut *big.Int }
                                 // priceLimit via priceLimitToCLOB(sqrtPriceLimitX96)
  route       []marketHop        // [] for a direct market; multi-hop for router paths

  // expectedAccess — the state this transition reads/writes, for replay binding:
  cAccess     { balanceOf(caller, assetIn), allowance/vault(assetIn),
                seamReserve[assetIn], seamReserve[assetOut] }
  dAccess     { market(marketID), book(marketID) resting set, positions touched }
}
```

The envelope is constructed at the TOP of the 0x9999.swap precompile `Run`, purely
from the decoded calldata and `AccessibleState` reads. `marketID = poolKey.ID()`
reuses the existing `PoolKey.ID()`/`key.ID()` already used by the PoolManager
(`pool_manager.go` Initialize/Swap). `priceLimitToCLOB` already exists for the MEV
floor (`native_dchain_client.go:734-798` references it). Nothing here is novel data;
it is the same inputs the async path already decodes, minus the intent/keeper hop.

### 3.2 The synchronous execution path (the heart of the change)

This runs ENTIRELY inside cEVM `0x9999.swap.Run`, which executes inside the cEVM
`Block.Verify` on every validator and inside `BuildBlock` on the proposer — the same
points where every other EVM tx runs. There is no ZAP, no shared memory, no second
chain in this path.

```
0x9999.swap.Run(state, caller, calldata):
  1. env := decodeV4SwapEnvelope(calldata, caller, state)      // pure, deterministic
  2. checkPauseState(state, env.marketID)                      // existing gate (pool_manager:checkPauseState)
  3. book := loadBook(state, env.marketID)                     // fold 0x9999 rows -> in-RAM lx.OrderBook
  4. LOCK input: debit caller's assetIn into the 0x9999 vault  // existing lockIntentInput discipline,
        observed-delta (fee-on-transfer safe)                  //   but NOT staged across blocks — local
  5. order := buildDeterministicOrder(env, blockCtx.Number, callIndex, blockCtx.Time)
        // id = blockDeterministicID(height, callIndex); ts = block time  (execute.go:42,79)
  6. fills := book.SubmitMarketable(order)                     // pkg/lx, pure-Go, no clock  (orderbook.go:282)
  7. enforceProceedsPriceFloor(env.priceLimit, ..., spent, out)// taker-authenticated MEV floor (native_dchain_client.go:734)
        // revert if violated -> whole tx reverts -> step 4 lock rolls back (EVM snapshot)
  8. APPLY D state in-trie:                                    // all 0x9999 storage writes
        - rewrite touched maker rows / delete filled makers    (dchain execute applyResult.Touched)
        - credit/debit balances+locks for taker and makers     (dchain settleOrderEffects analog)
        - persist resting remainder (if any) as a book row
  9. APPLY C state: credit taker assetOut from the vault       // existing creditSettlementOutput discipline,
        observed-delta; maintain the per-asset vault invariant //   in the SAME call, not a later import
 10. emit V4 events: Swap(poolId, caller, BalanceDelta, ...)   // existing emitSwapEvent (pool_manager:929)
        + DEXFill(bytes32,address,uint256,uint256) @0x9999     // the native fill log (see MEMORY: 0x9999 fill event)
 11. return BalanceDelta{amount0, amount1}  (or revert)
```

Key properties, each inherited from existing proven code:
- Steps 4 + 8 + 9 are ONE EVM call, covered by ONE EVM snapshot. A revert at 7/8/9
  rolls back the lock AND any partial D write atomically — the EVM snapshot does what
  `native_staging.go` had to emulate for the cross-process case.
- Step 6 is the proven deterministic matcher; step 8 mirrors the proven
  `settleOrderEffects`/`decrementMakerReserves` value moves (`dchain/block.go:706-…`).
- No `atomic.SharedMemory.Apply`, no staged window — the cEVM block's own state root
  carries the entire result.

### 3.3 Proposer BuildBlock vs validator VerifyBlock (the replay contract)

The directive's option of "ZAP in BuildBlock, fill vector carried in block bytes,
replayed in Verify" (Option B in the prompt) is NOT NEEDED in design (a), and should
be avoided:
- In design (a) the matcher runs in-process during BOTH BuildBlock and Verify, over
  in-trie state. There is no external venue to ZAP. The proposer does not learn the
  fills from anywhere except running the same pure function every validator runs.
- Therefore the "fill vector in block bytes" is just the ordinary EVM tx list + the
  resulting state root. The proposer commits a state root (the EVM block header
  `Root`); every validator re-executes the tx list in Verify and MUST derive the same
  root (`consensus/engine/chain/integration.go:400,411` — ParseBlock then Verify on
  the proposer's exact bytes; `evm` block Verify re-runs the txs). A proposer that
  fabricates a fill produces a state root no honest validator reproduces, so Verify
  rejects it — the SAME mechanism `dchain/block.go:181-184` uses (claimed root vs
  derived root), now expressed as the EVM state root.

So the deterministic-replay design is: proposer BuildBlock runs the swap (may use a
local quote/ZAP ONLY for off-consensus UI hints — see §3.4), commits the EVM state
root; validators ParseBlock + VerifyBlock re-run 0x9999.swap from block bytes + prior
trie state with NO live ZAP and MUST match the root. Option A from the prompt (ZAP as
an in-process/local pipe OUTSIDE consensus) is the chosen shape; Option B's
block-carried fill vector collapses into "the EVM state root already commits it."

### 3.4 ZAP's residual role (outside consensus only)

ZAP (`chains/dexvm/relay.go`, `dex/pkg/api`, `dex/pkg/zapwire`) is NOT on the
synchronous money path anymore. Its legitimate remaining uses are all OUTSIDE
consensus: quote/route/depth for the UI, session/websocket market data, and the
standalone test venue. A node MAY answer ZAP `clob_*` read frames from its in-trie
book state for compatibility, but a validator NEVER issues a live ZAP query during
Verify (that is the fork hazard, and design (a) removes the need entirely). The
standing memory rule "NEVER wire validators' dex-zap-endpoint to a venue" is upheld
and in fact made structural: there is no venue to wire.

---

## 4. The 8 required tests → concrete points in the design

These map onto the cEVM block lifecycle (`consensus/engine/chain/*`) + the in-process
matcher module. They assume the BFT-finality prereq (no proposer self-finalize).

1. **Test9999SyncSwap_FinalizesWithQuorum** — Build a block whose only tx is a
   0x9999.swap on a seeded market with crossable depth. Drive the cEVM chain engine
   with k>1 validators; assert `Block.Accept` fires ONLY after `acceptVotes >= alpha`
   (`consensus/engine/chain/consensus.go:110-114`), and that post-accept the taker's
   EVM balance reflects amountOut and the book reflects the fill. Anchor: the
   combined commit in §3.2 step 8+9 under the cEVM accept path.

2. **Test9999SyncSwap_NoProposerSelfFinalize** — The negative of #1 and the seam to
   the finality prereq: with the prereq applied, assert a proposer that has built and
   locally Verified the swap block does NOT transition it to Accepted until quorum
   votes arrive (i.e. `finalizeOwnProposal`/`ForceAccept`,
   `consensus/engine/chain/engine.go:891-938,1133-1137`, is gone/gated). This test
   FAILS on today's engine and PASSES once the prereq lands — it is the explicit
   guard that the DEX change is not shipped on top of self-finalization.

3. **Test9999SyncSwap_ByzantineProposerFakeFillRejected** — Proposer builds a block
   header whose state root claims a fill that the matcher would not produce (e.g.
   credits the taker more than `SubmitMarketable` yields, or fills against a maker
   that should not cross). Honest validators re-run 0x9999.swap in Verify, derive a
   different state root, and REJECT. Anchor: the EVM-state-root analog of
   `dchain/block.go:181-184` (claimed vs derived root), enforced by
   `integration.go:411` Verify on the proposer's bytes.

4. **Test9999SyncSwap_ValidatorReplayNoLiveZAP** — Run a validator's Verify with the
   ZAP dialer hard-disabled (inject a `zapDialer` that panics, mirroring the harness
   in `chains/dexvm/relay.go:102`). The swap MUST still Verify to the identical root,
   proving Verify reads only in-trie state + the in-process matcher and never a live
   venue. Anchor: §3.3 — no external input in the Verify path.

5. **Test9999SyncSwap_CAndDCommitAtomically** — After Accept, assert BOTH surfaces
   moved: C side (taker EVM balance debited assetIn / credited assetOut; vault
   invariant `realHolding(0x9999,a) == settleVault+makerLockedVault+seamReserve+
   committedPositions` holds, `native_state.go:218`) AND D side (book remainder,
   maker rows, locks). Then take the block state root and a fresh node re-deriving it:
   both surfaces appear or neither. Anchor: single EVM state commit, §2/§3.2.

6. **Test9999SyncSwap_RevertRollsBackCAndD** — Force a revert AFTER the lock and a
   partial D write (e.g. `enforceProceedsPriceFloor` fails at §3.2 step 7, or a
   forced error at step 8). Assert the taker's lock is fully refunded, NO maker row
   changed, NO balance moved, NO event emitted, and the block state root equals the
   pre-swap root for those slots. Anchor: ONE EVM snapshot covers steps 4/8/9 — the
   property `native_staging.go:18-33` had to engineer for the async case is now
   intrinsic.

7. **Test9999SyncSwap_SameInputSameFillAllValidators** — N independently-constructed
   VM instances (distinct process state, distinct in-RAM book caches rebuilt from the
   same rows) execute the SAME block; assert byte-identical fills, balance deltas, and
   state root across all N. Anchor: matcher purity (`execute.go:69-78`,
   `consensus.go:52` ConsensusAddOrder; the existing
   `orderbook_gpu_test.go:198` "RejectsNonDeterministicInput" guard) + block-
   derived ids/timestamps.

8. **Test9999SyncSwap_TradeVisibleInDStateAndCEvents** — After Accept, read the D
   state via the 0x9999 view selectors (book depth, balances — the in-trie analog of
   `dchain` `BookDepth`/`Balance`, `vm.go:325-367`) AND assert the C event log
   contains the V4 `Swap` event (`emitSwapEvent`, `pool_manager.go:929`) and the
   `DEXFill(bytes32,address,uint256,uint256)` log at 0x9999 (the indexer source per
   project memory). This is the test the current async path cannot satisfy in one
   block, and it is the acceptance gate for the indexer/lux.exchange pipeline.

---

## 5. Reconcile with what is built — carries forward vs deprecated

CARRIES FORWARD (reused, not rewritten):
- The matcher `dex/pkg/lx` (OrderBook, ConsensusAddOrder, SubmitMarketable) — the
  proven, deterministic core. Used verbatim in-process. (the proven devnet 4-path
  matcher.)
- The deterministic state-transition logic of `dex/pkg/dchain`: `applyTx`/`applyPlace`
  /`applyCancel`/`applySubmit` (`execute.go`), the settlement value moves
  (`settleOrderEffects`, `decrementMakerReserves`, `lockOrderSpend` in `block.go`),
  the book-rebuild-from-rows discipline (`vm.go:rebuildAllBooks`), and `ExecutionRoot`
  determinism. These are PROMOTED to a shared core library and called by the cEVM
  module. Matcher-at-Verify (`block.go:161`) IS the model; it just runs in the cEVM
  block now.
- The 0x9999 V4 ABI surface and PoolManager facade (`precompile/dex/pool_manager.go`,
  `engine.go`, `pool_manager_test.go`, the V4 PoolKey/SwapParams/BalanceDelta types,
  events, pause/freeze, Quoter/StateView/PositionManager modules). The ABI is
  invariant; web/mobile keep pointing at 0x9999.
- The vault accounting + conservation pots: `seamReserve`, `committedPositions`,
  `settleVault`, `makerLockedVault`, the per-asset vault invariant, the rail tags
  (railSwap/railLP), the observed-delta lock/credit helpers (`lockIntentInput`,
  `creditSettlementOutput`, fee-on-transfer safety). These remain the C-side balance
  model; the synchronous path debits/credits them in one call.
- The taker-authenticated MEV floor `enforceProceedsPriceFloor` + `priceLimitToCLOB`
  (`native_dchain_client.go:734-798`) — applied inline at §3.2 step 7.
- The ERC-20 vault env fix and `module_erc20.go` rail — unchanged; the synchronous
  credit uses the same `safeTransferTokenFrom/To`.
- The replay-idempotency discipline conceptually (`swapBindKey`/`storeSwapBinding`,
  `pool_manager.go:121-194`): in the synchronous single-block model a swap is one
  atomic tx so a partial double-fill is impossible, but the durable (txHash, poolId,
  params) bind still protects against the EVM's repeated executions of one tx
  (gas-estimate/validate/build/verify); keep it.
- The DEXFill 0x9999 event + indexer pipeline already designed in MEMORY
  (`fix/native-dex-indexing`): the synchronous path is what finally makes the fill
  appear in the SAME block, satisfying it.

DEPRECATED for normal swaps (removed from the money path; keep ONLY for genuine
cross-chain settlement, if any):
- The async two-phase intent/settle: `SubmitSwapIntent`, `ImportSettlement`,
  `SubmitPositionCommit`, `ImportPositionCollect`, `ReclaimIntent`
  (`native_dchain_client.go`). For a normal same-domain swap these are replaced by the
  single synchronous call. (If a true cross-CHAIN settlement to a DIFFERENT network is
  ever needed, this atomic import/export machinery is the right tool and stays — but
  it is NOT the spot-swap path.)
- The cross-process atomic shared-memory staging for swaps: `native_staging.go`'s
  Put/Remove staging + the host flush `evm/plugin/evm/vm.go:stageDexAtomic` for the
  DEX rails. The combined in-trie commit makes it unnecessary for swaps. (The flush
  hook may remain for other atomic users; it is no longer the DEX path.)
- The ZAP relay as a consensus participant: `chains/dexvm/relay.go` carry-fills, the
  `dexZapEndpoint` wiring, the standalone `dchain-venue`/`cmd/dvenue` as a production
  trade venue. ZAP survives only as an off-consensus read/quote/UI pipe and a test
  harness (§3.4).
- The standalone D-Chain as a separately-finalized production chain (`chains/dexvm`
  proxy + the `dexvm` OptionalVM consensus instance for trading). Retired in favor of
  the in-process module. (`chains/dexvm` was already the PROXY, not the matcher — per
  MEMORY it never matched; it relayed. Removing it from the swap path loses nothing
  the matcher provides.)
- DFillReceipt / BLS attestation layer for swap settlement: not built into the money
  path and not needed — the synchronous in-block fill is its own proof (every
  validator re-derived it). `precompile/dex/_deprecated_bls/` stays deprecated.

Explicitly nothing in the proven matcher or the V4 ABI is discarded. What is removed
is the cross-process plumbing that existed ONLY to bridge two consensus domains that
this design collapses into one.

---

## 6. Phased plan (assumes BFT quorum finality already wired)

Repos: `lux/dex` (matcher + state core), `lux/precompile` (0x9999 module),
`lux/evm` (cEVM block integration + host wiring), `lux/node` (chain topology / VM
registration), `lux/exchange` + `lux/graph` (indexer/UI, already scoped in MEMORY).
LP/PositionManager stays gated until the swap path is proven (per directive).

### Phase 0 — Promote the deterministic core to a shared library (lux/dex)
- Factor the deterministic execute/state/settlement logic out of `pkg/dchain`
  (`execute.go`, the settle helpers in `block.go`, `ExecutionRoot`, the
  rebuild-from-rows logic) into a caller-agnostic core package that takes a generic
  KV/StateDB-shaped store instead of `versiondb` directly. `pkg/lx` is already
  caller-agnostic; leave it.
- Keep the standalone `pkg/dchain` VM compiling against the new core (test-only home).
- Effort: M. Risk: LOW (pure refactor; the existing dchain tests are the safety net —
  conservation/ownership/settlement-identity suites must stay green).

### Phase 1 — In-process synchronous swap module (lux/precompile)
- Add `V4SwapEnvelope` (§3.1) + a synchronous `swap` path in the 0x9999 module that:
  decodes the envelope, loads the book from 0x9999 storage rows, runs
  `lx.SubmitMarketable`, applies D writes to 0x9999 storage + C balance moves to the
  vault pots, emits V4 `Swap` + `DEXFill`, returns BalanceDelta — all in one `Run`.
- Wire it as a real `Engine` implementation (the in-process matcher engine) that
  REPLACES `NativeDChainClient` on the swap selector. Keep `NativeDChainClient`
  compiled only behind the cross-chain-settlement seam (or delete from the swap
  dispatch).
- Storage layout: book/order/position rows under `settleStateNamespace`; reuse the
  vault pots + invariant unchanged.
- Effort: L (this is the bulk). Risk: MED — correctness of the in-trie book rebuild +
  the maker-row rewrite must byte-match the matcher's view. Mitigation: port the
  `dchain` conservation/ownership/settlement-identity tests onto the 0x9999 storage
  backend (they are the exact invariants).

### Phase 2 — cEVM block integration (lux/evm)
- Route 0x9999.swap through the in-process module on the C-Chain EVM. Confirm it runs
  inside `BuildBlock` and `Block.Verify` with no ZAP/shared-memory dependency. Ensure
  the matcher reads block context (height/time) via `state.GetBlockContext()` only.
- Decommission `stageDexAtomic` for the DEX swap rails (leave the hook if other
  atomic users exist; remove DEX Put/Remove staging from the swap path).
- Effort: M. Risk: MED — must prove the EVM state root commits the D rows (re-org /
  restart re-derivation). The 8 tests in §4 gate this phase.

### Phase 3 — Topology cleanup (lux/node)
- Stop scheduling the standalone `dexvm` consensus instance for production trading
  (remove from the active chain set / `--dex-validator` activation for the trade
  path). Keep the VM registrable for local/test only. No new chain; C-Chain now owns
  the DEX.
- Effort: S. Risk: LOW-MED (deployment/manifest change; coordinate with the operator
  repos per the standing CI/CD rule — no hand-built images).

### Phase 4 — Indexer + UI cutover (lux/graph, lux/exchange)
- Already designed (`fix/native-dex-indexing`, MEMORY): index `DEXFill`@0x9999 +
  the V4 `Swap` event from the C-Chain; point `exchange-api` at the native dex/amm
  subgraph; retire the dead Uniswap-gateway proxy. The synchronous block makes the
  fill present in-block, which is the data this pipeline was waiting on.
- Effort: M. Risk: LOW (read-side; the chain is the source of truth).

### Phase 5 — Gate-lift LP / PositionManager (after swap is proven)
- Only after Phases 1–4 are green on devnet+testnet, port `ModifyLiquidity` /
  PositionManager (0x9996) onto the same synchronous in-process model (LP commit/
  collect become in-block balanced moves on `committedPositions`, not C→D commit
  objects). Until then, LP stays on its current path or is paused.
- Effort: L. Risk: MED. Explicitly OUT of the first cut.

Benchmark/exit matrix (gate to ship): all 8 tests in §4 green on a k>1 cluster;
devnet 4-path proof re-run with 0x9999 synchronous fills (conservation OK); a swap's
`DEXFill` visible to the indexer in the SAME block it was mined; revert-rollback and
Byzantine-fake-fill tests green; no `atomic.SharedMemory.Apply` on the swap path
(grep-clean).

---

## Appendix — the one-paragraph summary for the next agent

Lux's C (cEVM, `evm`) and D (dexvm, `dex/pkg/dchain`) are today SEPARATE OS-process
plugins with SEPARATE per-chain consensus engines (`node/node/vms.go:117`,
`node/chains/manager.go:1223`), bridged by async `atomic.SharedMemory` intent/settle
(0x9999 `NativeDChainClient`) precisely because a synchronous live query across that
boundary forks consensus (`precompile/dex/engine.go:20-27`,
`native_dchain_client.go:918`). The one-consensus fix is to collapse them: make
the proven, pure-Go matcher (`dex/pkg/lx`) + the deterministic state logic of
`pkg/dchain` an IN-PROCESS module the cEVM 0x9999 precompile calls directly, with all
D book/balance/position state living in the cEVM state trie under the 0x9999
namespace, so one cEVM block commits BOTH surfaces atomically under ONE consensus
engine (the matcher-at-Verify discipline of `dchain/block.go:161` re-expressed as the
EVM state root). This deletes shared-memory staging, intents, the keeper, ZAP-in-
consensus, and the standalone D-Chain from the normal swap path; it keeps the matcher,
the V4 ABI, the vault pots, and the MEV floor unchanged. Assumes the separate BFT
quorum-finality fix (no proposer self-finalize, `engine.go:891-938`) is wired.
```
