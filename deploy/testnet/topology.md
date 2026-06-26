# dexd testnet — consensus & node topology

Read this before drawing conclusions from the numbers. What `dexd run` does, and
what it does NOT do, decides what "multi-node" means here.

## What one `dexd run` node is

`dexd run` (alias `standalone`) is a **self-contained single-node venue**
(`internal/standalone/standalone.go`). On start it:

1. opens its own on-disk `zapdb` (`-db`) — authoritative chainstate,
2. serves the FROZEN `dex_*` ZAP surface on ONE TCP socket (`-addr`),
3. runs the consensus sealer loop **itself**:
   `WaitForEvent → BuildBlock → Verify → Accept` (`standalone.go:sealer`).

Every write (`dex_place/cancel/submit/deposit/withdraw`) is queued in a mempool,
drained into a block, matched at `Verify` against a `versiondb` overlay, and
committed at `Accept`. The bytes a caller gets back are consensus-computed fills,
not a synchronous book mutation.

The VM has **no peer-to-peer layer**: `VM.Connected/Disconnected` are no-ops
(`pkg/dchain/vm.go`); there is no gossip, no AppGossip, no validator exchange.

## What N nodes are: a FLEET of independent venues

So `N × dexd run` on a box, `2N` across two boxes, is **N independent venues** —
each with its own zapdb and its own single-node sealer. **They do not gossip and
do not share a validator set.** This is deliberate and is exactly the right
substrate for the goal (cross-arch + cross-node ZAP throughput/determinism), but
do not mistake it for BFT replication.

### Genesis — every node is genesis, identically

There is nothing to coordinate. `VM.genesisBlock()` is a **pure function**
(height 0, all-zero parent, `ExecutionRoot` of empty state) — so every node, on
every box, on amd64 and arm64, bootstraps a **byte-identical genesis block id**
on a fresh `-db`. No genesis file, no bootstrap beacons, no "who's first".

### Bootstrap peers — none

Nodes never dial each other. The only network edges are **client → venue** (ZAP
order flow) and **operator → venue** (reads). The 2.5 GbE link / hanzozt overlay
carries that client traffic between boxes, not consensus gossip.

### What makes it "cross-node" and "cross-arch"

Two independent properties, both measurable:

1. **Cross-node throughput/latency** — drive ZAP order flow from a client on box
   A into venues on box B over the wire (`dexbench` load mode). Measures the real
   mempool→Accept + network round trip over the 2.5 GbE / overlay.

2. **Cross-arch determinism** — feed the **identical ordered input** to a venue on
   evo (amd64) and a venue on spark (arm64). The VM is deterministic and the ZAP
   wire is a frozen big-endian codec (`pkg/zapwire`), so the consensus output
   (order ids, fills, roots) is **byte-identical** across arches. `dexbench
   -verify` asserts this on the live boxes. (Verified locally: identical 3-fill
   multi-level sweep across two venues.)

## The other mode: real multi-validator BFT (luxd + `dexd plugin`)

`dexd` with **no subcommand** is the luxd rpcchainvm plugin
(`internal/plugin/plugin.go`). Here luxd — not dexd — drives
`BuildBlock/Verify/Accept` inside Snow\* consensus across a **real validator
set**, and the D-Chain is a P-chain-registered chain.

This is a different, heavier bring-up and **the compose in this directory does
NOT do it**. To stand up true BFT you need:

- the venue image `Dockerfile.dvenue` (bundles `luxd` + `dexd` + the EVM plugin;
  arm64+CUDA, spark only),
- a luxd genesis + staking keys per validator,
- `AddValidatorTx` to form the set and (for a sovereign L1) `ConvertNetworkToL1Tx`,
- bootstrap beacon wiring between the luxd nodes.

That path is owned by the operator (`~/work/lux` / the venue StatefulSet), not by
this testnet harness.

| | this testnet (`dexd run`) | BFT venue (`dexd` plugin + luxd) |
|---|---|---|
| consensus | per-node single sealer | Snow\* across validators |
| nodes share state? | no — independent venues | yes — replicated chain |
| peer gossip | none | luxd p2p |
| genesis | deterministic, per-node | luxd genesis + staking |
| image | `dexd-testnet` (CPU, any arch) | `Dockerfile.dvenue` (arm64+CUDA) |
| answers | cross-arch determinism, ZAP throughput | validator BFT, finality |

## Throughput characteristic (so the numbers make sense)

A single-node sealer is **commit-bound**: each block ends in a `zapdb`/badger
commit (an fsync). Per node you will see roughly the disk's durable-commit rate
(order 1–2k blocks/s on NVMe; the 5 ms `WaitForEvent` poll floors per-op
latency). Aggregate testnet throughput scales with the **number of venues** until
the box's disk or cores saturate — which is the point of running 3–11 per box and
measuring. The GPU matcher (spark, opt-in) accelerates the `Verify` match step
(`pkg/lx`), most visible under `-op cross` against deep books, not the commit
path. For the raw matcher ceiling (millions of ops/s, no consensus/disk) use
`make bench` in the repo root (`go test -bench`, `pkg/lx`), a different
measurement.
