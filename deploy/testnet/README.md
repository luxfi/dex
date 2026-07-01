# dexd multi-node testnet — evo (amd64) + spark (arm64)

Run a fleet of `dexd run` D-Chain DEX venues across two boxes and drive
cross-node, cross-arch ZAP order flow through them. `git pull` on each box, set
two IPs, run one command.

- **What this actually is** (independent venues, not a shared validator set; how
  genesis/peers work; the path to real BFT): **read [topology.md](topology.md)
  first.** It is short and it changes how you read the numbers.
- One node = one self-contained venue: own `zapdb`, own consensus sealer, one ZAP
  socket. N per box. Same pure-Go CPU image on both arches; the frozen big-endian
  ZAP wire is byte-identical amd64⇄arm64, which is what makes the fleet sound.

## Prerequisites (each box)

- Docker + Docker Compose v2 (`docker compose version`).
- This repo checked out (`git pull` to the testnet branch).
- The 2.5 GbE LAN reachable between the boxes; open TCP `BASE_PORT .. BASE_PORT+N-1`
  (default `9099..`) on each box's firewall.

## 1. Configure (each box)

```bash
cd deploy/testnet
cp .env.example .env
$EDITOR .env          # set N, and EVO_IP + SPARK_IP (the LAN IPs of the boxes)
```

Only `EVO_IP` / `SPARK_IP` are values **only you have**. Everything else has a
working default.

## 2. Bring up the venues (run on BOTH boxes)

```bash
# evo (amd64) and spark (arm64) — identical commands
make up N=8            # first run builds the image (native arch), then starts 8 venues
make ps                # all should be (healthy)
```

`make up`:
1. generates `compose.gen.yml` for N venues (ports `BASE_PORT..BASE_PORT+N-1`),
2. builds the `dexd-testnet` image from this repo if it is missing (pure-Go, native
   arch — works on amd64 and arm64),
3. starts the venues with per-node data volumes, resource caps, and a TCP
   healthcheck.

Each box now serves N venues on its LAN IP, e.g. `EVO_IP:9099 … EVO_IP:9106`.

> Private-module fetch: the build needs the luxfi/\* modules. If your box has them
> cached/proxied it builds tokenless; for a cold fetch export a GitHub token first:
> `export GH_TOKEN=ghp_…` then `make build`.

## 3. Benchmark across the boxes (run on ONE box)

```bash
make bench                          # uses BENCH_* from .env, all boxes' venues
# or override inline:
make bench BENCH_OP=cross BENCH_CONNS=64 BENCH_DURATION=60s
```

This runs `dexbench` (in the image, `--network host` so it reaches both boxes over
the LAN) against every venue on every configured box. It opens `-conns` signed
writer connections per venue, each its own account sending strictly-sequential
signed orders, and measures the full mempool → BuildBlock → Verify → Accept round
trip over the wire.

**Ops:**
- `place` (default) — rest a signed limit order. Clean, sustainable consensus-
  write unit; never depletes.
- `cross` — seed resting liquidity then submit marketable orders that cross it;
  exercises the matcher + settlement (reports fills). Best with the GPU build.

### Reading the output

```
=== dexbench RESULT ===
  op          : place
  endpoints   : 16  (evo + spark, 8 each)
  workers     : 512            # endpoints * conns
  ops (acked) : 240183         # consensus-accepted orders in the window
  errors      : 0              # >0 means rejects/timeouts — investigate
  throughput  : 8006 ops/sec   # aggregate across all venues
  latency (µs): p50=… p99=… p999=… max=…   # full wire+consensus round trip
```

- **throughput** is aggregate consensus-write rate across the fleet. It scales
  with the number of venues until a box's disk (commit fsync) or cores saturate —
  that ceiling is what you're hunting by going 3→11 nodes/box.
- **latency** is end-to-end (sign → wire → mempool → block accept → reply). With
  many closed-loop workers it becomes queue-dominated (≈ workers ÷ throughput,
  Little's law) — that's expected. For *unloaded* per-op latency use a small
  `-conns` (e.g. `make bench BENCH_CONNS=2`); the floor is ~one sealer commit.
- **errors must be 0.** Non-zero means rejects (bad wire/auth — shouldn't happen)
  or call timeouts (an overloaded/unreachable venue).

To compare arches, run the bench against each box alone and read the per-box
throughput:
```bash
make bench EVO_IP=$EVO_IP SPARK_IP=          # evo only
make bench EVO_IP= SPARK_IP=$SPARK_IP        # spark only
```

## 4. Cross-arch determinism proof (run on ONE box)

```bash
make verify                  # needs EVO_IP and SPARK_IP set
```

Drives the **identical** deterministic order sequence into one venue on evo and
one on spark and asserts **byte-identical** consensus output (order ids + fills).
A `PASS` is the live amd64⇄arm64 determinism proof — the same order bytes produce
the same fills on both arches. Expected output:

```
=== dexbench VERIFY ===
  submit fills @<evo>:9099 : 3 fill(s)
    fill 0: 10.0000 @ 100.0000 (takerSide=0)
    fill 1: 5.0000 @ 101.0000 (takerSide=0)
    fill 2: 3.0000 @ 102.0000 (takerSide=0)
  RESULT : PASS — byte-identical consensus output across all endpoints
```

## Sizing for a 128 GB box

Defaults are sized so up to ~11 venues fit a 128 GB box. Rules:

- **Node count:** `N ≤ min(11, physical_cores / 2)`. Each single-node sealer is
  commit-bound, so beyond ~1 venue per 2 cores you contend on disk/CPU rather than
  add throughput. Sweep N (3 → 7 → 11) and watch `make bench` throughput plateau —
  that plateau is your box's ceiling.
- **Memory:** `NODE_MEM × N ≤ RAM − 16 GB` (leave headroom for OS, page cache,
  the bench). Default `NODE_MEM=6g` ⇒ 11×6 = 66 GB, comfortable on 128 GB. Raise
  per-node memory and lower N for deep-book / `cross` runs (resting orders live in
  RAM + zapdb).
- **CPUs:** `NODE_CPUS` default 4.0. With N×4 > cores you oversubscribe (fine for a
  testnet; CPU shares). For clean per-node numbers keep N×NODE_CPUS ≤ cores.
- **Disk:** all N venues fsync to the same disk; aggregate throughput is often
  disk-commit-bound. Put the Docker volumes on NVMe. The "128 GB GPU RAM" helps
  only the GPU matcher (below), not the commit path.

Edit `.env` (`N`, `NODE_MEM`, `NODE_CPUS`) and `make up` again to re-shape.

## Ports & metrics (be precise)

`dexd run` exposes **exactly one** TCP port per node — the ZAP socket. The whole
`dex_*` surface (writes `ensure_market/place/cancel/submit/open_market/deposit/
withdraw`; reads `dex_depth/dex_balance`) multiplexes over it. There is **no**
separate http/ws/grpc port and **no built-in Prometheus endpoint** in this binary.

So node metrics are intentionally not wired here. If you need them, the honest
options (out of scope for this harness): poll `dex_depth`/`dex_balance` from a
sidecar and export, or add a metrics listener to the binary upstream. The bench
is the measurement surface for this testnet.

## GPU (spark, opt-in)

This testnet uses the **CPU** image (CGO=0) on both boxes — portable, and the
correctness/wire are identical to the GPU build. The GPU matcher (CUDA) is
**arm64-only** and lives in `../../Dockerfile.dvenue` (spark's NVIDIA GB10);
it needs the private luxcpp dependency tarball and the nvidia container runtime.
It accelerates the `Verify` match step (`pkg/lx`), most visible under `-op cross`
against deep books — not the commit path. To use it, build `Dockerfile.dvenue` on
spark, set `DEX_IMAGE` to it, and add the nvidia runtime to the venue services.
For the raw matcher ceiling (no consensus/disk) use `make bench` in the repo root.

## Overlay (NAT / multi-site)

On the direct 2.5 GbE link you do **not** need an overlay — the bench dials box
IPs directly. If the boxes are across NAT, wire the hanzozt (OpenZiti fork)
tunneler: see [overlay/README.md](overlay/README.md). It adds mTLS + a fabric hop,
so report overlay numbers separately from the direct-LAN baseline.

## Teardown

```bash
make down     # stop venues, keep data
make clean    # stop + delete data volumes + compose.gen.yml
```

## Owner-supplied values (the only blanks)

| value | where | needed for |
|---|---|---|
| `EVO_IP`, `SPARK_IP` | `.env` | cross-box `make bench` / `make verify` |
| `GH_TOKEN` (optional) | env | cold private-module build |
| hanzozt controller + enroll JWTs | overlay | overlay path only |
| `DEX_IMAGE` (optional) | `.env` | using a CI-published image instead of `make build` |

## Validated locally before shipping

The image binaries and bench were exercised end-to-end on a pure-Go venue:
- signed `place`/`cross` load: 0 rejects, `cross` = 1 fill/submit (matcher crosses);
- `-verify` across two independent venues: **PASS**, byte-identical 3-fill sweep.

## Troubleshooting

- `make ps` shows a venue unhealthy → `make logs`; look for the `DVENUE_READY` line.
- `address already in use` → another process holds `BASE_PORT+i`; change `BASE_PORT`.
- bench `errors > 0` with timeouts → box overloaded (lower N or `-conns`) or a
  venue unreachable (firewall on `BASE_PORT..`).
- Building the binary **natively** (not via Docker) inside `~/work/lux` → set
  `GOWORK=off` (the workspace `go.work` pulls unrelated modules). The Docker build
  is unaffected.
