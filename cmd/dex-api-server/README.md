# dex-api-server — Standalone DEX HTTP API

`dex-api-server` is a single-process daemon that serves a `pkg/lx.OrderBook`
over REST and WebSocket. It is the simplest deployment shape of the LX
matching engine and is what most local-dev workflows reach for first.

## Topology

There is exactly one operating mode: **standalone**.

| Aspect | Behavior |
|--------|----------|
| Orderbook | In-process `pkg/lx.OrderBook`. No replication. |
| Consensus | None. The daemon "mines" orders by calling `AddOrder` synchronously. |
| Persistence | None — the book is rebuilt from scratch on every start. |
| Auth | None (CORS open). Do not expose this binary to the public internet. |

Multi-node consensus deployments use a different binary,
[`cmd/dag-network`](../dag-network/main.go) (build tag `zmqtest`), which
wires `pkg/consensus.LuxDAGOrderBook` and broadcasts vertices over ZeroMQ.
That code path has its own multi-node test in `pkg/consensus/multinode_test.go`.

## Standalone start

```sh
dex-api-server -port 8080
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | HTML index (debug) |
| GET | `/api/orderbook` | Snapshot of bids / asks |
| GET | `/api/trades` | Recent trades |
| GET | `/api/stats` | Market stats |
| POST | `/api/order` | Place a limit or market order |
| GET (Upgrade) | `/ws` | WebSocket for live updates |

## Why no built-in single-node consensus mode

Adding a Quasar / LuxDAG self-loop here would be redundant: the orderbook
inside `dex-api-server` already finalises orders synchronously inside
`OrderBook.AddOrder`, so a single-validator consensus loop on top would only
add latency without altering semantics. The consensus engine
(`pkg/consensus`) is meaningful when there are real peers to vote with —
hence `cmd/dag-network` for that case and `dex-api-server` for the local
dev / single-process case.

If a future use-case requires a single binary that does both, the right
refactor is to extract a `Daemon` struct with a `--peers` flag where
`len(peers) == 0` selects the in-process orderbook and `len(peers) > 0`
selects the consensus-bound one. That work is **not** in this commit —
it would require defining clean interfaces between the two orderbook
implementations and is a separate change.

## Smoke test

```sh
go test -tags=integration ./cmd/dex-api-server/... -count=1 -timeout 60s
```

`standalone_test.go` builds the daemon, starts it on a free port, places a
limit order via `/api/order`, and confirms the orderbook is queryable. It
exercises the HTTP plumbing end-to-end.
