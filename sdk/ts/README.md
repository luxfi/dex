# @luxfi/dex-sdk

Clean-room **BSD-3-Clause** TypeScript client for the Lux DEX. One client
vocabulary over the four DEX paths:

| Subpath | Path | Engine source | Use |
|---|---|---|---|
| `@luxfi/dex-sdk/precompile` | V4 on-chain | Settlement precompile `0x9999` (LP-9999) | On-chain settlement via the MIT `@luxfi/exchange` ABIs + viem |
| `@luxfi/dex-sdk/zap` | Binary OrderBook wire | `pkg/zapwire` | Lowest-latency take/place |
| `@luxfi/dex-sdk/fix` | FIXT.1.1 / FIX.5.0SP2 | `pkg/fix` | Institutional order entry |
| `@luxfi/dex-sdk/ws` | JSON WebSocket | `pkg/api` | No server runs this today — see Status |

## License & provenance

BSD-3-Clause (`Copyright (C) 2019-2026, Lux Industries Inc.`). Written
clean-room: **no Uniswap/GPL-derived code**. Runtime dependencies are MIT
(`@luxfi/exchange`, `viem`). This package is the non-GPL client surface the apps
import; the GPL-3.0 Uniswap-interface forks live in the `luxexchange` org and
are never a dependency of this SDK.

## Status

Scaffold: types + path-uniform client interfaces + frozen wire constants
(grounded in the engine). Codec/transport bodies land in the implementation
pass. Typechecks under TypeScript 5.9 strict.

What each path can reach today, checked against the tree and the chain rather
than assumed:

- **precompile** — live. `eth_getCode` at `0x9999` returns code on Lux mainnet,
  and the address dispatches the V4 `PoolManager` selectors this path encodes.
- **zap** — live. `pkg/zapwire` is the frozen `dex_*` wire `dexd` serves in
  standalone mode, and `cmd/dexbench` drives it.
- **fix** — live. `pkg/fix` speaks FIXT.1.1 at the session layer and FIX.5.0SP2
  (Extension Pack 307) at the application layer.
- **ws** — no server. `pkg/api` compiles and is tested, but nothing constructs
  `NewWebSocketServer` or `NewVenueWS` outside its own tests, and no binary in
  `cmd/` imports the package. Treat this subpath as a client for a listener that
  has not been stood up.
