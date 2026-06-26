# @luxfi/dex-sdk

Clean-room **BSD-3-Clause** TypeScript client for the Lux DEX. One client
vocabulary over the four DEX paths:

| Subpath | Path | Engine source | Use |
|---|---|---|---|
| `@luxfi/dex-sdk/precompile` | V4 on-chain | LXPool precompile `0x9010` (LP-9010) | On-chain settlement via the MIT `@luxfi/exchange` ABIs + viem |
| `@luxfi/dex-sdk/zap` | Binary OrderBook wire | `pkg/zapwire` | Lowest-latency take/place |
| `@luxfi/dex-sdk/fix` | FIX 4.4 | `pkg/fix` | Institutional order entry |
| `@luxfi/dex-sdk/ws` | JSON WebSocket | `pkg/api` | Streaming market data + orders |

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
