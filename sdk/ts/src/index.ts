// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// @luxfi/dex-sdk — clean-room BSD-3-Clause client for the Lux DEX.
//
// One client vocabulary (`DexClient`, `OrderRequest`, `Fill`, ...) over the
// four DEX paths, each its own subpath export:
//
//   ./precompile  V4 on-chain via the settlement precompile 0x9999 (uses MIT
//                 @luxfi/exchange ABIs + viem). On-chain settlement.
//   ./zap         Binary OrderBook wire (pkg/zapwire). Lowest-latency take/place.
//   ./fix         FIXT.1.1 session, FIX.5.0SP2 application (pkg/fix).
//                 Institutional connectivity.
//   ./ws          JSON WebSocket (pkg/api). No server runs this today: nothing
//                 outside pkg/api's own tests constructs the handler, and no
//                 binary in cmd/ imports the package.
//
// No GPL/Uniswap-derived code. Dependencies are MIT (@luxfi/exchange, viem).

export * from './types.js'

export * as precompile from './precompile/index.js'
export * as zap from './zap/index.js'
export * as fix from './fix/index.js'
export * as ws from './ws/index.js'
