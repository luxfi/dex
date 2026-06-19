// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// V4 precompile path — the on-chain DEX surface.
//
// The native singleton AMM lives at the LXPool precompile, address 0x9010
// (LP-9010; see ~/work/lux/precompile/registry/registry.go). Pool keys, the
// PoolManager ABI, and the DEX swap-router ABI are provided by the MIT-licensed
// `@luxfi/exchange` package — this client does NOT redefine or copy them. A
// consumer constructs a viem `WalletClient`/`PublicClient`, imports the ABI +
// addresses from `@luxfi/exchange/contracts`, and calls through this thin,
// path-uniform wrapper.
//
// Clean-room: no Uniswap SDK code. The only on-chain encoding used is the ABI
// shipped by `@luxfi/exchange` (MIT) and viem (MIT).

import type { Address, Hex } from '../types.js'

/** Canonical LXPool singleton-AMM precompile address (LP-9010, all chains). */
export const LXPOOL_PRECOMPILE: Address =
  '0x0000000000000000000000000000000000009010'

/**
 * Pool key identifying a V4 pool. Field shape matches the MIT
 * `@luxfi/exchange` `PoolKey` so the two interoperate without conversion.
 */
export interface PoolKey {
  currency0: Address
  currency1: Address
  /** uint24 fee in hundredths of a bip (3000 = 0.30%). */
  fee: number
  /** int24 tick spacing. */
  tickSpacing: number
  /** Hook contract, or `address(0)` for none. */
  hooks: Address
}

/** A single exact-input swap intent against one pool. */
export interface ExactInputSingle {
  poolKey: PoolKey
  /** true: currency0 -> currency1. */
  zeroForOne: boolean
  amountIn: bigint
  amountOutMinimum: bigint
  /** 0n for no price limit. */
  sqrtPriceLimitX96: bigint
  hookData?: Hex
}

/** Minimal viem-shaped contract caller this client needs (structural type). */
export interface PrecompileTransport {
  /** Simulate/quote a swap; returns the expected output amount. */
  quoteExactInputSingle(params: ExactInputSingle): Promise<bigint>
  /** Execute a swap; returns the transaction hash. */
  swapExactInputSingle(params: ExactInputSingle): Promise<Hex>
}

/**
 * On-chain DEX client over the LXPool precompile. Construct with a transport
 * bound to a viem client + the `@luxfi/exchange` ABI at `LXPOOL_PRECOMPILE`.
 */
export class PrecompileClient {
  constructor(private readonly transport: PrecompileTransport) {}

  /** Quote an exact-input swap without sending a transaction. */
  quote(params: ExactInputSingle): Promise<bigint> {
    return this.transport.quoteExactInputSingle(params)
  }

  /** Execute an exact-input swap; resolves with the transaction hash. */
  swap(params: ExactInputSingle): Promise<Hex> {
    return this.transport.swapExactInputSingle(params)
  }
}
