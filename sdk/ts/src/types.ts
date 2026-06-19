// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// Shared vocabulary for the Lux DEX client surface. Clean-room: these are the
// hex/address template-literal aliases used everywhere in EVM tooling plus the
// minimal order model. No third-party code.

/** Lower-case hex string, `0x`-prefixed. Compatible with viem's `Hex`. */
export type Hex = `0x${string}`

/** 20-byte EVM address, `0x`-prefixed. Compatible with viem's `Address`. */
export type Address = `0x${string}`

/** `address(0)` — the native asset sentinel across every Lux chain (= LUX). */
export const NATIVE_ASSET: Address = '0x0000000000000000000000000000000000000000'

/** Order side. Numeric values mirror the engine wire (0 buy, 1 sell). */
export enum Side {
  Buy = 0,
  Sell = 1,
}

/** Order type. Numeric values mirror the engine wire. */
export enum OrderType {
  Limit = 0,
  Market = 1,
  Stop = 2,
  StopLimit = 3,
}

/** Time-in-force. */
export enum TimeInForce {
  GTC = 'GTC',
  IOC = 'IOC',
  FOK = 'FOK',
  Day = 'DAY',
}

/** Lifecycle status of a resting or completed order. */
export enum OrderStatus {
  New = 'new',
  PartiallyFilled = 'partial',
  Filled = 'filled',
  Canceled = 'canceled',
  Rejected = 'rejected',
}

/**
 * A single market on the CLOB. `id` is the 32-byte pool id used by the engine;
 * `base`/`quote` are the human symbols (e.g. `LUX`, `LUSD`).
 */
export interface Market {
  id: Hex
  base: string
  quote: string
}

/** A request to place a resting limit/market order. */
export interface OrderRequest {
  market: Hex
  side: Side
  type: OrderType
  /** Integer price in quote ticks. `0n` for pure market orders. */
  price: bigint
  /** Integer size in base lots. */
  size: bigint
  tif?: TimeInForce
  /** Optional client-supplied idempotency reference (≤32 bytes, hex). */
  clientRef?: Hex
}

/** Engine acknowledgement of an accepted/rejected order. */
export interface OrderAck {
  orderId: bigint
  status: OrderStatus
}

/** A single fill produced by a marketable order. */
export interface Fill {
  orderId: bigint
  price: bigint
  size: bigint
  side: Side
}

/** One side of an order-book depth snapshot: `[price, size]` levels. */
export type DepthLevels = ReadonlyArray<readonly [bigint, bigint]>

/** Order-book depth snapshot. */
export interface OrderBook {
  market: Hex
  bids: DepthLevels
  asks: DepthLevels
}

/**
 * Common shape every path client implements for the marketable-order ("take")
 * flow. Each transport (precompile / ZAP / FIX / WS) provides its own
 * connection semantics on top of this contract.
 */
export interface DexClient {
  /** Submit a marketable order and return the resulting fills. */
  submit(order: OrderRequest): Promise<readonly Fill[]>
  /** Place a resting order; returns the engine ack (order id + status). */
  place(order: OrderRequest): Promise<OrderAck>
  /** Cancel a resting order by engine order id. */
  cancel(market: Hex, orderId: bigint): Promise<OrderAck>
}
