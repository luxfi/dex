// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// ZAP path — the binary OrderBook wire (`pkg/zapwire`).
//
// ZAP is the low-latency off-chain twin of the on-chain settlement path: a fixed,
// frozen byte layout the matcher and the d-chain ledger exchange. Frame sizes
// here mirror the engine's frozen wire test (pkg/zapwire/zapwire_test.go) and
// MUST NOT drift. RPC method names mirror pkg/zapwire constants.
//
// Clean-room: the layout is the Lux engine's own frozen protocol; nothing here
// derives from any third party.

import type { DexClient, Fill, OrderAck, OrderRequest, Hex } from '../types.js'

/** Frozen frame sizes, in bytes (see pkg/zapwire/zapwire_test.go). */
export const WIRE = {
  PoolID: 32,
  User: 16,
  Asset: 32,
  Ref: 32,
  EnsureMarketReq: 32,
  /** PoolID + side(1) + price(8) + size(8) + User = 65. */
  PlaceReq: 65,
  /** PoolID + orderId(8) = 40. */
  CancelReq: 40,
  /** PoolID + side(1) + type(1) + price(8) + size(8) + User = 66. */
  SubmitReq: 66,
  Ack: 24,
  Fill: 17,
  BalanceReq: 48,
  BalanceResp: 9,
} as const

/** ZAP RPC method names (pkg/zapwire). */
export const METHOD = {
  EnsureMarket: 'orderBook_ensure_market',
  Place: 'orderBook_place',
  Cancel: 'orderBook_cancel',
  Submit: 'orderBook_submit',
  Deposit: 'orderBook_deposit',
  Withdraw: 'orderBook_withdraw',
} as const

export type ZapMethod = (typeof METHOD)[keyof typeof METHOD]

/**
 * Transport that ships a method + a single frozen-layout frame and returns the
 * raw response bytes. Implemented over TCP, WebSocket-binary, or in-process.
 */
export interface ZapTransport {
  call(method: ZapMethod, frame: Uint8Array): Promise<Uint8Array>
}

/**
 * Binary OrderBook client. Encodes requests to the frozen wire, decodes responses.
 * The encode/decode bodies are intentionally left for the implementation pass;
 * this scaffold pins the contract, sizes, and method names.
 */
export class ZapClient implements DexClient {
  constructor(private readonly transport: ZapTransport) {}

  async submit(order: OrderRequest): Promise<readonly Fill[]> {
    const frame = encodeSubmit(order)
    const resp = await this.transport.call(METHOD.Submit, frame)
    return decodeFills(resp)
  }

  async place(order: OrderRequest): Promise<OrderAck> {
    const frame = encodePlace(order)
    const resp = await this.transport.call(METHOD.Place, frame)
    return decodeAck(resp)
  }

  async cancel(market: Hex, orderId: bigint): Promise<OrderAck> {
    const frame = encodeCancel(market, orderId)
    const resp = await this.transport.call(METHOD.Cancel, frame)
    return decodeAck(resp)
  }
}

// --- codec stubs (frozen-layout encode/decode; bodies land in impl pass) ----

/** Encode a marketable order to the 66-byte Submit frame. */
export function encodeSubmit(_order: OrderRequest): Uint8Array {
  return new Uint8Array(WIRE.SubmitReq)
}

/** Encode a resting order to the 65-byte Place frame. */
export function encodePlace(_order: OrderRequest): Uint8Array {
  return new Uint8Array(WIRE.PlaceReq)
}

/** Encode a cancel to the 40-byte Cancel frame. */
export function encodeCancel(_market: Hex, _orderId: bigint): Uint8Array {
  return new Uint8Array(WIRE.CancelReq)
}

/** Decode a 24-byte Ack frame. */
export function decodeAck(_frame: Uint8Array): OrderAck {
  throw new Error('not implemented')
}

/** Decode a sequence of 17-byte Fill frames. */
export function decodeFills(_frame: Uint8Array): readonly Fill[] {
  throw new Error('not implemented')
}
