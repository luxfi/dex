// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// WebSocket path — the JSON streaming + order-entry surface.
//
// Message envelope and `type` discriminants mirror the engine WS server
// (pkg/api/websocket_server.go): every frame is `{ "type": "...", ... }`.
// Inbound it carries auth, market-data subscriptions, and order lifecycle.
//
// Clean-room: the protocol is the Lux engine's own JSON shape; nothing here is
// third-party derived.

import type {
  DexClient,
  Fill,
  OrderAck,
  OrderBook,
  OrderRequest,
  Hex,
} from '../types.js'

/** Outbound `type` discriminants accepted by the WS server. */
export const OUT = {
  Auth: 'auth',
  Ping: 'ping',
  Subscribe: 'subscribe',
  Unsubscribe: 'unsubscribe',
  PlaceOrder: 'place_order',
  CancelOrder: 'cancel_order',
  ModifyOrder: 'modify_order',
  GetOrders: 'get_orders',
  GetBalances: 'get_balances',
  GetPositions: 'get_positions',
} as const

/** Channels a client can subscribe to. */
export const CHANNEL = {
  OrderBook: 'orderbook',
  Trades: 'trades',
  Ticker: 'ticker',
} as const

export type Channel = (typeof CHANNEL)[keyof typeof CHANNEL]

/** Every WS frame is a JSON object with a `type` field. */
export interface WsFrame {
  type: string
  [key: string]: unknown
}

/** Minimal duplex JSON socket (structurally compatible with `ws`/browser). */
export interface WsTransport {
  send(frame: WsFrame): void
  /** Register a handler for inbound frames; returns an unsubscribe fn. */
  onMessage(handler: (frame: WsFrame) => void): () => void
  close(): void
}

/** Handler invoked for each depth snapshot/update on a subscribed market. */
export type DepthHandler = (book: OrderBook) => void

/**
 * Streaming + order-entry client over the JSON WebSocket. Request/response
 * correlation and frame (de)serialization land in the implementation pass;
 * this scaffold pins the envelope, discriminants, and channels.
 */
export class WsClient implements DexClient {
  constructor(private readonly transport: WsTransport) {}

  /** Authenticate the session with an API key/token. */
  auth(token: string): void {
    this.transport.send({ type: OUT.Auth, token })
  }

  /** Subscribe to a channel for a market; returns an unsubscribe fn. */
  subscribe(channel: Channel, market: Hex, handler: DepthHandler): () => void {
    this.transport.send({ type: OUT.Subscribe, channel, market })
    const off = this.transport.onMessage((frame) => {
      if (frame.type === channel && frame.market === market) {
        handler(frame as unknown as OrderBook)
      }
    })
    return () => {
      this.transport.send({ type: OUT.Unsubscribe, channel, market })
      off()
    }
  }

  submit(_order: OrderRequest): Promise<readonly Fill[]> {
    throw new Error('not implemented')
  }

  place(_order: OrderRequest): Promise<OrderAck> {
    throw new Error('not implemented')
  }

  cancel(_market: Hex, _orderId: bigint): Promise<OrderAck> {
    throw new Error('not implemented')
  }
}
