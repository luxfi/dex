// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// FIX path — institutional order entry.
//
// FIXT.1.1 is the session layer (tag 8) and FIX.5.0SP2 with Extension Pack 307
// is the application layer, named on the Logon. The retired FIX.4.4 profile is
// rejected outright: pkg/fix asserts Parse errors on both 8=FIX.4.4 and
// 8=FIX.4.2, so a session opened with either BeginString cannot connect.
//
// Tag numbers and enum values mirror the engine's FIX acceptor
// (pkg/fix/message.go). This client builds NewOrderSingle (35=D) and
// OrderCancelRequest (35=F) messages and parses ExecutionReport (35=8).
//
// Clean-room: FIX is a public standard (FIX Trading Community); the constants
// below are the protocol's own tag/enum numbers, not third-party code.

import type { DexClient, Fill, OrderAck, OrderRequest, Hex } from '../types.js'
import { OrderType, Side, TimeInForce } from '../types.js'

/** FIX tags used by the order-entry surface. */
export const TAG = {
  MsgType: 35,
  Symbol: 55,
  Side: 54,
  OrderQty: 38,
  OrdType: 40,
  Price: 44,
  TimeInForce: 59,
  ClOrdID: 11,
  OrderID: 37,
  OrdStatus: 39,
} as const

/** MsgType (tag 35) values. */
export const MSG_TYPE = {
  Heartbeat: '0',
  Logon: 'A',
  Logout: '5',
  ExecutionReport: '8',
  NewOrderSingle: 'D',
  OrderCancelRequest: 'F',
  OrderCancelReject: '9',
} as const

/** Map SDK Side -> FIX tag-54 value. */
export function fixSide(side: Side): '1' | '2' {
  return side === Side.Buy ? '1' : '2'
}

/** Map SDK OrderType -> FIX tag-40 value. */
export function fixOrdType(type: OrderType): '1' | '2' {
  return type === OrderType.Market ? '1' : '2'
}

/** Map SDK TimeInForce -> FIX tag-59 value (0 day, 3 IOC, 4 FOK, 1 GTC). */
export function fixTimeInForce(tif: TimeInForce): '0' | '1' | '3' | '4' {
  switch (tif) {
    case TimeInForce.Day:
      return '0'
    case TimeInForce.GTC:
      return '1'
    case TimeInForce.IOC:
      return '3'
    case TimeInForce.FOK:
      return '4'
  }
}

/** A decoded FIX message as an ordered tag->value map. */
export type FixMessage = ReadonlyMap<number, string>

/**
 * Session transport: sends an encoded FIX message body and yields inbound
 * messages (ExecutionReports, rejects) for the matching ClOrdID. Implemented
 * over a TCP FIX session (logon/heartbeat handled by the transport).
 */
export interface FixTransport {
  send(message: FixMessage): Promise<void>
  /** Resolve the next inbound message correlated to `clOrdId`. */
  await(clOrdId: string): Promise<FixMessage>
}

/**
 * FIX order-entry client. Builds messages from the shared order model and
 * decodes ExecutionReports back into acks/fills. Wire (de)serialization bodies
 * land in the implementation pass; this scaffold pins the contract + tags.
 */
export class FixClient implements DexClient {
  constructor(private readonly transport: FixTransport) {}

  async submit(order: OrderRequest): Promise<readonly Fill[]> {
    const clOrdId = newClOrdId()
    await this.transport.send(buildNewOrderSingle(order, clOrdId))
    return decodeFills(await this.transport.await(clOrdId))
  }

  async place(order: OrderRequest): Promise<OrderAck> {
    const clOrdId = newClOrdId()
    await this.transport.send(buildNewOrderSingle(order, clOrdId))
    return decodeAck(await this.transport.await(clOrdId))
  }

  async cancel(market: Hex, orderId: bigint): Promise<OrderAck> {
    const clOrdId = newClOrdId()
    await this.transport.send(buildOrderCancelRequest(market, orderId, clOrdId))
    return decodeAck(await this.transport.await(clOrdId))
  }
}

// --- message builders / parsers (bodies land in impl pass) ------------------

/** Build a 35=D NewOrderSingle from the shared order model. */
export function buildNewOrderSingle(
  order: OrderRequest,
  clOrdId: string,
): FixMessage {
  const m = new Map<number, string>()
  m.set(TAG.MsgType, MSG_TYPE.NewOrderSingle)
  m.set(TAG.ClOrdID, clOrdId)
  m.set(TAG.Symbol, order.market)
  m.set(TAG.Side, fixSide(order.side))
  m.set(TAG.OrdType, fixOrdType(order.type))
  m.set(TAG.OrderQty, order.size.toString())
  if (order.type !== OrderType.Market) m.set(TAG.Price, order.price.toString())
  if (order.tif) m.set(TAG.TimeInForce, fixTimeInForce(order.tif))
  return m
}

/** Build a 35=F OrderCancelRequest. */
export function buildOrderCancelRequest(
  market: Hex,
  orderId: bigint,
  clOrdId: string,
): FixMessage {
  const m = new Map<number, string>()
  m.set(TAG.MsgType, MSG_TYPE.OrderCancelRequest)
  m.set(TAG.ClOrdID, clOrdId)
  m.set(TAG.Symbol, market)
  m.set(TAG.OrderID, orderId.toString())
  return m
}

/** Decode an ExecutionReport (35=8) into an ack. */
export function decodeAck(_msg: FixMessage): OrderAck {
  throw new Error('not implemented')
}

/** Decode fills out of one or more ExecutionReports. */
export function decodeFills(_msg: FixMessage): readonly Fill[] {
  throw new Error('not implemented')
}

let clOrdSeq = 0
/** Monotonic client order id for the current session. */
export function newClOrdId(): string {
  clOrdSeq += 1
  return `lx-${clOrdSeq}`
}
