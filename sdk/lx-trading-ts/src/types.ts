/**
 * Core types for LX Trading SDK.
 */

import { Decimal } from 'decimal.js';

// =============================================================================
// Enums
// =============================================================================

export enum Side {
  BUY = 'buy',
  SELL = 'sell',
}

export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit',
  LIMIT_MAKER = 'limit_maker',
  STOP_LOSS = 'stop_loss',
  STOP_LOSS_LIMIT = 'stop_loss_limit',
  TAKE_PROFIT = 'take_profit',
  TAKE_PROFIT_LIMIT = 'take_profit_limit',
}

export enum TimeInForce {
  GTC = 'GTC', // Good till cancelled
  IOC = 'IOC', // Immediate or cancel
  FOK = 'FOK', // Fill or kill
  GTD = 'GTD', // Good till date
  POST_ONLY = 'POST_ONLY', // Maker only
}

export enum OrderStatus {
  PENDING = 'pending',
  OPEN = 'open',
  PARTIALLY_FILLED = 'partially_filled',
  FILLED = 'filled',
  CANCELLED = 'cancelled',
  REJECTED = 'rejected',
  EXPIRED = 'expired',
}

export enum VenueType {
  NATIVE = 'native',
  CCXT = 'ccxt',
  HUMMINGBOT = 'hummingbot',
  CUSTOM = 'custom',
}

// =============================================================================
// Trading Pair
// =============================================================================

export class TradingPair {
  constructor(
    public readonly base: string,
    public readonly quote: string,
  ) {}

  static fromSymbol(symbol: string): TradingPair | null {
    for (const sep of ['-', '/', '_']) {
      if (symbol.includes(sep)) {
        const parts = symbol.split(sep);
        if (parts.length === 2 && parts[0] && parts[1]) {
          return new TradingPair(parts[0], parts[1]);
        }
      }
    }
    return null;
  }

  toHummingbot(): string {
    return `${this.base}-${this.quote}`;
  }

  toCcxt(): string {
    return `${this.base}/${this.quote}`;
  }

  toString(): string {
    return this.toHummingbot();
  }
}

// =============================================================================
// Interfaces
// =============================================================================

export interface Fee {
  asset: string;
  amount: Decimal;
  rate?: Decimal;
}

export interface Balance {
  asset: string;
  venue: string;
  free: Decimal;
  locked: Decimal;
}

export function balanceTotal(b: Balance): Decimal {
  return b.free.plus(b.locked);
}

export interface AggregatedBalance {
  asset: string;
  totalFree: Decimal;
  totalLocked: Decimal;
  byVenue: Balance[];
}

export function aggregatedBalanceTotal(b: AggregatedBalance): Decimal {
  return b.totalFree.plus(b.totalLocked);
}

export interface OrderRequest {
  symbol: string;
  side: Side;
  orderType: OrderType;
  quantity: Decimal;
  price?: Decimal;
  stopPrice?: Decimal;
  timeInForce: TimeInForce;
  reduceOnly: boolean;
  postOnly: boolean;
  venue?: string;
  clientOrderId: string;
}

export function createMarketOrder(
  symbol: string,
  side: Side,
  quantity: Decimal,
): OrderRequest {
  return {
    symbol,
    side,
    orderType: OrderType.MARKET,
    quantity,
    timeInForce: TimeInForce.IOC,
    reduceOnly: false,
    postOnly: false,
    clientOrderId: crypto.randomUUID(),
  };
}

export function createLimitOrder(
  symbol: string,
  side: Side,
  quantity: Decimal,
  price: Decimal,
): OrderRequest {
  return {
    symbol,
    side,
    orderType: OrderType.LIMIT,
    quantity,
    price,
    timeInForce: TimeInForce.GTC,
    reduceOnly: false,
    postOnly: false,
    clientOrderId: crypto.randomUUID(),
  };
}

export interface Order {
  orderId: string;
  clientOrderId: string;
  symbol: string;
  venue: string;
  side: Side;
  orderType: OrderType;
  status: OrderStatus;
  quantity: Decimal;
  filledQuantity: Decimal;
  remainingQuantity: Decimal;
  price?: Decimal;
  averagePrice?: Decimal;
  createdAt: number;
  updatedAt: number;
  fees: Fee[];
}

export function orderIsOpen(o: Order): boolean {
  return [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING].includes(o.status);
}

export function orderIsDone(o: Order): boolean {
  return [
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
  ].includes(o.status);
}

export function orderFillPercent(o: Order): Decimal {
  if (o.quantity.isZero()) {
    return new Decimal(0);
  }
  return o.filledQuantity.div(o.quantity).mul(100);
}

export interface Trade {
  tradeId: string;
  orderId: string;
  symbol: string;
  venue: string;
  side: Side;
  price: Decimal;
  quantity: Decimal;
  fee: Fee;
  timestamp: number;
  isMaker: boolean;
}

export function tradeValue(t: Trade): Decimal {
  return t.price.mul(t.quantity);
}

export interface Ticker {
  symbol: string;
  venue: string;
  bid?: Decimal;
  ask?: Decimal;
  last?: Decimal;
  volume24h?: Decimal;
  high24h?: Decimal;
  low24h?: Decimal;
  change24h?: Decimal;
  timestamp: number;
}

export function tickerMidPrice(t: Ticker): Decimal | undefined {
  if (t.bid && t.ask) {
    return t.bid.plus(t.ask).div(2);
  }
  return t.last;
}

export function tickerSpread(t: Ticker): Decimal | undefined {
  if (t.bid && t.ask) {
    return t.ask.minus(t.bid);
  }
  return undefined;
}

export function tickerSpreadPercent(t: Ticker): Decimal | undefined {
  if (t.bid && t.ask && t.bid.gt(0)) {
    return t.ask.minus(t.bid).div(t.bid).mul(100);
  }
  return undefined;
}

export interface SwapQuote {
  baseToken: string;
  quoteToken: string;
  inputAmount: Decimal;
  outputAmount: Decimal;
  price: Decimal;
  priceImpact: Decimal;
  fee: Decimal;
  route: string[];
  expiresAt: number;
}

export interface PoolInfo {
  address: string;
  baseToken: string;
  quoteToken: string;
  baseReserve: Decimal;
  quoteReserve: Decimal;
  totalLiquidity: Decimal;
  feeRate: Decimal;
  apy?: Decimal;
}

export interface LpPosition {
  poolAddress: string;
  baseToken: string;
  quoteToken: string;
  lpTokens: Decimal;
  baseAmount: Decimal;
  quoteAmount: Decimal;
  sharePercent: Decimal;
  unrealizedPnl?: Decimal;
}

export interface LiquidityResult {
  txHash: string;
  poolAddress: string;
  baseAmount: Decimal;
  quoteAmount: Decimal;
  lpTokens: Decimal;
  sharePercent: Decimal;
}

export interface VenueInfo {
  name: string;
  venueType: VenueType;
  connected: boolean;
  latencyMs?: number;
  supportedPairs: string[];
  makerFee: Decimal;
  takerFee: Decimal;
}

export interface MarketInfo {
  symbol: string;
  base: string;
  quote: string;
  pricePrecision: number;
  quantityPrecision: number;
  minQuantity: Decimal;
  maxQuantity?: Decimal;
  minNotional?: Decimal;
  tickSize: Decimal;
  lotSize: Decimal;
}

export interface PriceLevel {
  price: Decimal;
  quantity: Decimal;
}

export function priceLevelValue(p: PriceLevel): Decimal {
  return p.price.mul(p.quantity);
}
