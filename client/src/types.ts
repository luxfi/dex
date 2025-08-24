/**
 * Lux Exchange Type Definitions
 * Supporting all asset classes: crypto, stocks, commodities, forex
 */

import Decimal from 'decimal.js';

// Asset Types
export enum AssetClass {
  CRYPTO = 'CRYPTO',
  STOCK = 'STOCK',
  COMMODITY = 'COMMODITY',
  FOREX = 'FOREX',
  INDEX = 'INDEX',
  OPTION = 'OPTION',
  FUTURE = 'FUTURE',
  BOND = 'BOND',
  ETF = 'ETF'
}

// Market identifiers
export enum Market {
  SPOT = 'SPOT',
  PERP = 'PERP',       // Perpetual futures
  FUTURES = 'FUTURES',
  OPTIONS = 'OPTIONS'
}

// Order types
export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT',
  TRAILING_STOP = 'TRAILING_STOP',
  ICEBERG = 'ICEBERG',
  POST_ONLY = 'POST_ONLY',
  IOC = 'IOC',  // Immediate or Cancel
  FOK = 'FOK',  // Fill or Kill
  GTX = 'GTX'   // Good Till Crossing
}

export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

export enum OrderStatus {
  NEW = 'NEW',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  FILLED = 'FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED'
}

export enum TimeInForce {
  GTC = 'GTC',  // Good Till Cancel
  IOC = 'IOC',  // Immediate or Cancel
  FOK = 'FOK',  // Fill or Kill
  GTX = 'GTX',  // Good Till Crossing
  DAY = 'DAY',  // Day order
  GTD = 'GTD'   // Good Till Date
}

// Asset definition
export interface Asset {
  symbol: string;
  name: string;
  assetClass: AssetClass;
  exchange?: string;
  baseCurrency?: string;
  quoteCurrency?: string;
  tickSize: Decimal;
  lotSize: Decimal;
  minOrderSize: Decimal;
  maxOrderSize: Decimal;
  makerFee: Decimal;
  takerFee: Decimal;
  marginRequirement?: Decimal;
  tradingHours?: TradingHours;
  metadata?: Record<string, any>;
}

export interface TradingHours {
  timezone: string;
  sessions: TradingSession[];
  holidays?: string[];
}

export interface TradingSession {
  day: string;
  open: string;
  close: string;
  preMarket?: string;
  afterHours?: string;
}

// Order definition
export interface Order {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  assetClass: AssetClass;
  market: Market;
  side: OrderSide;
  type: OrderType;
  price?: Decimal;
  stopPrice?: Decimal;
  quantity: Decimal;
  filledQuantity: Decimal;
  remainingQuantity: Decimal;
  status: OrderStatus;
  timeInForce: TimeInForce;
  postOnly?: boolean;
  reduceOnly?: boolean;
  closePosition?: boolean;
  leverage?: number;
  timestamp: Date;
  lastUpdateTime?: Date;
  fees?: Decimal;
  metadata?: Record<string, any>;
}

// Trade/Fill definition
export interface Trade {
  tradeId: string;
  orderId: string;
  symbol: string;
  side: OrderSide;
  price: Decimal;
  quantity: Decimal;
  fee: Decimal;
  feeCurrency: string;
  timestamp: Date;
  isMaker: boolean;
  metadata?: Record<string, any>;
}

// Order book levels
export interface OrderBookLevel {
  price: Decimal;
  quantity: Decimal;
  orderCount?: number;
}

// Order book snapshot
export interface OrderBookSnapshot {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: Date;
  sequenceNumber?: number;
}

// Market data tick
export interface MarketTick {
  symbol: string;
  bid: Decimal;
  bidSize: Decimal;
  ask: Decimal;
  askSize: Decimal;
  last: Decimal;
  lastSize: Decimal;
  volume24h: Decimal;
  high24h: Decimal;
  low24h: Decimal;
  open24h: Decimal;
  change24h: Decimal;
  changePercent24h: Decimal;
  timestamp: Date;
}

// Position (for margin/futures trading)
export interface Position {
  symbol: string;
  assetClass: AssetClass;
  market: Market;
  side: OrderSide;
  quantity: Decimal;
  entryPrice: Decimal;
  markPrice: Decimal;
  liquidationPrice?: Decimal;
  unrealizedPnl: Decimal;
  realizedPnl: Decimal;
  margin: Decimal;
  leverage: number;
  timestamp: Date;
}

// Account balance
export interface Balance {
  currency: string;
  available: Decimal;
  locked: Decimal;
  total: Decimal;
  usdValue?: Decimal;
}

// Portfolio summary
export interface PortfolioSummary {
  accountId: string;
  balances: Balance[];
  positions: Position[];
  totalValueUsd: Decimal;
  totalPnl: Decimal;
  margin?: Decimal;
  freeMargin?: Decimal;
  marginLevel?: Decimal;
  timestamp: Date;
}

// WebSocket subscription types
export interface Subscription {
  id: string;
  channel: string;
  symbols: string[];
  params?: Record<string, any>;
}

// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: Date;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}

// Client configuration
export interface LXClientConfig {
  apiKey?: string;
  apiSecret?: string;
  endpoint?: string;
  wsEndpoint?: string;
  testnet?: boolean;
  timeout?: number;
  reconnect?: boolean;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  rateLimit?: RateLimitConfig;
}

export interface RateLimitConfig {
  maxRequestsPerSecond: number;
  maxOrdersPerSecond: number;
  burstCapacity: number;
}

// Event types for WebSocket
export enum WSEventType {
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
  ORDER_UPDATE = 'order_update',
  TRADE = 'trade',
  ORDERBOOK = 'orderbook',
  TICKER = 'ticker',
  POSITION_UPDATE = 'position_update',
  BALANCE_UPDATE = 'balance_update'
}

// Kline/Candle data
export interface Kline {
  symbol: string;
  interval: string;
  openTime: Date;
  closeTime: Date;
  open: Decimal;
  high: Decimal;
  low: Decimal;
  close: Decimal;
  volume: Decimal;
  quoteVolume: Decimal;
  trades: number;
  isClosed: boolean;
}