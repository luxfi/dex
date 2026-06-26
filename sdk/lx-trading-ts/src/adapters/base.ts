/**
 * Base adapter interface.
 */

import { Decimal } from 'decimal.js';
import type { Orderbook } from '../orderbook.js';
import type {
  Balance,
  LiquidityResult,
  LpPosition,
  MarketInfo,
  Order,
  OrderRequest,
  PoolInfo,
  SwapQuote,
  Ticker,
  Trade,
  VenueInfo,
  VenueType,
} from '../types.js';

// =============================================================================
// Venue Capabilities
// =============================================================================

export interface VenueCapabilities {
  limitOrders: boolean;
  marketOrders: boolean;
  stopOrders: boolean;
  postOnly: boolean;
  cancelOrders: boolean;
  batchOrders: boolean;
  streaming: boolean;
  orderbook: boolean;
  trades: boolean;
  ammSwap: boolean;
  addLiquidity: boolean;
  removeLiquidity: boolean;
  lpPositions: boolean;
  maxBatchSize: number;
  supportedPairs: Set<string>;
}

export function orderBookCapabilities(): VenueCapabilities {
  return {
    limitOrders: true,
    marketOrders: true,
    stopOrders: true,
    postOnly: true,
    cancelOrders: true,
    batchOrders: true,
    streaming: true,
    orderbook: true,
    trades: true,
    ammSwap: false,
    addLiquidity: false,
    removeLiquidity: false,
    lpPositions: false,
    maxBatchSize: 10,
    supportedPairs: new Set(),
  };
}

export function ammCapabilities(): VenueCapabilities {
  return {
    limitOrders: false,
    marketOrders: true,
    stopOrders: false,
    postOnly: false,
    cancelOrders: false,
    batchOrders: false,
    streaming: true,
    orderbook: false,
    trades: true,
    ammSwap: true,
    addLiquidity: true,
    removeLiquidity: true,
    lpPositions: true,
    maxBatchSize: 1,
    supportedPairs: new Set(),
  };
}

// =============================================================================
// Venue Adapter Interface
// =============================================================================

export interface VenueAdapter {
  // Properties
  readonly name: string;
  readonly venueType: VenueType;
  readonly capabilities: VenueCapabilities;
  readonly isConnected: boolean;
  readonly latencyMs: number | undefined;

  // Info
  info(): VenueInfo;

  // Connection
  connect(): Promise<void>;
  disconnect(): Promise<void>;

  // Market data
  getMarkets(): Promise<MarketInfo[]>;
  getTicker(symbol: string): Promise<Ticker>;
  getOrderbook(symbol: string, depth?: number): Promise<Orderbook>;
  getTrades(symbol: string, limit?: number): Promise<Trade[]>;

  // Account
  getBalances(): Promise<Balance[]>;
  getBalance(asset: string): Promise<Balance>;
  getOpenOrders(symbol?: string): Promise<Order[]>;

  // Orders
  placeOrder(request: OrderRequest): Promise<Order>;
  cancelOrder(orderId: string, symbol: string): Promise<Order>;
  cancelAllOrders(symbol?: string): Promise<Order[]>;

  // AMM (optional)
  getSwapQuote?(
    baseToken: string,
    quoteToken: string,
    amount: Decimal,
    isBuy: boolean,
  ): Promise<SwapQuote>;

  executeSwap?(
    baseToken: string,
    quoteToken: string,
    amount: Decimal,
    isBuy: boolean,
    slippage: Decimal,
  ): Promise<Trade>;

  getPoolInfo?(baseToken: string, quoteToken: string): Promise<PoolInfo>;

  addLiquidity?(
    baseToken: string,
    quoteToken: string,
    baseAmount: Decimal,
    quoteAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult>;

  removeLiquidity?(
    poolAddress: string,
    liquidityAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult>;

  getLpPositions?(): Promise<LpPosition[]>;
}

// =============================================================================
// Base Adapter Implementation
// =============================================================================

export abstract class BaseAdapter implements VenueAdapter {
  abstract readonly name: string;
  abstract readonly venueType: VenueType;
  abstract readonly capabilities: VenueCapabilities;

  protected _connected = false;
  protected _latencyMs: number | undefined;

  get isConnected(): boolean {
    return this._connected;
  }

  get latencyMs(): number | undefined {
    return this._latencyMs;
  }

  info(): VenueInfo {
    return {
      name: this.name,
      venueType: this.venueType,
      connected: this.isConnected,
      latencyMs: this.latencyMs,
      supportedPairs: Array.from(this.capabilities.supportedPairs),
      makerFee: new Decimal('0.001'),
      takerFee: new Decimal('0.002'),
    };
  }

  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  abstract getMarkets(): Promise<MarketInfo[]>;
  abstract getTicker(symbol: string): Promise<Ticker>;
  abstract getOrderbook(symbol: string, depth?: number): Promise<Orderbook>;
  abstract getTrades(symbol: string, limit?: number): Promise<Trade[]>;
  abstract getBalances(): Promise<Balance[]>;
  abstract getBalance(asset: string): Promise<Balance>;
  abstract getOpenOrders(symbol?: string): Promise<Order[]>;
  abstract placeOrder(request: OrderRequest): Promise<Order>;
  abstract cancelOrder(orderId: string, symbol: string): Promise<Order>;
  abstract cancelAllOrders(symbol?: string): Promise<Order[]>;

  // Default AMM implementations throw
  async getSwapQuote(
    _baseToken: string,
    _quoteToken: string,
    _amount: Decimal,
    _isBuy: boolean,
  ): Promise<SwapQuote> {
    throw new Error('AMM swap not supported');
  }

  async executeSwap(
    _baseToken: string,
    _quoteToken: string,
    _amount: Decimal,
    _isBuy: boolean,
    _slippage: Decimal,
  ): Promise<Trade> {
    throw new Error('AMM swap not supported');
  }

  async getPoolInfo(_baseToken: string, _quoteToken: string): Promise<PoolInfo> {
    throw new Error('Pool info not supported');
  }

  async addLiquidity(
    _baseToken: string,
    _quoteToken: string,
    _baseAmount: Decimal,
    _quoteAmount: Decimal,
    _slippage: Decimal,
  ): Promise<LiquidityResult> {
    throw new Error('Add liquidity not supported');
  }

  async removeLiquidity(
    _poolAddress: string,
    _liquidityAmount: Decimal,
    _slippage: Decimal,
  ): Promise<LiquidityResult> {
    throw new Error('Remove liquidity not supported');
  }

  async getLpPositions(): Promise<LpPosition[]> {
    throw new Error('LP positions not supported');
  }
}
