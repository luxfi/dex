/**
 * Native LX DEX and LX AMM adapters.
 */

import axios, { type AxiosInstance } from 'axios';
import { Decimal } from 'decimal.js';
import type { NativeVenueConfig } from '../config.js';
import { Orderbook } from '../orderbook.js';
import {
  OrderStatus,
  OrderType,
  Side,
  TradingPair,
  VenueType,
  type Balance,
  type LiquidityResult,
  type LpPosition,
  type MarketInfo,
  type Order,
  type OrderRequest,
  type PoolInfo,
  type SwapQuote,
  type Ticker,
  type Trade,
} from '../types.js';
import { BaseAdapter, orderBookCapabilities, ammCapabilities, type VenueCapabilities } from './base.js';

// =============================================================================
// LX DEX Adapter (OrderBook)
// =============================================================================

export class LxDexAdapter extends BaseAdapter {
  readonly venueType = VenueType.NATIVE;
  readonly capabilities: VenueCapabilities;

  private client: AxiosInstance | null = null;

  constructor(
    readonly name: string,
    private readonly config: NativeVenueConfig,
  ) {
    super();
    this.capabilities = orderBookCapabilities();
  }

  async connect(): Promise<void> {
    this.client = axios.create({
      baseURL: this.config.apiUrl,
      timeout: 30000,
    });

    // Test connection
    const start = Date.now();
    await this.client.get('/api/v1/health');
    this._latencyMs = Date.now() - start;
    this._connected = true;
  }

  async disconnect(): Promise<void> {
    this.client = null;
    this._connected = false;
  }

  private async request<T>(method: string, path: string, data?: unknown): Promise<T> {
    if (!this.client) {
      throw new Error('Not connected');
    }

    const headers: Record<string, string> = {};
    if (this.config.apiKey) {
      headers['X-API-KEY'] = this.config.apiKey;
      headers['X-TIMESTAMP'] = String(Date.now());
    }

    const start = Date.now();
    const response = await this.client.request<T>({
      method,
      url: path,
      data,
      headers,
    });
    this._latencyMs = Date.now() - start;

    return response.data;
  }

  async getMarkets(): Promise<MarketInfo[]> {
    const data = await this.request<MarketData[]>('GET', '/api/v1/markets');
    return data.map((m) => ({
      symbol: m.symbol,
      base: m.base,
      quote: m.quote,
      pricePrecision: m.pricePrecision ?? 8,
      quantityPrecision: m.quantityPrecision ?? 8,
      minQuantity: new Decimal(m.minQuantity ?? 0),
      maxQuantity: m.maxQuantity ? new Decimal(m.maxQuantity) : undefined,
      minNotional: m.minNotional ? new Decimal(m.minNotional) : undefined,
      tickSize: new Decimal(m.tickSize ?? '0.00000001'),
      lotSize: new Decimal(m.lotSize ?? '0.00000001'),
    }));
  }

  async getTicker(symbol: string): Promise<Ticker> {
    const data = await this.request<TickerData>('GET', `/api/v1/ticker/${symbol}`);
    return {
      symbol: data.symbol,
      venue: this.name,
      bid: data.bid ? new Decimal(data.bid) : undefined,
      ask: data.ask ? new Decimal(data.ask) : undefined,
      last: data.last ? new Decimal(data.last) : undefined,
      volume24h: data.volume24h ? new Decimal(data.volume24h) : undefined,
      high24h: data.high24h ? new Decimal(data.high24h) : undefined,
      low24h: data.low24h ? new Decimal(data.low24h) : undefined,
      change24h: data.change24h ? new Decimal(data.change24h) : undefined,
      timestamp: data.timestamp ?? 0,
    };
  }

  async getOrderbook(symbol: string, depth?: number): Promise<Orderbook> {
    let path = `/api/v1/orderbook/${symbol}`;
    if (depth) {
      path += `?depth=${depth}`;
    }

    const data = await this.request<OrderbookData>('GET', path);
    const book = new Orderbook(symbol, this.name);

    for (const bid of data.bids ?? []) {
      book.addBid(new Decimal(bid[0]), new Decimal(bid[1]));
    }

    for (const ask of data.asks ?? []) {
      book.addAsk(new Decimal(ask[0]), new Decimal(ask[1]));
    }

    book.sort();
    return book;
  }

  async getTrades(symbol: string, limit?: number): Promise<Trade[]> {
    let path = `/api/v1/trades/${symbol}`;
    if (limit) {
      path += `?limit=${limit}`;
    }

    const data = await this.request<TradeData[]>('GET', path);
    return data.map((t) => ({
      tradeId: t.id,
      orderId: t.orderId ?? '',
      symbol,
      venue: this.name,
      side: t.side === 'buy' ? Side.BUY : Side.SELL,
      price: new Decimal(t.price),
      quantity: new Decimal(t.quantity),
      fee: {
        asset: t.feeAsset ?? '',
        amount: new Decimal(t.feeAmount ?? 0),
      },
      timestamp: t.timestamp,
      isMaker: t.isMaker ?? false,
    }));
  }

  async getBalances(): Promise<Balance[]> {
    const data = await this.request<BalanceData[]>('GET', '/api/v1/account/balances');
    return data.map((b) => ({
      asset: b.asset,
      venue: this.name,
      free: new Decimal(b.free ?? 0),
      locked: new Decimal(b.locked ?? 0),
    }));
  }

  async getBalance(asset: string): Promise<Balance> {
    const data = await this.request<BalanceData>('GET', `/api/v1/account/balance/${asset}`);
    return {
      asset: data.asset,
      venue: this.name,
      free: new Decimal(data.free ?? 0),
      locked: new Decimal(data.locked ?? 0),
    };
  }

  async getOpenOrders(symbol?: string): Promise<Order[]> {
    let path = '/api/v1/orders?status=open';
    if (symbol) {
      path += `&symbol=${symbol}`;
    }

    const data = await this.request<OrderData[]>('GET', path);
    return data.map((o) => this.convertOrder(o));
  }

  async placeOrder(request: OrderRequest): Promise<Order> {
    const body = {
      clientOrderId: request.clientOrderId,
      symbol: request.symbol,
      side: request.side,
      type: request.orderType,
      quantity: request.quantity.toString(),
      timeInForce: request.timeInForce,
      price: request.price?.toString(),
    };

    const data = await this.request<OrderData>('POST', '/api/v1/orders', body);
    return this.convertOrder(data);
  }

  async cancelOrder(orderId: string, symbol: string): Promise<Order> {
    const data = await this.request<OrderData>('DELETE', `/api/v1/orders/${orderId}`, { symbol });
    return this.convertOrder(data);
  }

  async cancelAllOrders(symbol?: string): Promise<Order[]> {
    const body = symbol ? { symbol } : {};
    const data = await this.request<OrderData[]>('DELETE', '/api/v1/orders/all', body);
    return data.map((o) => this.convertOrder(o));
  }

  private convertOrder(o: OrderData): Order {
    const statusMap: Record<string, OrderStatus> = {
      pending: OrderStatus.PENDING,
      open: OrderStatus.OPEN,
      partially_filled: OrderStatus.PARTIALLY_FILLED,
      filled: OrderStatus.FILLED,
      cancelled: OrderStatus.CANCELLED,
      rejected: OrderStatus.REJECTED,
      expired: OrderStatus.EXPIRED,
    };

    const typeMap: Record<string, OrderType> = {
      market: OrderType.MARKET,
      limit: OrderType.LIMIT,
      stop_loss: OrderType.STOP_LOSS,
      stop_loss_limit: OrderType.STOP_LOSS_LIMIT,
    };

    const quantity = new Decimal(o.quantity ?? 0);
    const filled = new Decimal(o.filledQuantity ?? 0);

    return {
      orderId: o.orderId,
      clientOrderId: o.clientOrderId ?? '',
      symbol: o.symbol,
      venue: this.name,
      side: o.side === 'buy' ? Side.BUY : Side.SELL,
      orderType: typeMap[o.type ?? 'limit'] ?? OrderType.LIMIT,
      status: statusMap[o.status ?? 'open'] ?? OrderStatus.OPEN,
      quantity,
      filledQuantity: filled,
      remainingQuantity: quantity.minus(filled),
      price: o.price ? new Decimal(o.price) : undefined,
      averagePrice: o.averagePrice ? new Decimal(o.averagePrice) : undefined,
      createdAt: o.createdAt ?? 0,
      updatedAt: o.updatedAt ?? 0,
      fees: [],
    };
  }
}

// =============================================================================
// LX AMM Adapter
// =============================================================================

export class LxAmmAdapter extends BaseAdapter {
  readonly venueType = VenueType.NATIVE;
  readonly capabilities: VenueCapabilities;

  private client: AxiosInstance | null = null;

  constructor(
    readonly name: string,
    private readonly config: NativeVenueConfig,
  ) {
    super();
    this.capabilities = ammCapabilities();
  }

  async connect(): Promise<void> {
    this.client = axios.create({
      baseURL: this.config.apiUrl,
      timeout: 30000,
    });
    this._connected = true;
  }

  async disconnect(): Promise<void> {
    this.client = null;
    this._connected = false;
  }

  private async request<T>(method: string, path: string, data?: unknown): Promise<T> {
    if (!this.client) {
      throw new Error('Not connected');
    }

    const start = Date.now();
    const response = await this.client.request<T>({
      method,
      url: path,
      data,
    });
    this._latencyMs = Date.now() - start;

    return response.data;
  }

  async getMarkets(): Promise<MarketInfo[]> {
    const data = await this.request<PoolData[]>('GET', '/api/v1/amm/pools');
    return data.map((p) => ({
      symbol: `${p.baseToken}-${p.quoteToken}`,
      base: p.baseToken,
      quote: p.quoteToken,
      pricePrecision: 8,
      quantityPrecision: 8,
      minQuantity: new Decimal(0),
      maxQuantity: undefined,
      minNotional: undefined,
      tickSize: new Decimal('0.00000001'),
      lotSize: new Decimal('0.00000001'),
    }));
  }

  async getTicker(symbol: string): Promise<Ticker> {
    const pair = TradingPair.fromSymbol(symbol);
    if (!pair) {
      throw new Error(`Invalid symbol: ${symbol}`);
    }

    const data = await this.request<PriceData>('GET', `/api/v1/amm/price/${pair.base}/${pair.quote}`);
    const price = data.price ? new Decimal(data.price) : undefined;

    return {
      symbol,
      venue: this.name,
      bid: price,
      ask: price,
      last: price,
      volume24h: data.volume24h ? new Decimal(data.volume24h) : undefined,
      high24h: undefined,
      low24h: undefined,
      change24h: undefined,
      timestamp: Date.now(),
    };
  }

  async getOrderbook(_symbol: string, _depth?: number): Promise<Orderbook> {
    throw new Error('AMM does not have orderbook');
  }

  async getTrades(symbol: string, limit?: number): Promise<Trade[]> {
    const pair = TradingPair.fromSymbol(symbol);
    if (!pair) {
      return [];
    }

    let path = `/api/v1/amm/swaps/${pair.base}/${pair.quote}`;
    if (limit) {
      path += `?limit=${limit}`;
    }

    const data = await this.request<SwapData[]>('GET', path);
    return data.map((t) => ({
      tradeId: t.txHash,
      orderId: t.txHash,
      symbol,
      venue: this.name,
      side: t.side === 'buy' ? Side.BUY : Side.SELL,
      price: new Decimal(t.price),
      quantity: new Decimal(t.amount),
      fee: {
        asset: '',
        amount: new Decimal(t.fee ?? 0),
      },
      timestamp: t.timestamp,
      isMaker: false,
    }));
  }

  async getBalances(): Promise<Balance[]> {
    const data = await this.request<BalanceData[]>('GET', '/api/v1/account/balances');
    return data.map((b) => ({
      asset: b.asset,
      venue: this.name,
      free: new Decimal(b.free),
      locked: new Decimal(b.locked ?? 0),
    }));
  }

  async getBalance(asset: string): Promise<Balance> {
    const data = await this.request<BalanceData>('GET', `/api/v1/account/balance/${asset}`);
    return {
      asset: data.asset,
      venue: this.name,
      free: new Decimal(data.free),
      locked: new Decimal(data.locked ?? 0),
    };
  }

  async getOpenOrders(_symbol?: string): Promise<Order[]> {
    return []; // AMM doesn't have orders
  }

  async placeOrder(request: OrderRequest): Promise<Order> {
    // Convert to swap
    const pair = TradingPair.fromSymbol(request.symbol);
    if (!pair) {
      throw new Error(`Invalid symbol: ${request.symbol}`);
    }

    const trade = await this.executeSwap(
      pair.base,
      pair.quote,
      request.quantity,
      request.side === Side.BUY,
      new Decimal('0.01'), // 1% default slippage
    );

    return {
      orderId: trade.tradeId,
      clientOrderId: request.clientOrderId,
      symbol: request.symbol,
      venue: this.name,
      side: request.side,
      orderType: OrderType.MARKET,
      status: OrderStatus.FILLED,
      quantity: request.quantity,
      filledQuantity: trade.quantity,
      remainingQuantity: new Decimal(0),
      price: trade.price,
      averagePrice: trade.price,
      createdAt: trade.timestamp,
      updatedAt: trade.timestamp,
      fees: [trade.fee],
    };
  }

  async cancelOrder(_orderId: string, _symbol: string): Promise<Order> {
    throw new Error('AMM swaps cannot be cancelled');
  }

  async cancelAllOrders(_symbol?: string): Promise<Order[]> {
    return [];
  }

  // AMM specific methods
  override async getSwapQuote(
    baseToken: string,
    quoteToken: string,
    amount: Decimal,
    isBuy: boolean,
  ): Promise<SwapQuote> {
    const data = await this.request<QuoteData>('POST', '/api/v1/amm/quote', {
      baseToken,
      quoteToken,
      amount: amount.toString(),
      side: isBuy ? 'buy' : 'sell',
    });

    return {
      baseToken,
      quoteToken,
      inputAmount: amount,
      outputAmount: new Decimal(data.outputAmount),
      price: new Decimal(data.price),
      priceImpact: new Decimal(data.priceImpact ?? 0),
      fee: new Decimal(data.fee ?? 0),
      route: data.route ?? [],
      expiresAt: Date.now() + 60000,
    };
  }

  override async executeSwap(
    baseToken: string,
    quoteToken: string,
    amount: Decimal,
    isBuy: boolean,
    slippage: Decimal,
  ): Promise<Trade> {
    const data = await this.request<SwapResultData>('POST', '/api/v1/amm/swap', {
      baseToken,
      quoteToken,
      amount: amount.toString(),
      side: isBuy ? 'buy' : 'sell',
      slippage: slippage.toString(),
    });

    return {
      tradeId: data.txHash,
      orderId: data.txHash,
      symbol: `${baseToken}-${quoteToken}`,
      venue: this.name,
      side: isBuy ? Side.BUY : Side.SELL,
      price: new Decimal(data.price),
      quantity: amount,
      fee: {
        asset: '',
        amount: new Decimal(data.fee ?? 0),
      },
      timestamp: Date.now(),
      isMaker: false,
    };
  }

  override async getPoolInfo(baseToken: string, quoteToken: string): Promise<PoolInfo> {
    const data = await this.request<PoolInfoData>('GET', `/api/v1/amm/pool/${baseToken}/${quoteToken}`);

    return {
      address: data.address,
      baseToken,
      quoteToken,
      baseReserve: new Decimal(data.baseReserve),
      quoteReserve: new Decimal(data.quoteReserve),
      totalLiquidity: new Decimal(data.totalLiquidity ?? 0),
      feeRate: new Decimal(data.feeRate ?? '0.003'),
      apy: data.apy ? new Decimal(data.apy) : undefined,
    };
  }

  override async addLiquidity(
    baseToken: string,
    quoteToken: string,
    baseAmount: Decimal,
    quoteAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult> {
    const data = await this.request<LiquidityResultData>('POST', '/api/v1/amm/liquidity/add', {
      baseToken,
      quoteToken,
      baseAmount: baseAmount.toString(),
      quoteAmount: quoteAmount.toString(),
      slippage: slippage.toString(),
    });

    return {
      txHash: data.txHash,
      poolAddress: data.poolAddress ?? '',
      baseAmount,
      quoteAmount,
      lpTokens: new Decimal(data.lpTokens ?? 0),
      sharePercent: new Decimal(data.sharePercent ?? 0),
    };
  }

  override async removeLiquidity(
    poolAddress: string,
    liquidityAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult> {
    const data = await this.request<LiquidityResultData>('POST', '/api/v1/amm/liquidity/remove', {
      poolAddress,
      liquidity: liquidityAmount.toString(),
      slippage: slippage.toString(),
    });

    return {
      txHash: data.txHash,
      poolAddress,
      baseAmount: new Decimal(data.baseAmount ?? 0),
      quoteAmount: new Decimal(data.quoteAmount ?? 0),
      lpTokens: liquidityAmount,
      sharePercent: new Decimal(0),
    };
  }

  override async getLpPositions(): Promise<LpPosition[]> {
    const data = await this.request<LpPositionData[]>('GET', '/api/v1/amm/positions');

    return data.map((p) => ({
      poolAddress: p.poolAddress,
      baseToken: p.baseToken,
      quoteToken: p.quoteToken,
      lpTokens: new Decimal(p.lpTokens ?? 0),
      baseAmount: new Decimal(p.baseAmount),
      quoteAmount: new Decimal(p.quoteAmount),
      sharePercent: new Decimal(p.sharePercent ?? 0),
      unrealizedPnl: p.unrealizedPnl ? new Decimal(p.unrealizedPnl) : undefined,
    }));
  }
}

// =============================================================================
// Data Types (API responses)
// =============================================================================

interface MarketData {
  symbol: string;
  base: string;
  quote: string;
  pricePrecision?: number;
  quantityPrecision?: number;
  minQuantity?: string | number;
  maxQuantity?: string | number;
  minNotional?: string | number;
  tickSize?: string;
  lotSize?: string;
}

interface TickerData {
  symbol: string;
  bid?: string;
  ask?: string;
  last?: string;
  volume24h?: string;
  high24h?: string;
  low24h?: string;
  change24h?: string;
  timestamp?: number;
}

interface OrderbookData {
  bids?: [string, string][];
  asks?: [string, string][];
}

interface TradeData {
  id: string;
  orderId?: string;
  side: string;
  price: string;
  quantity: string;
  feeAsset?: string;
  feeAmount?: string | number;
  timestamp: number;
  isMaker?: boolean;
}

interface BalanceData {
  asset: string;
  free: string;
  locked?: string;
}

interface OrderData {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  side: string;
  type?: string;
  status?: string;
  quantity?: string;
  filledQuantity?: string;
  price?: string;
  averagePrice?: string;
  createdAt?: number;
  updatedAt?: number;
}

interface PoolData {
  baseToken: string;
  quoteToken: string;
}

interface PriceData {
  price?: string;
  volume24h?: string;
}

interface SwapData {
  txHash: string;
  side?: string;
  price: string;
  amount: string;
  fee?: string | number;
  timestamp: number;
}

interface QuoteData {
  outputAmount: string;
  price: string;
  priceImpact?: string | number;
  fee?: string | number;
  route?: string[];
}

interface SwapResultData {
  txHash: string;
  price: string;
  fee?: string | number;
}

interface PoolInfoData {
  address: string;
  baseReserve: string;
  quoteReserve: string;
  totalLiquidity?: string;
  feeRate?: string;
  apy?: string;
}

interface LiquidityResultData {
  txHash: string;
  poolAddress?: string;
  baseAmount?: string;
  quoteAmount?: string;
  lpTokens?: string | number;
  sharePercent?: string | number;
}

interface LpPositionData {
  poolAddress: string;
  baseToken: string;
  quoteToken: string;
  lpTokens?: string | number;
  baseAmount: string;
  quoteAmount: string;
  sharePercent?: string | number;
  unrealizedPnl?: string;
}
