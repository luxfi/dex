/**
 * CCXT adapter - trade on 100+ exchanges.
 */

import ccxt, { type Exchange, type Balances, type Order as CcxtOrder, type Ticker as CcxtTicker, type Trade as CcxtTrade } from 'ccxt';
import { Decimal } from 'decimal.js';
import type { CcxtConfig } from '../config.js';
import { Orderbook } from '../orderbook.js';
import {
  OrderStatus,
  OrderType,
  Side,
  VenueType,
  type Balance,
  type MarketInfo,
  type Order,
  type OrderRequest,
  type Ticker,
  type Trade,
} from '../types.js';
import { BaseAdapter, clobCapabilities, type VenueCapabilities } from './base.js';

// =============================================================================
// CCXT Adapter
// =============================================================================

export class CcxtAdapter extends BaseAdapter {
  readonly venueType = VenueType.CCXT;
  readonly capabilities: VenueCapabilities;

  private exchange: Exchange | null = null;

  constructor(
    readonly name: string,
    private readonly config: CcxtConfig,
  ) {
    super();
    this.capabilities = {
      ...clobCapabilities(),
      batchOrders: false, // CCXT doesn't have unified batch
    };
  }

  async connect(): Promise<void> {
    // @ts-expect-error - dynamic exchange access
    const ExchangeClass = ccxt[this.config.exchangeId] as typeof Exchange;
    if (!ExchangeClass) {
      throw new Error(`Exchange not found: ${this.config.exchangeId}`);
    }

    this.exchange = new ExchangeClass({
      apiKey: this.config.apiKey,
      secret: this.config.apiSecret,
      password: this.config.password,
      enableRateLimit: this.config.rateLimit,
      options: this.config.options,
    });

    if (this.config.sandbox) {
      this.exchange.setSandboxMode(true);
    }

    const start = Date.now();
    await this.exchange.loadMarkets();
    this._latencyMs = Date.now() - start;
    this._connected = true;
  }

  async disconnect(): Promise<void> {
    if (this.exchange) {
      await this.exchange.close();
    }
    this._connected = false;
  }

  async getMarkets(): Promise<MarketInfo[]> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const markets: MarketInfo[] = [];
    for (const [symbol, market] of Object.entries(this.exchange.markets)) {
      markets.push({
        symbol,
        base: market.base ?? '',
        quote: market.quote ?? '',
        pricePrecision: market.precision?.price ?? 8,
        quantityPrecision: market.precision?.amount ?? 8,
        minQuantity: new Decimal(market.limits?.amount?.min ?? 0),
        maxQuantity: market.limits?.amount?.max ? new Decimal(market.limits.amount.max) : undefined,
        minNotional: market.limits?.cost?.min ? new Decimal(market.limits.cost.min) : undefined,
        tickSize: new Decimal('0.00000001'),
        lotSize: new Decimal('0.00000001'),
      });
    }
    return markets;
  }

  async getTicker(symbol: string): Promise<Ticker> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const start = Date.now();
    const ticker: CcxtTicker = await this.exchange.fetchTicker(symbol);
    this._latencyMs = Date.now() - start;

    return {
      symbol: ticker.symbol ?? symbol,
      venue: this.name,
      bid: ticker.bid !== undefined ? new Decimal(ticker.bid) : undefined,
      ask: ticker.ask !== undefined ? new Decimal(ticker.ask) : undefined,
      last: ticker.last !== undefined ? new Decimal(ticker.last) : undefined,
      volume24h: ticker.baseVolume !== undefined ? new Decimal(ticker.baseVolume) : undefined,
      high24h: ticker.high !== undefined ? new Decimal(ticker.high) : undefined,
      low24h: ticker.low !== undefined ? new Decimal(ticker.low) : undefined,
      change24h: ticker.percentage !== undefined ? new Decimal(ticker.percentage) : undefined,
      timestamp: ticker.timestamp ?? 0,
    };
  }

  async getOrderbook(symbol: string, depth?: number): Promise<Orderbook> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const start = Date.now();
    const book = await this.exchange.fetchOrderBook(symbol, depth);
    this._latencyMs = Date.now() - start;

    const orderbook = new Orderbook(symbol, this.name);

    for (const [price, qty] of book.bids ?? []) {
      orderbook.addBid(new Decimal(String(price)), new Decimal(String(qty)));
    }

    for (const [price, qty] of book.asks ?? []) {
      orderbook.addAsk(new Decimal(String(price)), new Decimal(String(qty)));
    }

    orderbook.sort();
    return orderbook;
  }

  async getTrades(symbol: string, limit?: number): Promise<Trade[]> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const trades: CcxtTrade[] = await this.exchange.fetchTrades(symbol, undefined, limit);

    return trades.map((t) => ({
      tradeId: t.id ?? '',
      orderId: t.order ?? '',
      symbol: t.symbol ?? symbol,
      venue: this.name,
      side: t.side === 'buy' ? Side.BUY : Side.SELL,
      price: new Decimal(t.price ?? 0),
      quantity: new Decimal(t.amount ?? 0),
      fee: {
        asset: t.fee?.currency ?? '',
        amount: new Decimal(t.fee?.cost ?? 0),
      },
      timestamp: t.timestamp ?? 0,
      isMaker: t.takerOrMaker === 'maker',
    }));
  }

  async getBalances(): Promise<Balance[]> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const start = Date.now();
    const balances: Balances = await this.exchange.fetchBalance();
    this._latencyMs = Date.now() - start;

    const result: Balance[] = [];
    const freeBalances = (balances.free ?? {}) as unknown as Record<string, number | string>;
    const usedBalances = (balances.used ?? {}) as unknown as Record<string, number | string>;
    for (const [asset, amount] of Object.entries(balances.total ?? {})) {
      if (amount && Number(amount) > 0) {
        result.push({
          asset,
          venue: this.name,
          free: new Decimal(freeBalances[asset] ?? 0),
          locked: new Decimal(usedBalances[asset] ?? 0),
        });
      }
    }
    return result;
  }

  async getBalance(asset: string): Promise<Balance> {
    const balances = await this.getBalances();
    const found = balances.find((b) => b.asset === asset);
    if (found) {
      return found;
    }
    return {
      asset,
      venue: this.name,
      free: new Decimal(0),
      locked: new Decimal(0),
    };
  }

  async getOpenOrders(symbol?: string): Promise<Order[]> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const orders: CcxtOrder[] = await this.exchange.fetchOpenOrders(symbol);
    return orders.map((o) => this.convertOrder(o));
  }

  async placeOrder(request: OrderRequest): Promise<Order> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const side = request.side === Side.BUY ? 'buy' : 'sell';
    const orderType = request.orderType === OrderType.MARKET ? 'market' : 'limit';

    const start = Date.now();
    const order = await this.exchange.createOrder(
      request.symbol,
      orderType,
      side,
      request.quantity.toNumber(),
      request.price?.toNumber(),
      { clientOrderId: request.clientOrderId },
    );
    this._latencyMs = Date.now() - start;

    return this.convertOrder(order);
  }

  async cancelOrder(orderId: string, symbol: string): Promise<Order> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    const order = await this.exchange.cancelOrder(orderId, symbol);
    return this.convertOrder(order);
  }

  async cancelAllOrders(symbol?: string): Promise<Order[]> {
    if (!this.exchange) {
      throw new Error('Not connected');
    }

    try {
      const orders = await this.exchange.cancelAllOrders(symbol);
      return orders.map((o: CcxtOrder) => this.convertOrder(o));
    } catch {
      // Fallback: cancel one by one
      const openOrders = await this.getOpenOrders(symbol);
      const cancelled: Order[] = [];
      for (const order of openOrders) {
        try {
          const o = await this.cancelOrder(order.orderId, order.symbol);
          cancelled.push(o);
        } catch {
          // Ignore individual cancellation errors
        }
      }
      return cancelled;
    }
  }

  private convertOrder(o: CcxtOrder): Order {
    const statusMap: Record<string, OrderStatus> = {
      open: OrderStatus.OPEN,
      closed: OrderStatus.FILLED,
      canceled: OrderStatus.CANCELLED,
      cancelled: OrderStatus.CANCELLED,
      expired: OrderStatus.EXPIRED,
      rejected: OrderStatus.REJECTED,
    };

    const typeMap: Record<string, OrderType> = {
      market: OrderType.MARKET,
      limit: OrderType.LIMIT,
      stop: OrderType.STOP_LOSS,
      stop_limit: OrderType.STOP_LOSS_LIMIT,
    };

    const quantity = new Decimal(o.amount ?? 0);
    const filled = new Decimal(o.filled ?? 0);

    return {
      orderId: o.id ?? '',
      clientOrderId: o.clientOrderId ?? '',
      symbol: o.symbol ?? '',
      venue: this.name,
      side: o.side === 'buy' ? Side.BUY : Side.SELL,
      orderType: typeMap[o.type ?? 'limit'] ?? OrderType.LIMIT,
      status: statusMap[o.status ?? 'open'] ?? OrderStatus.OPEN,
      quantity,
      filledQuantity: filled,
      remainingQuantity: quantity.minus(filled),
      price: o.price !== undefined ? new Decimal(o.price) : undefined,
      averagePrice: o.average !== undefined ? new Decimal(o.average) : undefined,
      createdAt: o.timestamp ?? 0,
      updatedAt: o.lastTradeTimestamp ?? o.timestamp ?? 0,
      fees: [],
    };
  }
}
