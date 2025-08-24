/**
 * Lux Exchange Client
 * Main client for interacting with LX trading engine
 */

import { EventEmitter } from 'eventemitter3';
import Decimal from 'decimal.js';
import { v4 as uuidv4 } from 'uuid';
import {
  LXClientConfig,
  Order,
  Trade,
  OrderBookSnapshot,
  MarketTick,
  Position,
  PortfolioSummary,
  ApiResponse,
  WSEventType,
  OrderType,
  OrderSide,
  OrderStatus,
  TimeInForce,
  AssetClass,
  Market,
  Asset
} from './types';
import { OrderBook } from './orderbook';
import { MarketData } from './market-data';
import { WebSocketManager } from './websocket';
import { RateLimiter } from './rate-limiter';

export class LXClient extends EventEmitter {
  private config: LXClientConfig;
  private ws: WebSocketManager;
  private rateLimiter: RateLimiter;
  private orderBook: OrderBook;
  private marketData: MarketData;
  private isConnected: boolean = false;

  constructor(config: LXClientConfig = {}) {
    super();
    
    this.config = {
      endpoint: config.endpoint || 'https://api.luxexchange.io',
      wsEndpoint: config.wsEndpoint || 'wss://stream.luxexchange.io',
      testnet: config.testnet || false,
      timeout: config.timeout || 10000,
      reconnect: config.reconnect !== false,
      reconnectDelay: config.reconnectDelay || 1000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      ...config
    };

    if (this.config.testnet) {
      this.config.endpoint = 'https://testnet-api.luxexchange.io';
      this.config.wsEndpoint = 'wss://testnet-stream.luxexchange.io';
    }

    this.ws = new WebSocketManager(this.config.wsEndpoint!, this.config);
    this.rateLimiter = new RateLimiter(this.config.rateLimit);
    this.orderBook = new OrderBook();
    this.marketData = new MarketData();

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.ws.on(WSEventType.CONNECTED, () => {
      this.isConnected = true;
      this.emit(WSEventType.CONNECTED);
    });

    this.ws.on(WSEventType.DISCONNECTED, () => {
      this.isConnected = false;
      this.emit(WSEventType.DISCONNECTED);
    });

    this.ws.on(WSEventType.ERROR, (error) => {
      this.emit(WSEventType.ERROR, error);
    });

    this.ws.on(WSEventType.ORDER_UPDATE, (order) => {
      this.emit(WSEventType.ORDER_UPDATE, order);
    });

    this.ws.on(WSEventType.TRADE, (trade) => {
      this.emit(WSEventType.TRADE, trade);
    });

    this.ws.on(WSEventType.ORDERBOOK, (snapshot) => {
      this.orderBook.update(snapshot);
      this.emit(WSEventType.ORDERBOOK, snapshot);
    });

    this.ws.on(WSEventType.TICKER, (tick) => {
      this.marketData.updateTicker(tick);
      this.emit(WSEventType.TICKER, tick);
    });
  }

  /**
   * Connect to the exchange
   */
  async connect(): Promise<void> {
    await this.ws.connect();
  }

  /**
   * Disconnect from the exchange
   */
  async disconnect(): Promise<void> {
    await this.ws.disconnect();
  }

  /**
   * Get available assets
   */
  async getAssets(assetClass?: AssetClass): Promise<Asset[]> {
    const response = await this.request<Asset[]>('GET', '/assets', {
      assetClass
    });
    return response.data || [];
  }

  /**
   * Submit a new order
   */
  async submitOrder(params: {
    symbol: string;
    side: OrderSide;
    type: OrderType;
    quantity: number | string;
    price?: number | string;
    stopPrice?: number | string;
    timeInForce?: TimeInForce;
    postOnly?: boolean;
    reduceOnly?: boolean;
    leverage?: number;
    clientOrderId?: string;
  }): Promise<Order> {
    await this.rateLimiter.checkOrderLimit();

    const order = {
      ...params,
      quantity: new Decimal(params.quantity),
      price: params.price ? new Decimal(params.price) : undefined,
      stopPrice: params.stopPrice ? new Decimal(params.stopPrice) : undefined,
      clientOrderId: params.clientOrderId || uuidv4(),
      timeInForce: params.timeInForce || TimeInForce.GTC
    };

    const response = await this.request<Order>('POST', '/orders', order);
    if (!response.data) {
      throw new Error(response.error?.message || 'Failed to submit order');
    }
    return response.data;
  }

  /**
   * Cancel an order
   */
  async cancelOrder(orderId: string): Promise<void> {
    await this.request('DELETE', `/orders/${orderId}`);
  }

  /**
   * Cancel all orders for a symbol
   */
  async cancelAllOrders(symbol?: string): Promise<void> {
    const params = symbol ? { symbol } : {};
    await this.request('DELETE', '/orders', params);
  }

  /**
   * Get open orders
   */
  async getOpenOrders(symbol?: string): Promise<Order[]> {
    const params = symbol ? { symbol } : {};
    const response = await this.request<Order[]>('GET', '/orders', params);
    return response.data || [];
  }

  /**
   * Get order by ID
   */
  async getOrder(orderId: string): Promise<Order> {
    const response = await this.request<Order>('GET', `/orders/${orderId}`);
    if (!response.data) {
      throw new Error('Order not found');
    }
    return response.data;
  }

  /**
   * Get recent trades
   */
  async getTrades(symbol: string, limit: number = 100): Promise<Trade[]> {
    const response = await this.request<Trade[]>('GET', '/trades', {
      symbol,
      limit
    });
    return response.data || [];
  }

  /**
   * Get order book snapshot
   */
  async getOrderBook(symbol: string, depth: number = 20): Promise<OrderBookSnapshot> {
    const response = await this.request<OrderBookSnapshot>('GET', `/orderbook/${symbol}`, {
      depth
    });
    if (!response.data) {
      throw new Error('Failed to get order book');
    }
    return response.data;
  }

  /**
   * Get market ticker
   */
  async getTicker(symbol: string): Promise<MarketTick> {
    const response = await this.request<MarketTick>('GET', `/ticker/${symbol}`);
    if (!response.data) {
      throw new Error('Failed to get ticker');
    }
    return response.data;
  }

  /**
   * Get all tickers
   */
  async getAllTickers(): Promise<MarketTick[]> {
    const response = await this.request<MarketTick[]>('GET', '/tickers');
    return response.data || [];
  }

  /**
   * Get positions (for margin/futures)
   */
  async getPositions(): Promise<Position[]> {
    const response = await this.request<Position[]>('GET', '/positions');
    return response.data || [];
  }

  /**
   * Get portfolio summary
   */
  async getPortfolio(): Promise<PortfolioSummary> {
    const response = await this.request<PortfolioSummary>('GET', '/portfolio');
    if (!response.data) {
      throw new Error('Failed to get portfolio');
    }
    return response.data;
  }

  /**
   * Subscribe to market data
   */
  subscribe(channel: string, symbols: string[]): void {
    this.ws.subscribe(channel, symbols);
  }

  /**
   * Unsubscribe from market data
   */
  unsubscribe(channel: string, symbols: string[]): void {
    this.ws.unsubscribe(channel, symbols);
  }

  /**
   * Make HTTP request to the API
   */
  private async request<T>(
    method: string,
    path: string,
    params?: any
  ): Promise<ApiResponse<T>> {
    await this.rateLimiter.checkRequestLimit();

    const url = `${this.config.endpoint}${path}`;
    const headers: any = {
      'Content-Type': 'application/json',
      'X-Client': 'lx-client-ts/1.0.0'
    };

    if (this.config.apiKey) {
      headers['X-API-Key'] = this.config.apiKey;
      // Add signature if needed
    }

    const options: RequestInit = {
      method,
      headers,
      signal: AbortSignal.timeout(this.config.timeout!)
    };

    if (method === 'GET' && params) {
      const queryString = new URLSearchParams(params).toString();
      options.method = 'GET';
    } else if (params) {
      options.body = JSON.stringify(params);
    }

    try {
      const response = await fetch(url, options);
      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: {
            code: data.code || 'UNKNOWN',
            message: data.message || 'Request failed'
          },
          timestamp: new Date()
        };
      }

      return {
        success: true,
        data,
        timestamp: new Date()
      };
    } catch (error: any) {
      return {
        success: false,
        error: {
          code: 'NETWORK_ERROR',
          message: error.message
        },
        timestamp: new Date()
      };
    }
  }

  /**
   * Get client status
   */
  getStatus(): {
    connected: boolean;
    endpoint: string;
    testnet: boolean;
  } {
    return {
      connected: this.isConnected,
      endpoint: this.config.endpoint!,
      testnet: this.config.testnet!
    };
  }
}