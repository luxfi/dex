import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';

// Types
export enum OrderType {
  LIMIT = 0,
  MARKET = 1,
  STOP = 2,
  STOP_LIMIT = 3,
  ICEBERG = 4,
  PEG = 5
}

export enum OrderSide {
  BUY = 0,
  SELL = 1
}

export enum OrderStatus {
  OPEN = 'open',
  PARTIAL = 'partial',
  FILLED = 'filled',
  CANCELLED = 'cancelled',
  REJECTED = 'rejected'
}

export enum TimeInForce {
  GTC = 'GTC',
  IOC = 'IOC',
  FOK = 'FOK',
  DAY = 'DAY'
}

export interface Order {
  orderId?: number;
  symbol: string;
  type: OrderType;
  side: OrderSide;
  price: number;
  size: number;
  userID?: string;
  clientID?: string;
  timeInForce?: TimeInForce;
  postOnly?: boolean;
  reduceOnly?: boolean;
  status?: OrderStatus;
  filled?: number;
  remaining?: number;
  timestamp?: number;
}

export interface Trade {
  tradeId: number;
  symbol: string;
  price: number;
  size: number;
  side: OrderSide;
  buyOrderId: number;
  sellOrderId: number;
  buyerId: string;
  sellerId: string;
  timestamp: number;
}

export interface OrderBookLevel {
  price: number;
  size: number;
  count?: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
}

export interface NodeInfo {
  version: string;
  network: string;
  orderCount: number;
  tradeCount: number;
  timestamp: number;
}

export interface LXDexConfig {
  jsonRpcUrl?: string;
  wsUrl?: string;
  grpcUrl?: string;
  apiKey?: string;
}

// JSON-RPC Client
class JSONRPCClient {
  private axios: AxiosInstance;
  private idCounter = 1;

  constructor(baseURL: string) {
    this.axios = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async call(method: string, params: any = {}): Promise<any> {
    const response = await this.axios.post('/rpc', {
      jsonrpc: '2.0',
      method,
      params,
      id: this.idCounter++
    });

    if (response.data.error) {
      throw new Error(response.data.error.message);
    }

    return response.data.result;
  }
}

// Main SDK Client
export class LXDexClient {
  private jsonRpc: JSONRPCClient;
  private ws: WebSocket | null = null;
  private wsCallbacks: Map<string, Function[]> = new Map();
  private config: LXDexConfig;

  constructor(config: LXDexConfig = {}) {
    this.config = {
      jsonRpcUrl: config.jsonRpcUrl || 'http://localhost:8080',
      wsUrl: config.wsUrl || 'ws://localhost:8081',
      ...config
    };

    this.jsonRpc = new JSONRPCClient(this.config.jsonRpcUrl!);
  }

  // Connection Management
  async connect(): Promise<void> {
    if (this.ws) return;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.wsUrl!);

      this.ws.on('open', () => {
        console.log('WebSocket connected');
        resolve();
      });

      this.ws.on('message', (data: string) => {
        try {
          const msg = JSON.parse(data);
          this.handleWebSocketMessage(msg);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      });

      this.ws.on('error', (err) => {
        console.error('WebSocket error:', err);
        reject(err);
      });

      this.ws.on('close', () => {
        console.log('WebSocket disconnected');
        this.ws = null;
      });
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // Order Management
  async placeOrder(order: Partial<Order>): Promise<{ orderId: number; status: string }> {
    return this.jsonRpc.call('lx_placeOrder', order);
  }

  async cancelOrder(orderId: number): Promise<{ success: boolean; message: string }> {
    return this.jsonRpc.call('lx_cancelOrder', { orderId });
  }

  async getOrder(orderId: number): Promise<Order> {
    return this.jsonRpc.call('lx_getOrder', { orderId });
  }

  // Market Data
  async getOrderBook(symbol: string = 'BTC-USD', depth: number = 10): Promise<OrderBook> {
    return this.jsonRpc.call('lx_getOrderBook', { symbol, depth });
  }

  async getBestBid(symbol: string = 'BTC-USD'): Promise<number> {
    const result = await this.jsonRpc.call('lx_getBestBid', { symbol });
    return result.price;
  }

  async getBestAsk(symbol: string = 'BTC-USD'): Promise<number> {
    const result = await this.jsonRpc.call('lx_getBestAsk', { symbol });
    return result.price;
  }

  async getTrades(symbol: string = 'BTC-USD', limit: number = 100): Promise<Trade[]> {
    return this.jsonRpc.call('lx_getTrades', { symbol, limit });
  }

  // Node Information
  async getInfo(): Promise<NodeInfo> {
    return this.jsonRpc.call('lx_getInfo');
  }

  async ping(): Promise<string> {
    return this.jsonRpc.call('lx_ping');
  }

  // WebSocket Subscriptions
  subscribe(channel: string, callback: Function): void {
    if (!this.wsCallbacks.has(channel)) {
      this.wsCallbacks.set(channel, []);
    }
    this.wsCallbacks.get(channel)!.push(callback);

    // Send subscription message
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channel
      }));
    }
  }

  unsubscribe(channel: string, callback?: Function): void {
    if (callback) {
      const callbacks = this.wsCallbacks.get(channel);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    } else {
      this.wsCallbacks.delete(channel);
    }

    // Send unsubscribe message
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        channel
      }));
    }
  }

  subscribeOrderBook(symbol: string, callback: (book: OrderBook) => void): void {
    this.subscribe(`orderbook:${symbol}`, callback);
  }

  subscribeTrades(symbol: string, callback: (trade: Trade) => void): void {
    this.subscribe(`trades:${symbol}`, callback);
  }

  // Private methods
  private handleWebSocketMessage(msg: any): void {
    const { channel, data } = msg;
    const callbacks = this.wsCallbacks.get(channel);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }

  // Utility methods
  static formatPrice(price: number, decimals: number = 2): string {
    return price.toFixed(decimals);
  }

  static formatSize(size: number, decimals: number = 8): string {
    return size.toFixed(decimals);
  }

  static calculateTotal(price: number, size: number): number {
    return price * size;
  }
}

// Export everything
export default LXDexClient;