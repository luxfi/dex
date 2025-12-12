#!/usr/bin/env node
/**
 * LX DEX Trading Client
 *
 * Programmatic trading client for LX DEX supporting multiple protocols:
 * - WebSocket: Real-time streaming, subscriptions, interactive trading
 * - gRPC: High-throughput RPC calls for automated trading systems
 *
 * Usage:
 * - As library: import { LXClient, GrpcClient } from '@luxfi/dex-client'
 * - Interactive REPL: npm start
 * - Command-line: npm start -- place_order BTC-USD buy limit 50000 0.1
 */

import { Command } from 'commander';
import * as readline from 'readline';
import WebSocket from 'ws';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import * as path from 'path';

// ============================================================================
// Common Types and Interfaces
// ============================================================================

/** Message structure for WebSocket communication */
export interface Message {
  type: string;
  data?: Record<string, unknown>;
  error?: string;
  request_id?: string;
  timestamp?: number;
}

/** Order parameters */
export interface Order {
  symbol: string;
  side: string;
  type: string;
  price: number;
  size: number;
  clientId?: string;
  timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  postOnly?: boolean;
  reduceOnly?: boolean;
  stopPrice?: number;
}

/** Order response from exchange */
export interface OrderResponse {
  orderId: string | number;
  status: string;
  message?: string;
  error?: string;
}

/** Position data */
export interface Position {
  symbol: string;
  size: number;
  entryPrice: number;
  markPrice: number;
  pnl: number;
  margin: number;
}

/** Balance data */
export interface Balance {
  asset: string;
  available: number;
  locked: number;
  total: number;
}

/** Price level in orderbook */
export interface PriceLevel {
  price: number;
  size: number;
  count?: number;
}

/** Orderbook snapshot */
export interface OrderBook {
  symbol: string;
  bids: PriceLevel[];
  asks: PriceLevel[];
  timestamp: number;
}

/** Protocol type */
export type Protocol = 'ws' | 'grpc';

// ============================================================================
// Trading Client Interface
// ============================================================================

/**
 * Common interface for trading operations.
 * Both WebSocket and gRPC clients implement this interface.
 */
export interface ITradingClient {
  connect(): Promise<void>;
  close(): void;
  auth(apiKey: string, apiSecret: string): Promise<void>;
  placeOrder(order: Order): Promise<OrderResponse>;
  cancelOrder(orderId: string | number): Promise<OrderResponse>;
  getOrders(): Promise<{ orders: Order[] }>;
  getPositions(): Promise<{ positions: Position[] }>;
  getBalances(): Promise<{ balances: Balance[] }>;
  getOrderBook(symbol: string, depth?: number): Promise<OrderBook>;
}

// ============================================================================
// WebSocket Client Implementation
// ============================================================================

/**
 * WebSocket-based trading client.
 * Best for: Real-time streaming, market data subscriptions, interactive trading.
 */
export class LXClient implements ITradingClient {
  private ws: WebSocket | null = null;
  private url: string;
  private verbose: boolean;
  private reqCounter = 0;
  private responses: Map<string, Message> = new Map();
  private pendingResolvers: Map<string, { resolve: (msg: Message) => void; reject: (err: Error) => void }> = new Map();
  private connected = false;
  private connectedPromise: Promise<void> | null = null;
  private connectedResolve: (() => void) | null = null;
  private messageCallback: ((msg: Message) => void) | null = null;

  constructor(url: string, verbose = false) {
    this.url = url;
    this.verbose = verbose;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.connectedPromise = new Promise((res) => {
        this.connectedResolve = res;
      });

      this.ws = new WebSocket(this.url);

      this.ws.on('open', () => {
        if (this.verbose) {
          console.log('WebSocket connected');
        }
      });

      this.ws.on('message', (data: WebSocket.RawData) => {
        const text = data.toString();
        if (this.verbose) {
          console.log(`<< ${text}`);
        }

        try {
          const msg: Message = JSON.parse(text);

          if (msg.type === 'connected') {
            this.connected = true;
            if (this.connectedResolve) {
              this.connectedResolve();
            }
            resolve();
            return;
          }

          if (msg.request_id) {
            const resolver = this.pendingResolvers.get(msg.request_id);
            if (resolver) {
              resolver.resolve(msg);
              this.pendingResolvers.delete(msg.request_id);
            } else {
              this.responses.set(msg.request_id, msg);
            }
          } else if (msg.type !== 'pong') {
            // Unsolicited message (subscription updates)
            if (this.messageCallback) {
              this.messageCallback(msg);
            } else {
              this.printMessage(msg);
            }
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      });

      this.ws.on('error', (error) => {
        console.error(`WebSocket error: ${error.message}`);
        reject(error);
      });

      this.ws.on('close', () => {
        this.connected = false;
        if (this.verbose) {
          console.log('WebSocket disconnected');
        }
        // Reject all pending requests
        for (const [, resolver] of this.pendingResolvers) {
          resolver.reject(new Error('Connection closed'));
        }
        this.pendingResolvers.clear();
      });

      // Timeout for initial connection
      setTimeout(() => {
        if (!this.connected) {
          reject(new Error('Connection timeout'));
        }
      }, 10000);
    });
  }

  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  setMessageCallback(cb: (msg: Message) => void): void {
    this.messageCallback = cb;
  }

  private nextReqId(): string {
    this.reqCounter++;
    return `req-${this.reqCounter}`;
  }

  private send(msgType: string, data?: Record<string, unknown>): string {
    const reqId = this.nextReqId();
    const msg: Record<string, unknown> = {
      type: msgType,
      request_id: reqId,
      ...data,
    };

    if (this.verbose) {
      console.log(`>> ${JSON.stringify(msg, null, 2)}`);
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }

    return reqId;
  }

  async waitResponse(reqId: string, timeout = 5000): Promise<Message> {
    // Check if already received
    const existing = this.responses.get(reqId);
    if (existing) {
      this.responses.delete(reqId);
      return existing;
    }

    // Wait for response
    return new Promise((resolve, reject) => {
      this.pendingResolvers.set(reqId, { resolve, reject });

      setTimeout(() => {
        if (this.pendingResolvers.has(reqId)) {
          this.pendingResolvers.delete(reqId);
          reject(new Error('Response timeout'));
        }
      }, timeout);
    });
  }

  async auth(apiKey: string, apiSecret: string): Promise<void> {
    const reqId = this.send('auth', { apiKey, apiSecret });
    const resp = await this.waitResponse(reqId);
    if (resp.error) {
      throw new Error(`Auth failed: ${resp.error}`);
    }
  }

  async placeOrder(order: Order): Promise<OrderResponse> {
    const reqId = this.send('place_order', { order });
    const resp = await this.waitResponse(reqId);
    if (resp.error) {
      return { orderId: '', status: 'rejected', error: resp.error };
    }
    const data = resp.data as { orderId?: string | number; status?: string; message?: string } | undefined;
    return {
      orderId: data?.orderId || '',
      status: data?.status || 'unknown',
      message: data?.message,
    };
  }

  async cancelOrder(orderId: string | number): Promise<OrderResponse> {
    const reqId = this.send('cancel_order', { orderID: orderId });
    const resp = await this.waitResponse(reqId);
    if (resp.error) {
      return { orderId, status: 'error', error: resp.error };
    }
    const data = resp.data as { success?: boolean; message?: string } | undefined;
    return {
      orderId,
      status: data?.success ? 'cancelled' : 'error',
      message: data?.message,
    };
  }

  async getPositions(): Promise<{ positions: Position[] }> {
    const reqId = this.send('get_positions', {});
    const resp = await this.waitResponse(reqId);
    const data = resp.data as { positions?: Position[] } | undefined;
    return { positions: data?.positions || [] };
  }

  async getOrders(): Promise<{ orders: Order[] }> {
    const reqId = this.send('get_orders', {});
    const resp = await this.waitResponse(reqId);
    const data = resp.data as { orders?: Order[] } | undefined;
    return { orders: data?.orders || [] };
  }

  async getBalances(): Promise<{ balances: Balance[] }> {
    const reqId = this.send('get_balances', {});
    const resp = await this.waitResponse(reqId);
    const data = resp.data as { balances?: Balance[] } | undefined;
    return { balances: data?.balances || [] };
  }

  async getOrderBook(symbol: string, _depth?: number): Promise<OrderBook> {
    // WebSocket uses subscribe for orderbook - single snapshot via request
    const reqId = this.send('get_orderbook', { symbol });
    const resp = await this.waitResponse(reqId);
    const data = resp.data as OrderBook | undefined;
    return data || { symbol, bids: [], asks: [], timestamp: Date.now() };
  }

  subscribe(symbol: string): void {
    this.send('subscribe', { symbols: [symbol] });
  }

  unsubscribe(symbol: string): void {
    this.send('unsubscribe', { symbols: [symbol] });
  }

  printMessage(msg: Message): void {
    if (msg.error) {
      console.error(`Error: ${msg.error}`);
      return;
    }

    switch (msg.type) {
      case 'orderbook': {
        const data = msg.data as { symbol?: string; bids?: Array<{ price: number; size: number }>; asks?: Array<{ price: number; size: number }> };
        console.log(`OrderBook [${data?.symbol || ''}]:`);
        if (data?.bids) {
          console.log(`  Bids: ${data.bids.length} levels`);
          for (const bid of data.bids.slice(0, 5)) {
            console.log(`    ${bid.price.toFixed(2)} @ ${bid.size.toFixed(4)}`);
          }
        }
        if (data?.asks) {
          console.log(`  Asks: ${data.asks.length} levels`);
          for (const ask of data.asks.slice(0, 5)) {
            console.log(`    ${ask.price.toFixed(2)} @ ${ask.size.toFixed(4)}`);
          }
        }
        break;
      }
      case 'order_update':
      case 'position_update':
        console.log(`${msg.type}: ${JSON.stringify(msg.data, null, 2)}`);
        break;
      default:
        console.log(JSON.stringify(msg, null, 2));
    }
  }
}

// ============================================================================
// gRPC Client Implementation
// ============================================================================

/**
 * gRPC-based trading client.
 * Best for: High-throughput automated trading, batch operations, low latency.
 */
export class GrpcClient implements ITradingClient {
  private client: grpc.Client | null = null;
  private service: Record<string, CallableFunction> | null = null;
  private host: string;
  private verbose: boolean;
  private protoPath: string;
  private credentials: { apiKey?: string; apiSecret?: string } = {};
  private metadata: grpc.Metadata = new grpc.Metadata();

  constructor(host: string, verbose = false, protoPath?: string) {
    this.host = host;
    this.verbose = verbose;
    // Default proto path relative to project root
    this.protoPath = protoPath || path.resolve(__dirname, '../../../proto/lxdex.proto');
  }

  async connect(): Promise<void> {
    // Load proto file
    const packageDefinition = await protoLoader.load(this.protoPath, {
      keepCase: false,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    });

    const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);

    // Get the service definition
    const lxdex = protoDescriptor.lxdex as { LXDEXService: grpc.ServiceClientConstructor };

    if (!lxdex || !lxdex.LXDEXService) {
      throw new Error('Failed to load LXDEXService from proto file');
    }

    // Create client
    this.client = new lxdex.LXDEXService(
      this.host,
      grpc.credentials.createInsecure()
    );

    // Store service methods for easy access
    this.service = this.client as unknown as Record<string, CallableFunction>;

    if (this.verbose) {
      console.log(`gRPC connected to ${this.host}`);
    }

    // Test connection with ping
    try {
      await this.ping();
    } catch (err) {
      if (this.verbose) {
        console.log('Ping failed (server may not support ping), continuing...');
      }
    }
  }

  close(): void {
    if (this.client) {
      grpc.closeClient(this.client);
      this.client = null;
      this.service = null;
    }
  }

  async auth(apiKey: string, apiSecret: string): Promise<void> {
    this.credentials = { apiKey, apiSecret };
    this.metadata = new grpc.Metadata();
    this.metadata.add('x-api-key', apiKey);
    this.metadata.add('x-api-secret', apiSecret);
    if (this.verbose) {
      console.log('gRPC credentials set');
    }
  }

  private promisifyCall<TReq, TRes>(method: string, request: TReq): Promise<TRes> {
    return new Promise((resolve, reject) => {
      if (!this.service) {
        reject(new Error('Client not connected'));
        return;
      }

      const fn = this.service[method];
      if (typeof fn !== 'function') {
        reject(new Error(`Method ${method} not found`));
        return;
      }

      if (this.verbose) {
        console.log(`>> ${method}: ${JSON.stringify(request)}`);
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (fn as any).call(this.client, request, this.metadata, (err: grpc.ServiceError | null, response: TRes) => {
        if (err) {
          if (this.verbose) {
            console.error(`<< ${method} error: ${err.message}`);
          }
          reject(err);
          return;
        }
        if (this.verbose) {
          console.log(`<< ${method}: ${JSON.stringify(response)}`);
        }
        resolve(response);
      });
    });
  }

  async ping(): Promise<{ timestamp: number; message: string }> {
    return this.promisifyCall('ping', { timestamp: Date.now() });
  }

  async placeOrder(order: Order): Promise<OrderResponse> {
    const sideMap: Record<string, number> = { buy: 0, sell: 1 };
    const typeMap: Record<string, number> = {
      limit: 0,
      market: 1,
      stop: 2,
      stop_limit: 3,
      iceberg: 4,
      peg: 5,
    };
    const tifMap: Record<string, number> = { GTC: 0, IOC: 1, FOK: 2, DAY: 3 };

    const request = {
      symbol: order.symbol,
      side: sideMap[order.side.toLowerCase()] ?? 0,
      type: typeMap[order.type.toLowerCase()] ?? 0,
      price: order.price,
      size: order.size,
      userId: this.credentials.apiKey || '',
      clientId: order.clientId || '',
      timeInForce: tifMap[order.timeInForce || 'GTC'] ?? 0,
      postOnly: order.postOnly || false,
      reduceOnly: order.reduceOnly || false,
      stopPrice: order.stopPrice || 0,
      limitPrice: order.price,
    };

    const response = await this.promisifyCall<typeof request, { orderId: string; status: number; message: string }>('placeOrder', request);

    const statusMap: Record<number, string> = {
      0: 'open',
      1: 'partial',
      2: 'filled',
      3: 'cancelled',
      4: 'rejected',
    };

    return {
      orderId: response.orderId,
      status: statusMap[response.status] || 'unknown',
      message: response.message,
    };
  }

  async cancelOrder(orderId: string | number): Promise<OrderResponse> {
    const request = {
      orderId: typeof orderId === 'string' ? parseInt(orderId, 10) : orderId,
      userId: this.credentials.apiKey || '',
    };

    const response = await this.promisifyCall<typeof request, { success: boolean; message: string }>('cancelOrder', request);

    return {
      orderId,
      status: response.success ? 'cancelled' : 'error',
      message: response.message,
    };
  }

  async getOrders(): Promise<{ orders: Order[] }> {
    const request = {
      userId: this.credentials.apiKey || '',
      symbol: '',
      status: 0, // OPEN
      limit: 100,
    };

    interface GrpcOrder {
      orderId: string;
      symbol: string;
      type: number;
      side: number;
      price: number;
      size: number;
      filled: number;
      remaining: number;
      status: number;
    }

    const response = await this.promisifyCall<typeof request, { orders: GrpcOrder[] }>('getOrders', request);

    const sideMap: Record<number, string> = { 0: 'buy', 1: 'sell' };
    const typeMap: Record<number, string> = {
      0: 'limit',
      1: 'market',
      2: 'stop',
      3: 'stop_limit',
    };

    return {
      orders: (response.orders || []).map((o) => ({
        symbol: o.symbol,
        side: sideMap[o.side] || 'buy',
        type: typeMap[o.type] || 'limit',
        price: o.price,
        size: o.size,
      })),
    };
  }

  async getPositions(): Promise<{ positions: Position[] }> {
    const request = {
      userId: this.credentials.apiKey || '',
    };

    interface GrpcPosition {
      symbol: string;
      size: number;
      entryPrice: number;
      markPrice: number;
      pnl: number;
      margin: number;
    }

    const response = await this.promisifyCall<typeof request, { positions: GrpcPosition[] }>('getPositions', request);

    return {
      positions: (response.positions || []).map((p) => ({
        symbol: p.symbol,
        size: p.size,
        entryPrice: p.entryPrice,
        markPrice: p.markPrice,
        pnl: p.pnl,
        margin: p.margin,
      })),
    };
  }

  async getBalances(): Promise<{ balances: Balance[] }> {
    const request = {
      userId: this.credentials.apiKey || '',
      asset: '',
    };

    interface GrpcBalance {
      asset: string;
      available: number;
      locked: number;
      total: number;
    }

    // Note: gRPC returns single balance, we may need to call multiple times
    // For now, return single balance as array
    const response = await this.promisifyCall<typeof request, GrpcBalance>('getBalance', request);

    return {
      balances: response ? [{
        asset: response.asset,
        available: response.available,
        locked: response.locked,
        total: response.total,
      }] : [],
    };
  }

  async getOrderBook(symbol: string, depth = 20): Promise<OrderBook> {
    const request = { symbol, depth };

    interface GrpcPriceLevel {
      price: number;
      size: number;
      count: number;
    }

    interface GrpcOrderBook {
      symbol: string;
      bids: GrpcPriceLevel[];
      asks: GrpcPriceLevel[];
      timestamp: string;
    }

    const response = await this.promisifyCall<typeof request, GrpcOrderBook>('getOrderBook', request);

    return {
      symbol: response.symbol,
      bids: (response.bids || []).map((l) => ({ price: l.price, size: l.size, count: l.count })),
      asks: (response.asks || []).map((l) => ({ price: l.price, size: l.size, count: l.count })),
      timestamp: parseInt(response.timestamp, 10) || Date.now(),
    };
  }

  async getNodeInfo(): Promise<Record<string, unknown>> {
    return this.promisifyCall('getNodeInfo', {});
  }
}

// ============================================================================
// Client Factory
// ============================================================================

/**
 * Create a trading client based on protocol selection.
 */
export function createClient(
  protocol: Protocol,
  endpoint: string,
  verbose = false
): ITradingClient {
  if (protocol === 'grpc') {
    return new GrpcClient(endpoint, verbose);
  }
  return new LXClient(endpoint, verbose);
}

// ============================================================================
// Interactive REPL
// ============================================================================

function printHelp(): void {
  console.log(`
LX DEX Trading Client Commands:

  place_order <symbol> <side> <type> <price> <size>
    Example: place_order BTC-USD buy limit 50000 0.1

  cancel_order <order_id>
    Example: cancel_order 12345

  get_orderbook <symbol>
    Example: get_orderbook BTC-USD

  get_positions
    Show all open positions

  get_orders
    Show all open orders

  get_balances
    Show account balances

  subscribe <symbol>
    Subscribe to orderbook updates (WebSocket only)

  unsubscribe <symbol>
    Unsubscribe from orderbook updates (WebSocket only)

  auth <api_key> <api_secret>
    Authenticate with credentials

  help
    Show this help message

  quit / exit
    Exit the client
`);
}

async function runInteractive(client: ITradingClient, isWebSocket: boolean): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('LX DEX Trading Client - Type \'help\' for commands');

  const prompt = (): void => {
    rl.question('> ', async (line) => {
      const trimmed = line.trim();
      if (!trimmed) {
        prompt();
        return;
      }

      const parts = trimmed.split(/\s+/);
      const cmd = parts[0].toLowerCase();

      try {
        switch (cmd) {
          case 'help':
            printHelp();
            break;

          case 'quit':
          case 'exit':
            console.log('Goodbye');
            rl.close();
            client.close();
            process.exit(0);

          case 'auth':
            if (parts.length < 3) {
              console.log('Usage: auth <api_key> <api_secret>');
            } else {
              await client.auth(parts[1], parts[2]);
              console.log('Authenticated successfully');
            }
            break;

          case 'place_order':
            if (parts.length < 6) {
              console.log('Usage: place_order <symbol> <side> <type> <price> <size>');
            } else {
              const price = parseFloat(parts[4]);
              const size = parseFloat(parts[5]);
              if (isNaN(price) || isNaN(size)) {
                console.log('Invalid price or size');
              } else {
                const order: Order = {
                  symbol: parts[1],
                  side: parts[2],
                  type: parts[3],
                  price,
                  size,
                };
                const resp = await client.placeOrder(order);
                console.log(JSON.stringify(resp, null, 2));
              }
            }
            break;

          case 'cancel_order':
            if (parts.length < 2) {
              console.log('Usage: cancel_order <order_id>');
            } else {
              const resp = await client.cancelOrder(parts[1]);
              console.log(JSON.stringify(resp, null, 2));
            }
            break;

          case 'get_orderbook':
            if (parts.length < 2) {
              console.log('Usage: get_orderbook <symbol>');
            } else {
              if (isWebSocket) {
                (client as LXClient).subscribe(parts[1]);
                console.log(`Subscribed to ${parts[1]} orderbook`);
              } else {
                const book = await client.getOrderBook(parts[1]);
                console.log(`OrderBook [${book.symbol}]:`);
                console.log(`  Bids: ${book.bids.length} levels`);
                for (const bid of book.bids.slice(0, 5)) {
                  console.log(`    ${bid.price.toFixed(2)} @ ${bid.size.toFixed(4)}`);
                }
                console.log(`  Asks: ${book.asks.length} levels`);
                for (const ask of book.asks.slice(0, 5)) {
                  console.log(`    ${ask.price.toFixed(2)} @ ${ask.size.toFixed(4)}`);
                }
              }
            }
            break;

          case 'get_positions': {
            const posResp = await client.getPositions();
            console.log(JSON.stringify(posResp, null, 2));
            break;
          }

          case 'get_orders': {
            const ordResp = await client.getOrders();
            console.log(JSON.stringify(ordResp, null, 2));
            break;
          }

          case 'get_balances': {
            const balResp = await client.getBalances();
            console.log(JSON.stringify(balResp, null, 2));
            break;
          }

          case 'subscribe':
            if (!isWebSocket) {
              console.log('subscribe is only available with WebSocket protocol');
            } else if (parts.length < 2) {
              console.log('Usage: subscribe <symbol>');
            } else {
              (client as LXClient).subscribe(parts[1]);
              console.log(`Subscribed to ${parts[1]}`);
            }
            break;

          case 'unsubscribe':
            if (!isWebSocket) {
              console.log('unsubscribe is only available with WebSocket protocol');
            } else if (parts.length < 2) {
              console.log('Usage: unsubscribe <symbol>');
            } else {
              (client as LXClient).unsubscribe(parts[1]);
              console.log(`Unsubscribed from ${parts[1]}`);
            }
            break;

          default:
            console.log(`Unknown command: ${cmd}. Type 'help' for commands.`);
        }
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
      }

      prompt();
    });
  };

  // Handle SIGINT for clean exit
  rl.on('close', () => {
    client.close();
    process.exit(0);
  });

  prompt();
}

// ============================================================================
// Main Entry Point
// ============================================================================

async function main(): Promise<void> {
  const program = new Command();

  program
    .name('lx-client')
    .description('LX DEX Trading Client - Programmatic trading via WebSocket or gRPC')
    .version('1.0.0')
    .option('-p, --protocol <protocol>', 'Protocol: ws or grpc', 'ws')
    .option('-u, --url <url>', 'Server URL (ws://host:port or host:port for gRPC)')
    .option('-k, --key <key>', 'API key for authentication')
    .option('-s, --secret <secret>', 'API secret for authentication')
    .option('-i, --interactive', 'Interactive mode')
    .option('-v, --verbose', 'Verbose output');

  // Helper to get endpoint based on protocol
  const getEndpoint = (opts: { protocol?: string; url?: string }): string => {
    if (opts.url) {
      return opts.url;
    }
    return opts.protocol === 'grpc' ? 'localhost:50051' : 'ws://localhost:8081';
  };

  // Helper to create the appropriate client
  const makeClient = (opts: { protocol?: string; url?: string; verbose?: boolean }): ITradingClient => {
    const endpoint = getEndpoint(opts);
    const protocol = (opts.protocol || 'ws') as Protocol;
    return createClient(protocol, endpoint, opts.verbose);
  };

  // Subcommand: place_order
  program
    .command('place_order <symbol> <side> <type> <price> <size>')
    .description('Place a new order')
    .action(async (symbol, side, type, price, size) => {
      const opts = program.opts();
      const client = makeClient(opts);

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        const order: Order = {
          symbol,
          side,
          type,
          price: parseFloat(price),
          size: parseFloat(size),
        };

        const resp = await client.placeOrder(order);
        console.log(JSON.stringify(resp, null, 2));
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Subcommand: cancel_order
  program
    .command('cancel_order <order_id>')
    .description('Cancel an order')
    .action(async (orderId) => {
      const opts = program.opts();
      const client = makeClient(opts);

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        const resp = await client.cancelOrder(orderId);
        console.log(JSON.stringify(resp, null, 2));
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Subcommand: get_orderbook
  program
    .command('get_orderbook <symbol>')
    .description('Get orderbook for a symbol')
    .action(async (symbol) => {
      const opts = program.opts();
      const client = makeClient(opts);
      const isWs = opts.protocol !== 'grpc';

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        if (isWs) {
          const wsClient = client as LXClient;
          wsClient.subscribe(symbol);

          // Wait for orderbook data
          await new Promise<void>((resolve) => {
            const timeout = setTimeout(() => {
              resolve();
            }, 2000);

            wsClient.setMessageCallback((msg) => {
              if (msg.type === 'orderbook') {
                wsClient.printMessage(msg);
                clearTimeout(timeout);
                resolve();
              }
            });
          });
        } else {
          const book = await client.getOrderBook(symbol);
          console.log(`OrderBook [${book.symbol}]:`);
          console.log(`  Bids: ${book.bids.length} levels`);
          for (const bid of book.bids.slice(0, 10)) {
            console.log(`    ${bid.price.toFixed(2)} @ ${bid.size.toFixed(4)}`);
          }
          console.log(`  Asks: ${book.asks.length} levels`);
          for (const ask of book.asks.slice(0, 10)) {
            console.log(`    ${ask.price.toFixed(2)} @ ${ask.size.toFixed(4)}`);
          }
        }
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Subcommand: get_positions
  program
    .command('get_positions')
    .description('Get all positions')
    .action(async () => {
      const opts = program.opts();
      const client = makeClient(opts);

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        const resp = await client.getPositions();
        console.log(JSON.stringify(resp, null, 2));
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Subcommand: get_orders
  program
    .command('get_orders')
    .description('Get all open orders')
    .action(async () => {
      const opts = program.opts();
      const client = makeClient(opts);

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        const resp = await client.getOrders();
        console.log(JSON.stringify(resp, null, 2));
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Subcommand: get_balances
  program
    .command('get_balances')
    .description('Get account balances')
    .action(async () => {
      const opts = program.opts();
      const client = makeClient(opts);

      try {
        await client.connect();

        if (opts.key && opts.secret) {
          await client.auth(opts.key, opts.secret);
        }

        const resp = await client.getBalances();
        console.log(JSON.stringify(resp, null, 2));
      } catch (e) {
        console.error(`Error: ${(e as Error).message}`);
        process.exit(1);
      } finally {
        client.close();
      }
    });

  // Parse arguments
  program.parse(process.argv);

  const opts = program.opts();

  // If no subcommand and no -i flag, but has positional args, fall through
  // If -i flag or no args, run interactive mode
  if (opts.interactive || process.argv.length <= 2) {
    const client = makeClient(opts);
    const isWs = opts.protocol !== 'grpc';

    try {
      await client.connect();
      if (opts.verbose) {
        console.log(`Connected to LX DEX via ${opts.protocol || 'ws'}`);
      }

      if (opts.key && opts.secret) {
        await client.auth(opts.key, opts.secret);
        if (opts.verbose) {
          console.log('Authenticated');
        }
      }

      await runInteractive(client, isWs);
    } catch (e) {
      console.error(`Connection failed: ${(e as Error).message}`);
      process.exit(1);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

// Export for library usage
export { createClient as default };
