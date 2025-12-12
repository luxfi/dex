/**
 * LX Gateway Connector
 *
 * Main connector class implementing Router, AMM, and CLMM schemas
 * for the LX decentralized exchange.
 *
 * Features:
 * - Ultra-low latency order matching (<100ns)
 * - Multiple orderbook backends (Pure Go, C++, GPU, FPGA)
 * - Central Limit Order Book (CLOB) with AMM integration
 * - Concentrated liquidity market making (CLMM)
 * - Cross-chain bridge support via Warp messaging
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import WebSocket from 'ws';
import { EventEmitter } from 'events';
import {
  LXDexConfig,
  getLXDexConfig,
  getNetworkEndpoints,
} from './lxdex.config';
import {
  TokenInfo,
  TradingPair,
  LXDexQuoteSwapRequest,
  LXDexQuoteSwapResponse,
  LXDexExecuteSwapRequest,
  LXDexExecuteQuoteRequest,
  LXDexSwapResponse,
  LXDexPoolInfoRequest,
  LXDexPoolInfoResponse,
  LXDexPositionInfoRequest,
  LXDexPositionInfoResponse,
  LXDexAddLiquidityRequest,
  LXDexAddLiquidityResponse,
  LXDexRemoveLiquidityRequest,
  LXDexRemoveLiquidityResponse,
  LXDexCLMMPoolInfoRequest,
  LXDexCLMMPoolInfoResponse,
  LXDexPositionsOwnedRequest,
  LXDexPositionsOwnedResponse,
  LXDexQuotePositionRequest,
  LXDexQuotePositionResponse,
  LXDexOpenPositionRequest,
  LXDexOpenPositionResponse,
  LXDexClosePositionRequest,
  LXDexClosePositionResponse,
  LXDexCollectFeesRequest,
  LXDexCollectFeesResponse,
  LXDexPlaceOrderRequest,
  LXDexPlaceOrderResponse,
  LXDexCancelOrderRequest,
  LXDexCancelOrderResponse,
  LXDexGetOrdersRequest,
  LXDexGetOrdersResponse,
  LXDexOrderBookRequest,
  LXDexOrderBookResponse,
} from './schemas';

// Error types
export class LXDexError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'LXDexError';
  }
}

export class InsufficientLiquidityError extends LXDexError {
  constructor(message: string = 'Insufficient liquidity for this trade') {
    super(message, 'INSUFFICIENT_LIQUIDITY');
  }
}

export class SlippageExceededError extends LXDexError {
  constructor(message: string = 'Price moved beyond slippage tolerance') {
    super(message, 'SLIPPAGE_EXCEEDED');
  }
}

export class OrderNotFoundError extends LXDexError {
  constructor(orderId: string) {
    super(`Order not found: ${orderId}`, 'ORDER_NOT_FOUND');
  }
}

/**
 * LX Connector
 *
 * Singleton class managing connections to LX.
 * Supports Router (swaps), AMM (liquidity), CLMM (concentrated liquidity),
 * and Order Book (limit orders) trading types.
 */
export class LXDex extends EventEmitter {
  private static _instances: Map<string, LXDex> = new Map();

  private readonly config: LXDexConfig;
  private readonly network: string;
  private readonly httpClient: AxiosInstance;
  private wsClient: WebSocket | null = null;
  private wsConnected: boolean = false;
  private tokenCache: Map<string, TokenInfo> = new Map();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;

  private constructor(network: string, config: LXDexConfig) {
    super();
    this.network = network;
    this.config = config;

    const endpoints = getNetworkEndpoints(network);

    this.httpClient = axios.create({
      baseURL: endpoints.api,
      timeout: config.connectionTimeout,
      headers: {
        'Content-Type': 'application/json',
        'X-Client': 'hummingbot-gateway',
        'X-Client-Version': '1.0.0',
      },
    });

    // Add request interceptor for logging
    this.httpClient.interceptors.request.use((request) => {
      console.log(`[LXDex] ${request.method?.toUpperCase()} ${request.url}`);
      return request;
    });

    // Add response interceptor for error handling
    this.httpClient.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => this.handleError(error)
    );
  }

  /**
   * Get or create a LXDex instance for the specified network
   */
  public static getInstance(
    network: string = 'mainnet',
    configOverrides?: Partial<LXDexConfig>
  ): LXDex {
    const key = network;

    if (!LXDex._instances.has(key)) {
      const config = getLXDexConfig(network, configOverrides);
      LXDex._instances.set(key, new LXDex(network, config));
    }

    return LXDex._instances.get(key)!;
  }

  /**
   * Check if the connector is ready
   */
  public async ready(): Promise<boolean> {
    try {
      const response = await this.httpClient.get('/health');
      return response.data?.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Get current network
   */
  public getNetwork(): string {
    return this.network;
  }

  /**
   * Get configuration
   */
  public getConfig(): LXDexConfig {
    return { ...this.config };
  }

  // ============================================================================
  // Router Schema - Swap Operations
  // ============================================================================

  /**
   * Get a swap quote
   */
  public async getQuote(
    request: LXDexQuoteSwapRequest
  ): Promise<LXDexQuoteSwapResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getQuote',
      params: {
        baseToken: request.baseToken,
        quoteToken: request.quoteToken,
        amount: request.amount,
        side: request.side,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
        maxHops: request.maxHops ?? this.config.maxHops,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Execute a swap directly
   */
  public async executeSwap(
    request: LXDexExecuteSwapRequest
  ): Promise<LXDexSwapResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_executeSwap',
      params: {
        walletAddress: request.walletAddress,
        baseToken: request.baseToken,
        quoteToken: request.quoteToken,
        amount: request.amount,
        side: request.side,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
        maxHops: request.maxHops ?? this.config.maxHops,
        gasPrice: request.gasPrice,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Execute a previously obtained quote
   */
  public async executeQuote(
    request: LXDexExecuteQuoteRequest
  ): Promise<LXDexSwapResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_executeQuote',
      params: {
        walletAddress: request.walletAddress,
        quoteId: request.quoteId,
        gasPrice: request.gasPrice,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  // ============================================================================
  // AMM Schema - Liquidity Pool Operations
  // ============================================================================

  /**
   * Get pool information
   */
  public async getPoolInfo(
    request: LXDexPoolInfoRequest
  ): Promise<LXDexPoolInfoResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getPoolInfo',
      params: {
        tokenA: request.tokenA,
        tokenB: request.tokenB,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Get LP position information
   */
  public async getPositionInfo(
    request: LXDexPositionInfoRequest
  ): Promise<LXDexPositionInfoResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getPositionInfo',
      params: {
        walletAddress: request.walletAddress,
        poolAddress: request.poolAddress,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Add liquidity to a pool
   */
  public async addLiquidity(
    request: LXDexAddLiquidityRequest
  ): Promise<LXDexAddLiquidityResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_addLiquidity',
      params: {
        walletAddress: request.walletAddress,
        tokenA: request.tokenA,
        tokenB: request.tokenB,
        amountA: request.amountA,
        amountB: request.amountB,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Remove liquidity from a pool
   */
  public async removeLiquidity(
    request: LXDexRemoveLiquidityRequest
  ): Promise<LXDexRemoveLiquidityResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_removeLiquidity',
      params: {
        walletAddress: request.walletAddress,
        poolAddress: request.poolAddress,
        liquidity: request.liquidity,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  // ============================================================================
  // CLMM Schema - Concentrated Liquidity Operations
  // ============================================================================

  /**
   * Get CLMM pool information
   */
  public async getCLMMPoolInfo(
    request: LXDexCLMMPoolInfoRequest
  ): Promise<LXDexCLMMPoolInfoResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getCLMMPoolInfo',
      params: {
        tokenA: request.tokenA,
        tokenB: request.tokenB,
        fee: request.fee,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Get all CLMM positions owned by an address
   */
  public async getPositionsOwned(
    request: LXDexPositionsOwnedRequest
  ): Promise<LXDexPositionsOwnedResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getPositionsOwned',
      params: {
        walletAddress: request.walletAddress,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Get a quote for opening a CLMM position
   */
  public async quotePosition(
    request: LXDexQuotePositionRequest
  ): Promise<LXDexQuotePositionResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_quotePosition',
      params: {
        tokenA: request.tokenA,
        tokenB: request.tokenB,
        fee: request.fee,
        tickLower: request.tickLower,
        tickUpper: request.tickUpper,
        amountA: request.amountA,
        amountB: request.amountB,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Open a new CLMM position
   */
  public async openPosition(
    request: LXDexOpenPositionRequest
  ): Promise<LXDexOpenPositionResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_openPosition',
      params: {
        walletAddress: request.walletAddress,
        tokenA: request.tokenA,
        tokenB: request.tokenB,
        fee: request.fee,
        tickLower: request.tickLower,
        tickUpper: request.tickUpper,
        amountA: request.amountA,
        amountB: request.amountB,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Close a CLMM position
   */
  public async closePosition(
    request: LXDexClosePositionRequest
  ): Promise<LXDexClosePositionResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_closePosition',
      params: {
        walletAddress: request.walletAddress,
        tokenId: request.tokenId,
        slippagePct: request.slippagePct ?? this.config.slippagePct,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Collect fees from a CLMM position
   */
  public async collectFees(
    request: LXDexCollectFeesRequest
  ): Promise<LXDexCollectFeesResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_collectFees',
      params: {
        walletAddress: request.walletAddress,
        tokenId: request.tokenId,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  // ============================================================================
  // Order Book Schema - Central Limit Order Book Operations
  // ============================================================================

  /**
   * Place an order on the order book
   */
  public async placeOrder(
    request: LXDexPlaceOrderRequest
  ): Promise<LXDexPlaceOrderResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_placeOrder',
      params: {
        walletAddress: request.walletAddress,
        symbol: request.symbol,
        side: request.side,
        type: request.type,
        price: request.price,
        size: request.size,
        timeInForce: request.timeInForce ?? 'GTC',
        clientOrderId: request.clientOrderId,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(
    request: LXDexCancelOrderRequest
  ): Promise<LXDexCancelOrderResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_cancelOrder',
      params: {
        walletAddress: request.walletAddress,
        orderId: request.orderId,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Get orders for an address
   */
  public async getOrders(
    request: LXDexGetOrdersRequest
  ): Promise<LXDexGetOrdersResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getOrders',
      params: {
        walletAddress: request.walletAddress,
        symbol: request.symbol,
        status: request.status,
        limit: request.limit ?? 100,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  /**
   * Get the current order book
   */
  public async getOrderBook(
    request: LXDexOrderBookRequest
  ): Promise<LXDexOrderBookResponse> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getOrderBook',
      params: {
        symbol: request.symbol,
        depth: request.depth ?? 100,
      },
      id: Date.now(),
    });

    return response.data.result;
  }

  // ============================================================================
  // WebSocket - Real-time Updates
  // ============================================================================

  /**
   * Connect to WebSocket for real-time updates
   */
  public async connectWebSocket(): Promise<void> {
    if (this.wsConnected) {
      return;
    }

    const endpoints = getNetworkEndpoints(this.network);

    return new Promise((resolve, reject) => {
      this.wsClient = new WebSocket(endpoints.ws);

      this.wsClient.on('open', () => {
        this.wsConnected = true;
        this.reconnectAttempts = 0;
        console.log('[LXDex] WebSocket connected');
        this.emit('connected');
        resolve();
      });

      this.wsClient.on('message', (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('[LXDex] Failed to parse WebSocket message:', error);
        }
      });

      this.wsClient.on('close', () => {
        this.wsConnected = false;
        console.log('[LXDex] WebSocket disconnected');
        this.emit('disconnected');
        this.attemptReconnect();
      });

      this.wsClient.on('error', (error) => {
        console.error('[LXDex] WebSocket error:', error);
        this.emit('error', error);
        reject(error);
      });
    });
  }

  /**
   * Subscribe to order book updates
   */
  public subscribeOrderBook(symbol: string): void {
    this.sendWebSocketMessage({
      type: 'subscribe',
      channel: 'orderbook',
      symbol,
    });
  }

  /**
   * Subscribe to trade updates
   */
  public subscribeTrades(symbol: string): void {
    this.sendWebSocketMessage({
      type: 'subscribe',
      channel: 'trades',
      symbol,
    });
  }

  /**
   * Subscribe to order updates for a wallet
   */
  public subscribeOrders(walletAddress: string): void {
    this.sendWebSocketMessage({
      type: 'subscribe',
      channel: 'orders',
      walletAddress,
    });
  }

  /**
   * Disconnect WebSocket
   */
  public disconnectWebSocket(): void {
    if (this.wsClient) {
      this.wsClient.close();
      this.wsClient = null;
      this.wsConnected = false;
    }
  }

  /**
   * Close the connector and cleanup resources
   */
  public async close(): Promise<void> {
    this.disconnectWebSocket();
    this.tokenCache.clear();
    LXDex._instances.delete(this.network);
  }

  // ============================================================================
  // Token Operations
  // ============================================================================

  /**
   * Get token information by symbol or address
   */
  public async getToken(symbolOrAddress: string): Promise<TokenInfo | null> {
    // Check cache first
    if (this.tokenCache.has(symbolOrAddress)) {
      return this.tokenCache.get(symbolOrAddress)!;
    }

    try {
      const response = await this.httpClient.post('/rpc', {
        jsonrpc: '2.0',
        method: 'lx_getToken',
        params: { token: symbolOrAddress },
        id: Date.now(),
      });

      const token = response.data.result as TokenInfo;
      this.tokenCache.set(symbolOrAddress, token);
      this.tokenCache.set(token.address, token);
      this.tokenCache.set(token.symbol, token);

      return token;
    } catch {
      return null;
    }
  }

  /**
   * Get all available trading pairs
   */
  public async getTradingPairs(): Promise<TradingPair[]> {
    const response = await this.httpClient.post('/rpc', {
      jsonrpc: '2.0',
      method: 'lx_getTradingPairs',
      params: {},
      id: Date.now(),
    });

    return response.data.result.pairs;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private handleError(error: AxiosError): never {
    const responseData = error.response?.data as { error?: { code?: string; message?: string } } | undefined;

    if (responseData?.error) {
      const { code, message } = responseData.error;

      switch (code) {
        case 'INSUFFICIENT_LIQUIDITY':
          throw new InsufficientLiquidityError(message);
        case 'SLIPPAGE_EXCEEDED':
          throw new SlippageExceededError(message);
        case 'ORDER_NOT_FOUND':
          throw new OrderNotFoundError(message || 'unknown');
        default:
          throw new LXDexError(
            message || 'Unknown error',
            code || 'UNKNOWN',
            responseData
          );
      }
    }

    throw new LXDexError(
      error.message || 'Request failed',
      'REQUEST_FAILED',
      { status: error.response?.status }
    );
  }

  private handleWebSocketMessage(message: {
    type: string;
    channel?: string;
    data?: unknown;
  }): void {
    switch (message.type) {
      case 'orderbook':
        this.emit('orderbook', message.data);
        break;
      case 'trade':
        this.emit('trade', message.data);
        break;
      case 'order':
        this.emit('order', message.data);
        break;
      case 'position':
        this.emit('position', message.data);
        break;
      default:
        console.log('[LXDex] Unknown message type:', message.type);
    }
  }

  private sendWebSocketMessage(message: object): void {
    if (this.wsClient && this.wsConnected) {
      this.wsClient.send(JSON.stringify(message));
    } else {
      console.warn('[LXDex] WebSocket not connected, message not sent');
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

      console.log(
        `[LXDex] Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
      );

      setTimeout(() => {
        this.connectWebSocket().catch((error) => {
          console.error('[LXDex] Reconnect failed:', error);
        });
      }, delay);
    } else {
      console.error('[LXDex] Max reconnect attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }
}

export default LXDex;
