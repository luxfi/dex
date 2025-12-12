/**
 * Hummingbot Gateway adapter.
 */

import axios, { type AxiosInstance } from 'axios';
import { Decimal } from 'decimal.js';
import type { HummingbotConfig } from '../config.js';
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
import { BaseAdapter, ammCapabilities, type VenueCapabilities } from './base.js';

// =============================================================================
// Hummingbot Gateway Adapter
// =============================================================================

export class HummingbotAdapter extends BaseAdapter {
  readonly venueType = VenueType.HUMMINGBOT;
  readonly capabilities: VenueCapabilities;

  private client: AxiosInstance | null = null;

  constructor(
    readonly name: string,
    private readonly config: HummingbotConfig,
  ) {
    super();
    this.capabilities = ammCapabilities();
  }

  async connect(): Promise<void> {
    this.client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: 30000,
    });

    const start = Date.now();
    const response = await this.client.get<{ status: string }>('/');
    if (response.data.status !== 'ok') {
      throw new Error('Gateway not ready');
    }
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

    const start = Date.now();
    const response = await this.client.request<T>({
      method,
      url: path,
      data,
    });
    this._latencyMs = Date.now() - start;

    return response.data;
  }

  private buildBody(body: Record<string, unknown>): Record<string, unknown> {
    return {
      ...body,
      chain: this.config.chain,
      network: this.config.network,
      connector: this.config.connector,
      address: this.config.walletAddress,
    };
  }

  async getMarkets(): Promise<MarketInfo[]> {
    const data = await this.request<TokensResponse>('POST', '/amm/tokens', this.buildBody({}));

    const markets: MarketInfo[] = [];
    const tokens = data.tokens ?? [];
    for (let i = 0; i < tokens.length; i++) {
      for (let j = i + 1; j < tokens.length; j++) {
        const t1 = tokens[i];
        const t2 = tokens[j];
        if (t1?.symbol && t2?.symbol) {
          markets.push({
            symbol: `${t1.symbol}-${t2.symbol}`,
            base: t1.symbol,
            quote: t2.symbol,
            pricePrecision: 8,
            quantityPrecision: 8,
            minQuantity: new Decimal(0),
            maxQuantity: undefined,
            minNotional: undefined,
            tickSize: new Decimal('0.00000001'),
            lotSize: new Decimal('0.00000001'),
          });
        }
      }
    }
    return markets;
  }

  async getTicker(symbol: string): Promise<Ticker> {
    const pair = TradingPair.fromSymbol(symbol);
    if (!pair) {
      throw new Error(`Invalid symbol: ${symbol}`);
    }

    const data = await this.request<PriceResponse>(
      'POST',
      '/amm/price',
      this.buildBody({
        base: pair.base,
        quote: pair.quote,
        amount: '1',
        side: 'BUY',
      }),
    );

    const price = data.price ? new Decimal(data.price) : undefined;

    return {
      symbol,
      venue: this.name,
      bid: price,
      ask: price,
      last: price,
      volume24h: undefined,
      high24h: undefined,
      low24h: undefined,
      change24h: undefined,
      timestamp: Date.now(),
    };
  }

  async getOrderbook(_symbol: string, _depth?: number): Promise<Orderbook> {
    throw new Error('Gateway AMM does not have orderbook');
  }

  async getTrades(_symbol: string, _limit?: number): Promise<Trade[]> {
    return []; // Gateway doesn't provide trade history
  }

  async getBalances(): Promise<Balance[]> {
    const data = await this.request<BalancesResponse>('POST', '/chain/balances', this.buildBody({}));

    return Object.entries(data.balances ?? {}).map(([asset, amount]) => ({
      asset,
      venue: this.name,
      free: new Decimal(amount as string | number),
      locked: new Decimal(0),
    }));
  }

  async getBalance(asset: string): Promise<Balance> {
    const balances = await this.getBalances();
    const found = balances.find((b) => b.asset.toLowerCase() === asset.toLowerCase());
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

  async getOpenOrders(_symbol?: string): Promise<Order[]> {
    return []; // AMM doesn't have orders
  }

  async placeOrder(request: OrderRequest): Promise<Order> {
    const pair = TradingPair.fromSymbol(request.symbol);
    if (!pair) {
      throw new Error(`Invalid symbol: ${request.symbol}`);
    }

    const trade = await this.executeSwap(
      pair.base,
      pair.quote,
      request.quantity,
      request.side === Side.BUY,
      new Decimal('0.01'),
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
    throw new Error('Gateway AMM swaps cannot be cancelled');
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
    const data = await this.request<PriceResponse>(
      'POST',
      '/amm/price',
      this.buildBody({
        base: baseToken,
        quote: quoteToken,
        amount: amount.toString(),
        side: isBuy ? 'BUY' : 'SELL',
      }),
    );

    return {
      baseToken,
      quoteToken,
      inputAmount: amount,
      outputAmount: new Decimal(data.expectedAmount ?? 0),
      price: new Decimal(data.price ?? 0),
      priceImpact: new Decimal(0),
      fee: new Decimal(0),
      route: [],
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
    const data = await this.request<TradeResponse>(
      'POST',
      '/amm/trade',
      this.buildBody({
        base: baseToken,
        quote: quoteToken,
        amount: amount.toString(),
        side: isBuy ? 'BUY' : 'SELL',
        limitPrice: '',
        allowedSlippage: `${slippage}/100`,
      }),
    );

    return {
      tradeId: data.txHash ?? '',
      orderId: data.txHash ?? '',
      symbol: `${baseToken}-${quoteToken}`,
      venue: this.name,
      side: isBuy ? Side.BUY : Side.SELL,
      price: new Decimal(data.price ?? 0),
      quantity: amount,
      fee: {
        asset: 'GAS',
        amount: new Decimal(data.gasPrice ?? 0),
      },
      timestamp: Date.now(),
      isMaker: false,
    };
  }

  override async getPoolInfo(baseToken: string, quoteToken: string): Promise<PoolInfo> {
    const data = await this.request<PoolPriceResponse>(
      'POST',
      '/amm/poolPrice',
      this.buildBody({
        token0: baseToken,
        token1: quoteToken,
      }),
    );

    return {
      address: data.token0Address ?? '',
      baseToken,
      quoteToken,
      baseReserve: new Decimal(data.token0Balance ?? 0),
      quoteReserve: new Decimal(data.token1Balance ?? 0),
      totalLiquidity: new Decimal(0),
      feeRate: new Decimal('0.003'),
      apy: undefined,
    };
  }

  override async addLiquidity(
    baseToken: string,
    quoteToken: string,
    baseAmount: Decimal,
    quoteAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult> {
    const data = await this.request<LiquidityResponse>(
      'POST',
      '/amm/liquidity/add',
      this.buildBody({
        token0: baseToken,
        token1: quoteToken,
        amount0: baseAmount.toString(),
        amount1: quoteAmount.toString(),
        allowedSlippage: `${slippage}/100`,
      }),
    );

    return {
      txHash: data.txHash ?? '',
      poolAddress: data.poolAddress ?? '',
      baseAmount,
      quoteAmount,
      lpTokens: new Decimal(0),
      sharePercent: new Decimal(0),
    };
  }

  override async removeLiquidity(
    poolAddress: string,
    liquidityAmount: Decimal,
    slippage: Decimal,
  ): Promise<LiquidityResult> {
    const data = await this.request<LiquidityResponse>(
      'POST',
      '/amm/liquidity/remove',
      this.buildBody({
        tokenId: poolAddress,
        decreasePercent: '100',
        allowedSlippage: `${slippage}/100`,
      }),
    );

    return {
      txHash: data.txHash ?? '',
      poolAddress,
      baseAmount: new Decimal(0),
      quoteAmount: new Decimal(0),
      lpTokens: liquidityAmount,
      sharePercent: new Decimal(0),
    };
  }

  override async getLpPositions(): Promise<LpPosition[]> {
    const data = await this.request<PositionData[]>('POST', '/amm/position', this.buildBody({}));

    if (!Array.isArray(data)) {
      return [];
    }

    return data.map((p) => ({
      poolAddress: p.tokenId ?? '',
      baseToken: p.token0 ?? '',
      quoteToken: p.token1 ?? '',
      lpTokens: new Decimal(0),
      baseAmount: new Decimal(p.amount0 ?? 0),
      quoteAmount: new Decimal(p.amount1 ?? 0),
      sharePercent: new Decimal(0),
      unrealizedPnl: p.unclaimedToken0 ? new Decimal(p.unclaimedToken0) : undefined,
    }));
  }
}

// =============================================================================
// Response Types
// =============================================================================

interface TokensResponse {
  tokens?: { symbol?: string }[];
}

interface PriceResponse {
  price?: string | number;
  expectedAmount?: string | number;
}

interface BalancesResponse {
  balances?: Record<string, string | number>;
}

interface TradeResponse {
  txHash?: string;
  price?: string | number;
  gasPrice?: string | number;
}

interface PoolPriceResponse {
  token0Address?: string;
  token0Balance?: string | number;
  token1Balance?: string | number;
}

interface LiquidityResponse {
  txHash?: string;
  poolAddress?: string;
}

interface PositionData {
  tokenId?: string;
  token0?: string;
  token1?: string;
  amount0?: string | number;
  amount1?: string | number;
  unclaimedToken0?: string | number;
}
