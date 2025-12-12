/**
 * Unified trading client.
 */

import { Decimal } from 'decimal.js';
import { CcxtAdapter, HummingbotAdapter, LxAmmAdapter, LxDexAdapter, type VenueAdapter } from './adapters/index.js';
import type { Config } from './config.js';
import { AggregatedOrderbook, Orderbook } from './orderbook.js';
import {
  createLimitOrder,
  createMarketOrder,
  Side,
  type AggregatedBalance,
  type Balance,
  type LiquidityResult,
  type LpPosition,
  type Order,
  type OrderRequest,
  type PoolInfo,
  type SwapQuote,
  type Ticker,
  type Trade,
  type VenueInfo,
} from './types.js';

/**
 * Unified trading client that abstracts multiple venues.
 *
 * Provides a single interface for:
 * - Native LX DEX and LX AMM
 * - CCXT exchanges (Binance, MEXC, OKX, etc.)
 * - Hummingbot Gateway connectors
 *
 * @example
 * ```typescript
 * const config = Config.fromObject(configData);
 * const client = new Client(config);
 * await client.connect();
 *
 * // Get aggregated orderbook
 * const book = await client.aggregatedOrderbook('BTC-USDC');
 *
 * // Smart order routing
 * const order = await client.buy('BTC-USDC', '0.1');
 *
 * // Target specific venue
 * const order = await client.buy('BTC-USDC', '0.1', 'binance');
 * ```
 */
export class Client {
  private venues: Map<string, VenueAdapter> = new Map();
  private defaultVenue: string | undefined;

  constructor(public readonly config: Config) {}

  /**
   * Connect to all configured venues.
   */
  async connect(): Promise<void> {
    // Native venues
    for (const [name, cfg] of this.config.native) {
      const adapter =
        cfg.venueType === 'amm' ? new LxAmmAdapter(name, cfg) : new LxDexAdapter(name, cfg);
      await adapter.connect();
      this.venues.set(name, adapter);
    }

    // CCXT exchanges
    for (const [name, cfg] of this.config.ccxt) {
      const adapter = new CcxtAdapter(name, cfg);
      await adapter.connect();
      this.venues.set(name, adapter);
    }

    // Hummingbot gateways
    for (const [name, cfg] of this.config.hummingbot) {
      const adapter = new HummingbotAdapter(name, cfg);
      await adapter.connect();
      this.venues.set(name, adapter);
    }

    // Set default venue
    if (this.config.general.venuePriority.length > 0) {
      this.defaultVenue = this.config.general.venuePriority[0];
    } else if (this.venues.size > 0) {
      this.defaultVenue = this.venues.keys().next().value;
    }
  }

  /**
   * Disconnect from all venues.
   */
  async disconnect(): Promise<void> {
    for (const adapter of this.venues.values()) {
      await adapter.disconnect();
    }
    this.venues.clear();
  }

  /**
   * Get specific venue adapter.
   */
  venue(name: string): VenueAdapter | undefined {
    return this.venues.get(name);
  }

  /**
   * List connected venues.
   */
  listVenues(): VenueInfo[] {
    return Array.from(this.venues.values()).map((adapter) => adapter.info());
  }

  // ===========================================================================
  // Market Data
  // ===========================================================================

  /**
   * Get orderbook from a venue.
   */
  async orderbook(symbol: string, venue?: string): Promise<Orderbook> {
    const venueName = venue ?? this.defaultVenue;
    if (!venueName) {
      throw new Error('No venue specified and no default venue');
    }

    const adapter = this.venues.get(venueName);
    if (!adapter) {
      throw new Error(`Venue not found: ${venueName}`);
    }

    return adapter.getOrderbook(symbol);
  }

  /**
   * Get orderbook aggregated from all venues.
   */
  async aggregatedOrderbook(symbol: string): Promise<AggregatedOrderbook> {
    const agg = new AggregatedOrderbook(symbol);

    for (const adapter of this.venues.values()) {
      if (adapter.capabilities.orderbook) {
        try {
          const book = await adapter.getOrderbook(symbol, 20);
          agg.addOrderbook(book);
        } catch {
          // Skip venues that don't support this pair
        }
      }
    }

    return agg;
  }

  /**
   * Get ticker from a venue.
   */
  async ticker(symbol: string, venue?: string): Promise<Ticker> {
    const venueName = venue ?? this.defaultVenue;
    if (!venueName) {
      throw new Error('No venue specified and no default venue');
    }

    const adapter = this.venues.get(venueName);
    if (!adapter) {
      throw new Error(`Venue not found: ${venueName}`);
    }

    return adapter.getTicker(symbol);
  }

  /**
   * Get tickers from all venues.
   */
  async tickers(symbol: string): Promise<Ticker[]> {
    const results: Ticker[] = [];
    for (const adapter of this.venues.values()) {
      try {
        const t = await adapter.getTicker(symbol);
        results.push(t);
      } catch {
        // Skip venues that don't support this pair
      }
    }
    return results;
  }

  // ===========================================================================
  // Account
  // ===========================================================================

  /**
   * Get balances aggregated across all venues.
   */
  async balances(): Promise<AggregatedBalance[]> {
    const byAsset: Map<string, Balance[]> = new Map();

    for (const adapter of this.venues.values()) {
      try {
        for (const balance of await adapter.getBalances()) {
          const existing = byAsset.get(balance.asset) ?? [];
          existing.push(balance);
          byAsset.set(balance.asset, existing);
        }
      } catch {
        // Skip failing venues
      }
    }

    const results: AggregatedBalance[] = [];
    for (const [asset, bals] of byAsset) {
      results.push({
        asset,
        totalFree: bals.reduce((sum, b) => sum.plus(b.free), new Decimal(0)),
        totalLocked: bals.reduce((sum, b) => sum.plus(b.locked), new Decimal(0)),
        byVenue: bals,
      });
    }

    return results;
  }

  /**
   * Get balance for an asset.
   */
  async balance(asset: string, venue?: string): Promise<Balance> {
    const venueName = venue ?? this.defaultVenue;
    if (!venueName) {
      throw new Error('No venue specified');
    }

    const adapter = this.venues.get(venueName);
    if (!adapter) {
      throw new Error(`Venue not found: ${venueName}`);
    }

    return adapter.getBalance(asset);
  }

  // ===========================================================================
  // Order Management
  // ===========================================================================

  /**
   * Place market buy order.
   */
  async buy(
    symbol: string,
    quantity: Decimal | number | string,
    venue?: string,
  ): Promise<Order> {
    const qty = new Decimal(quantity);
    const request = createMarketOrder(symbol, Side.BUY, qty);

    if (venue) {
      request.venue = venue;
      return this.placeOrderOn(request);
    } else if (this.config.general.smartRouting) {
      return this.smartRoute(request);
    }
    return this.placeOrderDefault(request);
  }

  /**
   * Place market sell order.
   */
  async sell(
    symbol: string,
    quantity: Decimal | number | string,
    venue?: string,
  ): Promise<Order> {
    const qty = new Decimal(quantity);
    const request = createMarketOrder(symbol, Side.SELL, qty);

    if (venue) {
      request.venue = venue;
      return this.placeOrderOn(request);
    } else if (this.config.general.smartRouting) {
      return this.smartRoute(request);
    }
    return this.placeOrderDefault(request);
  }

  /**
   * Place limit buy order.
   */
  async limitBuy(
    symbol: string,
    quantity: Decimal | number | string,
    price: Decimal | number | string,
    venue?: string,
  ): Promise<Order> {
    const request = createLimitOrder(symbol, Side.BUY, new Decimal(quantity), new Decimal(price));
    if (venue) {
      request.venue = venue;
    }
    return this.placeOrder(request);
  }

  /**
   * Place limit sell order.
   */
  async limitSell(
    symbol: string,
    quantity: Decimal | number | string,
    price: Decimal | number | string,
    venue?: string,
  ): Promise<Order> {
    const request = createLimitOrder(symbol, Side.SELL, new Decimal(quantity), new Decimal(price));
    if (venue) {
      request.venue = venue;
    }
    return this.placeOrder(request);
  }

  /**
   * Place order.
   */
  async placeOrder(request: OrderRequest): Promise<Order> {
    if (request.venue) {
      return this.placeOrderOn(request);
    }
    return this.placeOrderDefault(request);
  }

  /**
   * Cancel order.
   */
  async cancelOrder(orderId: string, symbol: string, venue: string): Promise<Order> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    return adapter.cancelOrder(orderId, symbol);
  }

  /**
   * Cancel all orders.
   */
  async cancelAllOrders(symbol?: string, venue?: string): Promise<Order[]> {
    if (venue) {
      const adapter = this.venues.get(venue);
      if (!adapter) {
        throw new Error(`Venue not found: ${venue}`);
      }
      return adapter.cancelAllOrders(symbol);
    }

    // Cancel on all venues
    const allOrders: Order[] = [];
    for (const adapter of this.venues.values()) {
      try {
        const orders = await adapter.cancelAllOrders(symbol);
        allOrders.push(...orders);
      } catch {
        // Ignore errors
      }
    }
    return allOrders;
  }

  /**
   * Get all open orders across venues.
   */
  async openOrders(symbol?: string): Promise<Order[]> {
    const allOrders: Order[] = [];
    for (const adapter of this.venues.values()) {
      try {
        const orders = await adapter.getOpenOrders(symbol);
        allOrders.push(...orders);
      } catch {
        // Ignore errors
      }
    }
    return allOrders;
  }

  // ===========================================================================
  // AMM Operations
  // ===========================================================================

  /**
   * Get swap quote.
   */
  async quote(
    baseToken: string,
    quoteToken: string,
    amount: Decimal | number | string,
    isBuy: boolean,
    venue: string,
  ): Promise<SwapQuote> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.getSwapQuote) {
      throw new Error(`Venue does not support swap quotes: ${venue}`);
    }
    return adapter.getSwapQuote(baseToken, quoteToken, new Decimal(amount), isBuy);
  }

  /**
   * Execute swap.
   */
  async swap(
    baseToken: string,
    quoteToken: string,
    amount: Decimal | number | string,
    isBuy: boolean,
    slippage: number | string,
    venue: string,
  ): Promise<Trade> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.executeSwap) {
      throw new Error(`Venue does not support swaps: ${venue}`);
    }
    return adapter.executeSwap(
      baseToken,
      quoteToken,
      new Decimal(amount),
      isBuy,
      new Decimal(slippage),
    );
  }

  /**
   * Get pool information.
   */
  async poolInfo(baseToken: string, quoteToken: string, venue: string): Promise<PoolInfo> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.getPoolInfo) {
      throw new Error(`Venue does not support pool info: ${venue}`);
    }
    return adapter.getPoolInfo(baseToken, quoteToken);
  }

  /**
   * Add liquidity to pool.
   */
  async addLiquidity(
    baseToken: string,
    quoteToken: string,
    baseAmount: Decimal | number | string,
    quoteAmount: Decimal | number | string,
    slippage: number | string,
    venue: string,
  ): Promise<LiquidityResult> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.addLiquidity) {
      throw new Error(`Venue does not support adding liquidity: ${venue}`);
    }
    return adapter.addLiquidity(
      baseToken,
      quoteToken,
      new Decimal(baseAmount),
      new Decimal(quoteAmount),
      new Decimal(slippage),
    );
  }

  /**
   * Remove liquidity from pool.
   */
  async removeLiquidity(
    poolAddress: string,
    liquidityAmount: Decimal | number | string,
    slippage: number | string,
    venue: string,
  ): Promise<LiquidityResult> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.removeLiquidity) {
      throw new Error(`Venue does not support removing liquidity: ${venue}`);
    }
    return adapter.removeLiquidity(poolAddress, new Decimal(liquidityAmount), new Decimal(slippage));
  }

  /**
   * Get LP positions.
   */
  async lpPositions(venue: string): Promise<LpPosition[]> {
    const adapter = this.venues.get(venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${venue}`);
    }
    if (!adapter.getLpPositions) {
      throw new Error(`Venue does not support LP positions: ${venue}`);
    }
    return adapter.getLpPositions();
  }

  // ===========================================================================
  // Internal
  // ===========================================================================

  private async placeOrderDefault(request: OrderRequest): Promise<Order> {
    if (!this.defaultVenue) {
      throw new Error('No default venue');
    }

    const adapter = this.venues.get(this.defaultVenue);
    if (!adapter) {
      throw new Error(`Default venue not found: ${this.defaultVenue}`);
    }

    return adapter.placeOrder(request);
  }

  private async placeOrderOn(request: OrderRequest): Promise<Order> {
    if (!request.venue) {
      throw new Error('No venue specified');
    }

    const adapter = this.venues.get(request.venue);
    if (!adapter) {
      throw new Error(`Venue not found: ${request.venue}`);
    }

    return adapter.placeOrder(request);
  }

  private async smartRoute(request: OrderRequest): Promise<Order> {
    const aggBook = await this.aggregatedOrderbook(request.symbol);

    const best =
      request.side === Side.BUY
        ? aggBook.bestVenueBuy(request.quantity)
        : aggBook.bestVenueSell(request.quantity);

    if (best) {
      request.venue = best.venue;
      return this.placeOrderOn(request);
    }

    // Fallback to default
    return this.placeOrderDefault(request);
  }
}
