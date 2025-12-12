/**
 * Unified Liquidity Arbitrage.
 *
 * Since LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks),
 * it becomes the price ORACLE. Other venues are always stale by comparison.
 *
 * Architecture:
 * 1. LX DEX prices are the TRUTH (most current)
 * 2. Other venues (CEX, external DEX) are STALE
 * 3. Arbitrage = exploiting stale venues before they catch up
 * 4. LX always wins because it sees/moves prices first
 *
 * NO SMART CONTRACTS - just coordinated trades through unified SDK.
 */

import { Decimal } from 'decimal.js';
import type { Client } from '../client.js';
import type { Order, Side, VenueInfo } from '../types.js';
import type {
  UnifiedArbConfig,
  UnifiedArbStats,
  UnifiedExecution,
  UnifiedOpportunity,
} from './types.js';

/** Aggregated level from the orderbook */
interface AggregatedLevel {
  price: Decimal;
  quantity: Decimal;
  venue: string;
  timestamp: number;
}

/**
 * Trading client interface for arbitrage.
 */
export interface ArbitrageTradingClient {
  aggregatedOrderbook(symbol: string): Promise<AggregatedBook>;
  placeOrder(request: {
    symbol: string;
    side: Side;
    orderType: 'market' | 'limit';
    quantity: Decimal;
    price?: Decimal;
    venue: string;
  }): Promise<Order>;
  listVenues(): VenueInfo[];
}

interface AggregatedBook {
  symbol: string;
  bids: AggregatedLevel[];
  asks: AggregatedLevel[];
}

export class UnifiedArbitrage {
  private totalPnL = new Decimal(0);
  private executions: UnifiedExecution[] = [];
  private opportunityCallbacks: ((opp: UnifiedOpportunity) => void)[] = [];
  private running = false;
  private scanTimer?: ReturnType<typeof setInterval>;
  private opportunityQueue: UnifiedOpportunity[] = [];

  constructor(
    private readonly client: ArbitrageTradingClient,
    public readonly config: UnifiedArbConfig
  ) {}

  /**
   * Start the arbitrage system.
   */
  start(): void {
    if (this.running) return;
    this.running = true;

    // Start scan loop
    this.scanTimer = setInterval(() => {
      this.scan();
    }, this.config.scanIntervalMs);

    // Start execute loop
    this.executeLoop();
  }

  /**
   * Stop the arbitrage system.
   */
  stop(): void {
    this.running = false;
    if (this.scanTimer) {
      clearInterval(this.scanTimer);
      this.scanTimer = undefined;
    }
  }

  /**
   * Subscribe to opportunity events.
   */
  onOpportunity(callback: (opp: UnifiedOpportunity) => void): void {
    this.opportunityCallbacks.push(callback);
  }

  /**
   * Get arbitrage statistics.
   */
  getStats(): UnifiedArbStats {
    const successful = this.executions.filter(
      (e) => e.status === 'completed' && e.actualProfit.gt(0)
    ).length;

    const winRate = this.executions.length > 0
      ? successful / this.executions.length
      : 0;

    return {
      totalExecutions: this.executions.length,
      successfulExecutions: successful,
      totalPnL: this.totalPnL,
      winRate,
    };
  }

  /**
   * Scan for opportunities.
   */
  private async scan(): Promise<void> {
    for (const symbol of this.config.symbols) {
      const opp = await this.findOpportunity(symbol);
      if (opp && opp.netProfit.gt(this.config.minProfit)) {
        this.opportunityQueue.push(opp);

        // Emit to callbacks
        for (const callback of this.opportunityCallbacks) {
          try {
            callback(opp);
          } catch (err) {
            console.error('Error in opportunity callback:', err);
          }
        }
      }
    }
  }

  /**
   * Find arbitrage opportunity for a symbol.
   */
  private async findOpportunity(symbol: string): Promise<UnifiedOpportunity | null> {
    try {
      const book = await this.client.aggregatedOrderbook(symbol);

      const bestBid = book.bids[0];
      const bestAsk = book.asks[0];

      if (!bestBid || !bestAsk) return null;

      // Cross-venue arbitrage: bid on one venue > ask on another
      if (bestBid.price.lte(bestAsk.price)) return null;

      const spread = bestBid.price.minus(bestAsk.price);
      const spreadBps = spread.div(bestAsk.price).mul(10000);

      if (spreadBps.lt(this.config.minSpreadBps)) return null;

      let maxSize = Decimal.min(bestBid.quantity, bestAsk.quantity);
      maxSize = Decimal.min(maxSize, this.config.maxPositionSize);

      const grossProfit = spread.mul(maxSize);
      const totalFees = bestAsk.price.mul(maxSize).mul(0.002); // ~0.2% total fees
      const netProfit = grossProfit.minus(totalFees);

      const now = Date.now();

      return {
        id: `arb-${symbol}-${now}`,
        symbol,
        timestamp: now,
        expiresAt: now + 5000,
        buyVenue: bestAsk.venue,
        buyPrice: bestAsk.price,
        buySize: bestAsk.quantity,
        sellVenue: bestBid.venue,
        sellPrice: bestBid.price,
        sellSize: bestBid.quantity,
        spread,
        spreadBps,
        maxSize,
        grossProfit,
        estFees: totalFees,
        netProfit,
        confidence: 0.8,
        latency: now - bestAsk.timestamp,
      };
    } catch {
      return null;
    }
  }

  /**
   * Execute loop - process opportunities from queue.
   */
  private async executeLoop(): Promise<void> {
    while (this.running) {
      const opp = this.opportunityQueue.shift();
      if (opp) {
        await this.execute(opp);
      } else {
        // Wait a bit before checking again
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }
  }

  /**
   * Execute an arbitrage opportunity.
   */
  private async execute(opp: UnifiedOpportunity): Promise<void> {
    const now = Date.now();
    if (now > opp.expiresAt) return;

    const exec: UnifiedExecution = {
      id: opp.id,
      opportunity: opp,
      startTime: now,
      endTime: 0,
      status: 'executing',
      actualProfit: new Decimal(0),
      fees: new Decimal(0),
    };

    try {
      // Execute both legs simultaneously
      const [buyResult, sellResult] = await Promise.all([
        this.client.placeOrder({
          symbol: opp.symbol,
          side: 'buy' as Side,
          orderType: 'limit',
          quantity: opp.maxSize,
          price: opp.buyPrice,
          venue: opp.buyVenue,
        }),
        this.client.placeOrder({
          symbol: opp.symbol,
          side: 'sell' as Side,
          orderType: 'limit',
          quantity: opp.maxSize,
          price: opp.sellPrice,
          venue: opp.sellVenue,
        }),
      ]);

      exec.endTime = Date.now();
      exec.buyOrderId = buyResult.orderId;
      exec.sellOrderId = sellResult.orderId;

      // Calculate actual profit
      if (buyResult.averagePrice && sellResult.averagePrice) {
        const buyValue = buyResult.averagePrice.mul(buyResult.filledQuantity);
        const sellValue = sellResult.averagePrice.mul(sellResult.filledQuantity);
        exec.actualProfit = sellValue.minus(buyValue);

        // Subtract fees
        for (const fee of [...buyResult.fees, ...sellResult.fees]) {
          exec.fees = exec.fees.plus(fee.amount);
        }
        exec.actualProfit = exec.actualProfit.minus(exec.fees);
      }

      exec.status = 'completed';
    } catch (err) {
      exec.endTime = Date.now();
      exec.status = 'failed';
      exec.error = err instanceof Error ? err : new Error(String(err));
    }

    // Update stats
    this.totalPnL = this.totalPnL.plus(exec.actualProfit);
    this.executions.push(exec);
  }
}

/**
 * Create a UnifiedArbitrage instance from a Client.
 */
export function createUnifiedArbitrage(
  client: Client,
  config: UnifiedArbConfig
): UnifiedArbitrage {
  // Adapt the Client to the ArbitrageTradingClient interface
  const adapter: ArbitrageTradingClient = {
    async aggregatedOrderbook(symbol: string): Promise<AggregatedBook> {
      const book = await client.aggregatedOrderbook(symbol);
      const bids: AggregatedLevel[] = [];
      const asks: AggregatedLevel[] = [];

      // Convert Map<string, VenueQuantity[]> to flat array
      for (const [priceStr, venues] of book.bids.entries()) {
        for (const vq of venues) {
          bids.push({
            price: new Decimal(priceStr),
            quantity: vq.quantity,
            venue: vq.venue,
            timestamp: Date.now(),
          });
        }
      }
      for (const [priceStr, venues] of book.asks.entries()) {
        for (const vq of venues) {
          asks.push({
            price: new Decimal(priceStr),
            quantity: vq.quantity,
            venue: vq.venue,
            timestamp: Date.now(),
          });
        }
      }

      // Sort bids descending, asks ascending
      bids.sort((a, b) => b.price.comparedTo(a.price));
      asks.sort((a, b) => a.price.comparedTo(b.price));

      return { symbol: book.symbol, bids, asks };
    },
    async placeOrder(request) {
      return client.placeOrder({
        ...request,
        orderType: request.orderType === 'market' ? 'market' : 'limit',
        timeInForce: 'IOC' as any,
        reduceOnly: false,
        postOnly: false,
        clientOrderId: crypto.randomUUID(),
      } as any);
    },
    listVenues(): VenueInfo[] {
      return client.listVenues();
    },
  };

  return new UnifiedArbitrage(adapter, config);
}
