/**
 * Execution algorithms.
 */

import { Decimal } from 'decimal.js';
import type { Client } from './client.js';
import { orderIsOpen, Side, type Order } from './types.js';

/**
 * Sleep utility.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// =============================================================================
// TWAP Executor
// =============================================================================

/**
 * Time-Weighted Average Price execution.
 *
 * Splits a large order into smaller slices executed at regular intervals.
 */
export class TwapExecutor {
  private readonly sliceQty: Decimal;
  private readonly intervalMs: number;

  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly totalQuantity: Decimal,
    durationSeconds: number,
    private readonly numSlices: number,
  ) {
    this.sliceQty = totalQuantity.div(numSlices);
    this.intervalMs = (durationSeconds / numSlices) * 1000;
  }

  /**
   * Execute TWAP strategy.
   */
  async execute(): Promise<Order[]> {
    const orders: Order[] = [];

    for (let i = 0; i < this.numSlices; i++) {
      const remaining = this.totalQuantity.minus(this.sliceQty.mul(i));
      const qty = Decimal.min(this.sliceQty, remaining);

      if (qty.lte(0)) {
        break;
      }

      const order =
        this.side === Side.BUY
          ? await this.client.buy(this.symbol, qty)
          : await this.client.sell(this.symbol, qty);

      orders.push(order);

      if (i < this.numSlices - 1) {
        await sleep(this.intervalMs);
      }
    }

    return orders;
  }
}

// =============================================================================
// VWAP Executor
// =============================================================================

/**
 * Volume-Weighted Average Price execution.
 *
 * Executes based on participation rate of market volume.
 */
export class VwapExecutor {
  private readonly checkIntervalMs = 5000;

  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly totalQuantity: Decimal,
    private readonly participationRate: Decimal, // e.g., 0.1 = 10% of volume
    private readonly maxDurationSeconds: number,
  ) {}

  /**
   * Execute VWAP strategy.
   */
  async execute(): Promise<Order[]> {
    const orders: Order[] = [];
    let remaining = this.totalQuantity;
    let elapsed = 0;

    while (remaining.gt(0) && elapsed < this.maxDurationSeconds * 1000) {
      const ticker = await this.client.ticker(this.symbol);
      const volume = ticker.volume24h ?? new Decimal(1000);

      // Calculate slice based on participation
      const hourlyVolume = volume.div(24);
      const sliceVolume = hourlyVolume
        .mul(this.participationRate)
        .div(3600000 / this.checkIntervalMs);
      const qty = Decimal.min(sliceVolume, remaining);

      if (qty.gt(0)) {
        const order =
          this.side === Side.BUY
            ? await this.client.buy(this.symbol, qty)
            : await this.client.sell(this.symbol, qty);

        orders.push(order);
        remaining = remaining.minus(qty);
      }

      await sleep(this.checkIntervalMs);
      elapsed += this.checkIntervalMs;
    }

    return orders;
  }
}

// =============================================================================
// Iceberg Executor
// =============================================================================

/**
 * Iceberg order execution.
 *
 * Hides large orders by only showing a small visible quantity at a time.
 */
export class IcebergExecutor {
  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly totalQuantity: Decimal,
    private readonly visibleQuantity: Decimal,
    private readonly price: Decimal,
    private readonly venue?: string,
  ) {}

  /**
   * Execute iceberg strategy.
   */
  async execute(): Promise<Order[]> {
    const orders: Order[] = [];
    let remaining = this.totalQuantity;

    while (remaining.gt(0)) {
      const qty = Decimal.min(this.visibleQuantity, remaining);

      const order =
        this.side === Side.BUY
          ? await this.client.limitBuy(this.symbol, qty, this.price, this.venue)
          : await this.client.limitSell(this.symbol, qty, this.price, this.venue);

      // Wait for fill
      let attempts = 0;
      while (orderIsOpen(order) && attempts < 100) {
        await sleep(500);
        attempts++;
        // Note: In production, refresh order status from venue
        break; // Simplified - just continue
      }

      remaining = remaining.minus(order.filledQuantity);
      orders.push(order);
    }

    return orders;
  }
}

// =============================================================================
// Sniper Executor
// =============================================================================

/**
 * Sniper execution - wait for price target then execute immediately.
 */
export class SniperExecutor {
  private readonly checkIntervalMs = 100; // 100ms

  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly quantity: Decimal,
    private readonly targetPrice: Decimal,
    private readonly timeoutSeconds: number,
  ) {}

  /**
   * Execute sniper strategy.
   * Returns the executed order, or undefined if timeout reached.
   */
  async execute(): Promise<Order | undefined> {
    let elapsed = 0;
    const timeoutMs = this.timeoutSeconds * 1000;

    while (elapsed < timeoutMs) {
      const ticker = await this.client.ticker(this.symbol);

      let shouldExecute = false;
      if (this.side === Side.BUY) {
        if (ticker.ask && ticker.ask.lte(this.targetPrice)) {
          shouldExecute = true;
        }
      } else {
        if (ticker.bid && ticker.bid.gte(this.targetPrice)) {
          shouldExecute = true;
        }
      }

      if (shouldExecute) {
        return this.side === Side.BUY
          ? await this.client.buy(this.symbol, this.quantity)
          : await this.client.sell(this.symbol, this.quantity);
      }

      await sleep(this.checkIntervalMs);
      elapsed += this.checkIntervalMs;
    }

    return undefined; // Timeout
  }
}

// =============================================================================
// POV Executor (Percent of Volume)
// =============================================================================

/**
 * Percent of Volume execution.
 *
 * Executes as a percentage of market volume over time.
 */
export class PovExecutor {
  private readonly checkIntervalMs = 10000; // 10 seconds

  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly totalQuantity: Decimal,
    private readonly targetPercent: Decimal, // e.g., 0.05 = 5% of volume
    private readonly maxDurationSeconds: number,
  ) {}

  /**
   * Execute POV strategy.
   */
  async execute(): Promise<Order[]> {
    const orders: Order[] = [];
    let remaining = this.totalQuantity;
    let elapsed = 0;
    let prevVolume = new Decimal(0);

    // Get initial volume
    const initialTicker = await this.client.ticker(this.symbol);
    prevVolume = initialTicker.volume24h ?? new Decimal(0);

    while (remaining.gt(0) && elapsed < this.maxDurationSeconds * 1000) {
      await sleep(this.checkIntervalMs);
      elapsed += this.checkIntervalMs;

      const ticker = await this.client.ticker(this.symbol);
      const currentVolume = ticker.volume24h ?? new Decimal(0);

      // Estimate volume in this interval (rough approximation)
      const volumeDelta = currentVolume.minus(prevVolume).abs();
      prevVolume = currentVolume;

      // Calculate our share
      const ourShare = volumeDelta.mul(this.targetPercent);
      const qty = Decimal.min(ourShare, remaining);

      if (qty.gt(new Decimal('0.0001'))) {
        const order =
          this.side === Side.BUY
            ? await this.client.buy(this.symbol, qty)
            : await this.client.sell(this.symbol, qty);

        orders.push(order);
        remaining = remaining.minus(order.filledQuantity);
      }
    }

    return orders;
  }
}

// =============================================================================
// DCA Executor (Dollar Cost Averaging)
// =============================================================================

/**
 * Dollar Cost Averaging execution.
 *
 * Invests fixed amounts at regular intervals.
 */
export class DcaExecutor {
  constructor(
    private readonly client: Client,
    private readonly symbol: string,
    private readonly side: Side,
    private readonly amountPerInterval: Decimal, // Amount in quote currency
    private readonly intervalMs: number,
    private readonly numIntervals: number,
    private readonly venue?: string,
  ) {}

  /**
   * Execute DCA strategy.
   */
  async execute(): Promise<Order[]> {
    const orders: Order[] = [];

    for (let i = 0; i < this.numIntervals; i++) {
      // Get current price to calculate quantity
      const ticker = await this.client.ticker(this.symbol, this.venue);
      const price = ticker.last ?? ticker.ask ?? ticker.bid;

      if (!price || price.isZero()) {
        continue;
      }

      const qty = this.amountPerInterval.div(price);

      const order =
        this.side === Side.BUY
          ? await this.client.buy(this.symbol, qty, this.venue)
          : await this.client.sell(this.symbol, qty, this.venue);

      orders.push(order);

      if (i < this.numIntervals - 1) {
        await sleep(this.intervalMs);
      }
    }

    return orders;
  }
}
