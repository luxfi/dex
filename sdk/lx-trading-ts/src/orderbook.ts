/**
 * Orderbook with aggregation support.
 */

import { Decimal } from 'decimal.js';
import { priceLevelValue, Side, type PriceLevel } from './types.js';

// =============================================================================
// Orderbook
// =============================================================================

export class Orderbook {
  bids: PriceLevel[] = [];
  asks: PriceLevel[] = [];
  timestamp: number;
  sequence = 0;

  constructor(
    public readonly symbol: string,
    public readonly venue: string,
  ) {
    this.timestamp = Date.now();
  }

  addBid(price: Decimal, quantity: Decimal): void {
    this.bids.push({ price, quantity });
  }

  addAsk(price: Decimal, quantity: Decimal): void {
    this.asks.push({ price, quantity });
  }

  sort(): void {
    // Bids descending (highest first)
    this.bids.sort((a, b) => b.price.cmp(a.price));
    // Asks ascending (lowest first)
    this.asks.sort((a, b) => a.price.cmp(b.price));
  }

  get bestBid(): Decimal | undefined {
    return this.bids[0]?.price;
  }

  get bestAsk(): Decimal | undefined {
    return this.asks[0]?.price;
  }

  get midPrice(): Decimal | undefined {
    if (this.bestBid && this.bestAsk) {
      return this.bestBid.plus(this.bestAsk).div(2);
    }
    return undefined;
  }

  get spread(): Decimal | undefined {
    if (this.bestBid && this.bestAsk) {
      return this.bestAsk.minus(this.bestBid);
    }
    return undefined;
  }

  get spreadPercent(): Decimal | undefined {
    const mid = this.midPrice;
    const spr = this.spread;
    if (spr && mid && mid.gt(0)) {
      return spr.div(mid).mul(100);
    }
    return undefined;
  }

  get bidLiquidity(): Decimal {
    return this.bids.reduce((sum, l) => sum.plus(priceLevelValue(l)), new Decimal(0));
  }

  get askLiquidity(): Decimal {
    return this.asks.reduce((sum, l) => sum.plus(priceLevelValue(l)), new Decimal(0));
  }

  bidDepth(levels: number): Decimal {
    return this.bids
      .slice(0, levels)
      .reduce((sum, l) => sum.plus(priceLevelValue(l)), new Decimal(0));
  }

  askDepth(levels: number): Decimal {
    return this.asks
      .slice(0, levels)
      .reduce((sum, l) => sum.plus(priceLevelValue(l)), new Decimal(0));
  }

  /**
   * VWAP for buying `amount` of base asset.
   */
  vwapBuy(amount: Decimal): Decimal | undefined {
    return this.calculateVwap(this.asks, amount);
  }

  /**
   * VWAP for selling `amount` of base asset.
   */
  vwapSell(amount: Decimal): Decimal | undefined {
    return this.calculateVwap(this.bids, amount);
  }

  private calculateVwap(levels: PriceLevel[], amount: Decimal): Decimal | undefined {
    let remaining = amount;
    let totalValue = new Decimal(0);
    let totalQty = new Decimal(0);

    for (const level of levels) {
      if (remaining.lte(0)) {
        break;
      }

      const fillQty = Decimal.min(remaining, level.quantity);
      totalValue = totalValue.plus(fillQty.mul(level.price));
      totalQty = totalQty.plus(fillQty);
      remaining = remaining.minus(fillQty);
    }

    if (totalQty.isZero()) {
      return undefined;
    }
    return totalValue.div(totalQty);
  }

  /**
   * Check if there's enough liquidity for the given side and amount.
   */
  hasLiquidity(side: Side, amount: Decimal): boolean {
    const levels = side === Side.BUY ? this.asks : this.bids;
    const total = levels.reduce((sum, l) => sum.plus(l.quantity), new Decimal(0));
    return total.gte(amount);
  }
}

// =============================================================================
// Aggregated Orderbook
// =============================================================================

export interface VenueQuantity {
  venue: string;
  quantity: Decimal;
}

export class AggregatedOrderbook {
  // price -> venue quantities
  bids: Map<string, VenueQuantity[]> = new Map();
  asks: Map<string, VenueQuantity[]> = new Map();
  timestamp = Date.now();

  constructor(public readonly symbol: string) {}

  /**
   * Add orderbook from a venue.
   */
  addOrderbook(book: Orderbook): void {
    for (const level of book.bids) {
      const key = level.price.toString();
      const existing = this.bids.get(key) ?? [];
      existing.push({ venue: book.venue, quantity: level.quantity });
      this.bids.set(key, existing);
    }

    for (const level of book.asks) {
      const key = level.price.toString();
      const existing = this.asks.get(key) ?? [];
      existing.push({ venue: book.venue, quantity: level.quantity });
      this.asks.set(key, existing);
    }

    this.timestamp = Math.max(this.timestamp, book.timestamp);
  }

  /**
   * Get best bid across all venues: { price, venue, quantity }
   */
  bestBid(): { price: Decimal; venue: string; quantity: Decimal } | undefined {
    if (this.bids.size === 0) {
      return undefined;
    }

    let bestPrice: Decimal | undefined;
    for (const key of this.bids.keys()) {
      const price = new Decimal(key);
      if (!bestPrice || price.gt(bestPrice)) {
        bestPrice = price;
      }
    }

    if (!bestPrice) {
      return undefined;
    }

    const venues = this.bids.get(bestPrice.toString());
    const first = venues?.[0];
    if (first) {
      return { price: bestPrice, venue: first.venue, quantity: first.quantity };
    }
    return undefined;
  }

  /**
   * Get best ask across all venues: { price, venue, quantity }
   */
  bestAsk(): { price: Decimal; venue: string; quantity: Decimal } | undefined {
    if (this.asks.size === 0) {
      return undefined;
    }

    let bestPrice: Decimal | undefined;
    for (const key of this.asks.keys()) {
      const price = new Decimal(key);
      if (!bestPrice || price.lt(bestPrice)) {
        bestPrice = price;
      }
    }

    if (!bestPrice) {
      return undefined;
    }

    const venues = this.asks.get(bestPrice.toString());
    const first = venues?.[0];
    if (first) {
      return { price: bestPrice, venue: first.venue, quantity: first.quantity };
    }
    return undefined;
  }

  /**
   * Get aggregated bid levels (total quantity at each price).
   */
  aggregatedBids(): PriceLevel[] {
    const prices = Array.from(this.bids.keys())
      .map((k) => new Decimal(k))
      .sort((a, b) => b.cmp(a));

    return prices.map((price) => {
      const venues = this.bids.get(price.toString()) ?? [];
      const totalQty = venues.reduce((sum, v) => sum.plus(v.quantity), new Decimal(0));
      return { price, quantity: totalQty };
    });
  }

  /**
   * Get aggregated ask levels (total quantity at each price).
   */
  aggregatedAsks(): PriceLevel[] {
    const prices = Array.from(this.asks.keys())
      .map((k) => new Decimal(k))
      .sort((a, b) => a.cmp(b));

    return prices.map((price) => {
      const venues = this.asks.get(price.toString()) ?? [];
      const totalQty = venues.reduce((sum, v) => sum.plus(v.quantity), new Decimal(0));
      return { price, quantity: totalQty };
    });
  }

  /**
   * Find best venue for buying `amount`: { venue, price }
   */
  bestVenueBuy(amount: Decimal): { venue: string; price: Decimal } | undefined {
    let bestVenue: string | undefined;
    let bestPrice = new Decimal(Infinity);
    let remaining = amount;

    const prices = Array.from(this.asks.keys())
      .map((k) => new Decimal(k))
      .sort((a, b) => a.cmp(b));

    for (const price of prices) {
      if (remaining.lte(0)) {
        break;
      }

      const venues = this.asks.get(price.toString()) ?? [];
      for (const { venue, quantity } of venues) {
        const fill = Decimal.min(remaining, quantity);
        if (price.lt(bestPrice)) {
          bestPrice = price;
          bestVenue = venue;
        }
        remaining = remaining.minus(fill);
        if (remaining.lte(0)) {
          break;
        }
      }
    }

    if (bestVenue) {
      return { venue: bestVenue, price: bestPrice };
    }
    return undefined;
  }

  /**
   * Find best venue for selling `amount`: { venue, price }
   */
  bestVenueSell(amount: Decimal): { venue: string; price: Decimal } | undefined {
    let bestVenue: string | undefined;
    let bestPrice = new Decimal(0);
    let remaining = amount;

    const prices = Array.from(this.bids.keys())
      .map((k) => new Decimal(k))
      .sort((a, b) => b.cmp(a));

    for (const price of prices) {
      if (remaining.lte(0)) {
        break;
      }

      const venues = this.bids.get(price.toString()) ?? [];
      for (const { venue, quantity } of venues) {
        const fill = Decimal.min(remaining, quantity);
        if (price.gt(bestPrice)) {
          bestPrice = price;
          bestVenue = venue;
        }
        remaining = remaining.minus(fill);
        if (remaining.lte(0)) {
          break;
        }
      }
    }

    if (bestVenue) {
      return { venue: bestVenue, price: bestPrice };
    }
    return undefined;
  }
}
