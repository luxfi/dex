/**
 * LX-First Arbitrage Strategy.
 *
 * Key Insight: LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks).
 * By the time other venues update, LX has already moved.
 *
 * This means:
 * 1. LX DEX price is the "TRUE" price (most current)
 * 2. Other venues are always STALE by comparison
 * 3. Arbitrage = correcting stale venues to match LX
 * 4. LX DEX is the ORACLE, not just another venue
 *
 * Strategy:
 * 1. Watch LX DEX prices (the reference)
 * 2. Compare against "slow" venues (CEX, external DEX)
 * 3. When slow venue diverges from LX, trade on SLOW venue
 * 4. You're essentially front-running slow venues with LX information
 *
 * Example:
 * - LX DEX BTC: $50,000 (current, true)
 * - Binance BTC: $49,990 (stale, 50ms behind)
 * - Uniswap BTC: $50,020 (stale, 12s behind)
 *
 * Action:
 * - Buy on Binance at $49,990 (they haven't caught up yet)
 * - Sell on Uniswap at $50,020 (they haven't corrected yet)
 * - Net: $30 profit per BTC
 *
 * Why LX wins: By the time Binance/Uniswap update, we've already executed.
 */

import { Decimal } from 'decimal.js';
import type {
  LxFirstConfig,
  LxFirstOpportunity,
  LxPrice,
  VenuePrice,
} from './types.js';

export class LxFirstArbitrage {
  private lxPrices: Map<string, LxPrice> = new Map();
  private venuePrices: Map<string, VenuePrice[]> = new Map();
  private opportunityCallbacks: ((opp: LxFirstOpportunity) => void)[] = [];
  private running = false;

  constructor(public readonly config: LxFirstConfig) {}

  /**
   * Update the LX DEX price (the oracle).
   */
  updateLxPrice(price: LxPrice): void {
    this.lxPrices.set(price.symbol, price);

    // Immediately check for opportunities against stale venues
    this.checkOpportunities(price.symbol);
  }

  /**
   * Update a price from a "slow" venue.
   */
  updateVenuePrice(price: VenuePrice): void {
    const prices = this.venuePrices.get(price.symbol) ?? [];

    // Update or append
    const idx = prices.findIndex((p) => p.venue === price.venue);
    if (idx >= 0) {
      prices[idx] = price;
    } else {
      prices.push(price);
    }

    this.venuePrices.set(price.symbol, prices);
  }

  /**
   * Subscribe to opportunity events.
   */
  onOpportunity(callback: (opp: LxFirstOpportunity) => void): void {
    this.opportunityCallbacks.push(callback);
  }

  /**
   * Start the arbitrage system.
   */
  start(): void {
    this.running = true;
  }

  /**
   * Stop the arbitrage system.
   */
  stop(): void {
    this.running = false;
  }

  /**
   * Check for opportunities against stale venues.
   */
  private checkOpportunities(symbol: string): void {
    if (!this.running) return;

    const lxPrice = this.lxPrices.get(symbol);
    const venuePrices = this.venuePrices.get(symbol);

    if (!lxPrice || !venuePrices) return;

    const now = Date.now();

    for (const vp of venuePrices) {
      // Calculate how stale the venue is
      const staleness = now - vp.timestamp;
      if (staleness > this.config.maxStalenessMs) {
        continue; // Too stale, might have updated by now
      }

      // Check for BUY opportunity (venue ask < LX mid)
      // The slow venue hasn't caught up to LX's higher price
      if (vp.ask.lt(lxPrice.mid)) {
        const divergence = lxPrice.mid.minus(vp.ask);
        const divergenceBps = divergence.div(lxPrice.mid).mul(10000);

        if (divergenceBps.gte(this.config.minDivergenceBps)) {
          const opp = this.createOpportunity(
            symbol,
            lxPrice,
            vp,
            staleness,
            'buy',
            divergence,
            divergenceBps
          );

          if (opp.expectedProfit.gte(this.config.minProfit)) {
            this.emitOpportunity(opp);
          }
        }
      }

      // Check for SELL opportunity (venue bid > LX mid)
      // The slow venue hasn't caught up to LX's lower price
      if (vp.bid.gt(lxPrice.mid)) {
        const divergence = vp.bid.minus(lxPrice.mid);
        const divergenceBps = divergence.div(lxPrice.mid).mul(10000);

        if (divergenceBps.gte(this.config.minDivergenceBps)) {
          const opp = this.createOpportunity(
            symbol,
            lxPrice,
            vp,
            staleness,
            'sell',
            divergence,
            divergenceBps
          );

          if (opp.expectedProfit.gte(this.config.minProfit)) {
            this.emitOpportunity(opp);
          }
        }
      }
    }
  }

  /**
   * Create an opportunity object.
   */
  private createOpportunity(
    symbol: string,
    lxPrice: LxPrice,
    vp: VenuePrice,
    staleness: number,
    side: 'buy' | 'sell',
    divergence: Decimal,
    divergenceBps: Decimal
  ): LxFirstOpportunity {
    const now = Date.now();
    const expectedProfit = divergence.mul(this.config.maxPositionSize);
    const confidence = this.calculateConfidence(staleness, divergenceBps);

    return {
      id: `${symbol}-${vp.venue}-${side}-${now}`,
      symbol,
      timestamp: now,
      lxPrice,
      staleVenue: vp.venue,
      stalePrice: vp,
      staleness,
      side,
      divergence,
      divergenceBps,
      expectedProfit,
      maxSize: this.config.maxPositionSize,
      confidence,
    };
  }

  /**
   * Calculate confidence score.
   *
   * Higher confidence when:
   * 1. Venue is more stale (hasn't had time to update)
   * 2. Divergence is larger (more room for profit)
   */
  private calculateConfidence(staleness: number, divergenceBps: Decimal): number {
    const stalenessScore = 1.0 - staleness / (5000); // 5s max
    const clampedStaleness = Math.max(0, stalenessScore);

    const divergenceScore = divergenceBps.toNumber() / 100; // 100bps = 1.0
    const clampedDivergence = Math.min(1, divergenceScore);

    return 0.5 * clampedStaleness + 0.5 * clampedDivergence;
  }

  /**
   * Emit an opportunity to all subscribers.
   */
  private emitOpportunity(opp: LxFirstOpportunity): void {
    for (const callback of this.opportunityCallbacks) {
      try {
        callback(opp);
      } catch (err) {
        console.error('Error in opportunity callback:', err);
      }
    }
  }
}

/*
TRADING EXECUTION STRATEGY

When an LxFirstOpportunity is detected:

1. DO NOT trade on LX DEX (it's the reference, not the opportunity)

2. Trade on the STALE venue:
   - If Side="buy": Buy on stale venue (their ask is behind LX)
   - If Side="sell": Sell on stale venue (their bid is behind LX)

3. Settlement options:
   a) Hold position until venues converge (market neutral)
   b) Immediately hedge on LX DEX (lock in profit)
   c) Bridge and sell on another venue (more complex)

4. The key insight:
   - You're NOT arbitraging between two venues
   - You're front-running the slow venue with LX information
   - LX price is where the slow venue WILL BE, you just got there first

Example execution:

  LX DEX shows BTC = $50,000 (current, true price)
  Binance shows BTC = $49,950 (50ms stale)

  Action: BUY on Binance at $49,950
  Why: Binance WILL update to ~$50,000, we bought before they did
  Profit: ~$50 per BTC (0.1%)

  Optional hedge: SELL on LX DEX at $50,000 to lock in profit immediately
*/
