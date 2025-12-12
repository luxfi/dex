/**
 * Arbitrage scanner for detecting cross-venue opportunities.
 *
 * Continuously scans for arbitrage opportunities across all venues.
 * Supports simple, triangular, and CEX-DEX arbitrage detection.
 */

import { Decimal } from 'decimal.js';
import type {
  ArbitrageOpportunity,
  ArbType,
  PriceSource,
  ScannerConfig,
  CrossChainInfo,
} from './types.js';

const CEX_VENUES = new Set([
  'binance', 'coinbase', 'kraken', 'okx', 'bybit',
  'kucoin', 'mexc', 'gate', 'huobi',
]);

export class Scanner {
  private prices: Map<string, PriceSource[]> = new Map();
  private chains: Map<string, CrossChainInfo> = new Map();
  private opportunityCallbacks: ((opp: ArbitrageOpportunity) => void)[] = [];
  private running = false;
  private timer?: ReturnType<typeof setInterval>;

  constructor(public readonly config: ScannerConfig) {}

  /**
   * Add a chain configuration.
   */
  addChain(info: CrossChainInfo): void {
    this.chains.set(info.chainId, info);
  }

  /**
   * Update a price feed.
   */
  updatePrice(source: PriceSource): void {
    const sources = this.prices.get(source.symbol) ?? [];

    // Update existing or append new
    const idx = sources.findIndex(
      (s) => s.chainId === source.chainId && s.venue === source.venue
    );

    if (idx >= 0) {
      sources[idx] = source;
    } else {
      sources.push(source);
    }

    this.prices.set(source.symbol, sources);
  }

  /**
   * Start scanning for opportunities.
   */
  start(): void {
    if (this.running) return;
    this.running = true;

    this.timer = setInterval(() => {
      this.scan();
    }, this.config.scanIntervalMs);
  }

  /**
   * Stop scanning.
   */
  stop(): void {
    this.running = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  /**
   * Subscribe to opportunity events.
   */
  onOpportunity(callback: (opp: ArbitrageOpportunity) => void): void {
    this.opportunityCallbacks.push(callback);
  }

  /**
   * Perform a single scan.
   */
  private scan(): void {
    const symbols = Array.from(this.prices.keys());

    // Process in parallel
    const promises: Promise<void>[] = [];

    for (const symbol of symbols) {
      const promise = this.findOpportunities(symbol).then((opps) => {
        for (const opp of opps) {
          this.emitOpportunity(opp);
        }
      });
      promises.push(promise);
    }

    Promise.all(promises).catch(console.error);
  }

  /**
   * Find all opportunities for a symbol.
   */
  private async findOpportunities(symbol: string): Promise<ArbitrageOpportunity[]> {
    const sources = this.prices.get(symbol);
    if (!sources || sources.length < 2) return [];

    const now = Date.now();

    // Filter stale prices
    const validSources = sources.filter(
      (s) => now - s.timestamp < this.config.maxPriceAgeMs
    );

    if (validSources.length < 2) return [];

    const opportunities: ArbitrageOpportunity[] = [];

    // Simple arbitrage
    opportunities.push(...this.findSimpleArb(symbol, validSources));

    // CEX-DEX arbitrage
    opportunities.push(...this.findCexDexArb(symbol, validSources));

    return opportunities;
  }

  /**
   * Find simple buy-low-sell-high opportunities.
   */
  private findSimpleArb(symbol: string, sources: PriceSource[]): ArbitrageOpportunity[] {
    const opportunities: ArbitrageOpportunity[] = [];

    // Sort by ask (lowest first for buying)
    const buyOrder = [...sources].sort((a, b) =>
      a.ask.comparedTo(b.ask)
    );

    // Sort by bid (highest first for selling)
    const sellOrder = [...sources].sort((a, b) =>
      b.bid.comparedTo(a.bid)
    );

    // Check each buy/sell combination
    for (const buySrc of buyOrder) {
      for (const sellSrc of sellOrder) {
        // Skip same venue/chain
        if (buySrc.chainId === sellSrc.chainId && buySrc.venue === sellSrc.venue) {
          continue;
        }

        // Calculate spread
        const spread = sellSrc.bid.minus(buySrc.ask);
        if (spread.lte(0)) continue;

        const spreadBps = spread.div(buySrc.ask).mul(10000);
        if (spreadBps.lt(this.config.minSpreadBps)) continue;

        // Calculate costs
        const [gasCost, bridgeCost] = this.calculateCosts(
          buySrc.chainId,
          sellSrc.chainId
        );

        // Maximum size limited by liquidity
        const maxSize = Decimal.min(buySrc.liquidity, sellSrc.liquidity);

        // Calculate PnL
        const grossPnL = spread.mul(maxSize);
        const netPnL = grossPnL.minus(gasCost).minus(bridgeCost);

        if (netPnL.lt(this.config.minProfitUSD)) continue;

        // Calculate confidence
        const confidence = this.calculateConfidence(buySrc, sellSrc);

        const opp: ArbitrageOpportunity = {
          id: `simple-${symbol}-${buySrc.venue}-${sellSrc.venue}-${Date.now()}`,
          type: 'simple' as ArbType,
          buySource: buySrc,
          sellSource: sellSrc,
          spreadBps,
          estimatedPnL: grossPnL,
          maxSize,
          gasCostUSD: gasCost,
          bridgeCostUSD: bridgeCost,
          netPnL,
          confidence,
          expiresAt: Date.now() + 5000,
          routes: [
            {
              chainId: buySrc.chainId,
              venue: buySrc.venue,
              action: 'buy',
              tokenIn: 'USDC',
              tokenOut: symbol,
              amountIn: maxSize.mul(buySrc.ask),
              expectedOut: maxSize,
              minAmountOut: maxSize.mul(0.99),
            },
            {
              chainId: sellSrc.chainId,
              venue: sellSrc.venue,
              action: 'sell',
              tokenIn: symbol,
              tokenOut: 'USDC',
              amountIn: maxSize,
              expectedOut: maxSize.mul(sellSrc.bid),
              minAmountOut: maxSize.mul(sellSrc.bid).mul(0.99),
            },
          ],
        };

        opportunities.push(opp);
      }
    }

    return opportunities;
  }

  /**
   * Find CEX-DEX arbitrage opportunities.
   */
  private findCexDexArb(symbol: string, sources: PriceSource[]): ArbitrageOpportunity[] {
    const opportunities: ArbitrageOpportunity[] = [];

    // Separate CEX and DEX sources
    const cexSources = sources.filter((s) => CEX_VENUES.has(s.venue));
    const dexSources = sources.filter((s) => !CEX_VENUES.has(s.venue));

    // Find CEX buy -> DEX sell opportunities
    for (const cex of cexSources) {
      for (const dex of dexSources) {
        const spread = dex.bid.minus(cex.ask);
        if (spread.lte(0)) continue;

        const spreadBps = spread.div(cex.ask).mul(10000);
        if (spreadBps.lt(this.config.minSpreadBps)) continue;

        const maxSize = Decimal.min(cex.liquidity, dex.liquidity);
        const grossPnL = spread.mul(maxSize);

        const opp: ArbitrageOpportunity = {
          id: `cexdex-${symbol}-${cex.venue}-${dex.venue}-${Date.now()}`,
          type: 'cex_dex' as ArbType,
          buySource: cex,
          sellSource: dex,
          spreadBps,
          estimatedPnL: grossPnL,
          maxSize,
          gasCostUSD: new Decimal(0.5),
          bridgeCostUSD: new Decimal(0),
          netPnL: grossPnL.minus(0.5),
          confidence: 0.7,
          expiresAt: Date.now() + 3000,
          routes: [],
        };

        opportunities.push(opp);
      }
    }

    // Find DEX buy -> CEX sell opportunities
    for (const dex of dexSources) {
      for (const cex of cexSources) {
        const spread = cex.bid.minus(dex.ask);
        if (spread.lte(0)) continue;

        const spreadBps = spread.div(dex.ask).mul(10000);
        if (spreadBps.lt(this.config.minSpreadBps)) continue;

        const maxSize = Decimal.min(dex.liquidity, cex.liquidity);
        const grossPnL = spread.mul(maxSize);

        const opp: ArbitrageOpportunity = {
          id: `cexdex-${symbol}-${dex.venue}-${cex.venue}-${Date.now()}`,
          type: 'cex_dex' as ArbType,
          buySource: dex,
          sellSource: cex,
          spreadBps,
          estimatedPnL: grossPnL,
          maxSize,
          gasCostUSD: new Decimal(0.5),
          bridgeCostUSD: new Decimal(0),
          netPnL: grossPnL.minus(0.5),
          confidence: 0.7,
          expiresAt: Date.now() + 3000,
          routes: [],
        };

        opportunities.push(opp);
      }
    }

    return opportunities;
  }

  /**
   * Calculate gas and bridge costs between chains.
   */
  private calculateCosts(sourceChain: string, destChain: string): [Decimal, Decimal] {
    const srcConfig = this.chains.get(sourceChain);
    const dstConfig = this.chains.get(destChain);

    // Estimate gas cost
    let gasCost = new Decimal(0.1); // Default
    if (srcConfig) {
      // Simplified gas estimation
      gasCost = new Decimal(0.05);
    }

    // Bridge cost if crossing chains
    let bridgeCost = new Decimal(0);
    if (sourceChain !== destChain && srcConfig && dstConfig) {
      if (srcConfig.warpSupported && dstConfig.warpSupported) {
        bridgeCost = new Decimal(0.01); // Warp is nearly free
      } else if (srcConfig.teleportSupported && dstConfig.teleportSupported) {
        bridgeCost = new Decimal(0.10); // Teleport for EVM chains
      } else {
        bridgeCost = new Decimal(1.0); // Generic bridge
      }
    }

    return [gasCost, bridgeCost];
  }

  /**
   * Calculate confidence score for an opportunity.
   */
  private calculateConfidence(buy: PriceSource, sell: PriceSource): number {
    const now = Date.now();

    // Freshness score
    const buyAge = (now - buy.timestamp) / 1000;
    const sellAge = (now - sell.timestamp) / 1000;
    const maxAge = this.config.maxPriceAgeMs / 1000;
    let freshnessScore = 1.0 - (buyAge + sellAge) / (2 * maxAge);
    if (freshnessScore < 0) freshnessScore = 0;

    // Liquidity score
    const minLiq = Decimal.min(buy.liquidity, sell.liquidity);
    let liquidityScore = 0.5;
    if (minLiq.gt(100000)) {
      liquidityScore = 1.0;
    } else if (minLiq.gt(10000)) {
      liquidityScore = 0.8;
    }

    // Latency score
    const avgLatency = (buy.latency + sell.latency) / 2;
    let latencyScore = 1.0 - avgLatency / 1000;
    if (latencyScore < 0) latencyScore = 0;

    // Weighted average
    return 0.4 * freshnessScore + 0.4 * liquidityScore + 0.2 * latencyScore;
  }

  /**
   * Emit an opportunity to all subscribers.
   */
  private emitOpportunity(opp: ArbitrageOpportunity): void {
    for (const callback of this.opportunityCallbacks) {
      try {
        callback(opp);
      } catch (err) {
        console.error('Error in opportunity callback:', err);
      }
    }
  }
}
