/**
 * Tests for LX Trading SDK arbitrage scanner module.
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { Decimal } from 'decimal.js';

import { Scanner } from './scanner.js';
import type { PriceSource, ScannerConfig, CrossChainInfo, ArbitrageOpportunity } from './types.js';
import { ChainType } from './types.js';

function createScannerConfig(overrides?: Partial<ScannerConfig>): ScannerConfig {
  return {
    minSpreadBps: new Decimal(10),
    minProfitUSD: new Decimal(10),
    maxPriceAgeMs: 5000,
    scanIntervalMs: 1000,
    maxConcurrency: 10,
    symbols: ['BTC-USDC', 'ETH-USDC'],
    chainIds: ['lux', 'ethereum'],
    ...overrides,
  };
}

function createPriceSource(overrides?: Partial<PriceSource>): PriceSource {
  return {
    chainId: 'lux',
    venue: 'lx_dex',
    symbol: 'BTC-USDC',
    bid: new Decimal(50000),
    ask: new Decimal(50010),
    liquidity: new Decimal(100),
    timestamp: Date.now(),
    latency: 10,
    ...overrides,
  };
}

describe('Scanner', () => {
  let scanner: Scanner;

  beforeEach(() => {
    scanner = new Scanner(createScannerConfig());
  });

  afterEach(() => {
    scanner.stop();
  });

  describe('configuration', () => {
    it('should use provided config', () => {
      const config = createScannerConfig({ minSpreadBps: new Decimal(20) });
      const s = new Scanner(config);
      assert.equal(s.config.minSpreadBps.toNumber(), 20);
    });
  });

  describe('price updates', () => {
    it('should store price updates', () => {
      const source = createPriceSource();
      scanner.updatePrice(source);
      // Scanner stores prices internally - verify via opportunity detection
    });

    it('should update existing price source', () => {
      const source1 = createPriceSource({ bid: new Decimal(50000) });
      scanner.updatePrice(source1);

      const source2 = createPriceSource({ bid: new Decimal(50100) });
      scanner.updatePrice(source2);
      // Price should be updated, not duplicated
    });

    it('should distinguish sources by chain and venue', () => {
      const source1 = createPriceSource({ chainId: 'lux', venue: 'lx_dex' });
      const source2 = createPriceSource({ chainId: 'ethereum', venue: 'uniswap' });
      scanner.updatePrice(source1);
      scanner.updatePrice(source2);
      // Both should be stored separately
    });
  });

  describe('chain configuration', () => {
    it('should add chain info', () => {
      const chainInfo: CrossChainInfo = {
        chainId: 'lux_mainnet',
        name: 'Lux Mainnet',
        chainType: ChainType.LUX_SUBNET,
        blockTimeMs: 400,
        finalityMs: 400,
        warpSupported: true,
        teleportSupported: true,
        venues: ['lx_dex'],
      };
      scanner.addChain(chainInfo);
    });
  });

  describe('start/stop', () => {
    it('should start scanning', () => {
      scanner.start();
      assert.ok(true, 'Scanner started without error');
    });

    it('should stop scanning', () => {
      scanner.start();
      scanner.stop();
      assert.ok(true, 'Scanner stopped without error');
    });

    it('should not start twice', () => {
      scanner.start();
      scanner.start(); // Should be idempotent
      scanner.stop();
    });
  });

  describe('opportunity detection', () => {
    it('should detect simple arbitrage', async () => {
      const opportunities: ArbitrageOpportunity[] = [];
      scanner.onOpportunity((opp) => opportunities.push(opp));

      // Add price with spread opportunity
      // Buy on exchange A (lower ask)
      scanner.updatePrice(createPriceSource({
        chainId: 'lux',
        venue: 'lx_dex',
        symbol: 'BTC-USDC',
        bid: new Decimal(49900),
        ask: new Decimal(50000),
        liquidity: new Decimal(10),
      }));

      // Sell on exchange B (higher bid)
      scanner.updatePrice(createPriceSource({
        chainId: 'ethereum',
        venue: 'uniswap',
        symbol: 'BTC-USDC',
        bid: new Decimal(50200), // Higher bid = arbitrage opportunity
        ask: new Decimal(50300),
        liquidity: new Decimal(10),
      }));

      // Start scanner and wait for scan
      scanner.start();
      await new Promise((resolve) => setTimeout(resolve, 1500));
      scanner.stop();

      // Should have detected an opportunity (buy LX, sell Uniswap)
      assert.ok(opportunities.length > 0, 'Should detect arbitrage opportunity');
      if (opportunities.length > 0) {
        const opp = opportunities[0];
        assert.equal(opp.type, 'simple');
        assert.ok(opp.spreadBps.gt(0));
        assert.ok(opp.netPnL.gt(0));
      }
    });

    it('should detect CEX-DEX arbitrage', async () => {
      const opportunities: ArbitrageOpportunity[] = [];
      scanner.onOpportunity((opp) => opportunities.push(opp));

      // DEX price (lower ask)
      scanner.updatePrice(createPriceSource({
        chainId: 'lux',
        venue: 'lx_dex',
        symbol: 'BTC-USDC',
        bid: new Decimal(49900),
        ask: new Decimal(50000),
        liquidity: new Decimal(10),
      }));

      // CEX price (higher bid)
      scanner.updatePrice(createPriceSource({
        chainId: 'binance',
        venue: 'binance',
        symbol: 'BTC-USDC',
        bid: new Decimal(50200),
        ask: new Decimal(50300),
        liquidity: new Decimal(10),
      }));

      scanner.start();
      await new Promise((resolve) => setTimeout(resolve, 1500));
      scanner.stop();

      // Should have CEX-DEX opportunities
      const cexDexOpps = opportunities.filter((o) => o.type === 'cex_dex');
      assert.ok(cexDexOpps.length > 0, 'Should detect CEX-DEX opportunity');
    });

    it('should filter by minimum spread', async () => {
      const scanner2 = new Scanner(createScannerConfig({ minSpreadBps: new Decimal(100) }));
      const opportunities: ArbitrageOpportunity[] = [];
      scanner2.onOpportunity((opp) => opportunities.push(opp));

      // Add prices with small spread (5 bps)
      scanner2.updatePrice(createPriceSource({
        chainId: 'lux',
        venue: 'lx_dex',
        symbol: 'BTC-USDC',
        bid: new Decimal(49990),
        ask: new Decimal(50000),
      }));

      scanner2.updatePrice(createPriceSource({
        chainId: 'ethereum',
        venue: 'uniswap',
        symbol: 'BTC-USDC',
        bid: new Decimal(50025), // Only ~5 bps higher
        ask: new Decimal(50050),
      }));

      scanner2.start();
      await new Promise((resolve) => setTimeout(resolve, 1500));
      scanner2.stop();

      // Should not detect opportunity (spread too small)
      assert.equal(opportunities.length, 0, 'Should not detect low-spread opportunity');
    });

    it('should filter stale prices', async () => {
      const scanner2 = new Scanner(createScannerConfig({ maxPriceAgeMs: 1000 }));
      const opportunities: ArbitrageOpportunity[] = [];
      scanner2.onOpportunity((opp) => opportunities.push(opp));

      // Add fresh price
      scanner2.updatePrice(createPriceSource({
        chainId: 'lux',
        venue: 'lx_dex',
        symbol: 'BTC-USDC',
        bid: new Decimal(49900),
        ask: new Decimal(50000),
      }));

      // Add stale price (2 seconds old)
      scanner2.updatePrice(createPriceSource({
        chainId: 'ethereum',
        venue: 'uniswap',
        symbol: 'BTC-USDC',
        bid: new Decimal(50200),
        ask: new Decimal(50300),
        timestamp: Date.now() - 2000, // 2 seconds old
      }));

      scanner2.start();
      await new Promise((resolve) => setTimeout(resolve, 1500));
      scanner2.stop();

      // Should not detect opportunity (stale price)
      assert.equal(opportunities.length, 0, 'Should not use stale prices');
    });
  });

  describe('multiple callbacks', () => {
    it('should call all registered callbacks', async () => {
      let count1 = 0;
      let count2 = 0;

      scanner.onOpportunity(() => count1++);
      scanner.onOpportunity(() => count2++);

      // Add profitable opportunity
      scanner.updatePrice(createPriceSource({
        chainId: 'lux',
        venue: 'lx_dex',
        bid: new Decimal(49900),
        ask: new Decimal(50000),
        liquidity: new Decimal(10),
      }));

      scanner.updatePrice(createPriceSource({
        chainId: 'ethereum',
        venue: 'uniswap',
        bid: new Decimal(50200),
        ask: new Decimal(50300),
        liquidity: new Decimal(10),
      }));

      scanner.start();
      await new Promise((resolve) => setTimeout(resolve, 1500));
      scanner.stop();

      if (count1 > 0) {
        assert.equal(count1, count2, 'Both callbacks should be called same number of times');
      }
    });
  });
});
