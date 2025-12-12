/**
 * Tests for LX Trading SDK arbitrage types module.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Decimal } from 'decimal.js';

import {
  CrossChainTransport,
  ChainType,
  ArbType,
  defaultUnifiedArbConfig,
  defaultLxFirstConfig,
  defaultScannerConfig,
  defaultCrossChainConfig,
} from './types.js';

describe('Arbitrage Enums', () => {
  describe('CrossChainTransport', () => {
    it('should have correct values', () => {
      assert.equal(CrossChainTransport.WARP, 'warp');
      assert.equal(CrossChainTransport.TELEPORT, 'teleport');
      assert.equal(CrossChainTransport.DIRECT, 'direct');
      assert.equal(CrossChainTransport.CEX_API, 'cex_api');
    });
  });

  describe('ChainType', () => {
    it('should have correct values', () => {
      assert.equal(ChainType.LUX_SUBNET, 'lux_subnet');
      assert.equal(ChainType.EVM, 'evm');
      assert.equal(ChainType.CEX, 'cex');
    });
  });

  describe('ArbType', () => {
    it('should have correct values', () => {
      assert.equal(ArbType.SIMPLE, 'simple');
      assert.equal(ArbType.TRIANGULAR, 'triangular');
      assert.equal(ArbType.MULTI_HOP, 'multi_hop');
      assert.equal(ArbType.CEX_DEX, 'cex_dex');
      assert.equal(ArbType.FLASH_SWAP, 'flash_swap');
    });
  });
});

describe('Default Configurations', () => {
  describe('defaultUnifiedArbConfig', () => {
    it('should return valid default config', () => {
      const config = defaultUnifiedArbConfig();

      assert.ok(config.minSpreadBps instanceof Decimal);
      assert.equal(config.minSpreadBps.toNumber(), 10);
      assert.equal(config.minProfit.toNumber(), 5);
      assert.equal(config.maxPositionSize.toNumber(), 10000);
      assert.equal(config.maxTotalExposure.toNumber(), 100000);
      assert.ok(Array.isArray(config.symbols));
      assert.ok(config.symbols.includes('BTC-USDC'));
      assert.ok(config.symbols.includes('ETH-USDC'));
      assert.ok(config.symbols.includes('LUX-USDC'));
      assert.ok(Array.isArray(config.venuePriority));
      assert.equal(config.venuePriority[0], 'lx_dex'); // LX DEX first
      assert.equal(config.scanIntervalMs, 100);
      assert.equal(config.executeTimeoutMs, 5000);
      assert.equal(config.maxDailyLoss.toNumber(), 1000);
      assert.equal(config.maxTradesPerDay, 100);
    });
  });

  describe('defaultLxFirstConfig', () => {
    it('should return valid default config', () => {
      const config = defaultLxFirstConfig();

      assert.equal(config.maxStalenessMs, 2000);
      assert.equal(config.minDivergenceBps.toNumber(), 10);
      assert.equal(config.minProfit.toNumber(), 5);
      assert.equal(config.maxPositionSize.toNumber(), 1000);
      assert.ok(Array.isArray(config.symbols));
      assert.ok(config.venueLatencies instanceof Map);
      assert.ok(config.venueLatencies.has('binance'));
      assert.equal(config.venueLatencies.get('binance'), 50);
    });

    it('should have venue latencies ordered by speed', () => {
      const config = defaultLxFirstConfig();

      // CEXes should be faster than DEXes
      const binanceLatency = config.venueLatencies.get('binance')!;
      const uniswapLatency = config.venueLatencies.get('uniswap')!;

      assert.ok(binanceLatency < uniswapLatency, 'CEX should be faster than DEX');
    });
  });

  describe('defaultScannerConfig', () => {
    it('should return valid default config', () => {
      const config = defaultScannerConfig();

      assert.equal(config.minSpreadBps.toNumber(), 10);
      assert.equal(config.minProfitUSD.toNumber(), 10);
      assert.equal(config.maxPriceAgeMs, 5000);
      assert.equal(config.scanIntervalMs, 100);
      assert.equal(config.maxConcurrency, 50);
      assert.ok(Array.isArray(config.symbols));
      assert.ok(Array.isArray(config.chainIds));
    });
  });

  describe('defaultCrossChainConfig', () => {
    it('should return valid default config', () => {
      const config = defaultCrossChainConfig();

      assert.equal(config.warpEnabled, true);
      assert.equal(config.warpTimeoutMs, 5000);
      assert.equal(config.teleportEnabled, true);
      assert.equal(config.teleportTimeoutMs, 60000);
      assert.ok(config.chains instanceof Map);
    });

    it('should have Lux chains with Warp support', () => {
      const config = defaultCrossChainConfig();

      const luxMainnet = config.chains.get('lux_mainnet');
      assert.ok(luxMainnet);
      assert.equal(luxMainnet.chainType, ChainType.LUX_SUBNET);
      assert.equal(luxMainnet.warpSupported, true);

      const lxDexSubnet = config.chains.get('lx_dex_subnet');
      assert.ok(lxDexSubnet);
      assert.equal(lxDexSubnet.chainType, ChainType.LUX_SUBNET);
      assert.equal(lxDexSubnet.warpSupported, true);
      assert.equal(lxDexSubnet.blockTimeMs, 200); // 200ms blocks
    });

    it('should have EVM chains with Teleport support', () => {
      const config = defaultCrossChainConfig();

      const ethereum = config.chains.get('ethereum');
      assert.ok(ethereum);
      assert.equal(ethereum.chainType, ChainType.EVM);
      assert.equal(ethereum.warpSupported, false);
      assert.equal(ethereum.teleportSupported, true);
      assert.equal(ethereum.blockTimeMs, 12000);

      const bsc = config.chains.get('bsc');
      assert.ok(bsc);
      assert.equal(bsc.chainType, ChainType.EVM);
      assert.equal(bsc.teleportSupported, true);
    });

    it('should have CEX chains without bridge support', () => {
      const config = defaultCrossChainConfig();

      const binance = config.chains.get('binance');
      assert.ok(binance);
      assert.equal(binance.chainType, ChainType.CEX);
      assert.equal(binance.warpSupported, false);
      assert.equal(binance.teleportSupported, false);
      assert.equal(binance.blockTimeMs, 0); // CEX has no blocks
    });

    it('should have LX DEX as fastest venue', () => {
      const config = defaultCrossChainConfig();

      // Get all block times
      const blockTimes: number[] = [];
      for (const chain of config.chains.values()) {
        if (chain.blockTimeMs > 0) {
          blockTimes.push(chain.blockTimeMs);
        }
      }

      const lxDex = config.chains.get('lx_dex_subnet')!;
      const minBlockTime = Math.min(...blockTimes);

      assert.equal(lxDex.blockTimeMs, minBlockTime, 'LX DEX should have fastest block time');
    });
  });
});
