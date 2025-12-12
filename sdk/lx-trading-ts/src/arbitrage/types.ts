/**
 * Arbitrage types for LX Trading SDK.
 *
 * LX-FIRST ARBITRAGE STRATEGY:
 * - LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks)
 * - By the time other venues update, LX has already moved
 * - LX DEX price is the "TRUTH" (most current)
 * - Other venues are always STALE by comparison
 * - Arbitrage = correcting stale venues to match LX
 */

import { Decimal } from 'decimal.js';

// =============================================================================
// Cross-Chain Transport
// =============================================================================

export enum CrossChainTransport {
  /** Lux native - between subnets only */
  WARP = 'warp',
  /** EVM bridge for external chains */
  TELEPORT = 'teleport',
  /** Same chain, no bridge needed */
  DIRECT = 'direct',
  /** CEX API calls */
  CEX_API = 'cex_api',
}

export enum ChainType {
  LUX_SUBNET = 'lux_subnet',
  EVM = 'evm',
  CEX = 'cex',
}

export enum ArbType {
  /** Buy A, sell B */
  SIMPLE = 'simple',
  /** A->B->C->A */
  TRIANGULAR = 'triangular',
  /** Complex routes */
  MULTI_HOP = 'multi_hop',
  /** CEX<->DEX arb */
  CEX_DEX = 'cex_dex',
  /** DEX flash swap */
  FLASH_SWAP = 'flash_swap',
}

// =============================================================================
// Price Sources
// =============================================================================

export interface PriceSource {
  chainId: string;
  venue: string;
  symbol: string;
  bid: Decimal;
  ask: Decimal;
  liquidity: Decimal;
  timestamp: number;
  latency: number; // milliseconds
}

/** LX DEX price - the reference/oracle */
export interface LxPrice {
  symbol: string;
  bid: Decimal;
  ask: Decimal;
  mid: Decimal;
  timestamp: number;
  blockNum: bigint;
}

/** Price from a "slow" venue */
export interface VenuePrice {
  venue: string;
  symbol: string;
  bid: Decimal;
  ask: Decimal;
  timestamp: number;
  latency: number; // How far behind LX this venue typically is (ms)
  stale: boolean; // Is this price stale relative to LX?
}

// =============================================================================
// Opportunities
// =============================================================================

export interface ArbitrageOpportunity {
  id: string;
  type: ArbType;
  routes: Route[];
  buySource: PriceSource;
  sellSource: PriceSource;
  spreadBps: Decimal;
  estimatedPnL: Decimal;
  maxSize: Decimal;
  gasCostUSD: Decimal;
  bridgeCostUSD: Decimal;
  netPnL: Decimal;
  confidence: number; // 0-1
  expiresAt: number;
}

export interface Route {
  chainId: string;
  venue: string;
  action: 'buy' | 'sell';
  tokenIn: string;
  tokenOut: string;
  amountIn: Decimal;
  expectedOut: Decimal;
  minAmountOut: Decimal;
  swapData?: Uint8Array;
}

/** LX-first arbitrage opportunity */
export interface LxFirstOpportunity {
  id: string;
  symbol: string;
  timestamp: number;
  lxPrice: LxPrice;
  staleVenue: string;
  stalePrice: VenuePrice;
  staleness: number; // milliseconds
  side: 'buy' | 'sell';
  divergence: Decimal;
  divergenceBps: Decimal;
  expectedProfit: Decimal;
  maxSize: Decimal;
  confidence: number;
}

/** Unified arbitrage opportunity */
export interface UnifiedOpportunity {
  id: string;
  symbol: string;
  timestamp: number;
  expiresAt: number;
  buyVenue: string;
  buyPrice: Decimal;
  buySize: Decimal;
  sellVenue: string;
  sellPrice: Decimal;
  sellSize: Decimal;
  spread: Decimal;
  spreadBps: Decimal;
  maxSize: Decimal;
  grossProfit: Decimal;
  estFees: Decimal;
  netProfit: Decimal;
  confidence: number;
  latency: number;
}

// =============================================================================
// Execution
// =============================================================================

export interface UnifiedExecution {
  id: string;
  opportunity: UnifiedOpportunity;
  startTime: number;
  endTime: number;
  status: 'executing' | 'completed' | 'failed';
  buyOrderId?: string;
  sellOrderId?: string;
  actualProfit: Decimal;
  fees: Decimal;
  error?: Error;
}

export interface UnifiedArbStats {
  totalExecutions: number;
  successfulExecutions: number;
  totalPnL: Decimal;
  winRate: number;
}

// =============================================================================
// Configuration
// =============================================================================

export interface UnifiedArbConfig {
  /** Minimum spread to trade (basis points) */
  minSpreadBps: Decimal;
  /** Minimum profit per trade (quote currency) */
  minProfit: Decimal;
  /** Maximum position size per asset */
  maxPositionSize: Decimal;
  /** Maximum total exposure */
  maxTotalExposure: Decimal;
  /** Trading pairs to monitor */
  symbols: string[];
  /** Venue priority for execution */
  venuePriority: string[];
  /** Scan interval in milliseconds */
  scanIntervalMs: number;
  /** Execute timeout in milliseconds */
  executeTimeoutMs: number;
  /** Maximum daily loss */
  maxDailyLoss: Decimal;
  /** Maximum trades per day */
  maxTradesPerDay: number;
}

export interface LxFirstConfig {
  /** How stale is "too stale" to trade (ms) */
  maxStalenessMs: number;
  /** Minimum divergence from LX price (bps) */
  minDivergenceBps: Decimal;
  /** Minimum expected profit */
  minProfit: Decimal;
  /** Venue latency estimates */
  venueLatencies: Map<string, number>;
  /** Maximum position per trade */
  maxPositionSize: Decimal;
  /** Symbols to monitor */
  symbols: string[];
}

export interface ScannerConfig {
  /** Minimum spread (bps) */
  minSpreadBps: Decimal;
  /** Minimum profit (USD) */
  minProfitUSD: Decimal;
  /** Maximum price age before stale (ms) */
  maxPriceAgeMs: number;
  /** Symbols to scan */
  symbols: string[];
  /** Chain IDs to scan */
  chainIds: string[];
  /** Scan interval (ms) */
  scanIntervalMs: number;
  /** Maximum concurrent scans */
  maxConcurrency: number;
}

export interface CrossChainInfo {
  chainId: string;
  name: string;
  chainType: ChainType;
  blockTimeMs: number;
  finalityMs: number;
  warpSupported: boolean;
  teleportSupported: boolean;
  venues: string[];
}

export interface CrossChainConfig {
  warpEnabled: boolean;
  warpEndpoint?: string;
  warpTimeoutMs: number;
  teleportEnabled: boolean;
  teleportEndpoint?: string;
  teleportTimeoutMs: number;
  chains: Map<string, CrossChainInfo>;
}

// =============================================================================
// Default Configs
// =============================================================================

export function defaultUnifiedArbConfig(): UnifiedArbConfig {
  return {
    minSpreadBps: new Decimal(10),
    minProfit: new Decimal(5),
    maxPositionSize: new Decimal(10000),
    maxTotalExposure: new Decimal(100000),
    symbols: ['BTC-USDC', 'ETH-USDC', 'LUX-USDC'],
    venuePriority: ['lx_dex', 'binance', 'mexc', 'lx_amm'],
    scanIntervalMs: 100,
    executeTimeoutMs: 5000,
    maxDailyLoss: new Decimal(1000),
    maxTradesPerDay: 100,
  };
}

export function defaultLxFirstConfig(): LxFirstConfig {
  return {
    maxStalenessMs: 2000,
    minDivergenceBps: new Decimal(10),
    minProfit: new Decimal(5),
    maxPositionSize: new Decimal(1000),
    symbols: ['BTC-USDC', 'ETH-USDC', 'LUX-USDC'],
    venueLatencies: new Map([
      ['binance', 50],
      ['mexc', 100],
      ['okx', 80],
      ['uniswap', 12000],
      ['pancakeswap', 3000],
    ]),
  };
}

export function defaultScannerConfig(): ScannerConfig {
  return {
    minSpreadBps: new Decimal(10),
    minProfitUSD: new Decimal(10),
    maxPriceAgeMs: 5000,
    scanIntervalMs: 100,
    maxConcurrency: 50,
    symbols: ['BTC', 'ETH', 'LUX', 'SOL', 'AVAX'],
    chainIds: ['lux', 'ethereum', 'bsc', 'arbitrum', 'polygon'],
  };
}

export function defaultCrossChainConfig(): CrossChainConfig {
  const chains = new Map<string, CrossChainInfo>();

  // Lux ecosystem (Warp enabled)
  chains.set('lux_mainnet', {
    chainId: 'lux_mainnet',
    name: 'Lux Mainnet',
    chainType: ChainType.LUX_SUBNET,
    blockTimeMs: 400,
    finalityMs: 400,
    warpSupported: true,
    teleportSupported: true,
    venues: ['lx_dex', 'lx_amm'],
  });

  chains.set('lx_dex_subnet', {
    chainId: 'lx_dex_subnet',
    name: 'LX DEX Subnet',
    chainType: ChainType.LUX_SUBNET,
    blockTimeMs: 200,
    finalityMs: 200,
    warpSupported: true,
    teleportSupported: false,
    venues: ['lx_dex'],
  });

  // EVM chains (Teleport enabled)
  chains.set('ethereum', {
    chainId: '1',
    name: 'Ethereum',
    chainType: ChainType.EVM,
    blockTimeMs: 12000,
    finalityMs: 15 * 60 * 1000,
    warpSupported: false,
    teleportSupported: true,
    venues: ['uniswap', 'sushiswap'],
  });

  chains.set('bsc', {
    chainId: '56',
    name: 'BNB Smart Chain',
    chainType: ChainType.EVM,
    blockTimeMs: 3000,
    finalityMs: 45000,
    warpSupported: false,
    teleportSupported: true,
    venues: ['pancakeswap'],
  });

  chains.set('arbitrum', {
    chainId: '42161',
    name: 'Arbitrum One',
    chainType: ChainType.EVM,
    blockTimeMs: 250,
    finalityMs: 15 * 60 * 1000,
    warpSupported: false,
    teleportSupported: true,
    venues: ['uniswap', 'camelot'],
  });

  // CEX (API only)
  chains.set('binance', {
    chainId: 'binance',
    name: 'Binance',
    chainType: ChainType.CEX,
    blockTimeMs: 0,
    finalityMs: 0,
    warpSupported: false,
    teleportSupported: false,
    venues: ['binance'],
  });

  chains.set('mexc', {
    chainId: 'mexc',
    name: 'MEXC',
    chainType: ChainType.CEX,
    blockTimeMs: 0,
    finalityMs: 0,
    warpSupported: false,
    teleportSupported: false,
    venues: ['mexc'],
  });

  return {
    warpEnabled: true,
    warpTimeoutMs: 5000,
    teleportEnabled: true,
    teleportTimeoutMs: 60000,
    chains,
  };
}
