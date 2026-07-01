/**
 * @luxfi/trading - High-frequency trading SDK with unified liquidity aggregation
 *
 * Features:
 * - Unified API for native LX DEX, CCXT exchanges, and Hummingbot Gateway
 * - Smart order routing with aggregated orderbook
 * - AMM support: swap, liquidity, LP positions
 * - Execution algorithms: TWAP, VWAP, Iceberg, Sniper
 * - Risk management with position limits and kill switch
 * - Financial math: Black-Scholes, Greeks, VaR/CVaR, AMM pricing
 */

// Types
export {
  Side,
  OrderType,
  TimeInForce,
  OrderStatus,
  VenueType,
  TradingPair,
  balanceTotal,
  aggregatedBalanceTotal,
  createMarketOrder,
  createLimitOrder,
  orderIsOpen,
  orderIsDone,
  orderFillPercent,
  tradeValue,
  tickerMidPrice,
  tickerSpread,
  tickerSpreadPercent,
  priceLevelValue,
  type Fee,
  type Balance,
  type AggregatedBalance,
  type OrderRequest,
  type Order,
  type Trade,
  type Ticker,
  type SwapQuote,
  type PoolInfo,
  type LpPosition,
  type LiquidityResult,
  type VenueInfo,
  type MarketInfo,
  type PriceLevel,
} from './types.js';

// Config
export {
  Config,
  NativeVenueConfig,
  CcxtConfig,
  HummingbotConfig,
  type GeneralConfig,
  type RiskConfig,
  type ConfigData,
} from './config.js';

// Adapters
export {
  BaseAdapter,
  orderBookCapabilities,
  ammCapabilities,
  LxDexAdapter,
  LxAmmAdapter,
  CcxtAdapter,
  HummingbotAdapter,
  type VenueAdapter,
  type VenueCapabilities,
} from './adapters/index.js';

// Orderbook
export { Orderbook, AggregatedOrderbook, type VenueQuantity } from './orderbook.js';

// Client
export { Client } from './client.js';

// Risk
export { RiskManager, RiskError } from './risk.js';

// Execution
export {
  TwapExecutor,
  VwapExecutor,
  IcebergExecutor,
  SniperExecutor,
  PovExecutor,
  DcaExecutor,
} from './execution.js';

// Math
export {
  // Options
  blackScholes,
  impliedVolatility,
  greeks,
  type Greeks,
  // AMM
  constantProductPrice,
  concentratedLiquidityPrice,
  calculateLiquidity,
  type AmmPriceResult,
  type ConcentratedLiquidityResult,
  // Risk metrics
  volatility,
  sharpeRatio,
  sortinoRatio,
  maxDrawdown,
  valueAtRisk,
  conditionalVaR,
  type MaxDrawdownResult,
  // Helpers
  normCdf,
  normPdf,
  priceToSqrtPrice,
  sqrtPriceToPrice,
  tickToSqrtPrice,
  sqrtPriceToTick,
  calculateReturns,
  calculateLogReturns,
} from './math.js';

// Arbitrage
export * as arbitrage from './arbitrage/index.js';

// ===========================================================================
// Convenience Exports - Simple single-venue clients
// ===========================================================================

import { Client } from './client.js';
import { Config, NativeVenueConfig } from './config.js';

/**
 * Options for DEX/AMM client initialization.
 */
export interface DexOptions {
  /** JSON-RPC endpoint URL */
  rpcUrl?: string;
  /** WebSocket endpoint URL */
  wsUrl?: string;
  /** API key for authenticated operations */
  apiKey?: string;
  /** Venue name (default: 'lx-dex' or 'lx-amm') */
  name?: string;
}

/**
 * Create a simple LX DEX client (OrderBook orderbook exchange).
 *
 * @example
 * ```typescript
 * import { DEX } from '@luxfi/trading';
 *
 * const dex = await DEX({ rpcUrl: 'http://localhost:8080/rpc' });
 * const order = await dex.limitBuy('BTC-USDC', '0.1', '50000');
 * ```
 */
export async function DEX(options: DexOptions = {}): Promise<Client> {
  const name = options.name ?? 'lx-dex';
  const nativeConfig = NativeVenueConfig.lxDex(options.rpcUrl ?? 'http://localhost:8080/rpc');
  if (options.wsUrl) {
    nativeConfig.wsUrl = options.wsUrl;
  }
  if (options.apiKey) {
    nativeConfig.apiKey = options.apiKey;
  }

  const config = new Config()
    .withNative(name, nativeConfig)
    .withVenuePriority([name])
    .withSmartRouting(false);

  const client = new Client(config);
  await client.connect();
  return client;
}

/**
 * Create a simple LX AMM client (C-chain AMM for swaps/liquidity).
 *
 * @example
 * ```typescript
 * import { AMM } from '@luxfi/trading';
 *
 * const amm = await AMM({ rpcUrl: 'http://localhost:8080/rpc' });
 * const quote = await amm.quote('BTC', 'USDC', '0.1', true, 'lx-amm');
 * ```
 */
export async function AMM(options: DexOptions = {}): Promise<Client> {
  const name = options.name ?? 'lx-amm';
  const nativeConfig = NativeVenueConfig.lxAmm(options.rpcUrl ?? 'http://localhost:8080/rpc');
  if (options.wsUrl) {
    nativeConfig.wsUrl = options.wsUrl;
  }
  if (options.apiKey) {
    nativeConfig.apiKey = options.apiKey;
  }

  const config = new Config()
    .withNative(name, nativeConfig)
    .withVenuePriority([name])
    .withSmartRouting(false);

  const client = new Client(config);
  await client.connect();
  return client;
}

/**
 * Alias for DEX() - LX DEX client.
 * @deprecated Use DEX() instead
 */
export const LxDex = DEX;
