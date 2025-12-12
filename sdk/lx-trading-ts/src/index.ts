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
  clobCapabilities,
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
