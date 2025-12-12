/**
 * LX Hummingbot Gateway Connector
 *
 * Ultra-low latency decentralized exchange connector supporting:
 * - Router Schema: DEX aggregation and optimal swap routing
 * - AMM Schema: Traditional xy=k liquidity pool operations
 * - CLMM Schema: Concentrated liquidity market maker positions
 * - Order Book: Central limit order book trading
 *
 * @packageDocumentation
 */

// Main connector class
export { LXDex } from './lxdex';

// Configuration
export { LXDexConfig, getLXDexConfig, LX_DEX_NETWORKS } from './lxdex.config';

// Route registration
export { lxdexRoutes } from './lxdex.routes';

// All schemas and types
export * from './schemas';

// Sub-route modules
export * as routerRoutes from './router-routes';
export * as ammRoutes from './amm-routes';
export * as clmmRoutes from './clmm-routes';
