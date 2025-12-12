/**
 * LX Trading SDK - Arbitrage Module.
 *
 * Omnichain arbitrage using LX DEX as the price oracle.
 * LX DEX is the fastest venue (nanosecond updates, 200ms blocks),
 * making it the "truth" while other venues are always stale.
 */

export * from './types.js';
export * from './scanner.js';
export * from './lx-first.js';
export * from './unified.js';
export * from './cross-chain.js';
