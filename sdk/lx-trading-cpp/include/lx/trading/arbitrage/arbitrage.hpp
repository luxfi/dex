// LX Trading SDK - Arbitrage Module
//
// LX-First Arbitrage Strategy:
// - LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks)
// - By the time other venues update, LX has already moved
// - LX DEX price is the "TRUTH" (most current)
// - Other venues are always STALE by comparison
// - Arbitrage = correcting stale venues to match LX
//
// Key Concepts:
// 1. LX as Oracle: LX DEX provides the reference price
// 2. Stale Venues: CEX and external DEX are behind by 50ms-12s
// 3. Front-running: Trade on stale venues before they catch up
// 4. Cross-chain: Warp for Lux subnets, Teleport for EVM
//
// NO SMART CONTRACTS - All arbitrage is executed through native RPC and the unified SDK.

#pragma once

#include <lx/trading/arbitrage/types.hpp>
#include <lx/trading/arbitrage/scanner.hpp>
#include <lx/trading/arbitrage/lx_first.hpp>
#include <lx/trading/arbitrage/unified.hpp>
#include <lx/trading/arbitrage/cross_chain.hpp>
