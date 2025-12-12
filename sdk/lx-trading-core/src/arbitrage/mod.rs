//! # Arbitrage Module
//!
//! LX-First Arbitrage Strategy:
//! - LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks)
//! - By the time other venues update, LX has already moved
//! - LX DEX price is the "TRUTH" (most current)
//! - Other venues are always STALE by comparison
//! - Arbitrage = correcting stale venues to match LX
//!
//! ## Key Concepts
//!
//! 1. **LX as Oracle**: LX DEX provides the reference price
//! 2. **Stale Venues**: CEX and external DEX are behind by 50ms-12s
//! 3. **Front-running**: Trade on stale venues before they catch up
//! 4. **Cross-chain**: Warp for Lux subnets, Teleport for EVM
//!
//! ## No Smart Contracts
//!
//! All arbitrage is executed through native RPC and the unified SDK.
//! No on-chain contracts needed - just coordinated trades.

mod cross_chain;
mod lx_first;
mod scanner;
mod types;
mod unified;

pub use cross_chain::*;
pub use lx_first::*;
pub use scanner::*;
pub use types::*;
pub use unified::*;
