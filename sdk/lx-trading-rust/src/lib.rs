//! # LX Trading SDK
//!
//! High-performance trading utilities for the LX DEX.
//!
//! ## Features
//!
//! - **Types**: Fixed-point decimals, orders, trades, positions
//! - **Orderbook**: VWAP, slippage, liquidity analysis, multi-venue aggregation
//! - **Risk**: Position limits, PnL tracking, kill switch
//! - **Math**: Black-Scholes, Greeks, implied volatility, AMM pricing
//! - **Stats**: Volatility, Sharpe ratio, VaR, max drawdown
//!
//! ## Quick Start
//!
//! ```rust
//! use lx_trading::{
//!     types::{Decimal, OrderRequest, Side},
//!     orderbook::Orderbook,
//!     risk::{RiskConfig, RiskManager},
//!     math::{black_scholes, greeks},
//!     stats::{volatility, sharpe_ratio},
//! };
//!
//! // Create an order
//! let order = OrderRequest::limit(
//!     "BTC-USDC",
//!     Side::Buy,
//!     Decimal::from_f64(0.1),
//!     Decimal::from_f64(100.0),
//! );
//!
//! // Build an orderbook
//! let mut book = Orderbook::new("BTC-USDC", "lx_dex");
//! book.add_bid(Decimal::from_f64(99.0), Decimal::from_f64(1.0));
//! book.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(1.0));
//! book.sort();
//!
//! // Calculate VWAP
//! if let Some(vwap) = book.vwap_buy(Decimal::from_f64(0.5)) {
//!     println!("VWAP for 0.5 BTC: {}", vwap);
//! }
//!
//! // Risk management
//! let config = RiskConfig {
//!     enabled: true,
//!     max_order_size: Decimal::from_f64(10.0),
//!     ..Default::default()
//! };
//! let rm = RiskManager::new(config);
//!
//! // Options pricing
//! let call_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, true);
//! let g = greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
//!
//! // Statistics
//! let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];
//! let vol = volatility(&returns, true, None);
//! let sharpe = sharpe_ratio(&returns, None, None);
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod math;
pub mod orderbook;
pub mod risk;
pub mod stats;
pub mod types;

// Re-export commonly used items
pub use math::{black_scholes, greeks, implied_volatility, Greeks};
pub use orderbook::{AggregatedOrderbook, Orderbook};
pub use risk::{RiskConfig, RiskError, RiskManager};
pub use stats::{calculate_returns, max_drawdown, sharpe_ratio, var, volatility};
pub use types::{
    Balance, Decimal, Fee, Order, OrderRequest, OrderStatus, OrderType, Position, PriceLevel, Side,
    Ticker, TimeInForce, Trade, TradingPair,
};
