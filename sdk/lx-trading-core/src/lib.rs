//! # LX Trading SDK
//!
//! High-frequency trading SDK with unified liquidity aggregation.
//! Supports native LX DEX, CCXT exchanges, and Hummingbot connectors.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    LX Trading SDK                        │
//! ├─────────────────────────────────────────────────────────┤
//! │  UnifiedClient  │  Execution  │  Risk   │  Orderbook    │
//! ├─────────────────────────────────────────────────────────┤
//! │  WebSocket  │  HTTP Pool  │  Metrics  │  Math           │
//! ├─────────────────────────────────────────────────────────┤
//! │                    Adapter Layer                         │
//! ├───────────┬───────────┬───────────┬────────────────────┤
//! │  Native   │   CCXT    │ Hummingbot│    Custom          │
//! │  LX DEX   │  Exchanges│  Gateway  │    Adapters        │
//! └───────────┴───────────┴───────────┴────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Unified API**: Same interface for all venues
//! - **Smart Order Routing**: Best execution across venues
//! - **Aggregated Orderbook**: Combined liquidity view
//! - **Low Latency**: Lock-free data structures, zero-copy
//! - **Risk Management**: Position limits, PnL tracking
//! - **Real-time Streaming**: WebSocket support for market data
//! - **Connection Pooling**: Efficient HTTP with retry logic
//! - **Metrics Collection**: Latency, throughput, PnL tracking
//! - **Financial Math**: VWAP, volatility, position sizing
//!
//! ## Example
//!
//! ```rust,ignore
//! use lx_trading::{UnifiedClient, Config};
//! use rust_decimal::Decimal;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load configuration from file
//!     let config = Config::from_file("config.toml")?;
//!     let client = UnifiedClient::new(config)?;
//!
//!     // Initialize all configured venues
//!     client.init().await?;
//!
//!     // Get aggregated orderbook
//!     let book = client.orderbook("BTC-USDC").await?;
//!
//!     // Smart order routing - best price across all venues
//!     let order = client.buy("BTC-USDC", Decimal::from(1), None).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod adapters;
pub mod arbitrage;
pub mod config;
pub mod engine;
pub mod error;
pub mod execution;
pub mod http;
pub mod math;
pub mod metrics;
pub mod orderbook;
pub mod risk;
pub mod stream;
pub mod types;
pub mod ws;

// Re-exports
pub use config::Config;
pub use engine::UnifiedClient;
pub use error::{Error, Result};
pub use http::{HttpClient, HttpConfig, RetryConfig};
pub use math::*;
pub use metrics::{global_metrics, MetricsCollector};
pub use stream::{FillStream, OrderStream, OrderbookStream, StreamBuilder, TradeStream};
pub use types::*;
pub use ws::{WsConfig, WsConnection, WsEvent};

// No FFI module. `ffi = []` and `#[cfg(feature = "ffi")] pub mod ffi;` were
// declared but src/ffi.rs has never existed in this repo and nothing references
// `ffi::`. The feature could therefore never build — and because CI runs
// `cargo clippy --all-features` and `cargo test --all-features`, every run
// enabled it and failed on the missing file. `cargo fmt` stumbled on it too:
//   Error writing files: failed to resolve mod `ffi`: src/ffi.rs does not exist

// No Python bindings module. `python = ["pyo3"]` and
// `#[cfg(feature = "python")] pub mod python;` were declared but src/python.rs
// has never existed here either — same dead shape as the ffi one above.
