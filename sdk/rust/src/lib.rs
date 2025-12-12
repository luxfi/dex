//! # LUX DEX Rust SDK
//!
//! High-performance Rust client for the LX decentralized exchange.
//!
//! ## Features
//!
//! - Async WebSocket client with automatic reconnection
//! - HTTP/JSON-RPC client for request-response operations
//! - Type-safe order, trade, and position management
//! - Order book utilities with slippage estimation
//! - Real-time market data subscriptions
//!
//! ## Quick Start
//!
//! ```no_run
//! use lux_dex::{Client, ClientConfig, Order, Side};
//!
//! #[tokio::main]
//! async fn main() -> lux_dex::Result<()> {
//!     // Create client with default config
//!     let client = Client::new();
//!
//!     // Connect to WebSocket
//!     client.connect().await?;
//!
//!     // Authenticate (if trading)
//!     client.authenticate("api_key", "api_secret").await?;
//!
//!     // Subscribe to market data
//!     client.subscribe_orderbook("BTC-USDT").await?;
//!
//!     // Place a limit order
//!     let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1);
//!     client.place_order(&order).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Event Handling
//!
//! ```no_run
//! use lux_dex::{Client, WsEvent};
//!
//! #[tokio::main]
//! async fn main() -> lux_dex::Result<()> {
//!     let client = Client::new();
//!     client.connect().await?;
//!
//!     // Take the event receiver (can only be called once)
//!     let mut events = client.take_event_receiver().await.unwrap();
//!
//!     // Process events in a loop
//!     while let Some(event) = events.recv().await {
//!         match event {
//!             WsEvent::OrderBook(book) => {
//!                 println!("Order book: {} bid={:?} ask={:?}",
//!                     book.symbol, book.best_bid(), book.best_ask());
//!             }
//!             WsEvent::Trade(trade) => {
//!                 println!("Trade: {} @ {}", trade.size, trade.price);
//!             }
//!             WsEvent::OrderUpdate { order, status } => {
//!                 println!("Order {}: {}", order.order_id.unwrap_or(0), status);
//!             }
//!             _ => {}
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod client;
pub mod error;
pub mod orderbook;
pub mod types;

// Re-export main types at crate root
pub use client::{Client, ClientConfig, WsEvent};
pub use error::{Error, Result};
pub use orderbook::{BookDepth, OrderBook, OrderBookUpdate};
pub use types::{
    Balance, Market, NodeInfo, Order, OrderResponse, OrderStatus, OrderType, Position, PriceLevel,
    Side, TimeInForce, Trade,
};
