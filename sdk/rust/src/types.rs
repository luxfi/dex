//! Core types for the LX DEX SDK.
//!
//! Defines orders, trades, positions, balances, and market data structures.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// Returns the opposite side.
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "buy"),
            Side::Sell => write!(f, "sell"),
        }
    }
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Limit,
    Market,
    Stop,
    StopLimit,
    Iceberg,
    Peg,
}

impl Default for OrderType {
    fn default() -> Self {
        Self::Limit
    }
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Open,
    Partial,
    Filled,
    Cancelled,
    Rejected,
}

impl OrderStatus {
    /// Returns true if the order is still active.
    pub fn is_active(self) -> bool {
        matches!(self, OrderStatus::Open | OrderStatus::Partial)
    }

    /// Returns true if the order is terminal (done).
    pub fn is_done(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected
        )
    }
}

/// Time in force for orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good till cancelled.
    GTC,
    /// Immediate or cancel.
    IOC,
    /// Fill or kill.
    FOK,
    /// Day order.
    DAY,
}

impl Default for TimeInForce {
    fn default() -> Self {
        Self::GTC
    }
}

/// A trading order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Order {
    /// Unique order ID assigned by the exchange.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_id: Option<u64>,

    /// Trading symbol (e.g., "BTC-USDT").
    pub symbol: String,

    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,

    /// Order side.
    pub side: Side,

    /// Limit price (ignored for market orders).
    pub price: f64,

    /// Order size in base currency.
    pub size: f64,

    /// Amount filled.
    #[serde(default)]
    pub filled: f64,

    /// Amount remaining.
    #[serde(default)]
    pub remaining: f64,

    /// Current status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OrderStatus>,

    /// User ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,

    /// Client-provided order ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,

    /// Unix timestamp in milliseconds.
    #[serde(default)]
    pub timestamp: i64,

    /// Time in force.
    #[serde(default)]
    pub time_in_force: TimeInForce,

    /// Post-only flag (maker only).
    #[serde(default)]
    pub post_only: bool,

    /// Reduce-only flag (close position only).
    #[serde(default)]
    pub reduce_only: bool,
}

impl Order {
    /// Create a new limit order.
    pub fn limit(symbol: impl Into<String>, side: Side, price: f64, size: f64) -> Self {
        Self {
            order_id: None,
            symbol: symbol.into(),
            order_type: OrderType::Limit,
            side,
            price,
            size,
            filled: 0.0,
            remaining: size,
            status: None,
            user_id: None,
            client_id: None,
            timestamp: now_millis(),
            time_in_force: TimeInForce::GTC,
            post_only: false,
            reduce_only: false,
        }
    }

    /// Create a new market order.
    pub fn market(symbol: impl Into<String>, side: Side, size: f64) -> Self {
        Self {
            order_id: None,
            symbol: symbol.into(),
            order_type: OrderType::Market,
            side,
            price: 0.0,
            size,
            filled: 0.0,
            remaining: size,
            status: None,
            user_id: None,
            client_id: None,
            timestamp: now_millis(),
            time_in_force: TimeInForce::IOC,
            post_only: false,
            reduce_only: false,
        }
    }

    /// Set client ID.
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// Set time in force.
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set post-only flag.
    pub fn post_only(mut self) -> Self {
        self.post_only = true;
        self
    }

    /// Set reduce-only flag.
    pub fn reduce_only(mut self) -> Self {
        self.reduce_only = true;
        self
    }

    /// Returns true if order is still open.
    pub fn is_open(&self) -> bool {
        self.status.map_or(true, |s| s.is_active())
    }

    /// Returns fill percentage (0.0 to 1.0).
    pub fn fill_rate(&self) -> f64 {
        if self.size > 0.0 {
            self.filled / self.size
        } else {
            0.0
        }
    }
}

/// Response to an order placement.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResponse {
    pub order_id: u64,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
}

/// An executed trade.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Trade {
    pub trade_id: u64,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: Side,
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    #[serde(default)]
    pub buyer_id: Option<String>,
    #[serde(default)]
    pub seller_id: Option<String>,
    pub timestamp: i64,
}

impl Trade {
    /// Total value of the trade (price * size).
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// A price level in the order book.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
    #[serde(default)]
    pub count: Option<i32>,
}

impl PriceLevel {
    pub fn new(price: f64, size: f64) -> Self {
        Self {
            price,
            size,
            count: None,
        }
    }
}

/// Account balance for an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub available: f64,
    pub locked: f64,
    pub total: f64,
}

impl Balance {
    /// Returns utilization ratio (locked / total).
    pub fn utilization(&self) -> f64 {
        if self.total > 0.0 {
            self.locked / self.total
        } else {
            0.0
        }
    }
}

/// A margin trading position.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    #[serde(default)]
    pub position_id: Option<String>,
    pub symbol: String,
    pub side: Side,
    pub size: f64,
    pub entry_price: f64,
    pub mark_price: f64,
    pub liquidation_price: f64,
    pub leverage: f64,
    pub margin: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

impl Position {
    /// Calculate unrealized PnL based on current mark price.
    pub fn calc_unrealized_pnl(&self) -> f64 {
        let diff = self.mark_price - self.entry_price;
        match self.side {
            Side::Buy => diff * self.size,
            Side::Sell => -diff * self.size,
        }
    }

    /// Calculate PnL as a percentage.
    pub fn pnl_percentage(&self) -> f64 {
        if self.entry_price > 0.0 {
            ((self.mark_price - self.entry_price) / self.entry_price) * 100.0
        } else {
            0.0
        }
    }

    /// Returns true if position is at risk of liquidation.
    pub fn is_at_risk(&self, threshold: f64) -> bool {
        let distance = (self.mark_price - self.liquidation_price).abs();
        let percent = distance / self.mark_price;
        percent < threshold
    }
}

/// Market information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Market {
    pub symbol: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub price_precision: u8,
    pub size_precision: u8,
    pub min_size: f64,
    pub max_size: f64,
    pub tick_size: f64,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub trading_enabled: bool,
}

/// Node/exchange information.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeInfo {
    pub version: String,
    pub network: String,
    pub order_count: i64,
    pub trade_count: i64,
    pub timestamp: i64,
    #[serde(default)]
    pub block_height: Option<i64>,
    pub syncing: bool,
    #[serde(default)]
    pub uptime: Option<i64>,
}

/// Get current unix timestamp in milliseconds.
fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_order() {
        let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1);
        assert_eq!(order.symbol, "BTC-USDT");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.price, 50000.0);
        assert_eq!(order.size, 0.1);
    }

    #[test]
    fn test_market_order() {
        let order = Order::market("ETH-USDT", Side::Sell, 1.0);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.time_in_force, TimeInForce::IOC);
    }

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn test_fill_rate() {
        let mut order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 1.0);
        order.filled = 0.5;
        assert!((order.fill_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_trade_value() {
        let trade = Trade {
            trade_id: 1,
            symbol: "BTC-USDT".into(),
            price: 50000.0,
            size: 0.1,
            side: Side::Buy,
            buy_order_id: 1,
            sell_order_id: 2,
            buyer_id: None,
            seller_id: None,
            timestamp: 0,
        };
        assert!((trade.value() - 5000.0).abs() < 0.001);
    }
}
