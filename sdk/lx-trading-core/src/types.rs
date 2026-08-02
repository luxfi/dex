//! Core types for the LX Trading SDK.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Trading side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "buy"),
            Side::Sell => write!(f, "sell"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Market,
    Limit,
    LimitMaker,
    StopLoss,
    StopLossLimit,
    TakeProfit,
    TakeProfitLimit,
}

impl Default for OrderType {
    fn default() -> Self {
        Self::Limit
    }
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Good till date
    GTD,
    /// Post only (maker only)
    PostOnly,
}

impl Default for TimeInForce {
    fn default() -> Self {
        Self::GTC
    }
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Venue/exchange type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VenueType {
    /// Native LX DEX
    Native,
    /// CCXT-compatible exchange
    Ccxt,
    /// Hummingbot Gateway
    Hummingbot,
    /// Custom adapter
    Custom,
}

/// Trading pair
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    pub base: String,
    pub quote: String,
}

impl TradingPair {
    pub fn new(base: impl Into<String>, quote: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            quote: quote.into(),
        }
    }

    pub fn from_symbol(symbol: &str) -> Option<Self> {
        let parts: Vec<&str> = symbol.split(&['-', '/', '_'][..]).collect();
        if parts.len() == 2 {
            Some(Self::new(parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Format as "BASE-QUOTE" (Hummingbot style)
    pub fn to_hummingbot(&self) -> String {
        format!("{}-{}", self.base, self.quote)
    }

    /// Format as "BASE/QUOTE" (CCXT style)
    pub fn to_ccxt(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }

    /// Format as "BASEQUOTE" (exchange style)
    pub fn to_exchange(&self) -> String {
        format!("{}{}", self.base, self.quote)
    }
}

impl fmt::Display for TradingPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.base, self.quote)
    }
}

/// Price level in orderbook
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

impl PriceLevel {
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self { price, quantity }
    }

    pub fn value(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// Order request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    /// Client order ID
    pub client_order_id: String,
    /// Trading pair
    pub symbol: String,
    /// Buy or sell
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Quantity
    pub quantity: Decimal,
    /// Price (required for limit orders)
    pub price: Option<Decimal>,
    /// Stop price (for stop orders)
    pub stop_price: Option<Decimal>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Reduce only (futures)
    pub reduce_only: bool,
    /// Post only (maker only)
    pub post_only: bool,
    /// Target venue (None = smart routing)
    pub venue: Option<String>,
}

impl OrderRequest {
    pub fn market(symbol: impl Into<String>, side: Side, quantity: Decimal) -> Self {
        Self {
            client_order_id: Uuid::new_v4().to_string(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
            post_only: false,
            venue: None,
        }
    }

    pub fn limit(symbol: impl Into<String>, side: Side, quantity: Decimal, price: Decimal) -> Self {
        Self {
            client_order_id: Uuid::new_v4().to_string(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            reduce_only: false,
            post_only: false,
            venue: None,
        }
    }

    pub fn with_venue(mut self, venue: impl Into<String>) -> Self {
        self.venue = Some(venue.into());
        self
    }

    pub fn with_client_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = id.into();
        self
    }

    pub fn post_only(mut self) -> Self {
        self.post_only = true;
        self.time_in_force = TimeInForce::PostOnly;
        self
    }
}

/// Order response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Exchange order ID
    pub order_id: String,
    /// Client order ID
    pub client_order_id: String,
    /// Trading pair
    pub symbol: String,
    /// Venue that executed the order
    pub venue: String,
    /// Buy or sell
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Order status
    pub status: OrderStatus,
    /// Original quantity
    pub quantity: Decimal,
    /// Filled quantity
    pub filled_quantity: Decimal,
    /// Remaining quantity
    pub remaining_quantity: Decimal,
    /// Order price
    pub price: Option<Decimal>,
    /// Average fill price
    pub average_price: Option<Decimal>,
    /// Creation timestamp (ms)
    pub created_at: i64,
    /// Last update timestamp (ms)
    pub updated_at: i64,
    /// Fees paid
    pub fees: Vec<Fee>,
}

impl Order {
    pub fn is_open(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Open | OrderStatus::PartiallyFilled | OrderStatus::Pending
        )
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    pub fn fill_percent(&self) -> Decimal {
        if self.quantity.is_zero() {
            Decimal::ZERO
        } else {
            (self.filled_quantity / self.quantity) * Decimal::from(100)
        }
    }
}

/// Trade/fill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: String,
    pub order_id: String,
    pub symbol: String,
    pub venue: String,
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub fee: Fee,
    pub timestamp: i64,
    pub is_maker: bool,
}

impl Trade {
    pub fn value(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// Fee information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fee {
    pub asset: String,
    pub amount: Decimal,
    pub rate: Option<Decimal>,
}

/// Balance for a single asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub venue: String,
    pub free: Decimal,
    pub locked: Decimal,
    pub total: Decimal,
}

impl Balance {
    pub fn new(
        asset: impl Into<String>,
        venue: impl Into<String>,
        free: Decimal,
        locked: Decimal,
    ) -> Self {
        Self {
            asset: asset.into(),
            venue: venue.into(),
            free,
            locked,
            total: free + locked,
        }
    }
}

/// Aggregated balance across venues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedBalance {
    pub asset: String,
    pub total_free: Decimal,
    pub total_locked: Decimal,
    pub total: Decimal,
    pub by_venue: Vec<Balance>,
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub venue: String,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub last: Option<Decimal>,
    pub volume_24h: Option<Decimal>,
    pub high_24h: Option<Decimal>,
    pub low_24h: Option<Decimal>,
    pub change_24h: Option<Decimal>,
    pub timestamp: i64,
}

impl Ticker {
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from(2)),
            _ => self.last,
        }
    }

    pub fn spread(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    pub fn spread_percent(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) if !bid.is_zero() => {
                Some(((ask - bid) / bid) * Decimal::from(100))
            }
            _ => None,
        }
    }
}

/// Venue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueInfo {
    pub name: String,
    pub venue_type: VenueType,
    pub connected: bool,
    pub latency_ms: Option<u64>,
    pub supported_pairs: Vec<String>,
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
}

/// Market info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketInfo {
    pub symbol: String,
    pub base: String,
    pub quote: String,
    pub price_precision: u32,
    pub quantity_precision: u32,
    pub min_quantity: Decimal,
    pub max_quantity: Option<Decimal>,
    pub min_notional: Option<Decimal>,
    pub tick_size: Decimal,
    pub lot_size: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_pair_parsing() {
        let pair = TradingPair::from_symbol("BTC-USDC").unwrap();
        assert_eq!(pair.base, "BTC");
        assert_eq!(pair.quote, "USDC");
        assert_eq!(pair.to_ccxt(), "BTC/USDC");
        assert_eq!(pair.to_hummingbot(), "BTC-USDC");
        assert_eq!(pair.to_exchange(), "BTCUSDC");
    }

    #[test]
    fn test_order_request_builder() {
        let order = OrderRequest::limit(
            "BTC-USDC",
            Side::Buy,
            Decimal::from(1),
            Decimal::from(50000),
        )
        .with_venue("binance")
        .post_only();

        assert_eq!(order.symbol, "BTC-USDC");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.venue, Some("binance".to_string()));
        assert!(order.post_only);
    }

    #[test]
    fn test_ticker_calculations() {
        let ticker = Ticker {
            symbol: "BTC-USDC".to_string(),
            venue: "lx_dex".to_string(),
            bid: Some(Decimal::from(50000)),
            ask: Some(Decimal::from(50010)),
            last: Some(Decimal::from(50005)),
            volume_24h: None,
            high_24h: None,
            low_24h: None,
            change_24h: None,
            timestamp: 0,
        };

        assert_eq!(ticker.mid_price(), Some(Decimal::from(50005)));
        assert_eq!(ticker.spread(), Some(Decimal::from(10)));
    }
}
