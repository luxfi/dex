//! Core trading types.
//!
//! Zero-copy, cache-friendly structures for high-frequency trading.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;

/// Fixed-point decimal for exact financial arithmetic.
/// Stores value as integer * 10^(-PRECISION).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Decimal {
    value: i64,
}

impl Decimal {
    /// Number of decimal places.
    pub const PRECISION: u32 = 8;
    /// Scale factor (10^PRECISION).
    pub const SCALE: i64 = 100_000_000;

    /// Create a new Decimal from a scaled value.
    #[inline]
    pub const fn from_scaled(scaled: i64) -> Self {
        Self { value: scaled }
    }

    /// Create a Decimal from a floating-point value.
    #[inline]
    pub fn from_f64(d: f64) -> Self {
        Self {
            value: (d * Self::SCALE as f64) as i64,
        }
    }

    /// Parse a Decimal from a string.
    pub fn from_str_decimal(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let negative = s.starts_with('-');
        let s = if negative || s.starts_with('+') {
            &s[1..]
        } else {
            s
        };

        let (int_part, frac_part) = match s.find('.') {
            Some(pos) => (&s[..pos], &s[pos + 1..]),
            None => (s, ""),
        };

        let int_val: i64 = if int_part.is_empty() {
            0
        } else {
            int_part.parse().ok()?
        };

        let frac_val: i64 = if frac_part.is_empty() {
            0
        } else {
            let mut frac_str = frac_part.to_string();
            if frac_str.len() < Self::PRECISION as usize {
                frac_str.push_str(&"0".repeat(Self::PRECISION as usize - frac_str.len()));
            } else if frac_str.len() > Self::PRECISION as usize {
                frac_str.truncate(Self::PRECISION as usize);
            }
            frac_str.parse().ok()?
        };

        let result = int_val * Self::SCALE + frac_val;
        Some(Self {
            value: if negative { -result } else { result },
        })
    }

    /// Convert to floating-point.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.value as f64 / Self::SCALE as f64
    }

    /// Get the scaled integer value.
    #[inline]
    pub const fn scaled_value(self) -> i64 {
        self.value
    }

    /// Absolute value.
    #[inline]
    pub const fn abs(self) -> Self {
        Self {
            value: if self.value < 0 {
                -self.value
            } else {
                self.value
            },
        }
    }

    /// Check if zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    /// Check if positive.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.value > 0
    }

    /// Check if negative.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.value < 0
    }

    /// Zero value.
    #[inline]
    pub const fn zero() -> Self {
        Self { value: 0 }
    }

    /// One value.
    #[inline]
    pub const fn one() -> Self {
        Self { value: Self::SCALE }
    }

    /// Minimum of two decimals.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.value <= other.value {
            self
        } else {
            other
        }
    }

    /// Maximum of two decimals.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.value >= other.value {
            self
        } else {
            other
        }
    }
}

impl Add for Decimal {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl Sub for Decimal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl Mul for Decimal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: (self.value * rhs.value) / Self::SCALE,
        }
    }
}

impl Div for Decimal {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            value: (self.value * Self::SCALE) / rhs.value,
        }
    }
}

impl Neg for Decimal {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { value: -self.value }
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs_val = self.value.abs();
        let int_part = abs_val / Self::SCALE;
        let frac_part = abs_val % Self::SCALE;

        if self.value < 0 {
            write!(f, "-")?;
        }
        write!(f, "{}", int_part)?;

        if frac_part != 0 {
            let frac_str = format!("{:08}", frac_part);
            let trimmed = frac_str.trim_end_matches('0');
            write!(f, ".{}", trimmed)?;
        }

        Ok(())
    }
}

impl FromStr for Decimal {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_decimal(s).ok_or("invalid decimal string")
    }
}

/// Trading side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// Returns the opposite side.
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// Returns string representation.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrderType {
    #[default]
    Market,
    Limit,
    LimitMaker,
    StopLoss,
    StopLossLimit,
    TakeProfit,
    TakeProfitLimit,
}

impl OrderType {
    /// Returns string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::LimitMaker => "limit_maker",
            OrderType::StopLoss => "stop_loss",
            OrderType::StopLossLimit => "stop_loss_limit",
            OrderType::TakeProfit => "take_profit",
            OrderType::TakeProfitLimit => "take_profit_limit",
        }
    }
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Time in force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TimeInForce {
    /// Good till cancelled.
    #[default]
    GTC,
    /// Immediate or cancel.
    IOC,
    /// Fill or kill.
    FOK,
    /// Good till date.
    GTD,
    /// Post only (maker only).
    PostOnly,
}

impl TimeInForce {
    /// Returns string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            TimeInForce::GTC => "GTC",
            TimeInForce::IOC => "IOC",
            TimeInForce::FOK => "FOK",
            TimeInForce::GTD => "GTD",
            TimeInForce::PostOnly => "POST_ONLY",
        }
    }
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrderStatus {
    #[default]
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl OrderStatus {
    /// Returns string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            OrderStatus::Pending => "pending",
            OrderStatus::Open => "open",
            OrderStatus::PartiallyFilled => "partially_filled",
            OrderStatus::Filled => "filled",
            OrderStatus::Cancelled => "cancelled",
            OrderStatus::Rejected => "rejected",
            OrderStatus::Expired => "expired",
        }
    }

    /// Check if order is still active.
    #[inline]
    pub const fn is_open(self) -> bool {
        matches!(
            self,
            OrderStatus::Pending | OrderStatus::Open | OrderStatus::PartiallyFilled
        )
    }

    /// Check if order is terminal.
    #[inline]
    pub const fn is_done(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Trading pair with inline storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TradingPair {
    base: [u8; 16],
    quote: [u8; 16],
}

impl TradingPair {
    /// Parse a trading pair from a symbol string.
    /// Supports separators: '-', '/', '_'
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        const SEPARATORS: [char; 3] = ['-', '/', '_'];

        for sep in SEPARATORS {
            if let Some(pos) = symbol.find(sep) {
                let base_str = &symbol[..pos];
                let quote_str = &symbol[pos + 1..];

                if base_str.len() > 15 || quote_str.len() > 15 {
                    return None;
                }

                let mut base = [0u8; 16];
                let mut quote = [0u8; 16];

                base[..base_str.len()].copy_from_slice(base_str.as_bytes());
                quote[..quote_str.len()].copy_from_slice(quote_str.as_bytes());

                return Some(Self { base, quote });
            }
        }

        None
    }

    /// Get base asset as string.
    pub fn base(&self) -> &str {
        let len = self.base.iter().position(|&b| b == 0).unwrap_or(16);
        std::str::from_utf8(&self.base[..len]).unwrap_or("")
    }

    /// Get quote asset as string.
    pub fn quote(&self) -> &str {
        let len = self.quote.iter().position(|&b| b == 0).unwrap_or(16);
        std::str::from_utf8(&self.quote[..len]).unwrap_or("")
    }

    /// Format as Hummingbot style (BASE-QUOTE).
    pub fn to_hummingbot(&self) -> String {
        format!("{}-{}", self.base(), self.quote())
    }

    /// Format as CCXT style (BASE/QUOTE).
    pub fn to_ccxt(&self) -> String {
        format!("{}/{}", self.base(), self.quote())
    }
}

impl fmt::Display for TradingPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hummingbot())
    }
}

/// Price level in orderbook.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

impl PriceLevel {
    /// Create a new price level.
    #[inline]
    pub const fn new(price: Decimal, quantity: Decimal) -> Self {
        Self { price, quantity }
    }

    /// Total value at this level (price * quantity).
    #[inline]
    pub fn value(self) -> Decimal {
        self.price * self.quantity
    }
}

/// Fee information.
#[derive(Debug, Clone, PartialEq)]
pub struct Fee {
    pub asset: String,
    pub amount: Decimal,
    pub rate: Option<Decimal>,
}

/// Balance for an asset.
#[derive(Debug, Clone, PartialEq)]
pub struct Balance {
    pub asset: String,
    pub venue: String,
    pub free: Decimal,
    pub locked: Decimal,
}

impl Balance {
    /// Total balance (free + locked).
    #[inline]
    pub fn total(&self) -> Decimal {
        self.free + self.locked
    }

    /// Utilization ratio (locked / total).
    pub fn utilization(&self) -> f64 {
        let total = self.total();
        if total.is_positive() {
            self.locked.to_f64() / total.to_f64()
        } else {
            0.0
        }
    }
}

/// Order request builder.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub time_in_force: TimeInForce,
    pub reduce_only: bool,
    pub post_only: bool,
    pub venue: Option<String>,
    pub client_order_id: Option<String>,
}

impl OrderRequest {
    /// Create a market order.
    pub fn market(symbol: impl Into<String>, side: Side, quantity: Decimal) -> Self {
        Self {
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
            client_order_id: None,
        }
    }

    /// Create a limit order.
    pub fn limit(symbol: impl Into<String>, side: Side, quantity: Decimal, price: Decimal) -> Self {
        Self {
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
            client_order_id: None,
        }
    }

    /// Set venue.
    pub fn with_venue(mut self, venue: impl Into<String>) -> Self {
        self.venue = Some(venue.into());
        self
    }

    /// Set post-only flag.
    pub fn with_post_only(mut self) -> Self {
        self.post_only = true;
        self.time_in_force = TimeInForce::PostOnly;
        self
    }

    /// Set client order ID.
    pub fn with_client_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = Some(id.into());
        self
    }

    /// Set time in force.
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set reduce-only flag.
    pub fn with_reduce_only(mut self) -> Self {
        self.reduce_only = true;
        self
    }
}

/// Order.
#[derive(Debug, Clone, PartialEq)]
pub struct Order {
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub venue: String,
    pub side: Side,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub quantity: Decimal,
    pub filled_quantity: Decimal,
    pub remaining_quantity: Decimal,
    pub price: Option<Decimal>,
    pub average_price: Option<Decimal>,
    pub created_at: i64,
    pub updated_at: i64,
    pub fees: Vec<Fee>,
}

impl Order {
    /// Check if order is still open.
    #[inline]
    pub fn is_open(&self) -> bool {
        self.status.is_open()
    }

    /// Check if order is terminal.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.status.is_done()
    }

    /// Fill percentage (0-100).
    pub fn fill_percent(&self) -> Decimal {
        if self.quantity.is_zero() {
            Decimal::zero()
        } else {
            (self.filled_quantity / self.quantity) * Decimal::from_f64(100.0)
        }
    }
}

/// Trade/fill.
#[derive(Debug, Clone, PartialEq)]
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
    /// Total value of trade.
    #[inline]
    pub fn value(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// Ticker data.
#[derive(Debug, Clone, PartialEq)]
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
    /// Mid price (average of bid and ask).
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from_f64(2.0)),
            _ => self.last,
        }
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Spread as percentage of mid price.
    pub fn spread_percent(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) if bid.is_positive() => {
                Some(((ask - bid) / bid) * Decimal::from_f64(100.0))
            }
            _ => None,
        }
    }
}

/// Position.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub symbol: String,
    pub side: Side,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub mark_price: Decimal,
    pub liquidation_price: Option<Decimal>,
    pub leverage: Decimal,
    pub margin: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
}

impl Position {
    /// Calculate unrealized PnL based on mark price.
    pub fn calc_unrealized_pnl(&self) -> Decimal {
        let diff = self.mark_price - self.entry_price;
        match self.side {
            Side::Buy => diff * self.size,
            Side::Sell => -diff * self.size,
        }
    }

    /// PnL as percentage of entry.
    pub fn pnl_percent(&self) -> f64 {
        if self.entry_price.is_positive() {
            ((self.mark_price - self.entry_price).to_f64() / self.entry_price.to_f64()) * 100.0
        } else {
            0.0
        }
    }

    /// Check if position is at risk of liquidation.
    pub fn is_at_risk(&self, threshold: f64) -> bool {
        if let Some(liq) = self.liquidation_price {
            let distance = (self.mark_price - liq).abs().to_f64();
            let percent = distance / self.mark_price.to_f64();
            percent < threshold
        } else {
            false
        }
    }
}

/// Get current timestamp in milliseconds.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Get current timestamp in microseconds.
pub fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Get current timestamp in nanoseconds.
pub fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_arithmetic() {
        let a = Decimal::from_f64(100.5);
        let b = Decimal::from_f64(50.25);

        assert!((a + b).to_f64() - 150.75 < 0.0001);
        assert!((a - b).to_f64() - 50.25 < 0.0001);
        assert!((a * Decimal::from_f64(2.0)).to_f64() - 201.0 < 0.0001);
        assert!((a / Decimal::from_f64(2.0)).to_f64() - 50.25 < 0.0001);
    }

    #[test]
    fn test_decimal_from_string() {
        let d = Decimal::from_str_decimal("123.456").unwrap();
        assert!((d.to_f64() - 123.456).abs() < 0.0001);

        let d2 = Decimal::from_str_decimal("-99.99").unwrap();
        assert!((d2.to_f64() - (-99.99)).abs() < 0.0001);
        assert!(d2.is_negative());
    }

    #[test]
    fn test_decimal_comparison() {
        let a = Decimal::from_f64(10.0);
        let b = Decimal::from_f64(20.0);

        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert!(a == a);
        assert!(a != b);
    }

    #[test]
    fn test_decimal_zero_one() {
        assert!(Decimal::zero().is_zero());
        assert!((Decimal::one().to_f64() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_trading_pair_parsing() {
        // Hyphen separator
        let pair = TradingPair::from_symbol("BTC-USDC").unwrap();
        assert_eq!(pair.base(), "BTC");
        assert_eq!(pair.quote(), "USDC");

        // Slash separator
        let pair = TradingPair::from_symbol("ETH/USD").unwrap();
        assert_eq!(pair.base(), "ETH");
        assert_eq!(pair.quote(), "USD");

        // Underscore separator
        let pair = TradingPair::from_symbol("LUX_USDT").unwrap();
        assert_eq!(pair.base(), "LUX");
        assert_eq!(pair.quote(), "USDT");

        // Invalid
        assert!(TradingPair::from_symbol("INVALID").is_none());
    }

    #[test]
    fn test_trading_pair_format() {
        let pair = TradingPair::from_symbol("BTC-USDC").unwrap();
        assert_eq!(pair.to_hummingbot(), "BTC-USDC");
        assert_eq!(pair.to_ccxt(), "BTC/USDC");
    }

    #[test]
    fn test_order_request_market() {
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1.5));
        assert_eq!(req.symbol, "BTC-USDC");
        assert_eq!(req.side, Side::Buy);
        assert_eq!(req.order_type, OrderType::Market);
        assert!((req.quantity.to_f64() - 1.5).abs() < 0.0001);
        assert_eq!(req.time_in_force, TimeInForce::IOC);
    }

    #[test]
    fn test_order_request_limit() {
        let req = OrderRequest::limit(
            "ETH-USDC",
            Side::Sell,
            Decimal::from_f64(10.0),
            Decimal::from_f64(2000.0),
        );
        assert_eq!(req.symbol, "ETH-USDC");
        assert_eq!(req.side, Side::Sell);
        assert_eq!(req.order_type, OrderType::Limit);
        assert!((req.price.unwrap().to_f64() - 2000.0).abs() < 0.0001);
        assert_eq!(req.time_in_force, TimeInForce::GTC);
    }

    #[test]
    fn test_order_request_builder() {
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1.0))
            .with_venue("lx_dex")
            .with_post_only()
            .with_client_id("my-order-123");

        assert_eq!(req.venue.as_deref(), Some("lx_dex"));
        assert!(req.post_only);
        assert_eq!(req.time_in_force, TimeInForce::PostOnly);
        assert_eq!(req.client_order_id.as_deref(), Some("my-order-123"));
    }

    #[test]
    fn test_order_status() {
        assert!(OrderStatus::Open.is_open());
        assert!(OrderStatus::PartiallyFilled.is_open());
        assert!(OrderStatus::Pending.is_open());

        assert!(OrderStatus::Filled.is_done());
        assert!(OrderStatus::Cancelled.is_done());
        assert!(OrderStatus::Rejected.is_done());
    }

    #[test]
    fn test_ticker_calculations() {
        let ticker = Ticker {
            symbol: "BTC-USDC".to_string(),
            venue: "test".to_string(),
            bid: Some(Decimal::from_f64(100.0)),
            ask: Some(Decimal::from_f64(101.0)),
            last: None,
            volume_24h: None,
            high_24h: None,
            low_24h: None,
            change_24h: None,
            timestamp: 0,
        };

        assert!((ticker.mid_price().unwrap().to_f64() - 100.5).abs() < 0.0001);
        assert!((ticker.spread().unwrap().to_f64() - 1.0).abs() < 0.0001);
        assert!((ticker.spread_percent().unwrap().to_f64() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn test_enum_display() {
        assert_eq!(Side::Buy.as_str(), "buy");
        assert_eq!(Side::Sell.as_str(), "sell");
        assert_eq!(OrderType::Market.as_str(), "market");
        assert_eq!(OrderType::Limit.as_str(), "limit");
        assert_eq!(TimeInForce::GTC.as_str(), "GTC");
        assert_eq!(OrderStatus::Filled.as_str(), "filled");
    }
}
