//! Integration tests for lx-trading-core.
//!
//! This module provides comprehensive integration tests for the LX Trading SDK.
//! Tests are organized by component and cover:
//!
//! - Configuration parsing and validation
//! - Trading types (pairs, orders, balances)
//! - Orderbook operations and aggregation
//! - Risk management and position tracking
//! - HTTP client and retry logic
//! - Metrics collection
//! - Financial mathematics
//! - WebSocket configuration
//! - Error handling and classification
//!
//! # Running Tests
//!
//! ```bash
//! cargo test --test integration_tests
//! ```
//!
//! # Test Coverage
//!
//! These tests validate the public API contracts and ensure correct behavior
//! across the SDK's core functionality.

use lx_trading::*;
use rust_decimal::prelude::FromStr;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// Config Tests
// =============================================================================

#[test]
fn test_config_default() {
    let config = Config::default();
    assert!(config.native.is_empty());
    assert!(config.ccxt.is_empty());
    assert!(config.hummingbot.is_empty());
    assert!(config.risk.enabled);
}

#[test]
fn test_config_from_toml() {
    let toml = r#"
[general]
log_level = "debug"
smart_routing = true
timeout_ms = 60000

[risk]
enabled = true
max_daily_loss = 1000
max_order_size = 10
max_open_orders = 50

[native.lx_dex]
venue_type = "dex"
api_url = "https://api.dex.lux.network"
network = "mainnet"

[ccxt.binance]
exchange_id = "binance"
sandbox = false
"#;

    let config = Config::from_toml(toml).unwrap();
    assert_eq!(config.general.log_level, "debug");
    assert_eq!(config.general.timeout_ms, 60000);
    assert!(config.native.contains_key("lx_dex"));
    assert!(config.ccxt.contains_key("binance"));
}

#[test]
fn test_native_venue_config_builder() {
    let config = config::NativeVenueConfig::lx_dex("https://api.dex.lux.network")
        .with_credentials("api_key", "api_secret")
        .with_websocket("wss://ws.dex.lux.network")
        .testnet();

    assert_eq!(config.venue_type, "dex");
    assert_eq!(config.api_key, Some("api_key".into()));
    assert_eq!(config.api_secret, Some("api_secret".into()));
    assert_eq!(config.ws_url, Some("wss://ws.dex.lux.network".into()));
    assert_eq!(config.network, "testnet");
}

// =============================================================================
// Types Tests
// =============================================================================

#[test]
fn test_trading_pair_parsing() {
    // Dash separator
    let pair = TradingPair::from_symbol("BTC-USDC").unwrap();
    assert_eq!(pair.base, "BTC");
    assert_eq!(pair.quote, "USDC");

    // Slash separator (CCXT style)
    let pair = TradingPair::from_symbol("ETH/USDT").unwrap();
    assert_eq!(pair.base, "ETH");
    assert_eq!(pair.quote, "USDT");

    // Underscore separator
    let pair = TradingPair::from_symbol("SOL_USD").unwrap();
    assert_eq!(pair.base, "SOL");
    assert_eq!(pair.quote, "USD");

    // Invalid format
    assert!(TradingPair::from_symbol("BTCUSDC").is_none());
}

#[test]
fn test_trading_pair_formatting() {
    let pair = TradingPair::new("BTC", "USDC");

    assert_eq!(pair.to_hummingbot(), "BTC-USDC");
    assert_eq!(pair.to_ccxt(), "BTC/USDC");
    assert_eq!(pair.to_exchange(), "BTCUSDC");
    assert_eq!(pair.to_string(), "BTC-USDC");
}

#[test]
fn test_order_request_market() {
    let order = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(1));

    assert_eq!(order.symbol, "BTC-USDC");
    assert_eq!(order.side, Side::Buy);
    assert_eq!(order.order_type, OrderType::Market);
    assert_eq!(order.quantity, Decimal::from(1));
    assert!(order.price.is_none());
    assert_eq!(order.time_in_force, TimeInForce::IOC);
}

#[test]
fn test_order_request_limit() {
    let order = OrderRequest::limit(
        "ETH-USDC",
        Side::Sell,
        Decimal::from(10),
        Decimal::from(2000),
    );

    assert_eq!(order.symbol, "ETH-USDC");
    assert_eq!(order.side, Side::Sell);
    assert_eq!(order.order_type, OrderType::Limit);
    assert_eq!(order.quantity, Decimal::from(10));
    assert_eq!(order.price, Some(Decimal::from(2000)));
    assert_eq!(order.time_in_force, TimeInForce::GTC);
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
    .with_client_id("my-order-123")
    .post_only();

    assert_eq!(order.venue, Some("binance".into()));
    assert_eq!(order.client_order_id, "my-order-123");
    assert!(order.post_only);
    assert_eq!(order.time_in_force, TimeInForce::PostOnly);
}

#[test]
fn test_price_level() {
    let level = PriceLevel::new(Decimal::from(50000), Decimal::from(2));
    assert_eq!(level.price, Decimal::from(50000));
    assert_eq!(level.quantity, Decimal::from(2));
    assert_eq!(level.value(), Decimal::from(100000));
}

#[test]
fn test_ticker_calculations() {
    let ticker = Ticker {
        symbol: "BTC-USDC".into(),
        venue: "test".into(),
        bid: Some(Decimal::from(50000)),
        ask: Some(Decimal::from(50100)),
        last: Some(Decimal::from(50050)),
        volume_24h: Some(Decimal::from(1000)),
        high_24h: Some(Decimal::from(51000)),
        low_24h: Some(Decimal::from(49000)),
        change_24h: Some(Decimal::from(2)),
        timestamp: 0,
    };

    assert_eq!(ticker.mid_price(), Some(Decimal::from(50050)));
    assert_eq!(ticker.spread(), Some(Decimal::from(100)));
    // spread_percent = 100/50000 * 100 = 0.2%
    let spread_pct = ticker.spread_percent().unwrap();
    assert!(
        (spread_pct - Decimal::from_str("0.2").unwrap()).abs()
            < Decimal::from_str("0.001").unwrap()
    );
}

#[test]
fn test_balance() {
    let balance = Balance::new("BTC", "binance", Decimal::from(10), Decimal::from(2));

    assert_eq!(balance.asset, "BTC");
    assert_eq!(balance.venue, "binance");
    assert_eq!(balance.free, Decimal::from(10));
    assert_eq!(balance.locked, Decimal::from(2));
    assert_eq!(balance.total, Decimal::from(12));
}

#[test]
fn test_order_status() {
    let mut order = Order {
        order_id: "123".into(),
        client_order_id: "client-123".into(),
        symbol: "BTC-USDC".into(),
        venue: "test".into(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        status: OrderStatus::PartiallyFilled,
        quantity: Decimal::from(10),
        filled_quantity: Decimal::from(5),
        remaining_quantity: Decimal::from(5),
        price: Some(Decimal::from(50000)),
        average_price: Some(Decimal::from(49990)),
        created_at: 0,
        updated_at: 0,
        fees: vec![],
    };

    assert!(order.is_open());
    assert!(!order.is_done());
    assert_eq!(order.fill_percent(), Decimal::from(50));

    order.status = OrderStatus::Filled;
    order.filled_quantity = Decimal::from(10);
    order.remaining_quantity = Decimal::ZERO;

    assert!(!order.is_open());
    assert!(order.is_done());
    assert_eq!(order.fill_percent(), Decimal::from(100));
}

// =============================================================================
// Orderbook Tests
// =============================================================================

#[test]
fn test_orderbook_basics() {
    let mut book = orderbook::Orderbook::new("BTC-USDC", "test");

    book.add_bid(Decimal::from(50000), Decimal::from(1));
    book.add_bid(Decimal::from(49900), Decimal::from(2));
    book.add_ask(Decimal::from(50100), Decimal::from(1));
    book.add_ask(Decimal::from(50200), Decimal::from(2));

    book.sort();

    assert_eq!(book.best_bid(), Some(Decimal::from(50000)));
    assert_eq!(book.best_ask(), Some(Decimal::from(50100)));
    assert_eq!(book.spread(), Some(Decimal::from(100)));
    assert_eq!(book.mid_price(), Some(Decimal::from(50050)));
}

#[test]
fn test_orderbook_vwap() {
    let mut book = orderbook::Orderbook::new("BTC-USDC", "test");

    // Bids: 100@50000, 200@49900
    book.add_bid(Decimal::from(50000), Decimal::from(100));
    book.add_bid(Decimal::from(49900), Decimal::from(200));

    // Asks: 100@50100, 200@50200
    book.add_ask(Decimal::from(50100), Decimal::from(100));
    book.add_ask(Decimal::from(50200), Decimal::from(200));

    book.sort();

    // VWAP for buying 150: 100@50100 + 50@50200 = (5010000 + 2510000) / 150 = 50133.33
    let vwap_buy = book.vwap_buy(Decimal::from(150)).unwrap();
    let expected = Decimal::from_str("50133.333333333333333333333333").unwrap();
    assert!((vwap_buy - expected).abs() < Decimal::from_str("0.01").unwrap());
}

#[test]
fn test_orderbook_liquidity() {
    let mut book = orderbook::Orderbook::new("BTC-USDC", "test");

    book.add_bid(Decimal::from(50000), Decimal::from(1)); // 50000 value
    book.add_bid(Decimal::from(49000), Decimal::from(2)); // 98000 value
    book.add_ask(Decimal::from(51000), Decimal::from(1)); // 51000 value

    assert_eq!(book.bid_liquidity(), Decimal::from(148000));
    assert_eq!(book.ask_liquidity(), Decimal::from(51000));

    assert!(book.has_liquidity(Side::Buy, Decimal::from(1)));
    assert!(!book.has_liquidity(Side::Buy, Decimal::from(10)));
}

#[test]
fn test_aggregated_orderbook() {
    let mut agg = orderbook::AggregatedOrderbook::new("BTC-USDC");

    let mut book1 = orderbook::Orderbook::new("BTC-USDC", "venue1");
    book1.add_bid(Decimal::from(50000), Decimal::from(1));
    book1.add_ask(Decimal::from(50100), Decimal::from(1));

    let mut book2 = orderbook::Orderbook::new("BTC-USDC", "venue2");
    book2.add_bid(Decimal::from(50050), Decimal::from(2)); // Better bid
    book2.add_ask(Decimal::from(50080), Decimal::from(2)); // Better ask

    agg.add_orderbook(&book1);
    agg.add_orderbook(&book2);

    // Best bid from venue2
    let (price, venue, qty) = agg.best_bid().unwrap();
    assert_eq!(price, Decimal::from(50050));
    assert_eq!(venue, "venue2");
    assert_eq!(qty, Decimal::from(2));

    // Best ask from venue2
    let (price, venue, qty) = agg.best_ask().unwrap();
    assert_eq!(price, Decimal::from(50080));
    assert_eq!(venue, "venue2");
    assert_eq!(qty, Decimal::from(2));
}

// =============================================================================
// Risk Tests
// =============================================================================

#[test]
fn test_risk_manager_order_validation() {
    let config = config::RiskConfig {
        enabled: true,
        max_position_size: Decimal::from(100),
        max_order_size: Decimal::from(10),
        max_daily_loss: Decimal::from(1000),
        max_open_orders: 5,
        kill_switch_enabled: true,
        position_limits: HashMap::new(),
    };

    let rm = risk::RiskManager::new(config);

    // Valid order
    let valid = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(5));
    assert!(rm.validate_order(&valid).is_ok());

    // Order too large
    let too_large = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(15));
    assert!(rm.validate_order(&too_large).is_err());
}

#[test]
fn test_risk_manager_position_tracking() {
    let config = config::RiskConfig::default();
    let rm = risk::RiskManager::new(config);

    assert_eq!(rm.position("BTC"), Decimal::ZERO);

    rm.update_position("BTC", Decimal::from(10), Side::Buy);
    assert_eq!(rm.position("BTC"), Decimal::from(10));

    rm.update_position("BTC", Decimal::from(3), Side::Sell);
    assert_eq!(rm.position("BTC"), Decimal::from(7));
}

#[test]
fn test_risk_manager_kill_switch() {
    let config = config::RiskConfig {
        enabled: true,
        kill_switch_enabled: true,
        max_daily_loss: Decimal::from(100),
        ..Default::default()
    };

    let rm = risk::RiskManager::new(config);

    assert!(!rm.is_killed());

    // Trigger kill switch via PnL
    rm.update_pnl(Decimal::from(-150)); // Exceeds max daily loss

    assert!(rm.is_killed());

    // Orders should be rejected
    let order = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(1));
    assert!(rm.validate_order(&order).is_err());

    // Reset
    rm.reset();
    assert!(!rm.is_killed());
}

// =============================================================================
// HTTP Client Tests
// =============================================================================

#[test]
fn test_http_config() {
    let config = HttpConfig::new("https://api.example.com")
        .with_timeout(Duration::from_secs(60))
        .with_connect_timeout(Duration::from_secs(5))
        .with_pool_size(64);

    assert_eq!(config.base_url, "https://api.example.com");
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.connect_timeout, Duration::from_secs(5));
    assert_eq!(config.pool_max_idle_per_host, 64);
}

#[test]
fn test_retry_config() {
    let config = RetryConfig::default()
        .with_max_retries(5)
        .with_initial_delay(Duration::from_millis(200))
        .with_max_delay(Duration::from_secs(60));

    assert_eq!(config.max_retries, 5);
    assert_eq!(config.initial_delay, Duration::from_millis(200));
    assert_eq!(config.max_delay, Duration::from_secs(60));
}

#[test]
fn test_retry_delay_exponential_backoff() {
    let config = RetryConfig {
        max_retries: 5,
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(10),
        multiplier: 2.0,
        jitter: 0.0, // No jitter for deterministic test
        retryable_statuses: vec![],
    };

    let delay0 = config.delay_for_attempt(0);
    let delay1 = config.delay_for_attempt(1);
    let delay2 = config.delay_for_attempt(2);

    // 100, 200, 400
    assert!(delay0.as_millis() >= 95 && delay0.as_millis() <= 105);
    assert!(delay1.as_millis() >= 195 && delay1.as_millis() <= 205);
    assert!(delay2.as_millis() >= 395 && delay2.as_millis() <= 405);
}

#[test]
fn test_rate_limiter() {
    let limiter = http::RateLimiter::new(10.0);

    // Should acquire first token
    assert!(limiter.try_acquire());

    // Available should be less than max
    let available = limiter.available();
    assert!(available < 10.0);
}

// =============================================================================
// Metrics Tests
// =============================================================================

#[test]
fn test_metrics_collector() {
    let metrics = MetricsCollector::new();

    metrics.record_order_submitted("venue1");
    metrics.record_order_submitted("venue1");
    metrics.record_order_filled("venue1", Decimal::from(1), Decimal::from(50000));

    let order_metrics = metrics.order_metrics();
    assert_eq!(
        order_metrics
            .submitted
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    assert_eq!(
        order_metrics
            .filled
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_latency_tracking() {
    let metrics = MetricsCollector::new();

    metrics.record_latency("get_orderbook", Duration::from_millis(10));
    metrics.record_latency("get_orderbook", Duration::from_millis(20));
    metrics.record_latency("get_orderbook", Duration::from_millis(30));

    let stats = metrics.latency_stats("get_orderbook").unwrap();
    assert_eq!(stats.count, 3);
    assert_eq!(stats.min, Duration::from_millis(10));
    assert_eq!(stats.max, Duration::from_millis(30));
}

#[test]
fn test_venue_metrics() {
    let metrics = MetricsCollector::new();

    metrics.record_order_submitted("binance");
    metrics.record_order_filled("binance", Decimal::from(1), Decimal::from(50000));
    metrics.record_trade(
        "binance",
        Side::Buy,
        Decimal::from(1),
        Decimal::from(50000),
        Decimal::from(5),
        true,
    );

    let venue = metrics.venue_metrics("binance");
    assert_eq!(
        venue
            .orders_submitted
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(venue.trades.load(std::sync::atomic::Ordering::Relaxed), 1);
}

// =============================================================================
// Math Tests
// =============================================================================

#[test]
fn test_mid_price_calculation() {
    let mid = mid_price(Decimal::from(100), Decimal::from(102));
    assert_eq!(mid, Decimal::from(101));
}

#[test]
fn test_spread_calculations() {
    let bid = Decimal::from(100);
    let ask = Decimal::from(101);

    assert_eq!(spread(bid, ask), Decimal::from(1));
    assert_eq!(spread_percent(bid, ask), Decimal::from(1));
    assert_eq!(spread_bps(bid, ask), Decimal::from(100));
}

#[test]
fn test_vwap_calculation() {
    let trades = vec![
        (Decimal::from(100), Decimal::from(10)),
        (Decimal::from(102), Decimal::from(20)),
        (Decimal::from(101), Decimal::from(10)),
    ];

    let result = vwap(&trades).unwrap();
    // (100*10 + 102*20 + 101*10) / 40 = 101.25
    assert_eq!(result, Decimal::from_str("101.25").unwrap());
}

#[test]
fn test_position_pnl() {
    // Long position profit
    let pnl = position_pnl(
        Decimal::from(100),
        Decimal::from(110),
        Decimal::from(10),
        Side::Buy,
    );
    assert_eq!(pnl, Decimal::from(100));

    // Short position profit
    let pnl = position_pnl(
        Decimal::from(100),
        Decimal::from(90),
        Decimal::from(10),
        Side::Sell,
    );
    assert_eq!(pnl, Decimal::from(100));

    // Long position loss
    let pnl = position_pnl(
        Decimal::from(100),
        Decimal::from(90),
        Decimal::from(10),
        Side::Buy,
    );
    assert_eq!(pnl, Decimal::from(-100));
}

#[test]
fn test_kelly_criterion() {
    // 60% win rate, 2:1 risk/reward
    let kelly = kelly_criterion(
        Decimal::from_str("0.6").unwrap(),
        Decimal::from(2),
        Decimal::from(1),
    );

    // f* = (0.6*2 - 0.4)/2 = 0.4
    let expected = Decimal::from_str("0.4").unwrap();
    assert!((kelly - expected).abs() < Decimal::from_str("0.01").unwrap());
}

#[test]
fn test_max_drawdown() {
    let equity: Vec<Decimal> = vec![100, 120, 100, 80, 90, 110]
        .into_iter()
        .map(Decimal::from)
        .collect();

    let dd = max_drawdown(&equity);
    // Peak 120, trough 80: (120-80)/120 = 33.33%
    let expected = Decimal::from_str("33.333333").unwrap();
    assert!((dd - expected).abs() < Decimal::from_str("0.01").unwrap());
}

#[test]
fn test_moving_averages() {
    let prices: Vec<Decimal> = vec![1, 2, 3, 4, 5].into_iter().map(Decimal::from).collect();

    let sma_val = sma(&prices).unwrap();
    assert_eq!(sma_val, Decimal::from(3));

    let ema_val = ema(&prices, 3).unwrap();
    assert!(ema_val > Decimal::from(3)); // EMA weights recent values higher
}

#[test]
fn test_rolling_stats() {
    let mut stats = math::RollingStats::new(3);

    stats.push(Decimal::from(1));
    stats.push(Decimal::from(2));
    stats.push(Decimal::from(3));

    assert_eq!(stats.mean(), Some(Decimal::from(2)));
    assert_eq!(stats.count(), 3);
    assert_eq!(stats.min(), Some(Decimal::from(1)));
    assert_eq!(stats.max(), Some(Decimal::from(3)));

    // Window rolls
    stats.push(Decimal::from(4));
    assert_eq!(stats.mean(), Some(Decimal::from(3))); // [2, 3, 4]
}

// =============================================================================
// WebSocket Tests
// =============================================================================

#[test]
fn test_ws_config() {
    let config = WsConfig::new("wss://stream.example.com/ws", "test_venue");

    assert_eq!(config.url, "wss://stream.example.com/ws");
    assert_eq!(config.venue, "test_venue");
    assert!(config.auto_reconnect);
    assert_eq!(config.ping_interval, Duration::from_secs(30));
}

// =============================================================================
// Error Tests
// =============================================================================

#[test]
fn test_error_retryable() {
    assert!(Error::RateLimited {
        venue: "test".into(),
        retry_after_ms: 1000
    }
    .is_retryable());

    assert!(Error::Timeout { timeout_ms: 5000 }.is_retryable());
    assert!(Error::NetworkError("connection reset".into()).is_retryable());

    assert!(!Error::OrderRejected("insufficient balance".into()).is_retryable());
    assert!(!Error::InvalidOrder("bad symbol".into()).is_retryable());
}

#[test]
fn test_error_retry_delay() {
    let err = Error::RateLimited {
        venue: "test".into(),
        retry_after_ms: 5000,
    };
    assert_eq!(err.retry_delay_ms(), Some(5000));

    let err = Error::Timeout { timeout_ms: 1000 };
    assert_eq!(err.retry_delay_ms(), Some(1000));

    let err = Error::OrderRejected("test".into());
    assert_eq!(err.retry_delay_ms(), None);
}

// =============================================================================
// Engine Tests
// =============================================================================

#[tokio::test]
async fn test_unified_client_creation() {
    let config = Config::default();
    let client = UnifiedClient::new(config).unwrap();

    // No venues connected initially
    assert!(client.venues().is_empty());
}

// =============================================================================
// Adapter Trait Tests
// =============================================================================

#[test]
fn test_venue_capabilities_order_book() {
    let caps = adapters::VenueCapabilities::order_book();

    assert!(caps.limit_orders);
    assert!(caps.market_orders);
    assert!(caps.cancel_orders);
    assert!(caps.orderbook);
    assert!(!caps.amm_swap);
}

#[test]
fn test_venue_capabilities_amm() {
    let caps = adapters::VenueCapabilities::amm();

    assert!(!caps.limit_orders);
    assert!(caps.market_orders);
    assert!(!caps.cancel_orders);
    assert!(!caps.orderbook);
    assert!(caps.amm_swap);
    assert!(caps.add_liquidity);
}

#[test]
fn test_venue_capabilities_hybrid() {
    let caps = adapters::VenueCapabilities::hybrid();

    assert!(caps.limit_orders);
    assert!(caps.market_orders);
    assert!(caps.amm_swap);
    assert!(caps.orderbook);
    assert!(caps.add_liquidity);
}
