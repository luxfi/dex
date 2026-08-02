//! Basic Trading Example
//!
//! This example demonstrates simple spot trading operations with the LX Trading SDK.
//! It shows how to:
//! - Configure and initialize the trading client
//! - Connect to trading venues
//! - Fetch market data (tickers, orderbooks)
//! - Place and manage orders
//!
//! # Running
//!
//! ```bash
//! cargo run --example basic_trading
//! ```

use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use lx_trading::{
    config::{NativeVenueConfig, RiskConfig},
    Config, OrderRequest, Side, UnifiedClient,
};

/// Example configuration for paper trading
fn create_config() -> Config {
    let mut config = Config::default();

    // Configure native LX DEX venue
    config.native.insert(
        "lx_dex".to_string(),
        NativeVenueConfig::lx_dex("https://api.dex.lux.network")
            .testnet()
            .with_websocket("wss://ws.dex.lux.network"),
    );

    // Configure risk management
    config.risk = RiskConfig {
        enabled: true,
        max_position_size: dec!(100),
        max_order_size: dec!(10),
        max_daily_loss: dec!(1000),
        max_open_orders: 50,
        kill_switch_enabled: true,
        ..Default::default()
    };

    // General settings
    config.general.log_level = "info".to_string();
    config.general.smart_routing = true;
    config.general.timeout_ms = 30000;

    config
}

/// Print orderbook summary
fn print_orderbook_summary(symbol: &str, book: &lx_trading::orderbook::Orderbook) {
    println!("\n=== Orderbook: {} ===", symbol);
    println!("Best Bid: {:?}", book.best_bid());
    println!("Best Ask: {:?}", book.best_ask());
    println!("Spread: {:?}", book.spread());
    println!("Mid Price: {:?}", book.mid_price());
    println!("Bid Levels: {}", book.bids.len());
    println!("Ask Levels: {}", book.asks.len());
}

/// Print ticker information
fn print_ticker(ticker: &lx_trading::Ticker) {
    println!("\n=== Ticker: {} ===", ticker.symbol);
    println!("Bid: {:?}", ticker.bid);
    println!("Ask: {:?}", ticker.ask);
    println!("Last: {:?}", ticker.last);
    println!("24h Volume: {:?}", ticker.volume_24h);
    println!("24h Change: {:?}%", ticker.change_24h);
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== LX Trading SDK - Basic Trading Example ===\n");

    // Create configuration
    let config = create_config();
    println!("[1] Configuration created");

    // Initialize client
    let client = UnifiedClient::new(config)?;
    println!("[2] UnifiedClient created");

    // In a real scenario, we would initialize venues
    // For this example, we'll demonstrate the API without actual connections
    println!("[3] Venues would be initialized with: client.init().await?");

    // Example: Market data operations (would work with connected venue)
    println!("\n--- Market Data Operations ---");

    // Simulated ticker data for demonstration
    let ticker = lx_trading::Ticker {
        symbol: "BTC-USDC".to_string(),
        venue: "lx_dex".to_string(),
        bid: Some(dec!(50000)),
        ask: Some(dec!(50010)),
        last: Some(dec!(50005)),
        volume_24h: Some(dec!(1000)),
        high_24h: Some(dec!(51000)),
        low_24h: Some(dec!(49000)),
        change_24h: Some(dec!(2.5)),
        timestamp: chrono::Utc::now().timestamp_millis(),
    };
    print_ticker(&ticker);

    // Simulated orderbook
    let mut book = lx_trading::orderbook::Orderbook::new("BTC-USDC", "lx_dex");
    book.add_bid(dec!(50000), dec!(1.5));
    book.add_bid(dec!(49990), dec!(2.0));
    book.add_bid(dec!(49980), dec!(3.0));
    book.add_ask(dec!(50010), dec!(1.0));
    book.add_ask(dec!(50020), dec!(2.5));
    book.add_ask(dec!(50030), dec!(4.0));
    book.sort();
    print_orderbook_summary("BTC-USDC", &book);

    // Example: Order creation
    println!("\n--- Order Operations ---");

    // Create a market buy order
    let market_order = OrderRequest::market("BTC-USDC", Side::Buy, dec!(0.1));
    println!("\nMarket Order: {:?}", market_order);

    // Create a limit buy order
    let limit_order = OrderRequest::limit("BTC-USDC", Side::Buy, dec!(0.5), dec!(49500))
        .with_venue("lx_dex")
        .with_client_id("my-order-001");
    println!("\nLimit Order: {:?}", limit_order);

    // Create a post-only limit order (maker only)
    let maker_order = OrderRequest::limit("ETH-USDC", Side::Sell, dec!(10.0), dec!(3100))
        .with_venue("lx_dex")
        .with_client_id("maker-001")
        .post_only();
    println!("\nPost-Only Order: {:?}", maker_order);

    // Example: Calculate execution cost using orderbook
    println!("\n--- Execution Analysis ---");
    let buy_quantity = dec!(2.5);
    if let Some(vwap) = book.vwap_buy(buy_quantity) {
        println!("VWAP to buy {} BTC: ${}", buy_quantity, vwap.round_dp(2));
        if let Some(best_ask) = book.best_ask() {
            let slippage = ((vwap - best_ask) / best_ask) * dec!(100);
            println!("Expected slippage: {}%", slippage.round_dp(4));
        }
    }

    // Example: Use math functions
    println!("\n--- Financial Math ---");
    let entry = dec!(50000);
    let current = dec!(52000);
    let quantity = dec!(1.0);
    let pnl = lx_trading::position_pnl(entry, current, quantity, Side::Buy);
    let pnl_pct = lx_trading::position_pnl_percent(entry, current, Side::Buy);
    println!("Position P&L: ${} ({:.2}%)", pnl, pnl_pct);

    // Kelly criterion for position sizing
    let kelly = lx_trading::kelly_criterion(dec!(0.55), dec!(2.0), dec!(1.0));
    println!("Kelly Criterion: {:.2}% of capital", kelly * dec!(100));

    println!("\n=== Example Complete ===");
    println!("\nIn production:");
    println!("1. Use real API credentials");
    println!("2. Connect to venues: client.init().await?");
    println!("3. Place orders: client.place_order(order).await?");
    println!("4. Monitor fills via WebSocket streams");

    Ok(())
}
