//! WebSocket Client Example
//!
//! This example demonstrates real-time market data streaming:
//! - Connecting to WebSocket feeds
//! - Subscribing to orderbook, trade, and ticker channels
//! - Processing streaming events
//! - Managing connection lifecycle
//!
//! # Running
//!
//! ```bash
//! cargo run --example websocket_client
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};

use lx_trading::{
    ws::{Fill, OrderUpdate, OrderbookUpdate, WsConfig, WsEvent},
    PriceLevel, Side, Ticker, Trade,
};

/// Event statistics
struct EventStats {
    orderbook_updates: AtomicU64,
    trades: AtomicU64,
    tickers: AtomicU64,
    order_updates: AtomicU64,
    fills: AtomicU64,
    errors: AtomicU64,
}

impl EventStats {
    fn new() -> Self {
        Self {
            orderbook_updates: AtomicU64::new(0),
            trades: AtomicU64::new(0),
            tickers: AtomicU64::new(0),
            order_updates: AtomicU64::new(0),
            fills: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    fn print_summary(&self) {
        println!("\n=== Event Statistics ===");
        println!(
            "Orderbook Updates: {}",
            self.orderbook_updates.load(Ordering::Relaxed)
        );
        println!("Trades: {}", self.trades.load(Ordering::Relaxed));
        println!("Tickers: {}", self.tickers.load(Ordering::Relaxed));
        println!("Order Updates: {}", self.order_updates.load(Ordering::Relaxed));
        println!("Fills: {}", self.fills.load(Ordering::Relaxed));
        println!("Errors: {}", self.errors.load(Ordering::Relaxed));
    }
}

/// Process an orderbook update
fn handle_orderbook_update(update: &OrderbookUpdate, stats: &EventStats) {
    stats.orderbook_updates.fetch_add(1, Ordering::Relaxed);

    let update_type = if update.is_snapshot { "SNAPSHOT" } else { "DELTA" };
    println!(
        "[ORDERBOOK] {} {} | Bids: {} | Asks: {}",
        update.symbol,
        update_type,
        update.bids.len(),
        update.asks.len()
    );

    // Show best levels
    if let (Some(best_bid), Some(best_ask)) = (update.bids.first(), update.asks.first()) {
        println!(
            "  Best: {} @ {} | {} @ {}",
            best_bid.quantity, best_bid.price, best_ask.quantity, best_ask.price
        );
    }
}

/// Process a trade
fn handle_trade(trade: &Trade, stats: &EventStats) {
    stats.trades.fetch_add(1, Ordering::Relaxed);

    let side_str = match trade.side {
        Side::Buy => "BUY ",
        Side::Sell => "SELL",
    };

    println!(
        "[TRADE] {} {} {} @ {} | Fee: {} {}",
        trade.symbol, side_str, trade.quantity, trade.price, trade.fee.amount, trade.fee.asset
    );
}

/// Process a ticker update
fn handle_ticker(ticker: &Ticker, stats: &EventStats) {
    stats.tickers.fetch_add(1, Ordering::Relaxed);

    let spread = match (ticker.bid, ticker.ask) {
        (Some(b), Some(a)) => Some(a - b),
        _ => None,
    };

    println!(
        "[TICKER] {} | Bid: {:?} | Ask: {:?} | Spread: {:?}",
        ticker.symbol, ticker.bid, ticker.ask, spread
    );
}

/// Process an order update
fn handle_order_update(update: &OrderUpdate, stats: &EventStats) {
    stats.order_updates.fetch_add(1, Ordering::Relaxed);

    println!(
        "[ORDER] {} | Status: {:?} | Filled: {} | ID: {}",
        update.symbol, update.status, update.filled_quantity, update.order_id
    );
}

/// Process a fill notification
fn handle_fill(fill: &Fill, stats: &EventStats) {
    stats.fills.fetch_add(1, Ordering::Relaxed);

    let side_str = match fill.side {
        Side::Buy => "BUY ",
        Side::Sell => "SELL",
    };

    println!(
        "[FILL] {} {} {} @ {} | Order: {}",
        fill.symbol, side_str, fill.quantity, fill.price, fill.order_id
    );
}

/// Simulate WebSocket events for demonstration
async fn simulate_events(tx: mpsc::Sender<WsEvent>, running: Arc<AtomicBool>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut interval = interval(Duration::from_millis(200));
    let mut rng = StdRng::from_entropy();
    let base_price = dec!(50000);
    let mut price_offset = dec!(0);

    while running.load(Ordering::Relaxed) {
        interval.tick().await;

        // Simulate price movement
        let change: f64 = rng.gen_range(-0.0005..0.0005);
        price_offset += Decimal::try_from(change * 50000.0).unwrap_or_default();
        let current_price = base_price + price_offset;

        // Generate random event type
        let event_type: u8 = rng.gen_range(0..5);

        let event = match event_type {
            0 => {
                // Orderbook update
                let is_snapshot = rng.gen_bool(0.1);
                WsEvent::OrderbookUpdate(OrderbookUpdate {
                    symbol: "BTC-USDC".to_string(),
                    venue: "lx_dex".to_string(),
                    is_snapshot,
                    bids: vec![
                        PriceLevel::new(current_price - dec!(5), Decimal::try_from(rng.gen_range(0.1..2.0) as f64).unwrap_or_default()),
                        PriceLevel::new(current_price - dec!(10), Decimal::try_from(rng.gen_range(0.5..3.0) as f64).unwrap_or_default()),
                    ],
                    asks: vec![
                        PriceLevel::new(current_price + dec!(5), Decimal::try_from(rng.gen_range(0.1..2.0) as f64).unwrap_or_default()),
                        PriceLevel::new(current_price + dec!(10), Decimal::try_from(rng.gen_range(0.5..3.0) as f64).unwrap_or_default()),
                    ],
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    sequence: rng.gen(),
                })
            }
            1 => {
                // Trade
                WsEvent::Trade(Trade {
                    trade_id: format!("trade-{}", rng.gen::<u64>()),
                    order_id: format!("order-{}", rng.gen::<u64>()),
                    symbol: "BTC-USDC".to_string(),
                    venue: "lx_dex".to_string(),
                    side: if rng.gen_bool(0.5) { Side::Buy } else { Side::Sell },
                    price: current_price,
                    quantity: Decimal::try_from(rng.gen_range(0.01..0.5) as f64).unwrap_or_default(),
                    fee: lx_trading::Fee {
                        asset: "USDC".to_string(),
                        amount: Decimal::try_from(rng.gen_range(0.01..1.0) as f64).unwrap_or_default(),
                        rate: Some(dec!(0.001)),
                    },
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    is_maker: rng.gen_bool(0.5),
                })
            }
            2 => {
                // Ticker
                WsEvent::Ticker(Ticker {
                    symbol: "BTC-USDC".to_string(),
                    venue: "lx_dex".to_string(),
                    bid: Some(current_price - dec!(5)),
                    ask: Some(current_price + dec!(5)),
                    last: Some(current_price),
                    volume_24h: Some(dec!(1000)),
                    high_24h: Some(base_price + dec!(500)),
                    low_24h: Some(base_price - dec!(500)),
                    change_24h: Some((price_offset / base_price) * dec!(100)),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                })
            }
            _ => continue,
        };

        if tx.send(event).await.is_err() {
            break;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== LX Trading SDK - WebSocket Client Example ===\n");

    // Create WebSocket configuration
    let ws_config = WsConfig::new("wss://ws.dex.lux.network/stream", "lx_dex");

    println!("WebSocket Configuration:");
    println!("  URL: {}", ws_config.url);
    println!("  Venue: {}", ws_config.venue);
    println!("  Auto Reconnect: {}", ws_config.auto_reconnect);
    println!("  Ping Interval: {:?}", ws_config.ping_interval);
    println!();

    // Initialize statistics
    let stats = Arc::new(EventStats::new());
    let running = Arc::new(AtomicBool::new(true));

    // Create event channel
    let (tx, mut rx) = mpsc::channel::<WsEvent>(1024);

    // Start event simulator (in production, this would be actual WebSocket connection)
    let running_clone = running.clone();
    let simulator = tokio::spawn(async move {
        simulate_events(tx, running_clone).await;
    });

    println!("Subscriptions:");
    println!("  - Orderbook (BTC-USDC)");
    println!("  - Trades (BTC-USDC)");
    println!("  - Ticker (BTC-USDC)");
    println!();

    println!("Processing events for 5 seconds...\n");

    // Process events with timeout
    let stats_clone = stats.clone();
    let event_processor = tokio::spawn(async move {
        let timeout_duration = Duration::from_secs(5);
        let result = timeout(timeout_duration, async {
            while let Some(event) = rx.recv().await {
                match event {
                    WsEvent::OrderbookUpdate(update) => {
                        handle_orderbook_update(&update, &stats_clone);
                    }
                    WsEvent::Trade(trade) => {
                        handle_trade(&trade, &stats_clone);
                    }
                    WsEvent::Ticker(ticker) => {
                        handle_ticker(&ticker, &stats_clone);
                    }
                    WsEvent::OrderUpdate(update) => {
                        handle_order_update(&update, &stats_clone);
                    }
                    WsEvent::Fill(fill) => {
                        handle_fill(&fill, &stats_clone);
                    }
                    WsEvent::Connected { venue } => {
                        println!("[CONNECTED] {}", venue);
                    }
                    WsEvent::Disconnected { venue, reason } => {
                        println!("[DISCONNECTED] {} - {}", venue, reason);
                    }
                    WsEvent::Error { venue, message } => {
                        stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                        println!("[ERROR] {} - {}", venue, message);
                    }
                    WsEvent::Subscribed { channel, symbol } => {
                        println!("[SUBSCRIBED] {}:{}", channel, symbol);
                    }
                    WsEvent::Unsubscribed { channel, symbol } => {
                        println!("[UNSUBSCRIBED] {}:{}", channel, symbol);
                    }
                }
            }
        })
        .await;

        if result.is_err() {
            println!("\nTimeout reached, stopping event processing...");
        }
    });

    // Wait for processor
    let _ = event_processor.await;

    // Stop simulator
    running.store(false, Ordering::Relaxed);
    let _ = simulator.await;

    // Print statistics
    stats.print_summary();

    println!("\n=== Example Complete ===");
    println!("\nKey concepts demonstrated:");
    println!("1. WebSocket configuration");
    println!("2. Channel subscriptions (orderbook, trades, ticker)");
    println!("3. Event processing patterns");
    println!("4. Statistics collection");

    Ok(())
}
