//! Basic trading example for LX DEX.
//!
//! Demonstrates:
//! - Connecting to WebSocket
//! - Authentication
//! - Subscribing to market data
//! - Placing and canceling orders
//! - Handling events

use lux_dex::{Client, ClientConfig, Order, Side, WsEvent};
use std::env;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> lux_dex::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Configuration from environment or defaults
    let ws_url = env::var("LX_WS_URL").unwrap_or_else(|_| "ws://localhost:8081".into());
    let http_url = env::var("LX_HTTP_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let api_key = env::var("LX_API_KEY").ok();
    let api_secret = env::var("LX_API_SECRET").ok();

    let mut config = ClientConfig::default()
        .with_ws_url(&ws_url)
        .with_http_url(&http_url);

    if let (Some(key), Some(secret)) = (api_key, api_secret) {
        config = config.with_credentials(key, secret);
    }

    // Create client
    let client = Client::with_config(config);

    // Take event receiver before connecting
    let mut events = client.take_event_receiver().await.unwrap();

    // Connect to WebSocket
    info!("Connecting to {}", ws_url);
    client.connect().await?;

    // Spawn event handler
    let event_handle = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                WsEvent::Connected { client_id } => {
                    info!("Connected with client_id: {}", client_id);
                }
                WsEvent::Authenticated { user_id } => {
                    info!("Authenticated as user: {}", user_id);
                }
                WsEvent::OrderBook(book) => {
                    info!(
                        "Order book {}: bid={:?} ask={:?} spread={:?}",
                        book.symbol,
                        book.best_bid(),
                        book.best_ask(),
                        book.spread()
                    );
                }
                WsEvent::Trade(trade) => {
                    info!(
                        "Trade: {} {} @ {} (value: {})",
                        trade.symbol,
                        trade.size,
                        trade.price,
                        trade.value()
                    );
                }
                WsEvent::OrderUpdate { order, status } => {
                    info!(
                        "Order update: {} {} {} @ {} - {}",
                        order.order_id.unwrap_or(0),
                        order.side,
                        order.size,
                        order.price,
                        status
                    );
                }
                WsEvent::Position(pos) => {
                    info!(
                        "Position: {} {} size={} entry={} pnl={}",
                        pos.symbol, pos.side, pos.size, pos.entry_price, pos.unrealized_pnl
                    );
                }
                WsEvent::Price { symbol, price } => {
                    info!("Price update: {} = {}", symbol, price);
                }
                WsEvent::Error { message, request_id } => {
                    warn!("Error: {} (request: {:?})", message, request_id);
                }
                WsEvent::Pong => {
                    info!("Pong received");
                }
                _ => {}
            }
        }
    });

    // Wait for connection
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Authenticate if credentials are configured
    if client.is_connected() {
        info!("Attempting authentication...");
        client.authenticate_configured().await.ok();
    }

    // Subscribe to market data
    info!("Subscribing to BTC-USDT order book");
    client.subscribe_orderbook("BTC-USDT").await?;

    info!("Subscribing to ETH-USDT trades");
    client.subscribe_trades("ETH-USDT").await?;

    // Get order book via HTTP
    info!("Fetching order book via HTTP...");
    match client.get_orderbook("BTC-USDT", 10).await {
        Ok(book) => {
            info!(
                "HTTP Order book: {} levels, mid={:?}",
                book.bids.len() + book.asks.len(),
                book.mid_price()
            );

            // Demonstrate slippage estimation
            if let Some(avg_price) = book.estimate_buy_slippage(1.0) {
                info!("Estimated buy price for 1 BTC: {}", avg_price);
            }
        }
        Err(e) => {
            warn!("Failed to fetch order book: {}", e);
        }
    }

    // Place a test order (only if authenticated)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    if client.is_authenticated() {
        info!("Placing test limit order...");
        let order = Order::limit("BTC-USDT", Side::Buy, 45000.0, 0.001)
            .with_client_id("rust-sdk-test-001")
            .post_only();

        client.place_order(&order).await?;
        info!("Order placed");

        // Wait for confirmation
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    // Ping to keep connection alive
    client.ping().await?;

    // Run for a while to receive events
    info!("Listening for events... (press Ctrl+C to exit)");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down...");
        }
        _ = event_handle => {
            info!("Event handler finished");
        }
    }

    // Disconnect
    client.disconnect().await?;
    info!("Disconnected");

    Ok(())
}
