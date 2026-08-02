//! WebSocket module for real-time market data streaming.
//!
//! Provides:
//! - Connection management with auto-reconnect
//! - Subscription management for orderbook, trades, and user data
//! - Async streams for consuming events

use futures::{SinkExt, Stream, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
};
use url::Url;

use crate::error::{Error, Result};
use crate::types::*;

/// WebSocket event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsEvent {
    /// Orderbook update (snapshot or delta)
    OrderbookUpdate(OrderbookUpdate),
    /// Trade executed
    Trade(Trade),
    /// Ticker update
    Ticker(Ticker),
    /// Order status update
    OrderUpdate(OrderUpdate),
    /// Fill notification
    Fill(Fill),
    /// Connection status change
    Connected {
        venue: String,
    },
    Disconnected {
        venue: String,
        reason: String,
    },
    /// Error
    Error {
        venue: String,
        message: String,
    },
    /// Subscription confirmation
    Subscribed {
        channel: String,
        symbol: String,
    },
    Unsubscribed {
        channel: String,
        symbol: String,
    },
}

/// Orderbook update (can be snapshot or delta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookUpdate {
    pub symbol: String,
    pub venue: String,
    pub is_snapshot: bool,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: i64,
    pub sequence: u64,
}

/// Order status update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub venue: String,
    pub status: OrderStatus,
    pub filled_quantity: rust_decimal::Decimal,
    pub remaining_quantity: rust_decimal::Decimal,
    pub average_price: Option<rust_decimal::Decimal>,
    pub timestamp: i64,
}

/// Fill notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub trade_id: String,
    pub order_id: String,
    pub symbol: String,
    pub venue: String,
    pub side: Side,
    pub price: rust_decimal::Decimal,
    pub quantity: rust_decimal::Decimal,
    pub fee: Fee,
    pub timestamp: i64,
    pub is_maker: bool,
}

/// WebSocket connection configuration
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// WebSocket URL
    pub url: String,
    /// Venue name
    pub venue: String,
    /// Reconnect on disconnect
    pub auto_reconnect: bool,
    /// Initial reconnect delay
    pub reconnect_delay: Duration,
    /// Maximum reconnect delay
    pub max_reconnect_delay: Duration,
    /// Ping interval
    pub ping_interval: Duration,
    /// Pong timeout
    pub pong_timeout: Duration,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            venue: String::new(),
            auto_reconnect: true,
            reconnect_delay: Duration::from_secs(1),
            max_reconnect_delay: Duration::from_secs(60),
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(10),
        }
    }
}

impl WsConfig {
    pub fn new(url: impl Into<String>, venue: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            venue: venue.into(),
            ..Default::default()
        }
    }
}

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// WebSocket subscription request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub channel: SubscriptionChannel,
    pub symbol: String,
}

/// Subscription channels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionChannel {
    /// Orderbook updates
    Orderbook,
    /// Trade stream
    Trades,
    /// Ticker updates
    Ticker,
    /// User order updates (requires auth)
    Orders,
    /// User fills (requires auth)
    Fills,
}

/// Internal command for WebSocket manager
enum WsCommand {
    Subscribe(Subscription),
    Unsubscribe(Subscription),
    #[allow(dead_code)]
    SendMessage(String),
    Disconnect,
}

/// WebSocket connection manager
pub struct WsConnection {
    config: WsConfig,
    state: Arc<RwLock<ConnectionState>>,
    connected: AtomicBool,
    reconnect_count: AtomicU64,
    subscriptions: Arc<RwLock<HashSet<String>>>,
    event_tx: broadcast::Sender<WsEvent>,
    command_tx: mpsc::Sender<WsCommand>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl WsConnection {
    /// Create a new WebSocket connection
    pub fn new(config: WsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        let (command_tx, _) = mpsc::channel(256);

        Self {
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            connected: AtomicBool::new(false),
            reconnect_count: AtomicU64::new(0),
            subscriptions: Arc::new(RwLock::new(HashSet::new())),
            event_tx,
            command_tx,
            handle: None,
        }
    }

    /// Connect to the WebSocket server
    pub async fn connect(&mut self) -> Result<()> {
        if self.connected.load(Ordering::Relaxed) {
            return Ok(());
        }

        *self.state.write() = ConnectionState::Connecting;

        let url = Url::parse(&self.config.url)?;
        let (ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| Error::WebsocketError(format!("Failed to connect: {e}")))?;

        let (write, read) = ws_stream.split();
        let (command_tx, command_rx) = mpsc::channel(256);
        self.command_tx = command_tx;

        let config = self.config.clone();
        let state = self.state.clone();
        let connected = AtomicBool::new(self.connected.load(Ordering::Relaxed));
        let reconnect_count = AtomicU64::new(0);
        let subscriptions = self.subscriptions.clone();
        let event_tx = self.event_tx.clone();

        self.handle = Some(tokio::spawn(async move {
            run_ws_loop(
                config,
                state,
                connected,
                reconnect_count,
                subscriptions,
                event_tx,
                command_rx,
                write,
                read,
            )
            .await;
        }));

        self.connected.store(true, Ordering::Relaxed);
        *self.state.write() = ConnectionState::Connected;

        // Send connected event
        let _ = self.event_tx.send(WsEvent::Connected {
            venue: self.config.venue.clone(),
        });

        Ok(())
    }

    /// Disconnect from the WebSocket server
    pub async fn disconnect(&mut self) -> Result<()> {
        if !self.connected.load(Ordering::Relaxed) {
            return Ok(());
        }

        let _ = self.command_tx.send(WsCommand::Disconnect).await;
        self.connected.store(false, Ordering::Relaxed);
        *self.state.write() = ConnectionState::Disconnected;

        if let Some(handle) = self.handle.take() {
            handle.abort();
        }

        let _ = self.event_tx.send(WsEvent::Disconnected {
            venue: self.config.venue.clone(),
            reason: "User requested disconnect".into(),
        });

        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Get connection state
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Subscribe to a channel
    pub async fn subscribe(&self, channel: SubscriptionChannel, symbol: &str) -> Result<()> {
        let sub = Subscription {
            channel,
            symbol: symbol.to_string(),
        };

        self.command_tx
            .send(WsCommand::Subscribe(sub.clone()))
            .await
            .map_err(|e| Error::WebsocketError(format!("Failed to send subscribe: {e}")))?;

        let key = format!("{}:{}", channel_to_str(channel), symbol);
        self.subscriptions.write().insert(key);

        Ok(())
    }

    /// Unsubscribe from a channel
    pub async fn unsubscribe(&self, channel: SubscriptionChannel, symbol: &str) -> Result<()> {
        let sub = Subscription {
            channel,
            symbol: symbol.to_string(),
        };

        self.command_tx
            .send(WsCommand::Unsubscribe(sub))
            .await
            .map_err(|e| Error::WebsocketError(format!("Failed to send unsubscribe: {e}")))?;

        let key = format!("{}:{}", channel_to_str(channel), symbol);
        self.subscriptions.write().remove(&key);

        Ok(())
    }

    /// Get event receiver
    pub fn subscribe_events(&self) -> broadcast::Receiver<WsEvent> {
        self.event_tx.subscribe()
    }

    /// Create an async stream of events
    pub fn event_stream(&self) -> WsEventStream {
        WsEventStream::new(self.event_tx.subscribe())
    }

    /// Get active subscriptions
    pub fn subscriptions(&self) -> Vec<String> {
        self.subscriptions.read().iter().cloned().collect()
    }

    /// Get reconnect count
    pub fn reconnect_count(&self) -> u64 {
        self.reconnect_count.load(Ordering::Relaxed)
    }
}

impl Drop for WsConnection {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// Async stream of WebSocket events
pub struct WsEventStream {
    rx: broadcast::Receiver<WsEvent>,
}

impl WsEventStream {
    fn new(rx: broadcast::Receiver<WsEvent>) -> Self {
        Self { rx }
    }
}

impl Stream for WsEventStream {
    type Item = WsEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.try_recv() {
            Ok(event) => Poll::Ready(Some(event)),
            Err(broadcast::error::TryRecvError::Empty) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Closed) => Poll::Ready(None),
        }
    }
}

/// WebSocket message handler
#[allow(clippy::too_many_arguments)]
async fn run_ws_loop<S, R>(
    config: WsConfig,
    state: Arc<RwLock<ConnectionState>>,
    connected: AtomicBool,
    reconnect_count: AtomicU64,
    _subscriptions: Arc<RwLock<HashSet<String>>>,
    event_tx: broadcast::Sender<WsEvent>,
    mut command_rx: mpsc::Receiver<WsCommand>,
    mut write: S,
    mut read: R,
) where
    S: SinkExt<Message, Error = tungstenite::Error> + Unpin,
    R: StreamExt<Item = std::result::Result<Message, tungstenite::Error>> + Unpin,
{
    let mut ping_interval = tokio::time::interval(config.ping_interval);
    let mut reconnect_delay = config.reconnect_delay;

    loop {
        tokio::select! {
            // Handle incoming messages
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(event) = parse_ws_message(&text, &config.venue) {
                            let _ = event_tx.send(event);
                        }
                    }
                    Some(Ok(Message::Binary(data))) => {
                        if let Ok(text) = String::from_utf8(data) {
                            if let Ok(event) = parse_ws_message(&text, &config.venue) {
                                let _ = event_tx.send(event);
                            }
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Pong(_))) => {
                        // Connection is alive
                        reconnect_delay = config.reconnect_delay;
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        connected.store(false, Ordering::Relaxed);
                        *state.write() = ConnectionState::Disconnected;

                        let _ = event_tx.send(WsEvent::Disconnected {
                            venue: config.venue.clone(),
                            reason: "Connection closed".into(),
                        });

                        if config.auto_reconnect {
                            // Attempt reconnect
                            *state.write() = ConnectionState::Reconnecting;
                            reconnect_count.fetch_add(1, Ordering::Relaxed);
                            tokio::time::sleep(reconnect_delay).await;
                            reconnect_delay = (reconnect_delay * 2).min(config.max_reconnect_delay);
                        } else {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = event_tx.send(WsEvent::Error {
                            venue: config.venue.clone(),
                            message: e.to_string(),
                        });
                    }
                    _ => {}
                }
            }

            // Handle commands
            cmd = command_rx.recv() => {
                match cmd {
                    Some(WsCommand::Subscribe(sub)) => {
                        let msg = format_subscribe_message(&sub);
                        let _ = write.send(Message::Text(msg)).await;
                    }
                    Some(WsCommand::Unsubscribe(sub)) => {
                        let msg = format_unsubscribe_message(&sub);
                        let _ = write.send(Message::Text(msg)).await;
                    }
                    Some(WsCommand::SendMessage(msg)) => {
                        let _ = write.send(Message::Text(msg)).await;
                    }
                    Some(WsCommand::Disconnect) | None => {
                        let _ = write.send(Message::Close(None)).await;
                        break;
                    }
                }
            }

            // Send ping
            _ = ping_interval.tick() => {
                let _ = write.send(Message::Ping(vec![])).await;
            }
        }
    }
}

/// Parse WebSocket message into event
fn parse_ws_message(text: &str, venue: &str) -> Result<WsEvent> {
    // Try to parse as generic JSON first
    let value: serde_json::Value = serde_json::from_str(text)?;

    // Detect message type from common patterns
    if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "orderbook" | "book" | "depth" => {
                let update = parse_orderbook_update(&value, venue)?;
                return Ok(WsEvent::OrderbookUpdate(update));
            }
            "trade" | "trades" => {
                let trade = parse_trade(&value, venue)?;
                return Ok(WsEvent::Trade(trade));
            }
            "ticker" => {
                let ticker = parse_ticker(&value, venue)?;
                return Ok(WsEvent::Ticker(ticker));
            }
            "order" | "order_update" => {
                let update = parse_order_update(&value, venue)?;
                return Ok(WsEvent::OrderUpdate(update));
            }
            "fill" | "execution" => {
                let fill = parse_fill(&value, venue)?;
                return Ok(WsEvent::Fill(fill));
            }
            _ => {}
        }
    }

    // Check for channel-based messages
    if let Some(channel) = value.get("channel").and_then(|v| v.as_str()) {
        if channel.contains("orderbook") || channel.contains("book") {
            let update = parse_orderbook_update(&value, venue)?;
            return Ok(WsEvent::OrderbookUpdate(update));
        }
        if channel.contains("trade") {
            let trade = parse_trade(&value, venue)?;
            return Ok(WsEvent::Trade(trade));
        }
        if channel.contains("ticker") {
            let ticker = parse_ticker(&value, venue)?;
            return Ok(WsEvent::Ticker(ticker));
        }
    }

    Err(Error::DeserializationError(format!(
        "Unknown message format: {text}"
    )))
}

fn parse_orderbook_update(value: &serde_json::Value, venue: &str) -> Result<OrderbookUpdate> {
    let symbol = value
        .get("symbol")
        .or_else(|| value.get("s"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let is_snapshot = value
        .get("snapshot")
        .or_else(|| value.get("is_snapshot"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut bids = Vec::new();
    let mut asks = Vec::new();

    if let Some(bid_array) = value
        .get("bids")
        .or_else(|| value.get("b"))
        .and_then(|v| v.as_array())
    {
        for bid in bid_array {
            if let (Some(price), Some(qty)) = (
                bid.get(0)
                    .or_else(|| bid.get("price"))
                    .and_then(parse_decimal),
                bid.get(1)
                    .or_else(|| bid.get("quantity"))
                    .and_then(parse_decimal),
            ) {
                bids.push(PriceLevel::new(price, qty));
            }
        }
    }

    if let Some(ask_array) = value
        .get("asks")
        .or_else(|| value.get("a"))
        .and_then(|v| v.as_array())
    {
        for ask in ask_array {
            if let (Some(price), Some(qty)) = (
                ask.get(0)
                    .or_else(|| ask.get("price"))
                    .and_then(parse_decimal),
                ask.get(1)
                    .or_else(|| ask.get("quantity"))
                    .and_then(parse_decimal),
            ) {
                asks.push(PriceLevel::new(price, qty));
            }
        }
    }

    let timestamp = value
        .get("timestamp")
        .or_else(|| value.get("T"))
        .or_else(|| value.get("t"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    let sequence = value
        .get("sequence")
        .or_else(|| value.get("u"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    Ok(OrderbookUpdate {
        symbol,
        venue: venue.to_string(),
        is_snapshot,
        bids,
        asks,
        timestamp,
        sequence,
    })
}

fn parse_trade(value: &serde_json::Value, venue: &str) -> Result<Trade> {
    use rust_decimal::Decimal;

    let trade_id = value
        .get("trade_id")
        .or_else(|| value.get("t"))
        .or_else(|| value.get("id"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let order_id = value
        .get("order_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let symbol = value
        .get("symbol")
        .or_else(|| value.get("s"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let side = match value
        .get("side")
        .or_else(|| value.get("S"))
        .and_then(|v| v.as_str())
    {
        Some("sell") | Some("SELL") | Some("s") => Side::Sell,
        _ => Side::Buy,
    };

    let price = value
        .get("price")
        .or_else(|| value.get("p"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let quantity = value
        .get("quantity")
        .or_else(|| value.get("q"))
        .or_else(|| value.get("amount"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let timestamp = value
        .get("timestamp")
        .or_else(|| value.get("T"))
        .or_else(|| value.get("t"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    let is_maker = value
        .get("is_maker")
        .or_else(|| value.get("m"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(Trade {
        trade_id,
        order_id,
        symbol,
        venue: venue.to_string(),
        side,
        price,
        quantity,
        fee: Fee {
            asset: String::new(),
            amount: Decimal::ZERO,
            rate: None,
        },
        timestamp,
        is_maker,
    })
}

fn parse_ticker(value: &serde_json::Value, venue: &str) -> Result<Ticker> {
    let symbol = value
        .get("symbol")
        .or_else(|| value.get("s"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let bid = value
        .get("bid")
        .or_else(|| value.get("b"))
        .and_then(parse_decimal);

    let ask = value
        .get("ask")
        .or_else(|| value.get("a"))
        .and_then(parse_decimal);

    let last = value
        .get("last")
        .or_else(|| value.get("c"))
        .and_then(parse_decimal);

    let volume_24h = value
        .get("volume")
        .or_else(|| value.get("v"))
        .and_then(parse_decimal);

    let high_24h = value
        .get("high")
        .or_else(|| value.get("h"))
        .and_then(parse_decimal);

    let low_24h = value
        .get("low")
        .or_else(|| value.get("l"))
        .and_then(parse_decimal);

    let change_24h = value
        .get("change")
        .or_else(|| value.get("P"))
        .and_then(parse_decimal);

    let timestamp = value
        .get("timestamp")
        .or_else(|| value.get("T"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    Ok(Ticker {
        symbol,
        venue: venue.to_string(),
        bid,
        ask,
        last,
        volume_24h,
        high_24h,
        low_24h,
        change_24h,
        timestamp,
    })
}

fn parse_order_update(value: &serde_json::Value, venue: &str) -> Result<OrderUpdate> {
    let order_id = value
        .get("order_id")
        .or_else(|| value.get("i"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let client_order_id = value
        .get("client_order_id")
        .or_else(|| value.get("c"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let symbol = value
        .get("symbol")
        .or_else(|| value.get("s"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let status = match value
        .get("status")
        .or_else(|| value.get("X"))
        .and_then(|v| v.as_str())
    {
        Some("NEW") | Some("new") | Some("open") => OrderStatus::Open,
        Some("PARTIALLY_FILLED") | Some("partially_filled") => OrderStatus::PartiallyFilled,
        Some("FILLED") | Some("filled") => OrderStatus::Filled,
        Some("CANCELED") | Some("CANCELLED") | Some("canceled") | Some("cancelled") => {
            OrderStatus::Cancelled
        }
        Some("EXPIRED") | Some("expired") => OrderStatus::Expired,
        Some("REJECTED") | Some("rejected") => OrderStatus::Rejected,
        _ => OrderStatus::Pending,
    };

    let filled_quantity = value
        .get("filled_quantity")
        .or_else(|| value.get("z"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let remaining_quantity = value
        .get("remaining_quantity")
        .and_then(parse_decimal)
        .unwrap_or_default();

    let average_price = value
        .get("average_price")
        .or_else(|| value.get("ap"))
        .and_then(parse_decimal);

    let timestamp = value
        .get("timestamp")
        .or_else(|| value.get("T"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    Ok(OrderUpdate {
        order_id,
        client_order_id,
        symbol,
        venue: venue.to_string(),
        status,
        filled_quantity,
        remaining_quantity,
        average_price,
        timestamp,
    })
}

fn parse_fill(value: &serde_json::Value, venue: &str) -> Result<Fill> {
    let trade_id = value
        .get("trade_id")
        .or_else(|| value.get("t"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let order_id = value
        .get("order_id")
        .or_else(|| value.get("i"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let symbol = value
        .get("symbol")
        .or_else(|| value.get("s"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let side = match value
        .get("side")
        .or_else(|| value.get("S"))
        .and_then(|v| v.as_str())
    {
        Some("sell") | Some("SELL") => Side::Sell,
        _ => Side::Buy,
    };

    let price = value
        .get("price")
        .or_else(|| value.get("p"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let quantity = value
        .get("quantity")
        .or_else(|| value.get("q"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let fee_amount = value
        .get("fee")
        .or_else(|| value.get("n"))
        .and_then(parse_decimal)
        .unwrap_or_default();

    let fee_asset = value
        .get("fee_asset")
        .or_else(|| value.get("N"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let timestamp = value
        .get("timestamp")
        .or_else(|| value.get("T"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    let is_maker = value
        .get("is_maker")
        .or_else(|| value.get("m"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(Fill {
        trade_id,
        order_id,
        symbol,
        venue: venue.to_string(),
        side,
        price,
        quantity,
        fee: Fee {
            asset: fee_asset,
            amount: fee_amount,
            rate: None,
        },
        timestamp,
        is_maker,
    })
}

fn parse_decimal(value: &serde_json::Value) -> Option<rust_decimal::Decimal> {
    use rust_decimal::prelude::FromStr;

    if let Some(s) = value.as_str() {
        rust_decimal::Decimal::from_str(s).ok()
    } else if let Some(f) = value.as_f64() {
        rust_decimal::Decimal::try_from(f).ok()
    } else {
        None
    }
}

fn channel_to_str(channel: SubscriptionChannel) -> &'static str {
    match channel {
        SubscriptionChannel::Orderbook => "orderbook",
        SubscriptionChannel::Trades => "trades",
        SubscriptionChannel::Ticker => "ticker",
        SubscriptionChannel::Orders => "orders",
        SubscriptionChannel::Fills => "fills",
    }
}

fn format_subscribe_message(sub: &Subscription) -> String {
    serde_json::json!({
        "method": "SUBSCRIBE",
        "params": [format!("{}@{}", sub.symbol.to_lowercase(), channel_to_str(sub.channel))],
        "id": chrono::Utc::now().timestamp_millis()
    })
    .to_string()
}

fn format_unsubscribe_message(sub: &Subscription) -> String {
    serde_json::json!({
        "method": "UNSUBSCRIBE",
        "params": [format!("{}@{}", sub.symbol.to_lowercase(), channel_to_str(sub.channel))],
        "id": chrono::Utc::now().timestamp_millis()
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_config() {
        let config = WsConfig::new("wss://stream.example.com/ws", "test_venue");
        assert_eq!(config.url, "wss://stream.example.com/ws");
        assert_eq!(config.venue, "test_venue");
        assert!(config.auto_reconnect);
    }

    #[test]
    fn test_parse_orderbook_update() {
        let json = serde_json::json!({
            "type": "orderbook",
            "symbol": "BTC-USDC",
            "bids": [["50000", "1.5"], ["49900", "2.0"]],
            "asks": [["50100", "1.0"], ["50200", "3.0"]],
            "timestamp": 1700000000000i64,
            "sequence": 12345u64
        });

        let update = parse_orderbook_update(&json, "test").unwrap();
        assert_eq!(update.symbol, "BTC-USDC");
        assert_eq!(update.bids.len(), 2);
        assert_eq!(update.asks.len(), 2);
    }

    #[test]
    fn test_parse_trade() {
        let json = serde_json::json!({
            "type": "trade",
            "trade_id": "12345",
            "symbol": "BTC-USDC",
            "side": "buy",
            "price": "50000",
            "quantity": "1.5",
            "timestamp": 1700000000000i64
        });

        let trade = parse_trade(&json, "test").unwrap();
        assert_eq!(trade.trade_id, "12345");
        assert_eq!(trade.symbol, "BTC-USDC");
        assert_eq!(trade.side, Side::Buy);
    }

    #[test]
    fn test_subscription_channel() {
        assert_eq!(channel_to_str(SubscriptionChannel::Orderbook), "orderbook");
        assert_eq!(channel_to_str(SubscriptionChannel::Trades), "trades");
    }
}
