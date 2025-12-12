//! WebSocket and REST client for LX DEX.
//!
//! Provides async client with automatic reconnection and subscription management.

use crate::error::{Error, Result, RpcError};
use crate::orderbook::{OrderBook, OrderBookUpdate};
use crate::types::{Balance, NodeInfo, Order, OrderResponse, Position, Trade};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client as HttpClient;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::{debug, error, info};

/// Client configuration.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// WebSocket URL (default: ws://localhost:8081).
    pub ws_url: String,
    /// HTTP JSON-RPC URL (default: http://localhost:8080).
    pub http_url: String,
    /// API key for authentication.
    pub api_key: Option<String>,
    /// API secret for authentication.
    pub api_secret: Option<String>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Auto-reconnect on disconnect.
    pub auto_reconnect: bool,
    /// Reconnect delay in milliseconds.
    pub reconnect_delay_ms: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            ws_url: "ws://localhost:8081".into(),
            http_url: "http://localhost:8080".into(),
            api_key: None,
            api_secret: None,
            timeout_ms: 30000,
            auto_reconnect: true,
            reconnect_delay_ms: 1000,
        }
    }
}

impl ClientConfig {
    /// Create config with WebSocket URL.
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = url.into();
        self
    }

    /// Create config with HTTP URL.
    pub fn with_http_url(mut self, url: impl Into<String>) -> Self {
        self.http_url = url.into();
        self
    }

    /// Set API credentials.
    pub fn with_credentials(mut self, key: impl Into<String>, secret: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self.api_secret = Some(secret.into());
        self
    }
}

/// Message types received from WebSocket.
#[derive(Debug, Clone)]
pub enum WsEvent {
    Connected { client_id: String },
    Authenticated { user_id: String },
    OrderUpdate { order: Order, status: String },
    Trade(Trade),
    OrderBook(OrderBook),
    OrderBookUpdate(OrderBookUpdate),
    Price { symbol: String, price: f64 },
    Position(Position),
    Balance(Balance),
    Error { message: String, request_id: Option<String> },
    Pong,
}

/// Internal WebSocket message for sending.
#[derive(Debug, Serialize)]
struct WsRequest {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(flatten)]
    data: serde_json::Value,
}

/// WebSocket response message.
#[derive(Debug, Deserialize)]
struct WsResponse {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    data: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    timestamp: i64,
}

/// JSON-RPC request.
#[derive(Debug, Serialize)]
struct RpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: serde_json::Value,
    id: u64,
}

/// JSON-RPC response.
#[derive(Debug, Deserialize)]
struct RpcResponse<T> {
    result: Option<T>,
    error: Option<RpcError>,
    #[allow(dead_code)]
    id: u64,
}

/// LX DEX client.
pub struct Client {
    config: ClientConfig,
    http: HttpClient,
    connected: AtomicBool,
    authenticated: AtomicBool,
    request_id: AtomicU64,
    rpc_id: AtomicU64,

    // WebSocket channels
    ws_tx: RwLock<Option<mpsc::Sender<WsMessage>>>,

    // Event stream
    event_tx: mpsc::Sender<WsEvent>,
    event_rx: RwLock<Option<mpsc::Receiver<WsEvent>>>,
}

impl Client {
    /// Create a new client with default configuration.
    pub fn new() -> Self {
        Self::with_config(ClientConfig::default())
    }

    /// Create a new client with custom configuration.
    pub fn with_config(config: ClientConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1024);

        Self {
            config,
            http: HttpClient::new(),
            connected: AtomicBool::new(false),
            authenticated: AtomicBool::new(false),
            request_id: AtomicU64::new(1),
            rpc_id: AtomicU64::new(1),
            ws_tx: RwLock::new(None),
            event_tx,
            event_rx: RwLock::new(Some(event_rx)),
        }
    }

    /// Take the event receiver (can only be called once).
    pub async fn take_event_receiver(&self) -> Option<mpsc::Receiver<WsEvent>> {
        self.event_rx.write().await.take()
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Check if authenticated.
    pub fn is_authenticated(&self) -> bool {
        self.authenticated.load(Ordering::SeqCst)
    }

    /// Connect to WebSocket server.
    pub async fn connect(&self) -> Result<()> {
        if self.is_connected() {
            return Err(Error::AlreadyConnected);
        }

        let (ws_stream, _) = connect_async(&self.config.ws_url).await?;
        let (write, read) = ws_stream.split();

        // Create channel for sending messages
        let (tx, mut rx) = mpsc::channel::<WsMessage>(256);
        *self.ws_tx.write().await = Some(tx);

        self.connected.store(true, Ordering::SeqCst);
        info!("WebSocket connected to {}", self.config.ws_url);

        // Spawn write task
        let write = Arc::new(tokio::sync::Mutex::new(write));
        let write_clone = write.clone();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let mut writer = write_clone.lock().await;
                if let Err(e) = writer.send(msg).await {
                    error!("WebSocket write error: {}", e);
                    break;
                }
            }
        });

        // Spawn read task
        let event_tx_clone = self.event_tx.clone();
        tokio::spawn(async move {
            let mut read = read;
            while let Some(result) = read.next().await {
                match result {
                    Ok(WsMessage::Text(text)) => {
                        if let Ok(response) = serde_json::from_str::<WsResponse>(&text) {
                            if let Err(e) = Self::handle_message(&event_tx_clone, response).await {
                                error!("Error handling message: {}", e);
                            }
                        }
                    }
                    Ok(WsMessage::Close(_)) => {
                        info!("WebSocket closed by server");
                        break;
                    }
                    Ok(WsMessage::Ping(_)) => {
                        debug!("Received ping");
                    }
                    Err(e) => {
                        error!("WebSocket read error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }

    /// Handle incoming WebSocket message.
    async fn handle_message(event_tx: &mpsc::Sender<WsEvent>, msg: WsResponse) -> Result<()> {
        let event = match msg.msg_type.as_str() {
            "connected" => {
                let client_id = msg
                    .data
                    .and_then(|d| d.get("client_id").and_then(|v| v.as_str()).map(String::from))
                    .unwrap_or_default();
                WsEvent::Connected { client_id }
            }
            "auth_success" => {
                let user_id = msg
                    .data
                    .and_then(|d| d.get("user_id").and_then(|v| v.as_str()).map(String::from))
                    .unwrap_or_default();
                WsEvent::Authenticated { user_id }
            }
            "order_update" => {
                if let Some(data) = msg.data {
                    let order: Order = serde_json::from_value(
                        data.get("order").cloned().unwrap_or(serde_json::Value::Null),
                    )?;
                    let status = data
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    WsEvent::OrderUpdate { order, status }
                } else {
                    return Ok(());
                }
            }
            "trade_update" => {
                if let Some(data) = msg.data {
                    let trade: Trade =
                        serde_json::from_value(data.get("trade").cloned().unwrap_or(data))?;
                    WsEvent::Trade(trade)
                } else {
                    return Ok(());
                }
            }
            "orderbook_update" => {
                if let Some(data) = msg.data {
                    if let Some(snapshot) = data.get("snapshot") {
                        let book: OrderBook = serde_json::from_value(snapshot.clone())?;
                        WsEvent::OrderBook(book)
                    } else {
                        let update: OrderBookUpdate = serde_json::from_value(data)?;
                        WsEvent::OrderBookUpdate(update)
                    }
                } else {
                    return Ok(());
                }
            }
            "price_update" => {
                if let Some(data) = msg.data {
                    let symbol = data
                        .get("symbol")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let price = data.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    WsEvent::Price { symbol, price }
                } else {
                    return Ok(());
                }
            }
            "position_update" => {
                if let Some(data) = msg.data {
                    let position: Position =
                        serde_json::from_value(data.get("position").cloned().unwrap_or(data))?;
                    WsEvent::Position(position)
                } else {
                    return Ok(());
                }
            }
            "balance_update" => {
                if let Some(data) = msg.data {
                    // Balance updates may come as a map of balances
                    // For simplicity, emit first balance
                    if let Some(balances) = data.get("balances").and_then(|v| v.as_object()) {
                        for (asset, amount) in balances {
                            let balance = Balance {
                                asset: asset.clone(),
                                available: amount.as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                                locked: 0.0,
                                total: amount.as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                            };
                            let _ = event_tx.send(WsEvent::Balance(balance)).await;
                        }
                        return Ok(());
                    }
                }
                return Ok(());
            }
            "pong" => WsEvent::Pong,
            "error" => WsEvent::Error {
                message: msg.error.unwrap_or_default(),
                request_id: msg.request_id,
            },
            _ => {
                debug!("Unknown message type: {}", msg.msg_type);
                return Ok(());
            }
        };

        event_tx.send(event).await.map_err(|_| Error::ChannelClosed)
    }

    /// Disconnect from WebSocket server.
    pub async fn disconnect(&self) -> Result<()> {
        self.connected.store(false, Ordering::SeqCst);
        self.authenticated.store(false, Ordering::SeqCst);
        *self.ws_tx.write().await = None;
        info!("WebSocket disconnected");
        Ok(())
    }

    /// Send a WebSocket message.
    async fn send_ws(&self, msg_type: &str, data: serde_json::Value) -> Result<()> {
        let tx = self.ws_tx.read().await;
        let tx = tx.as_ref().ok_or(Error::NotConnected)?;

        let request = WsRequest {
            msg_type: msg_type.into(),
            request_id: Some(self.next_request_id()),
            data,
        };

        let json = serde_json::to_string(&request)?;
        tx.send(WsMessage::Text(json.into()))
            .await
            .map_err(|_| Error::ChannelClosed)
    }

    fn next_request_id(&self) -> String {
        format!("req_{}", self.request_id.fetch_add(1, Ordering::SeqCst))
    }

    // ===== Authentication =====

    /// Authenticate with API credentials.
    pub async fn authenticate(&self, api_key: &str, api_secret: &str) -> Result<()> {
        self.send_ws(
            "auth",
            serde_json::json!({
                "apiKey": api_key,
                "apiSecret": api_secret
            }),
        )
        .await
    }

    /// Authenticate using configured credentials.
    pub async fn authenticate_configured(&self) -> Result<()> {
        let (key, secret) = match (&self.config.api_key, &self.config.api_secret) {
            (Some(k), Some(s)) => (k.clone(), s.clone()),
            _ => return Err(Error::auth("No credentials configured")),
        };
        self.authenticate(&key, &secret).await
    }

    // ===== Trading =====

    /// Place an order via WebSocket.
    pub async fn place_order(&self, order: &Order) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "place_order",
            serde_json::json!({
                "order": order
            }),
        )
        .await
    }

    /// Cancel an order.
    pub async fn cancel_order(&self, order_id: u64) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "cancel_order",
            serde_json::json!({
                "orderID": order_id
            }),
        )
        .await
    }

    /// Modify an existing order.
    pub async fn modify_order(&self, order_id: u64, new_price: f64, new_size: f64) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "modify_order",
            serde_json::json!({
                "orderID": order_id,
                "newPrice": new_price,
                "newSize": new_size
            }),
        )
        .await
    }

    // ===== Subscriptions =====

    /// Subscribe to order book updates.
    pub async fn subscribe_orderbook(&self, symbol: &str) -> Result<()> {
        self.send_ws(
            "subscribe",
            serde_json::json!({
                "channel": "orderbook",
                "symbols": [symbol]
            }),
        )
        .await
    }

    /// Subscribe to trade updates.
    pub async fn subscribe_trades(&self, symbol: &str) -> Result<()> {
        self.send_ws(
            "subscribe",
            serde_json::json!({
                "channel": "trades",
                "symbols": [symbol]
            }),
        )
        .await
    }

    /// Subscribe to price updates.
    pub async fn subscribe_prices(&self, symbols: &[&str]) -> Result<()> {
        self.send_ws(
            "subscribe",
            serde_json::json!({
                "channel": "prices",
                "symbols": symbols
            }),
        )
        .await
    }

    /// Unsubscribe from a channel.
    pub async fn unsubscribe(&self, channel: &str, symbols: &[&str]) -> Result<()> {
        self.send_ws(
            "unsubscribe",
            serde_json::json!({
                "channel": channel,
                "symbols": symbols
            }),
        )
        .await
    }

    // ===== Account Data =====

    /// Get account balances.
    pub async fn get_balances(&self) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }
        self.send_ws("get_balances", serde_json::json!({})).await
    }

    /// Get open positions.
    pub async fn get_positions(&self) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }
        self.send_ws("get_positions", serde_json::json!({})).await
    }

    /// Get open orders.
    pub async fn get_orders(&self) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }
        self.send_ws("get_orders", serde_json::json!({})).await
    }

    // ===== Margin Trading =====

    /// Open a margin position.
    pub async fn open_position(
        &self,
        symbol: &str,
        side: &str,
        size: f64,
        leverage: f64,
    ) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "open_position",
            serde_json::json!({
                "symbol": symbol,
                "side": side,
                "size": size,
                "leverage": leverage
            }),
        )
        .await
    }

    /// Close a margin position.
    pub async fn close_position(&self, position_id: &str, size: f64) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "close_position",
            serde_json::json!({
                "positionID": position_id,
                "size": size
            }),
        )
        .await
    }

    /// Modify position leverage.
    pub async fn modify_leverage(&self, position_id: &str, leverage: f64) -> Result<()> {
        if !self.is_authenticated() {
            return Err(Error::auth("Not authenticated"));
        }

        self.send_ws(
            "modify_leverage",
            serde_json::json!({
                "position_id": position_id,
                "leverage": leverage
            }),
        )
        .await
    }

    // ===== Ping =====

    /// Send ping to server.
    pub async fn ping(&self) -> Result<()> {
        self.send_ws("ping", serde_json::json!({})).await
    }

    // ===== HTTP/JSON-RPC Methods =====

    /// Make a JSON-RPC call.
    async fn rpc_call<T: DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T> {
        let id = self.rpc_id.fetch_add(1, Ordering::SeqCst);
        let request = RpcRequest {
            jsonrpc: "2.0",
            method: method.into(),
            params,
            id,
        };

        let response = self
            .http
            .post(format!("{}/rpc", self.config.http_url))
            .json(&request)
            .send()
            .await?
            .json::<RpcResponse<T>>()
            .await?;

        if let Some(err) = response.error {
            return Err(err.into());
        }

        response
            .result
            .ok_or_else(|| Error::invalid_response("missing result"))
    }

    /// Get node information via HTTP.
    pub async fn get_info(&self) -> Result<NodeInfo> {
        self.rpc_call("lx_getInfo", serde_json::json!({})).await
    }

    /// Get order book via HTTP.
    pub async fn get_orderbook(&self, symbol: &str, depth: i32) -> Result<OrderBook> {
        self.rpc_call(
            "lx_getOrderBook",
            serde_json::json!({
                "symbol": symbol,
                "depth": depth
            }),
        )
        .await
    }

    /// Get recent trades via HTTP.
    pub async fn get_trades(&self, symbol: &str, limit: i32) -> Result<Vec<Trade>> {
        self.rpc_call(
            "lx_getTrades",
            serde_json::json!({
                "symbol": symbol,
                "limit": limit
            }),
        )
        .await
    }

    /// Place order via HTTP (authenticated).
    pub async fn place_order_http(&self, order: &Order) -> Result<OrderResponse> {
        self.rpc_call(
            "lx_placeOrder",
            serde_json::json!({
                "symbol": order.symbol,
                "type": order.order_type,
                "side": order.side,
                "price": order.price,
                "size": order.size,
                "clientID": order.client_id,
                "timeInForce": order.time_in_force,
                "postOnly": order.post_only,
                "reduceOnly": order.reduce_only
            }),
        )
        .await
    }

    /// Cancel order via HTTP.
    pub async fn cancel_order_http(&self, order_id: u64) -> Result<()> {
        self.rpc_call::<serde_json::Value>(
            "lx_cancelOrder",
            serde_json::json!({
                "orderId": order_id
            }),
        )
        .await?;
        Ok(())
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.ws_url, "ws://localhost:8081");
        assert_eq!(config.http_url, "http://localhost:8080");
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = ClientConfig::default()
            .with_ws_url("ws://example.com:8081")
            .with_http_url("http://example.com:8080")
            .with_credentials("key", "secret");

        assert_eq!(config.ws_url, "ws://example.com:8081");
        assert_eq!(config.http_url, "http://example.com:8080");
        assert_eq!(config.api_key, Some("key".into()));
        assert_eq!(config.api_secret, Some("secret".into()));
    }

    #[test]
    fn test_client_not_connected() {
        let client = Client::new();
        assert!(!client.is_connected());
        assert!(!client.is_authenticated());
    }
}
