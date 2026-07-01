//! Native LX DEX and LX AMM adapters.

use async_trait::async_trait;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::adapters::adapter::*;
use crate::config::NativeVenueConfig;
use crate::error::{Error, Result};
use crate::orderbook::Orderbook;
use crate::types::*;

/// LX DEX adapter for OrderBook trading
pub struct LxDexAdapter {
    name: String,
    config: NativeVenueConfig,
    capabilities: VenueCapabilities,
    connected: AtomicBool,
    latency: AtomicU64,
    client: reqwest::Client,
    balances: RwLock<HashMap<String, Balance>>,
    #[allow(dead_code)]
    open_orders: RwLock<HashMap<String, Order>>,
}

impl LxDexAdapter {
    pub async fn new(name: &str, config: NativeVenueConfig) -> Result<Self> {
        let mut capabilities = VenueCapabilities::orderBook();
        capabilities.streaming = config.ws_url.is_some();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            name: name.to_string(),
            config,
            capabilities,
            connected: AtomicBool::new(false),
            latency: AtomicU64::new(0),
            client,
            balances: RwLock::new(HashMap::new()),
            open_orders: RwLock::new(HashMap::new()),
        })
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}{}", self.config.api_url.trim_end_matches('/'), path)
    }

    async fn signed_request<T: serde::de::DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<T> {
        let url = self.api_url(path);
        let timestamp = chrono::Utc::now().timestamp_millis();

        let mut request = self.client.request(method, &url);

        // Add auth headers if configured
        if let (Some(api_key), Some(_api_secret)) =
            (&self.config.api_key, &self.config.api_secret)
        {
            request = request
                .header("X-API-KEY", api_key.as_str())
                .header("X-TIMESTAMP", timestamp.to_string());
            // TODO: Add signature
        }

        if let Some(body) = body {
            request = request.json(&body);
        }

        let start = std::time::Instant::now();
        let response = request.send().await?;
        let latency = start.elapsed().as_millis() as u64;
        self.latency.store(latency, Ordering::Relaxed);

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError {
                venue: self.name.clone(),
                code: status.as_str().to_string(),
                message: text,
            });
        }

        response.json().await.map_err(Error::from)
    }
}

#[async_trait]
impl VenueAdapter for LxDexAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn venue_type(&self) -> VenueType {
        VenueType::Native
    }

    fn capabilities(&self) -> &VenueCapabilities {
        &self.capabilities
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn latency_ms(&self) -> Option<u64> {
        let lat = self.latency.load(Ordering::Relaxed);
        if lat > 0 {
            Some(lat)
        } else {
            None
        }
    }

    async fn connect(&mut self) -> Result<()> {
        // Test connection with a simple request
        let _: serde_json::Value = self.signed_request(
            reqwest::Method::GET,
            "/api/v1/health",
            None,
        ).await.map_err(|e| Error::ConnectionFailed {
            venue: self.name.clone(),
            message: e.to_string(),
        })?;

        self.connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn start_streaming(&mut self, _symbols: &[String]) -> Result<()> {
        // TODO: Implement WebSocket streaming
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        // TODO: Implement WebSocket streaming
        Ok(())
    }

    async fn get_markets(&self) -> Result<Vec<MarketInfo>> {
        self.signed_request(reqwest::Method::GET, "/api/v1/markets", None).await
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let path = format!("/api/v1/ticker/{symbol}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_tickers(&self, symbols: &[String]) -> Result<Vec<Ticker>> {
        let query = symbols.join(",");
        let path = format!("/api/v1/tickers?symbols={query}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_orderbook(&self, symbol: &str, depth: Option<usize>) -> Result<Orderbook> {
        let depth = depth.unwrap_or(100);
        let path = format!("/api/v1/orderbook/{symbol}?depth={depth}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>> {
        let limit = limit.unwrap_or(100);
        let path = format!("/api/v1/trades/{symbol}?limit={limit}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_balances(&self) -> Result<Vec<Balance>> {
        let balances: Vec<Balance> = self
            .signed_request(reqwest::Method::GET, "/api/v1/account/balances", None)
            .await?;

        // Cache balances
        let mut cache = self.balances.write();
        for balance in &balances {
            cache.insert(balance.asset.clone(), balance.clone());
        }

        Ok(balances)
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        // Try cache first
        if let Some(balance) = self.balances.read().get(asset) {
            return Ok(balance.clone());
        }

        let path = format!("/api/v1/account/balance/{asset}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let path = match symbol {
            Some(s) => format!("/api/v1/orders?symbol={s}&status=open"),
            None => "/api/v1/orders?status=open".to_string(),
        };
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_order(&self, order_id: &str, symbol: &str) -> Result<Order> {
        let path = format!("/api/v1/orders/{order_id}?symbol={symbol}");
        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn get_order_history(
        &self,
        symbol: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Order>> {
        let mut path = "/api/v1/orders/history".to_string();
        let mut params = vec![];

        if let Some(s) = symbol {
            params.push(format!("symbol={s}"));
        }
        if let Some(l) = limit {
            params.push(format!("limit={l}"));
        }

        if !params.is_empty() {
            path = format!("{}?{}", path, params.join("&"));
        }

        self.signed_request(reqwest::Method::GET, &path, None).await
    }

    async fn place_order(&self, request: OrderRequest) -> Result<Order> {
        let body = serde_json::to_value(&request)?;
        self.signed_request(reqwest::Method::POST, "/api/v1/orders", Some(body)).await
    }

    async fn place_orders(&self, requests: Vec<OrderRequest>) -> Result<Vec<Order>> {
        let body = serde_json::json!({ "orders": requests });
        self.signed_request(reqwest::Method::POST, "/api/v1/orders/batch", Some(body)).await
    }

    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<Order> {
        let path = format!("/api/v1/orders/{order_id}");
        let body = serde_json::json!({ "symbol": symbol });
        self.signed_request(reqwest::Method::DELETE, &path, Some(body)).await
    }

    async fn cancel_orders(&self, order_ids: &[(String, String)]) -> Result<Vec<Order>> {
        let body = serde_json::json!({
            "orders": order_ids.iter().map(|(id, sym)| {
                serde_json::json!({ "order_id": id, "symbol": sym })
            }).collect::<Vec<_>>()
        });
        self.signed_request(reqwest::Method::DELETE, "/api/v1/orders/batch", Some(body)).await
    }

    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let body = match symbol {
            Some(s) => serde_json::json!({ "symbol": s }),
            None => serde_json::json!({}),
        };
        self.signed_request(reqwest::Method::DELETE, "/api/v1/orders/all", Some(body)).await
    }
}

// =============================================================================
// LX AMM Adapter
// =============================================================================

/// LX AMM adapter for liquidity pool operations
pub struct LxAmmAdapter {
    name: String,
    config: NativeVenueConfig,
    capabilities: VenueCapabilities,
    connected: AtomicBool,
    latency: AtomicU64,
    client: reqwest::Client,
}

impl LxAmmAdapter {
    pub async fn new(name: &str, config: NativeVenueConfig) -> Result<Self> {
        let capabilities = VenueCapabilities::amm();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            name: name.to_string(),
            config,
            capabilities,
            connected: AtomicBool::new(false),
            latency: AtomicU64::new(0),
            client,
        })
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}{}", self.config.api_url.trim_end_matches('/'), path)
    }

    async fn request<T: serde::de::DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<T> {
        let url = self.api_url(path);
        let mut request = self.client.request(method, &url);

        if let Some(body) = body {
            request = request.json(&body);
        }

        let start = std::time::Instant::now();
        let response = request.send().await?;
        let latency = start.elapsed().as_millis() as u64;
        self.latency.store(latency, Ordering::Relaxed);

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError {
                venue: self.name.clone(),
                code: status.as_str().to_string(),
                message: text,
            });
        }

        response.json().await.map_err(Error::from)
    }
}

#[async_trait]
impl VenueAdapter for LxAmmAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn venue_type(&self) -> VenueType {
        VenueType::Native
    }

    fn capabilities(&self) -> &VenueCapabilities {
        &self.capabilities
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn latency_ms(&self) -> Option<u64> {
        let lat = self.latency.load(Ordering::Relaxed);
        if lat > 0 {
            Some(lat)
        } else {
            None
        }
    }

    async fn connect(&mut self) -> Result<()> {
        self.connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn start_streaming(&mut self, _symbols: &[String]) -> Result<()> {
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        Ok(())
    }

    async fn get_markets(&self) -> Result<Vec<MarketInfo>> {
        self.request(reqwest::Method::GET, "/api/v1/amm/pools", None).await
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let path = format!("/api/v1/amm/price/{symbol}");
        self.request(reqwest::Method::GET, &path, None).await
    }

    async fn get_tickers(&self, symbols: &[String]) -> Result<Vec<Ticker>> {
        let mut tickers = Vec::new();
        for symbol in symbols {
            if let Ok(ticker) = self.get_ticker(symbol).await {
                tickers.push(ticker);
            }
        }
        Ok(tickers)
    }

    async fn get_orderbook(&self, _symbol: &str, _depth: Option<usize>) -> Result<Orderbook> {
        Err(Error::NotImplemented("AMM does not have orderbook".into()))
    }

    async fn get_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>> {
        let limit = limit.unwrap_or(100);
        let path = format!("/api/v1/amm/swaps/{symbol}?limit={limit}");
        self.request(reqwest::Method::GET, &path, None).await
    }

    async fn get_balances(&self) -> Result<Vec<Balance>> {
        self.request(reqwest::Method::GET, "/api/v1/account/balances", None).await
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        let path = format!("/api/v1/account/balance/{asset}");
        self.request(reqwest::Method::GET, &path, None).await
    }

    async fn get_open_orders(&self, _symbol: Option<&str>) -> Result<Vec<Order>> {
        // AMM doesn't have traditional orders
        Ok(vec![])
    }

    async fn get_order(&self, _order_id: &str, _symbol: &str) -> Result<Order> {
        Err(Error::NotImplemented("AMM does not have orders".into()))
    }

    async fn get_order_history(
        &self,
        _symbol: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<Order>> {
        Ok(vec![])
    }

    async fn place_order(&self, request: OrderRequest) -> Result<Order> {
        // Convert order to swap
        let is_buy = matches!(request.side, Side::Buy);
        let pair = TradingPair::from_symbol(&request.symbol)
            .ok_or_else(|| Error::InvalidOrder("Invalid symbol format".into()))?;

        let trade = self.execute_swap(
            &pair.base,
            &pair.quote,
            request.quantity,
            is_buy,
            Decimal::from(1), // 1% default slippage
        ).await?;

        // Convert trade to order format
        Ok(Order {
            order_id: trade.trade_id.clone(),
            client_order_id: request.client_order_id,
            symbol: request.symbol,
            venue: self.name.clone(),
            side: request.side,
            order_type: OrderType::Market,
            status: OrderStatus::Filled,
            quantity: request.quantity,
            filled_quantity: trade.quantity,
            remaining_quantity: Decimal::ZERO,
            price: Some(trade.price),
            average_price: Some(trade.price),
            created_at: trade.timestamp,
            updated_at: trade.timestamp,
            fees: vec![trade.fee],
        })
    }

    async fn place_orders(&self, requests: Vec<OrderRequest>) -> Result<Vec<Order>> {
        let mut orders = Vec::new();
        for request in requests {
            orders.push(self.place_order(request).await?);
        }
        Ok(orders)
    }

    async fn cancel_order(&self, _order_id: &str, _symbol: &str) -> Result<Order> {
        Err(Error::NotImplemented("AMM swaps cannot be cancelled".into()))
    }

    async fn cancel_orders(&self, _order_ids: &[(String, String)]) -> Result<Vec<Order>> {
        Err(Error::NotImplemented("AMM swaps cannot be cancelled".into()))
    }

    async fn cancel_all_orders(&self, _symbol: Option<&str>) -> Result<Vec<Order>> {
        Ok(vec![])
    }

    // AMM-specific methods
    async fn get_swap_quote(
        &self,
        base_token: &str,
        quote_token: &str,
        amount: Decimal,
        is_buy: bool,
    ) -> Result<SwapQuote> {
        let body = serde_json::json!({
            "base_token": base_token,
            "quote_token": quote_token,
            "amount": amount.to_string(),
            "side": if is_buy { "buy" } else { "sell" }
        });
        self.request(reqwest::Method::POST, "/api/v1/amm/quote", Some(body)).await
    }

    async fn execute_swap(
        &self,
        base_token: &str,
        quote_token: &str,
        amount: Decimal,
        is_buy: bool,
        slippage_percent: Decimal,
    ) -> Result<Trade> {
        let body = serde_json::json!({
            "base_token": base_token,
            "quote_token": quote_token,
            "amount": amount.to_string(),
            "side": if is_buy { "buy" } else { "sell" },
            "slippage": slippage_percent.to_string()
        });
        self.request(reqwest::Method::POST, "/api/v1/amm/swap", Some(body)).await
    }

    async fn get_pool_info(&self, base_token: &str, quote_token: &str) -> Result<PoolInfo> {
        let path = format!("/api/v1/amm/pool/{base_token}/{quote_token}");
        self.request(reqwest::Method::GET, &path, None).await
    }

    async fn add_liquidity(
        &self,
        base_token: &str,
        quote_token: &str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        let body = serde_json::json!({
            "base_token": base_token,
            "quote_token": quote_token,
            "base_amount": base_amount.to_string(),
            "quote_amount": quote_amount.to_string(),
            "slippage": slippage_percent.to_string()
        });
        self.request(reqwest::Method::POST, "/api/v1/amm/liquidity/add", Some(body)).await
    }

    async fn remove_liquidity(
        &self,
        pool_address: &str,
        liquidity_amount: Decimal,
        slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        let body = serde_json::json!({
            "pool_address": pool_address,
            "liquidity": liquidity_amount.to_string(),
            "slippage": slippage_percent.to_string()
        });
        self.request(reqwest::Method::POST, "/api/v1/amm/liquidity/remove", Some(body)).await
    }

    async fn get_lp_positions(&self) -> Result<Vec<LpPosition>> {
        self.request(reqwest::Method::GET, "/api/v1/amm/positions", None).await
    }
}
