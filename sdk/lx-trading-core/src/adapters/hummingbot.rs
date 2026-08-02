//! Hummingbot Gateway adapter - connects to Hummingbot Gateway for DEX access.
//!
//! This adapter communicates with Hummingbot Gateway's REST API to access
//! DEX connectors like Uniswap, Jupiter, and our own LX DEX Gateway connector.

use async_trait::async_trait;
use rust_decimal::Decimal;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::adapters::adapter::*;
use crate::config::HummingbotConfig;
use crate::error::{Error, Result};
use crate::orderbook::Orderbook;
use crate::types::*;

/// Hummingbot Gateway adapter
pub struct HummingbotAdapter {
    name: String,
    config: HummingbotConfig,
    capabilities: VenueCapabilities,
    connected: AtomicBool,
    latency: AtomicU64,
    client: reqwest::Client,
}

impl HummingbotAdapter {
    pub async fn new(name: &str, config: HummingbotConfig) -> Result<Self> {
        // Gateway connectors are typically AMM-style
        let capabilities = VenueCapabilities::amm();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60)) // Gateway can be slow
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

    fn base_url(&self) -> String {
        self.config.base_url()
    }

    async fn gateway_request<T: serde::de::DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<T> {
        let url = format!("{}{}", self.base_url(), path);

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
            return Err(Error::HummingbotError(format!(
                "Gateway error [{status}]: {text}"
            )));
        }

        response.json().await.map_err(|e| {
            Error::DeserializationError(format!("Failed to parse Gateway response: {e}"))
        })
    }

    /// Build standard request body with chain/network/connector
    fn build_body(&self, mut body: serde_json::Value) -> serde_json::Value {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("chain".to_string(), serde_json::json!(self.config.chain));
            obj.insert(
                "network".to_string(),
                serde_json::json!(self.config.network),
            );
            obj.insert(
                "connector".to_string(),
                serde_json::json!(self.config.connector),
            );
            if let Some(wallet) = &self.config.wallet_address {
                obj.insert("address".to_string(), serde_json::json!(wallet));
            }
        }
        body
    }
}

#[async_trait]
impl VenueAdapter for HummingbotAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn venue_type(&self) -> VenueType {
        VenueType::Hummingbot
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
        // Check gateway status
        let status: serde_json::Value = self
            .gateway_request(reqwest::Method::GET, "/", None)
            .await
            .map_err(|e| Error::ConnectionFailed {
                venue: self.name.clone(),
                message: e.to_string(),
            })?;

        if status["status"].as_str() != Some("ok") {
            return Err(Error::ConnectionFailed {
                venue: self.name.clone(),
                message: "Gateway not ready".to_string(),
            });
        }

        self.connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn start_streaming(&mut self, _symbols: &[String]) -> Result<()> {
        // Gateway uses polling, not streaming
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        Ok(())
    }

    async fn get_markets(&self) -> Result<Vec<MarketInfo>> {
        let body = self.build_body(serde_json::json!({}));
        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/tokens", Some(body))
            .await?;

        // Convert tokens to markets (pairs are derived)
        let mut markets = Vec::new();
        if let Some(tokens) = response["tokens"].as_array() {
            // Create pairs from tokens
            for i in 0..tokens.len() {
                for j in (i + 1)..tokens.len() {
                    let base = tokens[i]["symbol"].as_str().unwrap_or("");
                    let quote = tokens[j]["symbol"].as_str().unwrap_or("");

                    if base.is_empty() || quote.is_empty() {
                        continue;
                    }

                    markets.push(MarketInfo {
                        symbol: format!("{base}-{quote}"),
                        base: base.to_string(),
                        quote: quote.to_string(),
                        price_precision: 8,
                        quantity_precision: 8,
                        min_quantity: Decimal::ZERO,
                        max_quantity: None,
                        min_notional: None,
                        tick_size: decimal_from_str("0.00000001")?,
                        lot_size: decimal_from_str("0.00000001")?,
                    });
                }
            }
        }

        Ok(markets)
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let pair = TradingPair::from_symbol(symbol)
            .ok_or_else(|| Error::InvalidOrder("Invalid symbol format".into()))?;

        // Get price quote for a small amount to determine current price
        let body = self.build_body(serde_json::json!({
            "base": pair.base,
            "quote": pair.quote,
            "amount": "1",
            "side": "BUY"
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/price", Some(body))
            .await?;

        let price = response["price"]
            .as_str()
            .and_then(|s| s.parse::<Decimal>().ok());

        Ok(Ticker {
            symbol: symbol.to_string(),
            venue: self.name.clone(),
            bid: price,
            ask: price,
            last: price,
            volume_24h: None,
            high_24h: None,
            low_24h: None,
            change_24h: None,
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
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
        // AMM doesn't have traditional orderbook
        Err(Error::NotImplemented(
            "Gateway AMM does not have orderbook".into(),
        ))
    }

    async fn get_trades(&self, symbol: &str, _limit: Option<usize>) -> Result<Vec<Trade>> {
        // Gateway doesn't provide historical trades directly
        // Would need to query on-chain events
        let _ = symbol;
        Ok(vec![])
    }

    async fn get_balances(&self) -> Result<Vec<Balance>> {
        let body = self.build_body(serde_json::json!({}));
        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/chain/balances", Some(body))
            .await?;

        let mut balances = Vec::new();
        if let Some(bal) = response["balances"].as_object() {
            for (asset, amount) in bal {
                let value = amount
                    .as_str()
                    .and_then(|s| s.parse::<Decimal>().ok())
                    .unwrap_or_default();
                balances.push(Balance::new(
                    asset.clone(),
                    &self.name,
                    value,
                    Decimal::ZERO,
                ));
            }
        }

        Ok(balances)
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        let balances = self.get_balances().await?;
        balances
            .into_iter()
            .find(|b| b.asset.eq_ignore_ascii_case(asset))
            .ok_or_else(|| Error::MarketNotFound(asset.to_string()))
    }

    async fn get_open_orders(&self, _symbol: Option<&str>) -> Result<Vec<Order>> {
        // AMM doesn't have open orders
        Ok(vec![])
    }

    async fn get_order(&self, _order_id: &str, _symbol: &str) -> Result<Order> {
        Err(Error::NotImplemented(
            "Gateway AMM does not have orders".into(),
        ))
    }

    async fn get_order_history(
        &self,
        _symbol: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<Order>> {
        Ok(vec![])
    }

    async fn place_order(&self, request: OrderRequest) -> Result<Order> {
        // Convert order to swap via Gateway
        let pair = TradingPair::from_symbol(&request.symbol)
            .ok_or_else(|| Error::InvalidOrder("Invalid symbol format".into()))?;

        let is_buy = matches!(request.side, Side::Buy);
        let trade = self
            .execute_swap(
                &pair.base,
                &pair.quote,
                request.quantity,
                is_buy,
                Decimal::from(1), // 1% default slippage
            )
            .await?;

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
        Err(Error::NotImplemented(
            "Gateway AMM swaps cannot be cancelled".into(),
        ))
    }

    async fn cancel_orders(&self, _order_ids: &[(String, String)]) -> Result<Vec<Order>> {
        Err(Error::NotImplemented(
            "Gateway AMM swaps cannot be cancelled".into(),
        ))
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
        let body = self.build_body(serde_json::json!({
            "base": base_token,
            "quote": quote_token,
            "amount": amount.to_string(),
            "side": if is_buy { "BUY" } else { "SELL" }
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/price", Some(body))
            .await?;

        let price = response["price"]
            .as_str()
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or_default();

        let expected_amount = response["expectedAmount"]
            .as_str()
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or_default();

        Ok(SwapQuote {
            base_token: base_token.to_string(),
            quote_token: quote_token.to_string(),
            input_amount: amount,
            output_amount: expected_amount,
            price,
            price_impact: Decimal::ZERO, // Gateway may not provide this
            fee: Decimal::ZERO,
            route: vec![],
            expires_at: chrono::Utc::now().timestamp_millis() + 60000, // 1 minute
        })
    }

    async fn execute_swap(
        &self,
        base_token: &str,
        quote_token: &str,
        amount: Decimal,
        is_buy: bool,
        slippage_percent: Decimal,
    ) -> Result<Trade> {
        let body = self.build_body(serde_json::json!({
            "base": base_token,
            "quote": quote_token,
            "amount": amount.to_string(),
            "side": if is_buy { "BUY" } else { "SELL" },
            "limitPrice": "",  // No limit for market swap
            "allowedSlippage": format!("{}/100", slippage_percent)
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/trade", Some(body))
            .await?;

        let tx_hash = response["txHash"].as_str().unwrap_or("").to_string();
        let price = response["price"]
            .as_str()
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or_default();

        let gas_price = response["gasPrice"]
            .as_str()
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or_default();

        Ok(Trade {
            trade_id: tx_hash.clone(),
            order_id: tx_hash,
            symbol: format!("{base_token}-{quote_token}"),
            venue: self.name.clone(),
            side: if is_buy { Side::Buy } else { Side::Sell },
            price,
            quantity: amount,
            fee: Fee {
                asset: "GAS".to_string(),
                amount: gas_price,
                rate: None,
            },
            timestamp: chrono::Utc::now().timestamp_millis(),
            is_maker: false,
        })
    }

    async fn get_pool_info(&self, base_token: &str, quote_token: &str) -> Result<PoolInfo> {
        let body = self.build_body(serde_json::json!({
            "token0": base_token,
            "token1": quote_token
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/poolPrice", Some(body))
            .await?;

        Ok(PoolInfo {
            address: response["token0Address"].as_str().unwrap_or("").to_string(),
            base_token: base_token.to_string(),
            quote_token: quote_token.to_string(),
            base_reserve: response["token0Balance"]
                .as_str()
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or_default(),
            quote_reserve: response["token1Balance"]
                .as_str()
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or_default(),
            total_liquidity: Decimal::ZERO,
            fee_rate: decimal_from_str("0.003")?, // 0.3% typical
            apy: None,
        })
    }

    async fn add_liquidity(
        &self,
        base_token: &str,
        quote_token: &str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        let body = self.build_body(serde_json::json!({
            "token0": base_token,
            "token1": quote_token,
            "amount0": base_amount.to_string(),
            "amount1": quote_amount.to_string(),
            "allowedSlippage": format!("{}/100", slippage_percent)
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/liquidity/add", Some(body))
            .await?;

        Ok(LiquidityResult {
            tx_hash: response["txHash"].as_str().unwrap_or("").to_string(),
            pool_address: response["poolAddress"].as_str().unwrap_or("").to_string(),
            base_amount,
            quote_amount,
            lp_tokens: Decimal::ZERO, // Gateway may not return this
            share_percent: Decimal::ZERO,
        })
    }

    async fn remove_liquidity(
        &self,
        pool_address: &str,
        liquidity_amount: Decimal,
        slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        let body = self.build_body(serde_json::json!({
            "tokenId": pool_address,
            "decreasePercent": "100", // Remove all
            "allowedSlippage": format!("{}/100", slippage_percent)
        }));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/liquidity/remove", Some(body))
            .await?;

        Ok(LiquidityResult {
            tx_hash: response["txHash"].as_str().unwrap_or("").to_string(),
            pool_address: pool_address.to_string(),
            base_amount: Decimal::ZERO,
            quote_amount: Decimal::ZERO,
            lp_tokens: liquidity_amount,
            share_percent: Decimal::ZERO,
        })
    }

    async fn get_lp_positions(&self) -> Result<Vec<LpPosition>> {
        let body = self.build_body(serde_json::json!({}));

        let response: serde_json::Value = self
            .gateway_request(reqwest::Method::POST, "/amm/position", Some(body))
            .await?;

        // Parse positions from response
        let mut positions = Vec::new();

        if let Some(pos_array) = response.as_array() {
            for pos in pos_array {
                positions.push(LpPosition {
                    pool_address: pos["tokenId"].as_str().unwrap_or("").to_string(),
                    base_token: pos["token0"].as_str().unwrap_or("").to_string(),
                    quote_token: pos["token1"].as_str().unwrap_or("").to_string(),
                    lp_tokens: Decimal::ZERO,
                    base_amount: pos["amount0"]
                        .as_str()
                        .and_then(|s| s.parse::<Decimal>().ok())
                        .unwrap_or_default(),
                    quote_amount: pos["amount1"]
                        .as_str()
                        .and_then(|s| s.parse::<Decimal>().ok())
                        .unwrap_or_default(),
                    share_percent: Decimal::ZERO,
                    unrealized_pnl: pos["unclaimedToken0"]
                        .as_str()
                        .and_then(|s| s.parse::<Decimal>().ok()),
                });
            }
        }

        Ok(positions)
    }
}

use rust_decimal::prelude::FromStr;

fn decimal_from_str(s: &str) -> Result<Decimal> {
    Decimal::from_str(s).map_err(|e| Error::DeserializationError(e.to_string()))
}
