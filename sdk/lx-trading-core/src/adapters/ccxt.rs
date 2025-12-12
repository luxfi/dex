//! CCXT adapter - enables trading on 100+ exchanges via CCXT library.
//!
//! This adapter wraps CCXT (either via ccxt-rs or by calling Python/Node CCXT)
//! to provide access to centralized exchanges like Binance, MEXC, OKX, etc.

use async_trait::async_trait;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::adapters::adapter::*;
use crate::config::CcxtConfig;
use crate::error::{Error, Result};
use crate::orderbook::Orderbook;
use crate::types::*;

/// CCXT adapter using Python CCXT via subprocess
///
/// This approach provides full CCXT compatibility without requiring
/// native Rust bindings for every exchange.
pub struct CcxtAdapter {
    name: String,
    config: CcxtConfig,
    capabilities: VenueCapabilities,
    connected: AtomicBool,
    latency: AtomicU64,
    markets_cache: RwLock<HashMap<String, MarketInfo>>,
}

impl CcxtAdapter {
    pub async fn new(name: &str, config: CcxtConfig) -> Result<Self> {
        let mut capabilities = VenueCapabilities::clob();
        // CCXT exchanges typically don't support batch orders through unified API
        capabilities.batch_orders = false;

        Ok(Self {
            name: name.to_string(),
            config,
            capabilities,
            connected: AtomicBool::new(false),
            latency: AtomicU64::new(0),
            markets_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Execute CCXT command via Python subprocess
    async fn ccxt_call<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T> {
        let script = format!(
            r#"
import ccxt
import json
import sys

config = json.loads('{config}')
exchange_class = getattr(ccxt, config['exchange_id'])
exchange = exchange_class({{
    'apiKey': config.get('api_key'),
    'secret': config.get('api_secret'),
    'password': config.get('password'),
    'enableRateLimit': config.get('rate_limit', True),
    'sandbox': config.get('sandbox', False),
    **config.get('options', {{}})
}})

params = json.loads('{params}')
method = getattr(exchange, '{method}')

if callable(method):
    if isinstance(params, list):
        result = method(*params)
    elif isinstance(params, dict):
        result = method(**params)
    else:
        result = method(params) if params else method()
else:
    result = method

print(json.dumps(result))
"#,
            config = serde_json::to_string(&self.config).unwrap_or_default(),
            params = serde_json::to_string(&params).unwrap_or_default(),
            method = method
        );

        let start = std::time::Instant::now();

        let output = tokio::process::Command::new("python3")
            .arg("-c")
            .arg(&script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Error::CcxtError(format!("Failed to execute Python: {}", e)))?;

        let latency = start.elapsed().as_millis() as u64;
        self.latency.store(latency, Ordering::Relaxed);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::CcxtError(format!(
                "CCXT {} failed: {}",
                method, stderr
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(&stdout).map_err(|e| {
            Error::DeserializationError(format!("Failed to parse CCXT response: {}", e))
        })
    }

    /// Convert CCXT order to our Order type
    fn convert_order(ccxt_order: serde_json::Value, venue: &str) -> Result<Order> {
        let status = match ccxt_order["status"].as_str().unwrap_or("unknown") {
            "open" => OrderStatus::Open,
            "closed" => OrderStatus::Filled,
            "canceled" | "cancelled" => OrderStatus::Cancelled,
            "expired" => OrderStatus::Expired,
            "rejected" => OrderStatus::Rejected,
            _ => OrderStatus::Pending,
        };

        let side = match ccxt_order["side"].as_str().unwrap_or("buy") {
            "sell" => Side::Sell,
            _ => Side::Buy,
        };

        let order_type = match ccxt_order["type"].as_str().unwrap_or("limit") {
            "market" => OrderType::Market,
            "stop" | "stop_loss" => OrderType::StopLoss,
            "take_profit" => OrderType::TakeProfit,
            _ => OrderType::Limit,
        };

        let quantity: Decimal = ccxt_order["amount"]
            .as_f64()
            .map(|f| Decimal::try_from(f).unwrap_or_default())
            .unwrap_or_default();

        let filled: Decimal = ccxt_order["filled"]
            .as_f64()
            .map(|f| Decimal::try_from(f).unwrap_or_default())
            .unwrap_or_default();

        let price: Option<Decimal> = ccxt_order["price"]
            .as_f64()
            .map(|f| Decimal::try_from(f).ok())
            .flatten();

        let average: Option<Decimal> = ccxt_order["average"]
            .as_f64()
            .map(|f| Decimal::try_from(f).ok())
            .flatten();

        Ok(Order {
            order_id: ccxt_order["id"].as_str().unwrap_or("").to_string(),
            client_order_id: ccxt_order["clientOrderId"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            symbol: ccxt_order["symbol"].as_str().unwrap_or("").to_string(),
            venue: venue.to_string(),
            side,
            order_type,
            status,
            quantity,
            filled_quantity: filled,
            remaining_quantity: quantity - filled,
            price,
            average_price: average,
            created_at: ccxt_order["timestamp"].as_i64().unwrap_or(0),
            updated_at: ccxt_order["lastTradeTimestamp"].as_i64().unwrap_or(0),
            fees: vec![],
        })
    }

    /// Convert CCXT trade to our Trade type
    fn convert_trade(ccxt_trade: serde_json::Value, venue: &str) -> Result<Trade> {
        let side = match ccxt_trade["side"].as_str().unwrap_or("buy") {
            "sell" => Side::Sell,
            _ => Side::Buy,
        };

        let price: Decimal = ccxt_trade["price"]
            .as_f64()
            .map(|f| Decimal::try_from(f).unwrap_or_default())
            .unwrap_or_default();

        let quantity: Decimal = ccxt_trade["amount"]
            .as_f64()
            .map(|f| Decimal::try_from(f).unwrap_or_default())
            .unwrap_or_default();

        let fee_amount: Decimal = ccxt_trade["fee"]["cost"]
            .as_f64()
            .map(|f| Decimal::try_from(f).unwrap_or_default())
            .unwrap_or_default();

        let fee_asset = ccxt_trade["fee"]["currency"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(Trade {
            trade_id: ccxt_trade["id"].as_str().unwrap_or("").to_string(),
            order_id: ccxt_trade["order"].as_str().unwrap_or("").to_string(),
            symbol: ccxt_trade["symbol"].as_str().unwrap_or("").to_string(),
            venue: venue.to_string(),
            side,
            price,
            quantity,
            fee: Fee {
                asset: fee_asset,
                amount: fee_amount,
                rate: None,
            },
            timestamp: ccxt_trade["timestamp"].as_i64().unwrap_or(0),
            is_maker: ccxt_trade["takerOrMaker"].as_str() == Some("maker"),
        })
    }
}

#[async_trait]
impl VenueAdapter for CcxtAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn venue_type(&self) -> VenueType {
        VenueType::Ccxt
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
        // Load markets to verify connection
        let _: serde_json::Value = self.ccxt_call("load_markets", serde_json::json!({})).await?;
        self.connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn start_streaming(&mut self, _symbols: &[String]) -> Result<()> {
        // CCXT Pro would be needed for streaming
        // For now, use polling
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        Ok(())
    }

    async fn get_markets(&self) -> Result<Vec<MarketInfo>> {
        let markets: serde_json::Value =
            self.ccxt_call("load_markets", serde_json::json!({})).await?;

        let mut result = Vec::new();
        if let Some(obj) = markets.as_object() {
            for (symbol, market) in obj {
                let info = MarketInfo {
                    symbol: symbol.clone(),
                    base: market["base"].as_str().unwrap_or("").to_string(),
                    quote: market["quote"].as_str().unwrap_or("").to_string(),
                    price_precision: market["precision"]["price"].as_u64().unwrap_or(8) as u32,
                    quantity_precision: market["precision"]["amount"].as_u64().unwrap_or(8) as u32,
                    min_quantity: market["limits"]["amount"]["min"]
                        .as_f64()
                        .map(|f| Decimal::try_from(f).unwrap_or_default())
                        .unwrap_or_default(),
                    max_quantity: market["limits"]["amount"]["max"]
                        .as_f64()
                        .map(|f| Decimal::try_from(f).ok())
                        .flatten(),
                    min_notional: market["limits"]["cost"]["min"]
                        .as_f64()
                        .map(|f| Decimal::try_from(f).ok())
                        .flatten(),
                    tick_size: market["precision"]["price"]
                        .as_f64()
                        .map(|p| Decimal::from(1) / Decimal::from(10u64.pow(p as u32)))
                        .unwrap_or_else(|| Decimal::from_str("0.00000001").unwrap_or_default()),
                    lot_size: market["precision"]["amount"]
                        .as_f64()
                        .map(|p| Decimal::from(1) / Decimal::from(10u64.pow(p as u32)))
                        .unwrap_or_else(|| Decimal::from_str("0.00000001").unwrap_or_default()),
                };
                result.push(info);
            }
        }

        Ok(result)
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let ticker: serde_json::Value = self
            .ccxt_call("fetch_ticker", serde_json::json!([symbol]))
            .await?;

        Ok(Ticker {
            symbol: ticker["symbol"].as_str().unwrap_or("").to_string(),
            venue: self.name.clone(),
            bid: ticker["bid"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            ask: ticker["ask"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            last: ticker["last"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            volume_24h: ticker["baseVolume"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            high_24h: ticker["high"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            low_24h: ticker["low"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            change_24h: ticker["percentage"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
            timestamp: ticker["timestamp"].as_i64().unwrap_or(0),
        })
    }

    async fn get_tickers(&self, symbols: &[String]) -> Result<Vec<Ticker>> {
        let tickers: serde_json::Value = self
            .ccxt_call("fetch_tickers", serde_json::json!([symbols]))
            .await?;

        let mut result = Vec::new();
        if let Some(obj) = tickers.as_object() {
            for (_, ticker) in obj {
                result.push(Ticker {
                    symbol: ticker["symbol"].as_str().unwrap_or("").to_string(),
                    venue: self.name.clone(),
                    bid: ticker["bid"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    ask: ticker["ask"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    last: ticker["last"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    volume_24h: ticker["baseVolume"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    high_24h: ticker["high"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    low_24h: ticker["low"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    change_24h: ticker["percentage"].as_f64().map(|f| Decimal::try_from(f).ok()).flatten(),
                    timestamp: ticker["timestamp"].as_i64().unwrap_or(0),
                });
            }
        }

        Ok(result)
    }

    async fn get_orderbook(&self, symbol: &str, depth: Option<usize>) -> Result<Orderbook> {
        let params = match depth {
            Some(d) => serde_json::json!([symbol, d]),
            None => serde_json::json!([symbol]),
        };

        let book: serde_json::Value = self.ccxt_call("fetch_order_book", params).await?;

        let mut orderbook = Orderbook::new(symbol, &self.name);

        if let Some(bids) = book["bids"].as_array() {
            for bid in bids {
                if let (Some(price), Some(qty)) = (bid[0].as_f64(), bid[1].as_f64()) {
                    orderbook.add_bid(
                        Decimal::try_from(price).unwrap_or_default(),
                        Decimal::try_from(qty).unwrap_or_default(),
                    );
                }
            }
        }

        if let Some(asks) = book["asks"].as_array() {
            for ask in asks {
                if let (Some(price), Some(qty)) = (ask[0].as_f64(), ask[1].as_f64()) {
                    orderbook.add_ask(
                        Decimal::try_from(price).unwrap_or_default(),
                        Decimal::try_from(qty).unwrap_or_default(),
                    );
                }
            }
        }

        Ok(orderbook)
    }

    async fn get_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>> {
        let params = match limit {
            Some(l) => serde_json::json!([symbol, null, l]),
            None => serde_json::json!([symbol]),
        };

        let trades: Vec<serde_json::Value> = self.ccxt_call("fetch_trades", params).await?;

        trades
            .into_iter()
            .map(|t| Self::convert_trade(t, &self.name))
            .collect()
    }

    async fn get_balances(&self) -> Result<Vec<Balance>> {
        let balances: serde_json::Value =
            self.ccxt_call("fetch_balance", serde_json::json!({})).await?;

        let mut result = Vec::new();
        if let Some(total) = balances["total"].as_object() {
            for (asset, amount) in total {
                let total_amount = amount.as_f64().unwrap_or(0.0);
                if total_amount > 0.0 {
                    let free = balances["free"][asset].as_f64().unwrap_or(0.0);
                    let used = balances["used"][asset].as_f64().unwrap_or(0.0);

                    result.push(Balance::new(
                        asset.clone(),
                        &self.name,
                        Decimal::try_from(free).unwrap_or_default(),
                        Decimal::try_from(used).unwrap_or_default(),
                    ));
                }
            }
        }

        Ok(result)
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        let balances = self.get_balances().await?;
        balances
            .into_iter()
            .find(|b| b.asset == asset)
            .ok_or_else(|| Error::MarketNotFound(asset.to_string()))
    }

    async fn get_open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let params = match symbol {
            Some(s) => serde_json::json!([s]),
            None => serde_json::json!([]),
        };

        let orders: Vec<serde_json::Value> = self.ccxt_call("fetch_open_orders", params).await?;

        orders
            .into_iter()
            .map(|o| Self::convert_order(o, &self.name))
            .collect()
    }

    async fn get_order(&self, order_id: &str, symbol: &str) -> Result<Order> {
        let order: serde_json::Value = self
            .ccxt_call("fetch_order", serde_json::json!([order_id, symbol]))
            .await?;

        Self::convert_order(order, &self.name)
    }

    async fn get_order_history(
        &self,
        symbol: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Order>> {
        let params = match (symbol, limit) {
            (Some(s), Some(l)) => serde_json::json!([s, null, l]),
            (Some(s), None) => serde_json::json!([s]),
            (None, Some(l)) => serde_json::json!([null, null, l]),
            (None, None) => serde_json::json!([]),
        };

        let orders: Vec<serde_json::Value> = self.ccxt_call("fetch_closed_orders", params).await?;

        orders
            .into_iter()
            .map(|o| Self::convert_order(o, &self.name))
            .collect()
    }

    async fn place_order(&self, request: OrderRequest) -> Result<Order> {
        let side = match request.side {
            Side::Buy => "buy",
            Side::Sell => "sell",
        };

        let order_type = match request.order_type {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::StopLoss => "stop",
            OrderType::StopLossLimit => "stop_limit",
            _ => "limit",
        };

        let params = serde_json::json!({
            "symbol": request.symbol,
            "type": order_type,
            "side": side,
            "amount": request.quantity.to_string().parse::<f64>().unwrap_or(0.0),
            "price": request.price.map(|p| p.to_string().parse::<f64>().unwrap_or(0.0)),
            "clientOrderId": request.client_order_id,
        });

        let order: serde_json::Value = self.ccxt_call("create_order", params).await?;

        Self::convert_order(order, &self.name)
    }

    async fn place_orders(&self, requests: Vec<OrderRequest>) -> Result<Vec<Order>> {
        // CCXT doesn't have unified batch order API
        // Execute sequentially
        let mut orders = Vec::new();
        for request in requests {
            orders.push(self.place_order(request).await?);
        }
        Ok(orders)
    }

    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<Order> {
        let order: serde_json::Value = self
            .ccxt_call("cancel_order", serde_json::json!([order_id, symbol]))
            .await?;

        Self::convert_order(order, &self.name)
    }

    async fn cancel_orders(&self, order_ids: &[(String, String)]) -> Result<Vec<Order>> {
        let mut orders = Vec::new();
        for (order_id, symbol) in order_ids {
            if let Ok(order) = self.cancel_order(order_id, symbol).await {
                orders.push(order);
            }
        }
        Ok(orders)
    }

    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let params = match symbol {
            Some(s) => serde_json::json!([s]),
            None => serde_json::json!([]),
        };

        let orders: Vec<serde_json::Value> =
            self.ccxt_call("cancel_all_orders", params).await?;

        orders
            .into_iter()
            .map(|o| Self::convert_order(o, &self.name))
            .collect()
    }
}

use rust_decimal::prelude::FromStr;

fn decimal_from_str(s: &str) -> Result<Decimal> {
    Decimal::from_str(s).map_err(|e| Error::DeserializationError(e.to_string()))
}
