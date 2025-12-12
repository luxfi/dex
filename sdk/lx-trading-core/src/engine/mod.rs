//! Unified trading engine with smart order routing.

use dashmap::DashMap;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::adapters::{
    create_ccxt_adapter, create_hummingbot_adapter, create_native_adapter, VenueAdapter,
};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::orderbook::{AggregatedOrderbook, Orderbook};
use crate::types::*;

/// Unified trading client that abstracts multiple venues.
///
/// Provides a single interface for:
/// - Native LX DEX and LX AMM
/// - CCXT exchanges (Binance, MEXC, OKX, etc.)
/// - Hummingbot Gateway connectors
pub struct UnifiedClient {
    config: Config,
    venues: DashMap<String, Arc<dyn VenueAdapter>>,
    default_venue: RwLock<Option<String>>,
}

impl UnifiedClient {
    /// Create a new unified client from configuration
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            venues: DashMap::new(),
            default_venue: RwLock::new(None),
        })
    }

    /// Initialize all venues from configuration
    pub async fn init(&self) -> Result<()> {
        // Connect native venues
        for (name, cfg) in &self.config.native {
            let adapter = create_native_adapter(name, cfg).await?;
            self.venues.insert(name.clone(), adapter);
        }

        // Connect CCXT exchanges
        for (name, cfg) in &self.config.ccxt {
            let adapter = create_ccxt_adapter(name, cfg).await?;
            self.venues.insert(name.clone(), adapter);
        }

        // Connect Hummingbot gateways
        for (name, cfg) in &self.config.hummingbot {
            let adapter = create_hummingbot_adapter(name, cfg).await?;
            self.venues.insert(name.clone(), adapter);
        }

        // Set default venue
        if let Some(first) = self.config.general.venue_priority.first() {
            *self.default_venue.write().await = Some(first.clone());
        } else if let Some(entry) = self.venues.iter().next() {
            *self.default_venue.write().await = Some(entry.key().clone());
        }

        // Connect all venues
        for entry in self.venues.iter() {
            let name = entry.key().clone();
            // We need mutable access, so we'll reconnect
            drop(entry);
            if let Some(mut adapter) = self.venues.get_mut(&name) {
                let adapter = Arc::get_mut(&mut adapter).ok_or_else(|| {
                    Error::Internal("Failed to get mutable adapter reference".into())
                })?;
                adapter.connect().await?;
            }
        }

        Ok(())
    }

    /// Get a specific venue adapter
    pub fn venue(&self, name: &str) -> Option<Arc<dyn VenueAdapter>> {
        self.venues.get(name).map(|v| v.clone())
    }

    /// List connected venues
    pub fn venues(&self) -> Vec<VenueInfo> {
        self.venues
            .iter()
            .map(|entry| {
                let adapter = entry.value();
                VenueInfo {
                    name: adapter.name().to_string(),
                    venue_type: adapter.venue_type(),
                    connected: adapter.is_connected(),
                    latency_ms: adapter.latency_ms(),
                    supported_pairs: adapter
                        .capabilities()
                        .supported_pairs
                        .iter()
                        .cloned()
                        .collect(),
                    maker_fee: Decimal::from_str("0.001").unwrap_or_default(),
                    taker_fee: Decimal::from_str("0.002").unwrap_or_default(),
                }
            })
            .collect()
    }

    /// Get orderbook from a specific venue
    pub async fn orderbook(&self, symbol: &str) -> Result<Orderbook> {
        let venue = self.default_venue.read().await;
        let venue_name = venue
            .as_ref()
            .ok_or_else(|| Error::VenueNotConnected("No default venue".into()))?;

        let adapter = self
            .venues
            .get(venue_name)
            .ok_or_else(|| Error::VenueNotFound(venue_name.clone()))?;

        adapter.get_orderbook(symbol, None).await
    }

    /// Get orderbook from specific venue
    pub async fn orderbook_from(&self, symbol: &str, venue: &str) -> Result<Orderbook> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter.get_orderbook(symbol, None).await
    }

    /// Get aggregated orderbook from all venues
    pub async fn aggregated_orderbook(&self, symbol: &str) -> Result<AggregatedOrderbook> {
        let mut agg = AggregatedOrderbook::new(symbol);

        for entry in self.venues.iter() {
            let adapter = entry.value();
            if adapter.capabilities().orderbook {
                if let Ok(book) = adapter.get_orderbook(symbol, Some(20)).await {
                    agg.add_orderbook(&book);
                }
            }
        }

        Ok(agg)
    }

    /// Get ticker from default venue
    pub async fn ticker(&self, symbol: &str) -> Result<Ticker> {
        let venue = self.default_venue.read().await;
        let venue_name = venue
            .as_ref()
            .ok_or_else(|| Error::VenueNotConnected("No default venue".into()))?;

        let adapter = self
            .venues
            .get(venue_name)
            .ok_or_else(|| Error::VenueNotFound(venue_name.clone()))?;

        adapter.get_ticker(symbol).await
    }

    /// Get tickers from all venues
    pub async fn tickers(&self, symbol: &str) -> Result<Vec<Ticker>> {
        let mut tickers = Vec::new();

        for entry in self.venues.iter() {
            let adapter = entry.value();
            if let Ok(ticker) = adapter.get_ticker(symbol).await {
                tickers.push(ticker);
            }
        }

        Ok(tickers)
    }

    /// Get aggregated balances across all venues
    pub async fn balances(&self) -> Result<Vec<AggregatedBalance>> {
        let mut by_asset: std::collections::HashMap<String, Vec<Balance>> =
            std::collections::HashMap::new();

        for entry in self.venues.iter() {
            let adapter = entry.value();
            if let Ok(balances) = adapter.get_balances().await {
                for balance in balances {
                    by_asset
                        .entry(balance.asset.clone())
                        .or_default()
                        .push(balance);
                }
            }
        }

        Ok(by_asset
            .into_iter()
            .map(|(asset, balances)| {
                let total_free: Decimal = balances.iter().map(|b| b.free).sum();
                let total_locked: Decimal = balances.iter().map(|b| b.locked).sum();

                AggregatedBalance {
                    asset,
                    total_free,
                    total_locked,
                    total: total_free + total_locked,
                    by_venue: balances,
                }
            })
            .collect())
    }

    /// Place market buy order with smart routing
    pub async fn buy(
        &self,
        symbol: &str,
        quantity: Decimal,
        venue: Option<&str>,
    ) -> Result<Order> {
        let request = OrderRequest::market(symbol, Side::Buy, quantity);

        if let Some(v) = venue {
            self.place_order_on(request.with_venue(v)).await
        } else if self.config.general.smart_routing {
            self.smart_route_order(request).await
        } else {
            self.place_order(request).await
        }
    }

    /// Place market sell order with smart routing
    pub async fn sell(
        &self,
        symbol: &str,
        quantity: Decimal,
        venue: Option<&str>,
    ) -> Result<Order> {
        let request = OrderRequest::market(symbol, Side::Sell, quantity);

        if let Some(v) = venue {
            self.place_order_on(request.with_venue(v)).await
        } else if self.config.general.smart_routing {
            self.smart_route_order(request).await
        } else {
            self.place_order(request).await
        }
    }

    /// Place limit buy order
    pub async fn limit_buy(
        &self,
        symbol: &str,
        quantity: Decimal,
        price: Decimal,
        venue: Option<&str>,
    ) -> Result<Order> {
        let request = OrderRequest::limit(symbol, Side::Buy, quantity, price);

        if let Some(v) = venue {
            self.place_order_on(request.with_venue(v)).await
        } else {
            self.place_order(request).await
        }
    }

    /// Place limit sell order
    pub async fn limit_sell(
        &self,
        symbol: &str,
        quantity: Decimal,
        price: Decimal,
        venue: Option<&str>,
    ) -> Result<Order> {
        let request = OrderRequest::limit(symbol, Side::Sell, quantity, price);

        if let Some(v) = venue {
            self.place_order_on(request.with_venue(v)).await
        } else {
            self.place_order(request).await
        }
    }

    /// Place order on default venue
    pub async fn place_order(&self, request: OrderRequest) -> Result<Order> {
        if let Some(venue) = &request.venue {
            return self.place_order_on(request.clone()).await;
        }

        let venue = self.default_venue.read().await;
        let venue_name = venue
            .as_ref()
            .ok_or_else(|| Error::VenueNotConnected("No default venue".into()))?;

        let adapter = self
            .venues
            .get(venue_name)
            .ok_or_else(|| Error::VenueNotFound(venue_name.clone()))?;

        adapter.place_order(request).await
    }

    /// Place order on specific venue
    pub async fn place_order_on(&self, request: OrderRequest) -> Result<Order> {
        let venue = request
            .venue
            .as_ref()
            .ok_or_else(|| Error::InvalidOrder("No venue specified".into()))?;

        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.clone()))?;

        adapter.place_order(request).await
    }

    /// Smart order routing - find best venue for execution
    pub async fn smart_route_order(&self, request: OrderRequest) -> Result<Order> {
        let agg_book = self.aggregated_orderbook(&request.symbol).await?;

        let best_venue = match request.side {
            Side::Buy => agg_book.best_venue_buy(request.quantity),
            Side::Sell => agg_book.best_venue_sell(request.quantity),
        };

        if let Some((venue, _price)) = best_venue {
            let routed_request = OrderRequest {
                venue: Some(venue),
                ..request
            };
            self.place_order_on(routed_request).await
        } else {
            // Fall back to default venue
            self.place_order(request).await
        }
    }

    /// Cancel order
    pub async fn cancel_order(&self, order_id: &str, symbol: &str, venue: &str) -> Result<Order> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter.cancel_order(order_id, symbol).await
    }

    /// Cancel all orders on a venue
    pub async fn cancel_all_orders(&self, symbol: Option<&str>, venue: &str) -> Result<Vec<Order>> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter.cancel_all_orders(symbol).await
    }

    /// Get all open orders across venues
    pub async fn open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let mut all_orders = Vec::new();

        for entry in self.venues.iter() {
            let adapter = entry.value();
            if let Ok(orders) = adapter.get_open_orders(symbol).await {
                all_orders.extend(orders);
            }
        }

        Ok(all_orders)
    }

    /// AMM: Execute swap
    pub async fn swap(
        &self,
        base_token: &str,
        quote_token: &str,
        amount: Decimal,
        is_buy: bool,
        slippage: Decimal,
        venue: &str,
    ) -> Result<Trade> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter
            .execute_swap(base_token, quote_token, amount, is_buy, slippage)
            .await
    }

    /// AMM: Get swap quote
    pub async fn quote(
        &self,
        base_token: &str,
        quote_token: &str,
        amount: Decimal,
        is_buy: bool,
        venue: &str,
    ) -> Result<crate::adapters::adapter::SwapQuote> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter
            .get_swap_quote(base_token, quote_token, amount, is_buy)
            .await
    }

    /// AMM: Add liquidity
    pub async fn add_liquidity(
        &self,
        base_token: &str,
        quote_token: &str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage: Decimal,
        venue: &str,
    ) -> Result<crate::adapters::adapter::LiquidityResult> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter
            .add_liquidity(base_token, quote_token, base_amount, quote_amount, slippage)
            .await
    }

    /// AMM: Remove liquidity
    pub async fn remove_liquidity(
        &self,
        pool_address: &str,
        liquidity_amount: Decimal,
        slippage: Decimal,
        venue: &str,
    ) -> Result<crate::adapters::adapter::LiquidityResult> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter
            .remove_liquidity(pool_address, liquidity_amount, slippage)
            .await
    }

    /// Get LP positions
    pub async fn lp_positions(&self, venue: &str) -> Result<Vec<crate::adapters::adapter::LpPosition>> {
        let adapter = self
            .venues
            .get(venue)
            .ok_or_else(|| Error::VenueNotFound(venue.into()))?;

        adapter.get_lp_positions().await
    }
}

use rust_decimal::prelude::FromStr;

fn decimal_from_str(s: &str) -> Result<Decimal> {
    Decimal::from_str(s).map_err(|e| Error::DeserializationError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let config = Config::default();
        let client = UnifiedClient::new(config).unwrap();
        assert!(client.venues().is_empty());
    }
}
