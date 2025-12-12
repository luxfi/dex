//! Execution algorithms for advanced order types.

use rust_decimal::Decimal;
use std::time::Duration;

use crate::engine::UnifiedClient;
use crate::error::Result;
use crate::types::*;

/// TWAP (Time-Weighted Average Price) execution
pub struct TwapExecutor {
    client: std::sync::Arc<UnifiedClient>,
    symbol: String,
    side: Side,
    total_quantity: Decimal,
    num_slices: usize,
    interval: Duration,
}

impl TwapExecutor {
    pub fn new(
        client: std::sync::Arc<UnifiedClient>,
        symbol: String,
        side: Side,
        total_quantity: Decimal,
        duration: Duration,
        num_slices: usize,
    ) -> Self {
        let interval = duration / num_slices as u32;
        Self {
            client,
            symbol,
            side,
            total_quantity,
            num_slices,
            interval,
        }
    }

    pub async fn execute(&self) -> Result<Vec<Order>> {
        let slice_quantity = self.total_quantity / Decimal::from(self.num_slices);
        let mut orders = Vec::new();

        for i in 0..self.num_slices {
            let remaining = self.total_quantity - slice_quantity * Decimal::from(i);
            let qty = slice_quantity.min(remaining);

            if qty <= Decimal::ZERO {
                break;
            }

            let order = match self.side {
                Side::Buy => self.client.buy(&self.symbol, qty, None).await?,
                Side::Sell => self.client.sell(&self.symbol, qty, None).await?,
            };

            orders.push(order);

            if i < self.num_slices - 1 {
                tokio::time::sleep(self.interval).await;
            }
        }

        Ok(orders)
    }
}

/// VWAP (Volume-Weighted Average Price) execution
pub struct VwapExecutor {
    client: std::sync::Arc<UnifiedClient>,
    symbol: String,
    side: Side,
    total_quantity: Decimal,
    participation_rate: Decimal, // e.g., 0.1 = 10% of volume
    max_duration: Duration,
}

impl VwapExecutor {
    pub fn new(
        client: std::sync::Arc<UnifiedClient>,
        symbol: String,
        side: Side,
        total_quantity: Decimal,
        participation_rate: Decimal,
        max_duration: Duration,
    ) -> Self {
        Self {
            client,
            symbol,
            side,
            total_quantity,
            participation_rate,
            max_duration,
        }
    }

    pub async fn execute(&self) -> Result<Vec<Order>> {
        let mut orders = Vec::new();
        let mut remaining = self.total_quantity;
        let start = std::time::Instant::now();
        let check_interval = Duration::from_secs(5);

        while remaining > Decimal::ZERO && start.elapsed() < self.max_duration {
            // Get current market volume (simplified - would need real volume data)
            let ticker = self.client.ticker(&self.symbol).await?;
            let volume = ticker.volume_24h.unwrap_or(Decimal::from(1000));

            // Calculate slice based on participation rate
            let hourly_volume = volume / Decimal::from(24);
            let slice_volume = hourly_volume * self.participation_rate / Decimal::from(720); // per 5 sec
            let qty = slice_volume.min(remaining);

            if qty > Decimal::ZERO {
                let order = match self.side {
                    Side::Buy => self.client.buy(&self.symbol, qty, None).await?,
                    Side::Sell => self.client.sell(&self.symbol, qty, None).await?,
                };
                orders.push(order);
                remaining -= qty;
            }

            tokio::time::sleep(check_interval).await;
        }

        Ok(orders)
    }
}

/// Iceberg order execution
pub struct IcebergExecutor {
    client: std::sync::Arc<UnifiedClient>,
    symbol: String,
    side: Side,
    total_quantity: Decimal,
    visible_quantity: Decimal,
    price: Decimal,
    venue: Option<String>,
}

impl IcebergExecutor {
    pub fn new(
        client: std::sync::Arc<UnifiedClient>,
        symbol: String,
        side: Side,
        total_quantity: Decimal,
        visible_quantity: Decimal,
        price: Decimal,
        venue: Option<String>,
    ) -> Self {
        Self {
            client,
            symbol,
            side,
            total_quantity,
            visible_quantity,
            price,
            venue,
        }
    }

    pub async fn execute(&self) -> Result<Vec<Order>> {
        let mut orders = Vec::new();
        let mut remaining = self.total_quantity;

        while remaining > Decimal::ZERO {
            let qty = self.visible_quantity.min(remaining);

            let request = OrderRequest::limit(&self.symbol, self.side, qty, self.price);
            let request = if let Some(ref v) = self.venue {
                request.with_venue(v)
            } else {
                request
            };

            let order = self.client.place_order(request).await?;

            // Wait for fill
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;

                let updated = self
                    .client
                    .venue(order.venue.as_str())
                    .ok_or_else(|| crate::error::Error::VenueNotFound(order.venue.clone()))?
                    .get_order(&order.order_id, &self.symbol)
                    .await?;

                if updated.is_done() {
                    remaining -= updated.filled_quantity;
                    orders.push(updated);
                    break;
                }
            }
        }

        Ok(orders)
    }
}

/// Sniper execution - wait for price target then execute
pub struct SniperExecutor {
    client: std::sync::Arc<UnifiedClient>,
    symbol: String,
    side: Side,
    quantity: Decimal,
    target_price: Decimal,
    timeout: Duration,
}

impl SniperExecutor {
    pub fn new(
        client: std::sync::Arc<UnifiedClient>,
        symbol: String,
        side: Side,
        quantity: Decimal,
        target_price: Decimal,
        timeout: Duration,
    ) -> Self {
        Self {
            client,
            symbol,
            side,
            quantity,
            target_price,
            timeout,
        }
    }

    pub async fn execute(&self) -> Result<Option<Order>> {
        let start = std::time::Instant::now();
        let check_interval = Duration::from_millis(100);

        while start.elapsed() < self.timeout {
            let ticker = self.client.ticker(&self.symbol).await?;

            let should_execute = match self.side {
                Side::Buy => ticker.ask.map(|a| a <= self.target_price).unwrap_or(false),
                Side::Sell => ticker.bid.map(|b| b >= self.target_price).unwrap_or(false),
            };

            if should_execute {
                let order = match self.side {
                    Side::Buy => self.client.buy(&self.symbol, self.quantity, None).await?,
                    Side::Sell => self.client.sell(&self.symbol, self.quantity, None).await?,
                };
                return Ok(Some(order));
            }

            tokio::time::sleep(check_interval).await;
        }

        Ok(None) // Timeout without execution
    }
}
