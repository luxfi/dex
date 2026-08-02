//! Risk management module.

use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::RiskConfig;
use crate::error::{Error, Result};
use crate::types::*;

/// Risk manager for enforcing trading limits
pub struct RiskManager {
    config: RiskConfig,
    positions: RwLock<HashMap<String, Decimal>>,
    daily_pnl: RwLock<Decimal>,
    open_orders_count: RwLock<HashMap<String, u32>>,
    kill_switch: AtomicBool,
}

impl RiskManager {
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            positions: RwLock::new(HashMap::new()),
            daily_pnl: RwLock::new(Decimal::ZERO),
            open_orders_count: RwLock::new(HashMap::new()),
            kill_switch: AtomicBool::new(false),
        }
    }

    /// Check if risk manager is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if kill switch is active
    pub fn is_killed(&self) -> bool {
        self.kill_switch.load(Ordering::Relaxed)
    }

    /// Activate kill switch - stops all trading
    pub fn kill(&self) {
        self.kill_switch.store(true, Ordering::Relaxed);
    }

    /// Deactivate kill switch
    pub fn reset(&self) {
        self.kill_switch.store(false, Ordering::Relaxed);
    }

    /// Validate an order against risk limits
    pub fn validate_order(&self, request: &OrderRequest) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if self.is_killed() {
            return Err(Error::RiskLimitExceeded("Kill switch is active".into()));
        }

        // Check order size
        if !self.config.max_order_size.is_zero() && request.quantity > self.config.max_order_size {
            return Err(Error::RiskLimitExceeded(format!(
                "Order size {} exceeds max {}",
                request.quantity, self.config.max_order_size
            )));
        }

        // Check position limit
        let pair = TradingPair::from_symbol(&request.symbol);
        if let Some(pair) = pair {
            let current_position = self
                .positions
                .read()
                .get(&pair.base)
                .copied()
                .unwrap_or_default();
            let new_position = match request.side {
                Side::Buy => current_position + request.quantity,
                Side::Sell => current_position - request.quantity,
            };

            // Check asset-specific limit
            if let Some(limit) = self.config.position_limits.get(&pair.base) {
                if new_position.abs() > *limit {
                    return Err(Error::PositionLimitExceeded {
                        asset: pair.base.clone(),
                        current: current_position.to_string(),
                        requested: request.quantity.to_string(),
                        limit: limit.to_string(),
                    });
                }
            }

            // Check global position limit
            if !self.config.max_position_size.is_zero()
                && new_position.abs() > self.config.max_position_size
            {
                return Err(Error::PositionLimitExceeded {
                    asset: pair.base.clone(),
                    current: current_position.to_string(),
                    requested: request.quantity.to_string(),
                    limit: self.config.max_position_size.to_string(),
                });
            }
        }

        // Check open orders count
        let count = self
            .open_orders_count
            .read()
            .get(&request.symbol)
            .copied()
            .unwrap_or(0);

        if count >= self.config.max_open_orders {
            return Err(Error::RiskLimitExceeded(format!(
                "Max open orders ({}) reached for {}",
                self.config.max_open_orders, request.symbol
            )));
        }

        // Check daily loss limit
        if !self.config.max_daily_loss.is_zero() {
            let pnl = *self.daily_pnl.read();
            if pnl < -self.config.max_daily_loss {
                return Err(Error::DailyLossLimitExceeded {
                    loss: pnl.abs().to_string(),
                    limit: self.config.max_daily_loss.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Update position after a trade
    pub fn update_position(&self, asset: &str, quantity: Decimal, side: Side) {
        let mut positions = self.positions.write();
        let current = positions.get(asset).copied().unwrap_or_default();
        let new_position = match side {
            Side::Buy => current + quantity,
            Side::Sell => current - quantity,
        };
        positions.insert(asset.to_string(), new_position);
    }

    /// Update PnL
    pub fn update_pnl(&self, pnl: Decimal) {
        let mut daily_pnl = self.daily_pnl.write();
        *daily_pnl += pnl;

        // Auto kill switch if enabled and loss limit exceeded
        if self.config.kill_switch_enabled
            && !self.config.max_daily_loss.is_zero()
            && *daily_pnl < -self.config.max_daily_loss
        {
            self.kill();
        }
    }

    /// Increment open orders count
    pub fn order_opened(&self, symbol: &str) {
        let mut counts = self.open_orders_count.write();
        *counts.entry(symbol.to_string()).or_insert(0) += 1;
    }

    /// Decrement open orders count
    pub fn order_closed(&self, symbol: &str) {
        let mut counts = self.open_orders_count.write();
        if let Some(count) = counts.get_mut(symbol) {
            *count = count.saturating_sub(1);
        }
    }

    /// Get current position for an asset
    pub fn position(&self, asset: &str) -> Decimal {
        self.positions
            .read()
            .get(asset)
            .copied()
            .unwrap_or_default()
    }

    /// Get all positions
    pub fn positions(&self) -> HashMap<String, Decimal> {
        self.positions.read().clone()
    }

    /// Get daily PnL
    pub fn daily_pnl(&self) -> Decimal {
        *self.daily_pnl.read()
    }

    /// Reset daily PnL (call at start of trading day)
    pub fn reset_daily_pnl(&self) {
        *self.daily_pnl.write() = Decimal::ZERO;
    }

    /// Get open orders count for a symbol
    pub fn open_orders(&self, symbol: &str) -> u32 {
        self.open_orders_count
            .read()
            .get(symbol)
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RiskConfig {
        RiskConfig {
            enabled: true,
            max_position_size: Decimal::from(100),
            max_order_size: Decimal::from(10),
            max_daily_loss: Decimal::from(1000),
            max_open_orders: 10,
            kill_switch_enabled: true,
            position_limits: HashMap::new(),
        }
    }

    #[test]
    fn test_order_size_limit() {
        let rm = RiskManager::new(test_config());

        let request = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(5));
        assert!(rm.validate_order(&request).is_ok());

        let request = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(15));
        assert!(rm.validate_order(&request).is_err());
    }

    #[test]
    fn test_position_tracking() {
        let rm = RiskManager::new(test_config());

        rm.update_position("BTC", Decimal::from(10), Side::Buy);
        assert_eq!(rm.position("BTC"), Decimal::from(10));

        rm.update_position("BTC", Decimal::from(5), Side::Sell);
        assert_eq!(rm.position("BTC"), Decimal::from(5));
    }

    #[test]
    fn test_kill_switch() {
        let rm = RiskManager::new(test_config());

        assert!(!rm.is_killed());
        rm.kill();
        assert!(rm.is_killed());

        let request = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from(1));
        assert!(rm.validate_order(&request).is_err());

        rm.reset();
        assert!(!rm.is_killed());
    }
}
