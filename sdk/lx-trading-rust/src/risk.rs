//! Risk management for trading.
//!
//! Thread-safe risk controls and position tracking.

use crate::types::{Decimal, OrderRequest, Side, TradingPair};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use thiserror::Error;

/// Risk validation error.
#[derive(Debug, Error)]
pub enum RiskError {
    #[error("Kill switch is active")]
    KillSwitchActive,

    #[error("Order size {0} exceeds max {1}")]
    OrderSizeExceeded(String, String),

    #[error("Position limit exceeded for {asset}: {current} + {order} > {limit}")]
    PositionLimitExceeded {
        asset: String,
        current: String,
        order: String,
        limit: String,
    },

    #[error("Max position size exceeded: {current} > {limit}")]
    MaxPositionExceeded { current: String, limit: String },

    #[error("Max open orders ({max}) reached for {symbol}")]
    MaxOpenOrdersReached { symbol: String, max: i32 },

    #[error("Daily loss limit exceeded: {loss} > {limit}")]
    DailyLossExceeded { loss: String, limit: String },
}

/// Risk configuration.
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Enable risk checks.
    pub enabled: bool,
    /// Maximum order size.
    pub max_order_size: Decimal,
    /// Maximum position size per asset.
    pub max_position_size: Decimal,
    /// Per-asset position limits.
    pub position_limits: HashMap<String, Decimal>,
    /// Maximum open orders per symbol.
    pub max_open_orders: i32,
    /// Maximum daily loss.
    pub max_daily_loss: Decimal,
    /// Auto-activate kill switch on daily loss.
    pub kill_switch_enabled: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_order_size: Decimal::zero(),
            max_position_size: Decimal::zero(),
            position_limits: HashMap::new(),
            max_open_orders: 100,
            max_daily_loss: Decimal::zero(),
            kill_switch_enabled: false,
        }
    }
}

/// Thread-safe risk manager.
pub struct RiskManager {
    config: RiskConfig,
    kill_switch: AtomicBool,
    positions: RwLock<HashMap<String, Decimal>>,
    daily_pnl: RwLock<Decimal>,
    open_orders: RwLock<HashMap<String, i32>>,
}

impl RiskManager {
    /// Create a new risk manager.
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            kill_switch: AtomicBool::new(false),
            positions: RwLock::new(HashMap::new()),
            daily_pnl: RwLock::new(Decimal::zero()),
            open_orders: RwLock::new(HashMap::new()),
        }
    }

    /// Check if risk management is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get configuration.
    #[inline]
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }

    /// Check if kill switch is active.
    #[inline]
    pub fn is_killed(&self) -> bool {
        self.kill_switch.load(Ordering::Acquire)
    }

    /// Activate kill switch.
    pub fn kill(&self) {
        self.kill_switch.store(true, Ordering::Release);
    }

    /// Reset kill switch.
    pub fn reset(&self) {
        self.kill_switch.store(false, Ordering::Release);
    }

    /// Validate an order request.
    pub fn validate_order(&self, request: &OrderRequest) -> Result<(), RiskError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check kill switch
        if self.is_killed() {
            return Err(RiskError::KillSwitchActive);
        }

        // Check order size
        if self.config.max_order_size.is_positive()
            && request.quantity > self.config.max_order_size
        {
            return Err(RiskError::OrderSizeExceeded(
                request.quantity.to_string(),
                self.config.max_order_size.to_string(),
            ));
        }

        // Check position limit
        if let Some(pair) = TradingPair::from_symbol(&request.symbol) {
            let base = pair.base().to_string();

            let current = {
                let positions = self.positions.read().unwrap();
                positions.get(&base).copied().unwrap_or(Decimal::zero())
            };

            let new_position = match request.side {
                Side::Buy => current + request.quantity,
                Side::Sell => current - request.quantity,
            };

            // Asset-specific limit
            if let Some(&limit) = self.config.position_limits.get(&base) {
                if new_position.abs() > limit {
                    return Err(RiskError::PositionLimitExceeded {
                        asset: base,
                        current: current.to_string(),
                        order: request.quantity.to_string(),
                        limit: limit.to_string(),
                    });
                }
            }

            // Global position limit
            if self.config.max_position_size.is_positive()
                && new_position.abs() > self.config.max_position_size
            {
                return Err(RiskError::MaxPositionExceeded {
                    current: new_position.abs().to_string(),
                    limit: self.config.max_position_size.to_string(),
                });
            }
        }

        // Check open orders count
        {
            let orders = self.open_orders.read().unwrap();
            let count = orders.get(&request.symbol).copied().unwrap_or(0);
            if count >= self.config.max_open_orders {
                return Err(RiskError::MaxOpenOrdersReached {
                    symbol: request.symbol.clone(),
                    max: self.config.max_open_orders,
                });
            }
        }

        // Check daily loss
        if self.config.max_daily_loss.is_positive() {
            let pnl = self.daily_pnl.read().unwrap();
            if pnl.is_negative() && pnl.abs() > self.config.max_daily_loss {
                return Err(RiskError::DailyLossExceeded {
                    loss: pnl.abs().to_string(),
                    limit: self.config.max_daily_loss.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Update position for an asset.
    pub fn update_position(&self, asset: &str, quantity: Decimal, side: Side) {
        let mut positions = self.positions.write().unwrap();
        let pos = positions.entry(asset.to_string()).or_insert(Decimal::zero());
        *pos = match side {
            Side::Buy => *pos + quantity,
            Side::Sell => *pos - quantity,
        };
    }

    /// Get position for an asset.
    pub fn position(&self, asset: &str) -> Decimal {
        self.positions
            .read()
            .unwrap()
            .get(asset)
            .copied()
            .unwrap_or(Decimal::zero())
    }

    /// Get all positions.
    pub fn positions(&self) -> HashMap<String, Decimal> {
        self.positions.read().unwrap().clone()
    }

    /// Update daily PnL.
    pub fn update_pnl(&self, pnl: Decimal) {
        let mut daily = self.daily_pnl.write().unwrap();
        *daily = *daily + pnl;

        // Auto kill switch
        if self.config.kill_switch_enabled
            && self.config.max_daily_loss.is_positive()
            && daily.is_negative()
            && daily.abs() > self.config.max_daily_loss
        {
            self.kill_switch.store(true, Ordering::Release);
        }
    }

    /// Get daily PnL.
    pub fn daily_pnl(&self) -> Decimal {
        *self.daily_pnl.read().unwrap()
    }

    /// Reset daily PnL.
    pub fn reset_daily_pnl(&self) {
        let mut daily = self.daily_pnl.write().unwrap();
        *daily = Decimal::zero();
    }

    /// Record an order being opened.
    pub fn order_opened(&self, symbol: &str) {
        let mut orders = self.open_orders.write().unwrap();
        *orders.entry(symbol.to_string()).or_insert(0) += 1;
    }

    /// Record an order being closed.
    pub fn order_closed(&self, symbol: &str) {
        let mut orders = self.open_orders.write().unwrap();
        if let Some(count) = orders.get_mut(symbol) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }

    /// Get open order count for a symbol.
    pub fn open_orders(&self, symbol: &str) -> i32 {
        self.open_orders
            .read()
            .unwrap()
            .get(symbol)
            .copied()
            .unwrap_or(0)
    }

    /// Check order size (non-throwing).
    pub fn check_order_size(&self, quantity: Decimal) -> bool {
        if !self.config.max_order_size.is_positive() {
            return true;
        }
        quantity <= self.config.max_order_size
    }

    /// Check position limit (non-throwing).
    pub fn check_position_limit(&self, asset: &str, new_position: Decimal) -> bool {
        if let Some(&limit) = self.config.position_limits.get(asset) {
            if new_position.abs() > limit {
                return false;
            }
        }

        if self.config.max_position_size.is_positive()
            && new_position.abs() > self.config.max_position_size
        {
            return false;
        }

        true
    }

    /// Check daily loss (non-throwing).
    pub fn check_daily_loss(&self) -> bool {
        if !self.config.max_daily_loss.is_positive() {
            return true;
        }

        let pnl = self.daily_pnl.read().unwrap();
        !pnl.is_negative() || pnl.abs() <= self.config.max_daily_loss
    }

    /// Check open orders limit (non-throwing).
    pub fn check_open_orders(&self, symbol: &str) -> bool {
        let orders = self.open_orders.read().unwrap();
        let count = orders.get(symbol).copied().unwrap_or(0);
        count < self.config.max_open_orders
    }
}

/// RAII order tracker.
pub struct OrderTracker<'a> {
    risk_manager: &'a RiskManager,
    symbol: String,
    released: bool,
}

impl<'a> OrderTracker<'a> {
    /// Create a new order tracker.
    pub fn new(risk_manager: &'a RiskManager, symbol: impl Into<String>) -> Self {
        let symbol = symbol.into();
        risk_manager.order_opened(&symbol);
        Self {
            risk_manager,
            symbol,
            released: false,
        }
    }

    /// Release the tracker without decrementing count.
    pub fn release(&mut self) {
        self.released = true;
    }
}

impl<'a> Drop for OrderTracker<'a> {
    fn drop(&mut self) {
        if !self.released {
            self.risk_manager.order_closed(&self.symbol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_risk_manager_basic_validation() {
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_order_size = Decimal::from_f64(100.0);
        config.max_position_size = Decimal::from_f64(1000.0);
        config.max_open_orders = 10;

        let rm = RiskManager::new(config);

        // Valid order passes
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(50.0));
        assert!(rm.validate_order(&req).is_ok());

        // Order size exceeded
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(150.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::OrderSizeExceeded(..))
        ));
    }

    #[test]
    fn test_risk_manager_position_limits() {
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_position_size = Decimal::from_f64(100.0);
        config.position_limits.insert("BTC".to_string(), Decimal::from_f64(50.0));

        let rm = RiskManager::new(config);

        // Within global limit
        rm.update_position("BTC", Decimal::from_f64(30.0), Side::Buy);
        assert!(approx_eq(rm.position("BTC").to_f64(), 30.0, 0.0001));

        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(10.0));
        assert!(rm.validate_order(&req).is_ok());

        // Exceeds asset-specific limit
        rm.update_position("BTC", Decimal::from_f64(15.0), Side::Buy); // Now at 45
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(10.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::PositionLimitExceeded { .. })
        ));

        // Exceeds global limit
        rm.update_position("ETH", Decimal::from_f64(90.0), Side::Buy);
        let req = OrderRequest::market("ETH-USDC", Side::Buy, Decimal::from_f64(20.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::MaxPositionExceeded { .. })
        ));
    }

    #[test]
    fn test_risk_manager_pnl_tracking() {
        // Test PnL tracking without kill switch
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_daily_loss = Decimal::from_f64(1000.0);
        config.kill_switch_enabled = false; // Disable auto kill for this test

        let rm = RiskManager::new(config);

        // Track PnL
        rm.update_pnl(Decimal::from_f64(100.0));
        rm.update_pnl(Decimal::from_f64(-50.0));
        assert!(approx_eq(rm.daily_pnl().to_f64(), 50.0, 0.0001));

        // Daily loss limit triggers rejection
        rm.update_pnl(Decimal::from_f64(-1051.0)); // Now at -1001
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::DailyLossExceeded { .. })
        ));

        // Kill switch NOT activated (disabled)
        assert!(!rm.is_killed());
    }

    #[test]
    fn test_risk_manager_auto_kill_switch() {
        // Test that kill switch activates on daily loss when enabled
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_daily_loss = Decimal::from_f64(1000.0);
        config.kill_switch_enabled = true;

        let rm = RiskManager::new(config);

        assert!(!rm.is_killed());
        rm.update_pnl(Decimal::from_f64(-1001.0));

        // Kill switch should be activated
        assert!(rm.is_killed());

        // Now orders fail with KillSwitchActive
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::KillSwitchActive)
        ));
    }

    #[test]
    fn test_risk_manager_order_tracking() {
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_open_orders = 5;

        let rm = RiskManager::new(config);

        // Track open orders
        rm.order_opened("BTC-USDC");
        rm.order_opened("BTC-USDC");
        assert_eq!(rm.open_orders("BTC-USDC"), 2);

        rm.order_closed("BTC-USDC");
        assert_eq!(rm.open_orders("BTC-USDC"), 1);

        // Max open orders limit
        for _ in 0..5 {
            rm.order_opened("ETH-USDC");
        }

        let req = OrderRequest::market("ETH-USDC", Side::Buy, Decimal::from_f64(1.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::MaxOpenOrdersReached { .. })
        ));
    }

    #[test]
    fn test_risk_manager_kill_switch() {
        let mut config = RiskConfig::default();
        config.enabled = true;

        let rm = RiskManager::new(config);

        // Manual kill
        assert!(!rm.is_killed());
        rm.kill();
        assert!(rm.is_killed());

        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1.0));
        assert!(matches!(
            rm.validate_order(&req),
            Err(RiskError::KillSwitchActive)
        ));

        // Reset
        rm.reset();
        assert!(!rm.is_killed());
    }

    #[test]
    fn test_risk_manager_disabled() {
        let mut config = RiskConfig::default();
        config.enabled = false;
        config.max_order_size = Decimal::from_f64(10.0);

        let rm = RiskManager::new(config);

        // Validation passes when disabled
        let req = OrderRequest::market("BTC-USDC", Side::Buy, Decimal::from_f64(1000.0));
        assert!(rm.validate_order(&req).is_ok());
    }

    #[test]
    fn test_risk_manager_pre_trade_checks() {
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_order_size = Decimal::from_f64(100.0);
        config.max_position_size = Decimal::from_f64(500.0);
        config.max_daily_loss = Decimal::from_f64(1000.0);
        config.max_open_orders = 5;

        let rm = RiskManager::new(config);

        // Check order size
        assert!(rm.check_order_size(Decimal::from_f64(50.0)));
        assert!(!rm.check_order_size(Decimal::from_f64(150.0)));

        // Check position limit
        assert!(rm.check_position_limit("BTC", Decimal::from_f64(100.0)));
        assert!(!rm.check_position_limit("BTC", Decimal::from_f64(600.0)));

        // Check daily loss
        assert!(rm.check_daily_loss());
        rm.update_pnl(Decimal::from_f64(-1001.0));
        assert!(!rm.check_daily_loss());

        // Check open orders
        assert!(rm.check_open_orders("BTC-USDC"));
        for _ in 0..5 {
            rm.order_opened("BTC-USDC");
        }
        assert!(!rm.check_open_orders("BTC-USDC"));
    }

    #[test]
    fn test_order_tracker_raii() {
        let mut config = RiskConfig::default();
        config.enabled = true;
        config.max_open_orders = 10;

        let rm = RiskManager::new(config);

        // Auto decrement on destruction
        {
            let _tracker = OrderTracker::new(&rm, "BTC-USDC");
            assert_eq!(rm.open_orders("BTC-USDC"), 1);
        }
        assert_eq!(rm.open_orders("BTC-USDC"), 0);

        // Release prevents decrement
        {
            let mut tracker = OrderTracker::new(&rm, "BTC-USDC");
            tracker.release();
        }
        assert_eq!(rm.open_orders("BTC-USDC"), 1);
    }

    #[test]
    fn test_reset_daily_pnl() {
        let mut config = RiskConfig::default();
        config.enabled = true;

        let rm = RiskManager::new(config);

        rm.update_pnl(Decimal::from_f64(-500.0));
        rm.reset_daily_pnl();
        assert!(rm.daily_pnl().is_zero());
    }
}
