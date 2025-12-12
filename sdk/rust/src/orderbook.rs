//! Order book types and operations.
//!
//! Provides order book snapshots and update handling.

use crate::types::PriceLevel;
use serde::{Deserialize, Serialize};

/// Order book snapshot for a trading pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol.
    pub symbol: String,
    /// Bid levels (descending by price).
    pub bids: Vec<PriceLevel>,
    /// Ask levels (ascending by price).
    pub asks: Vec<PriceLevel>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl OrderBook {
    /// Create a new empty order book.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: 0,
        }
    }

    /// Best bid price, or None if no bids.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Best ask price, or None if no asks.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Bid-ask spread, or None if either side is empty.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Mid price, or None if either side is empty.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Spread as a percentage of mid price.
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some((spread / mid) * 100.0),
            _ => None,
        }
    }

    /// Total bid liquidity up to depth levels.
    pub fn bid_liquidity(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.price * l.size).sum()
    }

    /// Total ask liquidity up to depth levels.
    pub fn ask_liquidity(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.price * l.size).sum()
    }

    /// Liquidity imbalance (-1.0 to 1.0).
    /// Positive means more bids, negative means more asks.
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_liq = self.bid_liquidity(depth);
        let ask_liq = self.ask_liquidity(depth);
        let total = bid_liq + ask_liq;
        if total > 0.0 {
            (bid_liq - ask_liq) / total
        } else {
            0.0
        }
    }

    /// Estimate slippage for a market buy of given size.
    /// Returns average fill price, or None if insufficient liquidity.
    pub fn estimate_buy_slippage(&self, size: f64) -> Option<f64> {
        let mut remaining = size;
        let mut total_cost = 0.0;

        for level in &self.asks {
            if remaining <= 0.0 {
                break;
            }
            let fill = remaining.min(level.size);
            total_cost += fill * level.price;
            remaining -= fill;
        }

        if remaining > 0.0 {
            None // Insufficient liquidity
        } else {
            Some(total_cost / size)
        }
    }

    /// Estimate slippage for a market sell of given size.
    /// Returns average fill price, or None if insufficient liquidity.
    pub fn estimate_sell_slippage(&self, size: f64) -> Option<f64> {
        let mut remaining = size;
        let mut total_proceeds = 0.0;

        for level in &self.bids {
            if remaining <= 0.0 {
                break;
            }
            let fill = remaining.min(level.size);
            total_proceeds += fill * level.price;
            remaining -= fill;
        }

        if remaining > 0.0 {
            None // Insufficient liquidity
        } else {
            Some(total_proceeds / size)
        }
    }

    /// Apply an incremental update to the order book.
    pub fn apply_update(&mut self, update: &OrderBookUpdate) {
        for level in &update.bids {
            Self::update_side(&mut self.bids, level, true);
        }
        for level in &update.asks {
            Self::update_side(&mut self.asks, level, false);
        }
        self.timestamp = update.timestamp;
    }

    fn update_side(levels: &mut Vec<PriceLevel>, update: &PriceLevel, is_bid: bool) {
        // Find position for this price
        let pos = levels.iter().position(|l| {
            if is_bid {
                l.price <= update.price
            } else {
                l.price >= update.price
            }
        });

        match pos {
            Some(i) if levels[i].price == update.price => {
                if update.size == 0.0 {
                    levels.remove(i);
                } else {
                    levels[i] = *update;
                }
            }
            Some(i) if update.size > 0.0 => {
                levels.insert(i, *update);
            }
            None if update.size > 0.0 => {
                levels.push(*update);
            }
            _ => {}
        }
    }
}

/// Incremental order book update.
#[derive(Debug, Clone, Deserialize)]
pub struct OrderBookUpdate {
    pub symbol: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: i64,
}

/// Order book subscription channel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookDepth {
    /// Top 5 levels.
    Top5,
    /// Top 10 levels.
    Top10,
    /// Top 20 levels.
    Top20,
    /// Full book.
    Full,
}

impl BookDepth {
    pub fn as_str(self) -> &'static str {
        match self {
            BookDepth::Top5 => "5",
            BookDepth::Top10 => "10",
            BookDepth::Top20 => "20",
            BookDepth::Full => "full",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_book() -> OrderBook {
        OrderBook {
            symbol: "BTC-USDT".into(),
            bids: vec![
                PriceLevel::new(50000.0, 1.0),
                PriceLevel::new(49900.0, 2.0),
                PriceLevel::new(49800.0, 3.0),
            ],
            asks: vec![
                PriceLevel::new(50100.0, 1.0),
                PriceLevel::new(50200.0, 2.0),
                PriceLevel::new(50300.0, 3.0),
            ],
            timestamp: 1000,
        }
    }

    #[test]
    fn test_best_prices() {
        let book = sample_book();
        assert_eq!(book.best_bid(), Some(50000.0));
        assert_eq!(book.best_ask(), Some(50100.0));
    }

    #[test]
    fn test_spread() {
        let book = sample_book();
        assert!((book.spread().unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_mid_price() {
        let book = sample_book();
        assert!((book.mid_price().unwrap() - 50050.0).abs() < 0.001);
    }

    #[test]
    fn test_spread_percent() {
        let book = sample_book();
        let pct = book.spread_percent().unwrap();
        // 100 / 50050 * 100 = ~0.1998%
        assert!(pct > 0.19 && pct < 0.21);
    }

    #[test]
    fn test_liquidity() {
        let book = sample_book();
        // First level: 50000 * 1.0 = 50000
        assert!((book.bid_liquidity(1) - 50000.0).abs() < 0.001);
        // Two levels: 50000 + 49900*2 = 149800
        assert!((book.bid_liquidity(2) - 149800.0).abs() < 0.001);
    }

    #[test]
    fn test_buy_slippage() {
        let book = sample_book();
        // Buy 1.0 at first ask level = 50100
        let avg = book.estimate_buy_slippage(1.0).unwrap();
        assert!((avg - 50100.0).abs() < 0.001);

        // Buy 2.0: 1.0*50100 + 1.0*50200 = 100300 / 2 = 50150
        let avg = book.estimate_buy_slippage(2.0).unwrap();
        assert!((avg - 50150.0).abs() < 0.001);
    }

    #[test]
    fn test_sell_slippage() {
        let book = sample_book();
        // Sell 1.0 at first bid = 50000
        let avg = book.estimate_sell_slippage(1.0).unwrap();
        assert!((avg - 50000.0).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_liquidity() {
        let book = sample_book();
        // Total ask liquidity is 6.0, try to buy 10.0
        assert!(book.estimate_buy_slippage(10.0).is_none());
    }

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new("BTC-USDT");
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert!(book.spread().is_none());
        assert!(book.mid_price().is_none());
    }
}
