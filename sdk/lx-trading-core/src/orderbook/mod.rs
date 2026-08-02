//! Orderbook implementation with aggregation support.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::types::PriceLevel;

/// Orderbook with bids and asks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orderbook {
    pub symbol: String,
    pub venue: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: i64,
    pub sequence: u64,
}

impl Orderbook {
    pub fn new(symbol: &str, venue: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            venue: venue.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            sequence: 0,
        }
    }

    pub fn add_bid(&mut self, price: Decimal, quantity: Decimal) {
        self.bids.push(PriceLevel::new(price, quantity));
    }

    pub fn add_ask(&mut self, price: Decimal, quantity: Decimal) {
        self.asks.push(PriceLevel::new(price, quantity));
    }

    /// Sort bids descending (best bid first), asks ascending (best ask first)
    pub fn sort(&mut self) {
        self.bids.sort_by_key(|l| std::cmp::Reverse(l.price));
        self.asks.sort_by_key(|l| l.price);
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread as percentage of mid price
    pub fn spread_percent(&self) -> Option<Decimal> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if !mid.is_zero() => {
                Some((spread / mid) * Decimal::from(100))
            }
            _ => None,
        }
    }

    /// Get total bid liquidity
    pub fn bid_liquidity(&self) -> Decimal {
        self.bids.iter().map(|l| l.value()).sum()
    }

    /// Get total ask liquidity
    pub fn ask_liquidity(&self) -> Decimal {
        self.asks.iter().map(|l| l.value()).sum()
    }

    /// Get bid depth at price levels
    pub fn bid_depth(&self, levels: usize) -> Decimal {
        self.bids.iter().take(levels).map(|l| l.value()).sum()
    }

    /// Get ask depth at price levels
    pub fn ask_depth(&self, levels: usize) -> Decimal {
        self.asks.iter().take(levels).map(|l| l.value()).sum()
    }

    /// Calculate volume-weighted average price for buying `amount`
    pub fn vwap_buy(&self, amount: Decimal) -> Option<Decimal> {
        self.calculate_vwap(&self.asks, amount)
    }

    /// Calculate volume-weighted average price for selling `amount`
    pub fn vwap_sell(&self, amount: Decimal) -> Option<Decimal> {
        self.calculate_vwap(&self.bids, amount)
    }

    fn calculate_vwap(&self, levels: &[PriceLevel], amount: Decimal) -> Option<Decimal> {
        let mut remaining = amount;
        let mut total_value = Decimal::ZERO;
        let mut total_quantity = Decimal::ZERO;

        for level in levels {
            if remaining <= Decimal::ZERO {
                break;
            }

            let fill_qty = remaining.min(level.quantity);
            total_value += fill_qty * level.price;
            total_quantity += fill_qty;
            remaining -= fill_qty;
        }

        if total_quantity.is_zero() {
            None
        } else {
            Some(total_value / total_quantity)
        }
    }

    /// Check if there's enough liquidity to fill an order
    pub fn has_liquidity(&self, side: crate::types::Side, amount: Decimal) -> bool {
        let levels = match side {
            crate::types::Side::Buy => &self.asks,
            crate::types::Side::Sell => &self.bids,
        };

        let total: Decimal = levels.iter().map(|l| l.quantity).sum();
        total >= amount
    }
}

/// Aggregated orderbook from multiple venues
#[derive(Debug, Clone)]
pub struct AggregatedOrderbook {
    pub symbol: String,
    pub bids: BTreeMap<Decimal, Vec<(String, Decimal)>>, // price -> [(venue, quantity)]
    pub asks: BTreeMap<Decimal, Vec<(String, Decimal)>>,
    pub timestamp: i64,
}

impl AggregatedOrderbook {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Add orderbook from a venue
    pub fn add_orderbook(&mut self, book: &Orderbook) {
        for level in &book.bids {
            self.bids
                .entry(level.price)
                .or_default()
                .push((book.venue.clone(), level.quantity));
        }

        for level in &book.asks {
            self.asks
                .entry(level.price)
                .or_default()
                .push((book.venue.clone(), level.quantity));
        }

        self.timestamp = self.timestamp.max(book.timestamp);
    }

    /// Get best bid across all venues
    pub fn best_bid(&self) -> Option<(Decimal, &str, Decimal)> {
        self.bids.iter().next_back().and_then(|(price, venues)| {
            venues
                .first()
                .map(|(venue, qty)| (*price, venue.as_str(), *qty))
        })
    }

    /// Get best ask across all venues
    pub fn best_ask(&self) -> Option<(Decimal, &str, Decimal)> {
        self.asks.iter().next().and_then(|(price, venues)| {
            venues
                .first()
                .map(|(venue, qty)| (*price, venue.as_str(), *qty))
        })
    }

    /// Get total liquidity at each price level
    pub fn aggregated_bids(&self) -> Vec<PriceLevel> {
        self.bids
            .iter()
            .rev()
            .map(|(price, venues)| {
                let total_qty: Decimal = venues.iter().map(|(_, q)| q).sum();
                PriceLevel::new(*price, total_qty)
            })
            .collect()
    }

    /// Get total liquidity at each price level
    pub fn aggregated_asks(&self) -> Vec<PriceLevel> {
        self.asks
            .iter()
            .map(|(price, venues)| {
                let total_qty: Decimal = venues.iter().map(|(_, q)| q).sum();
                PriceLevel::new(*price, total_qty)
            })
            .collect()
    }

    /// Find best venue for buying `amount`
    pub fn best_venue_buy(&self, amount: Decimal) -> Option<(String, Decimal)> {
        let mut best_venue = None;
        let mut best_price = Decimal::MAX;
        let mut remaining = amount;

        for (price, venues) in &self.asks {
            if remaining <= Decimal::ZERO {
                break;
            }

            for (venue, qty) in venues {
                let fill = remaining.min(*qty);
                if *price < best_price {
                    best_price = *price;
                    best_venue = Some(venue.clone());
                }
                remaining -= fill;
                if remaining <= Decimal::ZERO {
                    break;
                }
            }
        }

        best_venue.map(|v| (v, best_price))
    }

    /// Find best venue for selling `amount`
    pub fn best_venue_sell(&self, amount: Decimal) -> Option<(String, Decimal)> {
        let mut best_venue = None;
        let mut best_price = Decimal::ZERO;
        let mut remaining = amount;

        for (price, venues) in self.bids.iter().rev() {
            if remaining <= Decimal::ZERO {
                break;
            }

            for (venue, qty) in venues {
                let fill = remaining.min(*qty);
                if *price > best_price {
                    best_price = *price;
                    best_venue = Some(venue.clone());
                }
                remaining -= fill;
                if remaining <= Decimal::ZERO {
                    break;
                }
            }
        }

        best_venue.map(|v| (v, best_price))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_basics() {
        let mut book = Orderbook::new("BTC-USDC", "test");

        book.add_bid(Decimal::from(50000), Decimal::from(1));
        book.add_bid(Decimal::from(49900), Decimal::from(2));
        book.add_ask(Decimal::from(50100), Decimal::from(1));
        book.add_ask(Decimal::from(50200), Decimal::from(2));

        book.sort();

        assert_eq!(book.best_bid(), Some(Decimal::from(50000)));
        assert_eq!(book.best_ask(), Some(Decimal::from(50100)));
        assert_eq!(book.spread(), Some(Decimal::from(100)));
    }

    #[test]
    fn test_aggregated_orderbook() {
        let mut agg = AggregatedOrderbook::new("BTC-USDC");

        let mut book1 = Orderbook::new("BTC-USDC", "venue1");
        book1.add_bid(Decimal::from(50000), Decimal::from(1));
        book1.add_ask(Decimal::from(50100), Decimal::from(1));

        let mut book2 = Orderbook::new("BTC-USDC", "venue2");
        book2.add_bid(Decimal::from(50050), Decimal::from(2));
        book2.add_ask(Decimal::from(50080), Decimal::from(2));

        agg.add_orderbook(&book1);
        agg.add_orderbook(&book2);

        // Best bid should be from venue2 at 50050
        let (price, venue, _) = agg.best_bid().unwrap();
        assert_eq!(price, Decimal::from(50050));
        assert_eq!(venue, "venue2");

        // Best ask should be from venue2 at 50080
        let (price, venue, _) = agg.best_ask().unwrap();
        assert_eq!(price, Decimal::from(50080));
        assert_eq!(venue, "venue2");
    }
}
