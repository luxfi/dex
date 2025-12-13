//! Order book implementation with VWAP, slippage, and liquidity analysis.
//!
//! Provides both single-venue and aggregated multi-venue order books.

use crate::types::{Decimal, PriceLevel, Side};
use std::collections::HashMap;

/// Single venue order book.
#[derive(Debug, Clone)]
pub struct Orderbook {
    symbol: String,
    venue: String,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
    timestamp: i64,
    sequence: u64,
}

impl Orderbook {
    /// Create a new order book.
    pub fn new(symbol: impl Into<String>, venue: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            venue: venue.into(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: crate::types::now_ms(),
            sequence: 0,
        }
    }

    /// Get symbol.
    #[inline]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get venue.
    #[inline]
    pub fn venue(&self) -> &str {
        &self.venue
    }

    /// Get timestamp.
    #[inline]
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Get sequence number.
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Add a bid level.
    pub fn add_bid(&mut self, price: Decimal, quantity: Decimal) {
        self.bids.push(PriceLevel::new(price, quantity));
    }

    /// Add an ask level.
    pub fn add_ask(&mut self, price: Decimal, quantity: Decimal) {
        self.asks.push(PriceLevel::new(price, quantity));
    }

    /// Set/update a bid level.
    pub fn set_bid(&mut self, price: Decimal, quantity: Decimal) {
        if let Some(level) = self.bids.iter_mut().find(|l| l.price == price) {
            level.quantity = quantity;
        } else {
            self.bids.push(PriceLevel::new(price, quantity));
        }
    }

    /// Set/update an ask level.
    pub fn set_ask(&mut self, price: Decimal, quantity: Decimal) {
        if let Some(level) = self.asks.iter_mut().find(|l| l.price == price) {
            level.quantity = quantity;
        } else {
            self.asks.push(PriceLevel::new(price, quantity));
        }
    }

    /// Remove a bid level.
    pub fn remove_bid(&mut self, price: Decimal) {
        self.bids.retain(|l| l.price != price);
    }

    /// Remove an ask level.
    pub fn remove_ask(&mut self, price: Decimal) {
        self.asks.retain(|l| l.price != price);
    }

    /// Clear all levels.
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Sort bids (descending) and asks (ascending).
    pub fn sort(&mut self) {
        self.bids.sort_by(|a, b| b.price.cmp(&a.price));
        self.asks.sort_by(|a, b| a.price.cmp(&b.price));
        self.sequence += 1;
        self.timestamp = crate::types::now_ms();
    }

    /// Set timestamp.
    pub fn set_timestamp(&mut self, ts: i64) {
        self.timestamp = ts;
    }

    /// Set sequence.
    pub fn set_sequence(&mut self, seq: u64) {
        self.sequence = seq;
    }

    /// Get bids (sorted descending by price).
    #[inline]
    pub fn bids(&self) -> &[PriceLevel] {
        &self.bids
    }

    /// Get asks (sorted ascending by price).
    #[inline]
    pub fn asks(&self) -> &[PriceLevel] {
        &self.asks
    }

    /// Get best bid price.
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price.
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from_f64(2.0)),
            _ => None,
        }
    }

    /// Get bid-ask spread.
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread as percentage of mid price.
    pub fn spread_percent(&self) -> Option<Decimal> {
        match (self.spread(), self.mid_price()) {
            (Some(s), Some(m)) if !m.is_zero() => Some((s / m) * Decimal::from_f64(100.0)),
            _ => None,
        }
    }

    /// Total bid liquidity (value).
    pub fn bid_liquidity(&self) -> Decimal {
        self.bids.iter().fold(Decimal::zero(), |acc, l| acc + l.value())
    }

    /// Total ask liquidity (value).
    pub fn ask_liquidity(&self) -> Decimal {
        self.asks.iter().fold(Decimal::zero(), |acc, l| acc + l.value())
    }

    /// Bid depth for top N levels (value).
    pub fn bid_depth(&self, levels: usize) -> Decimal {
        self.bids
            .iter()
            .take(levels)
            .fold(Decimal::zero(), |acc, l| acc + l.value())
    }

    /// Ask depth for top N levels (value).
    pub fn ask_depth(&self, levels: usize) -> Decimal {
        self.asks
            .iter()
            .take(levels)
            .fold(Decimal::zero(), |acc, l| acc + l.value())
    }

    /// VWAP for buying `amount` quantity.
    pub fn vwap_buy(&self, amount: Decimal) -> Option<Decimal> {
        Self::calculate_vwap(&self.asks, amount)
    }

    /// VWAP for selling `amount` quantity.
    pub fn vwap_sell(&self, amount: Decimal) -> Option<Decimal> {
        Self::calculate_vwap(&self.bids, amount)
    }

    /// Calculate slippage for buying `amount` quantity.
    /// Returns (vwap, slippage_percent).
    pub fn slippage_buy(&self, amount: Decimal) -> Option<(Decimal, Decimal)> {
        let best = self.best_ask()?;
        let vwap = self.vwap_buy(amount)?;
        let slippage = ((vwap - best) / best) * Decimal::from_f64(100.0);
        Some((vwap, slippage))
    }

    /// Calculate slippage for selling `amount` quantity.
    /// Returns (vwap, slippage_percent).
    pub fn slippage_sell(&self, amount: Decimal) -> Option<(Decimal, Decimal)> {
        let best = self.best_bid()?;
        let vwap = self.vwap_sell(amount)?;
        let slippage = ((best - vwap) / best) * Decimal::from_f64(100.0);
        Some((vwap, slippage))
    }

    /// Check if sufficient liquidity exists.
    pub fn has_liquidity(&self, side: Side, amount: Decimal) -> bool {
        let levels = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };
        let total: Decimal = levels.iter().fold(Decimal::zero(), |acc, l| acc + l.quantity);
        total >= amount
    }

    /// Calculate VWAP across price levels.
    fn calculate_vwap(levels: &[PriceLevel], amount: Decimal) -> Option<Decimal> {
        let mut remaining = amount;
        let mut total_value = Decimal::zero();
        let mut total_qty = Decimal::zero();

        for level in levels {
            if remaining <= Decimal::zero() {
                break;
            }

            let fill_qty = remaining.min(level.quantity);
            total_value = total_value + (fill_qty * level.price);
            total_qty = total_qty + fill_qty;
            remaining = remaining - fill_qty;
        }

        if total_qty.is_zero() {
            None
        } else {
            Some(total_value / total_qty)
        }
    }
}

/// Aggregated order book from multiple venues.
#[derive(Debug, Clone)]
pub struct AggregatedOrderbook {
    symbol: String,
    timestamp: i64,
    // price -> [(venue, quantity), ...]
    bids: HashMap<i64, Vec<(String, Decimal)>>,
    asks: HashMap<i64, Vec<(String, Decimal)>>,
}

impl AggregatedOrderbook {
    /// Create a new aggregated order book.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            timestamp: 0,
            bids: HashMap::new(),
            asks: HashMap::new(),
        }
    }

    /// Get symbol.
    #[inline]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get timestamp.
    #[inline]
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Add an order book from a venue.
    pub fn add_orderbook(&mut self, book: &Orderbook) {
        for level in book.bids() {
            self.bids
                .entry(level.price.scaled_value())
                .or_default()
                .push((book.venue().to_string(), level.quantity));
        }

        for level in book.asks() {
            self.asks
                .entry(level.price.scaled_value())
                .or_default()
                .push((book.venue().to_string(), level.quantity));
        }

        self.timestamp = self.timestamp.max(book.timestamp());
    }

    /// Get best bid across all venues: (price, venue, quantity).
    pub fn best_bid(&self) -> Option<(Decimal, String, Decimal)> {
        self.bids
            .iter()
            .filter(|(_, venues)| !venues.is_empty())
            .max_by_key(|(price, _)| *price)
            .map(|(price, venues)| {
                let (venue, qty) = &venues[0];
                (Decimal::from_scaled(*price), venue.clone(), *qty)
            })
    }

    /// Get best ask across all venues: (price, venue, quantity).
    pub fn best_ask(&self) -> Option<(Decimal, String, Decimal)> {
        self.asks
            .iter()
            .filter(|(_, venues)| !venues.is_empty())
            .min_by_key(|(price, _)| *price)
            .map(|(price, venues)| {
                let (venue, qty) = &venues[0];
                (Decimal::from_scaled(*price), venue.clone(), *qty)
            })
    }

    /// Get aggregated bid levels (sorted descending by price).
    pub fn aggregated_bids(&self) -> Vec<PriceLevel> {
        let mut levels: Vec<_> = self
            .bids
            .iter()
            .map(|(price, venues)| {
                let total_qty = venues.iter().fold(Decimal::zero(), |acc, (_, qty)| acc + *qty);
                (Decimal::from_scaled(*price), total_qty)
            })
            .collect();

        levels.sort_by(|a, b| b.0.cmp(&a.0));
        levels
            .into_iter()
            .map(|(price, qty)| PriceLevel::new(price, qty))
            .collect()
    }

    /// Get aggregated ask levels (sorted ascending by price).
    pub fn aggregated_asks(&self) -> Vec<PriceLevel> {
        let mut levels: Vec<_> = self
            .asks
            .iter()
            .map(|(price, venues)| {
                let total_qty = venues.iter().fold(Decimal::zero(), |acc, (_, qty)| acc + *qty);
                (Decimal::from_scaled(*price), total_qty)
            })
            .collect();

        levels.sort_by(|a, b| a.0.cmp(&b.0));
        levels
            .into_iter()
            .map(|(price, qty)| PriceLevel::new(price, qty))
            .collect()
    }

    /// Find best venue for buying: (venue, price).
    pub fn best_venue_buy(&self, _amount: Decimal) -> Option<(String, Decimal)> {
        self.best_ask().map(|(price, venue, _)| (venue, price))
    }

    /// Find best venue for selling: (venue, price).
    pub fn best_venue_sell(&self, _amount: Decimal) -> Option<(String, Decimal)> {
        self.best_bid().map(|(price, venue, _)| (venue, price))
    }

    /// Clear all levels.
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.timestamp = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_orderbook_basic_operations() {
        let mut book = Orderbook::new("BTC-USDC", "test_venue");

        book.add_bid(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book.add_bid(Decimal::from_f64(99.0), Decimal::from_f64(2.0));
        book.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(1.5));
        book.add_ask(Decimal::from_f64(102.0), Decimal::from_f64(2.5));
        book.sort();

        assert_eq!(book.bids().len(), 2);
        assert_eq!(book.asks().len(), 2);

        // Bids sorted descending
        assert!(approx_eq(book.bids()[0].price.to_f64(), 100.0, 0.0001));
        assert!(approx_eq(book.bids()[1].price.to_f64(), 99.0, 0.0001));

        // Asks sorted ascending
        assert!(approx_eq(book.asks()[0].price.to_f64(), 101.0, 0.0001));
        assert!(approx_eq(book.asks()[1].price.to_f64(), 102.0, 0.0001));
    }

    #[test]
    fn test_orderbook_best_bid_ask() {
        let mut book = Orderbook::new("BTC-USDC", "test_venue");

        book.add_bid(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(1.0));
        book.sort();

        assert!(approx_eq(book.best_bid().unwrap().to_f64(), 100.0, 0.0001));
        assert!(approx_eq(book.best_ask().unwrap().to_f64(), 101.0, 0.0001));
    }

    #[test]
    fn test_orderbook_mid_price_and_spread() {
        let mut book = Orderbook::new("BTC-USDC", "test_venue");

        book.add_bid(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book.add_ask(Decimal::from_f64(102.0), Decimal::from_f64(1.0));
        book.sort();

        assert!(approx_eq(book.mid_price().unwrap().to_f64(), 101.0, 0.0001));
        assert!(approx_eq(book.spread().unwrap().to_f64(), 2.0, 0.0001));
        assert!(approx_eq(book.spread_percent().unwrap().to_f64(), 1.98, 0.1));
    }

    #[test]
    fn test_orderbook_vwap() {
        let mut book = Orderbook::new("BTC-USDC", "test_venue");

        book.add_ask(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(2.0));
        book.add_ask(Decimal::from_f64(102.0), Decimal::from_f64(3.0));
        book.sort();

        // VWAP for small amount
        let vwap = book.vwap_buy(Decimal::from_f64(0.5)).unwrap();
        assert!(approx_eq(vwap.to_f64(), 100.0, 0.0001));

        // VWAP across multiple levels: (1.0 * 100 + 1.5 * 101) / 2.5 = 100.6
        let vwap = book.vwap_buy(Decimal::from_f64(2.5)).unwrap();
        assert!(approx_eq(vwap.to_f64(), 100.6, 0.0001));

        // VWAP for full book: (1*100 + 2*101 + 3*102) / 6 = 101.333...
        let vwap = book.vwap_buy(Decimal::from_f64(6.0)).unwrap();
        assert!(approx_eq(vwap.to_f64(), 101.333, 0.01));
    }

    #[test]
    fn test_orderbook_liquidity() {
        let mut book = Orderbook::new("BTC-USDC", "test_venue");

        book.add_bid(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book.add_bid(Decimal::from_f64(99.0), Decimal::from_f64(2.0));
        book.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(1.5));
        book.add_ask(Decimal::from_f64(102.0), Decimal::from_f64(2.5));
        book.sort();

        // Bid liquidity: 1.0 * 100 + 2.0 * 99 = 298
        assert!(approx_eq(book.bid_liquidity().to_f64(), 298.0, 0.0001));

        // Ask liquidity: 1.5 * 101 + 2.5 * 102 = 406.5
        assert!(approx_eq(book.ask_liquidity().to_f64(), 406.5, 0.0001));

        // Depth for specific levels
        assert!(approx_eq(book.bid_depth(1).to_f64(), 100.0, 0.0001));
        assert!(approx_eq(book.ask_depth(1).to_f64(), 151.5, 0.0001));

        // Has liquidity
        assert!(book.has_liquidity(Side::Buy, Decimal::from_f64(3.0)));
        assert!(!book.has_liquidity(Side::Buy, Decimal::from_f64(10.0)));
    }

    #[test]
    fn test_aggregated_orderbook() {
        let mut agg = AggregatedOrderbook::new("BTC-USDC");

        // First venue
        let mut book1 = Orderbook::new("BTC-USDC", "venue1");
        book1.add_bid(Decimal::from_f64(100.0), Decimal::from_f64(1.0));
        book1.add_ask(Decimal::from_f64(102.0), Decimal::from_f64(1.0));
        book1.sort();

        // Second venue
        let mut book2 = Orderbook::new("BTC-USDC", "venue2");
        book2.add_bid(Decimal::from_f64(99.0), Decimal::from_f64(2.0));
        book2.add_ask(Decimal::from_f64(101.0), Decimal::from_f64(1.5));
        book2.sort();

        agg.add_orderbook(&book1);
        agg.add_orderbook(&book2);

        // Best bid across venues
        let (price, venue, _qty) = agg.best_bid().unwrap();
        assert!(approx_eq(price.to_f64(), 100.0, 0.0001));
        assert_eq!(venue, "venue1");

        // Best ask across venues
        let (price, venue, _qty) = agg.best_ask().unwrap();
        assert!(approx_eq(price.to_f64(), 101.0, 0.0001));
        assert_eq!(venue, "venue2");

        // Aggregated levels
        let asks = agg.aggregated_asks();
        assert_eq!(asks.len(), 2);
        assert!(approx_eq(asks[0].price.to_f64(), 101.0, 0.0001));
        assert!(approx_eq(asks[1].price.to_f64(), 102.0, 0.0001));

        // Best venue for buying
        let (venue, price) = agg.best_venue_buy(Decimal::from_f64(1.0)).unwrap();
        assert_eq!(venue, "venue2");
        assert!(approx_eq(price.to_f64(), 101.0, 0.0001));

        // Best venue for selling
        let (venue, price) = agg.best_venue_sell(Decimal::from_f64(0.5)).unwrap();
        assert_eq!(venue, "venue1");
        assert!(approx_eq(price.to_f64(), 100.0, 0.0001));
    }
}
