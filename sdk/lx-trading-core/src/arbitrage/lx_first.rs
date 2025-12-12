//! LX-First Arbitrage Strategy.
//!
//! Key Insight: LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks).
//! By the time other venues update, LX has already moved.
//!
//! This means:
//! 1. LX DEX price is the "TRUE" price (most current)
//! 2. Other venues are always STALE by comparison
//! 3. Arbitrage = correcting stale venues to match LX
//! 4. LX DEX is the ORACLE, not just another venue
//!
//! Strategy:
//! 1. Watch LX DEX prices (the reference)
//! 2. Compare against "slow" venues (CEX, external DEX)
//! 3. When slow venue diverges from LX, trade on SLOW venue
//! 4. You're essentially front-running slow venues with LX information
//!
//! Example:
//! - LX DEX BTC: $50,000 (current, true)
//! - Binance BTC: $49,990 (stale, 50ms behind)
//! - Uniswap BTC: $50,020 (stale, 12s behind)
//!
//! Action:
//! - Buy on Binance at $49,990 (they haven't caught up yet)
//! - Sell on Uniswap at $50,020 (they haven't corrected yet)
//! - Net: $30 profit per BTC
//!
//! Why LX wins: By the time Binance/Uniswap update, we've already executed.

use crate::arbitrage::types::*;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Callback type for opportunity events.
pub type LxFirstCallback = Box<dyn Fn(LxFirstOpportunity) + Send + Sync>;

/// LX-first arbitrage using LX DEX as the price oracle.
pub struct LxFirstArbitrage {
    config: LxFirstConfig,
    lx_prices: Arc<RwLock<HashMap<String, LxPrice>>>,
    venue_prices: Arc<RwLock<HashMap<String, Vec<VenuePrice>>>>,
    callbacks: Arc<RwLock<Vec<LxFirstCallback>>>,
    running: Arc<RwLock<bool>>,
}

impl LxFirstArbitrage {
    /// Create a new LX-first arbitrage instance.
    pub fn new(config: LxFirstConfig) -> Self {
        Self {
            config,
            lx_prices: Arc::new(RwLock::new(HashMap::new())),
            venue_prices: Arc::new(RwLock::new(HashMap::new())),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Update the LX DEX price (the oracle).
    pub async fn update_lx_price(&self, price: LxPrice) {
        let symbol = price.symbol.clone();
        {
            let mut prices = self.lx_prices.write().await;
            prices.insert(symbol.clone(), price);
        }

        // Immediately check for opportunities against stale venues
        self.check_opportunities(&symbol).await;
    }

    /// Update a price from a 'slow' venue.
    pub async fn update_venue_price(&self, price: VenuePrice) {
        let mut prices = self.venue_prices.write().await;
        let venue_prices = prices.entry(price.symbol.clone()).or_insert_with(Vec::new);

        // Update or append
        let mut found = false;
        for p in venue_prices.iter_mut() {
            if p.venue == price.venue {
                *p = price.clone();
                found = true;
                break;
            }
        }

        if !found {
            venue_prices.push(price);
        }
    }

    /// Subscribe to opportunity events.
    pub async fn on_opportunity(&self, callback: LxFirstCallback) {
        let mut callbacks = self.callbacks.write().await;
        callbacks.push(callback);
    }

    /// Start the arbitrage system.
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        *running = true;
    }

    /// Stop the arbitrage system.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Check for opportunities against stale venues.
    async fn check_opportunities(&self, symbol: &str) {
        {
            let running = self.running.read().await;
            if !*running {
                return;
            }
        }

        let lx_prices = self.lx_prices.read().await;
        let venue_prices = self.venue_prices.read().await;

        let lx_price = match lx_prices.get(symbol) {
            Some(p) => p,
            None => return,
        };

        let vps = match venue_prices.get(symbol) {
            Some(p) => p,
            None => return,
        };

        let now = current_time_ms();

        for vp in vps {
            // Calculate how stale the venue is
            let staleness = now - vp.timestamp;
            if staleness > self.config.max_staleness_ms {
                continue; // Too stale, might have updated by now
            }

            // Check for BUY opportunity (venue ask < LX mid)
            // The slow venue hasn't caught up to LX's higher price
            if vp.ask < lx_price.mid {
                let divergence = lx_price.mid - vp.ask;
                let divergence_bps = (divergence / lx_price.mid) * Decimal::from(10000);

                if divergence_bps >= self.config.min_divergence_bps {
                    let opp = self.create_opportunity(
                        symbol,
                        lx_price,
                        vp,
                        staleness,
                        "buy",
                        divergence,
                        divergence_bps,
                    );
                    if opp.expected_profit >= self.config.min_profit {
                        self.emit_opportunity(opp).await;
                    }
                }
            }

            // Check for SELL opportunity (venue bid > LX mid)
            // The slow venue hasn't caught up to LX's lower price
            if vp.bid > lx_price.mid {
                let divergence = vp.bid - lx_price.mid;
                let divergence_bps = (divergence / lx_price.mid) * Decimal::from(10000);

                if divergence_bps >= self.config.min_divergence_bps {
                    let opp = self.create_opportunity(
                        symbol,
                        lx_price,
                        vp,
                        staleness,
                        "sell",
                        divergence,
                        divergence_bps,
                    );
                    if opp.expected_profit >= self.config.min_profit {
                        self.emit_opportunity(opp).await;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_opportunity(
        &self,
        symbol: &str,
        lx_price: &LxPrice,
        vp: &VenuePrice,
        staleness: i64,
        side: &str,
        divergence: Decimal,
        divergence_bps: Decimal,
    ) -> LxFirstOpportunity {
        let now = current_time_ms();
        let expected_profit = divergence * self.config.max_position_size;
        let confidence = self.calculate_confidence(staleness, divergence_bps);

        LxFirstOpportunity {
            id: format!("{}-{}-{}-{}", symbol, vp.venue, side, now),
            symbol: symbol.to_string(),
            timestamp: now,
            lx_price: lx_price.clone(),
            stale_venue: vp.venue.clone(),
            stale_price: vp.clone(),
            staleness,
            side: side.to_string(),
            divergence,
            divergence_bps,
            expected_profit,
            max_size: self.config.max_position_size,
            confidence,
        }
    }

    fn calculate_confidence(&self, staleness: i64, divergence_bps: Decimal) -> f64 {
        // Higher confidence when:
        // 1. Venue is more stale (hasn't had time to update)
        // 2. Divergence is larger (more room for profit)
        let staleness_score = (1.0 - staleness as f64 / 5000.0).max(0.0); // 5s max
        let divergence_score = (divergence_bps / Decimal::from(100))
            .to_string()
            .parse::<f64>()
            .unwrap_or(0.0)
            .min(1.0); // 100bps = 1.0

        0.5 * staleness_score + 0.5 * divergence_score
    }

    async fn emit_opportunity(&self, opp: LxFirstOpportunity) {
        let callbacks = self.callbacks.read().await;
        for callback in callbacks.iter() {
            callback(opp.clone());
        }
    }

    /// Get current config.
    pub fn config(&self) -> &LxFirstConfig {
        &self.config
    }
}

fn current_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/*
TRADING EXECUTION STRATEGY

When an LxFirstOpportunity is detected:

1. DO NOT trade on LX DEX (it's the reference, not the opportunity)

2. Trade on the STALE venue:
   - If Side="buy": Buy on stale venue (their ask is behind LX)
   - If Side="sell": Sell on stale venue (their bid is behind LX)

3. Settlement options:
   a) Hold position until venues converge (market neutral)
   b) Immediately hedge on LX DEX (lock in profit)
   c) Bridge and sell on another venue (more complex)

4. The key insight:
   - You're NOT arbitraging between two venues
   - You're front-running the slow venue with LX information
   - LX price is where the slow venue WILL BE, you just got there first

Example execution:

  LX DEX shows BTC = $50,000 (current, true price)
  Binance shows BTC = $49,950 (50ms stale)

  Action: BUY on Binance at $49,950
  Why: Binance WILL update to ~$50,000, we bought before they did
  Profit: ~$50 per BTC (0.1%)

  Optional hedge: SELL on LX DEX at $50,000 to lock in profit immediately
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lx_first_creation() {
        let config = LxFirstConfig::default();
        let arb = LxFirstArbitrage::new(config);
        assert!(!*arb.running.read().await);
    }

    #[tokio::test]
    async fn test_update_lx_price() {
        let config = LxFirstConfig::default();
        let arb = LxFirstArbitrage::new(config);

        let price = LxPrice::new(
            "BTC-USDC".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            current_time_ms(),
            100,
        );

        arb.update_lx_price(price).await;

        let prices = arb.lx_prices.read().await;
        assert!(prices.contains_key("BTC-USDC"));
    }

    #[tokio::test]
    async fn test_update_venue_price() {
        let config = LxFirstConfig::default();
        let arb = LxFirstArbitrage::new(config);

        let price = VenuePrice {
            venue: "binance".to_string(),
            symbol: "BTC-USDC".to_string(),
            bid: Decimal::from(49990),
            ask: Decimal::from(50000),
            timestamp: current_time_ms(),
            latency: 50,
            stale: false,
        };

        arb.update_venue_price(price).await;

        let prices = arb.venue_prices.read().await;
        assert!(prices.contains_key("BTC-USDC"));
    }

    #[tokio::test]
    async fn test_detect_buy_opportunity() {
        let mut config = LxFirstConfig::default();
        config.min_divergence_bps = Decimal::from(5);
        config.min_profit = Decimal::from(1);

        let arb = LxFirstArbitrage::new(config);
        arb.start().await;

        // LX price is higher (the truth)
        let lx_price = LxPrice::new(
            "BTC-USDC".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            current_time_ms(),
            100,
        );

        // Binance is behind - their ask is lower than LX mid
        let venue_price = VenuePrice {
            venue: "binance".to_string(),
            symbol: "BTC-USDC".to_string(),
            bid: Decimal::from(49950),
            ask: Decimal::from(49960), // Lower than LX mid of 50005
            timestamp: current_time_ms(),
            latency: 50,
            stale: false,
        };

        arb.update_venue_price(venue_price).await;

        // Track opportunities
        let opportunities = Arc::new(RwLock::new(Vec::new()));
        let opps_clone = opportunities.clone();

        arb.on_opportunity(Box::new(move |opp| {
            let opps = opps_clone.clone();
            tokio::spawn(async move {
                let mut opps = opps.write().await;
                opps.push(opp);
            });
        }))
        .await;

        // Update LX price - should trigger opportunity check
        arb.update_lx_price(lx_price).await;

        // Give async tasks time to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let opps = opportunities.read().await;
        assert!(!opps.is_empty());
        assert_eq!(opps[0].side, "buy");
    }

    #[test]
    fn test_confidence_calculation() {
        let config = LxFirstConfig::default();
        let arb = LxFirstArbitrage::new(config);

        // High staleness (recent) + high divergence = high confidence
        let confidence = arb.calculate_confidence(1000, Decimal::from(50));
        assert!(confidence > 0.5);

        // Low staleness (old) + low divergence = low confidence
        let confidence = arb.calculate_confidence(4000, Decimal::from(5));
        assert!(confidence < 0.5);
    }
}
