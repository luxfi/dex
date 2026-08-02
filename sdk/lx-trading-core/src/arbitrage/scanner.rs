//! Arbitrage scanner for detecting cross-venue opportunities.
//!
//! Continuously scans for arbitrage opportunities across all venues.
//! Supports simple, triangular, and CEX-DEX arbitrage detection.

use crate::arbitrage::types::*;
use rust_decimal::Decimal;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;

/// Known CEX venues.
const CEX_VENUES: &[&str] = &[
    "binance", "coinbase", "kraken", "okx", "bybit", "kucoin", "mexc", "gate", "huobi",
];

/// Callback type for opportunity events.
pub type OpportunityCallback = Box<dyn Fn(ArbitrageOpportunity) + Send + Sync>;

/// Arbitrage scanner for detecting cross-venue opportunities.
pub struct Scanner {
    config: ScannerConfig,
    prices: Arc<RwLock<HashMap<String, Vec<PriceSource>>>>,
    chains: Arc<RwLock<HashMap<String, CrossChainInfo>>>,
    callbacks: Arc<RwLock<Vec<OpportunityCallback>>>,
    running: Arc<RwLock<bool>>,
    task: Option<JoinHandle<()>>,
    tx: Option<mpsc::Sender<ArbitrageOpportunity>>,
}

impl Scanner {
    /// Create a new scanner with the given configuration.
    pub fn new(config: ScannerConfig) -> Self {
        Self {
            config,
            prices: Arc::new(RwLock::new(HashMap::new())),
            chains: Arc::new(RwLock::new(HashMap::new())),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(RwLock::new(false)),
            task: None,
            tx: None,
        }
    }

    /// Add a chain configuration.
    pub async fn add_chain(&self, info: CrossChainInfo) {
        let mut chains = self.chains.write().await;
        chains.insert(info.chain_id.clone(), info);
    }

    /// Update a price feed.
    pub async fn update_price(&self, source: PriceSource) {
        let mut prices = self.prices.write().await;
        let sources = prices.entry(source.symbol.clone()).or_insert_with(Vec::new);

        // Update existing or append new
        let mut found = false;
        for s in sources.iter_mut() {
            if s.chain_id == source.chain_id && s.venue == source.venue {
                *s = source.clone();
                found = true;
                break;
            }
        }

        if !found {
            sources.push(source);
        }
    }

    /// Subscribe to opportunity events.
    pub async fn on_opportunity(&self, callback: OpportunityCallback) {
        let mut callbacks = self.callbacks.write().await;
        callbacks.push(callback);
    }

    /// Subscribe to opportunity channel.
    pub fn subscribe(&mut self) -> mpsc::Receiver<ArbitrageOpportunity> {
        let (tx, rx) = mpsc::channel(1000);
        self.tx = Some(tx);
        rx
    }

    /// Start scanning for opportunities.
    pub async fn start(&mut self) {
        {
            let running = self.running.read().await;
            if *running {
                return;
            }
        }

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let config = self.config.clone();
        let prices = self.prices.clone();
        let chains = self.chains.clone();
        let callbacks = self.callbacks.clone();
        let running = self.running.clone();
        let tx = self.tx.clone();

        let task = tokio::spawn(async move {
            scan_loop(config, prices, chains, callbacks, running, tx).await;
        });

        self.task = Some(task);
    }

    /// Stop scanning.
    pub async fn stop(&mut self) {
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }

    /// Get current config.
    pub fn config(&self) -> &ScannerConfig {
        &self.config
    }
}

async fn scan_loop(
    config: ScannerConfig,
    prices: Arc<RwLock<HashMap<String, Vec<PriceSource>>>>,
    chains: Arc<RwLock<HashMap<String, CrossChainInfo>>>,
    callbacks: Arc<RwLock<Vec<OpportunityCallback>>>,
    running: Arc<RwLock<bool>>,
    tx: Option<mpsc::Sender<ArbitrageOpportunity>>,
) {
    let interval = std::time::Duration::from_millis(config.scan_interval_ms);

    loop {
        {
            let is_running = running.read().await;
            if !*is_running {
                break;
            }
        }

        // Perform scan
        let opps = scan(&config, &prices, &chains).await;

        // Emit opportunities
        for opp in opps {
            // Send to callbacks
            {
                let cbs = callbacks.read().await;
                for cb in cbs.iter() {
                    cb(opp.clone());
                }
            }

            // Send to channel
            if let Some(ref tx) = tx {
                let _ = tx.send(opp).await;
            }
        }

        tokio::time::sleep(interval).await;
    }
}

async fn scan(
    config: &ScannerConfig,
    prices: &Arc<RwLock<HashMap<String, Vec<PriceSource>>>>,
    chains: &Arc<RwLock<HashMap<String, CrossChainInfo>>>,
) -> Vec<ArbitrageOpportunity> {
    let prices_guard = prices.read().await;
    let chains_guard = chains.read().await;

    let mut opportunities = Vec::new();

    for (symbol, sources) in prices_guard.iter() {
        if sources.len() < 2 {
            continue;
        }

        let now = current_time_ms();

        // Filter stale prices
        let valid_sources: Vec<_> = sources
            .iter()
            .filter(|s| now - s.timestamp < config.max_price_age_ms)
            .cloned()
            .collect();

        if valid_sources.len() < 2 {
            continue;
        }

        // Simple arbitrage
        opportunities.extend(find_simple_arb(
            symbol,
            &valid_sources,
            config,
            &chains_guard,
        ));

        // CEX-DEX arbitrage
        opportunities.extend(find_cex_dex_arb(symbol, &valid_sources, config));
    }

    opportunities
}

fn find_simple_arb(
    symbol: &str,
    sources: &[PriceSource],
    config: &ScannerConfig,
    chains: &HashMap<String, CrossChainInfo>,
) -> Vec<ArbitrageOpportunity> {
    let mut opportunities = Vec::new();

    // Sort by ask (lowest first for buying)
    let mut buy_order: Vec<_> = sources.to_vec();
    buy_order.sort_by_key(|s| s.ask);

    // Sort by bid (highest first for selling)
    let mut sell_order: Vec<_> = sources.to_vec();
    sell_order.sort_by_key(|s| std::cmp::Reverse(s.bid));

    for buy_src in &buy_order {
        for sell_src in &sell_order {
            // Skip same venue/chain
            if buy_src.chain_id == sell_src.chain_id && buy_src.venue == sell_src.venue {
                continue;
            }

            // Calculate spread
            let spread = sell_src.bid - buy_src.ask;
            if spread <= Decimal::ZERO {
                continue;
            }

            let spread_bps = (spread / buy_src.ask) * Decimal::from(10000);
            if spread_bps < config.min_spread_bps {
                continue;
            }

            // Calculate costs
            let (gas_cost, bridge_cost) =
                calculate_costs(&buy_src.chain_id, &sell_src.chain_id, chains);

            // Maximum size limited by liquidity
            let max_size = buy_src.liquidity.min(sell_src.liquidity);

            // Calculate PnL
            let gross_pnl = spread * max_size;
            let net_pnl = gross_pnl - gas_cost - bridge_cost;

            if net_pnl < config.min_profit_usd {
                continue;
            }

            // Calculate confidence
            let confidence = calculate_confidence(buy_src, sell_src, config.max_price_age_ms);

            let now = current_time_ms();
            let opp = ArbitrageOpportunity {
                id: format!(
                    "simple-{}-{}-{}-{}",
                    symbol, buy_src.venue, sell_src.venue, now
                ),
                arb_type: ArbType::Simple,
                buy_source: buy_src.clone(),
                sell_source: sell_src.clone(),
                spread_bps,
                estimated_pnl: gross_pnl,
                max_size,
                gas_cost_usd: gas_cost,
                bridge_cost_usd: bridge_cost,
                net_pnl,
                confidence,
                expires_at: now + 5000,
                routes: vec![
                    Route {
                        chain_id: buy_src.chain_id.clone(),
                        venue: buy_src.venue.clone(),
                        action: "buy".to_string(),
                        token_in: "USDC".to_string(),
                        token_out: symbol.to_string(),
                        amount_in: max_size * buy_src.ask,
                        expected_out: max_size,
                        min_amount_out: max_size * Decimal::new(99, 2), // 0.99
                        swap_data: None,
                    },
                    Route {
                        chain_id: sell_src.chain_id.clone(),
                        venue: sell_src.venue.clone(),
                        action: "sell".to_string(),
                        token_in: symbol.to_string(),
                        token_out: "USDC".to_string(),
                        amount_in: max_size,
                        expected_out: max_size * sell_src.bid,
                        min_amount_out: max_size * sell_src.bid * Decimal::new(99, 2),
                        swap_data: None,
                    },
                ],
            };
            opportunities.push(opp);
        }
    }

    opportunities
}

fn find_cex_dex_arb(
    symbol: &str,
    sources: &[PriceSource],
    config: &ScannerConfig,
) -> Vec<ArbitrageOpportunity> {
    let cex_set: HashSet<&str> = CEX_VENUES.iter().copied().collect();
    let mut opportunities = Vec::new();

    // Separate CEX and DEX sources
    let cex_sources: Vec<_> = sources
        .iter()
        .filter(|s| cex_set.contains(s.venue.as_str()))
        .cloned()
        .collect();
    let dex_sources: Vec<_> = sources
        .iter()
        .filter(|s| !cex_set.contains(s.venue.as_str()))
        .cloned()
        .collect();

    // CEX buy -> DEX sell
    for cex in &cex_sources {
        for dex in &dex_sources {
            let spread = dex.bid - cex.ask;
            if spread <= Decimal::ZERO {
                continue;
            }

            let spread_bps = (spread / cex.ask) * Decimal::from(10000);
            if spread_bps < config.min_spread_bps {
                continue;
            }

            let max_size = cex.liquidity.min(dex.liquidity);
            let gross_pnl = spread * max_size;

            let now = current_time_ms();
            let opp = ArbitrageOpportunity {
                id: format!("cexdex-{}-{}-{}-{}", symbol, cex.venue, dex.venue, now),
                arb_type: ArbType::CexDex,
                buy_source: cex.clone(),
                sell_source: dex.clone(),
                spread_bps,
                estimated_pnl: gross_pnl,
                max_size,
                gas_cost_usd: Decimal::new(5, 1), // 0.5
                bridge_cost_usd: Decimal::ZERO,
                net_pnl: gross_pnl - Decimal::new(5, 1),
                confidence: 0.7,
                expires_at: now + 3000,
                routes: Vec::new(),
            };
            opportunities.push(opp);
        }
    }

    // DEX buy -> CEX sell
    for dex in &dex_sources {
        for cex in &cex_sources {
            let spread = cex.bid - dex.ask;
            if spread <= Decimal::ZERO {
                continue;
            }

            let spread_bps = (spread / dex.ask) * Decimal::from(10000);
            if spread_bps < config.min_spread_bps {
                continue;
            }

            let max_size = dex.liquidity.min(cex.liquidity);
            let gross_pnl = spread * max_size;

            let now = current_time_ms();
            let opp = ArbitrageOpportunity {
                id: format!("cexdex-{}-{}-{}-{}", symbol, dex.venue, cex.venue, now),
                arb_type: ArbType::CexDex,
                buy_source: dex.clone(),
                sell_source: cex.clone(),
                spread_bps,
                estimated_pnl: gross_pnl,
                max_size,
                gas_cost_usd: Decimal::new(5, 1),
                bridge_cost_usd: Decimal::ZERO,
                net_pnl: gross_pnl - Decimal::new(5, 1),
                confidence: 0.7,
                expires_at: now + 3000,
                routes: Vec::new(),
            };
            opportunities.push(opp);
        }
    }

    opportunities
}

fn calculate_costs(
    source_chain: &str,
    dest_chain: &str,
    chains: &HashMap<String, CrossChainInfo>,
) -> (Decimal, Decimal) {
    let src_config = chains.get(source_chain);
    let dst_config = chains.get(dest_chain);

    // Estimate gas cost
    let gas_cost = if src_config.is_some() {
        Decimal::new(5, 2) // 0.05
    } else {
        Decimal::new(1, 1) // 0.1
    };

    // Bridge cost if crossing chains
    let bridge_cost = if source_chain != dest_chain {
        match (src_config, dst_config) {
            (Some(src), Some(dst)) if src.warp_supported && dst.warp_supported => {
                Decimal::new(1, 2) // 0.01 - Warp is nearly free
            }
            (Some(src), Some(dst)) if src.teleport_supported && dst.teleport_supported => {
                Decimal::new(1, 1) // 0.10 - Teleport for EVM
            }
            _ => Decimal::from(1), // Generic bridge
        }
    } else {
        Decimal::ZERO
    };

    (gas_cost, bridge_cost)
}

fn calculate_confidence(buy: &PriceSource, sell: &PriceSource, max_price_age_ms: i64) -> f64 {
    let now = current_time_ms();
    let max_age = max_price_age_ms as f64 / 1000.0;

    // Freshness score
    let buy_age = (now - buy.timestamp) as f64 / 1000.0;
    let sell_age = (now - sell.timestamp) as f64 / 1000.0;
    let freshness_score = (1.0 - (buy_age + sell_age) / (2.0 * max_age)).max(0.0);

    // Liquidity score
    let min_liq = buy.liquidity.min(sell.liquidity);
    let liquidity_score = if min_liq > Decimal::from(100000) {
        1.0
    } else if min_liq > Decimal::from(10000) {
        0.8
    } else {
        0.5
    };

    // Latency score
    let avg_latency = (buy.latency + sell.latency) as f64 / 2.0;
    let latency_score = (1.0 - avg_latency / 1000.0).max(0.0);

    // Weighted average
    0.4 * freshness_score + 0.4 * liquidity_score + 0.2 * latency_score
}

fn current_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scanner_creation() {
        let config = ScannerConfig::default();
        let scanner = Scanner::new(config);
        assert!(!*scanner.running.read().await);
    }

    #[tokio::test]
    async fn test_update_price() {
        let config = ScannerConfig::default();
        let scanner = Scanner::new(config);

        let source = PriceSource {
            chain_id: "lux".to_string(),
            venue: "lx_dex".to_string(),
            symbol: "BTC-USDC".to_string(),
            bid: Decimal::from(50000),
            ask: Decimal::from(50010),
            liquidity: Decimal::from(100),
            timestamp: current_time_ms(),
            latency: 5,
        };

        scanner.update_price(source).await;

        let prices = scanner.prices.read().await;
        assert!(prices.contains_key("BTC-USDC"));
    }

    #[tokio::test]
    async fn test_find_simple_arb() {
        let config = ScannerConfig {
            min_spread_bps: Decimal::from(5),
            min_profit_usd: Decimal::from(1),
            ..Default::default()
        };

        let sources = vec![
            PriceSource {
                chain_id: "lux".to_string(),
                venue: "lx_dex".to_string(),
                symbol: "BTC".to_string(),
                bid: Decimal::from(50000),
                ask: Decimal::from(50010),
                liquidity: Decimal::from(10),
                timestamp: current_time_ms(),
                latency: 5,
            },
            PriceSource {
                chain_id: "ethereum".to_string(),
                venue: "uniswap".to_string(),
                symbol: "BTC".to_string(),
                bid: Decimal::from(50100), // Higher bid - sell here
                ask: Decimal::from(50110),
                liquidity: Decimal::from(10),
                timestamp: current_time_ms(),
                latency: 100,
            },
        ];

        let chains = HashMap::new();
        let opps = find_simple_arb("BTC", &sources, &config, &chains);

        // Should find buy on lx_dex (lower ask), sell on uniswap (higher bid)
        assert!(!opps.is_empty());
    }
}
