//! Arbitrage types for LX Trading SDK.
//!
//! LX-FIRST ARBITRAGE STRATEGY:
//! - LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks)
//! - By the time other venues update, LX has already moved
//! - LX DEX price is the "TRUTH" (most current)
//! - Other venues are always STALE by comparison
//! - Arbitrage = correcting stale venues to match LX

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-chain transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossChainTransport {
    /// Warp - Lux native messaging between subnets only (<500ms)
    Warp,
    /// Teleport - EVM bridge for external chains (~30s)
    Teleport,
    /// Direct - Same chain, no bridge needed
    Direct,
    /// CEX API - API calls for centralized exchanges
    CexApi,
}

/// Type of blockchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainType {
    /// Lux subnet (Warp-enabled)
    LuxSubnet,
    /// EVM-compatible chain
    Evm,
    /// Centralized exchange
    Cex,
}

/// Type of arbitrage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArbType {
    /// Simple buy-low-sell-high
    Simple,
    /// Triangular A->B->C->A
    Triangular,
    /// Multi-hop complex routes
    MultiHop,
    /// CEX-DEX arbitrage
    CexDex,
    /// DEX flash swap
    FlashSwap,
}

/// Price feed from a specific venue/chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSource {
    pub chain_id: String,
    pub venue: String,
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub liquidity: Decimal,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Latency in milliseconds
    pub latency: i64,
}

impl PriceSource {
    pub fn mid_price(&self) -> Decimal {
        (self.bid + self.ask) / Decimal::from(2)
    }

    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }

    pub fn spread_bps(&self) -> Decimal {
        if self.bid.is_zero() {
            Decimal::ZERO
        } else {
            ((self.ask - self.bid) / self.bid) * Decimal::from(10000)
        }
    }
}

/// LX DEX price - the reference/oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LxPrice {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub mid: Decimal,
    pub timestamp: i64,
    pub block_num: u64,
}

impl LxPrice {
    pub fn new(symbol: String, bid: Decimal, ask: Decimal, timestamp: i64, block_num: u64) -> Self {
        let mid = (bid + ask) / Decimal::from(2);
        Self {
            symbol,
            bid,
            ask,
            mid,
            timestamp,
            block_num,
        }
    }
}

/// Price from a 'slow' venue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenuePrice {
    pub venue: String,
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub timestamp: i64,
    /// How far behind LX this venue typically is (ms)
    pub latency: i64,
    /// Is this price stale relative to LX?
    pub stale: bool,
}

/// Single leg of an arbitrage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub chain_id: String,
    pub venue: String,
    /// "buy" or "sell"
    pub action: String,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: Decimal,
    pub expected_out: Decimal,
    pub min_amount_out: Decimal,
    pub swap_data: Option<Vec<u8>>,
}

/// Detected arbitrage opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub arb_type: ArbType,
    pub routes: Vec<Route>,
    pub buy_source: PriceSource,
    pub sell_source: PriceSource,
    /// Spread in basis points
    pub spread_bps: Decimal,
    pub estimated_pnl: Decimal,
    /// Limited by liquidity
    pub max_size: Decimal,
    pub gas_cost_usd: Decimal,
    pub bridge_cost_usd: Decimal,
    pub net_pnl: Decimal,
    /// 0-1, based on price freshness and liquidity
    pub confidence: f64,
    pub expires_at: i64,
}

impl ArbitrageOpportunity {
    pub fn is_profitable(&self) -> bool {
        self.net_pnl > Decimal::ZERO
    }

    pub fn is_expired(&self, now: i64) -> bool {
        now > self.expires_at
    }
}

/// LX-first arbitrage opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LxFirstOpportunity {
    pub id: String,
    pub symbol: String,
    pub timestamp: i64,
    pub lx_price: LxPrice,
    pub stale_venue: String,
    pub stale_price: VenuePrice,
    /// Staleness in milliseconds
    pub staleness: i64,
    /// "buy" or "sell"
    pub side: String,
    pub divergence: Decimal,
    pub divergence_bps: Decimal,
    pub expected_profit: Decimal,
    pub max_size: Decimal,
    pub confidence: f64,
}

/// Unified arbitrage opportunity across venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedOpportunity {
    pub id: String,
    pub symbol: String,
    pub timestamp: i64,
    pub expires_at: i64,
    pub buy_venue: String,
    pub buy_price: Decimal,
    pub buy_size: Decimal,
    pub sell_venue: String,
    pub sell_price: Decimal,
    pub sell_size: Decimal,
    pub spread: Decimal,
    pub spread_bps: Decimal,
    pub max_size: Decimal,
    pub gross_profit: Decimal,
    pub est_fees: Decimal,
    pub net_profit: Decimal,
    pub confidence: f64,
    pub latency: i64,
}

/// Executed arbitrage result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExecution {
    pub id: String,
    pub opportunity: UnifiedOpportunity,
    pub start_time: i64,
    pub end_time: i64,
    /// "executing", "completed", "failed"
    pub status: String,
    pub buy_order_id: Option<String>,
    pub sell_order_id: Option<String>,
    pub actual_profit: Decimal,
    pub fees: Decimal,
    pub error: Option<String>,
}

/// Arbitrage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedArbStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub total_pnl: Decimal,
    pub win_rate: f64,
}

/// Configuration for unified arbitrage system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedArbConfig {
    pub min_spread_bps: Decimal,
    pub min_profit: Decimal,
    pub max_position_size: Decimal,
    pub max_total_exposure: Decimal,
    pub symbols: Vec<String>,
    pub venue_priority: Vec<String>,
    pub scan_interval_ms: u64,
    pub execute_timeout_ms: u64,
    pub max_daily_loss: Decimal,
    pub max_trades_per_day: u32,
}

impl Default for UnifiedArbConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: Decimal::from(10),
            min_profit: Decimal::from(5),
            max_position_size: Decimal::from(10000),
            max_total_exposure: Decimal::from(100000),
            symbols: vec![
                "BTC-USDC".to_string(),
                "ETH-USDC".to_string(),
                "LUX-USDC".to_string(),
            ],
            venue_priority: vec![
                "lx_dex".to_string(),
                "binance".to_string(),
                "mexc".to_string(),
                "lx_amm".to_string(),
            ],
            scan_interval_ms: 100,
            execute_timeout_ms: 5000,
            max_daily_loss: Decimal::from(1000),
            max_trades_per_day: 100,
        }
    }
}

/// Configuration for LX-first strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LxFirstConfig {
    pub max_staleness_ms: i64,
    pub min_divergence_bps: Decimal,
    pub min_profit: Decimal,
    pub max_position_size: Decimal,
    pub symbols: Vec<String>,
    pub venue_latencies: HashMap<String, i64>,
}

impl Default for LxFirstConfig {
    fn default() -> Self {
        let mut venue_latencies = HashMap::new();
        venue_latencies.insert("binance".to_string(), 50);
        venue_latencies.insert("mexc".to_string(), 100);
        venue_latencies.insert("okx".to_string(), 80);
        venue_latencies.insert("uniswap".to_string(), 12000);
        venue_latencies.insert("pancakeswap".to_string(), 3000);

        Self {
            max_staleness_ms: 2000,
            min_divergence_bps: Decimal::from(10),
            min_profit: Decimal::from(5),
            max_position_size: Decimal::from(1000),
            symbols: vec![
                "BTC-USDC".to_string(),
                "ETH-USDC".to_string(),
                "LUX-USDC".to_string(),
            ],
            venue_latencies,
        }
    }
}

/// Configuration for arbitrage scanner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerConfig {
    pub min_spread_bps: Decimal,
    pub min_profit_usd: Decimal,
    pub max_price_age_ms: i64,
    pub symbols: Vec<String>,
    pub chain_ids: Vec<String>,
    pub scan_interval_ms: u64,
    pub max_concurrency: usize,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: Decimal::from(10),
            min_profit_usd: Decimal::from(10),
            max_price_age_ms: 5000,
            symbols: vec![
                "BTC".to_string(),
                "ETH".to_string(),
                "LUX".to_string(),
                "SOL".to_string(),
                "AVAX".to_string(),
            ],
            chain_ids: vec![
                "lux".to_string(),
                "ethereum".to_string(),
                "bsc".to_string(),
                "arbitrum".to_string(),
                "polygon".to_string(),
            ],
            scan_interval_ms: 100,
            max_concurrency: 50,
        }
    }
}

/// Information about a chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainInfo {
    pub chain_id: String,
    pub name: String,
    pub chain_type: ChainType,
    pub block_time_ms: u64,
    pub finality_ms: u64,
    pub warp_supported: bool,
    pub teleport_supported: bool,
    pub venues: Vec<String>,
}

/// Configuration for cross-chain routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainConfig {
    pub warp_enabled: bool,
    pub warp_endpoint: Option<String>,
    pub warp_timeout_ms: u64,
    pub teleport_enabled: bool,
    pub teleport_endpoint: Option<String>,
    pub teleport_timeout_ms: u64,
    pub chains: HashMap<String, CrossChainInfo>,
}

impl Default for CrossChainConfig {
    fn default() -> Self {
        let mut chains = HashMap::new();

        // Lux ecosystem (Warp enabled)
        chains.insert(
            "lux_mainnet".to_string(),
            CrossChainInfo {
                chain_id: "lux_mainnet".to_string(),
                name: "Lux Mainnet".to_string(),
                chain_type: ChainType::LuxSubnet,
                block_time_ms: 400,
                finality_ms: 400,
                warp_supported: true,
                teleport_supported: true,
                venues: vec!["lx_dex".to_string(), "lx_amm".to_string()],
            },
        );

        chains.insert(
            "lx_dex_subnet".to_string(),
            CrossChainInfo {
                chain_id: "lx_dex_subnet".to_string(),
                name: "LX DEX Subnet".to_string(),
                chain_type: ChainType::LuxSubnet,
                block_time_ms: 200,
                finality_ms: 200,
                warp_supported: true,
                teleport_supported: false,
                venues: vec!["lx_dex".to_string()],
            },
        );

        // EVM chains (Teleport enabled)
        chains.insert(
            "ethereum".to_string(),
            CrossChainInfo {
                chain_id: "1".to_string(),
                name: "Ethereum".to_string(),
                chain_type: ChainType::Evm,
                block_time_ms: 12000,
                finality_ms: 15 * 60 * 1000, // 15 minutes
                warp_supported: false,
                teleport_supported: true,
                venues: vec!["uniswap".to_string(), "sushiswap".to_string()],
            },
        );

        chains.insert(
            "bsc".to_string(),
            CrossChainInfo {
                chain_id: "56".to_string(),
                name: "BNB Smart Chain".to_string(),
                chain_type: ChainType::Evm,
                block_time_ms: 3000,
                finality_ms: 45000,
                warp_supported: false,
                teleport_supported: true,
                venues: vec!["pancakeswap".to_string()],
            },
        );

        chains.insert(
            "arbitrum".to_string(),
            CrossChainInfo {
                chain_id: "42161".to_string(),
                name: "Arbitrum One".to_string(),
                chain_type: ChainType::Evm,
                block_time_ms: 250,
                finality_ms: 15 * 60 * 1000,
                warp_supported: false,
                teleport_supported: true,
                venues: vec!["uniswap".to_string(), "camelot".to_string()],
            },
        );

        // CEX (API only)
        chains.insert(
            "binance".to_string(),
            CrossChainInfo {
                chain_id: "binance".to_string(),
                name: "Binance".to_string(),
                chain_type: ChainType::Cex,
                block_time_ms: 0,
                finality_ms: 0,
                warp_supported: false,
                teleport_supported: false,
                venues: vec!["binance".to_string()],
            },
        );

        chains.insert(
            "mexc".to_string(),
            CrossChainInfo {
                chain_id: "mexc".to_string(),
                name: "MEXC".to_string(),
                chain_type: ChainType::Cex,
                block_time_ms: 0,
                finality_ms: 0,
                warp_supported: false,
                teleport_supported: false,
                venues: vec!["mexc".to_string()],
            },
        );

        Self {
            warp_enabled: true,
            warp_endpoint: None,
            warp_timeout_ms: 5000,
            teleport_enabled: true,
            teleport_endpoint: None,
            teleport_timeout_ms: 60000,
            chains,
        }
    }
}

/// Enhanced opportunity with routing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedOpportunity {
    pub base: UnifiedOpportunity,
    pub transport: CrossChainTransport,
    pub estimated_latency: i64,
    pub bridge_cost: Decimal,
    pub adjusted_net_profit: Decimal,
}

/// Bridge transaction status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub tx_id: String,
    /// pending, confirming, completed, failed
    pub status: String,
    pub source_chain: String,
    pub dest_chain: String,
    pub amount: Decimal,
    pub fee: Decimal,
    pub source_tx: String,
    pub dest_tx: Option<String>,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_source_calculations() {
        let source = PriceSource {
            chain_id: "lux".to_string(),
            venue: "lx_dex".to_string(),
            symbol: "BTC-USDC".to_string(),
            bid: Decimal::from(50000),
            ask: Decimal::from(50010),
            liquidity: Decimal::from(100),
            timestamp: 1000,
            latency: 5,
        };

        assert_eq!(source.mid_price(), Decimal::from(50005));
        assert_eq!(source.spread(), Decimal::from(10));
    }

    #[test]
    fn test_lx_price_creation() {
        let price = LxPrice::new(
            "BTC-USDC".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            1000,
            100,
        );

        assert_eq!(price.mid, Decimal::from(50005));
    }

    #[test]
    fn test_default_configs() {
        let unified = UnifiedArbConfig::default();
        assert_eq!(unified.min_spread_bps, Decimal::from(10));

        let lx_first = LxFirstConfig::default();
        assert_eq!(lx_first.max_staleness_ms, 2000);

        let scanner = ScannerConfig::default();
        assert_eq!(scanner.scan_interval_ms, 100);

        let cross_chain = CrossChainConfig::default();
        assert!(cross_chain.warp_enabled);
    }
}
