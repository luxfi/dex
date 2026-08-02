//! LX-First Arbitrage Bot Example
//!
//! This bot uses the LX-First strategy where LX DEX prices are treated
//! as the "truth" (fastest venue with nanosecond updates, 200ms blocks).
//! Other venues are always stale by comparison.
//!
//! Arbitrage = exploiting stale venues before they catch up to LX prices.
//!
//! Cross-chain transport options:
//! - Warp: For Lux subnet communication (<500ms)
//! - Teleport: For EVM chain bridging (~30s)
//! - CEX API: Direct trading (instant)
//!
//! NO SMART CONTRACTS - just coordinated trades through unified SDK.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::RwLock;
use tokio::time::interval;

use lx_trading::arbitrage::{
    ChainType, CrossChainConfig, CrossChainInfo, CrossChainRouter, LxFirstArbitrage, LxFirstConfig,
    LxFirstOpportunity, LxPrice, Scanner, ScannerConfig, VenuePrice,
};

// ============================================
// Arbitrage Bot
// ============================================

struct ArbitrageBot {
    lx_first: Arc<LxFirstArbitrage>,
    scanner: Arc<RwLock<Scanner>>,
    router: Arc<CrossChainRouter>,
    running: Arc<RwLock<bool>>,
    stats: Arc<RwLock<Stats>>,
}

#[derive(Default)]
struct Stats {
    total_opportunities: u64,
    total_executions: u64,
    total_pnl: Decimal,
}

impl ArbitrageBot {
    async fn new() -> Result<Self> {
        println!("{}", "=".repeat(60));
        println!("LX-FIRST ARBITRAGE BOT");
        println!("{}", "=".repeat(60));
        println!();

        // Initialize LX-First strategy
        let lx_config = LxFirstConfig {
            max_staleness_ms: 2000,
            min_divergence_bps: dec!(10),
            min_profit: dec!(5),
            max_position_size: dec!(10000),
            symbols: vec!["BTC-USDC".into(), "ETH-USDC".into(), "LUX-USDC".into()],
            venue_latencies: [
                ("binance".into(), 50),
                ("mexc".into(), 100),
                ("okx".into(), 80),
                ("uniswap".into(), 12000),
                ("pancakeswap".into(), 3000),
            ]
            .into_iter()
            .collect(),
        };
        let lx_first = Arc::new(LxFirstArbitrage::new(lx_config));
        println!("[OK] LX-First strategy initialized");

        // Initialize Scanner
        let scanner_config = ScannerConfig {
            min_spread_bps: dec!(10),
            min_profit_usd: dec!(10),
            max_price_age_ms: 5000,
            symbols: vec![
                "BTC".into(),
                "ETH".into(),
                "LUX".into(),
                "SOL".into(),
                "AVAX".into(),
            ],
            chain_ids: vec![
                "lux".into(),
                "ethereum".into(),
                "bsc".into(),
                "arbitrum".into(),
                "polygon".into(),
            ],
            scan_interval_ms: 100,
            max_concurrency: 50,
        };
        let scanner = Arc::new(RwLock::new(Scanner::new(scanner_config)));
        println!("[OK] Scanner initialized");

        // Initialize Cross-Chain Router
        let mut router_config = CrossChainConfig::default();
        router_config.warp_enabled = true;
        router_config.teleport_enabled = true;
        router_config.chains.insert(
            "lux_mainnet".into(),
            CrossChainInfo {
                chain_id: "lux_mainnet".into(),
                name: "Lux Mainnet".into(),
                chain_type: ChainType::LuxSubnet,
                block_time_ms: 400,
                finality_ms: 400,
                warp_supported: true,
                teleport_supported: true,
                venues: vec!["lx_dex".into(), "lx_amm".into()],
            },
        );
        router_config.chains.insert(
            "ethereum".into(),
            CrossChainInfo {
                chain_id: "1".into(),
                name: "Ethereum".into(),
                chain_type: ChainType::Evm,
                block_time_ms: 12000,
                finality_ms: 900000,
                warp_supported: false,
                teleport_supported: true,
                venues: vec!["uniswap".into(), "sushiswap".into()],
            },
        );
        router_config.chains.insert(
            "binance".into(),
            CrossChainInfo {
                chain_id: "binance".into(),
                name: "Binance".into(),
                chain_type: ChainType::Cex,
                block_time_ms: 0,
                finality_ms: 0,
                warp_supported: false,
                teleport_supported: false,
                venues: vec!["binance".into()],
            },
        );
        let router = Arc::new(CrossChainRouter::new(router_config));
        println!("[OK] Cross-chain router initialized");

        Ok(Self {
            lx_first,
            scanner,
            router,
            running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(Stats::default())),
        })
    }

    async fn start(&self) -> Result<()> {
        // Start systems
        self.lx_first.start().await;
        {
            let mut scanner = self.scanner.write().await;
            scanner.start().await;
        }

        *self.running.write().await = true;

        println!();
        println!("{}", "=".repeat(60));
        println!("BOT RUNNING - Press Ctrl+C to stop");
        println!("{}", "=".repeat(60));
        println!();
        println!("Monitoring symbols: BTC-USDC, ETH-USDC, LUX-USDC");
        println!("Min divergence: 10 bps");
        println!("Min profit: $5");
        println!();

        // Spawn price feed simulator
        let lx_first = self.lx_first.clone();
        let running = self.running.clone();
        tokio::spawn(async move {
            Self::simulate_price_feeds(lx_first, running).await;
        });

        // Spawn stats reporter
        let stats = self.stats.clone();
        let running = self.running.clone();
        tokio::spawn(async move {
            Self::report_stats(stats, running).await;
        });

        Ok(())
    }

    async fn stop(&self) {
        println!("\nShutting down...");
        *self.running.write().await = false;

        self.lx_first.stop().await;
        {
            let mut scanner = self.scanner.write().await;
            scanner.stop().await;
        }

        self.print_final_stats().await;
    }

    async fn simulate_price_feeds(lx_first: Arc<LxFirstArbitrage>, running: Arc<RwLock<bool>>) {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let base_prices = [
            ("BTC-USDC", dec!(50000)),
            ("ETH-USDC", dec!(3000)),
            ("LUX-USDC", dec!(25)),
        ];

        let mut interval = interval(Duration::from_millis(100));
        let mut rng = StdRng::from_entropy();

        loop {
            interval.tick().await;

            if !*running.read().await {
                break;
            }

            for (symbol, base) in &base_prices {
                // Simulate LX DEX price (the oracle)
                let variance: f64 = rng.gen_range(-0.001..0.001);
                let lx_mid = *base * Decimal::try_from(1.0 + variance).unwrap();

                lx_first
                    .update_lx_price(LxPrice {
                        symbol: symbol.to_string(),
                        bid: lx_mid * dec!(0.9999),
                        ask: lx_mid * dec!(1.0001),
                        mid: lx_mid,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        block_num: rng.gen_range(1000000..2000000),
                    })
                    .await;

                // Simulate stale CEX prices
                for (venue, latency) in [("binance", 50i64), ("mexc", 100i64)] {
                    let divergence: f64 = rng.gen_range(-0.002..0.002);
                    let venue_mid = *base * Decimal::try_from(1.0 + divergence).unwrap();

                    lx_first
                        .update_venue_price(VenuePrice {
                            venue: venue.to_string(),
                            symbol: symbol.to_string(),
                            bid: venue_mid * dec!(0.9998),
                            ask: venue_mid * dec!(1.0002),
                            timestamp: chrono::Utc::now().timestamp_millis() - latency,
                            latency,
                            stale: false,
                        })
                        .await;
                }
            }
        }
    }

    async fn handle_opportunity(
        opp: &LxFirstOpportunity,
        router: &CrossChainRouter,
        stats: &Arc<RwLock<Stats>>,
    ) {
        println!();
        println!("{}", "=".repeat(50));
        println!("LX-FIRST OPPORTUNITY DETECTED");
        println!("{}", "=".repeat(50));
        println!("Symbol:          {}", opp.symbol);
        println!("LX Price:        ${}", opp.lx_price.mid);
        println!("Stale Venue:     {}", opp.stale_venue);
        println!(
            "Stale Bid/Ask:   ${} / ${}",
            opp.stale_price.bid, opp.stale_price.ask
        );
        println!("Staleness:       {}ms", opp.staleness);
        println!("Side:            {}", opp.side.to_uppercase());
        println!("Divergence:      {} bps", opp.divergence_bps);
        println!("Expected Profit: ${}", opp.expected_profit);
        println!("Confidence:      {:.1}%", opp.confidence * 100.0);
        println!("{}", "=".repeat(50));

        if opp.confidence > 0.8 {
            println!("\n[EXECUTING] {}...", opp.id);

            // Determine cross-chain transport
            let buy_chain = router.venue_to_chain(&opp.stale_venue);
            let sell_chain = "lux_mainnet";
            let transport = router.determine_transport(&buy_chain, sell_chain);
            let latency = router.estimate_latency(&buy_chain, sell_chain);

            println!("  Transport: {transport:?}");
            println!("  Est. Latency: {latency}ms");

            if opp.side == "buy" {
                println!("  Buying on {}...", opp.stale_venue);
                println!("  Hedging on LX DEX...");
            } else {
                println!("  Selling on {}...", opp.stale_venue);
                println!("  Hedging on LX DEX...");
            }

            // Simulate successful execution
            let profit = opp.expected_profit * dec!(0.8);

            {
                let mut s = stats.write().await;
                s.total_executions += 1;
                s.total_pnl += profit;
            }

            println!("[SUCCESS] Executed {} | Profit: ${:.2}", opp.id, profit);
        }

        {
            let mut s = stats.write().await;
            s.total_opportunities += 1;
        }
    }

    async fn report_stats(stats: Arc<RwLock<Stats>>, running: Arc<RwLock<bool>>) {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            if !*running.read().await {
                break;
            }

            let s = stats.read().await;
            println!();
            println!("{}", "-".repeat(40));
            println!("STATS");
            println!("  Opportunities: {}", s.total_opportunities);
            println!("  Executions:    {}", s.total_executions);
            println!("  Total PnL:     ${:.2}", s.total_pnl);
            if s.total_executions > 0 {
                let avg_pnl = s.total_pnl / Decimal::from(s.total_executions);
                println!("  Avg PnL:       ${avg_pnl:.2}");
            }
            println!("{}", "-".repeat(40));
        }
    }

    async fn print_final_stats(&self) {
        let s = self.stats.read().await;
        println!();
        println!("{}", "=".repeat(50));
        println!("FINAL STATISTICS");
        println!("{}", "=".repeat(50));
        println!("Total Opportunities: {}", s.total_opportunities);
        println!("Total Executions:    {}", s.total_executions);
        println!("Total PnL:           ${:.2}", s.total_pnl);
        if s.total_executions > 0 && s.total_opportunities > 0 {
            let win_rate = (s.total_executions as f64 / s.total_opportunities as f64) * 100.0;
            let avg_pnl = s.total_pnl / Decimal::from(s.total_executions);
            println!("Execution Rate:      {win_rate:.1}%");
            println!("Avg PnL per Trade:   ${avg_pnl:.2}");
        }
        println!("{}", "=".repeat(50));
    }
}

// ============================================
// Main Entry Point
// ============================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let bot = ArbitrageBot::new().await?;
    bot.start().await?;

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    bot.stop().await;

    Ok(())
}
