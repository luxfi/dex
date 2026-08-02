//! Market Making Strategy Example
//!
//! This example demonstrates a simple market making strategy that:
//! - Maintains quotes on both sides of the orderbook
//! - Adjusts spread based on inventory risk
//! - Uses risk management to prevent excessive exposure
//!
//! # Strategy Overview
//!
//! Market makers profit from the bid-ask spread while managing inventory risk.
//! This example shows:
//! - Quote calculation with inventory skew
//! - Position-based spread adjustment
//! - Risk limits and monitoring
//!
//! # Running
//!
//! ```bash
//! cargo run --example market_maker
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::time::interval;

use lx_trading::{OrderRequest, Side};

/// Market maker configuration
#[derive(Debug, Clone)]
struct MakerConfig {
    /// Trading symbol
    symbol: String,
    /// Target spread in basis points
    target_spread_bps: Decimal,
    /// Maximum spread in basis points
    max_spread_bps: Decimal,
    /// Order size for each quote
    order_size: Decimal,
    /// Maximum position (long or short)
    max_position: Decimal,
    /// Inventory skew factor (how much to widen spread per unit of inventory)
    inventory_skew: Decimal,
    /// Quote refresh interval
    refresh_interval: Duration,
}

impl Default for MakerConfig {
    fn default() -> Self {
        Self {
            symbol: "BTC-USDC".to_string(),
            target_spread_bps: dec!(10), // 10 bps = 0.1%
            max_spread_bps: dec!(50),    // 50 bps = 0.5%
            order_size: dec!(0.1),       // 0.1 BTC per side
            max_position: dec!(1.0),     // Max 1 BTC long or short
            inventory_skew: dec!(5),     // 5 bps per 0.1 BTC inventory
            refresh_interval: Duration::from_millis(500),
        }
    }
}

/// Market maker state
struct MakerState {
    /// Current position (positive = long, negative = short)
    position: Decimal,
    /// Total realized PnL
    realized_pnl: Decimal,
    /// Number of trades executed
    trade_count: u64,
    /// Total volume traded
    volume: Decimal,
}

impl Default for MakerState {
    fn default() -> Self {
        Self {
            position: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            trade_count: 0,
            volume: Decimal::ZERO,
        }
    }
}

/// Calculate quotes based on mid price and inventory
fn calculate_quotes(
    config: &MakerConfig,
    state: &MakerState,
    reference_price: Decimal,
) -> (Decimal, Decimal) {
    // Base spread
    let base_spread = reference_price * config.target_spread_bps / dec!(10000);

    // Inventory adjustment: widen spread on the side where we have exposure
    let inventory_adjustment = state.position.abs() * config.inventory_skew / dec!(10000);
    let skew = state.position * config.inventory_skew / dec!(10000) * reference_price;

    // Calculate bid and ask with inventory skew
    // If long (position > 0): lower bid more, raise ask less (encourage sells)
    // If short (position < 0): lower bid less, raise ask more (encourage buys)
    let half_spread = base_spread / dec!(2) + inventory_adjustment * reference_price;

    let bid_price = reference_price - half_spread - skew;
    let ask_price = reference_price + half_spread - skew;

    // Enforce maximum spread
    let max_half_spread = reference_price * config.max_spread_bps / dec!(10000) / dec!(2);
    let clamped_bid = bid_price.max(reference_price - max_half_spread);
    let clamped_ask = ask_price.min(reference_price + max_half_spread);

    (clamped_bid.round_dp(2), clamped_ask.round_dp(2))
}

/// Check if position is within limits
fn check_position_limits(config: &MakerConfig, state: &MakerState, side: Side) -> bool {
    match side {
        Side::Buy => state.position < config.max_position,
        Side::Sell => state.position > -config.max_position,
    }
}

/// Print market maker status
fn print_status(config: &MakerConfig, state: &MakerState, bid: Decimal, ask: Decimal) {
    let spread_bps = (ask - bid) / ((ask + bid) / dec!(2)) * dec!(10000);

    println!("\n--- Market Maker Status ---");
    println!("Symbol: {}", config.symbol);
    println!("Position: {} BTC", state.position);
    println!("Bid: ${} | Ask: ${}", bid, ask);
    println!("Spread: {:.1} bps", spread_bps);
    println!(
        "Trades: {} | Volume: {} BTC",
        state.trade_count, state.volume
    );
    println!("Realized PnL: ${:.2}", state.realized_pnl);
    println!(
        "Position Limit: {:.1}%",
        (state.position.abs() / config.max_position) * dec!(100)
    );
}

/// Simulate a fill
fn simulate_fill(state: &mut MakerState, side: Side, price: Decimal, quantity: Decimal) {
    match side {
        Side::Buy => {
            state.position += quantity;
        }
        Side::Sell => {
            state.position -= quantity;
        }
    }
    state.trade_count += 1;
    state.volume += quantity;

    // Simple PnL estimation (in reality, need to track entry prices)
    println!(
        "  [FILL] {} {:.4} BTC @ ${}",
        if side == Side::Buy { "BUY" } else { "SELL" },
        quantity,
        price
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== LX Trading SDK - Market Maker Example ===\n");

    let config = MakerConfig::default();
    let state = Arc::new(RwLock::new(MakerState::default()));

    println!("Configuration:");
    println!("  Symbol: {}", config.symbol);
    println!("  Target Spread: {} bps", config.target_spread_bps);
    println!("  Order Size: {} BTC", config.order_size);
    println!("  Max Position: {} BTC", config.max_position);
    println!("  Refresh Interval: {:?}", config.refresh_interval);
    println!();

    // Simulate market data feed with some price variation
    let base_price = dec!(50000);
    let mut price_offset = dec!(0);
    let mut iteration = 0;

    // Run for a limited number of iterations for the example
    let max_iterations = 10;
    let mut ticker_interval = interval(config.refresh_interval);

    println!("Starting market making simulation...\n");

    while iteration < max_iterations {
        ticker_interval.tick().await;
        iteration += 1;

        // Simulate price movement
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let change: f64 = rng.gen_range(-0.001..0.001);
        price_offset += Decimal::try_from(change * 50000.0).unwrap_or_default();
        let current_price = base_price + price_offset;

        println!(
            "\n[Iteration {}] Reference Price: ${:.2}",
            iteration, current_price
        );

        // Calculate quotes
        let (bid_price, ask_price) = {
            let s = state.read();
            calculate_quotes(&config, &s, current_price)
        };

        // Create orders (in reality, would send to exchange)
        let mut s = state.write();

        // Check if we can quote on each side
        let can_bid = check_position_limits(&config, &s, Side::Buy);
        let can_ask = check_position_limits(&config, &s, Side::Sell);

        if can_bid {
            let _bid_order =
                OrderRequest::limit(&config.symbol, Side::Buy, config.order_size, bid_price)
                    .post_only();
            println!("  BID: {} @ ${}", config.order_size, bid_price);

            // Simulate random fill (30% chance)
            if rng.gen_bool(0.3) {
                simulate_fill(&mut s, Side::Buy, bid_price, config.order_size);
            }
        } else {
            println!("  BID: SKIPPED (position limit)");
        }

        if can_ask {
            let _ask_order =
                OrderRequest::limit(&config.symbol, Side::Sell, config.order_size, ask_price)
                    .post_only();
            println!("  ASK: {} @ ${}", config.order_size, ask_price);

            // Simulate random fill (30% chance)
            if rng.gen_bool(0.3) {
                simulate_fill(&mut s, Side::Sell, ask_price, config.order_size);
            }
        } else {
            println!("  ASK: SKIPPED (position limit)");
        }

        // Print status every 5 iterations
        if iteration % 5 == 0 {
            print_status(&config, &s, bid_price, ask_price);
        }
    }

    // Final status
    {
        let s = state.read();
        println!("\n=== Final Summary ===");
        println!("Total Trades: {}", s.trade_count);
        println!("Total Volume: {} BTC", s.volume);
        println!("Final Position: {} BTC", s.position);
        println!("Realized PnL: ${:.2}", s.realized_pnl);
    }

    println!("\n=== Example Complete ===");
    println!("\nKey concepts demonstrated:");
    println!("1. Bid-ask spread calculation");
    println!("2. Inventory-based quote adjustment");
    println!("3. Position limit checks");
    println!("4. Post-only orders for maker rebates");

    Ok(())
}
