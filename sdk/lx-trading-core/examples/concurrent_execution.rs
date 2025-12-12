//! Concurrent Execution Example
//!
//! This example demonstrates async patterns for high-performance trading:
//! - Concurrent order submission to multiple venues
//! - Parallel market data fetching
//! - Task coordination with tokio
//! - Error handling in concurrent contexts
//!
//! # Running
//!
//! ```bash
//! cargo run --example concurrent_execution
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::future::join_all;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::timeout;

use lx_trading::{OrderRequest, OrderStatus, Side};

/// Simulated order result
#[derive(Debug, Clone)]
struct OrderResult {
    venue: String,
    order_id: String,
    symbol: String,
    side: Side,
    quantity: Decimal,
    price: Decimal,
    status: OrderStatus,
    latency_ms: u64,
}

/// Execution statistics
struct ExecutionStats {
    orders_submitted: AtomicU64,
    orders_filled: AtomicU64,
    orders_rejected: AtomicU64,
    total_latency_us: AtomicU64,
}

impl ExecutionStats {
    fn new() -> Self {
        Self {
            orders_submitted: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }

    fn record_order(&self, result: &OrderResult) {
        self.orders_submitted.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add(result.latency_ms * 1000, Ordering::Relaxed);

        match result.status {
            OrderStatus::Filled | OrderStatus::PartiallyFilled => {
                self.orders_filled.fetch_add(1, Ordering::Relaxed);
            }
            OrderStatus::Rejected => {
                self.orders_rejected.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn print_summary(&self) {
        let submitted = self.orders_submitted.load(Ordering::Relaxed);
        let filled = self.orders_filled.load(Ordering::Relaxed);
        let rejected = self.orders_rejected.load(Ordering::Relaxed);
        let total_latency = self.total_latency_us.load(Ordering::Relaxed);

        let avg_latency = if submitted > 0 {
            total_latency / submitted / 1000 // Convert to ms
        } else {
            0
        };

        let fill_rate = if submitted > 0 {
            (filled as f64 / submitted as f64) * 100.0
        } else {
            0.0
        };

        println!("\n=== Execution Statistics ===");
        println!("Orders Submitted: {}", submitted);
        println!("Orders Filled: {}", filled);
        println!("Orders Rejected: {}", rejected);
        println!("Fill Rate: {:.1}%", fill_rate);
        println!("Average Latency: {}ms", avg_latency);
    }
}

/// Simulate order execution with random latency and outcome
async fn execute_order(
    venue: &str,
    request: OrderRequest,
    stats: Arc<ExecutionStats>,
) -> Result<OrderResult> {
    use rand::Rng;

    let start = Instant::now();

    // Generate random values before await
    let (latency_ms, outcome_filled, outcome_rejected): (u64, bool, bool) = {
        let mut rng = rand::thread_rng();
        (
            rng.gen_range(10..50),
            rng.gen_bool(0.85),
            rng.gen_bool(0.5),
        )
    };

    // Simulate network latency (10-50ms)
    tokio::time::sleep(Duration::from_millis(latency_ms)).await;

    // Generate order ID
    let order_id = {
        let mut rng = rand::thread_rng();
        format!("{}-{}", venue, rng.gen::<u32>())
    };

    // Use pre-generated random outcomes
    let status = if outcome_filled {
        OrderStatus::Filled
    } else if outcome_rejected {
        OrderStatus::Rejected
    } else {
        OrderStatus::PartiallyFilled
    };

    let result = OrderResult {
        venue: venue.to_string(),
        order_id,
        symbol: request.symbol.clone(),
        side: request.side.clone(),
        quantity: request.quantity,
        price: request.price.unwrap_or(dec!(50000)),
        status,
        latency_ms: start.elapsed().as_millis() as u64,
    };

    stats.record_order(&result);
    Ok(result)
}

/// Submit orders to multiple venues concurrently
async fn submit_to_venues(
    venues: &[&str],
    request: OrderRequest,
    stats: Arc<ExecutionStats>,
) -> Vec<Result<OrderResult>> {
    let futures: Vec<_> = venues
        .iter()
        .map(|venue| {
            let req = request.clone();
            let stats = Arc::clone(&stats);
            async move { execute_order(venue, req, stats).await }
        })
        .collect();

    join_all(futures).await
}

/// Fetch market data from multiple venues concurrently
async fn fetch_market_data(venues: &[&str], symbol: &str) -> Vec<(String, Decimal)> {
    use rand::Rng;

    let futures: Vec<_> = venues
        .iter()
        .map(|venue| async move {
            // Simulate fetch latency
            let mut rng = rand::thread_rng();
            let latency: u64 = rng.gen_range(5..30);
            tokio::time::sleep(Duration::from_millis(latency)).await;

            // Simulate price with small venue-specific offset
            let base = dec!(50000);
            let offset: f64 = rng.gen_range(-20.0..20.0);
            let price = base + Decimal::try_from(offset).unwrap_or_default();

            (venue.to_string(), price)
        })
        .collect();

    join_all(futures).await
}

/// Rate-limited order submission using semaphore
async fn rate_limited_orders(
    orders: Vec<OrderRequest>,
    venue: &str,
    max_concurrent: usize,
    stats: Arc<ExecutionStats>,
) -> Vec<Result<OrderResult>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));

    let futures: Vec<_> = orders
        .into_iter()
        .map(|order| {
            let permit = Arc::clone(&semaphore);
            let stats = Arc::clone(&stats);
            let venue = venue.to_string();
            async move {
                let _permit = permit.acquire().await.unwrap();
                execute_order(&venue, order, stats).await
            }
        })
        .collect();

    join_all(futures).await
}

/// Process stream of orders with bounded concurrency
async fn process_order_stream(
    mut rx: mpsc::Receiver<OrderRequest>,
    venue: String,
    stats: Arc<ExecutionStats>,
    max_concurrent: usize,
) {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let mut handles = vec![];

    while let Some(order) = rx.recv().await {
        let permit = Arc::clone(&semaphore);
        let stats = Arc::clone(&stats);
        let venue = venue.clone();

        let handle = tokio::spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            let result = execute_order(&venue, order, stats).await;
            if let Ok(r) = &result {
                println!(
                    "  [{}] {} {} @ {} -> {:?} ({}ms)",
                    r.venue,
                    if r.side == Side::Buy { "BUY" } else { "SELL" },
                    r.quantity,
                    r.price,
                    r.status,
                    r.latency_ms
                );
            }
        });

        handles.push(handle);
    }

    // Wait for all in-flight orders
    for handle in handles {
        let _ = handle.await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== LX Trading SDK - Concurrent Execution Example ===\n");

    let stats = Arc::new(ExecutionStats::new());
    let venues = vec!["lx_dex", "binance", "mexc"];

    // Example 1: Parallel market data fetching
    println!("--- Example 1: Parallel Market Data Fetch ---");
    println!("Fetching BTC-USDC prices from {} venues...", venues.len());

    let start = Instant::now();
    let prices = fetch_market_data(&venues, "BTC-USDC").await;
    let fetch_time = start.elapsed();

    for (venue, price) in &prices {
        println!("  {}: ${}", venue, price);
    }
    println!("Total fetch time: {:?} (parallel)", fetch_time);

    // Find best price
    if let Some((best_venue, best_price)) = prices.iter().min_by_key(|(_, p)| *p) {
        println!("Best ask: {} at ${}", best_venue, best_price);
    }

    // Example 2: Concurrent order submission
    println!("\n--- Example 2: Concurrent Order Submission ---");
    let order = OrderRequest::limit("BTC-USDC", Side::Buy, dec!(0.1), dec!(50000));

    println!("Submitting order to {} venues concurrently...", venues.len());
    let start = Instant::now();
    let results = submit_to_venues(&venues, order, Arc::clone(&stats)).await;
    let submit_time = start.elapsed();

    for result in results {
        match result {
            Ok(r) => {
                println!(
                    "  [{}] Order {} -> {:?} ({}ms)",
                    r.venue, r.order_id, r.status, r.latency_ms
                );
            }
            Err(e) => {
                println!("  [ERROR] {}", e);
            }
        }
    }
    println!("Total submit time: {:?} (parallel)", submit_time);

    // Example 3: Rate-limited batch orders
    println!("\n--- Example 3: Rate-Limited Batch Orders ---");

    let batch_orders: Vec<_> = (0..20)
        .map(|i| {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            let price = dec!(50000) + Decimal::from(i as i64 * 10);
            OrderRequest::limit("BTC-USDC", side, dec!(0.05), price)
        })
        .collect();

    println!(
        "Submitting {} orders with max 5 concurrent...",
        batch_orders.len()
    );
    let start = Instant::now();
    let results = rate_limited_orders(batch_orders, "lx_dex", 5, Arc::clone(&stats)).await;
    let batch_time = start.elapsed();

    let successful = results.iter().filter(|r| r.is_ok()).count();
    println!("Completed: {}/{} in {:?}", successful, results.len(), batch_time);

    // Example 4: Stream-based order processing
    println!("\n--- Example 4: Stream-Based Order Processing ---");

    let (tx, rx) = mpsc::channel(100);
    let stats_clone = Arc::clone(&stats);

    // Spawn order processor
    let processor = tokio::spawn(async move {
        process_order_stream(rx, "lx_dex".to_string(), stats_clone, 3).await;
    });

    // Send orders through stream
    println!("Sending 10 orders through stream processor...");
    for i in 0..10 {
        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
        let order = OrderRequest::market("ETH-USDC", side, dec!(0.5));
        tx.send(order).await?;
    }

    // Close channel and wait for completion
    drop(tx);
    let _ = timeout(Duration::from_secs(5), processor).await;

    // Example 5: Timeout handling
    println!("\n--- Example 5: Timeout Handling ---");

    let slow_order = async {
        tokio::time::sleep(Duration::from_secs(10)).await;
        OrderResult {
            venue: "slow_venue".to_string(),
            order_id: "timeout-test".to_string(),
            symbol: "BTC-USDC".to_string(),
            side: Side::Buy,
            quantity: dec!(1.0),
            price: dec!(50000),
            status: OrderStatus::Filled,
            latency_ms: 10000,
        }
    };

    match timeout(Duration::from_millis(100), slow_order).await {
        Ok(result) => {
            println!("Order completed: {:?}", result.status);
        }
        Err(_) => {
            println!("Order timed out after 100ms (expected)");
        }
    }

    // Print final statistics
    stats.print_summary();

    println!("\n=== Example Complete ===");
    println!("\nKey patterns demonstrated:");
    println!("1. join_all() for parallel execution");
    println!("2. Semaphore for rate limiting");
    println!("3. mpsc channels for stream processing");
    println!("4. timeout() for deadline enforcement");
    println!("5. Arc for shared state across tasks");

    Ok(())
}
