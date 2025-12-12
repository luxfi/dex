//! Unified Liquidity Arbitrage.
//!
//! Since LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks),
//! it becomes the price ORACLE. Other venues are always stale by comparison.
//!
//! Architecture:
//! 1. LX DEX prices are the TRUTH (most current)
//! 2. Other venues (CEX, external DEX) are STALE
//! 3. Arbitrage = exploiting stale venues before they catch up
//! 4. LX always wins because it sees/moves prices first
//!
//! NO SMART CONTRACTS - just coordinated trades through unified SDK.

use crate::arbitrage::types::*;
use crate::{Order, OrderRequest, Side};
use async_trait::async_trait;
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// Aggregated orderbook level.
#[derive(Debug, Clone)]
pub struct AggregatedLevel {
    pub price: Decimal,
    pub quantity: Decimal,
    pub venue: String,
    pub timestamp: i64,
}

/// Aggregated orderbook from all venues.
#[derive(Debug, Clone)]
pub struct AggregatedBook {
    pub symbol: String,
    pub bids: Vec<AggregatedLevel>,
    pub asks: Vec<AggregatedLevel>,
}

/// Trading client interface for arbitrage.
#[async_trait]
pub trait TradingClient: Send + Sync {
    /// Get aggregated orderbook from all venues.
    async fn aggregated_orderbook(&self, symbol: &str) -> Result<AggregatedBook, String>;

    /// Place an order on a specific venue.
    async fn place_order(&self, request: OrderRequest) -> Result<Order, String>;
}

/// Callback type for opportunity events.
pub type UnifiedCallback = Box<dyn Fn(UnifiedOpportunity) + Send + Sync>;

/// Unified arbitrage across all SDK-connected venues.
pub struct UnifiedArbitrage<C: TradingClient> {
    client: Arc<C>,
    config: UnifiedArbConfig,
    total_pnl: Arc<RwLock<Decimal>>,
    executions: Arc<RwLock<Vec<UnifiedExecution>>>,
    callbacks: Arc<RwLock<Vec<UnifiedCallback>>>,
    opportunity_queue: Arc<RwLock<VecDeque<UnifiedOpportunity>>>,
    running: Arc<RwLock<bool>>,
    scan_task: Option<JoinHandle<()>>,
    execute_task: Option<JoinHandle<()>>,
}

impl<C: TradingClient + 'static> UnifiedArbitrage<C> {
    /// Create a new unified arbitrage instance.
    pub fn new(client: C, config: UnifiedArbConfig) -> Self {
        Self {
            client: Arc::new(client),
            config,
            total_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            executions: Arc::new(RwLock::new(Vec::new())),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            opportunity_queue: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            running: Arc::new(RwLock::new(false)),
            scan_task: None,
            execute_task: None,
        }
    }

    /// Start the arbitrage system.
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

        // Start scan task
        let client = self.client.clone();
        let config = self.config.clone();
        let running = self.running.clone();
        let callbacks = self.callbacks.clone();
        let opportunity_queue = self.opportunity_queue.clone();

        let scan_task = tokio::spawn(async move {
            scan_loop(client, config, running, callbacks, opportunity_queue).await;
        });
        self.scan_task = Some(scan_task);

        // Start execute task
        let client = self.client.clone();
        let config = self.config.clone();
        let running = self.running.clone();
        let opportunity_queue = self.opportunity_queue.clone();
        let total_pnl = self.total_pnl.clone();
        let executions = self.executions.clone();

        let execute_task = tokio::spawn(async move {
            execute_loop(client, config, running, opportunity_queue, total_pnl, executions).await;
        });
        self.execute_task = Some(execute_task);
    }

    /// Stop the arbitrage system.
    pub async fn stop(&mut self) {
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        if let Some(task) = self.scan_task.take() {
            task.abort();
            let _ = task.await;
        }

        if let Some(task) = self.execute_task.take() {
            task.abort();
            let _ = task.await;
        }
    }

    /// Subscribe to opportunity events.
    pub async fn on_opportunity(&self, callback: UnifiedCallback) {
        let mut callbacks = self.callbacks.write().await;
        callbacks.push(callback);
    }

    /// Get arbitrage statistics.
    pub async fn get_stats(&self) -> UnifiedArbStats {
        let executions = self.executions.read().await;
        let total_pnl = *self.total_pnl.read().await;

        let successful = executions
            .iter()
            .filter(|e| e.status == "completed" && e.actual_profit > Decimal::ZERO)
            .count() as u64;

        let win_rate = if executions.is_empty() {
            0.0
        } else {
            successful as f64 / executions.len() as f64
        };

        UnifiedArbStats {
            total_executions: executions.len() as u64,
            successful_executions: successful,
            total_pnl,
            win_rate,
        }
    }

    /// Get current config.
    pub fn config(&self) -> &UnifiedArbConfig {
        &self.config
    }
}

async fn scan_loop<C: TradingClient>(
    client: Arc<C>,
    config: UnifiedArbConfig,
    running: Arc<RwLock<bool>>,
    callbacks: Arc<RwLock<Vec<UnifiedCallback>>>,
    opportunity_queue: Arc<RwLock<VecDeque<UnifiedOpportunity>>>,
) {
    let interval = std::time::Duration::from_millis(config.scan_interval_ms);

    loop {
        {
            let is_running = running.read().await;
            if !*is_running {
                break;
            }
        }

        // Scan each symbol
        for symbol in &config.symbols {
            if let Some(opp) = find_opportunity(&client, symbol, &config).await {
                if opp.net_profit > config.min_profit {
                    // Add to queue
                    {
                        let mut queue = opportunity_queue.write().await;
                        if queue.len() < 1000 {
                            queue.push_back(opp.clone());
                        }
                    }

                    // Emit to callbacks
                    {
                        let cbs = callbacks.read().await;
                        for cb in cbs.iter() {
                            cb(opp.clone());
                        }
                    }
                }
            }
        }

        tokio::time::sleep(interval).await;
    }
}

async fn find_opportunity<C: TradingClient>(
    client: &Arc<C>,
    symbol: &str,
    config: &UnifiedArbConfig,
) -> Option<UnifiedOpportunity> {
    let book = match client.aggregated_orderbook(symbol).await {
        Ok(b) => b,
        Err(_) => return None,
    };

    if book.bids.is_empty() || book.asks.is_empty() {
        return None;
    }

    let best_bid = &book.bids[0];
    let best_ask = &book.asks[0];

    // Cross-venue arbitrage: bid on one venue > ask on another
    if best_bid.price <= best_ask.price {
        return None;
    }

    let spread = best_bid.price - best_ask.price;
    let spread_bps = (spread / best_ask.price) * Decimal::from(10000);

    if spread_bps < config.min_spread_bps {
        return None;
    }

    let max_size = best_bid
        .quantity
        .min(best_ask.quantity)
        .min(config.max_position_size);

    let gross_profit = spread * max_size;
    let total_fees = best_ask.price * max_size * Decimal::new(2, 3); // ~0.2% total fees
    let net_profit = gross_profit - total_fees;

    let now = current_time_ms();

    Some(UnifiedOpportunity {
        id: format!("arb-{}-{}", symbol, now),
        symbol: symbol.to_string(),
        timestamp: now,
        expires_at: now + 5000,
        buy_venue: best_ask.venue.clone(),
        buy_price: best_ask.price,
        buy_size: best_ask.quantity,
        sell_venue: best_bid.venue.clone(),
        sell_price: best_bid.price,
        sell_size: best_bid.quantity,
        spread,
        spread_bps,
        max_size,
        gross_profit,
        est_fees: total_fees,
        net_profit,
        confidence: 0.8,
        latency: now - best_ask.timestamp,
    })
}

async fn execute_loop<C: TradingClient>(
    client: Arc<C>,
    _config: UnifiedArbConfig,
    running: Arc<RwLock<bool>>,
    opportunity_queue: Arc<RwLock<VecDeque<UnifiedOpportunity>>>,
    total_pnl: Arc<RwLock<Decimal>>,
    executions: Arc<RwLock<Vec<UnifiedExecution>>>,
) {
    loop {
        {
            let is_running = running.read().await;
            if !*is_running {
                break;
            }
        }

        // Get next opportunity
        let opp = {
            let mut queue = opportunity_queue.write().await;
            queue.pop_front()
        };

        if let Some(opp) = opp {
            let result = execute_opportunity(&client, opp).await;

            // Update stats
            {
                let mut pnl = total_pnl.write().await;
                *pnl += result.actual_profit;
            }
            {
                let mut execs = executions.write().await;
                execs.push(result);
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

async fn execute_opportunity<C: TradingClient>(
    client: &Arc<C>,
    opp: UnifiedOpportunity,
) -> UnifiedExecution {
    let now = current_time_ms();

    // Check if expired
    if now > opp.expires_at {
        return UnifiedExecution {
            id: opp.id.clone(),
            opportunity: opp,
            start_time: now,
            end_time: now,
            status: "expired".to_string(),
            buy_order_id: None,
            sell_order_id: None,
            actual_profit: Decimal::ZERO,
            fees: Decimal::ZERO,
            error: Some("Opportunity expired".to_string()),
        };
    }

    let mut exec_result = UnifiedExecution {
        id: opp.id.clone(),
        opportunity: opp.clone(),
        start_time: now,
        end_time: 0,
        status: "executing".to_string(),
        buy_order_id: None,
        sell_order_id: None,
        actual_profit: Decimal::ZERO,
        fees: Decimal::ZERO,
        error: None,
    };

    // Execute both legs simultaneously
    let buy_request = OrderRequest::limit(&opp.symbol, Side::Buy, opp.max_size, opp.buy_price)
        .with_venue(&opp.buy_venue);

    let sell_request = OrderRequest::limit(&opp.symbol, Side::Sell, opp.max_size, opp.sell_price)
        .with_venue(&opp.sell_venue);

    let (buy_result, sell_result) =
        tokio::join!(client.place_order(buy_request), client.place_order(sell_request));

    exec_result.end_time = current_time_ms();

    match (buy_result, sell_result) {
        (Ok(buy_order), Ok(sell_order)) => {
            exec_result.buy_order_id = Some(buy_order.order_id.clone());
            exec_result.sell_order_id = Some(sell_order.order_id.clone());

            // Calculate actual profit
            if let (Some(buy_price), Some(sell_price)) =
                (buy_order.average_price, sell_order.average_price)
            {
                let buy_value = buy_price * buy_order.filled_quantity;
                let sell_value = sell_price * sell_order.filled_quantity;
                exec_result.actual_profit = sell_value - buy_value;

                // Subtract fees
                let buy_fees: Decimal = buy_order.fees.iter().map(|f| f.amount).sum();
                let sell_fees: Decimal = sell_order.fees.iter().map(|f| f.amount).sum();
                exec_result.fees = buy_fees + sell_fees;
                exec_result.actual_profit -= exec_result.fees;
            }

            exec_result.status = "completed".to_string();
        }
        (Err(e), _) | (_, Err(e)) => {
            exec_result.status = "failed".to_string();
            exec_result.error = Some(e);
        }
    }

    exec_result
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
    use crate::Fee;

    struct MockClient;

    #[async_trait]
    impl TradingClient for MockClient {
        async fn aggregated_orderbook(&self, symbol: &str) -> Result<AggregatedBook, String> {
            Ok(AggregatedBook {
                symbol: symbol.to_string(),
                bids: vec![AggregatedLevel {
                    price: Decimal::from(50100), // Higher bid
                    quantity: Decimal::from(10),
                    venue: "binance".to_string(),
                    timestamp: current_time_ms(),
                }],
                asks: vec![AggregatedLevel {
                    price: Decimal::from(50000), // Lower ask
                    quantity: Decimal::from(10),
                    venue: "lx_dex".to_string(),
                    timestamp: current_time_ms(),
                }],
            })
        }

        async fn place_order(&self, request: OrderRequest) -> Result<Order, String> {
            Ok(Order {
                order_id: "test-order".to_string(),
                client_order_id: request.client_order_id,
                symbol: request.symbol,
                venue: request.venue.unwrap_or_default(),
                side: request.side,
                order_type: request.order_type,
                status: crate::OrderStatus::Filled,
                quantity: request.quantity,
                filled_quantity: request.quantity,
                remaining_quantity: Decimal::ZERO,
                price: request.price,
                average_price: request.price,
                created_at: current_time_ms(),
                updated_at: current_time_ms(),
                fees: vec![Fee {
                    asset: "USDC".to_string(),
                    amount: Decimal::new(1, 1), // 0.1
                    rate: None,
                }],
            })
        }
    }

    #[tokio::test]
    async fn test_unified_arbitrage_creation() {
        let client = MockClient;
        let config = UnifiedArbConfig::default();
        let arb = UnifiedArbitrage::new(client, config);

        let stats = arb.get_stats().await;
        assert_eq!(stats.total_executions, 0);
    }

    #[tokio::test]
    async fn test_find_opportunity() {
        let client = Arc::new(MockClient);
        let config = UnifiedArbConfig {
            min_spread_bps: Decimal::from(10),
            min_profit: Decimal::from(1),
            ..Default::default()
        };

        let opp = find_opportunity(&client, "BTC-USDC", &config).await;
        assert!(opp.is_some());

        let opp = opp.unwrap();
        assert_eq!(opp.buy_venue, "lx_dex");
        assert_eq!(opp.sell_venue, "binance");
        assert!(opp.spread > Decimal::ZERO);
    }
}
