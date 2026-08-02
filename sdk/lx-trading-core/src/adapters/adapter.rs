//! Core adapter trait and capabilities.

use async_trait::async_trait;
use rust_decimal::Decimal;
use std::collections::HashSet;

use crate::error::Result;
use crate::orderbook::Orderbook;
use crate::types::*;

/// Capabilities that a venue may support
#[derive(Debug, Clone, Default)]
pub struct VenueCapabilities {
    /// Supports limit orders
    pub limit_orders: bool,
    /// Supports market orders
    pub market_orders: bool,
    /// Supports stop orders
    pub stop_orders: bool,
    /// Supports post-only/maker-only orders
    pub post_only: bool,
    /// Supports order cancellation
    pub cancel_orders: bool,
    /// Supports batch order operations
    pub batch_orders: bool,
    /// Supports real-time WebSocket streaming
    pub streaming: bool,
    /// Supports orderbook snapshots
    pub orderbook: bool,
    /// Supports trade history
    pub trades: bool,
    /// Supports AMM swaps
    pub amm_swap: bool,
    /// Supports adding liquidity
    pub add_liquidity: bool,
    /// Supports removing liquidity
    pub remove_liquidity: bool,
    /// Supports LP position tracking
    pub lp_positions: bool,
    /// Maximum orders per batch
    pub max_batch_size: usize,
    /// Supported trading pairs
    pub supported_pairs: HashSet<String>,
}

impl VenueCapabilities {
    /// Create capabilities for a typical OrderBook/orderbook venue
    pub fn order_book() -> Self {
        Self {
            limit_orders: true,
            market_orders: true,
            stop_orders: true,
            post_only: true,
            cancel_orders: true,
            batch_orders: true,
            streaming: true,
            orderbook: true,
            trades: true,
            amm_swap: false,
            add_liquidity: false,
            remove_liquidity: false,
            lp_positions: false,
            max_batch_size: 10,
            supported_pairs: HashSet::new(),
        }
    }

    /// Create capabilities for a typical AMM venue
    pub fn amm() -> Self {
        Self {
            limit_orders: false,
            market_orders: true, // via swap
            stop_orders: false,
            post_only: false,
            cancel_orders: false,
            batch_orders: false,
            streaming: true,
            orderbook: false,
            trades: true,
            amm_swap: true,
            add_liquidity: true,
            remove_liquidity: true,
            lp_positions: true,
            max_batch_size: 1,
            supported_pairs: HashSet::new(),
        }
    }

    /// Create capabilities for a hybrid venue (OrderBook + AMM)
    pub fn hybrid() -> Self {
        Self {
            limit_orders: true,
            market_orders: true,
            stop_orders: true,
            post_only: true,
            cancel_orders: true,
            batch_orders: true,
            streaming: true,
            orderbook: true,
            trades: true,
            amm_swap: true,
            add_liquidity: true,
            remove_liquidity: true,
            lp_positions: true,
            max_batch_size: 10,
            supported_pairs: HashSet::new(),
        }
    }
}

/// Unified adapter interface for all trading venues.
///
/// This trait provides a consistent API regardless of whether the underlying
/// venue is:
/// - Native LX DEX (OrderBook)
/// - Native LX AMM (liquidity pools)
/// - CCXT exchange (Binance, MEXC, OKX, etc.)
/// - Hummingbot Gateway connector
///
/// Not all methods are supported by all venues. Check `capabilities()` before
/// calling venue-specific methods.
#[async_trait]
pub trait VenueAdapter: Send + Sync {
    // =========================================================================
    // Identity & Status
    // =========================================================================

    /// Get venue name/identifier
    fn name(&self) -> &str;

    /// Get venue type
    fn venue_type(&self) -> VenueType;

    /// Get venue capabilities
    fn capabilities(&self) -> &VenueCapabilities;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Get connection latency in milliseconds
    fn latency_ms(&self) -> Option<u64>;

    // =========================================================================
    // Connection
    // =========================================================================

    /// Connect to the venue
    async fn connect(&mut self) -> Result<()>;

    /// Disconnect from the venue
    async fn disconnect(&mut self) -> Result<()>;

    /// Start streaming updates (orderbook, trades, etc.)
    async fn start_streaming(&mut self, symbols: &[String]) -> Result<()>;

    /// Stop streaming
    async fn stop_streaming(&mut self) -> Result<()>;

    // =========================================================================
    // Market Data
    // =========================================================================

    /// Get supported trading pairs
    async fn get_markets(&self) -> Result<Vec<MarketInfo>>;

    /// Get ticker for a symbol
    async fn get_ticker(&self, symbol: &str) -> Result<Ticker>;

    /// Get tickers for multiple symbols
    async fn get_tickers(&self, symbols: &[String]) -> Result<Vec<Ticker>>;

    /// Get orderbook snapshot
    async fn get_orderbook(&self, symbol: &str, depth: Option<usize>) -> Result<Orderbook>;

    /// Get recent trades
    async fn get_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>>;

    // =========================================================================
    // Account Data
    // =========================================================================

    /// Get all balances
    async fn get_balances(&self) -> Result<Vec<Balance>>;

    /// Get balance for specific asset
    async fn get_balance(&self, asset: &str) -> Result<Balance>;

    /// Get open orders
    async fn get_open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>>;

    /// Get order by ID
    async fn get_order(&self, order_id: &str, symbol: &str) -> Result<Order>;

    /// Get order history
    async fn get_order_history(
        &self,
        symbol: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Order>>;

    // =========================================================================
    // Order Management
    // =========================================================================

    /// Place a single order
    async fn place_order(&self, request: OrderRequest) -> Result<Order>;

    /// Place multiple orders (batch)
    async fn place_orders(&self, requests: Vec<OrderRequest>) -> Result<Vec<Order>>;

    /// Cancel an order
    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<Order>;

    /// Cancel multiple orders
    async fn cancel_orders(&self, order_ids: &[(String, String)]) -> Result<Vec<Order>>;

    /// Cancel all orders for a symbol (or all if None)
    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>>;

    // =========================================================================
    // AMM Operations (optional - check capabilities)
    // =========================================================================

    /// Get swap quote
    async fn get_swap_quote(
        &self,
        _base_token: &str,
        _quote_token: &str,
        _amount: Decimal,
        _is_buy: bool,
    ) -> Result<SwapQuote> {
        Err(crate::error::Error::NotImplemented(
            "AMM swap not supported".into(),
        ))
    }

    /// Execute swap
    async fn execute_swap(
        &self,
        _base_token: &str,
        _quote_token: &str,
        _amount: Decimal,
        _is_buy: bool,
        _slippage_percent: Decimal,
    ) -> Result<Trade> {
        Err(crate::error::Error::NotImplemented(
            "AMM swap not supported".into(),
        ))
    }

    /// Get pool information
    async fn get_pool_info(&self, _base_token: &str, _quote_token: &str) -> Result<PoolInfo> {
        Err(crate::error::Error::NotImplemented(
            "Pool info not supported".into(),
        ))
    }

    /// Add liquidity to pool
    async fn add_liquidity(
        &self,
        _base_token: &str,
        _quote_token: &str,
        _base_amount: Decimal,
        _quote_amount: Decimal,
        _slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        Err(crate::error::Error::NotImplemented(
            "Add liquidity not supported".into(),
        ))
    }

    /// Remove liquidity from pool
    async fn remove_liquidity(
        &self,
        _pool_address: &str,
        _liquidity_amount: Decimal,
        _slippage_percent: Decimal,
    ) -> Result<LiquidityResult> {
        Err(crate::error::Error::NotImplemented(
            "Remove liquidity not supported".into(),
        ))
    }

    /// Get LP positions
    async fn get_lp_positions(&self) -> Result<Vec<LpPosition>> {
        Err(crate::error::Error::NotImplemented(
            "LP positions not supported".into(),
        ))
    }
}

/// Swap quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapQuote {
    pub base_token: String,
    pub quote_token: String,
    pub input_amount: Decimal,
    pub output_amount: Decimal,
    pub price: Decimal,
    pub price_impact: Decimal,
    pub fee: Decimal,
    pub route: Vec<String>,
    pub expires_at: i64,
}

/// Pool information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfo {
    pub address: String,
    pub base_token: String,
    pub quote_token: String,
    pub base_reserve: Decimal,
    pub quote_reserve: Decimal,
    pub total_liquidity: Decimal,
    pub fee_rate: Decimal,
    pub apy: Option<Decimal>,
}

/// Result of liquidity operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityResult {
    pub tx_hash: String,
    pub pool_address: String,
    pub base_amount: Decimal,
    pub quote_amount: Decimal,
    pub lp_tokens: Decimal,
    pub share_percent: Decimal,
}

/// LP position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpPosition {
    pub pool_address: String,
    pub base_token: String,
    pub quote_token: String,
    pub lp_tokens: Decimal,
    pub base_amount: Decimal,
    pub quote_amount: Decimal,
    pub share_percent: Decimal,
    pub unrealized_pnl: Option<Decimal>,
}

use serde::{Deserialize, Serialize};
