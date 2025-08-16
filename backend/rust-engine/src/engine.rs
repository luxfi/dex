use crate::orderbook::OrderBook;
use crate::types::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

pub struct Engine {
    orderbooks: Arc<RwLock<HashMap<String, Arc<OrderBook>>>>,
    order_counter: Arc<RwLock<u64>>,
}

impl Engine {
    pub fn new() -> Self {
        info!("Initializing Rust engine");
        Engine {
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
            order_counter: Arc::new(RwLock::new(0)),
        }
    }

    pub fn get_or_create_orderbook(&self, symbol: &str) -> Arc<OrderBook> {
        let mut books = self.orderbooks.write();
        books
            .entry(symbol.to_string())
            .or_insert_with(|| {
                info!("Creating new orderbook for symbol: {}", symbol);
                Arc::new(OrderBook::new(symbol.to_string()))
            })
            .clone()
    }

    pub fn submit_order(
        &self,
        user_id: u64,
        symbol: String,
        side: Side,
        order_type: OrderType,
        price: rust_decimal::Decimal,
        quantity: rust_decimal::Decimal,
    ) -> (Order, Vec<Trade>) {
        let mut order_counter = self.order_counter.write();
        *order_counter += 1;
        let order_id = OrderId(*order_counter);
        drop(order_counter);

        let order = Order::new(
            order_id,
            UserId(user_id),
            symbol.clone(),
            side,
            order_type,
            price,
            quantity,
        );

        let orderbook = self.get_or_create_orderbook(&symbol);
        let trades = orderbook.add_order(order.clone());

        info!(
            "Order {} submitted: {} {} {} @ {} - {} trades executed",
            order_id.0,
            if side == Side::Buy { "BUY" } else { "SELL" },
            quantity,
            symbol,
            price,
            trades.len()
        );

        (order, trades)
    }

    pub fn cancel_order(&self, symbol: &str, order_id: u64) -> Option<Order> {
        let orderbook = self.get_or_create_orderbook(symbol);
        let result = orderbook.cancel_order(OrderId(order_id));
        
        if result.is_some() {
            info!("Order {} cancelled", order_id);
        } else {
            warn!("Failed to cancel order {} - not found", order_id);
        }
        
        result
    }

    pub fn get_orderbook_snapshot(&self, symbol: &str, depth: usize) -> OrderBookSnapshot {
        let orderbook = self.get_or_create_orderbook(symbol);
        orderbook.get_snapshot(depth)
    }

    pub fn get_best_bid(&self, symbol: &str) -> Option<rust_decimal::Decimal> {
        let orderbook = self.get_or_create_orderbook(symbol);
        orderbook.get_best_bid()
    }

    pub fn get_best_ask(&self, symbol: &str) -> Option<rust_decimal::Decimal> {
        let orderbook = self.get_or_create_orderbook(symbol);
        orderbook.get_best_ask()
    }

    pub fn get_spread(&self, symbol: &str) -> Option<rust_decimal::Decimal> {
        let orderbook = self.get_or_create_orderbook(symbol);
        orderbook.get_spread()
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}