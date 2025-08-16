use crate::types::*;
use indexmap::IndexMap;
use parking_lot::RwLock;
use priority_queue::PriorityQueue;
use rust_decimal::Decimal;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::{debug, trace};

pub struct OrderBook {
    symbol: String,
    orders: Arc<RwLock<HashMap<OrderId, Order>>>,
    bids: Arc<RwLock<BTreeMap<Decimal, Vec<OrderId>>>>,
    asks: Arc<RwLock<BTreeMap<Decimal, Vec<OrderId>>>>,
    trade_counter: Arc<RwLock<u64>>,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        OrderBook {
            symbol,
            orders: Arc::new(RwLock::new(HashMap::new())),
            bids: Arc::new(RwLock::new(BTreeMap::new())),
            asks: Arc::new(RwLock::new(BTreeMap::new())),
            trade_counter: Arc::new(RwLock::new(0)),
        }
    }

    pub fn add_order(&self, mut order: Order) -> Vec<Trade> {
        debug!("Adding order: {:?}", order);
        
        let mut trades = Vec::new();
        
        // Try to match the order first
        if order.order_type == OrderType::Limit || order.order_type == OrderType::Market {
            trades = self.match_order(&mut order);
        }
        
        // If order is not fully filled, add to book
        if !order.is_filled() && order.order_type == OrderType::Limit {
            let mut orders = self.orders.write();
            let order_id = order.id;
            let price = order.price;
            let side = order.side;
            
            orders.insert(order_id, order.clone());
            drop(orders);
            
            match side {
                Side::Buy => {
                    let mut bids = self.bids.write();
                    bids.entry(price).or_insert_with(Vec::new).push(order_id);
                }
                Side::Sell => {
                    let mut asks = self.asks.write();
                    asks.entry(price).or_insert_with(Vec::new).push(order_id);
                }
            }
        }
        
        trades
    }

    fn match_order(&self, order: &mut Order) -> Vec<Trade> {
        let mut trades = Vec::new();
        
        match order.side {
            Side::Buy => {
                let mut asks = self.asks.write();
                let mut orders = self.orders.write();
                
                let prices: Vec<Decimal> = asks.keys().cloned().collect();
                for price in prices {
                    if order.order_type == OrderType::Limit && price > order.price {
                        break;
                    }
                    
                    if let Some(order_ids) = asks.get_mut(&price) {
                        let mut i = 0;
                        while i < order_ids.len() && !order.is_filled() {
                            let counter_id = order_ids[i];
                            if let Some(counter_order) = orders.get_mut(&counter_id) {
                                let trade_qty = order.remaining_quantity()
                                    .min(counter_order.remaining_quantity());
                                
                                // Update orders
                                order.filled_quantity += trade_qty;
                                counter_order.filled_quantity += trade_qty;
                                
                                if order.is_filled() {
                                    order.status = OrderStatus::Filled;
                                }
                                if counter_order.is_filled() {
                                    counter_order.status = OrderStatus::Filled;
                                    order_ids.remove(i);
                                } else {
                                    counter_order.status = OrderStatus::PartiallyFilled;
                                    i += 1;
                                }
                                
                                // Create trade
                                let mut trade_counter = self.trade_counter.write();
                                *trade_counter += 1;
                                trades.push(Trade {
                                    id: *trade_counter,
                                    symbol: self.symbol.clone(),
                                    price,
                                    quantity: trade_qty,
                                    buy_order_id: order.id,
                                    sell_order_id: counter_id,
                                    timestamp: chrono::Utc::now().timestamp_nanos(),
                                });
                            } else {
                                order_ids.remove(i);
                            }
                        }
                        
                        if order_ids.is_empty() {
                            asks.remove(&price);
                        }
                    }
                    
                    if order.is_filled() {
                        break;
                    }
                }
            }
            Side::Sell => {
                let mut bids = self.bids.write();
                let mut orders = self.orders.write();
                
                let prices: Vec<Decimal> = bids.keys().rev().cloned().collect();
                for price in prices {
                    if order.order_type == OrderType::Limit && price < order.price {
                        break;
                    }
                    
                    if let Some(order_ids) = bids.get_mut(&price) {
                        let mut i = 0;
                        while i < order_ids.len() && !order.is_filled() {
                            let counter_id = order_ids[i];
                            if let Some(counter_order) = orders.get_mut(&counter_id) {
                                let trade_qty = order.remaining_quantity()
                                    .min(counter_order.remaining_quantity());
                                
                                // Update orders
                                order.filled_quantity += trade_qty;
                                counter_order.filled_quantity += trade_qty;
                                
                                if order.is_filled() {
                                    order.status = OrderStatus::Filled;
                                }
                                if counter_order.is_filled() {
                                    counter_order.status = OrderStatus::Filled;
                                    order_ids.remove(i);
                                } else {
                                    counter_order.status = OrderStatus::PartiallyFilled;
                                    i += 1;
                                }
                                
                                // Create trade
                                let mut trade_counter = self.trade_counter.write();
                                *trade_counter += 1;
                                trades.push(Trade {
                                    id: *trade_counter,
                                    symbol: self.symbol.clone(),
                                    price,
                                    quantity: trade_qty,
                                    buy_order_id: counter_id,
                                    sell_order_id: order.id,
                                    timestamp: chrono::Utc::now().timestamp_nanos(),
                                });
                            } else {
                                order_ids.remove(i);
                            }
                        }
                        
                        if order_ids.is_empty() {
                            bids.remove(&price);
                        }
                    }
                    
                    if order.is_filled() {
                        break;
                    }
                }
            }
        }
        
        trades
    }

    pub fn cancel_order(&self, order_id: OrderId) -> Option<Order> {
        let mut orders = self.orders.write();
        
        if let Some(mut order) = orders.remove(&order_id) {
            order.status = OrderStatus::Cancelled;
            
            // Remove from price level
            match order.side {
                Side::Buy => {
                    let mut bids = self.bids.write();
                    if let Some(order_ids) = bids.get_mut(&order.price) {
                        order_ids.retain(|&id| id != order_id);
                        if order_ids.is_empty() {
                            bids.remove(&order.price);
                        }
                    }
                }
                Side::Sell => {
                    let mut asks = self.asks.write();
                    if let Some(order_ids) = asks.get_mut(&order.price) {
                        order_ids.retain(|&id| id != order_id);
                        if order_ids.is_empty() {
                            asks.remove(&order.price);
                        }
                    }
                }
            }
            
            Some(order)
        } else {
            None
        }
    }

    pub fn get_snapshot(&self, depth: usize) -> OrderBookSnapshot {
        let bids = self.bids.read();
        let asks = self.asks.read();
        let orders = self.orders.read();
        
        let mut bid_levels = Vec::new();
        for (price, order_ids) in bids.iter().rev().take(depth) {
            let mut total_qty = Decimal::ZERO;
            for order_id in order_ids {
                if let Some(order) = orders.get(order_id) {
                    total_qty += order.remaining_quantity();
                }
            }
            bid_levels.push(PriceLevel {
                price: *price,
                quantity: total_qty,
                order_count: order_ids.len(),
            });
        }
        
        let mut ask_levels = Vec::new();
        for (price, order_ids) in asks.iter().take(depth) {
            let mut total_qty = Decimal::ZERO;
            for order_id in order_ids {
                if let Some(order) = orders.get(order_id) {
                    total_qty += order.remaining_quantity();
                }
            }
            ask_levels.push(PriceLevel {
                price: *price,
                quantity: total_qty,
                order_count: order_ids.len(),
            });
        }
        
        OrderBookSnapshot {
            symbol: self.symbol.clone(),
            bids: bid_levels,
            asks: ask_levels,
            timestamp: chrono::Utc::now().timestamp_nanos(),
        }
    }

    pub fn get_best_bid(&self) -> Option<Decimal> {
        self.bids.read().keys().rev().next().copied()
    }

    pub fn get_best_ask(&self) -> Option<Decimal> {
        self.asks.read().keys().next().copied()
    }

    pub fn get_spread(&self) -> Option<Decimal> {
        let best_bid = self.get_best_bid()?;
        let best_ask = self.get_best_ask()?;
        Some(best_ask - best_bid)
    }
}