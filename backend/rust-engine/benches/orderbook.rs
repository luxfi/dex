use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lx_engine::{OrderBook, Order, OrderId, UserId, Side, OrderType};
use rust_decimal::prelude::*;
use std::sync::Arc;

fn benchmark_order_submission(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_submission");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let orderbook = Arc::new(OrderBook::new("BTC-USD".to_string()));
            let mut order_id = 0u64;
            
            // Pre-populate orderbook
            for i in 0..size/2 {
                order_id += 1;
                let price = Decimal::from(50000) + Decimal::from(i);
                let order = Order::new(
                    OrderId(order_id),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Buy,
                    OrderType::Limit,
                    price,
                    Decimal::from(1),
                );
                orderbook.add_order(order);
            }
            
            for i in 0..size/2 {
                order_id += 1;
                let price = Decimal::from(51000) + Decimal::from(i);
                let order = Order::new(
                    OrderId(order_id),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Sell,
                    OrderType::Limit,
                    price,
                    Decimal::from(1),
                );
                orderbook.add_order(order);
            }
            
            b.iter(|| {
                order_id += 1;
                let order = Order::new(
                    OrderId(order_id),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Buy,
                    OrderType::Limit,
                    Decimal::from(50500),
                    Decimal::from(1),
                );
                black_box(orderbook.add_order(order));
            });
        });
    }
    
    group.finish();
}

fn benchmark_order_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_matching");
    
    group.bench_function("market_order_full_match", |b| {
        let orderbook = Arc::new(OrderBook::new("BTC-USD".to_string()));
        let mut order_id = 0u64;
        
        b.iter(|| {
            // Add sell orders
            for i in 0..10 {
                order_id += 1;
                let order = Order::new(
                    OrderId(order_id),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Sell,
                    OrderType::Limit,
                    Decimal::from(50000) + Decimal::from(i),
                    Decimal::from(1),
                );
                orderbook.add_order(order);
            }
            
            // Submit market buy order
            order_id += 1;
            let order = Order::new(
                OrderId(order_id),
                UserId(2),
                "BTC-USD".to_string(),
                Side::Buy,
                OrderType::Market,
                Decimal::from(60000),
                Decimal::from(10),
            );
            black_box(orderbook.add_order(order));
        });
    });
    
    group.finish();
}

fn benchmark_orderbook_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_snapshot");
    
    for depth in [10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let orderbook = Arc::new(OrderBook::new("BTC-USD".to_string()));
            
            // Populate orderbook
            for i in 0..100 {
                let order = Order::new(
                    OrderId(i * 2),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Buy,
                    OrderType::Limit,
                    Decimal::from(50000) - Decimal::from(i),
                    Decimal::from(1),
                );
                orderbook.add_order(order);
                
                let order = Order::new(
                    OrderId(i * 2 + 1),
                    UserId(1),
                    "BTC-USD".to_string(),
                    Side::Sell,
                    OrderType::Limit,
                    Decimal::from(51000) + Decimal::from(i),
                    Decimal::from(1),
                );
                orderbook.add_order(order);
            }
            
            b.iter(|| {
                black_box(orderbook.get_snapshot(depth));
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_order_submission,
    benchmark_order_matching,
    benchmark_orderbook_snapshot
);
criterion_main!(benches);