// LX Trading SDK - Orderbook Demo
// Demonstrates lock-free orderbook and aggregation

#include <lx/trading/orderbook.hpp>
#include <lx/trading/math.hpp>
#include <iostream>
#include <thread>
#include <vector>

using namespace lx::trading;
using namespace lx::trading::math;

void print_orderbook(const Orderbook& book) {
    std::cout << "\n=== " << book.symbol() << " @ " << book.venue() << " ===\n";
    std::cout << "Timestamp: " << book.timestamp() << "\n";

    auto bids = book.bids();
    auto asks = book.asks();

    std::cout << "\nAsks:\n";
    for (auto it = asks.rbegin(); it != asks.rend(); ++it) {
        std::cout << "  " << it->price.to_string() << " x " << it->quantity.to_string() << "\n";
    }

    std::cout << "---\n";

    std::cout << "Bids:\n";
    for (const auto& bid : bids) {
        std::cout << "  " << bid.price.to_string() << " x " << bid.quantity.to_string() << "\n";
    }

    if (auto mid = book.mid_price()) {
        std::cout << "\nMid: " << mid->to_string() << "\n";
    }
    if (auto spread = book.spread_percent()) {
        std::cout << "Spread: " << spread->to_string() << "%\n";
    }
}

void concurrent_update_demo() {
    std::cout << "\n=== Concurrent Update Demo ===\n";

    Orderbook book("BTC-USDC", "test");

    // Multiple threads updating
    std::vector<std::thread> writers;

    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&book, t]() {
            for (int i = 0; i < 100; ++i) {
                double price = 100.0 + t * 10 + (i % 10);
                double qty = 1.0 + (i % 5) * 0.1;

                if (t % 2 == 0) {
                    book.add_bid(Decimal::from_double(price), Decimal::from_double(qty));
                } else {
                    book.add_ask(Decimal::from_double(price + 50), Decimal::from_double(qty));
                }
            }
        });
    }

    // Reader thread
    std::thread reader([&book]() {
        for (int i = 0; i < 10; ++i) {
            auto bids = book.bids();
            auto asks = book.asks();
            std::cout << "Read: " << bids.size() << " bids, " << asks.size() << " asks\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    for (auto& w : writers) w.join();
    reader.join();

    book.sort();
    std::cout << "Final: " << book.bids().size() << " bids, " << book.asks().size() << " asks\n";
}

void aggregation_demo() {
    std::cout << "\n=== Aggregation Demo ===\n";

    // Simulate orderbooks from multiple venues
    Orderbook binance("BTC-USDC", "binance");
    binance.add_bid(Decimal::from_double(50000.0), Decimal::from_double(1.5));
    binance.add_bid(Decimal::from_double(49990.0), Decimal::from_double(2.0));
    binance.add_ask(Decimal::from_double(50010.0), Decimal::from_double(1.0));
    binance.add_ask(Decimal::from_double(50020.0), Decimal::from_double(1.5));
    binance.sort();

    Orderbook lx_dex("BTC-USDC", "lx_dex");
    lx_dex.add_bid(Decimal::from_double(50005.0), Decimal::from_double(0.8));  // Best bid!
    lx_dex.add_bid(Decimal::from_double(49995.0), Decimal::from_double(1.2));
    lx_dex.add_ask(Decimal::from_double(50008.0), Decimal::from_double(0.5));  // Best ask!
    lx_dex.add_ask(Decimal::from_double(50015.0), Decimal::from_double(1.0));
    lx_dex.sort();

    print_orderbook(binance);
    print_orderbook(lx_dex);

    // Aggregate
    AggregatedOrderbook agg("BTC-USDC");
    agg.add_orderbook(binance);
    agg.add_orderbook(lx_dex);

    std::cout << "\n=== Aggregated ===\n";

    if (auto best = agg.best_bid()) {
        auto [price, venue, qty] = *best;
        std::cout << "Best bid: " << price.to_string() << " @ " << venue << "\n";
    }

    if (auto best = agg.best_ask()) {
        auto [price, venue, qty] = *best;
        std::cout << "Best ask: " << price.to_string() << " @ " << venue << "\n";
    }

    // Best venue for buying 1 BTC
    if (auto best = agg.best_venue_buy(Decimal::from_double(1.0))) {
        auto [venue, price] = *best;
        std::cout << "Buy 1 BTC best at: " << venue << " @ " << price.to_string() << "\n";
    }

    // Best venue for selling 1 BTC
    if (auto best = agg.best_venue_sell(Decimal::from_double(1.0))) {
        auto [venue, price] = *best;
        std::cout << "Sell 1 BTC best at: " << venue << " @ " << price.to_string() << "\n";
    }

    std::cout << "\nAggregated asks:\n";
    for (const auto& level : agg.aggregated_asks()) {
        std::cout << "  " << level.price.to_string() << " x " << level.quantity.to_string() << "\n";
    }
}

void vwap_demo() {
    std::cout << "\n=== VWAP Demo ===\n";

    Orderbook book("ETH-USDC", "test");

    // Create realistic orderbook
    book.add_ask(Decimal::from_double(2000.0), Decimal::from_double(5.0));
    book.add_ask(Decimal::from_double(2001.0), Decimal::from_double(10.0));
    book.add_ask(Decimal::from_double(2002.0), Decimal::from_double(15.0));
    book.add_ask(Decimal::from_double(2005.0), Decimal::from_double(20.0));
    book.add_ask(Decimal::from_double(2010.0), Decimal::from_double(50.0));
    book.sort();

    // Calculate VWAP for different sizes
    std::cout << "VWAP for buying:\n";
    for (double size : {1.0, 5.0, 10.0, 25.0, 50.0}) {
        auto vwap = book.vwap_buy(Decimal::from_double(size));
        if (vwap) {
            std::cout << "  " << size << " ETH: " << vwap->to_string() << "\n";
        } else {
            std::cout << "  " << size << " ETH: insufficient liquidity\n";
        }
    }
}

void amm_math_demo() {
    std::cout << "\n=== AMM Math Demo ===\n";

    // Constant product (Uniswap V2)
    double reserve_eth = 1000.0;
    double reserve_usdc = 2000000.0;  // Price = 2000

    std::cout << "Constant Product Pool:\n";
    std::cout << "ETH reserve: " << reserve_eth << "\n";
    std::cout << "USDC reserve: " << reserve_usdc << "\n";
    std::cout << "Implied price: " << (reserve_usdc / reserve_eth) << "\n\n";

    for (double amt : {1.0, 10.0, 50.0, 100.0}) {
        auto [out, price] = constant_product_price(reserve_eth, reserve_usdc, amt, 0.003, true);
        double slippage = (2000.0 - price * 1000) / 2000.0 * 100;
        std::cout << "Buy " << amt << " ETH -> " << out << " USDC"
                  << " (price: " << (out / amt) << ", slippage: " << slippage << "%)\n";
    }
}

int main() {
    std::cout << "LX Trading SDK - Orderbook Demo\n";
    std::cout << "================================\n";

    concurrent_update_demo();
    aggregation_demo();
    vwap_demo();
    amm_math_demo();

    return 0;
}
