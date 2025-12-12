// LX C++ SDK - Orderbook Interface
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#ifndef LXDEX_ORDERBOOK_HPP
#define LXDEX_ORDERBOOK_HPP

#include "types.hpp"
#include <map>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <atomic>

namespace lxdex {

/// Local orderbook for market data tracking
/// Thread-safe implementation for real-time updates
class LocalOrderBook {
public:
    explicit LocalOrderBook(std::string symbol);
    ~LocalOrderBook() = default;

    // Non-copyable, non-movable (contains atomic and mutex)
    LocalOrderBook(const LocalOrderBook&) = delete;
    LocalOrderBook& operator=(const LocalOrderBook&) = delete;
    LocalOrderBook(LocalOrderBook&&) = delete;
    LocalOrderBook& operator=(LocalOrderBook&&) = delete;

    /// Get symbol
    [[nodiscard]] const std::string& symbol() const noexcept { return symbol_; }

    /// Apply full snapshot
    void apply_snapshot(const OrderBook& snapshot);

    /// Update single price level
    void update_level(Side side, double price, double size);

    /// Remove price level
    void remove_level(Side side, double price);

    /// Get current snapshot
    [[nodiscard]] OrderBook get_snapshot(int32_t depth = 0) const;

    /// Get best bid
    [[nodiscard]] std::optional<PriceLevel> best_bid() const;

    /// Get best ask
    [[nodiscard]] std::optional<PriceLevel> best_ask() const;

    /// Get mid price
    [[nodiscard]] double mid_price() const;

    /// Get spread
    [[nodiscard]] double spread() const;

    /// Get total bid depth
    [[nodiscard]] double bid_depth() const;

    /// Get total ask depth
    [[nodiscard]] double ask_depth() const;

    /// Clear all levels
    void clear();

    /// Get last update timestamp
    [[nodiscard]] int64_t last_update() const noexcept { return last_update_.load(); }

    /// Set callback for orderbook updates
    void on_update(std::function<void(const OrderBook&)> callback);

private:
    std::string symbol_;

    // Bids: highest price first (descending)
    std::map<double, PriceLevel, std::greater<double>> bids_;

    // Asks: lowest price first (ascending)
    std::map<double, PriceLevel, std::less<double>> asks_;

    std::atomic<int64_t> last_update_{0};
    mutable std::mutex mutex_;
    std::function<void(const OrderBook&)> update_callback_;

    void notify_update();
};

/// Order tracker for managing local order state
class OrderTracker {
public:
    OrderTracker() = default;
    ~OrderTracker() = default;

    /// Add or update order
    void upsert(const Order& order);

    /// Remove order by ID
    bool remove(uint64_t order_id);

    /// Get order by ID
    [[nodiscard]] std::optional<Order> get(uint64_t order_id) const;

    /// Get all orders for symbol
    [[nodiscard]] std::vector<Order> get_by_symbol(const std::string& symbol) const;

    /// Get all open orders
    [[nodiscard]] std::vector<Order> get_open() const;

    /// Get all orders
    [[nodiscard]] std::vector<Order> get_all() const;

    /// Clear all orders
    void clear();

    /// Get order count
    [[nodiscard]] size_t count() const;

private:
    std::unordered_map<uint64_t, Order> orders_;
    mutable std::mutex mutex_;
};

/// Trade tracker for recent trades
class TradeTracker {
public:
    explicit TradeTracker(size_t max_trades = 1000);
    ~TradeTracker() = default;

    /// Add trade
    void add(const Trade& trade);

    /// Get recent trades for symbol
    [[nodiscard]] std::vector<Trade> get_by_symbol(
        const std::string& symbol,
        size_t limit = 100
    ) const;

    /// Get all recent trades
    [[nodiscard]] std::vector<Trade> get_recent(size_t limit = 100) const;

    /// Clear all trades
    void clear();

    /// Get trade count
    [[nodiscard]] size_t count() const;

    /// Set callback for new trades
    void on_trade(std::function<void(const Trade&)> callback);

private:
    std::vector<Trade> trades_;
    size_t max_trades_;
    mutable std::mutex mutex_;
    std::function<void(const Trade&)> trade_callback_;
};

/// Multi-symbol orderbook manager
class OrderBookManager {
public:
    OrderBookManager() = default;
    ~OrderBookManager() = default;

    /// Get or create orderbook for symbol
    LocalOrderBook& get_or_create(const std::string& symbol);

    /// Get orderbook for symbol (returns nullptr if not exists)
    LocalOrderBook* get(const std::string& symbol);

    /// Check if orderbook exists
    [[nodiscard]] bool has(const std::string& symbol) const;

    /// Remove orderbook
    bool remove(const std::string& symbol);

    /// Get all symbols
    [[nodiscard]] std::vector<std::string> symbols() const;

    /// Clear all orderbooks
    void clear();

private:
    std::unordered_map<std::string, std::unique_ptr<LocalOrderBook>> books_;
    mutable std::mutex mutex_;
};

} // namespace lxdex

#endif // LXDEX_ORDERBOOK_HPP
