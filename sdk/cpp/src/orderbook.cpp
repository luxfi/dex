// LX C++ SDK - Orderbook Implementation
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#include "lxdex/orderbook.hpp"
#include <algorithm>
#include <chrono>

namespace lxdex {

//------------------------------------------------------------------------------
// LocalOrderBook
//------------------------------------------------------------------------------

LocalOrderBook::LocalOrderBook(std::string symbol)
    : symbol_(std::move(symbol))
{}

void LocalOrderBook::apply_snapshot(const OrderBook& snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);

    bids_.clear();
    asks_.clear();

    for (const auto& level : snapshot.bids) {
        if (level.size > 0) {
            bids_[level.price] = level;
        }
    }

    for (const auto& level : snapshot.asks) {
        if (level.size > 0) {
            asks_[level.price] = level;
        }
    }

    last_update_.store(snapshot.timestamp > 0
        ? snapshot.timestamp
        : std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
          ).count()
    );

    notify_update();
}

void LocalOrderBook::update_level(Side side, double price, double size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (side == Side::Buy) {
        if (size > 0) {
            bids_[price] = PriceLevel{price, size, 1};
        } else {
            bids_.erase(price);
        }
    } else {
        if (size > 0) {
            asks_[price] = PriceLevel{price, size, 1};
        } else {
            asks_.erase(price);
        }
    }

    last_update_.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count());

    notify_update();
}

void LocalOrderBook::remove_level(Side side, double price) {
    update_level(side, price, 0);
}

OrderBook LocalOrderBook::get_snapshot(int32_t depth) const {
    std::lock_guard<std::mutex> lock(mutex_);

    OrderBook ob;
    ob.symbol = symbol_;
    ob.timestamp = last_update_.load();

    size_t limit = depth > 0 ? static_cast<size_t>(depth) : SIZE_MAX;

    ob.bids.reserve(std::min(bids_.size(), limit));
    size_t count = 0;
    for (const auto& [price, level] : bids_) {
        if (count >= limit) break;
        ob.bids.push_back(level);
        ++count;
    }

    ob.asks.reserve(std::min(asks_.size(), limit));
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count >= limit) break;
        ob.asks.push_back(level);
        ++count;
    }

    return ob;
}

std::optional<PriceLevel> LocalOrderBook::best_bid() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->second;
}

std::optional<PriceLevel> LocalOrderBook::best_ask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->second;
}

double LocalOrderBook::mid_price() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) return 0.0;
    return (bids_.begin()->first + asks_.begin()->first) / 2.0;
}

double LocalOrderBook::spread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) return 0.0;
    return asks_.begin()->first - bids_.begin()->first;
}

double LocalOrderBook::bid_depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    double total = 0.0;
    for (const auto& [price, level] : bids_) {
        total += level.size;
    }
    return total;
}

double LocalOrderBook::ask_depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    double total = 0.0;
    for (const auto& [price, level] : asks_) {
        total += level.size;
    }
    return total;
}

void LocalOrderBook::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    bids_.clear();
    asks_.clear();
    last_update_.store(0);
}

void LocalOrderBook::on_update(std::function<void(const OrderBook&)> callback) {
    update_callback_ = std::move(callback);
}

void LocalOrderBook::notify_update() {
    if (update_callback_) {
        // Create snapshot without lock (already held by caller)
        OrderBook ob;
        ob.symbol = symbol_;
        ob.timestamp = last_update_.load();

        for (const auto& [price, level] : bids_) {
            ob.bids.push_back(level);
        }
        for (const auto& [price, level] : asks_) {
            ob.asks.push_back(level);
        }

        update_callback_(ob);
    }
}

//------------------------------------------------------------------------------
// OrderTracker
//------------------------------------------------------------------------------

void OrderTracker::upsert(const Order& order) {
    std::lock_guard<std::mutex> lock(mutex_);
    orders_[order.order_id] = order;
}

bool OrderTracker::remove(uint64_t order_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return orders_.erase(order_id) > 0;
}

std::optional<Order> OrderTracker::get(uint64_t order_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return std::nullopt;
    return it->second;
}

std::vector<Order> OrderTracker::get_by_symbol(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Order> result;
    for (const auto& [id, order] : orders_) {
        if (order.symbol == symbol) {
            result.push_back(order);
        }
    }
    return result;
}

std::vector<Order> OrderTracker::get_open() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Order> result;
    for (const auto& [id, order] : orders_) {
        if (order.is_open()) {
            result.push_back(order);
        }
    }
    return result;
}

std::vector<Order> OrderTracker::get_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Order> result;
    result.reserve(orders_.size());
    for (const auto& [id, order] : orders_) {
        result.push_back(order);
    }
    return result;
}

void OrderTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    orders_.clear();
}

size_t OrderTracker::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return orders_.size();
}

//------------------------------------------------------------------------------
// TradeTracker
//------------------------------------------------------------------------------

TradeTracker::TradeTracker(size_t max_trades)
    : max_trades_(max_trades)
{
    trades_.reserve(max_trades_);
}

void TradeTracker::add(const Trade& trade) {
    std::lock_guard<std::mutex> lock(mutex_);

    trades_.push_back(trade);

    // Trim if over limit
    if (trades_.size() > max_trades_) {
        trades_.erase(trades_.begin(), trades_.begin() + (trades_.size() - max_trades_));
    }

    if (trade_callback_) {
        trade_callback_(trade);
    }
}

std::vector<Trade> TradeTracker::get_by_symbol(
    const std::string& symbol,
    size_t limit
) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Trade> result;
    result.reserve(std::min(limit, trades_.size()));

    // Iterate in reverse (most recent first)
    for (auto it = trades_.rbegin(); it != trades_.rend() && result.size() < limit; ++it) {
        if (it->symbol == symbol) {
            result.push_back(*it);
        }
    }

    return result;
}

std::vector<Trade> TradeTracker::get_recent(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t count = std::min(limit, trades_.size());
    std::vector<Trade> result;
    result.reserve(count);

    // Get most recent trades
    auto start = trades_.size() > limit ? trades_.end() - limit : trades_.begin();
    for (auto it = trades_.rbegin(); it != trades_.rend() && result.size() < limit; ++it) {
        result.push_back(*it);
    }

    return result;
}

void TradeTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    trades_.clear();
}

size_t TradeTracker::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trades_.size();
}

void TradeTracker::on_trade(std::function<void(const Trade&)> callback) {
    trade_callback_ = std::move(callback);
}

//------------------------------------------------------------------------------
// OrderBookManager
//------------------------------------------------------------------------------

LocalOrderBook& OrderBookManager::get_or_create(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = books_.find(symbol);
    if (it != books_.end()) {
        return *it->second;
    }

    auto [new_it, inserted] = books_.emplace(
        symbol,
        std::make_unique<LocalOrderBook>(symbol)
    );
    return *new_it->second;
}

LocalOrderBook* OrderBookManager::get(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = books_.find(symbol);
    return it != books_.end() ? it->second.get() : nullptr;
}

bool OrderBookManager::has(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return books_.find(symbol) != books_.end();
}

bool OrderBookManager::remove(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);
    return books_.erase(symbol) > 0;
}

std::vector<std::string> OrderBookManager::symbols() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(books_.size());
    for (const auto& [symbol, book] : books_) {
        result.push_back(symbol);
    }
    return result;
}

void OrderBookManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    books_.clear();
}

} // namespace lxdex
