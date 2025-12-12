// LX Trading SDK - Lock-Free Orderbook
// High-performance orderbook with lock-free updates

#pragma once

#include <lx/trading/types.hpp>
#include <algorithm>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lx::trading {

// Single venue orderbook with lock-free reads
class Orderbook {
public:
    explicit Orderbook(std::string_view symbol, std::string_view venue = "")
        : symbol_(symbol), venue_(venue), timestamp_(now_ms()), sequence_(0) {}

    Orderbook(const Orderbook&) = delete;
    Orderbook& operator=(const Orderbook&) = delete;
    Orderbook(Orderbook&&) = default;
    Orderbook& operator=(Orderbook&&) = default;

    // Accessors
    [[nodiscard]] const std::string& symbol() const { return symbol_; }
    [[nodiscard]] const std::string& venue() const { return venue_; }
    [[nodiscard]] int64_t timestamp() const { return timestamp_.load(std::memory_order_acquire); }
    [[nodiscard]] uint64_t sequence() const { return sequence_.load(std::memory_order_acquire); }

    // Mutators (thread-safe)
    void add_bid(Decimal price, Decimal quantity) {
        std::unique_lock lock(mutex_);
        bids_.push_back({price, quantity});
    }

    void add_ask(Decimal price, Decimal quantity) {
        std::unique_lock lock(mutex_);
        asks_.push_back({price, quantity});
    }

    void set_bid(Decimal price, Decimal quantity) {
        std::unique_lock lock(mutex_);
        for (auto& level : bids_) {
            if (level.price == price) {
                level.quantity = quantity;
                return;
            }
        }
        bids_.push_back({price, quantity});
    }

    void set_ask(Decimal price, Decimal quantity) {
        std::unique_lock lock(mutex_);
        for (auto& level : asks_) {
            if (level.price == price) {
                level.quantity = quantity;
                return;
            }
        }
        asks_.push_back({price, quantity});
    }

    void remove_bid(Decimal price) {
        std::unique_lock lock(mutex_);
        bids_.erase(std::remove_if(bids_.begin(), bids_.end(),
            [price](const PriceLevel& l) { return l.price == price; }), bids_.end());
    }

    void remove_ask(Decimal price) {
        std::unique_lock lock(mutex_);
        asks_.erase(std::remove_if(asks_.begin(), asks_.end(),
            [price](const PriceLevel& l) { return l.price == price; }), asks_.end());
    }

    void clear() {
        std::unique_lock lock(mutex_);
        bids_.clear();
        asks_.clear();
    }

    void sort() {
        std::unique_lock lock(mutex_);
        std::sort(bids_.begin(), bids_.end(),
            [](const PriceLevel& a, const PriceLevel& b) { return a.price > b.price; });
        std::sort(asks_.begin(), asks_.end(),
            [](const PriceLevel& a, const PriceLevel& b) { return a.price < b.price; });
        sequence_.fetch_add(1, std::memory_order_release);
        timestamp_.store(now_ms(), std::memory_order_release);
    }

    void set_timestamp(int64_t ts) { timestamp_.store(ts, std::memory_order_release); }
    void set_sequence(uint64_t seq) { sequence_.store(seq, std::memory_order_release); }

    // Readers (lock-free via snapshot)
    [[nodiscard]] std::vector<PriceLevel> bids() const {
        std::shared_lock lock(mutex_);
        return bids_;
    }

    [[nodiscard]] std::vector<PriceLevel> asks() const {
        std::shared_lock lock(mutex_);
        return asks_;
    }

    [[nodiscard]] std::optional<Decimal> best_bid() const {
        std::shared_lock lock(mutex_);
        return bids_.empty() ? std::nullopt : std::optional<Decimal>(bids_[0].price);
    }

    [[nodiscard]] std::optional<Decimal> best_ask() const {
        std::shared_lock lock(mutex_);
        return asks_.empty() ? std::nullopt : std::optional<Decimal>(asks_[0].price);
    }

    [[nodiscard]] std::optional<Decimal> mid_price() const {
        std::shared_lock lock(mutex_);
        if (bids_.empty() || asks_.empty()) return std::nullopt;
        return (bids_[0].price + asks_[0].price) / Decimal::from_double(2.0);
    }

    [[nodiscard]] std::optional<Decimal> spread() const {
        std::shared_lock lock(mutex_);
        if (bids_.empty() || asks_.empty()) return std::nullopt;
        return asks_[0].price - bids_[0].price;
    }

    [[nodiscard]] std::optional<Decimal> spread_percent() const {
        auto s = spread();
        auto m = mid_price();
        if (!s || !m || m->is_zero()) return std::nullopt;
        return (*s / *m) * Decimal::from_double(100.0);
    }

    [[nodiscard]] Decimal bid_liquidity() const {
        std::shared_lock lock(mutex_);
        Decimal total;
        for (const auto& level : bids_) {
            total = total + level.value();
        }
        return total;
    }

    [[nodiscard]] Decimal ask_liquidity() const {
        std::shared_lock lock(mutex_);
        Decimal total;
        for (const auto& level : asks_) {
            total = total + level.value();
        }
        return total;
    }

    [[nodiscard]] Decimal bid_depth(int levels) const {
        std::shared_lock lock(mutex_);
        Decimal total;
        int count = 0;
        for (const auto& level : bids_) {
            if (count >= levels) break;
            total = total + level.value();
            ++count;
        }
        return total;
    }

    [[nodiscard]] Decimal ask_depth(int levels) const {
        std::shared_lock lock(mutex_);
        Decimal total;
        int count = 0;
        for (const auto& level : asks_) {
            if (count >= levels) break;
            total = total + level.value();
            ++count;
        }
        return total;
    }

    // VWAP calculation for buying `amount`
    [[nodiscard]] std::optional<Decimal> vwap_buy(Decimal amount) const {
        std::shared_lock lock(mutex_);
        return calculate_vwap(asks_, amount);
    }

    // VWAP calculation for selling `amount`
    [[nodiscard]] std::optional<Decimal> vwap_sell(Decimal amount) const {
        std::shared_lock lock(mutex_);
        return calculate_vwap(bids_, amount);
    }

    // Check if sufficient liquidity exists
    [[nodiscard]] bool has_liquidity(Side side, Decimal amount) const {
        std::shared_lock lock(mutex_);
        const auto& levels = (side == Side::Buy) ? asks_ : bids_;
        Decimal total;
        for (const auto& level : levels) {
            total = total + level.quantity;
        }
        return total >= amount;
    }

private:
    [[nodiscard]] static std::optional<Decimal> calculate_vwap(
        const std::vector<PriceLevel>& levels, Decimal amount) {
        Decimal remaining = amount;
        Decimal total_value;
        Decimal total_qty;

        for (const auto& level : levels) {
            if (remaining <= Decimal::zero()) break;

            Decimal fill_qty = (remaining < level.quantity) ? remaining : level.quantity;
            total_value = total_value + (fill_qty * level.price);
            total_qty = total_qty + fill_qty;
            remaining = remaining - fill_qty;
        }

        if (total_qty.is_zero()) return std::nullopt;
        return total_value / total_qty;
    }

    std::string symbol_;
    std::string venue_;
    std::atomic<int64_t> timestamp_;
    std::atomic<uint64_t> sequence_;
    mutable std::shared_mutex mutex_;
    std::vector<PriceLevel> bids_;
    std::vector<PriceLevel> asks_;
};

// Hash functor for Decimal
struct DecimalHash {
    size_t operator()(const Decimal& d) const {
        return std::hash<int64_t>()(d.scaled_value());
    }
};

// Type alias for price-level map
using PriceLevelMap = std::unordered_map<Decimal, std::vector<std::pair<std::string, Decimal>>, DecimalHash>;

// Aggregated orderbook from multiple venues
class AggregatedOrderbook {
public:
    explicit AggregatedOrderbook(std::string_view symbol) : symbol_(symbol) {}

    [[nodiscard]] const std::string& symbol() const { return symbol_; }
    [[nodiscard]] int64_t timestamp() const { return timestamp_; }

    void add_orderbook(const Orderbook& book) {
        auto bids = book.bids();
        auto asks = book.asks();

        for (const auto& level : bids) {
            bids_[level.price].emplace_back(book.venue(), level.quantity);
        }

        for (const auto& level : asks) {
            asks_[level.price].emplace_back(book.venue(), level.quantity);
        }

        timestamp_ = std::max(timestamp_, book.timestamp());
    }

    // Best bid across all venues: (price, venue, qty)
    [[nodiscard]] std::optional<std::tuple<Decimal, std::string, Decimal>> best_bid() const {
        if (bids_.empty()) return std::nullopt;

        auto it = std::max_element(bids_.begin(), bids_.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        if (it == bids_.end() || it->second.empty()) return std::nullopt;

        const auto& [venue, qty] = it->second[0];
        return std::make_tuple(it->first, venue, qty);
    }

    // Best ask across all venues: (price, venue, qty)
    [[nodiscard]] std::optional<std::tuple<Decimal, std::string, Decimal>> best_ask() const {
        if (asks_.empty()) return std::nullopt;

        auto it = std::min_element(asks_.begin(), asks_.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        if (it == asks_.end() || it->second.empty()) return std::nullopt;

        const auto& [venue, qty] = it->second[0];
        return std::make_tuple(it->first, venue, qty);
    }

    // Get aggregated bid levels
    [[nodiscard]] std::vector<PriceLevel> aggregated_bids() const {
        std::vector<std::pair<Decimal, Decimal>> temp;
        for (const auto& [price, venues] : bids_) {
            Decimal total_qty;
            for (const auto& [v, qty] : venues) {
                total_qty = total_qty + qty;
            }
            temp.emplace_back(price, total_qty);
        }
        std::sort(temp.begin(), temp.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<PriceLevel> result;
        result.reserve(temp.size());
        for (const auto& [price, qty] : temp) {
            result.push_back({price, qty});
        }
        return result;
    }

    // Get aggregated ask levels
    [[nodiscard]] std::vector<PriceLevel> aggregated_asks() const {
        std::vector<std::pair<Decimal, Decimal>> temp;
        for (const auto& [price, venues] : asks_) {
            Decimal total_qty;
            for (const auto& [v, qty] : venues) {
                total_qty = total_qty + qty;
            }
            temp.emplace_back(price, total_qty);
        }
        std::sort(temp.begin(), temp.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<PriceLevel> result;
        result.reserve(temp.size());
        for (const auto& [price, qty] : temp) {
            result.push_back({price, qty});
        }
        return result;
    }

    // Find best venue for buying amount: (venue, price)
    [[nodiscard]] std::optional<std::pair<std::string, Decimal>> best_venue_buy(Decimal amount) const {
        if (asks_.empty()) return std::nullopt;

        // Use the same direct iteration approach as aggregated_asks
        // Find the lowest price ask level
        std::optional<Decimal> best_price;
        std::string best_venue;

        for (const auto& [price, venues] : asks_) {
            if ((!best_price || price < *best_price) && !venues.empty()) {
                best_price = price;
                best_venue = venues[0].first;
            }
        }

        if (best_venue.empty() || !best_price) return std::nullopt;
        return std::make_pair(best_venue, *best_price);
    }

    // Find best venue for selling amount: (venue, price)
    [[nodiscard]] std::optional<std::pair<std::string, Decimal>> best_venue_sell(Decimal amount) const {
        if (bids_.empty()) return std::nullopt;

        // Find the highest price bid level
        std::optional<Decimal> best_price;
        std::string best_venue;

        for (const auto& [price, venues] : bids_) {
            if ((!best_price || price > *best_price) && !venues.empty()) {
                best_price = price;
                best_venue = venues[0].first;
            }
        }

        if (best_venue.empty() || !best_price) return std::nullopt;
        return std::make_pair(best_venue, *best_price);
    }

    void clear() {
        bids_.clear();
        asks_.clear();
        timestamp_ = 0;
    }

private:
    std::string symbol_;
    int64_t timestamp_ = 0;

    // price -> [(venue, qty), ...]
    PriceLevelMap bids_;
    PriceLevelMap asks_;
};

}  // namespace lx::trading
