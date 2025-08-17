#ifndef ULTRA_FAST_MATCHER_ARM_HPP
#define ULTRA_FAST_MATCHER_ARM_HPP

#include <atomic>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <thread>
#include <chrono>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace lx {

// Cache-line aligned structures for optimal CPU performance
struct alignas(64) Order {
    uint64_t id;
    uint64_t price;      // Fixed point with 8 decimals
    uint64_t quantity;   // Fixed point with 8 decimals  
    uint64_t timestamp;  // Nanoseconds since epoch
    uint32_t trader_id;
    uint8_t  side;       // 0 = buy, 1 = sell
    uint8_t  type;       // 0 = limit, 1 = market
    uint8_t  flags;      // Various flags (post-only, IOC, etc)
    uint8_t  padding[1]; // Align to 64 bytes
};

static_assert(sizeof(Order) == 64, "Order must be exactly 64 bytes");

// Lock-free price level using atomic operations
struct alignas(64) PriceLevel {
    std::atomic<uint64_t> price{0};
    std::atomic<uint64_t> total_quantity{0};
    std::atomic<uint32_t> order_count{0};
    
    // Simplified order storage for demo
    static constexpr size_t MAX_ORDERS_PER_LEVEL = 1000;
    std::array<Order, MAX_ORDERS_PER_LEVEL> orders;
    std::atomic<uint32_t> order_index{0};
    
    // Padding to avoid false sharing
    uint8_t padding[8];
};

// Ultra-fast matching engine optimized for ARM
class UltraFastMatcher {
private:
    static constexpr size_t MAX_PRICE_LEVELS = 10000;
    static constexpr size_t PRICE_LEVEL_BUCKETS = 256;
    
    // Buy and sell books
    alignas(64) std::array<PriceLevel, MAX_PRICE_LEVELS> buy_levels;
    alignas(64) std::array<PriceLevel, MAX_PRICE_LEVELS> sell_levels;
    
    // Indices for fast lookup
    std::atomic<uint32_t> buy_level_count{0};
    std::atomic<uint32_t> sell_level_count{0};
    
    // Best bid/ask cached
    alignas(64) std::atomic<uint64_t> best_bid{0};
    alignas(64) std::atomic<uint64_t> best_ask{UINT64_MAX};
    
    // Statistics
    alignas(64) std::atomic<uint64_t> total_orders{0};
    alignas(64) std::atomic<uint64_t> total_trades{0};
    alignas(64) std::atomic<uint64_t> total_volume{0};
    alignas(64) std::atomic<uint64_t> total_latency_ns{0};
    
    // Get high-resolution timestamp
    inline uint64_t get_timestamp_ns() const noexcept {
        auto now = std::chrono::high_resolution_clock::now();
        auto nanos = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
        return nanos.time_since_epoch().count();
    }
    
    // Find or create price level
    PriceLevel* find_or_create_level(std::array<PriceLevel, MAX_PRICE_LEVELS>& levels, 
                                    std::atomic<uint32_t>& count, uint64_t price) {
        uint32_t level_count = count.load(std::memory_order_acquire);
        
        // Linear search (can be optimized to binary search)
        for (uint32_t i = 0; i < level_count; ++i) {
            uint64_t level_price = levels[i].price.load(std::memory_order_relaxed);
            if (level_price == price) {
                return &levels[i];
            }
        }
        
        // Create new level
        if (level_count < MAX_PRICE_LEVELS) {
            uint32_t new_idx = count.fetch_add(1, std::memory_order_acq_rel);
            if (new_idx < MAX_PRICE_LEVELS) {
                levels[new_idx].price.store(price, std::memory_order_release);
                return &levels[new_idx];
            }
        }
        
        return nullptr;
    }
    
public:
    struct Trade {
        uint64_t buy_order_id;
        uint64_t sell_order_id;
        uint64_t price;
        uint64_t quantity;
        uint64_t timestamp;
    };
    
    UltraFastMatcher() = default;
    ~UltraFastMatcher() = default;
    
    // Ultra-fast order insertion
    inline uint64_t add_order(const Order& order) noexcept {
        uint64_t start_ns = get_timestamp_ns();
        
        // Update statistics
        total_orders.fetch_add(1, std::memory_order_relaxed);
        
        // Determine which book to add to
        auto& levels = (order.side == 0) ? buy_levels : sell_levels;
        auto& level_count = (order.side == 0) ? buy_level_count : sell_level_count;
        
        // Market order - match immediately
        if (order.type == 1) {
            uint64_t matched = match_market_order(order);
            uint64_t end_ns = get_timestamp_ns();
            total_latency_ns.fetch_add(end_ns - start_ns, std::memory_order_relaxed);
            return end_ns - start_ns;
        }
        
        // Find or create price level
        PriceLevel* level = find_or_create_level(levels, level_count, order.price);
        if (!level) {
            return 0; // Failed to add
        }
        
        // Add order to level
        uint32_t idx = level->order_index.fetch_add(1, std::memory_order_acq_rel);
        if (idx < PriceLevel::MAX_ORDERS_PER_LEVEL) {
            level->orders[idx] = order;
            level->total_quantity.fetch_add(order.quantity, std::memory_order_relaxed);
            level->order_count.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Update best bid/ask
        if (order.side == 0) { // Buy
            uint64_t current_best = best_bid.load(std::memory_order_relaxed);
            while (order.price > current_best) {
                if (best_bid.compare_exchange_weak(current_best, order.price)) {
                    break;
                }
            }
        } else { // Sell
            uint64_t current_best = best_ask.load(std::memory_order_relaxed);
            while (order.price < current_best) {
                if (best_ask.compare_exchange_weak(current_best, order.price)) {
                    break;
                }
            }
        }
        
        // Try to match
        match_order(order);
        
        uint64_t end_ns = get_timestamp_ns();
        uint64_t latency = end_ns - start_ns;
        total_latency_ns.fetch_add(latency, std::memory_order_relaxed);
        return latency;
    }
    
    // Order matching
    inline std::vector<Trade> match_order(const Order& order) noexcept {
        std::vector<Trade> trades;
        
        auto& opposite_levels = (order.side == 0) ? sell_levels : buy_levels;
        auto& opposite_count = (order.side == 0) ? sell_level_count : buy_level_count;
        uint32_t level_count = opposite_count.load(std::memory_order_acquire);
        
        uint64_t remaining_qty = order.quantity;
        
        for (uint32_t i = 0; i < level_count && remaining_qty > 0; ++i) {
            PriceLevel& level = opposite_levels[i];
            uint64_t level_price = level.price.load(std::memory_order_relaxed);
            
            // Check if prices cross
            if ((order.side == 0 && level_price > order.price) ||
                (order.side == 1 && level_price < order.price)) {
                continue;
            }
            
            // Match against orders at this level
            uint32_t order_count = level.order_index.load(std::memory_order_acquire);
            for (uint32_t j = 0; j < order_count && remaining_qty > 0; ++j) {
                Order& match_order = level.orders[j];
                if (match_order.quantity == 0) continue;
                
                uint64_t match_qty = std::min(remaining_qty, match_order.quantity);
                
                // Create trade
                Trade trade;
                trade.price = level_price;
                trade.quantity = match_qty;
                trade.timestamp = get_timestamp_ns();
                
                if (order.side == 0) {
                    trade.buy_order_id = order.id;
                    trade.sell_order_id = match_order.id;
                } else {
                    trade.buy_order_id = match_order.id;
                    trade.sell_order_id = order.id;
                }
                
                trades.push_back(trade);
                
                // Update quantities
                remaining_qty -= match_qty;
                match_order.quantity -= match_qty;
                level.total_quantity.fetch_sub(match_qty, std::memory_order_relaxed);
            }
        }
        
        // Update statistics
        total_trades.fetch_add(trades.size(), std::memory_order_relaxed);
        for (const auto& trade : trades) {
            total_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
        }
        
        return trades;
    }
    
    // Market order matching
    inline uint64_t match_market_order(const Order& order) noexcept {
        auto& opposite_levels = (order.side == 0) ? sell_levels : buy_levels;
        auto& opposite_count = (order.side == 0) ? sell_level_count : buy_level_count;
        
        uint32_t level_count = opposite_count.load(std::memory_order_acquire);
        if (level_count == 0) return 0;
        
        uint64_t remaining_qty = order.quantity;
        uint64_t total_matched = 0;
        
        // Find best price level
        PriceLevel* best_level = nullptr;
        uint64_t best_price = (order.side == 0) ? UINT64_MAX : 0;
        
        for (uint32_t i = 0; i < level_count; ++i) {
            uint64_t level_price = opposite_levels[i].price.load(std::memory_order_relaxed);
            if (opposite_levels[i].total_quantity.load() == 0) continue;
            
            if ((order.side == 0 && level_price < best_price) ||
                (order.side == 1 && level_price > best_price)) {
                best_price = level_price;
                best_level = &opposite_levels[i];
            }
        }
        
        if (best_level) {
            uint32_t order_count = best_level->order_index.load(std::memory_order_acquire);
            for (uint32_t j = 0; j < order_count && remaining_qty > 0; ++j) {
                Order& match_order = best_level->orders[j];
                if (match_order.quantity == 0) continue;
                
                uint64_t match_qty = std::min(remaining_qty, match_order.quantity);
                remaining_qty -= match_qty;
                total_matched += match_qty;
                match_order.quantity -= match_qty;
                
                best_level->total_quantity.fetch_sub(match_qty, std::memory_order_relaxed);
            }
            
            total_trades.fetch_add(1, std::memory_order_relaxed);
            total_volume.fetch_add(total_matched, std::memory_order_relaxed);
        }
        
        return total_matched;
    }
    
    // Cancel order
    inline bool cancel_order(uint64_t order_id) noexcept {
        // Simplified - would need proper index
        return true;
    }
    
    // Get orderbook snapshot
    struct OrderBookSnapshot {
        std::vector<std::pair<uint64_t, uint64_t>> bids;
        std::vector<std::pair<uint64_t, uint64_t>> asks;
        uint64_t timestamp;
    };
    
    OrderBookSnapshot get_snapshot(size_t depth = 10) const noexcept {
        OrderBookSnapshot snapshot;
        snapshot.timestamp = get_timestamp_ns();
        
        // Collect bids
        uint32_t buy_count = buy_level_count.load(std::memory_order_acquire);
        for (uint32_t i = 0; i < std::min<uint32_t>(buy_count, depth); ++i) {
            uint64_t price = buy_levels[i].price.load(std::memory_order_relaxed);
            uint64_t qty = buy_levels[i].total_quantity.load(std::memory_order_relaxed);
            if (qty > 0) {
                snapshot.bids.push_back({price, qty});
            }
        }
        
        // Collect asks
        uint32_t sell_count = sell_level_count.load(std::memory_order_acquire);
        for (uint32_t i = 0; i < std::min<uint32_t>(sell_count, depth); ++i) {
            uint64_t price = sell_levels[i].price.load(std::memory_order_relaxed);
            uint64_t qty = sell_levels[i].total_quantity.load(std::memory_order_relaxed);
            if (qty > 0) {
                snapshot.asks.push_back({price, qty});
            }
        }
        
        return snapshot;
    }
    
    // Performance statistics
    struct Stats {
        uint64_t total_orders;
        uint64_t total_trades;
        uint64_t total_volume;
        uint64_t avg_latency_ns;
    };
    
    Stats get_stats() const noexcept {
        uint64_t orders = total_orders.load(std::memory_order_relaxed);
        uint64_t latency_sum = total_latency_ns.load(std::memory_order_relaxed);
        
        return {
            orders,
            total_trades.load(std::memory_order_relaxed),
            total_volume.load(std::memory_order_relaxed),
            orders > 0 ? latency_sum / orders : 0
        };
    }
};

} // namespace lx

#endif // ULTRA_FAST_MATCHER_ARM_HPP