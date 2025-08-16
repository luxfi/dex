// MLX C++ GPU Matching Engine
// Unified implementation for both Apple Silicon (Metal) and NVIDIA (CUDA)

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/devices.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace mx = mlx::core;

// Order structure matching Go side
struct Order {
    uint64_t order_id;
    float price;      // Fixed point representation
    float quantity;
    uint32_t timestamp;
    uint8_t side;     // 0=buy, 1=sell
    uint8_t status;   // 0=active, 1=filled, 2=cancelled
    uint16_t user_id;
};

// Trade result structure
struct Trade {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    float price;
    float quantity;
    uint32_t timestamp;
};

// Performance statistics
struct MLXStats {
    uint64_t orders_processed;
    uint64_t trades_executed;
    uint64_t total_latency_ns;
    double throughput_orders_per_sec;
    double throughput_trades_per_sec;
};

class MLXMatchingEngine {
private:
    static constexpr size_t MAX_ORDERS = 1000000;
    static constexpr size_t MAX_TRADES = 500000;
    
    mx::Device device;
    MLXStats stats;
    
    // Pre-allocated arrays for zero-copy
    mx::array bid_buffer;
    mx::array ask_buffer;
    mx::array trade_buffer;
    
public:
    MLXMatchingEngine() : stats{0, 0, 0, 0.0, 0.0} {
        // Select best available device (CUDA if available, else Metal on Mac)
        device = mx::default_device();
        
        std::cout << "MLX Matching Engine initialized on: " 
                  << (device.type == mx::Device::Type::gpu ? "GPU" : "CPU") 
                  << std::endl;
        
        // Pre-allocate buffers
        bid_buffer = mx::zeros({MAX_ORDERS, 7}, mx::float32);
        ask_buffer = mx::zeros({MAX_ORDERS, 7}, mx::float32);
        trade_buffer = mx::zeros({MAX_TRADES, 6}, mx::float32);
    }
    
    std::vector<Trade> match_orders(const std::vector<Order>& bids, 
                                   const std::vector<Order>& asks) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (bids.empty() || asks.empty()) {
            return {};
        }
        
        // Convert orders to MLX arrays (7 fields per order)
        auto bids_array = orders_to_array(bids);
        auto asks_array = orders_to_array(asks);
        
        // Extract price columns for matching
        auto bid_prices = mx::take(bids_array, mx::array({1}), 1);  // Column 1 is price
        auto ask_prices = mx::take(asks_array, mx::array({1}), 1);
        
        // Reshape for broadcasting
        bid_prices = mx::reshape(bid_prices, {static_cast<int>(bids.size()), 1});
        ask_prices = mx::reshape(ask_prices, {1, static_cast<int>(asks.size())});
        
        // Create price crossing matrix using broadcasting
        // This leverages MLX's unified GPU architecture
        auto can_match = mx::greater_equal(bid_prices, ask_prices);
        
        // Find matching pairs
        auto matches = mx::where(can_match);
        
        if (matches.shape()[0] == 0) {
            return {};
        }
        
        // Extract bid and ask indices
        auto bid_indices = mx::take(matches, mx::array({0}), 1);
        auto ask_indices = mx::take(matches, mx::array({1}), 1);
        
        // Get quantities for matched orders
        auto bid_quantities = mx::take(bids_array, mx::array({2}), 1);  // Column 2 is quantity
        auto ask_quantities = mx::take(asks_array, mx::array({2}), 1);
        
        auto matched_bid_qty = mx::take(bid_quantities, bid_indices, 0);
        auto matched_ask_qty = mx::take(ask_quantities, ask_indices, 0);
        
        // Calculate trade quantities (minimum of bid/ask)
        auto trade_quantities = mx::minimum(matched_bid_qty, matched_ask_qty);
        
        // Filter out zero-quantity trades
        auto valid_mask = mx::greater(trade_quantities, mx::array(0.0f));
        auto valid_indices = mx::where(valid_mask);
        
        // Build trades from valid matches
        std::vector<Trade> trades;
        auto num_trades = std::min(valid_indices.shape()[0], static_cast<int>(MAX_TRADES));
        
        // Convert back to CPU for output
        auto bid_idx_cpu = bid_indices.to(mx::Device::cpu());
        auto ask_idx_cpu = ask_indices.to(mx::Device::cpu());
        auto qty_cpu = trade_quantities.to(mx::Device::cpu());
        
        for (int i = 0; i < num_trades; i++) {
            if (qty_cpu.data<float>()[i] <= 0) continue;
            
            int bid_idx = static_cast<int>(bid_idx_cpu.data<float>()[i]);
            int ask_idx = static_cast<int>(ask_idx_cpu.data<float>()[i]);
            
            Trade trade;
            trade.trade_id = stats.trades_executed + i;
            trade.buy_order_id = bids[bid_idx].order_id;
            trade.sell_order_id = asks[ask_idx].order_id;
            trade.price = asks[ask_idx].price;  // Trade at ask price (maker)
            trade.quantity = qty_cpu.data<float>()[i];
            trade.timestamp = std::max(bids[bid_idx].timestamp, asks[ask_idx].timestamp);
            
            trades.push_back(trade);
        }
        
        // Update statistics
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        stats.orders_processed += bids.size() + asks.size();
        stats.trades_executed += trades.size();
        stats.total_latency_ns += latency_ns;
        
        return trades;
    }
    
    // Parallel matching across multiple order books
    std::vector<std::vector<Trade>> match_multiple_books(
        const std::vector<std::pair<std::vector<Order>, std::vector<Order>>>& books) {
        
        std::vector<std::vector<Trade>> all_trades;
        
        // Process each book (MLX handles GPU parallelism internally)
        for (const auto& [bids, asks] : books) {
            auto trades = match_orders(bids, asks);
            all_trades.push_back(trades);
        }
        
        return all_trades;
    }
    
    // Aggregate orders into price levels
    std::pair<std::vector<float>, std::vector<float>> aggregate_orderbook(
        const std::vector<Order>& orders) {
        
        if (orders.empty()) {
            return {{}, {}};
        }
        
        auto orders_array = orders_to_array(orders);
        auto prices = mx::take(orders_array, mx::array({1}), 1);
        auto quantities = mx::take(orders_array, mx::array({2}), 1);
        
        // Get unique prices
        auto unique_prices = mx::unique(prices);
        auto num_levels = unique_prices.shape()[0];
        
        std::vector<float> price_levels;
        std::vector<float> aggregated_quantities;
        
        // Aggregate quantities at each price level
        auto unique_cpu = unique_prices.to(mx::Device::cpu());
        
        for (int i = 0; i < num_levels; i++) {
            float price = unique_cpu.data<float>()[i];
            
            // Find all orders at this price
            auto mask = mx::equal(prices, mx::array(price));
            auto level_quantities = mx::where(mask, quantities, mx::array(0.0f));
            auto total_qty = mx::sum(level_quantities);
            
            price_levels.push_back(price);
            aggregated_quantities.push_back(total_qty.item<float>());
        }
        
        return {price_levels, aggregated_quantities};
    }
    
    MLXStats get_stats() const {
        MLXStats current_stats = stats;
        
        if (stats.total_latency_ns > 0) {
            double seconds = stats.total_latency_ns / 1e9;
            current_stats.throughput_orders_per_sec = stats.orders_processed / seconds;
            current_stats.throughput_trades_per_sec = stats.trades_executed / seconds;
        }
        
        return current_stats;
    }
    
    std::string get_device_info() const {
        if (device.type == mx::Device::Type::gpu) {
            return "MLX GPU (Metal/CUDA unified)";
        }
        return "MLX CPU";
    }
    
private:
    mx::array orders_to_array(const std::vector<Order>& orders) {
        // Convert orders to 2D array (N x 7)
        std::vector<float> data;
        data.reserve(orders.size() * 7);
        
        for (const auto& order : orders) {
            data.push_back(static_cast<float>(order.order_id));
            data.push_back(order.price);
            data.push_back(order.quantity);
            data.push_back(static_cast<float>(order.timestamp));
            data.push_back(static_cast<float>(order.side));
            data.push_back(static_cast<float>(order.status));
            data.push_back(static_cast<float>(order.user_id));
        }
        
        return mx::array(data.data(), {static_cast<int>(orders.size()), 7}, mx::float32);
    }
};

// C interface for Go integration
extern "C" {
    void* mlx_create_engine() {
        try {
            return new MLXMatchingEngine();
        } catch (const std::exception& e) {
            std::cerr << "Failed to create MLX engine: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    void mlx_destroy_engine(void* engine) {
        delete static_cast<MLXMatchingEngine*>(engine);
    }
    
    int mlx_match_orders(void* engine,
                        Order* bids, int bid_count,
                        Order* asks, int ask_count,
                        Trade* trades, int max_trades) {
        if (!engine) return 0;
        
        auto* mlx_engine = static_cast<MLXMatchingEngine*>(engine);
        
        std::vector<Order> bid_vec(bids, bids + bid_count);
        std::vector<Order> ask_vec(asks, asks + ask_count);
        
        auto matched_trades = mlx_engine->match_orders(bid_vec, ask_vec);
        
        int trade_count = std::min(static_cast<int>(matched_trades.size()), max_trades);
        for (int i = 0; i < trade_count; i++) {
            trades[i] = matched_trades[i];
        }
        
        return trade_count;
    }
    
    void mlx_get_stats(void* engine, MLXStats* stats) {
        if (!engine || !stats) return;
        
        auto* mlx_engine = static_cast<MLXMatchingEngine*>(engine);
        *stats = mlx_engine->get_stats();
    }
    
    const char* mlx_get_device_info(void* engine) {
        if (!engine) return "No engine";
        
        auto* mlx_engine = static_cast<MLXMatchingEngine*>(engine);
        static std::string info = mlx_engine->get_device_info();
        return info.c_str();
    }
    
    int mlx_is_available() {
        try {
            auto device = mx::default_device();
            return device.type == mx::Device::Type::gpu ? 1 : 0;
        } catch (...) {
            return 0;
        }
    }
}