#include <cstring>
#include <chrono>

#ifdef __ARM_ARCH
#include "../cpp/ultra_fast_matcher_arm.hpp"
#else
#include "../cpp/ultra_fast_matcher.hpp"
#endif

extern "C" {

// Opaque handle for Go
typedef void* matcher_handle_t;

// C-compatible order structure
struct COrder {
    uint64_t id;
    uint64_t price;
    uint64_t quantity;
    uint64_t timestamp;
    uint32_t trader_id;
    uint8_t  side;
    uint8_t  order_type;
    uint8_t  flags;
};

// C-compatible trade structure
struct CTrade {
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint64_t price;
    uint64_t quantity;
    uint64_t timestamp;
};

// Create a new ultra-fast matcher instance
matcher_handle_t ultra_matcher_create() {
    return new lx::UltraFastMatcher();
}

// Destroy matcher instance
void ultra_matcher_destroy(matcher_handle_t handle) {
    delete static_cast<lx::UltraFastMatcher*>(handle);
}

// Add order with nanosecond latency measurement
uint64_t ultra_matcher_add_order(matcher_handle_t handle, const COrder* c_order) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    
    // Convert C order to C++ order
    lx::Order order;
    order.id = c_order->id;
    order.price = c_order->price;
    order.quantity = c_order->quantity;
    order.timestamp = c_order->timestamp;
    order.trader_id = c_order->trader_id;
    order.side = c_order->side;
    order.type = c_order->order_type;
    order.flags = c_order->flags;
    
    // Add order and return latency in nanoseconds
    return matcher->add_order(order);
}

// Match orders and return trades
int ultra_matcher_match(matcher_handle_t handle, const COrder* c_order, CTrade* trades, int max_trades) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    
    // Convert C order to C++ order
    lx::Order order;
    order.id = c_order->id;
    order.price = c_order->price;
    order.quantity = c_order->quantity;
    order.timestamp = c_order->timestamp;
    order.trader_id = c_order->trader_id;
    order.side = c_order->side;
    order.type = c_order->order_type;
    order.flags = c_order->flags;
    
    // Match order
    auto matched_trades = matcher->match_order(order);
    
    // Copy trades to C array
    int num_trades = std::min(static_cast<int>(matched_trades.size()), max_trades);
    for (int i = 0; i < num_trades; ++i) {
        trades[i].buy_order_id = matched_trades[i].buy_order_id;
        trades[i].sell_order_id = matched_trades[i].sell_order_id;
        trades[i].price = matched_trades[i].price;
        trades[i].quantity = matched_trades[i].quantity;
        trades[i].timestamp = matched_trades[i].timestamp;
    }
    
    return num_trades;
}

// Cancel order
int ultra_matcher_cancel(matcher_handle_t handle, uint64_t order_id) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    return matcher->cancel_order(order_id) ? 1 : 0;
}

// Get best bid price
uint64_t ultra_matcher_best_bid(matcher_handle_t handle) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    auto snapshot = matcher->get_snapshot(1);
    return snapshot.bids.empty() ? 0 : snapshot.bids[0].first;
}

// Get best ask price
uint64_t ultra_matcher_best_ask(matcher_handle_t handle) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    auto snapshot = matcher->get_snapshot(1);
    return snapshot.asks.empty() ? UINT64_MAX : snapshot.asks[0].first;
}

// Get statistics
void ultra_matcher_stats(matcher_handle_t handle, uint64_t* total_orders, uint64_t* total_trades, uint64_t* total_volume) {
    auto* matcher = static_cast<lx::UltraFastMatcher*>(handle);
    auto stats = matcher->get_stats();
    *total_orders = stats.total_orders;
    *total_trades = stats.total_trades;
    *total_volume = stats.total_volume;
}

} // extern "C"