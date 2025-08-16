#ifndef LX_ORDERBOOK_BRIDGE_H
#define LX_ORDERBOOK_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Order structure compatible with both Go and C++
typedef struct {
    uint64_t order_id;
    uint64_t user_id;
    char symbol[32];
    double price;
    double quantity;
    double filled_quantity;
    uint8_t side;  // 0 = buy, 1 = sell
    uint8_t order_type;  // 0 = market, 1 = limit
    uint8_t status;  // 0 = pending, 1 = partial, 2 = filled, 3 = cancelled
    uint64_t timestamp;
} OrderC;

// Trade execution result
typedef struct {
    uint64_t trade_id;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    double quantity;
    uint64_t timestamp;
} TradeC;

// Order book handle (opaque pointer)
typedef void* OrderBookHandle;

// Order book management functions
OrderBookHandle orderbook_create(const char* symbol);
void orderbook_destroy(OrderBookHandle handle);

// Order operations
uint64_t orderbook_add_order(OrderBookHandle handle, const OrderC* order);
bool orderbook_cancel_order(OrderBookHandle handle, uint64_t order_id);
bool orderbook_modify_order(OrderBookHandle handle, uint64_t order_id, double new_price, double new_quantity);

// Matching engine
int orderbook_match_orders(OrderBookHandle handle, TradeC* trades, int max_trades);

// Book state queries
double orderbook_get_best_bid(OrderBookHandle handle);
double orderbook_get_best_ask(OrderBookHandle handle);
int orderbook_get_depth(OrderBookHandle handle, int level, double* bids, double* asks, double* bid_sizes, double* ask_sizes);
uint64_t orderbook_get_volume(OrderBookHandle handle);

// Snapshot and restore for persistence
int orderbook_get_snapshot(OrderBookHandle handle, char* buffer, int buffer_size);
bool orderbook_restore_snapshot(OrderBookHandle handle, const char* buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // LX_ORDERBOOK_BRIDGE_H