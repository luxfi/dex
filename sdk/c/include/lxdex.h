/*
 * LX C SDK
 * Copyright (c) 2025 Lux Partners Limited
 *
 * High-performance C client for LX trading.
 */

#ifndef LXDEX_H
#define LXDEX_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define LXDEX_VERSION_MAJOR 1
#define LXDEX_VERSION_MINOR 0
#define LXDEX_VERSION_PATCH 0

/* Error codes */
typedef enum {
    LXDEX_OK = 0,
    LXDEX_ERR_INVALID_ARG = -1,
    LXDEX_ERR_NO_MEMORY = -2,
    LXDEX_ERR_CONNECTION = -3,
    LXDEX_ERR_TIMEOUT = -4,
    LXDEX_ERR_AUTH = -5,
    LXDEX_ERR_PARSE = -6,
    LXDEX_ERR_PROTOCOL = -7,
    LXDEX_ERR_RATE_LIMIT = -8,
    LXDEX_ERR_ORDER_REJECTED = -9,
    LXDEX_ERR_NOT_CONNECTED = -10,
    LXDEX_ERR_INTERNAL = -99
} lxdex_error_t;

/* Order types */
typedef enum {
    LXDEX_ORDER_LIMIT = 0,
    LXDEX_ORDER_MARKET = 1,
    LXDEX_ORDER_STOP = 2,
    LXDEX_ORDER_STOP_LIMIT = 3,
    LXDEX_ORDER_ICEBERG = 4,
    LXDEX_ORDER_PEG = 5
} lxdex_order_type_t;

/* Order sides */
typedef enum {
    LXDEX_SIDE_BUY = 0,
    LXDEX_SIDE_SELL = 1
} lxdex_side_t;

/* Order status */
typedef enum {
    LXDEX_STATUS_OPEN = 0,
    LXDEX_STATUS_PARTIAL = 1,
    LXDEX_STATUS_FILLED = 2,
    LXDEX_STATUS_CANCELLED = 3,
    LXDEX_STATUS_REJECTED = 4
} lxdex_order_status_t;

/* Time in force */
typedef enum {
    LXDEX_TIF_GTC = 0,  /* Good Till Cancelled */
    LXDEX_TIF_IOC = 1,  /* Immediate Or Cancel */
    LXDEX_TIF_FOK = 2,  /* Fill Or Kill */
    LXDEX_TIF_DAY = 3   /* Day Order */
} lxdex_time_in_force_t;

/* Connection state */
typedef enum {
    LXDEX_STATE_DISCONNECTED = 0,
    LXDEX_STATE_CONNECTING = 1,
    LXDEX_STATE_CONNECTED = 2,
    LXDEX_STATE_AUTHENTICATED = 3,
    LXDEX_STATE_ERROR = 4
} lxdex_conn_state_t;

/* Forward declarations */
typedef struct lxdex_client lxdex_client_t;
typedef struct lxdex_order lxdex_order_t;
typedef struct lxdex_trade lxdex_trade_t;
typedef struct lxdex_orderbook lxdex_orderbook_t;
typedef struct lxdex_price_level lxdex_price_level_t;
typedef struct lxdex_balance lxdex_balance_t;
typedef struct lxdex_position lxdex_position_t;

/* Strings - fixed size for predictable memory */
#define LXDEX_SYMBOL_LEN 32
#define LXDEX_USER_ID_LEN 64
#define LXDEX_CLIENT_ID_LEN 64
#define LXDEX_MSG_LEN 256

/* Order structure */
struct lxdex_order {
    uint64_t order_id;
    char symbol[LXDEX_SYMBOL_LEN];
    lxdex_order_type_t type;
    lxdex_side_t side;
    double price;
    double size;
    double filled;
    double remaining;
    lxdex_order_status_t status;
    char user_id[LXDEX_USER_ID_LEN];
    char client_id[LXDEX_CLIENT_ID_LEN];
    int64_t timestamp;
    lxdex_time_in_force_t time_in_force;
    bool post_only;
    bool reduce_only;
};

/* Trade structure */
struct lxdex_trade {
    uint64_t trade_id;
    char symbol[LXDEX_SYMBOL_LEN];
    double price;
    double size;
    lxdex_side_t side;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    char buyer_id[LXDEX_USER_ID_LEN];
    char seller_id[LXDEX_USER_ID_LEN];
    int64_t timestamp;
};

/* Price level in orderbook */
struct lxdex_price_level {
    double price;
    double size;
    int32_t count;
};

/* Order book structure */
struct lxdex_orderbook {
    char symbol[LXDEX_SYMBOL_LEN];
    lxdex_price_level_t *bids;
    size_t bids_count;
    size_t bids_capacity;
    lxdex_price_level_t *asks;
    size_t asks_count;
    size_t asks_capacity;
    int64_t timestamp;
};

/* Balance structure */
struct lxdex_balance {
    char asset[LXDEX_SYMBOL_LEN];
    double available;
    double locked;
    double total;
};

/* Position structure */
struct lxdex_position {
    char symbol[LXDEX_SYMBOL_LEN];
    double size;
    double entry_price;
    double mark_price;
    double pnl;
    double margin;
};

/* Client configuration */
typedef struct {
    const char *ws_url;         /* WebSocket URL (default: ws://localhost:8081) */
    const char *api_key;        /* API key for authentication */
    const char *api_secret;     /* API secret for authentication */
    int connect_timeout_ms;     /* Connection timeout (default: 10000) */
    int recv_timeout_ms;        /* Receive timeout (default: 30000) */
    int reconnect_interval_ms;  /* Reconnect interval (default: 5000) */
    bool auto_reconnect;        /* Auto reconnect on disconnect */
} lxdex_config_t;

/* Callbacks */
typedef void (*lxdex_on_connect_t)(lxdex_client_t *client, void *user_data);
typedef void (*lxdex_on_disconnect_t)(lxdex_client_t *client, int code, const char *reason, void *user_data);
typedef void (*lxdex_on_error_t)(lxdex_client_t *client, lxdex_error_t error, const char *msg, void *user_data);
typedef void (*lxdex_on_order_update_t)(lxdex_client_t *client, const lxdex_order_t *order, void *user_data);
typedef void (*lxdex_on_trade_t)(lxdex_client_t *client, const lxdex_trade_t *trade, void *user_data);
typedef void (*lxdex_on_orderbook_t)(lxdex_client_t *client, const lxdex_orderbook_t *book, void *user_data);
typedef void (*lxdex_on_balance_t)(lxdex_client_t *client, const lxdex_balance_t *balance, void *user_data);
typedef void (*lxdex_on_position_t)(lxdex_client_t *client, const lxdex_position_t *position, void *user_data);

/* Callback structure */
typedef struct {
    lxdex_on_connect_t on_connect;
    lxdex_on_disconnect_t on_disconnect;
    lxdex_on_error_t on_error;
    lxdex_on_order_update_t on_order_update;
    lxdex_on_trade_t on_trade;
    lxdex_on_orderbook_t on_orderbook;
    lxdex_on_balance_t on_balance;
    lxdex_on_position_t on_position;
    void *user_data;
} lxdex_callbacks_t;

/*
 * Client lifecycle
 */

/* Create a new client with configuration */
lxdex_client_t *lxdex_client_new(const lxdex_config_t *config);

/* Set callbacks */
void lxdex_client_set_callbacks(lxdex_client_t *client, const lxdex_callbacks_t *callbacks);

/* Connect to the DEX */
lxdex_error_t lxdex_client_connect(lxdex_client_t *client);

/* Authenticate with API credentials */
lxdex_error_t lxdex_client_auth(lxdex_client_t *client);

/* Disconnect from the DEX */
void lxdex_client_disconnect(lxdex_client_t *client);

/* Free client resources */
void lxdex_client_free(lxdex_client_t *client);

/* Get connection state */
lxdex_conn_state_t lxdex_client_state(const lxdex_client_t *client);

/* Service the client (call in event loop) */
int lxdex_client_service(lxdex_client_t *client, int timeout_ms);

/*
 * Order operations
 */

/* Place a new order */
lxdex_error_t lxdex_place_order(
    lxdex_client_t *client,
    const lxdex_order_t *order,
    uint64_t *order_id_out
);

/* Cancel an order */
lxdex_error_t lxdex_cancel_order(
    lxdex_client_t *client,
    uint64_t order_id
);

/* Cancel all orders for a symbol */
lxdex_error_t lxdex_cancel_all_orders(
    lxdex_client_t *client,
    const char *symbol
);

/* Modify an existing order */
lxdex_error_t lxdex_modify_order(
    lxdex_client_t *client,
    uint64_t order_id,
    double new_price,
    double new_size
);

/*
 * Market data
 */

/* Subscribe to orderbook updates */
lxdex_error_t lxdex_subscribe_orderbook(
    lxdex_client_t *client,
    const char *symbol
);

/* Subscribe to trade updates */
lxdex_error_t lxdex_subscribe_trades(
    lxdex_client_t *client,
    const char *symbol
);

/* Unsubscribe from a channel */
lxdex_error_t lxdex_unsubscribe(
    lxdex_client_t *client,
    const char *channel
);

/* Get orderbook snapshot (blocking) */
lxdex_error_t lxdex_get_orderbook(
    lxdex_client_t *client,
    const char *symbol,
    int depth,
    lxdex_orderbook_t *book_out
);

/* Get recent trades (blocking) */
lxdex_error_t lxdex_get_trades(
    lxdex_client_t *client,
    const char *symbol,
    int limit,
    lxdex_trade_t **trades_out,
    size_t *count_out
);

/*
 * Account operations
 */

/* Get balances */
lxdex_error_t lxdex_get_balances(
    lxdex_client_t *client,
    lxdex_balance_t **balances_out,
    size_t *count_out
);

/* Get positions */
lxdex_error_t lxdex_get_positions(
    lxdex_client_t *client,
    lxdex_position_t **positions_out,
    size_t *count_out
);

/* Get open orders */
lxdex_error_t lxdex_get_orders(
    lxdex_client_t *client,
    const char *symbol,
    lxdex_order_t **orders_out,
    size_t *count_out
);

/*
 * Order book utilities
 */

/* Initialize an orderbook */
lxdex_error_t lxdex_orderbook_init(lxdex_orderbook_t *book, size_t initial_capacity);

/* Free orderbook resources */
void lxdex_orderbook_free(lxdex_orderbook_t *book);

/* Get best bid price */
double lxdex_orderbook_best_bid(const lxdex_orderbook_t *book);

/* Get best ask price */
double lxdex_orderbook_best_ask(const lxdex_orderbook_t *book);

/* Get spread */
double lxdex_orderbook_spread(const lxdex_orderbook_t *book);

/* Get mid price */
double lxdex_orderbook_mid(const lxdex_orderbook_t *book);

/*
 * Order utilities
 */

/* Initialize an order with defaults */
void lxdex_order_init(lxdex_order_t *order);

/* Create a limit order */
void lxdex_order_limit(
    lxdex_order_t *order,
    const char *symbol,
    lxdex_side_t side,
    double price,
    double size
);

/* Create a market order */
void lxdex_order_market(
    lxdex_order_t *order,
    const char *symbol,
    lxdex_side_t side,
    double size
);

/*
 * Memory management
 */

/* Free array allocated by SDK */
void lxdex_free(void *ptr);

/* Free trades array */
void lxdex_trades_free(lxdex_trade_t *trades, size_t count);

/* Free balances array */
void lxdex_balances_free(lxdex_balance_t *balances, size_t count);

/* Free positions array */
void lxdex_positions_free(lxdex_position_t *positions, size_t count);

/* Free orders array */
void lxdex_orders_free(lxdex_order_t *orders, size_t count);

/*
 * Error handling
 */

/* Get error string */
const char *lxdex_strerror(lxdex_error_t error);

/* Get last error message (thread-local) */
const char *lxdex_last_error(void);

/*
 * Utility functions
 */

/* Get library version string */
const char *lxdex_version(void);

/* Initialize library (call once at startup) */
lxdex_error_t lxdex_init(void);

/* Cleanup library (call once at shutdown) */
void lxdex_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* LXDEX_H */
