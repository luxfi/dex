/**
 * LX Trading SDK - Core Types
 * Fixed-point decimal, trading structs, and enums for HFT
 */

#ifndef LX_TRADING_TYPES_H
#define LX_TRADING_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Fixed-Point Decimal
 * ============================================================================
 * Stores value as integer * 10^(-8) for exact financial arithmetic.
 */

#define LX_DECIMAL_PRECISION 8
#define LX_DECIMAL_SCALE 100000000LL

typedef struct {
    int64_t value;
} LxDecimal;

/* Decimal constructors */
LxDecimal lx_decimal_zero(void);
LxDecimal lx_decimal_one(void);
LxDecimal lx_decimal_from_int(int64_t value);
LxDecimal lx_decimal_from_double(double value);
LxDecimal lx_decimal_from_string(const char* str);
LxDecimal lx_decimal_from_scaled(int64_t scaled_value);

/* Decimal conversions */
double lx_decimal_to_double(LxDecimal d);
int lx_decimal_to_string(LxDecimal d, char* buf, size_t buf_size);
int64_t lx_decimal_scaled_value(LxDecimal d);

/* Decimal arithmetic */
LxDecimal lx_decimal_add(LxDecimal a, LxDecimal b);
LxDecimal lx_decimal_sub(LxDecimal a, LxDecimal b);
LxDecimal lx_decimal_mul(LxDecimal a, LxDecimal b);
LxDecimal lx_decimal_div(LxDecimal a, LxDecimal b);
LxDecimal lx_decimal_abs(LxDecimal d);
LxDecimal lx_decimal_neg(LxDecimal d);

/* Decimal comparisons */
bool lx_decimal_eq(LxDecimal a, LxDecimal b);
bool lx_decimal_ne(LxDecimal a, LxDecimal b);
bool lx_decimal_lt(LxDecimal a, LxDecimal b);
bool lx_decimal_le(LxDecimal a, LxDecimal b);
bool lx_decimal_gt(LxDecimal a, LxDecimal b);
bool lx_decimal_ge(LxDecimal a, LxDecimal b);

/* Decimal predicates */
bool lx_decimal_is_zero(LxDecimal d);
bool lx_decimal_is_positive(LxDecimal d);
bool lx_decimal_is_negative(LxDecimal d);

/* ============================================================================
 * Trading Enums
 * ============================================================================ */

typedef enum {
    LX_SIDE_BUY = 0,
    LX_SIDE_SELL = 1
} LxSide;

typedef enum {
    LX_ORDER_TYPE_MARKET = 0,
    LX_ORDER_TYPE_LIMIT = 1,
    LX_ORDER_TYPE_LIMIT_MAKER = 2,
    LX_ORDER_TYPE_STOP_LOSS = 3,
    LX_ORDER_TYPE_STOP_LOSS_LIMIT = 4,
    LX_ORDER_TYPE_TAKE_PROFIT = 5,
    LX_ORDER_TYPE_TAKE_PROFIT_LIMIT = 6
} LxOrderType;

typedef enum {
    LX_TIF_GTC = 0,       /* Good till cancelled */
    LX_TIF_IOC = 1,       /* Immediate or cancel */
    LX_TIF_FOK = 2,       /* Fill or kill */
    LX_TIF_GTD = 3,       /* Good till date */
    LX_TIF_POST_ONLY = 4  /* Post only */
} LxTimeInForce;

typedef enum {
    LX_ORDER_STATUS_PENDING = 0,
    LX_ORDER_STATUS_OPEN = 1,
    LX_ORDER_STATUS_PARTIALLY_FILLED = 2,
    LX_ORDER_STATUS_FILLED = 3,
    LX_ORDER_STATUS_CANCELLED = 4,
    LX_ORDER_STATUS_REJECTED = 5,
    LX_ORDER_STATUS_EXPIRED = 6
} LxOrderStatus;

typedef enum {
    LX_VENUE_NATIVE = 0,
    LX_VENUE_CCXT = 1,
    LX_VENUE_HUMMINGBOT = 2,
    LX_VENUE_CUSTOM = 3
} LxVenueType;

/* Enum to string conversions */
const char* lx_side_to_string(LxSide side);
const char* lx_order_type_to_string(LxOrderType type);
const char* lx_tif_to_string(LxTimeInForce tif);
const char* lx_order_status_to_string(LxOrderStatus status);

/* ============================================================================
 * Trading Pair
 * ============================================================================ */

#define LX_SYMBOL_MAX_LEN 16

typedef struct {
    char base[LX_SYMBOL_MAX_LEN];
    char quote[LX_SYMBOL_MAX_LEN];
} LxTradingPair;

/* Parse symbol string (e.g., "BTC-USDC", "ETH/USD", "LUX_USDT") */
bool lx_trading_pair_from_symbol(const char* symbol, LxTradingPair* pair);

/* Format to different conventions */
int lx_trading_pair_to_hummingbot(const LxTradingPair* pair, char* buf, size_t buf_size);
int lx_trading_pair_to_ccxt(const LxTradingPair* pair, char* buf, size_t buf_size);

/* ============================================================================
 * Price Level
 * ============================================================================ */

typedef struct {
    LxDecimal price;
    LxDecimal quantity;
} LxPriceLevel;

/* Calculate notional value (price * quantity) */
LxDecimal lx_price_level_value(const LxPriceLevel* level);

/* ============================================================================
 * Fee
 * ============================================================================ */

#define LX_ASSET_MAX_LEN 32

typedef struct {
    char asset[LX_ASSET_MAX_LEN];
    LxDecimal amount;
    bool has_rate;
    LxDecimal rate;
} LxFee;

/* ============================================================================
 * Balance
 * ============================================================================ */

#define LX_VENUE_MAX_LEN 64

typedef struct {
    char asset[LX_ASSET_MAX_LEN];
    char venue[LX_VENUE_MAX_LEN];
    LxDecimal free;
    LxDecimal locked;
} LxBalance;

/* Get total balance (free + locked) */
LxDecimal lx_balance_total(const LxBalance* balance);

/* ============================================================================
 * Order Request
 * ============================================================================ */

#define LX_ORDER_ID_MAX_LEN 64
#define LX_CLIENT_ORDER_ID_MAX_LEN 64

typedef struct {
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];  /* "BASE-QUOTE" */
    LxSide side;
    LxOrderType order_type;
    LxDecimal quantity;
    bool has_price;
    LxDecimal price;
    bool has_stop_price;
    LxDecimal stop_price;
    LxTimeInForce time_in_force;
    bool reduce_only;
    bool post_only;
    bool has_venue;
    char venue[LX_VENUE_MAX_LEN];
    char client_order_id[LX_CLIENT_ORDER_ID_MAX_LEN];
} LxOrderRequest;

/* Initialize an order request */
void lx_order_request_init(LxOrderRequest* req);

/* Factory functions */
LxOrderRequest lx_order_request_market(const char* symbol, LxSide side, LxDecimal quantity);
LxOrderRequest lx_order_request_limit(const char* symbol, LxSide side, LxDecimal quantity, LxDecimal price);

/* Builder functions (mutate in place, return pointer for chaining) */
LxOrderRequest* lx_order_request_with_venue(LxOrderRequest* req, const char* venue);
LxOrderRequest* lx_order_request_with_post_only(LxOrderRequest* req);
LxOrderRequest* lx_order_request_with_client_id(LxOrderRequest* req, const char* client_id);
LxOrderRequest* lx_order_request_with_stop_price(LxOrderRequest* req, LxDecimal stop_price);

/* ============================================================================
 * Order
 * ============================================================================ */

typedef struct {
    char order_id[LX_ORDER_ID_MAX_LEN];
    char client_order_id[LX_CLIENT_ORDER_ID_MAX_LEN];
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    char venue[LX_VENUE_MAX_LEN];
    LxSide side;
    LxOrderType order_type;
    LxOrderStatus status;
    LxDecimal quantity;
    LxDecimal filled_quantity;
    LxDecimal remaining_quantity;
    bool has_price;
    LxDecimal price;
    bool has_average_price;
    LxDecimal average_price;
    int64_t created_at;
    int64_t updated_at;
    /* Fees stored separately (typically small count) */
    LxFee* fees;
    size_t fee_count;
} LxOrder;

/* Order status checks */
bool lx_order_is_open(const LxOrder* order);
bool lx_order_is_done(const LxOrder* order);
LxDecimal lx_order_fill_percent(const LxOrder* order);

/* Initialize/cleanup */
void lx_order_init(LxOrder* order);
void lx_order_cleanup(LxOrder* order);

/* ============================================================================
 * Trade/Fill
 * ============================================================================ */

typedef struct {
    char trade_id[LX_ORDER_ID_MAX_LEN];
    char order_id[LX_ORDER_ID_MAX_LEN];
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    char venue[LX_VENUE_MAX_LEN];
    LxSide side;
    LxDecimal price;
    LxDecimal quantity;
    LxFee fee;
    int64_t timestamp;
    bool is_maker;
} LxTrade;

/* Calculate trade value (price * quantity) */
LxDecimal lx_trade_value(const LxTrade* trade);

/* ============================================================================
 * Ticker
 * ============================================================================ */

typedef struct {
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    char venue[LX_VENUE_MAX_LEN];
    bool has_bid;
    LxDecimal bid;
    bool has_ask;
    LxDecimal ask;
    bool has_last;
    LxDecimal last;
    bool has_volume_24h;
    LxDecimal volume_24h;
    bool has_high_24h;
    LxDecimal high_24h;
    bool has_low_24h;
    LxDecimal low_24h;
    bool has_change_24h;
    LxDecimal change_24h;
    int64_t timestamp;
} LxTicker;

/* Ticker calculations */
bool lx_ticker_mid_price(const LxTicker* ticker, LxDecimal* result);
bool lx_ticker_spread(const LxTicker* ticker, LxDecimal* result);
bool lx_ticker_spread_percent(const LxTicker* ticker, LxDecimal* result);

/* ============================================================================
 * Pool Info (AMM)
 * ============================================================================ */

#define LX_ADDRESS_MAX_LEN 64

typedef struct {
    char address[LX_ADDRESS_MAX_LEN];
    char base_token[LX_ASSET_MAX_LEN];
    char quote_token[LX_ASSET_MAX_LEN];
    LxDecimal base_reserve;
    LxDecimal quote_reserve;
    LxDecimal total_liquidity;
    LxDecimal fee_rate;
    bool has_apy;
    LxDecimal apy;
} LxPoolInfo;

/* ============================================================================
 * Swap Quote
 * ============================================================================ */

typedef struct {
    char base_token[LX_ASSET_MAX_LEN];
    char quote_token[LX_ASSET_MAX_LEN];
    LxDecimal input_amount;
    LxDecimal output_amount;
    LxDecimal price;
    LxDecimal price_impact;
    LxDecimal fee;
    char** route;
    size_t route_count;
    int64_t expires_at;
} LxSwapQuote;

/* Cleanup swap quote (frees route) */
void lx_swap_quote_cleanup(LxSwapQuote* quote);

/* ============================================================================
 * Timestamp Utilities
 * ============================================================================ */

int64_t lx_now_ms(void);
int64_t lx_now_us(void);
int64_t lx_now_ns(void);

#ifdef __cplusplus
}
#endif

#endif /* LX_TRADING_TYPES_H */
