/**
 * LX Trading SDK - Types Implementation
 */

#include "lx_trading/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

/* ============================================================================
 * Decimal Implementation
 * ============================================================================ */

LxDecimal lx_decimal_zero(void) {
    LxDecimal d = {0};
    return d;
}

LxDecimal lx_decimal_one(void) {
    LxDecimal d = {LX_DECIMAL_SCALE};
    return d;
}

LxDecimal lx_decimal_from_int(int64_t value) {
    LxDecimal d = {value * LX_DECIMAL_SCALE};
    return d;
}

LxDecimal lx_decimal_from_double(double value) {
    LxDecimal d = {(int64_t)(value * LX_DECIMAL_SCALE)};
    return d;
}

LxDecimal lx_decimal_from_string(const char* str) {
    if (!str) return lx_decimal_zero();

    int negative = 0;
    const char* p = str;

    /* Skip whitespace */
    while (*p && isspace((unsigned char)*p)) p++;

    /* Handle sign */
    if (*p == '-') {
        negative = 1;
        p++;
    } else if (*p == '+') {
        p++;
    }

    int64_t integer_part = 0;
    int64_t decimal_part = 0;
    int decimal_places = 0;

    /* Parse integer part */
    while (*p && isdigit((unsigned char)*p)) {
        integer_part = integer_part * 10 + (*p - '0');
        p++;
    }

    /* Parse decimal part */
    if (*p == '.') {
        p++;
        while (*p && isdigit((unsigned char)*p) && decimal_places < LX_DECIMAL_PRECISION) {
            decimal_part = decimal_part * 10 + (*p - '0');
            decimal_places++;
            p++;
        }
    }

    /* Scale decimal part */
    int64_t scale_factor = LX_DECIMAL_SCALE;
    for (int i = 0; i < decimal_places; i++) {
        scale_factor /= 10;
    }

    int64_t value = integer_part * LX_DECIMAL_SCALE + decimal_part * scale_factor;
    if (negative) value = -value;

    LxDecimal d = {value};
    return d;
}

LxDecimal lx_decimal_from_scaled(int64_t scaled_value) {
    LxDecimal d = {scaled_value};
    return d;
}

double lx_decimal_to_double(LxDecimal d) {
    return (double)d.value / LX_DECIMAL_SCALE;
}

int lx_decimal_to_string(LxDecimal d, char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return -1;

    int negative = d.value < 0;
    int64_t abs_value = negative ? -d.value : d.value;

    int64_t integer_part = abs_value / LX_DECIMAL_SCALE;
    int64_t decimal_part = abs_value % LX_DECIMAL_SCALE;

    /* Remove trailing zeros */
    int decimal_digits = LX_DECIMAL_PRECISION;
    while (decimal_digits > 1 && decimal_part % 10 == 0) {
        decimal_part /= 10;
        decimal_digits--;
    }

    int written;
    if (decimal_part == 0) {
        written = snprintf(buf, buf_size, "%s%lld",
                          negative ? "-" : "",
                          (long long)integer_part);
    } else {
        written = snprintf(buf, buf_size, "%s%lld.%0*lld",
                          negative ? "-" : "",
                          (long long)integer_part,
                          decimal_digits,
                          (long long)decimal_part);
    }

    return written;
}

int64_t lx_decimal_scaled_value(LxDecimal d) {
    return d.value;
}

LxDecimal lx_decimal_add(LxDecimal a, LxDecimal b) {
    LxDecimal d = {a.value + b.value};
    return d;
}

LxDecimal lx_decimal_sub(LxDecimal a, LxDecimal b) {
    LxDecimal d = {a.value - b.value};
    return d;
}

LxDecimal lx_decimal_mul(LxDecimal a, LxDecimal b) {
    LxDecimal d = {(a.value * b.value) / LX_DECIMAL_SCALE};
    return d;
}

LxDecimal lx_decimal_div(LxDecimal a, LxDecimal b) {
    if (b.value == 0) {
        LxDecimal d = {0};
        return d;
    }
    LxDecimal d = {(a.value * LX_DECIMAL_SCALE) / b.value};
    return d;
}

LxDecimal lx_decimal_abs(LxDecimal d) {
    LxDecimal result = {d.value < 0 ? -d.value : d.value};
    return result;
}

LxDecimal lx_decimal_neg(LxDecimal d) {
    LxDecimal result = {-d.value};
    return result;
}

bool lx_decimal_eq(LxDecimal a, LxDecimal b) { return a.value == b.value; }
bool lx_decimal_ne(LxDecimal a, LxDecimal b) { return a.value != b.value; }
bool lx_decimal_lt(LxDecimal a, LxDecimal b) { return a.value < b.value; }
bool lx_decimal_le(LxDecimal a, LxDecimal b) { return a.value <= b.value; }
bool lx_decimal_gt(LxDecimal a, LxDecimal b) { return a.value > b.value; }
bool lx_decimal_ge(LxDecimal a, LxDecimal b) { return a.value >= b.value; }

bool lx_decimal_is_zero(LxDecimal d) { return d.value == 0; }
bool lx_decimal_is_positive(LxDecimal d) { return d.value > 0; }
bool lx_decimal_is_negative(LxDecimal d) { return d.value < 0; }

/* ============================================================================
 * Enum to String
 * ============================================================================ */

const char* lx_side_to_string(LxSide side) {
    return side == LX_SIDE_BUY ? "buy" : "sell";
}

const char* lx_order_type_to_string(LxOrderType type) {
    switch (type) {
        case LX_ORDER_TYPE_MARKET: return "market";
        case LX_ORDER_TYPE_LIMIT: return "limit";
        case LX_ORDER_TYPE_LIMIT_MAKER: return "limit_maker";
        case LX_ORDER_TYPE_STOP_LOSS: return "stop_loss";
        case LX_ORDER_TYPE_STOP_LOSS_LIMIT: return "stop_loss_limit";
        case LX_ORDER_TYPE_TAKE_PROFIT: return "take_profit";
        case LX_ORDER_TYPE_TAKE_PROFIT_LIMIT: return "take_profit_limit";
        default: return "unknown";
    }
}

const char* lx_tif_to_string(LxTimeInForce tif) {
    switch (tif) {
        case LX_TIF_GTC: return "GTC";
        case LX_TIF_IOC: return "IOC";
        case LX_TIF_FOK: return "FOK";
        case LX_TIF_GTD: return "GTD";
        case LX_TIF_POST_ONLY: return "POST_ONLY";
        default: return "unknown";
    }
}

const char* lx_order_status_to_string(LxOrderStatus status) {
    switch (status) {
        case LX_ORDER_STATUS_PENDING: return "pending";
        case LX_ORDER_STATUS_OPEN: return "open";
        case LX_ORDER_STATUS_PARTIALLY_FILLED: return "partially_filled";
        case LX_ORDER_STATUS_FILLED: return "filled";
        case LX_ORDER_STATUS_CANCELLED: return "cancelled";
        case LX_ORDER_STATUS_REJECTED: return "rejected";
        case LX_ORDER_STATUS_EXPIRED: return "expired";
        default: return "unknown";
    }
}

/* ============================================================================
 * Trading Pair
 * ============================================================================ */

bool lx_trading_pair_from_symbol(const char* symbol, LxTradingPair* pair) {
    if (!symbol || !pair) return false;

    memset(pair, 0, sizeof(LxTradingPair));

    /* Find separator */
    const char* sep = strchr(symbol, '-');
    if (!sep) sep = strchr(symbol, '/');
    if (!sep) sep = strchr(symbol, '_');
    if (!sep) return false;

    size_t base_len = sep - symbol;
    size_t quote_len = strlen(sep + 1);

    if (base_len >= LX_SYMBOL_MAX_LEN || quote_len >= LX_SYMBOL_MAX_LEN) {
        return false;
    }

    strncpy(pair->base, symbol, base_len);
    pair->base[base_len] = '\0';
    strncpy(pair->quote, sep + 1, quote_len);
    pair->quote[quote_len] = '\0';

    return true;
}

int lx_trading_pair_to_hummingbot(const LxTradingPair* pair, char* buf, size_t buf_size) {
    if (!pair || !buf || buf_size == 0) return -1;
    return snprintf(buf, buf_size, "%s-%s", pair->base, pair->quote);
}

int lx_trading_pair_to_ccxt(const LxTradingPair* pair, char* buf, size_t buf_size) {
    if (!pair || !buf || buf_size == 0) return -1;
    return snprintf(buf, buf_size, "%s/%s", pair->base, pair->quote);
}

/* ============================================================================
 * Price Level
 * ============================================================================ */

LxDecimal lx_price_level_value(const LxPriceLevel* level) {
    if (!level) return lx_decimal_zero();
    return lx_decimal_mul(level->price, level->quantity);
}

/* ============================================================================
 * Balance
 * ============================================================================ */

LxDecimal lx_balance_total(const LxBalance* balance) {
    if (!balance) return lx_decimal_zero();
    return lx_decimal_add(balance->free, balance->locked);
}

/* ============================================================================
 * Order Request
 * ============================================================================ */

void lx_order_request_init(LxOrderRequest* req) {
    if (!req) return;
    memset(req, 0, sizeof(LxOrderRequest));
    req->side = LX_SIDE_BUY;
    req->order_type = LX_ORDER_TYPE_MARKET;
    req->time_in_force = LX_TIF_GTC;
}

LxOrderRequest lx_order_request_market(const char* symbol, LxSide side, LxDecimal quantity) {
    LxOrderRequest req;
    lx_order_request_init(&req);

    if (symbol) {
        strncpy(req.symbol, symbol, sizeof(req.symbol) - 1);
    }
    req.side = side;
    req.order_type = LX_ORDER_TYPE_MARKET;
    req.quantity = quantity;
    req.time_in_force = LX_TIF_IOC;

    return req;
}

LxOrderRequest lx_order_request_limit(const char* symbol, LxSide side,
                                       LxDecimal quantity, LxDecimal price) {
    LxOrderRequest req;
    lx_order_request_init(&req);

    if (symbol) {
        strncpy(req.symbol, symbol, sizeof(req.symbol) - 1);
    }
    req.side = side;
    req.order_type = LX_ORDER_TYPE_LIMIT;
    req.quantity = quantity;
    req.has_price = true;
    req.price = price;
    req.time_in_force = LX_TIF_GTC;

    return req;
}

LxOrderRequest* lx_order_request_with_venue(LxOrderRequest* req, const char* venue) {
    if (!req) return NULL;
    if (venue) {
        req->has_venue = true;
        strncpy(req->venue, venue, sizeof(req->venue) - 1);
    }
    return req;
}

LxOrderRequest* lx_order_request_with_post_only(LxOrderRequest* req) {
    if (!req) return NULL;
    req->post_only = true;
    req->time_in_force = LX_TIF_POST_ONLY;
    return req;
}

LxOrderRequest* lx_order_request_with_client_id(LxOrderRequest* req, const char* client_id) {
    if (!req) return NULL;
    if (client_id) {
        strncpy(req->client_order_id, client_id, sizeof(req->client_order_id) - 1);
    }
    return req;
}

LxOrderRequest* lx_order_request_with_stop_price(LxOrderRequest* req, LxDecimal stop_price) {
    if (!req) return NULL;
    req->has_stop_price = true;
    req->stop_price = stop_price;
    return req;
}

/* ============================================================================
 * Order
 * ============================================================================ */

void lx_order_init(LxOrder* order) {
    if (!order) return;
    memset(order, 0, sizeof(LxOrder));
    order->side = LX_SIDE_BUY;
    order->order_type = LX_ORDER_TYPE_LIMIT;
    order->status = LX_ORDER_STATUS_PENDING;
}

void lx_order_cleanup(LxOrder* order) {
    if (!order) return;
    if (order->fees) {
        free(order->fees);
        order->fees = NULL;
    }
    order->fee_count = 0;
}

bool lx_order_is_open(const LxOrder* order) {
    if (!order) return false;
    return order->status == LX_ORDER_STATUS_OPEN ||
           order->status == LX_ORDER_STATUS_PARTIALLY_FILLED ||
           order->status == LX_ORDER_STATUS_PENDING;
}

bool lx_order_is_done(const LxOrder* order) {
    if (!order) return false;
    return order->status == LX_ORDER_STATUS_FILLED ||
           order->status == LX_ORDER_STATUS_CANCELLED ||
           order->status == LX_ORDER_STATUS_REJECTED ||
           order->status == LX_ORDER_STATUS_EXPIRED;
}

LxDecimal lx_order_fill_percent(const LxOrder* order) {
    if (!order || lx_decimal_is_zero(order->quantity)) {
        return lx_decimal_zero();
    }
    LxDecimal ratio = lx_decimal_div(order->filled_quantity, order->quantity);
    return lx_decimal_mul(ratio, lx_decimal_from_double(100.0));
}

/* ============================================================================
 * Trade
 * ============================================================================ */

LxDecimal lx_trade_value(const LxTrade* trade) {
    if (!trade) return lx_decimal_zero();
    return lx_decimal_mul(trade->price, trade->quantity);
}

/* ============================================================================
 * Ticker
 * ============================================================================ */

bool lx_ticker_mid_price(const LxTicker* ticker, LxDecimal* result) {
    if (!ticker || !result) return false;
    if (!ticker->has_bid || !ticker->has_ask) {
        if (ticker->has_last) {
            *result = ticker->last;
            return true;
        }
        return false;
    }
    *result = lx_decimal_div(lx_decimal_add(ticker->bid, ticker->ask),
                             lx_decimal_from_double(2.0));
    return true;
}

bool lx_ticker_spread(const LxTicker* ticker, LxDecimal* result) {
    if (!ticker || !result) return false;
    if (!ticker->has_bid || !ticker->has_ask) return false;
    *result = lx_decimal_sub(ticker->ask, ticker->bid);
    return true;
}

bool lx_ticker_spread_percent(const LxTicker* ticker, LxDecimal* result) {
    if (!ticker || !result) return false;
    if (!ticker->has_bid || !ticker->has_ask) return false;
    if (lx_decimal_is_zero(ticker->bid)) return false;

    LxDecimal spread = lx_decimal_sub(ticker->ask, ticker->bid);
    *result = lx_decimal_mul(lx_decimal_div(spread, ticker->bid),
                             lx_decimal_from_double(100.0));
    return true;
}

/* ============================================================================
 * Swap Quote
 * ============================================================================ */

void lx_swap_quote_cleanup(LxSwapQuote* quote) {
    if (!quote) return;
    if (quote->route) {
        for (size_t i = 0; i < quote->route_count; i++) {
            free(quote->route[i]);
        }
        free(quote->route);
        quote->route = NULL;
    }
    quote->route_count = 0;
}

/* ============================================================================
 * Timestamp Utilities
 * ============================================================================ */

int64_t lx_now_ms(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return (int64_t)((ull.QuadPart - 116444736000000000ULL) / 10000);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

int64_t lx_now_us(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return (int64_t)((ull.QuadPart - 116444736000000000ULL) / 10);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#endif
}

int64_t lx_now_ns(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return (int64_t)((ull.QuadPart - 116444736000000000ULL) * 100);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#endif
}
