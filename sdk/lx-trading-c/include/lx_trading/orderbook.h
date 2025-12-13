/**
 * LX Trading SDK - Orderbook
 * Thread-safe orderbook with VWAP, slippage, and liquidity calculations
 */

#ifndef LX_TRADING_ORDERBOOK_H
#define LX_TRADING_ORDERBOOK_H

#include "types.h"
#include <pthread.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Orderbook
 * ============================================================================
 * Single-venue orderbook with thread-safe operations.
 */

typedef struct {
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    char venue[LX_VENUE_MAX_LEN];

    /* Atomic timestamp and sequence */
    atomic_int_fast64_t timestamp;
    atomic_uint_fast64_t sequence;

    /* Price levels (dynamically allocated) */
    LxPriceLevel* bids;
    size_t bid_count;
    size_t bid_capacity;

    LxPriceLevel* asks;
    size_t ask_count;
    size_t ask_capacity;

    /* Thread safety */
    pthread_rwlock_t lock;
} LxOrderbook;

/* ============================================================================
 * Orderbook Lifecycle
 * ============================================================================ */

/* Initialize orderbook (returns 0 on success, -1 on error) */
int lx_orderbook_init(LxOrderbook* book, const char* symbol, const char* venue);

/* Cleanup orderbook (frees internal memory) */
void lx_orderbook_cleanup(LxOrderbook* book);

/* Create/destroy on heap */
LxOrderbook* lx_orderbook_create(const char* symbol, const char* venue);
void lx_orderbook_destroy(LxOrderbook* book);

/* ============================================================================
 * Orderbook Accessors (thread-safe reads)
 * ============================================================================ */

const char* lx_orderbook_symbol(const LxOrderbook* book);
const char* lx_orderbook_venue(const LxOrderbook* book);
int64_t lx_orderbook_timestamp(const LxOrderbook* book);
uint64_t lx_orderbook_sequence(const LxOrderbook* book);

/* ============================================================================
 * Orderbook Mutators (thread-safe writes)
 * ============================================================================ */

/* Add a price level (appends to list) */
int lx_orderbook_add_bid(LxOrderbook* book, LxDecimal price, LxDecimal quantity);
int lx_orderbook_add_ask(LxOrderbook* book, LxDecimal price, LxDecimal quantity);

/* Set a price level (updates existing or adds new) */
int lx_orderbook_set_bid(LxOrderbook* book, LxDecimal price, LxDecimal quantity);
int lx_orderbook_set_ask(LxOrderbook* book, LxDecimal price, LxDecimal quantity);

/* Remove a price level */
void lx_orderbook_remove_bid(LxOrderbook* book, LxDecimal price);
void lx_orderbook_remove_ask(LxOrderbook* book, LxDecimal price);

/* Clear all levels */
void lx_orderbook_clear(LxOrderbook* book);

/* Sort levels (bids descending, asks ascending) and update timestamp/sequence */
void lx_orderbook_sort(LxOrderbook* book);

/* Set timestamp/sequence directly */
void lx_orderbook_set_timestamp(LxOrderbook* book, int64_t ts);
void lx_orderbook_set_sequence(LxOrderbook* book, uint64_t seq);

/* ============================================================================
 * Orderbook Readers (thread-safe snapshots)
 * ============================================================================ */

/* Get copies of price levels (caller must free returned array) */
LxPriceLevel* lx_orderbook_bids_copy(const LxOrderbook* book, size_t* count);
LxPriceLevel* lx_orderbook_asks_copy(const LxOrderbook* book, size_t* count);

/* Get best bid/ask (returns false if empty) */
bool lx_orderbook_best_bid(const LxOrderbook* book, LxDecimal* price);
bool lx_orderbook_best_ask(const LxOrderbook* book, LxDecimal* price);

/* Get mid price (returns false if no bid or ask) */
bool lx_orderbook_mid_price(const LxOrderbook* book, LxDecimal* price);

/* Get spread (returns false if no bid or ask) */
bool lx_orderbook_spread(const LxOrderbook* book, LxDecimal* spread);
bool lx_orderbook_spread_percent(const LxOrderbook* book, LxDecimal* spread_pct);

/* ============================================================================
 * Liquidity Calculations
 * ============================================================================ */

/* Total notional liquidity on each side */
LxDecimal lx_orderbook_bid_liquidity(const LxOrderbook* book);
LxDecimal lx_orderbook_ask_liquidity(const LxOrderbook* book);

/* Notional depth for N levels */
LxDecimal lx_orderbook_bid_depth(const LxOrderbook* book, int levels);
LxDecimal lx_orderbook_ask_depth(const LxOrderbook* book, int levels);

/* Check if sufficient liquidity exists */
bool lx_orderbook_has_liquidity(const LxOrderbook* book, LxSide side, LxDecimal amount);

/* ============================================================================
 * VWAP Calculations
 * ============================================================================ */

/* Volume-weighted average price for buying amount (walks asks) */
bool lx_orderbook_vwap_buy(const LxOrderbook* book, LxDecimal amount, LxDecimal* vwap);

/* Volume-weighted average price for selling amount (walks bids) */
bool lx_orderbook_vwap_sell(const LxOrderbook* book, LxDecimal amount, LxDecimal* vwap);

/* ============================================================================
 * Slippage Calculation
 * ============================================================================ */

/* Calculate slippage for a given order size
 * Returns the difference between VWAP and best price as a percentage */
bool lx_orderbook_slippage_buy(const LxOrderbook* book, LxDecimal amount, LxDecimal* slippage_pct);
bool lx_orderbook_slippage_sell(const LxOrderbook* book, LxDecimal amount, LxDecimal* slippage_pct);

/* ============================================================================
 * Aggregated Orderbook
 * ============================================================================
 * Combines orderbooks from multiple venues.
 */

typedef struct {
    char venue[LX_VENUE_MAX_LEN];
    LxDecimal quantity;
} LxVenueQuantity;

typedef struct {
    LxDecimal price;
    LxVenueQuantity* venues;
    size_t venue_count;
    size_t venue_capacity;
} LxAggregatedLevel;

typedef struct {
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    int64_t timestamp;

    LxAggregatedLevel* bids;
    size_t bid_count;
    size_t bid_capacity;

    LxAggregatedLevel* asks;
    size_t ask_count;
    size_t ask_capacity;
} LxAggregatedOrderbook;

/* ============================================================================
 * Aggregated Orderbook Lifecycle
 * ============================================================================ */

int lx_aggregated_orderbook_init(LxAggregatedOrderbook* agg, const char* symbol);
void lx_aggregated_orderbook_cleanup(LxAggregatedOrderbook* agg);
LxAggregatedOrderbook* lx_aggregated_orderbook_create(const char* symbol);
void lx_aggregated_orderbook_destroy(LxAggregatedOrderbook* agg);

/* ============================================================================
 * Aggregated Orderbook Operations
 * ============================================================================ */

/* Add an orderbook to the aggregation */
int lx_aggregated_orderbook_add(LxAggregatedOrderbook* agg, const LxOrderbook* book);

/* Clear all aggregated data */
void lx_aggregated_orderbook_clear(LxAggregatedOrderbook* agg);

/* Best bid across all venues (price, venue, qty) */
bool lx_aggregated_orderbook_best_bid(const LxAggregatedOrderbook* agg,
                                       LxDecimal* price, char* venue, size_t venue_buf_size,
                                       LxDecimal* quantity);

/* Best ask across all venues (price, venue, qty) */
bool lx_aggregated_orderbook_best_ask(const LxAggregatedOrderbook* agg,
                                       LxDecimal* price, char* venue, size_t venue_buf_size,
                                       LxDecimal* quantity);

/* Get aggregated price levels (caller must free) */
LxPriceLevel* lx_aggregated_orderbook_aggregated_bids(const LxAggregatedOrderbook* agg, size_t* count);
LxPriceLevel* lx_aggregated_orderbook_aggregated_asks(const LxAggregatedOrderbook* agg, size_t* count);

/* Find best venue for buying/selling amount */
bool lx_aggregated_orderbook_best_venue_buy(const LxAggregatedOrderbook* agg, LxDecimal amount,
                                             char* venue, size_t venue_buf_size, LxDecimal* price);
bool lx_aggregated_orderbook_best_venue_sell(const LxAggregatedOrderbook* agg, LxDecimal amount,
                                              char* venue, size_t venue_buf_size, LxDecimal* price);

#ifdef __cplusplus
}
#endif

#endif /* LX_TRADING_ORDERBOOK_H */
