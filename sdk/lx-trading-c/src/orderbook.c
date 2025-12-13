/**
 * LX Trading SDK - Orderbook Implementation
 */

#include "lx_trading/orderbook.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 64

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static int ensure_bid_capacity(LxOrderbook* book, size_t required) {
    if (book->bid_capacity >= required) return 0;

    size_t new_capacity = book->bid_capacity * 2;
    if (new_capacity < required) new_capacity = required;
    if (new_capacity < INITIAL_CAPACITY) new_capacity = INITIAL_CAPACITY;

    LxPriceLevel* new_bids = realloc(book->bids, new_capacity * sizeof(LxPriceLevel));
    if (!new_bids) return -1;

    book->bids = new_bids;
    book->bid_capacity = new_capacity;
    return 0;
}

static int ensure_ask_capacity(LxOrderbook* book, size_t required) {
    if (book->ask_capacity >= required) return 0;

    size_t new_capacity = book->ask_capacity * 2;
    if (new_capacity < required) new_capacity = required;
    if (new_capacity < INITIAL_CAPACITY) new_capacity = INITIAL_CAPACITY;

    LxPriceLevel* new_asks = realloc(book->asks, new_capacity * sizeof(LxPriceLevel));
    if (!new_asks) return -1;

    book->asks = new_asks;
    book->ask_capacity = new_capacity;
    return 0;
}

static int compare_bids_desc(const void* a, const void* b) {
    const LxPriceLevel* la = (const LxPriceLevel*)a;
    const LxPriceLevel* lb = (const LxPriceLevel*)b;
    if (lx_decimal_gt(la->price, lb->price)) return -1;
    if (lx_decimal_lt(la->price, lb->price)) return 1;
    return 0;
}

static int compare_asks_asc(const void* a, const void* b) {
    const LxPriceLevel* la = (const LxPriceLevel*)a;
    const LxPriceLevel* lb = (const LxPriceLevel*)b;
    if (lx_decimal_lt(la->price, lb->price)) return -1;
    if (lx_decimal_gt(la->price, lb->price)) return 1;
    return 0;
}

static LxDecimal calculate_vwap(const LxPriceLevel* levels, size_t count, LxDecimal amount) {
    LxDecimal remaining = amount;
    LxDecimal total_value = lx_decimal_zero();
    LxDecimal total_qty = lx_decimal_zero();

    for (size_t i = 0; i < count && lx_decimal_gt(remaining, lx_decimal_zero()); i++) {
        LxDecimal fill_qty = lx_decimal_lt(remaining, levels[i].quantity)
                             ? remaining : levels[i].quantity;
        total_value = lx_decimal_add(total_value, lx_decimal_mul(fill_qty, levels[i].price));
        total_qty = lx_decimal_add(total_qty, fill_qty);
        remaining = lx_decimal_sub(remaining, fill_qty);
    }

    if (lx_decimal_is_zero(total_qty)) {
        return lx_decimal_zero();
    }

    return lx_decimal_div(total_value, total_qty);
}

/* ============================================================================
 * Orderbook Lifecycle
 * ============================================================================ */

int lx_orderbook_init(LxOrderbook* book, const char* symbol, const char* venue) {
    if (!book) return -1;

    memset(book, 0, sizeof(LxOrderbook));

    if (symbol) {
        strncpy(book->symbol, symbol, sizeof(book->symbol) - 1);
    }
    if (venue) {
        strncpy(book->venue, venue, sizeof(book->venue) - 1);
    }

    atomic_store(&book->timestamp, lx_now_ms());
    atomic_store(&book->sequence, 0);

    if (pthread_rwlock_init(&book->lock, NULL) != 0) {
        return -1;
    }

    return 0;
}

void lx_orderbook_cleanup(LxOrderbook* book) {
    if (!book) return;

    pthread_rwlock_destroy(&book->lock);

    free(book->bids);
    free(book->asks);

    book->bids = NULL;
    book->asks = NULL;
    book->bid_count = 0;
    book->ask_count = 0;
    book->bid_capacity = 0;
    book->ask_capacity = 0;
}

LxOrderbook* lx_orderbook_create(const char* symbol, const char* venue) {
    LxOrderbook* book = malloc(sizeof(LxOrderbook));
    if (!book) return NULL;

    if (lx_orderbook_init(book, symbol, venue) != 0) {
        free(book);
        return NULL;
    }

    return book;
}

void lx_orderbook_destroy(LxOrderbook* book) {
    if (!book) return;
    lx_orderbook_cleanup(book);
    free(book);
}

/* ============================================================================
 * Orderbook Accessors
 * ============================================================================ */

const char* lx_orderbook_symbol(const LxOrderbook* book) {
    return book ? book->symbol : "";
}

const char* lx_orderbook_venue(const LxOrderbook* book) {
    return book ? book->venue : "";
}

int64_t lx_orderbook_timestamp(const LxOrderbook* book) {
    return book ? atomic_load(&book->timestamp) : 0;
}

uint64_t lx_orderbook_sequence(const LxOrderbook* book) {
    return book ? atomic_load(&book->sequence) : 0;
}

/* ============================================================================
 * Orderbook Mutators
 * ============================================================================ */

int lx_orderbook_add_bid(LxOrderbook* book, LxDecimal price, LxDecimal quantity) {
    if (!book) return -1;

    pthread_rwlock_wrlock(&book->lock);

    if (ensure_bid_capacity(book, book->bid_count + 1) != 0) {
        pthread_rwlock_unlock(&book->lock);
        return -1;
    }

    book->bids[book->bid_count].price = price;
    book->bids[book->bid_count].quantity = quantity;
    book->bid_count++;

    pthread_rwlock_unlock(&book->lock);
    return 0;
}

int lx_orderbook_add_ask(LxOrderbook* book, LxDecimal price, LxDecimal quantity) {
    if (!book) return -1;

    pthread_rwlock_wrlock(&book->lock);

    if (ensure_ask_capacity(book, book->ask_count + 1) != 0) {
        pthread_rwlock_unlock(&book->lock);
        return -1;
    }

    book->asks[book->ask_count].price = price;
    book->asks[book->ask_count].quantity = quantity;
    book->ask_count++;

    pthread_rwlock_unlock(&book->lock);
    return 0;
}

int lx_orderbook_set_bid(LxOrderbook* book, LxDecimal price, LxDecimal quantity) {
    if (!book) return -1;

    pthread_rwlock_wrlock(&book->lock);

    /* Find existing level */
    for (size_t i = 0; i < book->bid_count; i++) {
        if (lx_decimal_eq(book->bids[i].price, price)) {
            book->bids[i].quantity = quantity;
            pthread_rwlock_unlock(&book->lock);
            return 0;
        }
    }

    /* Add new level */
    if (ensure_bid_capacity(book, book->bid_count + 1) != 0) {
        pthread_rwlock_unlock(&book->lock);
        return -1;
    }

    book->bids[book->bid_count].price = price;
    book->bids[book->bid_count].quantity = quantity;
    book->bid_count++;

    pthread_rwlock_unlock(&book->lock);
    return 0;
}

int lx_orderbook_set_ask(LxOrderbook* book, LxDecimal price, LxDecimal quantity) {
    if (!book) return -1;

    pthread_rwlock_wrlock(&book->lock);

    /* Find existing level */
    for (size_t i = 0; i < book->ask_count; i++) {
        if (lx_decimal_eq(book->asks[i].price, price)) {
            book->asks[i].quantity = quantity;
            pthread_rwlock_unlock(&book->lock);
            return 0;
        }
    }

    /* Add new level */
    if (ensure_ask_capacity(book, book->ask_count + 1) != 0) {
        pthread_rwlock_unlock(&book->lock);
        return -1;
    }

    book->asks[book->ask_count].price = price;
    book->asks[book->ask_count].quantity = quantity;
    book->ask_count++;

    pthread_rwlock_unlock(&book->lock);
    return 0;
}

void lx_orderbook_remove_bid(LxOrderbook* book, LxDecimal price) {
    if (!book) return;

    pthread_rwlock_wrlock(&book->lock);

    for (size_t i = 0; i < book->bid_count; i++) {
        if (lx_decimal_eq(book->bids[i].price, price)) {
            /* Shift remaining elements */
            memmove(&book->bids[i], &book->bids[i + 1],
                    (book->bid_count - i - 1) * sizeof(LxPriceLevel));
            book->bid_count--;
            break;
        }
    }

    pthread_rwlock_unlock(&book->lock);
}

void lx_orderbook_remove_ask(LxOrderbook* book, LxDecimal price) {
    if (!book) return;

    pthread_rwlock_wrlock(&book->lock);

    for (size_t i = 0; i < book->ask_count; i++) {
        if (lx_decimal_eq(book->asks[i].price, price)) {
            memmove(&book->asks[i], &book->asks[i + 1],
                    (book->ask_count - i - 1) * sizeof(LxPriceLevel));
            book->ask_count--;
            break;
        }
    }

    pthread_rwlock_unlock(&book->lock);
}

void lx_orderbook_clear(LxOrderbook* book) {
    if (!book) return;

    pthread_rwlock_wrlock(&book->lock);
    book->bid_count = 0;
    book->ask_count = 0;
    pthread_rwlock_unlock(&book->lock);
}

void lx_orderbook_sort(LxOrderbook* book) {
    if (!book) return;

    pthread_rwlock_wrlock(&book->lock);

    if (book->bid_count > 1) {
        qsort(book->bids, book->bid_count, sizeof(LxPriceLevel), compare_bids_desc);
    }
    if (book->ask_count > 1) {
        qsort(book->asks, book->ask_count, sizeof(LxPriceLevel), compare_asks_asc);
    }

    atomic_fetch_add(&book->sequence, 1);
    atomic_store(&book->timestamp, lx_now_ms());

    pthread_rwlock_unlock(&book->lock);
}

void lx_orderbook_set_timestamp(LxOrderbook* book, int64_t ts) {
    if (book) atomic_store(&book->timestamp, ts);
}

void lx_orderbook_set_sequence(LxOrderbook* book, uint64_t seq) {
    if (book) atomic_store(&book->sequence, seq);
}

/* ============================================================================
 * Orderbook Readers
 * ============================================================================ */

LxPriceLevel* lx_orderbook_bids_copy(const LxOrderbook* book, size_t* count) {
    if (!book || !count) {
        if (count) *count = 0;
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    *count = book->bid_count;
    if (*count == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
        return NULL;
    }

    LxPriceLevel* copy = malloc(*count * sizeof(LxPriceLevel));
    if (copy) {
        memcpy(copy, book->bids, *count * sizeof(LxPriceLevel));
    } else {
        *count = 0;
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return copy;
}

LxPriceLevel* lx_orderbook_asks_copy(const LxOrderbook* book, size_t* count) {
    if (!book || !count) {
        if (count) *count = 0;
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    *count = book->ask_count;
    if (*count == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
        return NULL;
    }

    LxPriceLevel* copy = malloc(*count * sizeof(LxPriceLevel));
    if (copy) {
        memcpy(copy, book->asks, *count * sizeof(LxPriceLevel));
    } else {
        *count = 0;
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return copy;
}

bool lx_orderbook_best_bid(const LxOrderbook* book, LxDecimal* price) {
    if (!book || !price) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    bool found = book->bid_count > 0;
    if (found) {
        *price = book->bids[0].price;
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return found;
}

bool lx_orderbook_best_ask(const LxOrderbook* book, LxDecimal* price) {
    if (!book || !price) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    bool found = book->ask_count > 0;
    if (found) {
        *price = book->asks[0].price;
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return found;
}

bool lx_orderbook_mid_price(const LxOrderbook* book, LxDecimal* price) {
    if (!book || !price) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    bool found = book->bid_count > 0 && book->ask_count > 0;
    if (found) {
        *price = lx_decimal_div(
            lx_decimal_add(book->bids[0].price, book->asks[0].price),
            lx_decimal_from_double(2.0)
        );
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return found;
}

bool lx_orderbook_spread(const LxOrderbook* book, LxDecimal* spread) {
    if (!book || !spread) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    bool found = book->bid_count > 0 && book->ask_count > 0;
    if (found) {
        *spread = lx_decimal_sub(book->asks[0].price, book->bids[0].price);
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return found;
}

bool lx_orderbook_spread_percent(const LxOrderbook* book, LxDecimal* spread_pct) {
    if (!book || !spread_pct) return false;

    LxDecimal spread, mid;
    if (!lx_orderbook_spread(book, &spread)) return false;
    if (!lx_orderbook_mid_price(book, &mid)) return false;
    if (lx_decimal_is_zero(mid)) return false;

    *spread_pct = lx_decimal_mul(
        lx_decimal_div(spread, mid),
        lx_decimal_from_double(100.0)
    );
    return true;
}

/* ============================================================================
 * Liquidity Calculations
 * ============================================================================ */

LxDecimal lx_orderbook_bid_liquidity(const LxOrderbook* book) {
    if (!book) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    LxDecimal total = lx_decimal_zero();
    for (size_t i = 0; i < book->bid_count; i++) {
        total = lx_decimal_add(total, lx_price_level_value(&book->bids[i]));
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return total;
}

LxDecimal lx_orderbook_ask_liquidity(const LxOrderbook* book) {
    if (!book) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    LxDecimal total = lx_decimal_zero();
    for (size_t i = 0; i < book->ask_count; i++) {
        total = lx_decimal_add(total, lx_price_level_value(&book->asks[i]));
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return total;
}

LxDecimal lx_orderbook_bid_depth(const LxOrderbook* book, int levels) {
    if (!book || levels <= 0) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    LxDecimal total = lx_decimal_zero();
    size_t max_levels = (size_t)levels < book->bid_count ? (size_t)levels : book->bid_count;
    for (size_t i = 0; i < max_levels; i++) {
        total = lx_decimal_add(total, lx_price_level_value(&book->bids[i]));
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return total;
}

LxDecimal lx_orderbook_ask_depth(const LxOrderbook* book, int levels) {
    if (!book || levels <= 0) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    LxDecimal total = lx_decimal_zero();
    size_t max_levels = (size_t)levels < book->ask_count ? (size_t)levels : book->ask_count;
    for (size_t i = 0; i < max_levels; i++) {
        total = lx_decimal_add(total, lx_price_level_value(&book->asks[i]));
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return total;
}

bool lx_orderbook_has_liquidity(const LxOrderbook* book, LxSide side, LxDecimal amount) {
    if (!book) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    const LxPriceLevel* levels = (side == LX_SIDE_BUY) ? book->asks : book->bids;
    size_t count = (side == LX_SIDE_BUY) ? book->ask_count : book->bid_count;

    LxDecimal total = lx_decimal_zero();
    for (size_t i = 0; i < count; i++) {
        total = lx_decimal_add(total, levels[i].quantity);
    }

    bool has_liq = lx_decimal_ge(total, amount);

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return has_liq;
}

/* ============================================================================
 * VWAP Calculations
 * ============================================================================ */

bool lx_orderbook_vwap_buy(const LxOrderbook* book, LxDecimal amount, LxDecimal* vwap) {
    if (!book || !vwap) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    if (book->ask_count == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
        return false;
    }

    *vwap = calculate_vwap(book->asks, book->ask_count, amount);
    bool valid = !lx_decimal_is_zero(*vwap);

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return valid;
}

bool lx_orderbook_vwap_sell(const LxOrderbook* book, LxDecimal amount, LxDecimal* vwap) {
    if (!book || !vwap) return false;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    if (book->bid_count == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
        return false;
    }

    *vwap = calculate_vwap(book->bids, book->bid_count, amount);
    bool valid = !lx_decimal_is_zero(*vwap);

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return valid;
}

/* ============================================================================
 * Slippage Calculations
 * ============================================================================ */

bool lx_orderbook_slippage_buy(const LxOrderbook* book, LxDecimal amount, LxDecimal* slippage_pct) {
    if (!book || !slippage_pct) return false;

    LxDecimal best_ask;
    if (!lx_orderbook_best_ask(book, &best_ask)) return false;

    LxDecimal vwap;
    if (!lx_orderbook_vwap_buy(book, amount, &vwap)) return false;

    if (lx_decimal_is_zero(best_ask)) return false;

    /* Slippage = (VWAP - best_ask) / best_ask * 100 */
    *slippage_pct = lx_decimal_mul(
        lx_decimal_div(lx_decimal_sub(vwap, best_ask), best_ask),
        lx_decimal_from_double(100.0)
    );

    return true;
}

bool lx_orderbook_slippage_sell(const LxOrderbook* book, LxDecimal amount, LxDecimal* slippage_pct) {
    if (!book || !slippage_pct) return false;

    LxDecimal best_bid;
    if (!lx_orderbook_best_bid(book, &best_bid)) return false;

    LxDecimal vwap;
    if (!lx_orderbook_vwap_sell(book, amount, &vwap)) return false;

    if (lx_decimal_is_zero(best_bid)) return false;

    /* Slippage = (best_bid - VWAP) / best_bid * 100 (positive means worse execution) */
    *slippage_pct = lx_decimal_mul(
        lx_decimal_div(lx_decimal_sub(best_bid, vwap), best_bid),
        lx_decimal_from_double(100.0)
    );

    return true;
}

/* ============================================================================
 * Aggregated Orderbook Implementation
 * ============================================================================ */

static int ensure_agg_bid_capacity(LxAggregatedOrderbook* agg, size_t required) {
    if (agg->bid_capacity >= required) return 0;

    size_t new_capacity = agg->bid_capacity * 2;
    if (new_capacity < required) new_capacity = required;
    if (new_capacity < INITIAL_CAPACITY) new_capacity = INITIAL_CAPACITY;

    LxAggregatedLevel* new_bids = realloc(agg->bids, new_capacity * sizeof(LxAggregatedLevel));
    if (!new_bids) return -1;

    agg->bids = new_bids;
    agg->bid_capacity = new_capacity;
    return 0;
}

static int ensure_agg_ask_capacity(LxAggregatedOrderbook* agg, size_t required) {
    if (agg->ask_capacity >= required) return 0;

    size_t new_capacity = agg->ask_capacity * 2;
    if (new_capacity < required) new_capacity = required;
    if (new_capacity < INITIAL_CAPACITY) new_capacity = INITIAL_CAPACITY;

    LxAggregatedLevel* new_asks = realloc(agg->asks, new_capacity * sizeof(LxAggregatedLevel));
    if (!new_asks) return -1;

    agg->asks = new_asks;
    agg->ask_capacity = new_capacity;
    return 0;
}

static int ensure_venue_capacity(LxAggregatedLevel* level, size_t required) {
    if (level->venue_capacity >= required) return 0;

    size_t new_capacity = level->venue_capacity * 2;
    if (new_capacity < required) new_capacity = required;
    if (new_capacity < 8) new_capacity = 8;

    LxVenueQuantity* new_venues = realloc(level->venues, new_capacity * sizeof(LxVenueQuantity));
    if (!new_venues) return -1;

    level->venues = new_venues;
    level->venue_capacity = new_capacity;
    return 0;
}

int lx_aggregated_orderbook_init(LxAggregatedOrderbook* agg, const char* symbol) {
    if (!agg) return -1;

    memset(agg, 0, sizeof(LxAggregatedOrderbook));

    if (symbol) {
        strncpy(agg->symbol, symbol, sizeof(agg->symbol) - 1);
    }

    return 0;
}

void lx_aggregated_orderbook_cleanup(LxAggregatedOrderbook* agg) {
    if (!agg) return;

    for (size_t i = 0; i < agg->bid_count; i++) {
        free(agg->bids[i].venues);
    }
    for (size_t i = 0; i < agg->ask_count; i++) {
        free(agg->asks[i].venues);
    }

    free(agg->bids);
    free(agg->asks);

    memset(agg, 0, sizeof(LxAggregatedOrderbook));
}

LxAggregatedOrderbook* lx_aggregated_orderbook_create(const char* symbol) {
    LxAggregatedOrderbook* agg = malloc(sizeof(LxAggregatedOrderbook));
    if (!agg) return NULL;

    if (lx_aggregated_orderbook_init(agg, symbol) != 0) {
        free(agg);
        return NULL;
    }

    return agg;
}

void lx_aggregated_orderbook_destroy(LxAggregatedOrderbook* agg) {
    if (!agg) return;
    lx_aggregated_orderbook_cleanup(agg);
    free(agg);
}

static int add_to_agg_level(LxAggregatedLevel* levels, size_t* count, size_t* capacity,
                            LxDecimal price, const char* venue, LxDecimal quantity,
                            int (*ensure_cap)(LxAggregatedOrderbook*, size_t),
                            LxAggregatedOrderbook* agg) {
    /* Find existing price level */
    for (size_t i = 0; i < *count; i++) {
        if (lx_decimal_eq(levels[i].price, price)) {
            if (ensure_venue_capacity(&levels[i], levels[i].venue_count + 1) != 0) {
                return -1;
            }
            strncpy(levels[i].venues[levels[i].venue_count].venue, venue, LX_VENUE_MAX_LEN - 1);
            levels[i].venues[levels[i].venue_count].quantity = quantity;
            levels[i].venue_count++;
            return 0;
        }
    }

    /* Add new price level */
    if (ensure_cap(agg, *count + 1) != 0) {
        return -1;
    }

    /* Re-get the pointer since it may have been reallocated */
    levels = (levels == agg->bids) ? agg->bids : agg->asks;

    levels[*count].price = price;
    levels[*count].venues = NULL;
    levels[*count].venue_count = 0;
    levels[*count].venue_capacity = 0;

    if (ensure_venue_capacity(&levels[*count], 1) != 0) {
        return -1;
    }

    strncpy(levels[*count].venues[0].venue, venue, LX_VENUE_MAX_LEN - 1);
    levels[*count].venues[0].quantity = quantity;
    levels[*count].venue_count = 1;
    (*count)++;

    return 0;
}

int lx_aggregated_orderbook_add(LxAggregatedOrderbook* agg, const LxOrderbook* book) {
    if (!agg || !book) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&book->lock);

    /* Add bids */
    for (size_t i = 0; i < book->bid_count; i++) {
        if (ensure_agg_bid_capacity(agg, agg->bid_count + 1) != 0) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
            return -1;
        }

        /* Find or create price level */
        bool found = false;
        for (size_t j = 0; j < agg->bid_count; j++) {
            if (lx_decimal_eq(agg->bids[j].price, book->bids[i].price)) {
                if (ensure_venue_capacity(&agg->bids[j], agg->bids[j].venue_count + 1) != 0) {
                    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
                    return -1;
                }
                strncpy(agg->bids[j].venues[agg->bids[j].venue_count].venue,
                        book->venue, LX_VENUE_MAX_LEN - 1);
                agg->bids[j].venues[agg->bids[j].venue_count].quantity = book->bids[i].quantity;
                agg->bids[j].venue_count++;
                found = true;
                break;
            }
        }

        if (!found) {
            agg->bids[agg->bid_count].price = book->bids[i].price;
            agg->bids[agg->bid_count].venues = NULL;
            agg->bids[agg->bid_count].venue_count = 0;
            agg->bids[agg->bid_count].venue_capacity = 0;

            if (ensure_venue_capacity(&agg->bids[agg->bid_count], 1) != 0) {
                pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
                return -1;
            }

            strncpy(agg->bids[agg->bid_count].venues[0].venue, book->venue, LX_VENUE_MAX_LEN - 1);
            agg->bids[agg->bid_count].venues[0].quantity = book->bids[i].quantity;
            agg->bids[agg->bid_count].venue_count = 1;
            agg->bid_count++;
        }
    }

    /* Add asks */
    for (size_t i = 0; i < book->ask_count; i++) {
        if (ensure_agg_ask_capacity(agg, agg->ask_count + 1) != 0) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
            return -1;
        }

        bool found = false;
        for (size_t j = 0; j < agg->ask_count; j++) {
            if (lx_decimal_eq(agg->asks[j].price, book->asks[i].price)) {
                if (ensure_venue_capacity(&agg->asks[j], agg->asks[j].venue_count + 1) != 0) {
                    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
                    return -1;
                }
                strncpy(agg->asks[j].venues[agg->asks[j].venue_count].venue,
                        book->venue, LX_VENUE_MAX_LEN - 1);
                agg->asks[j].venues[agg->asks[j].venue_count].quantity = book->asks[i].quantity;
                agg->asks[j].venue_count++;
                found = true;
                break;
            }
        }

        if (!found) {
            agg->asks[agg->ask_count].price = book->asks[i].price;
            agg->asks[agg->ask_count].venues = NULL;
            agg->asks[agg->ask_count].venue_count = 0;
            agg->asks[agg->ask_count].venue_capacity = 0;

            if (ensure_venue_capacity(&agg->asks[agg->ask_count], 1) != 0) {
                pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
                return -1;
            }

            strncpy(agg->asks[agg->ask_count].venues[0].venue, book->venue, LX_VENUE_MAX_LEN - 1);
            agg->asks[agg->ask_count].venues[0].quantity = book->asks[i].quantity;
            agg->asks[agg->ask_count].venue_count = 1;
            agg->ask_count++;
        }
    }

    int64_t book_ts = atomic_load(&book->timestamp);
    if (book_ts > agg->timestamp) {
        agg->timestamp = book_ts;
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&book->lock);
    return 0;
}

void lx_aggregated_orderbook_clear(LxAggregatedOrderbook* agg) {
    if (!agg) return;

    for (size_t i = 0; i < agg->bid_count; i++) {
        free(agg->bids[i].venues);
        agg->bids[i].venues = NULL;
        agg->bids[i].venue_count = 0;
        agg->bids[i].venue_capacity = 0;
    }
    for (size_t i = 0; i < agg->ask_count; i++) {
        free(agg->asks[i].venues);
        agg->asks[i].venues = NULL;
        agg->asks[i].venue_count = 0;
        agg->asks[i].venue_capacity = 0;
    }

    agg->bid_count = 0;
    agg->ask_count = 0;
    agg->timestamp = 0;
}

bool lx_aggregated_orderbook_best_bid(const LxAggregatedOrderbook* agg,
                                       LxDecimal* price, char* venue, size_t venue_buf_size,
                                       LxDecimal* quantity) {
    if (!agg || agg->bid_count == 0) return false;

    /* Find highest bid */
    LxDecimal best_price = agg->bids[0].price;
    size_t best_idx = 0;

    for (size_t i = 1; i < agg->bid_count; i++) {
        if (lx_decimal_gt(agg->bids[i].price, best_price)) {
            best_price = agg->bids[i].price;
            best_idx = i;
        }
    }

    if (agg->bids[best_idx].venue_count == 0) return false;

    if (price) *price = best_price;
    if (venue && venue_buf_size > 0) {
        strncpy(venue, agg->bids[best_idx].venues[0].venue, venue_buf_size - 1);
        venue[venue_buf_size - 1] = '\0';
    }
    if (quantity) *quantity = agg->bids[best_idx].venues[0].quantity;

    return true;
}

bool lx_aggregated_orderbook_best_ask(const LxAggregatedOrderbook* agg,
                                       LxDecimal* price, char* venue, size_t venue_buf_size,
                                       LxDecimal* quantity) {
    if (!agg || agg->ask_count == 0) return false;

    /* Find lowest ask */
    LxDecimal best_price = agg->asks[0].price;
    size_t best_idx = 0;

    for (size_t i = 1; i < agg->ask_count; i++) {
        if (lx_decimal_lt(agg->asks[i].price, best_price)) {
            best_price = agg->asks[i].price;
            best_idx = i;
        }
    }

    if (agg->asks[best_idx].venue_count == 0) return false;

    if (price) *price = best_price;
    if (venue && venue_buf_size > 0) {
        strncpy(venue, agg->asks[best_idx].venues[0].venue, venue_buf_size - 1);
        venue[venue_buf_size - 1] = '\0';
    }
    if (quantity) *quantity = agg->asks[best_idx].venues[0].quantity;

    return true;
}

static int compare_agg_bids_desc(const void* a, const void* b) {
    const LxAggregatedLevel* la = (const LxAggregatedLevel*)a;
    const LxAggregatedLevel* lb = (const LxAggregatedLevel*)b;
    if (lx_decimal_gt(la->price, lb->price)) return -1;
    if (lx_decimal_lt(la->price, lb->price)) return 1;
    return 0;
}

static int compare_agg_asks_asc(const void* a, const void* b) {
    const LxAggregatedLevel* la = (const LxAggregatedLevel*)a;
    const LxAggregatedLevel* lb = (const LxAggregatedLevel*)b;
    if (lx_decimal_lt(la->price, lb->price)) return -1;
    if (lx_decimal_gt(la->price, lb->price)) return 1;
    return 0;
}

LxPriceLevel* lx_aggregated_orderbook_aggregated_bids(const LxAggregatedOrderbook* agg, size_t* count) {
    if (!agg || !count) {
        if (count) *count = 0;
        return NULL;
    }

    if (agg->bid_count == 0) {
        *count = 0;
        return NULL;
    }

    LxPriceLevel* result = malloc(agg->bid_count * sizeof(LxPriceLevel));
    if (!result) {
        *count = 0;
        return NULL;
    }

    for (size_t i = 0; i < agg->bid_count; i++) {
        result[i].price = agg->bids[i].price;
        result[i].quantity = lx_decimal_zero();
        for (size_t j = 0; j < agg->bids[i].venue_count; j++) {
            result[i].quantity = lx_decimal_add(result[i].quantity, agg->bids[i].venues[j].quantity);
        }
    }

    /* Sort descending */
    qsort(result, agg->bid_count, sizeof(LxPriceLevel), compare_bids_desc);

    *count = agg->bid_count;
    return result;
}

LxPriceLevel* lx_aggregated_orderbook_aggregated_asks(const LxAggregatedOrderbook* agg, size_t* count) {
    if (!agg || !count) {
        if (count) *count = 0;
        return NULL;
    }

    if (agg->ask_count == 0) {
        *count = 0;
        return NULL;
    }

    LxPriceLevel* result = malloc(agg->ask_count * sizeof(LxPriceLevel));
    if (!result) {
        *count = 0;
        return NULL;
    }

    for (size_t i = 0; i < agg->ask_count; i++) {
        result[i].price = agg->asks[i].price;
        result[i].quantity = lx_decimal_zero();
        for (size_t j = 0; j < agg->asks[i].venue_count; j++) {
            result[i].quantity = lx_decimal_add(result[i].quantity, agg->asks[i].venues[j].quantity);
        }
    }

    /* Sort ascending */
    qsort(result, agg->ask_count, sizeof(LxPriceLevel), compare_asks_asc);

    *count = agg->ask_count;
    return result;
}

bool lx_aggregated_orderbook_best_venue_buy(const LxAggregatedOrderbook* agg, LxDecimal amount,
                                             char* venue, size_t venue_buf_size, LxDecimal* price) {
    if (!agg || agg->ask_count == 0) return false;

    /* Find lowest ask price */
    LxDecimal best_price = agg->asks[0].price;
    size_t best_idx = 0;

    for (size_t i = 1; i < agg->ask_count; i++) {
        if (lx_decimal_lt(agg->asks[i].price, best_price) && agg->asks[i].venue_count > 0) {
            best_price = agg->asks[i].price;
            best_idx = i;
        }
    }

    if (agg->asks[best_idx].venue_count == 0) return false;

    if (venue && venue_buf_size > 0) {
        strncpy(venue, agg->asks[best_idx].venues[0].venue, venue_buf_size - 1);
        venue[venue_buf_size - 1] = '\0';
    }
    if (price) *price = best_price;

    return true;
}

bool lx_aggregated_orderbook_best_venue_sell(const LxAggregatedOrderbook* agg, LxDecimal amount,
                                              char* venue, size_t venue_buf_size, LxDecimal* price) {
    if (!agg || agg->bid_count == 0) return false;

    /* Find highest bid price */
    LxDecimal best_price = agg->bids[0].price;
    size_t best_idx = 0;

    for (size_t i = 1; i < agg->bid_count; i++) {
        if (lx_decimal_gt(agg->bids[i].price, best_price) && agg->bids[i].venue_count > 0) {
            best_price = agg->bids[i].price;
            best_idx = i;
        }
    }

    if (agg->bids[best_idx].venue_count == 0) return false;

    if (venue && venue_buf_size > 0) {
        strncpy(venue, agg->bids[best_idx].venues[0].venue, venue_buf_size - 1);
        venue[venue_buf_size - 1] = '\0';
    }
    if (price) *price = best_price;

    return true;
}
