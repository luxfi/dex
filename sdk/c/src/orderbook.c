/*
 * LX C SDK - Orderbook utilities
 * Local orderbook management and helper functions.
 */

#include "lxdex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * Order utilities
 */

void lxdex_order_init(lxdex_order_t *order) {
    if (!order) return;
    memset(order, 0, sizeof(*order));
    order->type = LXDEX_ORDER_LIMIT;
    order->side = LXDEX_SIDE_BUY;
    order->time_in_force = LXDEX_TIF_GTC;
}

void lxdex_order_limit(lxdex_order_t *order, const char *symbol,
                       lxdex_side_t side, double price, double size) {
    if (!order || !symbol) return;

    lxdex_order_init(order);
    strncpy(order->symbol, symbol, LXDEX_SYMBOL_LEN - 1);
    order->symbol[LXDEX_SYMBOL_LEN - 1] = '\0';
    order->type = LXDEX_ORDER_LIMIT;
    order->side = side;
    order->price = price;
    order->size = size;
}

void lxdex_order_market(lxdex_order_t *order, const char *symbol,
                        lxdex_side_t side, double size) {
    if (!order || !symbol) return;

    lxdex_order_init(order);
    strncpy(order->symbol, symbol, LXDEX_SYMBOL_LEN - 1);
    order->symbol[LXDEX_SYMBOL_LEN - 1] = '\0';
    order->type = LXDEX_ORDER_MARKET;
    order->side = side;
    order->size = size;
    order->time_in_force = LXDEX_TIF_IOC; /* Market orders are IOC by default */
}

/*
 * Orderbook utilities
 */

lxdex_error_t lxdex_orderbook_init(lxdex_orderbook_t *book, size_t initial_capacity) {
    if (!book) return LXDEX_ERR_INVALID_ARG;

    memset(book, 0, sizeof(*book));

    if (initial_capacity == 0) {
        initial_capacity = 20; /* Default depth */
    }

    book->bids = calloc(initial_capacity, sizeof(lxdex_price_level_t));
    if (!book->bids) return LXDEX_ERR_NO_MEMORY;
    book->bids_capacity = initial_capacity;

    book->asks = calloc(initial_capacity, sizeof(lxdex_price_level_t));
    if (!book->asks) {
        free(book->bids);
        book->bids = NULL;
        return LXDEX_ERR_NO_MEMORY;
    }
    book->asks_capacity = initial_capacity;

    return LXDEX_OK;
}

void lxdex_orderbook_free(lxdex_orderbook_t *book) {
    if (!book) return;

    free(book->bids);
    free(book->asks);

    book->bids = NULL;
    book->asks = NULL;
    book->bids_count = 0;
    book->asks_count = 0;
    book->bids_capacity = 0;
    book->asks_capacity = 0;
}

double lxdex_orderbook_best_bid(const lxdex_orderbook_t *book) {
    if (!book || book->bids_count == 0) return 0.0;
    return book->bids[0].price;
}

double lxdex_orderbook_best_ask(const lxdex_orderbook_t *book) {
    if (!book || book->asks_count == 0) return 0.0;
    return book->asks[0].price;
}

double lxdex_orderbook_spread(const lxdex_orderbook_t *book) {
    double bid = lxdex_orderbook_best_bid(book);
    double ask = lxdex_orderbook_best_ask(book);

    if (bid > 0.0 && ask > 0.0) {
        return ask - bid;
    }
    return 0.0;
}

double lxdex_orderbook_mid(const lxdex_orderbook_t *book) {
    double bid = lxdex_orderbook_best_bid(book);
    double ask = lxdex_orderbook_best_ask(book);

    if (bid > 0.0 && ask > 0.0) {
        return (bid + ask) / 2.0;
    }
    return 0.0;
}

/*
 * Local orderbook operations (for client-side order management)
 */

/* Binary search for price level (bids sorted descending, asks sorted ascending) */
static size_t find_bid_level(const lxdex_orderbook_t *book, double price) {
    if (book->bids_count == 0) return 0;

    size_t lo = 0;
    size_t hi = book->bids_count;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (book->bids[mid].price > price) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static size_t find_ask_level(const lxdex_orderbook_t *book, double price) {
    if (book->asks_count == 0) return 0;

    size_t lo = 0;
    size_t hi = book->asks_count;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (book->asks[mid].price < price) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* Update a price level (insert/update/delete) */
lxdex_error_t lxdex_orderbook_update_bid(lxdex_orderbook_t *book,
                                          double price, double size, int count) {
    if (!book) return LXDEX_ERR_INVALID_ARG;

    size_t idx = find_bid_level(book, price);

    /* Check if price already exists */
    if (idx < book->bids_count && book->bids[idx].price == price) {
        if (size <= 0.0) {
            /* Delete level */
            memmove(&book->bids[idx], &book->bids[idx + 1],
                (book->bids_count - idx - 1) * sizeof(lxdex_price_level_t));
            book->bids_count--;
        } else {
            /* Update level */
            book->bids[idx].size = size;
            book->bids[idx].count = count;
        }
        return LXDEX_OK;
    }

    /* Insert new level */
    if (size <= 0.0) {
        return LXDEX_OK; /* Nothing to insert */
    }

    /* Expand if needed */
    if (book->bids_count >= book->bids_capacity) {
        size_t new_cap = book->bids_capacity * 2;
        lxdex_price_level_t *new_bids = realloc(book->bids,
            sizeof(lxdex_price_level_t) * new_cap);
        if (!new_bids) return LXDEX_ERR_NO_MEMORY;
        book->bids = new_bids;
        book->bids_capacity = new_cap;
    }

    /* Make room and insert */
    memmove(&book->bids[idx + 1], &book->bids[idx],
        (book->bids_count - idx) * sizeof(lxdex_price_level_t));

    book->bids[idx].price = price;
    book->bids[idx].size = size;
    book->bids[idx].count = count;
    book->bids_count++;

    return LXDEX_OK;
}

lxdex_error_t lxdex_orderbook_update_ask(lxdex_orderbook_t *book,
                                          double price, double size, int count) {
    if (!book) return LXDEX_ERR_INVALID_ARG;

    size_t idx = find_ask_level(book, price);

    /* Check if price already exists */
    if (idx < book->asks_count && book->asks[idx].price == price) {
        if (size <= 0.0) {
            /* Delete level */
            memmove(&book->asks[idx], &book->asks[idx + 1],
                (book->asks_count - idx - 1) * sizeof(lxdex_price_level_t));
            book->asks_count--;
        } else {
            /* Update level */
            book->asks[idx].size = size;
            book->asks[idx].count = count;
        }
        return LXDEX_OK;
    }

    /* Insert new level */
    if (size <= 0.0) {
        return LXDEX_OK; /* Nothing to insert */
    }

    /* Expand if needed */
    if (book->asks_count >= book->asks_capacity) {
        size_t new_cap = book->asks_capacity * 2;
        lxdex_price_level_t *new_asks = realloc(book->asks,
            sizeof(lxdex_price_level_t) * new_cap);
        if (!new_asks) return LXDEX_ERR_NO_MEMORY;
        book->asks = new_asks;
        book->asks_capacity = new_cap;
    }

    /* Make room and insert */
    memmove(&book->asks[idx + 1], &book->asks[idx],
        (book->asks_count - idx) * sizeof(lxdex_price_level_t));

    book->asks[idx].price = price;
    book->asks[idx].size = size;
    book->asks[idx].count = count;
    book->asks_count++;

    return LXDEX_OK;
}

/* Clear the orderbook */
void lxdex_orderbook_clear(lxdex_orderbook_t *book) {
    if (!book) return;
    book->bids_count = 0;
    book->asks_count = 0;
}

/* Get total bid volume */
double lxdex_orderbook_bid_volume(const lxdex_orderbook_t *book, int depth) {
    if (!book) return 0.0;

    double volume = 0.0;
    size_t count = (depth > 0 && (size_t)depth < book->bids_count)
        ? (size_t)depth : book->bids_count;

    for (size_t i = 0; i < count; i++) {
        volume += book->bids[i].size;
    }

    return volume;
}

/* Get total ask volume */
double lxdex_orderbook_ask_volume(const lxdex_orderbook_t *book, int depth) {
    if (!book) return 0.0;

    double volume = 0.0;
    size_t count = (depth > 0 && (size_t)depth < book->asks_count)
        ? (size_t)depth : book->asks_count;

    for (size_t i = 0; i < count; i++) {
        volume += book->asks[i].size;
    }

    return volume;
}

/* Get price for a given size (market impact) */
double lxdex_orderbook_price_for_size(const lxdex_orderbook_t *book,
                                       lxdex_side_t side, double size) {
    if (!book || size <= 0.0) return 0.0;

    const lxdex_price_level_t *levels;
    size_t count;

    if (side == LXDEX_SIDE_BUY) {
        /* Buying from asks */
        levels = book->asks;
        count = book->asks_count;
    } else {
        /* Selling into bids */
        levels = book->bids;
        count = book->bids_count;
    }

    double remaining = size;
    double total_cost = 0.0;

    for (size_t i = 0; i < count && remaining > 0.0; i++) {
        double fill = (levels[i].size < remaining) ? levels[i].size : remaining;
        total_cost += fill * levels[i].price;
        remaining -= fill;
    }

    if (remaining > 0.0) {
        return 0.0; /* Not enough liquidity */
    }

    return total_cost / size; /* VWAP */
}

/* Print orderbook to stdout (for debugging) */
void lxdex_orderbook_print(const lxdex_orderbook_t *book, int depth) {
    if (!book) return;

    printf("=== %s Orderbook ===\n", book->symbol);
    printf("Timestamp: %lld\n", (long long)book->timestamp);
    printf("\n        ASKS\n");
    printf("%-12s %-12s %-8s\n", "Price", "Size", "Count");

    /* Print asks (reversed, bottom to top) */
    size_t ask_depth = (depth > 0 && (size_t)depth < book->asks_count)
        ? (size_t)depth : book->asks_count;

    for (size_t i = ask_depth; i > 0; i--) {
        printf("%-12.8f %-12.8f %-8d\n",
            book->asks[i - 1].price,
            book->asks[i - 1].size,
            book->asks[i - 1].count);
    }

    double spread = lxdex_orderbook_spread(book);
    double mid = lxdex_orderbook_mid(book);
    printf("\n--- Spread: %.8f (%.4f%%) Mid: %.8f ---\n\n",
        spread, (mid > 0 ? (spread / mid) * 100.0 : 0.0), mid);

    printf("        BIDS\n");
    printf("%-12s %-12s %-8s\n", "Price", "Size", "Count");

    size_t bid_depth = (depth > 0 && (size_t)depth < book->bids_count)
        ? (size_t)depth : book->bids_count;

    for (size_t i = 0; i < bid_depth; i++) {
        printf("%-12.8f %-12.8f %-8d\n",
            book->bids[i].price,
            book->bids[i].size,
            book->bids[i].count);
    }

    printf("\n");
}
