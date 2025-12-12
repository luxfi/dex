/*
 * LX C SDK - Basic Example
 *
 * Demonstrates connecting to the DEX, subscribing to market data,
 * and placing orders.
 *
 * Build:
 *   cd sdk/c && mkdir build && cd build
 *   cmake .. && make
 *   ./basic
 */

#include "lxdex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

/* Global state */
static volatile sig_atomic_t g_running = 1;
static lxdex_client_t *g_client = NULL;

/* Signal handler for clean shutdown */
static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* Callbacks */

static void on_connect(lxdex_client_t *client, void *user_data) {
    (void)user_data;
    printf("[CONNECTED] Connected to DEX\n");

    /* Authenticate if credentials provided */
    lxdex_error_t err = lxdex_client_auth(client);
    if (err == LXDEX_OK) {
        printf("[AUTH] Authenticating...\n");
    }
}

static void on_disconnect(lxdex_client_t *client, int code, const char *reason, void *user_data) {
    (void)client;
    (void)user_data;
    printf("[DISCONNECTED] Code: %d, Reason: %s\n", code, reason ? reason : "unknown");
}

static void on_error(lxdex_client_t *client, lxdex_error_t error, const char *msg, void *user_data) {
    (void)client;
    (void)user_data;
    printf("[ERROR] %s: %s\n", lxdex_strerror(error), msg ? msg : "");
}

static void on_order_update(lxdex_client_t *client, const lxdex_order_t *order, void *user_data) {
    (void)client;
    (void)user_data;

    const char *status_str = "unknown";
    switch (order->status) {
        case LXDEX_STATUS_OPEN: status_str = "open"; break;
        case LXDEX_STATUS_PARTIAL: status_str = "partial"; break;
        case LXDEX_STATUS_FILLED: status_str = "filled"; break;
        case LXDEX_STATUS_CANCELLED: status_str = "cancelled"; break;
        case LXDEX_STATUS_REJECTED: status_str = "rejected"; break;
    }

    printf("[ORDER] ID: %llu, Symbol: %s, Side: %s, Price: %.8f, Size: %.8f, Status: %s\n",
        (unsigned long long)order->order_id,
        order->symbol,
        order->side == LXDEX_SIDE_BUY ? "BUY" : "SELL",
        order->price,
        order->size,
        status_str);
}

static void on_trade(lxdex_client_t *client, const lxdex_trade_t *trade, void *user_data) {
    (void)client;
    (void)user_data;

    printf("[TRADE] ID: %llu, Symbol: %s, Price: %.8f, Size: %.8f, Side: %s\n",
        (unsigned long long)trade->trade_id,
        trade->symbol,
        trade->price,
        trade->size,
        trade->side == LXDEX_SIDE_BUY ? "BUY" : "SELL");
}

static void on_orderbook(lxdex_client_t *client, const lxdex_orderbook_t *book, void *user_data) {
    (void)client;
    (void)user_data;

    printf("[ORDERBOOK] %s - Best Bid: %.8f, Best Ask: %.8f, Spread: %.8f\n",
        book->symbol,
        lxdex_orderbook_best_bid(book),
        lxdex_orderbook_best_ask(book),
        lxdex_orderbook_spread(book));

    /* Print top 3 levels */
    printf("  Bids: ");
    for (size_t i = 0; i < book->bids_count && i < 3; i++) {
        printf("%.8f@%.8f ", book->bids[i].price, book->bids[i].size);
    }
    printf("\n  Asks: ");
    for (size_t i = 0; i < book->asks_count && i < 3; i++) {
        printf("%.8f@%.8f ", book->asks[i].price, book->asks[i].size);
    }
    printf("\n");
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -u URL    WebSocket URL (default: ws://localhost:8081)\n");
    printf("  -k KEY    API key\n");
    printf("  -s SECRET API secret\n");
    printf("  -m SYMBOL Subscribe to market (default: BTC-USD)\n");
    printf("  -o        Place a test order\n");
    printf("  -h        Show this help\n");
}

int main(int argc, char *argv[]) {
    const char *ws_url = "ws://localhost:8081";
    const char *api_key = NULL;
    const char *api_secret = NULL;
    const char *symbol = "BTC-USD";
    bool place_order = false;

    /* Parse arguments */
    int opt;
    while ((opt = getopt(argc, argv, "u:k:s:m:oh")) != -1) {
        switch (opt) {
            case 'u':
                ws_url = optarg;
                break;
            case 'k':
                api_key = optarg;
                break;
            case 's':
                api_secret = optarg;
                break;
            case 'm':
                symbol = optarg;
                break;
            case 'o':
                place_order = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    printf("LX C SDK Example v%s\n", lxdex_version());
    printf("Connecting to: %s\n", ws_url);

    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Initialize library */
    lxdex_error_t err = lxdex_init();
    if (err != LXDEX_OK) {
        fprintf(stderr, "Failed to initialize: %s\n", lxdex_strerror(err));
        return 1;
    }

    /* Configure client */
    lxdex_config_t config = {
        .ws_url = ws_url,
        .api_key = api_key,
        .api_secret = api_secret,
        .connect_timeout_ms = 10000,
        .recv_timeout_ms = 30000,
        .reconnect_interval_ms = 5000,
        .auto_reconnect = true,
    };

    /* Create client */
    g_client = lxdex_client_new(&config);
    if (!g_client) {
        fprintf(stderr, "Failed to create client\n");
        lxdex_cleanup();
        return 1;
    }

    /* Set callbacks */
    lxdex_callbacks_t callbacks = {
        .on_connect = on_connect,
        .on_disconnect = on_disconnect,
        .on_error = on_error,
        .on_order_update = on_order_update,
        .on_trade = on_trade,
        .on_orderbook = on_orderbook,
        .user_data = NULL,
    };
    lxdex_client_set_callbacks(g_client, &callbacks);

    /* Connect */
    err = lxdex_client_connect(g_client);
    if (err != LXDEX_OK) {
        fprintf(stderr, "Failed to connect: %s\n", lxdex_strerror(err));
        lxdex_client_free(g_client);
        lxdex_cleanup();
        return 1;
    }

    /* Wait for connection */
    printf("Connecting...\n");
    int wait_count = 0;
    while (g_running && lxdex_client_state(g_client) == LXDEX_STATE_CONNECTING) {
        lxdex_client_service(g_client, 100);
        if (++wait_count > 100) { /* 10 second timeout */
            fprintf(stderr, "Connection timeout\n");
            break;
        }
    }

    /* Subscribe to market data */
    if (lxdex_client_state(g_client) >= LXDEX_STATE_CONNECTED) {
        printf("Subscribing to %s orderbook and trades...\n", symbol);

        err = lxdex_subscribe_orderbook(g_client, symbol);
        if (err != LXDEX_OK) {
            fprintf(stderr, "Failed to subscribe to orderbook: %s\n", lxdex_strerror(err));
        }

        err = lxdex_subscribe_trades(g_client, symbol);
        if (err != LXDEX_OK) {
            fprintf(stderr, "Failed to subscribe to trades: %s\n", lxdex_strerror(err));
        }
    }

    /* Wait for authentication if credentials provided */
    if (api_key && api_secret) {
        wait_count = 0;
        while (g_running &&
               lxdex_client_state(g_client) == LXDEX_STATE_CONNECTED &&
               wait_count < 50) {
            lxdex_client_service(g_client, 100);
            wait_count++;
        }

        if (lxdex_client_state(g_client) == LXDEX_STATE_AUTHENTICATED) {
            printf("Authenticated successfully\n");

            /* Place test order if requested */
            if (place_order) {
                printf("Placing test limit order...\n");

                lxdex_order_t order;
                lxdex_order_limit(&order, symbol, LXDEX_SIDE_BUY, 50000.0, 0.001);
                order.post_only = true;  /* Don't take liquidity */

                uint64_t order_id;
                err = lxdex_place_order(g_client, &order, &order_id);
                if (err != LXDEX_OK) {
                    fprintf(stderr, "Failed to place order: %s\n", lxdex_strerror(err));
                } else {
                    printf("Order submitted\n");
                }
            }
        } else {
            fprintf(stderr, "Authentication failed or timed out\n");
        }
    }

    /* Main event loop */
    printf("Running... Press Ctrl+C to exit\n");
    while (g_running) {
        lxdex_client_service(g_client, 100);

        /* Check connection state */
        lxdex_conn_state_t state = lxdex_client_state(g_client);
        if (state == LXDEX_STATE_DISCONNECTED || state == LXDEX_STATE_ERROR) {
            printf("Connection lost\n");
            break;
        }
    }

    /* Cleanup */
    printf("\nShutting down...\n");
    lxdex_client_disconnect(g_client);
    lxdex_client_free(g_client);
    lxdex_cleanup();

    printf("Done\n");
    return 0;
}
