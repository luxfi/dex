/*
 * LX C CLI Trading Client
 * Copyright (c) 2025 Lux Partners Limited
 *
 * Ultra-fast command-line trading client for HFT and embedded systems.
 * Connects to LX WebSocket API for real-time trading.
 *
 * Usage:
 *   lx-cli -i                          # Interactive REPL mode
 *   lx-cli place_order BTC-USD buy limit 50000 0.1
 *   lx-cli cancel_order 12345
 *   lx-cli get_orders
 */

/* Platform compatibility - must be before any includes */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#else
#define _POSIX_C_SOURCE 200809L
#endif

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <sys/time.h>

#include <libwebsockets.h>

/* Version */
#define CLI_VERSION "1.0.0"

/* Limits */
#define MAX_MSG_LEN     65536
#define MAX_CMD_LEN     1024
#define MAX_SYMBOL_LEN  32
#define MAX_TOKEN_LEN   16
#define MAX_URL_LEN     256
#define RECV_BUF_INIT   4096

/* Error codes */
typedef enum {
    CLI_OK = 0,
    CLI_ERR_ARGS = 1,
    CLI_ERR_CONN = 2,
    CLI_ERR_AUTH = 3,
    CLI_ERR_TIMEOUT = 4,
    CLI_ERR_PROTO = 5,
    CLI_ERR_MEMORY = 6,
} cli_error_t;

/* Order types */
typedef enum {
    ORDER_LIMIT = 0,
    ORDER_MARKET,
    ORDER_STOP,
} order_type_t;

/* Order sides */
typedef enum {
    SIDE_BUY = 0,
    SIDE_SELL,
} side_t;

/* Connection state */
typedef enum {
    STATE_DISCONNECTED = 0,
    STATE_CONNECTING,
    STATE_CONNECTED,
    STATE_AUTHENTICATED,
} conn_state_t;

/* Send buffer entry */
typedef struct send_buf {
    unsigned char *data;
    size_t len;
    struct send_buf *next;
} send_buf_t;

/* CLI context */
typedef struct {
    /* Config */
    char ws_url[MAX_URL_LEN];
    char api_key[128];
    char api_secret[128];
    bool verbose;
    bool interactive;
    int timeout_ms;

    /* Connection */
    struct lws_context *lws_ctx;
    struct lws *wsi;
    atomic_int state;

    /* Send queue */
    send_buf_t *send_head;
    send_buf_t *send_tail;
    pthread_mutex_t send_mutex;

    /* Receive buffer */
    char *recv_buf;
    size_t recv_len;
    size_t recv_cap;

    /* Request tracking */
    atomic_uint_fast64_t req_id;
    char pending_req[64];
    char *response;
    pthread_mutex_t resp_mutex;
    pthread_cond_t resp_cond;
    bool resp_ready;

    /* Control */
    volatile sig_atomic_t running;
} cli_ctx_t;

/* Global context for signal handler */
static cli_ctx_t *g_ctx = NULL;

/* Forward declarations */
static void print_help(void);
static int run_interactive(cli_ctx_t *ctx);
static int run_command(cli_ctx_t *ctx, int argc, char **argv);

/*
 * JSON helpers - minimal inline builders
 */

static char *json_escape(const char *s, char *buf, size_t buflen) {
    size_t i = 0, j = 0;
    while (s[i] && j < buflen - 2) {
        if (s[i] == '"' || s[i] == '\\') {
            buf[j++] = '\\';
        }
        buf[j++] = s[i++];
    }
    buf[j] = '\0';
    return buf;
}

static char *build_auth_msg(const char *key, const char *secret, const char *req_id) {
    static char buf[512];
    char ek[256], es[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"auth\",\"apiKey\":\"%s\",\"apiSecret\":\"%s\",\"request_id\":\"%s\"}",
        json_escape(key, ek, sizeof(ek)),
        json_escape(secret, es, sizeof(es)),
        req_id);
    return buf;
}

static char *build_place_order_msg(const char *symbol, const char *side,
                                    const char *type, double price, double size,
                                    const char *req_id) {
    static char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"place_order\",\"order\":{"
        "\"symbol\":\"%s\",\"side\":\"%s\",\"type\":\"%s\","
        "\"price\":%.15g,\"size\":%.15g},\"request_id\":\"%s\"}",
        symbol, side, type, price, size, req_id);
    return buf;
}

static char *build_cancel_order_msg(uint64_t order_id, const char *req_id) {
    static char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"cancel_order\",\"orderID\":%llu,\"request_id\":\"%s\"}",
        (unsigned long long)order_id, req_id);
    return buf;
}

static char *build_subscribe_msg(const char *symbols, const char *req_id) {
    static char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"subscribe\",\"symbols\":[\"%s\"],\"request_id\":\"%s\"}",
        symbols, req_id);
    return buf;
}

static char *build_simple_msg(const char *type, const char *req_id) {
    static char buf[128];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"%s\",\"request_id\":\"%s\"}", type, req_id);
    return buf;
}

/*
 * JSON parsing - minimal extraction
 */

static const char *json_get_string(const char *json, const char *key, char *out, size_t outlen) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;

    p += strlen(pattern);
    while (*p && isspace(*p)) p++;

    if (*p != '"') return NULL;
    p++;

    size_t i = 0;
    while (*p && *p != '"' && i < outlen - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n': out[i++] = '\n'; break;
                case 't': out[i++] = '\t'; break;
                case 'r': out[i++] = '\r'; break;
                default: out[i++] = *p;
            }
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return out;
}

static double json_get_number(const char *json, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0.0;

    p += strlen(pattern);
    while (*p && isspace(*p)) p++;

    return strtod(p, NULL);
}

static bool json_has_key(const char *json, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    return strstr(json, pattern) != NULL;
}

/*
 * Generate request ID
 */
static char *next_req_id(cli_ctx_t *ctx) {
    static char buf[32];
    uint64_t id = atomic_fetch_add(&ctx->req_id, 1);
    snprintf(buf, sizeof(buf), "req_%llu", (unsigned long long)id);
    return buf;
}

/*
 * Queue message for sending
 */
static cli_error_t queue_send(cli_ctx_t *ctx, const char *msg) {
    if (!ctx || !msg) return CLI_ERR_ARGS;

    size_t len = strlen(msg);
    send_buf_t *buf = malloc(sizeof(send_buf_t));
    if (!buf) return CLI_ERR_MEMORY;

    /* LWS requires LWS_PRE bytes before data */
    buf->data = malloc(LWS_PRE + len);
    if (!buf->data) {
        free(buf);
        return CLI_ERR_MEMORY;
    }

    memcpy(buf->data + LWS_PRE, msg, len);
    buf->len = len;
    buf->next = NULL;

    pthread_mutex_lock(&ctx->send_mutex);
    if (ctx->send_tail) {
        ctx->send_tail->next = buf;
        ctx->send_tail = buf;
    } else {
        ctx->send_head = ctx->send_tail = buf;
    }
    pthread_mutex_unlock(&ctx->send_mutex);

    if (ctx->wsi) {
        lws_callback_on_writable(ctx->wsi);
    }

    if (ctx->verbose) {
        fprintf(stderr, ">> %s\n", msg);
    }

    return CLI_OK;
}

/*
 * Send and wait for response
 */
static char *send_and_wait(cli_ctx_t *ctx, const char *msg, const char *req_id, int timeout_ms) {
    pthread_mutex_lock(&ctx->resp_mutex);
    strncpy(ctx->pending_req, req_id, sizeof(ctx->pending_req) - 1);
    ctx->resp_ready = false;
    if (ctx->response) {
        free(ctx->response);
        ctx->response = NULL;
    }
    pthread_mutex_unlock(&ctx->resp_mutex);

    cli_error_t err = queue_send(ctx, msg);
    if (err != CLI_OK) return NULL;

    /* Wait for response */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }

    pthread_mutex_lock(&ctx->resp_mutex);
    while (!ctx->resp_ready && ctx->running) {
        int rc = pthread_cond_timedwait(&ctx->resp_cond, &ctx->resp_mutex, &ts);
        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&ctx->resp_mutex);
            return NULL;
        }
    }
    char *resp = ctx->response;
    ctx->response = NULL;
    ctx->pending_req[0] = '\0';
    pthread_mutex_unlock(&ctx->resp_mutex);

    return resp;
}

/*
 * Process incoming message
 */
static void process_message(cli_ctx_t *ctx, const char *msg, size_t len) {
    if (!ctx || !msg || len == 0) return;

    char *json = malloc(len + 1);
    if (!json) return;
    memcpy(json, msg, len);
    json[len] = '\0';

    if (ctx->verbose) {
        fprintf(stderr, "<< %s\n", json);
    }

    char type[64] = {0};
    json_get_string(json, "type", type, sizeof(type));

    char req_id[64] = {0};
    json_get_string(json, "request_id", req_id, sizeof(req_id));

    /* Handle connection events */
    if (strcmp(type, "connected") == 0) {
        atomic_store(&ctx->state, STATE_CONNECTED);
        free(json);
        return;
    }

    if (strcmp(type, "auth_success") == 0) {
        atomic_store(&ctx->state, STATE_AUTHENTICATED);
    }

    /* Check if this is a response to pending request */
    pthread_mutex_lock(&ctx->resp_mutex);
    if (ctx->pending_req[0] && req_id[0] && strcmp(ctx->pending_req, req_id) == 0) {
        ctx->response = json;
        ctx->resp_ready = true;
        pthread_cond_signal(&ctx->resp_cond);
        pthread_mutex_unlock(&ctx->resp_mutex);
        return; /* Don't free, transferred to response */
    }
    pthread_mutex_unlock(&ctx->resp_mutex);

    /* Print non-response messages */
    if (strcmp(type, "order_update") == 0) {
        char symbol[32] = {0}, status[32] = {0};
        json_get_string(json, "symbol", symbol, sizeof(symbol));
        json_get_string(json, "status", status, sizeof(status));
        double price = json_get_number(json, "price");
        double size = json_get_number(json, "size");
        printf("Order Update: %s %s @ %.2f x %.4f\n", symbol, status, price, size);
    }
    else if (strcmp(type, "trade") == 0) {
        char symbol[32] = {0}, side[8] = {0};
        json_get_string(json, "symbol", symbol, sizeof(symbol));
        json_get_string(json, "side", side, sizeof(side));
        double price = json_get_number(json, "price");
        double size = json_get_number(json, "size");
        printf("Trade: %s %s %.4f @ %.2f\n", symbol, side, size, price);
    }
    else if (strcmp(type, "orderbook") == 0 || strcmp(type, "orderbook_update") == 0) {
        char symbol[32] = {0};
        json_get_string(json, "symbol", symbol, sizeof(symbol));
        printf("OrderBook [%s] updated\n", symbol);
    }
    else if (strcmp(type, "error") == 0) {
        char err[256] = {0};
        json_get_string(json, "error", err, sizeof(err));
        fprintf(stderr, "Error: %s\n", err);
    }
    else if (strcmp(type, "pong") != 0 && ctx->interactive) {
        /* Print other messages in interactive mode */
        printf("%s\n", json);
    }

    free(json);
}

/*
 * libwebsockets callback
 */
static int cli_lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                            void *user, void *in, size_t len) {
    (void)user; /* unused */

    cli_ctx_t *ctx = NULL;
    struct lws_context *lws_ctx = lws_get_context(wsi);
    if (lws_ctx) {
        ctx = lws_context_user(lws_ctx);
    }

    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            if (ctx) {
                atomic_store(&ctx->state, STATE_CONNECTED);
            }
            break;

        case LWS_CALLBACK_CLIENT_RECEIVE:
            if (ctx && in && len > 0) {
                /* Accumulate fragments */
                size_t needed = ctx->recv_len + len;
                if (needed > ctx->recv_cap) {
                    size_t new_cap = ctx->recv_cap * 2;
                    if (new_cap < needed) new_cap = needed + 1024;
                    char *new_buf = realloc(ctx->recv_buf, new_cap);
                    if (!new_buf) break;
                    ctx->recv_buf = new_buf;
                    ctx->recv_cap = new_cap;
                }

                memcpy(ctx->recv_buf + ctx->recv_len, in, len);
                ctx->recv_len += len;

                if (lws_is_final_fragment(wsi)) {
                    process_message(ctx, ctx->recv_buf, ctx->recv_len);
                    ctx->recv_len = 0;
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_WRITEABLE:
            if (ctx) {
                pthread_mutex_lock(&ctx->send_mutex);
                send_buf_t *buf = ctx->send_head;
                if (buf) {
                    ctx->send_head = buf->next;
                    if (!ctx->send_head) ctx->send_tail = NULL;
                }
                pthread_mutex_unlock(&ctx->send_mutex);

                if (buf) {
                    lws_write(wsi, buf->data + LWS_PRE, buf->len, LWS_WRITE_TEXT);
                    free(buf->data);
                    free(buf);

                    pthread_mutex_lock(&ctx->send_mutex);
                    bool more = (ctx->send_head != NULL);
                    pthread_mutex_unlock(&ctx->send_mutex);
                    if (more) {
                        lws_callback_on_writable(wsi);
                    }
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            if (ctx) {
                atomic_store(&ctx->state, STATE_DISCONNECTED);
                const char *err = in ? (const char *)in : "Connection error";
                fprintf(stderr, "Connection error: %s\n", err);
            }
            break;

        case LWS_CALLBACK_CLIENT_CLOSED:
            if (ctx) {
                atomic_store(&ctx->state, STATE_DISCONNECTED);
                ctx->wsi = NULL;
                if (ctx->interactive) {
                    printf("Disconnected\n");
                }
            }
            break;

        default:
            break;
    }

    return 0;
}

/* Protocol definition */
static struct lws_protocols protocols[] = {
    {
        .name = "lx-cli",
        .callback = cli_lws_callback,
        .per_session_data_size = sizeof(void *),
        .rx_buffer_size = 65536,
        .id = 0,
        .user = NULL,
        .tx_packet_size = 0,
    },
    { .name = NULL }  /* sentinel */
};

/*
 * Connect to server
 */
static cli_error_t cli_connect(cli_ctx_t *ctx) {
    if (!ctx) return CLI_ERR_ARGS;

    atomic_store(&ctx->state, STATE_CONNECTING);

    /* Parse URL */
    const char *address = "localhost";
    int port = 8081;
    const char *path = "/";
    bool use_ssl = false;

    if (strncmp(ctx->ws_url, "wss://", 6) == 0) {
        use_ssl = true;
        address = ctx->ws_url + 6;
        port = 443;
    } else if (strncmp(ctx->ws_url, "ws://", 5) == 0) {
        address = ctx->ws_url + 5;
    }

    char host[128] = {0};
    const char *p = address;
    int i = 0;
    while (*p && *p != ':' && *p != '/' && i < 127) {
        host[i++] = *p++;
    }
    host[i] = '\0';

    if (*p == ':') {
        p++;
        port = atoi(p);
        while (*p && *p != '/') p++;
    }

    if (*p == '/') {
        path = p;
    }

    /* Suppress noisy libwebsockets warnings */
    if (!ctx->verbose) {
        lws_set_log_level(LLL_ERR, NULL);
    } else {
        lws_set_log_level(LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_INFO, NULL);
    }

    /* Create context */
    struct lws_context_creation_info ctx_info;
    memset(&ctx_info, 0, sizeof(ctx_info));
    ctx_info.port = CONTEXT_PORT_NO_LISTEN;
    ctx_info.protocols = protocols;
    ctx_info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    ctx_info.user = ctx;

    ctx->lws_ctx = lws_create_context(&ctx_info);
    if (!ctx->lws_ctx) {
        fprintf(stderr, "Failed to create WebSocket context\n");
        return CLI_ERR_CONN;
    }

    /* Connect */
    struct lws_client_connect_info conn_info;
    memset(&conn_info, 0, sizeof(conn_info));
    conn_info.context = ctx->lws_ctx;
    conn_info.address = host;
    conn_info.port = port;
    conn_info.path = path;
    conn_info.host = host;
    conn_info.origin = host;
    conn_info.protocol = protocols[0].name;
    conn_info.ssl_connection = use_ssl ? LCCSCF_USE_SSL : 0;

    ctx->wsi = lws_client_connect_via_info(&conn_info);
    if (!ctx->wsi) {
        lws_context_destroy(ctx->lws_ctx);
        ctx->lws_ctx = NULL;
        fprintf(stderr, "Failed to initiate connection\n");
        return CLI_ERR_CONN;
    }

    /* Wait for connection */
    int wait_ms = 0;
    while (atomic_load(&ctx->state) == STATE_CONNECTING && wait_ms < ctx->timeout_ms) {
        lws_service(ctx->lws_ctx, 50);
        wait_ms += 50;
    }

    if (atomic_load(&ctx->state) != STATE_CONNECTED) {
        fprintf(stderr, "Connection timeout\n");
        return CLI_ERR_TIMEOUT;
    }

    return CLI_OK;
}

/*
 * Authenticate
 */
static cli_error_t cli_auth(cli_ctx_t *ctx) {
    if (!ctx->api_key[0] || !ctx->api_secret[0]) {
        return CLI_OK; /* No credentials, skip */
    }

    char *req_id = next_req_id(ctx);
    char *msg = build_auth_msg(ctx->api_key, ctx->api_secret, req_id);
    char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);

    if (!resp) {
        fprintf(stderr, "Auth timeout\n");
        return CLI_ERR_TIMEOUT;
    }

    char type[32] = {0};
    json_get_string(resp, "type", type, sizeof(type));

    if (strcmp(type, "auth_success") == 0) {
        atomic_store(&ctx->state, STATE_AUTHENTICATED);
        free(resp);
        return CLI_OK;
    }

    char err[256] = {0};
    json_get_string(resp, "error", err, sizeof(err));
    fprintf(stderr, "Auth failed: %s\n", err[0] ? err : "Unknown error");
    free(resp);
    return CLI_ERR_AUTH;
}

/*
 * Disconnect
 */
static void cli_disconnect(cli_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->lws_ctx) {
        lws_context_destroy(ctx->lws_ctx);
        ctx->lws_ctx = NULL;
    }
    ctx->wsi = NULL;
    atomic_store(&ctx->state, STATE_DISCONNECTED);
}

/*
 * Signal handler
 */
static void signal_handler(int sig) {
    (void)sig;
    if (g_ctx) {
        g_ctx->running = 0;
    }
}

/*
 * Service thread for async processing
 */
static void *service_thread(void *arg) {
    cli_ctx_t *ctx = (cli_ctx_t *)arg;
    while (ctx->running && ctx->lws_ctx) {
        lws_service(ctx->lws_ctx, 50);
    }
    return NULL;
}

/*
 * Print formatted response
 */
static void print_response(const char *json, const char *type) {
    if (!json) {
        printf("No response (timeout)\n");
        return;
    }

    if (json_has_key(json, "error")) {
        char err[256] = {0};
        json_get_string(json, "error", err, sizeof(err));
        printf("Error: %s\n", err);
        return;
    }

    if (strcmp(type, "place_order") == 0) {
        char symbol[32] = {0}, status[32] = {0};
        json_get_string(json, "symbol", symbol, sizeof(symbol));
        json_get_string(json, "status", status, sizeof(status));
        double order_id = json_get_number(json, "orderId");
        double price = json_get_number(json, "price");
        double size = json_get_number(json, "size");
        printf("Order placed: ID=%llu %s %.4f @ %.2f [%s]\n",
            (unsigned long long)order_id, symbol, size, price, status);
    }
    else if (strcmp(type, "cancel_order") == 0) {
        printf("Order cancelled\n");
    }
    else if (strcmp(type, "get_orders") == 0 || strcmp(type, "get_positions") == 0 ||
             strcmp(type, "get_balances") == 0) {
        /* Pretty print the JSON */
        printf("%s\n", json);
    }
    else {
        printf("%s\n", json);
    }
}

/*
 * Execute a single command
 */
static int exec_command(cli_ctx_t *ctx, int argc, char **argv) {
    if (argc < 1) return 0;

    const char *cmd = argv[0];

    /* Help */
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "?") == 0) {
        print_help();
        return 0;
    }

    /* Quit */
    if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0) {
        ctx->running = 0;
        return 0;
    }

    /* Auth */
    if (strcmp(cmd, "auth") == 0) {
        if (argc < 3) {
            printf("Usage: auth <api_key> <api_secret>\n");
            return 1;
        }
        strncpy(ctx->api_key, argv[1], sizeof(ctx->api_key) - 1);
        strncpy(ctx->api_secret, argv[2], sizeof(ctx->api_secret) - 1);
        cli_error_t err = cli_auth(ctx);
        if (err == CLI_OK) {
            printf("Authenticated successfully\n");
        }
        return err == CLI_OK ? 0 : 1;
    }

    /* Place order */
    if (strcmp(cmd, "place_order") == 0) {
        if (argc < 6) {
            printf("Usage: place_order <symbol> <side> <type> <price> <size>\n");
            printf("  Example: place_order BTC-USD buy limit 50000 0.1\n");
            return 1;
        }

        const char *symbol = argv[1];
        const char *side = argv[2];
        const char *type = argv[3];
        double price = atof(argv[4]);
        double size = atof(argv[5]);

        /* Validate side */
        if (strcmp(side, "buy") != 0 && strcmp(side, "sell") != 0) {
            printf("Invalid side: %s (use 'buy' or 'sell')\n", side);
            return 1;
        }

        /* Validate type */
        if (strcmp(type, "limit") != 0 && strcmp(type, "market") != 0 && strcmp(type, "stop") != 0) {
            printf("Invalid type: %s (use 'limit', 'market', or 'stop')\n", type);
            return 1;
        }

        char *req_id = next_req_id(ctx);
        char *msg = build_place_order_msg(symbol, side, type, price, size, req_id);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        print_response(resp, "place_order");
        free(resp);
        return 0;
    }

    /* Cancel order */
    if (strcmp(cmd, "cancel_order") == 0) {
        if (argc < 2) {
            printf("Usage: cancel_order <order_id>\n");
            return 1;
        }

        uint64_t order_id = strtoull(argv[1], NULL, 10);
        char *req_id = next_req_id(ctx);
        char *msg = build_cancel_order_msg(order_id, req_id);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        print_response(resp, "cancel_order");
        free(resp);
        return 0;
    }

    /* Get orders */
    if (strcmp(cmd, "get_orders") == 0) {
        char *req_id = next_req_id(ctx);
        char *msg = build_simple_msg("get_orders", req_id);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        print_response(resp, "get_orders");
        free(resp);
        return 0;
    }

    /* Get positions */
    if (strcmp(cmd, "get_positions") == 0) {
        char *req_id = next_req_id(ctx);
        char *msg = build_simple_msg("get_positions", req_id);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        print_response(resp, "get_positions");
        free(resp);
        return 0;
    }

    /* Get balances */
    if (strcmp(cmd, "get_balances") == 0) {
        char *req_id = next_req_id(ctx);
        char *msg = build_simple_msg("get_balances", req_id);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        print_response(resp, "get_balances");
        free(resp);
        return 0;
    }

    /* Subscribe to orderbook */
    if (strcmp(cmd, "subscribe") == 0 || strcmp(cmd, "get_orderbook") == 0) {
        if (argc < 2) {
            printf("Usage: subscribe <symbol>\n");
            return 1;
        }

        char *req_id = next_req_id(ctx);
        char *msg = build_subscribe_msg(argv[1], req_id);
        queue_send(ctx, msg);
        printf("Subscribed to %s\n", argv[1]);
        return 0;
    }

    /* Ping */
    if (strcmp(cmd, "ping") == 0) {
        char *req_id = next_req_id(ctx);
        char *msg = build_simple_msg("ping", req_id);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        char *resp = send_and_wait(ctx, msg, req_id, ctx->timeout_ms);
        clock_gettime(CLOCK_MONOTONIC, &end);

        if (resp) {
            double latency_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                               (end.tv_nsec - start.tv_nsec) / 1000000.0;
            printf("pong (%.2f ms)\n", latency_ms);
            free(resp);
        } else {
            printf("timeout\n");
        }
        return 0;
    }

    /* Status */
    if (strcmp(cmd, "status") == 0) {
        int state = atomic_load(&ctx->state);
        const char *states[] = {"disconnected", "connecting", "connected", "authenticated"};
        printf("Connection: %s\n", states[state]);
        printf("URL: %s\n", ctx->ws_url);
        if (ctx->api_key[0]) {
            printf("API Key: %s...\n", ctx->api_key);
        }
        return 0;
    }

    printf("Unknown command: %s (type 'help' for commands)\n", cmd);
    return 1;
}

/*
 * Parse command line into tokens
 */
static int tokenize(char *line, char **argv, int max_args) {
    int argc = 0;
    char *p = line;

    while (*p && argc < max_args) {
        /* Skip whitespace */
        while (*p && isspace(*p)) p++;
        if (!*p) break;

        /* Handle quoted strings */
        if (*p == '"' || *p == '\'') {
            char quote = *p++;
            argv[argc++] = p;
            while (*p && *p != quote) p++;
            if (*p) *p++ = '\0';
        } else {
            argv[argc++] = p;
            while (*p && !isspace(*p)) p++;
            if (*p) *p++ = '\0';
        }
    }

    return argc;
}

/*
 * Interactive REPL
 */
static int run_interactive(cli_ctx_t *ctx) {
    char line[MAX_CMD_LEN];
    char *argv[MAX_TOKEN_LEN];

    /* Start service thread */
    pthread_t svc_thread;
    pthread_create(&svc_thread, NULL, service_thread, ctx);

    printf("LX CLI v%s - Type 'help' for commands\n", CLI_VERSION);
    printf("> ");
    fflush(stdout);

    while (ctx->running && fgets(line, sizeof(line), stdin)) {
        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        /* Skip empty lines */
        if (line[0] == '\0') {
            printf("> ");
            fflush(stdout);
            continue;
        }

        /* Tokenize and execute */
        int argc = tokenize(line, argv, MAX_TOKEN_LEN);
        if (argc > 0) {
            exec_command(ctx, argc, argv);
        }

        if (ctx->running) {
            printf("> ");
            fflush(stdout);
        }
    }

    ctx->running = 0;
    pthread_join(svc_thread, NULL);

    printf("Goodbye\n");
    return 0;
}

/*
 * Run single command
 */
static int run_command(cli_ctx_t *ctx, int argc, char **argv) {
    /* Start service thread */
    pthread_t svc_thread;
    pthread_create(&svc_thread, NULL, service_thread, ctx);

    int result = exec_command(ctx, argc, argv);

    ctx->running = 0;
    pthread_join(svc_thread, NULL);

    return result;
}

/*
 * Print help
 */
static void print_help(void) {
    printf("\n");
    printf("LX CLI v%s - Trading Commands\n", CLI_VERSION);
    printf("=====================================\n\n");
    printf("Trading:\n");
    printf("  place_order <symbol> <side> <type> <price> <size>\n");
    printf("      Place a new order\n");
    printf("      Example: place_order BTC-USD buy limit 50000 0.1\n\n");
    printf("  cancel_order <order_id>\n");
    printf("      Cancel an existing order\n\n");
    printf("  get_orders\n");
    printf("      List all open orders\n\n");
    printf("Portfolio:\n");
    printf("  get_positions\n");
    printf("      Show all positions\n\n");
    printf("  get_balances\n");
    printf("      Show account balances\n\n");
    printf("Market Data:\n");
    printf("  subscribe <symbol>\n");
    printf("      Subscribe to orderbook updates\n\n");
    printf("Connection:\n");
    printf("  auth <api_key> <api_secret>\n");
    printf("      Authenticate with credentials\n\n");
    printf("  ping\n");
    printf("      Test connection latency\n\n");
    printf("  status\n");
    printf("      Show connection status\n\n");
    printf("General:\n");
    printf("  help\n");
    printf("      Show this help message\n\n");
    printf("  quit / exit\n");
    printf("      Exit the CLI\n\n");
}

/*
 * Print usage
 */
static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS] [COMMAND [ARGS...]]\n\n", prog);
    printf("Options:\n");
    printf("  -u, --url URL       WebSocket URL (default: ws://localhost:8081)\n");
    printf("  -k, --key KEY       API key for authentication\n");
    printf("  -s, --secret SECRET API secret for authentication\n");
    printf("  -i, --interactive   Run in interactive REPL mode\n");
    printf("  -v, --verbose       Enable verbose output\n");
    printf("  -t, --timeout MS    Request timeout in milliseconds (default: 5000)\n");
    printf("  -h, --help          Show this help message\n");
    printf("  -V, --version       Show version\n\n");
    printf("Commands:\n");
    printf("  place_order <symbol> <side> <type> <price> <size>\n");
    printf("  cancel_order <order_id>\n");
    printf("  get_orders\n");
    printf("  get_positions\n");
    printf("  get_balances\n");
    printf("  subscribe <symbol>\n\n");
    printf("Examples:\n");
    printf("  %s -i                                    # Interactive mode\n", prog);
    printf("  %s place_order BTC-USD buy limit 50000 0.1\n", prog);
    printf("  %s -k KEY -s SECRET get_orders\n", prog);
}

/*
 * Main entry point
 */
int main(int argc, char **argv) {
    cli_ctx_t ctx = {0};

    /* Defaults */
    strncpy(ctx.ws_url, "ws://localhost:8081", sizeof(ctx.ws_url) - 1);
    ctx.timeout_ms = 5000;
    ctx.running = 1;
    atomic_init(&ctx.state, STATE_DISCONNECTED);
    atomic_init(&ctx.req_id, 1);
    pthread_mutex_init(&ctx.send_mutex, NULL);
    pthread_mutex_init(&ctx.resp_mutex, NULL);
    pthread_cond_init(&ctx.resp_cond, NULL);

    /* Allocate receive buffer */
    ctx.recv_cap = RECV_BUF_INIT;
    ctx.recv_buf = malloc(ctx.recv_cap);
    if (!ctx.recv_buf) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    /* Parse arguments */
    int cmd_start = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--url") == 0) {
            if (++i < argc) {
                strncpy(ctx.ws_url, argv[i], sizeof(ctx.ws_url) - 1);
            }
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--key") == 0) {
            if (++i < argc) {
                strncpy(ctx.api_key, argv[i], sizeof(ctx.api_key) - 1);
            }
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--secret") == 0) {
            if (++i < argc) {
                strncpy(ctx.api_secret, argv[i], sizeof(ctx.api_secret) - 1);
            }
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            ctx.interactive = true;
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            ctx.verbose = true;
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) {
            if (++i < argc) {
                ctx.timeout_ms = atoi(argv[i]);
            }
            cmd_start = i + 1;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-V") == 0 || strcmp(argv[i], "--version") == 0) {
            printf("lx-cli v%s\n", CLI_VERSION);
            return 0;
        }
        else if (argv[i][0] != '-') {
            cmd_start = i;
            break;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Set up signal handlers */
    g_ctx = &ctx;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* If no command and not interactive, default to interactive */
    if (cmd_start >= argc && !ctx.interactive) {
        ctx.interactive = true;
    }

    /* Connect */
    cli_error_t err = cli_connect(&ctx);
    if (err != CLI_OK) {
        free(ctx.recv_buf);
        return err;
    }

    /* Authenticate if credentials provided */
    if (ctx.api_key[0] && ctx.api_secret[0]) {
        err = cli_auth(&ctx);
        if (err != CLI_OK && !ctx.interactive) {
            cli_disconnect(&ctx);
            free(ctx.recv_buf);
            return err;
        }
    }

    /* Run */
    int result = 0;
    if (ctx.interactive) {
        result = run_interactive(&ctx);
    } else {
        result = run_command(&ctx, argc - cmd_start, &argv[cmd_start]);
    }

    /* Cleanup */
    cli_disconnect(&ctx);

    /* Free send queue */
    pthread_mutex_lock(&ctx.send_mutex);
    send_buf_t *buf = ctx.send_head;
    while (buf) {
        send_buf_t *next = buf->next;
        free(buf->data);
        free(buf);
        buf = next;
    }
    pthread_mutex_unlock(&ctx.send_mutex);

    pthread_mutex_destroy(&ctx.send_mutex);
    pthread_mutex_destroy(&ctx.resp_mutex);
    pthread_cond_destroy(&ctx.resp_cond);

    free(ctx.recv_buf);
    free(ctx.response);

    return result;
}
