/*
 * LX C SDK - WebSocket Client
 * High-performance WebSocket client using libwebsockets.
 */

#include "lxdex.h"
#include <libwebsockets.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

/* Thread-local error message */
static _Thread_local char tls_error[LXDEX_MSG_LEN];

/* External JSON functions (from json.c) */
extern char *lxdex_json_auth(const char *api_key, const char *api_secret, const char *request_id);
extern char *lxdex_json_place_order(const lxdex_order_t *order, const char *request_id);
extern char *lxdex_json_cancel_order(uint64_t order_id, const char *request_id);
extern char *lxdex_json_subscribe(const char *channel, const char *request_id);
extern char *lxdex_json_unsubscribe(const char *channel, const char *request_id);
extern char *lxdex_json_ping(const char *request_id);
extern char *lxdex_json_get_balances(const char *request_id);
extern char *lxdex_json_get_positions(const char *request_id);
extern char *lxdex_json_get_orders(const char *request_id);
extern const char *lxdex_json_parse_type(const char *json);
extern lxdex_error_t lxdex_json_parse_order(const char *json, lxdex_order_t *order);
extern lxdex_error_t lxdex_json_parse_trade(const char *json, lxdex_trade_t *trade);
extern lxdex_error_t lxdex_json_parse_orderbook(const char *json, lxdex_orderbook_t *book);
extern lxdex_error_t lxdex_json_parse_error(const char *json, char *msg_out, size_t msg_len);

/* Send buffer entry */
typedef struct send_buf {
    unsigned char *data;
    size_t len;
    struct send_buf *next;
} send_buf_t;

/* Client structure */
struct lxdex_client {
    /* Configuration */
    char *ws_url;
    char *api_key;
    char *api_secret;
    int connect_timeout_ms;
    int recv_timeout_ms;
    int reconnect_interval_ms;
    bool auto_reconnect;

    /* libwebsockets */
    struct lws_context *lws_ctx;
    struct lws *wsi;

    /* State */
    atomic_int state;
    bool should_close;
    bool auth_pending;

    /* Send queue */
    send_buf_t *send_head;
    send_buf_t *send_tail;
    pthread_mutex_t send_mutex;

    /* Receive buffer */
    char *recv_buf;
    size_t recv_len;
    size_t recv_cap;

    /* Callbacks */
    lxdex_callbacks_t callbacks;

    /* Request ID counter */
    atomic_uint_fast64_t request_id;
};

/* Global initialization flag */
static atomic_bool g_initialized = false;

/* Protocol callback */
static int lxdex_lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                              void *user, void *in, size_t len);

/* Protocol definition */
static struct lws_protocols protocols[] = {
    {
        .name = "lxdex",
        .callback = lxdex_lws_callback,
        .per_session_data_size = sizeof(void *),
        .rx_buffer_size = 65536,
    },
    { NULL, NULL, 0, 0 }
};

/*
 * Utility functions
 */

static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tls_error, sizeof(tls_error), fmt, ap);
    va_end(ap);
}

static char *generate_request_id(lxdex_client_t *client) {
    static char buf[32];
    uint64_t id = atomic_fetch_add(&client->request_id, 1);
    snprintf(buf, sizeof(buf), "req_%llu", (unsigned long long)id);
    return buf;
}

static lxdex_error_t queue_send(lxdex_client_t *client, const char *msg) {
    if (!client || !msg) return LXDEX_ERR_INVALID_ARG;

    size_t len = strlen(msg);
    send_buf_t *buf = malloc(sizeof(send_buf_t));
    if (!buf) return LXDEX_ERR_NO_MEMORY;

    /* LWS requires LWS_PRE bytes before data */
    buf->data = malloc(LWS_PRE + len);
    if (!buf->data) {
        free(buf);
        return LXDEX_ERR_NO_MEMORY;
    }

    memcpy(buf->data + LWS_PRE, msg, len);
    buf->len = len;
    buf->next = NULL;

    pthread_mutex_lock(&client->send_mutex);
    if (client->send_tail) {
        client->send_tail->next = buf;
        client->send_tail = buf;
    } else {
        client->send_head = client->send_tail = buf;
    }
    pthread_mutex_unlock(&client->send_mutex);

    /* Request writable callback */
    if (client->wsi) {
        lws_callback_on_writable(client->wsi);
    }

    return LXDEX_OK;
}

static void process_message(lxdex_client_t *client, const char *msg, size_t len) {
    if (!client || !msg || len == 0) return;

    /* Null-terminate for parsing */
    char *json = malloc(len + 1);
    if (!json) return;
    memcpy(json, msg, len);
    json[len] = '\0';

    const char *type = lxdex_json_parse_type(json);
    if (!type) {
        free(json);
        return;
    }

    if (strcmp(type, "connected") == 0) {
        atomic_store(&client->state, LXDEX_STATE_CONNECTED);
        if (client->callbacks.on_connect) {
            client->callbacks.on_connect(client, client->callbacks.user_data);
        }
    }
    else if (strcmp(type, "auth_success") == 0) {
        atomic_store(&client->state, LXDEX_STATE_AUTHENTICATED);
        client->auth_pending = false;
    }
    else if (strcmp(type, "error") == 0) {
        char err_msg[LXDEX_MSG_LEN] = {0};
        lxdex_json_parse_error(json, err_msg, sizeof(err_msg));
        if (client->auth_pending) {
            client->auth_pending = false;
            atomic_store(&client->state, LXDEX_STATE_CONNECTED);
        }
        if (client->callbacks.on_error) {
            client->callbacks.on_error(client, LXDEX_ERR_PROTOCOL, err_msg,
                client->callbacks.user_data);
        }
    }
    else if (strcmp(type, "order_update") == 0) {
        if (client->callbacks.on_order_update) {
            lxdex_order_t order;
            if (lxdex_json_parse_order(json, &order) == LXDEX_OK) {
                client->callbacks.on_order_update(client, &order,
                    client->callbacks.user_data);
            }
        }
    }
    else if (strcmp(type, "trade") == 0) {
        if (client->callbacks.on_trade) {
            lxdex_trade_t trade;
            if (lxdex_json_parse_trade(json, &trade) == LXDEX_OK) {
                client->callbacks.on_trade(client, &trade,
                    client->callbacks.user_data);
            }
        }
    }
    else if (strcmp(type, "orderbook") == 0 || strcmp(type, "orderbook_update") == 0) {
        if (client->callbacks.on_orderbook) {
            lxdex_orderbook_t book;
            memset(&book, 0, sizeof(book));
            if (lxdex_json_parse_orderbook(json, &book) == LXDEX_OK) {
                client->callbacks.on_orderbook(client, &book,
                    client->callbacks.user_data);
                lxdex_orderbook_free(&book);
            }
        }
    }
    else if (strcmp(type, "pong") == 0) {
        /* Heartbeat response - no action needed */
    }

    free(json);
}

/*
 * libwebsockets callback
 */
static int lxdex_lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                              void *user, void *in, size_t len) {
    lxdex_client_t *client = NULL;

    /* Get client pointer from protocol user data */
    struct lws_context *ctx = lws_get_context(wsi);
    if (ctx) {
        client = lws_context_user(ctx);
    }

    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            if (client) {
                atomic_store(&client->state, LXDEX_STATE_CONNECTED);
            }
            break;

        case LWS_CALLBACK_CLIENT_RECEIVE:
            if (client && in && len > 0) {
                /* Accumulate message fragments */
                size_t needed = client->recv_len + len;
                if (needed > client->recv_cap) {
                    size_t new_cap = client->recv_cap * 2;
                    if (new_cap < needed) new_cap = needed + 1024;
                    char *new_buf = realloc(client->recv_buf, new_cap);
                    if (!new_buf) break;
                    client->recv_buf = new_buf;
                    client->recv_cap = new_cap;
                }

                memcpy(client->recv_buf + client->recv_len, in, len);
                client->recv_len += len;

                /* Check if this is the final fragment */
                if (lws_is_final_fragment(wsi)) {
                    process_message(client, client->recv_buf, client->recv_len);
                    client->recv_len = 0;
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_WRITEABLE:
            if (client) {
                pthread_mutex_lock(&client->send_mutex);
                send_buf_t *buf = client->send_head;
                if (buf) {
                    client->send_head = buf->next;
                    if (!client->send_head) {
                        client->send_tail = NULL;
                    }
                }
                pthread_mutex_unlock(&client->send_mutex);

                if (buf) {
                    lws_write(wsi, buf->data + LWS_PRE, buf->len, LWS_WRITE_TEXT);
                    free(buf->data);
                    free(buf);

                    /* More to send? */
                    pthread_mutex_lock(&client->send_mutex);
                    bool more = (client->send_head != NULL);
                    pthread_mutex_unlock(&client->send_mutex);
                    if (more) {
                        lws_callback_on_writable(wsi);
                    }
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            if (client) {
                atomic_store(&client->state, LXDEX_STATE_ERROR);
                const char *err = in ? (const char *)in : "Connection error";
                set_error("%s", err);
                if (client->callbacks.on_error) {
                    client->callbacks.on_error(client, LXDEX_ERR_CONNECTION, err,
                        client->callbacks.user_data);
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_CLOSED:
            if (client) {
                lxdex_conn_state_t prev = atomic_exchange(&client->state, LXDEX_STATE_DISCONNECTED);
                client->wsi = NULL;
                if (client->callbacks.on_disconnect && prev != LXDEX_STATE_DISCONNECTED) {
                    client->callbacks.on_disconnect(client, 0, "Connection closed",
                        client->callbacks.user_data);
                }
            }
            break;

        default:
            break;
    }

    return 0;
}

/*
 * Public API
 */

const char *lxdex_version(void) {
    static char ver[32];
    snprintf(ver, sizeof(ver), "%d.%d.%d",
        LXDEX_VERSION_MAJOR, LXDEX_VERSION_MINOR, LXDEX_VERSION_PATCH);
    return ver;
}

lxdex_error_t lxdex_init(void) {
    bool expected = false;
    if (atomic_compare_exchange_strong(&g_initialized, &expected, true)) {
        /* Initialize libwebsockets logging */
        lws_set_log_level(LLL_ERR | LLL_WARN, NULL);
    }
    return LXDEX_OK;
}

void lxdex_cleanup(void) {
    atomic_store(&g_initialized, false);
}

const char *lxdex_strerror(lxdex_error_t error) {
    switch (error) {
        case LXDEX_OK: return "Success";
        case LXDEX_ERR_INVALID_ARG: return "Invalid argument";
        case LXDEX_ERR_NO_MEMORY: return "Out of memory";
        case LXDEX_ERR_CONNECTION: return "Connection error";
        case LXDEX_ERR_TIMEOUT: return "Operation timed out";
        case LXDEX_ERR_AUTH: return "Authentication failed";
        case LXDEX_ERR_PARSE: return "Parse error";
        case LXDEX_ERR_PROTOCOL: return "Protocol error";
        case LXDEX_ERR_RATE_LIMIT: return "Rate limit exceeded";
        case LXDEX_ERR_ORDER_REJECTED: return "Order rejected";
        case LXDEX_ERR_NOT_CONNECTED: return "Not connected";
        case LXDEX_ERR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

const char *lxdex_last_error(void) {
    return tls_error[0] ? tls_error : NULL;
}

lxdex_client_t *lxdex_client_new(const lxdex_config_t *config) {
    lxdex_client_t *client = calloc(1, sizeof(lxdex_client_t));
    if (!client) return NULL;

    /* Set defaults */
    const char *ws_url = config && config->ws_url ? config->ws_url : "ws://localhost:8081";
    client->ws_url = strdup(ws_url);
    if (!client->ws_url) {
        free(client);
        return NULL;
    }

    if (config && config->api_key) {
        client->api_key = strdup(config->api_key);
    }
    if (config && config->api_secret) {
        client->api_secret = strdup(config->api_secret);
    }

    client->connect_timeout_ms = (config && config->connect_timeout_ms > 0)
        ? config->connect_timeout_ms : 10000;
    client->recv_timeout_ms = (config && config->recv_timeout_ms > 0)
        ? config->recv_timeout_ms : 30000;
    client->reconnect_interval_ms = (config && config->reconnect_interval_ms > 0)
        ? config->reconnect_interval_ms : 5000;
    client->auto_reconnect = config ? config->auto_reconnect : false;

    atomic_init(&client->state, LXDEX_STATE_DISCONNECTED);
    atomic_init(&client->request_id, 1);

    pthread_mutex_init(&client->send_mutex, NULL);

    /* Initial receive buffer */
    client->recv_cap = 4096;
    client->recv_buf = malloc(client->recv_cap);
    if (!client->recv_buf) {
        free(client->ws_url);
        free(client->api_key);
        free(client->api_secret);
        free(client);
        return NULL;
    }

    return client;
}

void lxdex_client_set_callbacks(lxdex_client_t *client, const lxdex_callbacks_t *callbacks) {
    if (client && callbacks) {
        client->callbacks = *callbacks;
    }
}

lxdex_conn_state_t lxdex_client_state(const lxdex_client_t *client) {
    return client ? atomic_load(&client->state) : LXDEX_STATE_DISCONNECTED;
}

lxdex_error_t lxdex_client_connect(lxdex_client_t *client) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    if (atomic_load(&client->state) != LXDEX_STATE_DISCONNECTED) {
        return LXDEX_OK; /* Already connected or connecting */
    }

    atomic_store(&client->state, LXDEX_STATE_CONNECTING);

    /* Parse URL */
    const char *protocol = "ws";
    const char *address = "localhost";
    int port = 8081;
    const char *path = "/";
    bool use_ssl = false;

    if (strncmp(client->ws_url, "wss://", 6) == 0) {
        use_ssl = true;
        protocol = "wss";
        address = client->ws_url + 6;
        port = 443;
    } else if (strncmp(client->ws_url, "ws://", 5) == 0) {
        address = client->ws_url + 5;
    }

    /* Parse host:port/path */
    char host[256] = {0};
    const char *p = address;
    int i = 0;
    while (*p && *p != ':' && *p != '/' && i < 255) {
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

    /* Create LWS context */
    struct lws_context_creation_info ctx_info;
    memset(&ctx_info, 0, sizeof(ctx_info));
    ctx_info.port = CONTEXT_PORT_NO_LISTEN;
    ctx_info.protocols = protocols;
    ctx_info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    ctx_info.user = client;

    client->lws_ctx = lws_create_context(&ctx_info);
    if (!client->lws_ctx) {
        atomic_store(&client->state, LXDEX_STATE_ERROR);
        set_error("Failed to create WebSocket context");
        return LXDEX_ERR_CONNECTION;
    }

    /* Connect */
    struct lws_client_connect_info conn_info;
    memset(&conn_info, 0, sizeof(conn_info));
    conn_info.context = client->lws_ctx;
    conn_info.address = host;
    conn_info.port = port;
    conn_info.path = path;
    conn_info.host = host;
    conn_info.origin = host;
    conn_info.protocol = protocols[0].name;
    conn_info.ssl_connection = use_ssl ? LCCSCF_USE_SSL : 0;

    client->wsi = lws_client_connect_via_info(&conn_info);
    if (!client->wsi) {
        lws_context_destroy(client->lws_ctx);
        client->lws_ctx = NULL;
        atomic_store(&client->state, LXDEX_STATE_ERROR);
        set_error("Failed to initiate connection");
        return LXDEX_ERR_CONNECTION;
    }

    return LXDEX_OK;
}

lxdex_error_t lxdex_client_auth(lxdex_client_t *client) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_CONNECTED) {
        set_error("Not connected");
        return LXDEX_ERR_NOT_CONNECTED;
    }

    if (!client->api_key || !client->api_secret) {
        set_error("Missing API credentials");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_auth(client->api_key, client->api_secret,
        generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    client->auth_pending = true;
    lxdex_error_t err = queue_send(client, msg);
    free(msg);

    return err;
}

void lxdex_client_disconnect(lxdex_client_t *client) {
    if (!client) return;

    client->should_close = true;

    if (client->wsi) {
        lws_set_timeout(client->wsi, PENDING_TIMEOUT_USER_OK, LWS_TO_KILL_SYNC);
    }

    if (client->lws_ctx) {
        lws_context_destroy(client->lws_ctx);
        client->lws_ctx = NULL;
    }

    client->wsi = NULL;
    atomic_store(&client->state, LXDEX_STATE_DISCONNECTED);
}

void lxdex_client_free(lxdex_client_t *client) {
    if (!client) return;

    lxdex_client_disconnect(client);

    /* Free send queue */
    pthread_mutex_lock(&client->send_mutex);
    send_buf_t *buf = client->send_head;
    while (buf) {
        send_buf_t *next = buf->next;
        free(buf->data);
        free(buf);
        buf = next;
    }
    pthread_mutex_unlock(&client->send_mutex);
    pthread_mutex_destroy(&client->send_mutex);

    free(client->recv_buf);
    free(client->ws_url);
    free(client->api_key);
    free(client->api_secret);
    free(client);
}

int lxdex_client_service(lxdex_client_t *client, int timeout_ms) {
    if (!client || !client->lws_ctx) return -1;
    return lws_service(client->lws_ctx, timeout_ms);
}

/*
 * Order operations
 */

lxdex_error_t lxdex_place_order(lxdex_client_t *client, const lxdex_order_t *order,
                                uint64_t *order_id_out) {
    if (!client || !order) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_place_order(order, generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);

    /* Note: actual order_id will come via callback */
    if (order_id_out) *order_id_out = 0;

    return err;
}

lxdex_error_t lxdex_cancel_order(lxdex_client_t *client, uint64_t order_id) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_cancel_order(order_id, generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);
    return err;
}

lxdex_error_t lxdex_cancel_all_orders(lxdex_client_t *client, const char *symbol) {
    /* Not directly supported - would need to get orders and cancel each */
    (void)client;
    (void)symbol;
    return LXDEX_ERR_PROTOCOL;
}

lxdex_error_t lxdex_modify_order(lxdex_client_t *client, uint64_t order_id,
                                  double new_price, double new_size) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    /* Build modify message manually */
    char msg[512];
    snprintf(msg, sizeof(msg),
        "{\"type\":\"modify_order\",\"orderId\":%llu,\"price\":%.15g,\"size\":%.15g,\"request_id\":\"%s\"}",
        (unsigned long long)order_id, new_price, new_size,
        generate_request_id(client));

    return queue_send(client, msg);
}

/*
 * Subscriptions
 */

lxdex_error_t lxdex_subscribe_orderbook(lxdex_client_t *client, const char *symbol) {
    if (!client || !symbol) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state < LXDEX_STATE_CONNECTED) {
        set_error("Not connected");
        return LXDEX_ERR_NOT_CONNECTED;
    }

    char channel[128];
    snprintf(channel, sizeof(channel), "orderbook:%s", symbol);

    char *msg = lxdex_json_subscribe(channel, generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);
    return err;
}

lxdex_error_t lxdex_subscribe_trades(lxdex_client_t *client, const char *symbol) {
    if (!client || !symbol) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state < LXDEX_STATE_CONNECTED) {
        set_error("Not connected");
        return LXDEX_ERR_NOT_CONNECTED;
    }

    char channel[128];
    snprintf(channel, sizeof(channel), "trades:%s", symbol);

    char *msg = lxdex_json_subscribe(channel, generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);
    return err;
}

lxdex_error_t lxdex_unsubscribe(lxdex_client_t *client, const char *channel) {
    if (!client || !channel) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state < LXDEX_STATE_CONNECTED) {
        return LXDEX_OK; /* Not connected, nothing to unsubscribe */
    }

    char *msg = lxdex_json_unsubscribe(channel, generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);
    return err;
}

/*
 * Data queries (blocking operations - requires async response handling)
 */

lxdex_error_t lxdex_get_orderbook(lxdex_client_t *client, const char *symbol,
                                   int depth, lxdex_orderbook_t *book_out) {
    /* For now, return not implemented - true blocking requires response correlation */
    (void)client;
    (void)symbol;
    (void)depth;
    (void)book_out;
    set_error("Blocking queries not implemented; use subscriptions");
    return LXDEX_ERR_PROTOCOL;
}

lxdex_error_t lxdex_get_trades(lxdex_client_t *client, const char *symbol,
                                int limit, lxdex_trade_t **trades_out,
                                size_t *count_out) {
    (void)client;
    (void)symbol;
    (void)limit;
    (void)trades_out;
    (void)count_out;
    set_error("Blocking queries not implemented; use subscriptions");
    return LXDEX_ERR_PROTOCOL;
}

lxdex_error_t lxdex_get_balances(lxdex_client_t *client,
                                  lxdex_balance_t **balances_out,
                                  size_t *count_out) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_get_balances(generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);

    /* Results come via callback */
    if (balances_out) *balances_out = NULL;
    if (count_out) *count_out = 0;

    return err;
}

lxdex_error_t lxdex_get_positions(lxdex_client_t *client,
                                   lxdex_position_t **positions_out,
                                   size_t *count_out) {
    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_get_positions(generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);

    if (positions_out) *positions_out = NULL;
    if (count_out) *count_out = 0;

    return err;
}

lxdex_error_t lxdex_get_orders(lxdex_client_t *client, const char *symbol,
                                lxdex_order_t **orders_out, size_t *count_out) {
    (void)symbol; /* Symbol filtering would be done client-side */

    if (!client) return LXDEX_ERR_INVALID_ARG;

    lxdex_conn_state_t state = atomic_load(&client->state);
    if (state != LXDEX_STATE_AUTHENTICATED) {
        set_error("Not authenticated");
        return LXDEX_ERR_AUTH;
    }

    char *msg = lxdex_json_get_orders(generate_request_id(client));
    if (!msg) return LXDEX_ERR_NO_MEMORY;

    lxdex_error_t err = queue_send(client, msg);
    free(msg);

    if (orders_out) *orders_out = NULL;
    if (count_out) *count_out = 0;

    return err;
}

/*
 * Memory management
 */

void lxdex_free(void *ptr) {
    free(ptr);
}

void lxdex_trades_free(lxdex_trade_t *trades, size_t count) {
    (void)count;
    free(trades);
}

void lxdex_balances_free(lxdex_balance_t *balances, size_t count) {
    (void)count;
    free(balances);
}

void lxdex_positions_free(lxdex_position_t *positions, size_t count) {
    (void)count;
    free(positions);
}

void lxdex_orders_free(lxdex_order_t *orders, size_t count) {
    (void)count;
    free(orders);
}
