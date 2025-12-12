/*
 * LX C SDK - JSON parsing
 * Minimal JSON parser for protocol messages.
 */

#include "lxdex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

/* Internal JSON types */
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

struct json_value {
    json_type_t type;
    union {
        bool bool_val;
        double num_val;
        char *str_val;
        struct {
            json_value_t **items;
            size_t count;
            size_t capacity;
        } array;
        struct {
            char **keys;
            json_value_t **values;
            size_t count;
            size_t capacity;
        } object;
    } data;
};

/* Forward declarations */
static json_value_t *json_parse_value(const char **p);
static void json_free(json_value_t *val);

/* Skip whitespace */
static void skip_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) {
        (*p)++;
    }
}

/* Parse string (after opening quote) */
static char *json_parse_string_content(const char **p) {
    size_t cap = 64;
    size_t len = 0;
    char *str = malloc(cap);
    if (!str) return NULL;

    while (**p && **p != '"') {
        if (len + 8 >= cap) {
            cap *= 2;
            char *tmp = realloc(str, cap);
            if (!tmp) { free(str); return NULL; }
            str = tmp;
        }

        if (**p == '\\') {
            (*p)++;
            switch (**p) {
                case '"': str[len++] = '"'; break;
                case '\\': str[len++] = '\\'; break;
                case '/': str[len++] = '/'; break;
                case 'b': str[len++] = '\b'; break;
                case 'f': str[len++] = '\f'; break;
                case 'n': str[len++] = '\n'; break;
                case 'r': str[len++] = '\r'; break;
                case 't': str[len++] = '\t'; break;
                case 'u': {
                    /* Unicode escape - simplified, just skip */
                    (*p)++;
                    for (int i = 0; i < 4 && **p; i++) (*p)++;
                    (*p)--;
                    str[len++] = '?';
                    break;
                }
                default:
                    str[len++] = **p;
            }
        } else {
            str[len++] = **p;
        }
        (*p)++;
    }

    if (**p == '"') (*p)++;
    str[len] = '\0';
    return str;
}

/* Parse number */
static double json_parse_number(const char **p) {
    char *end;
    double val = strtod(*p, &end);
    *p = end;
    return val;
}

/* Parse array */
static json_value_t *json_parse_array(const char **p) {
    json_value_t *arr = calloc(1, sizeof(*arr));
    if (!arr) return NULL;

    arr->type = JSON_ARRAY;
    arr->data.array.capacity = 8;
    arr->data.array.items = malloc(sizeof(json_value_t *) * arr->data.array.capacity);
    if (!arr->data.array.items) { free(arr); return NULL; }

    (*p)++; /* skip '[' */
    skip_ws(p);

    while (**p && **p != ']') {
        json_value_t *item = json_parse_value(p);
        if (!item) { json_free(arr); return NULL; }

        if (arr->data.array.count >= arr->data.array.capacity) {
            arr->data.array.capacity *= 2;
            json_value_t **tmp = realloc(arr->data.array.items,
                sizeof(json_value_t *) * arr->data.array.capacity);
            if (!tmp) { json_free(item); json_free(arr); return NULL; }
            arr->data.array.items = tmp;
        }

        arr->data.array.items[arr->data.array.count++] = item;

        skip_ws(p);
        if (**p == ',') {
            (*p)++;
            skip_ws(p);
        }
    }

    if (**p == ']') (*p)++;
    return arr;
}

/* Parse object */
static json_value_t *json_parse_object(const char **p) {
    json_value_t *obj = calloc(1, sizeof(*obj));
    if (!obj) return NULL;

    obj->type = JSON_OBJECT;
    obj->data.object.capacity = 8;
    obj->data.object.keys = malloc(sizeof(char *) * obj->data.object.capacity);
    obj->data.object.values = malloc(sizeof(json_value_t *) * obj->data.object.capacity);
    if (!obj->data.object.keys || !obj->data.object.values) {
        free(obj->data.object.keys);
        free(obj->data.object.values);
        free(obj);
        return NULL;
    }

    (*p)++; /* skip '{' */
    skip_ws(p);

    while (**p && **p != '}') {
        /* Parse key */
        if (**p != '"') { json_free(obj); return NULL; }
        (*p)++;
        char *key = json_parse_string_content(p);
        if (!key) { json_free(obj); return NULL; }

        skip_ws(p);
        if (**p != ':') { free(key); json_free(obj); return NULL; }
        (*p)++;
        skip_ws(p);

        /* Parse value */
        json_value_t *val = json_parse_value(p);
        if (!val) { free(key); json_free(obj); return NULL; }

        /* Expand if needed */
        if (obj->data.object.count >= obj->data.object.capacity) {
            obj->data.object.capacity *= 2;
            char **tmp_k = realloc(obj->data.object.keys,
                sizeof(char *) * obj->data.object.capacity);
            json_value_t **tmp_v = realloc(obj->data.object.values,
                sizeof(json_value_t *) * obj->data.object.capacity);
            if (!tmp_k || !tmp_v) {
                free(key);
                json_free(val);
                json_free(obj);
                return NULL;
            }
            obj->data.object.keys = tmp_k;
            obj->data.object.values = tmp_v;
        }

        obj->data.object.keys[obj->data.object.count] = key;
        obj->data.object.values[obj->data.object.count] = val;
        obj->data.object.count++;

        skip_ws(p);
        if (**p == ',') {
            (*p)++;
            skip_ws(p);
        }
    }

    if (**p == '}') (*p)++;
    return obj;
}

/* Parse any JSON value */
static json_value_t *json_parse_value(const char **p) {
    skip_ws(p);

    if (**p == 'n' && strncmp(*p, "null", 4) == 0) {
        *p += 4;
        json_value_t *v = calloc(1, sizeof(*v));
        if (v) v->type = JSON_NULL;
        return v;
    }

    if (**p == 't' && strncmp(*p, "true", 4) == 0) {
        *p += 4;
        json_value_t *v = calloc(1, sizeof(*v));
        if (v) { v->type = JSON_BOOL; v->data.bool_val = true; }
        return v;
    }

    if (**p == 'f' && strncmp(*p, "false", 5) == 0) {
        *p += 5;
        json_value_t *v = calloc(1, sizeof(*v));
        if (v) { v->type = JSON_BOOL; v->data.bool_val = false; }
        return v;
    }

    if (**p == '"') {
        (*p)++;
        char *str = json_parse_string_content(p);
        if (!str) return NULL;
        json_value_t *v = calloc(1, sizeof(*v));
        if (!v) { free(str); return NULL; }
        v->type = JSON_STRING;
        v->data.str_val = str;
        return v;
    }

    if (**p == '[') {
        return json_parse_array(p);
    }

    if (**p == '{') {
        return json_parse_object(p);
    }

    if (**p == '-' || isdigit((unsigned char)**p)) {
        json_value_t *v = calloc(1, sizeof(*v));
        if (!v) return NULL;
        v->type = JSON_NUMBER;
        v->data.num_val = json_parse_number(p);
        return v;
    }

    return NULL;
}

/* Free JSON value */
static void json_free(json_value_t *val) {
    if (!val) return;

    switch (val->type) {
        case JSON_STRING:
            free(val->data.str_val);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < val->data.array.count; i++) {
                json_free(val->data.array.items[i]);
            }
            free(val->data.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < val->data.object.count; i++) {
                free(val->data.object.keys[i]);
                json_free(val->data.object.values[i]);
            }
            free(val->data.object.keys);
            free(val->data.object.values);
            break;
        default:
            break;
    }
    free(val);
}

/* Get object field */
static json_value_t *json_get(json_value_t *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;

    for (size_t i = 0; i < obj->data.object.count; i++) {
        if (strcmp(obj->data.object.keys[i], key) == 0) {
            return obj->data.object.values[i];
        }
    }
    return NULL;
}

/* Get string value */
static const char *json_get_string(json_value_t *obj, const char *key) {
    json_value_t *v = json_get(obj, key);
    if (v && v->type == JSON_STRING) return v->data.str_val;
    return NULL;
}

/* Get number value */
static double json_get_number(json_value_t *obj, const char *key, double def) {
    json_value_t *v = json_get(obj, key);
    if (v && v->type == JSON_NUMBER) return v->data.num_val;
    return def;
}

/* Get bool value */
static bool json_get_bool(json_value_t *obj, const char *key, bool def) {
    json_value_t *v = json_get(obj, key);
    if (v && v->type == JSON_BOOL) return v->data.bool_val;
    return def;
}

/* Get array */
static json_value_t *json_get_array(json_value_t *obj, const char *key) {
    json_value_t *v = json_get(obj, key);
    if (v && v->type == JSON_ARRAY) return v;
    return NULL;
}

/* Get object */
static json_value_t *json_get_object(json_value_t *obj, const char *key) {
    json_value_t *v = json_get(obj, key);
    if (v && v->type == JSON_OBJECT) return v;
    return NULL;
}

/*
 * JSON builder for outgoing messages
 */

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} json_builder_t;

static void jb_init(json_builder_t *jb) {
    jb->cap = 256;
    jb->len = 0;
    jb->buf = malloc(jb->cap);
    if (jb->buf) jb->buf[0] = '\0';
}

static void jb_free(json_builder_t *jb) {
    free(jb->buf);
    jb->buf = NULL;
    jb->len = 0;
    jb->cap = 0;
}

static void jb_ensure(json_builder_t *jb, size_t need) {
    if (!jb->buf) return;
    if (jb->len + need >= jb->cap) {
        while (jb->len + need >= jb->cap) jb->cap *= 2;
        char *tmp = realloc(jb->buf, jb->cap);
        if (!tmp) { free(jb->buf); jb->buf = NULL; return; }
        jb->buf = tmp;
    }
}

static void jb_append(json_builder_t *jb, const char *s) {
    if (!jb->buf || !s) return;
    size_t n = strlen(s);
    jb_ensure(jb, n + 1);
    if (!jb->buf) return;
    memcpy(jb->buf + jb->len, s, n + 1);
    jb->len += n;
}

static void jb_append_char(json_builder_t *jb, char c) {
    if (!jb->buf) return;
    jb_ensure(jb, 2);
    if (!jb->buf) return;
    jb->buf[jb->len++] = c;
    jb->buf[jb->len] = '\0';
}

static void jb_append_string(json_builder_t *jb, const char *s) {
    if (!s) {
        jb_append(jb, "null");
        return;
    }
    jb_append_char(jb, '"');
    while (*s) {
        switch (*s) {
            case '"': jb_append(jb, "\\\""); break;
            case '\\': jb_append(jb, "\\\\"); break;
            case '\b': jb_append(jb, "\\b"); break;
            case '\f': jb_append(jb, "\\f"); break;
            case '\n': jb_append(jb, "\\n"); break;
            case '\r': jb_append(jb, "\\r"); break;
            case '\t': jb_append(jb, "\\t"); break;
            default:
                if ((unsigned char)*s < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", (unsigned char)*s);
                    jb_append(jb, hex);
                } else {
                    jb_append_char(jb, *s);
                }
        }
        s++;
    }
    jb_append_char(jb, '"');
}

static void jb_append_number(json_builder_t *jb, double n) {
    char buf[64];
    if (floor(n) == n && fabs(n) < 1e15) {
        snprintf(buf, sizeof(buf), "%.0f", n);
    } else {
        snprintf(buf, sizeof(buf), "%.15g", n);
    }
    jb_append(jb, buf);
}

static void jb_append_int(json_builder_t *jb, int64_t n) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)n);
    jb_append(jb, buf);
}

static void jb_append_uint(json_builder_t *jb, uint64_t n) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%llu", (unsigned long long)n);
    jb_append(jb, buf);
}

static void jb_append_bool(json_builder_t *jb, bool b) {
    jb_append(jb, b ? "true" : "false");
}

/*
 * Public JSON building functions for protocol messages
 */

/* Build authentication message */
char *lxdex_json_auth(const char *api_key, const char *api_secret, const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"auth\",\"apiKey\":");
    jb_append_string(&jb, api_key);
    jb_append(&jb, ",\"apiSecret\":");
    jb_append_string(&jb, api_secret);
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build place_order message */
char *lxdex_json_place_order(const lxdex_order_t *order, const char *request_id) {
    if (!order) return NULL;

    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"place_order\",\"order\":{");

    jb_append(&jb, "\"symbol\":");
    jb_append_string(&jb, order->symbol);

    jb_append(&jb, ",\"side\":");
    jb_append_string(&jb, order->side == LXDEX_SIDE_BUY ? "buy" : "sell");

    const char *type_str = "limit";
    switch (order->type) {
        case LXDEX_ORDER_MARKET: type_str = "market"; break;
        case LXDEX_ORDER_STOP: type_str = "stop"; break;
        case LXDEX_ORDER_STOP_LIMIT: type_str = "stop_limit"; break;
        case LXDEX_ORDER_ICEBERG: type_str = "iceberg"; break;
        case LXDEX_ORDER_PEG: type_str = "peg"; break;
        default: break;
    }
    jb_append(&jb, ",\"type\":");
    jb_append_string(&jb, type_str);

    jb_append(&jb, ",\"price\":");
    jb_append_number(&jb, order->price);

    jb_append(&jb, ",\"size\":");
    jb_append_number(&jb, order->size);

    if (order->client_id[0]) {
        jb_append(&jb, ",\"clientId\":");
        jb_append_string(&jb, order->client_id);
    }

    const char *tif_str = "GTC";
    switch (order->time_in_force) {
        case LXDEX_TIF_IOC: tif_str = "IOC"; break;
        case LXDEX_TIF_FOK: tif_str = "FOK"; break;
        case LXDEX_TIF_DAY: tif_str = "DAY"; break;
        default: break;
    }
    jb_append(&jb, ",\"timeInForce\":");
    jb_append_string(&jb, tif_str);

    if (order->post_only) {
        jb_append(&jb, ",\"postOnly\":true");
    }

    if (order->reduce_only) {
        jb_append(&jb, ",\"reduceOnly\":true");
    }

    jb_append(&jb, "}");

    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }

    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build cancel_order message */
char *lxdex_json_cancel_order(uint64_t order_id, const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"cancel_order\",\"orderId\":");
    jb_append_uint(&jb, order_id);
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build subscribe message */
char *lxdex_json_subscribe(const char *channel, const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"subscribe\",\"channel\":");
    jb_append_string(&jb, channel);
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build unsubscribe message */
char *lxdex_json_unsubscribe(const char *channel, const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"unsubscribe\",\"channel\":");
    jb_append_string(&jb, channel);
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build ping message */
char *lxdex_json_ping(const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"ping\"");
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build get_balances message */
char *lxdex_json_get_balances(const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"get_balances\"");
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build get_positions message */
char *lxdex_json_get_positions(const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"get_positions\"");
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/* Build get_orders message */
char *lxdex_json_get_orders(const char *request_id) {
    json_builder_t jb;
    jb_init(&jb);

    jb_append(&jb, "{\"type\":\"get_orders\"");
    if (request_id) {
        jb_append(&jb, ",\"request_id\":");
        jb_append_string(&jb, request_id);
    }
    jb_append_char(&jb, '}');

    return jb.buf;
}

/*
 * Parsing functions for incoming messages
 */

/* Parse message type */
const char *lxdex_json_parse_type(const char *json) {
    const char *p = json;
    json_value_t *root = json_parse_value(&p);
    if (!root) return NULL;

    const char *type = json_get_string(root, "type");
    /* Note: caller must use result immediately or copy it */
    static char type_buf[64];
    if (type) {
        strncpy(type_buf, type, sizeof(type_buf) - 1);
        type_buf[sizeof(type_buf) - 1] = '\0';
    } else {
        type_buf[0] = '\0';
    }

    json_free(root);
    return type_buf[0] ? type_buf : NULL;
}

/* Parse order from JSON */
lxdex_error_t lxdex_json_parse_order(const char *json, lxdex_order_t *order) {
    if (!json || !order) return LXDEX_ERR_INVALID_ARG;

    const char *p = json;
    json_value_t *root = json_parse_value(&p);
    if (!root) return LXDEX_ERR_PARSE;

    /* Get order data - could be at root or in "data.order" */
    json_value_t *data = json_get_object(root, "data");
    json_value_t *ord = data ? json_get_object(data, "order") : root;
    if (!ord) ord = root;

    memset(order, 0, sizeof(*order));

    order->order_id = (uint64_t)json_get_number(ord, "orderId", 0);
    if (order->order_id == 0) {
        order->order_id = (uint64_t)json_get_number(ord, "ID", 0);
    }

    const char *sym = json_get_string(ord, "symbol");
    if (!sym) sym = json_get_string(ord, "Symbol");
    if (sym) strncpy(order->symbol, sym, LXDEX_SYMBOL_LEN - 1);

    const char *side = json_get_string(ord, "side");
    if (!side) side = json_get_string(ord, "Side");
    if (side) {
        order->side = (strcmp(side, "sell") == 0 || strcmp(side, "SELL") == 0)
            ? LXDEX_SIDE_SELL : LXDEX_SIDE_BUY;
    }

    order->price = json_get_number(ord, "price", 0);
    if (order->price == 0) order->price = json_get_number(ord, "Price", 0);

    order->size = json_get_number(ord, "size", 0);
    if (order->size == 0) order->size = json_get_number(ord, "Size", 0);

    order->filled = json_get_number(ord, "filled", 0);
    order->remaining = json_get_number(ord, "remaining", order->size - order->filled);

    const char *status = json_get_string(ord, "status");
    if (status) {
        if (strcmp(status, "open") == 0) order->status = LXDEX_STATUS_OPEN;
        else if (strcmp(status, "partial") == 0) order->status = LXDEX_STATUS_PARTIAL;
        else if (strcmp(status, "filled") == 0) order->status = LXDEX_STATUS_FILLED;
        else if (strcmp(status, "cancelled") == 0) order->status = LXDEX_STATUS_CANCELLED;
        else if (strcmp(status, "rejected") == 0) order->status = LXDEX_STATUS_REJECTED;
    }

    order->timestamp = (int64_t)json_get_number(ord, "timestamp", 0);
    order->post_only = json_get_bool(ord, "postOnly", false);
    order->reduce_only = json_get_bool(ord, "reduceOnly", false);

    json_free(root);
    return LXDEX_OK;
}

/* Parse trade from JSON */
lxdex_error_t lxdex_json_parse_trade(const char *json, lxdex_trade_t *trade) {
    if (!json || !trade) return LXDEX_ERR_INVALID_ARG;

    const char *p = json;
    json_value_t *root = json_parse_value(&p);
    if (!root) return LXDEX_ERR_PARSE;

    json_value_t *data = json_get_object(root, "data");
    json_value_t *t = data ? data : root;

    memset(trade, 0, sizeof(*trade));

    trade->trade_id = (uint64_t)json_get_number(t, "tradeId", 0);

    const char *sym = json_get_string(t, "symbol");
    if (sym) strncpy(trade->symbol, sym, LXDEX_SYMBOL_LEN - 1);

    trade->price = json_get_number(t, "price", 0);
    trade->size = json_get_number(t, "size", 0);

    const char *side = json_get_string(t, "side");
    if (side) {
        trade->side = (strcmp(side, "sell") == 0) ? LXDEX_SIDE_SELL : LXDEX_SIDE_BUY;
    }

    trade->buy_order_id = (uint64_t)json_get_number(t, "buyOrderId", 0);
    trade->sell_order_id = (uint64_t)json_get_number(t, "sellOrderId", 0);

    const char *buyer = json_get_string(t, "buyerId");
    if (buyer) strncpy(trade->buyer_id, buyer, LXDEX_USER_ID_LEN - 1);

    const char *seller = json_get_string(t, "sellerId");
    if (seller) strncpy(trade->seller_id, seller, LXDEX_USER_ID_LEN - 1);

    trade->timestamp = (int64_t)json_get_number(t, "timestamp", 0);

    json_free(root);
    return LXDEX_OK;
}

/* Parse orderbook from JSON */
lxdex_error_t lxdex_json_parse_orderbook(const char *json, lxdex_orderbook_t *book) {
    if (!json || !book) return LXDEX_ERR_INVALID_ARG;

    const char *p = json;
    json_value_t *root = json_parse_value(&p);
    if (!root) return LXDEX_ERR_PARSE;

    json_value_t *data = json_get_object(root, "data");
    json_value_t *b = data ? data : root;

    const char *sym = json_get_string(b, "symbol");
    if (!sym) sym = json_get_string(b, "Symbol");
    if (sym) strncpy(book->symbol, sym, LXDEX_SYMBOL_LEN - 1);

    book->timestamp = (int64_t)json_get_number(b, "timestamp", 0);
    if (book->timestamp == 0) book->timestamp = (int64_t)json_get_number(b, "Timestamp", 0);

    /* Parse bids */
    json_value_t *bids = json_get_array(b, "bids");
    if (!bids) bids = json_get_array(b, "Bids");
    if (bids) {
        size_t count = bids->data.array.count;
        if (count > book->bids_capacity) {
            lxdex_price_level_t *new_bids = realloc(book->bids,
                sizeof(lxdex_price_level_t) * count);
            if (!new_bids) {
                json_free(root);
                return LXDEX_ERR_NO_MEMORY;
            }
            book->bids = new_bids;
            book->bids_capacity = count;
        }

        book->bids_count = count;
        for (size_t i = 0; i < count; i++) {
            json_value_t *level = bids->data.array.items[i];
            book->bids[i].price = json_get_number(level, "price", 0);
            if (book->bids[i].price == 0)
                book->bids[i].price = json_get_number(level, "Price", 0);
            book->bids[i].size = json_get_number(level, "size", 0);
            if (book->bids[i].size == 0)
                book->bids[i].size = json_get_number(level, "Size", 0);
            book->bids[i].count = (int32_t)json_get_number(level, "count", 1);
        }
    }

    /* Parse asks */
    json_value_t *asks = json_get_array(b, "asks");
    if (!asks) asks = json_get_array(b, "Asks");
    if (asks) {
        size_t count = asks->data.array.count;
        if (count > book->asks_capacity) {
            lxdex_price_level_t *new_asks = realloc(book->asks,
                sizeof(lxdex_price_level_t) * count);
            if (!new_asks) {
                json_free(root);
                return LXDEX_ERR_NO_MEMORY;
            }
            book->asks = new_asks;
            book->asks_capacity = count;
        }

        book->asks_count = count;
        for (size_t i = 0; i < count; i++) {
            json_value_t *level = asks->data.array.items[i];
            book->asks[i].price = json_get_number(level, "price", 0);
            if (book->asks[i].price == 0)
                book->asks[i].price = json_get_number(level, "Price", 0);
            book->asks[i].size = json_get_number(level, "size", 0);
            if (book->asks[i].size == 0)
                book->asks[i].size = json_get_number(level, "Size", 0);
            book->asks[i].count = (int32_t)json_get_number(level, "count", 1);
        }
    }

    json_free(root);
    return LXDEX_OK;
}

/* Parse error from JSON */
lxdex_error_t lxdex_json_parse_error(const char *json, char *msg_out, size_t msg_len) {
    if (!json) return LXDEX_ERR_INVALID_ARG;

    const char *p = json;
    json_value_t *root = json_parse_value(&p);
    if (!root) return LXDEX_ERR_PARSE;

    const char *err = json_get_string(root, "error");
    if (err && msg_out) {
        strncpy(msg_out, err, msg_len - 1);
        msg_out[msg_len - 1] = '\0';
    }

    json_free(root);
    return err ? LXDEX_OK : LXDEX_ERR_PARSE;
}
