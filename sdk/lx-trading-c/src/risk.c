/**
 * LX Trading SDK - Risk Manager Implementation
 */

#include "lx_trading/risk.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Risk Configuration
 * ============================================================================ */

void lx_risk_config_init(LxRiskConfig* config) {
    if (!config) return;
    memset(config, 0, sizeof(LxRiskConfig));
    config->enabled = true;
    config->max_open_orders = 100;
}

int lx_risk_config_add_position_limit(LxRiskConfig* config, const char* asset, LxDecimal limit) {
    if (!config || !asset) return -1;
    if (config->position_limit_count >= LX_RISK_MAX_POSITION_LIMITS) return -1;

    strncpy(config->position_limits[config->position_limit_count].asset,
            asset, LX_ASSET_MAX_LEN - 1);
    config->position_limits[config->position_limit_count].limit = limit;
    config->position_limit_count++;

    return 0;
}

/* ============================================================================
 * Risk Error Messages
 * ============================================================================ */

const char* lx_risk_error_message(LxRiskError error) {
    switch (error) {
        case LX_RISK_OK: return "OK";
        case LX_RISK_ERROR_KILLED: return "Kill switch activated";
        case LX_RISK_ERROR_ORDER_SIZE: return "Order size exceeds limit";
        case LX_RISK_ERROR_POSITION_LIMIT: return "Position limit exceeded";
        case LX_RISK_ERROR_DAILY_LOSS: return "Daily loss limit exceeded";
        case LX_RISK_ERROR_OPEN_ORDERS: return "Maximum open orders exceeded";
        case LX_RISK_ERROR_DISABLED: return "Risk management disabled";
        default: return "Unknown error";
    }
}

/* ============================================================================
 * Risk Manager Lifecycle
 * ============================================================================ */

int lx_risk_manager_init(LxRiskManager* rm, const LxRiskConfig* config) {
    if (!rm) return -1;

    memset(rm, 0, sizeof(LxRiskManager));

    if (config) {
        rm->config = *config;
    } else {
        lx_risk_config_init(&rm->config);
    }

    atomic_store(&rm->kill_switch, false);
    rm->daily_pnl = lx_decimal_zero();

    if (pthread_rwlock_init(&rm->position_lock, NULL) != 0) {
        return -1;
    }

    if (pthread_rwlock_init(&rm->pnl_lock, NULL) != 0) {
        pthread_rwlock_destroy(&rm->position_lock);
        return -1;
    }

    if (pthread_rwlock_init(&rm->orders_lock, NULL) != 0) {
        pthread_rwlock_destroy(&rm->position_lock);
        pthread_rwlock_destroy(&rm->pnl_lock);
        return -1;
    }

    return 0;
}

void lx_risk_manager_cleanup(LxRiskManager* rm) {
    if (!rm) return;

    pthread_rwlock_destroy(&rm->position_lock);
    pthread_rwlock_destroy(&rm->pnl_lock);
    pthread_rwlock_destroy(&rm->orders_lock);
}

LxRiskManager* lx_risk_manager_create(const LxRiskConfig* config) {
    LxRiskManager* rm = malloc(sizeof(LxRiskManager));
    if (!rm) return NULL;

    if (lx_risk_manager_init(rm, config) != 0) {
        free(rm);
        return NULL;
    }

    return rm;
}

void lx_risk_manager_destroy(LxRiskManager* rm) {
    if (!rm) return;
    lx_risk_manager_cleanup(rm);
    free(rm);
}

/* ============================================================================
 * Risk Manager Configuration
 * ============================================================================ */

bool lx_risk_manager_is_enabled(const LxRiskManager* rm) {
    return rm ? rm->config.enabled : false;
}

const LxRiskConfig* lx_risk_manager_config(const LxRiskManager* rm) {
    return rm ? &rm->config : NULL;
}

/* ============================================================================
 * Kill Switch
 * ============================================================================ */

bool lx_risk_manager_is_killed(const LxRiskManager* rm) {
    return rm ? atomic_load(&rm->kill_switch) : false;
}

void lx_risk_manager_kill(LxRiskManager* rm) {
    if (rm) atomic_store(&rm->kill_switch, true);
}

void lx_risk_manager_reset(LxRiskManager* rm) {
    if (rm) atomic_store(&rm->kill_switch, false);
}

/* ============================================================================
 * Helper: Extract asset from symbol
 * ============================================================================ */

static void extract_base_asset(const char* symbol, char* asset, size_t asset_size) {
    if (!symbol || !asset || asset_size == 0) return;

    const char* sep = strchr(symbol, '-');
    if (!sep) sep = strchr(symbol, '/');
    if (!sep) sep = strchr(symbol, '_');

    if (sep) {
        size_t len = sep - symbol;
        if (len >= asset_size) len = asset_size - 1;
        strncpy(asset, symbol, len);
        asset[len] = '\0';
    } else {
        strncpy(asset, symbol, asset_size - 1);
        asset[asset_size - 1] = '\0';
    }
}

/* ============================================================================
 * Order Validation
 * ============================================================================ */

LxRiskError lx_risk_manager_validate_order(LxRiskManager* rm, const LxOrderRequest* request) {
    if (!rm || !request) return LX_RISK_ERROR_DISABLED;

    /* Check if disabled */
    if (!rm->config.enabled) {
        return LX_RISK_OK;
    }

    /* Check kill switch */
    if (atomic_load(&rm->kill_switch)) {
        return LX_RISK_ERROR_KILLED;
    }

    /* Check order size */
    if (!lx_decimal_is_zero(rm->config.max_order_size)) {
        if (lx_decimal_gt(request->quantity, rm->config.max_order_size)) {
            return LX_RISK_ERROR_ORDER_SIZE;
        }
    }

    /* Check position limits */
    char asset[LX_ASSET_MAX_LEN];
    extract_base_asset(request->symbol, asset, sizeof(asset));

    if (!lx_decimal_is_zero(rm->config.max_position_size)) {
        pthread_rwlock_rdlock(&rm->position_lock);

        LxDecimal current_pos = lx_decimal_zero();
        for (size_t i = 0; i < rm->position_count; i++) {
            if (strcmp(rm->positions[i].asset, asset) == 0) {
                current_pos = rm->positions[i].position;
                break;
            }
        }

        pthread_rwlock_unlock(&rm->position_lock);

        /* Calculate new position */
        LxDecimal new_pos;
        if (request->side == LX_SIDE_BUY) {
            new_pos = lx_decimal_add(current_pos, request->quantity);
        } else {
            new_pos = lx_decimal_sub(current_pos, request->quantity);
        }
        new_pos = lx_decimal_abs(new_pos);

        /* Check global limit */
        if (lx_decimal_gt(new_pos, rm->config.max_position_size)) {
            return LX_RISK_ERROR_POSITION_LIMIT;
        }

        /* Check asset-specific limit */
        for (size_t i = 0; i < rm->config.position_limit_count; i++) {
            if (strcmp(rm->config.position_limits[i].asset, asset) == 0) {
                if (lx_decimal_gt(new_pos, rm->config.position_limits[i].limit)) {
                    return LX_RISK_ERROR_POSITION_LIMIT;
                }
                break;
            }
        }
    }

    /* Check daily loss */
    if (!lx_decimal_is_zero(rm->config.max_daily_loss)) {
        pthread_rwlock_rdlock(&rm->pnl_lock);
        LxDecimal daily_pnl = rm->daily_pnl;
        pthread_rwlock_unlock(&rm->pnl_lock);

        if (lx_decimal_is_negative(daily_pnl)) {
            LxDecimal loss = lx_decimal_neg(daily_pnl);
            if (lx_decimal_gt(loss, rm->config.max_daily_loss)) {
                return LX_RISK_ERROR_DAILY_LOSS;
            }
        }
    }

    /* Check open orders */
    if (rm->config.max_open_orders > 0) {
        pthread_rwlock_rdlock(&rm->orders_lock);

        int open_count = 0;
        for (size_t i = 0; i < rm->open_orders_count; i++) {
            if (strcmp(rm->open_orders[i].symbol, request->symbol) == 0) {
                open_count = rm->open_orders[i].count;
                break;
            }
        }

        pthread_rwlock_unlock(&rm->orders_lock);

        if (open_count >= rm->config.max_open_orders) {
            return LX_RISK_ERROR_OPEN_ORDERS;
        }
    }

    return LX_RISK_OK;
}

/* ============================================================================
 * Position Tracking
 * ============================================================================ */

void lx_risk_manager_update_position(LxRiskManager* rm, const char* asset,
                                      LxDecimal quantity, LxSide side) {
    if (!rm || !asset) return;

    pthread_rwlock_wrlock(&rm->position_lock);

    /* Find existing position */
    size_t idx = rm->position_count;
    for (size_t i = 0; i < rm->position_count; i++) {
        if (strcmp(rm->positions[i].asset, asset) == 0) {
            idx = i;
            break;
        }
    }

    /* Create new position entry if needed */
    if (idx == rm->position_count) {
        if (rm->position_count < LX_RISK_MAX_POSITIONS) {
            strncpy(rm->positions[idx].asset, asset, LX_ASSET_MAX_LEN - 1);
            rm->positions[idx].position = lx_decimal_zero();
            rm->position_count++;
        } else {
            pthread_rwlock_unlock(&rm->position_lock);
            return;
        }
    }

    /* Update position */
    if (side == LX_SIDE_BUY) {
        rm->positions[idx].position = lx_decimal_add(rm->positions[idx].position, quantity);
    } else {
        rm->positions[idx].position = lx_decimal_sub(rm->positions[idx].position, quantity);
    }

    pthread_rwlock_unlock(&rm->position_lock);
}

LxDecimal lx_risk_manager_position(const LxRiskManager* rm, const char* asset) {
    if (!rm || !asset) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&rm->position_lock);

    LxDecimal pos = lx_decimal_zero();
    for (size_t i = 0; i < rm->position_count; i++) {
        if (strcmp(rm->positions[i].asset, asset) == 0) {
            pos = rm->positions[i].position;
            break;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&rm->position_lock);
    return pos;
}

size_t lx_risk_manager_positions(const LxRiskManager* rm,
                                  LxPositionEntry* entries, size_t max_entries) {
    if (!rm || !entries || max_entries == 0) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&rm->position_lock);

    size_t count = rm->position_count < max_entries ? rm->position_count : max_entries;
    memcpy(entries, rm->positions, count * sizeof(LxPositionEntry));

    pthread_rwlock_unlock((pthread_rwlock_t*)&rm->position_lock);
    return count;
}

/* ============================================================================
 * PnL Tracking
 * ============================================================================ */

void lx_risk_manager_update_pnl(LxRiskManager* rm, LxDecimal pnl) {
    if (!rm) return;

    pthread_rwlock_wrlock(&rm->pnl_lock);
    rm->daily_pnl = lx_decimal_add(rm->daily_pnl, pnl);
    pthread_rwlock_unlock(&rm->pnl_lock);

    /* Check for kill switch trigger */
    if (rm->config.kill_switch_enabled && !lx_decimal_is_zero(rm->config.max_daily_loss)) {
        pthread_rwlock_rdlock(&rm->pnl_lock);
        LxDecimal daily_pnl = rm->daily_pnl;
        pthread_rwlock_unlock(&rm->pnl_lock);

        if (lx_decimal_is_negative(daily_pnl)) {
            LxDecimal loss = lx_decimal_neg(daily_pnl);
            if (lx_decimal_gt(loss, rm->config.max_daily_loss)) {
                atomic_store(&rm->kill_switch, true);
            }
        }
    }
}

LxDecimal lx_risk_manager_daily_pnl(const LxRiskManager* rm) {
    if (!rm) return lx_decimal_zero();

    pthread_rwlock_rdlock((pthread_rwlock_t*)&rm->pnl_lock);
    LxDecimal pnl = rm->daily_pnl;
    pthread_rwlock_unlock((pthread_rwlock_t*)&rm->pnl_lock);

    return pnl;
}

void lx_risk_manager_reset_daily_pnl(LxRiskManager* rm) {
    if (!rm) return;

    pthread_rwlock_wrlock(&rm->pnl_lock);
    rm->daily_pnl = lx_decimal_zero();
    pthread_rwlock_unlock(&rm->pnl_lock);
}

/* ============================================================================
 * Order Tracking
 * ============================================================================ */

void lx_risk_manager_order_opened(LxRiskManager* rm, const char* symbol) {
    if (!rm || !symbol) return;

    pthread_rwlock_wrlock(&rm->orders_lock);

    /* Find existing entry */
    for (size_t i = 0; i < rm->open_orders_count; i++) {
        if (strcmp(rm->open_orders[i].symbol, symbol) == 0) {
            rm->open_orders[i].count++;
            pthread_rwlock_unlock(&rm->orders_lock);
            return;
        }
    }

    /* Create new entry */
    if (rm->open_orders_count < LX_RISK_MAX_SYMBOLS) {
        strncpy(rm->open_orders[rm->open_orders_count].symbol, symbol,
                sizeof(rm->open_orders[rm->open_orders_count].symbol) - 1);
        rm->open_orders[rm->open_orders_count].count = 1;
        rm->open_orders_count++;
    }

    pthread_rwlock_unlock(&rm->orders_lock);
}

void lx_risk_manager_order_closed(LxRiskManager* rm, const char* symbol) {
    if (!rm || !symbol) return;

    pthread_rwlock_wrlock(&rm->orders_lock);

    for (size_t i = 0; i < rm->open_orders_count; i++) {
        if (strcmp(rm->open_orders[i].symbol, symbol) == 0) {
            if (rm->open_orders[i].count > 0) {
                rm->open_orders[i].count--;
            }
            break;
        }
    }

    pthread_rwlock_unlock(&rm->orders_lock);
}

int lx_risk_manager_open_orders(const LxRiskManager* rm, const char* symbol) {
    if (!rm || !symbol) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&rm->orders_lock);

    int count = 0;
    for (size_t i = 0; i < rm->open_orders_count; i++) {
        if (strcmp(rm->open_orders[i].symbol, symbol) == 0) {
            count = rm->open_orders[i].count;
            break;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t*)&rm->orders_lock);
    return count;
}

/* ============================================================================
 * Pre-Trade Checks
 * ============================================================================ */

bool lx_risk_manager_check_order_size(const LxRiskManager* rm, LxDecimal quantity) {
    if (!rm || !rm->config.enabled) return true;
    if (lx_decimal_is_zero(rm->config.max_order_size)) return true;
    return lx_decimal_le(quantity, rm->config.max_order_size);
}

bool lx_risk_manager_check_position_limit(const LxRiskManager* rm, const char* asset,
                                           LxDecimal new_position) {
    if (!rm || !rm->config.enabled || !asset) return true;
    if (lx_decimal_is_zero(rm->config.max_position_size)) return true;

    LxDecimal abs_pos = lx_decimal_abs(new_position);

    /* Check global limit */
    if (lx_decimal_gt(abs_pos, rm->config.max_position_size)) {
        return false;
    }

    /* Check asset-specific limit */
    for (size_t i = 0; i < rm->config.position_limit_count; i++) {
        if (strcmp(rm->config.position_limits[i].asset, asset) == 0) {
            if (lx_decimal_gt(abs_pos, rm->config.position_limits[i].limit)) {
                return false;
            }
            break;
        }
    }

    return true;
}

bool lx_risk_manager_check_daily_loss(const LxRiskManager* rm) {
    if (!rm || !rm->config.enabled) return true;
    if (lx_decimal_is_zero(rm->config.max_daily_loss)) return true;

    pthread_rwlock_rdlock((pthread_rwlock_t*)&rm->pnl_lock);
    LxDecimal daily_pnl = rm->daily_pnl;
    pthread_rwlock_unlock((pthread_rwlock_t*)&rm->pnl_lock);

    if (lx_decimal_is_negative(daily_pnl)) {
        LxDecimal loss = lx_decimal_neg(daily_pnl);
        if (lx_decimal_gt(loss, rm->config.max_daily_loss)) {
            return false;
        }
    }

    return true;
}

bool lx_risk_manager_check_open_orders(const LxRiskManager* rm, const char* symbol) {
    if (!rm || !rm->config.enabled || !symbol) return true;
    if (rm->config.max_open_orders <= 0) return true;

    int count = lx_risk_manager_open_orders(rm, symbol);
    return count < rm->config.max_open_orders;
}

/* ============================================================================
 * Order Tracker
 * ============================================================================ */

void lx_order_tracker_begin(LxOrderTracker* tracker, LxRiskManager* rm, const char* symbol) {
    if (!tracker || !rm || !symbol) return;

    tracker->risk_manager = rm;
    strncpy(tracker->symbol, symbol, sizeof(tracker->symbol) - 1);
    tracker->symbol[sizeof(tracker->symbol) - 1] = '\0';
    tracker->released = false;

    lx_risk_manager_order_opened(rm, symbol);
}

void lx_order_tracker_end(LxOrderTracker* tracker) {
    if (!tracker || tracker->released) return;

    lx_risk_manager_order_closed(tracker->risk_manager, tracker->symbol);
}

void lx_order_tracker_release(LxOrderTracker* tracker) {
    if (tracker) tracker->released = true;
}
