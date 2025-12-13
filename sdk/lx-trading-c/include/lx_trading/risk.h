/**
 * LX Trading SDK - Risk Management
 * Thread-safe risk controls and position tracking
 */

#ifndef LX_TRADING_RISK_H
#define LX_TRADING_RISK_H

#include "types.h"
#include <pthread.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Risk Configuration
 * ============================================================================ */

#define LX_RISK_MAX_POSITION_LIMITS 64

typedef struct {
    char asset[LX_ASSET_MAX_LEN];
    LxDecimal limit;
} LxPositionLimit;

typedef struct {
    bool enabled;
    LxDecimal max_position_size;
    LxDecimal max_order_size;
    LxDecimal max_daily_loss;
    int max_open_orders;
    bool kill_switch_enabled;

    /* Per-asset position limits */
    LxPositionLimit position_limits[LX_RISK_MAX_POSITION_LIMITS];
    size_t position_limit_count;
} LxRiskConfig;

/* Initialize default config */
void lx_risk_config_init(LxRiskConfig* config);

/* Add a position limit for a specific asset */
int lx_risk_config_add_position_limit(LxRiskConfig* config, const char* asset, LxDecimal limit);

/* ============================================================================
 * Risk Error Codes
 * ============================================================================ */

typedef enum {
    LX_RISK_OK = 0,
    LX_RISK_ERROR_KILLED = -1,
    LX_RISK_ERROR_ORDER_SIZE = -2,
    LX_RISK_ERROR_POSITION_LIMIT = -3,
    LX_RISK_ERROR_DAILY_LOSS = -4,
    LX_RISK_ERROR_OPEN_ORDERS = -5,
    LX_RISK_ERROR_DISABLED = -6
} LxRiskError;

/* Get error message for risk error code */
const char* lx_risk_error_message(LxRiskError error);

/* ============================================================================
 * Position Entry
 * ============================================================================ */

#define LX_RISK_MAX_POSITIONS 256

typedef struct {
    char asset[LX_ASSET_MAX_LEN];
    LxDecimal position;
} LxPositionEntry;

/* ============================================================================
 * Open Orders Entry
 * ============================================================================ */

#define LX_RISK_MAX_SYMBOLS 256

typedef struct {
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    int count;
} LxOpenOrdersEntry;

/* ============================================================================
 * Risk Manager
 * ============================================================================
 * Thread-safe risk manager with position tracking, PnL monitoring, and kill switch.
 */

typedef struct {
    LxRiskConfig config;
    atomic_bool kill_switch;

    /* Position tracking */
    LxPositionEntry positions[LX_RISK_MAX_POSITIONS];
    size_t position_count;
    pthread_rwlock_t position_lock;

    /* PnL tracking */
    LxDecimal daily_pnl;
    pthread_rwlock_t pnl_lock;

    /* Open orders tracking */
    LxOpenOrdersEntry open_orders[LX_RISK_MAX_SYMBOLS];
    size_t open_orders_count;
    pthread_rwlock_t orders_lock;
} LxRiskManager;

/* ============================================================================
 * Risk Manager Lifecycle
 * ============================================================================ */

/* Initialize risk manager (returns 0 on success) */
int lx_risk_manager_init(LxRiskManager* rm, const LxRiskConfig* config);

/* Cleanup risk manager */
void lx_risk_manager_cleanup(LxRiskManager* rm);

/* Create/destroy on heap */
LxRiskManager* lx_risk_manager_create(const LxRiskConfig* config);
void lx_risk_manager_destroy(LxRiskManager* rm);

/* ============================================================================
 * Risk Manager Configuration
 * ============================================================================ */

bool lx_risk_manager_is_enabled(const LxRiskManager* rm);
const LxRiskConfig* lx_risk_manager_config(const LxRiskManager* rm);

/* ============================================================================
 * Kill Switch
 * ============================================================================ */

bool lx_risk_manager_is_killed(const LxRiskManager* rm);
void lx_risk_manager_kill(LxRiskManager* rm);
void lx_risk_manager_reset(LxRiskManager* rm);

/* ============================================================================
 * Order Validation
 * ============================================================================
 * Returns LX_RISK_OK if valid, or error code if invalid.
 */

LxRiskError lx_risk_manager_validate_order(LxRiskManager* rm, const LxOrderRequest* request);

/* ============================================================================
 * Position Tracking
 * ============================================================================ */

/* Update position for an asset (adds delta based on side) */
void lx_risk_manager_update_position(LxRiskManager* rm, const char* asset,
                                      LxDecimal quantity, LxSide side);

/* Get current position for an asset */
LxDecimal lx_risk_manager_position(const LxRiskManager* rm, const char* asset);

/* Get all positions (caller provides buffer) */
size_t lx_risk_manager_positions(const LxRiskManager* rm,
                                  LxPositionEntry* entries, size_t max_entries);

/* ============================================================================
 * PnL Tracking
 * ============================================================================ */

/* Update daily PnL (accumulates) */
void lx_risk_manager_update_pnl(LxRiskManager* rm, LxDecimal pnl);

/* Get current daily PnL */
LxDecimal lx_risk_manager_daily_pnl(const LxRiskManager* rm);

/* Reset daily PnL to zero */
void lx_risk_manager_reset_daily_pnl(LxRiskManager* rm);

/* ============================================================================
 * Order Tracking
 * ============================================================================ */

/* Increment open orders for a symbol */
void lx_risk_manager_order_opened(LxRiskManager* rm, const char* symbol);

/* Decrement open orders for a symbol */
void lx_risk_manager_order_closed(LxRiskManager* rm, const char* symbol);

/* Get open order count for a symbol */
int lx_risk_manager_open_orders(const LxRiskManager* rm, const char* symbol);

/* ============================================================================
 * Pre-Trade Checks (return bool, don't throw)
 * ============================================================================ */

bool lx_risk_manager_check_order_size(const LxRiskManager* rm, LxDecimal quantity);
bool lx_risk_manager_check_position_limit(const LxRiskManager* rm, const char* asset,
                                           LxDecimal new_position);
bool lx_risk_manager_check_daily_loss(const LxRiskManager* rm);
bool lx_risk_manager_check_open_orders(const LxRiskManager* rm, const char* symbol);

/* ============================================================================
 * Order Tracker (RAII-style helper)
 * ============================================================================
 * Use lx_order_tracker_begin() to increment open orders, then
 * lx_order_tracker_end() to decrement (unless released).
 */

typedef struct {
    LxRiskManager* risk_manager;
    char symbol[LX_SYMBOL_MAX_LEN * 2 + 2];
    bool released;
} LxOrderTracker;

/* Begin tracking (increments open orders) */
void lx_order_tracker_begin(LxOrderTracker* tracker, LxRiskManager* rm, const char* symbol);

/* End tracking (decrements open orders unless released) */
void lx_order_tracker_end(LxOrderTracker* tracker);

/* Release tracker (prevents decrement on end) */
void lx_order_tracker_release(LxOrderTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif /* LX_TRADING_RISK_H */
