"""
LX DEX Constants

Constants and configuration for the LX DEX connector.
"""

from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit

# Exchange info
EXCHANGE_NAME = "lx_dex"
BROKER_ID = "hummingbot"
MAX_ORDER_ID_LEN = 32

# Base URLs
MAINNET_BASE_URL = "https://api.dex.lux.network"
TESTNET_BASE_URL = "https://api.testnet.dex.lux.network"

MAINNET_WS_URL = "wss://ws.dex.lux.network"
TESTNET_WS_URL = "wss://ws.testnet.dex.lux.network"

# REST API endpoints
REST_URLS = {
    "health": "/health",
    "time": "/time",
    "symbols": "/symbols",
    "ticker": "/ticker",
    "orderbook": "/orderbook",
    "trades": "/trades",
    "klines": "/klines",
    "account": "/account",
    "balances": "/balances",
    "order": "/order",
    "orders": "/orders",
    "open_orders": "/orders/open",
    "order_history": "/orders/history",
    "my_trades": "/trades/my",
}

# WebSocket channels
WS_CHANNELS = {
    "orderbook": "orderbook",
    "trades": "trades",
    "ticker": "ticker",
    "orders": "orders",
    "balances": "balances",
}

# Order types
ORDER_TYPES = {
    "LIMIT": "LIMIT",
    "MARKET": "MARKET",
    "LIMIT_MAKER": "LIMIT_MAKER",
}

# Order sides
ORDER_SIDES = {
    "BUY": "BUY",
    "SELL": "SELL",
}

# Order status mapping
ORDER_STATUS = {
    "open": "OPEN",
    "partial": "PARTIALLY_FILLED",
    "filled": "FILLED",
    "cancelled": "CANCELED",
    "rejected": "FAILED",
    "expired": "EXPIRED",
}

# Time in force
TIME_IN_FORCE = {
    "GTC": "GTC",  # Good Till Cancel
    "IOC": "IOC",  # Immediate Or Cancel
    "FOK": "FOK",  # Fill Or Kill
    "GTT": "GTT",  # Good Till Time
}

# Rate limits
RATE_LIMITS = [
    RateLimit(limit_id="general", limit=50, time_interval=1),
    RateLimit(limit_id="orders", limit=10, time_interval=1),
    RateLimit(
        limit_id="order_create",
        limit=10,
        time_interval=1,
        linked_limits=[LinkedLimitWeightPair(limit_id="orders", weight=1)],
    ),
    RateLimit(
        limit_id="order_cancel",
        limit=10,
        time_interval=1,
        linked_limits=[LinkedLimitWeightPair(limit_id="orders", weight=1)],
    ),
]

# Timeouts (seconds)
API_CALL_TIMEOUT = 10.0
WS_HEARTBEAT_INTERVAL = 30.0
ORDER_UPDATE_INTERVAL = 10.0
BALANCE_UPDATE_INTERVAL = 30.0

# Trading fees (default, actual fees from API)
DEFAULT_MAKER_FEE = 0.001  # 0.1%
DEFAULT_TAKER_FEE = 0.002  # 0.2%

# Precision
PRICE_PRECISION = 8
QUANTITY_PRECISION = 8
