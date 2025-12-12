"""
LX Trading SDK - High-frequency trading with unified liquidity aggregation.

Supports:
- Native LX DEX (CLOB) and LX AMM
- CCXT exchanges (Binance, MEXC, OKX, etc.)
- Hummingbot Gateway connectors

Example:
    >>> from lx_trading import Client, Config
    >>>
    >>> config = Config.from_file("config.toml")
    >>> client = Client(config)
    >>> await client.connect()
    >>>
    >>> # Smart order routing across all venues
    >>> order = await client.buy("BTC-USDC", 1.0)
    >>>
    >>> # Or target specific venue
    >>> order = await client.buy("BTC-USDC", 1.0, venue="binance")
"""

__version__ = "0.1.0"

from lx_trading.client import Client
from lx_trading.config import Config
from lx_trading.types import (
    Balance,
    Fee,
    LpPosition,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    PoolInfo,
    Side,
    SwapQuote,
    Ticker,
    TimeInForce,
    Trade,
    TradingPair,
)
from lx_trading.adapters import (
    CcxtAdapter,
    HummingbotAdapter,
    LxDexAdapter,
    LxAmmAdapter,
)
from lx_trading.orderbook import Orderbook, AggregatedOrderbook
from lx_trading.risk import RiskManager
from lx_trading.execution import TwapExecutor, VwapExecutor, IcebergExecutor, SniperExecutor
from lx_trading.math import (
    # Pricing
    black_scholes,
    implied_volatility,
    greeks,
    # AMM
    constant_product_price,
    concentrated_liquidity_price,
    # Statistics
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    var,
    cvar,
)

__all__ = [
    # Core
    "Client",
    "Config",
    # Types
    "Balance",
    "Fee",
    "LpPosition",
    "Order",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "PoolInfo",
    "Side",
    "SwapQuote",
    "Ticker",
    "TimeInForce",
    "Trade",
    "TradingPair",
    # Adapters
    "CcxtAdapter",
    "HummingbotAdapter",
    "LxDexAdapter",
    "LxAmmAdapter",
    # Orderbook
    "Orderbook",
    "AggregatedOrderbook",
    # Risk
    "RiskManager",
    # Execution
    "TwapExecutor",
    "VwapExecutor",
    "IcebergExecutor",
    "SniperExecutor",
    # Math
    "black_scholes",
    "implied_volatility",
    "greeks",
    "constant_product_price",
    "concentrated_liquidity_price",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "var",
    "cvar",
]
