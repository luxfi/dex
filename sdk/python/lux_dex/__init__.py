"""
LX Python SDK

A Python client library for interacting with the LX trading platform.
"""

from .client import LXDexClient
from .types import OrderType, OrderSide, OrderStatus, TimeInForce
from .exceptions import LXDexException, ConnectionError, OrderError

__version__ = "1.0.0"
__all__ = [
    "LXDexClient",
    "OrderType",
    "OrderSide", 
    "OrderStatus",
    "TimeInForce",
    "LXDexException",
    "ConnectionError",
    "OrderError",
]