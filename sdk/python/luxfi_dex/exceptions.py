"""
Exception definitions for LX DEX Python SDK
"""


class LXDexException(Exception):
    """Base exception for LX DEX SDK"""
    pass


class ConnectionError(LXDexException):
    """Connection-related errors"""
    pass


class OrderError(LXDexException):
    """Order-related errors"""
    pass


class AuthenticationError(LXDexException):
    """Authentication errors"""
    pass


class RateLimitError(LXDexException):
    """Rate limit exceeded"""
    pass


class InvalidParameterError(LXDexException):
    """Invalid parameter provided"""
    pass


class InsufficientBalanceError(OrderError):
    """Insufficient balance for order"""
    pass


class OrderNotFoundError(OrderError):
    """Order not found"""
    pass


class MarketClosedError(OrderError):
    """Market is closed"""
    pass