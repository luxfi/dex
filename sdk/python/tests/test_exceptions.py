"""
Tests for LX Python SDK exceptions
"""

import pytest

from lux_dex.exceptions import (
    LXDexException,
    ConnectionError,
    OrderError,
    AuthenticationError,
    RateLimitError,
    InvalidParameterError,
    InsufficientBalanceError,
    OrderNotFoundError,
    MarketClosedError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        exc = LXDexException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_connection_error(self):
        exc = ConnectionError("Connection failed")
        assert isinstance(exc, LXDexException)
        assert str(exc) == "Connection failed"

    def test_order_error(self):
        exc = OrderError("Order failed")
        assert isinstance(exc, LXDexException)

    def test_authentication_error(self):
        exc = AuthenticationError("Invalid credentials")
        assert isinstance(exc, LXDexException)

    def test_rate_limit_error(self):
        exc = RateLimitError("Too many requests")
        assert isinstance(exc, LXDexException)

    def test_invalid_parameter_error(self):
        exc = InvalidParameterError("Invalid symbol")
        assert isinstance(exc, LXDexException)

    def test_insufficient_balance_error(self):
        exc = InsufficientBalanceError("Not enough funds")
        assert isinstance(exc, OrderError)
        assert isinstance(exc, LXDexException)

    def test_order_not_found_error(self):
        exc = OrderNotFoundError("Order 123 not found")
        assert isinstance(exc, OrderError)

    def test_market_closed_error(self):
        exc = MarketClosedError("Market is closed")
        assert isinstance(exc, OrderError)


class TestExceptionRaising:
    def test_raise_and_catch_base(self):
        with pytest.raises(LXDexException):
            raise LXDexException("Base error")

    def test_catch_subclass_as_base(self):
        with pytest.raises(LXDexException):
            raise OrderError("Order error")

    def test_catch_specific_exception(self):
        with pytest.raises(InsufficientBalanceError):
            raise InsufficientBalanceError("Not enough funds")

    def test_catch_order_error_catches_subclasses(self):
        with pytest.raises(OrderError):
            raise InsufficientBalanceError("Not enough funds")

        with pytest.raises(OrderError):
            raise OrderNotFoundError("Order not found")

        with pytest.raises(OrderError):
            raise MarketClosedError("Market closed")
