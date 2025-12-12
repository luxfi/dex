"""Tests for LX Trading SDK types module."""

import pytest
from decimal import Decimal

from lx_trading.types import (
    TradingPair,
    Balance,
    AggregatedBalance,
    OrderRequest,
    Order,
    Trade,
    Ticker,
    PriceLevel,
    Fee,
    Side,
    OrderType,
    OrderStatus,
    TimeInForce,
)


class TestTradingPair:
    def test_parse_dash_separator(self):
        pair = TradingPair.from_symbol("BTC-USDC")
        assert pair is not None
        assert pair.base == "BTC"
        assert pair.quote == "USDC"

    def test_parse_slash_separator(self):
        pair = TradingPair.from_symbol("ETH/USD")
        assert pair is not None
        assert pair.base == "ETH"
        assert pair.quote == "USD"

    def test_parse_underscore_separator(self):
        pair = TradingPair.from_symbol("SOL_USDT")
        assert pair is not None
        assert pair.base == "SOL"
        assert pair.quote == "USDT"

    def test_parse_invalid_symbol(self):
        pair = TradingPair.from_symbol("BTCUSDC")
        assert pair is None

    def test_to_hummingbot(self):
        pair = TradingPair(base="BTC", quote="USDC")
        assert pair.to_hummingbot() == "BTC-USDC"

    def test_to_ccxt(self):
        pair = TradingPair(base="ETH", quote="USD")
        assert pair.to_ccxt() == "ETH/USD"

    def test_str(self):
        pair = TradingPair(base="BTC", quote="USDC")
        assert str(pair) == "BTC-USDC"


class TestBalance:
    def test_total(self):
        balance = Balance(
            asset="BTC",
            venue="lx",
            free=Decimal("1.5"),
            locked=Decimal("0.5"),
        )
        assert balance.total == Decimal("2")


class TestAggregatedBalance:
    def test_total(self):
        balance = AggregatedBalance(
            asset="ETH",
            total_free=Decimal("10"),
            total_locked=Decimal("5"),
            by_venue=[],
        )
        assert balance.total == Decimal("15")


class TestOrderRequest:
    def test_market_order(self):
        order = OrderRequest.market("BTC-USDC", Side.BUY, Decimal("0.5"))
        assert order.symbol == "BTC-USDC"
        assert order.side == Side.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == Decimal("0.5")
        assert order.time_in_force == TimeInForce.IOC
        assert order.client_order_id  # Should have UUID

    def test_limit_order(self):
        order = OrderRequest.limit("ETH-USD", Side.SELL, Decimal("1"), Decimal("2000"))
        assert order.symbol == "ETH-USD"
        assert order.side == Side.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("1")
        assert order.price == Decimal("2000")
        assert order.time_in_force == TimeInForce.GTC

    def test_with_venue(self):
        order = OrderRequest.market("BTC-USDC", Side.BUY, Decimal("1"))
        order.with_venue("binance")
        assert order.venue == "binance"

    def test_with_post_only(self):
        order = OrderRequest.limit("BTC-USDC", Side.BUY, Decimal("1"), Decimal("50000"))
        order.with_post_only()
        assert order.post_only is True
        assert order.time_in_force == TimeInForce.POST_ONLY


class TestOrder:
    def test_is_open(self):
        order = Order(
            order_id="1",
            client_order_id="c1",
            symbol="BTC-USDC",
            venue="lx",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.OPEN,
            quantity=Decimal("1"),
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("1"),
            price=Decimal("50000"),
            average_price=None,
            created_at=1000,
            updated_at=1000,
        )
        assert order.is_open is True
        assert order.is_done is False

    def test_is_done(self):
        order = Order(
            order_id="2",
            client_order_id="c2",
            symbol="ETH-USD",
            venue="lx",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=Decimal("1"),
            filled_quantity=Decimal("1"),
            remaining_quantity=Decimal("0"),
            price=None,
            average_price=Decimal("2000"),
            created_at=1000,
            updated_at=1000,
        )
        assert order.is_open is False
        assert order.is_done is True

    def test_fill_percent(self):
        order = Order(
            order_id="3",
            client_order_id="c3",
            symbol="BTC-USDC",
            venue="lx",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PARTIALLY_FILLED,
            quantity=Decimal("10"),
            filled_quantity=Decimal("3"),
            remaining_quantity=Decimal("7"),
            price=Decimal("50000"),
            average_price=Decimal("50000"),
            created_at=1000,
            updated_at=1000,
        )
        assert order.fill_percent == Decimal("30")


class TestTrade:
    def test_value(self):
        trade = Trade(
            trade_id="t1",
            order_id="o1",
            symbol="BTC-USDC",
            venue="lx",
            side=Side.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            fee=Fee(asset="USDC", amount=Decimal("5")),
            timestamp=1000,
            is_maker=False,
        )
        assert trade.value == Decimal("5000")


class TestTicker:
    def test_mid_price_with_bid_ask(self):
        ticker = Ticker(
            symbol="BTC-USDC",
            venue="lx",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last=Decimal("50000"),
            volume_24h=None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=1000,
        )
        assert ticker.mid_price == Decimal("50000")

    def test_mid_price_without_bid_ask(self):
        ticker = Ticker(
            symbol="BTC-USDC",
            venue="lx",
            bid=None,
            ask=None,
            last=Decimal("50000"),
            volume_24h=None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=1000,
        )
        assert ticker.mid_price == Decimal("50000")

    def test_spread(self):
        ticker = Ticker(
            symbol="BTC-USDC",
            venue="lx",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last=Decimal("50000"),
            volume_24h=None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=1000,
        )
        assert ticker.spread == Decimal("200")

    def test_spread_percent(self):
        ticker = Ticker(
            symbol="BTC-USDC",
            venue="lx",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last=Decimal("50000"),
            volume_24h=None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=1000,
        )
        # spread_percent = (200/49900) * 100 ≈ 0.401
        assert ticker.spread_percent is not None
        assert 0.4 < float(ticker.spread_percent) < 0.41


class TestPriceLevel:
    def test_value(self):
        level = PriceLevel(price=Decimal("100"), quantity=Decimal("5"))
        assert level.value == Decimal("500")
