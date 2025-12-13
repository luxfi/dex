"""
Tests for LX Python SDK types
"""

import pytest
from datetime import datetime

from lux_dex.types import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Order,
    Trade,
    OrderBookLevel,
    OrderBook,
    Balance,
    Position,
    NodeInfo,
)


class TestOrderType:
    def test_values(self):
        assert OrderType.LIMIT == 0
        assert OrderType.MARKET == 1
        assert OrderType.STOP == 2
        assert OrderType.STOP_LIMIT == 3
        assert OrderType.ICEBERG == 4
        assert OrderType.PEG == 5

    def test_is_int(self):
        assert isinstance(OrderType.LIMIT.value, int)


class TestOrderSide:
    def test_values(self):
        assert OrderSide.BUY == 0
        assert OrderSide.SELL == 1


class TestOrderStatus:
    def test_values(self):
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestTimeInForce:
    def test_values(self):
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.DAY.value == "DAY"


class TestOrder:
    def test_create_order(self):
        order = Order(
            symbol="BTC-USD",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=50000.0,
            size=1.0
        )
        assert order.symbol == "BTC-USD"
        assert order.type == OrderType.LIMIT
        assert order.side == OrderSide.BUY
        assert order.price == 50000.0
        assert order.size == 1.0

    def test_is_open(self):
        order = Order(status=OrderStatus.OPEN)
        assert order.is_open() is True

        order.status = OrderStatus.PARTIAL
        assert order.is_open() is True

        order.status = OrderStatus.FILLED
        assert order.is_open() is False

    def test_is_closed(self):
        order = Order(status=OrderStatus.FILLED)
        assert order.is_closed() is True

        order.status = OrderStatus.CANCELLED
        assert order.is_closed() is True

        order.status = OrderStatus.REJECTED
        assert order.is_closed() is True

        order.status = OrderStatus.OPEN
        assert order.is_closed() is False

    def test_fill_rate(self):
        order = Order(size=10.0, filled=5.0)
        assert order.fill_rate() == 0.5

        order.filled = 10.0
        assert order.fill_rate() == 1.0

        order.size = 0
        assert order.fill_rate() == 0.0


class TestTrade:
    def test_create_trade(self):
        trade = Trade(
            trade_id=1,
            symbol="BTC-USD",
            price=50000.0,
            size=1.0,
            side=OrderSide.BUY,
            buy_order_id=100,
            sell_order_id=101,
            buyer_id="buyer1",
            seller_id="seller1",
            timestamp=1704067200  # 2024-01-01
        )
        assert trade.trade_id == 1
        assert trade.symbol == "BTC-USD"
        assert trade.price == 50000.0

    def test_total_value(self):
        trade = Trade(
            trade_id=1,
            symbol="BTC-USD",
            price=50000.0,
            size=2.0,
            side=OrderSide.BUY,
            buy_order_id=100,
            sell_order_id=101,
            buyer_id="buyer1",
            seller_id="seller1",
            timestamp=1704067200
        )
        assert trade.total_value() == 100000.0

    def test_timestamp_datetime(self):
        trade = Trade(
            trade_id=1,
            symbol="BTC-USD",
            price=50000.0,
            size=1.0,
            side=OrderSide.BUY,
            buy_order_id=100,
            sell_order_id=101,
            buyer_id="buyer1",
            seller_id="seller1",
            timestamp=1704067200
        )
        dt = trade.timestamp_datetime()
        assert isinstance(dt, datetime)


class TestOrderBookLevel:
    def test_create_level(self):
        level = OrderBookLevel(price=50000.0, size=10.0, count=5)
        assert level.price == 50000.0
        assert level.size == 10.0
        assert level.count == 5

    def test_total_value(self):
        level = OrderBookLevel(price=50000.0, size=2.0)
        assert level.total_value() == 100000.0


class TestOrderBook:
    def test_create_orderbook(self):
        bids = [OrderBookLevel(50000, 1.0), OrderBookLevel(49999, 2.0)]
        asks = [OrderBookLevel(50001, 1.0), OrderBookLevel(50002, 2.0)]
        book = OrderBook(
            symbol="BTC-USD",
            bids=bids,
            asks=asks,
            timestamp=1704067200
        )
        assert book.symbol == "BTC-USD"
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    def test_best_bid(self):
        bids = [OrderBookLevel(50000, 1.0)]
        asks = [OrderBookLevel(50001, 1.0)]
        book = OrderBook(symbol="BTC-USD", bids=bids, asks=asks, timestamp=0)
        assert book.best_bid() == 50000

        book.bids = []
        assert book.best_bid() is None

    def test_best_ask(self):
        bids = [OrderBookLevel(50000, 1.0)]
        asks = [OrderBookLevel(50001, 1.0)]
        book = OrderBook(symbol="BTC-USD", bids=bids, asks=asks, timestamp=0)
        assert book.best_ask() == 50001

        book.asks = []
        assert book.best_ask() is None

    def test_spread(self):
        bids = [OrderBookLevel(50000, 1.0)]
        asks = [OrderBookLevel(50001, 1.0)]
        book = OrderBook(symbol="BTC-USD", bids=bids, asks=asks, timestamp=0)
        assert book.spread() == 1.0

        book.bids = []
        assert book.spread() is None

    def test_mid_price(self):
        bids = [OrderBookLevel(50000, 1.0)]
        asks = [OrderBookLevel(50002, 1.0)]
        book = OrderBook(symbol="BTC-USD", bids=bids, asks=asks, timestamp=0)
        assert book.mid_price() == 50001.0

    def test_spread_percentage(self):
        bids = [OrderBookLevel(50000, 1.0)]
        asks = [OrderBookLevel(50100, 1.0)]
        book = OrderBook(symbol="BTC-USD", bids=bids, asks=asks, timestamp=0)
        # spread = 100, mid = 50050, percentage = 100/50050 * 100 = ~0.2
        assert book.spread_percentage() is not None
        assert abs(book.spread_percentage() - 0.1998) < 0.01


class TestBalance:
    def test_create_balance(self):
        balance = Balance(
            asset="USD",
            available=10000.0,
            locked=5000.0,
            total=15000.0
        )
        assert balance.asset == "USD"
        assert balance.available == 10000.0

    def test_utilization(self):
        balance = Balance(asset="USD", available=5000.0, locked=5000.0, total=10000.0)
        assert balance.utilization() == 0.5

        balance.total = 0
        assert balance.utilization() == 0.0


class TestPosition:
    def test_create_position(self):
        position = Position(
            symbol="BTC-USD",
            size=1.0,
            entry_price=50000.0,
            mark_price=51000.0,
            pnl=1000.0,
            margin=5000.0
        )
        assert position.symbol == "BTC-USD"
        assert position.size == 1.0

    def test_unrealized_pnl(self):
        position = Position(
            symbol="BTC-USD",
            size=1.0,
            entry_price=50000.0,
            mark_price=51000.0,
            pnl=0.0,
            margin=5000.0
        )
        assert position.unrealized_pnl() == 1000.0

    def test_pnl_percentage(self):
        position = Position(
            symbol="BTC-USD",
            size=1.0,
            entry_price=50000.0,
            mark_price=51000.0,
            pnl=0.0,
            margin=5000.0
        )
        assert position.pnl_percentage() == 2.0

        position.entry_price = 0
        assert position.pnl_percentage() == 0.0


class TestNodeInfo:
    def test_create_node_info(self):
        info = NodeInfo(
            version="1.0.0",
            network="mainnet",
            order_count=1000,
            trade_count=500,
            timestamp=1704067200
        )
        assert info.version == "1.0.0"
        assert info.network == "mainnet"

    def test_timestamp_datetime(self):
        info = NodeInfo(
            version="1.0.0",
            network="mainnet",
            order_count=1000,
            trade_count=500,
            timestamp=1704067200
        )
        dt = info.timestamp_datetime()
        assert isinstance(dt, datetime)
