"""Tests for LX Trading SDK orderbook module."""

import pytest
from decimal import Decimal

from lx_trading.orderbook import Orderbook, AggregatedOrderbook
from lx_trading.types import Side


class TestOrderbook:
    def test_create_orderbook(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        assert book.symbol == "BTC-USDC"
        assert book.venue == "lx"
        assert book.timestamp > 0

    def test_add_and_retrieve_bids(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_bid(Decimal("99"), Decimal("2"))
        book.sort()

        assert len(book.bids) == 2
        assert book.bids[0].price == Decimal("100")  # Highest first
        assert book.bids[1].price == Decimal("99")

    def test_add_and_retrieve_asks(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("102"), Decimal("1.5"))
        book.add_ask(Decimal("101"), Decimal("2.5"))
        book.sort()

        assert len(book.asks) == 2
        assert book.asks[0].price == Decimal("101")  # Lowest first
        assert book.asks[1].price == Decimal("102")

    def test_sort_order(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("98"), Decimal("1"))
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_bid(Decimal("99"), Decimal("1"))
        book.add_ask(Decimal("103"), Decimal("1"))
        book.add_ask(Decimal("101"), Decimal("1"))
        book.add_ask(Decimal("102"), Decimal("1"))
        book.sort()

        assert book.bids[0].price == Decimal("100")
        assert book.bids[1].price == Decimal("99")
        assert book.bids[2].price == Decimal("98")

        assert book.asks[0].price == Decimal("101")
        assert book.asks[1].price == Decimal("102")
        assert book.asks[2].price == Decimal("103")


class TestOrderbookPrices:
    def test_best_bid(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_bid(Decimal("99"), Decimal("2"))
        book.sort()

        assert book.best_bid == Decimal("100")

    def test_best_ask(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("101"), Decimal("1"))
        book.add_ask(Decimal("102"), Decimal("2"))
        book.sort()

        assert book.best_ask == Decimal("101")

    def test_empty_book(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        assert book.best_bid is None
        assert book.best_ask is None

    def test_mid_price(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_ask(Decimal("102"), Decimal("1"))
        book.sort()

        assert book.mid_price == Decimal("101")

    def test_spread(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_ask(Decimal("102"), Decimal("1"))
        book.sort()

        assert book.spread == Decimal("2")

    def test_spread_percent(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_ask(Decimal("102"), Decimal("1"))
        book.sort()

        # spread = 2, mid = 101, spread% = 2/101*100 ≈ 1.98
        assert book.spread_percent is not None
        assert abs(float(book.spread_percent) - 1.98) < 0.01


class TestOrderbookLiquidity:
    def test_bid_liquidity(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))  # 100 value
        book.add_bid(Decimal("99"), Decimal("2"))  # 198 value
        book.sort()

        assert book.bid_liquidity == Decimal("298")

    def test_ask_liquidity(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("101"), Decimal("1.5"))  # 151.5 value
        book.add_ask(Decimal("102"), Decimal("2.5"))  # 255 value
        book.sort()

        assert book.ask_liquidity == Decimal("406.5")

    def test_bid_depth(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))  # 100 value
        book.add_bid(Decimal("99"), Decimal("2"))  # 198 value
        book.add_bid(Decimal("98"), Decimal("3"))  # 294 value
        book.sort()

        assert book.bid_depth(1) == Decimal("100")
        assert book.bid_depth(2) == Decimal("298")

    def test_ask_depth(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("101"), Decimal("1.5"))  # 151.5
        book.add_ask(Decimal("102"), Decimal("2.5"))  # 255
        book.sort()

        assert book.ask_depth(1) == Decimal("151.5")


class TestOrderbookVWAP:
    def test_vwap_small_buy(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("100"), Decimal("1"))
        book.add_ask(Decimal("101"), Decimal("2"))
        book.add_ask(Decimal("102"), Decimal("3"))
        book.sort()

        vwap = book.vwap_buy(Decimal("0.5"))
        assert vwap == Decimal("100")

    def test_vwap_across_levels(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("100"), Decimal("1"))
        book.add_ask(Decimal("101"), Decimal("2"))
        book.add_ask(Decimal("102"), Decimal("3"))
        book.sort()

        # Buying 2.5: 1@100 + 1.5@101 = 100 + 151.5 = 251.5 / 2.5 = 100.6
        vwap = book.vwap_buy(Decimal("2.5"))
        assert vwap is not None
        assert abs(float(vwap) - 100.6) < 0.01

    def test_vwap_sell(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("1"))
        book.add_bid(Decimal("99"), Decimal("2"))
        book.sort()

        vwap = book.vwap_sell(Decimal("0.5"))
        assert vwap == Decimal("100")

    def test_vwap_empty_book(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        assert book.vwap_buy(Decimal("1")) is None
        assert book.vwap_sell(Decimal("1")) is None


class TestOrderbookHasLiquidity:
    def test_sufficient_liquidity_buy(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_ask(Decimal("100"), Decimal("5"))
        book.sort()

        assert book.has_liquidity(Side.BUY, Decimal("3")) is True
        assert book.has_liquidity(Side.BUY, Decimal("10")) is False

    def test_sufficient_liquidity_sell(self):
        book = Orderbook(symbol="BTC-USDC", venue="lx")
        book.add_bid(Decimal("100"), Decimal("5"))
        book.sort()

        assert book.has_liquidity(Side.SELL, Decimal("3")) is True
        assert book.has_liquidity(Side.SELL, Decimal("10")) is False


class TestAggregatedOrderbook:
    @pytest.fixture
    def test_books(self):
        book1 = Orderbook(symbol="BTC-USDC", venue="venue1")
        book1.add_bid(Decimal("100"), Decimal("1"))
        book1.add_ask(Decimal("102"), Decimal("1"))
        book1.sort()

        book2 = Orderbook(symbol="BTC-USDC", venue="venue2")
        book2.add_bid(Decimal("99"), Decimal("2"))
        book2.add_ask(Decimal("101"), Decimal("1.5"))
        book2.sort()

        return book1, book2

    def test_aggregate_orderbooks(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")

        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        assert len(agg.bids) == 2
        assert len(agg.asks) == 2

    def test_best_bid_across_venues(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        best = agg.best_bid()
        assert best is not None
        price, venue, qty = best
        assert price == Decimal("100")
        assert venue == "venue1"

    def test_best_ask_across_venues(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        best = agg.best_ask()
        assert best is not None
        price, venue, qty = best
        assert price == Decimal("101")
        assert venue == "venue2"

    def test_empty_aggregated_book(self):
        agg = AggregatedOrderbook("BTC-USDC")
        assert agg.best_bid() is None
        assert agg.best_ask() is None

    def test_aggregated_bids(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        bids = agg.aggregated_bids()
        assert len(bids) == 2
        assert bids[0].price == Decimal("100")  # Highest first
        assert bids[1].price == Decimal("99")

    def test_aggregated_asks(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        asks = agg.aggregated_asks()
        assert len(asks) == 2
        assert asks[0].price == Decimal("101")  # Lowest first
        assert asks[1].price == Decimal("102")

    def test_sum_quantities_same_price(self):
        agg = AggregatedOrderbook("BTC-USDC")

        book1 = Orderbook(symbol="BTC-USDC", venue="venue1")
        book1.add_bid(Decimal("100"), Decimal("1"))
        book1.sort()

        book2 = Orderbook(symbol="BTC-USDC", venue="venue2")
        book2.add_bid(Decimal("100"), Decimal("2"))
        book2.sort()

        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        bids = agg.aggregated_bids()
        assert len(bids) == 1
        assert bids[0].quantity == Decimal("3")  # 1 + 2

    def test_best_venue_buy(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        best = agg.best_venue_buy(Decimal("1"))
        assert best is not None
        venue, price = best
        assert venue == "venue2"  # Lower ask at 101
        assert price == Decimal("101")

    def test_best_venue_sell(self, test_books):
        book1, book2 = test_books
        agg = AggregatedOrderbook("BTC-USDC")
        agg.add_orderbook(book1)
        agg.add_orderbook(book2)

        best = agg.best_venue_sell(Decimal("0.5"))
        assert best is not None
        venue, price = best
        assert venue == "venue1"  # Higher bid at 100
        assert price == Decimal("100")

    def test_best_venue_empty_book(self):
        agg = AggregatedOrderbook("BTC-USDC")
        assert agg.best_venue_buy(Decimal("1")) is None
        assert agg.best_venue_sell(Decimal("1")) is None
