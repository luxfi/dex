"""Orderbook with aggregation support."""

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import time

from lx_trading.types import PriceLevel, Side


@dataclass
class Orderbook:
    """Orderbook with bids and asks."""
    symbol: str
    venue: str
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    sequence: int = 0

    def add_bid(self, price: Decimal, quantity: Decimal) -> None:
        self.bids.append(PriceLevel(price=price, quantity=quantity))

    def add_ask(self, price: Decimal, quantity: Decimal) -> None:
        self.asks.append(PriceLevel(price=price, quantity=quantity))

    def sort(self) -> None:
        """Sort bids descending, asks ascending."""
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_percent(self) -> Optional[Decimal]:
        if self.spread and self.mid_price and self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return None

    @property
    def bid_liquidity(self) -> Decimal:
        return sum(l.value for l in self.bids)

    @property
    def ask_liquidity(self) -> Decimal:
        return sum(l.value for l in self.asks)

    def bid_depth(self, levels: int) -> Decimal:
        return sum(l.value for l in self.bids[:levels])

    def ask_depth(self, levels: int) -> Decimal:
        return sum(l.value for l in self.asks[:levels])

    def vwap_buy(self, amount: Decimal) -> Optional[Decimal]:
        """VWAP for buying `amount`."""
        return self._calculate_vwap(self.asks, amount)

    def vwap_sell(self, amount: Decimal) -> Optional[Decimal]:
        """VWAP for selling `amount`."""
        return self._calculate_vwap(self.bids, amount)

    def _calculate_vwap(self, levels: List[PriceLevel], amount: Decimal) -> Optional[Decimal]:
        remaining = amount
        total_value = Decimal(0)
        total_qty = Decimal(0)

        for level in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            total_value += fill_qty * level.price
            total_qty += fill_qty
            remaining -= fill_qty

        if total_qty == 0:
            return None
        return total_value / total_qty

    def has_liquidity(self, side: Side, amount: Decimal) -> bool:
        """Check if there's enough liquidity."""
        levels = self.asks if side == Side.BUY else self.bids
        total = sum(l.quantity for l in levels)
        return total >= amount


class AggregatedOrderbook:
    """Orderbook aggregated from multiple venues."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[Decimal, List[Tuple[str, Decimal]]] = defaultdict(list)  # price -> [(venue, qty)]
        self.asks: Dict[Decimal, List[Tuple[str, Decimal]]] = defaultdict(list)
        self.timestamp = int(time.time() * 1000)

    def add_orderbook(self, book: Orderbook) -> None:
        """Add orderbook from a venue."""
        for level in book.bids:
            self.bids[level.price].append((book.venue, level.quantity))

        for level in book.asks:
            self.asks[level.price].append((book.venue, level.quantity))

        self.timestamp = max(self.timestamp, book.timestamp)

    def best_bid(self) -> Optional[Tuple[Decimal, str, Decimal]]:
        """Get best bid across all venues: (price, venue, qty)."""
        if not self.bids:
            return None

        best_price = max(self.bids.keys())
        venues = self.bids[best_price]
        if venues:
            venue, qty = venues[0]
            return (best_price, venue, qty)
        return None

    def best_ask(self) -> Optional[Tuple[Decimal, str, Decimal]]:
        """Get best ask across all venues: (price, venue, qty)."""
        if not self.asks:
            return None

        best_price = min(self.asks.keys())
        venues = self.asks[best_price]
        if venues:
            venue, qty = venues[0]
            return (best_price, venue, qty)
        return None

    def aggregated_bids(self) -> List[PriceLevel]:
        """Get aggregated bid levels."""
        levels = []
        for price in sorted(self.bids.keys(), reverse=True):
            total_qty = sum(qty for _, qty in self.bids[price])
            levels.append(PriceLevel(price=price, quantity=total_qty))
        return levels

    def aggregated_asks(self) -> List[PriceLevel]:
        """Get aggregated ask levels."""
        levels = []
        for price in sorted(self.asks.keys()):
            total_qty = sum(qty for _, qty in self.asks[price])
            levels.append(PriceLevel(price=price, quantity=total_qty))
        return levels

    def best_venue_buy(self, amount: Decimal) -> Optional[Tuple[str, Decimal]]:
        """Find best venue for buying `amount`: (venue, price)."""
        best_venue = None
        best_price = Decimal("inf")
        remaining = amount

        for price in sorted(self.asks.keys()):
            if remaining <= 0:
                break

            for venue, qty in self.asks[price]:
                fill = min(remaining, qty)
                if price < best_price:
                    best_price = price
                    best_venue = venue
                remaining -= fill
                if remaining <= 0:
                    break

        if best_venue:
            return (best_venue, best_price)
        return None

    def best_venue_sell(self, amount: Decimal) -> Optional[Tuple[str, Decimal]]:
        """Find best venue for selling `amount`: (venue, price)."""
        best_venue = None
        best_price = Decimal(0)
        remaining = amount

        for price in sorted(self.bids.keys(), reverse=True):
            if remaining <= 0:
                break

            for venue, qty in self.bids[price]:
                fill = min(remaining, qty)
                if price > best_price:
                    best_price = price
                    best_venue = venue
                remaining -= fill
                if remaining <= 0:
                    break

        if best_venue:
            return (best_venue, best_price)
        return None
