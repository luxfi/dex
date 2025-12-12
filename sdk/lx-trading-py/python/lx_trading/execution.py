"""Execution algorithms."""

import asyncio
from decimal import Decimal
from typing import List, Optional

from lx_trading.types import Order, Side


class TwapExecutor:
    """Time-Weighted Average Price execution."""

    def __init__(
        self,
        client,  # Client
        symbol: str,
        side: Side,
        total_quantity: Decimal,
        duration_seconds: int,
        num_slices: int,
    ):
        self.client = client
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.num_slices = num_slices
        self.interval = duration_seconds / num_slices

    async def execute(self) -> List[Order]:
        """Execute TWAP strategy."""
        slice_qty = self.total_quantity / self.num_slices
        orders = []

        for i in range(self.num_slices):
            remaining = self.total_quantity - slice_qty * i
            qty = min(slice_qty, remaining)

            if qty <= 0:
                break

            if self.side == Side.BUY:
                order = await self.client.buy(self.symbol, qty)
            else:
                order = await self.client.sell(self.symbol, qty)

            orders.append(order)

            if i < self.num_slices - 1:
                await asyncio.sleep(self.interval)

        return orders


class VwapExecutor:
    """Volume-Weighted Average Price execution."""

    def __init__(
        self,
        client,  # Client
        symbol: str,
        side: Side,
        total_quantity: Decimal,
        participation_rate: Decimal,  # e.g., 0.1 = 10% of volume
        max_duration_seconds: int,
    ):
        self.client = client
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.participation_rate = participation_rate
        self.max_duration = max_duration_seconds

    async def execute(self) -> List[Order]:
        """Execute VWAP strategy."""
        orders = []
        remaining = self.total_quantity
        elapsed = 0
        check_interval = 5  # seconds

        while remaining > 0 and elapsed < self.max_duration:
            ticker = await self.client.ticker(self.symbol)
            volume = ticker.volume_24h or Decimal(1000)

            # Calculate slice based on participation
            hourly_volume = volume / 24
            slice_volume = hourly_volume * self.participation_rate / (3600 / check_interval)
            qty = min(slice_volume, remaining)

            if qty > 0:
                if self.side == Side.BUY:
                    order = await self.client.buy(self.symbol, qty)
                else:
                    order = await self.client.sell(self.symbol, qty)

                orders.append(order)
                remaining -= qty

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        return orders


class IcebergExecutor:
    """Iceberg order execution."""

    def __init__(
        self,
        client,  # Client
        symbol: str,
        side: Side,
        total_quantity: Decimal,
        visible_quantity: Decimal,
        price: Decimal,
        venue: Optional[str] = None,
    ):
        self.client = client
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.visible_quantity = visible_quantity
        self.price = price
        self.venue = venue

    async def execute(self) -> List[Order]:
        """Execute iceberg strategy."""
        orders = []
        remaining = self.total_quantity

        while remaining > 0:
            qty = min(self.visible_quantity, remaining)

            if self.side == Side.BUY:
                order = await self.client.limit_buy(
                    self.symbol, qty, self.price, venue=self.venue
                )
            else:
                order = await self.client.limit_sell(
                    self.symbol, qty, self.price, venue=self.venue
                )

            # Wait for fill
            while order.is_open:
                await asyncio.sleep(0.5)
                # Would need to refresh order status here
                break  # Simplified

            remaining -= order.filled_quantity
            orders.append(order)

        return orders


class SniperExecutor:
    """Sniper execution - wait for price target."""

    def __init__(
        self,
        client,  # Client
        symbol: str,
        side: Side,
        quantity: Decimal,
        target_price: Decimal,
        timeout_seconds: int,
    ):
        self.client = client
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.target_price = target_price
        self.timeout = timeout_seconds

    async def execute(self) -> Optional[Order]:
        """Execute sniper strategy."""
        elapsed = 0
        check_interval = 0.1  # 100ms

        while elapsed < self.timeout:
            ticker = await self.client.ticker(self.symbol)

            should_execute = False
            if self.side == Side.BUY:
                if ticker.ask and ticker.ask <= self.target_price:
                    should_execute = True
            else:
                if ticker.bid and ticker.bid >= self.target_price:
                    should_execute = True

            if should_execute:
                if self.side == Side.BUY:
                    return await self.client.buy(self.symbol, self.quantity)
                else:
                    return await self.client.sell(self.symbol, self.quantity)

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        return None  # Timeout
