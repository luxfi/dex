"""CCXT adapter - trade on 100+ exchanges."""

from decimal import Decimal
from typing import List, Optional
import time

import ccxt.async_support as ccxt

from lx_trading.adapters.base import VenueAdapter, VenueCapabilities
from lx_trading.config import CcxtConfig
from lx_trading.types import (
    Balance,
    Fee,
    MarketInfo,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
    Trade,
    VenueType,
)
from lx_trading.orderbook import Orderbook


class CcxtAdapter(VenueAdapter):
    """CCXT adapter for 100+ exchanges."""

    def __init__(self, name: str, config: CcxtConfig):
        self._name = name
        self._config = config
        self._capabilities = VenueCapabilities.orderBook()
        self._capabilities.batch_orders = False  # CCXT doesn't have unified batch
        self._connected = False
        self._latency: Optional[int] = None
        self._exchange: Optional[ccxt.Exchange] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def venue_type(self) -> VenueType:
        return VenueType.CCXT

    @property
    def capabilities(self) -> VenueCapabilities:
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def latency_ms(self) -> Optional[int]:
        return self._latency

    async def connect(self) -> None:
        exchange_class = getattr(ccxt, self._config.exchange_id)

        self._exchange = exchange_class({
            "apiKey": self._config.api_key,
            "secret": self._config.api_secret,
            "password": self._config.password,
            "enableRateLimit": self._config.rate_limit,
            "options": self._config.options,
        })

        if self._config.sandbox:
            self._exchange.set_sandbox_mode(True)

        start = time.time()
        await self._exchange.load_markets()
        self._latency = int((time.time() - start) * 1000)
        self._connected = True

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()
        self._connected = False

    async def get_markets(self) -> List[MarketInfo]:
        if not self._exchange:
            raise RuntimeError("Not connected")

        markets = []
        for symbol, market in self._exchange.markets.items():
            markets.append(MarketInfo(
                symbol=symbol,
                base=market.get("base", ""),
                quote=market.get("quote", ""),
                price_precision=market.get("precision", {}).get("price", 8),
                quantity_precision=market.get("precision", {}).get("amount", 8),
                min_quantity=Decimal(str(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)),
                max_quantity=Decimal(str(market.get("limits", {}).get("amount", {}).get("max", 0) or 0)) or None,
                min_notional=Decimal(str(market.get("limits", {}).get("cost", {}).get("min", 0) or 0)) or None,
                tick_size=Decimal("0.00000001"),
                lot_size=Decimal("0.00000001"),
            ))
        return markets

    async def get_ticker(self, symbol: str) -> Ticker:
        if not self._exchange:
            raise RuntimeError("Not connected")

        start = time.time()
        ticker = await self._exchange.fetch_ticker(symbol)
        self._latency = int((time.time() - start) * 1000)

        return Ticker(
            symbol=ticker.get("symbol", symbol),
            venue=self._name,
            bid=Decimal(str(ticker["bid"])) if ticker.get("bid") else None,
            ask=Decimal(str(ticker["ask"])) if ticker.get("ask") else None,
            last=Decimal(str(ticker["last"])) if ticker.get("last") else None,
            volume_24h=Decimal(str(ticker["baseVolume"])) if ticker.get("baseVolume") else None,
            high_24h=Decimal(str(ticker["high"])) if ticker.get("high") else None,
            low_24h=Decimal(str(ticker["low"])) if ticker.get("low") else None,
            change_24h=Decimal(str(ticker["percentage"])) if ticker.get("percentage") else None,
            timestamp=ticker.get("timestamp", 0),
        )

    async def get_orderbook(self, symbol: str, depth: Optional[int] = None) -> Orderbook:
        if not self._exchange:
            raise RuntimeError("Not connected")

        start = time.time()
        book = await self._exchange.fetch_order_book(symbol, depth)
        self._latency = int((time.time() - start) * 1000)

        orderbook = Orderbook(symbol, self._name)

        for price, qty in book.get("bids", []):
            orderbook.add_bid(Decimal(str(price)), Decimal(str(qty)))

        for price, qty in book.get("asks", []):
            orderbook.add_ask(Decimal(str(price)), Decimal(str(qty)))

        orderbook.sort()
        return orderbook

    async def get_trades(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        if not self._exchange:
            raise RuntimeError("Not connected")

        trades = await self._exchange.fetch_trades(symbol, limit=limit)

        return [
            Trade(
                trade_id=t.get("id", ""),
                order_id=t.get("order", ""),
                symbol=t.get("symbol", symbol),
                venue=self._name,
                side=Side.BUY if t.get("side") == "buy" else Side.SELL,
                price=Decimal(str(t.get("price", 0))),
                quantity=Decimal(str(t.get("amount", 0))),
                fee=Fee(
                    asset=t.get("fee", {}).get("currency", ""),
                    amount=Decimal(str(t.get("fee", {}).get("cost", 0) or 0)),
                ),
                timestamp=t.get("timestamp", 0),
                is_maker=t.get("takerOrMaker") == "maker",
            )
            for t in trades
        ]

    async def get_balances(self) -> List[Balance]:
        if not self._exchange:
            raise RuntimeError("Not connected")

        start = time.time()
        balances = await self._exchange.fetch_balance()
        self._latency = int((time.time() - start) * 1000)

        result = []
        for asset, amount in balances.get("total", {}).items():
            if amount and float(amount) > 0:
                result.append(Balance(
                    asset=asset,
                    venue=self._name,
                    free=Decimal(str(balances.get("free", {}).get(asset, 0) or 0)),
                    locked=Decimal(str(balances.get("used", {}).get(asset, 0) or 0)),
                ))
        return result

    async def get_balance(self, asset: str) -> Balance:
        balances = await self.get_balances()
        for b in balances:
            if b.asset == asset:
                return b
        return Balance(asset=asset, venue=self._name, free=Decimal(0), locked=Decimal(0))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        if not self._exchange:
            raise RuntimeError("Not connected")

        orders = await self._exchange.fetch_open_orders(symbol)
        return [self._convert_order(o) for o in orders]

    async def place_order(self, request: OrderRequest) -> Order:
        if not self._exchange:
            raise RuntimeError("Not connected")

        side = "buy" if request.side == Side.BUY else "sell"
        order_type = "market" if request.order_type == OrderType.MARKET else "limit"

        start = time.time()
        order = await self._exchange.create_order(
            symbol=request.symbol,
            type=order_type,
            side=side,
            amount=float(request.quantity),
            price=float(request.price) if request.price else None,
            params={"clientOrderId": request.client_order_id},
        )
        self._latency = int((time.time() - start) * 1000)

        return self._convert_order(order)

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        if not self._exchange:
            raise RuntimeError("Not connected")

        order = await self._exchange.cancel_order(order_id, symbol)
        return self._convert_order(order)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        if not self._exchange:
            raise RuntimeError("Not connected")

        try:
            orders = await self._exchange.cancel_all_orders(symbol)
            return [self._convert_order(o) for o in orders]
        except Exception:
            # Fallback: cancel one by one
            open_orders = await self.get_open_orders(symbol)
            cancelled = []
            for order in open_orders:
                try:
                    o = await self.cancel_order(order.order_id, order.symbol)
                    cancelled.append(o)
                except Exception:
                    pass
            return cancelled

    def _convert_order(self, o: dict) -> Order:
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }

        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_LOSS,
            "stop_limit": OrderType.STOP_LOSS_LIMIT,
        }

        quantity = Decimal(str(o.get("amount", 0)))
        filled = Decimal(str(o.get("filled", 0)))

        return Order(
            order_id=o.get("id", ""),
            client_order_id=o.get("clientOrderId", ""),
            symbol=o.get("symbol", ""),
            venue=self._name,
            side=Side.BUY if o.get("side") == "buy" else Side.SELL,
            order_type=type_map.get(o.get("type", "limit"), OrderType.LIMIT),
            status=status_map.get(o.get("status", "open"), OrderStatus.OPEN),
            quantity=quantity,
            filled_quantity=filled,
            remaining_quantity=quantity - filled,
            price=Decimal(str(o.get("price"))) if o.get("price") else None,
            average_price=Decimal(str(o.get("average"))) if o.get("average") else None,
            created_at=o.get("timestamp", 0),
            updated_at=o.get("lastTradeTimestamp", 0) or o.get("timestamp", 0),
        )
