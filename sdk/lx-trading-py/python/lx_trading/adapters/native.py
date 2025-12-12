"""Native LX DEX and LX AMM adapters."""

from decimal import Decimal
from typing import List, Optional
import time

import aiohttp

from lx_trading.adapters.base import VenueAdapter, VenueCapabilities
from lx_trading.config import NativeVenueConfig
from lx_trading.types import (
    Balance,
    Fee,
    LiquidityResult,
    LpPosition,
    MarketInfo,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    PoolInfo,
    Side,
    SwapQuote,
    Ticker,
    Trade,
    TradingPair,
    VenueType,
)
from lx_trading.orderbook import Orderbook


class LxDexAdapter(VenueAdapter):
    """LX DEX adapter for CLOB trading."""

    def __init__(self, name: str, config: NativeVenueConfig):
        self._name = name
        self._config = config
        self._capabilities = VenueCapabilities.clob()
        self._connected = False
        self._latency: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def venue_type(self) -> VenueType:
        return VenueType.NATIVE

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
        self._session = aiohttp.ClientSession()

        # Test connection
        start = time.time()
        async with self._session.get(f"{self._config.api_url}/api/v1/health") as resp:
            await resp.json()
        self._latency = int((time.time() - start) * 1000)
        self._connected = True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        self._connected = False

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        if not self._session:
            raise RuntimeError("Not connected")

        url = f"{self._config.api_url}{path}"
        headers = {}

        if self._config.api_key:
            headers["X-API-KEY"] = self._config.api_key
            headers["X-TIMESTAMP"] = str(int(time.time() * 1000))

        start = time.time()
        async with self._session.request(method, url, headers=headers, **kwargs) as resp:
            self._latency = int((time.time() - start) * 1000)
            resp.raise_for_status()
            return await resp.json()

    async def get_markets(self) -> List[MarketInfo]:
        data = await self._request("GET", "/api/v1/markets")
        return [
            MarketInfo(
                symbol=m["symbol"],
                base=m["base"],
                quote=m["quote"],
                price_precision=m.get("pricePrecision", 8),
                quantity_precision=m.get("quantityPrecision", 8),
                min_quantity=Decimal(str(m.get("minQuantity", 0))),
                max_quantity=Decimal(str(m.get("maxQuantity", 0))) if m.get("maxQuantity") else None,
                min_notional=Decimal(str(m.get("minNotional", 0))) if m.get("minNotional") else None,
                tick_size=Decimal(str(m.get("tickSize", "0.00000001"))),
                lot_size=Decimal(str(m.get("lotSize", "0.00000001"))),
            )
            for m in data
        ]

    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._request("GET", f"/api/v1/ticker/{symbol}")
        return Ticker(
            symbol=data["symbol"],
            venue=self._name,
            bid=Decimal(str(data["bid"])) if data.get("bid") else None,
            ask=Decimal(str(data["ask"])) if data.get("ask") else None,
            last=Decimal(str(data["last"])) if data.get("last") else None,
            volume_24h=Decimal(str(data["volume24h"])) if data.get("volume24h") else None,
            high_24h=Decimal(str(data["high24h"])) if data.get("high24h") else None,
            low_24h=Decimal(str(data["low24h"])) if data.get("low24h") else None,
            change_24h=Decimal(str(data["change24h"])) if data.get("change24h") else None,
            timestamp=data.get("timestamp", 0),
        )

    async def get_orderbook(self, symbol: str, depth: Optional[int] = None) -> Orderbook:
        path = f"/api/v1/orderbook/{symbol}"
        if depth:
            path += f"?depth={depth}"

        data = await self._request("GET", path)
        book = Orderbook(symbol, self._name)

        for bid in data.get("bids", []):
            book.add_bid(Decimal(str(bid[0])), Decimal(str(bid[1])))

        for ask in data.get("asks", []):
            book.add_ask(Decimal(str(ask[0])), Decimal(str(ask[1])))

        book.sort()
        return book

    async def get_trades(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        path = f"/api/v1/trades/{symbol}"
        if limit:
            path += f"?limit={limit}"

        data = await self._request("GET", path)
        return [
            Trade(
                trade_id=t["id"],
                order_id=t.get("orderId", ""),
                symbol=symbol,
                venue=self._name,
                side=Side.BUY if t["side"] == "buy" else Side.SELL,
                price=Decimal(str(t["price"])),
                quantity=Decimal(str(t["quantity"])),
                fee=Fee(
                    asset=t.get("feeAsset", ""),
                    amount=Decimal(str(t.get("feeAmount", 0))),
                ),
                timestamp=t["timestamp"],
                is_maker=t.get("isMaker", False),
            )
            for t in data
        ]

    async def get_balances(self) -> List[Balance]:
        data = await self._request("GET", "/api/v1/account/balances")
        return [
            Balance(
                asset=b["asset"],
                venue=self._name,
                free=Decimal(str(b["free"])),
                locked=Decimal(str(b["locked"])),
            )
            for b in data
        ]

    async def get_balance(self, asset: str) -> Balance:
        data = await self._request("GET", f"/api/v1/account/balance/{asset}")
        return Balance(
            asset=data["asset"],
            venue=self._name,
            free=Decimal(str(data["free"])),
            locked=Decimal(str(data["locked"])),
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        path = "/api/v1/orders?status=open"
        if symbol:
            path += f"&symbol={symbol}"

        data = await self._request("GET", path)
        return [self._convert_order(o) for o in data]

    async def place_order(self, request: OrderRequest) -> Order:
        body = {
            "clientOrderId": request.client_order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "type": request.order_type.value,
            "quantity": str(request.quantity),
            "timeInForce": request.time_in_force.value,
        }

        if request.price:
            body["price"] = str(request.price)

        data = await self._request("POST", "/api/v1/orders", json=body)
        return self._convert_order(data)

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        data = await self._request("DELETE", f"/api/v1/orders/{order_id}", json={"symbol": symbol})
        return self._convert_order(data)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        body = {}
        if symbol:
            body["symbol"] = symbol

        data = await self._request("DELETE", "/api/v1/orders/all", json=body)
        return [self._convert_order(o) for o in data]

    def _convert_order(self, o: dict) -> Order:
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }

        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "stop_loss_limit": OrderType.STOP_LOSS_LIMIT,
        }

        quantity = Decimal(str(o.get("quantity", 0)))
        filled = Decimal(str(o.get("filledQuantity", 0)))

        return Order(
            order_id=o["orderId"],
            client_order_id=o.get("clientOrderId", ""),
            symbol=o["symbol"],
            venue=self._name,
            side=Side.BUY if o["side"] == "buy" else Side.SELL,
            order_type=type_map.get(o.get("type", "limit"), OrderType.LIMIT),
            status=status_map.get(o.get("status", "open"), OrderStatus.OPEN),
            quantity=quantity,
            filled_quantity=filled,
            remaining_quantity=quantity - filled,
            price=Decimal(str(o.get("price"))) if o.get("price") else None,
            average_price=Decimal(str(o.get("averagePrice"))) if o.get("averagePrice") else None,
            created_at=o.get("createdAt", 0),
            updated_at=o.get("updatedAt", 0),
        )


class LxAmmAdapter(VenueAdapter):
    """LX AMM adapter for liquidity pool operations."""

    def __init__(self, name: str, config: NativeVenueConfig):
        self._name = name
        self._config = config
        self._capabilities = VenueCapabilities.amm()
        self._connected = False
        self._latency: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def venue_type(self) -> VenueType:
        return VenueType.NATIVE

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
        self._session = aiohttp.ClientSession()
        self._connected = True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        self._connected = False

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        if not self._session:
            raise RuntimeError("Not connected")

        url = f"{self._config.api_url}{path}"

        start = time.time()
        async with self._session.request(method, url, **kwargs) as resp:
            self._latency = int((time.time() - start) * 1000)
            resp.raise_for_status()
            return await resp.json()

    async def get_markets(self) -> List[MarketInfo]:
        data = await self._request("GET", "/api/v1/amm/pools")
        return [
            MarketInfo(
                symbol=f"{p['baseToken']}-{p['quoteToken']}",
                base=p["baseToken"],
                quote=p["quoteToken"],
                price_precision=8,
                quantity_precision=8,
                min_quantity=Decimal(0),
                max_quantity=None,
                min_notional=None,
                tick_size=Decimal("0.00000001"),
                lot_size=Decimal("0.00000001"),
            )
            for p in data
        ]

    async def get_ticker(self, symbol: str) -> Ticker:
        pair = TradingPair.from_symbol(symbol)
        if not pair:
            raise ValueError(f"Invalid symbol: {symbol}")

        data = await self._request("GET", f"/api/v1/amm/price/{pair.base}/{pair.quote}")
        price = Decimal(str(data.get("price", 0)))

        return Ticker(
            symbol=symbol,
            venue=self._name,
            bid=price,
            ask=price,
            last=price,
            volume_24h=Decimal(str(data.get("volume24h"))) if data.get("volume24h") else None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, depth: Optional[int] = None) -> Orderbook:
        raise NotImplementedError("AMM does not have orderbook")

    async def get_trades(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        pair = TradingPair.from_symbol(symbol)
        if not pair:
            return []

        path = f"/api/v1/amm/swaps/{pair.base}/{pair.quote}"
        if limit:
            path += f"?limit={limit}"

        data = await self._request("GET", path)
        return [
            Trade(
                trade_id=t["txHash"],
                order_id=t["txHash"],
                symbol=symbol,
                venue=self._name,
                side=Side.BUY if t.get("side") == "buy" else Side.SELL,
                price=Decimal(str(t["price"])),
                quantity=Decimal(str(t["amount"])),
                fee=Fee(asset="", amount=Decimal(str(t.get("fee", 0)))),
                timestamp=t["timestamp"],
                is_maker=False,
            )
            for t in data
        ]

    async def get_balances(self) -> List[Balance]:
        data = await self._request("GET", "/api/v1/account/balances")
        return [
            Balance(
                asset=b["asset"],
                venue=self._name,
                free=Decimal(str(b["free"])),
                locked=Decimal(str(b.get("locked", 0))),
            )
            for b in data
        ]

    async def get_balance(self, asset: str) -> Balance:
        data = await self._request("GET", f"/api/v1/account/balance/{asset}")
        return Balance(
            asset=data["asset"],
            venue=self._name,
            free=Decimal(str(data["free"])),
            locked=Decimal(str(data.get("locked", 0))),
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []  # AMM doesn't have orders

    async def place_order(self, request: OrderRequest) -> Order:
        # Convert to swap
        pair = TradingPair.from_symbol(request.symbol)
        if not pair:
            raise ValueError(f"Invalid symbol: {request.symbol}")

        trade = await self.execute_swap(
            pair.base,
            pair.quote,
            request.quantity,
            request.side == Side.BUY,
            Decimal("0.01"),  # 1% default slippage
        )

        return Order(
            order_id=trade.trade_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            venue=self._name,
            side=request.side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=request.quantity,
            filled_quantity=trade.quantity,
            remaining_quantity=Decimal(0),
            price=trade.price,
            average_price=trade.price,
            created_at=trade.timestamp,
            updated_at=trade.timestamp,
            fees=[trade.fee],
        )

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        raise NotImplementedError("AMM swaps cannot be cancelled")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    # AMM specific
    async def get_swap_quote(
        self, base_token: str, quote_token: str, amount: Decimal, is_buy: bool
    ) -> SwapQuote:
        data = await self._request(
            "POST",
            "/api/v1/amm/quote",
            json={
                "baseToken": base_token,
                "quoteToken": quote_token,
                "amount": str(amount),
                "side": "buy" if is_buy else "sell",
            },
        )

        return SwapQuote(
            base_token=base_token,
            quote_token=quote_token,
            input_amount=amount,
            output_amount=Decimal(str(data["outputAmount"])),
            price=Decimal(str(data["price"])),
            price_impact=Decimal(str(data.get("priceImpact", 0))),
            fee=Decimal(str(data.get("fee", 0))),
            route=data.get("route", []),
            expires_at=int(time.time() * 1000) + 60000,
        )

    async def execute_swap(
        self,
        base_token: str,
        quote_token: str,
        amount: Decimal,
        is_buy: bool,
        slippage: Decimal,
    ) -> Trade:
        data = await self._request(
            "POST",
            "/api/v1/amm/swap",
            json={
                "baseToken": base_token,
                "quoteToken": quote_token,
                "amount": str(amount),
                "side": "buy" if is_buy else "sell",
                "slippage": str(slippage),
            },
        )

        return Trade(
            trade_id=data["txHash"],
            order_id=data["txHash"],
            symbol=f"{base_token}-{quote_token}",
            venue=self._name,
            side=Side.BUY if is_buy else Side.SELL,
            price=Decimal(str(data["price"])),
            quantity=amount,
            fee=Fee(asset="", amount=Decimal(str(data.get("fee", 0)))),
            timestamp=int(time.time() * 1000),
            is_maker=False,
        )

    async def get_pool_info(self, base_token: str, quote_token: str) -> PoolInfo:
        data = await self._request("GET", f"/api/v1/amm/pool/{base_token}/{quote_token}")

        return PoolInfo(
            address=data["address"],
            base_token=base_token,
            quote_token=quote_token,
            base_reserve=Decimal(str(data["baseReserve"])),
            quote_reserve=Decimal(str(data["quoteReserve"])),
            total_liquidity=Decimal(str(data.get("totalLiquidity", 0))),
            fee_rate=Decimal(str(data.get("feeRate", "0.003"))),
            apy=Decimal(str(data.get("apy"))) if data.get("apy") else None,
        )

    async def add_liquidity(
        self,
        base_token: str,
        quote_token: str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage: Decimal,
    ) -> LiquidityResult:
        data = await self._request(
            "POST",
            "/api/v1/amm/liquidity/add",
            json={
                "baseToken": base_token,
                "quoteToken": quote_token,
                "baseAmount": str(base_amount),
                "quoteAmount": str(quote_amount),
                "slippage": str(slippage),
            },
        )

        return LiquidityResult(
            tx_hash=data["txHash"],
            pool_address=data.get("poolAddress", ""),
            base_amount=base_amount,
            quote_amount=quote_amount,
            lp_tokens=Decimal(str(data.get("lpTokens", 0))),
            share_percent=Decimal(str(data.get("sharePercent", 0))),
        )

    async def remove_liquidity(
        self, pool_address: str, liquidity_amount: Decimal, slippage: Decimal
    ) -> LiquidityResult:
        data = await self._request(
            "POST",
            "/api/v1/amm/liquidity/remove",
            json={
                "poolAddress": pool_address,
                "liquidity": str(liquidity_amount),
                "slippage": str(slippage),
            },
        )

        return LiquidityResult(
            tx_hash=data["txHash"],
            pool_address=pool_address,
            base_amount=Decimal(str(data.get("baseAmount", 0))),
            quote_amount=Decimal(str(data.get("quoteAmount", 0))),
            lp_tokens=liquidity_amount,
            share_percent=Decimal(0),
        )

    async def get_lp_positions(self) -> List[LpPosition]:
        data = await self._request("GET", "/api/v1/amm/positions")

        return [
            LpPosition(
                pool_address=p["poolAddress"],
                base_token=p["baseToken"],
                quote_token=p["quoteToken"],
                lp_tokens=Decimal(str(p.get("lpTokens", 0))),
                base_amount=Decimal(str(p["baseAmount"])),
                quote_amount=Decimal(str(p["quoteAmount"])),
                share_percent=Decimal(str(p.get("sharePercent", 0))),
                unrealized_pnl=Decimal(str(p.get("unrealizedPnl"))) if p.get("unrealizedPnl") else None,
            )
            for p in data
        ]
