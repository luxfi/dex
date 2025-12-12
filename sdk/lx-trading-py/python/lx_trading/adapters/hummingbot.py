"""Hummingbot Gateway adapter."""

from decimal import Decimal
from typing import List, Optional
import time

import aiohttp

from lx_trading.adapters.base import VenueAdapter, VenueCapabilities
from lx_trading.config import HummingbotConfig
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


class HummingbotAdapter(VenueAdapter):
    """Hummingbot Gateway adapter."""

    def __init__(self, name: str, config: HummingbotConfig):
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
        return VenueType.HUMMINGBOT

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

        start = time.time()
        async with self._session.get(self._config.base_url) as resp:
            data = await resp.json()
            if data.get("status") != "ok":
                raise ConnectionError("Gateway not ready")
        self._latency = int((time.time() - start) * 1000)
        self._connected = True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        self._connected = False

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        if not self._session:
            raise RuntimeError("Not connected")

        url = f"{self._config.base_url}{path}"

        start = time.time()
        async with self._session.request(method, url, **kwargs) as resp:
            self._latency = int((time.time() - start) * 1000)
            resp.raise_for_status()
            return await resp.json()

    def _build_body(self, body: dict) -> dict:
        body["chain"] = self._config.chain
        body["network"] = self._config.network
        body["connector"] = self._config.connector
        if self._config.wallet_address:
            body["address"] = self._config.wallet_address
        return body

    async def get_markets(self) -> List[MarketInfo]:
        data = await self._request("POST", "/amm/tokens", json=self._build_body({}))

        markets = []
        tokens = data.get("tokens", [])
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i + 1:]:
                symbol1 = t1.get("symbol", "")
                symbol2 = t2.get("symbol", "")
                if symbol1 and symbol2:
                    markets.append(MarketInfo(
                        symbol=f"{symbol1}-{symbol2}",
                        base=symbol1,
                        quote=symbol2,
                        price_precision=8,
                        quantity_precision=8,
                        min_quantity=Decimal(0),
                        max_quantity=None,
                        min_notional=None,
                        tick_size=Decimal("0.00000001"),
                        lot_size=Decimal("0.00000001"),
                    ))
        return markets

    async def get_ticker(self, symbol: str) -> Ticker:
        pair = TradingPair.from_symbol(symbol)
        if not pair:
            raise ValueError(f"Invalid symbol: {symbol}")

        data = await self._request(
            "POST",
            "/amm/price",
            json=self._build_body({
                "base": pair.base,
                "quote": pair.quote,
                "amount": "1",
                "side": "BUY",
            }),
        )

        price = Decimal(str(data.get("price", 0))) if data.get("price") else None

        return Ticker(
            symbol=symbol,
            venue=self._name,
            bid=price,
            ask=price,
            last=price,
            volume_24h=None,
            high_24h=None,
            low_24h=None,
            change_24h=None,
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, depth: Optional[int] = None) -> Orderbook:
        raise NotImplementedError("Gateway AMM does not have orderbook")

    async def get_trades(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        return []  # Gateway doesn't provide trade history

    async def get_balances(self) -> List[Balance]:
        data = await self._request("POST", "/chain/balances", json=self._build_body({}))

        return [
            Balance(
                asset=asset,
                venue=self._name,
                free=Decimal(str(amount)),
                locked=Decimal(0),
            )
            for asset, amount in data.get("balances", {}).items()
        ]

    async def get_balance(self, asset: str) -> Balance:
        balances = await self.get_balances()
        for b in balances:
            if b.asset.lower() == asset.lower():
                return b
        return Balance(asset=asset, venue=self._name, free=Decimal(0), locked=Decimal(0))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []  # AMM doesn't have orders

    async def place_order(self, request: OrderRequest) -> Order:
        pair = TradingPair.from_symbol(request.symbol)
        if not pair:
            raise ValueError(f"Invalid symbol: {request.symbol}")

        trade = await self.execute_swap(
            pair.base,
            pair.quote,
            request.quantity,
            request.side == Side.BUY,
            Decimal("0.01"),
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
        raise NotImplementedError("Gateway AMM swaps cannot be cancelled")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    # AMM specific
    async def get_swap_quote(
        self, base_token: str, quote_token: str, amount: Decimal, is_buy: bool
    ) -> SwapQuote:
        data = await self._request(
            "POST",
            "/amm/price",
            json=self._build_body({
                "base": base_token,
                "quote": quote_token,
                "amount": str(amount),
                "side": "BUY" if is_buy else "SELL",
            }),
        )

        return SwapQuote(
            base_token=base_token,
            quote_token=quote_token,
            input_amount=amount,
            output_amount=Decimal(str(data.get("expectedAmount", 0))),
            price=Decimal(str(data.get("price", 0))),
            price_impact=Decimal(0),
            fee=Decimal(0),
            route=[],
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
            "/amm/trade",
            json=self._build_body({
                "base": base_token,
                "quote": quote_token,
                "amount": str(amount),
                "side": "BUY" if is_buy else "SELL",
                "limitPrice": "",
                "allowedSlippage": f"{slippage}/100",
            }),
        )

        return Trade(
            trade_id=data.get("txHash", ""),
            order_id=data.get("txHash", ""),
            symbol=f"{base_token}-{quote_token}",
            venue=self._name,
            side=Side.BUY if is_buy else Side.SELL,
            price=Decimal(str(data.get("price", 0))),
            quantity=amount,
            fee=Fee(asset="GAS", amount=Decimal(str(data.get("gasPrice", 0)))),
            timestamp=int(time.time() * 1000),
            is_maker=False,
        )

    async def get_pool_info(self, base_token: str, quote_token: str) -> PoolInfo:
        data = await self._request(
            "POST",
            "/amm/poolPrice",
            json=self._build_body({
                "token0": base_token,
                "token1": quote_token,
            }),
        )

        return PoolInfo(
            address=data.get("token0Address", ""),
            base_token=base_token,
            quote_token=quote_token,
            base_reserve=Decimal(str(data.get("token0Balance", 0))),
            quote_reserve=Decimal(str(data.get("token1Balance", 0))),
            total_liquidity=Decimal(0),
            fee_rate=Decimal("0.003"),
            apy=None,
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
            "/amm/liquidity/add",
            json=self._build_body({
                "token0": base_token,
                "token1": quote_token,
                "amount0": str(base_amount),
                "amount1": str(quote_amount),
                "allowedSlippage": f"{slippage}/100",
            }),
        )

        return LiquidityResult(
            tx_hash=data.get("txHash", ""),
            pool_address=data.get("poolAddress", ""),
            base_amount=base_amount,
            quote_amount=quote_amount,
            lp_tokens=Decimal(0),
            share_percent=Decimal(0),
        )

    async def remove_liquidity(
        self, pool_address: str, liquidity_amount: Decimal, slippage: Decimal
    ) -> LiquidityResult:
        data = await self._request(
            "POST",
            "/amm/liquidity/remove",
            json=self._build_body({
                "tokenId": pool_address,
                "decreasePercent": "100",
                "allowedSlippage": f"{slippage}/100",
            }),
        )

        return LiquidityResult(
            tx_hash=data.get("txHash", ""),
            pool_address=pool_address,
            base_amount=Decimal(0),
            quote_amount=Decimal(0),
            lp_tokens=liquidity_amount,
            share_percent=Decimal(0),
        )

    async def get_lp_positions(self) -> List[LpPosition]:
        data = await self._request("POST", "/amm/position", json=self._build_body({}))

        positions = []
        if isinstance(data, list):
            for p in data:
                positions.append(LpPosition(
                    pool_address=p.get("tokenId", ""),
                    base_token=p.get("token0", ""),
                    quote_token=p.get("token1", ""),
                    lp_tokens=Decimal(0),
                    base_amount=Decimal(str(p.get("amount0", 0))),
                    quote_amount=Decimal(str(p.get("amount1", 0))),
                    share_percent=Decimal(0),
                    unrealized_pnl=Decimal(str(p.get("unclaimedToken0", 0))) if p.get("unclaimedToken0") else None,
                ))

        return positions
