"""Unified trading client."""

from decimal import Decimal
from typing import Dict, List, Optional, Union

from lx_trading.config import Config
from lx_trading.types import (
    AggregatedBalance,
    Balance,
    LiquidityResult,
    LpPosition,
    Order,
    OrderRequest,
    PoolInfo,
    Side,
    SwapQuote,
    Ticker,
    Trade,
    VenueInfo,
)
from lx_trading.orderbook import AggregatedOrderbook, Orderbook
from lx_trading.adapters import (
    CcxtAdapter,
    HummingbotAdapter,
    LxAmmAdapter,
    LxDexAdapter,
    VenueAdapter,
)


class Client:
    """
    Unified trading client that abstracts multiple venues.

    Provides a single interface for:
    - Native LX DEX and LX AMM
    - CCXT exchanges (Binance, MEXC, OKX, etc.)
    - Hummingbot Gateway connectors

    Example:
        >>> config = Config.from_file("config.toml")
        >>> client = Client(config)
        >>> await client.connect()
        >>>
        >>> # Get aggregated orderbook
        >>> book = await client.aggregated_orderbook("BTC-USDC")
        >>>
        >>> # Smart order routing
        >>> order = await client.buy("BTC-USDC", Decimal("0.1"))
        >>>
        >>> # Target specific venue
        >>> order = await client.buy("BTC-USDC", Decimal("0.1"), venue="binance")
    """

    def __init__(self, config: Config):
        """Initialize client with configuration."""
        self.config = config
        self._venues: Dict[str, VenueAdapter] = {}
        self._default_venue: Optional[str] = None

    async def connect(self) -> None:
        """Connect to all configured venues."""
        # Native venues
        for name, cfg in self.config.native.items():
            if cfg.venue_type == "dex":
                adapter = LxDexAdapter(name, cfg)
            else:
                adapter = LxAmmAdapter(name, cfg)
            await adapter.connect()
            self._venues[name] = adapter

        # CCXT exchanges
        for name, cfg in self.config.ccxt.items():
            adapter = CcxtAdapter(name, cfg)
            await adapter.connect()
            self._venues[name] = adapter

        # Hummingbot gateways
        for name, cfg in self.config.hummingbot.items():
            adapter = HummingbotAdapter(name, cfg)
            await adapter.connect()
            self._venues[name] = adapter

        # Set default venue
        if self.config.general.venue_priority:
            self._default_venue = self.config.general.venue_priority[0]
        elif self._venues:
            self._default_venue = next(iter(self._venues.keys()))

    async def disconnect(self) -> None:
        """Disconnect from all venues."""
        for adapter in self._venues.values():
            await adapter.disconnect()
        self._venues.clear()

    def venue(self, name: str) -> Optional[VenueAdapter]:
        """Get specific venue adapter."""
        return self._venues.get(name)

    def venues(self) -> List[VenueInfo]:
        """List connected venues."""
        return [adapter.info() for adapter in self._venues.values()]

    # =========================================================================
    # Market Data
    # =========================================================================

    async def orderbook(self, symbol: str, venue: Optional[str] = None) -> Orderbook:
        """Get orderbook from a venue."""
        venue_name = venue or self._default_venue
        if not venue_name:
            raise ValueError("No venue specified and no default venue")

        adapter = self._venues.get(venue_name)
        if not adapter:
            raise ValueError(f"Venue not found: {venue_name}")

        return await adapter.get_orderbook(symbol)

    async def aggregated_orderbook(self, symbol: str) -> AggregatedOrderbook:
        """Get orderbook aggregated from all venues."""
        agg = AggregatedOrderbook(symbol)

        for adapter in self._venues.values():
            if adapter.capabilities.orderbook:
                try:
                    book = await adapter.get_orderbook(symbol, depth=20)
                    agg.add_orderbook(book)
                except Exception:
                    pass  # Skip venues that don't support this pair

        return agg

    async def ticker(self, symbol: str, venue: Optional[str] = None) -> Ticker:
        """Get ticker from a venue."""
        venue_name = venue or self._default_venue
        if not venue_name:
            raise ValueError("No venue specified and no default venue")

        adapter = self._venues.get(venue_name)
        if not adapter:
            raise ValueError(f"Venue not found: {venue_name}")

        return await adapter.get_ticker(symbol)

    async def tickers(self, symbol: str) -> List[Ticker]:
        """Get tickers from all venues."""
        tickers = []
        for adapter in self._venues.values():
            try:
                ticker = await adapter.get_ticker(symbol)
                tickers.append(ticker)
            except Exception:
                pass
        return tickers

    # =========================================================================
    # Account
    # =========================================================================

    async def balances(self) -> List[AggregatedBalance]:
        """Get balances aggregated across all venues."""
        by_asset: Dict[str, List[Balance]] = {}

        for adapter in self._venues.values():
            try:
                for balance in await adapter.get_balances():
                    if balance.asset not in by_asset:
                        by_asset[balance.asset] = []
                    by_asset[balance.asset].append(balance)
            except Exception:
                pass

        return [
            AggregatedBalance(
                asset=asset,
                total_free=sum(b.free for b in bals),
                total_locked=sum(b.locked for b in bals),
                by_venue=bals,
            )
            for asset, bals in by_asset.items()
        ]

    async def balance(self, asset: str, venue: Optional[str] = None) -> Balance:
        """Get balance for an asset."""
        venue_name = venue or self._default_venue
        if not venue_name:
            raise ValueError("No venue specified")

        adapter = self._venues.get(venue_name)
        if not adapter:
            raise ValueError(f"Venue not found: {venue_name}")

        return await adapter.get_balance(asset)

    # =========================================================================
    # Order Management
    # =========================================================================

    async def buy(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        venue: Optional[str] = None,
    ) -> Order:
        """Place market buy order."""
        qty = Decimal(str(quantity))
        request = OrderRequest.market(symbol, Side.BUY, qty)

        if venue:
            request.venue = venue
            return await self._place_order_on(request)
        elif self.config.general.smart_routing:
            return await self._smart_route(request)
        else:
            return await self._place_order(request)

    async def sell(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        venue: Optional[str] = None,
    ) -> Order:
        """Place market sell order."""
        qty = Decimal(str(quantity))
        request = OrderRequest.market(symbol, Side.SELL, qty)

        if venue:
            request.venue = venue
            return await self._place_order_on(request)
        elif self.config.general.smart_routing:
            return await self._smart_route(request)
        else:
            return await self._place_order(request)

    async def limit_buy(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        price: Union[Decimal, float, str],
        venue: Optional[str] = None,
    ) -> Order:
        """Place limit buy order."""
        request = OrderRequest.limit(symbol, Side.BUY, Decimal(str(quantity)), Decimal(str(price)))
        if venue:
            request.venue = venue
        return await self._place_order(request)

    async def limit_sell(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        price: Union[Decimal, float, str],
        venue: Optional[str] = None,
    ) -> Order:
        """Place limit sell order."""
        request = OrderRequest.limit(symbol, Side.SELL, Decimal(str(quantity)), Decimal(str(price)))
        if venue:
            request.venue = venue
        return await self._place_order(request)

    async def place_order(self, request: OrderRequest) -> Order:
        """Place order."""
        if request.venue:
            return await self._place_order_on(request)
        return await self._place_order(request)

    async def cancel_order(self, order_id: str, symbol: str, venue: str) -> Order:
        """Cancel order."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.cancel_order(order_id, symbol)

    async def cancel_all_orders(
        self, symbol: Optional[str] = None, venue: Optional[str] = None
    ) -> List[Order]:
        """Cancel all orders."""
        if venue:
            adapter = self._venues.get(venue)
            if not adapter:
                raise ValueError(f"Venue not found: {venue}")
            return await adapter.cancel_all_orders(symbol)

        # Cancel on all venues
        all_orders = []
        for adapter in self._venues.values():
            try:
                orders = await adapter.cancel_all_orders(symbol)
                all_orders.extend(orders)
            except Exception:
                pass
        return all_orders

    async def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders across venues."""
        all_orders = []
        for adapter in self._venues.values():
            try:
                orders = await adapter.get_open_orders(symbol)
                all_orders.extend(orders)
            except Exception:
                pass
        return all_orders

    # =========================================================================
    # AMM Operations
    # =========================================================================

    async def quote(
        self,
        base_token: str,
        quote_token: str,
        amount: Union[Decimal, float, str],
        is_buy: bool,
        venue: str,
    ) -> SwapQuote:
        """Get swap quote."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.get_swap_quote(base_token, quote_token, Decimal(str(amount)), is_buy)

    async def swap(
        self,
        base_token: str,
        quote_token: str,
        amount: Union[Decimal, float, str],
        is_buy: bool,
        slippage: float,
        venue: str,
    ) -> Trade:
        """Execute swap."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.execute_swap(
            base_token, quote_token, Decimal(str(amount)), is_buy, Decimal(str(slippage))
        )

    async def pool_info(self, base_token: str, quote_token: str, venue: str) -> PoolInfo:
        """Get pool information."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.get_pool_info(base_token, quote_token)

    async def add_liquidity(
        self,
        base_token: str,
        quote_token: str,
        base_amount: Union[Decimal, float, str],
        quote_amount: Union[Decimal, float, str],
        slippage: float,
        venue: str,
    ) -> LiquidityResult:
        """Add liquidity to pool."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.add_liquidity(
            base_token,
            quote_token,
            Decimal(str(base_amount)),
            Decimal(str(quote_amount)),
            Decimal(str(slippage)),
        )

    async def remove_liquidity(
        self,
        pool_address: str,
        liquidity_amount: Union[Decimal, float, str],
        slippage: float,
        venue: str,
    ) -> LiquidityResult:
        """Remove liquidity from pool."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.remove_liquidity(
            pool_address, Decimal(str(liquidity_amount)), Decimal(str(slippage))
        )

    async def lp_positions(self, venue: str) -> List[LpPosition]:
        """Get LP positions."""
        adapter = self._venues.get(venue)
        if not adapter:
            raise ValueError(f"Venue not found: {venue}")
        return await adapter.get_lp_positions()

    # =========================================================================
    # Internal
    # =========================================================================

    async def _place_order(self, request: OrderRequest) -> Order:
        """Place order on default venue."""
        if not self._default_venue:
            raise ValueError("No default venue")

        adapter = self._venues.get(self._default_venue)
        if not adapter:
            raise ValueError(f"Default venue not found: {self._default_venue}")

        return await adapter.place_order(request)

    async def _place_order_on(self, request: OrderRequest) -> Order:
        """Place order on specific venue."""
        if not request.venue:
            raise ValueError("No venue specified")

        adapter = self._venues.get(request.venue)
        if not adapter:
            raise ValueError(f"Venue not found: {request.venue}")

        return await adapter.place_order(request)

    async def _smart_route(self, request: OrderRequest) -> Order:
        """Smart order routing."""
        agg_book = await self.aggregated_orderbook(request.symbol)

        if request.side == Side.BUY:
            best = agg_book.best_venue_buy(request.quantity)
        else:
            best = agg_book.best_venue_sell(request.quantity)

        if best:
            venue, _ = best
            request.venue = venue
            return await self._place_order_on(request)

        # Fallback to default
        return await self._place_order(request)
