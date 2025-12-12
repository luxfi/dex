"""Base adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Set

from lx_trading.types import (
    Balance,
    LiquidityResult,
    LpPosition,
    MarketInfo,
    Order,
    OrderRequest,
    PoolInfo,
    SwapQuote,
    Ticker,
    Trade,
    VenueInfo,
    VenueType,
)
from lx_trading.orderbook import Orderbook


@dataclass
class VenueCapabilities:
    """Venue capabilities."""
    limit_orders: bool = False
    market_orders: bool = False
    stop_orders: bool = False
    post_only: bool = False
    cancel_orders: bool = False
    batch_orders: bool = False
    streaming: bool = False
    orderbook: bool = False
    trades: bool = False
    amm_swap: bool = False
    add_liquidity: bool = False
    remove_liquidity: bool = False
    lp_positions: bool = False
    max_batch_size: int = 1
    supported_pairs: Set[str] = field(default_factory=set)

    @classmethod
    def clob(cls) -> "VenueCapabilities":
        """CLOB/orderbook venue capabilities."""
        return cls(
            limit_orders=True,
            market_orders=True,
            stop_orders=True,
            post_only=True,
            cancel_orders=True,
            batch_orders=True,
            streaming=True,
            orderbook=True,
            trades=True,
            max_batch_size=10,
        )

    @classmethod
    def amm(cls) -> "VenueCapabilities":
        """AMM venue capabilities."""
        return cls(
            market_orders=True,
            streaming=True,
            trades=True,
            amm_swap=True,
            add_liquidity=True,
            remove_liquidity=True,
            lp_positions=True,
        )


class VenueAdapter(ABC):
    """Base adapter interface for all venues."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Venue name."""
        pass

    @property
    @abstractmethod
    def venue_type(self) -> VenueType:
        """Venue type."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> VenueCapabilities:
        """Venue capabilities."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass

    @property
    def latency_ms(self) -> Optional[int]:
        """Connection latency in ms."""
        return None

    def info(self) -> VenueInfo:
        """Get venue info."""
        return VenueInfo(
            name=self.name,
            venue_type=self.venue_type,
            connected=self.is_connected,
            latency_ms=self.latency_ms,
            supported_pairs=list(self.capabilities.supported_pairs),
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.002"),
        )

    # Connection
    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    # Market data
    @abstractmethod
    async def get_markets(self) -> List[MarketInfo]:
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: Optional[int] = None) -> Orderbook:
        pass

    @abstractmethod
    async def get_trades(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        pass

    # Account
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Balance:
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        pass

    # Orders
    @abstractmethod
    async def place_order(self, request: OrderRequest) -> Order:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        pass

    # AMM (optional)
    async def get_swap_quote(
        self, base_token: str, quote_token: str, amount: Decimal, is_buy: bool
    ) -> SwapQuote:
        raise NotImplementedError("AMM swap not supported")

    async def execute_swap(
        self,
        base_token: str,
        quote_token: str,
        amount: Decimal,
        is_buy: bool,
        slippage: Decimal,
    ) -> Trade:
        raise NotImplementedError("AMM swap not supported")

    async def get_pool_info(self, base_token: str, quote_token: str) -> PoolInfo:
        raise NotImplementedError("Pool info not supported")

    async def add_liquidity(
        self,
        base_token: str,
        quote_token: str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage: Decimal,
    ) -> LiquidityResult:
        raise NotImplementedError("Add liquidity not supported")

    async def remove_liquidity(
        self, pool_address: str, liquidity_amount: Decimal, slippage: Decimal
    ) -> LiquidityResult:
        raise NotImplementedError("Remove liquidity not supported")

    async def get_lp_positions(self) -> List[LpPosition]:
        raise NotImplementedError("LP positions not supported")
