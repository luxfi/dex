"""
LX DEX Trading Client - Abstract Base Class and Types

Defines the common interface for all trading client implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, AsyncIterator
import time


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    PEG = "peg"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    DAY = "day"  # Day Order


@dataclass
class Order:
    """Order data model."""
    order_id: int
    symbol: str
    order_type: OrderType
    side: OrderSide
    price: float
    size: float
    filled: float = 0.0
    remaining: float = 0.0
    status: OrderStatus = OrderStatus.OPEN
    user_id: str = ""
    client_id: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    reduce_only: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "type": self.order_type.value,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "filled": self.filled,
            "remaining": self.remaining,
            "status": self.status.value,
            "user_id": self.user_id,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
            "time_in_force": self.time_in_force.value,
            "post_only": self.post_only,
            "reduce_only": self.reduce_only,
        }


@dataclass
class PriceLevel:
    """Price level in order book."""
    price: float
    size: float
    count: int = 1


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Trade:
    """Trade data model."""
    trade_id: int
    symbol: str
    price: float
    size: float
    side: OrderSide
    buy_order_id: int = 0
    sell_order_id: int = 0
    buyer_id: str = ""
    seller_id: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Position:
    """Position data model."""
    symbol: str
    size: float
    entry_price: float
    mark_price: float = 0.0
    pnl: float = 0.0
    margin: float = 0.0


@dataclass
class Balance:
    """Balance data model."""
    asset: str
    available: float
    locked: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        if self.total == 0.0:
            self.total = self.available + self.locked


class TradingClient(ABC):
    """
    Abstract base class for trading clients.

    All protocol-specific clients must implement this interface.
    Provides async/await support for all operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._authenticated = False

    @property
    def connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    @property
    def authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self._authenticated

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the trading server.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the trading server."""
        pass

    @abstractmethod
    async def authenticate(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with the server.

        Args:
            api_key: API key (uses instance key if not provided)
            api_secret: API secret (uses instance secret if not provided)

        Returns:
            True if authentication successful
        """
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        size: float,
        client_id: Optional[str] = None,
        time_in_force: str = "gtc",
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: Order side ("buy" or "sell")
            order_type: Order type ("limit", "market", etc.)
            price: Order price
            size: Order size
            client_id: Optional client order ID
            time_in_force: Time in force ("gtc", "ioc", "fok", "day")
            post_only: Post-only flag
            reduce_only: Reduce-only flag

        Returns:
            Order object with order details
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: int) -> Optional[Order]:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        Get list of orders.

        Args:
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders

        Returns:
            List of Order objects
        """
        pass

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> OrderBook:
        """
        Get order book snapshot.

        Args:
            symbol: Trading pair
            depth: Number of price levels

        Returns:
            OrderBook object
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Get balance for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Balance object or None
        """
        pass

    async def stream_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> AsyncIterator[OrderBook]:
        """
        Stream order book updates.

        Args:
            symbol: Trading pair
            depth: Number of price levels

        Yields:
            OrderBook objects
        """
        raise NotImplementedError("Streaming not supported by this client")

    async def stream_trades(
        self,
        symbol: str,
    ) -> AsyncIterator[Trade]:
        """
        Stream trades.

        Args:
            symbol: Trading pair

        Yields:
            Trade objects
        """
        raise NotImplementedError("Streaming not supported by this client")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        if self.api_key and self.api_secret:
            await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False
