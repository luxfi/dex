"""Core types for LX Trading SDK."""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional
from uuid import uuid4


class Side(str, Enum):
    """Trading side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_MAKER = "limit_maker"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTD = "GTD"  # Good till date
    POST_ONLY = "POST_ONLY"  # Maker only


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class VenueType(str, Enum):
    """Venue type."""
    NATIVE = "native"
    CCXT = "ccxt"
    HUMMINGBOT = "hummingbot"
    CUSTOM = "custom"


@dataclass
class TradingPair:
    """Trading pair."""
    base: str
    quote: str

    @classmethod
    def from_symbol(cls, symbol: str) -> Optional["TradingPair"]:
        """Parse symbol like 'BTC-USDC' or 'BTC/USDC'."""
        for sep in ["-", "/", "_"]:
            if sep in symbol:
                parts = symbol.split(sep)
                if len(parts) == 2:
                    return cls(base=parts[0], quote=parts[1])
        return None

    def to_hummingbot(self) -> str:
        """Format as 'BASE-QUOTE'."""
        return f"{self.base}-{self.quote}"

    def to_ccxt(self) -> str:
        """Format as 'BASE/QUOTE'."""
        return f"{self.base}/{self.quote}"

    def __str__(self) -> str:
        return self.to_hummingbot()


@dataclass
class Fee:
    """Fee information."""
    asset: str
    amount: Decimal
    rate: Optional[Decimal] = None


@dataclass
class Balance:
    """Asset balance."""
    asset: str
    venue: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


@dataclass
class AggregatedBalance:
    """Aggregated balance across venues."""
    asset: str
    total_free: Decimal
    total_locked: Decimal
    by_venue: List[Balance]

    @property
    def total(self) -> Decimal:
        return self.total_free + self.total_locked


@dataclass
class OrderRequest:
    """Order request."""
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    venue: Optional[str] = None
    client_order_id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def market(cls, symbol: str, side: Side, quantity: Decimal) -> "OrderRequest":
        """Create market order."""
        return cls(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
        )

    @classmethod
    def limit(cls, symbol: str, side: Side, quantity: Decimal, price: Decimal) -> "OrderRequest":
        """Create limit order."""
        return cls(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
        )

    def with_venue(self, venue: str) -> "OrderRequest":
        """Set target venue."""
        self.venue = venue
        return self

    def with_post_only(self) -> "OrderRequest":
        """Make post-only (maker only)."""
        self.post_only = True
        self.time_in_force = TimeInForce.POST_ONLY
        return self


@dataclass
class Order:
    """Order."""
    order_id: str
    client_order_id: str
    symbol: str
    venue: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    price: Optional[Decimal]
    average_price: Optional[Decimal]
    created_at: int
    updated_at: int
    fees: List[Fee] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING]

    @property
    def is_done(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    @property
    def fill_percent(self) -> Decimal:
        if self.quantity == 0:
            return Decimal(0)
        return (self.filled_quantity / self.quantity) * 100


@dataclass
class Trade:
    """Trade/fill."""
    trade_id: str
    order_id: str
    symbol: str
    venue: str
    side: Side
    price: Decimal
    quantity: Decimal
    fee: Fee
    timestamp: int
    is_maker: bool

    @property
    def value(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class Ticker:
    """Ticker data."""
    symbol: str
    venue: str
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    last: Optional[Decimal]
    volume_24h: Optional[Decimal]
    high_24h: Optional[Decimal]
    low_24h: Optional[Decimal]
    change_24h: Optional[Decimal]
    timestamp: int

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> Optional[Decimal]:
        if self.bid and self.ask:
            return self.ask - self.bid
        return None

    @property
    def spread_percent(self) -> Optional[Decimal]:
        if self.bid and self.ask and self.bid > 0:
            return ((self.ask - self.bid) / self.bid) * 100
        return None


@dataclass
class SwapQuote:
    """AMM swap quote."""
    base_token: str
    quote_token: str
    input_amount: Decimal
    output_amount: Decimal
    price: Decimal
    price_impact: Decimal
    fee: Decimal
    route: List[str]
    expires_at: int


@dataclass
class PoolInfo:
    """AMM pool information."""
    address: str
    base_token: str
    quote_token: str
    base_reserve: Decimal
    quote_reserve: Decimal
    total_liquidity: Decimal
    fee_rate: Decimal
    apy: Optional[Decimal] = None


@dataclass
class LpPosition:
    """Liquidity provider position."""
    pool_address: str
    base_token: str
    quote_token: str
    lp_tokens: Decimal
    base_amount: Decimal
    quote_amount: Decimal
    share_percent: Decimal
    unrealized_pnl: Optional[Decimal] = None


@dataclass
class LiquidityResult:
    """Result of liquidity add/remove."""
    tx_hash: str
    pool_address: str
    base_amount: Decimal
    quote_amount: Decimal
    lp_tokens: Decimal
    share_percent: Decimal


@dataclass
class VenueInfo:
    """Venue information."""
    name: str
    venue_type: VenueType
    connected: bool
    latency_ms: Optional[int]
    supported_pairs: List[str]
    maker_fee: Decimal
    taker_fee: Decimal


@dataclass
class MarketInfo:
    """Market/pair information."""
    symbol: str
    base: str
    quote: str
    price_precision: int
    quantity_precision: int
    min_quantity: Decimal
    max_quantity: Optional[Decimal]
    min_notional: Optional[Decimal]
    tick_size: Decimal
    lot_size: Decimal


@dataclass
class PriceLevel:
    """Orderbook price level."""
    price: Decimal
    quantity: Decimal

    @property
    def value(self) -> Decimal:
        return self.price * self.quantity
