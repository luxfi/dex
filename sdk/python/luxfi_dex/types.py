"""
Type definitions for LX DEX Python SDK
"""

from enum import IntEnum, Enum
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


class OrderType(IntEnum):
    """Order types"""
    LIMIT = 0
    MARKET = 1
    STOP = 2
    STOP_LIMIT = 3
    ICEBERG = 4
    PEG = 5


class OrderSide(IntEnum):
    """Order sides"""
    BUY = 0
    SELL = 1


class OrderStatus(Enum):
    """Order status"""
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    """Time in force options"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day Order


@dataclass
class Order:
    """Order data structure"""
    order_id: Optional[int] = None
    symbol: Optional[str] = None
    type: Optional[OrderType] = None
    side: Optional[OrderSide] = None
    price: Optional[float] = None
    size: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    status: Optional[OrderStatus] = None
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    timestamp: Optional[int] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    reduce_only: bool = False
    
    def is_open(self) -> bool:
        """Check if order is open"""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIAL]
    
    def is_closed(self) -> bool:
        """Check if order is closed"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
    def fill_rate(self) -> float:
        """Calculate fill rate"""
        if self.size and self.size > 0:
            return self.filled / self.size
        return 0.0


@dataclass
class Trade:
    """Trade data structure"""
    trade_id: int
    symbol: str
    price: float
    size: float
    side: OrderSide
    buy_order_id: int
    sell_order_id: int
    buyer_id: str
    seller_id: str
    timestamp: int
    
    def total_value(self) -> float:
        """Calculate total trade value"""
        return self.price * self.size
    
    def timestamp_datetime(self) -> datetime:
        """Convert timestamp to datetime"""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class OrderBookLevel:
    """Order book price level"""
    price: float
    size: float
    count: int = 1
    
    def total_value(self) -> float:
        """Calculate total value at this level"""
        return self.price * self.size


@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: int
    
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None
    
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None
    
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask - bid
        return None
    
    def spread_percentage(self) -> Optional[float]:
        """Calculate spread as percentage"""
        spread = self.spread()
        mid = self.mid_price()
        if spread and mid:
            return (spread / mid) * 100
        return None
    
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None


@dataclass
class Balance:
    """Account balance"""
    asset: str
    available: float
    locked: float
    total: float
    
    def utilization(self) -> float:
        """Calculate balance utilization"""
        if self.total > 0:
            return self.locked / self.total
        return 0.0


@dataclass
class Position:
    """Trading position"""
    symbol: str
    size: float
    entry_price: float
    mark_price: float
    pnl: float
    margin: float
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL"""
        return (self.mark_price - self.entry_price) * self.size
    
    def pnl_percentage(self) -> float:
        """Calculate PnL percentage"""
        if self.entry_price > 0:
            return ((self.mark_price - self.entry_price) / self.entry_price) * 100
        return 0.0


@dataclass
class NodeInfo:
    """Node information"""
    version: str
    network: str
    order_count: int
    trade_count: int
    timestamp: int
    block_height: Optional[int] = None
    syncing: bool = False
    uptime: Optional[int] = None
    
    def timestamp_datetime(self) -> datetime:
        """Convert timestamp to datetime"""
        return datetime.fromtimestamp(self.timestamp)