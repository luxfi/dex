"""Risk management."""

from decimal import Decimal
from typing import Dict
import threading

from lx_trading.config import RiskConfig
from lx_trading.types import OrderRequest, Side, TradingPair


class RiskError(Exception):
    """Risk limit exceeded."""
    pass


class RiskManager:
    """Risk manager for enforcing trading limits."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._positions: Dict[str, Decimal] = {}
        self._daily_pnl = Decimal(0)
        self._open_orders: Dict[str, int] = {}  # symbol -> count
        self._kill_switch = False
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @property
    def is_killed(self) -> bool:
        return self._kill_switch

    def kill(self) -> None:
        """Activate kill switch - stops all trading."""
        self._kill_switch = True

    def reset(self) -> None:
        """Deactivate kill switch."""
        self._kill_switch = False

    def validate_order(self, request: OrderRequest) -> None:
        """Validate order against risk limits. Raises RiskError if invalid."""
        if not self.config.enabled:
            return

        if self._kill_switch:
            raise RiskError("Kill switch is active")

        # Check order size
        if self.config.max_order_size > 0 and request.quantity > self.config.max_order_size:
            raise RiskError(
                f"Order size {request.quantity} exceeds max {self.config.max_order_size}"
            )

        # Check position limit
        pair = TradingPair.from_symbol(request.symbol)
        if pair:
            with self._lock:
                current = self._positions.get(pair.base, Decimal(0))

            new_position = (
                current + request.quantity
                if request.side == Side.BUY
                else current - request.quantity
            )

            # Asset-specific limit
            if pair.base in self.config.position_limits:
                limit = self.config.position_limits[pair.base]
                if abs(new_position) > limit:
                    raise RiskError(
                        f"Position limit exceeded for {pair.base}: "
                        f"{current} + {request.quantity} > {limit}"
                    )

            # Global position limit
            if self.config.max_position_size > 0 and abs(new_position) > self.config.max_position_size:
                raise RiskError(
                    f"Max position size exceeded: {abs(new_position)} > {self.config.max_position_size}"
                )

        # Check open orders count
        with self._lock:
            count = self._open_orders.get(request.symbol, 0)

        if count >= self.config.max_open_orders:
            raise RiskError(
                f"Max open orders ({self.config.max_open_orders}) reached for {request.symbol}"
            )

        # Check daily loss
        if self.config.max_daily_loss > 0:
            with self._lock:
                if self._daily_pnl < -self.config.max_daily_loss:
                    raise RiskError(
                        f"Daily loss limit exceeded: {abs(self._daily_pnl)} > {self.config.max_daily_loss}"
                    )

    def update_position(self, asset: str, quantity: Decimal, side: Side) -> None:
        """Update position after a trade."""
        with self._lock:
            current = self._positions.get(asset, Decimal(0))
            new_position = current + quantity if side == Side.BUY else current - quantity
            self._positions[asset] = new_position

    def update_pnl(self, pnl: Decimal) -> None:
        """Update daily PnL."""
        with self._lock:
            self._daily_pnl += pnl

            # Auto kill switch
            if (
                self.config.kill_switch_enabled
                and self.config.max_daily_loss > 0
                and self._daily_pnl < -self.config.max_daily_loss
            ):
                self._kill_switch = True

    def order_opened(self, symbol: str) -> None:
        """Increment open orders count."""
        with self._lock:
            self._open_orders[symbol] = self._open_orders.get(symbol, 0) + 1

    def order_closed(self, symbol: str) -> None:
        """Decrement open orders count."""
        with self._lock:
            if symbol in self._open_orders:
                self._open_orders[symbol] = max(0, self._open_orders[symbol] - 1)

    def position(self, asset: str) -> Decimal:
        """Get current position for an asset."""
        with self._lock:
            return self._positions.get(asset, Decimal(0))

    def positions(self) -> Dict[str, Decimal]:
        """Get all positions."""
        with self._lock:
            return self._positions.copy()

    @property
    def daily_pnl(self) -> Decimal:
        """Get daily PnL."""
        with self._lock:
            return self._daily_pnl

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL (call at start of trading day)."""
        with self._lock:
            self._daily_pnl = Decimal(0)

    def open_orders(self, symbol: str) -> int:
        """Get open orders count for a symbol."""
        with self._lock:
            return self._open_orders.get(symbol, 0)
