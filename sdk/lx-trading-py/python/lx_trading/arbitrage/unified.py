"""
Unified Liquidity Arbitrage.

Since LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks),
it becomes the price ORACLE. Other venues are always stale by comparison.

Architecture:
1. LX DEX prices are the TRUTH (most current)
2. Other venues (CEX, external DEX) are STALE
3. Arbitrage = exploiting stale venues before they catch up
4. LX always wins because it sees/moves prices first

NO SMART CONTRACTS - just coordinated trades through unified SDK.
"""

import asyncio
import time
from collections import deque
from decimal import Decimal
from typing import Callable, Deque, List, Optional, Protocol

from .types import (
    UnifiedArbConfig,
    UnifiedArbStats,
    UnifiedExecution,
    UnifiedOpportunity,
)


class AggregatedLevel:
    """Aggregated orderbook level."""

    def __init__(self, price: Decimal, quantity: Decimal, venue: str, timestamp: int):
        self.price = price
        self.quantity = quantity
        self.venue = venue
        self.timestamp = timestamp


class AggregatedBook:
    """Aggregated orderbook."""

    def __init__(self, symbol: str, bids: List[AggregatedLevel], asks: List[AggregatedLevel]):
        self.symbol = symbol
        self.bids = bids
        self.asks = asks


class TradingClient(Protocol):
    """Trading client interface for arbitrage."""

    async def aggregated_orderbook(self, symbol: str) -> AggregatedBook:
        """Get aggregated orderbook from all venues."""
        ...

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
        venue: str,
    ) -> "OrderResult":
        """Place an order on a specific venue."""
        ...


class OrderResult:
    """Result of an order placement."""

    def __init__(
        self,
        order_id: str,
        filled_quantity: Decimal,
        average_price: Optional[Decimal],
        fees: Decimal,
    ):
        self.order_id = order_id
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.fees = fees


class UnifiedArbitrage:
    """Unified arbitrage across all SDK-connected venues."""

    def __init__(self, client: TradingClient, config: UnifiedArbConfig):
        self.client = client
        self.config = config
        self._total_pnl = Decimal(0)
        self._executions: List[UnifiedExecution] = []
        self._callbacks: List[Callable[[UnifiedOpportunity], None]] = []
        self._opportunity_queue: Deque[UnifiedOpportunity] = deque(maxlen=1000)
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._execute_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the arbitrage system."""
        if self._running:
            return
        self._running = True

        self._scan_task = asyncio.create_task(self._scan_loop())
        self._execute_task = asyncio.create_task(self._execute_loop())

    async def stop(self) -> None:
        """Stop the arbitrage system."""
        self._running = False

        for task in [self._scan_task, self._execute_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._scan_task = None
        self._execute_task = None

    def on_opportunity(self, callback: Callable[[UnifiedOpportunity], None]) -> None:
        """Subscribe to opportunity events."""
        self._callbacks.append(callback)

    def get_stats(self) -> UnifiedArbStats:
        """Get arbitrage statistics."""
        successful = sum(
            1 for e in self._executions
            if e.status == "completed" and e.actual_profit > 0
        )

        win_rate = successful / len(self._executions) if self._executions else 0

        return UnifiedArbStats(
            total_executions=len(self._executions),
            successful_executions=successful,
            total_pnl=self._total_pnl,
            win_rate=win_rate,
        )

    async def _scan_loop(self) -> None:
        """Scan loop."""
        while self._running:
            try:
                await self._scan()
            except Exception as e:
                print(f"Scan error: {e}")
            await asyncio.sleep(self.config.scan_interval_ms / 1000)

    async def _scan(self) -> None:
        """Scan for opportunities."""
        for symbol in self.config.symbols:
            opp = await self._find_opportunity(symbol)
            if opp and opp.net_profit > self.config.min_profit:
                self._opportunity_queue.append(opp)

                # Emit to callbacks
                for callback in self._callbacks:
                    try:
                        callback(opp)
                    except Exception as e:
                        print(f"Error in opportunity callback: {e}")

    async def _find_opportunity(self, symbol: str) -> Optional[UnifiedOpportunity]:
        """Find arbitrage opportunity for a symbol."""
        try:
            book = await self.client.aggregated_orderbook(symbol)

            if not book.bids or not book.asks:
                return None

            best_bid = book.bids[0]
            best_ask = book.asks[0]

            # Cross-venue arbitrage: bid on one venue > ask on another
            if best_bid.price <= best_ask.price:
                return None

            spread = best_bid.price - best_ask.price
            spread_bps = (spread / best_ask.price) * 10000

            if spread_bps < self.config.min_spread_bps:
                return None

            max_size = min(best_bid.quantity, best_ask.quantity, self.config.max_position_size)

            gross_profit = spread * max_size
            total_fees = best_ask.price * max_size * Decimal("0.002")  # ~0.2% total fees
            net_profit = gross_profit - total_fees

            now = int(time.time() * 1000)

            return UnifiedOpportunity(
                id=f"arb-{symbol}-{now}",
                symbol=symbol,
                timestamp=now,
                expires_at=now + 5000,
                buy_venue=best_ask.venue,
                buy_price=best_ask.price,
                buy_size=best_ask.quantity,
                sell_venue=best_bid.venue,
                sell_price=best_bid.price,
                sell_size=best_bid.quantity,
                spread=spread,
                spread_bps=spread_bps,
                max_size=max_size,
                gross_profit=gross_profit,
                est_fees=total_fees,
                net_profit=net_profit,
                confidence=0.8,
                latency=now - best_ask.timestamp,
            )
        except Exception:
            return None

    async def _execute_loop(self) -> None:
        """Execute loop - process opportunities from queue."""
        while self._running:
            if self._opportunity_queue:
                opp = self._opportunity_queue.popleft()
                await self._execute(opp)
            else:
                await asyncio.sleep(0.01)

    async def _execute(self, opp: UnifiedOpportunity) -> None:
        """Execute an arbitrage opportunity."""
        now = int(time.time() * 1000)
        if now > opp.expires_at:
            return

        exec_result = UnifiedExecution(
            id=opp.id,
            opportunity=opp,
            start_time=now,
            end_time=0,
            status="executing",
        )

        try:
            # Execute both legs simultaneously
            buy_task = self.client.place_order(
                opp.symbol,
                "buy",
                "limit",
                opp.max_size,
                opp.buy_price,
                opp.buy_venue,
            )
            sell_task = self.client.place_order(
                opp.symbol,
                "sell",
                "limit",
                opp.max_size,
                opp.sell_price,
                opp.sell_venue,
            )

            buy_result, sell_result = await asyncio.gather(buy_task, sell_task)

            exec_result.end_time = int(time.time() * 1000)
            exec_result.buy_order_id = buy_result.order_id
            exec_result.sell_order_id = sell_result.order_id

            # Calculate actual profit
            if buy_result.average_price and sell_result.average_price:
                buy_value = buy_result.average_price * buy_result.filled_quantity
                sell_value = sell_result.average_price * sell_result.filled_quantity
                exec_result.actual_profit = sell_value - buy_value
                exec_result.fees = buy_result.fees + sell_result.fees
                exec_result.actual_profit -= exec_result.fees

            exec_result.status = "completed"

        except Exception as e:
            exec_result.end_time = int(time.time() * 1000)
            exec_result.status = "failed"
            exec_result.error = e

        # Update stats
        self._total_pnl += exec_result.actual_profit
        self._executions.append(exec_result)
