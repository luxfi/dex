"""
LX-First Arbitrage Strategy.

Key Insight: LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks).
By the time other venues update, LX has already moved.

This means:
1. LX DEX price is the "TRUE" price (most current)
2. Other venues are always STALE by comparison
3. Arbitrage = correcting stale venues to match LX
4. LX DEX is the ORACLE, not just another venue

Strategy:
1. Watch LX DEX prices (the reference)
2. Compare against "slow" venues (CEX, external DEX)
3. When slow venue diverges from LX, trade on SLOW venue
4. You're essentially front-running slow venues with LX information

Example:
- LX DEX BTC: $50,000 (current, true)
- Binance BTC: $49,990 (stale, 50ms behind)
- Uniswap BTC: $50,020 (stale, 12s behind)

Action:
- Buy on Binance at $49,990 (they haven't caught up yet)
- Sell on Uniswap at $50,020 (they haven't corrected yet)
- Net: $30 profit per BTC

Why LX wins: By the time Binance/Uniswap update, we've already executed.
"""

import time
from typing import Callable, Dict, List, Optional

from .types import LxFirstConfig, LxFirstOpportunity, LxPrice, VenuePrice


class LxFirstArbitrage:
    """LX-first arbitrage using LX DEX as the price oracle."""

    def __init__(self, config: LxFirstConfig):
        self.config = config
        self._lx_prices: Dict[str, LxPrice] = {}
        self._venue_prices: Dict[str, List[VenuePrice]] = {}
        self._callbacks: List[Callable[[LxFirstOpportunity], None]] = []
        self._running = False

    def update_lx_price(self, price: LxPrice) -> None:
        """Update the LX DEX price (the oracle)."""
        self._lx_prices[price.symbol] = price

        # Immediately check for opportunities against stale venues
        self._check_opportunities(price.symbol)

    def update_venue_price(self, price: VenuePrice) -> None:
        """Update a price from a 'slow' venue."""
        prices = self._venue_prices.get(price.symbol, [])

        # Update or append
        found = False
        for i, p in enumerate(prices):
            if p.venue == price.venue:
                prices[i] = price
                found = True
                break

        if not found:
            prices.append(price)

        self._venue_prices[price.symbol] = prices

    def on_opportunity(self, callback: Callable[[LxFirstOpportunity], None]) -> None:
        """Subscribe to opportunity events."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the arbitrage system."""
        self._running = True

    def stop(self) -> None:
        """Stop the arbitrage system."""
        self._running = False

    def _check_opportunities(self, symbol: str) -> None:
        """Check for opportunities against stale venues."""
        if not self._running:
            return

        lx_price = self._lx_prices.get(symbol)
        venue_prices = self._venue_prices.get(symbol)

        if not lx_price or not venue_prices:
            return

        now = int(time.time() * 1000)

        for vp in venue_prices:
            # Calculate how stale the venue is
            staleness = now - vp.timestamp
            if staleness > self.config.max_staleness_ms:
                continue  # Too stale, might have updated by now

            # Check for BUY opportunity (venue ask < LX mid)
            # The slow venue hasn't caught up to LX's higher price
            if vp.ask < lx_price.mid:
                divergence = lx_price.mid - vp.ask
                divergence_bps = (divergence / lx_price.mid) * 10000

                if divergence_bps >= self.config.min_divergence_bps:
                    opp = self._create_opportunity(
                        symbol, lx_price, vp, staleness,
                        "buy", divergence, divergence_bps
                    )
                    if opp.expected_profit >= self.config.min_profit:
                        self._emit_opportunity(opp)

            # Check for SELL opportunity (venue bid > LX mid)
            # The slow venue hasn't caught up to LX's lower price
            if vp.bid > lx_price.mid:
                divergence = vp.bid - lx_price.mid
                divergence_bps = (divergence / lx_price.mid) * 10000

                if divergence_bps >= self.config.min_divergence_bps:
                    opp = self._create_opportunity(
                        symbol, lx_price, vp, staleness,
                        "sell", divergence, divergence_bps
                    )
                    if opp.expected_profit >= self.config.min_profit:
                        self._emit_opportunity(opp)

    def _create_opportunity(
        self,
        symbol: str,
        lx_price: LxPrice,
        vp: VenuePrice,
        staleness: int,
        side: str,
        divergence,
        divergence_bps,
    ) -> LxFirstOpportunity:
        """Create an opportunity object."""
        now = int(time.time() * 1000)
        expected_profit = divergence * self.config.max_position_size
        confidence = self._calculate_confidence(staleness, divergence_bps)

        return LxFirstOpportunity(
            id=f"{symbol}-{vp.venue}-{side}-{now}",
            symbol=symbol,
            timestamp=now,
            lx_price=lx_price,
            stale_venue=vp.venue,
            stale_price=vp,
            staleness=staleness,
            side=side,
            divergence=divergence,
            divergence_bps=divergence_bps,
            expected_profit=expected_profit,
            max_size=self.config.max_position_size,
            confidence=confidence,
        )

    def _calculate_confidence(self, staleness: int, divergence_bps) -> float:
        """
        Calculate confidence score.

        Higher confidence when:
        1. Venue is more stale (hasn't had time to update)
        2. Divergence is larger (more room for profit)
        """
        staleness_score = max(0, 1.0 - staleness / 5000)  # 5s max
        divergence_score = min(1, float(divergence_bps) / 100)  # 100bps = 1.0

        return 0.5 * staleness_score + 0.5 * divergence_score

    def _emit_opportunity(self, opp: LxFirstOpportunity) -> None:
        """Emit an opportunity to all subscribers."""
        for callback in self._callbacks:
            try:
                callback(opp)
            except Exception as e:
                print(f"Error in opportunity callback: {e}")


"""
TRADING EXECUTION STRATEGY

When an LxFirstOpportunity is detected:

1. DO NOT trade on LX DEX (it's the reference, not the opportunity)

2. Trade on the STALE venue:
   - If Side="buy": Buy on stale venue (their ask is behind LX)
   - If Side="sell": Sell on stale venue (their bid is behind LX)

3. Settlement options:
   a) Hold position until venues converge (market neutral)
   b) Immediately hedge on LX DEX (lock in profit)
   c) Bridge and sell on another venue (more complex)

4. The key insight:
   - You're NOT arbitraging between two venues
   - You're front-running the slow venue with LX information
   - LX price is where the slow venue WILL BE, you just got there first

Example execution:

  LX DEX shows BTC = $50,000 (current, true price)
  Binance shows BTC = $49,950 (50ms stale)

  Action: BUY on Binance at $49,950
  Why: Binance WILL update to ~$50,000, we bought before they did
  Profit: ~$50 per BTC (0.1%)

  Optional hedge: SELL on LX DEX at $50,000 to lock in profit immediately
"""
