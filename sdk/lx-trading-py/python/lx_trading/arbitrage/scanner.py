"""
Arbitrage scanner for detecting cross-venue opportunities.

Continuously scans for arbitrage opportunities across all venues.
Supports simple, triangular, and CEX-DEX arbitrage detection.
"""

import asyncio
import time
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Set

from .types import (
    ArbitrageOpportunity,
    ArbType,
    CrossChainInfo,
    PriceSource,
    Route,
    ScannerConfig,
)

CEX_VENUES: Set[str] = {
    "binance", "coinbase", "kraken", "okx", "bybit",
    "kucoin", "mexc", "gate", "huobi",
}


class Scanner:
    """Arbitrage scanner for detecting cross-venue opportunities."""

    def __init__(self, config: ScannerConfig):
        self.config = config
        self._prices: Dict[str, List[PriceSource]] = {}
        self._chains: Dict[str, CrossChainInfo] = {}
        self._callbacks: List[Callable[[ArbitrageOpportunity], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_chain(self, info: CrossChainInfo) -> None:
        """Add a chain configuration."""
        self._chains[info.chain_id] = info

    def update_price(self, source: PriceSource) -> None:
        """Update a price feed."""
        sources = self._prices.get(source.symbol, [])

        # Update existing or append new
        found = False
        for i, s in enumerate(sources):
            if s.chain_id == source.chain_id and s.venue == source.venue:
                sources[i] = source
                found = True
                break

        if not found:
            sources.append(source)

        self._prices[source.symbol] = sources

    def on_opportunity(self, callback: Callable[[ArbitrageOpportunity], None]) -> None:
        """Subscribe to opportunity events."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start scanning for opportunities."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())

    async def stop(self) -> None:
        """Stop scanning."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _scan_loop(self) -> None:
        """Main scan loop."""
        while self._running:
            await self._scan()
            await asyncio.sleep(self.config.scan_interval_ms / 1000)

    async def _scan(self) -> None:
        """Perform a single scan."""
        symbols = list(self._prices.keys())

        # Process in parallel
        tasks = [self._find_opportunities(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                for opp in result:
                    self._emit_opportunity(opp)

    async def _find_opportunities(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Find all opportunities for a symbol."""
        sources = self._prices.get(symbol)
        if not sources or len(sources) < 2:
            return []

        now = int(time.time() * 1000)

        # Filter stale prices
        valid_sources = [
            s for s in sources
            if now - s.timestamp < self.config.max_price_age_ms
        ]

        if len(valid_sources) < 2:
            return []

        opportunities: List[ArbitrageOpportunity] = []

        # Simple arbitrage
        opportunities.extend(self._find_simple_arb(symbol, valid_sources))

        # CEX-DEX arbitrage
        opportunities.extend(self._find_cex_dex_arb(symbol, valid_sources))

        return opportunities

    def _find_simple_arb(
        self, symbol: str, sources: List[PriceSource]
    ) -> List[ArbitrageOpportunity]:
        """Find simple buy-low-sell-high opportunities."""
        opportunities: List[ArbitrageOpportunity] = []

        # Sort by ask (lowest first for buying)
        buy_order = sorted(sources, key=lambda s: s.ask)

        # Sort by bid (highest first for selling)
        sell_order = sorted(sources, key=lambda s: s.bid, reverse=True)

        for buy_src in buy_order:
            for sell_src in sell_order:
                # Skip same venue/chain
                if buy_src.chain_id == sell_src.chain_id and buy_src.venue == sell_src.venue:
                    continue

                # Calculate spread
                spread = sell_src.bid - buy_src.ask
                if spread <= 0:
                    continue

                spread_bps = (spread / buy_src.ask) * 10000
                if spread_bps < self.config.min_spread_bps:
                    continue

                # Calculate costs
                gas_cost, bridge_cost = self._calculate_costs(
                    buy_src.chain_id, sell_src.chain_id
                )

                # Maximum size limited by liquidity
                max_size = min(buy_src.liquidity, sell_src.liquidity)

                # Calculate PnL
                gross_pnl = spread * max_size
                net_pnl = gross_pnl - gas_cost - bridge_cost

                if net_pnl < self.config.min_profit_usd:
                    continue

                # Calculate confidence
                confidence = self._calculate_confidence(buy_src, sell_src)

                now = int(time.time() * 1000)
                opp = ArbitrageOpportunity(
                    id=f"simple-{symbol}-{buy_src.venue}-{sell_src.venue}-{now}",
                    type=ArbType.SIMPLE,
                    buy_source=buy_src,
                    sell_source=sell_src,
                    spread_bps=spread_bps,
                    estimated_pnl=gross_pnl,
                    max_size=max_size,
                    gas_cost_usd=gas_cost,
                    bridge_cost_usd=bridge_cost,
                    net_pnl=net_pnl,
                    confidence=confidence,
                    expires_at=now + 5000,
                    routes=[
                        Route(
                            chain_id=buy_src.chain_id,
                            venue=buy_src.venue,
                            action="buy",
                            token_in="USDC",
                            token_out=symbol,
                            amount_in=max_size * buy_src.ask,
                            expected_out=max_size,
                            min_amount_out=max_size * Decimal("0.99"),
                        ),
                        Route(
                            chain_id=sell_src.chain_id,
                            venue=sell_src.venue,
                            action="sell",
                            token_in=symbol,
                            token_out="USDC",
                            amount_in=max_size,
                            expected_out=max_size * sell_src.bid,
                            min_amount_out=max_size * sell_src.bid * Decimal("0.99"),
                        ),
                    ],
                )
                opportunities.append(opp)

        return opportunities

    def _find_cex_dex_arb(
        self, symbol: str, sources: List[PriceSource]
    ) -> List[ArbitrageOpportunity]:
        """Find CEX-DEX arbitrage opportunities."""
        opportunities: List[ArbitrageOpportunity] = []

        # Separate CEX and DEX sources
        cex_sources = [s for s in sources if s.venue in CEX_VENUES]
        dex_sources = [s for s in sources if s.venue not in CEX_VENUES]

        # Find CEX buy -> DEX sell opportunities
        for cex in cex_sources:
            for dex in dex_sources:
                spread = dex.bid - cex.ask
                if spread <= 0:
                    continue

                spread_bps = (spread / cex.ask) * 10000
                if spread_bps < self.config.min_spread_bps:
                    continue

                max_size = min(cex.liquidity, dex.liquidity)
                gross_pnl = spread * max_size

                now = int(time.time() * 1000)
                opp = ArbitrageOpportunity(
                    id=f"cexdex-{symbol}-{cex.venue}-{dex.venue}-{now}",
                    type=ArbType.CEX_DEX,
                    buy_source=cex,
                    sell_source=dex,
                    spread_bps=spread_bps,
                    estimated_pnl=gross_pnl,
                    max_size=max_size,
                    gas_cost_usd=Decimal("0.5"),
                    bridge_cost_usd=Decimal(0),
                    net_pnl=gross_pnl - Decimal("0.5"),
                    confidence=0.7,
                    expires_at=now + 3000,
                    routes=[],
                )
                opportunities.append(opp)

        # Find DEX buy -> CEX sell opportunities
        for dex in dex_sources:
            for cex in cex_sources:
                spread = cex.bid - dex.ask
                if spread <= 0:
                    continue

                spread_bps = (spread / dex.ask) * 10000
                if spread_bps < self.config.min_spread_bps:
                    continue

                max_size = min(dex.liquidity, cex.liquidity)
                gross_pnl = spread * max_size

                now = int(time.time() * 1000)
                opp = ArbitrageOpportunity(
                    id=f"cexdex-{symbol}-{dex.venue}-{cex.venue}-{now}",
                    type=ArbType.CEX_DEX,
                    buy_source=dex,
                    sell_source=cex,
                    spread_bps=spread_bps,
                    estimated_pnl=gross_pnl,
                    max_size=max_size,
                    gas_cost_usd=Decimal("0.5"),
                    bridge_cost_usd=Decimal(0),
                    net_pnl=gross_pnl - Decimal("0.5"),
                    confidence=0.7,
                    expires_at=now + 3000,
                    routes=[],
                )
                opportunities.append(opp)

        return opportunities

    def _calculate_costs(
        self, source_chain: str, dest_chain: str
    ) -> tuple[Decimal, Decimal]:
        """Calculate gas and bridge costs between chains."""
        src_config = self._chains.get(source_chain)
        dst_config = self._chains.get(dest_chain)

        # Estimate gas cost
        gas_cost = Decimal("0.1")  # Default
        if src_config:
            gas_cost = Decimal("0.05")

        # Bridge cost if crossing chains
        bridge_cost = Decimal(0)
        if source_chain != dest_chain and src_config and dst_config:
            if src_config.warp_supported and dst_config.warp_supported:
                bridge_cost = Decimal("0.01")  # Warp is nearly free
            elif src_config.teleport_supported and dst_config.teleport_supported:
                bridge_cost = Decimal("0.10")  # Teleport for EVM chains
            else:
                bridge_cost = Decimal("1.0")  # Generic bridge

        return gas_cost, bridge_cost

    def _calculate_confidence(
        self, buy: PriceSource, sell: PriceSource
    ) -> float:
        """Calculate confidence score for an opportunity."""
        now = int(time.time() * 1000)

        # Freshness score
        buy_age = (now - buy.timestamp) / 1000
        sell_age = (now - sell.timestamp) / 1000
        max_age = self.config.max_price_age_ms / 1000
        freshness_score = max(0, 1.0 - (buy_age + sell_age) / (2 * max_age))

        # Liquidity score
        min_liq = min(buy.liquidity, sell.liquidity)
        if min_liq > 100000:
            liquidity_score = 1.0
        elif min_liq > 10000:
            liquidity_score = 0.8
        else:
            liquidity_score = 0.5

        # Latency score
        avg_latency = (buy.latency + sell.latency) / 2
        latency_score = max(0, 1.0 - avg_latency / 1000)

        # Weighted average
        return 0.4 * freshness_score + 0.4 * liquidity_score + 0.2 * latency_score

    def _emit_opportunity(self, opp: ArbitrageOpportunity) -> None:
        """Emit an opportunity to all subscribers."""
        for callback in self._callbacks:
            try:
                callback(opp)
            except Exception as e:
                print(f"Error in opportunity callback: {e}")
