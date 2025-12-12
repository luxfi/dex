#!/usr/bin/env python3
"""
LX-First Arbitrage Bot Example

This bot uses the LX-First strategy where LX DEX prices are treated
as the "truth" (fastest venue with nanosecond updates, 200ms blocks).
Other venues are always stale by comparison.

Arbitrage = exploiting stale venues before they catch up to LX prices.

Cross-chain transport options:
- Warp: For Lux subnet communication (<500ms)
- Teleport: For EVM chain bridging (~30s)
- CEX API: Direct trading (instant)

NO SMART CONTRACTS - just coordinated trades through unified SDK.
"""

import asyncio
import os
import signal
import time
from decimal import Decimal
from typing import Optional

# LX Trading SDK imports
from lx_trading import LxDex
from lx_trading.arbitrage import (
    LxFirstArbitrage,
    LxFirstConfig,
    LxFirstOpportunity,
    LxPrice,
    VenuePrice,
    UnifiedArbitrage,
    UnifiedArbConfig,
    Scanner,
    ScannerConfig,
    CrossChainRouter,
    default_cross_chain_config,
)


class ArbitrageBot:
    """LX-First Arbitrage Bot."""

    def __init__(self):
        self.dex: Optional[LxDex] = None
        self.lx_first: Optional[LxFirstArbitrage] = None
        self.unified: Optional[UnifiedArbitrage] = None
        self.scanner: Optional[Scanner] = None
        self.router: Optional[CrossChainRouter] = None
        self.running = False
        self.total_opportunities = 0
        self.total_executions = 0
        self.total_pnl = Decimal("0")

    async def start(self):
        """Start the arbitrage bot."""
        print("=" * 60)
        print("LX-FIRST ARBITRAGE BOT")
        print("=" * 60)
        print()

        # Initialize DEX client
        self.dex = LxDex(
            endpoint=os.getenv("LX_DEX_ENDPOINT", "wss://dex.lux.network/ws"),
            api_key=os.getenv("LX_API_KEY"),
        )
        await self.dex.connect()
        print("[OK] Connected to LX DEX")

        # Initialize LX-First strategy
        lx_config = LxFirstConfig(
            max_staleness_ms=2000,
            min_divergence_bps=Decimal("10"),
            min_profit=Decimal("5"),
            max_position_size=Decimal("10000"),
            symbols=["BTC-USDC", "ETH-USDC", "LUX-USDC"],
            venue_latencies={
                "binance": 50,
                "mexc": 100,
                "okx": 80,
                "uniswap": 12000,
                "pancakeswap": 3000,
            },
        )
        self.lx_first = LxFirstArbitrage(lx_config)
        self.lx_first.on_opportunity(self._on_lx_first_opportunity)
        print("[OK] LX-First strategy initialized")

        # Initialize Unified Arbitrage
        unified_config = UnifiedArbConfig(
            min_spread_bps=Decimal("10"),
            min_profit=Decimal("5"),
            max_position_size=Decimal("10000"),
            max_total_exposure=Decimal("100000"),
            symbols=["BTC-USDC", "ETH-USDC", "LUX-USDC"],
            venue_priority=["lx_dex", "binance", "mexc", "lx_amm"],
            scan_interval_ms=100,
            execute_timeout_ms=5000,
        )
        self.unified = UnifiedArbitrage(self.dex, unified_config)
        self.unified.on_opportunity(self._on_unified_opportunity)
        print("[OK] Unified arbitrage initialized")

        # Initialize Scanner
        scanner_config = ScannerConfig(
            min_spread_bps=Decimal("10"),
            min_profit_usd=Decimal("10"),
            max_price_age_ms=5000,
            symbols=["BTC", "ETH", "LUX", "SOL", "AVAX"],
            chain_ids=["lux", "ethereum", "bsc", "arbitrum", "polygon"],
            scan_interval_ms=100,
        )
        self.scanner = Scanner(scanner_config)
        self.scanner.on_opportunity(self._on_scanner_opportunity)
        print("[OK] Scanner initialized")

        # Initialize Cross-Chain Router
        self.router = CrossChainRouter(default_cross_chain_config())
        print("[OK] Cross-chain router initialized")

        # Start all systems
        self.lx_first.start()
        await self.unified.start()
        await self.scanner.start()
        self.running = True

        print()
        print("=" * 60)
        print("BOT RUNNING - Press Ctrl+C to stop")
        print("=" * 60)
        print()
        print(f"Monitoring symbols: {lx_config.symbols}")
        print(f"Min divergence: {lx_config.min_divergence_bps} bps")
        print(f"Min profit: ${lx_config.min_profit}")
        print()

        # Start price feed simulation (in production, connect to real feeds)
        asyncio.create_task(self._simulate_price_feeds())

        # Start stats reporter
        asyncio.create_task(self._report_stats())

    async def stop(self):
        """Stop the arbitrage bot."""
        print("\nShutting down...")
        self.running = False

        self.lx_first.stop()
        await self.unified.stop()
        await self.scanner.stop()

        self._print_final_stats()

    def _on_lx_first_opportunity(self, opp: LxFirstOpportunity):
        """Handle LX-First opportunity."""
        self.total_opportunities += 1

        print()
        print("=" * 50)
        print("LX-FIRST OPPORTUNITY DETECTED")
        print("=" * 50)
        print(f"Symbol:          {opp.symbol}")
        print(f"LX Price:        ${opp.lx_price.mid}")
        print(f"Stale Venue:     {opp.stale_venue}")
        print(f"Stale Bid/Ask:   ${opp.stale_price.bid} / ${opp.stale_price.ask}")
        print(f"Staleness:       {opp.staleness}ms")
        print(f"Side:            {opp.side.upper()}")
        print(f"Divergence:      {opp.divergence_bps} bps")
        print(f"Expected Profit: ${opp.expected_profit}")
        print(f"Confidence:      {opp.confidence * 100:.1f}%")
        print("=" * 50)

        # Execute if confidence is high enough
        if opp.confidence > 0.8:
            asyncio.create_task(self._execute_lx_first(opp))

    def _on_unified_opportunity(self, opp):
        """Handle unified arbitrage opportunity."""
        print(f"[UNIFIED] {opp.symbol}: Buy {opp.buy_venue} @ ${opp.buy_price} -> "
              f"Sell {opp.sell_venue} @ ${opp.sell_price} | "
              f"Net: ${opp.net_profit}")

    def _on_scanner_opportunity(self, opp):
        """Handle scanner opportunity."""
        print(f"[SCANNER] {opp.type.value}: {opp.buy_source.venue} -> "
              f"{opp.sell_source.venue} | Spread: {opp.spread_bps} bps | "
              f"Net PnL: ${opp.net_pnl}")

    async def _execute_lx_first(self, opp: LxFirstOpportunity):
        """Execute an LX-First arbitrage opportunity."""
        try:
            print(f"\n[EXECUTING] {opp.id}...")

            # Determine cross-chain transport
            buy_chain = self.router.venue_to_chain(opp.stale_venue)
            sell_chain = "lux_mainnet"  # Always hedge on LX
            transport = self.router.determine_transport(buy_chain, sell_chain)
            latency = self.router.estimate_latency(buy_chain, sell_chain)

            print(f"  Transport: {transport.value}")
            print(f"  Est. Latency: {latency}ms")

            if opp.side == "buy":
                # Buy on stale venue (it's cheap)
                print(f"  Buying on {opp.stale_venue}...")
                # In production: place actual order
                # order = await cex_client.place_order(...)

                # Hedge on LX DEX
                print("  Hedging on LX DEX...")
                # In production: place hedge order
                # hedge = await self.dex.spot.sell(...)
            else:
                # Sell on stale venue (it's expensive)
                print(f"  Selling on {opp.stale_venue}...")
                # In production: place actual order

                # Hedge on LX DEX
                print("  Hedging on LX DEX...")
                # In production: place hedge order

            # Simulate successful execution
            self.total_executions += 1
            profit = opp.expected_profit * Decimal("0.8")  # Simulate slippage
            self.total_pnl += profit

            print(f"[SUCCESS] Executed {opp.id} | Profit: ${profit:.2f}")

        except Exception as e:
            print(f"[FAILED] {opp.id}: {e}")

    async def _simulate_price_feeds(self):
        """Simulate price feeds for testing."""
        import random

        base_prices = {
            "BTC-USDC": Decimal("50000"),
            "ETH-USDC": Decimal("3000"),
            "LUX-USDC": Decimal("25"),
        }

        while self.running:
            for symbol, base in base_prices.items():
                # Simulate LX DEX price (the oracle)
                lx_mid = base * Decimal(str(1 + random.uniform(-0.001, 0.001)))
                self.lx_first.update_lx_price(LxPrice(
                    symbol=symbol,
                    bid=lx_mid * Decimal("0.9999"),
                    ask=lx_mid * Decimal("1.0001"),
                    mid=lx_mid,
                    timestamp=int(time.time() * 1000),
                    block_num=random.randint(1000000, 2000000),
                ))

                # Simulate stale CEX prices
                for venue, latency in [("binance", 50), ("mexc", 100)]:
                    # Add some divergence occasionally
                    divergence = Decimal(str(random.uniform(-0.002, 0.002)))
                    venue_mid = base * (1 + divergence)

                    self.lx_first.update_venue_price(VenuePrice(
                        venue=venue,
                        symbol=symbol,
                        bid=venue_mid * Decimal("0.9998"),
                        ask=venue_mid * Decimal("1.0002"),
                        timestamp=int(time.time() * 1000) - latency,
                        latency=latency,
                    ))

            await asyncio.sleep(0.1)  # 10 updates per second

    async def _report_stats(self):
        """Report statistics periodically."""
        while self.running:
            await asyncio.sleep(30)
            print()
            print("-" * 40)
            print("STATS")
            print(f"  Opportunities: {self.total_opportunities}")
            print(f"  Executions:    {self.total_executions}")
            print(f"  Total PnL:     ${self.total_pnl:.2f}")
            if self.total_executions > 0:
                avg_pnl = self.total_pnl / self.total_executions
                print(f"  Avg PnL:       ${avg_pnl:.2f}")
            print("-" * 40)

    def _print_final_stats(self):
        """Print final statistics."""
        print()
        print("=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)
        print(f"Total Opportunities: {self.total_opportunities}")
        print(f"Total Executions:    {self.total_executions}")
        print(f"Total PnL:           ${self.total_pnl:.2f}")
        if self.total_executions > 0:
            win_rate = (self.total_executions / self.total_opportunities) * 100
            avg_pnl = self.total_pnl / self.total_executions
            print(f"Execution Rate:      {win_rate:.1f}%")
            print(f"Avg PnL per Trade:   ${avg_pnl:.2f}")
        print("=" * 50)


async def main():
    """Main entry point."""
    bot = ArbitrageBot()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()

        # Keep running until stopped
        while bot.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
