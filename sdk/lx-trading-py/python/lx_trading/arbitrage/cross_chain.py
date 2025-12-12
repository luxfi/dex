"""
Cross-Chain Arbitrage Transports.

1. WARP (Lux Native)
   - Only works WITHIN Lux ecosystem (between subnets)
   - Sub-second message delivery
   - Use for: LX DEX <-> LX AMM <-> Other Lux subnets
   - Cannot reach external chains

2. TELEPORT (EVM Bridge)
   - Works with ANY EVM-compatible chain
   - Lux <-> Ethereum, BSC, Arbitrum, Polygon, etc.
   - ~30 second finality (depends on source chain)
   - Uses validator attestations

3. CEX API
   - No bridging needed - just API calls
   - Sub-second execution
   - Settlement via withdraw/deposit (slow but doesn't block arb)

4. FOR OMNICHAIN ARBITRAGE:
   - Lux internal: Warp (instant)
   - External EVM: Teleport (~30s)
   - CEX: Direct API (instant trade, later settle)
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Protocol

from .types import (
    ChainType,
    CrossChainConfig,
    CrossChainTransport,
    UnifiedOpportunity,
)


class WarpClient(Protocol):
    """Warp client interface for Lux-native messaging."""

    async def send_message(self, dest_subnet: str, payload: bytes) -> str:
        """Send a Warp message to another Lux subnet."""
        ...

    async def receive_message(self, message_id: str) -> bytes:
        """Receive a Warp message."""
        ...

    def get_blockchain_id(self) -> str:
        """Get this subnet's ID."""
        ...


class TeleportClient(Protocol):
    """Teleport client interface for EVM bridging."""

    async def bridge(self, dest_chain: str, token: str, amount: Decimal) -> str:
        """Bridge assets to another EVM chain."""
        ...

    async def get_bridge_status(self, tx_id: str) -> "BridgeStatus":
        """Get bridge transaction status."""
        ...

    async def estimate_bridge_fee(
        self, dest_chain: str, token: str, amount: Decimal
    ) -> Decimal:
        """Estimate bridge fee."""
        ...


@dataclass
class BridgeStatus:
    """Bridge transaction status."""

    tx_id: str
    status: str  # pending, confirming, completed, failed
    source_chain: str
    dest_chain: str
    amount: Decimal
    fee: Decimal
    source_tx: str
    dest_tx: Optional[str] = None
    timestamp: int = 0


@dataclass
class EnhancedOpportunity:
    """Opportunity with routing information."""

    base: UnifiedOpportunity
    transport: CrossChainTransport
    estimated_latency: int
    bridge_cost: Decimal
    adjusted_net_profit: Decimal


class CrossChainRouter:
    """Cross-chain router for determining optimal transport."""

    def __init__(self, config: CrossChainConfig):
        self.config = config
        self._warp: Optional[WarpClient] = None
        self._teleport: Optional[TeleportClient] = None

    def set_warp_client(self, client: WarpClient) -> None:
        """Set the Warp client."""
        self._warp = client

    def set_teleport_client(self, client: TeleportClient) -> None:
        """Set the Teleport client."""
        self._teleport = client

    @property
    def warp(self) -> Optional[WarpClient]:
        """Get the Warp client."""
        return self._warp

    @property
    def teleport(self) -> Optional[TeleportClient]:
        """Get the Teleport client."""
        return self._teleport

    def determine_transport(
        self, source_chain: str, dest_chain: str
    ) -> CrossChainTransport:
        """Determine the best transport between two chains."""
        src = self.config.chains.get(source_chain)
        dst = self.config.chains.get(dest_chain)

        # Same chain = direct
        if source_chain == dest_chain:
            return CrossChainTransport.DIRECT

        # CEX = API
        if src and src.chain_type == ChainType.CEX:
            return CrossChainTransport.CEX_API
        if dst and dst.chain_type == ChainType.CEX:
            return CrossChainTransport.CEX_API

        # Both Lux subnets = Warp (fastest)
        if (
            src
            and dst
            and src.chain_type == ChainType.LUX_SUBNET
            and dst.chain_type == ChainType.LUX_SUBNET
        ):
            if src.warp_supported and dst.warp_supported and self.config.warp_enabled:
                return CrossChainTransport.WARP

        # Both EVM or mixed = Teleport
        if (
            src
            and dst
            and src.teleport_supported
            and dst.teleport_supported
            and self.config.teleport_enabled
        ):
            return CrossChainTransport.TELEPORT

        # No viable transport - return DIRECT as fallback
        return CrossChainTransport.DIRECT

    def estimate_latency(self, source_chain: str, dest_chain: str) -> int:
        """Estimate latency for cross-chain message (ms)."""
        transport = self.determine_transport(source_chain, dest_chain)

        if transport == CrossChainTransport.DIRECT:
            return 0
        elif transport == CrossChainTransport.WARP:
            return 500  # Sub-second
        elif transport == CrossChainTransport.CEX_API:
            return 100  # API call
        elif transport == CrossChainTransport.TELEPORT:
            src = self.config.chains.get(source_chain)
            return (src.finality_ms if src else 0) + 10000  # Finality + processing
        else:
            return 3600000  # Unknown/unsupported (1 hour)

    async def estimate_cost(
        self,
        source_chain: str,
        dest_chain: str,
        token: str,
        amount: Decimal,
    ) -> Decimal:
        """Estimate cost for cross-chain transfer."""
        transport = self.determine_transport(source_chain, dest_chain)

        if transport == CrossChainTransport.DIRECT:
            return Decimal(0)
        elif transport == CrossChainTransport.WARP:
            return Decimal("0.001")  # Nearly free
        elif transport == CrossChainTransport.CEX_API:
            return Decimal(0)  # No bridge cost
        elif transport == CrossChainTransport.TELEPORT:
            if self._teleport:
                return await self._teleport.estimate_bridge_fee(dest_chain, token, amount)
            return Decimal("1.0")  # Estimate $1
        else:
            return Decimal(0)

    def venue_to_chain(self, venue: str) -> str:
        """Get chain ID from venue name."""
        for chain_id, info in self.config.chains.items():
            if venue in info.venues:
                return chain_id
        return venue  # Fallback to venue name

    async def enhance_opportunity(
        self, opp: UnifiedOpportunity
    ) -> EnhancedOpportunity:
        """Enhance an opportunity with routing information."""
        buy_chain = self.venue_to_chain(opp.buy_venue)
        sell_chain = self.venue_to_chain(opp.sell_venue)

        transport = self.determine_transport(buy_chain, sell_chain)
        estimated_latency = self.estimate_latency(buy_chain, sell_chain)
        bridge_cost = await self.estimate_cost(
            buy_chain, sell_chain, opp.symbol, opp.max_size
        )

        return EnhancedOpportunity(
            base=opp,
            transport=transport,
            estimated_latency=estimated_latency,
            bridge_cost=bridge_cost,
            adjusted_net_profit=opp.net_profit - bridge_cost,
        )
