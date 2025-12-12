"""
LX Trading SDK - Arbitrage Module.

Omnichain arbitrage using LX DEX as the price oracle.
LX DEX is the fastest venue (nanosecond updates, 200ms blocks),
making it the "truth" while other venues are always stale.
"""

from .types import (
    ArbType,
    ChainType,
    CrossChainTransport,
    ArbitrageOpportunity,
    LxFirstOpportunity,
    UnifiedOpportunity,
    UnifiedExecution,
    UnifiedArbStats,
    UnifiedArbConfig,
    LxFirstConfig,
    ScannerConfig,
    CrossChainConfig,
    CrossChainInfo,
    PriceSource,
    LxPrice,
    VenuePrice,
    Route,
    default_unified_arb_config,
    default_lx_first_config,
    default_scanner_config,
    default_cross_chain_config,
)
from .scanner import Scanner
from .lx_first import LxFirstArbitrage
from .unified import UnifiedArbitrage
from .cross_chain import CrossChainRouter

__all__ = [
    # Types
    "ArbType",
    "ChainType",
    "CrossChainTransport",
    "ArbitrageOpportunity",
    "LxFirstOpportunity",
    "UnifiedOpportunity",
    "UnifiedExecution",
    "UnifiedArbStats",
    "UnifiedArbConfig",
    "LxFirstConfig",
    "ScannerConfig",
    "CrossChainConfig",
    "CrossChainInfo",
    "PriceSource",
    "LxPrice",
    "VenuePrice",
    "Route",
    # Defaults
    "default_unified_arb_config",
    "default_lx_first_config",
    "default_scanner_config",
    "default_cross_chain_config",
    # Classes
    "Scanner",
    "LxFirstArbitrage",
    "UnifiedArbitrage",
    "CrossChainRouter",
]
