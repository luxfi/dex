"""
Arbitrage types for LX Trading SDK.

LX-FIRST ARBITRAGE STRATEGY:
- LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks)
- By the time other venues update, LX has already moved
- LX DEX price is the "TRUTH" (most current)
- Other venues are always STALE by comparison
- Arbitrage = correcting stale venues to match LX
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional


class CrossChainTransport(str, Enum):
    """Cross-chain transport protocol."""
    WARP = "warp"  # Lux native - between subnets only
    TELEPORT = "teleport"  # EVM bridge for external chains
    DIRECT = "direct"  # Same chain, no bridge needed
    CEX_API = "cex_api"  # CEX API calls


class ChainType(str, Enum):
    """Type of blockchain."""
    LUX_SUBNET = "lux_subnet"
    EVM = "evm"
    CEX = "cex"


class ArbType(str, Enum):
    """Type of arbitrage."""
    SIMPLE = "simple"  # Buy A, sell B
    TRIANGULAR = "triangular"  # A->B->C->A
    MULTI_HOP = "multi_hop"  # Complex routes
    CEX_DEX = "cex_dex"  # CEX<->DEX arb
    FLASH_SWAP = "flash_swap"  # DEX flash swap


@dataclass
class PriceSource:
    """Price feed from a specific venue/chain."""
    chain_id: str
    venue: str
    symbol: str
    bid: Decimal
    ask: Decimal
    liquidity: Decimal
    timestamp: int  # Unix timestamp ms
    latency: int  # milliseconds


@dataclass
class LxPrice:
    """LX DEX price - the reference/oracle."""
    symbol: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    timestamp: int
    block_num: int


@dataclass
class VenuePrice:
    """Price from a 'slow' venue."""
    venue: str
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: int
    latency: int  # How far behind LX this venue typically is (ms)
    stale: bool = False  # Is this price stale relative to LX?


@dataclass
class Route:
    """Single leg of an arbitrage."""
    chain_id: str
    venue: str
    action: str  # "buy" or "sell"
    token_in: str
    token_out: str
    amount_in: Decimal
    expected_out: Decimal
    min_amount_out: Decimal
    swap_data: Optional[bytes] = None


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    id: str
    type: ArbType
    routes: List[Route]
    buy_source: PriceSource
    sell_source: PriceSource
    spread_bps: Decimal  # Spread in basis points
    estimated_pnl: Decimal
    max_size: Decimal  # Limited by liquidity
    gas_cost_usd: Decimal
    bridge_cost_usd: Decimal
    net_pnl: Decimal
    confidence: float  # 0-1, based on price freshness and liquidity
    expires_at: int


@dataclass
class LxFirstOpportunity:
    """LX-first arbitrage opportunity."""
    id: str
    symbol: str
    timestamp: int
    lx_price: LxPrice
    stale_venue: str
    stale_price: VenuePrice
    staleness: int  # milliseconds
    side: str  # "buy" or "sell"
    divergence: Decimal
    divergence_bps: Decimal
    expected_profit: Decimal
    max_size: Decimal
    confidence: float


@dataclass
class UnifiedOpportunity:
    """Unified arbitrage opportunity across venues."""
    id: str
    symbol: str
    timestamp: int
    expires_at: int
    buy_venue: str
    buy_price: Decimal
    buy_size: Decimal
    sell_venue: str
    sell_price: Decimal
    sell_size: Decimal
    spread: Decimal
    spread_bps: Decimal
    max_size: Decimal
    gross_profit: Decimal
    est_fees: Decimal
    net_profit: Decimal
    confidence: float
    latency: int


@dataclass
class UnifiedExecution:
    """Executed arbitrage."""
    id: str
    opportunity: UnifiedOpportunity
    start_time: int
    end_time: int
    status: str  # "executing", "completed", "failed"
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    actual_profit: Decimal = field(default_factory=lambda: Decimal(0))
    fees: Decimal = field(default_factory=lambda: Decimal(0))
    error: Optional[Exception] = None


@dataclass
class UnifiedArbStats:
    """Arbitrage statistics."""
    total_executions: int
    successful_executions: int
    total_pnl: Decimal
    win_rate: float


@dataclass
class UnifiedArbConfig:
    """Configuration for unified arbitrage system."""
    min_spread_bps: Decimal = field(default_factory=lambda: Decimal(10))
    min_profit: Decimal = field(default_factory=lambda: Decimal(5))
    max_position_size: Decimal = field(default_factory=lambda: Decimal(10000))
    max_total_exposure: Decimal = field(default_factory=lambda: Decimal(100000))
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDC", "ETH-USDC", "LUX-USDC"])
    venue_priority: List[str] = field(default_factory=lambda: ["lx_dex", "binance", "mexc", "lx_amm"])
    scan_interval_ms: int = 100
    execute_timeout_ms: int = 5000
    max_daily_loss: Decimal = field(default_factory=lambda: Decimal(1000))
    max_trades_per_day: int = 100


@dataclass
class LxFirstConfig:
    """Configuration for LX-first strategy."""
    max_staleness_ms: int = 2000
    min_divergence_bps: Decimal = field(default_factory=lambda: Decimal(10))
    min_profit: Decimal = field(default_factory=lambda: Decimal(5))
    max_position_size: Decimal = field(default_factory=lambda: Decimal(1000))
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDC", "ETH-USDC", "LUX-USDC"])
    venue_latencies: Dict[str, int] = field(default_factory=lambda: {
        "binance": 50,
        "mexc": 100,
        "okx": 80,
        "uniswap": 12000,
        "pancakeswap": 3000,
    })


@dataclass
class ScannerConfig:
    """Configuration for arbitrage scanner."""
    min_spread_bps: Decimal = field(default_factory=lambda: Decimal(10))
    min_profit_usd: Decimal = field(default_factory=lambda: Decimal(10))
    max_price_age_ms: int = 5000
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "LUX", "SOL", "AVAX"])
    chain_ids: List[str] = field(default_factory=lambda: ["lux", "ethereum", "bsc", "arbitrum", "polygon"])
    scan_interval_ms: int = 100
    max_concurrency: int = 50


@dataclass
class CrossChainInfo:
    """Information about a chain."""
    chain_id: str
    name: str
    chain_type: ChainType
    block_time_ms: int
    finality_ms: int
    warp_supported: bool
    teleport_supported: bool
    venues: List[str] = field(default_factory=list)


@dataclass
class CrossChainConfig:
    """Configuration for cross-chain routing."""
    warp_enabled: bool = True
    warp_endpoint: Optional[str] = None
    warp_timeout_ms: int = 5000
    teleport_enabled: bool = True
    teleport_endpoint: Optional[str] = None
    teleport_timeout_ms: int = 60000
    chains: Dict[str, CrossChainInfo] = field(default_factory=dict)


def default_unified_arb_config() -> UnifiedArbConfig:
    """Return default unified arbitrage configuration."""
    return UnifiedArbConfig()


def default_lx_first_config() -> LxFirstConfig:
    """Return default LX-first configuration."""
    return LxFirstConfig()


def default_scanner_config() -> ScannerConfig:
    """Return default scanner configuration."""
    return ScannerConfig()


def default_cross_chain_config() -> CrossChainConfig:
    """Return default cross-chain configuration with common chains."""
    config = CrossChainConfig()

    # Lux ecosystem (Warp enabled)
    config.chains["lux_mainnet"] = CrossChainInfo(
        chain_id="lux_mainnet",
        name="Lux Mainnet",
        chain_type=ChainType.LUX_SUBNET,
        block_time_ms=400,
        finality_ms=400,
        warp_supported=True,
        teleport_supported=True,
        venues=["lx_dex", "lx_amm"],
    )

    config.chains["lx_dex_subnet"] = CrossChainInfo(
        chain_id="lx_dex_subnet",
        name="LX DEX Subnet",
        chain_type=ChainType.LUX_SUBNET,
        block_time_ms=200,
        finality_ms=200,
        warp_supported=True,
        teleport_supported=False,
        venues=["lx_dex"],
    )

    # EVM chains (Teleport enabled)
    config.chains["ethereum"] = CrossChainInfo(
        chain_id="1",
        name="Ethereum",
        chain_type=ChainType.EVM,
        block_time_ms=12000,
        finality_ms=15 * 60 * 1000,
        warp_supported=False,
        teleport_supported=True,
        venues=["uniswap", "sushiswap"],
    )

    config.chains["bsc"] = CrossChainInfo(
        chain_id="56",
        name="BNB Smart Chain",
        chain_type=ChainType.EVM,
        block_time_ms=3000,
        finality_ms=45000,
        warp_supported=False,
        teleport_supported=True,
        venues=["pancakeswap"],
    )

    config.chains["arbitrum"] = CrossChainInfo(
        chain_id="42161",
        name="Arbitrum One",
        chain_type=ChainType.EVM,
        block_time_ms=250,
        finality_ms=15 * 60 * 1000,
        warp_supported=False,
        teleport_supported=True,
        venues=["uniswap", "camelot"],
    )

    # CEX (API only)
    config.chains["binance"] = CrossChainInfo(
        chain_id="binance",
        name="Binance",
        chain_type=ChainType.CEX,
        block_time_ms=0,
        finality_ms=0,
        warp_supported=False,
        teleport_supported=False,
        venues=["binance"],
    )

    config.chains["mexc"] = CrossChainInfo(
        chain_id="mexc",
        name="MEXC",
        chain_type=ChainType.CEX,
        block_time_ms=0,
        finality_ms=0,
        warp_supported=False,
        teleport_supported=False,
        venues=["mexc"],
    )

    return config
