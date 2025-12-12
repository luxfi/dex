"""Configuration for LX Trading SDK."""

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


@dataclass
class GeneralConfig:
    """General SDK settings."""
    log_level: str = "info"
    timeout_ms: int = 30000
    smart_routing: bool = True
    venue_priority: List[str] = field(default_factory=list)
    min_improvement_bps: int = 5


@dataclass
class RiskConfig:
    """Risk management settings."""
    enabled: bool = True
    max_position_size: Decimal = Decimal(0)
    max_order_size: Decimal = Decimal(0)
    max_daily_loss: Decimal = Decimal(0)
    max_open_orders: int = 100
    kill_switch_enabled: bool = False
    position_limits: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class NativeVenueConfig:
    """Native LX venue config (lx_dex or lx_amm)."""
    venue_type: str = "dex"  # "dex" or "amm"
    api_url: str = ""
    ws_url: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    wallet_address: Optional[str] = None
    private_key: Optional[str] = None
    network: str = "mainnet"
    chain_id: int = 96369
    streaming: bool = True
    maker_fee: Optional[Decimal] = None
    taker_fee: Optional[Decimal] = None

    @classmethod
    def lx_dex(cls, api_url: str) -> "NativeVenueConfig":
        """Create LX DEX config."""
        return cls(venue_type="dex", api_url=api_url)

    @classmethod
    def lx_amm(cls, api_url: str) -> "NativeVenueConfig":
        """Create LX AMM config."""
        return cls(venue_type="amm", api_url=api_url)

    def with_credentials(self, api_key: str, api_secret: str) -> "NativeVenueConfig":
        self.api_key = api_key
        self.api_secret = api_secret
        return self

    def with_wallet(self, address: str, private_key: str) -> "NativeVenueConfig":
        self.wallet_address = address
        self.private_key = private_key
        return self

    def testnet(self) -> "NativeVenueConfig":
        self.network = "testnet"
        self.chain_id = 8888
        return self


@dataclass
class CcxtConfig:
    """CCXT exchange config."""
    exchange_id: str = ""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    password: Optional[str] = None
    sandbox: bool = False
    rate_limit: bool = True
    options: Dict = field(default_factory=dict)

    @classmethod
    def new(cls, exchange_id: str) -> "CcxtConfig":
        return cls(exchange_id=exchange_id)

    def with_credentials(self, api_key: str, api_secret: str) -> "CcxtConfig":
        self.api_key = api_key
        self.api_secret = api_secret
        return self

    def with_password(self, password: str) -> "CcxtConfig":
        self.password = password
        return self


@dataclass
class HummingbotConfig:
    """Hummingbot Gateway config."""
    host: str = "localhost"
    port: int = 15888
    https: bool = False
    connector: str = ""
    chain: str = "lux"
    network: str = "mainnet"
    wallet_address: Optional[str] = None

    @classmethod
    def new(cls, connector: str) -> "HummingbotConfig":
        return cls(connector=connector)

    def with_wallet(self, address: str) -> "HummingbotConfig":
        self.wallet_address = address
        return self

    def with_endpoint(self, host: str, port: int) -> "HummingbotConfig":
        self.host = host
        self.port = port
        return self

    @property
    def base_url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"


@dataclass
class Config:
    """Main SDK configuration."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    native: Dict[str, NativeVenueConfig] = field(default_factory=dict)
    ccxt: Dict[str, CcxtConfig] = field(default_factory=dict)
    hummingbot: Dict[str, HummingbotConfig] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from TOML file."""
        content = Path(path).read_text()
        return cls.from_toml(content)

    @classmethod
    def from_toml(cls, content: str) -> "Config":
        """Parse configuration from TOML string."""
        data = tomllib.loads(content)

        config = cls()

        # General
        if "general" in data:
            g = data["general"]
            config.general = GeneralConfig(
                log_level=g.get("log_level", "info"),
                timeout_ms=g.get("timeout_ms", 30000),
                smart_routing=g.get("smart_routing", True),
                venue_priority=g.get("venue_priority", []),
                min_improvement_bps=g.get("min_improvement_bps", 5),
            )

        # Risk
        if "risk" in data:
            r = data["risk"]
            config.risk = RiskConfig(
                enabled=r.get("enabled", True),
                max_position_size=Decimal(str(r.get("max_position_size", 0))),
                max_order_size=Decimal(str(r.get("max_order_size", 0))),
                max_daily_loss=Decimal(str(r.get("max_daily_loss", 0))),
                max_open_orders=r.get("max_open_orders", 100),
                kill_switch_enabled=r.get("kill_switch_enabled", False),
                position_limits={k: Decimal(str(v)) for k, v in r.get("position_limits", {}).items()},
            )

        # Native venues
        if "native" in data:
            for name, cfg in data["native"].items():
                config.native[name] = NativeVenueConfig(
                    venue_type=cfg.get("venue_type", "dex"),
                    api_url=cfg.get("api_url", ""),
                    ws_url=cfg.get("ws_url"),
                    api_key=cfg.get("api_key"),
                    api_secret=cfg.get("api_secret"),
                    wallet_address=cfg.get("wallet_address"),
                    private_key=cfg.get("private_key"),
                    network=cfg.get("network", "mainnet"),
                    chain_id=cfg.get("chain_id", 96369),
                    streaming=cfg.get("streaming", True),
                )

        # CCXT exchanges
        if "ccxt" in data:
            for name, cfg in data["ccxt"].items():
                config.ccxt[name] = CcxtConfig(
                    exchange_id=cfg.get("exchange_id", name),
                    api_key=cfg.get("api_key"),
                    api_secret=cfg.get("api_secret"),
                    password=cfg.get("password"),
                    sandbox=cfg.get("sandbox", False),
                    rate_limit=cfg.get("rate_limit", True),
                    options=cfg.get("options", {}),
                )

        # Hummingbot gateways
        if "hummingbot" in data:
            for name, cfg in data["hummingbot"].items():
                config.hummingbot[name] = HummingbotConfig(
                    host=cfg.get("host", "localhost"),
                    port=cfg.get("port", 15888),
                    https=cfg.get("https", False),
                    connector=cfg.get("connector", ""),
                    chain=cfg.get("chain", "lux"),
                    network=cfg.get("network", "mainnet"),
                    wallet_address=cfg.get("wallet_address"),
                )

        return config

    def with_native(self, name: str, config: NativeVenueConfig) -> "Config":
        self.native[name] = config
        return self

    def with_ccxt(self, name: str, config: CcxtConfig) -> "Config":
        self.ccxt[name] = config
        return self

    def with_hummingbot(self, name: str, config: HummingbotConfig) -> "Config":
        self.hummingbot[name] = config
        return self
