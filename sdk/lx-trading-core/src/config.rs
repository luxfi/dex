//! Configuration for the LX Trading SDK.

use crate::error::{Error, Result};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main SDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// General settings
    #[serde(default)]
    pub general: GeneralConfig,

    /// Risk management settings
    #[serde(default)]
    pub risk: RiskConfig,

    /// Native LX venues (lx_dex, lx_amm)
    #[serde(default)]
    pub native: HashMap<String, NativeVenueConfig>,

    /// CCXT exchange configurations
    #[serde(default)]
    pub ccxt: HashMap<String, CcxtConfig>,

    /// Hummingbot Gateway configurations
    #[serde(default)]
    pub hummingbot: HashMap<String, HummingbotConfig>,

    /// Custom adapter configurations
    #[serde(default)]
    pub custom: HashMap<String, CustomAdapterConfig>,
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::ConfigError(format!("Failed to read config file: {}", e)))?;

        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Create from TOML string
    pub fn from_toml(content: &str) -> Result<Self> {
        let config: Config = toml::from_str(content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate risk limits
        if self.risk.max_position_size.is_sign_negative() {
            return Err(Error::InvalidConfig {
                field: "risk.max_position_size".into(),
                message: "Must be non-negative".into(),
            });
        }

        // Validate venue configs
        for (name, cfg) in &self.native {
            if cfg.api_url.is_empty() {
                return Err(Error::InvalidConfig {
                    field: format!("native.{}.api_url", name),
                    message: "API URL is required".into(),
                });
            }
        }

        for (name, cfg) in &self.ccxt {
            if cfg.exchange_id.is_empty() {
                return Err(Error::InvalidConfig {
                    field: format!("ccxt.{}.exchange_id", name),
                    message: "Exchange ID is required".into(),
                });
            }
        }

        Ok(())
    }

    /// Get default configuration
    pub fn default_config() -> Self {
        Self {
            general: GeneralConfig::default(),
            risk: RiskConfig::default(),
            native: HashMap::new(),
            ccxt: HashMap::new(),
            hummingbot: HashMap::new(),
            custom: HashMap::new(),
        }
    }

    /// Builder pattern: add native venue
    pub fn with_native(mut self, name: impl Into<String>, config: NativeVenueConfig) -> Self {
        self.native.insert(name.into(), config);
        self
    }

    /// Builder pattern: add CCXT exchange
    pub fn with_ccxt(mut self, name: impl Into<String>, config: CcxtConfig) -> Self {
        self.ccxt.insert(name.into(), config);
        self
    }

    /// Builder pattern: add Hummingbot gateway
    pub fn with_hummingbot(mut self, name: impl Into<String>, config: HummingbotConfig) -> Self {
        self.hummingbot.insert(name.into(), config);
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default_config()
    }
}

/// General SDK settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Request timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,

    /// Enable smart order routing
    #[serde(default = "default_true")]
    pub smart_routing: bool,

    /// Prefer venues in this order for routing
    #[serde(default)]
    pub venue_priority: Vec<String>,

    /// Minimum improvement (bps) to route to non-preferred venue
    #[serde(default = "default_min_improvement")]
    pub min_improvement_bps: u32,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: "info".into(),
            timeout_ms: 30000,
            smart_routing: true,
            venue_priority: vec![],
            min_improvement_bps: 5,
        }
    }
}

/// Risk management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Enable risk checks
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum position size per asset (0 = unlimited)
    #[serde(default)]
    pub max_position_size: Decimal,

    /// Maximum order size
    #[serde(default)]
    pub max_order_size: Decimal,

    /// Maximum daily loss (quote currency, 0 = unlimited)
    #[serde(default)]
    pub max_daily_loss: Decimal,

    /// Maximum open orders per symbol
    #[serde(default = "default_max_open_orders")]
    pub max_open_orders: u32,

    /// Kill switch: stop all trading if triggered
    #[serde(default)]
    pub kill_switch_enabled: bool,

    /// Asset-specific position limits
    #[serde(default)]
    pub position_limits: HashMap<String, Decimal>,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_position_size: Decimal::ZERO,
            max_order_size: Decimal::ZERO,
            max_daily_loss: Decimal::ZERO,
            max_open_orders: 100,
            kill_switch_enabled: false,
            position_limits: HashMap::new(),
        }
    }
}

/// Native LX venue configuration (lx_dex or lx_amm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeVenueConfig {
    /// Venue type: "dex" (CLOB) or "amm" (liquidity pools)
    #[serde(default = "default_dex_type")]
    pub venue_type: String,

    /// REST API URL
    pub api_url: String,

    /// WebSocket URL
    #[serde(default)]
    pub ws_url: Option<String>,

    /// API key
    #[serde(default)]
    pub api_key: Option<String>,

    /// API secret
    #[serde(default)]
    pub api_secret: Option<String>,

    /// Wallet address (for on-chain operations)
    #[serde(default)]
    pub wallet_address: Option<String>,

    /// Private key (for signing, use env var in production)
    #[serde(default)]
    pub private_key: Option<String>,

    /// Network: mainnet, testnet
    #[serde(default = "default_mainnet")]
    pub network: String,

    /// Chain ID
    #[serde(default = "default_chain_id")]
    pub chain_id: u64,

    /// Enable streaming via WebSocket
    #[serde(default = "default_true")]
    pub streaming: bool,

    /// Custom maker fee override
    #[serde(default)]
    pub maker_fee: Option<Decimal>,

    /// Custom taker fee override
    #[serde(default)]
    pub taker_fee: Option<Decimal>,
}

impl NativeVenueConfig {
    /// Create LX DEX (CLOB) config
    pub fn lx_dex(api_url: impl Into<String>) -> Self {
        Self {
            venue_type: "dex".into(),
            api_url: api_url.into(),
            ws_url: None,
            api_key: None,
            api_secret: None,
            wallet_address: None,
            private_key: None,
            network: "mainnet".into(),
            chain_id: 96369,
            streaming: true,
            maker_fee: None,
            taker_fee: None,
        }
    }

    /// Create LX AMM config
    pub fn lx_amm(api_url: impl Into<String>) -> Self {
        Self {
            venue_type: "amm".into(),
            api_url: api_url.into(),
            ws_url: None,
            api_key: None,
            api_secret: None,
            wallet_address: None,
            private_key: None,
            network: "mainnet".into(),
            chain_id: 96369,
            streaming: true,
            maker_fee: None,
            taker_fee: None,
        }
    }

    /// Builder: set credentials
    pub fn with_credentials(
        mut self,
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Self {
        self.api_key = Some(api_key.into());
        self.api_secret = Some(api_secret.into());
        self
    }

    /// Builder: set wallet
    pub fn with_wallet(
        mut self,
        address: impl Into<String>,
        private_key: impl Into<String>,
    ) -> Self {
        self.wallet_address = Some(address.into());
        self.private_key = Some(private_key.into());
        self
    }

    /// Builder: set WebSocket URL
    pub fn with_websocket(mut self, ws_url: impl Into<String>) -> Self {
        self.ws_url = Some(ws_url.into());
        self
    }

    /// Builder: use testnet
    pub fn testnet(mut self) -> Self {
        self.network = "testnet".into();
        self.chain_id = 8888;
        self
    }
}

/// CCXT exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcxtConfig {
    /// CCXT exchange ID (e.g., "binance", "mexc", "okx")
    pub exchange_id: String,

    /// API key
    #[serde(default)]
    pub api_key: Option<String>,

    /// API secret
    #[serde(default)]
    pub api_secret: Option<String>,

    /// Password/passphrase (for exchanges that require it)
    #[serde(default)]
    pub password: Option<String>,

    /// Use sandbox/testnet
    #[serde(default)]
    pub sandbox: bool,

    /// Enable rate limiting
    #[serde(default = "default_true")]
    pub rate_limit: bool,

    /// Custom options passed to CCXT
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

impl CcxtConfig {
    pub fn new(exchange_id: impl Into<String>) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            api_key: None,
            api_secret: None,
            password: None,
            sandbox: false,
            rate_limit: true,
            options: HashMap::new(),
        }
    }

    pub fn with_credentials(
        mut self,
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Self {
        self.api_key = Some(api_key.into());
        self.api_secret = Some(api_secret.into());
        self
    }

    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(password.into());
        self
    }

    pub fn sandbox(mut self) -> Self {
        self.sandbox = true;
        self
    }
}

/// Hummingbot Gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HummingbotConfig {
    /// Gateway host
    #[serde(default = "default_gateway_host")]
    pub host: String,

    /// Gateway port
    #[serde(default = "default_gateway_port")]
    pub port: u16,

    /// Use HTTPS
    #[serde(default)]
    pub https: bool,

    /// Connector name in Gateway (e.g., "lxdex", "uniswap")
    pub connector: String,

    /// Chain name
    #[serde(default = "default_chain")]
    pub chain: String,

    /// Network name
    #[serde(default = "default_mainnet")]
    pub network: String,

    /// Wallet address
    #[serde(default)]
    pub wallet_address: Option<String>,
}

impl HummingbotConfig {
    pub fn new(connector: impl Into<String>) -> Self {
        Self {
            host: "localhost".into(),
            port: 15888,
            https: false,
            connector: connector.into(),
            chain: "lux".into(),
            network: "mainnet".into(),
            wallet_address: None,
        }
    }

    pub fn with_wallet(mut self, address: impl Into<String>) -> Self {
        self.wallet_address = Some(address.into());
        self
    }

    pub fn with_endpoint(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Get base URL for Gateway
    pub fn base_url(&self) -> String {
        let scheme = if self.https { "https" } else { "http" };
        format!("{}://{}:{}", scheme, self.host, self.port)
    }
}

/// Custom adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomAdapterConfig {
    /// Adapter type/name
    pub adapter_type: String,

    /// Custom configuration as key-value pairs
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

// Default value helpers
fn default_log_level() -> String {
    "info".into()
}

fn default_timeout() -> u64 {
    30000
}

fn default_true() -> bool {
    true
}

fn default_min_improvement() -> u32 {
    5
}

fn default_max_open_orders() -> u32 {
    100
}

fn default_dex_type() -> String {
    "dex".into()
}

fn default_mainnet() -> String {
    "mainnet".into()
}

fn default_chain_id() -> u64 {
    96369
}

fn default_gateway_host() -> String {
    "localhost".into()
}

fn default_gateway_port() -> u16 {
    15888
}

fn default_chain() -> String {
    "lux".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_toml() {
        let toml = r#"
[general]
log_level = "debug"
smart_routing = true

[risk]
enabled = true
max_daily_loss = 1000

[native.lx_dex]
venue_type = "dex"
api_url = "https://api.dex.lux.network"
network = "mainnet"

[native.lx_amm]
venue_type = "amm"
api_url = "https://api.dex.lux.network"

[ccxt.binance]
exchange_id = "binance"
api_key = "key"
api_secret = "secret"

[hummingbot.gateway]
connector = "lxdex"
chain = "lux"
"#;

        let config = Config::from_toml(toml).unwrap();
        assert_eq!(config.general.log_level, "debug");
        assert!(config.native.contains_key("lx_dex"));
        assert!(config.native.contains_key("lx_amm"));
        assert!(config.ccxt.contains_key("binance"));
        assert!(config.hummingbot.contains_key("gateway"));
    }

    #[test]
    fn test_venue_config_builders() {
        let dex = NativeVenueConfig::lx_dex("https://api.dex.lux.network")
            .with_credentials("key", "secret")
            .with_websocket("wss://ws.dex.lux.network");

        assert_eq!(dex.venue_type, "dex");
        assert_eq!(dex.api_key, Some("key".into()));
        assert!(dex.ws_url.is_some());

        let amm = NativeVenueConfig::lx_amm("https://api.dex.lux.network").testnet();
        assert_eq!(amm.venue_type, "amm");
        assert_eq!(amm.network, "testnet");
    }
}
