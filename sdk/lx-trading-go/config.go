// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"os"

	"github.com/shopspring/decimal"
)

// GeneralConfig holds general SDK settings.
type GeneralConfig struct {
	LogLevel          string
	TimeoutMs         int
	SmartRouting      bool
	VenuePriority     []string
	MinImprovementBps int
}

// DefaultGeneralConfig returns sensible defaults.
func DefaultGeneralConfig() GeneralConfig {
	return GeneralConfig{
		LogLevel:          "info",
		TimeoutMs:         30000,
		SmartRouting:      true,
		VenuePriority:     nil,
		MinImprovementBps: 5,
	}
}

// RiskConfig holds risk management settings.
type RiskConfig struct {
	Enabled           bool
	MaxPositionSize   decimal.Decimal
	MaxOrderSize      decimal.Decimal
	MaxDailyLoss      decimal.Decimal
	MaxOpenOrders     int
	KillSwitchEnabled bool
	PositionLimits    map[string]decimal.Decimal
}

// DefaultRiskConfig returns sensible defaults.
func DefaultRiskConfig() RiskConfig {
	return RiskConfig{
		Enabled:           true,
		MaxPositionSize:   decimal.Zero,
		MaxOrderSize:      decimal.Zero,
		MaxDailyLoss:      decimal.Zero,
		MaxOpenOrders:     100,
		KillSwitchEnabled: false,
		PositionLimits:    make(map[string]decimal.Decimal),
	}
}

// NativeVenueConfig holds config for native LX venues (DEX or AMM).
type NativeVenueConfig struct {
	VenueType     string // "dex" or "amm"
	APIURL        string
	WSURL         string
	APIKey        string
	APISecret     string
	WalletAddress string
	PrivateKey    string
	Network       string
	ChainID       int
	Streaming     bool
	MakerFee      *decimal.Decimal
	TakerFee      *decimal.Decimal
}

// NewLxDexConfig creates config for LX DEX.
func NewLxDexConfig(apiURL string) *NativeVenueConfig {
	return &NativeVenueConfig{
		VenueType: "dex",
		APIURL:    apiURL,
		Network:   "mainnet",
		ChainID:   96369,
		Streaming: true,
	}
}

// NewLxAmmConfig creates config for LX AMM.
func NewLxAmmConfig(apiURL string) *NativeVenueConfig {
	return &NativeVenueConfig{
		VenueType: "amm",
		APIURL:    apiURL,
		Network:   "mainnet",
		ChainID:   96369,
		Streaming: true,
	}
}

// WithCredentials sets API credentials.
func (c *NativeVenueConfig) WithCredentials(apiKey, apiSecret string) *NativeVenueConfig {
	c.APIKey = apiKey
	c.APISecret = apiSecret
	return c
}

// WithWallet sets wallet credentials.
func (c *NativeVenueConfig) WithWallet(address, privateKey string) *NativeVenueConfig {
	c.WalletAddress = address
	c.PrivateKey = privateKey
	return c
}

// WithWebSocket sets WebSocket URL.
func (c *NativeVenueConfig) WithWebSocket(wsURL string) *NativeVenueConfig {
	c.WSURL = wsURL
	return c
}

// Testnet switches to testnet.
func (c *NativeVenueConfig) Testnet() *NativeVenueConfig {
	c.Network = "testnet"
	c.ChainID = 8888
	return c
}

// CcxtConfig holds config for CCXT exchanges.
type CcxtConfig struct {
	ExchangeID string
	APIKey     string
	APISecret  string
	Password   string
	Sandbox    bool
	RateLimit  bool
	Options    map[string]interface{}
	// BaseURL for CCXT proxy/gateway endpoint
	BaseURL string
}

// NewCcxtConfig creates CCXT exchange config.
func NewCcxtConfig(exchangeID string) *CcxtConfig {
	return &CcxtConfig{
		ExchangeID: exchangeID,
		RateLimit:  true,
		Options:    make(map[string]interface{}),
		BaseURL:    "http://localhost:8080", // Default CCXT gateway
	}
}

// WithCredentials sets API credentials.
func (c *CcxtConfig) WithCredentials(apiKey, apiSecret string) *CcxtConfig {
	c.APIKey = apiKey
	c.APISecret = apiSecret
	return c
}

// WithPassword sets API password (for exchanges like OKX).
func (c *CcxtConfig) WithPassword(password string) *CcxtConfig {
	c.Password = password
	return c
}

// WithSandbox enables sandbox mode.
func (c *CcxtConfig) WithSandbox() *CcxtConfig {
	c.Sandbox = true
	return c
}

// WithBaseURL sets custom CCXT gateway URL.
func (c *CcxtConfig) WithBaseURL(url string) *CcxtConfig {
	c.BaseURL = url
	return c
}

// WithOption sets a custom option.
func (c *CcxtConfig) WithOption(key string, value interface{}) *CcxtConfig {
	c.Options[key] = value
	return c
}

// HummingbotConfig holds config for Hummingbot Gateway.
type HummingbotConfig struct {
	Host          string
	Port          int
	HTTPS         bool
	Connector     string
	Chain         string
	Network       string
	WalletAddress string
}

// NewHummingbotConfig creates Hummingbot Gateway config.
func NewHummingbotConfig(connector string) *HummingbotConfig {
	return &HummingbotConfig{
		Host:      "localhost",
		Port:      15888,
		HTTPS:     false,
		Connector: connector,
		Chain:     "lux",
		Network:   "mainnet",
	}
}

// WithWallet sets wallet address.
func (c *HummingbotConfig) WithWallet(address string) *HummingbotConfig {
	c.WalletAddress = address
	return c
}

// WithEndpoint sets custom host and port.
func (c *HummingbotConfig) WithEndpoint(host string, port int) *HummingbotConfig {
	c.Host = host
	c.Port = port
	return c
}

// WithHTTPS enables HTTPS.
func (c *HummingbotConfig) WithHTTPS() *HummingbotConfig {
	c.HTTPS = true
	return c
}

// BaseURL returns the full base URL.
func (c *HummingbotConfig) BaseURL() string {
	scheme := "http"
	if c.HTTPS {
		scheme = "https"
	}
	return scheme + "://" + c.Host + ":" + itoa(c.Port)
}

// Config holds the complete SDK configuration.
type Config struct {
	General    GeneralConfig
	Risk       RiskConfig
	Native     map[string]*NativeVenueConfig
	Ccxt       map[string]*CcxtConfig
	Hummingbot map[string]*HummingbotConfig
}

// NewConfig creates an empty configuration.
func NewConfig() *Config {
	return &Config{
		General:    DefaultGeneralConfig(),
		Risk:       DefaultRiskConfig(),
		Native:     make(map[string]*NativeVenueConfig),
		Ccxt:       make(map[string]*CcxtConfig),
		Hummingbot: make(map[string]*HummingbotConfig),
	}
}

// WithNative adds a native venue.
func (c *Config) WithNative(name string, cfg *NativeVenueConfig) *Config {
	c.Native[name] = cfg
	return c
}

// WithCcxt adds a CCXT exchange.
func (c *Config) WithCcxt(name string, cfg *CcxtConfig) *Config {
	c.Ccxt[name] = cfg
	return c
}

// WithHummingbot adds a Hummingbot Gateway connector.
func (c *Config) WithHummingbot(name string, cfg *HummingbotConfig) *Config {
	c.Hummingbot[name] = cfg
	return c
}

// WithSmartRouting enables/disables smart routing.
func (c *Config) WithSmartRouting(enabled bool) *Config {
	c.General.SmartRouting = enabled
	return c
}

// WithVenuePriority sets venue priority order.
func (c *Config) WithVenuePriority(venues ...string) *Config {
	c.General.VenuePriority = venues
	return c
}

// WithRiskConfig sets risk configuration.
func (c *Config) WithRiskConfig(risk RiskConfig) *Config {
	c.Risk = risk
	return c
}

// FromEnv loads credentials from environment variables.
// Pattern: {NAME}_API_KEY, {NAME}_API_SECRET, {NAME}_PASSWORD
func (c *Config) FromEnv() *Config {
	for name, cfg := range c.Native {
		if key := os.Getenv(toEnvKey(name) + "_API_KEY"); key != "" {
			cfg.APIKey = key
		}
		if secret := os.Getenv(toEnvKey(name) + "_API_SECRET"); secret != "" {
			cfg.APISecret = secret
		}
		if wallet := os.Getenv(toEnvKey(name) + "_WALLET"); wallet != "" {
			cfg.WalletAddress = wallet
		}
		if pk := os.Getenv(toEnvKey(name) + "_PRIVATE_KEY"); pk != "" {
			cfg.PrivateKey = pk
		}
	}

	for name, cfg := range c.Ccxt {
		if key := os.Getenv(toEnvKey(name) + "_API_KEY"); key != "" {
			cfg.APIKey = key
		}
		if secret := os.Getenv(toEnvKey(name) + "_API_SECRET"); secret != "" {
			cfg.APISecret = secret
		}
		if pwd := os.Getenv(toEnvKey(name) + "_PASSWORD"); pwd != "" {
			cfg.Password = pwd
		}
	}

	for name, cfg := range c.Hummingbot {
		if wallet := os.Getenv(toEnvKey(name) + "_WALLET"); wallet != "" {
			cfg.WalletAddress = wallet
		}
	}

	return c
}

// Helper: convert name to ENV_KEY format
func toEnvKey(name string) string {
	result := make([]byte, 0, len(name))
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c >= 'a' && c <= 'z' {
			result = append(result, c-32) // to upper
		} else if c >= 'A' && c <= 'Z' {
			result = append(result, c)
		} else if c >= '0' && c <= '9' {
			result = append(result, c)
		} else {
			result = append(result, '_')
		}
	}
	return string(result)
}

// Helper: int to string without strconv import
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
