// LX Trading SDK - Configuration
// Builder pattern for fluent configuration

#pragma once

#include <lx/trading/types.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace lx::trading {

// General SDK settings
struct GeneralConfig {
    std::string log_level = "info";
    int timeout_ms = 30000;
    bool smart_routing = true;
    std::vector<std::string> venue_priority;
    int min_improvement_bps = 5;
};

// Risk management settings
struct RiskConfig {
    bool enabled = true;
    Decimal max_position_size;
    Decimal max_order_size;
    Decimal max_daily_loss;
    int max_open_orders = 100;
    bool kill_switch_enabled = false;
    std::unordered_map<std::string, Decimal> position_limits;
};

// Native venue config (LX DEX or LX AMM)
class NativeVenueConfig {
public:
    std::string venue_type = "dex";  // "dex" or "amm"
    std::string api_url;
    std::optional<std::string> ws_url;
    std::optional<std::string> api_key;
    std::optional<std::string> api_secret;
    std::optional<std::string> wallet_address;
    std::optional<std::string> private_key;
    std::string network = "mainnet";
    int chain_id = 96369;
    bool streaming = true;
    std::optional<Decimal> maker_fee;
    std::optional<Decimal> taker_fee;

    NativeVenueConfig() = default;

    static NativeVenueConfig lx_dex(std::string_view api_url) {
        NativeVenueConfig cfg;
        cfg.venue_type = "dex";
        cfg.api_url = std::string(api_url);
        return cfg;
    }

    static NativeVenueConfig lx_amm(std::string_view api_url) {
        NativeVenueConfig cfg;
        cfg.venue_type = "amm";
        cfg.api_url = std::string(api_url);
        return cfg;
    }

    NativeVenueConfig& with_credentials(std::string_view key, std::string_view secret) {
        api_key = std::string(key);
        api_secret = std::string(secret);
        return *this;
    }

    NativeVenueConfig& with_wallet(std::string_view address, std::string_view priv_key) {
        wallet_address = std::string(address);
        private_key = std::string(priv_key);
        return *this;
    }

    NativeVenueConfig& with_ws(std::string_view url) {
        ws_url = std::string(url);
        return *this;
    }

    NativeVenueConfig& testnet() {
        network = "testnet";
        chain_id = 8888;
        return *this;
    }
};

// CCXT exchange config
class CcxtConfig {
public:
    std::string exchange_id;
    std::optional<std::string> api_key;
    std::optional<std::string> api_secret;
    std::optional<std::string> password;
    bool sandbox = false;
    bool rate_limit = true;
    std::unordered_map<std::string, std::string> options;

    CcxtConfig() = default;

    static CcxtConfig create(std::string_view exchange) {
        CcxtConfig cfg;
        cfg.exchange_id = std::string(exchange);
        return cfg;
    }

    CcxtConfig& with_credentials(std::string_view key, std::string_view secret) {
        api_key = std::string(key);
        api_secret = std::string(secret);
        return *this;
    }

    CcxtConfig& with_password(std::string_view pwd) {
        password = std::string(pwd);
        return *this;
    }

    CcxtConfig& enable_sandbox() {
        sandbox = true;
        return *this;
    }

    CcxtConfig& with_option(std::string_view key, std::string_view value) {
        options[std::string(key)] = std::string(value);
        return *this;
    }
};

// Hummingbot Gateway config
class HummingbotConfig {
public:
    std::string host = "localhost";
    int port = 15888;
    bool https = false;
    std::string connector;
    std::string chain = "lux";
    std::string network = "mainnet";
    std::optional<std::string> wallet_address;

    HummingbotConfig() = default;

    static HummingbotConfig create(std::string_view conn) {
        HummingbotConfig cfg;
        cfg.connector = std::string(conn);
        return cfg;
    }

    HummingbotConfig& with_wallet(std::string_view address) {
        wallet_address = std::string(address);
        return *this;
    }

    HummingbotConfig& with_endpoint(std::string_view h, int p) {
        host = std::string(h);
        port = p;
        return *this;
    }

    HummingbotConfig& enable_https() {
        https = true;
        return *this;
    }

    [[nodiscard]] std::string base_url() const {
        return (https ? "https://" : "http://") + host + ":" + std::to_string(port);
    }
};

// Main SDK configuration
class Config {
public:
    GeneralConfig general;
    RiskConfig risk;
    std::unordered_map<std::string, NativeVenueConfig> native;
    std::unordered_map<std::string, CcxtConfig> ccxt;
    std::unordered_map<std::string, HummingbotConfig> hummingbot;

    Config() = default;

    // Load from TOML file
    static Config from_file(std::string_view path);

    // Load from TOML string
    static Config from_toml(std::string_view content);

    // Builder methods
    Config& with_native(std::string_view name, NativeVenueConfig cfg) {
        native[std::string(name)] = std::move(cfg);
        return *this;
    }

    Config& with_ccxt(std::string_view name, CcxtConfig cfg) {
        ccxt[std::string(name)] = std::move(cfg);
        return *this;
    }

    Config& with_hummingbot(std::string_view name, HummingbotConfig cfg) {
        hummingbot[std::string(name)] = std::move(cfg);
        return *this;
    }

    Config& enable_smart_routing(bool enabled = true) {
        general.smart_routing = enabled;
        return *this;
    }

    Config& set_timeout(int ms) {
        general.timeout_ms = ms;
        return *this;
    }

    Config& set_venue_priority(std::vector<std::string> priority) {
        general.venue_priority = std::move(priority);
        return *this;
    }

    Config& enable_risk_management(bool enabled = true) {
        risk.enabled = enabled;
        return *this;
    }

    Config& set_max_order_size(Decimal size) {
        risk.max_order_size = size;
        return *this;
    }

    Config& set_max_position_size(Decimal size) {
        risk.max_position_size = size;
        return *this;
    }

    Config& set_max_daily_loss(Decimal loss) {
        risk.max_daily_loss = loss;
        return *this;
    }

    Config& set_position_limit(std::string_view asset, Decimal limit) {
        risk.position_limits[std::string(asset)] = limit;
        return *this;
    }
};

}  // namespace lx::trading
