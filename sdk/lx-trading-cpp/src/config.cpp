// LX Trading SDK - Configuration Implementation

#include <lx/trading/config.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace lx::trading {

// Simple TOML parser (handles basic cases)
namespace {

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s[0] == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

}  // namespace

Config Config::from_file(std::string_view path) {
    std::string path_str{path};
    std::ifstream file{path_str};
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path_str);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_toml(buffer.str());
}

Config Config::from_toml(std::string_view content) {
    Config config;
    std::string current_section;
    std::string current_subsection;

    std::string content_str{content};
    std::istringstream stream{content_str};
    std::string line;

    while (std::getline(stream, line)) {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Section header
        if (line[0] == '[') {
            size_t end = line.find(']');
            if (end != std::string::npos) {
                std::string section = line.substr(1, end - 1);

                // Check for subsection [section.name]
                auto dot = section.find('.');
                if (dot != std::string::npos) {
                    current_section = section.substr(0, dot);
                    current_subsection = section.substr(dot + 1);
                } else {
                    current_section = section;
                    current_subsection.clear();
                }
            }
            continue;
        }

        // Key-value pair
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            value = unquote(value);

            // Parse based on section
            if (current_section == "general") {
                if (key == "log_level") config.general.log_level = value;
                else if (key == "timeout_ms") config.general.timeout_ms = std::stoi(value);
                else if (key == "smart_routing") config.general.smart_routing = (value == "true");
                else if (key == "min_improvement_bps") config.general.min_improvement_bps = std::stoi(value);
            }
            else if (current_section == "risk") {
                if (key == "enabled") config.risk.enabled = (value == "true");
                else if (key == "max_position_size") config.risk.max_position_size = Decimal::from_string(value);
                else if (key == "max_order_size") config.risk.max_order_size = Decimal::from_string(value);
                else if (key == "max_daily_loss") config.risk.max_daily_loss = Decimal::from_string(value);
                else if (key == "max_open_orders") config.risk.max_open_orders = std::stoi(value);
                else if (key == "kill_switch_enabled") config.risk.kill_switch_enabled = (value == "true");
            }
            else if (current_section == "native" && !current_subsection.empty()) {
                auto& native_cfg = config.native[current_subsection];
                if (key == "venue_type") native_cfg.venue_type = value;
                else if (key == "api_url") native_cfg.api_url = value;
                else if (key == "ws_url") native_cfg.ws_url = value;
                else if (key == "api_key") native_cfg.api_key = value;
                else if (key == "api_secret") native_cfg.api_secret = value;
                else if (key == "wallet_address") native_cfg.wallet_address = value;
                else if (key == "private_key") native_cfg.private_key = value;
                else if (key == "network") native_cfg.network = value;
                else if (key == "chain_id") native_cfg.chain_id = std::stoi(value);
                else if (key == "streaming") native_cfg.streaming = (value == "true");
            }
            else if (current_section == "ccxt" && !current_subsection.empty()) {
                auto& ccxt_cfg = config.ccxt[current_subsection];
                if (key == "exchange_id") ccxt_cfg.exchange_id = value;
                else if (key == "api_key") ccxt_cfg.api_key = value;
                else if (key == "api_secret") ccxt_cfg.api_secret = value;
                else if (key == "password") ccxt_cfg.password = value;
                else if (key == "sandbox") ccxt_cfg.sandbox = (value == "true");
                else if (key == "rate_limit") ccxt_cfg.rate_limit = (value == "true");
            }
            else if (current_section == "hummingbot" && !current_subsection.empty()) {
                auto& hb_cfg = config.hummingbot[current_subsection];
                if (key == "host") hb_cfg.host = value;
                else if (key == "port") hb_cfg.port = std::stoi(value);
                else if (key == "https") hb_cfg.https = (value == "true");
                else if (key == "connector") hb_cfg.connector = value;
                else if (key == "chain") hb_cfg.chain = value;
                else if (key == "network") hb_cfg.network = value;
                else if (key == "wallet_address") hb_cfg.wallet_address = value;
            }
        }
    }

    return config;
}

}  // namespace lx::trading
