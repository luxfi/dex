// LX Trading SDK - CCXT Adapter
// REST proxy to CCXT service for 100+ exchanges

#pragma once

#include <lx/trading/adapter.hpp>
#include <lx/trading/config.hpp>
#include <nlohmann/json_fwd.hpp>
#include <atomic>
#include <memory>

namespace lx::trading {

// CCXT Adapter - connects to a CCXT REST service
// This allows C++ to leverage CCXT's exchange coverage via a REST API
class CcxtAdapter : public VenueAdapter {
public:
    CcxtAdapter(std::string_view name, const CcxtConfig& config);
    ~CcxtAdapter() override;

    // VenueAdapter interface
    [[nodiscard]] std::string_view name() const override { return name_; }
    [[nodiscard]] VenueType venue_type() const override { return VenueType::Ccxt; }
    [[nodiscard]] const VenueCapabilities& capabilities() const override { return capabilities_; }
    [[nodiscard]] bool is_connected() const override { return connected_.load(); }
    [[nodiscard]] std::optional<int> latency_ms() const override {
        int lat = latency_.load();
        return lat > 0 ? std::optional<int>(lat) : std::nullopt;
    }

    std::future<void> connect() override;
    std::future<void> disconnect() override;

    std::future<std::vector<MarketInfo>> get_markets() override;
    std::future<Ticker> get_ticker(const std::string& symbol) override;
    std::future<std::unique_ptr<Orderbook>> get_orderbook(
        const std::string& symbol,
        std::optional<int> depth = std::nullopt) override;
    std::future<std::vector<Trade>> get_trades(
        const std::string& symbol,
        std::optional<int> limit = std::nullopt) override;

    std::future<std::vector<Balance>> get_balances() override;
    std::future<Balance> get_balance(const std::string& asset) override;
    std::future<std::vector<Order>> get_open_orders(
        const std::optional<std::string>& symbol = std::nullopt) override;

    std::future<Order> place_order(const OrderRequest& request) override;
    std::future<Order> cancel_order(const std::string& order_id,
                                     const std::string& symbol) override;
    std::future<std::vector<Order>> cancel_all_orders(
        const std::optional<std::string>& symbol = std::nullopt) override;

    // Set the CCXT service URL (default: http://localhost:3000)
    void set_service_url(std::string_view url);

private:
    Order convert_order(const nlohmann::json& json);
    void update_latency(int64_t start_ns);

    std::string name_;
    CcxtConfig config_;
    std::string service_url_ = "http://localhost:3000";
    VenueCapabilities capabilities_;
    std::atomic<bool> connected_{false};
    std::atomic<int> latency_{0};
};

}  // namespace lx::trading
