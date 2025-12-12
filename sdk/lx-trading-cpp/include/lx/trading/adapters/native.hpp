// LX Trading SDK - Native Adapters
// LX DEX (CLOB) and LX AMM adapters

#pragma once

#include <lx/trading/adapter.hpp>
#include <lx/trading/config.hpp>
#include <lx/trading/orderbook.hpp>
#include <nlohmann/json_fwd.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace lx::trading {

// HTTP client abstraction (implemented with cpr)
class HttpClient;

// LX DEX Adapter - Central Limit Order Book
class LxDexAdapter : public VenueAdapter {
public:
    LxDexAdapter(std::string_view name, const NativeVenueConfig& config);
    ~LxDexAdapter() override;

    // VenueAdapter interface
    [[nodiscard]] std::string_view name() const override { return name_; }
    [[nodiscard]] VenueType venue_type() const override { return VenueType::Native; }
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

private:
    Order convert_order(const nlohmann::json& json);
    void update_latency(int64_t start_ns);

    std::string name_;
    NativeVenueConfig config_;
    VenueCapabilities capabilities_;
    std::unique_ptr<HttpClient> http_;
    std::atomic<bool> connected_{false};
    std::atomic<int> latency_{0};
};

// LX AMM Adapter - Automated Market Maker
class LxAmmAdapter : public VenueAdapter {
public:
    LxAmmAdapter(std::string_view name, const NativeVenueConfig& config);
    ~LxAmmAdapter() override;

    // VenueAdapter interface
    [[nodiscard]] std::string_view name() const override { return name_; }
    [[nodiscard]] VenueType venue_type() const override { return VenueType::Native; }
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

    // AMM-specific operations
    std::future<SwapQuote> get_swap_quote(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy) override;

    std::future<Trade> execute_swap(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy,
        Decimal slippage) override;

    std::future<PoolInfo> get_pool_info(
        const std::string& base_token,
        const std::string& quote_token) override;

    std::future<LiquidityResult> add_liquidity(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal base_amount,
        Decimal quote_amount,
        Decimal slippage) override;

    std::future<LiquidityResult> remove_liquidity(
        const std::string& pool_address,
        Decimal liquidity_amount,
        Decimal slippage) override;

    std::future<std::vector<LpPosition>> get_lp_positions() override;

private:
    void update_latency(int64_t start_ns);

    std::string name_;
    NativeVenueConfig config_;
    VenueCapabilities capabilities_;
    std::unique_ptr<HttpClient> http_;
    std::atomic<bool> connected_{false};
    std::atomic<int> latency_{0};
};

}  // namespace lx::trading
