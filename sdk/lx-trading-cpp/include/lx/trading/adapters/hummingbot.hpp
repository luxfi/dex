// LX Trading SDK - Hummingbot Gateway Adapter
// Connect to Hummingbot Gateway for DEX trading

#pragma once

#include <lx/trading/adapter.hpp>
#include <lx/trading/config.hpp>
#include <nlohmann/json_fwd.hpp>
#include <atomic>
#include <memory>

namespace lx::trading {

// Hummingbot Gateway Adapter
// Connects to Hummingbot Gateway REST API for DEX operations
class HummingbotAdapter : public VenueAdapter {
public:
    HummingbotAdapter(std::string_view name, const HummingbotConfig& config);
    ~HummingbotAdapter() override;

    // VenueAdapter interface
    [[nodiscard]] std::string_view name() const override { return name_; }
    [[nodiscard]] VenueType venue_type() const override { return VenueType::Hummingbot; }
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

    // AMM operations
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
    nlohmann::json build_request_body();
    void update_latency(int64_t start_ns);

    std::string name_;
    HummingbotConfig config_;
    VenueCapabilities capabilities_;
    std::atomic<bool> connected_{false};
    std::atomic<int> latency_{0};
};

}  // namespace lx::trading
