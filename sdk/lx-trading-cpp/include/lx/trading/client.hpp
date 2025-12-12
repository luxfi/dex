// LX Trading SDK - Unified Client
// Smart routing across multiple venues

#pragma once

#include <lx/trading/adapter.hpp>
#include <lx/trading/config.hpp>
#include <lx/trading/orderbook.hpp>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lx::trading {

// Forward declarations
class RiskManager;

// Unified trading client
class Client {
public:
    explicit Client(const Config& config);
    ~Client();

    // Disallow copy
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Connection management
    std::future<void> connect();
    std::future<void> disconnect();

    // Venue access
    [[nodiscard]] VenueAdapter* venue(std::string_view name);
    [[nodiscard]] const VenueAdapter* venue(std::string_view name) const;
    [[nodiscard]] std::vector<VenueInfo> venues() const;

    // =========================================================================
    // Market Data
    // =========================================================================

    // Get orderbook from specific venue
    std::future<std::unique_ptr<Orderbook>> orderbook(
        const std::string& symbol,
        std::optional<std::string_view> venue = std::nullopt);

    // Get aggregated orderbook from all venues
    std::future<AggregatedOrderbook> aggregated_orderbook(const std::string& symbol);

    // Get ticker from specific venue
    std::future<Ticker> ticker(
        const std::string& symbol,
        std::optional<std::string_view> venue = std::nullopt);

    // Get tickers from all venues
    std::future<std::vector<Ticker>> tickers(const std::string& symbol);

    // =========================================================================
    // Account
    // =========================================================================

    // Get aggregated balances across all venues
    std::future<std::vector<AggregatedBalance>> balances();

    // Get balance for specific asset from venue
    std::future<Balance> balance(
        const std::string& asset,
        std::optional<std::string_view> venue = std::nullopt);

    // =========================================================================
    // Order Management
    // =========================================================================

    // Place market buy order
    std::future<Order> buy(
        const std::string& symbol,
        Decimal quantity,
        std::optional<std::string_view> venue = std::nullopt);

    // Place market sell order
    std::future<Order> sell(
        const std::string& symbol,
        Decimal quantity,
        std::optional<std::string_view> venue = std::nullopt);

    // Place limit buy order
    std::future<Order> limit_buy(
        const std::string& symbol,
        Decimal quantity,
        Decimal price,
        std::optional<std::string_view> venue = std::nullopt);

    // Place limit sell order
    std::future<Order> limit_sell(
        const std::string& symbol,
        Decimal quantity,
        Decimal price,
        std::optional<std::string_view> venue = std::nullopt);

    // Place order with full control
    std::future<Order> place_order(const OrderRequest& request);

    // Cancel order
    std::future<Order> cancel_order(
        const std::string& order_id,
        const std::string& symbol,
        const std::string& venue);

    // Cancel all orders
    std::future<std::vector<Order>> cancel_all_orders(
        std::optional<std::string_view> symbol = std::nullopt,
        std::optional<std::string_view> venue = std::nullopt);

    // Get all open orders
    std::future<std::vector<Order>> open_orders(
        std::optional<std::string_view> symbol = std::nullopt);

    // =========================================================================
    // AMM Operations
    // =========================================================================

    // Get swap quote
    std::future<SwapQuote> quote(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy,
        const std::string& venue);

    // Execute swap
    std::future<Trade> swap(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy,
        double slippage,
        const std::string& venue);

    // Get pool info
    std::future<PoolInfo> pool_info(
        const std::string& base_token,
        const std::string& quote_token,
        const std::string& venue);

    // Add liquidity
    std::future<LiquidityResult> add_liquidity(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal base_amount,
        Decimal quote_amount,
        double slippage,
        const std::string& venue);

    // Remove liquidity
    std::future<LiquidityResult> remove_liquidity(
        const std::string& pool_address,
        Decimal liquidity_amount,
        double slippage,
        const std::string& venue);

    // Get LP positions
    std::future<std::vector<LpPosition>> lp_positions(const std::string& venue);

    // =========================================================================
    // Streaming
    // =========================================================================

    void subscribe_ticker(
        const std::string& symbol,
        VenueAdapter::TickerCallback callback,
        std::optional<std::string_view> venue = std::nullopt);

    void subscribe_trades(
        const std::string& symbol,
        VenueAdapter::TradeCallback callback,
        std::optional<std::string_view> venue = std::nullopt);

    void subscribe_orderbook(
        const std::string& symbol,
        VenueAdapter::OrderbookCallback callback,
        std::optional<std::string_view> venue = std::nullopt);

    void subscribe_orders(VenueAdapter::OrderCallback callback);

    void unsubscribe_all();

    // =========================================================================
    // Risk Management
    // =========================================================================

    [[nodiscard]] RiskManager& risk_manager() { return *risk_manager_; }
    [[nodiscard]] const RiskManager& risk_manager() const { return *risk_manager_; }

private:
    Order place_order_impl(const OrderRequest& request);
    Order smart_route(const OrderRequest& request);
    VenueAdapter* get_venue(std::optional<std::string_view> name);

    Config config_;
    std::unordered_map<std::string, std::unique_ptr<VenueAdapter>> venues_;
    std::optional<std::string> default_venue_;
    std::unique_ptr<RiskManager> risk_manager_;
};

}  // namespace lx::trading
