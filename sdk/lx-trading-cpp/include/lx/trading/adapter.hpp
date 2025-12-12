// LX Trading SDK - Venue Adapter Interface
// Abstract interface for all trading venues

#pragma once

#include <lx/trading/types.hpp>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace lx::trading {

// Forward declarations
class Orderbook;

// Venue capabilities
struct VenueCapabilities {
    bool limit_orders = false;
    bool market_orders = false;
    bool stop_orders = false;
    bool post_only = false;
    bool cancel_orders = false;
    bool batch_orders = false;
    bool streaming = false;
    bool orderbook = false;
    bool trades = false;
    bool amm_swap = false;
    bool add_liquidity = false;
    bool remove_liquidity = false;
    bool lp_positions = false;
    int max_batch_size = 1;
    std::set<std::string> supported_pairs;

    static VenueCapabilities clob() {
        VenueCapabilities caps;
        caps.limit_orders = true;
        caps.market_orders = true;
        caps.stop_orders = true;
        caps.post_only = true;
        caps.cancel_orders = true;
        caps.batch_orders = true;
        caps.streaming = true;
        caps.orderbook = true;
        caps.trades = true;
        caps.max_batch_size = 10;
        return caps;
    }

    static VenueCapabilities amm() {
        VenueCapabilities caps;
        caps.market_orders = true;
        caps.streaming = true;
        caps.trades = true;
        caps.amm_swap = true;
        caps.add_liquidity = true;
        caps.remove_liquidity = true;
        caps.lp_positions = true;
        return caps;
    }
};

// Adapter error
class AdapterError : public std::runtime_error {
public:
    explicit AdapterError(const std::string& msg) : std::runtime_error(msg) {}
};

// Base adapter interface
class VenueAdapter {
public:
    virtual ~VenueAdapter() = default;

    // Properties
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual VenueType venue_type() const = 0;
    [[nodiscard]] virtual const VenueCapabilities& capabilities() const = 0;
    [[nodiscard]] virtual bool is_connected() const = 0;
    [[nodiscard]] virtual std::optional<int> latency_ms() const { return std::nullopt; }

    // Venue info
    [[nodiscard]] VenueInfo info() const {
        VenueInfo vi;
        vi.name = std::string(name());
        vi.venue_type = venue_type();
        vi.connected = is_connected();
        vi.latency_ms = latency_ms();
        const auto& caps = capabilities();
        vi.supported_pairs = std::vector<std::string>(caps.supported_pairs.begin(),
                                                       caps.supported_pairs.end());
        vi.maker_fee = Decimal::from_double(0.001);
        vi.taker_fee = Decimal::from_double(0.002);
        return vi;
    }

    // Connection
    virtual std::future<void> connect() = 0;
    virtual std::future<void> disconnect() = 0;

    // Market data
    virtual std::future<std::vector<MarketInfo>> get_markets() = 0;
    virtual std::future<Ticker> get_ticker(const std::string& symbol) = 0;
    virtual std::future<std::unique_ptr<Orderbook>> get_orderbook(
        const std::string& symbol,
        std::optional<int> depth = std::nullopt) = 0;
    virtual std::future<std::vector<Trade>> get_trades(
        const std::string& symbol,
        std::optional<int> limit = std::nullopt) = 0;

    // Account
    virtual std::future<std::vector<Balance>> get_balances() = 0;
    virtual std::future<Balance> get_balance(const std::string& asset) = 0;
    virtual std::future<std::vector<Order>> get_open_orders(
        const std::optional<std::string>& symbol = std::nullopt) = 0;

    // Orders
    virtual std::future<Order> place_order(const OrderRequest& request) = 0;
    virtual std::future<Order> cancel_order(const std::string& order_id,
                                             const std::string& symbol) = 0;
    virtual std::future<std::vector<Order>> cancel_all_orders(
        const std::optional<std::string>& symbol = std::nullopt) = 0;

    // AMM operations (optional - throw AdapterError if not supported)
    virtual std::future<SwapQuote> get_swap_quote(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy) {
        return std::async(std::launch::deferred, []() -> SwapQuote {
            throw AdapterError("AMM swap not supported");
        });
    }

    virtual std::future<Trade> execute_swap(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal amount,
        bool is_buy,
        Decimal slippage) {
        return std::async(std::launch::deferred, []() -> Trade {
            throw AdapterError("AMM swap not supported");
        });
    }

    virtual std::future<PoolInfo> get_pool_info(
        const std::string& base_token,
        const std::string& quote_token) {
        return std::async(std::launch::deferred, []() -> PoolInfo {
            throw AdapterError("Pool info not supported");
        });
    }

    virtual std::future<LiquidityResult> add_liquidity(
        const std::string& base_token,
        const std::string& quote_token,
        Decimal base_amount,
        Decimal quote_amount,
        Decimal slippage) {
        return std::async(std::launch::deferred, []() -> LiquidityResult {
            throw AdapterError("Add liquidity not supported");
        });
    }

    virtual std::future<LiquidityResult> remove_liquidity(
        const std::string& pool_address,
        Decimal liquidity_amount,
        Decimal slippage) {
        return std::async(std::launch::deferred, []() -> LiquidityResult {
            throw AdapterError("Remove liquidity not supported");
        });
    }

    virtual std::future<std::vector<LpPosition>> get_lp_positions() {
        return std::async(std::launch::deferred, []() -> std::vector<LpPosition> {
            throw AdapterError("LP positions not supported");
        });
    }

    // Streaming callbacks
    using TickerCallback = std::function<void(const Ticker&)>;
    using TradeCallback = std::function<void(const Trade&)>;
    using OrderbookCallback = std::function<void(const Orderbook&)>;
    using OrderCallback = std::function<void(const Order&)>;

    virtual void subscribe_ticker(const std::string& symbol, TickerCallback cb) {
        (void)symbol; (void)cb;  // Default: no streaming
    }
    virtual void subscribe_trades(const std::string& symbol, TradeCallback cb) {
        (void)symbol; (void)cb;
    }
    virtual void subscribe_orderbook(const std::string& symbol, OrderbookCallback cb) {
        (void)symbol; (void)cb;
    }
    virtual void subscribe_orders(OrderCallback cb) {
        (void)cb;
    }
    virtual void unsubscribe_all() {}
};

// Adapter factory type
using AdapterFactory = std::function<std::unique_ptr<VenueAdapter>()>;

}  // namespace lx::trading
