// LX C++ SDK - Client
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#ifndef LXDEX_CLIENT_HPP
#define LXDEX_CLIENT_HPP

#include "types.hpp"
#include "orderbook.hpp"
#include <functional>
#include <memory>
#include <future>
#include <atomic>
#include <chrono>

namespace lxdex {

/// Connection state
enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed
};

/// Client configuration
struct ClientConfig {
    std::string ws_url = "ws://localhost:8081";
    std::string api_key;
    std::string api_secret;

    // Connection settings
    std::chrono::seconds connect_timeout{10};
    std::chrono::seconds ping_interval{30};
    std::chrono::seconds reconnect_delay{5};
    int max_reconnect_attempts = 5;
    bool auto_reconnect = true;

    // Performance settings
    size_t send_queue_size = 10000;
    size_t recv_queue_size = 10000;
};

/// Client metrics
struct ClientMetrics {
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    uint64_t orders_sent = 0;
    uint64_t trades_received = 0;
    int64_t last_latency_us = 0;
    int64_t avg_latency_us = 0;
    uint64_t reconnect_count = 0;
    uint64_t error_count = 0;
};

/// Error information
struct Error {
    int code = 0;
    std::string message;
    std::string request_id;

    explicit operator bool() const noexcept { return code != 0; }
};

/// Result type for async operations
template<typename T>
struct Result {
    T value;
    Error error;

    [[nodiscard]] bool ok() const noexcept { return !error; }
    explicit operator bool() const noexcept { return ok(); }
};

/// Callback types
using ErrorCallback = std::function<void(const Error&)>;
using OrderCallback = std::function<void(const Order&)>;
using TradeCallback = std::function<void(const Trade&)>;
using OrderBookCallback = std::function<void(const OrderBook&)>;
using MessageCallback = std::function<void(const Message&)>;
using ConnectionCallback = std::function<void(ConnectionState)>;

/// LX WebSocket Client
/// Thread-safe, RAII-compliant client for trading operations
class Client {
public:
    /// Construct client with configuration
    explicit Client(ClientConfig config = {});

    /// Destructor - ensures clean disconnection
    ~Client();

    // Non-copyable, non-movable (owns thread resources)
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

    //--------------------------------------------------------------------------
    // Connection
    //--------------------------------------------------------------------------

    /// Connect to server
    /// @return Error if connection fails
    Error connect();

    /// Disconnect from server
    void disconnect();

    /// Check if connected
    [[nodiscard]] bool is_connected() const noexcept;

    /// Get connection state
    [[nodiscard]] ConnectionState state() const noexcept;

    /// Authenticate with API credentials
    /// @return Error if authentication fails
    Error authenticate();

    /// Check if authenticated
    [[nodiscard]] bool is_authenticated() const noexcept;

    //--------------------------------------------------------------------------
    // Trading Operations
    //--------------------------------------------------------------------------

    /// Place a new order
    /// @param order Order to place
    /// @return Result with order response
    Result<OrderResponse> place_order(const Order& order);

    /// Place order asynchronously
    /// @param order Order to place
    /// @return Future with result
    std::future<Result<OrderResponse>> place_order_async(const Order& order);

    /// Cancel an order
    /// @param order_id Order ID to cancel
    /// @return Error if cancellation fails
    Error cancel_order(uint64_t order_id);

    /// Cancel order asynchronously
    std::future<Error> cancel_order_async(uint64_t order_id);

    /// Modify an existing order
    /// @param order_id Order ID to modify
    /// @param new_price New price (0 to keep current)
    /// @param new_size New size (0 to keep current)
    /// @return Error if modification fails
    Error modify_order(uint64_t order_id, double new_price, double new_size);

    /// Cancel all orders for a symbol
    /// @param symbol Symbol to cancel orders for (empty for all)
    /// @return Number of orders cancelled
    Result<int> cancel_all_orders(const std::string& symbol = "");

    //--------------------------------------------------------------------------
    // Market Data
    //--------------------------------------------------------------------------

    /// Get current orderbook snapshot
    /// @param symbol Trading pair symbol
    /// @param depth Number of levels (0 for all)
    /// @return Result with orderbook
    Result<OrderBook> get_orderbook(const std::string& symbol, int32_t depth = 20);

    /// Get recent trades
    /// @param symbol Trading pair symbol
    /// @param limit Maximum number of trades
    /// @return Result with trades
    Result<std::vector<Trade>> get_trades(const std::string& symbol, int32_t limit = 100);

    /// Subscribe to orderbook updates
    /// @param symbols Symbols to subscribe to
    /// @param callback Callback for updates
    /// @return Error if subscription fails
    Error subscribe_orderbook(
        const std::vector<std::string>& symbols,
        OrderBookCallback callback
    );

    /// Subscribe to trade updates
    /// @param symbols Symbols to subscribe to
    /// @param callback Callback for trades
    /// @return Error if subscription fails
    Error subscribe_trades(
        const std::vector<std::string>& symbols,
        TradeCallback callback
    );

    /// Unsubscribe from channel
    /// @param channel Channel name (e.g., "orderbook", "trades")
    /// @param symbols Symbols to unsubscribe from
    /// @return Error if unsubscription fails
    Error unsubscribe(
        const std::string& channel,
        const std::vector<std::string>& symbols
    );

    //--------------------------------------------------------------------------
    // Account
    //--------------------------------------------------------------------------

    /// Get account balances
    /// @return Result with balances
    Result<std::vector<Balance>> get_balances();

    /// Get open positions
    /// @return Result with positions
    Result<std::vector<Position>> get_positions();

    /// Get open orders
    /// @return Result with orders
    Result<std::vector<Order>> get_orders();

    /// Get node information
    /// @return Result with node info
    Result<NodeInfo> get_info();

    //--------------------------------------------------------------------------
    // Callbacks
    //--------------------------------------------------------------------------

    /// Set error callback
    void on_error(ErrorCallback callback);

    /// Set order update callback
    void on_order(OrderCallback callback);

    /// Set trade callback
    void on_trade(TradeCallback callback);

    /// Set connection state callback
    void on_connection(ConnectionCallback callback);

    /// Set raw message callback (for debugging)
    void on_message(MessageCallback callback);

    //--------------------------------------------------------------------------
    // Local Data
    //--------------------------------------------------------------------------

    /// Get local orderbook manager
    [[nodiscard]] OrderBookManager& orderbooks() noexcept;

    /// Get order tracker
    [[nodiscard]] OrderTracker& orders() noexcept;

    /// Get trade tracker
    [[nodiscard]] TradeTracker& trades() noexcept;

    //--------------------------------------------------------------------------
    // Metrics
    //--------------------------------------------------------------------------

    /// Get client metrics
    [[nodiscard]] ClientMetrics metrics() const noexcept;

    /// Reset metrics
    void reset_metrics();

    //--------------------------------------------------------------------------
    // Utility
    //--------------------------------------------------------------------------

    /// Send ping and measure latency
    /// @return Round-trip time in microseconds
    Result<int64_t> ping();

    /// Generate client order ID
    [[nodiscard]] static std::string generate_client_id();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convenience factory function
inline std::unique_ptr<Client> make_client(ClientConfig config = {}) {
    return std::make_unique<Client>(std::move(config));
}

} // namespace lxdex

#endif // LXDEX_CLIENT_HPP
