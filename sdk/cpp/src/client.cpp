// LX C++ SDK - Client Implementation
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#include "lxdex/client.hpp"

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <thread>
#include <queue>
#include <condition_variable>
#include <random>
#include <sstream>
#include <iomanip>

namespace lxdex {

using WsClient = websocketpp::client<websocketpp::config::asio_client>;
using ConnectionHdl = websocketpp::connection_hdl;
using MessagePtr = websocketpp::config::asio_client::message_type::ptr;

//------------------------------------------------------------------------------
// Client::Impl
//------------------------------------------------------------------------------

class Client::Impl {
public:
    explicit Impl(ClientConfig config)
        : config_(std::move(config))
        , state_(ConnectionState::Disconnected)
        , authenticated_(false)
        , request_id_(0)
        , running_(false)
    {
        ws_client_.clear_access_channels(websocketpp::log::alevel::all);
        ws_client_.clear_error_channels(websocketpp::log::elevel::all);

        ws_client_.init_asio();

        ws_client_.set_open_handler([this](ConnectionHdl hdl) {
            on_open(hdl);
        });

        ws_client_.set_close_handler([this](ConnectionHdl hdl) {
            on_close(hdl);
        });

        ws_client_.set_fail_handler([this](ConnectionHdl hdl) {
            on_fail(hdl);
        });

        ws_client_.set_message_handler([this](ConnectionHdl hdl, MessagePtr msg) {
            on_message(hdl, msg);
        });
    }

    ~Impl() {
        disconnect();
    }

    Error connect() {
        if (state_ == ConnectionState::Connected) {
            return {};
        }

        state_ = ConnectionState::Connecting;

        websocketpp::lib::error_code ec;
        auto con = ws_client_.get_connection(config_.ws_url, ec);

        if (ec) {
            state_ = ConnectionState::Failed;
            return Error{-1, "Failed to create connection: " + ec.message()};
        }

        connection_ = con->get_handle();
        ws_client_.connect(con);

        // Start IO thread
        running_ = true;
        io_thread_ = std::thread([this]() {
            ws_client_.run();
        });

        // Wait for connection with timeout
        {
            std::unique_lock<std::mutex> lock(connect_mutex_);
            auto result = connect_cv_.wait_for(
                lock,
                config_.connect_timeout,
                [this]() {
                    return state_ == ConnectionState::Connected ||
                           state_ == ConnectionState::Failed;
                }
            );

            if (!result) {
                state_ = ConnectionState::Failed;
                return Error{-2, "Connection timeout"};
            }
        }

        if (state_ != ConnectionState::Connected) {
            return Error{-3, "Connection failed"};
        }

        // Notify callback
        if (connection_callback_) {
            connection_callback_(state_);
        }

        return {};
    }

    void disconnect() {
        if (state_ == ConnectionState::Disconnected) {
            return;
        }

        running_ = false;
        state_ = ConnectionState::Disconnected;
        authenticated_ = false;

        try {
            if (!connection_.expired()) {
                ws_client_.close(connection_, websocketpp::close::status::normal, "");
            }
        } catch (...) {
            // Ignore close errors
        }

        ws_client_.stop();

        if (io_thread_.joinable()) {
            io_thread_.join();
        }

        // Notify callback
        if (connection_callback_) {
            connection_callback_(state_);
        }
    }

    bool is_connected() const noexcept {
        return state_ == ConnectionState::Connected;
    }

    ConnectionState state() const noexcept {
        return state_;
    }

    Error authenticate() {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        if (config_.api_key.empty() || config_.api_secret.empty()) {
            return Error{-2, "API credentials not configured"};
        }

        nlohmann::json msg = {
            {"type", "auth"},
            {"apiKey", config_.api_key},
            {"apiSecret", config_.api_secret},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return result.error;
        }

        if (result.value.contains("type") && result.value["type"] == "auth_success") {
            authenticated_ = true;
            return {};
        }

        return Error{-3, "Authentication failed"};
    }

    bool is_authenticated() const noexcept {
        return authenticated_;
    }

    Result<OrderResponse> place_order(const Order& order) {
        if (!is_connected()) {
            return {{}, Error{-1, "Not connected"}};
        }

        if (!is_authenticated()) {
            return {{}, Error{-2, "Not authenticated"}};
        }

        nlohmann::json order_data = {
            {"symbol", order.symbol},
            {"type", order.type},
            {"side", order.side},
            {"price", order.price},
            {"size", order.size}
        };

        if (!order.client_id.empty()) {
            order_data["clientId"] = order.client_id;
        }
        if (order.time_in_force != TimeInForce::GTC) {
            order_data["timeInForce"] = order.time_in_force;
        }
        if (order.post_only) {
            order_data["postOnly"] = true;
        }
        if (order.reduce_only) {
            order_data["reduceOnly"] = true;
        }

        std::string req_id = next_request_id();
        nlohmann::json msg = {
            {"type", "place_order"},
            {"order", order_data},
            {"request_id", req_id}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return {{}, result.error};
        }

        metrics_.orders_sent++;

        OrderResponse resp;
        if (result.value.contains("data")) {
            auto& data = result.value["data"];
            if (data.contains("order") && data["order"].contains("ID")) {
                resp.order_id = data["order"]["ID"].get<uint64_t>();
            }
            if (data.contains("status")) {
                resp.status = data["status"].get<std::string>();
            }
        }

        return {resp, {}};
    }

    std::future<Result<OrderResponse>> place_order_async(const Order& order) {
        return std::async(std::launch::async, [this, order]() {
            return place_order(order);
        });
    }

    Error cancel_order(uint64_t order_id) {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        if (!is_authenticated()) {
            return Error{-2, "Not authenticated"};
        }

        nlohmann::json msg = {
            {"type", "cancel_order"},
            {"orderID", order_id},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return result.error;
        }

        return {};
    }

    std::future<Error> cancel_order_async(uint64_t order_id) {
        return std::async(std::launch::async, [this, order_id]() {
            return cancel_order(order_id);
        });
    }

    Error modify_order(uint64_t order_id, double new_price, double new_size) {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        if (!is_authenticated()) {
            return Error{-2, "Not authenticated"};
        }

        nlohmann::json msg = {
            {"type", "modify_order"},
            {"orderID", order_id},
            {"request_id", next_request_id()}
        };

        if (new_price > 0) {
            msg["newPrice"] = new_price;
        }
        if (new_size > 0) {
            msg["newSize"] = new_size;
        }

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        return result.error;
    }

    Result<int> cancel_all_orders(const std::string& symbol) {
        // Cancel all would need to get orders first, then cancel each
        auto orders_result = get_orders();
        if (!orders_result.ok()) {
            return {0, orders_result.error};
        }

        int cancelled = 0;
        for (const auto& order : orders_result.value) {
            if (symbol.empty() || order.symbol == symbol) {
                auto err = cancel_order(order.order_id);
                if (!err) {
                    cancelled++;
                }
            }
        }

        return {cancelled, {}};
    }

    Error subscribe_orderbook(
        const std::vector<std::string>& symbols,
        OrderBookCallback callback
    ) {
        orderbook_callback_ = std::move(callback);
        return subscribe("orderbook", symbols);
    }

    Error subscribe_trades(
        const std::vector<std::string>& symbols,
        TradeCallback callback
    ) {
        trade_callback_ = std::move(callback);
        return subscribe("trades", symbols);
    }

    Error subscribe(
        const std::string& channel,
        const std::vector<std::string>& symbols
    ) {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        nlohmann::json msg = {
            {"type", "subscribe"},
            {"channel", channel},
            {"symbols", symbols},
            {"request_id", next_request_id()}
        };

        return send(msg);
    }

    Error unsubscribe(
        const std::string& channel,
        const std::vector<std::string>& symbols
    ) {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        nlohmann::json msg = {
            {"type", "unsubscribe"},
            {"channel", channel},
            {"symbols", symbols},
            {"request_id", next_request_id()}
        };

        return send(msg);
    }

    Result<std::vector<Balance>> get_balances() {
        if (!is_connected()) {
            return {{}, Error{-1, "Not connected"}};
        }

        if (!is_authenticated()) {
            return {{}, Error{-2, "Not authenticated"}};
        }

        nlohmann::json msg = {
            {"type", "get_balances"},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return {{}, result.error};
        }

        std::vector<Balance> balances;
        if (result.value.contains("data") && result.value["data"].contains("balances")) {
            auto& bal_data = result.value["data"]["balances"];
            for (auto& [asset, amount] : bal_data.items()) {
                Balance b;
                b.asset = asset;
                if (amount.is_string()) {
                    b.available = std::stod(amount.get<std::string>());
                } else {
                    b.available = amount.get<double>();
                }
                b.total = b.available;
                balances.push_back(b);
            }
        }

        return {balances, {}};
    }

    Result<std::vector<Position>> get_positions() {
        if (!is_connected()) {
            return {{}, Error{-1, "Not connected"}};
        }

        if (!is_authenticated()) {
            return {{}, Error{-2, "Not authenticated"}};
        }

        nlohmann::json msg = {
            {"type", "get_positions"},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return {{}, result.error};
        }

        std::vector<Position> positions;
        if (result.value.contains("data") && result.value["data"].contains("positions")) {
            for (auto& pos_data : result.value["data"]["positions"]) {
                Position p;
                if (pos_data.contains("symbol")) p.symbol = pos_data["symbol"];
                if (pos_data.contains("size")) p.size = pos_data["size"];
                if (pos_data.contains("entryPrice")) p.entry_price = pos_data["entryPrice"];
                if (pos_data.contains("markPrice")) p.mark_price = pos_data["markPrice"];
                if (pos_data.contains("pnl")) p.pnl = pos_data["pnl"];
                if (pos_data.contains("margin")) p.margin = pos_data["margin"];
                positions.push_back(p);
            }
        }

        return {positions, {}};
    }

    Result<std::vector<Order>> get_orders() {
        if (!is_connected()) {
            return {{}, Error{-1, "Not connected"}};
        }

        if (!is_authenticated()) {
            return {{}, Error{-2, "Not authenticated"}};
        }

        nlohmann::json msg = {
            {"type", "get_orders"},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(10));
        if (!result.ok()) {
            return {{}, result.error};
        }

        std::vector<Order> orders;
        if (result.value.contains("data") && result.value["data"].contains("orders")) {
            for (auto& order_data : result.value["data"]["orders"]) {
                Order o = order_data.get<Order>();
                orders.push_back(o);
            }
        }

        return {orders, {}};
    }

    Result<OrderBook> get_orderbook(const std::string& symbol, int32_t depth) {
        // Return from local cache if available
        auto* book = orderbook_manager_.get(symbol);
        if (book) {
            return {book->get_snapshot(depth), {}};
        }

        // Otherwise subscribe and wait for first snapshot
        Error err = subscribe("orderbook", {symbol});
        if (err) {
            return {{}, err};
        }

        // Wait briefly for snapshot
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        book = orderbook_manager_.get(symbol);
        if (book) {
            return {book->get_snapshot(depth), {}};
        }

        return {{}, Error{-1, "Failed to get orderbook"}};
    }

    Result<std::vector<Trade>> get_trades(const std::string& symbol, int32_t limit) {
        return {trade_tracker_.get_by_symbol(symbol, limit), {}};
    }

    Result<NodeInfo> get_info() {
        if (!is_connected()) {
            return {{}, Error{-1, "Not connected"}};
        }

        nlohmann::json msg = {
            {"type", "ping"},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(5));
        if (!result.ok()) {
            return {{}, result.error};
        }

        NodeInfo info;
        info.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        return {info, {}};
    }

    Result<int64_t> ping() {
        if (!is_connected()) {
            return {0, Error{-1, "Not connected"}};
        }

        auto start = std::chrono::high_resolution_clock::now();

        nlohmann::json msg = {
            {"type", "ping"},
            {"request_id", next_request_id()}
        };

        auto result = send_and_wait(msg, std::chrono::seconds(5));
        if (!result.ok()) {
            return {0, result.error};
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        metrics_.last_latency_us = latency;

        return {latency, {}};
    }

    void on_error(ErrorCallback callback) {
        error_callback_ = std::move(callback);
    }

    void on_order(OrderCallback callback) {
        order_callback_ = std::move(callback);
    }

    void on_trade(TradeCallback callback) {
        trade_callback_ = std::move(callback);
    }

    void on_connection(ConnectionCallback callback) {
        connection_callback_ = std::move(callback);
    }

    void on_message(MessageCallback callback) {
        message_callback_ = std::move(callback);
    }

    OrderBookManager& orderbooks() noexcept {
        return orderbook_manager_;
    }

    OrderTracker& orders() noexcept {
        return order_tracker_;
    }

    TradeTracker& trades() noexcept {
        return trade_tracker_;
    }

    ClientMetrics metrics() const noexcept {
        return metrics_;
    }

    void reset_metrics() {
        metrics_ = {};
    }

    static std::string generate_client_id() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<uint64_t> dis;

        std::stringstream ss;
        ss << "cpp_" << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
        return ss.str();
    }

private:
    void on_open(ConnectionHdl hdl) {
        state_ = ConnectionState::Connected;
        connect_cv_.notify_all();
    }

    void on_close(ConnectionHdl hdl) {
        state_ = ConnectionState::Disconnected;
        authenticated_ = false;
        connect_cv_.notify_all();

        if (connection_callback_) {
            connection_callback_(state_);
        }

        // Auto reconnect
        if (config_.auto_reconnect && running_) {
            attempt_reconnect();
        }
    }

    void on_fail(ConnectionHdl hdl) {
        state_ = ConnectionState::Failed;
        connect_cv_.notify_all();

        if (connection_callback_) {
            connection_callback_(state_);
        }
    }

    void on_message(ConnectionHdl hdl, MessagePtr msg) {
        metrics_.messages_received++;

        try {
            auto json = nlohmann::json::parse(msg->get_payload());

            // Handle request responses
            if (json.contains("request_id")) {
                std::string req_id = json["request_id"];
                std::lock_guard<std::mutex> lock(pending_mutex_);
                auto it = pending_requests_.find(req_id);
                if (it != pending_requests_.end()) {
                    it->second.set_value(json);
                    pending_requests_.erase(it);
                }
            }

            // Handle specific message types
            if (json.contains("type")) {
                std::string type = json["type"];

                if (type == "orderbook_update" && json.contains("data")) {
                    handle_orderbook_update(json["data"]);
                } else if (type == "trade_update" && json.contains("data")) {
                    handle_trade_update(json["data"]);
                } else if (type == "order_update" && json.contains("data")) {
                    handle_order_update(json["data"]);
                } else if (type == "error") {
                    handle_error(json);
                }
            }

            // Raw message callback
            if (message_callback_) {
                Message m;
                if (json.contains("type")) m.type = json["type"];
                if (json.contains("data")) m.data = json["data"];
                if (json.contains("error")) m.error = json["error"];
                if (json.contains("request_id")) m.request_id = json["request_id"];
                if (json.contains("timestamp")) m.timestamp = json["timestamp"];
                message_callback_(m);
            }
        } catch (const std::exception& e) {
            metrics_.error_count++;
            if (error_callback_) {
                error_callback_(Error{-1, std::string("JSON parse error: ") + e.what()});
            }
        }
    }

    void handle_orderbook_update(const nlohmann::json& data) {
        if (!data.contains("symbol")) return;

        std::string symbol = data["symbol"];
        auto& book = orderbook_manager_.get_or_create(symbol);

        if (data.contains("snapshot")) {
            OrderBook snapshot = data["snapshot"].get<OrderBook>();
            snapshot.symbol = symbol;
            book.apply_snapshot(snapshot);
        }

        if (orderbook_callback_) {
            orderbook_callback_(book.get_snapshot());
        }
    }

    void handle_trade_update(const nlohmann::json& data) {
        if (data.contains("trade")) {
            Trade trade = data["trade"].get<Trade>();
            trade_tracker_.add(trade);
            metrics_.trades_received++;

            if (trade_callback_) {
                trade_callback_(trade);
            }
        }
    }

    void handle_order_update(const nlohmann::json& data) {
        if (data.contains("order")) {
            Order order = data["order"].get<Order>();
            order_tracker_.upsert(order);

            if (order_callback_) {
                order_callback_(order);
            }
        }
    }

    void handle_error(const nlohmann::json& json) {
        metrics_.error_count++;

        if (error_callback_) {
            Error err;
            if (json.contains("error")) err.message = json["error"];
            if (json.contains("request_id")) err.request_id = json["request_id"];
            err.code = -1;
            error_callback_(err);
        }
    }

    void attempt_reconnect() {
        if (metrics_.reconnect_count >= static_cast<uint64_t>(config_.max_reconnect_attempts)) {
            state_ = ConnectionState::Failed;
            return;
        }

        state_ = ConnectionState::Reconnecting;
        metrics_.reconnect_count++;

        std::this_thread::sleep_for(config_.reconnect_delay);

        ws_client_.reset();
        ws_client_.init_asio();

        connect();
    }

    Error send(const nlohmann::json& msg) {
        if (!is_connected()) {
            return Error{-1, "Not connected"};
        }

        try {
            std::string payload = msg.dump();
            ws_client_.send(connection_, payload, websocketpp::frame::opcode::text);
            metrics_.messages_sent++;
            return {};
        } catch (const std::exception& e) {
            return Error{-1, std::string("Send failed: ") + e.what()};
        }
    }

    Result<nlohmann::json> send_and_wait(
        const nlohmann::json& msg,
        std::chrono::seconds timeout
    ) {
        std::string req_id = msg["request_id"];

        std::promise<nlohmann::json> promise;
        auto future = promise.get_future();

        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_[req_id] = std::move(promise);
        }

        auto send_err = send(msg);
        if (send_err) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_.erase(req_id);
            return {{}, send_err};
        }

        auto status = future.wait_for(timeout);
        if (status == std::future_status::timeout) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_.erase(req_id);
            return {{}, Error{-2, "Request timeout"}};
        }

        try {
            return {future.get(), {}};
        } catch (const std::exception& e) {
            return {{}, Error{-3, std::string("Request failed: ") + e.what()}};
        }
    }

    std::string next_request_id() {
        return std::to_string(++request_id_);
    }

    ClientConfig config_;
    WsClient ws_client_;
    ConnectionHdl connection_;
    std::thread io_thread_;

    std::atomic<ConnectionState> state_;
    std::atomic<bool> authenticated_;
    std::atomic<uint64_t> request_id_;
    std::atomic<bool> running_;

    std::mutex connect_mutex_;
    std::condition_variable connect_cv_;

    std::mutex pending_mutex_;
    std::unordered_map<std::string, std::promise<nlohmann::json>> pending_requests_;

    // Callbacks
    ErrorCallback error_callback_;
    OrderCallback order_callback_;
    TradeCallback trade_callback_;
    OrderBookCallback orderbook_callback_;
    ConnectionCallback connection_callback_;
    MessageCallback message_callback_;

    // Local data
    OrderBookManager orderbook_manager_;
    OrderTracker order_tracker_;
    TradeTracker trade_tracker_;

    // Metrics
    mutable ClientMetrics metrics_;
};

//------------------------------------------------------------------------------
// Client public interface
//------------------------------------------------------------------------------

Client::Client(ClientConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{}

Client::~Client() = default;

Error Client::connect() {
    return impl_->connect();
}

void Client::disconnect() {
    impl_->disconnect();
}

bool Client::is_connected() const noexcept {
    return impl_->is_connected();
}

ConnectionState Client::state() const noexcept {
    return impl_->state();
}

Error Client::authenticate() {
    return impl_->authenticate();
}

bool Client::is_authenticated() const noexcept {
    return impl_->is_authenticated();
}

Result<OrderResponse> Client::place_order(const Order& order) {
    return impl_->place_order(order);
}

std::future<Result<OrderResponse>> Client::place_order_async(const Order& order) {
    return impl_->place_order_async(order);
}

Error Client::cancel_order(uint64_t order_id) {
    return impl_->cancel_order(order_id);
}

std::future<Error> Client::cancel_order_async(uint64_t order_id) {
    return impl_->cancel_order_async(order_id);
}

Error Client::modify_order(uint64_t order_id, double new_price, double new_size) {
    return impl_->modify_order(order_id, new_price, new_size);
}

Result<int> Client::cancel_all_orders(const std::string& symbol) {
    return impl_->cancel_all_orders(symbol);
}

Result<OrderBook> Client::get_orderbook(const std::string& symbol, int32_t depth) {
    return impl_->get_orderbook(symbol, depth);
}

Result<std::vector<Trade>> Client::get_trades(const std::string& symbol, int32_t limit) {
    return impl_->get_trades(symbol, limit);
}

Error Client::subscribe_orderbook(
    const std::vector<std::string>& symbols,
    OrderBookCallback callback
) {
    return impl_->subscribe_orderbook(symbols, std::move(callback));
}

Error Client::subscribe_trades(
    const std::vector<std::string>& symbols,
    TradeCallback callback
) {
    return impl_->subscribe_trades(symbols, std::move(callback));
}

Error Client::unsubscribe(
    const std::string& channel,
    const std::vector<std::string>& symbols
) {
    return impl_->unsubscribe(channel, symbols);
}

Result<std::vector<Balance>> Client::get_balances() {
    return impl_->get_balances();
}

Result<std::vector<Position>> Client::get_positions() {
    return impl_->get_positions();
}

Result<std::vector<Order>> Client::get_orders() {
    return impl_->get_orders();
}

Result<NodeInfo> Client::get_info() {
    return impl_->get_info();
}

void Client::on_error(ErrorCallback callback) {
    impl_->on_error(std::move(callback));
}

void Client::on_order(OrderCallback callback) {
    impl_->on_order(std::move(callback));
}

void Client::on_trade(TradeCallback callback) {
    impl_->on_trade(std::move(callback));
}

void Client::on_connection(ConnectionCallback callback) {
    impl_->on_connection(std::move(callback));
}

void Client::on_message(MessageCallback callback) {
    impl_->on_message(std::move(callback));
}

OrderBookManager& Client::orderbooks() noexcept {
    return impl_->orderbooks();
}

OrderTracker& Client::orders() noexcept {
    return impl_->orders();
}

TradeTracker& Client::trades() noexcept {
    return impl_->trades();
}

ClientMetrics Client::metrics() const noexcept {
    return impl_->metrics();
}

void Client::reset_metrics() {
    impl_->reset_metrics();
}

Result<int64_t> Client::ping() {
    return impl_->ping();
}

std::string Client::generate_client_id() {
    return Impl::generate_client_id();
}

} // namespace lxdex
