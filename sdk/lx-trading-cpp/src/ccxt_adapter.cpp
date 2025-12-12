// LX Trading SDK - CCXT Adapter Implementation

#include <lx/trading/adapters/ccxt.hpp>
#include <lx/trading/orderbook.hpp>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

namespace lx::trading {

using json = nlohmann::json;

CcxtAdapter::CcxtAdapter(std::string_view name, const CcxtConfig& config)
    : name_(name), config_(config), capabilities_(VenueCapabilities::clob()) {
    capabilities_.batch_orders = false;  // CCXT REST doesn't have unified batch
}

CcxtAdapter::~CcxtAdapter() = default;

void CcxtAdapter::set_service_url(std::string_view url) {
    service_url_ = std::string(url);
}

void CcxtAdapter::update_latency(int64_t start_ns) {
    int64_t elapsed = now_ns() - start_ns;
    latency_.store(static_cast<int>(elapsed / 1000000), std::memory_order_release);
}

std::future<void> CcxtAdapter::connect() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        json body = {
            {"exchange", config_.exchange_id},
            {"apiKey", config_.api_key.value_or("")},
            {"secret", config_.api_secret.value_or("")},
            {"sandbox", config_.sandbox}
        };

        if (config_.password) {
            body["password"] = *config_.password;
        }

        auto response = cpr::Post(
            cpr::Url{service_url_ + "/connect"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        if (response.status_code != 200) {
            throw AdapterError("CCXT connect failed: " + response.text);
        }

        update_latency(start);
        connected_.store(true, std::memory_order_release);
    });
}

std::future<void> CcxtAdapter::disconnect() {
    return std::async(std::launch::async, [this]() {
        connected_.store(false, std::memory_order_release);
    });
}

std::future<std::vector<MarketInfo>> CcxtAdapter::get_markets() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        auto response = cpr::Get(
            cpr::Url{service_url_ + "/markets/" + config_.exchange_id});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get markets: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<MarketInfo> markets;

        for (const auto& m : data) {
            MarketInfo info;
            info.symbol = m["symbol"];
            info.base = m.value("base", "");
            info.quote = m.value("quote", "");
            info.price_precision = m.value("precision", json{}).value("price", 8);
            info.quantity_precision = m.value("precision", json{}).value("amount", 8);

            auto limits = m.value("limits", json{});
            auto amount_limits = limits.value("amount", json{});
            if (amount_limits.contains("min") && !amount_limits["min"].is_null()) {
                info.min_quantity = Decimal::from_double(amount_limits["min"].get<double>());
            }
            if (amount_limits.contains("max") && !amount_limits["max"].is_null()) {
                info.max_quantity = Decimal::from_double(amount_limits["max"].get<double>());
            }

            info.tick_size = Decimal::from_double(0.00000001);
            info.lot_size = Decimal::from_double(0.00000001);
            markets.push_back(std::move(info));
        }

        return markets;
    });
}

std::future<Ticker> CcxtAdapter::get_ticker(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto start = now_ns();

        auto response = cpr::Get(
            cpr::Url{service_url_ + "/ticker/" + config_.exchange_id + "/" + symbol});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get ticker: " + response.text);
        }

        auto data = json::parse(response.text);

        Ticker ticker;
        ticker.symbol = data.value("symbol", symbol);
        ticker.venue = name_;

        if (data.contains("bid") && !data["bid"].is_null())
            ticker.bid = Decimal::from_double(data["bid"].get<double>());
        if (data.contains("ask") && !data["ask"].is_null())
            ticker.ask = Decimal::from_double(data["ask"].get<double>());
        if (data.contains("last") && !data["last"].is_null())
            ticker.last = Decimal::from_double(data["last"].get<double>());
        if (data.contains("baseVolume") && !data["baseVolume"].is_null())
            ticker.volume_24h = Decimal::from_double(data["baseVolume"].get<double>());
        if (data.contains("high") && !data["high"].is_null())
            ticker.high_24h = Decimal::from_double(data["high"].get<double>());
        if (data.contains("low") && !data["low"].is_null())
            ticker.low_24h = Decimal::from_double(data["low"].get<double>());
        if (data.contains("percentage") && !data["percentage"].is_null())
            ticker.change_24h = Decimal::from_double(data["percentage"].get<double>());

        ticker.timestamp = data.value("timestamp", now_ms());

        return ticker;
    });
}

std::future<std::unique_ptr<Orderbook>> CcxtAdapter::get_orderbook(
    const std::string& symbol, std::optional<int> depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        auto start = now_ns();

        std::string url = service_url_ + "/orderbook/" + config_.exchange_id + "/" + symbol;
        if (depth) url += "?limit=" + std::to_string(*depth);

        auto response = cpr::Get(cpr::Url{url});
        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get orderbook: " + response.text);
        }

        auto data = json::parse(response.text);
        auto book = std::make_unique<Orderbook>(symbol, name_);

        for (const auto& bid : data["bids"]) {
            book->add_bid(
                Decimal::from_double(bid[0].get<double>()),
                Decimal::from_double(bid[1].get<double>()));
        }

        for (const auto& ask : data["asks"]) {
            book->add_ask(
                Decimal::from_double(ask[0].get<double>()),
                Decimal::from_double(ask[1].get<double>()));
        }

        book->sort();
        return book;
    });
}

std::future<std::vector<Trade>> CcxtAdapter::get_trades(
    const std::string& symbol, std::optional<int> limit) {
    return std::async(std::launch::async, [this, symbol, limit]() {
        auto start = now_ns();

        std::string url = service_url_ + "/trades/" + config_.exchange_id + "/" + symbol;
        if (limit) url += "?limit=" + std::to_string(*limit);

        auto response = cpr::Get(cpr::Url{url});
        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get trades: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<Trade> trades;

        for (const auto& t : data) {
            Trade trade;
            trade.trade_id = t.value("id", "");
            trade.order_id = t.value("order", "");
            trade.symbol = t.value("symbol", symbol);
            trade.venue = name_;
            trade.side = (t.value("side", "buy") == "buy") ? Side::Buy : Side::Sell;
            trade.price = Decimal::from_double(t.value("price", 0.0));
            trade.quantity = Decimal::from_double(t.value("amount", 0.0));

            auto fee = t.value("fee", json{});
            trade.fee.asset = fee.value("currency", "");
            trade.fee.amount = Decimal::from_double(fee.value("cost", 0.0));

            trade.timestamp = t.value("timestamp", int64_t(0));
            trade.is_maker = (t.value("takerOrMaker", "taker") == "maker");
            trades.push_back(std::move(trade));
        }

        return trades;
    });
}

std::future<std::vector<Balance>> CcxtAdapter::get_balances() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        json body = {
            {"exchange", config_.exchange_id},
            {"apiKey", config_.api_key.value_or("")},
            {"secret", config_.api_secret.value_or("")}
        };

        auto response = cpr::Post(
            cpr::Url{service_url_ + "/balance"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get balances: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<Balance> balances;

        auto total = data.value("total", json{});
        auto free = data.value("free", json{});
        auto used = data.value("used", json{});

        for (auto& [asset, amount] : total.items()) {
            if (amount.is_null() || amount.get<double>() <= 0) continue;

            Balance bal;
            bal.asset = asset;
            bal.venue = name_;
            bal.free = Decimal::from_double(free.value(asset, 0.0));
            bal.locked = Decimal::from_double(used.value(asset, 0.0));
            balances.push_back(std::move(bal));
        }

        return balances;
    });
}

std::future<Balance> CcxtAdapter::get_balance(const std::string& asset) {
    return std::async(std::launch::async, [this, asset]() {
        auto balances = get_balances().get();
        for (const auto& b : balances) {
            if (b.asset == asset) return b;
        }
        return Balance{asset, name_, Decimal::zero(), Decimal::zero()};
    });
}

std::future<std::vector<Order>> CcxtAdapter::get_open_orders(
    const std::optional<std::string>& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto start = now_ns();

        json body = {
            {"exchange", config_.exchange_id},
            {"apiKey", config_.api_key.value_or("")},
            {"secret", config_.api_secret.value_or("")}
        };
        if (symbol) body["symbol"] = *symbol;

        auto response = cpr::Post(
            cpr::Url{service_url_ + "/openOrders"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get open orders: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<Order> orders;

        for (const auto& o : data) {
            orders.push_back(convert_order(o));
        }

        return orders;
    });
}

Order CcxtAdapter::convert_order(const json& o) {
    Order order;
    order.order_id = o.value("id", "");
    order.client_order_id = o.value("clientOrderId", "");
    order.symbol = o.value("symbol", "");
    order.venue = name_;
    order.side = (o.value("side", "buy") == "buy") ? Side::Buy : Side::Sell;

    std::string type_str = o.value("type", "limit");
    if (type_str == "market") order.order_type = OrderType::Market;
    else if (type_str == "limit") order.order_type = OrderType::Limit;
    else if (type_str == "stop") order.order_type = OrderType::StopLoss;
    else if (type_str == "stop_limit") order.order_type = OrderType::StopLossLimit;

    std::string status_str = o.value("status", "open");
    if (status_str == "open") order.status = OrderStatus::Open;
    else if (status_str == "closed") order.status = OrderStatus::Filled;
    else if (status_str == "canceled" || status_str == "cancelled")
        order.status = OrderStatus::Cancelled;
    else if (status_str == "expired") order.status = OrderStatus::Expired;
    else if (status_str == "rejected") order.status = OrderStatus::Rejected;

    order.quantity = Decimal::from_double(o.value("amount", 0.0));
    order.filled_quantity = Decimal::from_double(o.value("filled", 0.0));
    order.remaining_quantity = order.quantity - order.filled_quantity;

    if (o.contains("price") && !o["price"].is_null())
        order.price = Decimal::from_double(o["price"].get<double>());
    if (o.contains("average") && !o["average"].is_null())
        order.average_price = Decimal::from_double(o["average"].get<double>());

    order.created_at = o.value("timestamp", int64_t(0));
    order.updated_at = o.value("lastTradeTimestamp", order.created_at);

    return order;
}

std::future<Order> CcxtAdapter::place_order(const OrderRequest& request) {
    return std::async(std::launch::async, [this, request]() -> Order {
        auto start = now_ns();

        json body = {
            {"exchange", config_.exchange_id},
            {"apiKey", config_.api_key.value_or("")},
            {"secret", config_.api_secret.value_or("")},
            {"symbol", request.symbol},
            {"side", request.side == Side::Buy ? "buy" : "sell"},
            {"type", request.order_type == OrderType::Market ? "market" : "limit"},
            {"amount", request.quantity.to_double()}
        };

        if (request.price) {
            body["price"] = request.price->to_double();
        }

        if (!request.client_order_id.empty()) {
            body["clientOrderId"] = request.client_order_id;
        }

        auto response = cpr::Post(
            cpr::Url{service_url_ + "/order"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200 && response.status_code != 201) {
            throw AdapterError("Failed to place order: " + response.text);
        }

        auto data = json::parse(response.text);
        return convert_order(data);
    });
}

std::future<Order> CcxtAdapter::cancel_order(
    const std::string& order_id, const std::string& symbol) {
    return std::async(std::launch::async, [this, order_id, symbol]() -> Order {
        auto start = now_ns();

        json body = {
            {"exchange", config_.exchange_id},
            {"apiKey", config_.api_key.value_or("")},
            {"secret", config_.api_secret.value_or("")},
            {"orderId", order_id},
            {"symbol", symbol}
        };

        auto response = cpr::Post(
            cpr::Url{service_url_ + "/cancelOrder"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to cancel order: " + response.text);
        }

        auto data = json::parse(response.text);
        return convert_order(data);
    });
}

std::future<std::vector<Order>> CcxtAdapter::cancel_all_orders(
    const std::optional<std::string>& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        // Get open orders then cancel each
        auto open = get_open_orders(symbol).get();
        std::vector<Order> cancelled;

        for (const auto& order : open) {
            try {
                auto result = cancel_order(order.order_id, order.symbol).get();
                cancelled.push_back(std::move(result));
            } catch (...) {
                // Continue with other orders
            }
        }

        return cancelled;
    });
}

}  // namespace lx::trading
