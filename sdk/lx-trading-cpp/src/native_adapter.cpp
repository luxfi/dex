// LX Trading SDK - Native Adapters Implementation

#include <lx/trading/adapters/native.hpp>
#include <lx/trading/orderbook.hpp>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

namespace lx::trading {

using json = nlohmann::json;

// HTTP client wrapper
class HttpClient {
public:
    explicit HttpClient(const std::string& base_url) : base_url_(base_url) {}

    json get(const std::string& path,
             const std::optional<std::string>& api_key = std::nullopt) {
        cpr::Header headers;
        if (api_key) {
            headers["X-API-KEY"] = *api_key;
            headers["X-TIMESTAMP"] = std::to_string(now_ms());
        }

        auto response = cpr::Get(
            cpr::Url{base_url_ + path},
            headers);

        if (response.status_code != 200) {
            throw AdapterError("HTTP " + std::to_string(response.status_code) +
                              ": " + response.text);
        }

        return json::parse(response.text);
    }

    json post(const std::string& path,
              const json& body,
              const std::optional<std::string>& api_key = std::nullopt) {
        cpr::Header headers{{"Content-Type", "application/json"}};
        if (api_key) {
            headers["X-API-KEY"] = *api_key;
            headers["X-TIMESTAMP"] = std::to_string(now_ms());
        }

        auto response = cpr::Post(
            cpr::Url{base_url_ + path},
            headers,
            cpr::Body{body.dump()});

        if (response.status_code != 200 && response.status_code != 201) {
            throw AdapterError("HTTP " + std::to_string(response.status_code) +
                              ": " + response.text);
        }

        return json::parse(response.text);
    }

    json del(const std::string& path,
             const json& body = {},
             const std::optional<std::string>& api_key = std::nullopt) {
        cpr::Header headers{{"Content-Type", "application/json"}};
        if (api_key) {
            headers["X-API-KEY"] = *api_key;
            headers["X-TIMESTAMP"] = std::to_string(now_ms());
        }

        auto response = cpr::Delete(
            cpr::Url{base_url_ + path},
            headers,
            cpr::Body{body.dump()});

        if (response.status_code != 200) {
            throw AdapterError("HTTP " + std::to_string(response.status_code) +
                              ": " + response.text);
        }

        return json::parse(response.text);
    }

private:
    std::string base_url_;
};

// =============================================================================
// LxDexAdapter Implementation
// =============================================================================

LxDexAdapter::LxDexAdapter(std::string_view name, const NativeVenueConfig& config)
    : name_(name), config_(config), capabilities_(VenueCapabilities::clob()) {
    http_ = std::make_unique<HttpClient>(config.api_url);
}

LxDexAdapter::~LxDexAdapter() = default;

void LxDexAdapter::update_latency(int64_t start_ns) {
    int64_t elapsed = now_ns() - start_ns;
    latency_.store(static_cast<int>(elapsed / 1000000), std::memory_order_release);
}

std::future<void> LxDexAdapter::connect() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        http_->get("/api/v1/health");
        update_latency(start);
        connected_.store(true, std::memory_order_release);
    });
}

std::future<void> LxDexAdapter::disconnect() {
    return std::async(std::launch::async, [this]() {
        connected_.store(false, std::memory_order_release);
    });
}

std::future<std::vector<MarketInfo>> LxDexAdapter::get_markets() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/markets", config_.api_key);
        update_latency(start);

        std::vector<MarketInfo> markets;
        for (const auto& m : data) {
            MarketInfo info;
            info.symbol = m["symbol"];
            info.base = m["base"];
            info.quote = m["quote"];
            info.price_precision = m.value("pricePrecision", 8);
            info.quantity_precision = m.value("quantityPrecision", 8);
            info.min_quantity = Decimal::from_string(m.value("minQuantity", "0"));
            if (m.contains("maxQuantity") && !m["maxQuantity"].is_null()) {
                info.max_quantity = Decimal::from_string(m["maxQuantity"].get<std::string>());
            }
            if (m.contains("minNotional") && !m["minNotional"].is_null()) {
                info.min_notional = Decimal::from_string(m["minNotional"].get<std::string>());
            }
            info.tick_size = Decimal::from_string(m.value("tickSize", "0.00000001"));
            info.lot_size = Decimal::from_string(m.value("lotSize", "0.00000001"));
            markets.push_back(std::move(info));
        }
        return markets;
    });
}

std::future<Ticker> LxDexAdapter::get_ticker(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/ticker/" + symbol, config_.api_key);
        update_latency(start);

        Ticker ticker;
        ticker.symbol = data.value("symbol", symbol);
        ticker.venue = name_;
        if (data.contains("bid") && !data["bid"].is_null())
            ticker.bid = Decimal::from_string(data["bid"].get<std::string>());
        if (data.contains("ask") && !data["ask"].is_null())
            ticker.ask = Decimal::from_string(data["ask"].get<std::string>());
        if (data.contains("last") && !data["last"].is_null())
            ticker.last = Decimal::from_string(data["last"].get<std::string>());
        if (data.contains("volume24h") && !data["volume24h"].is_null())
            ticker.volume_24h = Decimal::from_string(data["volume24h"].get<std::string>());
        ticker.timestamp = data.value("timestamp", now_ms());
        return ticker;
    });
}

std::future<std::unique_ptr<Orderbook>> LxDexAdapter::get_orderbook(
    const std::string& symbol, std::optional<int> depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        auto start = now_ns();
        std::string path = "/api/v1/orderbook/" + symbol;
        if (depth) path += "?depth=" + std::to_string(*depth);

        auto data = http_->get(path, config_.api_key);
        update_latency(start);

        auto book = std::make_unique<Orderbook>(symbol, name_);

        for (const auto& bid : data["bids"]) {
            book->add_bid(
                Decimal::from_string(bid[0].get<std::string>()),
                Decimal::from_string(bid[1].get<std::string>()));
        }

        for (const auto& ask : data["asks"]) {
            book->add_ask(
                Decimal::from_string(ask[0].get<std::string>()),
                Decimal::from_string(ask[1].get<std::string>()));
        }

        book->sort();
        return book;
    });
}

std::future<std::vector<Trade>> LxDexAdapter::get_trades(
    const std::string& symbol, std::optional<int> limit) {
    return std::async(std::launch::async, [this, symbol, limit]() {
        auto start = now_ns();
        std::string path = "/api/v1/trades/" + symbol;
        if (limit) path += "?limit=" + std::to_string(*limit);

        auto data = http_->get(path, config_.api_key);
        update_latency(start);

        std::vector<Trade> trades;
        for (const auto& t : data) {
            Trade trade;
            trade.trade_id = t["id"];
            trade.order_id = t.value("orderId", "");
            trade.symbol = symbol;
            trade.venue = name_;
            trade.side = (t["side"] == "buy") ? Side::Buy : Side::Sell;
            trade.price = Decimal::from_string(t["price"].get<std::string>());
            trade.quantity = Decimal::from_string(t["quantity"].get<std::string>());
            trade.fee.asset = t.value("feeAsset", "");
            trade.fee.amount = Decimal::from_string(t.value("feeAmount", "0"));
            trade.timestamp = t["timestamp"];
            trade.is_maker = t.value("isMaker", false);
            trades.push_back(std::move(trade));
        }
        return trades;
    });
}

std::future<std::vector<Balance>> LxDexAdapter::get_balances() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/account/balances", config_.api_key);
        update_latency(start);

        std::vector<Balance> balances;
        for (const auto& b : data) {
            Balance bal;
            bal.asset = b["asset"];
            bal.venue = name_;
            bal.free = Decimal::from_string(b["free"].get<std::string>());
            bal.locked = Decimal::from_string(b["locked"].get<std::string>());
            balances.push_back(std::move(bal));
        }
        return balances;
    });
}

std::future<Balance> LxDexAdapter::get_balance(const std::string& asset) {
    return std::async(std::launch::async, [this, asset]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/account/balance/" + asset, config_.api_key);
        update_latency(start);

        Balance bal;
        bal.asset = data["asset"];
        bal.venue = name_;
        bal.free = Decimal::from_string(data["free"].get<std::string>());
        bal.locked = Decimal::from_string(data["locked"].get<std::string>());
        return bal;
    });
}

std::future<std::vector<Order>> LxDexAdapter::get_open_orders(
    const std::optional<std::string>& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto start = now_ns();
        std::string path = "/api/v1/orders?status=open";
        if (symbol) path += "&symbol=" + *symbol;

        auto data = http_->get(path, config_.api_key);
        update_latency(start);

        std::vector<Order> orders;
        for (const auto& o : data) {
            orders.push_back(convert_order(o));
        }
        return orders;
    });
}

Order LxDexAdapter::convert_order(const json& o) {
    Order order;
    order.order_id = o["orderId"];
    order.client_order_id = o.value("clientOrderId", "");
    order.symbol = o["symbol"];
    order.venue = name_;
    order.side = (o["side"] == "buy") ? Side::Buy : Side::Sell;

    std::string type_str = o.value("type", "limit");
    if (type_str == "market") order.order_type = OrderType::Market;
    else if (type_str == "limit") order.order_type = OrderType::Limit;
    else if (type_str == "stop_loss") order.order_type = OrderType::StopLoss;
    else if (type_str == "stop_loss_limit") order.order_type = OrderType::StopLossLimit;

    std::string status_str = o.value("status", "open");
    if (status_str == "pending") order.status = OrderStatus::Pending;
    else if (status_str == "open") order.status = OrderStatus::Open;
    else if (status_str == "partially_filled") order.status = OrderStatus::PartiallyFilled;
    else if (status_str == "filled") order.status = OrderStatus::Filled;
    else if (status_str == "cancelled") order.status = OrderStatus::Cancelled;
    else if (status_str == "rejected") order.status = OrderStatus::Rejected;
    else if (status_str == "expired") order.status = OrderStatus::Expired;

    order.quantity = Decimal::from_string(o.value("quantity", "0"));
    order.filled_quantity = Decimal::from_string(o.value("filledQuantity", "0"));
    order.remaining_quantity = order.quantity - order.filled_quantity;

    if (o.contains("price") && !o["price"].is_null())
        order.price = Decimal::from_string(o["price"].get<std::string>());
    if (o.contains("averagePrice") && !o["averagePrice"].is_null())
        order.average_price = Decimal::from_string(o["averagePrice"].get<std::string>());

    order.created_at = o.value("createdAt", int64_t(0));
    order.updated_at = o.value("updatedAt", int64_t(0));

    return order;
}

std::future<Order> LxDexAdapter::place_order(const OrderRequest& request) {
    return std::async(std::launch::async, [this, request]() -> Order {
        auto start = now_ns();

        json body = {
            {"clientOrderId", request.client_order_id},
            {"symbol", request.symbol},
            {"side", to_string(request.side)},
            {"type", to_string(request.order_type)},
            {"quantity", request.quantity.to_string()},
            {"timeInForce", to_string(request.time_in_force)}
        };

        if (request.price) {
            body["price"] = request.price->to_string();
        }

        auto data = http_->post("/api/v1/orders", body, config_.api_key);
        update_latency(start);

        return convert_order(data);
    });
}

std::future<Order> LxDexAdapter::cancel_order(
    const std::string& order_id, const std::string& symbol) {
    return std::async(std::launch::async, [this, order_id, symbol]() -> Order {
        auto start = now_ns();
        json body = {{"symbol", symbol}};
        auto data = http_->del("/api/v1/orders/" + order_id, body, config_.api_key);
        update_latency(start);
        return convert_order(data);
    });
}

std::future<std::vector<Order>> LxDexAdapter::cancel_all_orders(
    const std::optional<std::string>& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto start = now_ns();
        json body = {};
        if (symbol) body["symbol"] = *symbol;

        auto data = http_->del("/api/v1/orders/all", body, config_.api_key);
        update_latency(start);

        std::vector<Order> orders;
        for (const auto& o : data) {
            orders.push_back(convert_order(o));
        }
        return orders;
    });
}

// =============================================================================
// LxAmmAdapter Implementation
// =============================================================================

LxAmmAdapter::LxAmmAdapter(std::string_view name, const NativeVenueConfig& config)
    : name_(name), config_(config), capabilities_(VenueCapabilities::amm()) {
    http_ = std::make_unique<HttpClient>(config.api_url);
}

LxAmmAdapter::~LxAmmAdapter() = default;

void LxAmmAdapter::update_latency(int64_t start_ns) {
    int64_t elapsed = now_ns() - start_ns;
    latency_.store(static_cast<int>(elapsed / 1000000), std::memory_order_release);
}

std::future<void> LxAmmAdapter::connect() {
    return std::async(std::launch::async, [this]() {
        connected_.store(true, std::memory_order_release);
    });
}

std::future<void> LxAmmAdapter::disconnect() {
    return std::async(std::launch::async, [this]() {
        connected_.store(false, std::memory_order_release);
    });
}

std::future<std::vector<MarketInfo>> LxAmmAdapter::get_markets() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/amm/pools");
        update_latency(start);

        std::vector<MarketInfo> markets;
        for (const auto& p : data) {
            MarketInfo info;
            std::string base = p["baseToken"];
            std::string quote = p["quoteToken"];
            info.symbol = base + "-" + quote;
            info.base = base;
            info.quote = quote;
            info.price_precision = 8;
            info.quantity_precision = 8;
            info.tick_size = Decimal::from_double(0.00000001);
            info.lot_size = Decimal::from_double(0.00000001);
            markets.push_back(std::move(info));
        }
        return markets;
    });
}

std::future<Ticker> LxAmmAdapter::get_ticker(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto pair = TradingPair::from_symbol(symbol);
        if (!pair) throw AdapterError("Invalid symbol: " + symbol);

        std::string base(pair->base.data());
        std::string quote(pair->quote.data());

        auto start = now_ns();
        auto data = http_->get("/api/v1/amm/price/" + base + "/" + quote);
        update_latency(start);

        Decimal price = Decimal::from_string(data.value("price", "0"));

        Ticker ticker;
        ticker.symbol = symbol;
        ticker.venue = name_;
        ticker.bid = price;
        ticker.ask = price;
        ticker.last = price;
        ticker.timestamp = now_ms();
        return ticker;
    });
}

std::future<std::unique_ptr<Orderbook>> LxAmmAdapter::get_orderbook(
    const std::string& symbol, std::optional<int> depth) {
    (void)symbol; (void)depth;
    return std::async(std::launch::deferred, []() -> std::unique_ptr<Orderbook> {
        throw AdapterError("AMM does not have orderbook");
    });
}

std::future<std::vector<Trade>> LxAmmAdapter::get_trades(
    const std::string& symbol, std::optional<int> limit) {
    return std::async(std::launch::async, [this, symbol, limit]() {
        auto pair = TradingPair::from_symbol(symbol);
        if (!pair) return std::vector<Trade>{};

        std::string base(pair->base.data());
        std::string quote(pair->quote.data());

        std::string path = "/api/v1/amm/swaps/" + base + "/" + quote;
        if (limit) path += "?limit=" + std::to_string(*limit);

        auto start = now_ns();
        auto data = http_->get(path);
        update_latency(start);

        std::vector<Trade> trades;
        for (const auto& t : data) {
            Trade trade;
            trade.trade_id = t["txHash"];
            trade.order_id = t["txHash"];
            trade.symbol = symbol;
            trade.venue = name_;
            trade.side = (t.value("side", "buy") == "buy") ? Side::Buy : Side::Sell;
            trade.price = Decimal::from_string(t["price"].get<std::string>());
            trade.quantity = Decimal::from_string(t["amount"].get<std::string>());
            trade.fee.amount = Decimal::from_string(t.value("fee", "0"));
            trade.timestamp = t["timestamp"];
            trade.is_maker = false;
            trades.push_back(std::move(trade));
        }
        return trades;
    });
}

std::future<std::vector<Balance>> LxAmmAdapter::get_balances() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/account/balances");
        update_latency(start);

        std::vector<Balance> balances;
        for (const auto& b : data) {
            Balance bal;
            bal.asset = b["asset"];
            bal.venue = name_;
            bal.free = Decimal::from_string(b["free"].get<std::string>());
            bal.locked = Decimal::from_string(b.value("locked", "0"));
            balances.push_back(std::move(bal));
        }
        return balances;
    });
}

std::future<Balance> LxAmmAdapter::get_balance(const std::string& asset) {
    return std::async(std::launch::async, [this, asset]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/account/balance/" + asset);
        update_latency(start);

        Balance bal;
        bal.asset = data["asset"];
        bal.venue = name_;
        bal.free = Decimal::from_string(data["free"].get<std::string>());
        bal.locked = Decimal::from_string(data.value("locked", "0"));
        return bal;
    });
}

std::future<std::vector<Order>> LxAmmAdapter::get_open_orders(
    const std::optional<std::string>& symbol) {
    (void)symbol;
    return std::async(std::launch::deferred, []() { return std::vector<Order>{}; });
}

std::future<Order> LxAmmAdapter::place_order(const OrderRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        auto pair = TradingPair::from_symbol(request.symbol);
        if (!pair) throw AdapterError("Invalid symbol: " + request.symbol);

        std::string base(pair->base.data());
        std::string quote(pair->quote.data());

        auto trade = execute_swap(
            base, quote, request.quantity,
            request.side == Side::Buy,
            Decimal::from_double(0.01)).get();

        Order order;
        order.order_id = trade.trade_id;
        order.client_order_id = request.client_order_id;
        order.symbol = request.symbol;
        order.venue = name_;
        order.side = request.side;
        order.order_type = OrderType::Market;
        order.status = OrderStatus::Filled;
        order.quantity = request.quantity;
        order.filled_quantity = trade.quantity;
        order.remaining_quantity = Decimal::zero();
        order.price = trade.price;
        order.average_price = trade.price;
        order.created_at = trade.timestamp;
        order.updated_at = trade.timestamp;
        order.fees.push_back(trade.fee);
        return order;
    });
}

std::future<Order> LxAmmAdapter::cancel_order(
    const std::string& order_id, const std::string& symbol) {
    (void)order_id; (void)symbol;
    return std::async(std::launch::deferred, []() -> Order {
        throw AdapterError("AMM swaps cannot be cancelled");
    });
}

std::future<std::vector<Order>> LxAmmAdapter::cancel_all_orders(
    const std::optional<std::string>& symbol) {
    (void)symbol;
    return std::async(std::launch::deferred, []() { return std::vector<Order>{}; });
}

std::future<SwapQuote> LxAmmAdapter::get_swap_quote(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal amount,
    bool is_buy) {
    return std::async(std::launch::async, [this, base_token, quote_token, amount, is_buy]() {
        auto start = now_ns();

        json body = {
            {"baseToken", base_token},
            {"quoteToken", quote_token},
            {"amount", amount.to_string()},
            {"side", is_buy ? "buy" : "sell"}
        };

        auto data = http_->post("/api/v1/amm/quote", body);
        update_latency(start);

        SwapQuote quote;
        quote.base_token = base_token;
        quote.quote_token = quote_token;
        quote.input_amount = amount;
        quote.output_amount = Decimal::from_string(data["outputAmount"].get<std::string>());
        quote.price = Decimal::from_string(data["price"].get<std::string>());
        quote.price_impact = Decimal::from_string(data.value("priceImpact", "0"));
        quote.fee = Decimal::from_string(data.value("fee", "0"));
        quote.expires_at = now_ms() + 60000;
        return quote;
    });
}

std::future<Trade> LxAmmAdapter::execute_swap(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal amount,
    bool is_buy,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, base_token, quote_token, amount, is_buy, slippage]() {
        auto start = now_ns();

        json body = {
            {"baseToken", base_token},
            {"quoteToken", quote_token},
            {"amount", amount.to_string()},
            {"side", is_buy ? "buy" : "sell"},
            {"slippage", slippage.to_string()}
        };

        auto data = http_->post("/api/v1/amm/swap", body);
        update_latency(start);

        Trade trade;
        trade.trade_id = data["txHash"];
        trade.order_id = data["txHash"];
        trade.symbol = base_token + "-" + quote_token;
        trade.venue = name_;
        trade.side = is_buy ? Side::Buy : Side::Sell;
        trade.price = Decimal::from_string(data["price"].get<std::string>());
        trade.quantity = amount;
        trade.fee.amount = Decimal::from_string(data.value("fee", "0"));
        trade.timestamp = now_ms();
        trade.is_maker = false;
        return trade;
    });
}

std::future<PoolInfo> LxAmmAdapter::get_pool_info(
    const std::string& base_token,
    const std::string& quote_token) {
    return std::async(std::launch::async, [this, base_token, quote_token]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/amm/pool/" + base_token + "/" + quote_token);
        update_latency(start);

        PoolInfo info;
        info.address = data["address"];
        info.base_token = base_token;
        info.quote_token = quote_token;
        info.base_reserve = Decimal::from_string(data["baseReserve"].get<std::string>());
        info.quote_reserve = Decimal::from_string(data["quoteReserve"].get<std::string>());
        info.total_liquidity = Decimal::from_string(data.value("totalLiquidity", "0"));
        info.fee_rate = Decimal::from_string(data.value("feeRate", "0.003"));
        if (data.contains("apy") && !data["apy"].is_null()) {
            info.apy = Decimal::from_string(data["apy"].get<std::string>());
        }
        return info;
    });
}

std::future<LiquidityResult> LxAmmAdapter::add_liquidity(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal base_amount,
    Decimal quote_amount,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, base_token, quote_token, base_amount, quote_amount, slippage]() {
        auto start = now_ns();

        json body = {
            {"baseToken", base_token},
            {"quoteToken", quote_token},
            {"baseAmount", base_amount.to_string()},
            {"quoteAmount", quote_amount.to_string()},
            {"slippage", slippage.to_string()}
        };

        auto data = http_->post("/api/v1/amm/liquidity/add", body);
        update_latency(start);

        LiquidityResult result;
        result.tx_hash = data["txHash"];
        result.pool_address = data.value("poolAddress", "");
        result.base_amount = base_amount;
        result.quote_amount = quote_amount;
        result.lp_tokens = Decimal::from_string(data.value("lpTokens", "0"));
        result.share_percent = Decimal::from_string(data.value("sharePercent", "0"));
        return result;
    });
}

std::future<LiquidityResult> LxAmmAdapter::remove_liquidity(
    const std::string& pool_address,
    Decimal liquidity_amount,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, pool_address, liquidity_amount, slippage]() {
        auto start = now_ns();

        json body = {
            {"poolAddress", pool_address},
            {"liquidity", liquidity_amount.to_string()},
            {"slippage", slippage.to_string()}
        };

        auto data = http_->post("/api/v1/amm/liquidity/remove", body);
        update_latency(start);

        LiquidityResult result;
        result.tx_hash = data["txHash"];
        result.pool_address = pool_address;
        result.base_amount = Decimal::from_string(data.value("baseAmount", "0"));
        result.quote_amount = Decimal::from_string(data.value("quoteAmount", "0"));
        result.lp_tokens = liquidity_amount;
        return result;
    });
}

std::future<std::vector<LpPosition>> LxAmmAdapter::get_lp_positions() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();
        auto data = http_->get("/api/v1/amm/positions");
        update_latency(start);

        std::vector<LpPosition> positions;
        for (const auto& p : data) {
            LpPosition pos;
            pos.pool_address = p["poolAddress"];
            pos.base_token = p["baseToken"];
            pos.quote_token = p["quoteToken"];
            pos.lp_tokens = Decimal::from_string(p.value("lpTokens", "0"));
            pos.base_amount = Decimal::from_string(p["baseAmount"].get<std::string>());
            pos.quote_amount = Decimal::from_string(p["quoteAmount"].get<std::string>());
            pos.share_percent = Decimal::from_string(p.value("sharePercent", "0"));
            if (p.contains("unrealizedPnl") && !p["unrealizedPnl"].is_null()) {
                pos.unrealized_pnl = Decimal::from_string(p["unrealizedPnl"].get<std::string>());
            }
            positions.push_back(std::move(pos));
        }
        return positions;
    });
}

}  // namespace lx::trading
