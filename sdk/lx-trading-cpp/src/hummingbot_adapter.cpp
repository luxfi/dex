// LX Trading SDK - Hummingbot Adapter Implementation

#include <lx/trading/adapters/hummingbot.hpp>
#include <lx/trading/orderbook.hpp>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

namespace lx::trading {

using json = nlohmann::json;

HummingbotAdapter::HummingbotAdapter(std::string_view name, const HummingbotConfig& config)
    : name_(name), config_(config), capabilities_(VenueCapabilities::amm()) {}

HummingbotAdapter::~HummingbotAdapter() = default;

void HummingbotAdapter::update_latency(int64_t start_ns) {
    int64_t elapsed = now_ns() - start_ns;
    latency_.store(static_cast<int>(elapsed / 1000000), std::memory_order_release);
}

json HummingbotAdapter::build_request_body() {
    json body = {
        {"chain", config_.chain},
        {"network", config_.network},
        {"connector", config_.connector}
    };

    if (config_.wallet_address) {
        body["address"] = *config_.wallet_address;
    }

    return body;
}

std::future<void> HummingbotAdapter::connect() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        auto response = cpr::Get(cpr::Url{config_.base_url()});

        if (response.status_code != 200) {
            throw AdapterError("Gateway not ready: " + response.text);
        }

        auto data = json::parse(response.text);
        if (data.value("status", "") != "ok") {
            throw AdapterError("Gateway not ready");
        }

        update_latency(start);
        connected_.store(true, std::memory_order_release);
    });
}

std::future<void> HummingbotAdapter::disconnect() {
    return std::async(std::launch::async, [this]() {
        connected_.store(false, std::memory_order_release);
    });
}

std::future<std::vector<MarketInfo>> HummingbotAdapter::get_markets() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/tokens"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{build_request_body().dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get markets: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<MarketInfo> markets;

        auto tokens = data.value("tokens", json::array());
        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = i + 1; j < tokens.size(); ++j) {
                std::string sym1 = tokens[i].value("symbol", "");
                std::string sym2 = tokens[j].value("symbol", "");
                if (sym1.empty() || sym2.empty()) continue;

                MarketInfo info;
                info.symbol = sym1 + "-" + sym2;
                info.base = sym1;
                info.quote = sym2;
                info.price_precision = 8;
                info.quantity_precision = 8;
                info.tick_size = Decimal::from_double(0.00000001);
                info.lot_size = Decimal::from_double(0.00000001);
                markets.push_back(std::move(info));
            }
        }

        return markets;
    });
}

std::future<Ticker> HummingbotAdapter::get_ticker(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        auto pair = TradingPair::from_symbol(symbol);
        if (!pair) throw AdapterError("Invalid symbol: " + symbol);

        std::string base(pair->base.data());
        std::string quote(pair->quote.data());

        auto start = now_ns();

        json body = build_request_body();
        body["base"] = base;
        body["quote"] = quote;
        body["amount"] = "1";
        body["side"] = "BUY";

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/price"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get ticker: " + response.text);
        }

        auto data = json::parse(response.text);

        Ticker ticker;
        ticker.symbol = symbol;
        ticker.venue = name_;

        if (data.contains("price") && !data["price"].is_null()) {
            auto price = Decimal::from_string(data["price"].get<std::string>());
            ticker.bid = price;
            ticker.ask = price;
            ticker.last = price;
        }

        ticker.timestamp = now_ms();
        return ticker;
    });
}

std::future<std::unique_ptr<Orderbook>> HummingbotAdapter::get_orderbook(
    const std::string& symbol, std::optional<int> depth) {
    (void)symbol; (void)depth;
    return std::async(std::launch::deferred, []() -> std::unique_ptr<Orderbook> {
        throw AdapterError("Gateway AMM does not have orderbook");
    });
}

std::future<std::vector<Trade>> HummingbotAdapter::get_trades(
    const std::string& symbol, std::optional<int> limit) {
    (void)symbol; (void)limit;
    return std::async(std::launch::deferred, []() {
        return std::vector<Trade>{};  // Gateway doesn't provide trade history
    });
}

std::future<std::vector<Balance>> HummingbotAdapter::get_balances() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/chain/balances"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{build_request_body().dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get balances: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<Balance> balances;

        auto bals = data.value("balances", json::object());
        for (auto& [asset, amount] : bals.items()) {
            Balance bal;
            bal.asset = asset;
            bal.venue = name_;
            bal.free = Decimal::from_string(amount.get<std::string>());
            bal.locked = Decimal::zero();
            balances.push_back(std::move(bal));
        }

        return balances;
    });
}

std::future<Balance> HummingbotAdapter::get_balance(const std::string& asset) {
    return std::async(std::launch::async, [this, asset]() {
        auto balances = get_balances().get();
        for (const auto& b : balances) {
            if (b.asset == asset) return b;
        }
        return Balance{asset, name_, Decimal::zero(), Decimal::zero()};
    });
}

std::future<std::vector<Order>> HummingbotAdapter::get_open_orders(
    const std::optional<std::string>& symbol) {
    (void)symbol;
    return std::async(std::launch::deferred, []() {
        return std::vector<Order>{};  // AMM doesn't have orders
    });
}

std::future<Order> HummingbotAdapter::place_order(const OrderRequest& request) {
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

std::future<Order> HummingbotAdapter::cancel_order(
    const std::string& order_id, const std::string& symbol) {
    (void)order_id; (void)symbol;
    return std::async(std::launch::deferred, []() -> Order {
        throw AdapterError("Gateway AMM swaps cannot be cancelled");
    });
}

std::future<std::vector<Order>> HummingbotAdapter::cancel_all_orders(
    const std::optional<std::string>& symbol) {
    (void)symbol;
    return std::async(std::launch::deferred, []() {
        return std::vector<Order>{};
    });
}

std::future<SwapQuote> HummingbotAdapter::get_swap_quote(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal amount,
    bool is_buy) {
    return std::async(std::launch::async,
        [this, base_token, quote_token, amount, is_buy]() {
        auto start = now_ns();

        json body = build_request_body();
        body["base"] = base_token;
        body["quote"] = quote_token;
        body["amount"] = amount.to_string();
        body["side"] = is_buy ? "BUY" : "SELL";

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/price"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get quote: " + response.text);
        }

        auto data = json::parse(response.text);

        SwapQuote quote;
        quote.base_token = base_token;
        quote.quote_token = quote_token;
        quote.input_amount = amount;
        quote.output_amount = Decimal::from_string(data.value("expectedAmount", "0"));
        quote.price = Decimal::from_string(data.value("price", "0"));
        quote.price_impact = Decimal::zero();
        quote.fee = Decimal::zero();
        quote.expires_at = now_ms() + 60000;
        return quote;
    });
}

std::future<Trade> HummingbotAdapter::execute_swap(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal amount,
    bool is_buy,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, base_token, quote_token, amount, is_buy, slippage]() {
        auto start = now_ns();

        json body = build_request_body();
        body["base"] = base_token;
        body["quote"] = quote_token;
        body["amount"] = amount.to_string();
        body["side"] = is_buy ? "BUY" : "SELL";
        body["limitPrice"] = "";
        body["allowedSlippage"] = slippage.to_string() + "/100";

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/trade"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to execute swap: " + response.text);
        }

        auto data = json::parse(response.text);

        Trade trade;
        trade.trade_id = data.value("txHash", "");
        trade.order_id = trade.trade_id;
        trade.symbol = base_token + "-" + quote_token;
        trade.venue = name_;
        trade.side = is_buy ? Side::Buy : Side::Sell;
        trade.price = Decimal::from_string(data.value("price", "0"));
        trade.quantity = amount;
        trade.fee.asset = "GAS";
        trade.fee.amount = Decimal::from_string(data.value("gasPrice", "0"));
        trade.timestamp = now_ms();
        trade.is_maker = false;
        return trade;
    });
}

std::future<PoolInfo> HummingbotAdapter::get_pool_info(
    const std::string& base_token,
    const std::string& quote_token) {
    return std::async(std::launch::async, [this, base_token, quote_token]() {
        auto start = now_ns();

        json body = build_request_body();
        body["token0"] = base_token;
        body["token1"] = quote_token;

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/poolPrice"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get pool info: " + response.text);
        }

        auto data = json::parse(response.text);

        PoolInfo info;
        info.address = data.value("token0Address", "");
        info.base_token = base_token;
        info.quote_token = quote_token;
        info.base_reserve = Decimal::from_string(data.value("token0Balance", "0"));
        info.quote_reserve = Decimal::from_string(data.value("token1Balance", "0"));
        info.total_liquidity = Decimal::zero();
        info.fee_rate = Decimal::from_double(0.003);
        return info;
    });
}

std::future<LiquidityResult> HummingbotAdapter::add_liquidity(
    const std::string& base_token,
    const std::string& quote_token,
    Decimal base_amount,
    Decimal quote_amount,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, base_token, quote_token, base_amount, quote_amount, slippage]() {
        auto start = now_ns();

        json body = build_request_body();
        body["token0"] = base_token;
        body["token1"] = quote_token;
        body["amount0"] = base_amount.to_string();
        body["amount1"] = quote_amount.to_string();
        body["allowedSlippage"] = slippage.to_string() + "/100";

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/liquidity/add"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to add liquidity: " + response.text);
        }

        auto data = json::parse(response.text);

        LiquidityResult result;
        result.tx_hash = data.value("txHash", "");
        result.pool_address = data.value("poolAddress", "");
        result.base_amount = base_amount;
        result.quote_amount = quote_amount;
        result.lp_tokens = Decimal::zero();
        result.share_percent = Decimal::zero();
        return result;
    });
}

std::future<LiquidityResult> HummingbotAdapter::remove_liquidity(
    const std::string& pool_address,
    Decimal liquidity_amount,
    Decimal slippage) {
    return std::async(std::launch::async,
        [this, pool_address, liquidity_amount, slippage]() {
        auto start = now_ns();

        json body = build_request_body();
        body["tokenId"] = pool_address;
        body["decreasePercent"] = "100";
        body["allowedSlippage"] = slippage.to_string() + "/100";

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/liquidity/remove"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to remove liquidity: " + response.text);
        }

        auto data = json::parse(response.text);

        LiquidityResult result;
        result.tx_hash = data.value("txHash", "");
        result.pool_address = pool_address;
        result.base_amount = Decimal::zero();
        result.quote_amount = Decimal::zero();
        result.lp_tokens = liquidity_amount;
        result.share_percent = Decimal::zero();
        return result;
    });
}

std::future<std::vector<LpPosition>> HummingbotAdapter::get_lp_positions() {
    return std::async(std::launch::async, [this]() {
        auto start = now_ns();

        auto response = cpr::Post(
            cpr::Url{config_.base_url() + "/amm/position"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{build_request_body().dump()});

        update_latency(start);

        if (response.status_code != 200) {
            throw AdapterError("Failed to get LP positions: " + response.text);
        }

        auto data = json::parse(response.text);
        std::vector<LpPosition> positions;

        if (data.is_array()) {
            for (const auto& p : data) {
                LpPosition pos;
                pos.pool_address = p.value("tokenId", "");
                pos.base_token = p.value("token0", "");
                pos.quote_token = p.value("token1", "");
                pos.lp_tokens = Decimal::zero();
                pos.base_amount = Decimal::from_string(p.value("amount0", "0"));
                pos.quote_amount = Decimal::from_string(p.value("amount1", "0"));
                pos.share_percent = Decimal::zero();
                if (p.contains("unclaimedToken0") && !p["unclaimedToken0"].is_null()) {
                    pos.unrealized_pnl = Decimal::from_string(
                        p["unclaimedToken0"].get<std::string>());
                }
                positions.push_back(std::move(pos));
            }
        }

        return positions;
    });
}

}  // namespace lx::trading
