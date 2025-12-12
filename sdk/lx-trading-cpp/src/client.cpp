// LX Trading SDK - Unified Client Implementation

#include <lx/trading/client.hpp>
#include <lx/trading/adapters/native.hpp>
#include <lx/trading/adapters/ccxt.hpp>
#include <lx/trading/adapters/hummingbot.hpp>
#include <lx/trading/risk.hpp>

namespace lx::trading {

Client::Client(const Config& config)
    : config_(config),
      risk_manager_(std::make_unique<RiskManager>(config.risk)) {}

Client::~Client() = default;

std::future<void> Client::connect() {
    return std::async(std::launch::async, [this]() {
        // Connect native venues
        for (const auto& [name, cfg] : config_.native) {
            std::unique_ptr<VenueAdapter> adapter;
            if (cfg.venue_type == "dex") {
                adapter = std::make_unique<LxDexAdapter>(name, cfg);
            } else {
                adapter = std::make_unique<LxAmmAdapter>(name, cfg);
            }
            adapter->connect().get();
            venues_[name] = std::move(adapter);
        }

        // Connect CCXT exchanges
        for (const auto& [name, cfg] : config_.ccxt) {
            auto adapter = std::make_unique<CcxtAdapter>(name, cfg);
            adapter->connect().get();
            venues_[name] = std::move(adapter);
        }

        // Connect Hummingbot gateways
        for (const auto& [name, cfg] : config_.hummingbot) {
            auto adapter = std::make_unique<HummingbotAdapter>(name, cfg);
            adapter->connect().get();
            venues_[name] = std::move(adapter);
        }

        // Set default venue
        if (!config_.general.venue_priority.empty()) {
            default_venue_ = config_.general.venue_priority[0];
        } else if (!venues_.empty()) {
            default_venue_ = venues_.begin()->first;
        }
    });
}

std::future<void> Client::disconnect() {
    return std::async(std::launch::async, [this]() {
        for (auto& [name, adapter] : venues_) {
            adapter->disconnect().get();
        }
        venues_.clear();
        default_venue_.reset();
    });
}

VenueAdapter* Client::venue(std::string_view name) {
    auto it = venues_.find(std::string(name));
    return it != venues_.end() ? it->second.get() : nullptr;
}

const VenueAdapter* Client::venue(std::string_view name) const {
    auto it = venues_.find(std::string(name));
    return it != venues_.end() ? it->second.get() : nullptr;
}

std::vector<VenueInfo> Client::venues() const {
    std::vector<VenueInfo> result;
    for (const auto& [name, adapter] : venues_) {
        result.push_back(adapter->info());
    }
    return result;
}

VenueAdapter* Client::get_venue(std::optional<std::string_view> name) {
    std::string venue_name;
    if (name) {
        venue_name = std::string(*name);
    } else if (default_venue_) {
        venue_name = *default_venue_;
    } else {
        throw AdapterError("No venue specified and no default venue");
    }

    auto it = venues_.find(venue_name);
    if (it == venues_.end()) {
        throw AdapterError("Venue not found: " + venue_name);
    }

    return it->second.get();
}

// Market Data
std::future<std::unique_ptr<Orderbook>> Client::orderbook(
    const std::string& symbol, std::optional<std::string_view> venue) {
    return get_venue(venue)->get_orderbook(symbol);
}

std::future<AggregatedOrderbook> Client::aggregated_orderbook(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        AggregatedOrderbook agg(symbol);

        for (auto& [name, adapter] : venues_) {
            if (adapter->capabilities().orderbook) {
                try {
                    auto book = adapter->get_orderbook(symbol, 20).get();
                    agg.add_orderbook(*book);
                } catch (...) {
                    // Skip venues that don't support this pair
                }
            }
        }

        return agg;
    });
}

std::future<Ticker> Client::ticker(
    const std::string& symbol, std::optional<std::string_view> venue) {
    return get_venue(venue)->get_ticker(symbol);
}

std::future<std::vector<Ticker>> Client::tickers(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        std::vector<Ticker> result;

        for (auto& [name, adapter] : venues_) {
            try {
                auto t = adapter->get_ticker(symbol).get();
                result.push_back(std::move(t));
            } catch (...) {
                // Skip
            }
        }

        return result;
    });
}

// Account
std::future<std::vector<AggregatedBalance>> Client::balances() {
    return std::async(std::launch::async, [this]() {
        std::unordered_map<std::string, std::vector<Balance>> by_asset;

        for (auto& [name, adapter] : venues_) {
            try {
                for (auto& bal : adapter->get_balances().get()) {
                    by_asset[bal.asset].push_back(std::move(bal));
                }
            } catch (...) {
                // Skip
            }
        }

        std::vector<AggregatedBalance> result;
        for (auto& [asset, bals] : by_asset) {
            AggregatedBalance agg;
            agg.asset = asset;
            agg.by_venue = std::move(bals);

            for (const auto& b : agg.by_venue) {
                agg.total_free = agg.total_free + b.free;
                agg.total_locked = agg.total_locked + b.locked;
            }

            result.push_back(std::move(agg));
        }

        return result;
    });
}

std::future<Balance> Client::balance(
    const std::string& asset, std::optional<std::string_view> venue) {
    return get_venue(venue)->get_balance(asset);
}

// Orders
std::future<Order> Client::buy(
    const std::string& symbol, Decimal quantity, std::optional<std::string_view> venue) {
    OrderRequest req = OrderRequest::market(symbol, Side::Buy, quantity);
    if (venue) req.venue = std::string(*venue);

    return place_order(req);
}

std::future<Order> Client::sell(
    const std::string& symbol, Decimal quantity, std::optional<std::string_view> venue) {
    OrderRequest req = OrderRequest::market(symbol, Side::Sell, quantity);
    if (venue) req.venue = std::string(*venue);

    return place_order(req);
}

std::future<Order> Client::limit_buy(
    const std::string& symbol, Decimal quantity, Decimal price,
    std::optional<std::string_view> venue) {
    OrderRequest req = OrderRequest::limit(symbol, Side::Buy, quantity, price);
    if (venue) req.venue = std::string(*venue);

    return place_order(req);
}

std::future<Order> Client::limit_sell(
    const std::string& symbol, Decimal quantity, Decimal price,
    std::optional<std::string_view> venue) {
    OrderRequest req = OrderRequest::limit(symbol, Side::Sell, quantity, price);
    if (venue) req.venue = std::string(*venue);

    return place_order(req);
}

std::future<Order> Client::place_order(const OrderRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        // Validate with risk manager
        risk_manager_->validate_order(request);

        // Route order
        if (request.venue) {
            auto adapter = get_venue(request.venue);
            auto order = adapter->place_order(request).get();
            risk_manager_->order_opened(request.symbol);
            return order;
        }

        if (config_.general.smart_routing &&
            request.order_type == OrderType::Market) {
            return smart_route(request);
        }

        auto adapter = get_venue(std::nullopt);
        auto order = adapter->place_order(request).get();
        risk_manager_->order_opened(request.symbol);
        return order;
    });
}

Order Client::smart_route(const OrderRequest& request) {
    auto agg_book = aggregated_orderbook(request.symbol).get();

    std::optional<std::pair<std::string, Decimal>> best;
    if (request.side == Side::Buy) {
        best = agg_book.best_venue_buy(request.quantity);
    } else {
        best = agg_book.best_venue_sell(request.quantity);
    }

    if (best) {
        OrderRequest routed = request;
        routed.venue = best->first;
        auto adapter = get_venue(routed.venue);
        auto order = adapter->place_order(routed).get();
        risk_manager_->order_opened(request.symbol);
        return order;
    }

    // Fallback to default
    auto adapter = get_venue(std::nullopt);
    auto order = adapter->place_order(request).get();
    risk_manager_->order_opened(request.symbol);
    return order;
}

std::future<Order> Client::cancel_order(
    const std::string& order_id, const std::string& symbol, const std::string& venue) {
    return std::async(std::launch::async, [this, order_id, symbol, venue]() {
        auto adapter = get_venue(venue);
        auto order = adapter->cancel_order(order_id, symbol).get();
        risk_manager_->order_closed(symbol);
        return order;
    });
}

std::future<std::vector<Order>> Client::cancel_all_orders(
    std::optional<std::string_view> symbol, std::optional<std::string_view> venue) {
    return std::async(std::launch::async, [this, symbol, venue]() {
        std::optional<std::string> sym;
        if (symbol) sym = std::string(*symbol);

        if (venue) {
            auto adapter = get_venue(venue);
            auto orders = adapter->cancel_all_orders(sym).get();
            for (const auto& o : orders) {
                risk_manager_->order_closed(o.symbol);
            }
            return orders;
        }

        std::vector<Order> all_orders;
        for (auto& [name, adapter] : venues_) {
            try {
                auto orders = adapter->cancel_all_orders(sym).get();
                for (const auto& o : orders) {
                    risk_manager_->order_closed(o.symbol);
                }
                all_orders.insert(all_orders.end(), orders.begin(), orders.end());
            } catch (...) {
                // Continue
            }
        }
        return all_orders;
    });
}

std::future<std::vector<Order>> Client::open_orders(std::optional<std::string_view> symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        std::optional<std::string> sym;
        if (symbol) sym = std::string(*symbol);

        std::vector<Order> all_orders;
        for (auto& [name, adapter] : venues_) {
            try {
                auto orders = adapter->get_open_orders(sym).get();
                all_orders.insert(all_orders.end(), orders.begin(), orders.end());
            } catch (...) {
                // Continue
            }
        }
        return all_orders;
    });
}

// AMM Operations
std::future<SwapQuote> Client::quote(
    const std::string& base_token, const std::string& quote_token,
    Decimal amount, bool is_buy, const std::string& venue) {
    return get_venue(venue)->get_swap_quote(base_token, quote_token, amount, is_buy);
}

std::future<Trade> Client::swap(
    const std::string& base_token, const std::string& quote_token,
    Decimal amount, bool is_buy, double slippage, const std::string& venue) {
    return get_venue(venue)->execute_swap(
        base_token, quote_token, amount, is_buy, Decimal::from_double(slippage));
}

std::future<PoolInfo> Client::pool_info(
    const std::string& base_token, const std::string& quote_token,
    const std::string& venue) {
    return get_venue(venue)->get_pool_info(base_token, quote_token);
}

std::future<LiquidityResult> Client::add_liquidity(
    const std::string& base_token, const std::string& quote_token,
    Decimal base_amount, Decimal quote_amount,
    double slippage, const std::string& venue) {
    return get_venue(venue)->add_liquidity(
        base_token, quote_token, base_amount, quote_amount,
        Decimal::from_double(slippage));
}

std::future<LiquidityResult> Client::remove_liquidity(
    const std::string& pool_address, Decimal liquidity_amount,
    double slippage, const std::string& venue) {
    return get_venue(venue)->remove_liquidity(
        pool_address, liquidity_amount, Decimal::from_double(slippage));
}

std::future<std::vector<LpPosition>> Client::lp_positions(const std::string& venue) {
    return get_venue(venue)->get_lp_positions();
}

// Streaming
void Client::subscribe_ticker(
    const std::string& symbol, VenueAdapter::TickerCallback callback,
    std::optional<std::string_view> venue) {
    get_venue(venue)->subscribe_ticker(symbol, std::move(callback));
}

void Client::subscribe_trades(
    const std::string& symbol, VenueAdapter::TradeCallback callback,
    std::optional<std::string_view> venue) {
    get_venue(venue)->subscribe_trades(symbol, std::move(callback));
}

void Client::subscribe_orderbook(
    const std::string& symbol, VenueAdapter::OrderbookCallback callback,
    std::optional<std::string_view> venue) {
    get_venue(venue)->subscribe_orderbook(symbol, std::move(callback));
}

void Client::subscribe_orders(VenueAdapter::OrderCallback callback) {
    for (auto& [name, adapter] : venues_) {
        adapter->subscribe_orders(callback);
    }
}

void Client::unsubscribe_all() {
    for (auto& [name, adapter] : venues_) {
        adapter->unsubscribe_all();
    }
}

}  // namespace lx::trading
