// LX Trading SDK - Unified Arbitrage Implementation

#include <lx/trading/arbitrage/unified.hpp>
#include <algorithm>
#include <chrono>

namespace lx::trading::arbitrage {

UnifiedArbitrage::UnifiedArbitrage(std::shared_ptr<TradingClient> client, UnifiedArbConfig config)
    : client_(std::move(client)),
      config_(std::move(config)),
      total_pnl_(Decimal::zero()) {}

UnifiedArbitrage::~UnifiedArbitrage() {
    stop();
}

void UnifiedArbitrage::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    scan_thread_ = std::make_unique<std::thread>(&UnifiedArbitrage::scan_loop, this);
    execute_thread_ = std::make_unique<std::thread>(&UnifiedArbitrage::execute_loop, this);
}

void UnifiedArbitrage::stop() {
    running_.store(false);

    if (scan_thread_ && scan_thread_->joinable()) {
        scan_thread_->join();
    }
    scan_thread_.reset();

    if (execute_thread_ && execute_thread_->joinable()) {
        execute_thread_->join();
    }
    execute_thread_.reset();
}

void UnifiedArbitrage::on_opportunity(UnifiedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(std::move(callback));
}

UnifiedArbStats UnifiedArbitrage::get_stats() const {
    std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
    std::lock_guard<std::mutex> exec_lock(executions_mutex_);

    uint64_t successful = 0;
    for (const auto& exec : executions_) {
        if (exec.status == "completed" && exec.actual_profit.is_positive()) {
            ++successful;
        }
    }

    double win_rate = executions_.empty()
        ? 0.0
        : static_cast<double>(successful) / static_cast<double>(executions_.size());

    return UnifiedArbStats{
        .total_executions = executions_.size(),
        .successful_executions = successful,
        .total_pnl = total_pnl_,
        .win_rate = win_rate
    };
}

void UnifiedArbitrage::scan_loop() {
    while (running_.load()) {
        // Scan each symbol
        for (const auto& symbol : config_.symbols) {
            auto opp = find_opportunity(symbol);
            if (opp && opp->net_profit > config_.min_profit) {
                // Add to queue
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (opportunity_queue_.size() < 1000) {
                        opportunity_queue_.push_back(*opp);
                    }
                }

                // Emit to callbacks
                {
                    std::lock_guard<std::mutex> lock(callbacks_mutex_);
                    for (const auto& callback : callbacks_) {
                        callback(*opp);
                    }
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.scan_interval_ms));
    }
}

std::optional<UnifiedOpportunity> UnifiedArbitrage::find_opportunity(const std::string& symbol) {
    AggregatedBook book;
    try {
        book = client_->aggregated_orderbook(symbol);
    } catch (...) {
        return std::nullopt;
    }

    if (book.bids.empty() || book.asks.empty()) {
        return std::nullopt;
    }

    const auto& best_bid = book.bids[0];
    const auto& best_ask = book.asks[0];

    // Cross-venue arbitrage: bid on one venue > ask on another
    if (best_bid.price <= best_ask.price) {
        return std::nullopt;
    }

    Decimal spread = best_bid.price - best_ask.price;
    Decimal spread_bps = (spread / best_ask.price) * Decimal::from_double(10000.0);

    if (spread_bps < config_.min_spread_bps) {
        return std::nullopt;
    }

    Decimal max_size = std::min({
        best_bid.quantity,
        best_ask.quantity,
        config_.max_position_size
    });

    Decimal gross_profit = spread * max_size;
    Decimal total_fees = best_ask.price * max_size * Decimal::from_double(0.002);  // ~0.2% total fees
    Decimal net_profit = gross_profit - total_fees;

    int64_t now = now_ms();

    UnifiedOpportunity opp;
    opp.id = "arb-" + symbol + "-" + std::to_string(now);
    opp.symbol = symbol;
    opp.timestamp = now;
    opp.expires_at = now + 5000;
    opp.buy_venue = best_ask.venue;
    opp.buy_price = best_ask.price;
    opp.buy_size = best_ask.quantity;
    opp.sell_venue = best_bid.venue;
    opp.sell_price = best_bid.price;
    opp.sell_size = best_bid.quantity;
    opp.spread = spread;
    opp.spread_bps = spread_bps;
    opp.max_size = max_size;
    opp.gross_profit = gross_profit;
    opp.est_fees = total_fees;
    opp.net_profit = net_profit;
    opp.confidence = 0.8;
    opp.latency = now - best_ask.timestamp;

    return opp;
}

void UnifiedArbitrage::execute_loop() {
    while (running_.load()) {
        UnifiedOpportunity opp;
        bool has_opp = false;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!opportunity_queue_.empty()) {
                opp = opportunity_queue_.front();
                opportunity_queue_.pop_front();
                has_opp = true;
            }
        }

        if (has_opp) {
            auto result = execute_opportunity(opp);

            // Update stats
            {
                std::lock_guard<std::mutex> lock(pnl_mutex_);
                total_pnl_ = total_pnl_ + result.actual_profit;
            }
            {
                std::lock_guard<std::mutex> lock(executions_mutex_);
                executions_.push_back(std::move(result));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

UnifiedExecution UnifiedArbitrage::execute_opportunity(const UnifiedOpportunity& opp) {
    int64_t now = now_ms();

    // Check if expired
    if (now > opp.expires_at) {
        return UnifiedExecution{
            .id = opp.id,
            .opportunity = opp,
            .start_time = now,
            .end_time = now,
            .status = "expired",
            .buy_order_id = std::nullopt,
            .sell_order_id = std::nullopt,
            .actual_profit = Decimal::zero(),
            .fees = Decimal::zero(),
            .error = "Opportunity expired"
        };
    }

    UnifiedExecution exec_result;
    exec_result.id = opp.id;
    exec_result.opportunity = opp;
    exec_result.start_time = now;
    exec_result.status = "executing";
    exec_result.actual_profit = Decimal::zero();
    exec_result.fees = Decimal::zero();

    try {
        // Execute both legs
        OrderRequest buy_request = OrderRequest::limit(opp.symbol, Side::Buy, opp.max_size, opp.buy_price)
            .with_venue(opp.buy_venue);

        OrderRequest sell_request = OrderRequest::limit(opp.symbol, Side::Sell, opp.max_size, opp.sell_price)
            .with_venue(opp.sell_venue);

        // Execute orders (in production, these would be async/parallel)
        Order buy_order = client_->place_order(buy_request);
        Order sell_order = client_->place_order(sell_request);

        exec_result.end_time = now_ms();
        exec_result.buy_order_id = buy_order.order_id;
        exec_result.sell_order_id = sell_order.order_id;

        // Calculate actual profit
        if (buy_order.average_price && sell_order.average_price) {
            Decimal buy_value = *buy_order.average_price * buy_order.filled_quantity;
            Decimal sell_value = *sell_order.average_price * sell_order.filled_quantity;
            exec_result.actual_profit = sell_value - buy_value;

            // Subtract fees
            Decimal buy_fees = Decimal::zero();
            for (const auto& f : buy_order.fees) {
                buy_fees = buy_fees + f.amount;
            }
            Decimal sell_fees = Decimal::zero();
            for (const auto& f : sell_order.fees) {
                sell_fees = sell_fees + f.amount;
            }
            exec_result.fees = buy_fees + sell_fees;
            exec_result.actual_profit = exec_result.actual_profit - exec_result.fees;
        }

        exec_result.status = "completed";
    } catch (const std::exception& e) {
        exec_result.end_time = now_ms();
        exec_result.status = "failed";
        exec_result.error = e.what();
    }

    return exec_result;
}

}  // namespace lx::trading::arbitrage
