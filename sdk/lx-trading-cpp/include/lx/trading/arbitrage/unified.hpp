// LX Trading SDK - Unified Liquidity Arbitrage
//
// Since LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks),
// it becomes the price ORACLE. Other venues are always stale by comparison.
//
// Architecture:
// 1. LX DEX prices are the TRUTH (most current)
// 2. Other venues (CEX, external DEX) are STALE
// 3. Arbitrage = exploiting stale venues before they catch up
// 4. LX always wins because it sees/moves prices first
//
// NO SMART CONTRACTS - just coordinated trades through unified SDK.

#pragma once

#include <lx/trading/arbitrage/types.hpp>
#include <lx/trading/types.hpp>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lx::trading::arbitrage {

/// Aggregated orderbook level
struct AggregatedLevel {
    Decimal price;
    Decimal quantity;
    std::string venue;
    int64_t timestamp{0};
};

/// Aggregated orderbook from all venues
struct AggregatedBook {
    std::string symbol;
    std::vector<AggregatedLevel> bids;
    std::vector<AggregatedLevel> asks;
};

/// Trading client interface for arbitrage
class TradingClient {
public:
    virtual ~TradingClient() = default;

    /// Get aggregated orderbook from all venues
    virtual AggregatedBook aggregated_orderbook(const std::string& symbol) = 0;

    /// Place an order on a specific venue
    virtual Order place_order(const OrderRequest& request) = 0;
};

/// Unified arbitrage across all SDK-connected venues
class UnifiedArbitrage {
public:
    UnifiedArbitrage(std::shared_ptr<TradingClient> client, UnifiedArbConfig config);
    ~UnifiedArbitrage();

    // Non-copyable
    UnifiedArbitrage(const UnifiedArbitrage&) = delete;
    UnifiedArbitrage& operator=(const UnifiedArbitrage&) = delete;

    /// Start the arbitrage system
    void start();

    /// Stop the arbitrage system
    void stop();

    /// Subscribe to opportunity events
    void on_opportunity(UnifiedCallback callback);

    /// Get arbitrage statistics
    [[nodiscard]] UnifiedArbStats get_stats() const;

    /// Check if system is running
    [[nodiscard]] bool is_running() const noexcept { return running_.load(); }

    /// Get current config
    [[nodiscard]] const UnifiedArbConfig& config() const noexcept { return config_; }

private:
    void scan_loop();
    void execute_loop();
    std::optional<UnifiedOpportunity> find_opportunity(const std::string& symbol);
    UnifiedExecution execute_opportunity(const UnifiedOpportunity& opp);

    std::shared_ptr<TradingClient> client_;
    UnifiedArbConfig config_;
    Decimal total_pnl_;
    std::vector<UnifiedExecution> executions_;
    std::vector<UnifiedCallback> callbacks_;
    std::deque<UnifiedOpportunity> opportunity_queue_;
    mutable std::mutex pnl_mutex_;
    mutable std::mutex executions_mutex_;
    mutable std::mutex callbacks_mutex_;
    mutable std::mutex queue_mutex_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> scan_thread_;
    std::unique_ptr<std::thread> execute_thread_;
};

}  // namespace lx::trading::arbitrage
