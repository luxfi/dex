// LX Trading SDK - Arbitrage Scanner
// Continuously scans for arbitrage opportunities across all venues.
// Supports simple, triangular, and CEX-DEX arbitrage detection.

#pragma once

#include <lx/trading/arbitrage/types.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lx::trading::arbitrage {

/// Known CEX venues
inline const std::set<std::string> CEX_VENUES = {
    "binance", "coinbase", "kraken", "okx", "bybit",
    "kucoin", "mexc", "gate", "huobi"
};

/// Arbitrage scanner for detecting cross-venue opportunities
class Scanner {
public:
    explicit Scanner(ScannerConfig config);
    ~Scanner();

    // Non-copyable
    Scanner(const Scanner&) = delete;
    Scanner& operator=(const Scanner&) = delete;

    /// Add a chain configuration
    void add_chain(const CrossChainInfo& info);

    /// Update a price feed
    void update_price(const PriceSource& source);

    /// Subscribe to opportunity events
    void on_opportunity(OpportunityCallback callback);

    /// Start scanning for opportunities
    void start();

    /// Stop scanning
    void stop();

    /// Check if scanner is running
    [[nodiscard]] bool is_running() const noexcept { return running_.load(); }

    /// Get current config
    [[nodiscard]] const ScannerConfig& config() const noexcept { return config_; }

private:
    void scan_loop();
    std::vector<ArbitrageOpportunity> scan();
    std::vector<ArbitrageOpportunity> find_simple_arb(
        const std::string& symbol,
        const std::vector<PriceSource>& sources) const;
    std::vector<ArbitrageOpportunity> find_cex_dex_arb(
        const std::string& symbol,
        const std::vector<PriceSource>& sources) const;
    std::pair<Decimal, Decimal> calculate_costs(
        const std::string& source_chain,
        const std::string& dest_chain) const;
    double calculate_confidence(
        const PriceSource& buy,
        const PriceSource& sell) const;

    ScannerConfig config_;
    std::unordered_map<std::string, std::vector<PriceSource>> prices_;
    std::unordered_map<std::string, CrossChainInfo> chains_;
    std::vector<OpportunityCallback> callbacks_;
    mutable std::mutex prices_mutex_;
    mutable std::mutex chains_mutex_;
    mutable std::mutex callbacks_mutex_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> scan_thread_;
};

}  // namespace lx::trading::arbitrage
