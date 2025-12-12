// LX Trading SDK - LX-First Arbitrage Strategy
//
// Key Insight: LX DEX is the FASTEST venue (nanosecond price updates, 200ms blocks).
// By the time other venues update, LX has already moved.
//
// This means:
// 1. LX DEX price is the "TRUE" price (most current)
// 2. Other venues are always STALE by comparison
// 3. Arbitrage = correcting stale venues to match LX
// 4. LX DEX is the ORACLE, not just another venue
//
// Strategy:
// 1. Watch LX DEX prices (the reference)
// 2. Compare against "slow" venues (CEX, external DEX)
// 3. When slow venue diverges from LX, trade on SLOW venue
// 4. You're essentially front-running slow venues with LX information
//
// Example:
// - LX DEX BTC: $50,000 (current, true)
// - Binance BTC: $49,990 (stale, 50ms behind)
// - Uniswap BTC: $50,020 (stale, 12s behind)
//
// Action:
// - Buy on Binance at $49,990 (they haven't caught up yet)
// - Sell on Uniswap at $50,020 (they haven't corrected yet)
// - Net: $30 profit per BTC
//
// Why LX wins: By the time Binance/Uniswap update, we've already executed.

#pragma once

#include <lx/trading/arbitrage/types.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lx::trading::arbitrage {

/// LX-first arbitrage using LX DEX as the price oracle
class LxFirstArbitrage {
public:
    explicit LxFirstArbitrage(LxFirstConfig config);
    ~LxFirstArbitrage() = default;

    // Non-copyable
    LxFirstArbitrage(const LxFirstArbitrage&) = delete;
    LxFirstArbitrage& operator=(const LxFirstArbitrage&) = delete;

    /// Update the LX DEX price (the oracle)
    /// Immediately checks for opportunities against stale venues
    void update_lx_price(const LxPrice& price);

    /// Update a price from a 'slow' venue
    void update_venue_price(const VenuePrice& price);

    /// Subscribe to opportunity events
    void on_opportunity(LxFirstCallback callback);

    /// Start the arbitrage system
    void start();

    /// Stop the arbitrage system
    void stop();

    /// Check if system is running
    [[nodiscard]] bool is_running() const noexcept { return running_.load(); }

    /// Get current config
    [[nodiscard]] const LxFirstConfig& config() const noexcept { return config_; }

private:
    void check_opportunities(const std::string& symbol);
    LxFirstOpportunity create_opportunity(
        const std::string& symbol,
        const LxPrice& lx_price,
        const VenuePrice& vp,
        int64_t staleness,
        const std::string& side,
        Decimal divergence,
        Decimal divergence_bps) const;
    double calculate_confidence(int64_t staleness, Decimal divergence_bps) const;

    LxFirstConfig config_;
    std::unordered_map<std::string, LxPrice> lx_prices_;
    std::unordered_map<std::string, std::vector<VenuePrice>> venue_prices_;
    std::vector<LxFirstCallback> callbacks_;
    mutable std::mutex lx_prices_mutex_;
    mutable std::mutex venue_prices_mutex_;
    mutable std::mutex callbacks_mutex_;
    std::atomic<bool> running_{false};
};

/*
TRADING EXECUTION STRATEGY

When an LxFirstOpportunity is detected:

1. DO NOT trade on LX DEX (it's the reference, not the opportunity)

2. Trade on the STALE venue:
   - If Side="buy": Buy on stale venue (their ask is behind LX)
   - If Side="sell": Sell on stale venue (their bid is behind LX)

3. Settlement options:
   a) Hold position until venues converge (market neutral)
   b) Immediately hedge on LX DEX (lock in profit)
   c) Bridge and sell on another venue (more complex)

4. The key insight:
   - You're NOT arbitraging between two venues
   - You're front-running the slow venue with LX information
   - LX price is where the slow venue WILL BE, you just got there first

Example execution:

  LX DEX shows BTC = $50,000 (current, true price)
  Binance shows BTC = $49,950 (50ms stale)

  Action: BUY on Binance at $49,950
  Why: Binance WILL update to ~$50,000, we bought before they did
  Profit: ~$50 per BTC (0.1%)

  Optional hedge: SELL on LX DEX at $50,000 to lock in profit immediately
*/

}  // namespace lx::trading::arbitrage
