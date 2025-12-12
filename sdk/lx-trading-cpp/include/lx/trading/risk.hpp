// LX Trading SDK - Risk Management
// Thread-safe risk controls and position tracking

#pragma once

#include <lx/trading/config.hpp>
#include <lx/trading/types.hpp>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lx::trading {

// Risk limit violation exception
class RiskError : public std::runtime_error {
public:
    explicit RiskError(const std::string& msg) : std::runtime_error(msg) {}
};

// Thread-safe risk manager
class RiskManager {
public:
    explicit RiskManager(const RiskConfig& config);

    // Configuration
    [[nodiscard]] bool is_enabled() const noexcept { return config_.enabled; }
    [[nodiscard]] const RiskConfig& config() const noexcept { return config_; }

    // Kill switch
    [[nodiscard]] bool is_killed() const noexcept { return kill_switch_.load(std::memory_order_acquire); }
    void kill() noexcept { kill_switch_.store(true, std::memory_order_release); }
    void reset() noexcept { kill_switch_.store(false, std::memory_order_release); }

    // Order validation - throws RiskError if invalid
    void validate_order(const OrderRequest& request);

    // Position tracking
    void update_position(const std::string& asset, Decimal quantity, Side side);
    [[nodiscard]] Decimal position(const std::string& asset) const;
    [[nodiscard]] std::unordered_map<std::string, Decimal> positions() const;

    // PnL tracking
    void update_pnl(Decimal pnl);
    [[nodiscard]] Decimal daily_pnl() const;
    void reset_daily_pnl();

    // Order tracking
    void order_opened(const std::string& symbol);
    void order_closed(const std::string& symbol);
    [[nodiscard]] int open_orders(const std::string& symbol) const;

    // Pre-trade checks (return false instead of throwing)
    [[nodiscard]] bool check_order_size(Decimal quantity) const noexcept;
    [[nodiscard]] bool check_position_limit(const std::string& asset, Decimal new_position) const noexcept;
    [[nodiscard]] bool check_daily_loss() const noexcept;
    [[nodiscard]] bool check_open_orders(const std::string& symbol) const noexcept;

private:
    RiskConfig config_;
    std::atomic<bool> kill_switch_{false};

    mutable std::shared_mutex position_mutex_;
    std::unordered_map<std::string, Decimal> positions_;

    mutable std::shared_mutex pnl_mutex_;
    Decimal daily_pnl_;

    mutable std::shared_mutex orders_mutex_;
    std::unordered_map<std::string, int> open_orders_;
};

// RAII order tracker
class OrderTracker {
public:
    OrderTracker(RiskManager& rm, const std::string& symbol)
        : risk_manager_(rm), symbol_(symbol) {
        risk_manager_.order_opened(symbol_);
    }

    ~OrderTracker() {
        if (!released_) {
            risk_manager_.order_closed(symbol_);
        }
    }

    void release() { released_ = true; }

private:
    RiskManager& risk_manager_;
    std::string symbol_;
    bool released_ = false;
};

}  // namespace lx::trading
