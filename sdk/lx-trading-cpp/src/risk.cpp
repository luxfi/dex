// LX Trading SDK - Risk Manager Implementation

#include <lx/trading/risk.hpp>

namespace lx::trading {

RiskManager::RiskManager(const RiskConfig& config) : config_(config) {}

void RiskManager::validate_order(const OrderRequest& request) {
    if (!config_.enabled) return;

    if (kill_switch_.load(std::memory_order_acquire)) {
        throw RiskError("Kill switch is active");
    }

    // Check order size
    if (config_.max_order_size.is_positive() &&
        request.quantity > config_.max_order_size) {
        throw RiskError(
            "Order size " + request.quantity.to_string() +
            " exceeds max " + config_.max_order_size.to_string());
    }

    // Check position limit
    auto pair = TradingPair::from_symbol(request.symbol);
    if (pair) {
        std::string base(pair->base.data());

        Decimal current;
        {
            std::shared_lock lock(position_mutex_);
            auto it = positions_.find(base);
            if (it != positions_.end()) {
                current = it->second;
            }
        }

        Decimal new_position = (request.side == Side::Buy)
            ? current + request.quantity
            : current - request.quantity;

        // Asset-specific limit
        auto limit_it = config_.position_limits.find(base);
        if (limit_it != config_.position_limits.end()) {
            if (new_position.abs() > limit_it->second) {
                throw RiskError(
                    "Position limit exceeded for " + base + ": " +
                    current.to_string() + " + " + request.quantity.to_string() +
                    " > " + limit_it->second.to_string());
            }
        }

        // Global position limit
        if (config_.max_position_size.is_positive() &&
            new_position.abs() > config_.max_position_size) {
            throw RiskError(
                "Max position size exceeded: " +
                new_position.abs().to_string() + " > " +
                config_.max_position_size.to_string());
        }
    }

    // Check open orders count
    {
        std::shared_lock lock(orders_mutex_);
        auto it = open_orders_.find(request.symbol);
        int count = (it != open_orders_.end()) ? it->second : 0;
        if (count >= config_.max_open_orders) {
            throw RiskError(
                "Max open orders (" + std::to_string(config_.max_open_orders) +
                ") reached for " + request.symbol);
        }
    }

    // Check daily loss
    if (config_.max_daily_loss.is_positive()) {
        std::shared_lock lock(pnl_mutex_);
        if (daily_pnl_.is_negative() &&
            daily_pnl_.abs() > config_.max_daily_loss) {
            throw RiskError(
                "Daily loss limit exceeded: " +
                daily_pnl_.abs().to_string() + " > " +
                config_.max_daily_loss.to_string());
        }
    }
}

void RiskManager::update_position(const std::string& asset, Decimal quantity, Side side) {
    std::unique_lock lock(position_mutex_);
    auto& pos = positions_[asset];
    pos = (side == Side::Buy) ? pos + quantity : pos - quantity;
}

Decimal RiskManager::position(const std::string& asset) const {
    std::shared_lock lock(position_mutex_);
    auto it = positions_.find(asset);
    return (it != positions_.end()) ? it->second : Decimal::zero();
}

std::unordered_map<std::string, Decimal> RiskManager::positions() const {
    std::shared_lock lock(position_mutex_);
    return positions_;
}

void RiskManager::update_pnl(Decimal pnl) {
    std::unique_lock lock(pnl_mutex_);
    daily_pnl_ = daily_pnl_ + pnl;

    // Auto kill switch
    if (config_.kill_switch_enabled &&
        config_.max_daily_loss.is_positive() &&
        daily_pnl_.is_negative() &&
        daily_pnl_.abs() > config_.max_daily_loss) {
        kill_switch_.store(true, std::memory_order_release);
    }
}

Decimal RiskManager::daily_pnl() const {
    std::shared_lock lock(pnl_mutex_);
    return daily_pnl_;
}

void RiskManager::reset_daily_pnl() {
    std::unique_lock lock(pnl_mutex_);
    daily_pnl_ = Decimal::zero();
}

void RiskManager::order_opened(const std::string& symbol) {
    std::unique_lock lock(orders_mutex_);
    open_orders_[symbol]++;
}

void RiskManager::order_closed(const std::string& symbol) {
    std::unique_lock lock(orders_mutex_);
    auto it = open_orders_.find(symbol);
    if (it != open_orders_.end() && it->second > 0) {
        it->second--;
    }
}

int RiskManager::open_orders(const std::string& symbol) const {
    std::shared_lock lock(orders_mutex_);
    auto it = open_orders_.find(symbol);
    return (it != open_orders_.end()) ? it->second : 0;
}

bool RiskManager::check_order_size(Decimal quantity) const noexcept {
    if (!config_.max_order_size.is_positive()) return true;
    return quantity <= config_.max_order_size;
}

bool RiskManager::check_position_limit(
    const std::string& asset, Decimal new_position) const noexcept {
    auto it = config_.position_limits.find(asset);
    if (it != config_.position_limits.end()) {
        if (new_position.abs() > it->second) return false;
    }

    if (config_.max_position_size.is_positive()) {
        if (new_position.abs() > config_.max_position_size) return false;
    }

    return true;
}

bool RiskManager::check_daily_loss() const noexcept {
    if (!config_.max_daily_loss.is_positive()) return true;

    std::shared_lock lock(pnl_mutex_);
    return !daily_pnl_.is_negative() ||
           daily_pnl_.abs() <= config_.max_daily_loss;
}

bool RiskManager::check_open_orders(const std::string& symbol) const noexcept {
    std::shared_lock lock(orders_mutex_);
    auto it = open_orders_.find(symbol);
    int count = (it != open_orders_.end()) ? it->second : 0;
    return count < config_.max_open_orders;
}

}  // namespace lx::trading
