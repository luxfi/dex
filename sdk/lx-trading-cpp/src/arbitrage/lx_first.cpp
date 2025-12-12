// LX Trading SDK - LX-First Arbitrage Implementation

#include <lx/trading/arbitrage/lx_first.hpp>
#include <algorithm>
#include <cmath>

namespace lx::trading::arbitrage {

LxFirstArbitrage::LxFirstArbitrage(LxFirstConfig config)
    : config_(std::move(config)) {}

void LxFirstArbitrage::update_lx_price(const LxPrice& price) {
    std::string symbol = price.symbol;
    {
        std::lock_guard<std::mutex> lock(lx_prices_mutex_);
        lx_prices_[symbol] = price;
    }

    // Immediately check for opportunities against stale venues
    check_opportunities(symbol);
}

void LxFirstArbitrage::update_venue_price(const VenuePrice& price) {
    std::lock_guard<std::mutex> lock(venue_prices_mutex_);
    auto& prices = venue_prices_[price.symbol];

    // Update or append
    bool found = false;
    for (auto& p : prices) {
        if (p.venue == price.venue) {
            p = price;
            found = true;
            break;
        }
    }

    if (!found) {
        prices.push_back(price);
    }
}

void LxFirstArbitrage::on_opportunity(LxFirstCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(std::move(callback));
}

void LxFirstArbitrage::start() {
    running_.store(true);
}

void LxFirstArbitrage::stop() {
    running_.store(false);
}

void LxFirstArbitrage::check_opportunities(const std::string& symbol) {
    if (!running_.load()) {
        return;
    }

    LxPrice lx_price;
    {
        std::lock_guard<std::mutex> lock(lx_prices_mutex_);
        auto it = lx_prices_.find(symbol);
        if (it == lx_prices_.end()) {
            return;
        }
        lx_price = it->second;
    }

    std::vector<VenuePrice> vps;
    {
        std::lock_guard<std::mutex> lock(venue_prices_mutex_);
        auto it = venue_prices_.find(symbol);
        if (it == venue_prices_.end()) {
            return;
        }
        vps = it->second;
    }

    int64_t now = now_ms();

    for (const auto& vp : vps) {
        // Calculate how stale the venue is
        int64_t staleness = now - vp.timestamp;
        if (staleness > config_.max_staleness_ms) {
            continue;  // Too stale, might have updated by now
        }

        // Check for BUY opportunity (venue ask < LX mid)
        // The slow venue hasn't caught up to LX's higher price
        if (vp.ask < lx_price.mid) {
            Decimal divergence = lx_price.mid - vp.ask;
            Decimal divergence_bps = (divergence / lx_price.mid) * Decimal::from_double(10000.0);

            if (divergence_bps >= config_.min_divergence_bps) {
                auto opp = create_opportunity(
                    symbol, lx_price, vp, staleness,
                    "buy", divergence, divergence_bps
                );
                if (opp.expected_profit >= config_.min_profit) {
                    std::lock_guard<std::mutex> lock(callbacks_mutex_);
                    for (const auto& callback : callbacks_) {
                        callback(opp);
                    }
                }
            }
        }

        // Check for SELL opportunity (venue bid > LX mid)
        // The slow venue hasn't caught up to LX's lower price
        if (vp.bid > lx_price.mid) {
            Decimal divergence = vp.bid - lx_price.mid;
            Decimal divergence_bps = (divergence / lx_price.mid) * Decimal::from_double(10000.0);

            if (divergence_bps >= config_.min_divergence_bps) {
                auto opp = create_opportunity(
                    symbol, lx_price, vp, staleness,
                    "sell", divergence, divergence_bps
                );
                if (opp.expected_profit >= config_.min_profit) {
                    std::lock_guard<std::mutex> lock(callbacks_mutex_);
                    for (const auto& callback : callbacks_) {
                        callback(opp);
                    }
                }
            }
        }
    }
}

LxFirstOpportunity LxFirstArbitrage::create_opportunity(
    const std::string& symbol,
    const LxPrice& lx_price,
    const VenuePrice& vp,
    int64_t staleness,
    const std::string& side,
    Decimal divergence,
    Decimal divergence_bps) const {

    int64_t now = now_ms();
    Decimal expected_profit = divergence * config_.max_position_size;
    double confidence = calculate_confidence(staleness, divergence_bps);

    LxFirstOpportunity opp;
    opp.id = symbol + "-" + vp.venue + "-" + side + "-" + std::to_string(now);
    opp.symbol = symbol;
    opp.timestamp = now;
    opp.lx_price = lx_price;
    opp.stale_venue = vp.venue;
    opp.stale_price = vp;
    opp.staleness = staleness;
    opp.side = side;
    opp.divergence = divergence;
    opp.divergence_bps = divergence_bps;
    opp.expected_profit = expected_profit;
    opp.max_size = config_.max_position_size;
    opp.confidence = confidence;

    return opp;
}

double LxFirstArbitrage::calculate_confidence(int64_t staleness, Decimal divergence_bps) const {
    // Higher confidence when:
    // 1. Venue is more stale (hasn't had time to update)
    // 2. Divergence is larger (more room for profit)
    double staleness_score = std::max(0.0, 1.0 - static_cast<double>(staleness) / 5000.0);  // 5s max
    double divergence_score = std::min(1.0, divergence_bps.to_double() / 100.0);  // 100bps = 1.0

    return 0.5 * staleness_score + 0.5 * divergence_score;
}

}  // namespace lx::trading::arbitrage
