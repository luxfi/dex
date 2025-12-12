// LX Trading SDK - Arbitrage Scanner Implementation

#include <lx/trading/arbitrage/scanner.hpp>
#include <algorithm>
#include <chrono>

namespace lx::trading::arbitrage {

Scanner::Scanner(ScannerConfig config)
    : config_(std::move(config)) {}

Scanner::~Scanner() {
    stop();
}

void Scanner::add_chain(const CrossChainInfo& info) {
    std::lock_guard<std::mutex> lock(chains_mutex_);
    chains_[info.chain_id] = info;
}

void Scanner::update_price(const PriceSource& source) {
    std::lock_guard<std::mutex> lock(prices_mutex_);
    auto& sources = prices_[source.symbol];

    // Update existing or append new
    bool found = false;
    for (auto& s : sources) {
        if (s.chain_id == source.chain_id && s.venue == source.venue) {
            s = source;
            found = true;
            break;
        }
    }

    if (!found) {
        sources.push_back(source);
    }
}

void Scanner::on_opportunity(OpportunityCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(std::move(callback));
}

void Scanner::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    scan_thread_ = std::make_unique<std::thread>(&Scanner::scan_loop, this);
}

void Scanner::stop() {
    running_.store(false);

    if (scan_thread_ && scan_thread_->joinable()) {
        scan_thread_->join();
    }
    scan_thread_.reset();
}

void Scanner::scan_loop() {
    while (running_.load()) {
        auto opportunities = scan();

        // Emit opportunities
        {
            std::lock_guard<std::mutex> lock(callbacks_mutex_);
            for (const auto& opp : opportunities) {
                for (const auto& callback : callbacks_) {
                    callback(opp);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.scan_interval_ms));
    }
}

std::vector<ArbitrageOpportunity> Scanner::scan() {
    std::vector<ArbitrageOpportunity> opportunities;

    std::lock_guard<std::mutex> lock(prices_mutex_);

    for (const auto& [symbol, sources] : prices_) {
        if (sources.size() < 2) {
            continue;
        }

        int64_t now = now_ms();

        // Filter stale prices
        std::vector<PriceSource> valid_sources;
        for (const auto& s : sources) {
            if (now - s.timestamp < config_.max_price_age_ms) {
                valid_sources.push_back(s);
            }
        }

        if (valid_sources.size() < 2) {
            continue;
        }

        // Simple arbitrage
        auto simple_opps = find_simple_arb(symbol, valid_sources);
        opportunities.insert(opportunities.end(), simple_opps.begin(), simple_opps.end());

        // CEX-DEX arbitrage
        auto cex_dex_opps = find_cex_dex_arb(symbol, valid_sources);
        opportunities.insert(opportunities.end(), cex_dex_opps.begin(), cex_dex_opps.end());
    }

    return opportunities;
}

std::vector<ArbitrageOpportunity> Scanner::find_simple_arb(
    const std::string& symbol,
    const std::vector<PriceSource>& sources) const {

    std::vector<ArbitrageOpportunity> opportunities;

    // Sort by ask (lowest first for buying)
    std::vector<PriceSource> buy_order = sources;
    std::sort(buy_order.begin(), buy_order.end(),
              [](const auto& a, const auto& b) { return a.ask < b.ask; });

    // Sort by bid (highest first for selling)
    std::vector<PriceSource> sell_order = sources;
    std::sort(sell_order.begin(), sell_order.end(),
              [](const auto& a, const auto& b) { return a.bid > b.bid; });

    for (const auto& buy_src : buy_order) {
        for (const auto& sell_src : sell_order) {
            // Skip same venue/chain
            if (buy_src.chain_id == sell_src.chain_id && buy_src.venue == sell_src.venue) {
                continue;
            }

            // Calculate spread
            Decimal spread = sell_src.bid - buy_src.ask;
            if (spread <= Decimal::zero()) {
                continue;
            }

            Decimal spread_bps = (spread / buy_src.ask) * Decimal::from_double(10000.0);
            if (spread_bps < config_.min_spread_bps) {
                continue;
            }

            // Calculate costs
            auto [gas_cost, bridge_cost] = calculate_costs(buy_src.chain_id, sell_src.chain_id);

            // Maximum size limited by liquidity
            Decimal max_size = std::min(buy_src.liquidity, sell_src.liquidity);

            // Calculate PnL
            Decimal gross_pnl = spread * max_size;
            Decimal net_pnl = gross_pnl - gas_cost - bridge_cost;

            if (net_pnl < config_.min_profit_usd) {
                continue;
            }

            // Calculate confidence
            double confidence = calculate_confidence(buy_src, sell_src);

            int64_t now = now_ms();
            ArbitrageOpportunity opp;
            opp.id = "simple-" + symbol + "-" + buy_src.venue + "-" + sell_src.venue + "-" + std::to_string(now);
            opp.arb_type = ArbType::Simple;
            opp.buy_source = buy_src;
            opp.sell_source = sell_src;
            opp.spread_bps = spread_bps;
            opp.estimated_pnl = gross_pnl;
            opp.max_size = max_size;
            opp.gas_cost_usd = gas_cost;
            opp.bridge_cost_usd = bridge_cost;
            opp.net_pnl = net_pnl;
            opp.confidence = confidence;
            opp.expires_at = now + 5000;

            // Build routes
            opp.routes.push_back(Route{
                .chain_id = buy_src.chain_id,
                .venue = buy_src.venue,
                .action = "buy",
                .token_in = "USDC",
                .token_out = symbol,
                .amount_in = max_size * buy_src.ask,
                .expected_out = max_size,
                .min_amount_out = max_size * Decimal::from_double(0.99)
            });

            opp.routes.push_back(Route{
                .chain_id = sell_src.chain_id,
                .venue = sell_src.venue,
                .action = "sell",
                .token_in = symbol,
                .token_out = "USDC",
                .amount_in = max_size,
                .expected_out = max_size * sell_src.bid,
                .min_amount_out = max_size * sell_src.bid * Decimal::from_double(0.99)
            });

            opportunities.push_back(std::move(opp));
        }
    }

    return opportunities;
}

std::vector<ArbitrageOpportunity> Scanner::find_cex_dex_arb(
    const std::string& symbol,
    const std::vector<PriceSource>& sources) const {

    std::vector<ArbitrageOpportunity> opportunities;

    // Separate CEX and DEX sources
    std::vector<PriceSource> cex_sources;
    std::vector<PriceSource> dex_sources;

    for (const auto& s : sources) {
        if (CEX_VENUES.count(s.venue)) {
            cex_sources.push_back(s);
        } else {
            dex_sources.push_back(s);
        }
    }

    // CEX buy -> DEX sell
    for (const auto& cex : cex_sources) {
        for (const auto& dex : dex_sources) {
            Decimal spread = dex.bid - cex.ask;
            if (spread <= Decimal::zero()) {
                continue;
            }

            Decimal spread_bps = (spread / cex.ask) * Decimal::from_double(10000.0);
            if (spread_bps < config_.min_spread_bps) {
                continue;
            }

            Decimal max_size = std::min(cex.liquidity, dex.liquidity);
            Decimal gross_pnl = spread * max_size;

            int64_t now = now_ms();
            ArbitrageOpportunity opp;
            opp.id = "cexdex-" + symbol + "-" + cex.venue + "-" + dex.venue + "-" + std::to_string(now);
            opp.arb_type = ArbType::CexDex;
            opp.buy_source = cex;
            opp.sell_source = dex;
            opp.spread_bps = spread_bps;
            opp.estimated_pnl = gross_pnl;
            opp.max_size = max_size;
            opp.gas_cost_usd = Decimal::from_double(0.5);
            opp.bridge_cost_usd = Decimal::zero();
            opp.net_pnl = gross_pnl - Decimal::from_double(0.5);
            opp.confidence = 0.7;
            opp.expires_at = now + 3000;

            opportunities.push_back(std::move(opp));
        }
    }

    // DEX buy -> CEX sell
    for (const auto& dex : dex_sources) {
        for (const auto& cex : cex_sources) {
            Decimal spread = cex.bid - dex.ask;
            if (spread <= Decimal::zero()) {
                continue;
            }

            Decimal spread_bps = (spread / dex.ask) * Decimal::from_double(10000.0);
            if (spread_bps < config_.min_spread_bps) {
                continue;
            }

            Decimal max_size = std::min(dex.liquidity, cex.liquidity);
            Decimal gross_pnl = spread * max_size;

            int64_t now = now_ms();
            ArbitrageOpportunity opp;
            opp.id = "cexdex-" + symbol + "-" + dex.venue + "-" + cex.venue + "-" + std::to_string(now);
            opp.arb_type = ArbType::CexDex;
            opp.buy_source = dex;
            opp.sell_source = cex;
            opp.spread_bps = spread_bps;
            opp.estimated_pnl = gross_pnl;
            opp.max_size = max_size;
            opp.gas_cost_usd = Decimal::from_double(0.5);
            opp.bridge_cost_usd = Decimal::zero();
            opp.net_pnl = gross_pnl - Decimal::from_double(0.5);
            opp.confidence = 0.7;
            opp.expires_at = now + 3000;

            opportunities.push_back(std::move(opp));
        }
    }

    return opportunities;
}

std::pair<Decimal, Decimal> Scanner::calculate_costs(
    const std::string& source_chain,
    const std::string& dest_chain) const {

    std::lock_guard<std::mutex> lock(chains_mutex_);

    auto src_it = chains_.find(source_chain);
    auto dst_it = chains_.find(dest_chain);

    // Estimate gas cost
    Decimal gas_cost = src_it != chains_.end()
        ? Decimal::from_double(0.05)
        : Decimal::from_double(0.1);

    // Bridge cost if crossing chains
    Decimal bridge_cost = Decimal::zero();
    if (source_chain != dest_chain && src_it != chains_.end() && dst_it != chains_.end()) {
        const auto& src = src_it->second;
        const auto& dst = dst_it->second;

        if (src.warp_supported && dst.warp_supported) {
            bridge_cost = Decimal::from_double(0.01);  // Warp is nearly free
        } else if (src.teleport_supported && dst.teleport_supported) {
            bridge_cost = Decimal::from_double(0.10);  // Teleport for EVM
        } else {
            bridge_cost = Decimal::from_double(1.0);   // Generic bridge
        }
    }

    return {gas_cost, bridge_cost};
}

double Scanner::calculate_confidence(
    const PriceSource& buy,
    const PriceSource& sell) const {

    int64_t now = now_ms();
    double max_age = static_cast<double>(config_.max_price_age_ms) / 1000.0;

    // Freshness score
    double buy_age = static_cast<double>(now - buy.timestamp) / 1000.0;
    double sell_age = static_cast<double>(now - sell.timestamp) / 1000.0;
    double freshness_score = std::max(0.0, 1.0 - (buy_age + sell_age) / (2.0 * max_age));

    // Liquidity score
    Decimal min_liq = std::min(buy.liquidity, sell.liquidity);
    double liquidity_score;
    if (min_liq > Decimal::from_double(100000.0)) {
        liquidity_score = 1.0;
    } else if (min_liq > Decimal::from_double(10000.0)) {
        liquidity_score = 0.8;
    } else {
        liquidity_score = 0.5;
    }

    // Latency score
    double avg_latency = static_cast<double>(buy.latency + sell.latency) / 2.0;
    double latency_score = std::max(0.0, 1.0 - avg_latency / 1000.0);

    // Weighted average
    return 0.4 * freshness_score + 0.4 * liquidity_score + 0.2 * latency_score;
}

}  // namespace lx::trading::arbitrage
