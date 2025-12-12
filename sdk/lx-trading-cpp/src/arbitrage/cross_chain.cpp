// LX Trading SDK - Cross-Chain Router Implementation

#include <lx/trading/arbitrage/cross_chain.hpp>
#include <algorithm>

namespace lx::trading::arbitrage {

CrossChainRouter::CrossChainRouter(CrossChainConfig config)
    : config_(std::move(config)) {}

void CrossChainRouter::set_warp_client(std::shared_ptr<WarpClient> client) {
    std::lock_guard<std::mutex> lock(warp_mutex_);
    warp_client_ = std::move(client);
}

void CrossChainRouter::set_teleport_client(std::shared_ptr<TeleportClient> client) {
    std::lock_guard<std::mutex> lock(teleport_mutex_);
    teleport_client_ = std::move(client);
}

std::shared_ptr<WarpClient> CrossChainRouter::warp() const {
    std::lock_guard<std::mutex> lock(warp_mutex_);
    return warp_client_;
}

std::shared_ptr<TeleportClient> CrossChainRouter::teleport() const {
    std::lock_guard<std::mutex> lock(teleport_mutex_);
    return teleport_client_;
}

CrossChainTransport CrossChainRouter::determine_transport(
    const std::string& source_chain,
    const std::string& dest_chain) const {

    auto src_it = config_.chains.find(source_chain);
    auto dst_it = config_.chains.find(dest_chain);

    // Same chain = direct
    if (source_chain == dest_chain) {
        return CrossChainTransport::Direct;
    }

    // CEX = API
    if (src_it != config_.chains.end() && src_it->second.chain_type == ChainType::Cex) {
        return CrossChainTransport::CexApi;
    }
    if (dst_it != config_.chains.end() && dst_it->second.chain_type == ChainType::Cex) {
        return CrossChainTransport::CexApi;
    }

    // Both Lux subnets = Warp (fastest)
    if (src_it != config_.chains.end() && dst_it != config_.chains.end()) {
        const auto& src = src_it->second;
        const auto& dst = dst_it->second;

        if (src.chain_type == ChainType::LuxSubnet &&
            dst.chain_type == ChainType::LuxSubnet &&
            src.warp_supported && dst.warp_supported &&
            config_.warp_enabled) {
            return CrossChainTransport::Warp;
        }
    }

    // Both EVM or mixed = Teleport
    if (src_it != config_.chains.end() && dst_it != config_.chains.end()) {
        const auto& src = src_it->second;
        const auto& dst = dst_it->second;

        if (src.teleport_supported && dst.teleport_supported && config_.teleport_enabled) {
            return CrossChainTransport::Teleport;
        }
    }

    // No viable transport - return Direct as fallback
    return CrossChainTransport::Direct;
}

int64_t CrossChainRouter::estimate_latency(
    const std::string& source_chain,
    const std::string& dest_chain) const {

    CrossChainTransport transport = determine_transport(source_chain, dest_chain);

    switch (transport) {
        case CrossChainTransport::Direct:
            return 0;
        case CrossChainTransport::Warp:
            return 500;  // Sub-second
        case CrossChainTransport::CexApi:
            return 100;  // API call
        case CrossChainTransport::Teleport: {
            auto src_it = config_.chains.find(source_chain);
            if (src_it != config_.chains.end()) {
                return static_cast<int64_t>(src_it->second.finality_ms) + 10000;  // Finality + processing
            }
            return 3600000;  // 1 hour fallback
        }
    }
    return 3600000;
}

Decimal CrossChainRouter::estimate_cost(
    const std::string& source_chain,
    const std::string& dest_chain,
    const std::string& token,
    Decimal amount) const {

    CrossChainTransport transport = determine_transport(source_chain, dest_chain);

    switch (transport) {
        case CrossChainTransport::Direct:
            return Decimal::zero();
        case CrossChainTransport::Warp:
            return Decimal::from_double(0.001);  // Nearly free
        case CrossChainTransport::CexApi:
            return Decimal::zero();  // No bridge cost
        case CrossChainTransport::Teleport: {
            std::lock_guard<std::mutex> lock(teleport_mutex_);
            if (teleport_client_) {
                try {
                    return teleport_client_->estimate_bridge_fee(dest_chain, token, amount);
                } catch (...) {
                    return Decimal::from_double(1.0);  // Fallback estimate
                }
            }
            return Decimal::from_double(1.0);  // Estimate $1
        }
    }
    return Decimal::zero();
}

std::string CrossChainRouter::venue_to_chain(const std::string& venue) const {
    for (const auto& [chain_id, info] : config_.chains) {
        for (const auto& v : info.venues) {
            if (v == venue) {
                return chain_id;
            }
        }
    }
    return venue;  // Fallback to venue name
}

EnhancedOpportunity CrossChainRouter::enhance_opportunity(const UnifiedOpportunity& opp) const {
    std::string buy_chain = venue_to_chain(opp.buy_venue);
    std::string sell_chain = venue_to_chain(opp.sell_venue);

    CrossChainTransport transport = determine_transport(buy_chain, sell_chain);
    int64_t estimated_latency = estimate_latency(buy_chain, sell_chain);
    Decimal bridge_cost = estimate_cost(buy_chain, sell_chain, opp.symbol, opp.max_size);

    return EnhancedOpportunity{
        .base = opp,
        .transport = transport,
        .estimated_latency = estimated_latency,
        .bridge_cost = bridge_cost,
        .adjusted_net_profit = opp.net_profit - bridge_cost
    };
}

}  // namespace lx::trading::arbitrage
