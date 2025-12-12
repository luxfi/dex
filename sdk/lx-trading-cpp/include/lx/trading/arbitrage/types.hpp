// LX Trading SDK - Arbitrage Types
// LX-First Strategy: LX DEX is the price oracle
//
// Key Insight: LX DEX is the FASTEST venue (nanosecond updates, 200ms blocks)
// By the time other venues update, LX has already moved.
// LX DEX price is the "TRUTH" - other venues are always STALE.
// Arbitrage = correcting stale venues to match LX

#pragma once

#include <lx/trading/types.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lx::trading::arbitrage {

/// Cross-chain transport protocol
enum class CrossChainTransport : uint8_t {
    /// Warp - Lux native messaging between subnets (<500ms)
    Warp = 0,
    /// Teleport - EVM bridge for external chains (~30s)
    Teleport = 1,
    /// Direct - Same chain, no bridge needed
    Direct = 2,
    /// CEX API - API calls for centralized exchanges
    CexApi = 3
};

inline constexpr const char* to_string(CrossChainTransport t) noexcept {
    switch (t) {
        case CrossChainTransport::Warp: return "warp";
        case CrossChainTransport::Teleport: return "teleport";
        case CrossChainTransport::Direct: return "direct";
        case CrossChainTransport::CexApi: return "cex_api";
    }
    return "unknown";
}

/// Type of blockchain
enum class ChainType : uint8_t {
    /// Lux subnet (Warp-enabled)
    LuxSubnet = 0,
    /// EVM-compatible chain
    Evm = 1,
    /// Centralized exchange
    Cex = 2
};

inline constexpr const char* to_string(ChainType t) noexcept {
    switch (t) {
        case ChainType::LuxSubnet: return "lux_subnet";
        case ChainType::Evm: return "evm";
        case ChainType::Cex: return "cex";
    }
    return "unknown";
}

/// Type of arbitrage
enum class ArbType : uint8_t {
    /// Simple buy-low-sell-high
    Simple = 0,
    /// Triangular A->B->C->A
    Triangular = 1,
    /// Multi-hop complex routes
    MultiHop = 2,
    /// CEX-DEX arbitrage
    CexDex = 3,
    /// DEX flash swap
    FlashSwap = 4
};

inline constexpr const char* to_string(ArbType t) noexcept {
    switch (t) {
        case ArbType::Simple: return "simple";
        case ArbType::Triangular: return "triangular";
        case ArbType::MultiHop: return "multi_hop";
        case ArbType::CexDex: return "cex_dex";
        case ArbType::FlashSwap: return "flash_swap";
    }
    return "unknown";
}

/// Price feed from a specific venue/chain
struct PriceSource {
    std::string chain_id;
    std::string venue;
    std::string symbol;
    Decimal bid;
    Decimal ask;
    Decimal liquidity;
    int64_t timestamp{0};  // Unix timestamp in milliseconds
    int64_t latency{0};    // Latency in milliseconds

    [[nodiscard]] Decimal mid_price() const noexcept {
        return (bid + ask) / Decimal::from_double(2.0);
    }

    [[nodiscard]] Decimal spread() const noexcept {
        return ask - bid;
    }

    [[nodiscard]] Decimal spread_bps() const noexcept {
        if (bid.is_zero()) return Decimal::zero();
        return ((ask - bid) / bid) * Decimal::from_double(10000.0);
    }
};

/// LX DEX price - the reference/oracle
struct LxPrice {
    std::string symbol;
    Decimal bid;
    Decimal ask;
    Decimal mid;
    int64_t timestamp{0};
    uint64_t block_num{0};

    static LxPrice create(const std::string& sym, Decimal b, Decimal a, int64_t ts, uint64_t block) {
        LxPrice price;
        price.symbol = sym;
        price.bid = b;
        price.ask = a;
        price.mid = (b + a) / Decimal::from_double(2.0);
        price.timestamp = ts;
        price.block_num = block;
        return price;
    }
};

/// Price from a 'slow' venue
struct VenuePrice {
    std::string venue;
    std::string symbol;
    Decimal bid;
    Decimal ask;
    int64_t timestamp{0};
    int64_t latency{0};  // How far behind LX this venue typically is (ms)
    bool stale{false};   // Is this price stale relative to LX?
};

/// Single leg of an arbitrage
struct Route {
    std::string chain_id;
    std::string venue;
    std::string action;  // "buy" or "sell"
    std::string token_in;
    std::string token_out;
    Decimal amount_in;
    Decimal expected_out;
    Decimal min_amount_out;
    std::vector<uint8_t> swap_data;
};

/// Detected arbitrage opportunity
struct ArbitrageOpportunity {
    std::string id;
    ArbType arb_type{ArbType::Simple};
    std::vector<Route> routes;
    PriceSource buy_source;
    PriceSource sell_source;
    Decimal spread_bps;      // Spread in basis points
    Decimal estimated_pnl;
    Decimal max_size;        // Limited by liquidity
    Decimal gas_cost_usd;
    Decimal bridge_cost_usd;
    Decimal net_pnl;
    double confidence{0.0};  // 0-1, based on price freshness and liquidity
    int64_t expires_at{0};

    [[nodiscard]] bool is_profitable() const noexcept {
        return net_pnl.is_positive();
    }

    [[nodiscard]] bool is_expired(int64_t now) const noexcept {
        return now > expires_at;
    }
};

/// LX-first arbitrage opportunity
struct LxFirstOpportunity {
    std::string id;
    std::string symbol;
    int64_t timestamp{0};
    LxPrice lx_price;
    std::string stale_venue;
    VenuePrice stale_price;
    int64_t staleness{0};   // Staleness in milliseconds
    std::string side;       // "buy" or "sell"
    Decimal divergence;
    Decimal divergence_bps;
    Decimal expected_profit;
    Decimal max_size;
    double confidence{0.0};
};

/// Unified arbitrage opportunity across venues
struct UnifiedOpportunity {
    std::string id;
    std::string symbol;
    int64_t timestamp{0};
    int64_t expires_at{0};
    std::string buy_venue;
    Decimal buy_price;
    Decimal buy_size;
    std::string sell_venue;
    Decimal sell_price;
    Decimal sell_size;
    Decimal spread;
    Decimal spread_bps;
    Decimal max_size;
    Decimal gross_profit;
    Decimal est_fees;
    Decimal net_profit;
    double confidence{0.0};
    int64_t latency{0};
};

/// Executed arbitrage result
struct UnifiedExecution {
    std::string id;
    UnifiedOpportunity opportunity;
    int64_t start_time{0};
    int64_t end_time{0};
    std::string status;  // "executing", "completed", "failed"
    std::optional<std::string> buy_order_id;
    std::optional<std::string> sell_order_id;
    Decimal actual_profit;
    Decimal fees;
    std::optional<std::string> error;
};

/// Arbitrage statistics
struct UnifiedArbStats {
    uint64_t total_executions{0};
    uint64_t successful_executions{0};
    Decimal total_pnl;
    double win_rate{0.0};
};

/// Configuration for unified arbitrage system
struct UnifiedArbConfig {
    Decimal min_spread_bps{Decimal::from_double(10.0)};
    Decimal min_profit{Decimal::from_double(5.0)};
    Decimal max_position_size{Decimal::from_double(10000.0)};
    Decimal max_total_exposure{Decimal::from_double(100000.0)};
    std::vector<std::string> symbols{"BTC-USDC", "ETH-USDC", "LUX-USDC"};
    std::vector<std::string> venue_priority{"lx_dex", "binance", "mexc", "lx_amm"};
    uint64_t scan_interval_ms{100};
    uint64_t execute_timeout_ms{5000};
    Decimal max_daily_loss{Decimal::from_double(1000.0)};
    uint32_t max_trades_per_day{100};

    static UnifiedArbConfig defaults() {
        return UnifiedArbConfig{};
    }
};

/// Configuration for LX-first strategy
struct LxFirstConfig {
    int64_t max_staleness_ms{2000};
    Decimal min_divergence_bps{Decimal::from_double(10.0)};
    Decimal min_profit{Decimal::from_double(5.0)};
    Decimal max_position_size{Decimal::from_double(1000.0)};
    std::vector<std::string> symbols{"BTC-USDC", "ETH-USDC", "LUX-USDC"};
    std::map<std::string, int64_t> venue_latencies{
        {"binance", 50},
        {"mexc", 100},
        {"okx", 80},
        {"uniswap", 12000},
        {"pancakeswap", 3000}
    };

    static LxFirstConfig defaults() {
        return LxFirstConfig{};
    }
};

/// Configuration for arbitrage scanner
struct ScannerConfig {
    Decimal min_spread_bps{Decimal::from_double(10.0)};
    Decimal min_profit_usd{Decimal::from_double(10.0)};
    int64_t max_price_age_ms{5000};
    std::vector<std::string> symbols{"BTC", "ETH", "LUX", "SOL", "AVAX"};
    std::vector<std::string> chain_ids{"lux", "ethereum", "bsc", "arbitrum", "polygon"};
    uint64_t scan_interval_ms{100};
    size_t max_concurrency{50};

    static ScannerConfig defaults() {
        return ScannerConfig{};
    }
};

/// Information about a chain
struct CrossChainInfo {
    std::string chain_id;
    std::string name;
    ChainType chain_type{ChainType::LuxSubnet};
    uint64_t block_time_ms{0};
    uint64_t finality_ms{0};
    bool warp_supported{false};
    bool teleport_supported{false};
    std::vector<std::string> venues;
};

/// Configuration for cross-chain routing
struct CrossChainConfig {
    bool warp_enabled{true};
    std::optional<std::string> warp_endpoint;
    uint64_t warp_timeout_ms{5000};
    bool teleport_enabled{true};
    std::optional<std::string> teleport_endpoint;
    uint64_t teleport_timeout_ms{60000};
    std::map<std::string, CrossChainInfo> chains;

    static CrossChainConfig defaults();
};

/// Enhanced opportunity with routing information
struct EnhancedOpportunity {
    UnifiedOpportunity base;
    CrossChainTransport transport{CrossChainTransport::Direct};
    int64_t estimated_latency{0};
    Decimal bridge_cost;
    Decimal adjusted_net_profit;
};

/// Bridge transaction status
struct BridgeStatus {
    std::string tx_id;
    std::string status;  // pending, confirming, completed, failed
    std::string source_chain;
    std::string dest_chain;
    Decimal amount;
    Decimal fee;
    std::string source_tx;
    std::optional<std::string> dest_tx;
    int64_t timestamp{0};
};

// Callback types
using OpportunityCallback = std::function<void(const ArbitrageOpportunity&)>;
using LxFirstCallback = std::function<void(const LxFirstOpportunity&)>;
using UnifiedCallback = std::function<void(const UnifiedOpportunity&)>;

}  // namespace lx::trading::arbitrage
