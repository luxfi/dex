/**
 * LX-First Arbitrage Bot Example
 *
 * This bot uses the LX-First strategy where LX DEX prices are treated
 * as the "truth" (fastest venue with nanosecond updates, 200ms blocks).
 * Other venues are always stale by comparison.
 *
 * Arbitrage = exploiting stale venues before they catch up to LX prices.
 *
 * Cross-chain transport options:
 * - Warp: For Lux subnet communication (<500ms)
 * - Teleport: For EVM chain bridging (~30s)
 * - CEX API: Direct trading (instant)
 *
 * NO SMART CONTRACTS - just coordinated trades through unified SDK.
 */

#include <lx/trading/lx_dex.hpp>
#include <lx/trading/arbitrage/arbitrage.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <csignal>
#include <mutex>

using namespace lx::trading;
using namespace lx::trading::arbitrage;

// ============================================
// Global state
// ============================================

std::atomic<bool> g_running{false};
std::mutex g_stats_mutex;

struct Stats {
    uint64_t total_opportunities = 0;
    uint64_t total_executions = 0;
    Decimal total_pnl{"0"};
};

Stats g_stats;

// ============================================
// Arbitrage Bot Class
// ============================================

class ArbitrageBot {
public:
    ArbitrageBot() = default;

    bool start() {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "LX-FIRST ARBITRAGE BOT" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::endl;

        // Initialize DEX client
        LxDexConfig dex_config;
        dex_config.endpoint = get_env("LX_DEX_ENDPOINT", "wss://dex.lux.network/ws");
        dex_config.api_key = get_env("LX_API_KEY", "");

        dex_ = std::make_shared<LxDex>(dex_config);
        if (!dex_->connect()) {
            std::cerr << "Failed to connect to LX DEX" << std::endl;
            return false;
        }
        std::cout << "[OK] Connected to LX DEX" << std::endl;

        // Initialize LX-First strategy
        LxFirstConfig lx_config;
        lx_config.max_staleness_ms = 2000;
        lx_config.min_divergence_bps = Decimal("10");
        lx_config.min_profit = Decimal("5");
        lx_config.max_position_size = Decimal("10000");
        lx_config.symbols = {"BTC-USDC", "ETH-USDC", "LUX-USDC"};
        lx_config.venue_latencies = {
            {"binance", 50},
            {"mexc", 100},
            {"okx", 80},
            {"uniswap", 12000},
            {"pancakeswap", 3000}
        };

        lx_first_ = std::make_shared<LxFirstArbitrage>(lx_config);
        lx_first_->on_opportunity([this](const LxFirstOpportunity& opp) {
            on_lx_first_opportunity(opp);
        });
        std::cout << "[OK] LX-First strategy initialized" << std::endl;

        // Initialize Scanner
        ScannerConfig scanner_config;
        scanner_config.min_spread_bps = Decimal("10");
        scanner_config.min_profit_usd = Decimal("10");
        scanner_config.max_price_age_ms = 5000;
        scanner_config.symbols = {"BTC", "ETH", "LUX", "SOL", "AVAX"};
        scanner_config.chain_ids = {"lux", "ethereum", "bsc", "arbitrum", "polygon"};
        scanner_config.scan_interval_ms = 100;

        scanner_ = std::make_shared<Scanner>(scanner_config);
        scanner_->on_opportunity([](const ArbitrageOpportunity& opp) {
            std::cout << "[SCANNER] " << to_string(opp.type) << ": "
                      << opp.buy_source.venue << " -> " << opp.sell_source.venue
                      << " | Spread: " << opp.spread_bps.to_string() << " bps"
                      << " | Net PnL: $" << opp.net_pnl.to_string()
                      << std::endl;
        });
        std::cout << "[OK] Scanner initialized" << std::endl;

        // Initialize Cross-Chain Router
        router_ = std::make_shared<CrossChainRouter>(default_cross_chain_config());
        std::cout << "[OK] Cross-chain router initialized" << std::endl;

        // Start all systems
        lx_first_->start();
        scanner_->start();
        g_running = true;

        std::cout << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "BOT RUNNING - Press Ctrl+C to stop" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::endl;
        std::cout << "Monitoring symbols: BTC-USDC, ETH-USDC, LUX-USDC" << std::endl;
        std::cout << "Min divergence: 10 bps" << std::endl;
        std::cout << "Min profit: $5" << std::endl;
        std::cout << std::endl;

        // Start price feed simulator thread
        price_feed_thread_ = std::thread([this]() {
            simulate_price_feeds();
        });

        // Start stats reporter thread
        stats_thread_ = std::thread([this]() {
            report_stats();
        });

        return true;
    }

    void stop() {
        std::cout << "\nShutting down..." << std::endl;
        g_running = false;

        lx_first_->stop();
        scanner_->stop();

        if (price_feed_thread_.joinable()) {
            price_feed_thread_.join();
        }
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }

        print_final_stats();
    }

private:
    void on_lx_first_opportunity(const LxFirstOpportunity& opp) {
        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            g_stats.total_opportunities++;
        }

        std::cout << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "LX-FIRST OPPORTUNITY DETECTED" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Symbol:          " << opp.symbol << std::endl;
        std::cout << "LX Price:        $" << opp.lx_price.mid.to_string() << std::endl;
        std::cout << "Stale Venue:     " << opp.stale_venue << std::endl;
        std::cout << "Stale Bid/Ask:   $" << opp.stale_price.bid.to_string()
                  << " / $" << opp.stale_price.ask.to_string() << std::endl;
        std::cout << "Staleness:       " << opp.staleness << "ms" << std::endl;
        std::cout << "Side:            " << opp.side << std::endl;
        std::cout << "Divergence:      " << opp.divergence_bps.to_string() << " bps" << std::endl;
        std::cout << "Expected Profit: $" << opp.expected_profit.to_string() << std::endl;
        std::cout << "Confidence:      " << (opp.confidence * 100) << "%" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Execute if confidence is high enough
        if (opp.confidence > 0.8) {
            execute_lx_first(opp);
        }
    }

    void execute_lx_first(const LxFirstOpportunity& opp) {
        std::cout << "\n[EXECUTING] " << opp.id << "..." << std::endl;

        // Determine cross-chain transport
        std::string buy_chain = router_->venue_to_chain(opp.stale_venue);
        std::string sell_chain = "lux_mainnet";
        auto transport = router_->determine_transport(buy_chain, sell_chain);
        auto latency = router_->estimate_latency(buy_chain, sell_chain);

        std::cout << "  Transport: " << to_string(transport) << std::endl;
        std::cout << "  Est. Latency: " << latency << "ms" << std::endl;

        if (opp.side == "buy") {
            std::cout << "  Buying on " << opp.stale_venue << "..." << std::endl;
            // In production: place actual order
            // auto order = cex_client->place_order(...)

            std::cout << "  Hedging on LX DEX..." << std::endl;
            // In production: place hedge order
            // auto hedge = dex_->spot()->sell(...)
        } else {
            std::cout << "  Selling on " << opp.stale_venue << "..." << std::endl;
            // In production: place actual order

            std::cout << "  Hedging on LX DEX..." << std::endl;
            // In production: place hedge order
        }

        // Simulate successful execution
        Decimal profit = opp.expected_profit * Decimal("0.8");  // Simulate slippage

        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            g_stats.total_executions++;
            g_stats.total_pnl = g_stats.total_pnl + profit;
        }

        std::cout << "[SUCCESS] Executed " << opp.id
                  << " | Profit: $" << profit.to_string() << std::endl;
    }

    void simulate_price_feeds() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(-0.001, 0.001);
        std::uniform_real_distribution<> divergence_dist(-0.002, 0.002);
        std::uniform_int_distribution<> block_dist(1000000, 2000000);

        struct BasePair {
            std::string symbol;
            Decimal base;
        };

        std::vector<BasePair> base_prices = {
            {"BTC-USDC", Decimal("50000")},
            {"ETH-USDC", Decimal("3000")},
            {"LUX-USDC", Decimal("25")}
        };

        while (g_running) {
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();

            for (const auto& pair : base_prices) {
                // Simulate LX DEX price (the oracle)
                double variance = price_dist(gen);
                Decimal lx_mid = pair.base * Decimal(std::to_string(1.0 + variance));

                LxPrice lx_price;
                lx_price.symbol = pair.symbol;
                lx_price.bid = lx_mid * Decimal("0.9999");
                lx_price.ask = lx_mid * Decimal("1.0001");
                lx_price.mid = lx_mid;
                lx_price.timestamp = timestamp;
                lx_price.block_num = block_dist(gen);

                lx_first_->update_lx_price(lx_price);

                // Simulate stale CEX prices
                std::vector<std::pair<std::string, int>> venues = {
                    {"binance", 50},
                    {"mexc", 100}
                };

                for (const auto& [venue, latency] : venues) {
                    double divergence = divergence_dist(gen);
                    Decimal venue_mid = pair.base * Decimal(std::to_string(1.0 + divergence));

                    VenuePrice venue_price;
                    venue_price.venue = venue;
                    venue_price.symbol = pair.symbol;
                    venue_price.bid = venue_mid * Decimal("0.9998");
                    venue_price.ask = venue_mid * Decimal("1.0002");
                    venue_price.timestamp = timestamp - latency;
                    venue_price.latency = latency;
                    venue_price.stale = false;

                    lx_first_->update_venue_price(venue_price);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void report_stats() {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(30));

            if (!g_running) break;

            std::lock_guard<std::mutex> lock(g_stats_mutex);
            std::cout << std::endl;
            std::cout << std::string(40, '-') << std::endl;
            std::cout << "STATS" << std::endl;
            std::cout << "  Opportunities: " << g_stats.total_opportunities << std::endl;
            std::cout << "  Executions:    " << g_stats.total_executions << std::endl;
            std::cout << "  Total PnL:     $" << g_stats.total_pnl.to_string() << std::endl;
            if (g_stats.total_executions > 0) {
                Decimal avg_pnl = g_stats.total_pnl / Decimal(std::to_string(g_stats.total_executions));
                std::cout << "  Avg PnL:       $" << avg_pnl.to_string() << std::endl;
            }
            std::cout << std::string(40, '-') << std::endl;
        }
    }

    void print_final_stats() {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        std::cout << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "FINAL STATISTICS" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Total Opportunities: " << g_stats.total_opportunities << std::endl;
        std::cout << "Total Executions:    " << g_stats.total_executions << std::endl;
        std::cout << "Total PnL:           $" << g_stats.total_pnl.to_string() << std::endl;
        if (g_stats.total_executions > 0 && g_stats.total_opportunities > 0) {
            double win_rate = (static_cast<double>(g_stats.total_executions) /
                              static_cast<double>(g_stats.total_opportunities)) * 100.0;
            Decimal avg_pnl = g_stats.total_pnl / Decimal(std::to_string(g_stats.total_executions));
            std::cout << "Execution Rate:      " << win_rate << "%" << std::endl;
            std::cout << "Avg PnL per Trade:   $" << avg_pnl.to_string() << std::endl;
        }
        std::cout << std::string(50, '=') << std::endl;
    }

    static std::string get_env(const std::string& key, const std::string& default_value) {
        const char* value = std::getenv(key.c_str());
        return value ? value : default_value;
    }

    std::shared_ptr<LxDex> dex_;
    std::shared_ptr<LxFirstArbitrage> lx_first_;
    std::shared_ptr<Scanner> scanner_;
    std::shared_ptr<CrossChainRouter> router_;
    std::thread price_feed_thread_;
    std::thread stats_thread_;
};

// ============================================
// Main Entry Point
// ============================================

ArbitrageBot* g_bot = nullptr;

void signal_handler(int signal) {
    if (g_bot) {
        g_bot->stop();
    }
    std::exit(0);
}

int main() {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    ArbitrageBot bot;
    g_bot = &bot;

    if (!bot.start()) {
        return 1;
    }

    // Keep running until signal
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
