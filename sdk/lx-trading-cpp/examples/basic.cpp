// LX Trading SDK - Basic Example
// Demonstrates client setup, market data, and order placement

#include <lx/trading/client.hpp>
#include <lx/trading/config.hpp>
#include <iostream>

using namespace lx::trading;

int main() {
    // Build configuration
    Config config;
    config.general.smart_routing = true;
    config.general.timeout_ms = 30000;

    // Add LX DEX
    config.with_native("lx_dex",
        NativeVenueConfig::lx_dex("https://api.lx.exchange")
            .with_credentials("your-api-key", "your-api-secret"));

    // Add LX AMM
    config.with_native("lx_amm",
        NativeVenueConfig::lx_amm("https://amm.lx.exchange"));

    // Risk management
    config.set_max_order_size(Decimal::from_double(10.0))
          .set_max_position_size(Decimal::from_double(100.0))
          .set_max_daily_loss(Decimal::from_double(1000.0));

    // Create client
    Client client(config);

    try {
        // Connect to all venues
        std::cout << "Connecting to venues...\n";
        client.connect().get();

        // List connected venues
        for (const auto& venue : client.venues()) {
            std::cout << "Connected: " << venue.name
                      << " (latency: " << venue.latency_ms.value_or(0) << "ms)\n";
        }

        // Get aggregated orderbook
        std::cout << "\nFetching BTC-USDC orderbook...\n";
        auto book = client.aggregated_orderbook("BTC-USDC").get();

        if (auto best_bid = book.best_bid()) {
            auto [price, venue, qty] = *best_bid;
            std::cout << "Best bid: " << price.to_string()
                      << " @ " << venue << "\n";
        }

        if (auto best_ask = book.best_ask()) {
            auto [price, venue, qty] = *best_ask;
            std::cout << "Best ask: " << price.to_string()
                      << " @ " << venue << "\n";
        }

        // Get ticker
        std::cout << "\nFetching ticker...\n";
        auto ticker = client.ticker("BTC-USDC").get();
        std::cout << "Mid price: " << ticker.mid_price().value_or(Decimal::zero()).to_string() << "\n";
        std::cout << "Spread: " << ticker.spread().value_or(Decimal::zero()).to_string() << "\n";

        // Get balances
        std::cout << "\nFetching balances...\n";
        auto balances = client.balances().get();
        for (const auto& bal : balances) {
            std::cout << bal.asset << ": " << bal.total().to_string() << "\n";
        }

        // Place a limit order
        std::cout << "\nPlacing limit buy order...\n";
        auto order = client.limit_buy(
            "BTC-USDC",
            Decimal::from_double(0.1),
            Decimal::from_double(40000.0)).get();

        std::cout << "Order placed: " << order.order_id << "\n";
        std::cout << "Status: " << to_string(order.status) << "\n";

        // Cancel the order
        std::cout << "\nCancelling order...\n";
        auto cancelled = client.cancel_order(order.order_id, order.symbol, order.venue).get();
        std::cout << "Cancelled: " << to_string(cancelled.status) << "\n";

        // Disconnect
        client.disconnect().get();
        std::cout << "\nDisconnected.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
