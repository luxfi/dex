// LX C++ SDK - Basic Usage Example
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT

#include <lxdex/client.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <csignal>
#include <atomic>

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down...\n";
    g_running = false;
}

void print_orderbook(const lxdex::OrderBook& ob) {
    std::cout << "\n=== " << ob.symbol << " Orderbook ===\n";
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Asks:\n";
    size_t ask_count = std::min(ob.asks.size(), size_t{5});
    for (size_t i = ask_count; i > 0; --i) {
        const auto& level = ob.asks[i - 1];
        std::cout << "  " << std::setw(12) << level.price
                  << " | " << std::setw(10) << level.size << "\n";
    }

    std::cout << "----------------------------\n";

    std::cout << "Bids:\n";
    for (size_t i = 0; i < std::min(ob.bids.size(), size_t{5}); ++i) {
        const auto& level = ob.bids[i];
        std::cout << "  " << std::setw(12) << level.price
                  << " | " << std::setw(10) << level.size << "\n";
    }

    std::cout << "\nSpread: " << ob.spread() << " ("
              << std::setprecision(4) << ob.spread_percentage() << "%)\n";
    std::cout << "Mid: " << std::setprecision(2) << ob.mid_price() << "\n";
}

void print_trade(const lxdex::Trade& trade) {
    std::cout << "[Trade] " << trade.symbol
              << " " << (trade.side == lxdex::Side::Buy ? "BUY" : "SELL")
              << " " << trade.size << " @ " << trade.price
              << " (id: " << trade.trade_id << ")\n";
}

void print_order(const lxdex::Order& order) {
    std::cout << "[Order] " << order.symbol
              << " " << (order.side == lxdex::Side::Buy ? "BUY" : "SELL")
              << " " << order.size << " @ " << order.price
              << " status: ";

    switch (order.status) {
        case lxdex::OrderStatus::Open: std::cout << "OPEN"; break;
        case lxdex::OrderStatus::Partial: std::cout << "PARTIAL"; break;
        case lxdex::OrderStatus::Filled: std::cout << "FILLED"; break;
        case lxdex::OrderStatus::Cancelled: std::cout << "CANCELLED"; break;
        case lxdex::OrderStatus::Rejected: std::cout << "REJECTED"; break;
    }
    std::cout << " (id: " << order.order_id << ")\n";
}

int main(int argc, char* argv[]) {
    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "LX C++ SDK Example\n";
    std::cout << "======================\n\n";

    // Configure client
    lxdex::ClientConfig config;
    config.ws_url = "ws://localhost:8081";
    config.auto_reconnect = true;
    config.max_reconnect_attempts = 5;

    // Parse command line args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--url" && i + 1 < argc) {
            config.ws_url = argv[++i];
        } else if (arg == "--key" && i + 1 < argc) {
            config.api_key = argv[++i];
        } else if (arg == "--secret" && i + 1 < argc) {
            config.api_secret = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --url URL      WebSocket URL (default: ws://localhost:8081)\n"
                      << "  --key KEY      API key for authentication\n"
                      << "  --secret SEC   API secret for authentication\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }

    // Create client
    auto client = lxdex::make_client(config);

    // Setup callbacks
    client->on_error([](const lxdex::Error& err) {
        std::cerr << "[Error] " << err.message;
        if (!err.request_id.empty()) {
            std::cerr << " (request: " << err.request_id << ")";
        }
        std::cerr << "\n";
    });

    client->on_connection([](lxdex::ConnectionState state) {
        std::cout << "[Connection] State: ";
        switch (state) {
            case lxdex::ConnectionState::Disconnected:
                std::cout << "DISCONNECTED";
                break;
            case lxdex::ConnectionState::Connecting:
                std::cout << "CONNECTING";
                break;
            case lxdex::ConnectionState::Connected:
                std::cout << "CONNECTED";
                break;
            case lxdex::ConnectionState::Reconnecting:
                std::cout << "RECONNECTING";
                break;
            case lxdex::ConnectionState::Failed:
                std::cout << "FAILED";
                break;
        }
        std::cout << "\n";
    });

    client->on_order([](const lxdex::Order& order) {
        print_order(order);
    });

    // Connect
    std::cout << "Connecting to " << config.ws_url << "...\n";
    auto err = client->connect();
    if (err) {
        std::cerr << "Failed to connect: " << err.message << "\n";
        return 1;
    }
    std::cout << "Connected!\n\n";

    // Ping test
    auto ping_result = client->ping();
    if (ping_result.ok()) {
        std::cout << "Ping: " << ping_result.value << " us\n\n";
    }

    // Authenticate if credentials provided
    if (!config.api_key.empty()) {
        std::cout << "Authenticating...\n";
        err = client->authenticate();
        if (err) {
            std::cerr << "Authentication failed: " << err.message << "\n";
        } else {
            std::cout << "Authenticated!\n\n";
        }
    }

    // Subscribe to orderbook updates
    std::vector<std::string> symbols = {"BTC-USDT", "ETH-USDT"};

    std::cout << "Subscribing to orderbook updates...\n";
    err = client->subscribe_orderbook(symbols, [](const lxdex::OrderBook& ob) {
        print_orderbook(ob);
    });
    if (err) {
        std::cerr << "Failed to subscribe: " << err.message << "\n";
    }

    // Subscribe to trades
    std::cout << "Subscribing to trade updates...\n";
    err = client->subscribe_trades(symbols, [](const lxdex::Trade& trade) {
        print_trade(trade);
    });
    if (err) {
        std::cerr << "Failed to subscribe: " << err.message << "\n";
    }

    // Example: Place a test order (if authenticated)
    if (client->is_authenticated()) {
        std::cout << "\nPlacing test order...\n";

        lxdex::Order order;
        order.symbol = "BTC-USDT";
        order.side = lxdex::Side::Buy;
        order.type = lxdex::OrderType::Limit;
        order.price = 40000.0;
        order.size = 0.001;
        order.time_in_force = lxdex::TimeInForce::GTC;
        order.client_id = lxdex::Client::generate_client_id();

        auto result = client->place_order(order);
        if (result.ok()) {
            std::cout << "Order placed! ID: " << result.value.order_id
                      << " Status: " << result.value.status << "\n";

            // Cancel the order after a brief delay
            std::this_thread::sleep_for(std::chrono::seconds(2));

            std::cout << "Cancelling order...\n";
            err = client->cancel_order(result.value.order_id);
            if (err) {
                std::cerr << "Failed to cancel: " << err.message << "\n";
            } else {
                std::cout << "Order cancelled!\n";
            }
        } else {
            std::cerr << "Failed to place order: " << result.error.message << "\n";
        }
    }

    // Main loop - wait for data
    std::cout << "\nWaiting for market data (Ctrl+C to exit)...\n\n";

    int update_count = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Periodic metrics report
        if (++update_count % 30 == 0) {
            auto metrics = client->metrics();
            std::cout << "\n--- Metrics ---\n"
                      << "Messages sent:     " << metrics.messages_sent << "\n"
                      << "Messages received: " << metrics.messages_received << "\n"
                      << "Orders sent:       " << metrics.orders_sent << "\n"
                      << "Trades received:   " << metrics.trades_received << "\n"
                      << "Last latency:      " << metrics.last_latency_us << " us\n"
                      << "Reconnects:        " << metrics.reconnect_count << "\n"
                      << "Errors:            " << metrics.error_count << "\n\n";
        }

        // Check connection
        if (!client->is_connected()) {
            std::cout << "Connection lost, waiting for reconnect...\n";
        }
    }

    // Cleanup
    std::cout << "\nDisconnecting...\n";
    client->disconnect();
    std::cout << "Done.\n";

    return 0;
}
