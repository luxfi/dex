// LX C++ CLI Trading Client
// Copyright (c) 2025 Lux Partners Limited
// SPDX-License-Identifier: MIT
//
// Command-line trading interface for LX WebSocket API.
// Connect to ws://localhost:8081 for real-time trading.

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

using json = nlohmann::json;
using WsClient = websocketpp::client<websocketpp::config::asio_client>;
using ConnectionHdl = websocketpp::connection_hdl;
using MessagePtr = websocketpp::config::asio_client::message_type::ptr;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

struct Config {
    std::string ws_url = "ws://localhost:8081";
    std::string api_key;
    std::string api_secret;
    bool verbose = false;
    bool interactive = true;
    std::vector<std::string> command_args;
};

//------------------------------------------------------------------------------
// WebSocket Client
//------------------------------------------------------------------------------

class Client {
public:
    explicit Client(const Config& config)
        : config_(config)
        , connected_(false)
        , authenticated_(false)
        , request_counter_(0)
        , running_(false)
    {
        ws_.clear_access_channels(websocketpp::log::alevel::all);
        ws_.clear_error_channels(websocketpp::log::elevel::all);
        ws_.init_asio();

        ws_.set_open_handler([this](ConnectionHdl) {
            std::lock_guard<std::mutex> lock(connect_mutex_);
            connected_ = true;
            connect_cv_.notify_all();
        });

        ws_.set_close_handler([this](ConnectionHdl) {
            std::lock_guard<std::mutex> lock(connect_mutex_);
            connected_ = false;
            connect_cv_.notify_all();
        });

        ws_.set_fail_handler([this](ConnectionHdl) {
            std::lock_guard<std::mutex> lock(connect_mutex_);
            connected_ = false;
            connect_cv_.notify_all();
        });

        ws_.set_message_handler([this](ConnectionHdl, MessagePtr msg) {
            on_message(msg);
        });
    }

    ~Client() {
        disconnect();
    }

    bool connect() {
        websocketpp::lib::error_code ec;
        auto con = ws_.get_connection(config_.ws_url, ec);
        if (ec) {
            std::cerr << "Connection error: " << ec.message() << "\n";
            return false;
        }

        connection_ = con->get_handle();
        ws_.connect(con);

        running_ = true;
        io_thread_ = std::thread([this]() { ws_.run(); });

        // Wait for connection
        {
            std::unique_lock<std::mutex> lock(connect_mutex_);
            bool ok = connect_cv_.wait_for(lock, std::chrono::seconds(10), [this]() {
                return connected_.load();
            });
            if (!ok) {
                std::cerr << "Connection timeout\n";
                return false;
            }
        }

        // Wait for connected message from server
        auto resp = wait_response("", std::chrono::seconds(5));
        if (resp && resp->contains("type") && (*resp)["type"] == "connected") {
            if (config_.verbose) {
                std::cout << "Connected to LX\n";
            }
        }

        return true;
    }

    void disconnect() {
        if (!running_) return;
        running_ = false;

        try {
            if (!connection_.expired()) {
                ws_.close(connection_, websocketpp::close::status::normal, "");
            }
        } catch (...) {}

        ws_.stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    bool is_connected() const { return connected_.load(); }

    bool authenticate() {
        if (config_.api_key.empty() || config_.api_secret.empty()) {
            return true; // No auth required
        }

        json msg = {
            {"type", "auth"},
            {"apiKey", config_.api_key},
            {"apiSecret", config_.api_secret},
            {"request_id", next_request_id()}
        };

        auto resp = send_and_wait(msg);
        if (!resp) {
            std::cerr << "Auth timeout\n";
            return false;
        }

        if (resp->contains("error") && !(*resp)["error"].is_null()) {
            std::cerr << "Auth failed: " << (*resp)["error"] << "\n";
            return false;
        }

        authenticated_ = true;
        return true;
    }

    // Place order
    std::optional<json> place_order(const std::string& symbol, const std::string& side,
                                     const std::string& type, double price, double size) {
        json order = {
            {"symbol", symbol},
            {"side", side},
            {"type", type},
            {"price", price},
            {"size", size}
        };

        json msg = {
            {"type", "place_order"},
            {"order", order},
            {"request_id", next_request_id()}
        };

        return send_and_wait(msg);
    }

    // Cancel order
    std::optional<json> cancel_order(uint64_t order_id) {
        json msg = {
            {"type", "cancel_order"},
            {"orderID", order_id},
            {"request_id", next_request_id()}
        };
        return send_and_wait(msg);
    }

    // Get positions
    std::optional<json> get_positions() {
        json msg = {
            {"type", "get_positions"},
            {"request_id", next_request_id()}
        };
        return send_and_wait(msg);
    }

    // Get open orders
    std::optional<json> get_orders() {
        json msg = {
            {"type", "get_orders"},
            {"request_id", next_request_id()}
        };
        return send_and_wait(msg);
    }

    // Get balances
    std::optional<json> get_balances() {
        json msg = {
            {"type", "get_balances"},
            {"request_id", next_request_id()}
        };
        return send_and_wait(msg);
    }

    // Subscribe to orderbook
    bool subscribe(const std::string& symbol) {
        json msg = {
            {"type", "subscribe"},
            {"symbols", json::array({symbol})},
            {"request_id", next_request_id()}
        };
        return send(msg);
    }

    // Get orderbook (via subscribe)
    std::optional<json> get_orderbook(const std::string& symbol) {
        subscribe(symbol);
        // Wait for orderbook message
        return wait_response("", std::chrono::seconds(5));
    }

    // Ping
    std::optional<json> ping() {
        json msg = {
            {"type", "ping"},
            {"request_id", next_request_id()}
        };
        return send_and_wait(msg);
    }

private:
    void on_message(MessagePtr msg) {
        try {
            json j = json::parse(msg->get_payload());

            if (config_.verbose) {
                std::cout << "<< " << j.dump(2) << "\n";
            }

            std::lock_guard<std::mutex> lock(response_mutex_);
            responses_.push(j);
            response_cv_.notify_all();
        } catch (const std::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << "\n";
        }
    }

    bool send(const json& msg) {
        if (!is_connected()) return false;

        try {
            std::string payload = msg.dump();
            if (config_.verbose) {
                std::cout << ">> " << msg.dump(2) << "\n";
            }
            ws_.send(connection_, payload, websocketpp::frame::opcode::text);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Send error: " << e.what() << "\n";
            return false;
        }
    }

    std::optional<json> wait_response(const std::string& request_id,
                                       std::chrono::seconds timeout = std::chrono::seconds(5)) {
        std::unique_lock<std::mutex> lock(response_mutex_);

        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!responses_.empty()) {
                json resp = responses_.front();
                responses_.pop();

                // If request_id specified, match it
                if (!request_id.empty()) {
                    if (resp.contains("request_id") && resp["request_id"] == request_id) {
                        return resp;
                    }
                    // Put it back if not matching (simple approach)
                    responses_.push(resp);
                } else {
                    return resp;
                }
            }

            response_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }

        return std::nullopt;
    }

    std::optional<json> send_and_wait(const json& msg,
                                       std::chrono::seconds timeout = std::chrono::seconds(5)) {
        std::string req_id = msg["request_id"];
        if (!send(msg)) {
            return std::nullopt;
        }

        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            std::unique_lock<std::mutex> lock(response_mutex_);
            if (!responses_.empty()) {
                json resp = responses_.front();
                responses_.pop();

                if (resp.contains("request_id") && resp["request_id"] == req_id) {
                    return resp;
                }

                // Handle non-matching responses (subscriptions, etc)
                if (resp.contains("type")) {
                    std::string type = resp["type"];
                    if (type != "connected" && type != "pong") {
                        print_response(resp);
                    }
                }
            }
            response_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }

        return std::nullopt;
    }

    std::string next_request_id() {
        return "req-" + std::to_string(++request_counter_);
    }

    void print_response(const json& resp) {
        if (resp.contains("error") && !resp["error"].is_null()) {
            std::cout << "Error: " << resp["error"] << "\n";
            return;
        }

        if (!resp.contains("type")) {
            std::cout << resp.dump(2) << "\n";
            return;
        }

        std::string type = resp["type"];

        if (type == "order_update" && resp.contains("data")) {
            std::cout << "Order Update: " << resp["data"].dump() << "\n";
        } else if (type == "position_update" && resp.contains("data")) {
            std::cout << "Position Update: " << resp["data"].dump() << "\n";
        } else if (type == "orderbook" && resp.contains("data")) {
            auto& data = resp["data"];
            if (data.contains("symbol")) {
                std::cout << "OrderBook [" << data["symbol"] << "]:\n";
            }
            if (data.contains("bids") && data["bids"].is_array()) {
                std::cout << "  Bids: " << data["bids"].size() << " levels\n";
                int count = 0;
                for (const auto& bid : data["bids"]) {
                    if (count++ >= 5) break;
                    if (bid.contains("price") && bid.contains("size")) {
                        std::cout << "    " << std::fixed << std::setprecision(2)
                                  << bid["price"].get<double>() << " @ "
                                  << std::setprecision(4) << bid["size"].get<double>() << "\n";
                    }
                }
            }
            if (data.contains("asks") && data["asks"].is_array()) {
                std::cout << "  Asks: " << data["asks"].size() << " levels\n";
                int count = 0;
                for (const auto& ask : data["asks"]) {
                    if (count++ >= 5) break;
                    if (ask.contains("price") && ask.contains("size")) {
                        std::cout << "    " << std::fixed << std::setprecision(2)
                                  << ask["price"].get<double>() << " @ "
                                  << std::setprecision(4) << ask["size"].get<double>() << "\n";
                    }
                }
            }
        } else {
            std::cout << resp.dump(2) << "\n";
        }
    }

    Config config_;
    WsClient ws_;
    ConnectionHdl connection_;
    std::thread io_thread_;

    std::atomic<bool> connected_;
    std::atomic<bool> authenticated_;
    std::atomic<uint64_t> request_counter_;
    std::atomic<bool> running_;

    std::mutex connect_mutex_;
    std::condition_variable connect_cv_;

    std::mutex response_mutex_;
    std::condition_variable response_cv_;
    std::queue<json> responses_;

    friend void print_message(Client& client, const json& msg);
};

//------------------------------------------------------------------------------
// CLI Interface
//------------------------------------------------------------------------------

void print_help() {
    std::cout << R"(
LX CLI Commands:

  place_order <symbol> <side> <type> <price> <size>
    Example: place_order BTC-USD buy limit 50000 0.1

  cancel_order <order_id>
    Example: cancel_order 12345

  get_orderbook <symbol>
    Example: get_orderbook BTC-USD

  get_positions
    Show all open positions

  get_orders
    Show all open orders

  get_balances
    Show account balances

  subscribe <symbol>
    Subscribe to orderbook updates

  ping
    Test connection latency

  help
    Show this help message

  quit / exit
    Exit the CLI
)";
}

void print_message(const json& msg) {
    if (msg.contains("error") && !msg["error"].is_null()) {
        std::cout << "Error: " << msg["error"] << "\n";
        return;
    }
    std::cout << msg.dump(2) << "\n";
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void run_interactive(Client& client) {
    std::cout << "LX CLI - Type 'help' for commands\n> ";

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) {
            std::cout << "> ";
            continue;
        }

        auto parts = split(line);
        if (parts.empty()) {
            std::cout << "> ";
            continue;
        }

        std::string cmd = parts[0];
        // Convert to lowercase
        for (auto& c : cmd) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (cmd == "help") {
            print_help();
        } else if (cmd == "quit" || cmd == "exit") {
            std::cout << "Goodbye\n";
            break;
        } else if (cmd == "place_order") {
            if (parts.size() < 6) {
                std::cout << "Usage: place_order <symbol> <side> <type> <price> <size>\n";
            } else {
                try {
                    double price = std::stod(parts[4]);
                    double size = std::stod(parts[5]);
                    auto resp = client.place_order(parts[1], parts[2], parts[3], price, size);
                    if (resp) {
                        print_message(*resp);
                    } else {
                        std::cout << "Timeout waiting for response\n";
                    }
                } catch (const std::exception& e) {
                    std::cout << "Invalid price or size: " << e.what() << "\n";
                }
            }
        } else if (cmd == "cancel_order") {
            if (parts.size() < 2) {
                std::cout << "Usage: cancel_order <order_id>\n";
            } else {
                try {
                    uint64_t order_id = std::stoull(parts[1]);
                    auto resp = client.cancel_order(order_id);
                    if (resp) {
                        print_message(*resp);
                    } else {
                        std::cout << "Timeout waiting for response\n";
                    }
                } catch (const std::exception& e) {
                    std::cout << "Invalid order ID: " << e.what() << "\n";
                }
            }
        } else if (cmd == "get_orderbook") {
            if (parts.size() < 2) {
                std::cout << "Usage: get_orderbook <symbol>\n";
            } else {
                auto resp = client.get_orderbook(parts[1]);
                if (resp) {
                    print_message(*resp);
                } else {
                    std::cout << "Subscribed to " << parts[1] << " orderbook\n";
                }
            }
        } else if (cmd == "get_positions") {
            auto resp = client.get_positions();
            if (resp) {
                print_message(*resp);
            } else {
                std::cout << "Timeout waiting for response\n";
            }
        } else if (cmd == "get_orders") {
            auto resp = client.get_orders();
            if (resp) {
                print_message(*resp);
            } else {
                std::cout << "Timeout waiting for response\n";
            }
        } else if (cmd == "get_balances") {
            auto resp = client.get_balances();
            if (resp) {
                print_message(*resp);
            } else {
                std::cout << "Timeout waiting for response\n";
            }
        } else if (cmd == "subscribe") {
            if (parts.size() < 2) {
                std::cout << "Usage: subscribe <symbol>\n";
            } else {
                if (client.subscribe(parts[1])) {
                    std::cout << "Subscribed to " << parts[1] << "\n";
                } else {
                    std::cout << "Failed to subscribe\n";
                }
            }
        } else if (cmd == "ping") {
            auto start = std::chrono::high_resolution_clock::now();
            auto resp = client.ping();
            auto end = std::chrono::high_resolution_clock::now();
            if (resp) {
                auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "Pong: " << latency.count() << " us\n";
            } else {
                std::cout << "Ping timeout\n";
            }
        } else {
            std::cout << "Unknown command: " << cmd << ". Type 'help' for commands.\n";
        }

        std::cout << "> ";
    }
}

void run_command(Client& client, const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "No command specified. Use -h for help.\n";
        return;
    }

    std::string cmd = args[0];
    // Convert to lowercase
    for (auto& c : cmd) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (cmd == "place_order") {
        if (args.size() < 6) {
            std::cerr << "Usage: lx-cli place_order <symbol> <side> <type> <price> <size>\n";
            std::exit(1);
        }
        try {
            double price = std::stod(args[4]);
            double size = std::stod(args[5]);
            auto resp = client.place_order(args[1], args[2], args[3], price, size);
            if (resp) {
                std::cout << resp->dump(2) << "\n";
            } else {
                std::cerr << "Timeout\n";
                std::exit(1);
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid price or size: " << e.what() << "\n";
            std::exit(1);
        }
    } else if (cmd == "cancel_order") {
        if (args.size() < 2) {
            std::cerr << "Usage: lx-cli cancel_order <order_id>\n";
            std::exit(1);
        }
        try {
            uint64_t order_id = std::stoull(args[1]);
            auto resp = client.cancel_order(order_id);
            if (resp) {
                std::cout << resp->dump(2) << "\n";
            } else {
                std::cerr << "Timeout\n";
                std::exit(1);
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid order ID: " << e.what() << "\n";
            std::exit(1);
        }
    } else if (cmd == "get_orderbook") {
        if (args.size() < 2) {
            std::cerr << "Usage: lx-cli get_orderbook <symbol>\n";
            std::exit(1);
        }
        auto resp = client.get_orderbook(args[1]);
        if (resp) {
            std::cout << resp->dump(2) << "\n";
        } else {
            std::cerr << "Timeout\n";
            std::exit(1);
        }
    } else if (cmd == "get_positions") {
        auto resp = client.get_positions();
        if (resp) {
            std::cout << resp->dump(2) << "\n";
        } else {
            std::cerr << "Timeout\n";
            std::exit(1);
        }
    } else if (cmd == "get_orders") {
        auto resp = client.get_orders();
        if (resp) {
            std::cout << resp->dump(2) << "\n";
        } else {
            std::cerr << "Timeout\n";
            std::exit(1);
        }
    } else if (cmd == "get_balances") {
        auto resp = client.get_balances();
        if (resp) {
            std::cout << resp->dump(2) << "\n";
        } else {
            std::cerr << "Timeout\n";
            std::exit(1);
        }
    } else if (cmd == "ping") {
        auto start = std::chrono::high_resolution_clock::now();
        auto resp = client.ping();
        auto end = std::chrono::high_resolution_clock::now();
        if (resp) {
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "{\"latency_us\": " << latency.count() << "}\n";
        } else {
            std::cerr << "Timeout\n";
            std::exit(1);
        }
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        std::exit(1);
    }
}

void print_usage(const char* prog) {
    std::cout << "LX C++ CLI Trading Client\n\n"
              << "Usage: " << prog << " [options] [command] [args...]\n\n"
              << "Options:\n"
              << "  -u, --url <url>      WebSocket server URL (default: ws://localhost:8081)\n"
              << "  -k, --key <key>      API key for authentication\n"
              << "  -s, --secret <secret> API secret for authentication\n"
              << "  -i, --interactive    Interactive mode (default if no command)\n"
              << "  -v, --verbose        Verbose output\n"
              << "  -h, --help           Show this help message\n\n"
              << "Commands:\n"
              << "  place_order <symbol> <side> <type> <price> <size>\n"
              << "  cancel_order <order_id>\n"
              << "  get_orderbook <symbol>\n"
              << "  get_positions\n"
              << "  get_orders\n"
              << "  get_balances\n"
              << "  ping\n\n"
              << "Examples:\n"
              << "  " << prog << " -i                           # Interactive mode\n"
              << "  " << prog << " place_order BTC-USD buy limit 50000 0.1\n"
              << "  " << prog << " cancel_order 12345\n"
              << "  " << prog << " get_orderbook BTC-USD\n"
              << "  " << prog << " -v ping                      # Ping with verbose output\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;

    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-u" || arg == "--url") {
            if (i + 1 >= argc) {
                std::cerr << "Missing URL argument\n";
                std::exit(1);
            }
            config.ws_url = argv[++i];
        } else if (arg == "-k" || arg == "--key") {
            if (i + 1 >= argc) {
                std::cerr << "Missing API key argument\n";
                std::exit(1);
            }
            config.api_key = argv[++i];
        } else if (arg == "-s" || arg == "--secret") {
            if (i + 1 >= argc) {
                std::cerr << "Missing API secret argument\n";
                std::exit(1);
            }
            config.api_secret = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            config.interactive = true;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg[0] != '-') {
            // Command and its arguments
            config.interactive = false;
            while (i < argc) {
                config.command_args.push_back(argv[i++]);
            }
            break;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
        ++i;
    }

    // Default to interactive if no command
    if (config.command_args.empty()) {
        config.interactive = true;
    }

    return config;
}

int main(int argc, char* argv[]) {
    Config config = parse_args(argc, argv);

    // Create and connect client
    Client client(config);
    if (!client.connect()) {
        std::cerr << "Failed to connect to " << config.ws_url << "\n";
        return 1;
    }

    // Authenticate if credentials provided
    if (!config.api_key.empty() && !config.api_secret.empty()) {
        if (!client.authenticate()) {
            std::cerr << "Authentication failed\n";
            return 1;
        }
        if (config.verbose) {
            std::cout << "Authenticated\n";
        }
    }

    // Run in interactive or command mode
    if (config.interactive) {
        run_interactive(client);
    } else {
        run_command(client, config.command_args);
    }

    return 0;
}
