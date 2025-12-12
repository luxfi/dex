# LX C++ SDK

Production-ready C++17 SDK for the LX Decentralized Exchange.

## Features

- WebSocket client with automatic reconnection
- Full trading API (orders, positions, balances)
- Real-time market data subscriptions
- Thread-safe local orderbook management
- RAII-compliant resource management
- Zero external runtime dependencies (header-only deps fetched at build)

## Requirements

- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.16+
- Git (for dependency fetching)

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Build Options

```bash
cmake -DLXDEX_BUILD_EXAMPLES=ON ..   # Build examples (default: ON)
cmake -DLXDEX_BUILD_TESTS=ON ..      # Build tests (default: OFF)
```

## Quick Start

```cpp
#include <lxdex/client.hpp>
#include <iostream>

int main() {
    // Configure
    lxdex::ClientConfig config;
    config.ws_url = "ws://localhost:8081";
    config.api_key = "your_key";
    config.api_secret = "your_secret";

    // Create client
    auto client = lxdex::make_client(config);

    // Connect
    if (auto err = client->connect(); err) {
        std::cerr << "Connect failed: " << err.message << "\n";
        return 1;
    }

    // Authenticate
    if (auto err = client->authenticate(); err) {
        std::cerr << "Auth failed: " << err.message << "\n";
        return 1;
    }

    // Place order
    lxdex::Order order;
    order.symbol = "BTC-USDT";
    order.side = lxdex::Side::Buy;
    order.type = lxdex::OrderType::Limit;
    order.price = 40000.0;
    order.size = 0.01;

    auto result = client->place_order(order);
    if (result.ok()) {
        std::cout << "Order ID: " << result.value.order_id << "\n";
    }

    // Subscribe to orderbook
    client->subscribe_orderbook({"BTC-USDT"}, [](const lxdex::OrderBook& ob) {
        std::cout << "Best bid: " << ob.best_bid()
                  << " Best ask: " << ob.best_ask() << "\n";
    });

    // Keep running...
    return 0;
}
```

## API Reference

### Client Configuration

```cpp
struct ClientConfig {
    std::string ws_url = "ws://localhost:8081";
    std::string api_key;
    std::string api_secret;

    std::chrono::seconds connect_timeout{10};
    std::chrono::seconds ping_interval{30};
    std::chrono::seconds reconnect_delay{5};
    int max_reconnect_attempts = 5;
    bool auto_reconnect = true;
};
```

### Connection

```cpp
// Connect to server
Error connect();

// Disconnect
void disconnect();

// Check connection state
bool is_connected() const;
ConnectionState state() const;

// Authenticate
Error authenticate();
bool is_authenticated() const;
```

### Trading

```cpp
// Place order
Result<OrderResponse> place_order(const Order& order);
std::future<Result<OrderResponse>> place_order_async(const Order& order);

// Cancel order
Error cancel_order(uint64_t order_id);
std::future<Error> cancel_order_async(uint64_t order_id);

// Modify order
Error modify_order(uint64_t order_id, double new_price, double new_size);

// Cancel all
Result<int> cancel_all_orders(const std::string& symbol = "");
```

### Market Data

```cpp
// Get orderbook snapshot
Result<OrderBook> get_orderbook(const std::string& symbol, int32_t depth = 20);

// Get recent trades
Result<std::vector<Trade>> get_trades(const std::string& symbol, int32_t limit = 100);

// Subscribe to updates
Error subscribe_orderbook(const std::vector<std::string>& symbols, OrderBookCallback cb);
Error subscribe_trades(const std::vector<std::string>& symbols, TradeCallback cb);

// Unsubscribe
Error unsubscribe(const std::string& channel, const std::vector<std::string>& symbols);
```

### Account

```cpp
Result<std::vector<Balance>> get_balances();
Result<std::vector<Position>> get_positions();
Result<std::vector<Order>> get_orders();
Result<NodeInfo> get_info();
```

### Callbacks

```cpp
void on_error(ErrorCallback callback);
void on_order(OrderCallback callback);
void on_trade(TradeCallback callback);
void on_connection(ConnectionCallback callback);
void on_message(MessageCallback callback);  // Raw messages
```

### Local Data

```cpp
// Access local caches
OrderBookManager& orderbooks();
OrderTracker& orders();
TradeTracker& trades();
```

### Metrics

```cpp
ClientMetrics metrics() const;
void reset_metrics();
Result<int64_t> ping();  // Returns latency in microseconds
```

## Types

### Order

```cpp
struct Order {
    uint64_t order_id;
    std::string symbol;
    OrderType type;       // Limit, Market, Stop, StopLimit, Iceberg, Peg
    Side side;            // Buy, Sell
    double price;
    double size;
    double filled;
    double remaining;
    OrderStatus status;   // Open, Partial, Filled, Cancelled, Rejected
    std::string user_id;
    std::string client_id;
    int64_t timestamp;
    TimeInForce time_in_force;  // GTC, IOC, FOK, DAY
    bool post_only;
    bool reduce_only;

    bool is_open() const;
    bool is_closed() const;
    double fill_rate() const;
};
```

### Trade

```cpp
struct Trade {
    uint64_t trade_id;
    std::string symbol;
    double price;
    double size;
    Side side;
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    std::string buyer_id;
    std::string seller_id;
    int64_t timestamp;

    double total_value() const;
};
```

### OrderBook

```cpp
struct OrderBook {
    std::string symbol;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    int64_t timestamp;

    double best_bid() const;
    double best_ask() const;
    double spread() const;
    double mid_price() const;
    double spread_percentage() const;
};
```

### Error Handling

```cpp
struct Error {
    int code;
    std::string message;
    std::string request_id;

    explicit operator bool() const;  // true if error
};

template<typename T>
struct Result {
    T value;
    Error error;

    bool ok() const;
    explicit operator bool() const;  // true if ok
};
```

## Thread Safety

- All client methods are thread-safe
- Callbacks are invoked from the IO thread; avoid blocking
- Local orderbook/trade managers use internal locking

## Example: High-Frequency Trading

```cpp
#include <lxdex/client.hpp>

int main() {
    lxdex::ClientConfig config;
    config.ws_url = "ws://localhost:8081";
    config.api_key = "key";
    config.api_secret = "secret";

    auto client = lxdex::make_client(config);
    client->connect();
    client->authenticate();

    // Track local orderbook
    client->subscribe_orderbook({"BTC-USDT"}, [&](const lxdex::OrderBook& ob) {
        double mid = ob.mid_price();
        double spread = ob.spread();

        // Simple market making logic
        if (spread > 10.0) {
            // Place bid slightly below mid
            lxdex::Order bid;
            bid.symbol = "BTC-USDT";
            bid.side = lxdex::Side::Buy;
            bid.type = lxdex::OrderType::Limit;
            bid.price = mid - 2.0;
            bid.size = 0.001;
            bid.post_only = true;
            bid.client_id = lxdex::Client::generate_client_id();

            client->place_order_async(bid);

            // Place ask slightly above mid
            lxdex::Order ask = bid;
            ask.side = lxdex::Side::Sell;
            ask.price = mid + 2.0;
            ask.client_id = lxdex::Client::generate_client_id();

            client->place_order_async(ask);
        }
    });

    // Handle fills
    client->on_order([](const lxdex::Order& order) {
        if (order.status == lxdex::OrderStatus::Filled) {
            std::cout << "Filled: " << order.order_id << "\n";
        }
    });

    // Run until interrupted
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
```

## Integration with CMake Projects

```cmake
find_package(lxdex REQUIRED)
target_link_libraries(your_target PRIVATE lxdex::lxdex)
```

Or use FetchContent:

```cmake
FetchContent_Declare(
    lxdex
    GIT_REPOSITORY https://github.com/luxfi/dex.git
    SOURCE_SUBDIR sdk/cpp
)
FetchContent_MakeAvailable(lxdex)
target_link_libraries(your_target PRIVATE lxdex)
```

## License

MIT License - Copyright (c) 2025 Lux Partners Limited
