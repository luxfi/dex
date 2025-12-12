# LX Trading SDK for C++

Modern C++17/20 high-frequency trading library for LX Exchange.

## Features

- **Zero-copy where possible**: Fixed-point decimal, inline string storage
- **Lock-free orderbook**: Thread-safe reads with minimal contention
- **SIMD optimizations**: AVX2/SSE4 for financial math when available
- **Smart order routing**: Aggregate orderbooks across multiple venues
- **Execution algorithms**: TWAP, VWAP, Iceberg, Sniper, POV
- **Risk management**: Thread-safe position and PnL tracking
- **Multi-venue support**: Native LX DEX/AMM, CCXT, Hummingbot Gateway

## Quick Start

```cpp
#include <lx/trading/client.hpp>
#include <lx/trading/config.hpp>

using namespace lx::trading;

int main() {
    // Configure venues
    Config config;
    config.with_native("lx_dex",
        NativeVenueConfig::lx_dex("https://api.lx.exchange")
            .with_credentials("key", "secret"));

    // Create client
    Client client(config);
    client.connect().get();

    // Get aggregated orderbook
    auto book = client.aggregated_orderbook("BTC-USDC").get();

    // Place order with smart routing
    auto order = client.buy("BTC-USDC", Decimal::from_double(0.1)).get();

    client.disconnect().get();
}
```

## Building

Requirements:
- CMake 3.16+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `LX_TRADING_BUILD_TESTS` | ON | Build test suite |
| `LX_TRADING_BUILD_EXAMPLES` | ON | Build examples |
| `LX_TRADING_USE_SIMD` | ON | Enable SIMD optimizations |

## API Overview

### Configuration

```cpp
Config config;

// Native venues
config.with_native("lx_dex",
    NativeVenueConfig::lx_dex("https://api.lx.exchange")
        .with_credentials("key", "secret")
        .with_wallet("0x...", "private_key"));

// CCXT exchanges (via REST proxy)
config.with_ccxt("binance",
    CcxtConfig::create("binance")
        .with_credentials("key", "secret")
        .enable_sandbox());

// Hummingbot Gateway
config.with_hummingbot("uniswap",
    HummingbotConfig::create("uniswap_lux_mainnet")
        .with_wallet("0x...")
        .with_endpoint("localhost", 15888));

// Risk management
config.set_max_order_size(Decimal::from_double(10.0))
      .set_max_position_size(Decimal::from_double(100.0))
      .set_max_daily_loss(Decimal::from_double(1000.0));
```

### Decimal Arithmetic

Fixed-point decimal with 8 decimal places for exact financial math:

```cpp
Decimal a = Decimal::from_double(100.5);
Decimal b = Decimal::from_string("50.25");
Decimal c = a + b;           // 150.75
Decimal d = a * b;           // 5050.125
std::string s = c.to_string(); // "150.75"
```

### Orderbook

```cpp
// Single venue
Orderbook book("BTC-USDC", "lx_dex");
book.add_bid(Decimal::from_double(50000), Decimal::from_double(1.5));
book.add_ask(Decimal::from_double(50010), Decimal::from_double(1.0));
book.sort();

auto vwap = book.vwap_buy(Decimal::from_double(2.0)); // VWAP for buying 2 BTC

// Aggregated across venues
AggregatedOrderbook agg("BTC-USDC");
agg.add_orderbook(book1);
agg.add_orderbook(book2);

auto best = agg.best_venue_buy(Decimal::from_double(1.0)); // Find best execution venue
```

### Order Placement

```cpp
// Market orders
auto order = client.buy("BTC-USDC", Decimal::from_double(0.1)).get();
auto order = client.sell("BTC-USDC", Decimal::from_double(0.1)).get();

// Limit orders
auto order = client.limit_buy("BTC-USDC",
    Decimal::from_double(0.1),
    Decimal::from_double(50000.0)).get();

// Full control
OrderRequest req = OrderRequest::limit("BTC-USDC", Side::Buy,
    Decimal::from_double(0.1), Decimal::from_double(50000.0))
    .with_venue("lx_dex")
    .with_post_only()
    .with_client_id("my-order-123");

auto order = client.place_order(req).get();
```

### AMM Operations

```cpp
// Get quote
auto quote = client.quote("ETH", "USDC",
    Decimal::from_double(1.0), true, "lx_amm").get();

// Execute swap
auto trade = client.swap("ETH", "USDC",
    Decimal::from_double(1.0), true, 0.01, "lx_amm").get();

// Liquidity management
auto result = client.add_liquidity("ETH", "USDC",
    Decimal::from_double(1.0),
    Decimal::from_double(2000.0),
    0.01, "lx_amm").get();
```

### Execution Algorithms

```cpp
#include <lx/trading/execution.hpp>

// TWAP - spread order over time
auto twap = make_twap(client, "BTC-USDC", Side::Buy,
    Decimal::from_double(10.0),
    std::chrono::minutes(30),
    10);  // 10 slices
auto result = twap->execute().get();

// VWAP - participate in market volume
auto vwap = make_vwap(client, "BTC-USDC", Side::Buy,
    Decimal::from_double(10.0),
    Decimal::from_double(0.1),  // 10% participation
    std::chrono::hours(1));

// Iceberg - hidden large order
auto iceberg = make_iceberg(client, "BTC-USDC", Side::Buy,
    Decimal::from_double(100.0),  // total
    Decimal::from_double(5.0),    // visible
    Decimal::from_double(50000.0));

// Sniper - wait for price then execute
auto sniper = make_sniper(client, "BTC-USDC", Side::Buy,
    Decimal::from_double(1.0),
    Decimal::from_double(49000.0),  // target price
    std::chrono::seconds(60));
```

### Risk Management

```cpp
auto& risk = client.risk_manager();

// Check before trading
if (!risk.check_order_size(qty)) {
    // Order too large
}

// Manual control
risk.kill();   // Stop all trading
risk.reset();  // Resume trading

// Position tracking
risk.update_position("BTC", Decimal::from_double(1.0), Side::Buy);
Decimal pos = risk.position("BTC");
```

### Financial Math

```cpp
#include <lx/trading/math.hpp>

using namespace lx::trading::math;

// Options pricing
double call_price = black_scholes(100, 100, 1.0, 0.05, 0.2, true);
double iv = implied_volatility(10.45, 100, 100, 1.0, 0.05, true);
Greeks g = greeks(100, 100, 1.0, 0.05, 0.2, true);

// AMM math
auto [out, price] = constant_product_price(1000, 1000, 10, 0.003, true);
auto [out, new_sqrt_p, impact] = concentrated_liquidity_price(...);

// Risk metrics
double vol = volatility(returns, true, 252);
double sharpe = sharpe_ratio(returns, 0.05);
auto [dd, peak, trough] = max_drawdown(prices);
double var95 = var(returns, 0.95);
double cvar95 = cvar(returns, 0.95);
```

## Architecture

```
lx-trading-cpp/
├── include/lx/trading/
│   ├── types.hpp          # Core types (Decimal, Side, Order, etc.)
│   ├── config.hpp         # Configuration with builder pattern
│   ├── adapter.hpp        # VenueAdapter interface
│   ├── adapters/
│   │   ├── native.hpp     # LX DEX and AMM adapters
│   │   ├── ccxt.hpp       # CCXT REST proxy adapter
│   │   └── hummingbot.hpp # Hummingbot Gateway adapter
│   ├── orderbook.hpp      # Lock-free orderbook
│   ├── client.hpp         # Unified trading client
│   ├── risk.hpp           # Risk management
│   ├── execution.hpp      # Execution algorithms
│   └── math.hpp           # Financial mathematics
├── src/                   # Implementation
├── tests/                 # Catch2 tests
└── examples/              # Usage examples
```

## Thread Safety

- `Orderbook`: Thread-safe reads/writes via `std::shared_mutex`
- `RiskManager`: Thread-safe via atomic operations and locks
- `Client`: Future-based async API, safe for concurrent calls
- `Decimal`: Immutable, thread-safe by design

## Performance Considerations

1. **Fixed-point arithmetic**: `Decimal` uses int64 internally for exact math
2. **Inline storage**: `TradingPair` uses fixed arrays, no heap allocation
3. **Lock-free reads**: Orderbook allows concurrent readers
4. **SIMD**: Batch operations use AVX2 when available
5. **Zero-copy**: Futures return by value, move semantics throughout

## Dependencies

- [nlohmann/json](https://github.com/nlohmann/json) - JSON parsing
- [cpr](https://github.com/libcpr/cpr) - HTTP client
- [websocketpp](https://github.com/zaphoyd/websocketpp) - WebSocket (streaming)
- [Catch2](https://github.com/catchorg/Catch2) - Testing (optional)

All dependencies are fetched via CMake FetchContent.

## License

MIT
