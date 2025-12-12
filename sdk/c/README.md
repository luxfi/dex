# LX C SDK

High-performance C client for LX trading.

## Features

- WebSocket client using libwebsockets
- Async event-driven architecture
- Full orderbook management
- Order placement, modification, cancellation
- Market data subscriptions
- Thread-safe with minimal locking
- C11 with no external dependencies beyond libwebsockets

## Requirements

- CMake 3.10+
- C11 compiler (GCC, Clang)
- libwebsockets 4.0+
- pthread

### macOS

```bash
brew install libwebsockets cmake
```

### Linux (Debian/Ubuntu)

```bash
apt install libwebsockets-dev cmake build-essential
```

## Build

```bash
cd sdk/c
mkdir build && cd build
cmake ..
make
```

### Options

```bash
cmake -DLXDEX_BUILD_EXAMPLES=ON ..   # Build examples (default: ON)
cmake -DLXDEX_BUILD_TESTS=ON ..      # Build tests (default: OFF)
```

## Usage

### Basic Connection

```c
#include "lxdex.h"

int main(void) {
    // Initialize library
    lxdex_init();

    // Configure client
    lxdex_config_t config = {
        .ws_url = "ws://localhost:8081",
        .api_key = "your-api-key",
        .api_secret = "your-api-secret",
    };

    // Create client
    lxdex_client_t *client = lxdex_client_new(&config);
    if (!client) return 1;

    // Set callbacks
    lxdex_callbacks_t callbacks = {
        .on_connect = my_on_connect,
        .on_error = my_on_error,
        .on_orderbook = my_on_orderbook,
    };
    lxdex_client_set_callbacks(client, &callbacks);

    // Connect
    lxdex_client_connect(client);

    // Event loop
    while (running) {
        lxdex_client_service(client, 100);
    }

    // Cleanup
    lxdex_client_free(client);
    lxdex_cleanup();
    return 0;
}
```

### Placing Orders

```c
// Limit order
lxdex_order_t order;
lxdex_order_limit(&order, "BTC-USD", LXDEX_SIDE_BUY, 50000.0, 0.1);
order.post_only = true;

uint64_t order_id;
lxdex_error_t err = lxdex_place_order(client, &order, &order_id);
if (err != LXDEX_OK) {
    fprintf(stderr, "Order failed: %s\n", lxdex_strerror(err));
}

// Market order
lxdex_order_t market;
lxdex_order_market(&market, "BTC-USD", LXDEX_SIDE_SELL, 0.05);
lxdex_place_order(client, &market, NULL);

// Cancel
lxdex_cancel_order(client, order_id);
```

### Market Data Subscriptions

```c
// Subscribe to orderbook
lxdex_subscribe_orderbook(client, "BTC-USD");

// Subscribe to trades
lxdex_subscribe_trades(client, "BTC-USD");

// Handle in callback
void on_orderbook(lxdex_client_t *client, const lxdex_orderbook_t *book, void *user_data) {
    printf("Best bid: %.2f, Best ask: %.2f\n",
        lxdex_orderbook_best_bid(book),
        lxdex_orderbook_best_ask(book));
}
```

### Local Orderbook Management

```c
// Initialize local orderbook
lxdex_orderbook_t book;
lxdex_orderbook_init(&book, 100);

// Update from feed
lxdex_orderbook_update_bid(&book, 50000.0, 1.5, 3);
lxdex_orderbook_update_ask(&book, 50001.0, 0.8, 2);

// Query
double spread = lxdex_orderbook_spread(&book);
double mid = lxdex_orderbook_mid(&book);

// Calculate market impact
double vwap = lxdex_orderbook_price_for_size(&book, LXDEX_SIDE_BUY, 10.0);

// Cleanup
lxdex_orderbook_free(&book);
```

## API Reference

### Client Lifecycle

| Function | Description |
|----------|-------------|
| `lxdex_init()` | Initialize library (call once) |
| `lxdex_cleanup()` | Cleanup library (call once) |
| `lxdex_client_new(config)` | Create client |
| `lxdex_client_set_callbacks(client, cb)` | Set event callbacks |
| `lxdex_client_connect(client)` | Connect to DEX |
| `lxdex_client_auth(client)` | Authenticate |
| `lxdex_client_disconnect(client)` | Disconnect |
| `lxdex_client_free(client)` | Free client |
| `lxdex_client_state(client)` | Get connection state |
| `lxdex_client_service(client, timeout_ms)` | Process events |

### Orders

| Function | Description |
|----------|-------------|
| `lxdex_place_order(client, order, &id)` | Place order |
| `lxdex_cancel_order(client, id)` | Cancel order |
| `lxdex_modify_order(client, id, price, size)` | Modify order |
| `lxdex_order_init(order)` | Initialize order struct |
| `lxdex_order_limit(order, sym, side, price, size)` | Create limit order |
| `lxdex_order_market(order, sym, side, size)` | Create market order |

### Market Data

| Function | Description |
|----------|-------------|
| `lxdex_subscribe_orderbook(client, symbol)` | Subscribe to orderbook |
| `lxdex_subscribe_trades(client, symbol)` | Subscribe to trades |
| `lxdex_unsubscribe(client, channel)` | Unsubscribe |

### Orderbook Utilities

| Function | Description |
|----------|-------------|
| `lxdex_orderbook_init(book, capacity)` | Initialize orderbook |
| `lxdex_orderbook_free(book)` | Free orderbook |
| `lxdex_orderbook_best_bid(book)` | Get best bid |
| `lxdex_orderbook_best_ask(book)` | Get best ask |
| `lxdex_orderbook_spread(book)` | Get spread |
| `lxdex_orderbook_mid(book)` | Get mid price |

### Error Handling

| Function | Description |
|----------|-------------|
| `lxdex_strerror(err)` | Error code to string |
| `lxdex_last_error()` | Get last error message |

## Error Codes

| Code | Description |
|------|-------------|
| `LXDEX_OK` | Success |
| `LXDEX_ERR_INVALID_ARG` | Invalid argument |
| `LXDEX_ERR_NO_MEMORY` | Out of memory |
| `LXDEX_ERR_CONNECTION` | Connection error |
| `LXDEX_ERR_TIMEOUT` | Operation timed out |
| `LXDEX_ERR_AUTH` | Authentication failed |
| `LXDEX_ERR_PARSE` | JSON parse error |
| `LXDEX_ERR_PROTOCOL` | Protocol error |
| `LXDEX_ERR_RATE_LIMIT` | Rate limit exceeded |
| `LXDEX_ERR_ORDER_REJECTED` | Order rejected |
| `LXDEX_ERR_NOT_CONNECTED` | Not connected |

## Connection States

| State | Description |
|-------|-------------|
| `LXDEX_STATE_DISCONNECTED` | Not connected |
| `LXDEX_STATE_CONNECTING` | Connection in progress |
| `LXDEX_STATE_CONNECTED` | Connected, not authenticated |
| `LXDEX_STATE_AUTHENTICATED` | Connected and authenticated |
| `LXDEX_STATE_ERROR` | Error state |

## Thread Safety

- The client is thread-safe for sending messages from multiple threads
- Callbacks are invoked from the service thread
- Use external synchronization if callbacks access shared data

## Performance

- Event-driven, non-blocking I/O
- Lock-free message queue for sends
- Minimal allocations in hot path
- Binary search for orderbook operations

## Example

Run the basic example:

```bash
cd build
./basic -u ws://localhost:8081 -m BTC-USD
```

With authentication and test order:

```bash
./basic -k YOUR_API_KEY -s YOUR_API_SECRET -o
```

## License

Copyright (c) 2025 Lux Partners Limited. All rights reserved.
