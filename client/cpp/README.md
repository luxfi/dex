# LX C++ CLI Trading Client

Command-line trading interface for LX WebSocket API.

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Interactive mode (default)
./lx-cli

# Interactive mode with authentication
./lx-cli -k YOUR_API_KEY -s YOUR_API_SECRET

# Single command mode
./lx-cli place_order BTC-USD buy limit 50000 0.1
./lx-cli cancel_order 12345
./lx-cli get_orderbook BTC-USD
./lx-cli get_balances
./lx-cli ping

# Custom server
./lx-cli -u ws://trading.example.com:8081 get_positions

# Verbose output
./lx-cli -v ping
```

## Options

| Option | Description |
|--------|-------------|
| `-u, --url <url>` | WebSocket server URL (default: ws://localhost:8081) |
| `-k, --key <key>` | API key for authentication |
| `-s, --secret <secret>` | API secret for authentication |
| `-i, --interactive` | Interactive REPL mode |
| `-v, --verbose` | Show raw JSON messages |
| `-h, --help` | Show help |

## Commands

### Trading

```
place_order <symbol> <side> <type> <price> <size>
  Place a new order
  Example: place_order BTC-USD buy limit 50000 0.1

cancel_order <order_id>
  Cancel an existing order
  Example: cancel_order 12345
```

### Market Data

```
get_orderbook <symbol>
  Get orderbook snapshot
  Example: get_orderbook BTC-USD

subscribe <symbol>
  Subscribe to orderbook updates
  Example: subscribe ETH-USD
```

### Account

```
get_positions
  Show all open positions

get_orders
  Show all open orders

get_balances
  Show account balances
```

### Utility

```
ping
  Test connection latency

help
  Show available commands

quit / exit
  Exit the CLI
```

## Interactive Example

```
$ ./lx-cli -k mykey -s mysecret
LX CLI - Type 'help' for commands
> ping
Pong: 1234 us
> place_order BTC-USD buy limit 50000 0.1
{
  "type": "order_placed",
  "data": {
    "orderId": 12345,
    "status": "open"
  }
}
> get_orders
{
  "type": "orders",
  "data": {
    "orders": [...]
  }
}
> quit
Goodbye
```

## Dependencies

Fetched automatically via CMake FetchContent:
- [nlohmann/json](https://github.com/nlohmann/json) - JSON parsing
- [websocketpp](https://github.com/zaphoyd/websocketpp) - WebSocket client
- [asio](https://github.com/chriskohlhoff/asio) - Async I/O (standalone, no Boost)

## Performance

Optimized for HFT:
- Native compilation with `-march=native`
- Lock-free message queue
- Microsecond-precision latency measurement
- Minimal allocations in hot path

## License

MIT License - Copyright (c) 2025 Lux Partners Limited
