# LX C CLI Trading Client

Ultra-fast command-line trading client for LX, designed for HFT traders and embedded systems.

## Features

- Direct WebSocket connection to LX API
- Sub-millisecond latency (limited only by network)
- Interactive REPL mode for manual trading
- Single-command mode for scripting/automation
- Minimal dependencies (libwebsockets only)
- Zero heap allocations in hot path
- POSIX-compliant, runs on Linux/macOS/BSD

## Build

### Dependencies

**macOS:**
```bash
brew install libwebsockets
```

**Ubuntu/Debian:**
```bash
apt install libwebsockets-dev
```

**Fedora/RHEL:**
```bash
dnf install libwebsockets-devel
```

### Compile

```bash
make          # Optimized build
make debug    # Debug build with symbols
make static   # Static binary for portability
```

## Usage

### Interactive Mode

```bash
./build/lx-cli -i
```

Starts a REPL where you can enter commands interactively:

```
LX CLI v1.0.0 - Type 'help' for commands
> auth mykey mysecret
Authenticated successfully
> place_order BTC-USD buy limit 50000 0.1
Order placed: ID=12345 BTC-USD 0.1000 @ 50000.00 [open]
> get_orders
{"type":"orders","data":[...]}
> quit
Goodbye
```

### Single Command Mode

Execute a single command and exit:

```bash
# Place an order
./build/lx-cli -k KEY -s SECRET place_order BTC-USD buy limit 50000 0.1

# Cancel an order
./build/lx-cli -k KEY -s SECRET cancel_order 12345

# Get open orders
./build/lx-cli -k KEY -s SECRET get_orders

# Get positions
./build/lx-cli -k KEY -s SECRET get_positions

# Get balances
./build/lx-cli -k KEY -s SECRET get_balances

# Measure latency
./build/lx-cli ping
```

### Options

| Option | Description |
|--------|-------------|
| `-u, --url URL` | WebSocket URL (default: `ws://localhost:8081`) |
| `-k, --key KEY` | API key for authentication |
| `-s, --secret SECRET` | API secret for authentication |
| `-i, --interactive` | Run in interactive REPL mode |
| `-v, --verbose` | Enable verbose output (show raw messages) |
| `-t, --timeout MS` | Request timeout in milliseconds (default: 5000) |
| `-h, --help` | Show help message |
| `-V, --version` | Show version |

## Commands

### Trading

**place_order** `<symbol> <side> <type> <price> <size>`

Place a new order.

```bash
place_order BTC-USD buy limit 50000 0.1
place_order ETH-USD sell market 0 1.5
place_order BTC-USD buy stop 48000 0.5
```

- `symbol`: Trading pair (e.g., BTC-USD, ETH-USD)
- `side`: `buy` or `sell`
- `type`: `limit`, `market`, or `stop`
- `price`: Order price (use 0 for market orders)
- `size`: Order size

**cancel_order** `<order_id>`

Cancel an existing order.

```bash
cancel_order 12345
```

**get_orders**

List all open orders.

### Portfolio

**get_positions**

Show all positions.

**get_balances**

Show account balances.

### Market Data

**subscribe** `<symbol>`

Subscribe to orderbook updates for a symbol.

```bash
subscribe BTC-USD
```

### Connection

**auth** `<api_key> <api_secret>`

Authenticate with credentials (can also use `-k` and `-s` flags).

**ping**

Test connection and measure latency.

**status**

Show connection status.

## Scripting Examples

### Bash Script

```bash
#!/bin/bash
LX="./build/lx-cli -k $API_KEY -s $API_SECRET"

# Place multiple orders
$LX place_order BTC-USD buy limit 49000 0.1
$LX place_order BTC-USD buy limit 48000 0.1
$LX place_order BTC-USD buy limit 47000 0.1

# Check orders
$LX get_orders
```

### Environment Variables

```bash
export LX_URL="wss://api.lx.exchange/ws"
export LX_KEY="your-api-key"
export LX_SECRET="your-api-secret"

./build/lx-cli -u "$LX_URL" -k "$LX_KEY" -s "$LX_SECRET" get_orders
```

## Performance

The C client is optimized for minimal latency:

- No JSON library overhead (inline parsing)
- Lock-free message queuing
- Zero-copy receive path
- Direct WebSocket frames
- Minimal system calls

Typical round-trip latency: **< 1ms** (on localhost)

## Comparison

| Client | Language | Binary Size | Startup Time | Latency |
|--------|----------|-------------|--------------|---------|
| lx-cli | C | ~50KB | < 5ms | < 1ms |
| lx-cli-go | Go | ~8MB | ~50ms | ~1ms |
| lx-cli-ts | TypeScript | N/A (Node) | ~500ms | ~5ms |

## Building for Embedded Systems

For embedded Linux systems, use static linking:

```bash
make static
```

Cross-compilation:

```bash
CC=arm-linux-gnueabihf-gcc make
```

## Protocol

The client communicates with LX using WebSocket JSON messages.

### Request Format

```json
{
  "type": "place_order",
  "order": {
    "symbol": "BTC-USD",
    "side": "buy",
    "type": "limit",
    "price": 50000,
    "size": 0.1
  },
  "request_id": "req_1"
}
```

### Response Format

```json
{
  "type": "order_update",
  "data": {
    "orderId": 12345,
    "symbol": "BTC-USD",
    "status": "open"
  },
  "request_id": "req_1"
}
```

## License

Copyright (c) 2025 Lux Partners Limited

## See Also

- [LX SDK (C)](/sdk/c/) - Library for embedding
- [LX CLI (Go)](/client/go/) - Go CLI client
- [WebSocket API Documentation](/docs/api/websocket.md)
