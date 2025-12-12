# LX DEX Rust CLI Client

Command-line trading interface for LX DEX.

## Build

```bash
cargo build --release
```

The binary will be at `target/release/lx-cli`.

## Usage

### Interactive Mode

```bash
./lx-cli -i
```

### Single Commands

```bash
# Place a limit buy order
./lx-cli place-order BTC-USD buy limit 50000 0.1

# Cancel an order
./lx-cli cancel-order 12345

# Get orderbook
./lx-cli get-orderbook BTC-USD

# Get positions
./lx-cli get-positions

# Get open orders
./lx-cli get-orders
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-u, --url` | `ws://localhost:8081` | WebSocket server URL |
| `-k, --key` | | API key for authentication |
| `-s, --secret` | | API secret for authentication |
| `-i, --interactive` | | Interactive mode |
| `-v, --verbose` | | Verbose output |

### Examples

```bash
# Connect to testnet with authentication
./lx-cli -u ws://testnet.lux.exchange:8081 -k mykey -s mysecret -i

# Place order with verbose output
./lx-cli -v place-order ETH-USD sell limit 3500 1.0

# Get positions from custom endpoint
./lx-cli -u ws://custom.host:8081 get-positions
```

## Interactive Commands

| Command | Description |
|---------|-------------|
| `place_order <symbol> <side> <type> <price> <size>` | Place a new order |
| `cancel_order <order_id>` | Cancel an existing order |
| `get_orderbook <symbol>` | Subscribe to orderbook updates |
| `get_positions` | List all positions |
| `get_orders` | List all open orders |
| `auth <key> <secret>` | Authenticate with API credentials |
| `help` | Show help |
| `quit` | Exit |

## Order Types

- `limit` - Limit order
- `market` - Market order
- `stop` - Stop order
- `stop_limit` - Stop-limit order

## Order Sides

- `buy` - Buy order
- `sell` - Sell order

## Dependencies

- `clap` - Command-line argument parsing
- `serde` / `serde_json` - JSON serialization
- `tungstenite` - WebSocket client
- `anyhow` - Error handling
