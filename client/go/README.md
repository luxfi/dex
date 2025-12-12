# LX DEX Go CLI Client

Command-line trading interface for LX DEX.

## Build

```bash
go build -o lx-cli .
```

## Usage

### Interactive Mode

```bash
./lx-cli -i
```

### Single Commands

```bash
# Place a limit buy order
./lx-cli place_order BTC-USD buy limit 50000 0.1

# Cancel an order
./lx-cli cancel_order 12345

# Get orderbook
./lx-cli get_orderbook BTC-USD

# Get positions
./lx-cli get_positions

# Get open orders
./lx-cli get_orders
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-url` | `ws://localhost:8081` | WebSocket server URL |
| `-key` | | API key for authentication |
| `-secret` | | API secret for authentication |
| `-i` | false | Interactive mode |
| `-v` | false | Verbose output |

### Examples

```bash
# Connect to testnet with authentication
./lx-cli -url ws://testnet.lux.exchange:8081 -key mykey -secret mysecret -i

# Place order with verbose output
./lx-cli -v place_order ETH-USD sell limit 3500 1.0

# Get positions from custom endpoint
./lx-cli -url ws://custom.host:8081 get_positions
```

## Interactive Commands

| Command | Description |
|---------|-------------|
| `place_order <symbol> <side> <type> <price> <size>` | Place a new order |
| `cancel_order <order_id>` | Cancel an existing order |
| `get_orderbook <symbol>` | Subscribe to orderbook updates |
| `get_positions` | List all positions |
| `get_orders` | List all open orders |
| `subscribe <symbol>` | Subscribe to market data |
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
