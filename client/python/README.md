# LX DEX Python CLI Client

Command-line trading interface for LX DEX.

## Install

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python main.py -i
```

### Single Commands

```bash
# Place a limit buy order
python main.py place_order BTC-USD buy limit 50000 0.1

# Cancel an order
python main.py cancel_order 12345

# Get orderbook
python main.py get_orderbook BTC-USD

# Get positions
python main.py get_positions

# Get open orders
python main.py get_orders
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
python main.py -u ws://testnet.lux.exchange:8081 -k mykey -s mysecret -i

# Place order with verbose output
python main.py -v place_order ETH-USD sell limit 3500 1.0

# Get positions from custom endpoint
python main.py -u ws://custom.host:8081 get_positions
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

## Using with SDK

This CLI uses the WebSocket API directly. For programmatic access, see the Python SDK at `sdk/python/`:

```python
from luxfi_dex import LXDexClient, OrderType, OrderSide

client = LXDexClient(
    json_rpc_url="http://localhost:8080",
    ws_url="ws://localhost:8081"
)

# Place order
order = client.place_order(
    symbol="BTC-USD",
    order_type=OrderType.LIMIT,
    side=OrderSide.BUY,
    price=50000.0,
    size=0.1
)
```
