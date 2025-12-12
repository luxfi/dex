# LX DEX Python Trading Client

Multi-protocol trading client for LX DEX. Supports WebSocket and gRPC protocols with async/await.

## Installation

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Programmatic Usage

The client can be imported as a module for programmatic trading:

```python
import asyncio
from lx_client import WebSocketClient, GrpcClient, create_client

async def main():
    # Using WebSocket client
    async with WebSocketClient("ws://localhost:8081") as client:
        # Place an order
        order = await client.place_order(
            symbol="BTC-USD",
            side="buy",
            order_type="limit",
            price=50000.0,
            size=0.1,
        )
        print(f"Order placed: {order.order_id}")

        # Get order book
        book = await client.get_orderbook("BTC-USD", depth=20)
        print(f"Best bid: {book.bids[0].price}")
        print(f"Best ask: {book.asks[0].price}")

        # Get positions
        positions = await client.get_positions()
        for pos in positions:
            print(f"{pos.symbol}: {pos.size} @ {pos.entry_price}")

asyncio.run(main())
```

### Using gRPC Client

```python
import asyncio
from lx_client import GrpcClient

async def main():
    async with GrpcClient("localhost", 50051) as client:
        # Authenticate
        await client.authenticate("api_key", "api_secret")

        # Place order
        order = await client.place_order(
            symbol="ETH-USD",
            side="sell",
            order_type="limit",
            price=3500.0,
            size=1.0,
        )

        # Ping server
        latency = await client.ping()
        print(f"Latency: {latency}ms")

        # Get node info
        info = await client.get_node_info()
        print(f"Node: {info['node_id']}")

asyncio.run(main())
```

### Factory Function

```python
from lx_client import create_client

# Create WebSocket client
ws_client = create_client("ws", url="ws://localhost:8081")

# Create gRPC client
grpc_client = create_client("grpc", host="localhost", port=50051)
```

### Streaming Data

```python
import asyncio
from lx_client import WebSocketClient

async def stream_orderbook():
    async with WebSocketClient("ws://localhost:8081") as client:
        async for book in client.stream_orderbook("BTC-USD"):
            print(f"Bid: {book.bids[0].price} Ask: {book.asks[0].price}")

asyncio.run(stream_orderbook())
```

## Command Line Interface

### Interactive Mode

```bash
# WebSocket (default)
python main.py -i

# gRPC
python main.py --protocol grpc -i
```

### Single Commands

```bash
# Place a limit buy order
python main.py place_order BTC-USD buy limit 50000 0.1

# Cancel an order
python main.py cancel_order 12345

# Get order book
python main.py get_orderbook BTC-USD

# Get positions
python main.py get_positions

# Get balance
python main.py get_balance USD

# Using gRPC protocol
python main.py -p grpc ping
python main.py -p grpc info
```

## CLI Options

| Flag | Description |
|------|-------------|
| `-p, --protocol` | Protocol: `ws` (default) or `grpc` |
| `-u, --url` | WebSocket URL (default: ws://localhost:8081) |
| `--host` | gRPC host (default: localhost) |
| `--port` | gRPC port (default: 50051) |
| `--tls` | Enable TLS for gRPC |
| `-k, --key` | API key |
| `-s, --secret` | API secret |
| `-i, --interactive` | Interactive mode |
| `-v, --verbose` | Verbose output |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LX_API_KEY` | API key for authentication |
| `LX_API_SECRET` | API secret for authentication |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `place_order <symbol> <side> <type> <price> <size>` | Place order |
| `cancel_order <order_id>` | Cancel order |
| `get_order <order_id>` | Get order details |
| `get_orders [symbol] [status]` | List orders |
| `get_orderbook <symbol> [depth]` | Get order book |
| `get_positions` | List positions |
| `get_balance <asset>` | Get balance |
| `auth <key> <secret>` | Authenticate |
| `ping` | Ping server (gRPC only) |
| `info` | Node info (gRPC only) |
| `help` | Show help |
| `quit` | Exit |

## Order Types

- `limit` - Limit order
- `market` - Market order
- `stop` - Stop order
- `stop_limit` - Stop-limit order

## Data Types

### Order

```python
@dataclass
class Order:
    order_id: int
    symbol: str
    order_type: OrderType
    side: OrderSide
    price: float
    size: float
    filled: float
    remaining: float
    status: OrderStatus
    user_id: str
    client_id: str
    timestamp: int
    time_in_force: TimeInForce
    post_only: bool
    reduce_only: bool
```

### OrderBook

```python
@dataclass
class OrderBook:
    symbol: str
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    timestamp: int

@dataclass
class PriceLevel:
    price: float
    size: float
    count: int
```

### Position

```python
@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    mark_price: float
    pnl: float
    margin: float
```

### Balance

```python
@dataclass
class Balance:
    asset: str
    available: float
    locked: float
    total: float
```

## Protocol Comparison

| Feature | WebSocket | gRPC |
|---------|-----------|------|
| Streaming | Yes | Yes |
| Latency | Low | Lower |
| Binary protocol | No | Yes |
| Connection overhead | Higher | Lower |
| Ping/Info | No | Yes |
| TLS support | wss:// | --tls |

## Examples

### Authenticated Trading

```bash
export LX_API_KEY="your-key"
export LX_API_SECRET="your-secret"
python main.py -i
```

### Custom Endpoint

```bash
# WebSocket
python main.py -u ws://testnet.lux.exchange:8081 -i

# gRPC
python main.py -p grpc --host testnet.lux.exchange --port 50051 --tls -i
```

### Verbose Mode

```bash
python main.py -v place_order BTC-USD buy limit 50000 0.1
```

## Project Structure

```
client/python/
  main.py              # CLI entry point
  requirements.txt     # Dependencies
  README.md            # This file
  lx_client/
    __init__.py        # Package exports
    base.py            # Abstract base class and types
    websocket_client.py # WebSocket implementation
    grpc_client.py     # gRPC implementation
```

## Related

- [Python SDK](/sdk/python/) - Full SDK with additional features
- [Go SDK](/sdk/go/) - Go client library
- [TypeScript Engine](/ts-engine/) - TypeScript implementation
