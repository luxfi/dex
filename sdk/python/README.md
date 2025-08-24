# LX DEX Python SDK

Official Python SDK for the LX DEX - High-performance decentralized exchange with perpetual contracts.

## Installation

```bash
pip install luxfi-dex
```

## Quick Start

```python
from luxfi_dex import LXClient

# Initialize client
client = LXClient(
    json_rpc_url="http://localhost:8080",
    websocket_url="ws://localhost:8081"
)

# Place an order
order = client.place_order(
    symbol="BTC-USD-PERP",
    side="buy",
    order_type="limit",
    price=50000,
    size=0.1,
    leverage=10
)

# Get funding rate
funding = client.get_funding_rate("BTC-USD-PERP")
print(f"Current funding rate: {funding['rate']}")
print(f"Next funding time: {funding['nextFundingTime']}")

# Subscribe to real-time data
def on_orderbook(data):
    print(f"Orderbook update: {data}")

client.subscribe("orderbook:BTC-USD-PERP", on_orderbook)
```

## Features

- Full perpetual contract support
- 8-hour funding mechanism (00:00, 08:00, 16:00 UTC)
- All order types (limit, market, stop, iceberg, etc.)
- Real-time WebSocket subscriptions
- Cross and isolated margin modes

## API Documentation

See [API Documentation](../../API_DOCUMENTATION.md) for complete reference.

## License

MIT