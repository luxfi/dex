# LX DEX SDK Documentation

## Overview

LX DEX provides multiple SDKs and APIs for integration:
- **JSON-RPC 2.0** - Standard HTTP API for web and mobile clients
- **gRPC** - High-performance binary protocol for internal Lux Network IPC
- **QZMQ** - Quantum-secure ZeroMQ for node-to-node communication
- **WebSocket** - Real-time market data streaming

## Quick Start

### 1. Start the DEX Node

```bash
# Build the node
make build

# Run the node
./bin/luxd
```

This starts:
- JSON-RPC API on port 8080
- gRPC server on port 50051  
- WebSocket server on port 8081
- P2P networking on port 5000

## API Protocols

### JSON-RPC API (Port 8080)

The JSON-RPC API is the primary interface for trading operations.

#### Endpoint
```
POST http://localhost:8080/rpc
```

#### Available Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `lx_ping` | Test connectivity | `{}` |
| `lx_getInfo` | Get node information | `{}` |
| `lx_placeOrder` | Place a new order | `{symbol, type, side, price, size, userID}` |
| `lx_cancelOrder` | Cancel an order | `{orderId}` |
| `lx_getOrder` | Get order details | `{orderId}` |
| `lx_getOrderBook` | Get order book snapshot | `{depth}` |
| `lx_getBestBid` | Get best bid price | `{}` |
| `lx_getBestAsk` | Get best ask price | `{}` |
| `lx_getTrades` | Get recent trades | `{limit}` |

#### Example: Place Order

```bash
curl -X POST http://localhost:8080/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_placeOrder",
    "params": {
      "symbol": "BTC-USD",
      "type": 0,        # 0=Limit, 1=Market, 2=Stop
      "side": 0,        # 0=Buy, 1=Sell
      "price": 50000,
      "size": 1.0,
      "userID": "trader1"
    },
    "id": 1
  }'
```

### gRPC API (Port 50051)

The gRPC API provides high-performance binary protocol for internal services.

#### Proto Definition
See `proto/lxdex.proto` for the complete service definition.

#### Key Services

- **Order Management**: PlaceOrder, CancelOrder, GetOrder
- **Market Data**: GetOrderBook, StreamOrderBook, GetTrades, StreamTrades
- **Account Management**: GetBalance, GetPositions
- **Node Management**: GetNodeInfo, GetPeers, Ping

#### Example: gRPC Client (Go)

```go
import (
    "context"
    "google.golang.org/grpc"
    pb "github.com/luxfi/dex/pkg/grpc/pb"
)

// Connect to gRPC server
conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
defer conn.Close()

client := pb.NewLXDEXServiceClient(conn)

// Place an order
resp, err := client.PlaceOrder(context.Background(), &pb.PlaceOrderRequest{
    Symbol: "BTC-USD",
    Type:   pb.OrderType_LIMIT,
    Side:   pb.OrderSide_BUY,
    Price:  50000,
    Size:   1.0,
    UserId: "trader1",
})
```

### WebSocket API (Port 8081)

Real-time market data streaming.

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:8081');
```

#### Subscribe to Market Data
```javascript
ws.send(JSON.stringify({
    type: 'subscribe',
    symbols: ['BTC-USD'],
    depth: 10
}));
```

## SDK Libraries

### TypeScript/JavaScript SDK

```bash
npm install @luxfi/dex-sdk
```

```typescript
import { LXDexClient } from '@luxfi/dex-sdk';

const client = new LXDexClient({
    jsonRpcUrl: 'http://localhost:8080/rpc',
    wsUrl: 'ws://localhost:8081'
});

// Place an order
const order = await client.placeOrder({
    symbol: 'BTC-USD',
    type: 'limit',
    side: 'buy',
    price: 50000,
    size: 1.0
});

// Subscribe to order book
client.subscribeOrderBook('BTC-USD', (book) => {
    console.log('Best bid:', book.bids[0]);
    console.log('Best ask:', book.asks[0]);
});
```

### Python SDK

```bash
pip install luxfi-dex
```

```python
from luxfi_dex import LXDexClient

client = LXDexClient(
    json_rpc_url='http://localhost:8080/rpc',
    ws_url='ws://localhost:8081'
)

# Place an order
order = client.place_order(
    symbol='BTC-USD',
    order_type='limit',
    side='buy',
    price=50000,
    size=1.0
)

# Get order book
book = client.get_order_book('BTC-USD', depth=10)
print(f"Best bid: {book['bids'][0]}")
print(f"Best ask: {book['asks'][0]}")
```

### Go SDK

```bash
go get github.com/luxfi/dex/sdk/go
```

```go
import (
    "github.com/luxfi/dex/sdk/go/lxdex"
)

client := lxdex.NewClient(
    lxdex.WithJSONRPC("http://localhost:8080/rpc"),
    lxdex.WithGRPC("localhost:50051"),
)

// Place an order
order, err := client.PlaceOrder(ctx, &lxdex.Order{
    Symbol: "BTC-USD",
    Type:   lxdex.OrderTypeLimit,
    Side:   lxdex.Buy,
    Price:  50000,
    Size:   1.0,
})

// Stream order book
stream, err := client.StreamOrderBook(ctx, "BTC-USD")
for {
    update, err := stream.Recv()
    if err != nil {
        break
    }
    fmt.Printf("Order book update: %+v\n", update)
}
```

## Order Types

| Type | Value | Description |
|------|-------|-------------|
| LIMIT | 0 | Order with specific price |
| MARKET | 1 | Immediate execution at best price |
| STOP | 2 | Trigger at stop price |
| STOP_LIMIT | 3 | Stop that becomes limit |
| ICEBERG | 4 | Hidden quantity order |
| PEG | 5 | Pegged to best bid/ask |

## Order Sides

| Side | Value | Description |
|------|-------|-------------|
| BUY | 0 | Buy order |
| SELL | 1 | Sell order |

## Time in Force

| TIF | Description |
|-----|-------------|
| GTC | Good Till Cancelled |
| IOC | Immediate Or Cancel |
| FOK | Fill Or Kill |
| DAY | Day Order |

## Error Codes

### JSON-RPC Error Codes
- `-32700` Parse Error
- `-32600` Invalid Request
- `-32601` Method Not Found
- `-32602` Invalid Params
- `-32603` Internal Error

### gRPC Status Codes
- `OK` Success
- `CANCELLED` Operation cancelled
- `INVALID_ARGUMENT` Invalid parameters
- `NOT_FOUND` Resource not found
- `ALREADY_EXISTS` Resource already exists
- `PERMISSION_DENIED` Permission denied
- `RESOURCE_EXHAUSTED` Rate limit exceeded
- `INTERNAL` Internal server error

## Rate Limits

- **Orders**: 100 per second per user
- **Cancellations**: 100 per second per user
- **Market Data**: 1000 requests per minute
- **WebSocket Connections**: 10 per IP

## Authentication

Currently, the DEX uses `userID` for identification. In production:
- JWT tokens for API authentication
- API keys for programmatic access
- Signature verification for orders

## Testing

### Test Network
```bash
# Start test node
./bin/luxd --testnet

# Use test endpoints
JSON-RPC: http://localhost:18080/rpc
gRPC: localhost:15051
WebSocket: ws://localhost:18081
```

### Example Test Script
```bash
# Run comprehensive tests
./curl-examples.sh

# Run SDK tests
npm test           # TypeScript
python -m pytest   # Python
go test ./...      # Go
```

## Performance

- **Order Matching**: 597ns latency (C++ engine)
- **Throughput**: 100M+ orders/second (with MLX GPU)
- **Block Time**: 1ms finality
- **Network**: Quantum-secure with BLS signatures

## Architecture

```
┌─────────────────────────────────────────────┐
│              Client Applications             │
├─────────────────────────────────────────────┤
│     JSON-RPC │ gRPC │ WebSocket │ QZMQ     │
├─────────────────────────────────────────────┤
│              LX DEX Core Engine              │
│  • Order Matching (Go/C++/GPU)              │
│  • Risk Management                          │
│  • State Management (BadgerDB)              │
├─────────────────────────────────────────────┤
│           Lux Consensus Layer                │
│  • K=3 Validators                           │
│  • 1ms Block Finality                       │
│  • Quantum-Resistant Signatures             │
└─────────────────────────────────────────────┘
```

## Support

- GitHub: https://github.com/luxfi/dex
- Documentation: https://docs.luxfi.com/dex
- Discord: https://discord.gg/luxfi

## License

MIT License - See LICENSE file for details