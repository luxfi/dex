# LX DEX API Documentation

## Complete API & Protocol Reference

The LX DEX supports multiple protocols for maximum compatibility and performance:

## 1. Protocol Overview

| Protocol | Port | Use Case | Latency | Throughput |
|----------|------|----------|---------|------------|
| **JSON-RPC 2.0** | 8080 | REST API, Web clients | ~10ms | 10K req/s |
| **gRPC** | 50051 | High-performance clients | ~1ms | 100K req/s |
| **WebSocket** | 8081 | Real-time streaming | ~1ms | 1M msg/s |
| **FIX Binary/QZMQ** | 4444 | Ultra-low latency trading | ~100μs | 6.8M msg/s |

## 2. JSON-RPC API

### Base URL
```
http://localhost:8080/rpc
```

### Authentication
```json
{
  "jsonrpc": "2.0",
  "method": "lx_authenticate",
  "params": {
    "apiKey": "your-api-key",
    "apiSecret": "your-secret"
  },
  "id": 1
}
```

### Order Management

#### Place Order
```json
{
  "jsonrpc": "2.0",
  "method": "lx_placeOrder",
  "params": {
    "symbol": "BTC-USD-PERP",
    "type": "limit",        // limit, market, stop, stop_limit
    "side": "buy",          // buy, sell
    "price": 50000.00,
    "size": 0.1,
    "leverage": 10,         // For perpetuals
    "timeInForce": "GTC",   // GTC, IOC, FOK, GTX
    "postOnly": false,
    "reduceOnly": false,
    "clientOrderId": "custom-123"
  },
  "id": 2
}
```

#### Cancel Order
```json
{
  "jsonrpc": "2.0",
  "method": "lx_cancelOrder",
  "params": {
    "orderId": "1234567890",
    "symbol": "BTC-USD-PERP"
  },
  "id": 3
}
```

#### Get Orders
```json
{
  "jsonrpc": "2.0",
  "method": "lx_getOrders",
  "params": {
    "symbol": "BTC-USD-PERP",
    "status": "open",  // open, filled, cancelled
    "limit": 100
  },
  "id": 4
}
```

### Perpetual Positions

#### Open Position
```json
{
  "jsonrpc": "2.0",
  "method": "lx_openPosition",
  "params": {
    "symbol": "BTC-USD-PERP",
    "side": "long",         // long, short
    "size": 1.0,
    "leverage": 25,
    "marginMode": "isolated" // isolated, cross
  },
  "id": 5
}
```

#### Close Position
```json
{
  "jsonrpc": "2.0",
  "method": "lx_closePosition",
  "params": {
    "symbol": "BTC-USD-PERP",
    "size": 0.5  // Partial close
  },
  "id": 6
}
```

#### Get Positions
```json
{
  "jsonrpc": "2.0",
  "method": "lx_getPositions",
  "params": {
    "symbol": "BTC-USD-PERP"  // Optional, omit for all
  },
  "id": 7
}
```

#### Adjust Margin
```json
{
  "jsonrpc": "2.0",
  "method": "lx_adjustMargin",
  "params": {
    "symbol": "BTC-USD-PERP",
    "amount": 1000.00,
    "type": "add"  // add, remove
  },
  "id": 8
}
```

### Market Data

#### Get Order Book
```json
{
  "jsonrpc": "2.0",
  "method": "lx_getOrderBook",
  "params": {
    "symbol": "BTC-USD-PERP",
    "depth": 20
  },
  "id": 9
}
```

#### Get Trades
```json
{
  "jsonrpc": "2.0",
  "method": "lx_getTrades",
  "params": {
    "symbol": "BTC-USD-PERP",
    "limit": 100
  },
  "id": 10
}
```

#### Get Funding Rate
```json
{
  "jsonrpc": "2.0",
  "method": "lx_getFundingRate",
  "params": {
    "symbol": "BTC-USD-PERP"
  },
  "id": 11
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "symbol": "BTC-USD-PERP",
    "fundingRate": 0.0001,
    "fundingTime": "2025-01-20T08:00:00Z",
    "nextFundingTime": "2025-01-20T16:00:00Z",
    "markPrice": 50500.00,
    "indexPrice": 50000.00,
    "premium": 0.01
  },
  "id": 11
}
```

## 3. WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8081/ws');

// Subscribe to channels
ws.send(JSON.stringify({
  "op": "subscribe",
  "channels": [
    "orderbook:BTC-USD-PERP",
    "trades:BTC-USD-PERP",
    "funding:BTC-USD-PERP",
    "liquidations:BTC-USD-PERP"
  ]
}));
```

### Real-time Streams

#### Order Book Updates
```json
{
  "channel": "orderbook:BTC-USD-PERP",
  "type": "snapshot",  // or "update"
  "data": {
    "bids": [[50000, 1.5], [49999, 2.0]],
    "asks": [[50001, 1.2], [50002, 1.8]],
    "timestamp": 1705745600000
  }
}
```

#### Trade Stream
```json
{
  "channel": "trades:BTC-USD-PERP",
  "data": {
    "id": "123456",
    "price": 50000.50,
    "size": 0.5,
    "side": "buy",
    "timestamp": 1705745600123
  }
}
```

#### Funding Updates (Every 8 Hours)
```json
{
  "channel": "funding:BTC-USD-PERP",
  "data": {
    "fundingRate": 0.0001,
    "markPrice": 50500.00,
    "indexPrice": 50000.00,
    "nextFundingTime": "2025-01-20T16:00:00Z",
    "openInterest": 1000000,
    "timestamp": 1705745600000
  }
}
```

#### Liquidation Feed
```json
{
  "channel": "liquidations:BTC-USD-PERP",
  "data": {
    "orderId": "789012",
    "side": "long",
    "price": 45000.00,
    "size": 10.0,
    "timestamp": 1705745600456
  }
}
```

## 4. gRPC API

### Proto Definition
```protobuf
syntax = "proto3";

package lx;

service TradingService {
  rpc PlaceOrder(OrderRequest) returns (OrderResponse);
  rpc CancelOrder(CancelRequest) returns (CancelResponse);
  rpc GetOrderBook(MarketRequest) returns (OrderBookResponse);
  rpc StreamOrderBook(MarketRequest) returns (stream OrderBookUpdate);
  rpc GetPositions(PositionRequest) returns (PositionsResponse);
  rpc StreamFunding(FundingRequest) returns (stream FundingUpdate);
}

message OrderRequest {
  string symbol = 1;
  OrderType type = 2;
  OrderSide side = 3;
  double price = 4;
  double size = 5;
  double leverage = 6;
  string client_order_id = 7;
}

message FundingUpdate {
  string symbol = 1;
  double funding_rate = 2;
  double mark_price = 3;
  double index_price = 4;
  int64 next_funding_time = 5;
  double open_interest = 6;
}
```

### gRPC Client Example (Go)
```go
conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewTradingServiceClient(conn)

// Place order
resp, _ := client.PlaceOrder(ctx, &pb.OrderRequest{
    Symbol: "BTC-USD-PERP",
    Type:   pb.OrderType_LIMIT,
    Side:   pb.OrderSide_BUY,
    Price:  50000.00,
    Size:   0.1,
    Leverage: 10,
})

// Stream funding updates
stream, _ := client.StreamFunding(ctx, &pb.FundingRequest{
    Symbol: "BTC-USD-PERP",
})
for {
    update, _ := stream.Recv()
    fmt.Printf("Funding: %.4f%%, Next: %v\n", 
        update.FundingRate*100, update.NextFundingTime)
}
```

## 5. FIX Binary over QZMQ

### Connection Details
- **Protocol**: FIX 4.4 Binary
- **Transport**: ZeroMQ (Quantum-resistant)
- **Port**: 4444
- **Throughput**: 6.8M messages/second
- **Latency**: ~100 microseconds

### Binary Message Format (60 bytes fixed)
```
Offset  Size  Field
0       8     Timestamp (nanoseconds)
8       8     OrderID
16      8     UserID
24      8     Price (fixed-point)
32      8     Quantity (fixed-point)
40      8     Symbol (padded)
48      2     Type (0=Limit, 1=Market, 2=Stop)
50      1     Side (0=Buy, 1=Sell)
51      1     TimeInForce
52      8     Reserved
```

### QZMQ Connection (C++)
```cpp
zmq::context_t context(1);
zmq::socket_t socket(context, ZMQ_DEALER);
socket.connect("tcp://localhost:4444");

// Send order
OrderMessage order;
order.timestamp = get_nanos();
order.price = 50000.00 * 1e7;  // Fixed-point
order.quantity = 0.1 * 1e8;
order.type = TYPE_LIMIT;
order.side = SIDE_BUY;

socket.send(&order, sizeof(order), 0);
```

## 6. Order Types & Semantics

### Supported Order Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| **Limit** | Execute at specified price or better | price, size |
| **Market** | Execute immediately at best price | size |
| **Stop** | Trigger market order at stop price | stopPrice, size |
| **Stop Limit** | Trigger limit order at stop price | stopPrice, price, size |
| **Iceberg** | Show only partial size | price, size, displaySize |
| **Post-Only** | Maker-only, cancel if would take | price, size |
| **Reduce-Only** | Only reduce position size | price, size |

### Time in Force

| TIF | Description |
|-----|-------------|
| **GTC** | Good Till Cancelled |
| **IOC** | Immediate or Cancel |
| **FOK** | Fill or Kill (all or nothing) |
| **GTX** | Good Till Crossing (Post-Only) |

## 7. Perpetual Semantics

### Funding Mechanism
- **Interval**: Every 8 hours (00:00, 08:00, 16:00 UTC)
- **Rate Calculation**: `(Mark - Index) / Index / 3`
- **Rate Cap**: ±0.1% per 8 hours
- **Payment**: Position Size × Mark Price × Funding Rate

### Position Management
- **Max Leverage**: Configurable per market (up to 100x)
- **Margin Modes**: Isolated or Cross
- **Liquidation**: When margin ratio < maintenance margin
- **Auto-Deleveraging**: In extreme market conditions

### Mark Price Calculation
```
Mark Price = Index Price + 30-second EMA(Last - Index)
```

### Liquidation Price
```
Long: Entry × (1 - 1/Leverage + MaintenanceMargin)
Short: Entry × (1 + 1/Leverage - MaintenanceMargin)
```

## 8. Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 1001 | INSUFFICIENT_BALANCE | Not enough margin/balance |
| 1002 | ORDER_NOT_FOUND | Order ID doesn't exist |
| 1003 | POSITION_NOT_FOUND | No position for symbol |
| 1004 | EXCESSIVE_LEVERAGE | Leverage exceeds maximum |
| 1005 | MARKET_CLOSED | Market not accepting orders |
| 1006 | RATE_LIMIT | Too many requests |
| 1007 | INVALID_PRICE | Price outside valid range |
| 1008 | POSITION_LIMIT | Position size exceeds limit |
| 1009 | MARGIN_INSUFFICIENT | Below maintenance margin |
| 1010 | LIQUIDATION_PENDING | Position being liquidated |

## 9. Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| Place Order | 100 | 1 second |
| Cancel Order | 200 | 1 second |
| Get Orders | 10 | 1 second |
| Market Data | 100 | 1 second |
| WebSocket Subscribe | 10 | 1 minute |

## 10. SDKs

### TypeScript/JavaScript
```bash
npm install @luxfi/dex-sdk
```

```typescript
import { LXClient } from '@luxfi/dex-sdk';

const client = new LXClient({
  apiKey: 'your-key',
  apiSecret: 'your-secret',
  testnet: false
});

// Place perpetual order
const order = await client.perpetuals.placeOrder({
  symbol: 'BTC-USD-PERP',
  side: 'buy',
  type: 'limit',
  price: 50000,
  size: 0.1,
  leverage: 10
});

// Subscribe to funding updates
client.ws.subscribe('funding:BTC-USD-PERP', (data) => {
  console.log(`Funding Rate: ${data.fundingRate}`);
});
```

### Python
```bash
pip install luxfi-dex
```

```python
from luxfi_dex import Client

client = Client(api_key='your-key', api_secret='your-secret')

# Open perpetual position
position = client.perpetuals.open_position(
    symbol='BTC-USD-PERP',
    side='long',
    size=1.0,
    leverage=25,
    margin_mode='isolated'
)

# Get funding history
funding = client.perpetuals.get_funding_history('BTC-USD-PERP')
for payment in funding:
    print(f"Rate: {payment['rate']}, Time: {payment['time']}")
```

### Go
```go
import "github.com/luxfi/dex/sdk/go/client"

c := client.New(client.Config{
    APIKey: "your-key",
    APISecret: "your-secret",
})

// Place order with funding awareness
order, _ := c.PlaceOrder(ctx, &client.OrderRequest{
    Symbol: "BTC-USD-PERP",
    Type: client.OrderTypeLimit,
    Side: client.OrderSideBuy,
    Price: 50000.00,
    Size: 0.1,
    Leverage: 10,
})

// Monitor funding
funding, _ := c.GetFundingRate("BTC-USD-PERP")
fmt.Printf("Next funding in %v at %.4f%%\n", 
    funding.NextFundingTime, funding.Rate*100)
```

## 11. Testing Endpoints

### Testnet
- JSON-RPC: `https://testnet.lx.dex/rpc`
- WebSocket: `wss://testnet.lx.dex/ws`
- gRPC: `testnet.lx.dex:50051`
- FIX/QZMQ: `testnet.lx.dex:4444`

### Health Check
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "block": 1234567,
  "orders": 45678,
  "positions": 1234,
  "markets": 50,
  "uptime": 3600
}
```

## 12. Example: Complete Trading Flow

```javascript
// 1. Connect
const client = new LXClient({ /* config */ });

// 2. Get market info
const market = await client.getMarket('BTC-USD-PERP');
console.log(`Max Leverage: ${market.maxLeverage}`);
console.log(`Funding Rate: ${market.fundingRate}`);

// 3. Open position
const position = await client.openPosition({
  symbol: 'BTC-USD-PERP',
  side: 'long',
  size: 1.0,
  leverage: 10,
  marginMode: 'isolated'
});

// 4. Monitor position
client.ws.subscribe(`position:${position.id}`, (update) => {
  console.log(`PnL: ${update.unrealizedPnL}`);
  console.log(`Liquidation Price: ${update.liquidationPrice}`);
});

// 5. Monitor funding (every 8 hours)
client.ws.subscribe('funding:BTC-USD-PERP', (funding) => {
  console.log(`Funding Payment: ${funding.payment}`);
});

// 6. Close position
await client.closePosition({
  symbol: 'BTC-USD-PERP',
  size: 0.5  // Partial close
});
```

---

## Support

- Documentation: https://docs.lx.dex
- API Status: https://status.lx.dex
- Support: support@lx.dex

---

*Last Updated: January 2025*