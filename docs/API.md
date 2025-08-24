# LX DEX API Documentation

## Table of Contents
- [WebSocket API](#websocket-api)
- [REST API](#rest-api)
- [gRPC API](#grpc-api)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)

## WebSocket API

### Connection
```
ws://localhost:8080/ws
wss://api.lux.network/ws (production)
```

### Message Format
All WebSocket messages use JSON format:
```json
{
  "type": "message_type",
  "data": {},
  "timestamp": 1234567890
}
```

### Subscription Messages

#### Subscribe to Order Book
```json
{
  "type": "subscribe",
  "channel": "orderbook",
  "symbol": "BTC-USDT",
  "depth": 20
}
```

Response:
```json
{
  "type": "orderbook_snapshot",
  "symbol": "BTC-USDT",
  "bids": [[50000, 1.5], [49999, 2.0]],
  "asks": [[50001, 1.2], [50002, 0.8]],
  "timestamp": 1234567890
}
```

#### Subscribe to Trades
```json
{
  "type": "subscribe",
  "channel": "trades",
  "symbol": "BTC-USDT"
}
```

Response:
```json
{
  "type": "trade",
  "symbol": "BTC-USDT",
  "price": 50000,
  "size": 0.5,
  "side": "buy",
  "timestamp": 1234567890
}
```

### Trading Messages

#### Place Order
```json
{
  "type": "place_order",
  "symbol": "BTC-USDT",
  "side": "buy",
  "order_type": "limit",
  "price": 50000,
  "size": 0.1,
  "time_in_force": "GTC",
  "post_only": false
}
```

Response:
```json
{
  "type": "order_placed",
  "order_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "open",
  "filled_size": 0,
  "remaining_size": 0.1,
  "timestamp": 1234567890
}
```

#### Cancel Order
```json
{
  "type": "cancel_order",
  "order_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

## REST API

### Base URL
```
http://localhost:8080/api/v1
https://api.lux.network/v1 (production)
```

### Endpoints

#### GET /orderbook/{symbol}
Get order book for a symbol.

**Parameters:**
- `symbol` (path): Trading pair symbol (e.g., BTC-USDT)
- `depth` (query): Order book depth (default: 20, max: 100)

**Response:**
```json
{
  "symbol": "BTC-USDT",
  "bids": [
    {"price": 50000, "size": 1.5},
    {"price": 49999, "size": 2.0}
  ],
  "asks": [
    {"price": 50001, "size": 1.2},
    {"price": 50002, "size": 0.8}
  ],
  "timestamp": 1234567890
}
```

#### POST /orders
Place a new order.

**Request Body:**
```json
{
  "symbol": "BTC-USDT",
  "side": "buy",
  "type": "limit",
  "price": 50000,
  "size": 0.1,
  "time_in_force": "GTC"
}
```

**Response:**
```json
{
  "order_id": "123e4567-e89b-12d3-a456-426614174000",
  "symbol": "BTC-USDT",
  "side": "buy",
  "type": "limit",
  "price": 50000,
  "size": 0.1,
  "status": "open",
  "created_at": 1234567890
}
```

#### DELETE /orders/{order_id}
Cancel an order.

**Parameters:**
- `order_id` (path): Order ID to cancel

**Response:**
```json
{
  "order_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "cancelled",
  "cancelled_at": 1234567890
}
```

#### GET /trades
Get recent trades.

**Parameters:**
- `symbol` (query): Trading pair symbol
- `limit` (query): Number of trades (default: 100, max: 1000)

**Response:**
```json
{
  "trades": [
    {
      "trade_id": "abc123",
      "symbol": "BTC-USDT",
      "price": 50000,
      "size": 0.5,
      "side": "buy",
      "timestamp": 1234567890
    }
  ]
}
```

## gRPC API

### Proto Definition
```protobuf
syntax = "proto3";
package lxdex;

service TradingService {
  rpc PlaceOrder(OrderRequest) returns (OrderResponse);
  rpc CancelOrder(CancelRequest) returns (CancelResponse);
  rpc GetOrderBook(OrderBookRequest) returns (OrderBookResponse);
  rpc StreamTrades(StreamRequest) returns (stream Trade);
}

message OrderRequest {
  string symbol = 1;
  string side = 2;
  string type = 3;
  double price = 4;
  double size = 5;
}

message OrderResponse {
  string order_id = 1;
  string status = 2;
  int64 timestamp = 3;
}
```

### Connection
```go
conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewTradingServiceClient(conn)
```

## Authentication

### API Key Authentication
Include API key in headers:
```
X-API-Key: your_api_key_here
X-API-Secret: your_api_secret_here
```

### JWT Authentication
After login, include JWT token:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Rate Limiting

### Limits
- **Public endpoints**: 100 requests per minute
- **Private endpoints**: 300 requests per minute
- **WebSocket connections**: 5 concurrent connections
- **Order placement**: 10 orders per second

### Headers
Rate limit information in response headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1234567890
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance for order",
    "details": {
      "required": 100.5,
      "available": 50.0
    }
  }
}
```

### Common Error Codes
| Code | Description |
|------|-------------|
| `INVALID_SYMBOL` | Trading pair not supported |
| `INVALID_ORDER_TYPE` | Order type not recognized |
| `INSUFFICIENT_BALANCE` | Not enough balance |
| `ORDER_NOT_FOUND` | Order ID does not exist |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `UNAUTHORIZED` | Authentication required |
| `FORBIDDEN` | Access denied |
| `INTERNAL_ERROR` | Server error |

## WebSocket Error Handling
```json
{
  "type": "error",
  "error": {
    "code": "SUBSCRIPTION_FAILED",
    "message": "Failed to subscribe to channel",
    "channel": "orderbook",
    "symbol": "INVALID-PAIR"
  }
}
```

## SDKs

### Go SDK
```go
import "github.com/luxfi/dex/sdk/go"

client := lxdex.NewClient("api_key", "api_secret")
order, err := client.PlaceOrder("BTC-USDT", "buy", "limit", 50000, 0.1)
```

### Python SDK
```python
from luxfi_dex import Client

client = Client(api_key="your_key", api_secret="your_secret")
order = client.place_order("BTC-USDT", "buy", "limit", 50000, 0.1)
```

### TypeScript SDK
```typescript
import { LXDexClient } from '@luxfi/dex-sdk';

const client = new LXDexClient({
  apiKey: 'your_key',
  apiSecret: 'your_secret'
});

const order = await client.placeOrder({
  symbol: 'BTC-USDT',
  side: 'buy',
  type: 'limit',
  price: 50000,
  size: 0.1
});
```