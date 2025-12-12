# LX TypeScript Trading Client

Programmatic trading client for LX supporting multiple protocols:
- **WebSocket**: Real-time streaming, market data subscriptions, interactive trading
- **gRPC**: High-throughput RPC calls for automated trading systems

## Installation

```bash
cd client/typescript
npm install
npm run build
```

## Usage

### As a Library

```typescript
import { LXClient, GrpcClient, createClient, ITradingClient } from '@luxfi/dex-client';

// WebSocket client for real-time streaming
const wsClient = new LXClient('ws://localhost:8081');
await wsClient.connect();
await wsClient.auth('api_key', 'api_secret');

// Place order
const response = await wsClient.placeOrder({
  symbol: 'BTC-USD',
  side: 'buy',
  type: 'limit',
  price: 50000,
  size: 0.1,
});

// Subscribe to orderbook updates
wsClient.subscribe('BTC-USD');

// gRPC client for high-throughput trading
const grpcClient = new GrpcClient('localhost:50051');
await grpcClient.connect();
await grpcClient.auth('api_key', 'api_secret');

// Get orderbook snapshot
const orderbook = await grpcClient.getOrderBook('BTC-USD', 20);

// Factory function for protocol selection
const client = createClient('grpc', 'localhost:50051', true);
await client.connect();
```

### Interactive Mode

```bash
# Default interactive mode (WebSocket)
npm start

# Interactive mode with gRPC
npm start -- -p grpc -i

# With options
npm start -- -i -v -u ws://localhost:8081

# With authentication
npm start -- -k YOUR_API_KEY -s YOUR_API_SECRET
```

### Command Mode

```bash
# WebSocket (default)
npm start -- place_order BTC-USD buy limit 50000 0.1
npm start -- cancel_order 12345
npm start -- get_orderbook BTC-USD

# gRPC
npm start -- -p grpc place_order BTC-USD buy limit 50000 0.1
npm start -- -p grpc get_orderbook BTC-USD
npm start -- -p grpc get_positions
```

## Protocol Selection

| Protocol | Flag | Default Endpoint | Best For |
|----------|------|------------------|----------|
| WebSocket | `-p ws` | `ws://localhost:8081` | Real-time streaming, subscriptions |
| gRPC | `-p grpc` | `localhost:50051` | High-throughput automated trading |

## Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --protocol <protocol>` | Protocol: ws or grpc | `ws` |
| `-u, --url <url>` | Server URL | Protocol-specific |
| `-k, --key <key>` | API key for authentication | - |
| `-s, --secret <secret>` | API secret for authentication | - |
| `-i, --interactive` | Force interactive mode | - |
| `-v, --verbose` | Enable verbose output | - |

## Interactive Commands

Once in interactive mode:

```
place_order <symbol> <side> <type> <price> <size>
  Place a new order
  Example: place_order BTC-USD buy limit 50000 0.1

cancel_order <order_id>
  Cancel an order
  Example: cancel_order 12345

get_orderbook <symbol>
  Get orderbook (gRPC: snapshot, WebSocket: subscribe)
  Example: get_orderbook BTC-USD

get_positions
  Show all open positions

get_orders
  Show all open orders

get_balances
  Show account balances

subscribe <symbol>
  Subscribe to market data (WebSocket only)
  Example: subscribe ETH-USD

unsubscribe <symbol>
  Unsubscribe from market data (WebSocket only)

auth <api_key> <api_secret>
  Authenticate with credentials

help
  Show help message

quit / exit
  Exit the client
```

## Order Types

- `limit` - Limit order at specified price
- `market` - Market order (immediate execution)
- `stop` - Stop order (triggers at stop price)
- `stop_limit` - Stop-limit order
- `iceberg` - Iceberg order (hidden quantity)
- `peg` - Pegged order

## Order Sides

- `buy` - Buy order
- `sell` - Sell order

## Time in Force

- `GTC` - Good Till Cancelled (default)
- `IOC` - Immediate Or Cancel
- `FOK` - Fill Or Kill
- `DAY` - Day Order

## Examples

### Place Orders

```bash
# Limit buy 0.1 BTC at $50,000 via WebSocket
npm start -- place_order BTC-USD buy limit 50000 0.1

# Limit sell 1 ETH at $3,000 via gRPC
npm start -- -p grpc place_order ETH-USD sell limit 3000 1

# Market buy 0.5 BTC
npm start -- place_order BTC-USD buy market 0 0.5
```

### Monitor Markets

```bash
# Interactive mode with verbose output (WebSocket)
npm start -- -v -i

# Then in the REPL:
> subscribe BTC-USD
> subscribe ETH-USD
```

### High-Throughput Trading (gRPC)

```bash
# Use gRPC for automated systems
npm start -- -p grpc -k my_api_key -s my_api_secret -i
```

## Client Interface

Both `LXClient` (WebSocket) and `GrpcClient` implement the `ITradingClient` interface:

```typescript
interface ITradingClient {
  connect(): Promise<void>;
  close(): void;
  auth(apiKey: string, apiSecret: string): Promise<void>;
  placeOrder(order: Order): Promise<OrderResponse>;
  cancelOrder(orderId: string | number): Promise<OrderResponse>;
  getOrders(): Promise<{ orders: Order[] }>;
  getPositions(): Promise<{ positions: Position[] }>;
  getBalances(): Promise<{ balances: Balance[] }>;
  getOrderBook(symbol: string, depth?: number): Promise<OrderBook>;
}
```

## Development

```bash
# Run in development mode
npm run dev

# Build
npm run build

# Clean
npm run clean
```

## Requirements

- Node.js >= 18.0.0
- npm >= 8.0.0

## License

MIT
