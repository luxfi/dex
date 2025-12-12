# LX DEX Hummingbot Gateway Connector

Official Hummingbot Gateway connector for [LX DEX](https://dex.lux.network), the ultra-low latency decentralized exchange on Lux Network.

## Features

- **Ultra-Low Latency**: Sub-100ns matching engine latency
- **Multiple Trading Schemas**:
  - **Router**: DEX aggregation with optimal swap routing
  - **AMM**: Traditional xy=k constant product liquidity pools
  - **CLMM**: Concentrated liquidity market maker (Uniswap V3-style)
  - **CLOB**: Central limit order book with on-chain settlement
- **Real-Time Updates**: WebSocket subscriptions for order book, trades, and order status
- **Cross-Chain**: Native support for Lux C-Chain and bridged assets

## Installation

```bash
# Install dependencies
npm install @lxdex/hummingbot-gateway-connector

# Or add to your Gateway installation
cd hummingbot-gateway
npm install
```

## Quick Start

### Register the Connector

```typescript
import { FastifyInstance } from 'fastify';
import { lxdexRoutes } from '@lxdex/hummingbot-gateway-connector';

async function registerConnectors(fastify: FastifyInstance) {
  await fastify.register(lxdexRoutes);
}
```

### Use the Connector Directly

```typescript
import { LXDex } from '@lxdex/hummingbot-gateway-connector';

// Get connector instance (singleton per network)
const connector = LXDex.getInstance('mainnet');

// Get a swap quote
const quote = await connector.getQuote({
  baseToken: 'LUX',
  quoteToken: 'USDC',
  amount: '1000000000000000000', // 1 LUX in wei
  side: 'SELL',
});

console.log(`Quote: ${quote.amountOut} USDC`);
console.log(`Price Impact: ${quote.priceImpactPct}%`);
```

## API Endpoints

### Router Schema (Swaps)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/lxdex/router/quote-swap` | Get optimal swap quote |
| POST | `/lxdex/router/execute-swap` | Execute swap directly |
| POST | `/lxdex/router/execute-quote` | Execute pre-fetched quote |

### AMM Schema (Liquidity Pools)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/lxdex/amm/pool-info` | Get pool reserves and pricing |
| GET | `/lxdex/amm/position-info` | Get LP position details |
| POST | `/lxdex/amm/quote-liquidity` | Calculate liquidity provision |
| POST | `/lxdex/amm/add-liquidity` | Add liquidity to pool |
| POST | `/lxdex/amm/remove-liquidity` | Remove liquidity from pool |

### CLMM Schema (Concentrated Liquidity)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/lxdex/clmm/pool-info` | Get CLMM pool information |
| GET | `/lxdex/clmm/positions-owned` | Get user's CLMM positions |
| POST | `/lxdex/clmm/quote-position` | Calculate position amounts |
| POST | `/lxdex/clmm/open-position` | Open new position |
| POST | `/lxdex/clmm/close-position` | Close existing position |
| POST | `/lxdex/clmm/collect-fees` | Collect accumulated fees |

### Order Book Schema

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/lxdex/health` | Health check |
| POST | `/lxdex/orderbook` | Get order book snapshot |
| POST | `/lxdex/order` | Place new order |
| DELETE | `/lxdex/order/:orderId` | Cancel order |
| GET | `/lxdex/orders` | Get user orders |
| GET | `/lxdex/pairs` | Get trading pairs |
| GET | `/lxdex/token/:token` | Get token info |

## Example Usage

### Get Swap Quote

```bash
curl "http://localhost:15888/lxdex/router/quote-swap?\
baseToken=LUX&\
quoteToken=USDC&\
amount=1000000000000000000&\
side=SELL"
```

Response:
```json
{
  "quoteId": "q-abc123",
  "tokenIn": { "symbol": "LUX", "address": "0x...", "decimals": 18 },
  "tokenOut": { "symbol": "USDC", "address": "0x...", "decimals": 6 },
  "amountIn": "1000000000000000000",
  "amountOut": "12500000",
  "price": "12.50",
  "priceImpactPct": "0.05",
  "minAmountOut": "12375000",
  "route": {
    "path": ["LUX", "USDC"],
    "pools": ["0x..."],
    "expectedOutput": "12500000",
    "priceImpact": "0.05",
    "fee": "0.003"
  },
  "estimatedGas": "150000",
  "expiresAt": 1702300000000
}
```

### Execute Swap

```bash
curl -X POST http://localhost:15888/lxdex/router/execute-swap \
  -H "Content-Type: application/json" \
  -d '{
    "walletAddress": "0x...",
    "baseToken": "LUX",
    "quoteToken": "USDC",
    "amount": "1000000000000000000",
    "side": "SELL",
    "slippagePct": 0.5
  }'
```

### Add Liquidity to AMM Pool

```bash
curl -X POST http://localhost:15888/lxdex/amm/add-liquidity \
  -H "Content-Type: application/json" \
  -d '{
    "walletAddress": "0x...",
    "tokenA": "LUX",
    "tokenB": "USDC",
    "amountA": "1000000000000000000",
    "amountB": "12500000",
    "slippagePct": 0.5
  }'
```

### Open CLMM Position

```bash
curl -X POST http://localhost:15888/lxdex/clmm/open-position \
  -H "Content-Type: application/json" \
  -d '{
    "walletAddress": "0x...",
    "tokenA": "LUX",
    "tokenB": "USDC",
    "fee": 3000,
    "tickLower": -887220,
    "tickUpper": 887220,
    "amountA": "1000000000000000000",
    "amountB": "12500000"
  }'
```

### Place Limit Order

```bash
curl -X POST http://localhost:15888/lxdex/order \
  -H "Content-Type: application/json" \
  -d '{
    "walletAddress": "0x...",
    "symbol": "LUX/USDC",
    "side": "BUY",
    "type": "LIMIT",
    "price": "12.00",
    "size": "100"
  }'
```

## WebSocket Subscriptions

Connect to the WebSocket endpoint for real-time updates:

```typescript
import { LXDex } from '@lxdex/hummingbot-gateway-connector';

const connector = LXDex.getInstance('mainnet');
await connector.connectWebSocket();

// Subscribe to order book updates
connector.on('orderbook', (data) => {
  console.log('Order book update:', data);
});
await connector.subscribeOrderBook('LUX/USDC');

// Subscribe to trades
connector.on('trade', (data) => {
  console.log('Trade:', data);
});
await connector.subscribeTrades('LUX/USDC');

// Subscribe to order updates
connector.on('order', (data) => {
  console.log('Order update:', data);
});
await connector.subscribeOrders('0xYourWalletAddress');
```

## Configuration

The connector can be configured via environment variables or configuration file:

```typescript
// Environment variables
LX_DEX_API_ENDPOINT=https://api.dex.lux.network
LX_DEX_WS_ENDPOINT=wss://ws.dex.lux.network
LX_DEX_GRPC_ENDPOINT=grpc.dex.lux.network:443

// Or programmatically
import { getLXDexConfig } from '@lxdex/hummingbot-gateway-connector';

const config = getLXDexConfig('mainnet', {
  slippagePct: 0.5,
  rateLimitPerSecond: 100,
});
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiEndpoint` | string | Auto | LX DEX API endpoint |
| `wsEndpoint` | string | Auto | WebSocket endpoint |
| `grpcEndpoint` | string | Auto | gRPC endpoint |
| `slippagePct` | number | 0.5 | Default slippage tolerance |
| `defaultGasPrice` | string | Auto | Default gas price in gwei |
| `defaultGasLimit` | string | "500000" | Default gas limit |
| `rateLimitPerSecond` | number | 50 | API rate limit |
| `maxHops` | number | 3 | Maximum routing hops |

## Networks

| Network | Chain ID | API Endpoint |
|---------|----------|--------------|
| mainnet | 93 | https://api.dex.lux.network |
| testnet | 9393 | https://api.testnet.dex.lux.network |

## Error Handling

The connector provides detailed error responses:

```typescript
try {
  const result = await connector.executeSwap(request);
} catch (error) {
  if (error.code === 'INSUFFICIENT_LIQUIDITY') {
    console.log('Not enough liquidity for this trade');
  } else if (error.code === 'SLIPPAGE_EXCEEDED') {
    console.log('Price moved too much');
  } else if (error.code === 'QUOTE_EXPIRED') {
    console.log('Quote has expired, fetch a new one');
  }
}
```

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Lint
npm run lint

# Lint and fix
npm run lint:fix
```

## License

Apache-2.0

## Links

- [LX DEX Documentation](https://dex.lux.network)
- [Hummingbot Gateway](https://github.com/hummingbot/gateway)
- [Lux Network](https://lux.network)
- [GitHub Issues](https://github.com/luxfi/dex/issues)
