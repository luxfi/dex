# LX Hummingbot Gateway Plugin

Official Hummingbot Gateway plugin for [LX](https://dex.lux.network).

## Installation

### Via npm

```bash
npm install hummingbot-gateway-lxdex-plugin
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/luxfi/dex.git
cd dex/hummingbot-gateway/plugins/lxdex
```

2. Install dependencies and build:
```bash
npm install
npm run build
```

3. Link to your Gateway installation:
```bash
npm link
cd /path/to/hummingbot-gateway
npm link hummingbot-gateway-lxdex-plugin
```

## Configuration

Copy the configuration file to your Gateway config directory:

```bash
cp config/lxdex.yml /path/to/hummingbot-gateway/conf/
```

### Configuration Options

Edit `conf/lxdex.yml` to customize:

- **Network endpoints**: API, WebSocket, gRPC URLs
- **Trading parameters**: Slippage, max hops, quote expiration
- **Gas settings**: Gas limit, price source
- **Rate limits**: Requests per second
- **WebSocket**: Auto-reconnect, ping intervals

## Usage

### Register with Gateway

In your Gateway startup file:

```typescript
import { FastifyInstance } from 'fastify';
import { lxdexPlugin } from 'hummingbot-gateway-lxdex-plugin';

async function start(fastify: FastifyInstance) {
  // Register LX plugin
  await fastify.register(lxdexPlugin);

  await fastify.listen({ port: 15888 });
}
```

### Test Endpoints

```bash
# Health check
curl http://localhost:15888/lxdex/health

# Get swap quote
curl "http://localhost:15888/lxdex/router/quote-swap?baseToken=LUX&quoteToken=USDC&amount=1000000000000000000&side=SELL"

# Get trading pairs
curl http://localhost:15888/lxdex/pairs
```

## Supported Trading Operations

| Schema | Endpoint Prefix | Operations |
|--------|-----------------|------------|
| Router | `/lxdex/router` | Quote swap, Execute swap |
| AMM | `/lxdex/amm` | Pool info, Add/Remove liquidity |
| CLMM | `/lxdex/clmm` | Open/Close positions, Collect fees |
| OrderBook | `/lxdex` | Place/Cancel orders, Order book |

## Links

- [Full Documentation](https://dex.lux.network/docs/integrations/hummingbot)
- [API Reference](https://dex.lux.network/docs/api)
- [Hummingbot Documentation](https://docs.hummingbot.org)

## License

Apache-2.0
