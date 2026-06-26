# @luxfi/trading - TypeScript SDK

High-frequency trading SDK with unified liquidity aggregation for TypeScript/Node.js.

## Features

- **Unified API**: Same interface for native LX DEX, CCXT exchanges, and Hummingbot Gateway
- **Smart Order Routing**: Automatic best-price execution across venues
- **Aggregated Orderbook**: Combined liquidity view from all venues
- **AMM Support**: Swap, add/remove liquidity, LP position tracking
- **Execution Algos**: TWAP, VWAP, Iceberg, Sniper, POV, DCA
- **Risk Management**: Position limits, daily loss limits, kill switch
- **Financial Math**: Black-Scholes, Greeks, VaR/CVaR, AMM pricing

## Installation

```bash
npm install @luxfi/trading
```

## Quick Start

```typescript
import Decimal from 'decimal.js';
import { Client, Config, NativeVenueConfig, CcxtConfig } from '@luxfi/trading';

async function main() {
  // Build config programmatically
  const config = new Config()
    .withNative('lx_dex', NativeVenueConfig.lxDex('https://api.dex.lux.network'))
    .withCcxt('binance', CcxtConfig.create('binance').withCredentials('key', 'secret'))
    .withSmartRouting(true);

  // Create client and connect
  const client = new Client(config);
  await client.connect();

  // Get aggregated orderbook
  const book = await client.aggregatedOrderbook('BTC-USDC');
  console.log(`Best bid: ${book.bestBid()?.price}`);
  console.log(`Best ask: ${book.bestAsk()?.price}`);

  // Smart order routing - best price across all venues
  const order = await client.buy('BTC-USDC', '0.1');
  console.log(`Filled on ${order.venue} at ${order.averagePrice}`);

  // Or target specific venue
  const order2 = await client.buy('BTC-USDC', '0.1', 'binance');

  // AMM swap
  const trade = await client.swap('LUX', 'USDC', '100', true, 0.01, 'lx_amm');

  await client.disconnect();
}

main();
```

## Configuration

### Programmatic

```typescript
import { Config, NativeVenueConfig, CcxtConfig, HummingbotConfig } from '@luxfi/trading';
import Decimal from 'decimal.js';

const config = new Config()
  // Native LX DEX
  .withNative(
    'lx_dex',
    NativeVenueConfig.lxDex('https://api.dex.lux.network')
      .withCredentials('api-key', 'api-secret')
  )
  // Native LX AMM
  .withNative(
    'lx_amm',
    NativeVenueConfig.lxAmm('https://api.dex.lux.network')
  )
  // CCXT exchange
  .withCcxt(
    'binance',
    CcxtConfig.create('binance')
      .withCredentials('key', 'secret')
  )
  // Hummingbot Gateway
  .withHummingbot(
    'gateway',
    HummingbotConfig.create('lxdex')
      .withWallet('0x...')
      .withEndpoint('localhost', 15888)
  )
  // Settings
  .withSmartRouting(true)
  .withVenuePriority(['lx_dex', 'binance'])
  .withRisk({
    enabled: true,
    maxPositionSize: new Decimal(1000),
    maxDailyLoss: new Decimal(5000),
    maxOpenOrders: 100,
    killSwitchEnabled: true,
    positionLimits: new Map([['BTC', new Decimal(10)]]),
  });
```

### From Object

```typescript
import { Config, type ConfigData } from '@luxfi/trading';

const data: ConfigData = {
  general: {
    log_level: 'info',
    smart_routing: true,
    venue_priority: ['lx_dex', 'binance'],
  },
  risk: {
    enabled: true,
    max_position_size: 1000,
    max_daily_loss: 5000,
  },
  native: {
    lx_dex: {
      venue_type: 'dex',
      api_url: 'https://api.dex.lux.network',
      api_key: 'your-api-key',
      api_secret: 'your-api-secret',
    },
    lx_amm: {
      venue_type: 'amm',
      api_url: 'https://api.dex.lux.network',
    },
  },
  ccxt: {
    binance: {
      exchange_id: 'binance',
      api_key: 'your-key',
      api_secret: 'your-secret',
    },
  },
};

const config = Config.fromObject(data);
```

## Financial Mathematics

```typescript
import {
  blackScholes,
  impliedVolatility,
  greeks,
  constantProductPrice,
  volatility,
  sharpeRatio,
  valueAtRisk,
  conditionalVaR,
} from '@luxfi/trading';

// Options pricing
const callPrice = blackScholes(100, 100, 1, 0.05, 0.2, 'call');
const iv = impliedVolatility(10.45, 100, 100, 1, 0.05, 'call');
const g = greeks(100, 100, 1, 0.05, 0.2, 'call');
console.log(`Delta: ${g.delta.toFixed(4)}, Gamma: ${g.gamma.toFixed(6)}`);

// AMM pricing
const result = constantProductPrice(1000000, 1000000, 1000, 0.003);
console.log(`Output: ${result.outputAmount.toFixed(4)}`);

// Risk metrics
const returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.005, 0.008, -0.012, 0.025];
console.log(`Volatility: ${(volatility(returns) * 100).toFixed(2)}%`);
console.log(`Sharpe: ${sharpeRatio(returns).toFixed(2)}`);
console.log(`VaR 95%: ${(valueAtRisk(returns, 0.95) * 100).toFixed(2)}%`);
console.log(`CVaR 95%: ${(conditionalVaR(returns, 0.95) * 100).toFixed(2)}%`);
```

## Execution Algorithms

```typescript
import Decimal from 'decimal.js';
import { Client, Config, TwapExecutor, VwapExecutor, IcebergExecutor, Side } from '@luxfi/trading';

const client = new Client(config);
await client.connect();

// TWAP - spread order over time
const twap = new TwapExecutor(
  client,
  'BTC-USDC',
  Side.BUY,
  new Decimal(10),
  3600,  // 1 hour
  12,    // 12 slices
);
const twapOrders = await twap.execute();

// VWAP - track market volume
const vwap = new VwapExecutor(
  client,
  'BTC-USDC',
  Side.BUY,
  new Decimal(10),
  new Decimal(0.1),  // 10% participation
  3600,
);
const vwapOrders = await vwap.execute();

// Iceberg - hide large order
const iceberg = new IcebergExecutor(
  client,
  'BTC-USDC',
  Side.BUY,
  new Decimal(100),
  new Decimal(5),
  new Decimal(50000),
);
const icebergOrders = await iceberg.execute();
```

## Risk Management

```typescript
import Decimal from 'decimal.js';
import { RiskManager, RiskError, Side, createMarketOrder } from '@luxfi/trading';

const risk = new RiskManager({
  enabled: true,
  maxPositionSize: new Decimal(100),
  maxOrderSize: new Decimal(10),
  maxDailyLoss: new Decimal(1000),
  maxOpenOrders: 50,
  killSwitchEnabled: true,
  positionLimits: new Map([['BTC', new Decimal(5)]]),
});

// Validate before placing order
const request = createMarketOrder('BTC-USDC', Side.BUY, new Decimal(1));

try {
  risk.validateOrder(request);
} catch (e) {
  if (e instanceof RiskError) {
    console.log(`Order rejected: ${e.message}`);
  }
}

// Update after trade
risk.updatePosition('BTC', new Decimal(1), Side.BUY);
risk.updatePnl(new Decimal(-50)); // Loss

// Check status
console.log(`BTC position: ${risk.position('BTC')}`);
console.log(`Daily PnL: ${risk.getDailyPnl()}`);
console.log(`Kill switch active: ${risk.isKilled}`);
```

## Supported Venues

### Native
- LX DEX (OrderBook)
- LX AMM

### CCXT (100+ exchanges)
- Binance
- MEXC
- OKX
- Bybit
- KuCoin
- [All CCXT exchanges](https://github.com/ccxt/ccxt)

### Hummingbot Gateway
- Any Gateway-supported DEX

## API Reference

### Client

```typescript
class Client {
  // Connection
  connect(): Promise<void>
  disconnect(): Promise<void>
  venue(name: string): VenueAdapter | undefined
  listVenues(): VenueInfo[]

  // Market Data
  orderbook(symbol: string, venue?: string): Promise<Orderbook>
  aggregatedOrderbook(symbol: string): Promise<AggregatedOrderbook>
  ticker(symbol: string, venue?: string): Promise<Ticker>
  tickers(symbol: string): Promise<Ticker[]>

  // Account
  balances(): Promise<AggregatedBalance[]>
  balance(asset: string, venue?: string): Promise<Balance>

  // Orders
  buy(symbol: string, quantity: Decimal | number | string, venue?: string): Promise<Order>
  sell(symbol: string, quantity: Decimal | number | string, venue?: string): Promise<Order>
  limitBuy(symbol: string, quantity, price, venue?: string): Promise<Order>
  limitSell(symbol: string, quantity, price, venue?: string): Promise<Order>
  placeOrder(request: OrderRequest): Promise<Order>
  cancelOrder(orderId: string, symbol: string, venue: string): Promise<Order>
  cancelAllOrders(symbol?: string, venue?: string): Promise<Order[]>
  openOrders(symbol?: string): Promise<Order[]>

  // AMM
  quote(baseToken, quoteToken, amount, isBuy, venue): Promise<SwapQuote>
  swap(baseToken, quoteToken, amount, isBuy, slippage, venue): Promise<Trade>
  poolInfo(baseToken, quoteToken, venue): Promise<PoolInfo>
  addLiquidity(baseToken, quoteToken, baseAmount, quoteAmount, slippage, venue): Promise<LiquidityResult>
  removeLiquidity(poolAddress, liquidityAmount, slippage, venue): Promise<LiquidityResult>
  lpPositions(venue: string): Promise<LpPosition[]>
}
```

## Links

- [LX DEX](https://dex.lux.network)
- [Documentation](https://dex.lux.network/docs/sdk/typescript)
- [GitHub](https://github.com/luxfi/dex)
- [Python SDK](https://github.com/luxfi/dex/tree/main/sdk/lx-trading-py)
