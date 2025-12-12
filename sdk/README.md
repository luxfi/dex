# LX Trading SDK

High-frequency trading SDK with unified liquidity aggregation across multiple venues.

## Overview

LX Trading SDK provides a unified interface for trading across:
- **LX DEX** - Native central limit order book
- **LX AMM** - Automated market maker pools
- **CCXT** - 100+ centralized exchanges
- **Hummingbot Gateway** - DEX aggregation

## SDKs

| Language | Package | Directory | Throughput |
|----------|---------|-----------|------------|
| Rust | `lx-trading` | `lx-trading-core/` | 5M+ msgs/sec |
| C++ | `lx-trading` | `lx-trading-cpp/` | 10M+ msgs/sec |
| Go | `github.com/luxfi/trading` | `lx-trading-go/` | 1M+ msgs/sec |
| TypeScript | `@luxfi/trading` | `lx-trading-ts/` | 100K+ msgs/sec |
| Python | `lx-trading` | `lx-trading-py/` | 50K+ msgs/sec |

## Features

### Unified API
Same interface across all venues - switch between DEX, CEX, AMM without code changes.

### Smart Order Routing (SOR)
Automatic best-price execution across all connected venues.

### Aggregated Orderbook
Combined liquidity view from multiple sources with real-time updates.

### Execution Algorithms
- **TWAP** - Time-weighted average price
- **VWAP** - Volume-weighted average price
- **Iceberg** - Hidden size orders
- **Sniper** - Target price execution

### Risk Management
- Position limits per asset
- Max order size enforcement
- Daily loss limits with kill switch
- Open orders tracking

### Financial Mathematics
- Black-Scholes options pricing
- Greeks (delta, gamma, theta, vega, rho)
- Implied volatility calculation
- AMM pricing (constant product, concentrated liquidity)
- Risk metrics (VaR, CVaR, Sharpe, Sortino)

## Quick Start

### Python

```python
import asyncio
from decimal import Decimal
from lx_trading import Client, Config, NativeVenueConfig, CcxtConfig

async def main():
    config = Config()
    config.with_native("lx_dex", NativeVenueConfig.lx_dex("https://api.dex.lux.network"))
    config.with_ccxt("binance", CcxtConfig.new("binance").with_credentials("key", "secret"))

    client = Client(config)
    await client.connect()

    # Smart routing - best price across all venues
    order = await client.buy("BTC-USDC", Decimal("0.1"))
    print(f"Filled on {order.venue} at {order.average_price}")

    await client.disconnect()

asyncio.run(main())
```

### Go

```go
package main

import (
    "fmt"
    "github.com/luxfi/trading"
    "github.com/shopspring/decimal"
)

func main() {
    config := trading.NewConfig().
        WithNative("lx_dex", trading.NewNativeConfig("https://api.dex.lux.network")).
        WithCCXT("binance", trading.NewCCXTConfig("binance").WithCredentials("key", "secret"))

    client := trading.NewClient(config)
    if err := client.Connect(); err != nil {
        panic(err)
    }
    defer client.Disconnect()

    order, err := client.Buy("BTC-USDC", decimal.NewFromFloat(0.1))
    if err != nil {
        panic(err)
    }
    fmt.Printf("Filled on %s at %s\n", order.Venue, order.AveragePrice)
}
```

### TypeScript

```typescript
import { Client, Config, NativeVenueConfig, CcxtConfig } from '@luxfi/trading';
import Decimal from 'decimal.js';

const config = new Config()
    .withNative('lx_dex', NativeVenueConfig.lxDex('https://api.dex.lux.network'))
    .withCcxt('binance', new CcxtConfig('binance').withCredentials('key', 'secret'));

const client = new Client(config);
await client.connect();

const order = await client.buy('BTC-USDC', new Decimal('0.1'));
console.log(`Filled on ${order.venue} at ${order.averagePrice}`);

await client.disconnect();
```

### C++

```cpp
#include <lx/trading/client.hpp>
#include <lx/trading/config.hpp>

int main() {
    auto config = lx::trading::Config()
        .with_native("lx_dex", lx::trading::NativeConfig("https://api.dex.lux.network"))
        .with_ccxt("binance", lx::trading::CcxtConfig("binance").with_credentials("key", "secret"));

    lx::trading::Client client(config);
    client.connect();

    auto order = client.buy("BTC-USDC", "0.1");
    std::cout << "Filled on " << order.venue << " at " << order.average_price << std::endl;

    client.disconnect();
    return 0;
}
```

### Rust

```rust
use lx_trading::{Client, Config, NativeVenueConfig, CcxtConfig};
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::new()
        .with_native("lx_dex", NativeVenueConfig::lx_dex("https://api.dex.lux.network"))
        .with_ccxt("binance", CcxtConfig::new("binance").with_credentials("key", "secret"));

    let client = Client::new(config);
    client.connect().await?;

    let order = client.buy("BTC-USDC", dec!(0.1)).await?;
    println!("Filled on {} at {:?}", order.venue, order.average_price);

    client.disconnect().await?;
    Ok(())
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LX Trading Client                        │
├─────────────────────────────────────────────────────────────┤
│  Smart Order Router  │  Risk Manager  │  Execution Algos   │
├─────────────────────────────────────────────────────────────┤
│                   Aggregated Orderbook                      │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│  LX DEX     │   LX AMM    │    CCXT      │   Hummingbot    │
│  Adapter    │   Adapter   │   Adapter    │    Adapter      │
├─────────────┼─────────────┼──────────────┼─────────────────┤
│ Native API  │ Native API  │ 100+ CEXs    │  Gateway DEXs   │
└─────────────┴─────────────┴──────────────┴─────────────────┘
```

## Configuration

### TOML File

```toml
[general]
log_level = "info"
smart_routing = true
venue_priority = ["lx_dex", "binance"]

[risk]
enabled = true
max_position_size = 1000
max_order_size = 100
max_daily_loss = 5000
kill_switch_enabled = true

[native.lx_dex]
venue_type = "dex"
api_url = "https://api.dex.lux.network"
api_key = "your-api-key"
api_secret = "your-api-secret"

[native.lx_amm]
venue_type = "amm"
api_url = "https://api.dex.lux.network"

[ccxt.binance]
exchange_id = "binance"
api_key = "your-key"
api_secret = "your-secret"

[hummingbot.gateway]
host = "localhost"
port = 15888
connector = "lxdex"
chain = "lux"
network = "mainnet"
```

## Execution Algorithms

### TWAP (Time-Weighted Average Price)

Split large orders over time to minimize market impact:

```python
from lx_trading.execution import TwapExecutor

twap = TwapExecutor(
    client=client,
    symbol="BTC-USDC",
    side=Side.BUY,
    total_quantity=Decimal("10"),
    duration_seconds=3600,  # 1 hour
    num_slices=12,
)
orders = await twap.execute()
```

### Iceberg Orders

Hide large order size:

```python
from lx_trading.execution import IcebergExecutor

iceberg = IcebergExecutor(
    client=client,
    symbol="BTC-USDC",
    side=Side.BUY,
    total_quantity=Decimal("100"),
    visible_quantity=Decimal("5"),
    price=Decimal("50000"),
)
orders = await iceberg.execute()
```

## Financial Math

```python
from lx_trading.math import (
    black_scholes,
    greeks,
    implied_volatility,
    constant_product_price,
    var,
    cvar,
    sharpe_ratio,
)

# Options pricing
call = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
g = greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)

# AMM pricing
output, price = constant_product_price(
    reserve_x=1000000,
    reserve_y=1000000,
    amount_in=1000,
    fee_rate=0.003
)

# Risk metrics
returns = [0.01, -0.02, 0.03, -0.01, 0.02]
print(f"VaR 95%: {var(returns, 0.95):.2%}")
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
```

## Performance Comparison

| Metric | LX Trading | CCXT | Hummingbot |
|--------|------------|------|------------|
| Order Latency | <1ms | 10-50ms | 5-20ms |
| Throughput | 1M+ orders/sec | 100 orders/sec | 1000 orders/sec |
| Languages | 5 | 2 (Python, JS) | 1 (Python) |
| Smart Routing | ✅ Built-in | ❌ | ✅ Gateway |
| Risk Management | ✅ Built-in | ❌ | ✅ Basic |
| Execution Algos | ✅ TWAP/VWAP/Iceberg | ❌ | ✅ Basic |

## Installation

### Python
```bash
pip install lx-trading
```

### Go
```bash
go get github.com/luxfi/trading
```

### TypeScript/Node.js
```bash
npm install @luxfi/trading
```

### Rust
```toml
[dependencies]
lx-trading = "0.1"
```

### C++
```cmake
find_package(lx-trading REQUIRED)
target_link_libraries(myapp lx::trading)
```

## Links

- [LX DEX](https://dex.lux.network)
- [Documentation](https://dex.lux.network/docs/sdk)
- [GitHub](https://github.com/luxfi/dex)
- [Discord](https://discord.gg/luxfi)
