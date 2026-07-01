# LX Trading SDK - Python

High-frequency trading SDK with unified liquidity aggregation for Python.

## Features

- **Unified API**: Same interface for native LX DEX, CCXT exchanges, and Hummingbot Gateway
- **Smart Order Routing**: Automatic best-price execution across venues
- **Aggregated Orderbook**: Combined liquidity view from all venues
- **AMM Support**: Swap, add/remove liquidity, LP position tracking
- **Execution Algos**: TWAP, VWAP, Iceberg, Sniper
- **Risk Management**: Position limits, daily loss limits, kill switch
- **Financial Math**: Black-Scholes, Greeks, VaR/CVaR, AMM pricing

## Installation

```bash
pip install lx-trading
```

## Quick Start

```python
import asyncio
from decimal import Decimal
from lx_trading import Client, Config

async def main():
    # Load config
    config = Config.from_file("config.toml")

    # Or build programmatically
    config = Config()
    config.with_native("lx_dex", NativeVenueConfig.lx_dex("https://api.dex.lux.network"))
    config.with_ccxt("binance", CcxtConfig.new("binance").with_credentials("key", "secret"))

    # Create client and connect
    client = Client(config)
    await client.connect()

    # Get aggregated orderbook
    book = await client.aggregated_orderbook("BTC-USDC")
    print(f"Best bid: {book.best_bid()}")
    print(f"Best ask: {book.best_ask()}")

    # Smart order routing - best price across all venues
    order = await client.buy("BTC-USDC", Decimal("0.1"))
    print(f"Filled on {order.venue} at {order.average_price}")

    # Or target specific venue
    order = await client.buy("BTC-USDC", Decimal("0.1"), venue="binance")

    # AMM swap
    trade = await client.swap("LUX", "USDC", Decimal("100"), is_buy=True, slippage=0.01, venue="lx_amm")

    await client.disconnect()

asyncio.run(main())
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
max_daily_loss = 5000

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

[ccxt.mexc]
exchange_id = "mexc"
api_key = "your-key"
api_secret = "your-secret"

[hummingbot.gateway]
connector = "lxdex"
chain = "lux"
network = "mainnet"
```

## Financial Mathematics

```python
from lx_trading.math import (
    black_scholes,
    implied_volatility,
    greeks,
    constant_product_price,
    volatility,
    sharpe_ratio,
    var,
    cvar,
)

# Options pricing
call_price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
iv = implied_volatility(price=10.45, S=100, K=100, T=1, r=0.05, option_type="call")
g = greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"Delta: {g['delta']:.4f}, Gamma: {g['gamma']:.6f}")

# AMM pricing
output, price = constant_product_price(
    reserve_x=1000000,
    reserve_y=1000000,
    amount_in=1000,
    fee_rate=0.003
)

# Risk metrics
returns = [0.01, -0.02, 0.03, -0.01, 0.02]
print(f"Volatility: {volatility(returns):.2%}")
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"VaR 95%: {var(returns, 0.95):.2%}")
print(f"CVaR 95%: {cvar(returns, 0.95):.2%}")
```

## Execution Algorithms

```python
from lx_trading.execution import TwapExecutor, VwapExecutor, IcebergExecutor

# TWAP - spread order over time
twap = TwapExecutor(
    client=client,
    symbol="BTC-USDC",
    side=Side.BUY,
    total_quantity=Decimal("10"),
    duration_seconds=3600,  # 1 hour
    num_slices=12,  # 12 slices
)
orders = await twap.execute()

# Iceberg - hide large order
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

## Risk Management

```python
from lx_trading import RiskManager
from lx_trading.config import RiskConfig

risk = RiskManager(RiskConfig(
    enabled=True,
    max_position_size=Decimal("100"),
    max_order_size=Decimal("10"),
    max_daily_loss=Decimal("1000"),
    kill_switch_enabled=True,
))

# Validate before placing order
try:
    risk.validate_order(order_request)
except RiskError as e:
    print(f"Order rejected: {e}")

# Update after trade
risk.update_position("BTC", Decimal("1"), Side.BUY)
risk.update_pnl(Decimal("-50"))  # Loss

# Check status
print(f"BTC position: {risk.position('BTC')}")
print(f"Daily PnL: {risk.daily_pnl}")
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

## Links

- [LX DEX](https://dex.lux.network)
- [Documentation](https://dex.lux.network/docs/sdk/python)
- [GitHub](https://github.com/luxfi/dex)
