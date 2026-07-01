# LX DEX Hummingbot Connectors

Official Hummingbot connectors for [LX DEX](https://dex.lux.network).

## Connectors

### 1. LX DEX Spot (`lx_dex`)

Central Limit Order Book (OrderBook) connector for spot trading.

**Features:**
- Limit and market orders
- Real-time order book streaming
- Order management (place, cancel, track)
- Balance tracking

**Usage:**
```
>>> connect lx_dex
>>> create --strategy pure_market_making --exchange lx_dex --market LUX-USDC
>>> start
```

### 2. LX AMM (`lx_amm`)

Automated Market Maker connector for liquidity pool operations.

**Features:**
- Token swaps via AMM pools
- Add/remove liquidity
- LP position tracking
- Real-time price quotes

**Usage:**
```
>>> connect lx_amm
>>> create --strategy amm_arb --connector lx_amm --market LUX-USDC
>>> start
```

## Configuration

### API Keys

1. Generate API keys at [dex.lux.network](https://dex.lux.network)
2. Connect in Hummingbot:
```
>>> connect lx_dex
Enter your LX DEX API key: ***
Enter your LX DEX API secret: ***
Enter your wallet address: 0x...
```

### Networks

| Network | Description |
|---------|-------------|
| mainnet | Production network |
| testnet | Test network (free tokens via faucet) |

## Strategies

### Pure Market Making (OrderBook)
```yaml
strategy: pure_market_making
exchange: lx_dex
market: LUX-USDC
bid_spread: 0.5%
ask_spread: 0.5%
order_amount: 100
```

### AMM Arbitrage
```yaml
strategy: amm_arb
connector_1: lx_amm
market_1: LUX-USDC
connector_2: binance
market_2: LUX-USDC
min_profitability: 0.5%
```

### Cross-Exchange Market Making
```yaml
strategy: cross_exchange_market_making
maker_exchange: lx_dex
taker_exchange: binance
market: LUX-USDC
min_profitability: 0.2%
```

## Installation

### From Source
```bash
# Clone Hummingbot
git clone https://github.com/hummingbot/hummingbot.git
cd hummingbot

# Copy LX connectors
cp -r /path/to/lx/dex/hummingbot/connector/exchange/lx_dex hummingbot/connector/exchange/
cp -r /path/to/lx/dex/hummingbot/connector/gateway/amm/lx_amm hummingbot/connector/gateway/amm/

# Install
./install
```

### With Gateway (for AMM)
```bash
# Start Gateway
cd gateway
npm install
npm run start

# Configure Gateway for LX
# Add to conf/lxdex.yml
```

## API Reference

### Spot Connector Methods

| Method | Description |
|--------|-------------|
| `buy()` | Place buy order |
| `sell()` | Place sell order |
| `cancel()` | Cancel order |
| `get_balance()` | Get wallet balance |
| `get_order_book()` | Get order book snapshot |

### AMM Connector Methods

| Method | Description |
|--------|-------------|
| `buy()` | Execute buy swap |
| `sell()` | Execute sell swap |
| `get_quote_price()` | Get swap quote |
| `get_pool_info()` | Get pool reserves |
| `add_liquidity()` | Add LP position |
| `remove_liquidity()` | Remove LP position |
| `get_lp_positions()` | Get LP positions |

## Links

- [LX DEX Documentation](https://dex.lux.network/docs)
- [Hummingbot Documentation](https://docs.hummingbot.org)
- [Discord Support](https://discord.gg/lux)
