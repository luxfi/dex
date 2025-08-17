# LX DEX API Documentation

## Overview

The LX DEX provides a comprehensive WebSocket API for real-time trading, including spot trading, margin trading with up to 100x leverage, perpetual futures, lending/borrowing, and vault management. The API supports both REST endpoints for queries and WebSocket connections for real-time updates.

## WebSocket API

### Connection

Connect to the WebSocket endpoint:
```
wss://api.lx-dex.com/ws
```

For local development:
```
ws://localhost:8080/ws
```

### Authentication

After connecting, authenticate using your API credentials:

```json
{
  "type": "auth",
  "apiKey": "your_api_key",
  "apiSecret": "your_api_secret",
  "timestamp": 1234567890
}
```

Response:
```json
{
  "type": "auth_success",
  "data": {
    "user_id": "user123"
  },
  "timestamp": 1234567890
}
```

### Message Format

All messages follow this structure:

```json
{
  "type": "message_type",
  "data": {
    // Message-specific data
  },
  "request_id": "optional_request_id",
  "timestamp": 1234567890
}
```

Error responses:
```json
{
  "type": "error",
  "error": "Error message",
  "request_id": "matching_request_id",
  "timestamp": 1234567890
}
```

## Trading Operations

### Place Order

Submit a new order to the order book.

**Request:**
```json
{
  "type": "place_order",
  "order": {
    "symbol": "BTC-USDT",
    "side": "buy",
    "type": "limit",
    "price": 50000,
    "size": 0.1,
    "time_in_force": "GTC",
    "reduce_only": false,
    "post_only": false
  },
  "request_id": "order_001"
}
```

**Order Types:**
- `limit` - Limit order at specified price
- `market` - Market order (immediate execution)
- `stop` - Stop order (triggers at stop price)
- `stop_limit` - Stop limit order
- `trailing_stop` - Trailing stop with offset

**Time in Force:**
- `GTC` - Good Till Cancelled
- `IOC` - Immediate or Cancel
- `FOK` - Fill or Kill
- `GTT` - Good Till Time (requires expiry)

**Response:**
```json
{
  "type": "order_update",
  "data": {
    "order": {
      "id": 12345,
      "symbol": "BTC-USDT",
      "side": "buy",
      "type": "limit",
      "price": 50000,
      "size": 0.1,
      "filled": 0,
      "status": "open",
      "timestamp": 1234567890
    },
    "status": "submitted"
  },
  "request_id": "order_001"
}
```

### Cancel Order

Cancel an existing order.

**Request:**
```json
{
  "type": "cancel_order",
  "orderID": 12345,
  "request_id": "cancel_001"
}
```

### Modify Order

Modify an existing order's price or size.

**Request:**
```json
{
  "type": "modify_order",
  "orderID": 12345,
  "newPrice": 51000,
  "newSize": 0.2,
  "request_id": "modify_001"
}
```

## Margin Trading

### Open Margin Position

Open a leveraged position.

**Request:**
```json
{
  "type": "open_position",
  "symbol": "BTC-USDT",
  "side": "buy",
  "size": 1.0,
  "leverage": 50,
  "request_id": "pos_001"
}
```

**Leverage Limits:**
- BTC/ETH: Up to 100x
- Major Altcoins: Up to 50x
- Other Pairs: Up to 20x

**Response:**
```json
{
  "type": "position_update",
  "data": {
    "position": {
      "id": "pos_abc123",
      "symbol": "BTC-USDT",
      "side": "buy",
      "size": 1.0,
      "entry_price": 50000,
      "mark_price": 50100,
      "liquidation_price": 47500,
      "leverage": 50,
      "margin": "1000",
      "unrealized_pnl": "100",
      "realized_pnl": "0"
    },
    "action": "opened"
  }
}
```

### Close Position

Close or partially close a position.

**Request:**
```json
{
  "type": "close_position",
  "positionID": "pos_abc123",
  "size": 0.5,
  "request_id": "close_001"
}
```

### Modify Leverage

Adjust leverage for an existing position.

**Request:**
```json
{
  "type": "modify_leverage",
  "positionID": "pos_abc123",
  "newLeverage": 25,
  "request_id": "leverage_001"
}
```

## Vault Operations

### Deposit to Vault

Deposit funds into a liquidity vault.

**Request:**
```json
{
  "type": "vault_deposit",
  "vaultID": "vault_hlp",
  "amount": "10000",
  "request_id": "deposit_001"
}
```

**Available Vaults:**
- `vault_hlp` - Hyperliquid LP Vault
- `vault_mm` - Market Making Vault
- `vault_yield` - Yield Farming Vault
- `vault_insurance` - Insurance Fund Vault

### Withdraw from Vault

Withdraw funds from a vault.

**Request:**
```json
{
  "type": "vault_withdraw",
  "vaultID": "vault_hlp",
  "shares": "1000",
  "request_id": "withdraw_001"
}
```

## Lending/Borrowing

### Supply Assets

Supply assets to the lending pool.

**Request:**
```json
{
  "type": "lending_supply",
  "asset": "USDT",
  "amount": "10000",
  "request_id": "supply_001"
}
```

### Borrow Assets

Borrow from the lending pool.

**Request:**
```json
{
  "type": "lending_borrow",
  "asset": "BTC",
  "amount": "1000000000",
  "request_id": "borrow_001"
}
```

### Repay Loan

Repay borrowed amount.

**Request:**
```json
{
  "type": "lending_repay",
  "asset": "BTC",
  "amount": "1000000000",
  "request_id": "repay_001"
}
```

## Market Data Subscriptions

### Subscribe to Channels

Subscribe to real-time market data updates.

**Request:**
```json
{
  "type": "subscribe",
  "channel": "orderbook",
  "symbols": ["BTC-USDT", "ETH-USDT"],
  "request_id": "sub_001"
}
```

**Available Channels:**
- `orderbook` - Order book updates (snapshot + diffs)
- `trades` - Trade executions
- `prices` - Price updates
- `ticker` - 24h ticker data
- `funding` - Funding rates (perpetuals)
- `liquidations` - Liquidation events

### Order Book Updates

Real-time order book snapshots and updates.

**Update Message:**
```json
{
  "type": "orderbook_update",
  "data": {
    "symbol": "BTC-USDT",
    "snapshot": {
      "bids": [
        [50000, 1.5],
        [49999, 2.0]
      ],
      "asks": [
        [50001, 1.2],
        [50002, 0.8]
      ],
      "timestamp": 1234567890
    }
  }
}
```

### Trade Updates

Real-time trade executions.

**Update Message:**
```json
{
  "type": "trade_update",
  "data": {
    "trade": {
      "id": "trade_123",
      "symbol": "BTC-USDT",
      "price": 50000,
      "size": 0.1,
      "side": "buy",
      "timestamp": 1234567890
    }
  }
}
```

### Price Updates

Real-time price updates.

**Update Message:**
```json
{
  "type": "price_update",
  "data": {
    "symbol": "BTC-USDT",
    "price": 50000,
    "bid": 49999,
    "ask": 50001,
    "volume": 1234.56,
    "timestamp": 1234567890
  }
}
```

## Account Data

### Get Balances

Request current account balances.

**Request:**
```json
{
  "type": "get_balances",
  "request_id": "bal_001"
}
```

**Response:**
```json
{
  "type": "balance_update",
  "data": {
    "balances": {
      "BTC": "1000000000",
      "USDT": "50000000000",
      "ETH": "10000000000"
    }
  }
}
```

### Get Positions

Request current margin positions.

**Request:**
```json
{
  "type": "get_positions",
  "request_id": "pos_001"
}
```

### Get Orders

Request open orders.

**Request:**
```json
{
  "type": "get_orders",
  "request_id": "orders_001"
}
```

## Position Updates

### Position Update Events

Automatic updates when positions change.

**Update Message:**
```json
{
  "type": "position_update",
  "data": {
    "position": {
      "id": "pos_abc123",
      "symbol": "BTC-USDT",
      "side": "buy",
      "size": 1.0,
      "mark_price": 48000,
      "liquidation_price": 47500,
      "unrealized_pnl": "-2000"
    },
    "action": "modified"
  }
}
```

**Actions:**
- `opened` - New position opened
- `modified` - Position size/leverage changed
- `closed` - Position closed
- `liquidated` - Position liquidated

### Liquidation Notification

Alert when position is liquidated.

**Notification:**
```json
{
  "type": "position_update",
  "data": {
    "position": {
      "id": "pos_abc123"
    },
    "action": "liquidated",
    "message": "Position pos_abc123 liquidated at 47500.00"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 1001 | Invalid message format |
| 1002 | Authentication required |
| 1003 | Authentication failed |
| 1004 | Rate limit exceeded |
| 2001 | Invalid order parameters |
| 2002 | Insufficient balance |
| 2003 | Order not found |
| 2004 | Market closed |
| 3001 | Invalid position |
| 3002 | Leverage too high |
| 3003 | Position at risk |
| 3004 | Liquidation in progress |
| 4001 | Vault not found |
| 4002 | Vault capacity exceeded |
| 4003 | Withdrawal locked |
| 5001 | Lending pool exhausted |
| 5002 | Collateral insufficient |
| 5003 | Borrow limit exceeded |

## Rate Limits

- **Orders:** 100 requests per minute
- **Market Data:** 1000 requests per minute
- **Account Data:** 60 requests per minute
- **WebSocket Messages:** 100 messages per second

## SDK Examples

### JavaScript/TypeScript

```typescript
import { TraderClient } from '@lx-dex/trader-client';

const client = new TraderClient({
  apiEndpoint: 'https://api.lx-dex.com',
  wsEndpoint: 'wss://api.lx-dex.com/ws',
  apiKey: 'your_api_key',
  apiSecret: 'your_api_secret'
});

// Connect
await client.connect();

// Place order
const order = await client.placeOrder({
  symbol: 'BTC-USDT',
  side: 'buy',
  type: 'limit',
  price: 50000,
  size: 0.1
});

// Open leveraged position
const position = await client.openMarginPosition(
  'BTC-USDT',
  'buy',
  1.0,
  50 // 50x leverage
);

// Subscribe to price updates
client.subscribe('prices', ['BTC-USDT']);
client.onPriceUpdate((update) => {
  console.log(`BTC Price: ${update.price}`);
});
```

### Python

```python
from lx_dex import TraderClient

client = TraderClient(
    api_endpoint='https://api.lx-dex.com',
    ws_endpoint='wss://api.lx-dex.com/ws',
    api_key='your_api_key',
    api_secret='your_api_secret'
)

# Connect
await client.connect()

# Place order
order = await client.place_order(
    symbol='BTC-USDT',
    side='buy',
    order_type='limit',
    price=50000,
    size=0.1
)

# Open leveraged position
position = await client.open_margin_position(
    symbol='BTC-USDT',
    side='buy',
    size=1.0,
    leverage=50
)

# Subscribe to trades
await client.subscribe('trades', ['BTC-USDT'])

async for trade in client.trade_stream():
    print(f"Trade: {trade.price} x {trade.size}")
```

### Go

```go
import "github.com/luxfi/dex/client"

config := client.ClientConfig{
    APIEndpoint: "https://api.lx-dex.com",
    WSEndpoint:  "wss://api.lx-dex.com/ws",
    APIKey:      "your_api_key",
    APISecret:   "your_api_secret",
}

client, err := client.NewTraderClient(config)
if err != nil {
    log.Fatal(err)
}

// Connect
err = client.Connect()

// Place order
order, err := client.PlaceOrder(&lx.Order{
    Symbol: "BTC-USDT",
    Side:   lx.Buy,
    Type:   lx.Limit,
    Price:  50000,
    Size:   0.1,
})

// Open leveraged position
position, err := client.OpenMarginPosition(
    "BTC-USDT",
    lx.Buy,
    1.0,
    50.0, // 50x leverage
)

// Subscribe to order book
client.Subscribe("orderbook", []string{"BTC-USDT"})

// Handle updates
for update := range client.GetPriceUpdates() {
    fmt.Printf("Price: %f\n", update.Price)
}
```

## Risk Management

### Margin Requirements

| Account Type | Initial Margin | Maintenance Margin | Max Leverage |
|--------------|----------------|-------------------|--------------|
| Cross Margin | 10% | 5% | 10x |
| Isolated Margin | 5% | 2.5% | 20x |
| Portfolio Margin | 1% | 0.5% | 100x |

### Liquidation Process

1. **Warning Level** (Margin Level < 150%): Notification sent
2. **Margin Call** (Margin Level < 120%): Reduce-only mode
3. **Liquidation** (Margin Level < 100%): Position liquidated
4. **Insurance Fund**: Covers losses if liquidation price not achieved
5. **Auto-Deleveraging**: If insurance fund depleted by 20%
6. **Socialized Loss**: Last resort, distributed among profitable traders

### Position Limits

| Asset | Max Position Size | Max Leverage |
|-------|------------------|--------------|
| BTC | 100 BTC | 100x |
| ETH | 1000 ETH | 100x |
| Major Alts | $1M notional | 50x |
| Other | $500K notional | 20x |

## Security

### API Key Permissions

Configure API key permissions in account settings:
- **Read**: View balances, positions, orders
- **Trade**: Place, modify, cancel orders
- **Margin**: Open/close leveraged positions
- **Withdraw**: Not available via API (web only)

### Best Practices

1. **Use unique API keys** for each application
2. **Implement request signing** for additional security
3. **Monitor rate limits** to avoid throttling
4. **Handle disconnections** with exponential backoff
5. **Subscribe selectively** to reduce bandwidth
6. **Use request IDs** for request/response matching
7. **Implement heartbeat** to detect stale connections

## Support

- **Documentation**: https://docs.lx-dex.com
- **API Status**: https://status.lx-dex.com
- **Support Email**: api-support@lx-dex.com
- **Discord**: https://discord.gg/lx-dex
- **GitHub**: https://github.com/luxfi/dex

## Changelog

### Version 2.0.0 (Current)
- Added 100x leverage for BTC/ETH
- Unified liquidity pools
- Advanced order types (iceberg, hidden)
- Vault strategies
- X-Chain settlement integration

### Version 1.5.0
- Lending/borrowing protocol
- Insurance fund
- Auto-deleveraging mechanism
- Circuit breakers

### Version 1.0.0
- Initial release
- Basic spot and margin trading
- WebSocket API
- Order book and trades