# LUX DEX Rust SDK

High-performance Rust client for the LX decentralized exchange.

## Features

- Async WebSocket client with automatic reconnection
- HTTP/JSON-RPC client for request-response operations
- Type-safe order, trade, and position management
- Order book utilities with slippage estimation
- Real-time market data subscriptions
- Margin trading support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lux-dex = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use lux_dex::{Client, Order, Side};

#[tokio::main]
async fn main() -> lux_dex::Result<()> {
    // Create client
    let client = Client::new();

    // Connect to WebSocket
    client.connect().await?;

    // Authenticate
    client.authenticate("api_key", "api_secret").await?;

    // Subscribe to market data
    client.subscribe_orderbook("BTC-USDT").await?;

    // Place a limit order
    let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1);
    client.place_order(&order).await?;

    Ok(())
}
```

## Configuration

```rust
use lux_dex::{Client, ClientConfig};

let config = ClientConfig::default()
    .with_ws_url("ws://localhost:8081")
    .with_http_url("http://localhost:8080")
    .with_credentials("api_key", "api_secret");

let client = Client::with_config(config);
```

Environment variables:
- `LX_WS_URL` - WebSocket endpoint (default: `ws://localhost:8081`)
- `LX_HTTP_URL` - HTTP endpoint (default: `http://localhost:8080`)
- `LX_API_KEY` - API key for authentication
- `LX_API_SECRET` - API secret for authentication

## Event Handling

```rust
use lux_dex::{Client, WsEvent};

let client = Client::new();
client.connect().await?;

// Take the event receiver (can only be called once)
let mut events = client.take_event_receiver().await.unwrap();

// Process events
while let Some(event) = events.recv().await {
    match event {
        WsEvent::OrderBook(book) => {
            println!("Best bid: {:?}, Best ask: {:?}",
                book.best_bid(), book.best_ask());
        }
        WsEvent::Trade(trade) => {
            println!("Trade: {} @ {}", trade.size, trade.price);
        }
        WsEvent::OrderUpdate { order, status } => {
            println!("Order {}: {}", order.order_id.unwrap_or(0), status);
        }
        WsEvent::Error { message, .. } => {
            eprintln!("Error: {}", message);
        }
        _ => {}
    }
}
```

## Order Types

```rust
use lux_dex::{Order, Side, TimeInForce};

// Limit order
let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1);

// Market order
let order = Order::market("BTC-USDT", Side::Sell, 0.1);

// Post-only limit order
let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1)
    .post_only();

// IOC order
let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1)
    .with_time_in_force(TimeInForce::IOC);

// With client ID
let order = Order::limit("BTC-USDT", Side::Buy, 50000.0, 0.1)
    .with_client_id("my-order-001");
```

## Order Book Operations

```rust
use lux_dex::OrderBook;

let book = client.get_orderbook("BTC-USDT", 20).await?;

// Best prices
println!("Best bid: {:?}", book.best_bid());
println!("Best ask: {:?}", book.best_ask());
println!("Mid price: {:?}", book.mid_price());
println!("Spread: {:?}", book.spread());
println!("Spread %: {:?}", book.spread_percent());

// Liquidity analysis
println!("Bid liquidity (5 levels): {}", book.bid_liquidity(5));
println!("Ask liquidity (5 levels): {}", book.ask_liquidity(5));
println!("Imbalance: {}", book.imbalance(5));

// Slippage estimation
if let Some(avg_price) = book.estimate_buy_slippage(1.0) {
    println!("Avg price to buy 1 BTC: {}", avg_price);
}
```

## Margin Trading

```rust
// Open a long position with 10x leverage
client.open_position("BTC-USDT", "buy", 0.1, 10.0).await?;

// Close position
client.close_position("position_id", 0.1).await?;

// Modify leverage
client.modify_leverage("position_id", 5.0).await?;
```

## HTTP API

For request-response operations without WebSocket:

```rust
// Get node info
let info = client.get_info().await?;
println!("Version: {}", info.version);

// Get order book
let book = client.get_orderbook("BTC-USDT", 20).await?;

// Get recent trades
let trades = client.get_trades("BTC-USDT", 100).await?;

// Place order via HTTP
let response = client.place_order_http(&order).await?;
println!("Order ID: {}", response.order_id);
```

## Error Handling

```rust
use lux_dex::{Error, Result};

match client.place_order(&order).await {
    Ok(()) => println!("Order placed"),
    Err(Error::Auth(msg)) => eprintln!("Auth error: {}", msg),
    Err(Error::Server { code, message }) => {
        eprintln!("Server error {}: {}", code, message);
    }
    Err(Error::NotConnected) => {
        client.connect().await?;
    }
    Err(e) if e.is_recoverable() => {
        // Retry logic
    }
    Err(e) => return Err(e),
}
```

## Running the Example

```bash
# Start LX server
# Then run the example:
cargo run --example basic

# With credentials:
LX_API_KEY=your_key LX_API_SECRET=your_secret cargo run --example basic
```

## API Reference

### Client Methods

| Method | Description |
|--------|-------------|
| `connect()` | Connect to WebSocket server |
| `disconnect()` | Disconnect from server |
| `authenticate(key, secret)` | Authenticate with API credentials |
| `place_order(&order)` | Place a new order |
| `cancel_order(id)` | Cancel an existing order |
| `modify_order(id, price, size)` | Modify an order |
| `subscribe_orderbook(symbol)` | Subscribe to order book updates |
| `subscribe_trades(symbol)` | Subscribe to trade updates |
| `subscribe_prices(symbols)` | Subscribe to price updates |
| `get_balances()` | Get account balances |
| `get_positions()` | Get open positions |
| `get_orders()` | Get open orders |

### WebSocket Events

| Event | Description |
|-------|-------------|
| `Connected` | Successfully connected to server |
| `Authenticated` | Successfully authenticated |
| `OrderBook` | Order book snapshot |
| `OrderBookUpdate` | Incremental order book update |
| `Trade` | Trade executed |
| `OrderUpdate` | Order status change |
| `Position` | Position update |
| `Balance` | Balance update |
| `Price` | Price update |
| `Error` | Error message |
| `Pong` | Ping response |

## License

MIT OR Apache-2.0
