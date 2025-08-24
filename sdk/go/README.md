# LX DEX Go SDK

Official Go SDK for interacting with the LX DEX trading platform.

## Features

- **Multiple Protocol Support**: JSON-RPC, gRPC, and WebSocket
- **High Performance**: Direct gRPC connection for low-latency operations
- **Real-time Data**: WebSocket subscriptions for live market data
- **Type Safety**: Strongly typed Go interfaces
- **Concurrent Safe**: Thread-safe operations with proper locking
- **Automatic Failover**: Falls back to JSON-RPC if gRPC unavailable

## Installation

```bash
go get github.com/luxfi/dex/sdk/go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/luxfi/dex/sdk/go/client"
)

func main() {
    // Create client
    c, err := client.NewClient(
        client.WithJSONRPCURL("http://localhost:8080"),
        client.WithWebSocketURL("ws://localhost:8081"),
        client.WithGRPCURL("localhost:50051"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer c.Disconnect()
    
    ctx := context.Background()
    
    // Connect to gRPC for best performance
    if err := c.ConnectGRPC(ctx); err != nil {
        log.Printf("Using JSON-RPC fallback: %v", err)
    }
    
    // Place an order
    order := &client.Order{
        Symbol: "BTC-USD",
        Type:   client.OrderTypeLimit,
        Side:   client.OrderSideBuy,
        Price:  50000.00,
        Size:   0.1,
    }
    
    resp, err := c.PlaceOrder(ctx, order)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Order placed: %d\n", resp.OrderID)
}
```

## Client Options

Configure the client with various options:

```go
client := client.NewClient(
    client.WithJSONRPCURL("http://localhost:8080"),
    client.WithWebSocketURL("ws://localhost:8081"),
    client.WithGRPCURL("localhost:50051"),
    client.WithAPIKey("your-api-key"),
)
```

## Order Management

### Place Order

```go
order := &client.Order{
    Symbol:      "BTC-USD",
    Type:        client.OrderTypeLimit,
    Side:        client.OrderSideBuy,
    Price:       50000.00,
    Size:        0.1,
    UserID:      "user123",
    ClientID:    "my-order-001",
    TimeInForce: client.TimeInForceGTC,
    PostOnly:    false,
    ReduceOnly:  false,
}

resp, err := c.PlaceOrder(ctx, order)
if err != nil {
    // Handle error
}

fmt.Printf("Order ID: %d, Status: %s\n", resp.OrderID, resp.Status)
```

### Cancel Order

```go
err := c.CancelOrder(ctx, orderID)
if err != nil {
    // Handle error
}
```

## Market Data

### Get Order Book

```go
orderBook, err := c.GetOrderBook(ctx, "BTC-USD", 10)
if err != nil {
    // Handle error
}

fmt.Printf("Best Bid: %.2f\n", orderBook.BestBid())
fmt.Printf("Best Ask: %.2f\n", orderBook.BestAsk())
fmt.Printf("Spread: %.2f (%.3f%%)\n", orderBook.Spread(), orderBook.SpreadPercentage())
```

### Get Recent Trades

```go
trades, err := c.GetTrades(ctx, "BTC-USD", 100)
if err != nil {
    // Handle error
}

for _, trade := range trades {
    fmt.Printf("Trade: %.2f @ %.8f\n", trade.Price, trade.Size)
}
```

## Real-time Data

### WebSocket Subscriptions

```go
// Connect WebSocket
if err := c.ConnectWebSocket(ctx); err != nil {
    log.Fatal(err)
}

// Subscribe to order book updates
err := c.SubscribeOrderBook("BTC-USD", func(ob *client.OrderBook) {
    fmt.Printf("Order book update: Bid %.2f, Ask %.2f\n",
        ob.BestBid(), ob.BestAsk())
})

// Subscribe to trades
err = c.SubscribeTrades("BTC-USD", func(trade *client.Trade) {
    fmt.Printf("New trade: %.2f @ %.8f\n", trade.Price, trade.Size)
})
```

### gRPC Streaming

```go
// Stream order book via gRPC
obChan, err := c.StreamOrderBook(ctx, "BTC-USD")
if err != nil {
    log.Fatal(err)
}

for ob := range obChan {
    fmt.Printf("Streamed update: Bid %.2f, Ask %.2f\n",
        ob.BestBid(), ob.BestAsk())
}
```

## Order Types

The SDK supports all LX DEX order types:

```go
const (
    OrderTypeLimit     // Basic limit order
    OrderTypeMarket    // Market order
    OrderTypeStop      // Stop order
    OrderTypeStopLimit // Stop-limit order
    OrderTypeIceberg   // Iceberg order
    OrderTypePeg       // Pegged order
)
```

## Time in Force

Control order lifetime:

```go
const (
    TimeInForceGTC // Good Till Cancelled
    TimeInForceIOC // Immediate Or Cancel
    TimeInForceFOK // Fill Or Kill
    TimeInForceDAY // Day Order
)
```

## Error Handling

The SDK returns standard Go errors:

```go
resp, err := c.PlaceOrder(ctx, order)
if err != nil {
    // Check error type
    switch err.Error() {
    case "insufficient balance":
        // Handle insufficient balance
    case "market closed":
        // Handle market closed
    default:
        // Handle other errors
    }
}
```

## Helper Methods

The SDK provides convenient helper methods:

```go
// Order helpers
order.IsOpen()     // Check if order is open
order.IsClosed()   // Check if order is closed
order.FillRate()   // Calculate fill rate

// Trade helpers
trade.TotalValue()    // Calculate trade value
trade.TimestampTime() // Get timestamp as time.Time

// OrderBook helpers
orderBook.BestBid()          // Get best bid price
orderBook.BestAsk()          // Get best ask price
orderBook.Spread()           // Calculate spread
orderBook.MidPrice()         // Calculate mid price
orderBook.SpreadPercentage() // Spread as percentage

// Position helpers
position.UnrealizedPnL()  // Calculate unrealized P&L
position.PnLPercentage()   // P&L as percentage

// Balance helpers
balance.Utilization() // Calculate balance utilization
```

## Performance Tips

1. **Use gRPC**: Connect via gRPC for lowest latency
2. **Reuse Client**: Create one client and reuse it
3. **Batch Operations**: Use concurrent requests when possible
4. **Stream vs Poll**: Use streaming for real-time data instead of polling

## Thread Safety

The client is thread-safe and can be used concurrently:

```go
var wg sync.WaitGroup

// Place multiple orders concurrently
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(i int) {
        defer wg.Done()
        
        order := &client.Order{
            Symbol: "BTC-USD",
            Type:   client.OrderTypeLimit,
            Side:   client.OrderSideBuy,
            Price:  50000.00 + float64(i),
            Size:   0.1,
        }
        
        _, err := c.PlaceOrder(ctx, order)
        if err != nil {
            log.Printf("Order %d failed: %v", i, err)
        }
    }(i)
}

wg.Wait()
```

## Examples

See the [examples](examples/) directory for complete working examples:

- Basic order placement
- Market data retrieval
- WebSocket subscriptions
- gRPC streaming
- Error handling

## Development

### Running Tests

```bash
go test ./...
```

### Building

```bash
go build ./...
```

## Support

For issues and questions:
- GitHub: https://github.com/luxfi/dex
- Documentation: https://docs.lux.network

## License

MIT License - see LICENSE file for details