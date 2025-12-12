# LX Trading Client

Multi-protocol programmatic trading client for LX. Supports WebSocket and gRPC protocols with a unified interface.

## Features

- **Dual Protocol Support**: WebSocket and gRPC with runtime switching
- **Unified Interface**: Same API across both protocols
- **Multiple Usage Modes**: Package import, CLI commands, or interactive shell
- **Real-time Streaming**: Market data subscriptions via WebSocket or gRPC streams

## Installation

```bash
go build -o lx-client .
```

## Quick Start

### WebSocket (Default)

```bash
# Interactive mode
./lx-client -i

# Single command
./lx-client place_order BTC-USD buy limit 50000 0.1

# With authentication
./lx-client -key mykey -secret mysecret -i
```

### gRPC

```bash
# Interactive mode via gRPC
./lx-client -protocol grpc -i

# Single command via gRPC
./lx-client -protocol grpc get_positions

# Custom gRPC endpoint
./lx-client -protocol grpc -grpc-addr exchange.lux.network:9090 ping
```

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `-protocol` | `ws` | Protocol: `ws` or `grpc` |
| `-ws-url` | `ws://localhost:8081` | WebSocket server URL |
| `-grpc-addr` | `localhost:9090` | gRPC server address |
| `-key` | | API key for authentication |
| `-secret` | | API secret for authentication |
| `-i` | false | Interactive mode |
| `-v` | false | Verbose output |

## Commands

### Trading

| Command | Description |
|---------|-------------|
| `place_order <symbol> <side> <type> <price> <size>` | Place a new order |
| `cancel_order <order_id>` | Cancel an existing order |
| `get_orders` | List all open orders |
| `get_positions` | List all positions |

### Market Data

| Command | Description |
|---------|-------------|
| `subscribe <symbol>` | Subscribe to orderbook updates |
| `get_orderbook <symbol>` | Get orderbook snapshot |

### Connection

| Command | Description |
|---------|-------------|
| `auth <key> <secret>` | Authenticate (WebSocket only) |
| `switch <ws\|grpc>` | Switch active protocol |
| `protocol` | Show current protocol |
| `ping` | Test connectivity (gRPC only) |
| `info` | Show node info (gRPC only) |

## Programmatic Usage

Import the client package for programmatic trading:

```go
package main

import (
    "context"
    "fmt"
    "log"

    client "github.com/luxfi/dex/client/go"
)

func main() {
    ctx := context.Background()

    // --- WebSocket Client ---
    wsClient, err := client.NewWsClient("ws://localhost:8081", false)
    if err != nil {
        log.Fatal(err)
    }
    defer wsClient.Close()

    // Wait for connection
    if err := wsClient.WaitConnected(5 * time.Second); err != nil {
        log.Fatal(err)
    }

    // Authenticate
    if err := wsClient.Auth("api_key", "api_secret"); err != nil {
        log.Fatal(err)
    }

    // Place order
    order := &client.Order{
        Symbol: "BTC-USD",
        Side:   "buy",
        Type:   "limit",
        Price:  50000,
        Size:   0.1,
    }
    resp, err := wsClient.PlaceOrder(ctx, order)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Order placed: %d\n", resp.OrderID)

    // --- gRPC Client ---
    grpcClient, err := client.NewGrpcClient("localhost:9090", false)
    if err != nil {
        log.Fatal(err)
    }
    defer grpcClient.Close()

    // Test connectivity
    latency, err := grpcClient.Ping(ctx)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Latency: %v\n", latency)

    // Get positions
    positions, err := grpcClient.GetPositions(ctx)
    if err != nil {
        log.Fatal(err)
    }
    for _, pos := range positions {
        fmt.Printf("Position: %s %.4f @ %.2f\n", pos.Symbol, pos.Size, pos.EntryPrice)
    }
}
```

## Using the Client Interface

Both `WsClient` and `GrpcClient` implement the `Client` interface for protocol-agnostic code:

```go
package main

import (
    "context"
    "log"

    client "github.com/luxfi/dex/client/go"
)

func trade(c client.Client) error {
    ctx := context.Background()

    // Works with either WebSocket or gRPC
    positions, err := c.GetPositions(ctx)
    if err != nil {
        return err
    }

    for _, pos := range positions {
        log.Printf("[%s] Position: %s %.4f", c.Protocol(), pos.Symbol, pos.Size)
    }
    return nil
}

func main() {
    // WebSocket
    ws, _ := client.NewWsClient("ws://localhost:8081", false)
    defer ws.Close()
    trade(ws)

    // gRPC
    grpc, _ := client.NewGrpcClient("localhost:9090", false)
    defer grpc.Close()
    trade(grpc)
}
```

## Client Manager for Multi-Protocol

Use `ClientManager` to manage multiple connections and switch protocols at runtime:

```go
package main

import (
    "context"
    "log"

    client "github.com/luxfi/dex/client/go"
)

func main() {
    ctx := context.Background()
    mgr := client.NewClientManager()

    // Connect both protocols
    mgr.ConnectWs("ws://localhost:8081", false)
    mgr.ConnectGrpc("localhost:9090", false)
    defer mgr.Close()

    // Use WebSocket by default
    orders, _ := mgr.Active().GetOrders(ctx)
    log.Printf("Orders via %s: %d", mgr.Active().Protocol(), len(orders))

    // Switch to gRPC
    mgr.SwitchProtocol(client.ProtocolGRPC)
    orders, _ = mgr.Active().GetOrders(ctx)
    log.Printf("Orders via %s: %d", mgr.Active().Protocol(), len(orders))
}
```

## Protocol Comparison

| Feature | WebSocket | gRPC |
|---------|-----------|------|
| Connection | Persistent | Per-call or streaming |
| Auth | API key/secret | Metadata (extensible) |
| Streaming | Native | Server-side streaming |
| Latency | Low | Very low |
| Best for | Real-time updates | High-throughput RPC |

## Examples

### Place and Monitor Order

```bash
# Place order and watch for updates
./lx-client -i
> subscribe BTC-USD
> place_order BTC-USD buy limit 50000 0.1
> get_orders
```

### High-Frequency Trading Setup

```bash
# Use gRPC for lowest latency
./lx-client -protocol grpc -grpc-addr exchange.lux.network:9090 -v ping
```

### Protocol Switching

```bash
./lx-client -i
> protocol
Active protocol: ws
> switch grpc
Switch failed: grpc client not connected
# Must connect to both protocols for runtime switching
```

## Order Types

- `limit` - Limit order
- `market` - Market order
- `stop` - Stop order
- `stop_limit` - Stop-limit order

## Order Sides

- `buy` - Buy order
- `sell` - Sell order

## Server Endpoints

| Protocol | Default | Environment |
|----------|---------|-------------|
| WebSocket | `ws://localhost:8081` | Development |
| WebSocket | `wss://api.lux.exchange` | Production |
| gRPC | `localhost:9090` | Development |
| gRPC | `grpc.lux.exchange:443` | Production |

## Error Handling

The client returns descriptive errors for common issues:

```go
resp, err := client.PlaceOrder(ctx, order)
if err != nil {
    // Check error type
    switch {
    case strings.Contains(err.Error(), "connection closed"):
        // Reconnect
    case strings.Contains(err.Error(), "timeout"):
        // Retry
    default:
        log.Printf("Order failed: %v", err)
    }
}
```

## Building from Source

```bash
cd /path/to/lx/dex/client/go
go mod tidy
go build -o lx-client .
```

## Testing

```bash
go test -v ./...
```
