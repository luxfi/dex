# 🌐 Multi-Node Trading Setup Guide

## Overview
The DEX system can run distributed across multiple machines. Traders connect to the server via HTTP (currently), making it easy to scale horizontally.

## Current Architecture (HTTP)
- **Server**: Listens on port 8080 (HTTP)
- **Traders**: Connect via `-server` flag
- **Protocol**: JSON over HTTP (simple but works)

## 🚀 How to Run on Multiple Nodes

### Node 1 - Start Server
```bash
# On your server machine (e.g., 192.168.1.100)
make turbo-server

# Or manually:
backend/bin/turbo-dex -port 8080 -workers 40 -shards 80
```

### Node 2+ - Start Traders
```bash
# On trader machines
backend/bin/dex-trader \
  -server http://192.168.1.100:8080 \
  -workers 10 \
  -batch 1000 \
  -duration 60s
```

## 📡 Network Discovery Options

### 1. Manual Configuration (Current)
Traders connect using the `-server` flag:
```bash
# Each trader specifies the server
dex-trader -server http://server-ip:8080
```

### 2. Environment Variable
Set server location via environment:
```bash
export DEX_SERVER=http://192.168.1.100:8080
make dex-trader
```

### 3. DNS/Service Discovery
Use a hostname instead of IP:
```bash
# Set up DNS entry: dex-server.local -> 192.168.1.100
dex-trader -server http://dex-server.local:8080
```

## 🔥 Maximum Performance Setup

### Server Machine (Beefy Box)
```bash
# Start TURBO server with max settings
backend/bin/turbo-dex \
  -port 8080 \
  -workers 100 \
  -shards 200 \
  -buffer 10000000
```

### Trader Machines (Multiple)
```bash
# Machine 1
backend/bin/dex-trader -server http://server:8080 -workers 20 -batch 1000

# Machine 2
backend/bin/dex-trader -server http://server:8080 -workers 20 -batch 1000

# Machine 3
backend/bin/dex-trader -server http://server:8080 -workers 20 -batch 1000

# ... etc
```

## 📊 Performance Expectations

### Single Machine
- **Server + 1 Trader**: ~30K orders/sec
- **Bottleneck**: HTTP overhead, context switching

### Multi-Machine
- **1 Server + 5 Traders**: ~150K orders/sec
- **1 Server + 10 Traders**: ~300K orders/sec
- **Bottleneck**: Network bandwidth, server CPU

## 🚀 Future: NATS/gRPC/LX

For higher performance, we could add:

### Option 1: NATS
```go
// Server publishes to NATS
nc.Publish("orders", orderData)

// Traders subscribe
nc.Subscribe("orders", func(msg *nats.Msg) {
    // Process order
})
```

### Option 2: gRPC Streaming
```go
// Bidirectional streaming for orders
stream, _ := client.OrderStream(ctx)
stream.Send(&Order{...})
```

### Option 3: LX (Already partially implemented)
```bash
make zmq-server  # Start ZMQ exchange
make zmq-trader  # Start ZMQ traders
```

## 🔧 Quick Test Commands

### Test on Same Machine
```bash
# Terminal 1
make turbo-server

# Terminal 2
make dex-trader
```

### Test Across Network
```bash
# Server (192.168.1.100)
make turbo-server

# Client (any machine)
backend/bin/dex-trader -server http://192.168.1.100:8080
```

### Benchmark Multiple Traders
```bash
# Start server
make turbo-server

# Start 5 traders in parallel (different terminals or machines)
for i in {1..5}; do
  backend/bin/dex-trader -server http://localhost:8080 &
done
```

## 📈 Monitoring

While running distributed:
```bash
# Check server stats
curl http://server:8080/stats | jq .

# Watch real-time
watch -n 1 'curl -s http://server:8080/stats | jq .'
```

## 🎯 Tips for Maximum Throughput

1. **Use Batch Mode**: Send 1000+ orders per batch
2. **More Workers**: Use `-workers 20` or higher
3. **Local Network**: Keep latency < 1ms
4. **Jumbo Frames**: Enable on network for less overhead
5. **CPU Affinity**: Pin server/trader processes to cores

## Example: 10-Node Setup

```bash
# Server (most powerful machine)
./turbo-dex -port 8080 -workers 200 -shards 400

# 9 Trader nodes (run on each)
./dex-trader \
  -server http://10.0.0.2:8080 \
  -workers 20 \
  -batch 2000 \
  -duration 60s
```

Expected: **500K+ orders/sec** across the cluster!

---

**Note**: Current implementation uses HTTP for simplicity. For production, consider NATS, gRPC, or LX for better performance and pub/sub capabilities.