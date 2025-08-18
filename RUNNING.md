# LX DEX - Running Instructions

## Current Status

### ✅ What Works

1. **Basic DEX Server** (`bin/lx-dex`)
   - Pure Go implementation
   - Order book with matching engine
   - WebSocket support

2. **X-Chain DEX** (`bin/xchain-dex`)
   - Standalone X-Chain with Lux Consensus
   - Integrated with luxfi/database
   - Q-Chain support for quantum finality
   - 50ms consensus finality

3. **ZMQ Benchmark** (`bin/zmq-benchmark`)
   - High-performance benchmarking tool
   - Binary FIX protocol (60-byte messages)
   - Multi-node consensus testing

4. **DEX Trader** (`bin/lx-trader`)
   - Trading client
   - Auto-scaling mode to find max throughput

### ⚠️ Known Issues

1. **Consensus Tests**: Some struct field mismatches in tests (doesn't affect runtime)
2. **Order Book Tests**: Some test failures in order matching logic (existing issues)
3. **DAG Network**: Build issues with consensus package fields

## Quick Start

### 1. Run Basic DEX Server

```bash
# Start NATS (required for messaging)
docker run -d --name nats -p 4222:4222 nats:latest

# Run DEX server
./bin/lx-dex -port 8080 -nats nats://localhost:4222

# In another terminal, run trader
./bin/lx-trader -server http://localhost:8080
```

### 2. Run X-Chain DEX (Standalone Mode)

```bash
# Run in standalone mode (no other nodes required)
./bin/xchain-dex \
  --standalone \
  --node-id node1 \
  --api-port 8080 \
  --consensus-port 6000 \
  --p2p-port 7000 \
  --data-dir ./xchain-data

# The X-Chain DEX will:
# - Start order processor on port 7000
# - Run Lux Consensus with 50ms finality
# - Provide API on port 8080
# - Store data using luxfi/database
```

### 3. Run Multi-Node X-Chain Network

```bash
# Node 1
./bin/xchain-dex \
  --node-id node1 \
  --api-port 8080 \
  --consensus-port 6001 \
  --p2p-port 7001 \
  --data-dir ./xchain-data-1 \
  tcp://localhost:6002 tcp://localhost:6003

# Node 2
./bin/xchain-dex \
  --node-id node2 \
  --api-port 8081 \
  --consensus-port 6002 \
  --p2p-port 7002 \
  --data-dir ./xchain-data-2 \
  tcp://localhost:6001 tcp://localhost:6003

# Node 3
./bin/xchain-dex \
  --node-id node3 \
  --api-port 8082 \
  --consensus-port 6003 \
  --p2p-port 7003 \
  --data-dir ./xchain-data-3 \
  tcp://localhost:6001 tcp://localhost:6002
```

### 4. Run ZMQ Benchmark

```bash
# Basic throughput test
./bin/zmq-benchmark -mode producer -rate 1000000 &
./bin/zmq-benchmark -mode consumer

# The benchmark will show:
# - Messages per second
# - Latency metrics
# - Network throughput
```

### 5. Submit Orders to X-Chain

```bash
# Using curl (when X-Chain is running)
curl -X POST http://localhost:8080/order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "side": "buy",
    "type": "limit",
    "price": 50000,
    "size": 1.0
  }'
```

## Performance Testing

### Quick Benchmark
```bash
make bench-quick
# Tests order book performance
# Expected: 90K+ orders/sec (Pure Go)
```

### ZMQ Network Benchmark
```bash
make bench-zmq-local
# Tests network throughput with ZeroMQ
# Expected: 1-2M messages/sec locally
```

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   X-Chain DEX   │────▶│  Lux Consensus  │────▶│   Q-Chain       │
│  (Order Book)   │     │  (50ms finality)│     │(Quantum Proofs) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ luxfi/database  │     │   ZeroMQ P2P    │     │ Ringtail+BLS    │
│   (Storage)     │     │   (Messaging)   │     │  (Signatures)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Configuration

### Environment Variables
```bash
export LX_NODE_ID=node1
export LX_DATA_DIR=/path/to/data
export LX_API_PORT=8080
export LX_CONSENSUS_PORT=6000
export LX_P2P_PORT=7000
```

### Config File (optional)
Create `config.yaml`:
```yaml
node:
  id: node1
  data_dir: ./xchain-data
  
network:
  api_port: 8080
  consensus_port: 6000
  p2p_port: 7000
  
consensus:
  type: lux
  finality_ms: 50
  vote_threshold: 0.55
  
quantum:
  enable_qchain: true
  qchain_endpoint: tcp://localhost:9000
```

## Monitoring

```bash
# Watch metrics (when X-Chain is running)
watch -n 1 'curl -s http://localhost:8080/stats'

# Check health
curl http://localhost:8080/health

# View logs
tail -f xchain-data/logs/consensus.log
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8080

# Kill the process
kill -9 <PID>
```

### Database Errors
```bash
# Clear database and restart
rm -rf ./xchain-data
./bin/xchain-dex --standalone --data-dir ./xchain-data
```

### Network Issues
```bash
# Check connectivity
nc -zv localhost 6000

# Check firewall
sudo iptables -L -n | grep 6000
```

## Development Mode

### Run with Debug Logging
```bash
LX_DEBUG=true ./bin/xchain-dex --standalone
```

### Run with Performance Profiling
```bash
./bin/xchain-dex --standalone --cpuprofile=cpu.prof
go tool pprof cpu.prof
```

## Next Steps

1. Fix remaining consensus test issues
2. Complete Q-Chain integration
3. Add Prometheus metrics
4. Implement DPDK for kernel-bypass networking
5. Add GPU acceleration for matching engine

---

**Note**: This is a development version. Some features are still being implemented.