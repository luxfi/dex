# 🚀 Quick Start: Multi-Node Testing with ZeroMQ

## Prerequisites

1. **Install ZeroMQ:**
   ```bash
   # macOS
   brew install zeromq
   
   # Ubuntu/Debian
   sudo apt-get install libzmq3-dev
   
   # CentOS/RHEL
   sudo yum install zeromq-devel
   ```

2. **Install Go ZMQ bindings:**
   ```bash
   go get github.com/pebbe/zmq4
   ```

## Run Multi-Node Test

### Quick Test (10 seconds, 3 nodes)
```bash
cd backend
./run-multi-node-test.sh
```

### Full Test (30 seconds, custom settings)
```bash
cd backend

# Build binaries
go build -o bin/multi-node ./cmd/multi-node/main.go
go build -o bin/test-client ./cmd/test-client/main.go

# Terminal 1 - Start Node 0 (Leader)
./bin/multi-node --node node0 --port 5000 --peers "tcp://localhost:5100,tcp://localhost:5200" --leader

# Terminal 2 - Start Node 1
./bin/multi-node --node node1 --port 5100 --peers "tcp://localhost:5000,tcp://localhost:5200"

# Terminal 3 - Start Node 2
./bin/multi-node --node node2 --port 5200 --peers "tcp://localhost:5000,tcp://localhost:5100"

# Terminal 4 - Run Test Client
./bin/test-client --nodes "tcp://localhost:5002,tcp://localhost:5102,tcp://localhost:5202" --duration 30s --rate 5000
```

## What It Does

The multi-node test demonstrates:

1. **Distributed Order Processing** - Orders distributed across 3 nodes
2. **ZeroMQ Messaging** - High-performance inter-node communication
3. **State Synchronization** - Nodes sync order book state
4. **Load Balancing** - Client randomly selects nodes
5. **Fault Tolerance** - System continues if a node fails

## Architecture

```
┌─────────────┐
│   Client    │
│  (DEALER)   │
└──────┬──────┘
       │
   Load Balance
       │
┌──────▼──────────────────────────┐
│          ZMQ ROUTER/DEALER      │
├──────────┬──────────┬───────────┤
│  Node 0  │  Node 1  │  Node 2   │
│ (Leader) │          │           │
├──────────┴──────────┴───────────┤
│      ZMQ PUB/SUB Network        │
└──────────────────────────────────┘
```

## Performance Expectations

With 3 nodes you should see:
- **Throughput**: 30,000-50,000 orders/sec
- **Latency P50**: < 1ms
- **Latency P99**: < 10ms

## Monitor Progress

```bash
# Watch all logs
tail -f logs/*.log

# Check metrics
grep "metrics" logs/*.log | tail -3

# Monitor specific node
tail -f logs/node0.log
```

## Docker Alternative

```bash
# Using Docker Compose
docker-compose -f ../docker-compose.multi-node.yml up

# Scale to 5 nodes
docker-compose -f ../docker-compose.multi-node.yml up --scale node=5
```

## Troubleshooting

If you see connection errors:
```bash
# Check if ports are free
lsof -i :5000-5300

# Kill any existing processes
pkill -f multi-node
```

## Next Steps

1. **Increase Load**: Try `--rate 10000` for 10K orders/sec
2. **Add More Nodes**: Run 5+ nodes for higher throughput
3. **Test Fault Tolerance**: Kill a node mid-test
4. **Monitor Metrics**: Set up Prometheus/Grafana

## Files Created

- `cmd/multi-node/main.go` - Multi-node server implementation
- `cmd/test-client/main.go` - Test client for load testing
- `scripts/run-multi-node.sh` - Full test automation script
- `run-multi-node-test.sh` - Quick test script
- `docs/multi-node-testing.md` - Detailed documentation
- `docker-compose.multi-node.yml` - Docker setup
- `Dockerfile.multi-node` - Docker build file