# Multi-Node Testing with ZeroMQ

This guide explains how to run multi-node testing for the LX DEX using ZeroMQ for high-performance message passing.

## Architecture

The multi-node setup uses ZeroMQ's advanced messaging patterns:

- **PUB/SUB**: For broadcasting events between nodes
- **ROUTER/DEALER**: For load-balanced request/reply patterns
- **Push/PULL**: For distributed work queues (optional)

Each node maintains:
- Local order book
- Event broadcasting to peers
- State synchronization
- Leader election (first node is leader)

## Prerequisites

### Install ZeroMQ

**macOS:**
```bash
brew install zeromq
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libzmq3-dev
```

**CentOS/RHEL:**
```bash
sudo yum install -y zeromq-devel
```

### Install Go ZMQ Bindings
```bash
go get -u github.com/pebbe/zmq4
```

## Running Multi-Node Tests

### Method 1: Using the Shell Script (Recommended)

```bash
# Make script executable
chmod +x backend/scripts/run-multi-node.sh

# Run with default settings (3 nodes, 1000 orders/sec, 30s duration)
./backend/scripts/run-multi-node.sh

# Custom configuration
NODES=5 ORDERS_PER_SEC=5000 TEST_DURATION=60s ./backend/scripts/run-multi-node.sh

# Keep nodes running after test
KEEP_ALIVE=true ./backend/scripts/run-multi-node.sh
```

### Method 2: Manual Setup

#### Step 1: Build the binaries
```bash
cd backend

# Build multi-node server
go build -o bin/multi-node ./cmd/multi-node/main.go

# Build test client
go build -o bin/test-client ./cmd/test-client/main.go
```

#### Step 2: Start Node 0 (Leader)
```bash
./bin/multi-node \
  --node node0 \
  --port 5000 \
  --peers "tcp://localhost:5100,tcp://localhost:5200" \
  --leader
```

#### Step 3: Start Node 1
```bash
./bin/multi-node \
  --node node1 \
  --port 5100 \
  --peers "tcp://localhost:5000,tcp://localhost:5200"
```

#### Step 4: Start Node 2
```bash
./bin/multi-node \
  --node node2 \
  --port 5200 \
  --peers "tcp://localhost:5000,tcp://localhost:5100"
```

#### Step 5: Run Test Client
```bash
./bin/test-client \
  --nodes "tcp://localhost:5002,tcp://localhost:5102,tcp://localhost:5202" \
  --duration 30s \
  --rate 1000
```

### Method 3: Using Docker Compose

```bash
# Build and start all nodes
docker-compose -f docker-compose.multi-node.yml up --build

# Scale to more nodes
docker-compose -f docker-compose.multi-node.yml up --scale node=5

# View logs
docker-compose -f docker-compose.multi-node.yml logs -f

# Stop all nodes
docker-compose -f docker-compose.multi-node.yml down
```

## Port Configuration

Each node uses 4 ports:
- **Base Port + 0**: PUB socket (event broadcasting)
- **Base Port + 1**: SUB socket (event receiving)
- **Base Port + 2**: ROUTER socket (order receiving)
- **Base Port + 3**: DEALER socket (order sending)

Example for 3 nodes:
- Node 0: 5000-5003
- Node 1: 5100-5103
- Node 2: 5200-5203

## Message Flow

1. **Order Submission**: Client sends order to any node via DEALER/ROUTER
2. **Order Matching**: Node processes order locally
3. **Trade Broadcasting**: Executed trades broadcast via PUB/SUB
4. **State Sync**: Leader periodically broadcasts full state
5. **Heartbeat**: Nodes send periodic heartbeats for health monitoring

## Performance Tuning

### ZeroMQ Socket Options
```go
// Increase high water mark for better throughput
socket.SetSndhwm(10000)
socket.SetRcvhwm(10000)

// Set linger to avoid message loss on shutdown
socket.SetLinger(1000) // 1 second

// Enable TCP keepalive
socket.SetTcpKeepalive(1)
socket.SetTcpKeepaliveIdle(300)
```

### System Tuning
```bash
# Increase file descriptor limits
ulimit -n 65536

# Increase socket buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

## Monitoring

### View Node Metrics
```bash
# Check node logs
tail -f logs/node0.log
tail -f logs/node1.log
tail -f logs/node2.log

# Monitor in real-time
watch -n 1 'grep "metrics" logs/*.log | tail -n 3'
```

### Metrics Available
- Orders processed per node
- Trades executed per node
- Messages sent/received
- Latency measurements
- Throughput (orders/sec)

## Testing Scenarios

### High Throughput Test
```bash
# 10,000 orders/sec across 5 nodes
NODES=5 ORDERS_PER_SEC=10000 TEST_DURATION=60s ./scripts/run-multi-node.sh
```

### Fault Tolerance Test
```bash
# Start nodes
./scripts/run-multi-node.sh &

# Kill a node mid-test
kill $(pgrep -f "node1")

# Observe recovery and continued operation
```

### Network Partition Test
```bash
# Use iptables to simulate network partition
sudo iptables -A INPUT -p tcp --dport 5100 -j DROP

# Remove partition
sudo iptables -D INPUT -p tcp --dport 5100 -j DROP
```

## Expected Performance

With proper tuning, the multi-node setup should achieve:

- **3 Nodes**: 50,000+ orders/sec aggregate
- **5 Nodes**: 100,000+ orders/sec aggregate
- **10 Nodes**: 200,000+ orders/sec aggregate

Latency:
- **P50**: < 1ms
- **P99**: < 10ms
- **P99.9**: < 50ms

## Troubleshooting

### ZMQ Library Not Found
```bash
# Check if ZMQ is installed
pkg-config --modversion libzmq

# Set PKG_CONFIG_PATH if needed
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### Port Already in Use
```bash
# Find process using port
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Connection Refused
- Check firewall settings
- Ensure all nodes are started
- Verify correct peer addresses

### High CPU Usage
- Reduce order rate
- Enable batch processing
- Profile with pprof

## Advanced Configuration

### Custom Network Topology
```go
// Star topology (all nodes connect to leader)
config := &NodeConfig{
    NodeID: "node1",
    Peers: []string{"tcp://leader:5000"},
}

// Mesh topology (all nodes connect to all)
config := &NodeConfig{
    NodeID: "node1",
    Peers: []string{
        "tcp://node0:5000",
        "tcp://node2:5200",
        "tcp://node3:5300",
    },
}

// Ring topology (each node connects to neighbors)
config := &NodeConfig{
    NodeID: "node1",
    Peers: []string{
        "tcp://node0:5000",
        "tcp://node2:5200",
    },
}
```

### Consensus Mechanisms

The system supports different consensus modes:

1. **Leader-based**: First node is leader, broadcasts authoritative state
2. **Byzantine**: Requires 2/3 agreement for trade execution
3. **Raft**: Leader election with log replication
4. **PBFT**: Practical Byzantine Fault Tolerance

Configure in node startup:
```bash
./bin/multi-node --consensus raft --node node0 --port 5000
```

## Integration with X-Chain

For production deployment on Lux X-Chain:

1. Each validator runs a node
2. Orders settled on-chain after matching
3. State channels for batch settlement
4. Cross-chain bridge for asset transfers

See [X-Chain Integration Guide](./x-chain-integration.md) for details.