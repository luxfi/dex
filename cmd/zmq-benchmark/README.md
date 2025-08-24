# ZMQ Binary FIX Benchmark Suite

High-performance benchmarking tool for testing LX DEX throughput using ZeroMQ and binary FIX protocol.

## Features

- **Binary FIX Protocol**: Compact 60-byte messages for maximum throughput
- **Multi-node Consensus**: Test distributed consensus with BadgerDB persistence
- **Load Generation**: Pummel testing with millions of messages/second
- **Latency Tracking**: Microsecond-precision latency measurements
- **BadgerDB Storage**: Persistent blockchain storage for finalized orders

## Quick Start

```bash
# Build the benchmark
go build -o zmq-benchmark

# Run basic throughput test
./zmq-benchmark -mode producer -rate 1000000 &
./zmq-benchmark -mode consumer

# Run 2-node consensus
# Node 1
./zmq-benchmark -mode consensus -node 1 -endpoint "tcp://0.0.0.0:5555" "tcp://node2:6002"

# Node 2
./zmq-benchmark -mode consensus -node 2 -endpoint "tcp://0.0.0.0:5555" "tcp://node1:6001"
```

## Benchmark Modes

### Producer Mode
Generates synthetic binary FIX orders at specified rate:
```bash
./zmq-benchmark -mode producer \
  -endpoint "tcp://127.0.0.1:5555" \
  -rate 1000000 \              # 1M messages/sec
  -batch 100 \                 # 100 messages per batch
  -producers 4 \               # 4 producer threads
  -duration 60s                # Run for 60 seconds
```

### Consumer Mode
Receives and processes orders, tracks metrics:
```bash
./zmq-benchmark -mode consumer \
  -endpoint "tcp://0.0.0.0:5555" \
  -consumers 4 \               # 4 consumer threads
  -latency \                   # Track latency metrics
  -duration 60s
```

### Consensus Mode
Runs a consensus node with BadgerDB storage:
```bash
./zmq-benchmark -mode consensus \
  -node 1 \                    # Node ID
  -endpoint "tcp://0.0.0.0:5555" \
  tcp://peer1:6001 \           # Peer consensus nodes
  tcp://peer2:6002
```

### Relay Mode
Acts as a proxy/load balancer:
```bash
./zmq-benchmark -mode relay \
  -endpoint "tcp://0.0.0.0:5555"  # Frontend
  # Backend on :5556 automatically
```

## Binary FIX Message Format

```go
type BinaryFIXOrder struct {
    MsgType      uint8     // 1 byte: 'D'=NewOrder, 'F'=Cancel
    Side         uint8     // 1 byte: 1=Buy, 2=Sell
    OrdType      uint8     // 1 byte: 1=Market, 2=Limit
    TimeInForce  uint8     // 1 byte: 0=Day, 1=IOC, 2=FOK, 3=GTC
    Symbol       [8]byte   // 8 bytes: "BTC-USD\0"
    OrderID      uint64    // 8 bytes
    ClOrdID      uint64    // 8 bytes
    Price        uint64    // 8 bytes: Fixed point 8 decimals
    OrderQty     uint64    // 8 bytes: Fixed point 8 decimals
    TransactTime uint64    // 8 bytes: Unix nanos
    Account      uint32    // 4 bytes
    ExecInst     uint32    // 4 bytes
    Checksum     uint32    // 4 bytes: CRC32
}
// Total: 60 bytes
```

## Multi-Node Test Setup

### Lab Configuration
```bash
# Set environment variables
export NODE1_HOST="192.168.1.100"
export NODE2_HOST="192.168.1.101"
export LOAD1_HOST="192.168.1.102"
export LOAD2_HOST="192.168.1.103"

# Run the benchmark suite
cd backend/scripts
./run-zmq-benchmark.sh all
```

### Test Scenarios

#### 1. Basic Throughput Test
Single node throughput measurement:
```bash
./run-zmq-benchmark.sh basic
```
Expected: 1-2M messages/sec on 10 cores

#### 2. Consensus Test
Two-node consensus with persistence:
```bash
./run-zmq-benchmark.sh consensus
```
Expected: 100K consensus rounds/sec

#### 3. Pummel Test
Maximum load from multiple sources:
```bash
./run-zmq-benchmark.sh pummel
```
Expected: 5-10M messages/sec aggregate

#### 4. Latency Test
Measure end-to-end latency:
```bash
./run-zmq-benchmark.sh latency
```
Expected: <10μs average, <100μs P99

## Performance Metrics

### Throughput Metrics
- **Messages Out**: Total messages sent
- **Messages In**: Total messages received
- **Bytes Out/In**: Network bandwidth utilization
- **Trades Executed**: Simulated trade matching rate

### Latency Metrics
- **Average Latency**: Mean message transit time
- **Min Latency**: Best-case latency
- **Max Latency**: Worst-case latency
- **P50/P95/P99**: Latency percentiles

### Consensus Metrics
- **Consensus Rounds**: Blocks finalized per second
- **Block Height**: Current blockchain height
- **DB Size**: BadgerDB storage utilization

## BadgerDB Configuration

Optimized for high throughput:
```go
opts := badger.DefaultOptions(dataDir)
opts.SyncWrites = false          // Async writes
opts.ValueLogFileSize = 1 << 30  // 1GB files
opts.NumMemtables = 5
opts.NumCompactors = 4
opts.BlockCacheSize = 256 << 20  // 256MB cache
```

## Network Requirements

### Bandwidth Calculation
```
60 bytes/message × 1M messages/sec = 60 MB/sec = 480 Mbps
```

### For 100M messages/sec:
```
60 bytes × 100M = 6 GB/sec = 48 Gbps (fits in 100 Gbps)
```

## Expected Results

### Single Machine (10 cores)
- **Throughput**: 2M+ messages/sec
- **Latency**: <1 microsecond average
- **CPU Usage**: ~80% at peak
- **Memory**: <1GB resident

### Two-Node Consensus
- **Throughput**: 500K messages/sec per node
- **Consensus**: 100 rounds/sec
- **Finality**: 50ms
- **Storage**: 100MB/minute to BadgerDB

### Network Saturation (100 Gbps)
- **Theoretical**: 208M messages/sec
- **Practical**: 100M messages/sec
- **With batching**: 200M+ messages/sec

## Troubleshooting

### High Latency
- Check network congestion
- Verify CPU frequency scaling disabled
- Ensure sufficient receive buffers

### Low Throughput
- Increase batch size
- Add more producer/consumer threads
- Check ZMQ high water marks

### BadgerDB Issues
- Ensure sufficient disk space
- Run periodic compaction
- Adjust cache sizes for available RAM

## Integration with LX DEX

This benchmark simulates the production message flow:

1. **Clients** submit orders via ZMQ (producers)
2. **Consensus nodes** receive and batch orders
3. **FPC consensus** finalizes order blocks
4. **BadgerDB** persists finalized blocks
5. **Execution reports** sent back to clients

## Building for Production

```bash
# Optimized build
go build -ldflags="-s -w" \
  -gcflags="-l=4" \
  -o zmq-benchmark

# With race detector (testing only)
go build -race -o zmq-benchmark-race

# Cross-compile for Linux
GOOS=linux GOARCH=amd64 go build -o zmq-benchmark-linux
```

## Contributing

Run benchmarks and submit results:
```bash
# Run full suite
make bench-all > results.txt

# Submit results
gh issue create --title "Benchmark Results: [System Specs]" \
  --body-file results.txt
```

---

Copyright © 2025 Lux Industries Inc. All rights reserved.