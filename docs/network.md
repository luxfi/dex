# 🌐 Network Benchmarking Guide - ZeroMQ Distributed Trading

## Overview

Test real network performance on your 10Gbps network using ZeroMQ for ultra-fast message passing between trading nodes and exchange servers.

## Quick Start

### Local Test (Same Machine)
```bash
cd /Users/z/work/lx/engine/backend
make bench-zmq-local
```

### Distributed Test (Multiple Machines)
```bash
# Get instructions
make bench-zmq-dist

# Or run manually - see below
```

### Network Saturation Test
```bash
# Test how many orders/sec to saturate 10Gbps
make bench-network
```

## Architecture

```
┌─────────────────┐      10Gbps Network      ┌─────────────────┐
│  Trader Node 1  │◄──────────────────────────►│                 │
│  (zmq-trader)   │                            │                 │
└─────────────────┘                            │                 │
                                               │                 │
┌─────────────────┐                            │    Exchange     │
│  Trader Node 2  │◄──────────────────────────►│     Server      │
│  (zmq-trader)   │          ZeroMQ            │  (zmq-exchange) │
└─────────────────┘          PUSH/PULL         │                 │
                                               │                 │
┌─────────────────┐                            │                 │
│  Trader Node N  │◄──────────────────────────►│                 │
│  (zmq-trader)   │                            └─────────────────┘
└─────────────────┘
```

## Distributed Deployment

### Step 1: Build on All Nodes
```bash
# On each machine
cd /Users/z/work/lx/engine/backend
go get github.com/pebbe/zmq4
go build -o zmq-exchange ./cmd/zmq-exchange
go build -o zmq-trader ./cmd/zmq-trader
```

### Step 2: Start Exchange Server
```bash
# On exchange node (e.g., 10.0.0.10)
./zmq-exchange -bind 'tcp://*:5555'
```

### Step 3: Start Trader Nodes
```bash
# On trader node 1 (e.g., 10.0.0.11)
./zmq-trader -server 'tcp://10.0.0.10:5555' -traders 1000 -rate 100 -duration 60s

# On trader node 2 (e.g., 10.0.0.12)
./zmq-trader -server 'tcp://10.0.0.10:5555' -traders 1000 -rate 100 -duration 60s

# On trader node 3 (e.g., 10.0.0.13)
./zmq-trader -server 'tcp://10.0.0.10:5555' -traders 1000 -rate 100 -duration 60s
```

## Network Tuning

### Linux TCP Buffer Tuning
```bash
# Run on all nodes for maximum performance
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 134217728'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 134217728'
sudo sysctl -w net.core.netdev_max_backlog=30000
sudo sysctl -w net.ipv4.tcp_congestion_control=htcp
sudo sysctl -w net.ipv4.tcp_mtu_probing=1
```

### macOS Tuning
```bash
# Increase socket buffer sizes
sudo sysctl -w kern.ipc.maxsockbuf=16777216
sudo sysctl -w net.inet.tcp.sendspace=1048576
sudo sysctl -w net.inet.tcp.recvspace=1048576
```

## Performance Expectations

### Message Size vs Throughput (10Gbps)

| Message Size | Theoretical Max | Realistic | Orders/sec |
|--------------|----------------|-----------|------------|
| 50 bytes | 25M msgs/sec | 10M msgs/sec | 10M |
| 100 bytes | 12.5M msgs/sec | 5M msgs/sec | 5M |
| 200 bytes | 6.25M msgs/sec | 2.5M msgs/sec | 2.5M |
| 500 bytes | 2.5M msgs/sec | 1M msgs/sec | 1M |
| 1KB | 1.25M msgs/sec | 500K msgs/sec | 500K |

### Scaling Guidelines

To saturate 10Gbps network:

1. **Small Messages (100 bytes)**
   - Need ~50,000 concurrent traders
   - Each doing 100 orders/sec
   - Total: 5M orders/sec

2. **Medium Messages (500 bytes)**
   - Need ~10,000 concurrent traders  
   - Each doing 100 orders/sec
   - Total: 1M orders/sec

3. **With Batching**
   - Batch 10 orders per message
   - Need only 5,000 traders
   - Can reach 5M logical orders/sec

## Advanced Configurations

### Multiple Exchange Servers (Sharding)
```bash
# Exchange 1 - handles BTC, ETH
./zmq-exchange -bind 'tcp://*:5555' -symbols BTC,ETH

# Exchange 2 - handles SOL, AVAX
./zmq-exchange -bind 'tcp://*:5556' -symbols SOL,AVAX

# Traders connect to appropriate exchange
./zmq-trader -server 'tcp://10.0.0.10:5555' -symbols BTC,ETH
./zmq-trader -server 'tcp://10.0.0.10:5556' -symbols SOL,AVAX
```

### Message Batching
```bash
# Enable batching for higher throughput
./zmq-trader -server 'tcp://10.0.0.10:5555' -batch 10 -batch-timeout 1ms
```

### Monitoring
```bash
# Monitor network usage
iftop -i eth0

# Monitor ZeroMQ stats
./zmq-exchange -bind 'tcp://*:5555' -stats-port 8080

# Check in browser
curl http://localhost:8080/stats
```

## Benchmarking Checklist

- [ ] Build ZMQ tools on all nodes
- [ ] Tune TCP buffers on all nodes
- [ ] Start exchange server
- [ ] Start trader nodes
- [ ] Monitor network usage (should see ~10Gbps)
- [ ] Monitor CPU usage (should not be bottleneck)
- [ ] Check error rates (should be <0.1%)
- [ ] Test different message sizes
- [ ] Test with batching
- [ ] Test with multiple exchanges

## Troubleshooting

### Low Performance (<100K msgs/sec)
- Check TCP buffer sizes
- Verify network is 10Gbps (not 1Gbps)
- Check CPU usage (might be CPU bound)
- Try fewer traders with higher rate

### High Error Rate (>1%)
- Reduce send rate
- Increase ZMQ high water mark
- Check network packet loss

### Can't Connect
- Check firewall rules
- Verify IP addresses
- Test with telnet first

## Results Interpretation

### Good Performance
```
📊 Orders: 5000000 | Rate: 500000/sec | Network: 95.37 MB/s (0.763 Gbps)
```
- Using 7.6% of 10Gbps
- Room for 10x more traffic

### Saturated Network
```
📊 Orders: 50000000 | Rate: 5000000/sec | Network: 1192.09 MB/s (9.537 Gbps)
```
- Using 95% of 10Gbps
- Network is bottleneck
- Consider compression or batching

## Next Steps

1. **Test with real order book processing**
   - Replace counter with actual matching engine
   - Measure end-to-end latency

2. **Test with market data distribution**
   - Add PUB/SUB for market data
   - Measure fan-out performance

3. **Test failover scenarios**
   - Kill exchange, measure recovery
   - Test with backup exchanges

4. **Compare with gRPC**
   - Run same test with gRPC
   - Compare throughput and latency

---

For questions or issues, check the main README or run `make help`
