# 🚀 High-Performance Trading System Guide

## Overview
The LX Engine now supports multiple high-performance messaging systems for distributed trading:
- **HTTP** - Simple, works everywhere (30K orders/sec)
- **LX** - High-throughput messaging (80K+ orders/sec)
- **NATS** - Auto-discovery pub/sub (50K+ orders/sec)
- **C++ Implementations** - Ultra-fast (100K+ orders/sec)

## 🔥 Performance Summary

| System | Protocol | Performance | Auto-Discovery | Use Case |
|--------|----------|------------|----------------|----------|
| **Ultra-FIX C++** | In-Process | **5.21M msgs/sec** | N/A | Benchmarking |
| **C++ LX** | ZMQ | **500K+ orders/sec** | No (need IP) | Ultra-low latency |
| **Go LX** | ZMQ | **80K orders/sec** | No (need IP) | High throughput |
| **NATS System** | NATS | **50K+ orders/sec** | **Yes!** | Auto-discovery |
| **HTTP Turbo** | HTTP | **30K orders/sec** | No (need IP) | Simple setup |

## 📡 LX Setup (High Performance)

### Start ZMQ Server
```bash
# Terminal 1 - Server
make zmq-server
# Runs on port 5555
```

### Start ZMQ Traders
```bash
# Terminal 2 - Go Trader (80K orders/sec)
make zmq-trader

# OR for C++ Ultra-Fast Trader (500K+ orders/sec)
make zmq-cpp-trader
```

### Multi-Node ZMQ
```bash
# Server (192.168.1.100)
backend/bin/zmq-exchange -bind tcp://*:5555 -workers 20

# Trader Nodes
backend/bin/zmq-trader -server tcp://192.168.1.100:5555 -traders 100 -rate 1000
```

## 🔌 NATS Setup (Auto-Discovery!)

### Install NATS Server
```bash
# Install if needed
go install github.com/nats-io/nats-server/v2@latest
```

### Start NATS System
```bash
# Terminal 1 - NATS Message Broker
make nats-server

# Terminal 2 - DEX Server (auto-discoverable)
make nats-dex

# Terminal 3 - Traders (auto-find server!)
make nats-trader
```

### How NATS Auto-Discovery Works
1. **Server announces** on `dex.announce` channel
2. **Traders listen** for announcements
3. **Auto-connect** when server found
4. **No IP needed!** Just NATS URL

### Multi-Node NATS
```bash
# All nodes just need NATS server address
nats-trader -nats nats://nats-server.local:4222
```

## ⚡ C++ High-Performance

### Build C++ ZMQ Trader
```bash
# Build
cd backend
g++ -std=c++17 -O3 -march=native -pthread cpp/zmq_turbo_trader.cpp -lzmq -o bin/zmq-cpp-trader

# Run
./bin/zmq-cpp-trader tcp://localhost:5555 40 10000 30
#                     ^server          ^traders ^rate ^duration
```

### Expected Performance
- **Single Machine**: 200K-500K orders/sec
- **Distributed**: 1M+ orders/sec possible

## 🎯 Quick Benchmarks

### LX Benchmark
```bash
make zmq-cpp-bench
# Runs server + C++ trader automatically
# Expected: 200K+ orders/sec
```

### NATS Benchmark
```bash
make nats-bench
# Auto-discovery test
# Expected: 50K+ orders/sec
```

### HTTP Turbo Benchmark
```bash
make turbo-bench
# HTTP with max CPU usage
# Expected: 30K+ orders/sec
```

## 🌐 Multi-Machine Setup

### Option 1: LX (Fastest)
```bash
# Machine 1 (Server)
./zmq-exchange -bind tcp://*:5555 -workers 50

# Machine 2-10 (Traders)
./zmq-cpp-trader tcp://server-ip:5555 100 10000 60
```

### Option 2: NATS (Auto-Discovery)
```bash
# Machine 1 (NATS + DEX)
nats-server -p 4222
./nats-dex -nats nats://localhost:4222

# Machine 2-10 (Traders - auto-find server!)
./nats-trader -nats nats://machine1:4222 -traders 50
```

### Option 3: HTTP (Simplest)
```bash
# Machine 1 (Server)
./turbo-dex -port 8080 -workers 100

# Machine 2-10 (Traders)
./dex-trader -server http://machine1:8080 -workers 20 -batch 1000
```

## 📊 Performance Tuning

### For Maximum Throughput
1. **Use C++ implementations** when possible
2. **Batch orders** (1000+ per batch)
3. **Multiple workers per CPU core** (2-10x)
4. **Jumbo frames** on network (9000 MTU)
5. **CPU affinity** - pin processes to cores
6. **Disable power saving** - performance governor

### Network Optimization
```bash
# Increase socket buffers (Linux)
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
```

## 🔥 Expected Performance by Setup

### Single Machine
- **Ultra-FIX C++**: 5.21M msgs/sec (in-process)
- **C++ ZMQ**: 500K orders/sec
- **Go ZMQ**: 80K orders/sec
- **NATS**: 50K orders/sec
- **HTTP**: 30K orders/sec

### 10-Machine Cluster
- **C++ ZMQ**: 2-5M orders/sec
- **Go ZMQ**: 500K-1M orders/sec
- **NATS**: 300-500K orders/sec
- **HTTP**: 200-300K orders/sec

## 🎯 Which to Use?

### Use LX When:
- Need absolute maximum performance
- Point-to-point connections are OK
- Can manage IP addresses manually

### Use NATS When:
- Need auto-discovery
- Want pub/sub patterns
- Multiple services need to communicate
- Easier cluster management

### Use HTTP When:
- Simple setup needed
- Behind firewalls/proxies
- Compatibility is important

## 📝 Testing Commands

```bash
# Test everything quickly
make zmq-cpp-bench   # Test C++ ZMQ
make nats-bench       # Test NATS auto-discovery
make turbo-bench      # Test HTTP turbo

# See all options
make help
```

---

**Current Record**: 5.21M messages/sec with Ultra-FIX C++ engine!
**Network Record**: 80K orders/sec with LX (can go higher with C++)