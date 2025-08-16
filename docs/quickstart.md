# 🚀 LX Engine Quick Start

## Default Command (Builds Everything + Tests + Benchmarks)
```bash
make
# or
make all
```

This will:
1. Build all engines (Go, C++, Hybrid, LX, FIX)
2. Run tests
3. Execute quick benchmark
4. Show best performance results

## Essential Commands

### 🏁 Quick Performance Test
```bash
make bench-quick   # or: make q
```

### 🏆 Find Maximum Performance  
```bash
make bench-max     # or: make m
```

### 🌐 Test Network (10Gbps)
```bash
make bench-network # or: make n
```

### 📄 FIX Protocol Testing
```bash
make bench-fix     # Test with C++ FIX clients
make fix-demo      # See FIX messages in action
```

## What Gets Built

### Engines
- **Pure Go**: 163K orders/sec
- **Hybrid Go/C++**: 180K orders/sec  
- **Pure C++**: 1.3M orders/sec
- **TypeScript**: 50K orders/sec

### Trading Clients
- **C++ FIX Trader**: High-performance FIX client
- **LX Trader**: Network benchmarking client
- **Mega Traders**: Massive concurrent load testing

### Tools
- **FIX Generator**: Creates standard FIX 4.4 messages
- **ZMQ Exchange**: High-throughput message broker
- **Max Perf Bench**: Finds optimal trader count

## Performance Summary

| Implementation | Max Throughput | Use Case |
|----------------|---------------|----------|
| Pure C++ | 1,328,880/sec | HFT, Market Making |
| Hybrid Go/C++ | 180,585/sec | Production DEX |
| Pure Go | 162,969/sec | General Exchange |
| TypeScript | ~50,000/sec | Web Trading |

## Network Testing

### Local Network Test
```bash
make bench-network
```

### Distributed Setup
```bash
# On exchange server:
make setup-exchange

# On trader nodes:
make setup-trader EXCHANGE=10.0.0.10
```

## FIX Protocol

The system now supports standard FIX 4.4 messages:
- New Order Single (D)
- Execution Report (8)  
- Market Data Request (V)
- Cancel Request (F)

### Test FIX Performance
```bash
# 100 C++ traders sending FIX messages
make bench-fix

# Stress test with 10,000 traders
make bench-fix-stress
```

## Shortcuts

- `make q` - Quick benchmark
- `make m` - Max throughput
- `make n` - Network test
- `make f` - Full suite
- `make b` - Build all
- `make c` - Clean

## Architecture

```
C++ FIX Clients ──┐
                  ├──► Trading Engine ──► Order Book
ZMQ Traders ──────┘         │
                           │
                      Matching Engine
```

## Next Steps

1. Run `make` to build and test everything
2. Use `make bench-fix` to test with real FIX messages
3. Use `make bench-network` to test your 10Gbps network
4. Check logs in `/tmp/` for server output

---
For detailed docs, see README.md and NETWORK_BENCHMARK.md
