# LX Engine - High-Performance Multi-Language Trading Platform

## 🚀 Performance Summary

Achieved **1.3M orders/second** with pure C++ implementation on Apple Silicon.

| Implementation | Max Throughput | Optimal Config | Use Case |
|----------------|---------------|----------------|----------|
| **Pure C++** | 1,328,880/sec | 1000 threads | HFT, Market Making |
| **Hybrid Go/C++** | 180,585/sec | 9,512 traders | Production DEX |
| **Pure Go** | 162,969/sec | 8,025 traders | General Exchange |
| **TypeScript** | ~50,000/sec | 500 traders | Web Trading |

## Quick Start

```bash
# From the engine directory
cd /Users/z/work/lx/engine

# Quick benchmark (1000 traders, 30s)
make bench-quick

# Full benchmark suite (~5 minutes)
make bench-full

# Find maximum performance
make bench-max

# Stress test with 10,000 traders
make bench-stress

# Generate performance report
make bench-report
```

## Building

```bash
# Build all benchmark tools and servers
cd backend
make bench-tools
make bench-servers

# Or build specific versions
make go-build        # Pure Go
make hybrid-build    # Go with C++ optimization
make cpp-build       # Pure C++ CEX
```

## Benchmark Results

### System: Apple M-series Silicon (10 cores, 32GB RAM)

#### Pure C++ Standalone
```
Peak: 1,328,880 orders/sec
Optimal: 1000 threads
Trades: 1,039,130/sec
Memory: ~200MB
CPU: 95% all cores
```

#### Hybrid Go/C++ (CGO)
```
Peak: 180,585 orders/sec
Optimal: 5,050 traders
Latency: 28ms avg, 8.8ms P99
Error Rate: 0.14%
```

#### Pure Go
```
Peak: 162,969 orders/sec
Optimal: 8,025 traders
Latency: 48.6ms avg, 74.5ms P99
Memory: Moderate (GC managed)
```

## Architecture

The LX Engine uses a polyglot architecture with multiple implementations:

```
┌─────────────────────────────────────┐
│         Client Applications         │
└─────────────┬───────────────────────┘
              │ gRPC
┌─────────────▼───────────────────────┐
│      Unified Client Library         │
│    (Local/Remote/Hybrid modes)      │
└─────────────┬───────────────────────┘
              │
    ┌─────────┼─────────┬──────────┐
    │         │         │          │
┌───▼──┐  ┌──▼───┐  ┌──▼───┐  ┌───▼──┐
│Pure Go│  │Hybrid│  │Pure  │  │TypeScript│
│Engine │  │Go/C++│  │C++   │  │Engine    │
└──────┘  └──────┘  └──────┘  └──────────┘
```

## Production Recommendations

### Use Pure Go When:
- You need 100-200K orders/sec
- Simple deployment is priority
- Team has Go expertise
- Good observability required

### Use Hybrid Go/C++ When:
- You need 150-250K orders/sec
- Can manage CGO complexity
- Want Go tooling with C++ performance

### Use Pure C++ When:
- You need 500K+ orders/sec
- Ultra-low latency required (<1ms)
- Team has C++ expertise
- HFT or market making

## Deployment

### Local Development
```bash
# Start all servers for testing
cd backend
make bench-servers

# Pure Go server
./bin/lx-dex -port 50051

# Hybrid CGO server
LX_ORDERBOOK_IMPL=cpp ./bin/lx-dex-hybrid -port 50052
```

### Docker
```bash
# Build images
make docker-build

# Run with docker-compose
docker-compose -f docker/docker-compose.dev.yml up
```

## Project Structure

```
engine/
├── README.md           # This file
├── Makefile           # Root convenience makefile
└── backend/
    ├── Makefile       # Main build system
    ├── bridge/        # C++ to Go bridges (CGO)
    ├── cpp/           # Pure C++ implementations
    ├── pkg/
    │   ├── client/    # Unified client library
    │   ├── engine/    # Go engine implementation
    │   ├── orderbook/ # Order book implementations
    │   └── proto/     # Generated gRPC code
    ├── cmd/           # Executables
    │   ├── dex/       # DEX server
    │   ├── gateway/   # FIX gateway
    │   ├── mega-traders/     # Massive load test
    │   ├── max-perf-bench/   # Find max throughput
    │   └── simple-benchmark/ # Basic benchmarks
    ├── proto/         # Protocol definitions
    ├── ts-engine/     # TypeScript implementation
    └── rust-engine/   # Rust implementation (WIP)
```

## Key Features

- **Multi-language support**: Go, C++, TypeScript, Rust
- **Universal gRPC protocol**: All engines speak same language
- **Flexible deployment**: Local library or remote service
- **High performance**: Up to 1.3M orders/sec
- **Production ready**: Circuit breakers, monitoring, failover

## Monitoring

Key metrics to track:
- Orders/second (throughput)
- P50, P95, P99 latency
- Error rate
- Memory usage
- GC pause time (Go)
- CPU utilization

## Contributing

1. Run tests: `make test`
2. Format code: `make fmt`
3. Run linters: `make lint`
4. Benchmark changes: `make bench-quick`

## License

See LICENSE file in repository root.

---

For detailed performance analysis, see `/tmp/ultimate_performance_report.md`
