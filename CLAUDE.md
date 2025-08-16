# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LX DEX is a high-performance, multi-language trading platform achieving **1.3M orders/second** with pure C++. It implements a polyglot architecture with five distinct engine implementations (Go, C++, Hybrid Go/C++, TypeScript, Rust) unified under a common gRPC interface.

## Quick Start Commands

### Build Commands
```bash
# Build all binaries (default Pure Go)
make build

# Build with different engines
CGO_ENABLED=0 make build    # Pure Go
CGO_ENABLED=1 make build    # Hybrid Go/C++ 

# Backend-specific builds
cd backend
make go-build        # Pure Go engine
make hybrid-build    # Go with C++ optimization
make cpp-build       # Pure C++ CEX
make typescript-build # TypeScript engine
make rust-build      # Rust engine
```

### Running the System
```bash
# Run DEX server
make server

# Run trader client  
make trader          # Normal mode
make trader-auto     # Auto-scaling mode

# Docker deployment
docker-compose up    # Full multi-engine environment
```

### Testing Commands
```bash
# Run all tests
make test

# Test with coverage
make test-coverage

# Test with race detection
cd backend && go test -race -v ./...

# Run specific test
cd backend && go test -v -run TestOrderBook ./pkg/orderbook
```

### Benchmarking Commands
```bash
# Quick benchmark (1000 traders, 30s)
cd backend && make bench-quick

# Full benchmark suite (~5 minutes)
cd backend && make bench-full

# Find maximum performance
cd backend && make bench-max

# Stress test (10,000 traders)
cd backend && make bench-stress

# Compare Go vs C++ performance
make bench-compare
```

### Development Commands
```bash
# Format code
cd backend && go fmt ./...

# Run linter
cd backend && golangci-lint run

# Generate protobuf
cd backend && make proto-gen

# Clean build artifacts
make clean

# Run CI tests locally
make ci
```

## Architecture Overview

### Multi-Engine Design

The project implements five distinct trading engines, all implementing the same gRPC interface:

1. **Pure Go** (`CGO_ENABLED=0`): Simple, maintainable, 162K orders/sec
2. **Hybrid Go/C++** (`CGO_ENABLED=1`): Balanced performance, 180K orders/sec  
3. **Pure C++**: Ultra-low latency, 1.3M orders/sec
4. **TypeScript**: Browser/Node.js compatible, 50K orders/sec
5. **Rust**: Memory-safe alternative, 450K orders/sec

### Key Technologies

- **Languages**: Go 1.23+, C++17, TypeScript, Rust
- **Messaging**: NATS, ZeroMQ for high-throughput
- **Protocols**: gRPC, Protocol Buffers, FIX
- **Performance**: Fixed-point arithmetic (7 decimals), lock-free data structures
- **Monitoring**: Prometheus, Grafana

### Directory Structure

```
backend/
├── bridge/          # C++ to Go CGO bridges
├── cmd/             # Executables (dex, gateway, benchmarks)
├── cpp/             # Pure C++ implementations
├── pkg/
│   ├── client/      # Unified client library
│   ├── engine/      # Go engine implementation
│   ├── orderbook/   # Order book implementations
│   ├── fix/         # FIX protocol support
│   └── proto/       # Generated gRPC code
├── proto/           # Protocol buffer definitions
├── rust-engine/     # Rust implementation
└── ts-engine/       # TypeScript implementation
```

## Engine Selection Strategy

### When to Use Each Engine

**Pure Go**:
- Development and testing
- Need 100-200K orders/sec
- Simple deployment required
- Good observability needed

**Hybrid Go/C++**:
- Production DEX on Lux X-Chain
- Need 150-250K orders/sec
- Balance performance and maintainability
- Can manage CGO complexity

**Pure C++**:
- CEX backend or HFT
- Need 500K+ orders/sec
- Ultra-low latency (<100μs)
- Team has C++ expertise

**TypeScript**:
- Browser/Edge execution
- Web trading interfaces
- Need 50K orders/sec
- JavaScript ecosystem integration

## Performance Characteristics

| Engine | Throughput | Latency | Memory | Use Case |
|--------|------------|---------|--------|----------|
| Pure C++ | 1,328,880/sec | <100μs | ~200MB | HFT, Market Making |
| Hybrid Go/C++ | 180,585/sec | <200μs | Moderate | Production DEX |
| Pure Go | 162,969/sec | <1ms | Moderate | Development |
| TypeScript | ~50,000/sec | <5ms | High | Web Trading |

## Protocol Definitions

All engines implement the same gRPC interface defined in `backend/proto/lx_engine.proto`:

- `SubmitOrder`: Submit new order
- `CancelOrder`: Cancel existing order
- `GetOrderBook`: Get current order book state
- `StreamOrderBook`: Stream real-time updates

## Client Library Usage

The unified client library supports multiple modes:

```go
// Remote engine via gRPC
client, _ := NewLXClient(ClientConfig{
    Mode: ModeRemote,
    ServerAddress: "engine.lx.com:50051",
})

// Local embedded engine
client, _ := NewLXClient(ClientConfig{
    Mode: ModeLocal,
    EngineConfig: engine.EngineConfig{
        OrderBookImpl: orderbook.ImplCpp,
    },
})
```

## Environment Variables

- `CGO_ENABLED`: Control CGO (0=Pure Go, 1=Hybrid)
- `LX_ORDERBOOK_IMPL`: Select orderbook implementation (go/cpp)
- `LX_ENGINE_PORT`: Server port (default: 50051)
- `LX_METRICS_PORT`: Prometheus metrics port (default: 9090)

## Docker Deployment

```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production with all engines
docker-compose up

# Build specific engine images
docker build -f Dockerfile.go -t lx-dex:go .
docker build -f Dockerfile.hybrid -t lx-dex:hybrid .
docker build -f Dockerfile.cpp -t lx-dex:cpp .
```

## Integration with Lux Blockchain

For DEX operations on X-Chain:
- Orders matched off-chain for speed
- Settlement happens on-chain
- State channels batch trades
- Validators run embedded engines

## Common Development Tasks

### Adding a New Order Type

1. Update proto definition in `backend/proto/lx_engine.proto`
2. Regenerate code: `cd backend && make proto-gen`
3. Implement in `backend/pkg/orderbook/orderbook.go`
4. Add C++ implementation in `backend/bridge/orderbook_bridge.cpp`
5. Update tests in `backend/pkg/orderbook/orderbook_test.go`
6. Run benchmarks to verify performance

### Debugging Performance Issues

1. Run profiling: `go test -cpuprofile cpu.prof -bench=.`
2. Analyze: `go tool pprof cpu.prof`
3. Check memory: `go test -memprofile mem.prof -bench=.`
4. For C++: Use `perf record` and `perf report`

### Updating Dependencies

```bash
cd backend
go mod tidy
go mod verify
```

## Security Considerations

- All engines validate input with fixed-point arithmetic
- No floating point in critical paths
- Rate limiting implemented at gateway
- TLS/mTLS for production gRPC

## Monitoring and Observability

- Prometheus metrics on `:9090`
- Key metrics: orders/sec, latency percentiles, error rate
- Grafana dashboards in `docker/grafana/`
- Health checks on `/health` endpoint

## Testing Strategy

1. **Unit tests**: Core logic in isolation
2. **Integration tests**: Engine interaction
3. **Benchmark tests**: Performance validation
4. **Stress tests**: 10,000+ concurrent traders
5. **Race detection**: `go test -race`

## Deployment Recommendations

- **Development**: Pure Go for simplicity
- **Testing**: Run bench-compare to validate
- **Production DEX**: Hybrid Go/C++ with monitoring
- **Production CEX**: Pure C++ with redundancy
- **Web**: TypeScript engine or gRPC-Web

## Known Optimizations

- Lock-free order book for high concurrency
- Fixed-point arithmetic (7 decimals) for precision
- Buffer pools for zero-copy networking
- TCP_NODELAY for low latency
- Batch matching to reduce contention

## Troubleshooting

- Check server logs: `tail -f /tmp/go-server.log`
- Verify ports: `lsof -i :50051`
- Test connectivity: `grpcurl -plaintext localhost:50051 list`
- Profile CPU: `go tool pprof http://localhost:6060/debug/pprof/profile`