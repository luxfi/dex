# DEX Architecture

## Core Principles
- **DRY**: Don't Repeat Yourself
- **KISS**: Keep It Simple
- **YAGNI**: You Aren't Gonna Need It
- **Single Responsibility**: Each component does ONE thing well

## Directory Structure

```
dex/
├── api/
│   ├── dex.yaml          # OpenAPI spec (source of truth)
│   └── generated/        # ogen-generated server/client
├── cmd/
│   ├── server/           # DEX server (ONE server)
│   ├── trader/           # Trading client (ONE client)
│   └── benchmark/        # Performance testing (ONE benchmark)
├── pkg/
│   ├── orderbook/        # Order matching engine
│   ├── engine/           # Trading engine
│   └── transport/        # Network transports (NATS, ZMQ)
├── internal/
│   └── service/          # Business logic
└── generate.go           # Code generation

```

## Components

### 1. API Layer
- **Source**: OpenAPI spec (`api/dex.yaml`)
- **Generation**: ogen for Go server/client
- **Clients**: OpenAPI Generator for other languages

### 2. Transport Layer
- **NATS**: Service discovery & pub/sub
- **ZeroMQ**: High-throughput point-to-point
- **HTTP**: REST API via ogen

### 3. Engine Layer
- **Pure Go**: Default implementation
- **CGO**: C++ orderbook when `CGO_ENABLED=1`
- **Automatic**: Runtime selection based on build flags

## Commands

```bash
# Generate API
go generate ./...

# Run server
go run cmd/server/main.go

# Run trader
go run cmd/trader/main.go

# Benchmark
go run cmd/benchmark/main.go

# Build optimized
CGO_ENABLED=1 go build ./cmd/server
```

## Performance Targets

| Component | Target | Actual |
|-----------|--------|--------|
| Orderbook | 1M ops/sec | TBD |
| Matching | 100K trades/sec | TBD |
| API | 50K req/sec | TBD |
| Latency | <1ms p99 | TBD |