# LX Backend - Multi-Engine Trading System

High-performance trading engine system with multiple language implementations for both DEX (on X-Chain) and CEX deployments.

## Architecture

The LX Backend provides multiple engine implementations with a unified gRPC interface:

- **Pure Go Engine** - Fast development, good performance, easy deployment
- **Hybrid Go/C++ Engine** - Go wrapper with C++ core for critical paths (CGO=1)
- **Pure C++ Engine** - Maximum performance for CEX deployments
- **TypeScript Engine** - Browser/Node.js compatible for web integrations
- **Rust Engine** - Memory-safe, high-performance alternative

All engines implement the same gRPC protocol, allowing seamless switching based on deployment requirements.

## Quick Start

### Development Environment

```bash
# Start development environment with single hybrid engine
make dev

# Or use Docker Compose
docker-compose -f docker-compose.dev.yml up
```

### Building All Engines

```bash
# Build all engine implementations
make all

# Build specific engines
make go-build        # Pure Go
make hybrid-build    # Go/C++ with CGO
make cpp-build       # Pure C++
make typescript-build # TypeScript
make rust-build      # Rust
```

### Running Benchmarks

```bash
# Download FIX test data
./scripts/download-fix-data.sh

# Run comprehensive benchmarks
make benchmark-all

# Compare engines
./scripts/run-comprehensive-benchmark.sh
```

## Docker Deployment

### Full Stack

```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Available Services

- **go-engine** (port 50051) - Pure Go engine
- **hybrid-engine** (port 50052) - Hybrid Go/C++ engine
- **cpp-engine** (port 50053) - Pure C++ engine
- **rust-engine** (port 50054) - Rust engine
- **ts-engine** (port 50055) - TypeScript engine
- **engine-router** (port 50050) - Load balancer/router
- **fix-gateway** (port 9878) - FIX protocol gateway
- **prometheus** (port 9090) - Metrics collection
- **grafana** (port 3000) - Metrics visualization

## Performance Characteristics

| Engine | Throughput | Latency (p99) | Memory | Startup Time | Dev Velocity |
|--------|------------|---------------|--------|--------------|--------------|
| Pure Go | 500K ops/s | 5ms | 200MB | 1s | Excellent |
| Hybrid Go/C++ | 1M ops/s | 2ms | 150MB | 2s | Good |
| Pure C++ | 2M ops/s | 500μs | 100MB | 1s | Moderate |
| Rust | 1.5M ops/s | 1ms | 120MB | 1s | Good |
| TypeScript | 100K ops/s | 20ms | 300MB | 3s | Excellent |

## Client Usage

### Unified Client Library

```go
import "github.com/lux/lx-backend/pkg/client"

// Create client (auto-detects local vs remote)
client, err := client.New(client.Config{
    Mode: client.ModeHybrid,  // Try local first, fallback to remote
    LocalEngine: "cpp",        // Use C++ engine locally
    RemoteEndpoint: "engine-router:50050",
})

// Submit order (works same for all engines)
order, trades, err := client.SubmitOrder(ctx, &SubmitOrderRequest{
    Symbol: "BTC-USD",
    Side: "BUY",
    Type: "LIMIT",
    Price: 50000.00,
    Quantity: 1.5,
})
```

## FIX Gateway

The FIX gateway provides standard FIX 4.4 protocol support:

```bash
# Connect to FIX gateway
telnet localhost 9878

# Or use your FIX client library
fix_client.Connect("localhost:9878")
```

## Monitoring

### Prometheus Metrics

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Prometheus
open http://localhost:9090

# Access Grafana (admin/admin)
open http://localhost:3000
```

### Available Metrics

- `lx_orders_submitted_total` - Total orders submitted
- `lx_trades_executed_total` - Total trades executed
- `lx_orderbook_depth` - Current orderbook depth
- `lx_matching_latency_seconds` - Order matching latency
- `lx_grpc_request_duration_seconds` - gRPC request duration

## Development

### Adding New Features

1. Update proto definition in `proto/lx_engine.proto`
2. Regenerate code: `make proto-gen`
3. Implement in each engine
4. Add tests
5. Update benchmarks

### Testing

```bash
# Run all tests
make test

# Test specific engine
CGO_ENABLED=0 go test ./...  # Pure Go
CGO_ENABLED=1 go test ./...  # Hybrid
cd rust-engine && cargo test # Rust
```

### Code Quality

```bash
# Format code
make fmt

# Run linters
make lint

# Run security scan
gosec ./...
```

## Configuration

### Environment Variables

- `CGO_ENABLED` - Enable/disable CGO (0 or 1)
- `ENGINE_PORT` - gRPC server port
- `LOG_LEVEL` - Logging level (debug, info, warn, error)
- `FIX_PORT` - FIX gateway port
- `METRICS_PORT` - Prometheus metrics port

### Config Files

See `configs/` directory for:
- `engine.yaml` - Engine configuration
- `fix.yaml` - FIX gateway settings
- `router.yaml` - Load balancer rules

## Deployment

### X-Chain DEX

```bash
# Use hybrid engine for optimal performance/development balance
CGO_ENABLED=1 ENGINE=hybrid make deploy-dex
```

### CEX

```bash
# Use pure C++ for maximum performance
ENGINE=cpp make deploy-cex
```

### Cloud

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Scale engines
kubectl scale deployment lx-engine --replicas=10
```

## Troubleshooting

### Common Issues

1. **CGO build fails**
   ```bash
   # Install C++ dependencies
   apt-get install g++ libboost-dev
   ```

2. **Rust build fails**
   ```bash
   # Update Rust
   rustup update
   ```

3. **TypeScript build fails**
   ```bash
   # Clear cache and reinstall
   cd ts-engine && rm -rf node_modules && npm install
   ```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=debug make dev

# Enable pprof for Go engines
PPROF=true ./bin/lx-engine

# Enable sanitizers for C++
CXXFLAGS="-fsanitize=address" make cpp-build
```

## License

Proprietary - Lux Exchange

## Support

For issues or questions:
- GitHub Issues: github.com/lux/lx-backend
- Slack: #lx-backend
- Email: backend@lux.exchange