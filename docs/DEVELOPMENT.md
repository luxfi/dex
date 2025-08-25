# LX DEX Development Guide

## Table of Contents
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)

## Getting Started

### Prerequisites
- Go 1.21+
- Docker & Docker Compose
- Make
- Git

### Initial Setup
```bash
# Clone repository
git clone https://github.com/luxfi/dex.git
cd dex

# Install dependencies
go mod download

# Run tests to verify setup
make test

# Build the project
make build
```

## Project Structure

```
dex/
├── pkg/                    # Core packages
│   ├── lx/                # Trading engine
│   │   ├── orderbook.go  # Order matching engine
│   │   ├── margin.go      # Margin trading
│   │   ├── vaults.go      # DeFi vaults
│   │   └── bridge.go      # Cross-chain bridge
│   ├── api/               # API servers
│   ├── consensus/         # DAG consensus
│   └── client/            # Client libraries
├── cmd/                   # Executables
├── examples/              # Example code
├── test/                  # Test suites
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── sdk/                   # Language SDKs
├── docs/                  # Documentation
└── .github/workflows/     # CI/CD pipelines
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
Follow the coding standards:
- Use meaningful variable names
- Add comments for complex logic
- Follow Go conventions
- Keep functions small and focused

### 3. Write Tests
Every new feature must include tests:
```go
func TestYourFeature(t *testing.T) {
    // Arrange
    setup := setupTest()
    
    // Act
    result := YourFeature(input)
    
    // Assert
    assert.Equal(t, expected, result)
}
```

### 4. Run Tests Locally
```bash
# Run all tests
make test

# Run specific test
go test -run TestOrderBook ./pkg/lx/

# Run with coverage
go test -cover ./pkg/lx/
```

### 5. Commit Changes
```bash
git add .
git commit -m "feat: add new feature

- Detailed description
- What was changed
- Why it was changed"
```

### 6. Push and Create PR
```bash
git push origin feature/your-feature-name
```

## Testing

### Unit Tests
```bash
# Run unit tests
go test ./pkg/...

# With coverage
go test -cover ./pkg/...

# Generate coverage report
go test -coverprofile=coverage.out ./pkg/...
go tool cover -html=coverage.out
```

### Integration Tests
```bash
# Run integration tests
go test ./test/integration/...

# Skip in CI (long-running)
go test -short ./test/integration/...
```

### Benchmarks
```bash
# Run all benchmarks
make bench

# Run specific benchmark
go test -bench=BenchmarkOrderBook -benchtime=10s ./pkg/lx/

# With memory profiling
go test -bench=. -benchmem ./pkg/lx/
```

### Load Testing
```bash
# Run load test
go run cmd/stress-test/main.go

# Monitor performance
watch -n 1 'go run cmd/monitor/main.go'
```

## Performance Optimization

### Profiling
```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=. ./pkg/lx/
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof -bench=. ./pkg/lx/
go tool pprof mem.prof

# Trace execution
go test -trace=trace.out ./pkg/lx/
go tool trace trace.out
```

### Optimization Tips
1. **Use sync.Pool** for frequently allocated objects
2. **Minimize allocations** in hot paths
3. **Use atomic operations** instead of mutexes where possible
4. **Profile before optimizing** - measure, don't guess
5. **Batch operations** when possible

### Example Optimization
```go
// Before: Allocates on every call
func processOrder(order Order) {
    result := &Result{} // Allocation
    // process...
}

// After: Reuse with sync.Pool
var resultPool = sync.Pool{
    New: func() interface{} {
        return &Result{}
    },
}

func processOrder(order Order) {
    result := resultPool.Get().(*Result)
    defer resultPool.Put(result)
    result.Reset() // Clear previous data
    // process...
}
```

## Contributing

### Code Style
- Run `gofmt` before committing
- Use `golangci-lint` for linting
- Follow [Effective Go](https://golang.org/doc/effective_go.html)

```bash
# Format code
make fmt

# Run linters
make lint

# Run all checks
make check
```

### Commit Convention
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Testing
- `perf:` Performance improvement
- `refactor:` Code refactoring
- `chore:` Maintenance

### Pull Request Process
1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review
5. Address feedback
6. Squash commits if needed

### Review Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No security issues
- [ ] Performance impact considered
- [ ] Breaking changes documented
- [ ] Error handling appropriate

## Debugging

### Local Debugging
```bash
# Run with debug logs
LOG_LEVEL=debug go run cmd/dex-server/main.go

# Use Delve debugger
dlv debug cmd/dex-server/main.go

# Attach to running process
dlv attach <pid>
```

### Remote Debugging
```bash
# Start with debug server
dlv debug --headless --listen=:2345 cmd/dex-server/main.go

# Connect from IDE or CLI
dlv connect :2345
```

## Performance Metrics

### Key Metrics to Monitor
- **Latency**: Order matching time
- **Throughput**: Orders/second
- **Memory**: Heap usage and GC pressure
- **Goroutines**: Active count and leaks
- **Network**: Message rate and bandwidth

### Monitoring Tools
```bash
# Built-in metrics
curl http://localhost:9090/metrics

# Prometheus
docker-compose up prometheus grafana

# Custom dashboard
go run cmd/monitor/main.go
```

## Deployment

### Local Development
```bash
# Single node
make run

# Multi-node cluster
make run-cluster

# With Docker
docker-compose up
```

### Production
```bash
# Build production binary
CGO_ENABLED=0 make build

# Build Docker image
docker build -t lux-dex:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Troubleshooting

### Common Issues

**1. Tests timing out**
```bash
# Increase timeout
go test -timeout=30s ./pkg/lx/

# Run in short mode
go test -short ./pkg/lx/
```

**2. Memory leaks**
```bash
# Check for goroutine leaks
go test -run=. -count=1 ./pkg/lx/

# Profile memory
go tool pprof http://localhost:6060/debug/pprof/heap
```

**3. Race conditions**
```bash
# Run with race detector
go test -race ./pkg/lx/
```

## Resources

### Documentation
- [Go Documentation](https://go.dev/doc/)
- [Effective Go](https://golang.org/doc/effective_go.html)
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)

### Tools
- [Delve Debugger](https://github.com/go-delve/delve)
- [pprof](https://github.com/google/pprof)
- [golangci-lint](https://golangci-lint.run/)
- [go-torch](https://github.com/uber/go-torch)

### Community
- Discord: [discord.gg/luxfi](https://discord.gg/luxfi)
- GitHub Issues: [github.com/luxfi/dex/issues](https://github.com/luxfi/dex/issues)
- Twitter: [@luxfi](https://twitter.com/luxfi)