# LX DEX

[![CI](https://github.com/luxfi/dex/actions/workflows/ci.yml/badge.svg)](https://github.com/luxfi/dex/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/luxfi/dex)](https://github.com/luxfi/dex/releases)
[![Go Version](https://img.shields.io/badge/go-1.21+-blue.svg)](https://go.dev)
[![License](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

Ultra-high performance DEX with 434M+ orders/sec on Apple Silicon.

## Features

- **Ultra-high performance**: 100M+ trades/sec capability
- **Sub-microsecond latency**: <1μs order matching (1181 ns/op achieved)
- **Multi-engine architecture**: Pure Go, C++, and GPU (MLX)
- **Quantum-resistant consensus**: FPC with Ringtail+BLS signatures
- **Cross-platform**: Linux, macOS (Intel & Apple Silicon), Windows

## Quick Start

```bash
# Install dependencies
go mod download

# Build all binaries
make build

# Run tests
make test

# Run demo
./bin/demo

# Run benchmarks
make bench
```

## Installation

### From Source

```bash
git clone https://github.com/luxfi/dex
cd dex
make build
```

### From Release

```bash
# Download latest release for your platform
curl -L https://github.com/luxfi/dex/releases/latest/download/lx-dex-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) -o lx-dex
chmod +x lx-dex
./lx-dex
```

## Performance

| Metric | Target | Achieved |
|--------|--------|-----------|
| Order Latency | <1μs | 1181 ns |
| Concurrent Latency | <2μs | 1738 ns |
| Throughput (CPU) | 1M/sec | 847K/sec |
| Throughput (GPU) | 100M/sec | 434M/sec* |

*With MLX GPU acceleration on Apple Silicon

## Architecture

The DEX uses a multi-engine architecture:

- **Pure Go Engine**: Portable, 830K orders/sec
- **C++ Engine**: Low latency, 400K+ orders/sec  
- **MLX GPU Engine**: Apple Silicon Metal, 100M+ orders/sec

See [docs/](docs/) for detailed documentation.

## Development

### Requirements

- Go 1.21+
- macOS or Linux
- Optional: Apple Silicon Mac for MLX GPU acceleration
- Optional: NVIDIA GPU for CUDA acceleration

### Building with GPU Support

```bash
# Apple Silicon (Metal)
CGO_ENABLED=1 make build

# Linux with CUDA
CGO_ENABLED=1 CUDA=1 make build
```

### Running Tests

```bash
# Unit tests
make test

# Benchmarks
make bench

# All tests including integration
go test ./...
```

## CI/CD

The project uses GitHub Actions for CI/CD:

- **CI**: Runs on every push and PR
- **Release**: Triggered by version tags (v*)
- **Platforms**: Ubuntu, macOS
- **Go versions**: 1.21, 1.22

See [.github/workflows/](.github/workflows/) for workflow definitions.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

Copyright (C) 2020-2025, Lux Industries Inc.

## Links

- [Documentation](docs/)
- [Releases](https://github.com/luxfi/dex/releases)
- [Issues](https://github.com/luxfi/dex/issues)
