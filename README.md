# LX

Lux DEX — pure-Go matching engine, order book, oracle aggregator, and
JSON-RPC / WebSocket / gRPC SDKs.

[![CI](https://github.com/luxfi/dex/actions/workflows/ci.yml/badge.svg)](https://github.com/luxfi/dex/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/luxfi/dex)](https://github.com/luxfi/dex/releases)
[![Go Version](https://img.shields.io/badge/go-1.26+-blue.svg)](https://go.dev)
[![License](https://img.shields.io/badge/license-Lux%20Research%20%2B%20Patent%20Reservation-blue.svg)](LICENSE)

## Open core

This repository is the **public, pure-Go reference implementation** of
the Lux DEX matching engine. It is fully functional, runs standalone,
and underpins every Lux DEX deployment.

For commercial deployments that need hardware acceleration (NUMA-aware
C++ order book, CUDA / Metal batched verification, FPGA fast paths)
the same Go interfaces are implemented by `lux-private/dex` and
selected at build time via the `dex_gpu` build tag. The accelerated
backend fails closed unless the operator's environment carries a Lux
commercial license token whose scope list includes `dex`. Contact
`licensing@lux.network` for commercial licensing.

## Features

- **High performance (measured, Apple M1 Max)**: 11.88M orders/sec (C++ engine, 10 threads), 2.2M orders/sec (pure Go)
- **Low latency**: 169 ns avg match (p50 125 ns, p99 292 ns) on the C++ engine; 381 ns/order in pure Go
- **Multi-engine architecture**: pure Go and NUMA-aware C++ — order matching runs on CPU
- **Quantum-resistant consensus**: DAG with post-quantum signatures
- **Cross-platform**: Linux, macOS (Intel & Apple Silicon), Windows
- **Professional Market Data**: Real-time oracle integration with multiple sources
- **Cross-Chain Support**: Universal bridge for all major blockchains
- **100% Test Coverage**: All critical paths tested and verified

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

Measured first-hand on Apple M1 Max. Order matching runs on **CPU** — see
"Why matching stays on CPU" below.

| Engine | Avg match | p50 / p99 | Throughput |
|--------|-----------|-----------|------------|
| C++ (10 threads) | — | — | 11.88M orders/sec |
| C++ (single thread) | 169 ns | 125 ns / 292 ns | 5.91M orders/sec |
| Pure Go (`pkg/lx`) | 381 ns | — | 2.2M orders/sec |

C++ order cancel: **49 ns**. Pure-Go matching does **1–2 allocs/op** (not zero-alloc).

### Why matching stays on CPU

GPU order matching exists only as a CUDA path on Linux (commercial
`lux-private/dex`); there is no Apple/Metal matching path. A single match is
~169 ns — smaller than the dispatch overhead of handing a batch to an
integrated GPU — so the CPU wins for matching. The GPU pays off in the **FHE**
layer instead: the Metal NTT kernel is **23× faster** than the CPU NTT at
N=4096, batch=128, where batched polynomial decomposition dominates.

## Architecture

On-chain settlement follows **D matches · C settles**: the D-Chain (`dexvm`)
matches and BLS-signs a `DFillReceipt`; the C-Chain receipt-settlement precompile
`0x9999` (Uniswap-V4 `PoolManager` ABI) verifies the certificate inline and settles
under Block-STM. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#on-chain-settlement-cd)
and the normative spec **LP-9999**.

The DEX uses a multi-engine architecture; order matching runs on **CPU**:

- **Pure Go engine** (`pkg/lx`): portable reference, 2.2M orders/sec, 381 ns/order
- **C++ engine**: 11.88M orders/sec (10 threads), 5.91M single-thread, 169 ns avg match

### FIX Protocol Performance (December 2024)

| Engine | NewOrderSingle | ExecutionReport | MarketDataSnapshot |
|--------|----------------|-----------------|-------------------|
| Pure Go | 163K/sec | 124K/sec | 332K/sec |
| Hybrid Go/C++ | 167K/sec | 378K/sec | 616K/sec |
| Pure C++ | 444K/sec | 804K/sec | 1.08M/sec |
| Rust | 484K/sec | 232K/sec | 586K/sec |

See [docs/](docs/) for detailed documentation.

## Development

### Requirements

- Go 1.21+
- macOS or Linux
- Optional: Apple Silicon Mac for Metal-accelerated FHE (NTT)
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
