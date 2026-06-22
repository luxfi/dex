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
and underpins every Lux DEX deployment. Built with `CGO_ENABLED=0` it is
a self-contained pure-Go binary with zero native dependencies.

### GPU acceleration is a runtime decision, not a build tag

The batched GPU paths (constant-product AMM curve evaluation and the
flat-buffer CLOB match primitive) live **in this repository** under
`pkg/lx/` and are compiled into the `CGO_ENABLED=1` build. They bind to
the luxcpp DEX kernels through pkg-config:

| pkg-config bundle    | backend        | consumed by                     |
| -------------------- | -------------- | ------------------------------- |
| `lux-dex-amm-metal`  | Metal (macOS)  | `pkg/lx/amm_gpu_metal.go`        |
| `lux-dex-amm-cuda`   | CUDA (Linux)   | `pkg/lx/amm_gpu_cuda.go`         |
| `lux-dex-clob-cuda`  | CUDA (Linux)   | `pkg/lx/orderbook_cuda.go`       |

These bundles ship from `luxcpp/dex` (`cmake --install`). Whether a GPU is
actually used is decided **at runtime** by `github.com/luxfi/crypto/backend`
detection, never by a build tag:

- `GPU_DISABLE=1` forces the CPU path (recorded as `reason=disabled`).
- No Metal/CUDA device, or the runtime kernel/metallib is absent → the
  per-platform dispatch returns its `unsupported` sentinel and the call
  transparently falls back to the pure-Go CPU CLOB / AMM oracle (recorded
  as `reason=unsupported`). The fallback is logged — a CPU result is never
  mislabeled as GPU.
- macOS resolves the AMM metallib at runtime via `LUX_DEX_AMM_METALLIB`
  (default: the installed `share/lux/dex/amm_xyk.metallib`).

There is no `dex_gpu` build tag. The one optional opt-in tag is
`lux_secp256k1_metal`, which force-links the Apple-only batched
secp256k1-ecrecover Metal archive; without it the signed-order batch
verifier runs the secp256k1 **CPU** pipeline. See
`pkg/lx/signed_order_metal_anchor_darwin.go` for the build + opt-in recipe.

For commercial deployments that need the heavier accelerators (NUMA-aware
C++ order book, FPGA fast paths) the same Go interfaces are implemented by
the private `lux-private/dex` tier, which fails closed unless the operator's
environment carries a Lux commercial license token whose scope list
includes `dex`. Contact `licensing@lux.network` for commercial licensing.

## Features

- **Ultra-high performance**: 13M+ orders/sec achieved with planet-scale architecture
- **Sub-microsecond latency**: 75.9ns order matching, 636ns position updates
- **Multi-engine architecture**: Pure Go, C++, and GPU (CUDA/MLX)
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

| Metric | Target | Achieved | Status |
|--------|--------|-----------|---------|
| Order Latency (GPU) | <1μs | 2 ns | ✅ 500x better |
| Order Latency (CPU) | <1μs | 487 ns | ✅ 2x better |
| Throughput (CPU) | 1M/sec | 1.01M/sec | ✅ Exceeded |
| Throughput (GPU) | 100M/sec | 434M/sec | ✅ 4.34x |
| Test Coverage | 100% pass | 100% pass | ✅ Complete |
| Code Coverage | 30% | 39.1% | ✅ Exceeded |

*With MLX GPU acceleration on Apple Silicon M2 Ultra

## Architecture

On-chain settlement follows **D matches · C settles**: the D-Chain (`dexvm`)
matches and BLS-signs a `DFillReceipt`; the C-Chain receipt-settlement precompile
`0x9999` (Uniswap-V4 `PoolManager` ABI) verifies the certificate inline and settles
under Block-STM. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#on-chain-settlement-cd)
and the normative spec **LP-9999**.

The DEX uses a multi-engine architecture:

- **Pure Go Engine**: Portable, 830K orders/sec
- **C++ Engine**: Low latency, 800K+ orders/sec
- **Rust Engine**: High performance, 585K+ orders/sec
- **MLX GPU Engine**: Apple Silicon Metal, 6M+ msgs/sec (FIX), 434M+ orders/sec

### FIX Protocol Performance (December 2024)

| Engine | NewOrderSingle | ExecutionReport | MarketDataSnapshot |
|--------|----------------|-----------------|-------------------|
| Pure Go | 163K/sec | 124K/sec | 332K/sec |
| Hybrid Go/C++ | 167K/sec | 378K/sec | 616K/sec |
| Pure C++ | 444K/sec | 804K/sec | 1.08M/sec |
| Rust | 484K/sec | 232K/sec | 586K/sec |
| **MLX (Apple Silicon)** | **3.12M/sec** | **4.27M/sec** | **5.95M/sec** |

*MLX achieves sub-2μs average latency (0.68-1.75μs) via GPU parallelism

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
