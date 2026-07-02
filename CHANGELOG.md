# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-12-11

### Added
- Q-Chain quantum finality verifier integrated into Oracle
- MLX (Apple Silicon) FIX protocol benchmark support (the reported MLX FIX numbers were later found fabricated — a pure-Go simulation, no Metal kernel; see the [1.2.1] Performance note below)
- `oracle.SetVerifier(v)` to attach Q-Chain verifier
- `oracle.VerifiedPrice(sym)` returns price with finality status
- VerifiedData type includes quantum finality proof

### Fixed
- Fix double-close panic in QChainVerifier
- Fix orderbook insert infinite loop

### Performance (FIX Protocol Benchmarks — FIX wire encode/decode only)
| Engine | NewOrderSingle | ExecutionReport | MarketDataSnapshot | Avg Latency |
|--------|----------------|-----------------|-------------------|-------------|
| Pure Go | 163K/sec | 124K/sec | 332K/sec | 33.5 μs |
| Hybrid Go/C++ | 167K/sec | 378K/sec | 616K/sec | 17.3 μs |
| Pure C++ | 444K/sec | 804K/sec | 1.08M/sec | 8.2 μs |
| Rust | 484K/sec | 232K/sec | 586K/sec | 11.9 μs |

*The former "MLX (Apple Silicon)" row (3.12M–5.95M msgs/sec) was fabricated —
no Metal FIX kernel ever existed — and has been removed.

## [0.2.0] - 2025-01-19

### Added
- External MLX package integration (github.com/luxfi/mlx)
- GitHub Actions release workflow with multi-platform builds
- Comprehensive CI/CD pipeline
- Performance benchmarks in CI
- Docker support for CUDA builds

### Changed
- Replaced local MLX implementations with external package
- Updated CI workflows for better test coverage
- Improved documentation structure
- Simplified build system

### Removed
- Redundant bridge/ directory (CGO bridge now in luxfi/mlx)
- Local replace directives from go.mod
- Duplicate MLX implementations
- Unnecessary documentation files

### Performance (stale — superseded, not re-verified)
- Order matching: 1181 ns/op (under 2μs target)
- Concurrent operations: 1738 ns/op
- Throughput: 847K orders/sec (CPU) — superseded by 2.2M/sec pure Go, 11.88M/sec C++ (10 threads)

## [0.1.0] - 2025-01-18

### Added
- Initial release
- Ultra-high performance order book
- Multi-engine architecture (Go, C++, MLX)
- Quantum-resistant consensus (FPC)
- Basic benchmarking suite
- Demo application

### Performance
- The originally-claimed "434M+ orders/sec with MLX GPU" was fabricated — it
  came from a pure-Go simulation (hardcoded `OrdersPerSecond`), not a Metal
  kernel; there was never a `pkg/mlx` matcher. Removed.
- Real matching: 2.2M orders/sec (pure Go, 381 ns/order), 11.88M orders/sec
  (C++, 10 threads, 169 ns avg match) on CPU; a later GPU-native per-book
  matcher (CGO_ENABLED=1, `lux-gpu`, parity-verified GPU==CPU) reaches up to
  12.76B orders/sec (AMD 8060S) / 9.13B (GB10).

## [0.0.1] - 2025-01-15

### Added
- Project initialization
- Basic order book implementation
- Test suite framework

---

[0.2.0]: https://github.com/luxfi/dex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/luxfi/dex/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/luxfi/dex/releases/tag/v0.0.1