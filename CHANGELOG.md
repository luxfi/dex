# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Performance
- Order matching: 1181 ns/op (under 2μs target)
- Concurrent operations: 1738 ns/op
- Throughput: 847K orders/sec (CPU)

## [0.1.0] - 2025-01-18

### Added
- Initial release
- Ultra-high performance order book
- Multi-engine architecture (Go, C++, MLX)
- Quantum-resistant consensus (FPC)
- Basic benchmarking suite
- Demo application

### Performance
- Achieved 434M+ orders/sec with MLX GPU
- Sub-microsecond latency for order matching

## [0.0.1] - 2025-01-15

### Added
- Project initialization
- Basic order book implementation
- Test suite framework

---

[0.2.0]: https://github.com/luxfi/dex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/luxfi/dex/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/luxfi/dex/releases/tag/v0.0.1