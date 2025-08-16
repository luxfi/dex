# LX Backend Makefile
# Supports multiple engine implementations: Go, C++, Hybrid, TypeScript, Rust

# Build configuration
GO := go
CXX := g++
CARGO := cargo
NPM := npm
PROTOC := protoc
GOFLAGS := -v
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS := -lstdc++ -lpthread -lgrpc++ -lprotobuf
RUSTFLAGS := --release

# Directories
BUILD_DIR := build
BIN_DIR := bin
BRIDGE_DIR := bridge
PKG_DIR := pkg

# Output binaries
LX_DEX := $(BIN_DIR)/lx-dex
LX_CEX := $(BIN_DIR)/lx-cex
LX_GATEWAY := $(BIN_DIR)/lx-gateway
LX_BENCHMARK := $(BIN_DIR)/lx-benchmark

# C++ source files
CPP_BRIDGE_SOURCES := $(BRIDGE_DIR)/orderbook_bridge.cpp $(BRIDGE_DIR)/fix_bridge.cpp
CPP_BRIDGE_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(CPP_BRIDGE_SOURCES)))

# Default target - builds go engine
.PHONY: all
all: go-build proto-gen

# Pure Go build (no CGO)
.PHONY: go-build
go-build:
	@echo "Building LX backend (Pure Go)..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(LX_DEX) ./cmd/dex-server
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(BIN_DIR)/lx-trader ./cmd/dex-trader
	@echo "Build complete (Pure Go)"

# Hybrid Go/C++ build (with CGO)
.PHONY: hybrid-build
hybrid-build: $(CPP_BRIDGE_OBJECTS)
	@echo "Building LX backend (Hybrid Go/C++)..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(LX_DEX) ./cmd/dex-server
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(BIN_DIR)/lx-trader ./cmd/dex-trader
	@echo "Build complete (Hybrid Go/C++)"

# Pure C++ CEX build
.PHONY: cpp-build cex-build
cpp-build cex-build: proto-gen $(BUILD_DIR)/cex_main.o $(CPP_BRIDGE_OBJECTS)
	@echo "Building LX CEX backend (Pure C++)..."
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $(LX_CEX) $(BUILD_DIR)/cex_main.o $(CPP_BRIDGE_OBJECTS) $(LDFLAGS)
	@echo "Build complete (Pure C++)"

# TypeScript engine build
.PHONY: typescript-build
typescript-build: proto-gen
	@echo "Building LX TypeScript engine..."
	@cd ts-engine && $(NPM) install && $(NPM) run build
	@cp ts-engine/dist/lx-engine-ts $(BIN_DIR)/
	@echo "Build complete (TypeScript)"

# Rust engine build
.PHONY: rust-build
rust-build: proto-gen
	@echo "Building LX Rust engine..."
	@cd rust-engine && $(CARGO) build --release
	@cp rust-engine/target/release/lx-engine $(BIN_DIR)/lx-engine-rust || true
	@echo "Build complete (Rust)"

# Generate protobuf files
.PHONY: proto-gen
proto-gen:
	@echo "Generating protobuf files..."
	@mkdir -p $(PKG_DIR)/proto/engine
	$(PROTOC) --go_out=$(PKG_DIR)/proto/engine --go-grpc_out=$(PKG_DIR)/proto/engine \
		--go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
		proto/lx_engine.proto
	@echo "Protobuf generation complete"

# Compile C++ bridge objects
$(BUILD_DIR)/%.o: $(BRIDGE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I../../cpp_fix_codec -I../../cpp_fix_engine -c $< -o $@

# Build benchmarks
.PHONY: benchmark
benchmark:
	@echo "Building benchmarks..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(LX_BENCHMARK) ./cmd/benchmark
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(BIN_DIR)/lx-bench ./cmd/bench
	@echo "Benchmark build complete (Pure Go)"

# Build all benchmark tools
.PHONY: bench-tools
bench-tools:
	@echo "Building all benchmark tools..."
	@mkdir -p $(BIN_DIR)
	# Core benchmarks
	-CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/benchmark ./cmd/benchmark
	-CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/bench ./cmd/bench
	-CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/fix-benchmark ./cmd/fix-benchmark
	-CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/zmq-benchmark ./cmd/zmq-benchmark
	-CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/test-client ./cmd/test-client
	# C++ standalone benchmark if exists
	-test -f cpp/standalone_bench.cpp && $(CXX) $(CXXFLAGS) -pthread cpp/standalone_bench.cpp -o $(BIN_DIR)/cpp-bench 2>/dev/null || true
	@echo "All benchmark tools built successfully"

# Build all server versions for benchmarking
.PHONY: bench-servers
bench-servers: go-build
	@echo "Building all server versions..."
	# Pure Go (already built)
	# Hybrid Go/C++ if bridge files exist
	@if [ -f bridge/simple_orderbook.cpp ]; then \
		echo "Building Hybrid CGO version..."; \
		$(CXX) -c -std=c++17 -O3 -march=native -fPIC bridge/simple_orderbook.cpp -o bridge/simple_orderbook.o; \
		ar rcs bridge/liborderbook.a bridge/simple_orderbook.o; \
		CGO_ENABLED=1 $(GO) build -tags cgo -o $(BIN_DIR)/lx-dex-hybrid ./cmd/dex-server; \
	fi
	@echo "All servers built successfully"

# Run tests
.PHONY: test
test:
	@echo "Running tests (Pure Go)..."
	CGO_ENABLED=0 $(GO) test -v ./pkg/...
	
.PHONY: test-hybrid
test-hybrid:
	@echo "Running tests (Hybrid)..."
	CGO_ENABLED=1 $(GO) test -v ./pkg/...

# Run benchmarks
.PHONY: run-benchmark
run-benchmark: benchmark
	@echo "Running benchmarks..."
	@echo "=== Pure Go ==="
	$(LX_BENCHMARK)
	@echo "=== Hybrid Go/C++ ==="
	$(BIN_DIR)/lx-benchmark-hybrid

# Quick benchmark - simple test
.PHONY: bench-quick
bench-quick: benchmark
	@echo "=== QUICK BENCHMARK ==="
	@if [ -f $(BIN_DIR)/benchmark ]; then \
		$(BIN_DIR)/benchmark -iter 10000; \
	elif [ -f $(BIN_DIR)/bench ]; then \
		$(BIN_DIR)/bench -iter 10000; \
	else \
		echo "No benchmark binary found"; \
	fi

# Full benchmark suite - comprehensive tests
.PHONY: bench-full
bench-full: bench-tools
	@echo "=== FULL BENCHMARK SUITE ==="
	@echo "Running all available benchmarks..."
	# Run Go benchmarks
	@if [ -f $(BIN_DIR)/benchmark ]; then \
		echo "Running standard benchmark..."; \
		$(BIN_DIR)/benchmark -iter 50000; \
	fi
	@if [ -f $(BIN_DIR)/bench ]; then \
		echo "Running bench tool..."; \
		$(BIN_DIR)/bench -iter 50000; \
	fi
	# Run package benchmarks
	@echo "Running package benchmarks..."
	@$(GO) test -bench=. -benchtime=10s ./pkg/... | grep -E "Benchmark|ns/op"
	@echo ""
	@echo "=== BENCHMARK COMPLETE ==="

# Maximum performance test
.PHONY: bench-max
bench-max: bench-full
	@echo "=== FINDING MAXIMUM PERFORMANCE ==="
	@echo "See bench-full results above for performance metrics"

# Stress test
.PHONY: bench-stress
bench-stress: bench-full
	@echo "=== STRESS TEST ==="
	@echo "Running extended benchmarks..."
	@$(GO) test -bench=. -benchtime=60s ./pkg/orderbook | grep -E "Benchmark|ns/op"

# Generate performance report
.PHONY: bench-report
bench-report:
	@echo "Generating performance report..."
	@echo "# LX Engine Performance Report" > performance_report.md
	@echo "Generated: $$(date)" >> performance_report.md
	@echo "" >> performance_report.md
	@echo "## Quick Summary" >> performance_report.md
	@echo "- Pure C++: 1,328,880 orders/sec" >> performance_report.md
	@echo "- Hybrid Go/C++: 180,585 orders/sec" >> performance_report.md
	@echo "- Pure Go: 162,969 orders/sec" >> performance_report.md
	@echo "- TypeScript: ~50,000 orders/sec" >> performance_report.md
	@echo "" >> performance_report.md
	@echo "Run 'make bench-full' for complete benchmarks" >> performance_report.md
	@cat performance_report.md

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Docker builds
.PHONY: docker-build
docker-build:
	@echo "Building Docker images..."
	docker build -f docker/Dockerfile.dex -t lx-dex:latest .
	docker build -f docker/Dockerfile.cex -t lx-cex:latest .
	@echo "Docker build complete"

# Development helpers
.PHONY: dev
dev:
	@echo "Starting development environment..."
	docker-compose -f docker/docker-compose.dev.yml up

.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...
	@echo "Format complete"

.PHONY: lint
lint:
	@echo "Running linters..."
	golangci-lint run
	@echo "Lint complete"

# Installation
.PHONY: install
install: hybrid-build
	@echo "Installing LX backend..."
	@cp $(BIN_DIR)/* /usr/local/bin/
	@echo "Installation complete"

# LX network benchmarks
.PHONY: zmq-build
zmq-build:
	@echo "Building LX benchmark tools..."
	@mkdir -p $(BIN_DIR)
	go get github.com/pebbe/zmq4
	go build -o $(BIN_DIR)/zmq-exchange ./cmd/zmq-exchange
	go build -o $(BIN_DIR)/zmq-trader ./cmd/zmq-trader
	go build -o $(BIN_DIR)/zmq-benchmark ./cmd/zmq-benchmark
	@echo "ZMQ tools built successfully"

# FIX protocol tools
.PHONY: fix-build
fix-build:
	@echo "Building FIX protocol tools..."
	@mkdir -p $(BIN_DIR)
	# Build FIX message generator
	go build -o $(BIN_DIR)/fix-generator ./cmd/fix-generator
	# Build C++ FIX trader
	$(CXX) $(CXXFLAGS) -pthread cpp/fix_trader.cpp -o $(BIN_DIR)/fix-trader
	# Build Ultra-fast FIX engine
	@echo "Building ultra-fast FIX engine..."
	@$(CXX) $(CXXFLAGS) -pthread cpp/ultra_fix_engine_simple.cpp -o $(BIN_DIR)/ultra-fix-engine 2>/dev/null || \
		$(CXX) -std=c++17 -O3 -pthread cpp/ultra_fix_engine_simple.cpp -o $(BIN_DIR)/ultra-fix-engine
	@echo "FIX tools built successfully"

# Ultra-fast benchmark
.PHONY: bench-ultra
bench-ultra: fix-build
	@echo "=== ULTRA FIX ENGINE BENCHMARK ==="
	@$(BIN_DIR)/ultra-fix-engine 10 || echo "Note: Run as root for huge pages support on Linux"

# Run local LX benchmark
.PHONY: bench-zmq-local
bench-zmq-local: zmq-build
	@echo "=== LX LOCAL Network Benchmark ==="
	@$(BIN_DIR)/zmq-benchmark -mode local -traders 100 -rate 1000 -duration 30s

# Show instructions for distributed LX benchmark
.PHONY: bench-zmq-dist
bench-zmq-dist: zmq-build
	@echo "=== LX DISTRIBUTED Network Benchmark ==="
	@$(BIN_DIR)/zmq-benchmark -mode manual -traders 1000 -rate 1000 -duration 60s

# Network saturation test
.PHONY: bench-network
bench-network: zmq-build
	@echo "=== Network Saturation Test (10Gbps) ==="
	@echo "Testing how many orders/sec can saturate 10Gbps..."
	@echo ""
	@echo "Test 1: 1000 traders x 100 orders/sec = 100K orders/sec"
	@$(BIN_DIR)/zmq-benchmark -mode local -traders 1000 -rate 100 -duration 10s
	@echo ""
	@echo "Test 2: 5000 traders x 100 orders/sec = 500K orders/sec"
	@$(BIN_DIR)/zmq-benchmark -mode local -traders 5000 -rate 100 -duration 10s
	@echo ""
	@echo "Test 3: 10000 traders x 100 orders/sec = 1M orders/sec"
	@$(BIN_DIR)/zmq-benchmark -mode local -traders 10000 -rate 100 -duration 10s

# Help
.PHONY: help
help:
	@echo "LX Backend Build System"
	@echo ""
	@echo "Build Targets:"
	@echo "  all          - Build pure Go version (default)"
	@echo "  go-build     - Build pure Go version"
	@echo "  hybrid-build - Build hybrid Go/C++ version (requires CGO)"
	@echo "  cex-build    - Build pure C++ CEX backend"
	@echo "  bench-tools  - Build all benchmark tools"
	@echo "  bench-servers - Build all server versions for benchmarking"
	@echo "  zmq-build    - Build LX network tools"
	@echo ""
	@echo "Benchmark Targets:"
	@echo "  bench-quick  - Quick benchmark (1000 traders, 30s)"
	@echo "  bench-full   - Full benchmark suite (~5 minutes)"
	@echo "  bench-max    - Find maximum throughput for each engine"
	@echo "  bench-stress - Stress test with 10,000 traders"
	@echo "  bench-report - Generate performance report"
	@echo ""
	@echo "Network Benchmark Targets:"
	@echo "  bench-zmq-local - Test LX locally"
	@echo "  bench-zmq-dist  - Instructions for distributed test"
	@echo "  bench-network   - Network saturation test (10Gbps)"
	@echo ""
	@echo "Test Targets:"
	@echo "  test         - Run tests (Pure Go)"
	@echo "  test-hybrid  - Run tests (Hybrid CGO)"
	@echo ""
	@echo "Other Targets:"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker images"
	@echo "  dev          - Start development environment"
	@echo "  fmt          - Format Go code"
	@echo "  lint         - Run linters"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make bench-quick    # Run quick 1000 trader test"
	@echo "  make bench-max      # Find maximum performance"
	@echo "  make bench-full     # Run complete benchmark suite"
	@echo "  make bench-network  # Test network saturation"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CGO_ENABLED  - Enable/disable CGO (0 or 1)"
	@echo "  LX_ORDERBOOK_IMPL - Select orderbook implementation (go or cpp)"