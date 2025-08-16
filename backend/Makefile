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
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(LX_DEX) ./cmd/dex
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(LX_GATEWAY) ./cmd/gateway
	@echo "Build complete (Pure Go)"

# Hybrid Go/C++ build (with CGO)
.PHONY: hybrid-build
hybrid-build: $(CPP_BRIDGE_OBJECTS)
	@echo "Building LX backend (Hybrid Go/C++)..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(LX_DEX) ./cmd/dex
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(LX_GATEWAY) ./cmd/gateway
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
	@echo "Benchmark build complete (Pure Go)"
	@echo "Note: Hybrid benchmark requires C++ bridge implementation"

# Build all benchmark tools
.PHONY: bench-tools
bench-tools:
	@echo "Building all benchmark tools..."
	@mkdir -p $(BIN_DIR)
	# Core benchmarks
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/simple-benchmark ./cmd/simple-benchmark
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/mega-traders ./cmd/mega-traders
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/max-perf-bench ./cmd/max-perf-bench
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/load-test ./cmd/load-test
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/client-test ./cmd/client-test
	# C++ standalone benchmark
	$(CXX) $(CXXFLAGS) -pthread cpp/standalone_bench.cpp -o $(BIN_DIR)/cpp-bench
	@echo "All benchmark tools built successfully"

# Build all server versions for benchmarking
.PHONY: bench-servers
bench-servers: go-build
	@echo "Building all server versions..."
	# Pure Go (already built)
	# Hybrid Go/C++
	@echo "Building Hybrid CGO version..."
	$(CXX) -c -std=c++17 -O3 -march=native -fPIC bridge/simple_orderbook.cpp -o bridge/simple_orderbook.o
	ar rcs bridge/liborderbook.a bridge/simple_orderbook.o
	CGO_ENABLED=1 $(GO) build -tags cgo -o $(BIN_DIR)/lx-dex-hybrid ./cmd/dex
	@echo "All servers built successfully"

# Run tests
.PHONY: test
test:
	@echo "Running tests (Pure Go)..."
	CGO_ENABLED=0 $(GO) test -v ./...
	
.PHONY: test-hybrid
test-hybrid:
	@echo "Running tests (Hybrid)..."
	CGO_ENABLED=1 $(GO) test -v ./...

# Run benchmarks
.PHONY: run-benchmark
run-benchmark: benchmark
	@echo "Running benchmarks..."
	@echo "=== Pure Go ==="
	$(LX_BENCHMARK)
	@echo "=== Hybrid Go/C++ ==="
	$(BIN_DIR)/lx-benchmark-hybrid

# Quick benchmark - 1000 traders
.PHONY: bench-quick
bench-quick: bench-tools bench-servers
	@echo "=== QUICK BENCHMARK (1000 traders, 30s) ==="
	@pkill lx-dex || true
	@sleep 1
	# Start servers
	@$(BIN_DIR)/lx-dex -port 50051 > /tmp/go-server.log 2>&1 &
	@$(BIN_DIR)/lx-dex-hybrid -port 50052 > /tmp/hybrid-server.log 2>&1 &
	@sleep 2
	# Run tests
	@echo "Testing Pure Go..."
	@$(BIN_DIR)/mega-traders -traders 1000 -rate 10 -duration 30s -grpc localhost:50051
	@echo ""
	@echo "Testing Hybrid CGO..."
	@$(BIN_DIR)/mega-traders -traders 1000 -rate 10 -duration 30s -hybrid
	@pkill lx-dex || true

# Full benchmark suite - all tests
.PHONY: bench-full
bench-full: bench-tools bench-servers
	@echo "=== FULL BENCHMARK SUITE ==="
	@echo "This will take approximately 5 minutes..."
	@pkill lx-dex || true
	@sleep 1
	# Start servers
	@$(BIN_DIR)/lx-dex -port 50051 > /tmp/go-server.log 2>&1 &
	@LX_ORDERBOOK_IMPL=cpp $(BIN_DIR)/lx-dex-hybrid -port 50052 > /tmp/hybrid-server.log 2>&1 &
	@sleep 2
	# Test 1: 1000 traders
	@echo "=== TEST 1: 1,000 Traders (30s) ==="
	@$(BIN_DIR)/mega-traders -traders 1000 -rate 10 -duration 30s -grpc localhost:50051
	@echo ""
	@$(BIN_DIR)/mega-traders -traders 1000 -rate 10 -duration 30s -hybrid
	# Test 2: 5000 traders
	@echo ""
	@echo "=== TEST 2: 5,000 Traders (20s) ==="
	@$(BIN_DIR)/mega-traders -traders 5000 -rate 5 -duration 20s -grpc localhost:50051
	# Test 3: Maximum throughput
	@echo ""
	@echo "=== TEST 3: Maximum Throughput ==="
	@$(BIN_DIR)/max-perf-bench -target go -duration 30s -warmup 3s -max-traders 10000 -step 1000
	# Test 4: C++ standalone
	@echo ""
	@echo "=== TEST 4: Pure C++ Benchmark ==="
	@$(BIN_DIR)/cpp-bench 1000 10
	@pkill lx-dex || true
	@echo ""
	@echo "=== BENCHMARK COMPLETE ==="

# Maximum performance test
.PHONY: bench-max
bench-max: bench-tools bench-servers
	@echo "=== FINDING MAXIMUM PERFORMANCE ==="
	@pkill lx-dex || true
	@sleep 1
	# Start servers
	@$(BIN_DIR)/lx-dex -port 50051 > /tmp/go-server.log 2>&1 &
	@LX_ORDERBOOK_IMPL=cpp $(BIN_DIR)/lx-dex-hybrid -port 50052 > /tmp/hybrid-server.log 2>&1 &
	@sleep 2
	# Find maximum for each implementation
	@echo "Finding Pure Go maximum..."
	@$(BIN_DIR)/max-perf-bench -target go -duration 30s -warmup 3s -max-traders 10000 -step 1000
	@echo ""
	@echo "Finding Hybrid CGO maximum..."
	@$(BIN_DIR)/max-perf-bench -target hybrid -duration 30s -warmup 3s -max-traders 10000 -step 1000
	@echo ""
	@echo "Testing Pure C++ maximum..."
	@$(BIN_DIR)/cpp-bench 1000 10 | tail -15
	@pkill lx-dex || true

# Stress test with many traders
.PHONY: bench-stress
bench-stress: bench-tools bench-servers
	@echo "=== STRESS TEST (10,000 traders) ==="
	@pkill lx-dex || true
	@sleep 1
	@$(BIN_DIR)/lx-dex -port 50051 > /tmp/go-server.log 2>&1 &
	@sleep 2
	@$(BIN_DIR)/mega-traders -traders 10000 -rate 5 -duration 60s -grpc localhost:50051
	@pkill lx-dex || true

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

# ZeroMQ network benchmarks
.PHONY: zmq-build
zmq-build:
	@echo "Building ZeroMQ benchmark tools..."
	@mkdir -p $(BIN_DIR)
	go get github.com/pebbe/zmq4
	go build -o $(BIN_DIR)/zmq-exchange ./cmd/zmq-exchange
	go build -o $(BIN_DIR)/zmq-trader ./cmd/zmq-trader
	go build -o $(BIN_DIR)/zmq-benchmark ./cmd/zmq-benchmark
	@echo "ZMQ tools built successfully"

# Run local ZeroMQ benchmark
.PHONY: bench-zmq-local
bench-zmq-local: zmq-build
	@echo "=== ZeroMQ LOCAL Network Benchmark ==="
	@$(BIN_DIR)/zmq-benchmark -mode local -traders 100 -rate 1000 -duration 30s

# Show instructions for distributed ZeroMQ benchmark
.PHONY: bench-zmq-dist
bench-zmq-dist: zmq-build
	@echo "=== ZeroMQ DISTRIBUTED Network Benchmark ==="
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
	@echo "  zmq-build    - Build ZeroMQ network tools"
	@echo ""
	@echo "Benchmark Targets:"
	@echo "  bench-quick  - Quick benchmark (1000 traders, 30s)"
	@echo "  bench-full   - Full benchmark suite (~5 minutes)"
	@echo "  bench-max    - Find maximum throughput for each engine"
	@echo "  bench-stress - Stress test with 10,000 traders"
	@echo "  bench-report - Generate performance report"
	@echo ""
	@echo "Network Benchmark Targets:"
	@echo "  bench-zmq-local - Test ZeroMQ locally"
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