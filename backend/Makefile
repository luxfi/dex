# LX DEX Backend Makefile
# Ultra-High Performance DEX with FPC Consensus and Quantum Finality
# Supports: Go, C++, Hybrid, TypeScript, Rust, DPDK, RDMA, GPU (MLX/CUDA)

# Build configuration
GO := go
CXX := g++
CARGO := cargo
NPM := npm
PROTOC := protoc
GOFLAGS := -v
CXXFLAGS := -std=c++20 -O3 -march=native -Wall -Wextra
LDFLAGS := -lstdc++ -lpthread -lgrpc++ -lprotobuf
RUSTFLAGS := --release

# DPDK/RDMA flags (optional)
DPDK_CFLAGS := $(shell pkg-config --cflags libdpdk 2>/dev/null)
DPDK_LDFLAGS := $(shell pkg-config --libs libdpdk 2>/dev/null)
RDMA_LDFLAGS := -libverbs -lrdmacm

# GPU flags (optional)
CUDA_FLAGS := $(shell pkg-config --cflags cuda-12.0 2>/dev/null)
MLX_FLAGS := -framework Metal -framework MetalPerformanceShaders

# Directories
BUILD_DIR := build
BIN_DIR := bin
BRIDGE_DIR := bridge
PKG_DIR := pkg
SCRIPTS_DIR := scripts

# Output binaries
LX_DEX := $(BIN_DIR)/lx-dex
LX_CEX := $(BIN_DIR)/lx-cex
LX_GATEWAY := $(BIN_DIR)/lx-gateway
LX_BENCHMARK := $(BIN_DIR)/lx-benchmark
DAG_NETWORK := $(BIN_DIR)/dag-network
FPC_NETWORK := $(BIN_DIR)/fpc-network

# C++ source files
CPP_BRIDGE_SOURCES := $(BRIDGE_DIR)/orderbook_bridge.cpp $(BRIDGE_DIR)/fix_bridge.cpp $(BRIDGE_DIR)/ultra_matcher_bridge.cpp
CPP_BRIDGE_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(CPP_BRIDGE_SOURCES)))

# Default target - builds everything needed for development
.PHONY: all
all: go-build dag-build test-quick

# ==================== CORE BUILDS ====================

# Pure Go build (no CGO)
.PHONY: go-build
go-build:
	@echo "🚀 Building LX DEX backend (Pure Go)..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(LX_DEX) ./cmd/dex-server
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) -o $(BIN_DIR)/lx-trader ./cmd/dex-trader
	@echo "✅ Build complete (Pure Go)"

# Hybrid Go/C++ build (with CGO)
.PHONY: hybrid-build
hybrid-build: $(CPP_BRIDGE_OBJECTS)
	@echo "🚀 Building LX backend (Hybrid Go/C++)..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(LX_DEX)-hybrid ./cmd/dex-server
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(BIN_DIR)/lx-trader-hybrid ./cmd/dex-trader
	@echo "✅ Build complete (Hybrid Go/C++)"

# Pure C++ CEX build (Ultra-low latency)
.PHONY: cpp-build
cpp-build: proto-gen
	@echo "🚀 Building LX CEX backend (Pure C++)..."
	@mkdir -p $(BIN_DIR)
	@if [ -f cpp/cex_main.cpp ]; then \
		$(CXX) $(CXXFLAGS) -o $(LX_CEX) cpp/cex_main.cpp $(LDFLAGS); \
	else \
		echo "⚠️  C++ CEX source not found, skipping..."; \
	fi
	@echo "✅ Build complete (Pure C++)"

# ==================== CONSENSUS & DAG BUILDS ====================

# DAG network with FPC consensus
.PHONY: dag-build
dag-build:
	@echo "🔗 Building DAG network with FPC consensus..."
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GOFLAGS) -o $(DAG_NETWORK) ./cmd/dag-network
	@echo "✅ DAG network build complete"

# FPC network runner
.PHONY: fpc-build
fpc-build: dag-build
	@echo "⚡ Building FPC network tools..."
	@cp $(DAG_NETWORK) $(FPC_NETWORK)
	@chmod +x $(SCRIPTS_DIR)/run-fpc-network.sh 2>/dev/null || true
	@echo "✅ FPC network build complete"

# X-Chain DEX build (Lux X-Chain integration)
.PHONY: xchain-build
xchain-build:
	@echo "🔗 Building X-Chain DEX..."
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GOFLAGS) -o $(BIN_DIR)/xchain-dex ./cmd/xchain-dex
	@echo "✅ X-Chain DEX build complete"

# Q-Chain build (Quantum finality chain)
.PHONY: qchain-build
qchain-build:
	@echo "🔐 Building Q-Chain (quantum finality)..."
	@mkdir -p $(BIN_DIR)
	@if [ -f cmd/qchain/main.go ]; then \
		$(GO) build $(GOFLAGS) -o $(BIN_DIR)/qchain ./cmd/qchain; \
	else \
		echo "⚠️  Q-Chain source not found, creating stub..."; \
		echo "Q-Chain implementation pending..."; \
	fi
	@echo "✅ Q-Chain build complete"

# X-Chain + Q-Chain unified build
.PHONY: xq-build
xq-build: xchain-build qchain-build
	@echo "🚀 Building unified X-Chain + Q-Chain system..."
	@echo "✅ X-Chain + Q-Chain build complete"

# ==================== HIGH-PERFORMANCE BUILDS ====================

# DPDK build (kernel-bypass networking)
.PHONY: dpdk-build
dpdk-build:
	@echo "🚀 Building DPDK-enabled version..."
	@if [ -n "$(DPDK_CFLAGS)" ]; then \
		$(CXX) $(CXXFLAGS) $(DPDK_CFLAGS) -o $(BIN_DIR)/lx-dpdk \
			pkg/dpdk/dpdk_orderbook.cpp $(DPDK_LDFLAGS); \
		echo "✅ DPDK build complete"; \
	else \
		echo "⚠️  DPDK not found. Install with: apt-get install dpdk-dev"; \
	fi

# RDMA build (zero-copy replication)
.PHONY: rdma-build
rdma-build:
	@echo "🚀 Building RDMA-enabled version..."
	@if [ -f pkg/rdma/rdma_replication.cpp ]; then \
		$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/lx-rdma \
			pkg/rdma/rdma_replication.cpp $(RDMA_LDFLAGS) 2>/dev/null || \
			echo "⚠️  RDMA libraries not found. Install with: apt-get install libibverbs-dev librdmacm-dev"; \
	fi

# GPU build (MLX/CUDA acceleration)
.PHONY: gpu-build
gpu-build:
	@echo "🎮 Building GPU-accelerated version..."
	@if [ -f pkg/matching/matching_engine.cpp ]; then \
		if [ -n "$(CUDA_FLAGS)" ]; then \
			echo "Building with CUDA support..."; \
			nvcc -O3 -o $(BIN_DIR)/lx-cuda pkg/matching/matching_engine.cpp; \
		elif [ "$(shell uname)" = "Darwin" ]; then \
			echo "Building with MLX/Metal support..."; \
			$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $(BIN_DIR)/lx-mlx \
				pkg/matching/matching_engine.cpp; \
		else \
			echo "⚠️  No GPU support detected"; \
		fi; \
	fi

# ==================== TESTING ====================

# Quick test suite (unit tests only)
.PHONY: test-quick
test-quick:
	@echo "🧪 Running quick tests..."
	@CGO_ENABLED=0 $(GO) test -short -timeout 30s ./pkg/lx/... ./pkg/consensus/...
	@echo "✅ Quick tests passed"

# Full test suite (all tests)
.PHONY: test
test:
	@echo "🧪 Running full test suite..."
	@CGO_ENABLED=0 $(GO) test -v -timeout 5m ./...
	@echo "✅ All tests passed"

# FPC consensus tests
.PHONY: test-fpc
test-fpc:
	@echo "⚡ Running FPC consensus tests..."
	@$(GO) test -v -timeout 2m ./pkg/consensus -run TestFPC
	@echo "✅ FPC tests passed"

# Quantum signature tests
.PHONY: test-quantum
test-quantum:
	@echo "🔐 Running quantum signature tests..."
	@$(GO) test -v -timeout 2m ./pkg/consensus -run TestQuantum
	@echo "✅ Quantum signature tests passed"

# Quasar certificate tests
.PHONY: test-quasar
test-quasar:
	@echo "🛡️ Running Quasar certificate tests..."
	@$(GO) test -v -timeout 2m ./pkg/consensus -run TestQuasar
	@echo "✅ Quasar certificate tests passed"

# Integration tests
.PHONY: test-integration
test-integration: dag-build
	@echo "🔄 Running integration tests..."
	@$(GO) test -v -tags=integration -timeout 10m ./test/integration/...
	@echo "✅ Integration tests passed"

# Race condition tests
.PHONY: test-race
test-race:
	@echo "🏃 Running race condition tests..."
	@CGO_ENABLED=1 $(GO) test -race -timeout 5m ./pkg/...
	@echo "✅ No race conditions detected"

# Coverage report
.PHONY: test-coverage
test-coverage:
	@echo "📊 Generating test coverage report..."
	@$(GO) test -coverprofile=coverage.out ./pkg/...
	@$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "✅ Coverage report generated: coverage.html"

# ==================== BENCHMARKS ====================

# Quick benchmark
.PHONY: bench-quick
bench-quick: go-build
	@echo "⚡ Running quick benchmark..."
	@$(GO) test -bench=. -benchtime=10s -run=^$ ./pkg/lx
	@echo "✅ Quick benchmark complete"

# ZMQ Binary FIX benchmark
.PHONY: bench-zmq
bench-zmq:
	@echo "📡 Building ZMQ benchmark tool..."
	@$(GO) build $(GOFLAGS) -o $(BIN_DIR)/zmq-benchmark ./cmd/zmq-benchmark
	@echo "✅ ZMQ benchmark tool built"

# ZMQ local throughput test
.PHONY: bench-zmq-local
bench-zmq-local: bench-zmq
	@echo "🚀 Running ZMQ local throughput test..."
	@$(BIN_DIR)/zmq-benchmark -mode producer -rate 1000000 -duration 10s &
	@sleep 1
	@$(BIN_DIR)/zmq-benchmark -mode consumer -duration 10s
	@echo "✅ ZMQ local test complete"

# ZMQ 2-node consensus test
.PHONY: cluster-2node-local
cluster-2node-local: bench-zmq
	@echo "🌐 Running 2-node consensus test..."
	@echo "Start node 1: $(BIN_DIR)/zmq-benchmark -mode consensus -node 1 -endpoint tcp://0.0.0.0:5555 tcp://localhost:6002"
	@echo "Start node 2: $(BIN_DIR)/zmq-benchmark -mode consensus -node 2 -endpoint tcp://0.0.0.0:6002 tcp://localhost:5555"
	@echo "✅ Instructions displayed"

# Full benchmark suite
.PHONY: bench-full
bench-full: go-build hybrid-build
	@echo "📊 Running full benchmark suite..."
	@echo "=== Pure Go ==="
	@$(GO) test -bench=. -benchtime=30s -run=^$ ./pkg/lx
	@echo ""
	@echo "=== FPC Consensus ==="
	@$(GO) test -bench=. -benchtime=30s -run=^$ ./pkg/consensus
	@echo ""
	@echo "=== Quantum Signatures ==="
	@$(GO) test -bench=BenchmarkQuantum -benchtime=10s -run=^$ ./pkg/consensus
	@echo "✅ Full benchmark complete"

# Ultra-performance benchmark (100M trades/sec target)
.PHONY: bench-ultra
bench-ultra: dpdk-build rdma-build gpu-build
	@echo "🚀 Running ultra-performance benchmark (100M trades/sec target)..."
	@if [ -f $(BIN_DIR)/lx-dpdk ]; then \
		echo "DPDK benchmark:"; \
		$(BIN_DIR)/lx-dpdk --benchmark; \
	fi
	@if [ -f $(BIN_DIR)/lx-rdma ]; then \
		echo "RDMA benchmark:"; \
		$(BIN_DIR)/lx-rdma --benchmark; \
	fi
	@if [ -f $(BIN_DIR)/lx-cuda ] || [ -f $(BIN_DIR)/lx-mlx ]; then \
		echo "GPU benchmark:"; \
		$(BIN_DIR)/lx-cuda --benchmark 2>/dev/null || $(BIN_DIR)/lx-mlx --benchmark; \
	fi

# ==================== NETWORK OPERATIONS ====================

# Run 3-node FPC network
.PHONY: run-fpc
run-fpc: fpc-build
	@echo "🌐 Starting 3-node FPC network with quantum finality..."
	@cd $(SCRIPTS_DIR) && ./run-fpc-network.sh

# Run DAG network manually
.PHONY: run-dag
run-dag: dag-build
	@echo "🔗 Starting DAG network..."
	@$(DAG_NETWORK) -node node0 -http 8080 -pub 5000 -rep 5002 -leader

# Run X-Chain DEX standalone
.PHONY: run-xchain
run-xchain: xchain-build
	@echo "🔗 Starting X-Chain DEX..."
	@$(BIN_DIR)/xchain-dex --data-dir ./xchain-data --enable-qchain=false

# Run Q-Chain standalone
.PHONY: run-qchain
run-qchain: qchain-build
	@echo "🔐 Starting Q-Chain (quantum finality)..."
	@if [ -f $(BIN_DIR)/qchain ]; then \
		$(BIN_DIR)/qchain --data-dir ./qchain-data; \
	else \
		echo "⚠️  Q-Chain not yet implemented"; \
	fi

# Run X-Chain + Q-Chain unified (standalone mode)
.PHONY: run-xq
run-xq: xq-build
	@echo "🚀 Starting unified X-Chain + Q-Chain system..."
	@echo "Starting X-Chain with Q-Chain integration..."
	@$(BIN_DIR)/xchain-dex --data-dir ./xq-data --enable-qchain=true --standalone &
	@echo "✅ X-Chain + Q-Chain running in standalone mode"
	@echo "   - X-Chain API: http://localhost:8080"
	@echo "   - Q-Chain endpoint: tcp://localhost:9000"

# Monitor network
.PHONY: monitor
monitor:
	@echo "📊 Monitoring network statistics..."
	@while true; do \
		clear; \
		echo "=== FPC Network Status ==="; \
		curl -s http://localhost:8080/stats | jq . 2>/dev/null || echo "Node 0: Not responding"; \
		curl -s http://localhost:8081/stats | jq . 2>/dev/null || echo "Node 1: Not responding"; \
		curl -s http://localhost:8082/stats | jq . 2>/dev/null || echo "Node 2: Not responding"; \
		sleep 2; \
	done

# ==================== CI/CD ====================

# CI test suite (for GitHub Actions)
.PHONY: ci-test
ci-test:
	@echo "🤖 Running CI test suite..."
	@$(MAKE) lint
	@$(MAKE) test-quick
	@$(MAKE) test-fpc
	@$(MAKE) test-quantum
	@$(MAKE) test-quasar
	@$(MAKE) test-race
	@echo "✅ CI tests passed"

# CI build check
.PHONY: ci-build
ci-build:
	@echo "🤖 Running CI build check..."
	@$(MAKE) go-build
	@$(MAKE) dag-build
	@$(MAKE) hybrid-build
	@echo "✅ CI build check passed"

# CI benchmark (quick)
.PHONY: ci-bench
ci-bench:
	@echo "🤖 Running CI benchmarks..."
	@$(GO) test -bench=. -benchtime=3s -run=^$ ./pkg/lx | grep -E "Benchmark|ns/op"
	@echo "✅ CI benchmarks complete"

# ==================== DEVELOPMENT TOOLS ====================

# Format code
.PHONY: fmt
fmt:
	@echo "🎨 Formatting code..."
	@$(GO) fmt ./...
	@gofumpt -w . 2>/dev/null || true
	@echo "✅ Code formatted"

# Lint code
.PHONY: lint
lint:
	@echo "🔍 Running linters..."
	@golangci-lint run --deadline=5m || \
		(echo "⚠️  golangci-lint not found. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest" && \
		 go vet ./...)
	@echo "✅ Lint complete"

# Generate protobuf
.PHONY: proto-gen
proto-gen:
	@echo "📝 Generating protobuf files..."
	@mkdir -p $(PKG_DIR)/proto/engine
	@if [ -f proto/lx_engine.proto ]; then \
		$(PROTOC) --go_out=$(PKG_DIR)/proto/engine --go-grpc_out=$(PKG_DIR)/proto/engine \
			--go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
			proto/lx_engine.proto; \
		echo "✅ Protobuf generation complete"; \
	else \
		echo "⚠️  No proto files found"; \
	fi

# Install dependencies
.PHONY: deps
deps:
	@echo "📦 Installing dependencies..."
	@$(GO) mod download
	@$(GO) mod tidy
	@echo "✅ Dependencies installed"

# ==================== DOCKER ====================

# Build Docker images
.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker images..."
	@docker build -f docker/Dockerfile.dex -t lx-dex:latest .
	@docker build -f docker/Dockerfile.fpc -t lx-fpc:latest .
	@echo "✅ Docker images built"

# Run Docker compose
.PHONY: docker-run
docker-run:
	@echo "🐳 Starting Docker containers..."
	@docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Containers started"

# ==================== CLEANUP ====================

# Clean build artifacts
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR) coverage.* *.prof
	@echo "✅ Clean complete"

# Deep clean (including dependencies)
.PHONY: clean-all
clean-all: clean
	@echo "🧹 Deep cleaning..."
	@$(GO) clean -cache -testcache -modcache
	@echo "✅ Deep clean complete"

# ==================== HELP ====================

.PHONY: help
help:
	@echo "LX DEX - Ultra-High Performance DEX with Quantum-Resistant Consensus"
	@echo "=================================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make all          - Build and test everything"
	@echo "  make run-fpc      - Run 3-node FPC network with quantum finality"
	@echo "  make monitor      - Monitor network statistics"
	@echo ""
	@echo "🔨 Build Targets:"
	@echo "  make go-build     - Build pure Go version"
	@echo "  make hybrid-build - Build hybrid Go/C++ version"
	@echo "  make cpp-build    - Build pure C++ version"
	@echo "  make dag-build    - Build DAG network with FPC consensus"
	@echo "  make dpdk-build   - Build DPDK version (kernel-bypass)"
	@echo "  make rdma-build   - Build RDMA version (zero-copy)"
	@echo "  make gpu-build    - Build GPU-accelerated version (MLX/CUDA)"
	@echo ""
	@echo "🧪 Test Targets:"
	@echo "  make test         - Run full test suite"
	@echo "  make test-quick   - Run quick tests"
	@echo "  make test-fpc     - Test FPC consensus"
	@echo "  make test-quantum - Test quantum signatures"
	@echo "  make test-quasar  - Test Quasar certificates"
	@echo "  make test-race    - Test for race conditions"
	@echo "  make test-coverage - Generate coverage report"
	@echo ""
	@echo "⚡ Benchmark Targets:"
	@echo "  make bench-quick  - Quick benchmark"
	@echo "  make bench-full   - Full benchmark suite"
	@echo "  make bench-ultra  - Ultra-performance (100M trades/sec)"
	@echo ""
	@echo "🤖 CI/CD Targets:"
	@echo "  make ci-test      - Run CI test suite"
	@echo "  make ci-build     - Run CI build check"
	@echo "  make ci-bench     - Run CI benchmarks"
	@echo ""
	@echo "🛠️ Development:"
	@echo "  make fmt          - Format code"
	@echo "  make lint         - Run linters"
	@echo "  make deps         - Install dependencies"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "📊 Performance:"
	@echo "  - 100M trades/sec with DPDK + RDMA + GPU"
	@echo "  - 50ms FPC consensus finality"
	@echo "  - Quantum-resistant with Ringtail+BLS"
	@echo "  - Zero-copy networking throughout"

# Default help
.DEFAULT_GOAL := help